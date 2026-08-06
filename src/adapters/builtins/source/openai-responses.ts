import type {
  ProviderNativeItem,
  SourceAdapter,
  StandardResponse,
  StandardResponseOutputItem,
  StandardResponseReasoning
} from '../../../types';
import { ok } from '../../../types';
import { asBoolean, isObject } from '../../../utils';
import { buildOpenAIHeaders, normalizeOpenAIResponsesCompletedResponse } from '../common';
import {
  decodeReasoningTransportEnvelope,
  encodeReasoningTransportEnvelope,
  isProviderNativePayloadStructurallyValid,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import { parseOpenAIResponsesRequest } from './parsers';
import { addNamespaceFieldsToStandardResponse } from '../target/tools';

export const openAIResponsesSourceAdapter: SourceAdapter = {
  key: 'openai_responses',
  provider: 'openai',
  toStandardRequest(input) {
    return parseOpenAIResponsesRequest(
      input.body,
      input.source.metadata?.operation === 'compact' ? 'compact' : 'create'
    );
  },
  fromStandardResponse(input) {
    const response = prepareOpenAIResponsesClientResponse(
      addNamespaceFieldsToStandardResponse(input.response, input.standardRequest?.tools)
    );
    return normalizeOpenAIResponsesCompletedResponse({
      ...response
    });
  },
  isStreamingRequest(input) {
    return asBoolean(input.body.stream) === true;
  },
  buildPassthroughRequest(input) {
    const headersResult = buildOpenAIHeaders(input.request.headers, input.config);
    if (!headersResult.ok) {
      return headersResult;
    }

    return ok({
      url: `${input.config.openaiBaseUrl}/responses${
        input.source.metadata?.operation === 'compact' ? '/compact' : ''
      }`,
      headers: headersResult.value,
      body: input.body
    });
  }
};

export function prepareOpenAIResponsesClientResponse(response: StandardResponse): StandardResponse {
  const output: StandardResponseOutputItem[] = [];

  for (const item of response.output) {
    if (item.type === 'provider_native_item') {
      if (
        item.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
        isProviderNativePayloadStructurallyValid(item, OPENAI_RESPONSES_REASONING_FORMAT)
      ) {
        output.push(asStandardResponseOutputItem(encodeProviderNativePayloadForClient(item)));
      }
      continue;
    }

    if (item.type === 'reasoning') {
      if (
        item.native_item?.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
        isProviderNativePayloadStructurallyValid(item.native_item, OPENAI_RESPONSES_REASONING_FORMAT)
      ) {
        output.push(asStandardResponseOutputItem(
          encodeProviderNativePayloadForClient(item.native_item)
        ));
        continue;
      }
      output.push(prepareReasoningItemForOpenAIResponsesClient(item));
      continue;
    }

    if (item.type === 'function_call') {
      if (
        item.native_item?.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
        isProviderNativePayloadStructurallyValid(item.native_item, OPENAI_RESPONSES_REASONING_FORMAT)
      ) {
        output.push(asStandardResponseOutputItem(
          encodeProviderNativePayloadForClient(item.native_item, {
            name: item.name,
            arguments: item.arguments,
            ...(item.namespace ? { namespace: item.namespace } : {}),
            ...(item.caller ? { caller: item.caller } : {})
          })
        ));
        continue;
      }
      if (item.thought_signature && item.thought_signature_format) {
        const carrierId = `rs_ccr_${item.id.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
        output.push({
          id: carrierId,
          type: 'reasoning',
          status: 'completed',
          summary: [],
          encrypted_content: encodeReasoningTransportEnvelope(
            item.thought_signature_format,
            item.thought_signature,
            carrierId,
            'signature',
            item.thought_signature_origin,
            item.native_item ? { nativeItem: item.native_item } : undefined
          )
        });
      }

      const {
        thought_signature: _thoughtSignature,
        thought_signature_format: _thoughtSignatureFormat,
        thought_signature_origin: _thoughtSignatureOrigin,
        native_item: _nativeItem,
        ...functionCall
      } = item;
      output.push(functionCall);
      continue;
    }

    if (
      item.type === 'message' &&
      item.native_item?.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
      isProviderNativePayloadStructurallyValid(item.native_item, OPENAI_RESPONSES_REASONING_FORMAT)
    ) {
      output.push(asStandardResponseOutputItem(
        encodeProviderNativePayloadForClient(item.native_item, {
          ...(item.phase ? { phase: item.phase } : {})
        })
      ));
      continue;
    }

    if (item.type === 'message') {
      const { native_item: _nativeItem, ...message } = item;
      output.push(message);
    } else {
      output.push(item);
    }
  }

  return {
    ...response,
    output
  };
}

function encodeProviderNativePayloadForClient(
  item: ProviderNativeItem,
  overrides: Record<string, unknown> = {}
): Record<string, unknown> {
  const payload = {
    ...item.raw_payload,
    ...overrides
  };
  if (item.source_origin.endpoint === 'pending' || item.source_origin.endpoint === 'unverified') {
    return payload;
  }
  const encode = (value: unknown, key?: string): unknown => {
    if (typeof value === 'string') {
      const carrierField = key === 'encrypted_content' || key === 'signature' ||
        key === 'thoughtSignature' || key === 'thought_signature' ||
        key === 'fingerprint' || key === 'caller';
      if (!carrierField || !value || decodeReasoningTransportEnvelope(value)) {
        return value;
      }
      return encodeReasoningTransportEnvelope(
        item.source_format,
        value,
        item.native_id,
        key === 'encrypted_content' ? 'encrypted' : 'signature',
        item.source_origin,
        { nativeItem: item }
      );
    }
    if (Array.isArray(value)) {
      return value.map((entry) => encode(entry));
    }
    if (!isObject(value)) {
      return value;
    }
    return Object.fromEntries(
      Object.entries(value).map(([childKey, child]) => [childKey, encode(child, childKey)])
    );
  };
  return encode(payload) as Record<string, unknown>;
}

function asStandardResponseOutputItem(
  value: Record<string, unknown>
): StandardResponseOutputItem {
  return value as unknown as StandardResponseOutputItem;
}

function prepareReasoningItemForOpenAIResponsesClient(
  item: StandardResponseReasoning
): StandardResponseReasoning {
  const opaqueState = readReasoningOpaqueState(item);
  const {
    reasoning_details: _reasoningDetails,
    source_format: _sourceFormat,
    source_origin: _sourceOrigin,
    native_item: _nativeItem,
    ...reasoning
  } = item;

  if (!opaqueState) {
    const { encrypted_content: _encryptedContent, ...withoutUnknownOpaqueState } = reasoning;
    return withoutUnknownOpaqueState;
  }

  return {
    ...reasoning,
    encrypted_content:
      opaqueState.format === OPENAI_RESPONSES_REASONING_FORMAT && !opaqueState.origin
        ? opaqueState.data
        : encodeReasoningTransportEnvelope(
            opaqueState.format,
            opaqueState.data,
            opaqueState.id || item.id,
            opaqueState.kind,
            opaqueState.origin,
            item.native_item ? { nativeItem: item.native_item } : undefined
          )
  };
}

function readReasoningOpaqueState(item: StandardResponseReasoning): {
  format: string;
  data: string;
  id?: string;
  kind: 'signature' | 'encrypted';
  origin?: StandardResponseReasoning['source_origin'];
} | undefined {
  for (const detail of item.reasoning_details || []) {
    if (!isObject(detail)) {
      continue;
    }
    const format = typeof detail.format === 'string' ? detail.format : item.source_format;
    const signature =
      typeof detail.signature === 'string'
        ? detail.signature
        : typeof detail.thoughtSignature === 'string'
          ? detail.thoughtSignature
          : typeof detail.thought_signature === 'string'
            ? detail.thought_signature
            : undefined;
    const encrypted =
      typeof detail.data === 'string'
        ? detail.data
        : typeof detail.encrypted_content === 'string'
          ? detail.encrypted_content
          : undefined;
    const data = signature || encrypted;
    if (format && data) {
      return {
        format,
        data,
        id: typeof detail.id === 'string' ? detail.id : item.id,
        kind: signature ? 'signature' : 'encrypted',
        ...(item.source_origin ? { origin: item.source_origin } : {})
      };
    }
  }

  if (item.source_format && item.encrypted_content) {
    return {
      format: item.source_format,
      data: item.encrypted_content,
      id: item.id,
      kind: 'encrypted',
      ...(item.source_origin ? { origin: item.source_origin } : {})
    };
  }

  return undefined;
}
