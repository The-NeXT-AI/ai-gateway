import type {
  SourceAdapter,
  StandardResponse,
  StandardResponseOutputItem,
  StandardResponseReasoning
} from '../../../types';
import { ok } from '../../../types';
import { asBoolean, isObject } from '../../../utils';
import { buildOpenAIHeaders, normalizeOpenAIResponsesCompletedResponse } from '../common';
import {
  encodeReasoningTransportEnvelope,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import { parseOpenAIResponsesRequest } from './parsers';
import { addNamespaceFieldsToStandardResponse } from '../target/tools';

export const openAIResponsesSourceAdapter: SourceAdapter = {
  key: 'openai_responses',
  provider: 'openai',
  toStandardRequest(input) {
    return parseOpenAIResponsesRequest(input.body);
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
      url: `${input.config.openaiBaseUrl}/responses`,
      headers: headersResult.value,
      body: input.body
    });
  }
};

function prepareOpenAIResponsesClientResponse(response: StandardResponse): StandardResponse {
  const output: StandardResponseOutputItem[] = [];

  for (const item of response.output) {
    if (item.type === 'reasoning') {
      output.push(prepareReasoningItemForOpenAIResponsesClient(item));
      continue;
    }

    if (item.type === 'function_call') {
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
            'signature'
          )
        });
      }

      const {
        thought_signature: _thoughtSignature,
        thought_signature_format: _thoughtSignatureFormat,
        ...functionCall
      } = item;
      output.push(functionCall);
      continue;
    }

    output.push(item);
  }

  return {
    ...response,
    output
  };
}

function prepareReasoningItemForOpenAIResponsesClient(
  item: StandardResponseReasoning
): StandardResponseReasoning {
  const opaqueState = readReasoningOpaqueState(item);
  const {
    reasoning_details: _reasoningDetails,
    source_format: _sourceFormat,
    ...reasoning
  } = item;

  if (!opaqueState) {
    const { encrypted_content: _encryptedContent, ...withoutUnknownOpaqueState } = reasoning;
    return withoutUnknownOpaqueState;
  }

  return {
    ...reasoning,
    encrypted_content:
      opaqueState.format === OPENAI_RESPONSES_REASONING_FORMAT
        ? opaqueState.data
        : encodeReasoningTransportEnvelope(
            opaqueState.format,
            opaqueState.data,
            opaqueState.id || item.id,
            opaqueState.kind
          )
  };
}

function readReasoningOpaqueState(item: StandardResponseReasoning): {
  format: string;
  data: string;
  id?: string;
  kind: 'signature' | 'encrypted';
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
        kind: signature ? 'signature' : 'encrypted'
      };
    }
  }

  if (item.source_format && item.encrypted_content) {
    return {
      format: item.source_format,
      data: item.encrypted_content,
      id: item.id,
      kind: 'encrypted'
    };
  }

  return undefined;
}
