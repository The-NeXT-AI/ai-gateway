import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import type {
  GatewayConfig,
  Provider,
  ProviderConfig,
  ReasoningStateOrigin,
  StandardRequest,
  StandardRequestInputContent,
  StandardResponse
} from '../types';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  decodeReasoningTransportEnvelope,
  encodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../adapters/builtins/reasoning-envelope';
import { parseSseChunks } from '../sse';

export function buildReasoningStateOrigin(
  provider: Provider,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig,
  model: string | undefined
): ReasoningStateOrigin {
  const providerFamily = resolveProviderFamily(provider, providerConfig);
  const baseUrl = resolveProviderBaseUrl(providerFamily, providerConfig, config);
  const normalizedEndpoint = normalizeReasoningEndpoint(baseUrl);
  const endpoint = createHash('sha256')
    .update(`${providerFamily}\0${normalizedEndpoint}`, 'utf8')
    .digest('base64url');

  return {
    provider: providerFamily,
    endpoint,
    ...(model?.trim() ? { model: model.trim() } : {})
  };
}

export function normalizeReasoningEndpoint(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return '';
  }

  try {
    const parsed = new URL(trimmed);
    parsed.protocol = parsed.protocol.toLowerCase();
    parsed.hostname = parsed.hostname.toLowerCase();
    parsed.username = '';
    parsed.password = '';
    parsed.search = '';
    parsed.hash = '';
    if (
      (parsed.protocol === 'https:' && parsed.port === '443') ||
      (parsed.protocol === 'http:' && parsed.port === '80')
    ) {
      parsed.port = '';
    }
    const path = parsed.pathname.replace(/\/+$/, '');
    return `${parsed.protocol}//${parsed.host}${path}`;
  } catch {
    return trimmed.replace(/\/+$/, '').toLowerCase();
  }
}

export function resolveReasoningTargetFormat(
  provider: Provider,
  providerConfig?: ProviderConfig
): string | undefined {
  switch (providerConfig?.type) {
    case 'openai_responses':
      return OPENAI_RESPONSES_REASONING_FORMAT;
    case 'anthropic_messages':
      return ANTHROPIC_CLAUDE_REASONING_FORMAT;
    case 'gemini_generate_content':
      return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
    case 'gemini_interactions':
      return GEMINI_INTERACTIONS_REASONING_FORMAT;
    case 'openai_chat_completions':
      return undefined;
  }

  if (provider === 'openai') {
    return OPENAI_RESPONSES_REASONING_FORMAT;
  }
  if (provider === 'anthropic') {
    return ANTHROPIC_CLAUDE_REASONING_FORMAT;
  }
  if (provider === 'gemini') {
    return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
  }
  return undefined;
}

export function attachReasoningStateOrigin(
  response: StandardResponse,
  origin: ReasoningStateOrigin
): StandardResponse {
  return {
    ...response,
    output: response.output.map((item) => {
      if (item.type === 'reasoning' && hasOpaqueReasoningState(item)) {
        return {
          ...item,
          source_origin: origin
        };
      }
      if (item.type === 'function_call' && item.thought_signature) {
        return {
          ...item,
          thought_signature_origin: origin
        };
      }
      return item;
    })
  };
}

export function prepareReasoningStateForTarget(
  request: StandardRequest,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin
): StandardRequest {
  if (typeof request.input === 'string') {
    return request;
  }

  return {
    ...request,
    input: request.input.map((message) => ({
      ...message,
      content: message.content.flatMap<StandardRequestInputContent>((item) => {
        if (item.type === 'tool_use') {
          if (!item.thought_signature) {
            return [item];
          }
          if (
            targetFormat &&
            canReplayReasoningState(
              item.thought_signature_format,
              item.thought_signature_origin,
              targetFormat,
              targetOrigin
            )
          ) {
            return [item];
          }

          const {
            thought_signature: _thoughtSignature,
            thought_signature_format: _thoughtSignatureFormat,
            thought_signature_origin: _thoughtSignatureOrigin,
            ...toolUse
          } = item;
          return [toolUse];
        }

        if (item.type !== 'reasoning' || !hasOpaqueReasoningState(item)) {
          return [item];
        }

        if (
          targetFormat &&
          canReplayReasoningState(
            item.source_format,
            item.source_origin,
            targetFormat,
            targetOrigin
          )
        ) {
          return [item];
        }

        if (targetFormat) {
          // Strict protocols bind their native reasoning blocks to provider state.
          // Keep normal assistant text/tool calls, but do not synthesize an unsigned block.
          return [];
        }

        const readableDetails = sanitizeReadableReasoningDetails(item.reasoning_details);
        const {
          encrypted_content: _encryptedContent,
          source_origin: _sourceOrigin,
          ...readableReasoning
        } = item;
        return [
          {
            ...readableReasoning,
            ...(readableDetails.length > 0
              ? { reasoning_details: readableDetails }
              : { reasoning_details: undefined })
          }
        ];
      })
    }))
  };
}

export function canReplayReasoningState(
  sourceFormat: string | undefined,
  sourceOrigin: ReasoningStateOrigin | undefined,
  targetFormat: string,
  targetOrigin: ReasoningStateOrigin
): boolean {
  if (
    sourceFormat !== targetFormat ||
    !sourceOrigin ||
    sourceOrigin.provider !== targetOrigin.provider ||
    sourceOrigin.endpoint !== targetOrigin.endpoint
  ) {
    return false;
  }

  return Boolean(
    sourceOrigin.model &&
    targetOrigin.model &&
    sourceOrigin.model === targetOrigin.model
  );
}

export function wrapPassthroughReasoningPayload(
  payload: unknown,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin
): { payload: unknown; changed: boolean } {
  const format = sourceFormatForAdapter(sourceAdapterKey);
  if (!format || (typeof payload !== 'object' && !Array.isArray(payload))) {
    return { payload, changed: false };
  }

  let changed = false;
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (typeof value !== 'object' || value === null) {
      return;
    }

    const record = value as Record<string, unknown>;
    if (format === OPENAI_RESPONSES_REASONING_FORMAT && record.type === 'reasoning') {
      changed = wrapRecordField(record, 'encrypted_content', format, origin, {
        id: typeof record.id === 'string' ? record.id : undefined,
        kind: 'encrypted'
      }) || changed;
    } else if (format === ANTHROPIC_CLAUDE_REASONING_FORMAT) {
      if (record.type === 'thinking') {
        changed = wrapRecordField(record, 'signature', format, origin, { kind: 'signature' }) || changed;
      } else if (record.type === 'redacted_thinking') {
        changed = wrapRecordField(record, 'data', format, origin, { kind: 'encrypted' }) || changed;
      }
    } else if (format === GEMINI_GENERATE_CONTENT_REASONING_FORMAT) {
      changed = wrapRecordField(record, 'thoughtSignature', format, origin, { kind: 'signature' }) || changed;
      changed = wrapRecordField(record, 'thought_signature', format, origin, { kind: 'signature' }) || changed;
    } else if (format === GEMINI_INTERACTIONS_REASONING_FORMAT) {
      if (record.type === 'thought' || record.type === 'thought_signature') {
        changed = wrapRecordField(record, 'signature', format, origin, { kind: 'signature' }) || changed;
      }
    }

    Object.values(record).forEach(visit);
  };

  visit(payload);
  return { payload, changed };
}

export function createReasoningAwarePassthroughSseStream(
  response: Response,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin,
  abortSignal?: AbortSignal
): Readable {
  return Readable.from(
    relayReasoningAwarePassthroughSse(response, sourceAdapterKey, origin, abortSignal)
  );
}

async function* relayReasoningAwarePassthroughSse(
  response: Response,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin,
  abortSignal?: AbortSignal
): AsyncGenerator<string> {
  const anthropicSignatureByIndex = new Map<number, string>();

  const flushAnthropicSignature = (index: number): string | undefined => {
    const signature = anthropicSignatureByIndex.get(index);
    if (!signature) {
      return undefined;
    }
    anthropicSignatureByIndex.delete(index);
    return encodeSseChunk('content_block_delta', {
      type: 'content_block_delta',
      index,
      delta: {
        type: 'signature_delta',
        signature: encodeReasoningTransportEnvelope(
          ANTHROPIC_CLAUDE_REASONING_FORMAT,
          signature,
          undefined,
          'signature',
          origin
        )
      }
    });
  };

  for await (const chunk of parseSseChunks(response, abortSignal)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }
    if (data === '[DONE]') {
      for (const index of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
        const frame = flushAnthropicSignature(index);
        if (frame) {
          yield frame;
        }
      }
      yield encodeRawSseChunk(chunk.event, data);
      continue;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data) as unknown;
    } catch {
      yield encodeRawSseChunk(chunk.event, data);
      continue;
    }

    if (sourceAdapterKey === 'anthropic_messages' && isRecord(payload)) {
      const eventType = typeof payload.type === 'string' ? payload.type : chunk.event;
      const index = typeof payload.index === 'number' ? payload.index : undefined;
      const delta = isRecord(payload.delta) ? payload.delta : undefined;
      if (
        eventType === 'content_block_delta' &&
        index !== undefined &&
        delta?.type === 'signature_delta' &&
        typeof delta.signature === 'string'
      ) {
        anthropicSignatureByIndex.set(
          index,
          `${anthropicSignatureByIndex.get(index) || ''}${delta.signature}`
        );
        continue;
      }
      if (eventType === 'content_block_stop' && index !== undefined) {
        const frame = flushAnthropicSignature(index);
        if (frame) {
          yield frame;
        }
      } else if (eventType === 'message_stop') {
        for (const pendingIndex of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
          const frame = flushAnthropicSignature(pendingIndex);
          if (frame) {
            yield frame;
          }
        }
      }
    }

    const wrapped = wrapPassthroughReasoningPayload(payload, sourceAdapterKey, origin);
    yield encodeSseChunk(chunk.event, wrapped.payload);
  }

  for (const index of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
    const frame = flushAnthropicSignature(index);
    if (frame) {
      yield frame;
    }
  }
}

function resolveProviderFamily(provider: Provider, providerConfig?: ProviderConfig): string {
  switch (providerConfig?.type) {
    case 'openai_responses':
    case 'openai_chat_completions':
      return 'openai';
    case 'anthropic_messages':
      return 'anthropic';
    case 'gemini_generate_content':
    case 'gemini_interactions':
      return 'gemini';
    default:
      return provider.trim().toLowerCase();
  }
}

function resolveProviderBaseUrl(
  providerFamily: string,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig
): string {
  if (providerConfig?.baseurl?.trim()) {
    return providerConfig.baseurl.trim();
  }
  if (providerFamily === 'openai') {
    return config.openaiBaseUrl;
  }
  if (providerFamily === 'anthropic') {
    return config.anthropicBaseUrl;
  }
  if (providerFamily === 'gemini') {
    return config.geminiBaseUrl;
  }
  return providerFamily;
}

function hasOpaqueReasoningState(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }> | StandardResponse['output'][number]
): boolean {
  if (item.type !== 'reasoning') {
    return false;
  }
  if (item.encrypted_content) {
    return true;
  }
  return (item.reasoning_details || []).some((detail) => {
    if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) {
      return false;
    }
    const record = detail as Record<string, unknown>;
    return Boolean(
      record.data ||
      record.encrypted_content ||
      record.signature ||
      record.thoughtSignature ||
      record.thought_signature
    );
  });
}

function sanitizeReadableReasoningDetails(value: unknown[] | undefined): unknown[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const readableDetails: unknown[] = [];
  for (const detail of value) {
    if (typeof detail === 'string') {
      if (detail.trim()) {
        readableDetails.push(detail);
      }
      continue;
    }
    if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) {
      continue;
    }
    const record = detail as Record<string, unknown>;
    const {
      data: _data,
      encrypted_content: _encryptedContent,
      signature: _signature,
      thoughtSignature: _thoughtSignature,
      thought_signature: _thoughtSignatureSnake,
      ...readable
    } = record;
    const hasReadableText = ['text', 'summary', 'reasoning', 'thinking'].some(
      (key) => typeof readable[key] === 'string' && (readable[key] as string).trim()
    );
    if (hasReadableText) {
      readableDetails.push(readable);
    }
  }
  return readableDetails;
}

function sourceFormatForAdapter(sourceAdapterKey: string): string | undefined {
  if (sourceAdapterKey === 'openai_responses') {
    return OPENAI_RESPONSES_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'anthropic_messages') {
    return ANTHROPIC_CLAUDE_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'gemini_generate' || sourceAdapterKey === 'gemini_stream') {
    return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'gemini_interactions') {
    return GEMINI_INTERACTIONS_REASONING_FORMAT;
  }
  return undefined;
}

function wrapRecordField(
  record: Record<string, unknown>,
  field: string,
  format: string,
  origin: ReasoningStateOrigin,
  options: { id?: string; kind: 'signature' | 'encrypted' }
): boolean {
  const value = typeof record[field] === 'string' ? record[field] as string : '';
  if (!value || decodeReasoningTransportEnvelope(value)) {
    return false;
  }
  record[field] = encodeReasoningTransportEnvelope(
    format,
    value,
    options.id,
    options.kind,
    origin
  );
  return true;
}

function encodeSseChunk(event: string | undefined, payload: unknown): string {
  return encodeRawSseChunk(event, JSON.stringify(payload));
}

function encodeRawSseChunk(event: string | undefined, data: string): string {
  return `${event ? `event: ${event}\n` : ''}${data
    .split('\n')
    .map((line) => `data: ${line}`)
    .join('\n')}\n\n`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
