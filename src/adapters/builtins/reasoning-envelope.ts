import type { ReasoningStateOrigin } from '../../types';

export const OPENAI_RESPONSES_REASONING_FORMAT = 'openai-responses-v1';
export const ANTHROPIC_CLAUDE_REASONING_FORMAT = 'anthropic-claude-v1';
export const GEMINI_GENERATE_CONTENT_REASONING_FORMAT = 'google-generate-content-v1';
export const GEMINI_INTERACTIONS_REASONING_FORMAT = 'google-interactions-v1';

const OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX = 'ccr-openai-responses-reasoning-v1:';
const REASONING_TRANSPORT_ENVELOPE_PREFIX = 'ccr-reasoning-transport-v1:';
const REASONING_TRANSPORT_ENVELOPE_V2_PREFIX = 'ccr-reasoning-transport-v2:';
export const GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR = '__thought__';

export interface OpenAIResponsesReasoningEnvelope {
  id: string;
  encryptedContent: string;
}

export interface ReasoningTransportEnvelope {
  format: string;
  data: string;
  id?: string;
  kind?: 'signature' | 'encrypted';
  origin?: ReasoningStateOrigin;
}

export function encodeOpenAIResponsesReasoningEnvelope(id: string, encryptedContent: string): string {
  const normalizedId = id.trim();
  if (!normalizedId || !encryptedContent) {
    return encryptedContent;
  }

  const payload = Buffer.from(
    JSON.stringify({
      id: normalizedId,
      encrypted_content: encryptedContent
    }),
    'utf8'
  ).toString('base64url');

  return `${OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX}${payload}`;
}

export function decodeOpenAIResponsesReasoningEnvelope(
  value: string
): OpenAIResponsesReasoningEnvelope | undefined {
  if (!value.startsWith(OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX)) {
    return undefined;
  }

  const payload = value.slice(OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX.length);
  if (!payload) {
    return undefined;
  }

  try {
    const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as unknown;
    if (typeof decoded !== 'object' || decoded === null || Array.isArray(decoded)) {
      return undefined;
    }

    const record = decoded as Record<string, unknown>;
    const id = typeof record.id === 'string' ? record.id.trim() : '';
    const encryptedContent =
      typeof record.encrypted_content === 'string' ? record.encrypted_content : '';
    if (!id || !encryptedContent) {
      return undefined;
    }

    return {
      id,
      encryptedContent
    };
  } catch {
    return undefined;
  }
}

export function encodeReasoningTransportEnvelope(
  format: string,
  data: string,
  id?: string,
  kind?: ReasoningTransportEnvelope['kind'],
  origin?: ReasoningStateOrigin
): string {
  const normalizedFormat = format.trim();
  const normalizedId = id?.trim();
  if (!normalizedFormat || !data) {
    return data;
  }

  const normalizedOrigin = normalizeReasoningStateOrigin(origin);
  if (!normalizedOrigin && normalizedFormat === OPENAI_RESPONSES_REASONING_FORMAT && normalizedId) {
    return encodeOpenAIResponsesReasoningEnvelope(normalizedId, data);
  }

  const payload = Buffer.from(
    JSON.stringify({
      format: normalizedFormat,
      data,
      ...(normalizedId ? { id: normalizedId } : {}),
      ...(kind ? { kind } : {}),
      ...(normalizedOrigin ? { origin: normalizedOrigin } : {})
    }),
    'utf8'
  ).toString('base64url');

  return `${normalizedOrigin ? REASONING_TRANSPORT_ENVELOPE_V2_PREFIX : REASONING_TRANSPORT_ENVELOPE_PREFIX}${payload}`;
}

export function decodeReasoningTransportEnvelope(
  value: string
): ReasoningTransportEnvelope | undefined {
  const v2Envelope = decodeReasoningTransportEnvelopeWithPrefix(
    value,
    REASONING_TRANSPORT_ENVELOPE_V2_PREFIX,
    true
  );
  if (v2Envelope) {
    return v2Envelope;
  }

  const openAIEnvelope = decodeOpenAIResponsesReasoningEnvelope(value);
  if (openAIEnvelope) {
    return {
      format: OPENAI_RESPONSES_REASONING_FORMAT,
      id: openAIEnvelope.id,
      data: openAIEnvelope.encryptedContent
    };
  }

  return decodeReasoningTransportEnvelopeWithPrefix(
    value,
    REASONING_TRANSPORT_ENVELOPE_PREFIX,
    false
  );
}

export function appendGeminiThoughtSignatureToToolCallId(
  toolCallId: string,
  encodedSignature: string
): string {
  const envelope = decodeReasoningTransportEnvelope(encodedSignature);
  if (
    !toolCallId ||
    !envelope?.origin ||
    envelope.format !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT ||
    envelope.kind !== 'signature'
  ) {
    return toolCallId;
  }

  return `${toolCallId}${GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR}${encodedSignature}`;
}

export function decodeGeminiThoughtSignatureToolCallId(value: string):
  | {
      toolCallId: string;
      envelope: ReasoningTransportEnvelope;
      encodedSignature: string;
    }
  | undefined {
  const separatorIndex = value.lastIndexOf(GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR);
  if (separatorIndex <= 0) {
    return undefined;
  }

  const toolCallId = value.slice(0, separatorIndex);
  const encodedSignature = value.slice(
    separatorIndex + GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR.length
  );
  const envelope = decodeReasoningTransportEnvelope(encodedSignature);
  if (
    !toolCallId ||
    !envelope?.origin ||
    envelope.format !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT ||
    envelope.kind !== 'signature'
  ) {
    return undefined;
  }

  return { toolCallId, envelope, encodedSignature };
}

export function containsReasoningTransportCarrier(value: unknown): boolean {
  if (typeof value === 'string') {
    return Boolean(
      decodeReasoningTransportEnvelope(value) ||
      decodeGeminiThoughtSignatureToolCallId(value)
    );
  }

  if (Array.isArray(value)) {
    return value.some((item) => containsReasoningTransportCarrier(item));
  }

  if (typeof value !== 'object' || value === null) {
    return false;
  }

  return Object.values(value as Record<string, unknown>).some((item) =>
    containsReasoningTransportCarrier(item)
  );
}

function decodeReasoningTransportEnvelopeWithPrefix(
  value: string,
  prefix: string,
  requireOrigin: boolean
): ReasoningTransportEnvelope | undefined {
  if (!value.startsWith(prefix)) {
    return undefined;
  }

  const payload = value.slice(prefix.length);
  if (!payload) {
    return undefined;
  }

  try {
    const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as unknown;
    if (typeof decoded !== 'object' || decoded === null || Array.isArray(decoded)) {
      return undefined;
    }

    const record = decoded as Record<string, unknown>;
    const format = typeof record.format === 'string' ? record.format.trim() : '';
    const data = typeof record.data === 'string' ? record.data : '';
    const id = typeof record.id === 'string' ? record.id.trim() : '';
    const kind =
      record.kind === 'signature' || record.kind === 'encrypted'
        ? record.kind
        : undefined;
    const origin = normalizeReasoningStateOrigin(record.origin);
    if (!format || !data || (requireOrigin && !origin)) {
      return undefined;
    }

    return {
      format,
      data,
      ...(id ? { id } : {}),
      ...(kind ? { kind } : {}),
      ...(origin ? { origin } : {})
    };
  } catch {
    return undefined;
  }
}

function normalizeReasoningStateOrigin(value: unknown): ReasoningStateOrigin | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  const provider = typeof record.provider === 'string' ? record.provider.trim() : '';
  const endpoint = typeof record.endpoint === 'string' ? record.endpoint.trim() : '';
  const model = typeof record.model === 'string' ? record.model.trim() : '';
  if (!provider || !endpoint) {
    return undefined;
  }

  return {
    provider,
    endpoint,
    ...(model ? { model } : {})
  };
}
