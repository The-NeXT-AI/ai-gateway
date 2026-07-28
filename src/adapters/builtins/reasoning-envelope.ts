export const OPENAI_RESPONSES_REASONING_FORMAT = 'openai-responses-v1';
export const ANTHROPIC_CLAUDE_REASONING_FORMAT = 'anthropic-claude-v1';
export const GEMINI_GENERATE_CONTENT_REASONING_FORMAT = 'google-generate-content-v1';
export const GEMINI_INTERACTIONS_REASONING_FORMAT = 'google-interactions-v1';

const OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX = 'ccr-openai-responses-reasoning-v1:';
const REASONING_TRANSPORT_ENVELOPE_PREFIX = 'ccr-reasoning-transport-v1:';

export interface OpenAIResponsesReasoningEnvelope {
  id: string;
  encryptedContent: string;
}

export interface ReasoningTransportEnvelope {
  format: string;
  data: string;
  id?: string;
  kind?: 'signature' | 'encrypted';
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
  kind?: ReasoningTransportEnvelope['kind']
): string {
  const normalizedFormat = format.trim();
  const normalizedId = id?.trim();
  if (!normalizedFormat || !data) {
    return data;
  }

  if (normalizedFormat === OPENAI_RESPONSES_REASONING_FORMAT && normalizedId) {
    return encodeOpenAIResponsesReasoningEnvelope(normalizedId, data);
  }

  const payload = Buffer.from(
    JSON.stringify({
      format: normalizedFormat,
      data,
      ...(normalizedId ? { id: normalizedId } : {}),
      ...(kind ? { kind } : {})
    }),
    'utf8'
  ).toString('base64url');

  return `${REASONING_TRANSPORT_ENVELOPE_PREFIX}${payload}`;
}

export function decodeReasoningTransportEnvelope(
  value: string
): ReasoningTransportEnvelope | undefined {
  const openAIEnvelope = decodeOpenAIResponsesReasoningEnvelope(value);
  if (openAIEnvelope) {
    return {
      format: OPENAI_RESPONSES_REASONING_FORMAT,
      id: openAIEnvelope.id,
      data: openAIEnvelope.encryptedContent
    };
  }

  if (!value.startsWith(REASONING_TRANSPORT_ENVELOPE_PREFIX)) {
    return undefined;
  }

  const payload = value.slice(REASONING_TRANSPORT_ENVELOPE_PREFIX.length);
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
    if (!format || !data) {
      return undefined;
    }

    return {
      format,
      data,
      ...(id ? { id } : {}),
      ...(kind ? { kind } : {})
    };
  } catch {
    return undefined;
  }
}
