export const OPENAI_RESPONSES_REASONING_FORMAT = 'openai-responses-v1';

const OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX = 'ccr-openai-responses-reasoning-v1:';

export interface OpenAIResponsesReasoningEnvelope {
  id: string;
  encryptedContent: string;
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
