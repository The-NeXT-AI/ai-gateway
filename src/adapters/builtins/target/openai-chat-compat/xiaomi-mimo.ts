import type {
  OpenAIChatProviderThinkingAdapter,
  OpenAIChatProviderThinkingRewriteInput
} from './types';

type XiaomiMimoThinkingType = 'enabled' | 'disabled';

export const xiaomiMimoOpenAIChatProviderThinkingAdapter: OpenAIChatProviderThinkingAdapter = {
  key: 'xiaomi-mimo',
  matches(input) {
    return matchesXiaomiMimoHost(input.providerConfig.baseurl);
  },
  rewriteRequest(input) {
    const thinkingType = readXiaomiMimoThinkingType(input.standardRequest?.thinking ?? input.body.thinking);
    const hasReasoningRequest = hasXiaomiMimoReasoningRequest(input);

    delete input.body.reasoning_split;
    delete input.body.interleaved_thinking;
    delete input.body.interleavedThinking;
    delete input.body.reasoning_effort;
    delete input.body.output_config;
    input.body.messages = rewriteXiaomiMimoMessageReasoningFields(input.body.messages);

    if (thinkingType === 'disabled') {
      input.body.thinking = { type: 'disabled' };
      return;
    }

    if (thinkingType || hasReasoningRequest) {
      input.body.thinking = { type: 'enabled' };
    }
  }
};

function matchesXiaomiMimoHost(baseUrl: string | undefined): boolean {
  const host = normalizeXiaomiMimoBaseUrlHost(baseUrl);
  return Boolean(
    host &&
      (
        host === 'xiaomimimo.com' ||
        host.endsWith('.xiaomimimo.com') ||
        host === 'mimo.mi.com' ||
        host.endsWith('.mimo.mi.com') ||
        host === 'mimo.xiaomi.com' ||
        host.endsWith('.mimo.xiaomi.com')
      )
  );
}

function readXiaomiMimoThinkingType(value: unknown): XiaomiMimoThinkingType | undefined {
  if (typeof value === 'boolean') {
    return value ? 'enabled' : 'disabled';
  }

  if (typeof value === 'string') {
    return normalizeXiaomiMimoThinkingType(value);
  }

  if (!isXiaomiMimoRecord(value)) {
    return undefined;
  }

  return normalizeXiaomiMimoThinkingType(value.type);
}

function normalizeXiaomiMimoThinkingType(value: unknown): XiaomiMimoThinkingType | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (['enabled', 'enable', 'on', 'true'].includes(normalized)) {
    return 'enabled';
  }
  if (['disabled', 'disable', 'off', 'false'].includes(normalized)) {
    return 'disabled';
  }
  return undefined;
}

function hasXiaomiMimoReasoningRequest(input: OpenAIChatProviderThinkingRewriteInput): boolean {
  const standardRequest = input.standardRequest;
  return (
    input.body.reasoning_effort !== undefined ||
    readXiaomiMimoNestedValue(standardRequest?.reasoning, 'effort') !== undefined ||
    readXiaomiMimoNestedValue(standardRequest?.output_config, 'effort') !== undefined ||
    readXiaomiMimoNestedValue(input.body.reasoning, 'effort') !== undefined ||
    readXiaomiMimoNestedValue(input.body.output_config, 'effort') !== undefined
  );
}

function readXiaomiMimoNestedValue(value: unknown, key: string): unknown {
  return isXiaomiMimoRecord(value) ? value[key] : undefined;
}

function rewriteXiaomiMimoMessageReasoningFields(messages: unknown): unknown {
  if (!Array.isArray(messages)) {
    return messages;
  }

  let changed = false;
  const nextMessages = messages.map((message) => {
    if (
      !isXiaomiMimoRecord(message) ||
      (
        !Object.prototype.hasOwnProperty.call(message, 'reasoning_details') &&
        !Object.prototype.hasOwnProperty.call(message, 'reasoning') &&
        !Object.prototype.hasOwnProperty.call(message, 'thinking')
      )
    ) {
      return message;
    }

    changed = true;
    const nextMessage = { ...message };
    delete nextMessage.reasoning_details;
    delete nextMessage.reasoning;
    delete nextMessage.thinking;
    return nextMessage;
  });

  return changed ? nextMessages.filter((message) => !isEmptyXiaomiMimoAssistantMessage(message)) : messages;
}

function isEmptyXiaomiMimoAssistantMessage(message: unknown): boolean {
  if (!isXiaomiMimoRecord(message) || message.role !== 'assistant') {
    return false;
  }

  if (
    Object.prototype.hasOwnProperty.call(message, 'tool_calls') ||
    Object.prototype.hasOwnProperty.call(message, 'function_call') ||
    Object.prototype.hasOwnProperty.call(message, 'reasoning_content')
  ) {
    return false;
  }

  const content = message.content;
  return content === undefined || content === '' || (Array.isArray(content) && content.length === 0);
}

function normalizeXiaomiMimoBaseUrlHost(baseUrl: string | undefined): string | undefined {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = parseXiaomiMimoUrlHost(trimmed) || parseXiaomiMimoUrlHost(`https://${trimmed}`);
  return parsed?.replace(/\.$/, '').toLowerCase();
}

function parseXiaomiMimoUrlHost(value: string): string | undefined {
  try {
    return new URL(value).hostname || undefined;
  } catch {
    return undefined;
  }
}

function isXiaomiMimoRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
