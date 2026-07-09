import type {
  OpenAIChatProviderThinkingAdapter,
  OpenAIChatProviderThinkingRewriteInput
} from './types';

type ZhipuReasoningEffort = 'max' | 'xhigh' | 'high' | 'medium' | 'low' | 'minimal' | 'none';
type ZhipuThinkingType = 'enabled' | 'disabled';

export const zhipuOpenAIChatProviderThinkingAdapter: OpenAIChatProviderThinkingAdapter = {
  key: 'zhipu',
  matches(input) {
    return matchesZhipuHost(input.providerConfig.baseurl);
  },
  rewriteRequest(input) {
    delete input.body.reasoning_split;
    delete input.body.interleaved_thinking;
    delete input.body.interleavedThinking;
    input.body.messages = stripZhipuMessageReasoningFields(input.body.messages);

    const thinkingType = readZhipuThinkingType(input.standardRequest?.thinking ?? input.body.thinking);
    const effort = readZhipuReasoningEffort(input);

    if (thinkingType === 'disabled') {
      input.body.thinking = { type: 'disabled' };
      delete input.body.reasoning_effort;
      stripZhipuOutputConfigEffort(input.body);
      return;
    }

    if (thinkingType || effort) {
      input.body.thinking = { type: thinkingType || 'enabled' };
    }

    if (effort) {
      input.body.reasoning_effort = effort;
      stripZhipuOutputConfigEffort(input.body);
    }
  }
};

function matchesZhipuHost(baseUrl: string | undefined): boolean {
  const host = normalizeZhipuBaseUrlHost(baseUrl);
  return Boolean(
    host &&
      (
        host === 'z.ai' ||
        host.endsWith('.z.ai') ||
        host === 'bigmodel.cn' ||
        host.endsWith('.bigmodel.cn') ||
        host === 'zhipuai.cn' ||
        host.endsWith('.zhipuai.cn')
      )
  );
}

function readZhipuThinkingType(value: unknown): ZhipuThinkingType | undefined {
  if (typeof value === 'boolean') {
    return value ? 'enabled' : 'disabled';
  }

  if (typeof value === 'string') {
    return normalizeZhipuThinkingType(value);
  }

  if (!isZhipuRecord(value)) {
    return undefined;
  }

  return normalizeZhipuThinkingType(value.type);
}

function normalizeZhipuThinkingType(value: unknown): ZhipuThinkingType | undefined {
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

function readZhipuReasoningEffort(input: OpenAIChatProviderThinkingRewriteInput): ZhipuReasoningEffort | undefined {
  const standardRequest = input.standardRequest;
  return (
    normalizeZhipuReasoningEffort(input.body.reasoning_effort) ||
    normalizeZhipuReasoningEffort(readZhipuNestedValue(standardRequest?.reasoning, 'effort')) ||
    normalizeZhipuReasoningEffort(readZhipuNestedValue(standardRequest?.output_config, 'effort')) ||
    normalizeZhipuReasoningEffort(readZhipuNestedValue(input.body.reasoning, 'effort')) ||
    normalizeZhipuReasoningEffort(readZhipuNestedValue(input.body.output_config, 'effort'))
  );
}

function normalizeZhipuReasoningEffort(value: unknown): ZhipuReasoningEffort | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase().replace(/[-_\s]+/g, '');
  if (normalized === 'minimal') {
    return 'minimal';
  }
  if (['max', 'xhigh', 'high', 'medium', 'low', 'none'].includes(normalized)) {
    return normalized as ZhipuReasoningEffort;
  }
  return undefined;
}

function readZhipuNestedValue(value: unknown, key: string): unknown {
  return isZhipuRecord(value) ? value[key] : undefined;
}

function stripZhipuMessageReasoningFields(messages: unknown): unknown {
  if (!Array.isArray(messages)) {
    return messages;
  }

  let changed = false;
  const nextMessages = messages.map((message) => {
    if (!isZhipuRecord(message)) {
      return message;
    }

    if (
      !Object.prototype.hasOwnProperty.call(message, 'reasoning_content') &&
      !Object.prototype.hasOwnProperty.call(message, 'reasoning_details') &&
      !Object.prototype.hasOwnProperty.call(message, 'reasoning') &&
      !Object.prototype.hasOwnProperty.call(message, 'thinking')
    ) {
      return message;
    }

    changed = true;
    const nextMessage = { ...message };
    delete nextMessage.reasoning_content;
    delete nextMessage.reasoning_details;
    delete nextMessage.reasoning;
    delete nextMessage.thinking;
    return nextMessage;
  });

  return changed ? nextMessages.filter((message) => !isEmptyZhipuAssistantMessage(message)) : messages;
}

function isEmptyZhipuAssistantMessage(message: unknown): boolean {
  if (!isZhipuRecord(message) || message.role !== 'assistant') {
    return false;
  }

  if (
    Object.prototype.hasOwnProperty.call(message, 'tool_calls') ||
    Object.prototype.hasOwnProperty.call(message, 'function_call')
  ) {
    return false;
  }

  const content = message.content;
  return content === undefined || content === '' || (Array.isArray(content) && content.length === 0);
}

function stripZhipuOutputConfigEffort(body: Record<string, unknown>): void {
  const outputConfig = body.output_config;
  if (!isZhipuRecord(outputConfig) || !Object.prototype.hasOwnProperty.call(outputConfig, 'effort')) {
    return;
  }

  const nextOutputConfig = { ...outputConfig };
  delete nextOutputConfig.effort;
  if (Object.keys(nextOutputConfig).length === 0) {
    delete body.output_config;
    return;
  }

  body.output_config = nextOutputConfig;
}

function normalizeZhipuBaseUrlHost(baseUrl: string | undefined): string | undefined {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = parseZhipuUrlHost(trimmed) || parseZhipuUrlHost(`https://${trimmed}`);
  return parsed?.replace(/\.$/, '').toLowerCase();
}

function parseZhipuUrlHost(value: string): string | undefined {
  try {
    return new URL(value).hostname || undefined;
  } catch {
    return undefined;
  }
}

function isZhipuRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
