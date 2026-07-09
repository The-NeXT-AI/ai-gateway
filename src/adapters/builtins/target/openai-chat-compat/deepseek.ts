import type {
  OpenAIChatProviderThinkingAdapter,
  OpenAIChatProviderThinkingRewriteInput
} from './types';

type DeepSeekReasoningEffort = 'high' | 'max';
type DeepSeekThinkingType = 'enabled' | 'disabled';

export const deepSeekOpenAIChatProviderThinkingAdapter: OpenAIChatProviderThinkingAdapter = {
  key: 'deepseek',
  matches(input) {
    return matchesDeepSeekHost(input.providerConfig.baseurl);
  },
  rewriteRequest(input) {
    delete input.body.reasoning_split;
    delete input.body.interleaved_thinking;
    delete input.body.interleavedThinking;

    const thinkingType = readDeepSeekThinkingType(input.standardRequest?.thinking ?? input.body.thinking);
    const effort = readDeepSeekReasoningEffort(input);

    if (thinkingType === 'disabled') {
      input.body.thinking = { type: 'disabled' };
      delete input.body.reasoning_effort;
      stripDeepSeekOutputConfigEffort(input.body);
      return;
    }

    if (thinkingType || effort) {
      input.body.thinking = { type: thinkingType || 'enabled' };
    }

    if (effort) {
      input.body.reasoning_effort = effort;
      stripDeepSeekOutputConfigEffort(input.body);
    }
  }
};

function matchesDeepSeekHost(baseUrl: string | undefined): boolean {
  const host = normalizeDeepSeekBaseUrlHost(baseUrl);
  return Boolean(host && (host === 'deepseek.com' || host.endsWith('.deepseek.com')));
}

function readDeepSeekThinkingType(value: unknown): DeepSeekThinkingType | undefined {
  if (typeof value === 'boolean') {
    return value ? 'enabled' : 'disabled';
  }

  if (typeof value === 'string') {
    return normalizeDeepSeekThinkingType(value);
  }

  if (!isDeepSeekRecord(value)) {
    return undefined;
  }

  return normalizeDeepSeekThinkingType(value.type);
}

function normalizeDeepSeekThinkingType(value: unknown): DeepSeekThinkingType | undefined {
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

function readDeepSeekReasoningEffort(input: OpenAIChatProviderThinkingRewriteInput): DeepSeekReasoningEffort | undefined {
  const standardRequest = input.standardRequest;
  return (
    normalizeDeepSeekReasoningEffort(input.body.reasoning_effort) ||
    normalizeDeepSeekReasoningEffort(readDeepSeekNestedValue(standardRequest?.reasoning, 'effort')) ||
    normalizeDeepSeekReasoningEffort(readDeepSeekNestedValue(standardRequest?.output_config, 'effort')) ||
    normalizeDeepSeekReasoningEffort(readDeepSeekNestedValue(input.body.reasoning, 'effort')) ||
    normalizeDeepSeekReasoningEffort(readDeepSeekNestedValue(input.body.output_config, 'effort'))
  );
}

function normalizeDeepSeekReasoningEffort(value: unknown): DeepSeekReasoningEffort | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase().replace(/[-_\s]+/g, '');
  if (normalized === 'max' || normalized === 'xhigh') {
    return 'max';
  }
  if (normalized === 'high' || normalized === 'medium' || normalized === 'low') {
    return 'high';
  }
  return undefined;
}

function readDeepSeekNestedValue(value: unknown, key: string): unknown {
  return isDeepSeekRecord(value) ? value[key] : undefined;
}

function stripDeepSeekOutputConfigEffort(body: Record<string, unknown>): void {
  const outputConfig = body.output_config;
  if (!isDeepSeekRecord(outputConfig) || !Object.prototype.hasOwnProperty.call(outputConfig, 'effort')) {
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

function normalizeDeepSeekBaseUrlHost(baseUrl: string | undefined): string | undefined {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = parseDeepSeekUrlHost(trimmed) || parseDeepSeekUrlHost(`https://${trimmed}`);
  return parsed?.replace(/\.$/, '').toLowerCase();
}

function parseDeepSeekUrlHost(value: string): string | undefined {
  try {
    return new URL(value).hostname || undefined;
  } catch {
    return undefined;
  }
}

function isDeepSeekRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
