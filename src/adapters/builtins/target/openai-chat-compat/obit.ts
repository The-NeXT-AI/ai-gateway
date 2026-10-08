import type { OpenAIChatProviderThinkingAdapter } from './types';

const obitReasoningEfforts = new Set(['none', 'minimal', 'low', 'medium', 'high', 'xhigh']);

// Obit's OpenAI-compatible relay consumes reasoning_effort. It does not consume
// Anthropic thinking/output_config, and the worker owns the effort-to-template map.
export const obitOpenAIChatProviderThinkingAdapter: OpenAIChatProviderThinkingAdapter = {
  key: 'obit',
  useGenericRequestOptions: true,
  matches(input) {
    try {
      return new URL(input.providerConfig.baseurl || '').hostname.toLowerCase().replace(/\.$/, '') === 'relay.obitmc.com';
    } catch {
      return false;
    }
  },
  rewriteRequest(input) {
    const thinking = input.standardRequest?.thinking ?? input.body.thinking;
    const thinkingDisabled = isRecord(thinking) && thinking.type === 'disabled';

    // Explicitly disabling thinking wins even when the caller also sends effort.
    // Requests without controls retain the provider's current default.
    if (thinkingDisabled) {
      input.body.reasoning_effort = 'none';
    } else if (input.body.reasoning_effort === undefined) {
      const effort = readEffort(input.standardRequest?.reasoning) ??
        readEffort(input.standardRequest?.output_config) ??
        readEffort(input.body.reasoning) ?? readEffort(input.body.output_config);
      if (effort !== undefined) input.body.reasoning_effort = effort;
    }

    delete input.body.thinking;
    if (isRecord(input.body.output_config) && 'effort' in input.body.output_config) {
      const outputConfig = { ...input.body.output_config };
      delete outputConfig.effort;
      if (Object.keys(outputConfig).length) input.body.output_config = outputConfig;
      else delete input.body.output_config;
    }
  }
};

function readEffort(value: unknown): string | undefined {
  if (!isRecord(value) || typeof value.effort !== 'string') return undefined;
  return obitReasoningEfforts.has(value.effort) ? value.effort : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
