import type { OpenAIChatProviderThinkingAdapter } from './types';

export const minimaxOpenAIChatProviderThinkingAdapter: OpenAIChatProviderThinkingAdapter = {
  key: 'minimax',
  matches(input) {
    return matchesMinimaxHost(input.providerConfig.baseurl);
  },
  rewriteRequest(input) {
    input.body.reasoning_split = true;
    delete input.body.interleaved_thinking;
    delete input.body.interleavedThinking;
    delete input.body.thinking;
    delete input.body.output_config;
    delete input.body.reasoning_effort;
  }
};

function matchesMinimaxHost(baseUrl: string | undefined): boolean {
  const host = normalizeMinimaxBaseUrlHost(baseUrl);
  return Boolean(
    host &&
      (
        host === 'minimax.io' ||
        host.endsWith('.minimax.io') ||
        host === 'minimax.chat' ||
        host.endsWith('.minimax.chat') ||
        host === 'minimaxi.com' ||
        host.endsWith('.minimaxi.com')
      )
  );
}

function normalizeMinimaxBaseUrlHost(baseUrl: string | undefined): string | undefined {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = parseMinimaxUrlHost(trimmed) || parseMinimaxUrlHost(`https://${trimmed}`);
  return parsed?.replace(/\.$/, '').toLowerCase();
}

function parseMinimaxUrlHost(value: string): string | undefined {
  try {
    return new URL(value).hostname || undefined;
  } catch {
    return undefined;
  }
}
