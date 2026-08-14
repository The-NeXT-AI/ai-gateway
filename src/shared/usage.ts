import type { StandardUsage } from '../types';

/**
 * Anthropic and OpenAI/Gemini disagree on what the "input tokens" counter means.
 *
 * - Anthropic `messages`: `usage.input_tokens` EXCLUDES `cache_read_input_tokens`
 *   and `cache_creation_input_tokens`. Clients are expected to add them up.
 * - OpenAI (`prompt_tokens`) and Gemini (`promptTokenCount`): the counter INCLUDES
 *   the cached prefix, which is additionally broken out in
 *   `prompt_tokens_details.cached_tokens` / `cachedContentTokenCount`.
 *
 * `StandardUsage.input_tokens` is populated from whichever upstream answered, so the
 * flag records which convention the value follows. Anthropic-facing formatters use it
 * to emit a spec-conformant exclusive count instead of forwarding an inclusive one,
 * which downstream clients would otherwise add the cache counters to a second time.
 */
export function anthropicInputTokens(usage: StandardUsage): number | undefined {
  if (usage.input_tokens === undefined) {
    return undefined;
  }
  if (!usage.input_includes_cache_tokens) {
    return usage.input_tokens;
  }
  return subtractCacheTokens(usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens);
}

/**
 * Same conversion as {@link anthropicInputTokens} for call sites that track the counters
 * as loose locals (streaming relay state) rather than a {@link StandardUsage} object.
 */
export function toAnthropicInputTokens(
  inputTokens: number | undefined,
  cacheReadTokens: number | undefined,
  cacheWriteTokens: number | undefined,
  inputIncludesCacheTokens: boolean | undefined
): number | undefined {
  if (inputTokens === undefined) {
    return undefined;
  }
  if (!inputIncludesCacheTokens) {
    return inputTokens;
  }
  return subtractCacheTokens(inputTokens, cacheReadTokens, cacheWriteTokens);
}

function subtractCacheTokens(
  inputTokens: number,
  cacheReadTokens: number | undefined,
  cacheWriteTokens: number | undefined
): number {
  const cacheTokens = (cacheReadTokens ?? 0) + (cacheWriteTokens ?? 0);
  if (cacheTokens <= 0) {
    return inputTokens;
  }
  return Math.max(0, inputTokens - cacheTokens);
}
