import { describe, expect, it } from 'vitest';
import { formatAnthropicMessagesResponse } from '../adapters/builtins/source/formatters';
import { parseAnthropicToStandardResponse, parseOpenAIToStandardResponse } from '../adapters/builtins/target/shared';
import { anthropicInputTokens, toAnthropicInputTokens } from './usage';
import type { StandardResponse } from '../types';

function unwrap(result: { ok: true; value: StandardResponse } | { ok: false; error: string }): StandardResponse {
  if (!result.ok) {
    throw new Error(`expected ok, got: ${result.error}`);
  }
  return result.value;
}

describe('anthropicInputTokens', () => {
  it('passes an already-exclusive counter through untouched', () => {
    expect(
      anthropicInputTokens({ input_tokens: 48, cache_read_tokens: 8960 })
    ).toBe(48);
  });

  it('subtracts the cached prefix from an inclusive counter', () => {
    expect(
      anthropicInputTokens({
        input_tokens: 9008,
        cache_read_tokens: 8960,
        input_includes_cache_tokens: true
      })
    ).toBe(48);
  });

  it('subtracts cache writes as well', () => {
    expect(
      anthropicInputTokens({
        input_tokens: 1000,
        cache_read_tokens: 600,
        cache_write_tokens: 300,
        input_includes_cache_tokens: true
      })
    ).toBe(100);
  });

  it('never returns a negative count when an upstream over-reports cache', () => {
    expect(
      anthropicInputTokens({
        input_tokens: 100,
        cache_read_tokens: 500,
        input_includes_cache_tokens: true
      })
    ).toBe(0);
  });

  it('leaves an undefined counter undefined', () => {
    expect(anthropicInputTokens({ cache_read_tokens: 10 })).toBeUndefined();
    expect(toAnthropicInputTokens(undefined, 10, undefined, true)).toBeUndefined();
  });
});

describe('OpenAI -> Anthropic usage conversion', () => {
  it('does not let a downstream client double-count the cached prefix', () => {
    // sglang / vLLM / any OpenAI-compatible upstream: prompt_tokens INCLUDES cached_tokens.
    const standard = unwrap(parseOpenAIToStandardResponse({
      id: 'chatcmpl-1',
      model: 'some-model',
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
      usage: {
        prompt_tokens: 9008,
        completion_tokens: 3,
        total_tokens: 9011,
        prompt_tokens_details: { cached_tokens: 8960 }
      }
    }));

    const body = formatAnthropicMessagesResponse(standard);
    const usage = body.usage as Record<string, number>;

    // Anthropic semantics: input_tokens EXCLUDES cache_read_input_tokens, so a client
    // summing the two must land back on the real prompt size.
    expect(usage.input_tokens).toBe(48);
    expect(usage.cache_read_input_tokens).toBe(8960);
    expect(usage.input_tokens + usage.cache_read_input_tokens).toBe(9008);
  });
});

describe('Anthropic -> Anthropic usage passthrough', () => {
  it('round-trips an already-exclusive counter without shrinking it', () => {
    const standard = unwrap(parseAnthropicToStandardResponse({
      id: 'msg_1',
      model: 'claude-sonnet-4-5',
      content: [{ type: 'text', text: 'ok' }],
      stop_reason: 'end_turn',
      usage: {
        input_tokens: 48,
        output_tokens: 3,
        cache_read_input_tokens: 8960
      }
    }));

    const body = formatAnthropicMessagesResponse(standard);
    const usage = body.usage as Record<string, number>;

    expect(usage.input_tokens).toBe(48);
    expect(usage.cache_read_input_tokens).toBe(8960);
  });
});
