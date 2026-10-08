import { describe, expect, it } from 'vitest';
import { parseAnthropicMessagesRequest } from '../../source/parsers';
import { openAIResponsesTargetAdapter } from '../openai-responses';
import { rewriteOpenAIChatCompatibleRequest } from '../shared';

const baseurl = 'https://relay.obitmc.com/v1';

describe('Obit reasoning controls', () => {
  it.each([
    [{ thinking: { type: 'disabled' } }, 'none'],
    [{ thinking: { type: 'disabled' }, output_config: { effort: 'high' } }, 'none'],
    [{ thinking: { type: 'adaptive', display: 'omitted' }, output_config: { effort: 'high' } }, 'high'],
    ...['none', 'minimal', 'low', 'medium', 'high', 'xhigh'].map((effort) => [
      { output_config: { effort } }, effort
    ])
  ])('preserves explicit Anthropic controls %j', (controls, effort) => {
    const body = convertAnthropic(controls as Record<string, unknown>);
    expect(body.reasoning_effort).toBe(effort);
    expect(body).not.toHaveProperty('thinking');
    expect(body).not.toHaveProperty('output_config');
  });

  it.each([{}, { thinking: { type: 'adaptive' } }, { thinking: { type: 'enabled', budget_tokens: 1024 } }])(
    'keeps absent effort at the existing default for %j', (controls) => {
      expect(convertAnthropic(controls)).not.toHaveProperty('reasoning_effort');
    }
  );

  it.each(['https://example.test/v1', 'https://relay.obitmc.com.example/v1', 'https://private.relay.obitmc.com/v1', 'invalid']) (
    'leaves unrelated provider defaults unchanged for %s', (otherBaseurl) => {
      expect(convertAnthropic({ thinking: { type: 'disabled' }, output_config: { effort: 'high' } }, otherBaseurl))
        .not.toHaveProperty('reasoning_effort');
    }
  );

  it('preserves direct OpenAI effort precedence and unrelated output configuration', () => {
    const body = rewriteOpenAIChatCompatibleRequest({
      reasoning_effort: 'xhigh', output_config: { effort: 'low', format: { type: 'json_schema' } }
    }, { type: 'openai_chat_completions', baseurl } as never) as Record<string, unknown>;
    expect(body.reasoning_effort).toBe('xhigh');
    expect(body.output_config).toEqual({ format: { type: 'json_schema' } });
  });

  it('honors the existing provider-level thinking-option opt-out', () => {
    const body = rewriteOpenAIChatCompatibleRequest({ thinking: { type: 'disabled' } }, {
      type: 'openai_chat_completions', baseurl, openaiChatThinkingOptions: 'disabled'
    } as never) as Record<string, unknown>;
    expect(body).not.toHaveProperty('thinking');
    expect(body).not.toHaveProperty('reasoning_effort');
  });

  it('keeps the generic reasoning-history filtering when no controls are sent', () => {
    const body = convertAnthropic({ messages: [
      { role: 'assistant', content: [
        { type: 'thinking', thinking: 'Prior reasoning.', signature: 'test-only' },
        { type: 'text', text: 'Prior answer.' }
      ] },
      { role: 'user', content: 'Continue.' }
    ] });
    expect(body.messages).toEqual([
      { role: 'assistant', content: 'Prior answer.' },
      { role: 'user', content: 'Continue.' }
    ]);
    expect(body).not.toHaveProperty('reasoning_effort');
  });
});

function convertAnthropic(controls: Record<string, unknown>, targetBaseurl = baseurl): Record<string, unknown> {
  const parsed = parseAnthropicMessagesRequest({
    model: 'unsloth/Qwen3.8-27B-GGUF', max_tokens: 64,
    messages: [{ role: 'user', content: 'Reply with OK.' }], ...controls
  });
  if (!parsed.ok) throw new Error(parsed.error);
  const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
    request: { headers: {} } as never, standardRequest: parsed.value,
    config: { openaiApiKey: 'test-only', openaiBaseUrl: targetBaseurl } as never,
    targetProviderConfig: { type: 'openai_chat_completions', baseurl: targetBaseurl } as never
  });
  if (!built.ok) throw new Error(built.error);
  return built.value.body as Record<string, unknown>;
}
