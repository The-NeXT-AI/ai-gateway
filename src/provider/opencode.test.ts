import { describe, expect, it } from 'vitest';
import type { ProviderPlugin, ProviderPluginRequestInput } from '../types';
import { createOpenCodeProviderPlugin } from './opencode';

function createMockInput(overrides?: Partial<ProviderPluginRequestInput>): ProviderPluginRequestInput {
  return {
    config: {} as any,
    request: {} as any,
    sourceProvider: 'openai',
    sourceAdapterKey: 'openai-chat-completions',
    targetProvider: 'openai',
    targetProviderConfig: {} as any,
    model: 'gpt-4',
    upstreamRequest: {
      url: 'https://opencode.ai/v1/chat/completions',
      headers: {},
      body: {
        model: 'gpt-4',
        messages: [],
        reasoning_split: true,
      },
    },
    ...overrides,
  } as any;
}

describe('createOpenCodeProviderPlugin', () => {
  const plugin: ProviderPlugin = createOpenCodeProviderPlugin();

  it('has the correct key and provider', () => {
    expect(plugin.key).toBe('builtin:opencode');
    expect(plugin.provider).toBe('openai');
  });

  it('has transformRequest defined', () => {
    expect(plugin.transformRequest).toBeDefined();
  });
});

describe('transformRequest', () => {
  const plugin: ProviderPlugin = createOpenCodeProviderPlugin();

  it('strips reasoning_split from the body', async () => {
    const input = createMockInput();
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.reasoning_split).toBeUndefined();
    }
  });

  it('strips reasoning_effort from the body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          reasoning_effort: 'high',
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.reasoning_effort).toBeUndefined();
    }
  });

  it('strips reasoning from the body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          reasoning: { type: 'thinking' },
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.reasoning).toBeUndefined();
    }
  });

  it('strips thinking from the body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
        headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          thinking: { type: 'enabled', budget_tokens: 2048 },
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.thinking).toBeUndefined();
    }
  });

  it('strips output_config from the body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
        headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          output_config: { structured_outputs: {} },
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.output_config).toBeUndefined();
    }
  });

  it('strips cache_control from the top-level body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          cache_control: { type: 'ephemeral' },
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.cache_control).toBeUndefined();
    }
  });

  it('strips cache_control from messages', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [
            { role: 'user', content: 'hello', cache_control: { type: 'ephemeral' } },
            { role: 'assistant', content: [{ type: 'text', text: 'hi', cache_control: { type: 'ephemeral' } }] },
          ],
          reasoning_split: true,
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      const messages = body.messages as Record<string, unknown>[];
      expect(messages[0].cache_control).toBeUndefined();
      const content = messages[1].content as Record<string, unknown>[];
      expect(content[0].cache_control).toBeUndefined();
    }
  });

  it('preserves other fields in the body', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'hello' }],
          temperature: 0.7,
          max_tokens: 100,
          reasoning_split: true,
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.model).toBe('gpt-4');
      expect(body.temperature).toBe(0.7);
      expect(body.max_tokens).toBe(100);
      expect(body.reasoning_split).toBeUndefined();
    }
  });

  it('returns upstreamRequest unchanged when body is missing', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: undefined,
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.body).toBeUndefined();
    }
  });

  it('does not modify non-opencode requests', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://api.openai.com/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          reasoning_split: true,
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.reasoning_split).toBe(true);
    }
  });

  it('does not modify non-openai target provider requests', async () => {
    const input = createMockInput({
      targetProvider: 'anthropic',
      upstreamRequest: {
        url: 'https://opencode.ai/v1/chat/completions',
          headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          reasoning_split: true,
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      const body = result.value.body as Record<string, unknown>;
      expect(body.reasoning_split).toBe(true);
    }
  });

  it('does not modify non-chat-completions requests', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'https://opencode.ai/v1/models',
        headers: {},
        body: undefined,
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe(input.upstreamRequest);
    }
  });

  it('does not mutate the original upstreamRequest', async () => {
    const originalBody = {
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'hi' }],
      reasoning_split: true,
    };
    const upstreamRequest = {
      url: 'https://opencode.ai/v1/chat/completions',
      headers: {},
      body: originalBody,
    };
    const input: any = createMockInput({ upstreamRequest });
    await plugin.transformRequest!(input);
    expect(originalBody.reasoning_split).toBe(true);
  });

  it('handles malformed URL gracefully', async () => {
    const input = createMockInput({
      upstreamRequest: {
        url: 'not-a-valid-url-that-contains-opencode.ai-in-it',
        headers: {},
        body: {
          model: 'gpt-4',
          messages: [],
          reasoning_split: true,
        },
      },
    });
    const result = await plugin.transformRequest!(input);
    expect(result.ok).toBe(true);
  });
});
