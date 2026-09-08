import { describe, expect, it } from 'vitest';
import {
  buildAnthropicHeaders,
  buildGeminiUrl,
  buildOpenAIHeaders,
  buildOpenAIPassthroughHeaders,
  normalizeOpenAIResponsesUsage
} from './common';

describe('buildOpenAIHeaders', () => {
  it('uses x-api-key when authorization header is missing', () => {
    const result = buildOpenAIHeaders(
      {
        'x-api-key': 'x-api-key-token'
      } as never,
      {
        auth: {
          enabled: false,
          mode: 'trusted_header'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.authorization).toBe('Bearer x-api-key-token');
  });

  it('prefers authorization bearer over x-api-key in trusted header mode', () => {
    const result = buildOpenAIHeaders(
      {
        authorization: 'Bearer bearer-token',
        'x-api-key': 'x-api-key-token'
      } as never,
      {
        openaiApiKey: 'managed-key',
        auth: {
          enabled: true,
          mode: 'trusted_header'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.authorization).toBe('Bearer bearer-token');
  });

  it('prefers managed key in introspection mode', () => {
    const result = buildOpenAIHeaders(
      {
        authorization: 'Bearer bearer-token',
        'x-api-key': 'x-api-key-token'
      } as never,
      {
        openaiApiKey: 'managed-key',
        auth: {
          enabled: true,
          mode: 'http_introspection'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.authorization).toBe('Bearer managed-key');
  });

  it('prefers managed key in static API key mode', () => {
    const result = buildOpenAIHeaders(
      {
        authorization: 'Bearer gateway-client-key',
        'x-api-key': 'gateway-client-key'
      } as never,
      {
        openaiApiKey: 'managed-key',
        auth: {
          enabled: true,
          mode: 'static_api_key'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.authorization).toBe('Bearer managed-key');
  });

  it('can disable OPENAI_API_KEY fallback in introspection mode', () => {
    const previousOpenAIKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'env-managed-key';

    try {
      const result = buildOpenAIHeaders(
        {
          authorization: 'Bearer bearer-token'
        } as never,
        {
          auth: {
            enabled: true,
            mode: 'http_introspection'
          },
          allowEnvApiKeyFallback: false
        } as never
      );

      expect(result.ok).toBe(true);
      if (!result.ok) {
        return;
      }

      expect(result.value.authorization).toBe('Bearer bearer-token');
    } finally {
      if (previousOpenAIKey === undefined) {
        delete process.env.OPENAI_API_KEY;
      } else {
        process.env.OPENAI_API_KEY = previousOpenAIKey;
      }
    }
  });
});

describe('buildOpenAIPassthroughHeaders', () => {
  it('preserves selected Codex headers while replacing credentials and dropping gateway headers', () => {
    const result = buildOpenAIPassthroughHeaders(
      {
        accept: 'text/event-stream',
        authorization: 'Bearer client-key',
        cookie: 'session=secret',
        originator: 'Codex Desktop',
        'session-id': 'session-123',
        'thread-id': 'thread-123',
        'user-agent': 'Codex Desktop/0.153.4',
        'x-auth-sub': 'subject-123',
        'x-client-request-id': 'request-123',
        'x-codex-beta-features': 'remote_compaction_v2',
        'x-codex-refresh-token': 'refresh-secret',
        'x-codex-turn-metadata': '{"cwd":"/private/repo"}',
        'x-codex-window-id': 'window-123',
        'x-target-provider': 'openai-main'
      } as never,
      {
        openaiApiKey: 'managed-key',
        auth: {
          enabled: true,
          mode: 'static_api_key'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value).toMatchObject({
      accept: 'text/event-stream',
      authorization: 'Bearer managed-key',
      originator: 'Codex Desktop',
      'session-id': 'session-123',
      'thread-id': 'thread-123',
      'user-agent': 'Codex Desktop/0.153.4',
      'x-client-request-id': 'request-123',
      'x-codex-beta-features': 'remote_compaction_v2',
      'x-codex-turn-metadata': '{"cwd":"/private/repo"}',
      'x-codex-window-id': 'window-123'
    });
    expect(result.value.cookie).toBeUndefined();
    expect(result.value['x-auth-sub']).toBeUndefined();
    expect(result.value['x-codex-refresh-token']).toBeUndefined();
    expect(result.value['x-target-provider']).toBeUndefined();
  });
});

describe('buildAnthropicHeaders', () => {
  it('preserves incoming user-agent for Anthropic upstream requests', () => {
    const result = buildAnthropicHeaders(
      {
        'x-api-key': 'x-api-key-token',
        'anthropic-beta': 'claude-code-20250219',
        'user-agent': 'claude-cli/2.1.205 (external, cli)'
      } as never,
      {
        auth: {
          enabled: false,
          mode: 'trusted_header'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value['x-api-key']).toBe('x-api-key-token');
    expect(result.value['anthropic-beta']).toBe('claude-code-20250219');
    expect(result.value['user-agent']).toBe('claude-cli/2.1.205 (external, cli)');
  });

  it('does not add a user-agent when one is not provided', () => {
    const result = buildAnthropicHeaders(
      {
        'x-api-key': 'x-api-key-token'
      } as never,
      {
        auth: {
          enabled: false,
          mode: 'trusted_header'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value).not.toHaveProperty('user-agent');
  });

  it('emits authorization Bearer alongside x-api-key for Anthropic upstreams', () => {
    const result = buildAnthropicHeaders(
      {
        'x-api-key': 'x-api-key-token'
      } as never,
      {
        auth: {
          enabled: false,
          mode: 'trusted_header'
        }
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value['x-api-key']).toBe('x-api-key-token');
    // OpenAI-compatible-hosted Anthropic endpoints (e.g. Ollama Cloud) reject
    // x-api-key and require the Bearer form. Regression: this was missing and
    // caused 401 from Ollama on the anthropic_messages forward.
    expect(result.value['authorization']).toBe('Bearer x-api-key-token');
  });
});

describe('buildGeminiUrl', () => {
  it('drops non-Gemini query parameters when constructing upstream URLs', () => {
    const result = buildGeminiUrl(
      {
        url: '/v1/messages?beta=tools-2024-04-04&target_provider=gemini-main&alt=sse',
        headers: {}
      } as never,
      'gemini-2.5-pro',
      'generateContent',
      'v1beta',
      {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local'
      } as never
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value).toBe('https://mock.local/v1beta/models/gemini-2.5-pro:generateContent?alt=sse&key=sk-test');
  });
});

describe('normalizeOpenAIResponsesUsage', () => {
  it('preserves cache creation and server tool counters', () => {
    expect(
      normalizeOpenAIResponsesUsage({
        input_tokens: 10,
        output_tokens: 3,
        total_tokens: 13,
        input_tokens_details: {
          cached_tokens: 4
        },
        cache_creation_input_tokens: 2,
        server_tool_use: {
          web_search_requests: 1,
          web_fetch_requests: 0
        }
      })
    ).toEqual({
      input_tokens: 10,
      input_tokens_details: {
        cached_tokens: 4,
        cache_write_tokens: 2,
        cache_creation_tokens: 2
      },
      output_tokens: 3,
      output_tokens_details: {
        reasoning_tokens: 0
      },
      total_tokens: 13,
      server_tool_use: {
        web_search_requests: 1,
        web_fetch_requests: 0
      }
    });
  });

  it('normalizes GPT-5.6 cache write counters from input token details', () => {
    expect(
      normalizeOpenAIResponsesUsage({
        input_tokens: 10,
        output_tokens: 3,
        total_tokens: 13,
        input_tokens_details: {
          cached_tokens: 4,
          cache_write_tokens: 2
        }
      })
    ).toEqual({
      input_tokens: 10,
      input_tokens_details: {
        cached_tokens: 4,
        cache_write_tokens: 2,
        cache_creation_tokens: 2
      },
      output_tokens: 3,
      output_tokens_details: {
        reasoning_tokens: 0
      },
      total_tokens: 13
    });
  });
});
