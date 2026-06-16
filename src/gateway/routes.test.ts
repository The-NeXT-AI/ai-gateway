import { createCipheriv, randomBytes } from 'node:crypto';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Fastify from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { closeBillingPublisher, initializeBillingPublisher } from '../billing';
import { createGatewayRuntime } from './runtime';
import { registerGatewayRoutes } from './routes';
import { registerGatewayIdempotencyHooks, resetGatewayIdempotencyForTests } from './idempotency';
import { resetProviderCircuitBreakerForTests } from './upstream-circuit-breaker';
import { resetProviderConcurrencyForTests } from './upstream-concurrency';
import { resetGatewayPrecheckStateForTests } from './precheck';
import { closeRawTraceManager, initializeRawTraceManager } from '../raw-trace';
import { closeCodexOauthStateStore, updateDistributedCredentialEncryption } from '../provider/plugins';
import type { GatewayConfig, ProviderConfig, ProviderPluginConfig, TargetAdapter } from '../types';

describe('gateway routes protocol conversion', () => {
  afterEach(async () => {
    updateDistributedCredentialEncryption(undefined);
    resetGatewayIdempotencyForTests();
    resetProviderCircuitBreakerForTests();
    resetProviderConcurrencyForTests();
    resetGatewayPrecheckStateForTests();
    await closeCodexOauthStateStore();
    await closeBillingPublisher();
    await closeRawTraceManager();
    vi.restoreAllMocks();
  });

  it('lists gateway models in OpenAI format by default', async () => {
    const config = createConfig(
      [
        createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5']),
        createProviderConfig('anthropic-main', 'anthropic_messages', ['claude-sonnet-4'])
      ],
      undefined,
      [
        {
          id: 'virtual-search',
          key: 'search',
          displayName: 'Search',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':search']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          tools: [],
          execution: {
            mode: 'decorate_only',
            maxTurns: 1,
            maxToolCalls: 0,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/v1/models'
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toMatchObject({
        object: 'list'
      });
      expect(body.data).toEqual([
        {
          id: 'openai-main/glm-5',
          object: 'model',
          created: 0,
          owned_by: 'openai-main'
        },
        {
          id: 'anthropic-main/claude-sonnet-4',
          object: 'model',
          created: 0,
          owned_by: 'anthropic-main'
        },
        {
          id: 'openai-main/glm-5:search',
          object: 'model',
          created: 0,
          owned_by: 'openai-main'
        },
        {
          id: 'anthropic-main/claude-sonnet-4:search',
          object: 'model',
          created: 0,
          owned_by: 'anthropic-main'
        }
      ]);
    } finally {
      await app.close();
    }
  });

  it('lists gateway models in Anthropic format when Anthropic headers are present', async () => {
    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('anthropic-main', 'anthropic_messages', ['claude-sonnet-4'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/v1/models',
        headers: {
          'anthropic-version': '2023-06-01',
          'x-api-key': 'test-key'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        data: [
          {
            created_at: '1970-01-01T00:00:00Z',
            display_name: 'claude-sonnet-4',
            id: 'anthropic-main/claude-sonnet-4',
            type: 'model'
          }
        ],
        first_id: 'anthropic-main/claude-sonnet-4',
        has_more: false,
        last_id: 'anthropic-main/claude-sonnet-4'
      });
    } finally {
      await app.close();
    }
  });

  it('allows the model list format to be selected explicitly', async () => {
    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/v1/models?format=anthropic'
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.object).toBeUndefined();
      expect(body.data[0]).toMatchObject({
        id: 'openai-main/glm-5',
        type: 'model'
      });
    } finally {
      await app.close();
    }
  });

  it('returns a single configured model in OpenAI format', async () => {
    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/v1/models/openai-main/glm-5'
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        id: 'openai-main/glm-5',
        object: 'model',
        created: 0,
        owned_by: 'openai-main'
      });
    } finally {
      await app.close();
    }
  });

  it('returns 404 for unknown single model lookups', async () => {
    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/v1/models/openai-main/missing'
      });

      expect(response.statusCode).toBe(404);
      expect(JSON.parse(response.body)).toEqual({
        error: {
          message: 'Model not found: openai-main/missing',
          type: 'invalid_request_error',
          code: 'model_not_found'
        }
      });
    } finally {
      await app.close();
    }
  });

  it('converts /v1/responses to chat/completions for openai_chat_completions targets', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_123',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'converted to chat'
              }
            }
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello from responses'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/chat/completions');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.messages).toEqual([{ role: 'user', content: 'hello from responses' }]);
      expect(upstreamBody.input).toBeUndefined();

      const responseBody = JSON.parse(response.body);
      expect(responseBody.object).toBe('response');
      expect(responseBody.output_text).toBe('converted to chat');
    } finally {
      await app.close();
    }
  });

  it('replays an idempotent converted JSON request without dispatching upstream again', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_idempotent',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'cached reply'
              }
            }
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.idempotency = {
      enabled: true,
      headerName: 'idempotency-key',
      ttlMs: 60000,
      maxEntries: 100,
      cacheErrorResponses: false
    };
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

      const request = {
      method: 'POST' as const,
      url: '/v1/responses',
      headers: {
        'content-type': 'application/json',
        'x-target-provider': 'openai-main',
        'idempotency-key': 'chat-retry-key'
      },
      payload: {
        model: 'glm-5',
        input: 'hello'
      }
    };

    try {
      const first = await app.inject(request);
      const second = await app.inject(request);

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(JSON.parse(second.body)).toMatchObject({
        output: [
          {
            type: 'message'
          }
        ]
      });
      expect(first.headers['x-gateway-idempotency-status']).toBe('stored');
      expect(second.headers['x-gateway-idempotency-status']).toBe('replayed');
    } finally {
      await app.close();
    }
  });

  it('rejects a second upstream request when provider concurrency is saturated', async () => {
    let resolveFetch!: (response: Response) => void;
    const fetchMock = vi.fn(async () => {
      return await new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.upstreamConcurrency = {
      enabled: true,
      maxInFlightPerProvider: 1,
      queueTimeoutMs: 1
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    const request = {
      method: 'POST' as const,
      url: '/v1/responses',
      headers: {
        'content-type': 'application/json',
        'x-target-provider': 'openai-main'
      },
      payload: {
        model: 'glm-5',
        input: 'hello'
      }
    };

    try {
      const first = app.inject(request);
      await waitForCondition(() => fetchMock.mock.calls.length === 1);
      const second = await app.inject(request);

      expect(second.statusCode).toBe(429);
      const secondBody = JSON.parse(second.body);
      expect(secondBody.error.message).toBe('All target providers failed.');
      expect(secondBody.error.attempts[0]).toMatchObject({
        stage: 'upstream_concurrency',
        status: 429,
        message: 'Provider upstream concurrency limit exceeded.'
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);

      resolveFetch(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_concurrency',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'first done'
                }
              }
            ],
            usage: {
              prompt_tokens: 10,
              completion_tokens: 5,
              total_tokens: 15
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
      const firstResponse = await first;
      expect(firstResponse.statusCode).toBe(200);
    } finally {
      await app.close();
    }
  });

  it('opens the upstream circuit breaker after a provider failure and rejects the next request', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({ error: { message: 'upstream unavailable' } }),
        {
          status: 500,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.upstreamCircuitBreaker = {
      enabled: true,
      failureThreshold: 1,
      cooldownMs: 60000,
      failureStatusCodes: [500]
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    const request = {
      method: 'POST' as const,
      url: '/v1/responses',
      headers: {
        'content-type': 'application/json',
        'x-target-provider': 'openai-main'
      },
      payload: {
        model: 'glm-5',
        input: 'hello'
      }
    };

    try {
      const first = await app.inject(request);
      const second = await app.inject(request);

      expect(first.statusCode).toBe(500);
      expect(second.statusCode).toBe(503);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const secondBody = JSON.parse(second.body);
      expect(secondBody.error.attempts[0]).toMatchObject({
        stage: 'upstream_circuit_open',
        status: 503,
        message: 'Provider upstream circuit breaker is open.',
        details: {
          provider: 'openai',
          providerName: 'openai-main',
          failureThreshold: 1,
          cooldownMs: 60000
        }
      });
    } finally {
      await app.close();
    }
  });

  it('retries configured upstream response statuses before falling back', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ error: { message: 'try again' } }), {
          status: 503,
          headers: {
            'content-type': 'application/json'
          }
        })
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_retry_status',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'retried successfully'
                }
              }
            ],
            usage: {
              prompt_tokens: 10,
              completion_tokens: 5,
              total_tokens: 15
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.upstreamRetry = {
      enabled: true,
      maxAttempts: 2,
      baseDelayMs: 0,
      maxDelayMs: 0,
      backoffMultiplier: 1,
      jitterMs: 0,
      retryStatusCodes: [503]
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(JSON.parse(response.body).output_text).toBe('retried successfully');
    } finally {
      await app.close();
    }
  });

  it('rejects requests before upstream dispatch when quota precheck fails', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.precheck = {
      enabled: true,
      rateLimit: {
        enabled: false,
        windowMs: 60000,
        maxRequests: 0,
        rpm: 0,
        rpd: 0,
        tpm: 0,
        tpd: 0,
        ipm: 0,
        limits: [],
        subject: 'identity',
        scope: 'global'
      },
      quota: {
        enabled: true,
        windowMs: 60000,
        maxTokens: 1,
        subject: 'global',
        scope: 'model'
      },
      budget: {
        enabled: false,
        windowMs: 86400000,
        maxCostUsd: 0,
        subject: 'identity',
        scope: 'global'
      },
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      },
      storage: {
        type: 'memory'
      }
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(429);
      expect(fetchMock).not.toHaveBeenCalled();
      expect(JSON.parse(response.body).error.code).toBe('quota_exceeded');
    } finally {
      await app.close();
    }
  });

  it('routes to a healthy provider when the first configured provider is down', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_healthy',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'healthy backup'
              }
            }
          ],
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const primary = createProviderConfig('openai-down', 'openai_chat_completions', ['glm-5']);
    primary.baseurl = 'https://down.example/v1';
    primary.health = {
      status: 'down',
      available: false
    };
    const backup = createProviderConfig('openai-backup', 'openai_chat_completions', ['glm-5']);
    backup.baseurl = 'https://backup.example/v1';
    backup.health = {
      status: 'healthy',
      available: true,
      latencyMs: 25
    };
    const config = createConfig([primary, backup]);
    config.healthAwareRouting = {
      enabled: true,
      skipUnavailable: true,
      unhealthyStatuses: ['down'],
      preferHealthy: true,
      preferLowerLatency: true
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'openai-down,openai-backup'
        },
        payload: {
          model: 'glm-5',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-backup');
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://backup.example/v1/chat/completions');
    } finally {
      await app.close();
    }
  });

  it('preserves adapter path versions when applying a named OpenAI provider base url override', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          'content-type': 'application/json'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-versioned', 'openai_chat_completions', ['glm-5']);
    provider.baseurl = 'https://vendor.example/api';
    const config = createConfig([provider]);
    const runtime = createGatewayRuntime(config);
    const targetAdapter: TargetAdapter = {
      provider: 'openai',
      buildRequestFromStandard() {
        return {
          ok: true,
          value: {
            url: 'https://adapter.example/v1/chat/completions?trace=1',
            headers: {
              'content-type': 'application/json',
              authorization: 'Bearer adapter-key'
            },
            body: {
              model: 'glm-5',
              messages: [{ role: 'user', content: 'hello' }]
            }
          }
        };
      },
      toStandardResponse() {
        return {
          ok: true,
          value: {
            id: 'resp_versioned_path',
            object: 'response',
            status: 'completed',
            model: 'glm-5',
            output_text: 'ok',
            output: [
              {
                id: 'msg_versioned_path',
                type: 'message',
                role: 'assistant',
                status: 'completed',
                content: [
                  {
                    type: 'output_text',
                    text: 'ok',
                    annotations: []
                  }
                ]
              }
            ],
            usage: {}
          }
        };
      }
    };
    runtime.targetAdapters.register(targetAdapter, { overwrite: true });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-versioned'
        },
        payload: {
          model: 'glm-5',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://vendor.example/api/v1/chat/completions?trace=1');
    } finally {
      await app.close();
    }
  });

  it('applies tenant gateway policy before upstream dispatch and falls back to an allowed provider', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_policy_allowed',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'allowed provider'
              }
            }
          ],
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const denied = createProviderConfig('openai-denied', 'openai_chat_completions', ['glm-5']);
    denied.baseurl = 'https://denied.example/v1';
    const allowed = createProviderConfig('openai-allowed', 'openai_chat_completions', ['glm-5']);
    allowed.baseurl = 'https://allowed.example/v1';
    const config = createConfig([denied, allowed]);
    config.auth.enabled = true;
    config.auth.required = false;
    config.policy = createPolicyConfig({
      enabled: true,
      byTenant: {
        'tenant-a': {
          ...createPolicyRuleConfig(),
          denyProviderNames: ['openai-denied']
        }
      }
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'openai-denied,openai-allowed',
          'x-auth-user-id': 'user-1',
          'x-auth-tenant-id': 'tenant-a'
        },
        payload: {
          model: 'glm-5',
          messages: [{ role: 'user', content: 'hello' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-allowed');
      expect(response.headers['x-gateway-fallback-used']).toBe('true');
      expect(response.headers['x-gateway-fallback-count']).toBe('1');
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://allowed.example/v1/chat/completions');
    } finally {
      await app.close();
    }
  });

  it('maps chat/completions reasoning_content into Responses reasoning output', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_reasoning_123',
          model: 'deepseek-r1',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                reasoning_content: '先分析问题。',
                content: '这是可见回答。'
              }
            }
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 8,
            total_tokens: 18
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['deepseek-r1'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'deepseek-r1',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.output_text).toBe('这是可见回答。');

      const reasoning = body.output.find((item: Record<string, unknown>) => item.type === 'reasoning');
      expect(reasoning).toMatchObject({
        type: 'reasoning',
        status: 'completed',
        summary: [],
        content: [
          {
            type: 'reasoning_text',
            text: '先分析问题。'
          }
        ]
      });
    } finally {
      await app.close();
    }
  });

  it('maps chat/completions reasoning_details into Responses reasoning output', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_reasoning_details_123',
          model: 'MiniMax-M2.7',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                reasoning_content: 'MiniMax thinking block',
                reasoning_details: [
                  {
                    type: 'reasoning.text',
                    text: 'MiniMax thinking block',
                    id: 'reasoning-text-1',
                    format: 'anthropic-claude-v1',
                    index: 0
                  }
                ],
                content: 'visible minimax answer'
              }
            }
          ],
          usage: {
            prompt_tokens: 9,
            completion_tokens: 7,
            total_tokens: 16
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['MiniMax-M2.7'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'MiniMax-M2.7',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.output_text).toBe('visible minimax answer');

      const reasoning = body.output.find((item: Record<string, unknown>) => item.type === 'reasoning');
      expect(reasoning).toMatchObject({
        id: 'reasoning-text-1',
        type: 'reasoning',
        status: 'completed',
        content: [
          {
            type: 'reasoning_text',
            text: 'MiniMax thinking block'
          }
        ]
      });
      expect(reasoning.content).toEqual([
        {
          type: 'reasoning_text',
          text: 'MiniMax thinking block'
        }
      ]);
    } finally {
      await app.close();
    }
  });

  it('streams /v1/responses incrementally when converting from chat/completions stream', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n',
        'data: {"id":"chatcmpl_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
        'data: {"id":"chatcmpl_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello world',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.stream).toBe(true);

      expect(response.body).toContain('"type":"response.created"');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"hello "');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"world"');
      expect(response.body).toContain('"type":"response.output_text.done","text":"hello world"');
      expect(response.body).toContain('"type":"response.completed"');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('stores the original upstream SSE body in raw trace wire_raw mode', async () => {
    const upstreamFrames = [
      'data: {"id":"chatcmpl_trace_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
      'data: {"id":"chatcmpl_trace_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"raw "}}]}\n\n',
      'data: {"id":"chatcmpl_trace_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"stream"}}]}\n\n',
      'data: {"id":"chatcmpl_trace_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
      'data: [DONE]\n\n',
    ];
    const syncedManifests: unknown[] = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === 'https://trace.example.com/sync') {
        syncedManifests.push(JSON.parse(String(init?.body || '{}')));
        return new Response(null, { status: 204 });
      }

      return createSseResponse(upstreamFrames);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const spoolDir = await mkdtemp(join(tmpdir(), 'gateway-raw-trace-'));
    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.rawTrace = {
      ...config.rawTrace,
      enabled: true,
      mode: 'wire_raw',
      spoolDir,
      sync: {
        ...config.rawTrace.sync,
        enabled: true,
        transport: 'http',
        endpoint: 'https://trace.example.com/sync'
      }
    };
    await initializeRawTraceManager(config.rawTrace);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime());
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
        },
        payload: {
          model: 'glm-5',
          input: 'hello world',
          stream: true,
        },
      });

      expect(response.statusCode).toBe(200);
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"raw "');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"stream"');

      const { bundleDir, manifest } = await waitForRawTraceManifest(spoolDir);
      await waitForCondition(() => syncedManifests.length === 1);
      await closeRawTraceManager();
      expect(manifest.parts.map((part: { partType: string }) => part.partType)).toContain('response_stream');
      expect(manifest.parts.map((part: { partType: string }) => part.partType)).toContain('upstream_response_metadata');
      expect((syncedManifests[0] as { parts: Array<{ storageBackend: string }> }).parts[0]?.storageBackend).toBe('local');

      const rawStream = await readFile(join(bundleDir, 'response_stream.txt'), 'utf8');
      expect(rawStream).toBe(upstreamFrames.join(''));
      expect(rawStream).toContain('data: [DONE]');
    } finally {
      await closeRawTraceManager();
      await app.close();
      await rm(spoolDir, { recursive: true, force: true });
    }
  });

  it('streams chat/completions reasoning_content as Responses reasoning_text events', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_reasoning_stream_1","object":"chat.completion.chunk","model":"deepseek-r1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_stream_1","object":"chat.completion.chunk","model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"think "}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_stream_1","object":"chat.completion.chunk","model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"first"}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_stream_1","object":"chat.completion.chunk","model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_stream_1","object":"chat.completion.chunk","model":"deepseek-r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['deepseek-r1'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'deepseek-r1',
          input: 'hello',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.body).toContain('"type":"response.reasoning_text.delta"');
      expect(response.body).toContain('"delta":"think "');
      expect(response.body).toContain('"delta":"first"');
      expect(response.body).toContain('"type":"response.reasoning_text.done"');
      expect(response.body).toContain('"text":"think first"');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"answer"');

      const completedLine = response.body
        .split('\n')
        .find((line) => line.startsWith('data: ') && line.includes('"type":"response.completed"'));
      expect(completedLine).toBeDefined();
      const completed = JSON.parse(String(completedLine).slice('data: '.length));
      expect(completed.response.output_text).toBe('answer');
      expect(completed.response.output[0]).toMatchObject({
        type: 'reasoning',
        content: [
          {
            type: 'reasoning_text',
            text: 'think first'
          }
        ]
      });
    } finally {
      await app.close();
    }
  });

  it('streams chat/completions reasoning_details as Responses reasoning_text events', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_reasoning_details_stream_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{"reasoning_content":"mini ","reasoning_details":[{"type":"reasoning.text","text":"mini ","id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_details_stream_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{"reasoning_content":"thinking","reasoning_details":[{"type":"reasoning.text","text":"thinking","id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_details_stream_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{"content":"visible"}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_details_stream_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['MiniMax-M2.7'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'MiniMax-M2.7',
          input: 'hello',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.body).toContain('"type":"response.reasoning_text.delta"');
      expect(response.body).toContain('"delta":"mini "');
      expect(response.body).toContain('"delta":"thinking"');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"visible"');

      const completedLine = response.body
        .split('\n')
        .find((line) => line.startsWith('data: ') && line.includes('"type":"response.completed"'));
      expect(completedLine).toBeDefined();
      const completed = JSON.parse(String(completedLine).slice('data: '.length));
      expect(completed.response.output_text).toBe('visible');
      expect(completed.response.output[0]).toMatchObject({
        type: 'reasoning',
        content: [
          {
            type: 'reasoning_text',
            text: 'mini thinking'
          }
        ]
      });
    } finally {
      await app.close();
    }
  });

  it('streams chat/completions reasoning_details as Anthropic thinking deltas without duplicating reasoning_content', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_reasoning_anthropic_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{"reasoning_content":"mini ","reasoning_details":[{"type":"reasoning.text","text":"mini ","id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_anthropic_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{"content":"visible"}}]}\n\n',
        'data: {"id":"chatcmpl_reasoning_anthropic_1","object":"chat.completion.chunk","model":"MiniMax-M2.7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['MiniMax-M2.7'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'MiniMax-M2.7',
          max_tokens: 128,
          stream: true,
          messages: [{ role: 'user', content: 'hello' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.body).toContain('"type":"thinking_delta","thinking":"mini "');
      const thinkingMatches = response.body.match(/"thinking":"mini "/g) || [];
      expect(thinkingMatches).toHaveLength(1);
      expect(response.body).toContain('"type":"text_delta","text":"visible"');
      expect(response.body).toContain('event: message_stop');
    } finally {
      await app.close();
    }
  });

  it('defaults Responses completed usage when chat/completions stream omits usage', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_stream_no_usage","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_stream_no_usage","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n',
        'data: {"id":"chatcmpl_stream_no_usage","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      const completedLine = response.body
        .split('\n')
        .find((line) => line.startsWith('data: ') && line.includes('"type":"response.completed"'));
      expect(completedLine).toBeDefined();
      const completed = JSON.parse(String(completedLine).slice('data: '.length));
      expect(completed.response.usage).toMatchObject({
        input_tokens: 0,
        input_tokens_details: {
          cached_tokens: 0
        },
        output_tokens: 0,
        output_tokens_details: {
          reasoning_tokens: 0
        },
        total_tokens: 0
      });
    } finally {
      await app.close();
    }
  });

  it('completes Responses stream promptly when chat/completions usage arrives after finish', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_stream_late_usage","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n',
        'data: {"id":"chatcmpl_stream_late_usage","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: {"id":"chatcmpl_stream_late_usage","object":"chat.completion.chunk","model":"glm-5","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      const completedLine = response.body
        .split('\n')
        .find((line) => line.startsWith('data: ') && line.includes('"type":"response.completed"'));
      expect(completedLine).toBeDefined();
      const completed = JSON.parse(String(completedLine).slice('data: '.length));
      expect(completed.response.usage).toMatchObject({
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0
      });
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('streams /v1/chat/completions incrementally when converting to anthropic stream', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"glm-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('anthropic-main', 'anthropic_messages', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'anthropic-main'
        },
        payload: {
          model: 'glm-5',
          stream: true,
          messages: [{ role: 'user', content: 'Say hello world' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.anthropic.com/v1/messages');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.stream).toBe(true);
      expect(upstreamBody.max_tokens).toBe(1024);

      expect(response.body).toContain('"object":"chat.completion.chunk"');
      expect(response.body).toContain('"delta":{"role":"assistant"}');
      expect(response.body).toContain('"delta":{"content":"hello "}');
      expect(response.body).toContain('"delta":{"content":"world"}');
      expect(response.body).toContain('"finish_reason":"stop"');
      expect(response.body).toContain('"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('converts anthropic tool_use responses into openai chat tool_calls', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'msg_tool_1',
          type: 'message',
          role: 'assistant',
          model: 'glm-5',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_weather',
              name: 'get_weather',
              input: {
                city: 'Shanghai'
              }
            }
          ],
          stop_reason: 'tool_use',
          stop_sequence: null,
          usage: {
            input_tokens: 7,
            output_tokens: 4
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('anthropic-main', 'anthropic_messages', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'anthropic-main'
        },
        payload: {
          model: 'glm-5',
          tools: [
            {
              type: 'function',
              function: {
                name: 'get_weather',
                description: 'Get current weather.',
                parameters: {
                  type: 'object',
                  properties: {
                    city: { type: 'string' }
                  },
                  required: ['city']
                }
              }
            }
          ],
          tool_choice: 'required',
          messages: [{ role: 'user', content: 'What is the weather in Shanghai?' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.anthropic.com/v1/messages');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.tools).toEqual([
        {
          name: 'get_weather',
          description: 'Get current weather.',
          input_schema: {
            type: 'object',
            properties: {
              city: { type: 'string' }
            },
            required: ['city']
          }
        }
      ]);
      expect(upstreamBody.tool_choice).toEqual({
        type: 'any'
      });

      const body = JSON.parse(response.body);
      expect(body.object).toBe('chat.completion');
      expect(body.choices[0]?.message).toEqual({
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'toolu_weather',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"city":"Shanghai"}'
            }
          }
        ]
      });
      expect(body.choices[0]?.finish_reason).toBe('tool_calls');
      expect(body.usage).toEqual({
        prompt_tokens: 7,
        completion_tokens: 4,
        total_tokens: 11
      });
    } finally {
      await app.close();
    }
  });

  it('streams anthropic tool_use events as openai chat tool_calls', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_tool_stream","type":"message","role":"assistant","model":"glm-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_weather","name":"get_weather","input":{}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Sh"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"anghai\\"}"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":4}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('anthropic-main', 'anthropic_messages', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'anthropic-main'
        },
        payload: {
          model: 'glm-5',
          stream: true,
          tools: [
            {
              type: 'function',
              function: {
                name: 'get_weather',
                description: 'Get current weather.',
                parameters: {
                  type: 'object',
                  properties: {
                    city: { type: 'string' }
                  },
                  required: ['city']
                }
              }
            }
          ],
          messages: [{ role: 'user', content: 'What is the weather in Shanghai?' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.body).toContain('"delta":{"role":"assistant"}');
      expect(response.body).toContain('"tool_calls":[{"index":0,"id":"toolu_weather","type":"function","function":{"name":"get_weather","arguments":""}}]');
      expect(response.body).toContain('{\\"city\\":\\"Sh');
      expect(response.body).toContain('anghai\\"}');
      expect(response.body).toContain('"finish_reason":"tool_calls"');
      expect(response.body).toContain('"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('streams /v1/chat/completions incrementally when converting to openai responses stream', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_stream_1","object":"response","model":"gpt-5.4"}}\n\n',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hello ","output_index":0,"content_index":0}\n\n',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"world","output_index":0,"content_index":0}\n\n',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_stream_1","object":"response","model":"gpt-5.4","status":"completed","output":[{"id":"msg_stream_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['gpt-5.4'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-5.4',
          stream: true,
          messages: [{ role: 'user', content: 'Say hello world' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/responses');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.stream).toBe(true);

      expect(response.body).toContain('"object":"chat.completion.chunk"');
      expect(response.body).toContain('"delta":{"role":"assistant"}');
      expect(response.body).toContain('"delta":{"content":"hello "}');
      expect(response.body).toContain('"delta":{"content":"world"}');
      expect(response.body).toContain('"finish_reason":"stop"');
      expect(response.body).toContain('"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('forces SSE headers for /v1/chat/completions stream conversion when upstream content-type is plain text', async () => {
    const fetchMock = vi.fn(async () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_plain_chat_1","object":"response","model":"gpt-5.4"}}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hello ","output_index":0,"content_index":0}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"world","output_index":0,"content_index":0}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_plain_chat_1","object":"response","model":"gpt-5.4","status":"completed","output":[{"id":"msg_plain_chat_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n'
            )
          );
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();
        }
      });

      return new Response(stream, {
        status: 200,
        headers: {
          'content-type': 'text/plain; charset=utf-8'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['gpt-5.4'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-5.4',
          stream: true,
          messages: [{ role: 'user', content: 'Say hello world' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.headers['cache-control']).toContain('no-transform');
      expect(response.headers['x-accel-buffering']).toBe('no');
      expect(response.body).toContain('"object":"chat.completion.chunk"');
      expect(response.body).toContain('"delta":{"content":"hello "}');
      expect(response.body).toContain('"delta":{"content":"world"}');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('treats non-json upstream content-type as live stream for /v1/chat/completions conversion', async () => {
    const fetchMock = vi.fn(async () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_octet_chat_1","object":"response","model":"gpt-5.4"}}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hello ","output_index":0,"content_index":0}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"world","output_index":0,"content_index":0}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_octet_chat_1","object":"response","model":"gpt-5.4","status":"completed","output":[{"id":"msg_octet_chat_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n'
            )
          );
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();
        }
      });

      return new Response(stream, {
        status: 200,
        headers: {
          'content-type': 'application/octet-stream'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['gpt-5.4'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-5.4',
          stream: true,
          messages: [{ role: 'user', content: 'Say hello world' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.headers['cache-control']).toContain('no-transform');
      expect(response.headers['x-accel-buffering']).toBe('no');
      expect(response.body).toContain('"object":"chat.completion.chunk"');
      expect(response.body).toContain('"delta":{"content":"hello "}');
      expect(response.body).toContain('"delta":{"content":"world"}');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('streams tool call events for /v1/responses when converting from chat/completions stream', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Sh"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"arguments":"anghai\\"}"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'call get_weather with city Shanghai',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.body).toContain('"type":"response.output_item.added","output_index":0');
      expect(response.body).toContain('"type":"response.function_call_arguments.delta","output_index":0');
      expect(response.body).toContain('"type":"response.function_call_arguments.done","output_index":0');
      expect(response.body).toContain('"type":"response.output_item.done","output_index":0');
      expect(response.body).toContain('"type":"response.completed"');
      expect(response.body).toContain('"call_id":"call_weather"');
      expect(response.body).toContain('"name":"get_weather"');
      expect(response.body).toContain('"arguments":"{\\"city\\":\\"Shanghai\\"}"');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('streams namespace tool call events for /v1/responses when converting from chat/completions stream', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_app_state","type":"function","function":{"name":"mcp__computer_use__.get_app_state","arguments":"{\\"app\\":\\"Sla"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_app_state","type":"function","function":{"arguments":"ck\\"}"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'inspect Slack',
          stream: true,
          tools: [
            {
              name: 'mcp__computer_use__',
              type: 'namespace',
              tools: [
                {
                  name: 'get_app_state',
                  type: 'function',
                  parameters: {
                    type: 'object',
                    properties: {
                      app: {
                        type: 'string'
                      }
                    },
                    required: ['app'],
                    additionalProperties: false
                  }
                }
              ]
            }
          ]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.body).toContain('"type":"response.output_item.done","output_index":0');
      expect(response.body).toContain('"call_id":"call_app_state"');
      expect(response.body).toContain('"name":"get_app_state"');
      expect(response.body).toContain('"namespace":"mcp__computer_use__"');
      expect(response.body).toContain('"arguments":"{\\"app\\":\\"Slack\\"}"');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('keeps passthrough for /v1/responses when target protocol is openai_responses', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'resp_123',
          object: 'response',
          status: 'completed',
          model: 'gpt-4.1-mini',
          output_text: 'native responses',
          output: [
            {
              id: 'msg_1',
              type: 'message',
              role: 'assistant',
              status: 'completed',
              content: [
                {
                  type: 'output_text',
                  text: 'native responses'
                }
              ]
            }
          ],
          usage: {
            input_tokens: 8,
            output_tokens: 4,
            total_tokens: 12
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['gpt-4.1-mini'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-4.1-mini',
          input: 'hello native'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/responses');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.input).toBe('hello native');
      expect(upstreamBody.model).toBe('gpt-4.1-mini');
    } finally {
      await app.close();
    }
  });

  it('forces SSE response headers for openai passthrough streaming when upstream content-type is plain text', async () => {
    const fetchMock = vi.fn(async () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_plain_sse","object":"response"}}\n\n'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_plain_sse","object":"response","status":"completed","output":[{"id":"msg_plain_sse","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}}\n\n'
            )
          );
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();
        }
      });

      return new Response(stream, {
        status: 200,
        headers: {
          'content-type': 'text/plain; charset=utf-8'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['gpt-4.1-mini'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-4.1-mini',
          input: 'hello native',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.headers['cache-control']).toContain('no-transform');
      expect(response.headers['x-accel-buffering']).toBe('no');
      expect(response.body).toContain('event: response.created');
      expect(response.body).toContain('event: response.completed');
    } finally {
      await app.close();
    }
  });

  it('returns non-stream JSON for /v1/responses when upstream chat endpoint responds with SSE', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_nonstream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_nonstream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n',
        'data: {"id":"chatcmpl_nonstream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
        'data: {"id":"chatcmpl_nonstream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello world'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('application/json');
      const body = JSON.parse(response.body);
      expect(body.object).toBe('response');
      expect(body.output_text).toBe('hello world');
      expect(body.usage).toMatchObject({
        input_tokens: 3,
        output_tokens: 2,
        total_tokens: 5
      });
    } finally {
      await app.close();
    }
  });

  it('streams Gemini SSE incrementally when routing streamGenerateContent to openai chat target', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_gem_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_gem_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n',
        'data: {"id":"chatcmpl_gem_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
        'data: {"id":"chatcmpl_gem_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/models/openai-main/glm-5:streamGenerateContent?alt=sse',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          contents: [
            {
              role: 'user',
              parts: [{ text: 'hello world' }]
            }
          ]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/chat/completions');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.stream).toBe(true);
      expect(upstreamBody.messages).toEqual([{ role: 'user', content: 'hello world' }]);

      expect(response.body).toContain('"parts":[{"text":"hello "}]');
      expect(response.body).toContain('"parts":[{"text":"world"}]');
      expect(response.body).toContain('"finishReason":"STOP"');
      expect(response.body).toContain('"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}');
    } finally {
      await app.close();
    }
  });

  it('converts Gemini tools and streamed tool-call arguments when routing to openai chat target', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_gem_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_gem_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Sh"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_gem_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"arguments":"anghai\\",\\"unit\\":\\"C\\"}"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_gem_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\n',
        'data: [DONE]\n\n'
      ]);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/models/openai-main/glm-5:streamGenerateContent?alt=sse',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          contents: [
            {
              role: 'user',
              parts: [{ text: 'What is the weather in Shanghai?' }]
            }
          ],
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'get_weather',
                  description: 'Get current weather.',
                  parameters: {
                    type: 'object',
                    properties: {
                      city: { type: 'string' },
                      unit: { type: 'string' }
                    },
                    required: ['city']
                  }
                }
              ]
            }
          ],
          toolConfig: {
            functionCallingConfig: {
              mode: 'ANY',
              allowedFunctionNames: ['get_weather']
            }
          }
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/chat/completions');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.messages).toEqual([{ role: 'user', content: 'What is the weather in Shanghai?' }]);
      expect(upstreamBody.tools).toEqual([
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get current weather.',
            parameters: {
              type: 'object',
              properties: {
                city: { type: 'string' },
                unit: { type: 'string' }
              },
              required: ['city']
            }
          }
        }
      ]);
      expect(upstreamBody.tool_choice).toEqual({
        type: 'function',
        function: {
          name: 'get_weather'
        }
      });

      expect(response.body).toContain(
        '"functionCall":{"name":"get_weather","args":{"city":"Shanghai","unit":"C"}}'
      );
      expect(response.body).toContain('"finishReason":"STOP"');
      expect(response.body).toContain('"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":4,"totalTokenCount":11}');
    } finally {
      await app.close();
    }
  });

  it('rejects requests when model is not configured for the target provider', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          'content-type': 'application/json'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([createProviderConfig('openai-main', 'openai_responses', ['glm-5'])]),
      createGatewayRuntime()
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'gpt-4.1-mini',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(fetchMock).not.toHaveBeenCalled();
      const body = JSON.parse(response.body);
      expect(body.error.message).toBe('All target providers failed.');
      expect(body.error.attempts[0].stage).toBe('model_resolution');
      expect(body.error.attempts[0].message).toContain(
        'Model "gpt-4.1-mini" is not configured for target provider openai-main.'
      );
      expect(body.error.attempts[0].message).toContain('Allowed models: glm-5.');
    } finally {
      await app.close();
    }
  });

  it('maps Anthropic thinking controls to DeepSeek OpenAI chat fields', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_deepseek_thinking',
          object: 'chat.completion',
          model: 'deepseek-v4-pro',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'ok'
              }
            }
          ],
          usage: {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    const config = createConfig(
      [createProviderConfig('router-main', 'openai_chat_completions', ['v4-pro'])],
      [
        {
          key: 'deepseek-thinking',
          enabled: true,
          providerName: 'router-main',
          deepseekThinking: {
            enabled: true
          }
        }
      ]
    );
    registerGatewayRoutes(
      app,
      config,
      createGatewayRuntime(config)
    );
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'anthropic-version': '2023-06-01',
          'x-target-provider': 'router-main'
        },
        payload: {
          model: 'v4-pro',
          max_tokens: 128,
          thinking: {
            type: 'enabled'
          },
          output_config: {
            effort: 'xhigh'
          },
          messages: [
            {
              role: 'user',
              content: 'hello'
            }
          ]
        }
      });

      expect(response.statusCode).toBe(200);
      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.thinking).toEqual({ type: 'enabled' });
      expect(upstreamBody.reasoning_effort).toBe('max');
      expect(upstreamBody.output_config).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('normalizes OpenAI reasoning effort for DeepSeek and honors disabled thinking', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_deepseek_reasoning',
          object: 'chat.completion',
          model: 'deepseek-v4-pro',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'ok'
              }
            }
          ],
          usage: {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const app = Fastify({ logger: false });
    const config = createConfig(
      [createProviderConfig('router-main', 'openai_chat_completions', ['v4-pro'])],
      [
        {
          key: 'deepseek-thinking',
          enabled: true,
          providerName: 'router-main',
          deepseekThinking: {
            enabled: true
          }
        }
      ]
    );
    registerGatewayRoutes(
      app,
      config,
      createGatewayRuntime(config)
    );
    await app.ready();

    try {
      const enabledResponse = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'router-main'
        },
        payload: {
          model: 'v4-pro',
          input: 'hello',
          reasoning: {
            effort: 'medium'
          }
        }
      });

      expect(enabledResponse.statusCode).toBe(200);
      const [, enabledUpstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const enabledUpstreamBody = JSON.parse(String(enabledUpstreamInit.body));
      expect(enabledUpstreamBody.thinking).toEqual({ type: 'enabled' });
      expect(enabledUpstreamBody.reasoning_effort).toBe('high');
      expect(enabledUpstreamBody.output_config).toBeUndefined();

      const disabledResponse = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'router-main'
        },
        payload: {
          model: 'v4-pro',
          input: 'hello',
          thinking: {
            type: 'disabled'
          },
          output_config: {
            effort: 'max'
          }
        }
      });

      expect(disabledResponse.statusCode).toBe(200);
      const [, disabledUpstreamInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const disabledUpstreamBody = JSON.parse(String(disabledUpstreamInit.body));
      expect(disabledUpstreamBody.thinking).toEqual({ type: 'disabled' });
      expect(disabledUpstreamBody.reasoning_effort).toBeUndefined();
      expect(disabledUpstreamBody.output_config).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('does not enable DeepSeek thinking conversion from provider name alone', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_deepseek_no_plugin',
          object: 'chat.completion',
          model: 'v4-pro',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'ok'
              }
            }
          ],
          usage: {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([createProviderConfig('deepseek-main', 'openai_chat_completions', ['v4-pro'])]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'deepseek-main'
        },
        payload: {
          model: 'v4-pro',
          input: 'hello',
          reasoning: {
            effort: 'medium'
          }
        }
      });

      expect(response.statusCode).toBe(200);
      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.reasoning_effort).toBeUndefined();
      expect(upstreamBody.output_config).toEqual({ effort: 'medium' });
    } finally {
      await app.close();
    }
  });

  it('applies provider plugin auth/request/response transforms by provider name', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const headers = (init?.headers || {}) as Record<string, string>;
      const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'upstream',
          auth_header: headers['x-custom-auth'],
          request_marker: body.custom_plugin_field
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const runtime = createGatewayRuntime();
    runtime.providerPlugins.register({
      key: 'openai-main-custom-plugin',
      providerName: 'OPENAI-MAIN',
      authenticate({ upstreamRequest }) {
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            headers: {
              ...upstreamRequest.headers,
              'x-custom-auth': 'signed-main'
            }
          }
        };
      },
      transformRequest({ upstreamRequest }) {
        const payload = upstreamRequest.body as Record<string, unknown>;
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            body: {
              ...payload,
              custom_plugin_field: 'request-main'
            }
          }
        };
      },
      transformResponse({ upstreamPayload }) {
        const payload = upstreamPayload as Record<string, unknown>;
        return {
          ok: true,
          value: {
            ...payload,
            output_text: 'plugin-main',
            plugin_response: true
          }
        };
      }
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(
      app,
      createConfig([
        createProviderConfig('openai-main', 'openai_responses', ['glm-5']),
        createProviderConfig('openai-backup', 'openai_responses', ['glm-5'])
      ]),
      runtime
    );
    await app.ready();

    try {
      const mainResponse = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello main'
        }
      });
      expect(mainResponse.statusCode).toBe(200);

      const [, mainUpstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const mainUpstreamHeaders = mainUpstreamInit.headers as Record<string, string>;
      const mainUpstreamBody = JSON.parse(String(mainUpstreamInit.body));
      expect(mainUpstreamHeaders['x-custom-auth']).toBe('signed-main');
      expect(mainUpstreamBody.custom_plugin_field).toBe('request-main');

      const mainBody = JSON.parse(mainResponse.body);
      expect(mainBody.output_text).toBe('plugin-main');
      expect(mainBody.plugin_response).toBe(true);
      expect(mainBody.auth_header).toBe('signed-main');
      expect(mainBody.request_marker).toBe('request-main');

      const backupResponse = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-backup'
        },
        payload: {
          model: 'glm-5',
          input: 'hello backup'
        }
      });
      expect(backupResponse.statusCode).toBe(200);

      const [, backupUpstreamInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const backupUpstreamHeaders = backupUpstreamInit.headers as Record<string, string>;
      const backupUpstreamBody = JSON.parse(String(backupUpstreamInit.body));
      expect(backupUpstreamHeaders['x-custom-auth']).toBeUndefined();
      expect(backupUpstreamBody.custom_plugin_field).toBeUndefined();

      const backupBody = JSON.parse(backupResponse.body);
      expect(backupBody.output_text).toBe('upstream');
      expect(backupBody.plugin_response).toBeUndefined();
      expect(backupBody.auth_header).toBeUndefined();
      expect(backupBody.request_marker).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('loads configured providerPlugins from gateway config automatically', async () => {
    process.env.OPENAI_MAIN_DYNAMIC_AUTH = 'dynamic-auth-token';
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const headers = (init?.headers || {}) as Record<string, string>;
      const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'upstream-config-plugin',
          auth_header: headers['x-config-auth'],
          request_marker: body.config_plugin_marker,
          data: {
            output: 'output-from-nested-field'
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-dynamic',
          enabled: true,
          providerName: 'openai-main',
          auth: {
            strict: true,
            headers: {
              'x-config-auth': {
                from: 'env.OPENAI_MAIN_DYNAMIC_AUTH'
              }
            },
            query: {},
            removeHeaders: [],
            removeQuery: [],
            bodySet: {},
            bodyMerge: {},
            bodyRemove: []
          },
          request: {
            strict: false,
            headers: {},
            query: {},
            removeHeaders: [],
            removeQuery: [],
            bodySet: {
              config_plugin_marker: {
                from: 'request.headers.x-auth-user-id'
              }
            },
            bodyMerge: {},
            bodyRemove: []
          },
          response: {
            strict: true,
            bodySet: {
              output_text: {
                from: 'upstreamPayload.data.output'
              },
              plugin_loaded_from_config: true
            },
            bodyMerge: {},
            bodyRemove: ['data']
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-auth-user-id': 'u-config-001'
        },
        payload: {
          model: 'glm-5',
          input: 'hello config plugin'
        }
      });

      expect(response.statusCode).toBe(200);
      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamHeaders = upstreamInit.headers as Record<string, string>;
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamHeaders['x-config-auth']).toBe('dynamic-auth-token');
      expect(upstreamBody.config_plugin_marker).toBe('u-config-001');

      const payload = JSON.parse(response.body);
      expect(payload.output_text).toBe('output-from-nested-field');
      expect(payload.plugin_loaded_from_config).toBe(true);
      expect(payload.data).toBeUndefined();
    } finally {
      delete process.env.OPENAI_MAIN_DYNAMIC_AUTH;
      await app.close();
    }
  });

  it('refreshes codex oauth token, rewrites upstream URL and injects codex headers', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const urlString = String(url);
      if (urlString === 'https://auth.openai.com/oauth/token') {
        const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
        expect(init?.method).toBe('POST');
        expect(body.client_id).toBe('app_EMoamEEZ73f0CkXaXp7hrann');
        expect(body.scope).toBe(
          'openid profile email offline_access api.connectors.read api.connectors.invoke'
        );
        expect(body.grant_type).toBe('refresh_token');
        expect(body.refresh_token).toBe('rtk-from-request');
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-codex-refresh'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      const headers = (init?.headers || {}) as Record<string, string>;
      const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'codex-oauth-upstream',
          authorization: headers.authorization,
          account_id: headers['ChatGPT-Account-ID'],
          store: body.store,
          stream: body.stream,
          instructions: body.instructions
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const providerConfig = createProviderConfig('openai-main', 'openai_responses', ['glm-5']);
    providerConfig.extraBody.default = {
      store: true
    };

    const config = createConfig(
      [providerConfig],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            },
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            accountId: {
              from: 'request.headers.x-codex-account-id'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-account-id': 'acct-test-001',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          input: 'hello codex oauth'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://auth.openai.com/oauth/token');
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      const [, upstreamInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const upstreamHeaders = upstreamInit.headers as Record<string, string>;
      expect(upstreamHeaders.authorization).toBe('Bearer atk-from-codex-refresh');
      expect(upstreamHeaders['ChatGPT-Account-ID']).toBe('acct-test-001');
      expect(upstreamHeaders.accept).toBe('text/event-stream');

      const payload = JSON.parse(response.body);
      expect(payload.authorization).toBe('Bearer atk-from-codex-refresh');
      expect(payload.account_id).toBe('acct-test-001');
      expect(payload.store).toBe(false);
      expect(payload.stream).toBe(true);
      expect(payload.instructions).toBe('You are a helpful assistant.');
      expect(payload.output_text).toBe('codex-oauth-upstream');
    } finally {
      await app.close();
    }
  });

  it('injects default instructions and forces store=false for chat/completions requests via codex oauth', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const urlString = String(url);
      if (urlString === 'https://auth.openai.com/oauth/token') {
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-codex-refresh'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'chat-codex-oauth-upstream',
          store: body.store,
          stream: body.stream,
          instructions: body.instructions
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          messages: [{ role: 'user', content: '天为什么是蓝的' }],
          stream: false
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      const [, upstreamInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const upstreamBody = JSON.parse(String(upstreamInit.body || '{}')) as Record<string, unknown>;
      const upstreamHeaders = upstreamInit.headers as Record<string, string>;
      expect(upstreamHeaders.accept).toBe('text/event-stream');
      expect(upstreamBody.store).toBe(false);
      expect(upstreamBody.stream).toBe(true);
      expect(upstreamBody.instructions).toBe('You are a helpful assistant.');

      const payload = JSON.parse(response.body);
      expect(payload.choices?.[0]?.message?.content).toBe('chat-codex-oauth-upstream');
    } finally {
      await app.close();
    }
  });

  it('decrypts encrypted codex oauth access token from distributed key', async () => {
    const key = Buffer.from('0123456789abcdef0123456789abcdef', 'utf8');
    updateDistributedCredentialEncryption({
      key: key.toString('base64'),
      keyVersion: 'v1',
      algorithm: 'aes-256-gcm'
    });
    const encryptedAccessToken = encryptCredentialForTest('atk-encrypted', key, 'v1');

    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const headers = (init?.headers || {}) as Record<string, string>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'decrypted-token-ok',
          authorization: headers.authorization
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            accessToken: encryptedAccessToken,
            refreshIfMissingAccessToken: false,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'hello encrypted codex oauth'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');

      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamHeaders = (upstreamInit.headers || {}) as Record<string, string>;
      expect(upstreamHeaders.authorization).toBe('Bearer atk-encrypted');

      const payload = JSON.parse(response.body);
      expect(payload.authorization).toBe('Bearer atk-encrypted');
      expect(payload.output_text).toBe('decrypted-token-ok');
    } finally {
      await app.close();
    }
  });

  it('caches codex oauth state in memory and reuses refreshed token on next request', async () => {
    const encryptionKey = Buffer.from('0123456789abcdef0123456789abcdef', 'utf8');
    updateDistributedCredentialEncryption({
      key: encryptionKey.toString('base64'),
      keyVersion: 'v1',
      algorithm: 'aes-256-gcm'
    });

    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url) === 'https://auth.openai.com/oauth/token') {
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-refresh',
            refresh_token: 'rtk-from-refresh'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      const headers = (init?.headers || {}) as Record<string, string>;
      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'memory-state-ok',
          authorization: headers.authorization
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const first = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          input: 'first'
        }
      });
      expect(first.statusCode).toBe(200);

      const second = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'glm-5',
          input: 'second'
        }
      });
      expect(second.statusCode).toBe(200);

      expect(fetchMock).toHaveBeenCalledTimes(3);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://auth.openai.com/oauth/token');
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      expect(String(fetchMock.mock.calls[2]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      const secondHeaders = (fetchMock.mock.calls[2]?.[1]?.headers || {}) as Record<string, string>;
      expect(secondHeaders.authorization).toBe('Bearer atk-from-refresh');
    } finally {
      await app.close();
    }
  });

  it('recovers non-stream chat/completions response when upstream returns raw SSE text payload', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      const urlString = String(url);
      if (urlString === 'https://auth.openai.com/oauth/token') {
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-codex-refresh'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      const rawSsePayload =
        'event: response.created\n' +
        'data: {"type":"response.created","response":{"id":"resp_raw_sse","object":"response","model":"gpt-5.4"}}\n\n' +
        'event: response.output_text.delta\n' +
        'data: {"type":"response.output_text.delta","delta":"raw-sse-ok","output_index":0,"content_index":0}\n\n' +
        'event: response.completed\n' +
        'data: {"type":"response.completed","response":{"id":"resp_raw_sse","object":"response","model":"gpt-5.4","output":[{"id":"msg_raw_sse","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"raw-sse-ok"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n' +
        'data: [DONE]\n\n';

      return new Response(rawSsePayload, {
        status: 200,
        headers: {
          'content-type': 'text/plain; charset=utf-8'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          messages: [{ role: 'user', content: '天为什么是蓝的' }],
          stream: false
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      const payload = JSON.parse(response.body);
      expect(payload.choices?.[0]?.message?.content).toBe('raw-sse-ok');
    } finally {
      await app.close();
    }
  });

  it('forces codex oauth refresh and retries once when upstream returns 401', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const urlString = String(url);
      if (urlString === 'https://auth.openai.com/oauth/token') {
        const body = JSON.parse(String(init?.body || '{}')) as Record<string, unknown>;
        expect(body.grant_type).toBe('refresh_token');
        expect(body.refresh_token).toBe('rtk-from-request');
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-codex-refresh'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      const headers = (init?.headers || {}) as Record<string, string>;
      if (headers.authorization === 'Bearer atk-stale') {
        return new Response(
          JSON.stringify({
            error: {
              message: 'You have insufficient permissions for this operation.'
            }
          }),
          {
            status: 401,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'retry-success',
          authorization: headers.authorization
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            },
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-access-token': 'atk-stale',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          input: 'hello codex oauth retry'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(3);

      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://auth.openai.com/oauth/token');
      expect(String(fetchMock.mock.calls[2]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');

      const firstUpstreamHeaders = (fetchMock.mock.calls[0]?.[1]?.headers || {}) as Record<string, string>;
      const secondUpstreamHeaders = (fetchMock.mock.calls[2]?.[1]?.headers || {}) as Record<string, string>;
      expect(firstUpstreamHeaders.authorization).toBe('Bearer atk-stale');
      expect(secondUpstreamHeaders.authorization).toBe('Bearer atk-from-codex-refresh');

      const payload = JSON.parse(response.body);
      expect(payload.output_text).toBe('retry-success');
      expect(payload.authorization).toBe('Bearer atk-from-codex-refresh');
    } finally {
      await app.close();
    }
  });

  it('returns provider_auth error when refreshed codex oauth scope misses required codex scopes', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === 'https://auth.openai.com/oauth/token') {
        return new Response(
          JSON.stringify({
            access_token: 'atk-from-codex-refresh',
            scope: 'openid profile email offline_access'
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        );
      }

      return new Response(
        JSON.stringify({
          object: 'response',
          output_text: 'unexpected-upstream-call'
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_responses', ['glm-5'])],
      [
        {
          key: 'openai-main-codex-oauth',
          enabled: true,
          providerName: 'openai-main',
          codexOauth: {
            enabled: true,
            tokenEndpoint: 'https://auth.openai.com/oauth/token',
            clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
            scope:
              'openid profile email offline_access api.connectors.read api.connectors.invoke',
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            },
            refreshIfMissingAccessToken: true,
            forceRefresh: false,
            required: true,
            timeoutMs: 3000,
            authHeader: 'authorization',
            authScheme: 'Bearer'
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
          'x-codex-refresh-token': 'rtk-from-request'
        },
        payload: {
          model: 'glm-5',
          input: 'hello codex oauth scope'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://auth.openai.com/oauth/token');

      const payload = JSON.parse(response.body);
      expect(payload.error.message).toBe('All target providers failed.');
      expect(payload.error.attempts?.[0]?.stage).toBe('provider_auth');
      expect(String(payload.error.attempts?.[0]?.message || '')).toContain(
        'missing required scopes'
      );
      expect(String(payload.error.attempts?.[0]?.message || '')).toContain(
        'api.connectors.invoke'
      );
    } finally {
      await app.close();
    }
  });

  it('publishes a single request-level billing event for failed requests', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url) === 'http://billing.local/events') {
        return new Response(null, { status: 200 });
      }

      return new Response(
        JSON.stringify({
          error: {
            message: 'upstream boom',
          },
        }),
        {
          status: 500,
          headers: {
            'content-type': 'application/json',
          },
        },
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5']),
    ]);
    config.billingWebhook = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://billing.local/events',
      timeoutMs: 1000,
      maxAttempts: 1,
      baseDelayMs: 10,
      maxDelayMs: 10,
      requireAck: false,
      headers: {},
    };
    await initializeBillingPublisher(config.billingQueue, config.billingWebhook);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
        },
        payload: {
          model: 'glm-5',
          messages: [{ role: 'user', content: 'fail please' }],
        },
      });

      expect(response.statusCode).toBe(500);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
        'https://api.openai.com/v1/chat/completions',
      );
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe(
        'http://billing.local/events',
      );

      const billingPayload = JSON.parse(
        String(((fetchMock.mock.calls[1] as unknown as [string, RequestInit])?.[1])?.body),
      );
      expect(billingPayload.target).toMatchObject({
        provider: 'openai',
        providerName: 'openai-main',
        model: 'glm-5',
      });
      expect(billingPayload.outcome).toMatchObject({
        status: 'error',
        statusCode: 500,
        errorMessage: 'Upstream request failed.',
      });
      expect(billingPayload.attempt).toBeUndefined();
      expect(billingPayload.attempts).toHaveLength(1);
      expect(billingPayload.attempts[0]).toMatchObject({
        provider: 'openai',
        stage: 'upstream_response',
        status: 500,
      });
      expect(billingPayload.billing.cost.total).toBe(0);
    } finally {
      await app.close();
    }
  });

  it('publishes success billing events without legacy attempt metadata', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === 'http://billing.local/events') {
        return new Response(null, { status: 200 });
      }

      return new Response(
        JSON.stringify({
          id: 'chatcmpl_success_1',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'hello',
              },
            },
          ],
          usage: {
            prompt_tokens: 12,
            completion_tokens: 8,
            total_tokens: 20,
          },
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json',
          },
        },
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5']),
    ]);
    config.billing.enabled = true;
    config.billingWebhook = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://billing.local/events',
      timeoutMs: 1000,
      maxAttempts: 1,
      baseDelayMs: 10,
      maxDelayMs: 10,
      requireAck: false,
      headers: {},
    };
    await initializeBillingPublisher(config.billingQueue, config.billingWebhook);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main',
        },
        payload: {
          model: 'glm-5',
          messages: [{ role: 'user', content: 'hello' }],
        },
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe(
        'http://billing.local/events',
      );

      const billingPayload = JSON.parse(
        String(((fetchMock.mock.calls[1] as unknown as [string, RequestInit])?.[1])?.body),
      );
      expect(billingPayload.target).toMatchObject({
        provider: 'openai',
        providerName: 'openai-main',
        model: 'glm-5',
      });
      expect(billingPayload.outcome).toMatchObject({
        status: 'success',
        statusCode: 200,
      });
      expect(billingPayload.attempt).toBeUndefined();
      expect(billingPayload.billing.usage.total_tokens).toBe(20);
    } finally {
      await app.close();
    }
  });

  it('executes declared gateway tools transparently for ordinary chat completion requests', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_transparent_tool_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'tool_calls',
                message: {
                  role: 'assistant',
                  content: '',
                  tool_calls: [
                    {
                      id: 'call_search_transparent_1',
                      type: 'function',
                      function: {
                        name: 'search_web',
                        arguments: '{"query":"latest ai news"}'
                      }
                    }
                  ]
                }
              }
            ],
            usage: {
              prompt_tokens: 5,
              completion_tokens: 3,
              total_tokens: 8
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_transparent_final_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'fresh answer'
                }
              }
            ],
            usage: {
              prompt_tokens: 6,
              completion_tokens: 4,
              total_tokens: 10
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const executedToolCalls: Array<{ name: string; args: unknown }> = [];
    const toolProvider = {
      listDefinitions: async () => [
        {
          name: 'search_web',
          description: 'Search the web',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string'
              }
            }
          }
        }
      ],
      has: async () => true,
      execute: async (name: string, input: { args: unknown }) => {
        executedToolCalls.push({ name, args: input.args });
        return {
          hits: ['fresh result']
        };
      },
      close: async () => undefined
    };

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.transparentToolExecution = {
      enabled: true,
      maxTurns: 4,
      maxToolCalls: 4,
      requireClientDeclaration: true,
      unknownToolPolicy: 'return_to_client',
      allowTools: [],
      denyTools: []
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config, toolProvider as any));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5',
          messages: [{ role: 'user', content: 'What happened today?' }],
          tools: [
            {
              type: 'function',
              function: {
                name: 'search_web',
                description: 'Search the web',
                parameters: {
                  type: 'object',
                  properties: {
                    query: {
                      type: 'string'
                    }
                  }
                }
              }
            }
          ],
          tool_choice: 'auto'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(executedToolCalls).toEqual([
        {
          name: 'search_web',
          args: {
            query: 'latest ai news'
          }
        }
      ]);

      const [, secondInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const secondBody = JSON.parse(String(secondInit.body));
      expect(JSON.stringify(secondBody.messages)).toContain('search_web');
      expect(JSON.stringify(secondBody.messages)).toContain('fresh result');

      const payload = JSON.parse(response.body);
      expect(payload.choices[0]?.message?.content).toBe('fresh answer');
      expect(payload.choices[0]?.message?.tool_calls).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('returns ordinary tool calls to the client when transparent tools are unknown', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          id: 'chatcmpl_transparent_unknown_1',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'tool_calls',
              message: {
                role: 'assistant',
                content: '',
                tool_calls: [
                  {
                    id: 'call_private_1',
                    type: 'function',
                    function: {
                      name: 'private_client_tool',
                      arguments: '{"value":1}'
                    }
                  }
                ]
              }
            }
          ],
          usage: {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      )
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const toolProvider = {
      listDefinitions: async () => [],
      has: async () => false,
      execute: async () => {
        throw new Error('should not execute');
      },
      close: async () => undefined
    };

    const config = createConfig([createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])]);
    config.transparentToolExecution = {
      enabled: true,
      maxTurns: 4,
      maxToolCalls: 4,
      requireClientDeclaration: true,
      unknownToolPolicy: 'return_to_client',
      allowTools: [],
      denyTools: []
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config, toolProvider as any));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5',
          messages: [{ role: 'user', content: 'Call my private tool' }],
          tools: [
            {
              type: 'function',
              function: {
                name: 'private_client_tool',
                parameters: {
                  type: 'object',
                  properties: {
                    value: {
                      type: 'number'
                    }
                  }
                }
              }
            }
          ]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const payload = JSON.parse(response.body);
      expect(payload.choices[0]?.message?.tool_calls?.[0]?.function?.name).toBe('private_client_tool');
    } finally {
      await app.close();
    }
  });

  it('executes internal virtual model tools without exposing tool calls to the client', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_tool_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'tool_calls',
                message: {
                  role: 'assistant',
                  content: '',
                  tool_calls: [
                    {
                      id: 'call_search_1',
                      type: 'function',
                      function: {
                        name: 'search_web',
                        arguments: '{"query":"latest ai news"}'
                      }
                    }
                  ]
                }
              }
            ],
            usage: {
              prompt_tokens: 5,
              completion_tokens: 3,
              total_tokens: 8
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_final_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'fresh answer'
                }
              }
            ],
            usage: {
              prompt_tokens: 6,
              completion_tokens: 4,
              total_tokens: 10
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const toolProvider = {
      listDefinitions: async () => [
        {
          name: 'search_web',
          description: 'Search the web',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string'
              }
            }
          }
        }
      ],
      has: async () => true,
      execute: async () => ({
        hits: ['fresh result']
      }),
      close: async () => undefined
    };

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])],
      undefined,
      [
        {
          id: 'virtual-search',
          key: 'search',
          displayName: 'Search',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':search']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          instructions: {
            prepend: 'Use search_web before answering time-sensitive questions.'
          },
          tools: [
            {
              name: 'search_web',
              visibility: 'internal'
            }
          ],
          toolChoice: 'auto',
          execution: {
            mode: 'tool_loop',
            maxTurns: 4,
            maxToolCalls: 4,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config, toolProvider as any));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5:search',
          messages: [{ role: 'user', content: 'What happened today?' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);

      const [, firstInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const firstBody = JSON.parse(String(firstInit.body));
      expect(firstBody.model).toBe('glm-5');
      expect(firstBody.tools?.[0]?.function?.name || firstBody.tools?.[0]?.name).toBe('search_web');

      const [, secondInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const secondBody = JSON.parse(String(secondInit.body));
      expect(JSON.stringify(secondBody.messages)).toContain('search_web');
      expect(JSON.stringify(secondBody.messages)).toContain('fresh result');

      const payload = JSON.parse(response.body);
      expect(payload.choices[0]?.message?.content).toBe('fresh answer');
      expect(payload.choices[0]?.message?.tool_calls).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('streams buffered virtual model responses as OpenAI Responses events', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_stream_tool_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'tool_calls',
                message: {
                  role: 'assistant',
                  content: '',
                  tool_calls: [
                    {
                      id: 'call_search_stream_1',
                      type: 'function',
                      function: {
                        name: 'search_web',
                        arguments: '{"query":"latest ai news"}'
                      }
                    }
                  ]
                }
              }
            ],
            usage: {
              prompt_tokens: 5,
              completion_tokens: 3,
              total_tokens: 8
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_stream_final_1',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'fresh answer'
                }
              }
            ],
            usage: {
              prompt_tokens: 6,
              completion_tokens: 4,
              total_tokens: 10
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const toolProvider = {
      listDefinitions: async () => [
        {
          name: 'search_web',
          description: 'Search the web',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string'
              }
            }
          }
        }
      ],
      has: async () => true,
      execute: async () => ({
        hits: ['fresh result']
      }),
      close: async () => undefined
    };

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])],
      undefined,
      [
        {
          id: 'virtual-search-stream',
          key: 'search',
          displayName: 'Search',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':search']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          instructions: {
            prepend: 'Use search_web before answering time-sensitive questions.'
          },
          tools: [
            {
              name: 'search_web',
              visibility: 'internal'
            }
          ],
          toolChoice: 'auto',
          execution: {
            mode: 'tool_loop',
            maxTurns: 4,
            maxToolCalls: 4,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config, toolProvider as any));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5:search',
          input: 'What happened today?',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(2);

      const [, firstInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const firstBody = JSON.parse(String(firstInit.body));
      expect(firstBody.stream).toBeUndefined();

      const [, secondInit] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
      const secondBody = JSON.parse(String(secondInit.body));
      expect(secondBody.stream).toBeUndefined();

      expect(response.body).toContain('"type":"response.created"');
      expect(response.body).toContain('"type":"response.output_item.added","output_index":0');
      expect(response.body).toContain('"type":"response.content_part.added","output_index":0');
      expect(response.body).toContain('"type":"response.output_text.delta","delta":"fresh answer"');
      expect(response.body).toContain('"type":"response.output_text.done","text":"fresh answer"');
      expect(response.body).toContain('"type":"response.content_part.done","output_index":0');
      expect(response.body).toContain('"type":"response.output_item.done","output_index":0');
      expect(response.body).toContain('"type":"response.completed"');
      expect(response.body).toContain('data: [DONE]');
      expect(response.body).not.toContain('search_web');
    } finally {
      await app.close();
    }
  });

  it('streams client-visible virtual model tool calls as OpenAI Responses function events', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          id: 'chatcmpl_virtual_client_tool_1',
          model: 'glm-5',
          choices: [
            {
              index: 0,
              finish_reason: 'tool_calls',
              message: {
                role: 'assistant',
                content: '',
                tool_calls: [
                  {
                    id: 'call_weather_1',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"city":"Shanghai"}'
                    }
                  }
                ]
              }
            }
          ],
          usage: {
            prompt_tokens: 7,
            completion_tokens: 4,
            total_tokens: 11
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      )
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])],
      undefined,
      [
        {
          id: 'virtual-client-tools-stream',
          key: 'client-tools',
          displayName: 'Client Tools',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':client-tools']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          tools: [],
          toolChoice: 'auto',
          execution: {
            mode: 'tool_loop',
            maxTurns: 2,
            maxToolCalls: 0,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5:client-tools',
          input: 'Check weather in Shanghai',
          stream: true,
          tools: [
            {
              type: 'function',
              name: 'get_weather',
              parameters: {
                type: 'object',
                properties: {
                  city: {
                    type: 'string'
                  }
                },
                required: ['city']
              }
            }
          ]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(response.body).toContain('"type":"response.output_item.added","output_index":0');
      expect(response.body).toContain('"type":"response.function_call_arguments.delta","output_index":0');
      expect(response.body).toContain('"type":"response.function_call_arguments.done","output_index":0');
      expect(response.body).toContain('"type":"response.output_item.done","output_index":0');
      expect(response.body).toContain('"type":"response.completed"');
      expect(response.body).toContain('"call_id":"call_weather_1"');
      expect(response.body).toContain('"name":"get_weather"');
      expect(response.body).toContain('"arguments":"{\\"city\\":\\"Shanghai\\"}"');
      expect(response.body).toContain('data: [DONE]');
    } finally {
      await app.close();
    }
  });

  it('resolves virtual model tool bindings from remote tool names to canonical MCP tool names', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_tool_2',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'tool_calls',
                message: {
                  role: 'assistant',
                  content: '',
                  tool_calls: [
                    {
                      id: 'call_search_2',
                      type: 'function',
                      function: {
                        name: 'search_web',
                        arguments: '{"query":"latest ai news"}'
                      }
                    }
                  ]
                }
              }
            ],
            usage: {
              prompt_tokens: 5,
              completion_tokens: 3,
              total_tokens: 8
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'chatcmpl_virtual_final_2',
            model: 'glm-5',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: {
                  role: 'assistant',
                  content: 'fresh answer'
                }
              }
            ],
            usage: {
              prompt_tokens: 6,
              completion_tokens: 4,
              total_tokens: 10
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'application/json'
            }
          }
        )
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const executedToolNames: string[] = [];
    const toolProvider = {
      listDefinitions: async () => [
        {
          name: 'browser.search_web',
          description: 'Search the web',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string'
              }
            }
          }
        }
      ],
      has: async () => true,
      execute: async (name: string) => {
        executedToolNames.push(name);
        return {
          hits: ['fresh result']
        };
      },
      close: async () => undefined
    };

    const config = createConfig(
      [createProviderConfig('openai-main', 'openai_chat_completions', ['glm-5'])],
      undefined,
      [
        {
          id: 'virtual-search-canonical',
          key: 'search-canonical',
          displayName: 'Search Canonical',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':search']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          tools: [
            {
              name: 'search_web',
              visibility: 'internal'
            }
          ],
          toolChoice: 'auto',
          execution: {
            mode: 'tool_loop',
            maxTurns: 4,
            maxToolCalls: 4,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config, toolProvider as any));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/glm-5:search',
          messages: [{ role: 'user', content: 'What happened today?' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(executedToolNames).toEqual(['browser.search_web']);

      const [, firstInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const firstBody = JSON.parse(String(firstInit.body));
      expect(firstBody.tools?.[0]?.function?.name || firstBody.tools?.[0]?.name).toBe('search_web');

      const payload = JSON.parse(response.body);
      expect(payload.choices[0]?.message?.content).toBe('fresh answer');
      expect(payload.choices[0]?.message?.tool_calls).toBeUndefined();
    } finally {
      await app.close();
    }
  });

  it('strips a virtual-model suffix before validating the target provider model', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_minimax_1',
          model: 'MiniMax-M2.7',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'vision answer'
              }
            }
          ],
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig(
      [createProviderConfig('Minimax', 'openai_chat_completions', ['MiniMax-M2.7'])],
      undefined,
      [
        {
          id: 'virtual-vision',
          key: 'vision',
          displayName: 'Vision',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':vision']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          tools: [],
          execution: {
            mode: 'decorate_only',
            maxTurns: 4,
            maxToolCalls: 4,
            clientToolsPolicy: 'allow',
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'Minimax'
        },
        payload: {
          model: 'MiniMax-M2.7:vision',
          messages: [{ role: 'user', content: 'Describe the image' }]
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://api.openai.com/v1/chat/completions');
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.model).toBe('MiniMax-M2.7');

      const payload = JSON.parse(response.body);
      expect(payload.choices[0]?.message?.content).toBe('vision answer');
    } finally {
      await app.close();
    }
  });

  it('extracts OpenAI chat image_url blocks as text for virtual-model multimodal matching', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_minimax_vision_1',
          model: 'MiniMax-M2.7',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'image description'
              }
            }
          ],
          usage: {
            prompt_tokens: 12,
            completion_tokens: 3,
            total_tokens: 15
          }
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const imageUrl =
      'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg';
    const config = createConfig(
      [createProviderConfig('Minimax', 'openai_chat_completions', ['MiniMax-M2.7'])],
      undefined,
      [
        {
          id: 'virtual-vision',
          key: 'vision',
          displayName: 'Vision',
          enabled: true,
          match: {
            exactAliases: [],
            prefixes: [],
            suffixes: [':vision']
          },
          baseModel: {
            mode: 'strip_suffix'
          },
          tools: [],
          execution: {
            mode: 'decorate_only',
            maxTurns: 4,
            maxToolCalls: 4,
            clientToolsPolicy: 'allow',
            matchMultimodal: true,
            streamMode: 'buffered'
          },
          materialization: {
            enabled: true,
            includeInGatewayModels: true
          }
        }
      ]
    );

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'Minimax'
        },
        payload: {
          model: 'MiniMax-M2.7:vision',
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text: 'What is in this image?'
                },
                {
                  type: 'image_url',
                  image_url: {
                    url: imageUrl
                  }
                }
              ]
            }
          ],
          max_tokens: 300
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody.model).toBe('MiniMax-M2.7');
      expect(upstreamBody.messages).toHaveLength(1);
      expect(upstreamBody.messages[0]?.role).toBe('user');
      expect(upstreamBody.messages[0]?.content).toContain('What is in this image?');
      expect(upstreamBody.messages[0]?.content).toContain(
        'Multimodal inputs available to tools. Use the media_ref value when calling tools:'
      );
      expect(upstreamBody.messages[0]?.content).toMatch(
        /- image: \[media_ref:mm_[a-f0-9]{10}\] \(url\)/
      );
    } finally {
      await app.close();
    }
  });
});

function createConfig(
  providers: ProviderConfig[],
  providerPlugins?: ProviderPluginConfig[],
  virtualModelProfiles?: GatewayConfig['virtualModelProfiles']
): GatewayConfig {
  return {
    providers,
    providerPlugins,
    virtualModelProfiles,
    defaultTargetProvider: 'openai',
    defaultTargetProviders: ['openai'],
    openaiApiKey: 'openai-test-key',
    anthropicApiKey: 'anthropic-test-key',
    geminiApiKey: 'gemini-test-key',
    openaiBaseUrl: 'https://api.openai.com/v1',
    anthropicBaseUrl: 'https://api.anthropic.com',
    geminiBaseUrl: 'https://generativelanguage.googleapis.com',
    geminiApiVersion: 'v1beta',
    upstreamTimeoutMs: 15000,
    auth: {
      enabled: false,
      mode: 'trusted_header',
      required: false,
      trustedCidrs: [],
      identityHeaders: {
        userId: 'x-auth-user-id',
        tenantId: 'x-auth-tenant-id',
        subject: 'x-auth-sub',
        organizationId: 'x-auth-organization-id',
        plan: 'x-auth-plan'
      },
      signature: {
        enabled: false,
        header: 'x-auth-signature',
        timestampHeader: 'x-auth-ts',
        secretEnv: 'AUTH_HEADER_SIGNING_SECRET',
        maxSkewSec: 120
      },
      introspection: {
        endpoint: undefined,
        timeoutMs: 3000,
        tokenHeader: 'authorization',
        tokenBearerOnly: true,
        requestTokenField: 'token',
        credentialHeader: 'x-gateway-auth',
        credentialEnv: 'AUTH_INTROSPECTION_SHARED_SECRET',
        responseMap: {
          active: 'active',
          userId: 'userId',
          tenantId: 'tenantId',
          subject: 'sub',
          organizationId: 'organizationId',
          plan: 'plan'
        }
      }
    },
    precheck: {
      enabled: false,
      rateLimit: {
        enabled: false,
        windowMs: 60000,
        maxRequests: 0,
        rpm: 0,
        rpd: 0,
        tpm: 0,
        tpd: 0,
        ipm: 0,
        limits: [],
        subject: 'identity',
        scope: 'global'
      },
      quota: {
        enabled: false,
        windowMs: 86400000,
        maxTokens: 0,
        subject: 'identity',
        scope: 'global'
      },
      budget: {
        enabled: false,
        windowMs: 86400000,
        maxCostUsd: 0,
        subject: 'identity',
        scope: 'global'
      },
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 1024
      },
      storage: {
        type: 'memory',
        failOpen: false
      }
    },
    healthAwareRouting: {
      enabled: false,
      skipUnavailable: true,
      unhealthyStatuses: ['down'],
      preferHealthy: true,
      preferLowerLatency: true
    },
    providerHealthCheck: {
      enabled: false,
      intervalMs: 60000,
      timeoutMs: 5000,
      initialDelayMs: 0
    },
    metrics: {
      enabled: false,
      includeProviderHealth: true
    },
    idempotency: {
      enabled: false,
      headerName: 'idempotency-key',
      ttlMs: 86400000,
      maxEntries: 10000,
      cacheErrorResponses: false
    },
    upstreamConcurrency: {
      enabled: false,
      maxInFlightPerProvider: 10,
      queueTimeoutMs: 1000
    },
    upstreamCircuitBreaker: {
      enabled: false,
      failureThreshold: 5,
      cooldownMs: 30000,
      failureStatusCodes: [429, 500, 502, 503, 504]
    },
    upstreamRetry: {
      enabled: true,
      maxAttempts: 2,
      baseDelayMs: 150,
      maxDelayMs: 150,
      backoffMultiplier: 1,
      jitterMs: 0,
      retryStatusCodes: []
    },
    billing: {
      enabled: false,
      currency: 'USD',
      rates: {
        openai: {
          inputPerMillionUsd: 0,
          outputPerMillionUsd: 0
        },
        anthropic: {
          inputPerMillionUsd: 0,
          outputPerMillionUsd: 0
        },
        gemini: {
          inputPerMillionUsd: 0,
          outputPerMillionUsd: 0
        }
      }
    },
    billingQueue: {
      enabled: false,
      queueName: 'gateway-billing',
      jobName: 'billing.usage',
      removeOnComplete: 1000,
      removeOnFail: 1000
    },
    billingWebhook: {
      enabled: false,
      endpoint: undefined,
      timeoutMs: 5000,
      headers: {}
    },
    rawTrace: {
      enabled: false,
      mode: 'disabled',
      spoolDir: '/tmp',
      maxPartBytes: 1024 * 1024,
      uploaderConcurrency: 1,
      maxAttempts: 1,
      baseDelayMs: 10,
      sync: {
        enabled: false,
        transport: 'http',
        endpoint: undefined,
        timeoutMs: 3000,
        apiKeyHeader: 'x-api-key',
        headers: {}
      }
    }
  } as unknown as GatewayConfig;
}

async function waitForCondition(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 1000;
  while (Date.now() < deadline) {
    if (predicate()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }

  throw new Error('Timed out waiting for condition.');
}

function createSseResponse(chunks: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(new TextEncoder().encode(chunk));
      }
      controller.close();
    }
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream; charset=utf-8'
    }
  });
}

async function waitForRawTraceManifest(
  spoolDir: string,
): Promise<{ bundleDir: string; manifest: { parts: Array<{ partType: string }> } }> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const bundles = await readdir(spoolDir);
    for (const bundle of bundles) {
      const bundleDir = join(spoolDir, bundle);
      try {
        const manifest = JSON.parse(await readFile(join(bundleDir, 'manifest.json'), 'utf8'));
        return { bundleDir, manifest };
      } catch {
        // Raw trace writing is async; keep polling until the manifest lands.
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  throw new Error('Timed out waiting for raw trace manifest.');
}

function createProviderConfig(
  name: string,
  type: ProviderConfig['type'],
  models: string[]
): ProviderConfig {
  return {
    name,
    type,
    models,
    extraHeaders: {
      default: {},
      byModel: {}
    },
    extraBody: {
      default: {},
      byModel: {}
    },
    billing: {
      byModel: {}
    }
  };
}

function createPolicyConfig(
  overrides: Partial<GatewayConfig['policy']> = {}
): GatewayConfig['policy'] {
  return {
    enabled: false,
    defaults: createPolicyRuleConfig(),
    byUser: {},
    byTenant: {},
    byOrganization: {},
    bySubject: {},
    byPlan: {},
    byApiKey: {},
    ...overrides
  };
}

function createPolicyRuleConfig(
  overrides: Partial<GatewayConfig['policy']['defaults']> = {}
): GatewayConfig['policy']['defaults'] {
  return {
    allowProviders: [],
    denyProviders: [],
    allowProviderNames: [],
    denyProviderNames: [],
    allowModels: [],
    denyModels: [],
    allowProviderModels: [],
    denyProviderModels: [],
    ...overrides
  };
}

function encryptCredentialForTest(value: string, key: Buffer, keyVersion = 'v1'): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', key, iv);
  const encrypted = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  const payload = {
    iv: iv.toString('base64'),
    tag: tag.toString('base64'),
    data: encrypted.toString('base64'),
    keyVersion
  };
  return `enc:v1:${Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url')}`;
}
