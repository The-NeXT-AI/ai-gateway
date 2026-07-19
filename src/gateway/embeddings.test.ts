import Fastify, { type FastifyRequest } from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { closeBillingPublisher, initializeBillingPublisher } from '../billing';
import type { GatewayConfig, ProviderConfig, ProviderPlugin } from '../types';
import { registerGatewayRoutes } from './routes';
import { createGatewayRuntime } from './runtime';
import { resetGatewayPrecheckStateForTests } from './precheck';
import { resetProviderCircuitBreakerForTests } from './upstream-circuit-breaker';
import { resetProviderConcurrencyForTests } from './upstream-concurrency';
import {
  decodeGatewayVideoId,
  encodeGatewayVideoId,
  resetGatewayVideoReferencesForTests
} from './video-compat';
import { applyGatewayScheduling, resetGatewaySchedulingStateForTests } from './scheduler';

describe('openai embeddings gateway route', () => {
  afterEach(async () => {
    resetGatewayVideoReferencesForTests();
    resetGatewayPrecheckStateForTests();
    resetProviderCircuitBreakerForTests();
    resetProviderConcurrencyForTests();
    resetGatewaySchedulingStateForTests();
    await closeBillingPublisher();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('routes providerName/model embeddings requests to the named OpenAI-compatible provider', async () => {
    const fetchMock = vi.fn(async () => {
      return jsonResponse({
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1, 0.2] }],
        model: 'text-embedding-3-small',
        usage: {
          prompt_tokens: 7,
          total_tokens: 7
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['text-embedding-3-small'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    provider.extraHeaders.default = {
      'x-provider-header': 'configured'
    };
    provider.extraBody.default = {
      encoding_format: 'float'
    };
    provider.billing.default = {
      inputPerMillionUsd: 0.13,
      outputPerMillionUsd: 0
    };
    const config = createConfig([provider]);
    config.billing.enabled = true;

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/text-embedding-3-small',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider']).toBe('openai');
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-main');
      expect(response.headers['x-gateway-billing-input-tokens']).toBe('7');
      expect(response.headers['x-gateway-billing-output-tokens']).toBe('0');
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://openai.example/v1/embeddings');
      expect(upstreamInit.headers).toMatchObject({
        authorization: 'Bearer provider-key',
        'x-provider-header': 'configured'
      });
      const upstreamBody = JSON.parse(String(upstreamInit.body));
      expect(upstreamBody).toMatchObject({
        model: 'text-embedding-3-small',
        input: 'hello',
        encoding_format: 'float'
      });
      expect(provider.health).toMatchObject({
        status: 'healthy',
        available: true
      });
    } finally {
      await app.close();
    }
  });

  it('keeps default text provider credentials aligned when a media provider is configured first', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        object: 'list',
        data: [],
        model: 'text-embedding-3-small',
        usage: { prompt_tokens: 1, total_tokens: 1 }
      })
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-video', [], {
        apikey: 'video-key',
        baseurl: 'https://video.example/v1',
        type: 'openai_video_generations'
      }),
      createProviderConfig('openai-text', ['text-embedding-3-small'], {
        apikey: 'text-key',
        baseurl: 'https://text.example/v1',
        type: 'openai_responses'
      })
    ]);
    config.openaiApiKey = 'text-key';
    config.openaiBaseUrl = 'https://text.example/v1';
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: { 'content-type': 'application/json' },
        payload: { model: 'text-embedding-3-small', input: 'hello' }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-text');
      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [
        string,
        RequestInit
      ];
      expect(upstreamUrl).toBe('https://text.example/v1/embeddings');
      expect(upstreamInit.headers).toMatchObject({ authorization: 'Bearer text-key' });
    } finally {
      await app.close();
    }
  });

  it('routes public provider model selectors to the matching credential-qualified OpenAI provider', async () => {
    const fetchMock = vi.fn(async () => {
      return jsonResponse({
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.3, 0.4] }],
        model: 'text-embedding-3-small',
        usage: {
          prompt_tokens: 5,
          total_tokens: 5
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const wrongProvider = createProviderConfig(
      'Other Embeddings::openai_responses::cred:test-1',
      ['other-embedding-model'],
      {
        apikey: 'wrong-key',
        baseurl: 'https://wrong.example/v1/'
      }
    );
    const targetProvider = createProviderConfig(
      'Zhipu AI (China) - Coding Plan::openai_responses::cred:test-1',
      ['text-embedding-3-small'],
      {
        apikey: 'target-key',
        baseurl: 'https://zhipu.example/v1/'
      }
    );
    const config = createConfig([wrongProvider, targetProvider]);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'Zhipu AI (China) - Coding Plan/text-embedding-3-small',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe(targetProvider.name);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://zhipu.example/v1/embeddings');
      expect(upstreamInit.headers).toMatchObject({
        authorization: 'Bearer target-key'
      });
      expect(JSON.parse(String(upstreamInit.body)).model).toBe('text-embedding-3-small');
    } finally {
      await app.close();
    }
  });

  it('rejects a concurrent OpenAI JSON request when provider concurrency is saturated', async () => {
    let resolveFetch!: (response: Response) => void;
    const fetchMock = vi.fn(async () => {
      return await new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['text-embedding-3-small'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    const config = createConfig([provider]);
    config.upstreamConcurrency = {
      enabled: true,
      maxInFlightPerProvider: 1,
      queueTimeoutMs: 1
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const request = {
      method: 'POST' as const,
      url: '/v1/embeddings',
      headers: {
        'content-type': 'application/json'
      },
      payload: {
        model: 'openai-main/text-embedding-3-small',
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
        jsonResponse({
          object: 'list',
          data: [{ object: 'embedding', index: 0, embedding: [0.1, 0.2] }],
          model: 'text-embedding-3-small',
          usage: {
            prompt_tokens: 7,
            total_tokens: 7
          }
        })
      );
      const firstResponse = await first;
      expect(firstResponse.statusCode).toBe(200);
    } finally {
      await app.close();
    }
  });

  it('does not expose the upstream error cause chain in OpenAI JSON 502 response details', async () => {
    const cause = Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:443'), {
      code: 'ECONNREFUSED'
    });
    const fetchMock = vi.fn(async () => {
      throw new TypeError('fetch failed', { cause });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['text-embedding-3-small'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    const config = createConfig([provider]);
    config.upstreamRetry.enabled = false;

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/text-embedding-3-small',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(502);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const attempt = JSON.parse(response.body).error.attempts[0];
      expect(attempt).toMatchObject({
        stage: 'upstream_connect',
        status: 502,
        message: 'Failed to reach upstream provider.'
      });
      expect(attempt).not.toHaveProperty('details');
    } finally {
      await app.close();
    }
  });

  it('opens the upstream circuit breaker for OpenAI JSON endpoints after a provider failure', async () => {
    const fetchMock = vi.fn(async () => {
      return jsonResponse({ error: { message: 'upstream unavailable' } }, 500);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['text-embedding-3-small'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    const config = createConfig([provider]);
    config.upstreamCircuitBreaker = {
      enabled: true,
      failureThreshold: 1,
      cooldownMs: 60000,
      failureStatusCodes: [500]
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const request = {
      method: 'POST' as const,
      url: '/v1/embeddings',
      headers: {
        'content-type': 'application/json'
      },
      payload: {
        model: 'openai-main/text-embedding-3-small',
        input: 'hello'
      }
    };

    try {
      const first = await app.inject(request);
      const second = await app.inject(request);

      expect(first.statusCode).toBe(500);
      expect(second.statusCode).toBe(503);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(JSON.parse(second.body).error.attempts[0]).toMatchObject({
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

  it('retries configured upstream response statuses for OpenAI JSON endpoints', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: { message: 'rate limited' } }, 429))
      .mockResolvedValueOnce(
        jsonResponse({
          object: 'list',
          data: [{ object: 'embedding', index: 0, embedding: [0.1, 0.2] }],
          model: 'text-embedding-3-small',
          usage: {
            prompt_tokens: 7,
            total_tokens: 7
          }
        })
      );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['text-embedding-3-small'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    const config = createConfig([provider]);
    config.upstreamRetry = {
      enabled: true,
      maxAttempts: 2,
      baseDelayMs: 0,
      maxDelayMs: 0,
      backoffMultiplier: 1,
      jitterMs: 0,
      retryStatusCodes: [429]
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/text-embedding-3-small',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(JSON.parse(response.body).data[0].embedding).toEqual([0.1, 0.2]);
    } finally {
      await app.close();
    }
  });

  it('falls back across named OpenAI-compatible providers', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes('primary.example')) {
        return jsonResponse({ error: { message: 'primary unavailable' } }, 500);
      }

      return jsonResponse({
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.3] }],
        model: 'embedding-model',
        usage: {
          prompt_tokens: 2,
          total_tokens: 2
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const primary = createProviderConfig('primary-openai', ['embedding-model'], {
      baseurl: 'https://primary.example/v1',
      apikey: 'primary-key'
    });
    const backup = createProviderConfig('backup-openai', ['embedding-model'], {
      baseurl: 'https://backup.example/v1',
      apikey: 'backup-key'
    });
    const config = createConfig([primary, backup]);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'primary-openai, backup-openai'
        },
        payload: {
          model: 'embedding-model',
          input: 'hi'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('backup-openai');
      expect(response.headers['x-gateway-fallback-used']).toBe('true');
      expect(response.headers['x-gateway-fallback-count']).toBe('1');
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[0]?.[0])).toBe('https://primary.example/v1/embeddings');
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://backup.example/v1/embeddings');
      expect(primary.health).toMatchObject({
        status: 'degraded',
        available: true
      });
      expect(backup.health).toMatchObject({
        status: 'healthy',
        available: true
      });
    } finally {
      await app.close();
    }
  });

  it('applies provider request and response plugins', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.headers).toMatchObject({
        'x-plugin': 'applied'
      });
      const upstreamBody = JSON.parse(String(init.body));
      expect(upstreamBody.plugin_model).toBe('embedding-model');

      return jsonResponse({
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.5] }],
        model: 'embedding-model',
        usage: {
          prompt_tokens: 3,
          total_tokens: 3
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['embedding-model'], {
        baseurl: 'https://openai.example/v1',
        apikey: 'provider-key'
      })
    ]);
    const runtime = createGatewayRuntime(config);
    const plugin: ProviderPlugin = {
      key: 'embeddings-plugin',
      providerName: 'openai-main',
      transformRequest({ upstreamRequest, standardRequest }) {
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            headers: {
              ...upstreamRequest.headers,
              'x-plugin': 'applied'
            },
            body: {
              ...(upstreamRequest.body as Record<string, unknown>),
              plugin_model: standardRequest?.model
            }
          }
        };
      },
      transformResponse({ upstreamPayload }) {
        return {
          ok: true,
          value: {
            ...(upstreamPayload as Record<string, unknown>),
            pluginHandled: true
          }
        };
      }
    };
    runtime.providerPlugins.register(plugin);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'embedding-model',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toMatchObject({
        pluginHandled: true
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('does not count default output tokens during embeddings precheck', async () => {
    const fetchMock = vi.fn(async () => {
      return jsonResponse({
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1] }],
        model: 'm',
        usage: {
          prompt_tokens: 3,
          total_tokens: 3
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['m'], {
        baseurl: 'https://openai.example/v1',
        apikey: 'provider-key'
      })
    ]);
    config.precheck.enabled = true;
    config.precheck.quota = {
      enabled: true,
      windowMs: 60000,
      maxTokens: 3,
      subject: 'global',
      scope: 'provider_model'
    };
    config.precheck.estimation = {
      charsPerToken: 1,
      defaultMaxOutputTokens: 1000
    };

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/m',
          input: 'hi'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('rejects embeddings requests denied by gateway model policy before upstream dispatch', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['blocked-embedding'], {
        baseurl: 'https://openai.example/v1',
        apikey: 'provider-key'
      })
    ]);
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        denyModels: ['blocked-*']
      })
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/blocked-embedding',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(403);
      expect(fetchMock).not.toHaveBeenCalled();
      const body = JSON.parse(response.body);
      expect(body.error.message).toBe('All target providers failed.');
      expect(body.error.attempts[0]).toMatchObject({
        provider: 'openai',
        provider_name: 'openai-main',
        stage: 'gateway_policy',
        status: 403
      });
      expect(body.error.attempts[0].details).toMatchObject({
        code: 'gateway_policy_denied',
        rule: 'defaults',
        model: 'blocked-embedding'
      });
    } finally {
      await app.close();
    }
  });
});

describe('openai moderations gateway route', () => {
  afterEach(async () => {
    resetGatewayPrecheckStateForTests();
    await closeBillingPublisher();
    vi.restoreAllMocks();
  });

  it('routes providerName/model moderation requests to the named OpenAI-compatible provider', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.headers).toMatchObject({
        authorization: 'Bearer provider-key',
        'x-provider-header': 'configured'
      });
      expect(JSON.parse(String(init.body))).toMatchObject({
        model: 'omni-moderation-latest',
        input: 'screen this text',
        metadata: {
          source: 'gateway-test'
        }
      });

      return jsonResponse({
        id: 'modr_123',
        model: 'omni-moderation-latest',
        results: [
          {
            flagged: false,
            categories: {},
            category_scores: {}
          }
        ]
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-main', ['omni-moderation-latest'], {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    provider.extraHeaders.default = {
      'x-provider-header': 'configured'
    };
    provider.extraBody.default = {
      metadata: {
        source: 'gateway-test'
      }
    };
    const config = createConfig([provider]);
    config.billing.enabled = true;

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/moderations',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/omni-moderation-latest',
          input: 'screen this text'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider']).toBe('openai');
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-main');
      expect(response.headers['x-gateway-billing-input-tokens']).toBeUndefined();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://openai.example/v1/moderations');
      expect(provider.health).toMatchObject({
        status: 'healthy',
        available: true
      });
    } finally {
      await app.close();
    }
  });

  it('uses the moderations provider plugin context', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.headers).toMatchObject({
        'x-plugin': 'moderations'
      });
      expect(JSON.parse(String(init.body))).toMatchObject({
        plugin_adapter: 'openai_moderations'
      });

      return jsonResponse({
        id: 'modr_plugin',
        model: 'omni-moderation-latest',
        results: [{ flagged: false, categories: {}, category_scores: {} }]
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['omni-moderation-latest'], {
        baseurl: 'https://openai.example/v1',
        apikey: 'provider-key'
      })
    ]);
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'moderations-plugin',
      providerName: 'openai-main',
      transformRequest({ upstreamRequest, sourceAdapterKey }) {
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            headers: {
              ...upstreamRequest.headers,
              'x-plugin': 'moderations'
            },
            body: {
              ...(upstreamRequest.body as Record<string, unknown>),
              plugin_adapter: sourceAdapterKey
            }
          }
        };
      },
      transformResponse({ upstreamPayload }) {
        return {
          ok: true,
          value: {
            ...(upstreamPayload as Record<string, unknown>),
            pluginHandled: true
          }
        };
      }
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/moderations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-main'
        },
        payload: {
          model: 'omni-moderation-latest',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toMatchObject({
        pluginHandled: true
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('rejects moderation requests denied by gateway model policy before upstream dispatch', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['omni-moderation-latest'], {
        baseurl: 'https://openai.example/v1',
        apikey: 'provider-key'
      })
    ]);
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        denyProviderModels: ['openai-main/omni-*']
      })
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/moderations',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-main/omni-moderation-latest',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(403);
      expect(fetchMock).not.toHaveBeenCalled();
      expect(JSON.parse(response.body).error.attempts[0]).toMatchObject({
        provider: 'openai',
        provider_name: 'openai-main',
        stage: 'gateway_policy',
        status: 403
      });
    } finally {
      await app.close();
    }
  });
});

describe('openai image generations gateway route', () => {
  afterEach(async () => {
    resetGatewayPrecheckStateForTests();
    await closeBillingPublisher();
    vi.restoreAllMocks();
  });

  it('routes providerName/model image generation requests and attaches usage billing when present', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.headers).toMatchObject({
        authorization: 'Bearer provider-key',
        'x-image-provider': 'configured'
      });
      expect(JSON.parse(String(init.body))).toMatchObject({
        model: 'gpt-image-1',
        prompt: 'A blue ceramic cup',
        size: '1024x1024'
      });

      return jsonResponse({
        created: 1713833628,
        data: [{ b64_json: 'abc123' }],
        usage: {
          input_tokens: 50,
          output_tokens: 150,
          total_tokens: 200
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-image', ['gpt-image-1'], {
      apikey: 'provider-key',
      baseurl: 'https://images.example/v1/'
    });
    provider.extraHeaders.default = {
      'x-image-provider': 'configured'
    };
    provider.billing.default = {
      inputPerMillionUsd: 5,
      outputPerMillionUsd: 40
    };
    const config = createConfig([provider]);
    config.billing.enabled = true;

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-image/gpt-image-1',
          prompt: 'A blue ceramic cup',
          size: '1024x1024'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider']).toBe('openai');
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-image');
      expect(response.headers['x-gateway-billing-input-tokens']).toBe('50');
      expect(response.headers['x-gateway-billing-output-tokens']).toBe('150');
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://images.example/v1/images/generations');
      expect(provider.health).toMatchObject({
        status: 'healthy',
        available: true
      });
    } finally {
      await app.close();
    }
  });

  it('does not retry or fall back a non-idempotent media create after an upstream response', async () => {
    const fetchMock = vi.fn(async (_url: string) => jsonResponse({ error: 'busy' }, 503));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('image-a', ['gpt-image-1'], {
        baseurl: 'https://image-a.example/v1',
        type: 'openai_image_generations'
      }),
      createProviderConfig('image-b', ['gpt-image-1'], {
        baseurl: 'https://image-b.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.upstreamRetry = {
      ...config.upstreamRetry,
      enabled: true,
      maxAttempts: 3,
      retryStatusCodes: [503]
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'image-a,image-b'
        },
        payload: { model: 'gpt-image-1', prompt: 'Create exactly one image' }
      });

      expect(response.statusCode).toBe(503);
      expect(JSON.parse(response.body)).toEqual({ error: 'busy' });
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(String(fetchMock.mock.calls[0]?.[0])).toContain('image-a.example');
    } finally {
      await app.close();
    }
  });

  it('fails closed when image model deny rules cannot evaluate an omitted model', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', [], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({ denyModels: ['gpt-image-*'] })
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: { 'content-type': 'application/json' },
        payload: { prompt: 'Do not let the upstream choose a denied model' }
      });

      expect(response.statusCode, response.body).toBe(403);
      expect(JSON.parse(response.body).error.attempts[0]).toMatchObject({
        provider_name: 'openai-image',
        stage: 'gateway_policy'
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('does not replay a media create after an ambiguous transport failure', async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error('socket closed after request upload');
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('image-a', ['gpt-image-1'], {
        baseurl: 'https://image-a.example/v1',
        type: 'openai_image_generations'
      }),
      createProviderConfig('image-b', ['gpt-image-1'], {
        baseurl: 'https://image-b.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'image-a,image-b'
        },
        payload: { model: 'gpt-image-1', prompt: 'Create exactly one image' }
      });

      expect(response.statusCode).toBe(502);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('passes image edit event streams through without buffering them into JSON', async () => {
    const sse = 'event: image_generation.partial_image\ndata: {"b64_json":"abc"}\n\n';
    const fetchMock = vi.fn(async () =>
      new Response(sse, {
        headers: { 'content-type': 'text/event-stream; charset=utf-8' }
      })
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'gpt-image-1',
          prompt: 'Stream the edit',
          image: { image_url: 'data:image/png;base64,abc' },
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.body).toBe(sse);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('publishes billing usage from completed image event streams', async () => {
    const billingEvents: Array<Record<string, unknown>> = [];
    const largeImagePayload = 'a'.repeat(300 * 1024);
    const sse = [
      'event: image_generation.partial_image',
      `data: {"type":"image_generation.partial_image","b64_json":"${largeImagePayload}","partial_image_index":0}`,
      '',
      'event: image_generation.completed',
      `data: {"type":"image_generation.completed","b64_json":"${largeImagePayload}","usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}`,
      '',
      ''
    ].join('\n');
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url === 'http://billing.local/events') {
        billingEvents.push(JSON.parse(String(init.body)) as Record<string, unknown>);
        return new Response(null, { status: 204 });
      }
      return new Response(sse, {
        headers: { 'content-type': 'text/event-stream; charset=utf-8' }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-image', ['gpt-image-1'], {
      apikey: 'provider-key',
      baseurl: 'https://images.example/v1',
      type: 'openai_image_generations'
    });
    provider.billing.default = {
      inputPerMillionUsd: 1,
      outputPerMillionUsd: 2
    };
    const config = createConfig([provider]);
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
      headers: {}
    };
    await initializeBillingPublisher(config.billingQueue, config.billingWebhook);

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'gpt-image-1',
          prompt: 'Stream an image',
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.body).toBe(sse);
      await waitForCondition(() => billingEvents.length === 1);
      expect(billingEvents[0]?.billing).toMatchObject({
        usage: {
          input_tokens: 10,
          output_tokens: 20,
          total_tokens: 30
        },
        cost: { total: 0.00005 }
      });
    } finally {
      await app.close();
    }
  });

  it('preserves missing image generation model instead of injecting defaultOpenAIModel', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const upstreamBody = JSON.parse(String(init.body));
      expect(upstreamBody).toMatchObject({
        prompt: 'A monochrome logo'
      });
      expect(upstreamBody.model).toBeUndefined();

      return jsonResponse({
        created: 1713833628,
        data: [{ url: 'https://example.test/image.png' }]
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'provider-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.defaultOpenAIModel = 'not-an-image-model';

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-image'
        },
        payload: {
          prompt: 'A monochrome logo'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-image');
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('applies image generation policy before upstream dispatch', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        baseurl: 'https://images.example/v1',
        apikey: 'provider-key'
      })
    ]);
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        denyProviderModels: ['openai-image/gpt-image-*']
      })
    });

    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: {
          'content-type': 'application/json'
        },
        payload: {
          model: 'openai-image/gpt-image-1',
          prompt: 'A policy blocked image'
        }
      });

      expect(response.statusCode).toBe(403);
      expect(fetchMock).not.toHaveBeenCalled();
      expect(JSON.parse(response.body).error.attempts[0]).toMatchObject({
        stage: 'gateway_policy',
        status: 403
      });
    } finally {
      await app.close();
    }
  });
});

describe('openai media gateway routes', () => {
  afterEach(async () => {
    resetGatewayVideoReferencesForTests();
    resetGatewayPrecheckStateForTests();
    resetProviderCircuitBreakerForTests();
    resetProviderConcurrencyForTests();
    resetGatewaySchedulingStateForTests();
    await closeBillingPublisher();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('hashes binary scheduling bodies without enumerating their properties', () => {
    const provider = createProviderConfig('openai-image', ['gpt-image-1'], {
      type: 'openai_image_generations'
    });
    provider.credentials = [
      {
        id: 'image-account',
        apikey: 'image-key',
        enabled: true,
        priority: 1,
        weight: 1
      }
    ];
    const config = createConfig([provider]);
    enableGatewayScheduling(config);
    const body = Buffer.from('binary-image-data');
    Object.defineProperty(body, 'unexpected-enumeration', {
      enumerable: true,
      get() {
        throw new Error('binary properties must not be enumerated');
      }
    });
    const request = {
      body,
      headers: {
        'x-gateway-cache-affinity-key': 'binary-request'
      }
    } as unknown as FastifyRequest;

    expect(() =>
      applyGatewayScheduling(
        [{ provider: 'openai' as const, providerConfig: provider }],
        { config, request, requestModel: 'gpt-image-1' }
      )
    ).not.toThrow();
  });

  it('routes JSON and multipart image edits without changing multipart bytes', async () => {
    const upstreamBodies: Array<RequestInit['body']> = [];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://images.example/v1/images/edits');
      expect(init.headers).toMatchObject({ authorization: 'Bearer provider-key' });
      upstreamBodies.push(init.body);
      return jsonResponse({ data: [{ url: 'https://example.test/edited.png' }] });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'provider-key',
        baseurl: 'https://images.example/v1'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const jsonResponseResult = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-image/gpt-image-1',
          prompt: 'Add a blue frame',
          image: { url: 'data:image/png;base64,abc123' }
        }
      });
      expect(jsonResponseResult.statusCode).toBe(200);
      expect(JSON.parse(String(upstreamBodies[0]))).toMatchObject({
        model: 'gpt-image-1',
        prompt: 'Add a blue frame'
      });

      const boundary = 'gateway-media-test-boundary';
      const multipartPayload = Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\ngpt-image-1\r\n--${boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nAdd a red frame\r\n--${boundary}--\r\n`
      );
      const multipartResponse = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`,
          'x-target-model': 'openai-image/gpt-image-1'
        },
        payload: multipartPayload
      });
      expect(multipartResponse.statusCode).toBe(200);
      expect(Buffer.from(upstreamBodies[1] as ArrayBuffer)).toEqual(multipartPayload);
    } finally {
      await app.close();
    }
  });

  it('counts JSON image edit file references during precheck', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'provider-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.precheck.enabled = true;
    config.precheck.rateLimit = {
      ...config.precheck.rateLimit,
      enabled: true,
      limits: [
        {
          enabled: true,
          name: 'ipm',
          metric: 'images',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ],
      subject: 'global',
      scope: 'global'
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'gpt-image-1',
          prompt: 'Replace the background',
          images: [{ file_id: 'file-source' }],
          mask: { file_id: 'file-mask' }
        }
      });

      expect(response.statusCode, response.body).toBe(429);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'rate_limit_exceeded',
        details: {
          metric: 'images',
          requested: 2,
          estimated: { imageCount: 2 }
        }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('uses multipart image metadata when enforcing scheduled credential limits', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ data: [{ url: 'https://example.test/edited.png' }] })
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-image', ['gpt-image-1'], {
      baseurl: 'https://images.example/v1',
      type: 'openai_image_generations'
    });
    provider.credentials = [
      {
        id: 'preferred-account',
        apikey: 'preferred-key',
        enabled: true,
        priority: 1,
        weight: 1,
        limits: { ipm: 1 }
      },
      {
        id: 'spillover-account',
        apikey: 'spillover-key',
        enabled: true,
        priority: 2,
        weight: 1,
        limits: { ipm: 10 }
      }
    ];
    const config = createConfig([provider]);
    enableGatewayScheduling(config);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-scheduling-multipart-boundary';
    const multipartPayload = buildMultipartImageEditPayload(boundary, {
      model: 'gpt-image-1',
      prompt: 'Use parsed multipart metadata for limits'
    });
    const sendRequest = () => app.inject({
      method: 'POST',
      url: '/v1/images/edits',
      headers: {
        'content-type': `multipart/form-data; boundary=${boundary}`,
        'x-target-provider': 'openai-image'
      },
      payload: multipartPayload
    });

    try {
      const first = await sendRequest();
      const second = await sendRequest();

      expect(first.statusCode, first.body).toBe(200);
      expect(first.headers['x-gateway-scheduled-credential-id']).toBe('preferred-account');
      expect(second.statusCode, second.body).toBe(200);
      expect(second.headers['x-gateway-scheduled-credential-id']).toBe('spillover-account');
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      await app.close();
    }
  });

  it('uses JSON image edit file references when enforcing scheduled credential limits', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ data: [{ url: 'https://example.test/edited.png' }] })
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-image', ['gpt-image-1'], {
      baseurl: 'https://images.example/v1',
      type: 'openai_image_generations'
    });
    provider.credentials = [
      {
        id: 'preferred-account',
        apikey: 'preferred-key',
        enabled: true,
        priority: 1,
        weight: 1,
        limits: { ipm: 1 }
      },
      {
        id: 'spillover-account',
        apikey: 'spillover-key',
        enabled: true,
        priority: 2,
        weight: 1,
        limits: { ipm: 10 }
      }
    ];
    const config = createConfig([provider]);
    enableGatewayScheduling(config);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const sendRequest = () =>
      app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-image'
        },
        payload: {
          model: 'gpt-image-1',
          prompt: 'Use file references for limits',
          images: [{ file_id: 'file-source' }]
        }
      });

    try {
      const first = await sendRequest();
      const second = await sendRequest();

      expect(first.statusCode, first.body).toBe(200);
      expect(first.headers['x-gateway-scheduled-credential-id']).toBe('preferred-account');
      expect(second.statusCode, second.body).toBe(200);
      expect(second.headers['x-gateway-scheduled-credential-id']).toBe('spillover-account');
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      await app.close();
    }
  });

  it('returns a committed video create response when a response plugin fails', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        id: 'video-committed-upstream',
        object: 'video',
        status: 'queued',
        model: 'sora-2'
      })
    );
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'provider-key',
        baseurl: 'https://videos.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.media = {
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'broken-video-response-transform',
      providerName: 'openai-video',
      transformResponse() {
        return { ok: false, error: 'cannot transform the committed response' };
      }
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'Create this exactly once',
          seconds: '4',
          size: '1280x720'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(response.headers['x-gateway-provider-response-transform']).toBe(
        'bypassed-after-upstream-commit'
      );
      expect(JSON.parse(response.body).id).toMatch(/^gv3\./);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('schedules media credentials and pins video follow-up requests to the creating credential', async () => {
    const authorizationHeaders: string[] = [];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      authorizationHeaders.push(
        String((init.headers as Record<string, string>).authorization || '')
      );
      if (url.endsWith('/videos')) {
        return jsonResponse({
          id: 'video-scheduled-upstream',
          object: 'video',
          status: 'queued',
          model: 'sora-2'
        });
      }
      if (url.endsWith('/content')) {
        return new Response('scheduled-video-bytes', {
          headers: { 'content-type': 'video/mp4' }
        });
      }
      return jsonResponse({
        id: 'video-scheduled-upstream',
        object: 'video',
        status: 'completed',
        model: 'sora-2'
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-video', ['sora-2'], {
      baseurl: 'https://videos.example/v1',
      type: 'openai_video_generations'
    });
    provider.credentials = [
      {
        id: 'account-a',
        apikey: 'credential-a',
        enabled: true,
        priority: 1,
        weight: 1
      },
      {
        id: 'account-b',
        apikey: 'credential-b',
        enabled: true,
        priority: 1,
        weight: 1
      }
    ];
    const config = createConfig([provider]);
    enableGatewayScheduling(config);
    config.media = {
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'Keep this task on its account',
          seconds: '4',
          size: '1280x720'
        }
      });
      expect(createResponse.statusCode, createResponse.body).toBe(200);
      expect(createResponse.headers['x-gateway-scheduled-credential-id']).toBe('account-a');
      const publicId = JSON.parse(createResponse.body).id as string;

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}`
      });
      expect(statusResponse.statusCode, statusResponse.body).toBe(200);
      expect(statusResponse.headers['x-gateway-scheduled-credential-id']).toBe('account-a');

      const contentResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}/content`
      });
      expect(contentResponse.statusCode, contentResponse.body).toBe(200);
      expect(contentResponse.headers['x-gateway-scheduled-credential-id']).toBe('account-a');
      expect(authorizationHeaders).toEqual([
        'Bearer credential-a',
        'Bearer credential-a',
        'Bearer credential-a'
      ]);
    } finally {
      await app.close();
    }
  });

  it('uses multipart model metadata to select the media-capable provider', async () => {
    let upstreamBody: RequestInit['body'];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://images.example/v1/images/edits');
      upstreamBody = init.body;
      return jsonResponse({ data: [{ url: 'https://example.test/edited.png' }] });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-main', ['gpt-5'], {
        apikey: 'main-key',
        baseurl: 'https://main.example/v1',
        type: 'openai_responses'
      }),
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-provider-selection-boundary';
    const multipartPayload = buildMultipartImageEditPayload(boundary, {
      model: 'gpt-image-1',
      prompt: 'Route using the standard multipart model field'
    });
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`
        },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-image');
      expect(Buffer.from(upstreamBody as ArrayBuffer)).toEqual(multipartPayload);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('rejects multipart model metadata that differs from the governed target model', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1', 'gpt-image-expensive'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-model-conflict-boundary';
    const multipartPayload = buildMultipartImageEditPayload(boundary, {
      model: 'gpt-image-expensive',
      prompt: 'Do not let this bypass model governance'
    });
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`,
          'x-target-model': 'openai-image/gpt-image-1'
        },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('Multipart model must resolve');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('rejects non-canonical multipart control field names before forwarding bytes', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-uppercase-model-boundary';
    const multipartPayload = Buffer.from(
      [
        `--${boundary}\r\nContent-Disposition: form-data; name="MODEL"\r\n\r\ngpt-image-1\r\n`,
        `--${boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nReject ambiguous governance\r\n`,
        `--${boundary}--\r\n`
      ].join('')
    );
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('exact lowercase name');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('rejects ambiguous multipart disposition parameters', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-ambiguous-disposition-boundary';
    const multipartPayload = Buffer.from(
      [
        `--${boundary}\r\nContent-Disposition: form-data; name="model"; name="prompt"\r\n\r\ngpt-image-1\r\n`,
        `--${boundary}--\r\n`
      ].join('')
    );
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('parameter \\"name\\" must not be repeated');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('keeps the multipart parser isolated from JSON-only routes', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-main', ['text-embedding-3-small'], {
        apikey: 'provider-key',
        baseurl: 'https://openai.example/v1'
      })
    ]);
    config.defaultOpenAIModel = 'text-embedding-3-small';
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/embeddings',
        headers: {
          'content-type': 'multipart/form-data; boundary=not-for-embeddings'
        },
        payload: Buffer.from('--not-for-embeddings--\r\n')
      });

      expect(response.statusCode).toBe(415);
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('uses multipart prompt and image metadata during precheck', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.precheck.enabled = true;
    config.precheck.quota = {
      enabled: true,
      windowMs: 60_000,
      maxTokens: 4,
      subject: 'global',
      scope: 'global'
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-precheck-boundary';
    const multipartPayload = buildMultipartImageEditPayload(boundary, {
      model: 'gpt-image-1',
      prompt: 'This multipart prompt is intentionally long enough to exceed the configured quota.'
    });
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`
        },
        payload: multipartPayload
      });

      expect(response.statusCode, response.body).toBe(429);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'quota_exceeded',
        details: {
          estimated: {
            imageCount: 1
          }
        }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('preserves binary multipart bodies when a non-strict plugin declares body mutations', async () => {
    let upstreamBody: RequestInit['body'];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.headers).toMatchObject({ 'x-plugin-applied': 'true' });
      upstreamBody = init.body;
      return jsonResponse({ data: [{ url: 'https://example.test/edited.png' }] });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    config.providerPlugins = [
      {
        key: 'multipart-safe-plugin',
        enabled: true,
        providerName: 'openai-image',
        request: {
          strict: false,
          headers: { 'x-plugin-applied': 'true' },
          query: {},
          removeHeaders: [],
          removeQuery: [],
          bodySet: { 'metadata.source': 'plugin' },
          bodyMerge: {},
          bodyRemove: []
        }
      }
    ];
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'gateway-plugin-boundary';
    const multipartPayload = buildMultipartImageEditPayload(boundary, {
      model: 'gpt-image-1',
      prompt: 'Keep these exact bytes'
    });
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/edits',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`
        },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(200);
      expect(Buffer.from(upstreamBody as ArrayBuffer)).toEqual(multipartPayload);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('includes the signed video model in status-request auth introspection', async () => {
    const introspectionBodies: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url === 'https://auth.example/introspect') {
        introspectionBodies.push(JSON.parse(String(init.body)) as Record<string, unknown>);
        return jsonResponse({ active: true, userId: 'video-user' });
      }
      if (url.endsWith('/videos')) {
        return jsonResponse({
          id: 'upstream-video-id',
          object: 'video',
          status: 'queued',
          model: 'sora-2',
          progress: 0,
          seconds: '4',
          size: '720x1280'
        });
      }
      expect(url).toBe('https://video.example/v1/videos/upstream-video-id');
      return jsonResponse({
        id: 'upstream-video-id',
        object: 'video',
        status: 'completed',
        model: 'sora-2',
        progress: 100,
        seconds: '4',
        size: '720x1280'
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-video', [], {
        apikey: 'provider-key',
        baseurl: 'https://video.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.auth = {
      ...config.auth,
      enabled: true,
      mode: 'http_introspection',
      required: true,
      introspection: {
        ...config.auth.introspection,
        endpoint: 'https://auth.example/introspect'
      }
    };
    config.media = {
      videoIdSigningSecret: 'video-introspection-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          authorization: 'Bearer client-token',
          'content-type': 'application/json'
        },
        payload: { model: 'sora-2', prompt: 'A calm ocean' }
      });
      expect(createResponse.statusCode, createResponse.body).toBe(200);
      const publicId = JSON.parse(createResponse.body).id as string;

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}`,
        headers: { authorization: 'Bearer client-token' }
      });

      expect(statusResponse.statusCode, statusResponse.body).toBe(200);
      expect(introspectionBodies).toHaveLength(2);
      expect(introspectionBodies[1]).toMatchObject({
        model: 'sora-2',
        models: ['sora-2']
      });

      const conflictingResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}`,
        headers: {
          authorization: 'Bearer client-token',
          'x-target-model': 'cheap-video-model'
        }
      });
      expect(conflictingResponse.statusCode, conflictingResponse.body).toBe(400);
      expect(conflictingResponse.body).toContain('conflicts with the signed video model');
      expect(introspectionBodies[2]).toMatchObject({
        model: 'sora-2',
        models: ['sora-2', 'cheap-video-model']
      });
    } finally {
      await app.close();
    }
  });

  it('routes video generation and stateless status requests to the selected provider', async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url.endsWith('/videos/generations')) {
        expect(init.method).toBe('POST');
        expect(JSON.parse(String(init.body))).toMatchObject({
          model: 'grok-imagine-video',
          prompt: 'A paper boat sailing'
        });
        return jsonResponse({ request_id: 'video-request-1', status: 'pending' });
      }
      if (url.endsWith('/videos/existing-xai-request')) {
        expect(init.method).toBe('GET');
        return jsonResponse({ status: 'pending', model: 'grok-imagine-video' });
      }
      expect(url).toBe('https://videos.example/v1/videos/video-request-1');
      expect(init.method).toBe('GET');
      expect(init.body).toBeUndefined();
      return jsonResponse({
        status: 'done',
        video: { url: 'https://example.test/video.mp4', duration: 8 },
        model: 'grok-imagine-video',
        progress: 100
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'provider-key',
        baseurl: 'https://videos.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const startResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai_video_generations'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'A paper boat sailing'
        }
      });
      expect(startResponse.statusCode).toBe(200);
      expect(startResponse.headers['x-gateway-target-provider-name']).toBe('xai-video');
      const requestId = JSON.parse(startResponse.body).request_id as string;
      expect(requestId).toMatch(/^gv3\./);

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(requestId)}`
      });
      expect(statusResponse.statusCode).toBe(200);
      expect(JSON.parse(statusResponse.body)).toMatchObject({
        status: 'done',
        model: 'grok-imagine-video',
        video: { url: 'https://example.test/video.mp4' }
      });

      const existingStatusResponse = await app.inject({
        method: 'GET',
        url: '/v1/videos/existing-xai-request',
        headers: { 'x-target-provider': 'xai_video_generations' }
      });
      expect(existingStatusResponse.statusCode).toBe(200);
      expect(JSON.parse(existingStatusResponse.body)).toEqual({
        status: 'pending',
        model: 'grok-imagine-video'
      });
      expect(fetchMock).toHaveBeenCalledTimes(3);
    } finally {
      await app.close();
    }
  });

  it('preserves conditional and range response semantics for video content', async () => {
    const upstreamVideoId = 'video-conditional-content';
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url.endsWith('/videos')) {
        return jsonResponse({
          id: upstreamVideoId,
          object: 'video',
          status: 'completed',
          model: 'sora-2'
        });
      }
      expect(url).toBe(`https://openai.example/v1/videos/${upstreamVideoId}/content`);
      const headers = init.headers as Record<string, string>;
      if (headers.range) {
        return new Response('requested range is outside the asset', {
          status: 416,
          headers: {
            'content-type': 'text/plain',
            'content-range': 'bytes */128',
            etag: '"video-etag"'
          }
        });
      }
      expect(headers['if-none-match']).toBe('"video-etag"');
      return new Response(null, {
        status: 304,
        headers: {
          etag: '"video-etag"',
          'cache-control': 'private, max-age=60'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': 'application/json' },
        payload: { model: 'sora-2', prompt: 'A cached video asset' }
      });
      expect(createResponse.statusCode, createResponse.body).toBe(200);
      const publicId = JSON.parse(createResponse.body).id as string;

      const rangeResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}/content`,
        headers: { range: 'bytes=999-1000' }
      });
      expect(rangeResponse.statusCode).toBe(416);
      expect(rangeResponse.headers['content-range']).toBe('bytes */128');
      expect(rangeResponse.headers.etag).toBe('"video-etag"');
      expect(rangeResponse.body).toBe('requested range is outside the asset');

      const notModifiedResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}/content`,
        headers: { 'if-none-match': '"video-etag"' }
      });
      expect(notModifiedResponse.statusCode).toBe(304);
      expect(notModifiedResponse.headers.etag).toBe('"video-etag"');
      expect(notModifiedResponse.headers['cache-control']).toBe('private, max-age=60');
      expect(notModifiedResponse.body).toBe('');
      expect(fetchMock).toHaveBeenCalledTimes(3);
    } finally {
      await app.close();
    }
  });

  it('uses the xAI default base URL when a video provider omits baseurl', async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://api.x.ai/v1/videos/generations');
      expect(init.headers).toMatchObject({ authorization: 'Bearer provider-xai-key' });
      return jsonResponse({ request_id: 'xai-default-base-request' });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'provider-xai-key',
        type: 'xai_video_generations'
      })
    ]);
    config.openaiBaseUrl = 'https://must-not-receive-xai-traffic.example/v1';
    config.openaiApiKey = 'must-not-reach-xai';
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai-video'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Use the protocol default endpoint'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('uses XAI_API_KEY instead of an OpenAI credential when provider apikey is omitted', async () => {
    vi.stubEnv('XAI_API_KEY', 'xai-env-key');
    vi.stubEnv('OPENAI_API_KEY', 'openai-env-key');
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://xai.example/v1/videos/generations');
      expect(init.headers).toMatchObject({ authorization: 'Bearer xai-env-key' });
      expect(init.headers).not.toHaveProperty('openai-organization');
      expect(init.headers).not.toHaveProperty('openai-project');
      return jsonResponse({ request_id: 'xai-env-key-request' });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.openaiApiKey = 'openai-config-key';
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          authorization: 'Bearer client-openai-key',
          'content-type': 'application/json',
          'openai-organization': 'org-must-not-reach-xai',
          'openai-project': 'project-must-not-reach-xai',
          'x-target-provider': 'xai-video'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Use the xAI environment credential'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('does not fall back to an OpenAI key when xAI credentials are missing', async () => {
    vi.stubEnv('XAI_API_KEY', '');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.openaiApiKey = 'openai-config-key';
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai-video'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Reject the wrong credential family'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('XAI_API_KEY is missing');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('does not forward a Gateway static auth token as an xAI credential', async () => {
    vi.stubEnv('XAI_API_KEY', '');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.auth = {
      ...config.auth,
      enabled: true,
      required: true,
      mode: 'static_api_key',
      staticApiKeys: {
        keys: ['gateway-static-key'],
        keyHeader: 'authorization',
        keyBearerOnly: true
      }
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          authorization: 'Bearer gateway-static-key',
          'content-type': 'application/json',
          'x-target-provider': 'xai-video'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Never forward the Gateway credential'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('XAI_API_KEY is missing');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('prefers a dedicated media provider over an unrelated default provider family', async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://videos.example/v1/videos/generations');
      expect(JSON.parse(String(init.body))).toMatchObject({
        model: 'grok-imagine-video',
        prompt: 'A dedicated media route'
      });
      return jsonResponse({ request_id: 'dedicated-video-request', status: 'pending' });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-text', ['gpt-5'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_responses'
      }),
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://videos.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    expect(config.defaultTargetProviders).toEqual(['openai']);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'A dedicated media route'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('xai-video');
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('keeps a generic default provider eligible when dedicated media providers do not serve the model', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      expect(url).toBe('https://openai.example/v1/images/generations');
      return jsonResponse({ data: [{ url: 'https://example.test/generated.png' }] });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-generic', ['dall-e-3'], {
        apikey: 'generic-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_responses'
      }),
      createProviderConfig('openai-image', ['gpt-image-1'], {
        apikey: 'image-key',
        baseurl: 'https://images.example/v1',
        type: 'openai_image_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/images/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'dall-e-3',
          prompt: 'Use the matching generic provider'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(response.headers['x-gateway-target-provider-name']).toBe('openai-generic');
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('rejects video status access from a different authenticated identity', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ request_id: 'identity-video' }));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://videos.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.auth.enabled = true;
    config.auth.required = true;
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-auth-user-id': 'user-a',
          'x-target-provider': 'xai_video_generations'
        },
        payload: { model: 'grok-imagine-video', prompt: 'Private video' }
      });
      expect(createResponse.statusCode).toBe(200);
      const requestId = JSON.parse(createResponse.body).request_id as string;

      const response = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(requestId)}`,
        headers: { 'x-auth-user-id': 'user-b' }
      });
      expect(response.statusCode).toBe(403);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      const rawIdResponse = await app.inject({
        method: 'GET',
        url: '/v1/videos/raw-upstream-video-id',
        headers: {
          'x-auth-user-id': 'user-a',
          'x-target-provider': 'xai_video_generations'
        }
      });
      expect(rawIdResponse.statusCode).toBe(403);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('rejects legacy signed video ids without an owner when authentication is enabled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.auth.enabled = true;
    config.auth.required = true;
    config.media = {
      videoIdSigningSecret: 'owner-test-secret',
      videoIdTtlMs: 86_400_000
    };
    const ownerlessId = encodeGatewayVideoId(
      {
        version: 2,
        upstreamId: 'legacy-ownerless-request',
        sourceProtocol: 'xai',
        targetProtocol: 'xai',
        targetProvider: 'xai',
        targetProviderName: 'xai-video',
        model: 'grok-imagine-video',
        createdAt: Math.floor(Date.now() / 1000)
      },
      { signingSecret: config.media.videoIdSigningSecret, ttlMs: config.media.videoIdTtlMs }
    );
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(ownerlessId)}`,
        headers: { 'x-auth-user-id': 'user-a' }
      });

      expect(response.statusCode).toBe(403);
      expect(response.body).toContain('different identity');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('requires an identity before creating video ids when optional auth is enabled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.auth.enabled = true;
    config.auth.required = false;
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai-video'
        },
        payload: { model: 'grok-imagine-video', prompt: 'Owner is required' }
      });

      expect(response.statusCode).toBe(403);
      expect(response.body).toContain('authenticated identity');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('does not override an explicit video provider with a cross-provider match', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      }),
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Keep this request on the explicitly selected provider',
          seconds: '4',
          size: '1280x720'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(JSON.parse(response.body).error.attempts).toEqual([
        expect.objectContaining({
          provider: 'openai',
          provider_name: 'openai-video',
          stage: 'model_resolution'
        })
      ]);
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('fails closed when video policy needs a model that cannot be inferred', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-video', [], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        denyModels: ['restricted-video-model']
      })
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'openai-video'
        },
        payload: {
          prompt: 'Do not bypass model policy when the upstream default is unknown',
          seconds: '4',
          size: '1280x720'
        }
      });

      expect(response.statusCode).toBe(403);
      expect(JSON.parse(response.body).error.attempts[0]).toMatchObject({
        provider_name: 'openai-video',
        stage: 'gateway_policy',
        message: 'Model is missing, so model deny rules cannot be evaluated safely.'
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('runs video budget precheck for the first locally dispatchable fallback', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const xaiProvider = createProviderConfig('xai-video', [], {
      apikey: 'xai-key',
      baseurl: 'https://xai.example/v1',
      type: 'xai_video_generations'
    });
    xaiProvider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0
    };
    const openAIProvider = createProviderConfig('openai-video', ['sora-2'], {
      apikey: 'openai-key',
      baseurl: 'https://openai.example/v1',
      type: 'openai_video_generations'
    });
    openAIProvider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.05
    };
    const config = createConfig([xaiProvider, openAIProvider]);
    config.precheck.enabled = true;
    config.precheck.budget = {
      enabled: true,
      windowMs: 60_000,
      maxCostUsd: 0.2,
      subject: 'global',
      scope: 'global'
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'xai-video,openai-video'
        },
        payload: {
          model: 'sora-2',
          prompt: 'Charge the provider that can actually accept this request',
          seconds: '16',
          size: '1280x720'
        }
      });

      expect(response.statusCode, response.body).toBe(402);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'budget_exceeded',
        details: {
          estimated: {
            videoSeconds: 16,
            estimatedCostUsd: 0.8
          }
        }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('runs video budget precheck after provider plugins select the actual fallback', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const xaiProvider = createProviderConfig('xai-video', ['shared-video'], {
      apikey: 'xai-key',
      baseurl: 'https://xai.example/v1',
      type: 'xai_video_generations'
    });
    xaiProvider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0
    };
    const openAIProvider = createProviderConfig('openai-video', ['shared-video'], {
      apikey: 'openai-key',
      baseurl: 'https://openai.example/v1',
      type: 'openai_video_generations'
    });
    openAIProvider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.05
    };
    const config = createConfig([xaiProvider, openAIProvider]);
    config.precheck.enabled = true;
    config.precheck.budget = {
      enabled: true,
      windowMs: 60_000,
      maxCostUsd: 0.2,
      subject: 'global',
      scope: 'global'
    };
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'reject-xai-auth',
      providerName: 'xai-video',
      authenticate() {
        return { ok: false, error: 'xAI credentials are unavailable' };
      }
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-providers': 'xai-video,openai-video'
        },
        payload: {
          model: 'shared-video',
          prompt: 'Use the first provider that can really dispatch',
          seconds: '8',
          size: '1280x720'
        }
      });

      expect(response.statusCode, response.body).toBe(402);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'budget_exceeded',
        details: { estimated: { videoSeconds: 8, estimatedCostUsd: 0.4 } }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('uses the final provider-transformed video request for budget precheck', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const provider = createProviderConfig('openai-video', ['sora-2', 'sora-2-pro'], {
      apikey: 'openai-key',
      baseurl: 'https://openai.example/v1',
      type: 'openai_video_generations'
    });
    provider.extraBody.default = {
      seconds: '12',
      size: '1024x1792'
    };
    provider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.1
    };
    provider.billing.byModel['sora-2-pro'] = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.3,
      videoPerSecondUsdBySize: { '1024x1792': 0.5 }
    };
    const config = createConfig([provider]);
    config.precheck.enabled = true;
    config.precheck.budget = {
      enabled: true,
      windowMs: 60_000,
      maxCostUsd: 2,
      subject: 'global',
      scope: 'global'
    };
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'select-pro-video-model',
      providerName: 'openai-video',
      transformRequest({ upstreamRequest }) {
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            body: {
              ...(upstreamRequest.body as Record<string, unknown>),
              model: 'sora-2-pro'
            }
          }
        };
      }
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'sora-2',
          prompt: 'Use the final transformed media parameters',
          seconds: '4',
          size: '1280x720'
        }
      });

      expect(response.statusCode, response.body).toBe(402);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'budget_exceeded',
        details: {
          estimated: {
            videoSeconds: 12,
            estimatedCostUsd: 6
          }
        }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('uses final video parameters for billing headers and signed request metadata', async () => {
    const upstreamVideoId = 'video_final_parameters';
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(JSON.parse(String(init.body))).toMatchObject({
        model: 'sora-2-pro',
        seconds: '12',
        size: '1024x1792'
      });
      return jsonResponse({
        id: upstreamVideoId,
        object: 'video',
        status: 'queued',
        model: 'sora-2-pro'
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const provider = createProviderConfig('openai-video', ['sora-2', 'sora-2-pro'], {
      apikey: 'openai-key',
      baseurl: 'https://openai.example/v1',
      type: 'openai_video_generations'
    });
    provider.extraBody.default = {
      seconds: '12',
      size: '1024x1792'
    };
    provider.billing.byModel['sora-2-pro'] = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.3,
      videoPerSecondUsdBySize: { '1024x1792': 0.5 }
    };
    const config = createConfig([provider]);
    config.billing.enabled = true;
    config.media = {
      videoIdSigningSecret: 'final-parameters-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'select-pro-video-model',
      providerName: 'openai-video',
      transformRequest({ upstreamRequest }) {
        return {
          ok: true,
          value: {
            ...upstreamRequest,
            body: {
              ...(upstreamRequest.body as Record<string, unknown>),
              model: 'sora-2-pro'
            }
          }
        };
      }
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'sora-2',
          prompt: 'Bill the final request',
          seconds: '4',
          size: '1280x720'
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      expect(response.headers['x-gateway-billing-video-seconds']).toBe('12');
      expect(response.headers['x-gateway-billing-video-size']).toBe('1024x1792');
      expect(response.headers['x-gateway-billing-video-per-second-usd']).toBe('0.50000000');
      expect(response.headers['x-gateway-billing-media-cost']).toBe('6.00000000');
      const created = JSON.parse(response.body) as { id: string };
      const reference = decodeGatewayVideoId(created.id, {
        signingSecret: 'final-parameters-signing-secret',
        ttlMs: 86_400_000
      });
      expect(reference).toMatchObject({
        upstreamId: upstreamVideoId,
        model: 'sora-2-pro',
        duration: 12,
        size: '1024x1792'
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      await app.close();
    }
  });

  it('requires explicit xAI duration when a per-second budget check is active', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const provider = createProviderConfig('xai-video', ['grok-imagine-video'], {
      apikey: 'xai-key',
      baseurl: 'https://xai.example/v1',
      type: 'xai_video_generations'
    });
    provider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.05
    };
    const config = createConfig([provider]);
    config.precheck.enabled = true;
    config.precheck.budget = {
      enabled: true,
      windowMs: 60_000,
      maxCostUsd: 1,
      subject: 'global',
      scope: 'global'
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'Duration is intentionally omitted'
        }
      });

      expect(response.statusCode, response.body).toBe(400);
      expect(JSON.parse(response.body).error.message).toContain('duration must be explicit');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('counts xAI reference images during video precheck', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'provider-key',
        baseurl: 'https://videos.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    config.precheck.enabled = true;
    config.precheck.quota = {
      enabled: true,
      windowMs: 60_000,
      maxTokens: 4,
      subject: 'global',
      scope: 'global'
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai-video'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'This video prompt is intentionally long enough to exceed the quota.',
          reference_images: [
            { url: 'https://example.test/first.png' },
            { url: 'https://example.test/second.png' }
          ]
        }
      });

      expect(response.statusCode, response.body).toBe(429);
      expect(JSON.parse(response.body).error).toMatchObject({
        code: 'quota_exceeded',
        details: { estimated: { imageCount: 2 } }
      });
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('rejects an inferred multipart video model that is absent from the forwarded bytes', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const provider = createProviderConfig('openai-video', ['custom-video-model'], {
      apikey: 'openai-key',
      baseurl: 'https://openai.example/v1',
      type: 'openai_video_generations'
    });
    provider.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.05
    };
    const config = createConfig([provider]);
    config.billing.enabled = true;
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'openai-video-boundary';
    const multipartPayload = Buffer.from(
      [
        `--${boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nA fox in the snow\r\n`,
        `--${boundary}\r\nContent-Disposition: form-data; name="seconds"\r\n\r\n8\r\n`,
        `--${boundary}\r\nContent-Disposition: form-data; name="size"\r\n\r\n1280x720\r\n`,
        `--${boundary}--\r\n`
      ].join('')
    );
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('multipart bytes are forwarded unchanged');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('rejects unsupported multipart fields during OpenAI-to-xAI conversion', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
    ]);
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    const boundary = 'openai-video-unsupported-field-boundary';
    const multipartPayload = Buffer.from(
      [
        `--${boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\ngrok-imagine-video\r\n`,
        `--${boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nA reusable character\r\n`,
        `--${boundary}\r\nContent-Disposition: form-data; name="characters[0][id]"\r\n\r\nchar_123\r\n`,
        `--${boundary}--\r\n`
      ].join('')
    );
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`,
          'x-target-provider': 'xai-video'
        },
        payload: multipartPayload
      });

      expect(response.statusCode).toBe(400);
      expect(response.body).toContain('cannot be converted to xAI without data loss');
      expect(response.body).toContain('characters[0][id]');
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('converts OpenAI video create and status requests to xAI', async () => {
    const billingEvents: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url === 'http://billing.local/events') {
        billingEvents.push(JSON.parse(String(init.body)) as Record<string, unknown>);
        return new Response(null, { status: 204 });
      }
      if (url.endsWith('/videos/generations')) {
        expect(init.method).toBe('POST');
        expect(JSON.parse(String(init.body))).toEqual({
          model: 'grok-imagine-video',
          prompt: 'A paper city at sunrise',
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p',
          image: { url: 'https://example.test/frame.png' }
        });
        return jsonResponse({ request_id: 'xai-request-1' });
      }

      expect(url).toBe('https://xai.example/v1/videos/xai-request-1');
      expect(init.method).toBe('GET');
      return jsonResponse({
        status: 'done',
        model: 'grok-imagine-video',
        progress: 100,
        video: { url: 'https://example.test/result.mp4', duration: 8 },
        usage: { cost_in_usd_ticks: 500_000_000 }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
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
      headers: {}
    };
    await initializeBillingPublisher(config.billingQueue, config.billingWebhook);
    const runtime = createGatewayRuntime(config);
    runtime.providerPlugins.register({
      key: 'strip-provider-usage',
      providerName: 'xai-video',
      transformResponse({ upstreamPayload }) {
        const transformed = { ...(upstreamPayload as Record<string, unknown>) };
        delete transformed.usage;
        return { ok: true, value: transformed };
      }
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, runtime);
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'A paper city at sunrise',
          seconds: '8',
          size: '1280x720',
          input_reference: { image_url: 'https://example.test/frame.png' }
        }
      });

      expect(createResponse.statusCode).toBe(200);
      const created = JSON.parse(createResponse.body);
      expect(created).toMatchObject({
        object: 'video',
        status: 'queued',
        model: 'grok-imagine-video',
        seconds: '8',
        size: '1280x720'
      });
      expect(created.id).toMatch(/^gv3\./);
      expect(billingEvents).toHaveLength(0);

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(created.id)}`
      });
      expect(statusResponse.statusCode).toBe(200);
      expect(statusResponse.headers['x-gateway-billing-media-cost']).toBe('0.05000000');
      expect(statusResponse.headers['x-gateway-billing-total-cost']).toBe('0.05000000');
      expect(JSON.parse(statusResponse.body)).toMatchObject({
        id: created.id,
        object: 'video',
        status: 'completed',
        progress: 100,
        model: 'grok-imagine-video',
        seconds: '8',
        size: '1280x720'
      });
      expect(billingEvents).toHaveLength(1);
      expect(billingEvents[0]?.billing).toMatchObject({
        cost: { total: 0.05 }
      });

      const tamperedId = `${created.id.slice(0, -1)}${created.id.endsWith('a') ? 'b' : 'a'}`;
      const tamperedResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(tamperedId)}`
      });
      expect(tamperedResponse.statusCode).toBe(400);

      const invalidVariantResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(created.id)}/content?variant=preview`
      });
      expect(invalidVariantResponse.statusCode).toBe(400);

      const contentResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(created.id)}/content`
      });
      expect(contentResponse.statusCode).toBe(302);
      expect(contentResponse.headers.location).toBe('https://example.test/result.mp4');
      expect(contentResponse.headers['x-gateway-billing-media-cost']).toBe('0.05000000');
      expect(billingEvents).toHaveLength(1);
      expect(fetchMock).toHaveBeenCalledTimes(4);
    } finally {
      await app.close();
    }
  });

  it('returns terminal xAI video failures instead of reporting content as pending', async () => {
    const billingEvents: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === 'http://billing.local/events') {
        billingEvents.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        return new Response(null, { status: 204 });
      }
      if (url.endsWith('/videos/generations')) {
        return jsonResponse({ request_id: 'xai-failed-request' });
      }
      expect(url).toBe('https://xai.example/v1/videos/xai-failed-request');
      return jsonResponse({
        status: 'failed',
        error: { message: 'The request was rejected by moderation.' },
        usage: { cost_in_usd_ticks: 200_000_000 }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('xai-video', ['grok-imagine-video'], {
        apikey: 'xai-key',
        baseurl: 'https://xai.example/v1',
        type: 'xai_video_generations'
      })
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
      headers: {}
    };
    await initializeBillingPublisher(config.billingQueue, config.billingWebhook);
    config.media = {
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'xai'
        },
        payload: {
          model: 'grok-imagine-video',
          prompt: 'A request that will fail upstream',
          seconds: '4',
          size: '1280x720'
        }
      });
      expect(createResponse.statusCode).toBe(200);
      const publicId = JSON.parse(createResponse.body).id as string;

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}`
      });
      expect(statusResponse.statusCode).toBe(200);
      expect(statusResponse.headers['x-gateway-billing-media-cost']).toBe('0.02000000');
      expect(JSON.parse(statusResponse.body)).toMatchObject({
        status: 'failed',
        error: { message: 'The request was rejected by moderation.' }
      });
      await waitForCondition(() => billingEvents.length === 1);
      expect(billingEvents[0]?.outcome).toMatchObject({
        status: 'error',
        statusCode: 422,
        errorMessage: 'The request was rejected by moderation.'
      });

      const contentResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(publicId)}/content`
      });
      expect(contentResponse.statusCode).toBe(422);
      expect(contentResponse.headers['x-gateway-billing-media-cost']).toBe('0.02000000');
      expect(JSON.parse(contentResponse.body)).toMatchObject({
        error: {
          code: 'video_generation_failed',
          details: {
            message: 'The request was rejected by moderation.'
          }
        }
      });
      expect(billingEvents).toHaveLength(1);
      expect(fetchMock).toHaveBeenCalledTimes(4);
    } finally {
      await app.close();
    }
  });

  it('rejects xAI-to-OpenAI video routing without an absolute public content base URL', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.media = {
      publicBaseUrl: '/relative-gateway',
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'This requires a client-reachable content URL',
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(JSON.parse(response.body).error.attempts[0]).toMatchObject({
        stage: 'upstream_request_build',
        status: 400
      });
      expect(JSON.parse(response.body).error.attempts[0].message).toContain(
        'media.publicBaseUrl'
      );
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it.each([
    'https://gateway.example/base?',
    'https://gateway.example/base#',
    'https://gateway.example/base?tenant=a',
    'https://gateway.example/base#content'
  ])('rejects a media public base URL with query or fragment: %s', async (publicBaseUrl) => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.media = {
      publicBaseUrl,
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'Reject an ambiguous public content base URL',
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p'
        }
      });

      expect(response.statusCode).toBe(400);
      expect(JSON.parse(response.body).error.attempts[0].message).toContain(
        'without query parameters or fragments'
      );
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('converts xAI video create and status requests to OpenAI', async () => {
    const upstreamVideoId = 'video_68d7512d07848190b3e45da0ecbebcde004da08e1e0678d5';
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url.endsWith(`/videos/${upstreamVideoId}/content`)) {
        expect(init.method).toBe('GET');
        expect(init.headers).toMatchObject({
          range: 'bytes=0-7',
          'if-range': '"video-etag"'
        });
        return new Response('mock-mp', {
          status: 206,
          headers: {
            'content-type': 'video/mp4',
            'content-range': 'bytes 0-7/14'
          }
        });
      }
      if (url.endsWith('/videos')) {
        expect(init.method).toBe('POST');
        expect(JSON.parse(String(init.body))).toEqual({
          model: 'sora-2',
          prompt: 'A lantern floating over water',
          seconds: '8',
          size: '1280x720'
        });
        return jsonResponse({
          id: upstreamVideoId,
          object: 'video',
          status: 'queued',
          model: 'sora-2',
          progress: 0,
          seconds: '8',
          size: '1280x720'
        });
      }

      expect(url).toBe(`https://openai.example/v1/videos/${upstreamVideoId}`);
      expect(init.method).toBe('GET');
      return jsonResponse({
        id: upstreamVideoId,
        object: 'video',
        status: 'completed',
        model: 'sora-2',
        progress: 100,
        seconds: '8',
        size: '1280x720'
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const config = createConfig([
      createProviderConfig('openai-video', ['sora-2'], {
        apikey: 'openai-key',
        baseurl: 'https://openai.example/v1',
        type: 'openai_video_generations'
      })
    ]);
    config.media = {
      publicBaseUrl: 'https://gateway.example',
      videoIdSigningSecret: 'test-video-signing-secret',
      videoIdTtlMs: 86_400_000
    };
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    await app.ready();

    try {
      const createResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'A lantern floating over water',
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p'
        }
      });

      expect(createResponse.statusCode).toBe(200);
      const created = JSON.parse(createResponse.body);
      expect(created.request_id).toMatch(/^gv3\./);

      const statusResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(created.request_id)}`,
        headers: { host: 'attacker-controlled.example' }
      });
      expect(statusResponse.statusCode).toBe(200);
      expect(JSON.parse(statusResponse.body)).toMatchObject({
        status: 'done',
        model: 'sora-2',
        progress: 100,
        video: {
          url: `https://gateway.example/v1/videos/${encodeURIComponent(created.request_id)}/content`,
          duration: 8
        }
      });

      const contentResponse = await app.inject({
        method: 'GET',
        url: `/v1/videos/${encodeURIComponent(created.request_id)}/content`,
        headers: {
          range: 'bytes=0-7',
          'if-range': '"video-etag"'
        }
      });
      expect(contentResponse.statusCode).toBe(206);
      expect(contentResponse.headers['content-type']).toContain('video/mp4');
      expect(contentResponse.headers['content-range']).toBe('bytes 0-7/14');
      expect(contentResponse.body).toBe('mock-mp');

      const imageConversionResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'A lantern floating over water',
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p',
          image: { url: 'https://example.test/lantern.png' }
        }
      });
      expect(imageConversionResponse.statusCode).toBe(400);
      expect(
        JSON.parse(imageConversionResponse.body).error.attempts[0].message
      ).toContain('requires a multipart file upload');

      const unsupportedResponse = await app.inject({
        method: 'POST',
        url: '/v1/videos/generations',
        headers: { 'content-type': 'application/json' },
        payload: {
          model: 'openai-video/sora-2',
          prompt: 'An unsupported six-second square video',
          duration: 6,
          aspect_ratio: '1:1',
          resolution: '720p'
        }
      });
      expect(unsupportedResponse.statusCode).toBe(400);
      expect(JSON.parse(unsupportedResponse.body).error.attempts[0].message).toContain(
        'cannot be represented by OpenAI'
      );
      expect(fetchMock).toHaveBeenCalledTimes(3);
    } finally {
      await app.close();
    }
  });
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'content-type': 'application/json'
    }
  });
}

function buildMultipartImageEditPayload(
  boundary: string,
  fields: { model: string; prompt: string }
): Buffer {
  return Buffer.from(
    [
      `--${boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n${fields.model}\r\n`,
      `--${boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n${fields.prompt}\r\n`,
      `--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\n`,
      'Content-Type: image/png\r\n\r\n',
      'fake-png-bytes\r\n',
      `--${boundary}--\r\n`
    ].join('')
  );
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

function createConfig(providers: ProviderConfig[]): GatewayConfig {
  return {
    providers,
    providerPlugins: [],
    virtualModelProfiles: [],
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

function enableGatewayScheduling(config: GatewayConfig): void {
  config.scheduling = {
    enabled: true,
    cacheAffinity: {
      enabled: true,
      ttlMs: 600_000,
      defaultScope: 'credential_model',
      minPrefixTokens: 0,
      maxWaitMs: 3_000
    },
    credentialScheduler: {
      enabled: true,
      spilloverUtilization: 0.8,
      cooldownMs: {
        auth: 300_000,
        rateLimit: 60_000,
        serverError: 60_000,
        network: 30_000
      }
    },
    fallback: {
      mode: 'adaptive',
      maxAttempts: 4,
      retryStatusCodes: [408, 409, 429, 500, 502, 503, 504],
      crossProviderStatusCodes: [401, 403, 404, 429, 500, 502, 503, 504],
      preserveCache: 'prefer',
      maxCacheWaitMs: 3_000
    }
  };
}

function createProviderConfig(
  name: string,
  models: string[],
  options: {
    apikey?: string;
    baseurl?: string;
    type?: ProviderConfig['type'];
  } = {}
): ProviderConfig {
  return {
    name,
    type: options.type ?? 'openai_responses',
    apikey: options.apikey,
    baseurl: options.baseurl,
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
    },
    health: {
      status: 'unknown'
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
