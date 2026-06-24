import Fastify from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { closeBillingPublisher } from '../billing';
import type { GatewayConfig, ProviderConfig, ProviderPlugin } from '../types';
import { registerGatewayRoutes } from './routes';
import { createGatewayRuntime } from './runtime';
import { resetGatewayPrecheckStateForTests } from './precheck';
import { resetProviderCircuitBreakerForTests } from './upstream-circuit-breaker';
import { resetProviderConcurrencyForTests } from './upstream-concurrency';

describe('openai embeddings gateway route', () => {
  afterEach(async () => {
    resetGatewayPrecheckStateForTests();
    resetProviderCircuitBreakerForTests();
    resetProviderConcurrencyForTests();
    await closeBillingPublisher();
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
        baseurl: 'https://images.example/v1'
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

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'content-type': 'application/json'
    }
  });
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

function createProviderConfig(
  name: string,
  models: string[],
  options: {
    apikey?: string;
    baseurl?: string;
  } = {}
): ProviderConfig {
  return {
    name,
    type: 'openai_responses',
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
