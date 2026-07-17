import type { FastifyRequest } from 'fastify';
import { afterEach, describe, expect, it } from 'vitest';
import type { GatewayConfig, GatewayPrecheckConfig, ProviderConfig, StandardRequest } from '../types';
import {
  evaluateGatewayPrecheck,
  resetGatewayPrecheckStateForTests,
  setGatewayPrecheckRedisReservationExecutorForTests
} from './precheck';

type TestPrecheckConfig = Omit<GatewayPrecheckConfig, 'storage'> &
  Partial<Pick<GatewayPrecheckConfig, 'storage'>>;

describe('evaluateGatewayPrecheck', () => {
  afterEach(() => {
    resetGatewayPrecheckStateForTests();
  });

  it('rejects requests once the fixed-window rate limit is exceeded', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: {
        enabled: true,
        windowMs: 60_000,
        maxRequests: 1,
        rpm: 0,
        rpd: 0,
        tpm: 0,
        tpd: 0,
        ipm: 0,
        limits: [
          {
            enabled: true,
            name: 'requests',
            metric: 'requests',
            windowMs: 60_000,
            max: 1,
            subject: 'global',
            scope: 'global'
          }
        ],
        subject: 'global',
        scope: 'global'
      },
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });
    const input = {
      request: createRequest(),
      config,
      targetProvider: 'openai' as const,
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    };

    expect((await evaluateGatewayPrecheck(input)).ok).toBe(true);

    const rejected = await evaluateGatewayPrecheck(input);
    expect(rejected.ok).toBe(false);
    if (!rejected.ok) {
      expect(rejected.statusCode).toBe(429);
      expect(rejected.code).toBe('rate_limit_exceeded');
      expect(rejected.details.used).toBe(1);
      expect(rejected.details.requested).toBe(1);
    }
  });

  it('enforces RPM and RPD request dimensions independently', async () => {
    const rpmConfig = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'rpm',
          metric: 'requests',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });
    const rpmInput = {
      request: createRequest(),
      config: rpmConfig,
      targetProvider: 'openai' as const,
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    };

    expect((await evaluateGatewayPrecheck(rpmInput)).ok).toBe(true);
    const rpmRejected = await evaluateGatewayPrecheck(rpmInput);
    expect(rpmRejected.ok).toBe(false);
    if (!rpmRejected.ok) {
      expect(rpmRejected.details.limit_name).toBe('rpm');
      expect(rpmRejected.details.metric).toBe('requests');
      expect(rpmRejected.details.window_ms).toBe(60_000);
    }

    resetGatewayPrecheckStateForTests();
    const rpdConfig = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'rpd',
          metric: 'requests',
          windowMs: 86_400_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });
    const rpdInput = {
      ...rpmInput,
      config: rpdConfig
    };

    expect((await evaluateGatewayPrecheck(rpdInput)).ok).toBe(true);
    const rpdRejected = await evaluateGatewayPrecheck(rpdInput);
    expect(rpdRejected.ok).toBe(false);
    if (!rpdRejected.ok) {
      expect(rpdRejected.details.limit_name).toBe('rpd');
      expect(rpdRejected.details.window_ms).toBe(86_400_000);
    }
  });

  it('enforces TPM and TPD token dimensions', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'tpm',
          metric: 'tokens',
          windowMs: 60_000,
          max: 5,
          subject: 'global',
          scope: 'model'
        },
        {
          enabled: true,
          name: 'tpd',
          metric: 'tokens',
          windowMs: 86_400_000,
          max: 100,
          subject: 'global',
          scope: 'model'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      }
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('123456')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe('rate_limit_exceeded');
      expect(result.details.limit_name).toBe('tpm');
      expect(result.details.metric).toBe('tokens');
      expect(result.details.requested).toBe(14);
    }
  });

  it('enforces IPM image dimensions from the original request body', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'ipm',
          metric: 'images',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'model'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('describe'),
      requestBody: {
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'compare' },
              { type: 'image_url', image_url: { url: 'https://example.com/1.png' } },
              { type: 'input_image', image_url: 'data:image/png;base64,abcd' }
            ]
          }
        ]
      }
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.details.limit_name).toBe('ipm');
      expect(result.details.metric).toBe('images');
      expect(result.details.requested).toBe(2);
      expect(result.details.estimated?.imageCount).toBe(2);
    }
  });

  it('rejects requests whose estimated token usage exceeds quota', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: disabledRateLimit(),
      quota: {
        enabled: true,
        windowMs: 60_000,
        maxTokens: 5,
        subject: 'global',
        scope: 'model'
      },
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      }
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('123456')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe('quota_exceeded');
      expect(result.details.scope).toBe('model:gpt-test');
      expect(result.details.requested).toBe(14);
      expect(result.details.estimated?.inputTokens).toBe(14);
    }
  });

  it('includes Responses text options in estimated token usage', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: disabledRateLimit(),
      quota: {
        enabled: true,
        windowMs: 60_000,
        maxTokens: 20,
        subject: 'global',
        scope: 'model'
      },
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      }
    });
    const standardRequest = createStandardRequest('123456');
    standardRequest.text = {
      verbosity: 'low'
    };

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe('quota_exceeded');
      expect(result.details.scope).toBe('model:gpt-test');
      expect(result.details.requested).toBe(33);
      expect(result.details.estimated?.inputTokens).toBe(33);
    }
  });

  it('rejects requests whose estimated cost exceeds budget', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: disabledRateLimit(),
      quota: disabledQuota(),
      budget: {
        enabled: true,
        windowMs: 86_400_000,
        maxCostUsd: 0.01,
        subject: 'global',
        scope: 'provider_model'
      },
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 1000
      }
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('prompt')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.statusCode).toBe(402);
      expect(result.code).toBe('budget_exceeded');
      expect(result.details.scope).toBe('provider:openai:model:gpt-test');
      expect(result.details.estimated?.estimatedCostUsd).toBeGreaterThan(0.01);
    }
  });

  it('includes configured video seconds in budget estimates', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: disabledRateLimit(),
      quota: disabledQuota(),
      budget: {
        enabled: true,
        windowMs: 86_400_000,
        maxCostUsd: 0.2,
        subject: 'global',
        scope: 'global'
      },
      estimation: { charsPerToken: 4, defaultMaxOutputTokens: 0 }
    });
    config.providers[0]!.billing.default = {
      inputPerMillionUsd: 0,
      outputPerMillionUsd: 0,
      videoPerSecondUsd: 0.05
    };

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      targetProviderConfig: config.providers[0],
      model: 'gpt-test',
      standardRequest: createStandardRequest('video prompt'),
      videoSeconds: 8
    });

    expect(result).toMatchObject({
      ok: false,
      statusCode: 402,
      code: 'budget_exceeded',
      details: {
        requested: 0.4,
        estimated: { videoSeconds: 8, estimatedCostUsd: 0.4 }
      }
    });
  });

  it('enforces API key restriction rate limits even when static precheck is disabled', async () => {
    const config = createConfig({
      enabled: false,
      rateLimit: disabledRateLimit(),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });
    const request = createRequest() as FastifyRequest & {
      gatewayIdentity?: { apiKeyId: string; source: 'http_introspection'; billingSubjectKey: string };
      gatewayApiKeyRestrictions?: { rateLimit: number; rateLimitWindowSeconds: number };
    };
    request.gatewayIdentity = {
      source: 'http_introspection',
      billingSubjectKey: 'user-1',
      apiKeyId: 'key-1'
    };
    request.gatewayApiKeyRestrictions = {
      rateLimit: 1,
      rateLimitWindowSeconds: 60
    };
    const input = {
      request,
      config,
      targetProvider: 'openai' as const,
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    };

    expect((await evaluateGatewayPrecheck(input)).ok).toBe(true);
    const rejected = await evaluateGatewayPrecheck(input);

    expect(rejected.ok).toBe(false);
    if (!rejected.ok) {
      expect(rejected.statusCode).toBe(429);
      expect(rejected.details.limit_name).toBe('api_key_restriction');
      expect(rejected.details.subject).toBe('key-1');
      expect(rejected.details.limit).toBe(1);
      expect(rejected.details.window_ms).toBe(60_000);
    }
  });

  it('does not enable static precheck rules when only API key restrictions are active', async () => {
    const config = createConfig({
      enabled: false,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'static-global',
          metric: 'requests',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      }
    });
    const request = createRequest() as FastifyRequest & {
      gatewayIdentity?: { apiKeyId: string; source: 'http_introspection'; billingSubjectKey: string };
      gatewayApiKeyRestrictions?: { requestsPerMinute: number };
    };
    request.gatewayIdentity = {
      source: 'http_introspection',
      billingSubjectKey: 'user-1',
      apiKeyId: 'key-2'
    };
    request.gatewayApiKeyRestrictions = {
      requestsPerMinute: 2
    };
    const input = {
      request,
      config,
      targetProvider: 'openai' as const,
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    };

    expect((await evaluateGatewayPrecheck(input)).ok).toBe(true);
    expect((await evaluateGatewayPrecheck(input)).ok).toBe(true);
    const rejected = await evaluateGatewayPrecheck(input);

    expect(rejected.ok).toBe(false);
    if (!rejected.ok) {
      expect(rejected.details.limit_name).toBe('api_key_restriction');
      expect(rejected.details.limit).toBe(2);
    }
  });

  it('enforces API key token limits even when static precheck is disabled', async () => {
    const config = createConfig({
      enabled: false,
      rateLimit: disabledRateLimit(),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      }
    });
    const request = createRequest() as FastifyRequest & {
      gatewayIdentity?: { apiKeyId: string; source: 'http_introspection'; billingSubjectKey: string };
      gatewayApiKeyRestrictions?: { tokensPerMinute: number; tokenLimitWindowSeconds: number };
    };
    request.gatewayIdentity = {
      source: 'http_introspection',
      billingSubjectKey: 'user-1',
      apiKeyId: 'key-tpm'
    };
    request.gatewayApiKeyRestrictions = {
      tokensPerMinute: 5,
      tokenLimitWindowSeconds: 120
    };

    const result = await evaluateGatewayPrecheck({
      request,
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('123456')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.statusCode).toBe(429);
      expect(result.details.limit_name).toBe('api_key_tpm');
      expect(result.details.metric).toBe('tokens');
      expect(result.details.subject).toBe('key-tpm');
      expect(result.details.window_ms).toBe(120_000);
      expect(result.details.requested).toBe(14);
    }
  });

  it('enforces API key cost limits even when static precheck is disabled', async () => {
    const config = createConfig({
      enabled: false,
      rateLimit: disabledRateLimit(),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 1,
        defaultMaxOutputTokens: 0
      }
    });
    const request = createRequest() as FastifyRequest & {
      gatewayIdentity?: { apiKeyId: string; source: 'http_introspection'; billingSubjectKey: string };
      gatewayApiKeyRestrictions?: { costLimitUsd: number; costLimitWindowSeconds: number };
    };
    request.gatewayIdentity = {
      source: 'http_introspection',
      billingSubjectKey: 'user-1',
      apiKeyId: 'key-cost'
    };
    request.gatewayApiKeyRestrictions = {
      costLimitUsd: 0.01,
      costLimitWindowSeconds: 60
    };

    const result = await evaluateGatewayPrecheck({
      request,
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('123456')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.statusCode).toBe(402);
      expect(result.code).toBe('budget_exceeded');
      expect(result.details.limit_name).toBe('api_key_cost');
      expect(result.details.metric).toBe('cost_usd');
      expect(result.details.subject).toBe('key-cost');
      expect(result.details.requested).toBeGreaterThan(0.01);
      expect(result.details.estimated?.estimatedCostUsd).toBeGreaterThan(0.01);
    }
  });

  it('uses Redis storage reservation when precheck storage is redis', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'global-rpm',
          metric: 'requests',
          windowMs: 60_000,
          max: 5,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      },
      storage: {
        type: 'redis',
        url: 'redis://cache.local:6379/2',
        keyPrefix: 'test:precheck',
        connectTimeoutMs: 50,
        commandTimeoutMs: 50
      }
    });
    const calls: Array<{ storage: GatewayPrecheckConfig['storage']; limitNames: Array<string | undefined> }> = [];
    setGatewayPrecheckRedisReservationExecutorForTests(async (storage, checks) => {
      calls.push({
        storage,
        limitNames: checks.map((check) => check.limitName)
      });
      return { ok: true };
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    });

    expect(result.ok).toBe(true);
    expect(calls).toHaveLength(1);
    expect(calls[0].storage).toMatchObject({
      type: 'redis',
      url: 'redis://cache.local:6379/2',
      keyPrefix: 'test:precheck'
    });
    expect(calls[0].limitNames).toEqual(['global-rpm']);
  });

  it('returns a rate-limit failure from Redis reservation denial', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'global-rpm',
          metric: 'requests',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      },
      storage: redisPrecheckStorage()
    });
    setGatewayPrecheckRedisReservationExecutorForTests(async () => ({
      ok: false,
      failedIndex: 1,
      used: 1
    }));

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.statusCode).toBe(429);
      expect(result.code).toBe('rate_limit_exceeded');
      expect(result.details.limit_name).toBe('global-rpm');
      expect(result.details.used).toBe(1);
    }
  });

  it('fails closed when Redis precheck storage is unavailable', async () => {
    const config = createConfig({
      enabled: true,
      rateLimit: createRateLimit([
        {
          enabled: true,
          name: 'global-rpm',
          metric: 'requests',
          windowMs: 60_000,
          max: 1,
          subject: 'global',
          scope: 'global'
        }
      ]),
      quota: disabledQuota(),
      budget: disabledBudget(),
      estimation: {
        charsPerToken: 4,
        defaultMaxOutputTokens: 0
      },
      storage: redisPrecheckStorage()
    });
    setGatewayPrecheckRedisReservationExecutorForTests(async () => {
      throw new Error('redis down');
    });

    const result = await evaluateGatewayPrecheck({
      request: createRequest(),
      config,
      targetProvider: 'openai',
      model: 'gpt-test',
      standardRequest: createStandardRequest('hello')
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.statusCode).toBe(503);
      expect(result.code).toBe('precheck_store_unavailable');
      expect(result.details.limit_name).toBe('global-rpm');
    }
  });

});

function disabledRateLimit(): GatewayPrecheckConfig['rateLimit'] {
  return createRateLimit([]);
}

function createRateLimit(
  limits: GatewayPrecheckConfig['rateLimit']['limits']
): GatewayPrecheckConfig['rateLimit'] {
  return {
    enabled: false,
    windowMs: 60_000,
    maxRequests: 0,
    rpm: 0,
    rpd: 0,
    tpm: 0,
    tpd: 0,
    ipm: 0,
    limits: [],
    ...(limits.length > 0
      ? {
          enabled: true,
          limits
        }
      : {}),
    subject: 'identity',
    scope: 'global'
  };
}

function disabledQuota(): GatewayPrecheckConfig['quota'] {
  return {
    enabled: false,
    windowMs: 86_400_000,
    maxTokens: 0,
    subject: 'identity',
    scope: 'global'
  };
}

function disabledBudget(): GatewayPrecheckConfig['budget'] {
  return {
    enabled: false,
    windowMs: 86_400_000,
    maxCostUsd: 0,
    subject: 'identity',
    scope: 'global'
  };
}

function createStandardRequest(input: string): StandardRequest {
  return {
    model: 'gpt-test',
    input,
    max_output_tokens: 0
  };
}

function createRequest(): FastifyRequest {
  return {
    headers: {},
    ip: '127.0.0.1',
    socket: {
      remoteAddress: '127.0.0.1'
    }
  } as unknown as FastifyRequest;
}

function createConfig(precheck: TestPrecheckConfig): GatewayConfig {
  const provider = createProviderConfig();
  return {
    providers: [provider],
    precheck: {
      ...precheck,
      storage: precheck.storage || memoryPrecheckStorage()
    },
    billing: {
      enabled: true,
      currency: 'USD',
      rates: {
        openai: {
          inputPerMillionUsd: 1000,
          outputPerMillionUsd: 1000
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
    }
  } as GatewayConfig;
}

function memoryPrecheckStorage(): GatewayPrecheckConfig['storage'] {
  return {
    type: 'memory'
  };
}

function redisPrecheckStorage(): GatewayPrecheckConfig['storage'] {
  return {
    type: 'redis',
    url: 'redis://127.0.0.1:6379/0',
    keyPrefix: 'test:precheck',
    connectTimeoutMs: 50,
    commandTimeoutMs: 50
  };
}

function createProviderConfig(): ProviderConfig {
  return {
    name: 'openai-main',
    type: 'openai_responses',
    models: ['gpt-test'],
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
