import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Fastify from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GatewayConfig, ProviderConfig } from '../types';
import { registerManagerRoutes } from './routes';

const ORIGINAL_GATEWAY_CONFIG_PATH = process.env.GATEWAY_CONFIG_PATH;
const ORIGINAL_MANAGER_API_KEY = process.env.MANAGER_API_KEY;

describe('manager config routes', () => {
  let tempDir: string | undefined;

  afterEach(() => {
    process.env.GATEWAY_CONFIG_PATH = ORIGINAL_GATEWAY_CONFIG_PATH;
    process.env.MANAGER_API_KEY = ORIGINAL_MANAGER_API_KEY;

    if (tempDir) {
      rmSync(tempDir, { recursive: true, force: true });
      tempDir = undefined;
    }
    vi.restoreAllMocks();
  });

  it('returns config file and effective config', async () => {
    const setup = prepareTempConfig({
      host: '127.0.0.1',
      port: 3900
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/manager/config'
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.path).toBe(setup.configPath);
      expect(body.fileConfig.host).toBe('127.0.0.1');
      expect(body.fileConfig.port).toBe(3900);
      expect(body.effectiveConfig.host).toBe(runtimeConfig.host);
    } finally {
      await app.close();
    }
  });

  it('redacts manager config secrets by default and reveals them explicitly', async () => {
    const setup = prepareTempConfig({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'file-provider-key'
        }
      ],
      providerPlugins: [
        {
          key: 'openai-main-codex-oauth',
          providerName: 'openai-main',
          codexOauth: {
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            },
            refreshToken: 'file-refresh-token'
          }
        }
      ],
      billingWebhook: {
        enabled: true,
        endpoint: 'https://billing.example/events',
        headers: {
          authorization: 'Bearer file-billing-secret'
        }
      }
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    runtimeConfig.openaiApiKey = 'runtime-openai-key';
    runtimeConfig.providers = [
      createProviderConfig('openai-main', {
        apikey: 'runtime-provider-key'
      })
    ];
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const redacted = await app.inject({
        method: 'GET',
        url: '/manager/config'
      });
      expect(redacted.statusCode).toBe(200);
      const redactedBody = JSON.parse(redacted.body);
      expect(redactedBody.secretsRedacted).toBe(true);
      expect(redactedBody.fileConfig.Providers[0].apikey).toBe('[REDACTED]');
      expect(redactedBody.fileConfig.providerPlugins[0].codexOauth.accessToken).toEqual({
        from: 'request.headers.x-codex-access-token'
      });
      expect(redactedBody.fileConfig.providerPlugins[0].codexOauth.refreshToken).toBe('[REDACTED]');
      expect(redactedBody.fileConfig.billingWebhook.headers.authorization).toBe('[REDACTED]');
      expect(redactedBody.effectiveConfig.openaiApiKey).toBe('[REDACTED]');
      expect(redactedBody.effectiveConfig.providers[0].apikey).toBe('[REDACTED]');

      const revealed = await app.inject({
        method: 'GET',
        url: '/manager/config?revealSecrets=true'
      });
      expect(revealed.statusCode).toBe(200);
      const revealedBody = JSON.parse(revealed.body);
      expect(revealedBody.secretsRedacted).toBe(false);
      expect(revealedBody.fileConfig.Providers[0].apikey).toBe('file-provider-key');
      expect(revealedBody.fileConfig.providerPlugins[0].codexOauth.refreshToken).toBe('file-refresh-token');
      expect(revealedBody.effectiveConfig.openaiApiKey).toBe('runtime-openai-key');
      expect(revealedBody.effectiveConfig.providers[0].apikey).toBe('runtime-provider-key');
    } finally {
      await app.close();
    }
  });

  it('writes gateway.config.json and reloads in-memory config', async () => {
    const setup = prepareTempConfig({
      host: '0.0.0.0',
      port: 3000,
      upstreamTimeoutMs: 120000
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    const nextConfigPayload = {
      host: '127.0.0.1',
      port: 3011,
      upstreamTimeoutMs: 5000,
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'test-key',
          baseurl: 'https://api.openai.com/v1'
        }
      ],
      billingQueue: {
        enabled: false
      },
      billingWebhook: {
        enabled: false
      }
    };

    try {
      const response = await app.inject({
        method: 'PUT',
        url: '/manager/config',
        payload: nextConfigPayload
      });

      expect(response.statusCode).toBe(200);
      const fileContent = JSON.parse(readFileSync(setup.configPath, 'utf8'));
      expect(fileContent).toEqual(nextConfigPayload);
      expect(runtimeConfig.upstreamTimeoutMs).toBe(5000);
      expect(runtimeConfig.host).toBe('127.0.0.1');
      expect(runtimeConfig.port).toBe(3011);
      expect(runtimeConfig.providers[0]?.name).toBe('openai-main');
    } finally {
      await app.close();
    }
  });

  it('preserves existing secrets when updating a redacted manager config', async () => {
    const setup = prepareTempConfig({
      host: '0.0.0.0',
      port: 3000,
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'old-main-key'
        },
        {
          name: 'openai-backup',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'old-backup-key'
        }
      ],
      providerPlugins: [
        {
          key: 'openai-main-oauth',
          providerName: 'openai-main',
          codexOauth: {
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            },
            refreshToken: 'old-refresh-token'
          }
        }
      ],
      billingWebhook: {
        enabled: false,
        headers: {
          authorization: 'Bearer old-billing-secret'
        }
      }
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'PUT',
        url: '/manager/config',
        payload: {
          host: '127.0.0.1',
          port: 3000,
          Providers: [
            {
              name: 'openai-backup',
              type: 'openai_responses',
              models: ['gpt-4.1-mini'],
              apikey: '[REDACTED]'
            },
            {
              name: 'openai-main',
              type: 'openai_responses',
              models: ['gpt-4.1-mini'],
              apikey: '[REDACTED]',
              baseurl: 'https://openai.example/v1'
            }
          ],
          providerPlugins: [
            {
              key: 'openai-main-oauth',
              providerName: 'openai-main',
              codexOauth: {
                accessToken: {
                  from: 'request.headers.x-codex-access-token'
                },
                refreshToken: '[REDACTED]'
              }
            }
          ],
          billingWebhook: {
            enabled: false,
            headers: {
              authorization: '[REDACTED]'
            }
          }
        }
      });

      expect(response.statusCode).toBe(200);
      const fileContent = JSON.parse(readFileSync(setup.configPath, 'utf8'));
      expect(fileContent.Providers[0].name).toBe('openai-backup');
      expect(fileContent.Providers[0].apikey).toBe('old-backup-key');
      expect(fileContent.Providers[1].name).toBe('openai-main');
      expect(fileContent.Providers[1].apikey).toBe('old-main-key');
      expect(fileContent.providerPlugins[0].codexOauth.refreshToken).toBe('old-refresh-token');
      expect(fileContent.billingWebhook.headers.authorization).toBe('Bearer old-billing-secret');
      expect(runtimeConfig.providers.find((provider) => provider.name === 'openai-main')?.apikey).toBe('old-main-key');
      expect(runtimeConfig.providers.find((provider) => provider.name === 'openai-backup')?.apikey).toBe('old-backup-key');
    } finally {
      await app.close();
    }
  });

  it('validates gateway config without writing or reloading', async () => {
    const initialConfig = {
      host: '0.0.0.0',
      port: 3000,
      upstreamTimeoutMs: 120000
    };
    const setup = prepareTempConfig(initialConfig);
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    const beforeApplyConfig = vi.fn();
    const onConfigReload = vi.fn();
    registerManagerRoutes(app, { config: runtimeConfig, beforeApplyConfig, onConfigReload });
    await app.ready();

    const candidate = {
      host: '127.0.0.1',
      port: 3011,
      upstreamTimeoutMs: 5000,
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'candidate-provider-key'
        }
      ],
      billingQueue: {
        enabled: false
      },
      billingWebhook: {
        enabled: false
      }
    };

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/manager/config/validate',
        payload: candidate
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.ok).toBe(true);
      expect(body.valid).toBe(true);
      expect(body.effectiveConfig.host).toBe('127.0.0.1');
      expect(body.effectiveConfig.port).toBe(3011);
      expect(body.secretsRedacted).toBe(true);
      expect(body.effectiveConfig.providers[0].apikey).toBe('[REDACTED]');
      expect(body.warnings).toContain(
        'host/port updated in file, but listener address will take effect after process restart.'
      );
      expect(JSON.parse(readFileSync(setup.configPath, 'utf8'))).toEqual(initialConfig);
      expect(runtimeConfig.host).toBe('0.0.0.0');
      expect(runtimeConfig.port).toBe(3000);
      expect(beforeApplyConfig).not.toHaveBeenCalled();
      expect(onConfigReload).not.toHaveBeenCalled();
    } finally {
      await app.close();
    }
  });

  it('requires manager api key when MANAGER_API_KEY is set', async () => {
    const setup = prepareTempConfig({});
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    process.env.MANAGER_API_KEY = 'manager-secret';

    const app = Fastify({ logger: false });
    registerManagerRoutes(app, { config: createRuntimeConfig() });
    await app.ready();

    try {
      const missing = await app.inject({
        method: 'GET',
        url: '/manager/config'
      });
      expect(missing.statusCode).toBe(401);

      const invalid = await app.inject({
        method: 'GET',
        url: '/manager/config',
        headers: {
          'x-manager-key': 'wrong'
        }
      });
      expect(invalid.statusCode).toBe(403);

      const allowed = await app.inject({
        method: 'GET',
        url: '/manager/config',
        headers: {
          authorization: 'Bearer manager-secret'
        }
      });
      expect(allowed.statusCode).toBe(200);
    } finally {
      await app.close();
    }
  });

  it('returns restart warnings for non-hot-reload fields', async () => {
    const setup = prepareTempConfig({
      host: '0.0.0.0',
      port: 3000
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    registerManagerRoutes(app, { config: createRuntimeConfig() });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'PUT',
        url: '/manager/config',
        payload: {
          host: '127.0.0.1',
          port: 3010,
          agent: {
            mcpServers: [
              {
                name: 'filesystem',
                transport: 'stdio',
                command: 'node',
                args: ['server.js'],
                env: {}
              }
            ]
          },
          billingQueue: {
            enabled: false
          },
          billingWebhook: {
            enabled: false
          }
        }
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(Array.isArray(body.warnings)).toBe(true);
      expect(body.warnings.length).toBeGreaterThan(0);
    } finally {
      await app.close();
    }
  });

  it('disables /manager/providers when provider external source is enabled', async () => {
    const setup = prepareTempConfig({});
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    runtimeConfig.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/manager/providers'
      });

      expect(response.statusCode).toBe(405);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Provider management API is disabled');
    } finally {
      await app.close();
    }
  });

  it('redacts provider secrets in /manager/providers by default', async () => {
    const setup = prepareTempConfig({});
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    runtimeConfig.providers = [
      createProviderConfig('openai-main', {
        apikey: 'provider-secret'
      })
    ];
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/manager/providers'
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.secretsRedacted).toBe(true);
      expect(body.providers[0].apikey).toBe('[REDACTED]');
    } finally {
      await app.close();
    }
  });

  it('actively checks provider health and updates runtime provider state', async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: {
          'content-type': 'application/json'
        }
      });
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const setup = prepareTempConfig({});
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    const provider = createProviderConfig('openai-main', {
      apikey: 'provider-key',
      baseurl: 'https://openai.example/v1/'
    });
    runtimeConfig.providers = [provider];
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/manager/providers/health/check',
        payload: {
          providerName: 'openai-main',
          timeoutMs: 1000
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(upstreamUrl).toBe('https://openai.example/v1/models');
      expect(upstreamInit.method).toBe('GET');
      expect(upstreamInit.headers).toMatchObject({
        authorization: 'Bearer provider-key'
      });

      const body = JSON.parse(response.body);
      expect(body.results[0]).toMatchObject({
        provider: 'openai',
        providerName: 'openai-main',
        ok: true,
        statusCode: 200
      });
      expect(provider.health).toMatchObject({
        status: 'healthy',
        available: true
      });
    } finally {
      await app.close();
    }
  });

  it('returns current provider health snapshots', async () => {
    const setup = prepareTempConfig({});
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    const runtimeConfig = createRuntimeConfig();
    runtimeConfig.providers = [
      createProviderConfig('openai-main', {
        health: {
          status: 'degraded',
          available: true,
          latencyMs: 123,
          checkedAt: '2026-06-08T00:00:00.000Z'
        }
      })
    ];
    registerManagerRoutes(app, { config: runtimeConfig });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'GET',
        url: '/manager/providers/health'
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        providers: [
          {
            name: 'openai-main',
            provider: 'openai',
            type: 'openai_responses',
            models: ['gpt-4.1-mini'],
            health: {
              status: 'degraded',
              available: true,
              latencyMs: 123,
              checkedAt: '2026-06-08T00:00:00.000Z'
            }
          }
        ]
      });
    } finally {
      await app.close();
    }
  });

  it('blocks provider updates in /manager/config when provider external source is enabled', async () => {
    const setup = prepareTempConfig({
      host: '0.0.0.0',
      port: 3000
    });
    tempDir = setup.tempDir;
    process.env.GATEWAY_CONFIG_PATH = setup.configPath;
    delete process.env.MANAGER_API_KEY;

    const app = Fastify({ logger: false });
    registerManagerRoutes(app, { config: createRuntimeConfig() });
    await app.ready();

    try {
      const response = await app.inject({
        method: 'PUT',
        url: '/manager/config',
        payload: {
          host: '127.0.0.1',
          port: 3012,
          providerExternal: {
            enabled: true,
            endpoint: 'http://localhost:3001/gateway/providers'
          },
          Providers: [
            {
              name: 'openai-main',
              type: 'openai_responses',
              models: ['gpt-4.1-mini'],
              apikey: 'test-key'
            }
          ]
        }
      });

      expect(response.statusCode).toBe(405);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Provider management API is disabled');
    } finally {
      await app.close();
    }
  });
});

function prepareTempConfig(payload: Record<string, unknown>): { tempDir: string; configPath: string } {
  const tempDir = mkdtempSync(join(tmpdir(), 'gateway-manager-test-'));
  const configPath = join(tempDir, 'gateway.config.json');
  writeFileSync(configPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return {
    tempDir,
    configPath
  };
}

function createRuntimeConfig(): GatewayConfig {
  return {
    host: '0.0.0.0',
    port: 3000,
    providers: [],
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
    defaultOpenAIModel: 'gpt-4.1-mini',
    defaultAnthropicModel: 'claude-3-5-sonnet-latest',
    defaultGeminiModel: 'gemini-2.0-flash',
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
    policy: {
      enabled: false,
      defaults: {
        allowProviders: [],
        denyProviders: [],
        allowProviderNames: [],
        denyProviderNames: [],
        allowModels: [],
        denyModels: [],
        allowProviderModels: [],
        denyProviderModels: []
      },
      byUser: {},
      byTenant: {},
      byOrganization: {},
      bySubject: {},
      byPlan: {},
      byApiKey: {}
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
      transport: 'http',
      endpoint: undefined,
      timeoutMs: 5000,
      maxAttempts: 1,
      baseDelayMs: 10,
      maxDelayMs: 10,
      requireAck: false,
      headers: {}
    },
    agent: {
      storage: {
        type: 'filesystem',
        dir: '.agent-data'
      },
      mcpServers: [],
      runtime: {
        sessionLockTimeoutMs: 15000,
        eventWorkerConcurrency: 16,
        llmRetry: {
          maxAttempts: 3,
          baseDelayMs: 200,
          maxDelayMs: 2000,
          backoffMultiplier: 2,
          jitterMs: 100
        },
        toolRetry: {
          maxAttempts: 2,
          baseDelayMs: 150,
          maxDelayMs: 1500,
          backoffMultiplier: 2,
          jitterMs: 50
        }
      }
    },
    mcpGateway: {
      enabled: false,
      endpoint: '/mcp',
      websocket: {
        enabled: false,
        endpoint: '/mcp/ws',
        auth: {
          allowQueryToken: true,
          queryTokenParam: 'token'
        }
      },
      principals: [],
      serverExposure: {},
      internalCidrs: [],
      guardrails: {
        enabled: true,
        maxArgumentBytes: 65536,
        blockedTools: [],
        blockedArgumentKeys: [],
        redactArgumentKeys: []
      },
      oauth: {
        enabled: false,
        scopesSupported: []
      }
    }
  } as unknown as GatewayConfig;
}

function createProviderConfig(
  name: string,
  options: {
    apikey?: string;
    baseurl?: string;
    health?: ProviderConfig['health'];
  } = {}
): ProviderConfig {
  return {
    name,
    type: 'openai_responses',
    apikey: options.apikey,
    baseurl: options.baseurl,
    models: ['gpt-4.1-mini'],
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
    health: options.health
  };
}
