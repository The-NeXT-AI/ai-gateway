import { createCipheriv, randomBytes } from 'node:crypto';
import Fastify from 'fastify';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import { registerGatewayRoutes } from '../gateway/routes';
import { createGatewayRuntime } from '../gateway/runtime';
import { hydrateProvidersFromExternalSource } from './external';
import { updateDistributedCredentialEncryption } from './plugins';

const ENV_KEYS = [
  'DEFAULT_TARGET_PROVIDER',
  'DEFAULT_TARGET_PROVIDERS',
  'DEFAULT_OPENAI_MODEL',
  'DEFAULT_ANTHROPIC_MODEL',
  'DEFAULT_GEMINI_MODEL',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'GEMINI_API_KEY',
  'OPENAI_BASE_URL',
  'ANTHROPIC_BASE_URL',
  'GEMINI_BASE_URL',
  'PROVIDER_EXTERNAL_ENABLED',
  'PROVIDER_EXTERNAL_ENDPOINT'
] as const;

const ORIGINAL_ENV = new Map<string, string | undefined>();

describe('provider external source', () => {
  beforeEach(() => {
    for (const key of ENV_KEYS) {
      ORIGINAL_ENV.set(key, process.env[key]);
      delete process.env[key];
    }
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      const value = ORIGINAL_ENV.get(key);
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
    updateDistributedCredentialEncryption(undefined);
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('skips fetch when provider external source is disabled', async () => {
    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'openai-main-key'
        }
      ]
    });
    config.providerExternal = {
      enabled: false,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await hydrateProvidersFromExternalSource(config);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(config.providers[0]?.name).toBe('openai-main');
  });

  it('hydrates providers from external endpoint and refreshes derived defaults', async () => {
    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'openai-main-key'
        }
      ]
    });
    config.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          providers: [
            {
              name: 'anthropic-main',
              type: 'anthropic_messages',
              models: ['claude-3-7-sonnet'],
              apikey: 'anthropic-main-key',
              baseurl: 'https://proxy.anthropic.local'
            }
          ]
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await hydrateProvidersFromExternalSource(config);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(config.providers).toHaveLength(1);
    expect(config.providers[0]?.name).toBe('anthropic-main');
    expect(config.defaultTargetProviders).toEqual(['anthropic']);
    expect(config.defaultTargetProvider).toBe('anthropic');
    expect(config.anthropicApiKey).toBe('anthropic-main-key');
    expect(config.defaultAnthropicModel).toBe('claude-3-7-sonnet');
  });

  it('accepts array payload from external endpoint', async () => {
    const config = parseGatewayConfigFromRaw({
      Providers: []
    });
    config.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            name: 'gemini-main',
            type: 'gemini_generate_content',
            models: ['gemini-2.0-flash'],
            apikey: 'gemini-main-key'
          }
        ]),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await hydrateProvidersFromExternalSource(config);

    expect(config.providers[0]?.name).toBe('gemini-main');
    expect(config.defaultTargetProviders).toEqual(['gemini']);
    expect(config.defaultTargetProvider).toBe('gemini');
  });

  it('hydrates providers from stdio external source', async () => {
    const config = parseGatewayConfigFromRaw({
      providerExternal: {
        enabled: true,
        transport: 'stdio',
        command: process.execPath,
        args: [
          '-e',
          'let input="";process.stdin.on("data",c=>input+=c);process.stdin.on("end",()=>process.stdout.write(JSON.stringify({providers:[{name:"openai-stdio",type:"openai_responses",models:["gpt-4.1-mini"],apikey:"stdio-key"}]})));'
        ]
      }
    });

    await hydrateProvidersFromExternalSource(config);

    expect(config.providers).toHaveLength(1);
    expect(config.providers[0]?.name).toBe('openai-stdio');
    expect(config.defaultTargetProvider).toBe('openai');
  });

  it('hydrates providerPlugins from external endpoint payload', async () => {
    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-5.4'],
          apikey: 'openai-main-key'
        }
      ],
      providerPlugins: [
        {
          key: 'local-plugin',
          providerName: 'openai-main',
          auth: {
            headers: {
              authorization: 'Bearer local-token'
            }
          }
        }
      ]
    });
    config.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          providers: [
            {
              name: 'openai-main',
              type: 'openai_responses',
              models: ['gpt-5.4'],
              apikey: 'openai-main-key'
            }
          ],
          providerPlugins: [
            {
              key: 'openai-main-codex-oauth',
              providerName: 'openai-main',
              codexOauth: {
                accessToken: 'atk-from-server',
                refreshToken: 'rtk-from-server',
                accountId: 'acct-from-server'
              }
            }
          ]
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await hydrateProvidersFromExternalSource(config);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(config.providerPlugins).toHaveLength(1);
    expect(config.providerPlugins?.[0]?.key).toBe('openai-main-codex-oauth');
    expect(config.providerPlugins?.[0]?.providerName).toBe('openai-main');
    expect(config.providerPlugins?.[0]?.codexOauth?.accessToken).toBe('atk-from-server');
    expect(config.providerPlugins?.[0]?.codexOauth?.refreshToken).toBe('rtk-from-server');
    expect(config.providerPlugins?.[0]?.codexOauth?.accountId).toBe('acct-from-server');
  });

  it('distributes credential encryption key from external payload for codex oauth token decryption', async () => {
    const encryptionKey = Buffer.from('0123456789abcdef0123456789abcdef', 'utf8');
    const encryptedAccessToken = encryptCredentialForTest('atk-encrypted-from-server', encryptionKey, 'v1');

    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-5.4'],
          apikey: 'openai-main-key'
        }
      ]
    });
    config.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const urlString = String(url);
      if (urlString === 'http://localhost:3001/gateway/providers') {
        return new Response(
          JSON.stringify({
            providers: [
              {
                name: 'openai-main',
                type: 'openai_responses',
                models: ['gpt-5.4'],
                apikey: 'openai-main-key'
              }
            ],
            providerPlugins: [
              {
                key: 'openai-main-codex-oauth',
                providerName: 'openai-main',
                codexOauth: {
                  enabled: true,
                  tokenEndpoint: 'https://auth.openai.com/oauth/token',
                  clientId: 'app_EMoamEEZ73f0CkXaXp7hrann',
                  scope:
                    'openid profile email offline_access api.connectors.read api.connectors.invoke',
                  accessToken: encryptedAccessToken,
                  required: true,
                  refreshIfMissingAccessToken: false
                }
              }
            ],
            credentialEncryption: {
              algorithm: 'aes-256-gcm',
              keyVersion: 'v1',
              key: encryptionKey.toString('base64')
            }
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
          output_text: 'external-key-distribution-ok',
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

    await hydrateProvidersFromExternalSource(config);

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
          model: 'gpt-5.4',
          input: 'hello'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe('https://chatgpt.com/backend-api/codex/responses');
      const [, upstreamInit] = fetchMock.mock.calls[1] as [string, RequestInit];
      const upstreamHeaders = (upstreamInit.headers || {}) as Record<string, string>;
      expect(upstreamHeaders.authorization).toBe('Bearer atk-encrypted-from-server');
    } finally {
      await app.close();
    }
  });

  it('decrypts encrypted provider API keys from external provider payloads', async () => {
    const encryptionKey = Buffer.from('abcdef0123456789abcdef0123456789', 'utf8');
    const encryptedProviderKey = encryptCredentialForTest(
      'openai-provider-key',
      encryptionKey,
      'v1'
    );
    const encryptedCredentialKey = encryptCredentialForTest(
      'openai-credential-key',
      encryptionKey,
      'v1'
    );
    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'old-openai-key'
        }
      ]
    });
    config.providerExternal = {
      enabled: true,
      transport: 'http',
      endpoint: 'http://localhost:3001/gateway/providers',
      timeoutMs: 5000,
      apiKeyHeader: 'x-provider-external-key',
      headers: {}
    };

    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          providers: [
            {
              name: 'openai-main',
              type: 'openai_responses',
              models: ['gpt-5.4'],
              apikey: encryptedProviderKey,
              credentials: [
                {
                  id: 'primary',
                  apikey: encryptedCredentialKey,
                  enabled: true
                }
              ]
            }
          ],
          credentialEncryption: {
            algorithm: 'aes-256-gcm',
            keyVersion: 'v1',
            key: encryptionKey.toString('base64')
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
    vi.stubGlobal('fetch', fetchMock);

    await hydrateProvidersFromExternalSource(config);

    expect(config.providers[0]?.apikey).toBe('openai-provider-key');
    expect(config.openaiApiKey).toBe('openai-provider-key');
    expect(config.providers[0]?.credentials?.[0]?.apikey).toBe(
      'openai-credential-key'
    );
  });
});

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
