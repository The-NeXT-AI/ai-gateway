import Fastify from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { registerGatewayRoutes } from './routes';
import { createGatewayRuntime } from './runtime';
import type { GatewayConfig, ProviderConfig } from '../types';

const runDeepSeekLive = isTruthy(process.env.RUN_DEEPSEEK_LIVE) && Boolean(readEnv('DEEPSEEK_API_KEY'));
const liveIt = runDeepSeekLive ? it : it.skip;

describe('deepseek live integration', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  liveIt('converts interleaved reasoning/tool history and receives a real DeepSeek response', async () => {
    const { app, capturedRequests, model } = await createLiveGateway();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/responses',
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'deepseek-live'
        },
        payload: {
          model,
          max_output_tokens: 64,
          temperature: 0,
          reasoning: {
            effort: 'medium'
          },
          input: [
            {
              type: 'reasoning',
              id: 'rs_live_previous',
              status: 'completed',
              content: [
                {
                  type: 'reasoning_text',
                  text: 'Need to use the prior tool result and answer briefly.'
                }
              ]
            },
            {
              type: 'function_call',
              call_id: 'call_live_lookup',
              name: 'lookup_value',
              arguments: '{"key":"live"}'
            },
            {
              type: 'function_call_output',
              call_id: 'call_live_lookup',
              output: '{"value":"live-ok"}'
            },
            {
              type: 'message',
              role: 'user',
              content: [
                {
                  type: 'input_text',
                  text: 'Use the tool result above and answer with exactly: live-ok'
                }
              ]
            }
          ]
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      const responseBody = JSON.parse(response.body) as Record<string, unknown>;
      expect(responseBody.output_text).toEqual(expect.any(String));

      const upstreamBody = expectCapturedDeepSeekRequest(capturedRequests, model, 'high');
      expect(upstreamBody?.messages).toEqual([
        {
          role: 'assistant',
          content: '',
          reasoning_content: 'Need to use the prior tool result and answer briefly.',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Need to use the prior tool result and answer briefly.',
              format: 'openai-responses-v1',
              index: 0
            }
          ],
          tool_calls: [
            {
              id: 'call_live_lookup',
              type: 'function',
              function: {
                name: 'lookup_value',
                arguments: '{"key":"live"}'
              }
            }
          ]
        },
        {
          role: 'tool',
          tool_call_id: 'call_live_lookup',
          content: '{"value":"live-ok"}'
        },
        {
          role: 'user',
          content: 'Use the tool result above and answer with exactly: live-ok'
        }
      ]);
    } finally {
      await app.close();
    }
  });

  liveIt('converts /v1/messages thinking/tool history into a real DeepSeek request', async () => {
    const { app, capturedRequests, model } = await createLiveGateway();

    try {
      const response = await app.inject({
        method: 'POST',
        url: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'anthropic-version': '2023-06-01',
          'x-target-provider': 'deepseek-live'
        },
        payload: {
          model,
          max_tokens: 64,
          temperature: 0,
          thinking: {
            type: 'enabled'
          },
          output_config: {
            effort: 'medium'
          },
          messages: [
            {
              role: 'user',
              content: 'Use a tool'
            },
            {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: 'Need to use the prior tool result and answer briefly.',
                  signature: 'sig_live_anthropic'
                },
                {
                  type: 'tool_use',
                  id: 'call_live_lookup',
                  name: 'lookup_value',
                  input: {
                    key: 'live'
                  }
                }
              ]
            },
            {
              role: 'user',
              content: [
                {
                  type: 'tool_result',
                  tool_use_id: 'call_live_lookup',
                  content: '{"value":"live-ok"}'
                },
                {
                  type: 'text',
                  text: 'Use the tool result above and answer with exactly: live-ok'
                }
              ]
            }
          ]
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      const responseBody = JSON.parse(response.body) as Record<string, unknown>;
      expect(responseBody.content).toEqual(expect.any(Array));

      const upstreamBody = expectCapturedDeepSeekRequest(capturedRequests, model, 'high');
      expect(upstreamBody?.messages).toEqual([
        {
          role: 'user',
          content: 'Use a tool'
        },
        {
          role: 'assistant',
          content: '',
          reasoning_content: 'Need to use the prior tool result and answer briefly.',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Need to use the prior tool result and answer briefly.',
              format: 'anthropic-claude-v1',
              index: 0,
              signature: 'sig_live_anthropic'
            }
          ],
          tool_calls: [
            {
              id: 'call_live_lookup',
              type: 'function',
              function: {
                name: 'lookup_value',
                arguments: '{"key":"live"}'
              }
            }
          ]
        },
        {
          role: 'tool',
          tool_call_id: 'call_live_lookup',
          content: '{"value":"live-ok"}'
        },
        {
          role: 'user',
          content: 'Use the tool result above and answer with exactly: live-ok'
        }
      ]);
    } finally {
      await app.close();
    }
  });

  liveIt('converts Gemini generateContent thinking/tool history into a real DeepSeek request', async () => {
    const { app, capturedRequests, model } = await createLiveGateway();

    try {
      const response = await app.inject({
        method: 'POST',
        url: `/v1/models/${model}:generateContent`,
        headers: {
          'content-type': 'application/json',
          'x-target-provider': 'deepseek-live'
        },
        payload: {
          generationConfig: {
            maxOutputTokens: 64,
            temperature: 0,
            thinkingConfig: {
              thinkingBudget: 1024
            }
          },
          contents: [
            {
              role: 'model',
              parts: [
                {
                  text: 'Need to use the prior tool result and answer briefly.',
                  thought: true
                },
                {
                  functionCall: {
                    id: 'call_live_lookup',
                    name: 'lookup_value',
                    args: {
                      key: 'live'
                    }
                  }
                }
              ]
            },
            {
              role: 'user',
              parts: [
                {
                  functionResponse: {
                    id: 'call_live_lookup',
                    name: 'lookup_value',
                    response: {
                      content: '{"value":"live-ok"}'
                    }
                  }
                },
                {
                  text: 'Use the tool result above and answer with exactly: live-ok'
                }
              ]
            }
          ]
        }
      });

      expect(response.statusCode, response.body).toBe(200);
      const responseBody = JSON.parse(response.body) as Record<string, unknown>;
      expect(responseBody.candidates).toEqual(expect.any(Array));

      const upstreamBody = expectCapturedDeepSeekRequest(capturedRequests, model);
      expect(upstreamBody?.messages).toEqual([
        {
          role: 'assistant',
          content: '',
          reasoning_content: 'Need to use the prior tool result and answer briefly.',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Need to use the prior tool result and answer briefly.',
              format: 'google-generate-content-v1',
              index: 0
            }
          ],
          tool_calls: [
            {
              id: 'call_live_lookup',
              type: 'function',
              function: {
                name: 'lookup_value',
                arguments: '{"key":"live"}'
              }
            }
          ]
        },
        {
          role: 'tool',
          tool_call_id: 'call_live_lookup',
          content: '{"value":"live-ok"}'
        },
        {
          role: 'user',
          content: 'Use the tool result above and answer with exactly: live-ok'
        }
      ]);
    } finally {
      await app.close();
    }
  });
});

async function createLiveGateway(): Promise<{
  app: ReturnType<typeof Fastify>;
  capturedRequests: Array<{ url: string; body?: Record<string, unknown> }>;
  model: string;
}> {
  const apiKey = readEnv('DEEPSEEK_API_KEY');
  expect(apiKey).toBeTruthy();

  const model = readEnv('DEEPSEEK_LIVE_MODEL') || 'deepseek-v4-flash';
  const capturedRequests: Array<{ url: string; body?: Record<string, unknown> }> = [];
  const realFetch = globalThis.fetch.bind(globalThis);
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const body = parseJsonBody(init?.body);
    capturedRequests.push({
      url: typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url,
      body
    });
    return realFetch(input, init);
  });

  const provider = createProviderConfig('deepseek-live', 'openai_chat_completions', [model]);
  provider.baseurl = 'https://api.deepseek.com';
  provider.apikey = apiKey;

  const app = Fastify({ logger: false });
  const config = createConfig([provider]);
  registerGatewayRoutes(app, config, createGatewayRuntime(config));
  await app.ready();

  return { app, capturedRequests, model };
}

function expectCapturedDeepSeekRequest(
  capturedRequests: Array<{ url: string; body?: Record<string, unknown> }>,
  model: string,
  expectedReasoningEffort?: string
): Record<string, unknown> | undefined {
  expect(capturedRequests).toHaveLength(1);
  expect(capturedRequests[0]?.url).toBe('https://api.deepseek.com/chat/completions');
  const upstreamBody = capturedRequests[0]?.body;
  expect(upstreamBody).toMatchObject({
    model,
    thinking: {
      type: 'enabled'
    }
  });
  expect(upstreamBody?.reasoning_effort).toBe(expectedReasoningEffort);
  expect(upstreamBody?.reasoning_split).toBeUndefined();
  expect(upstreamBody?.interleaved_thinking).toBeUndefined();
  expect(upstreamBody?.interleavedThinking).toBeUndefined();
  expect(upstreamBody?.output_config).toBeUndefined();
  return upstreamBody;
}

function readEnv(name: string): string | undefined {
  const value = process.env[name];
  if (typeof value !== 'string') {
    return undefined;
  }
  return value.trim() || undefined;
}

function isTruthy(value: string | undefined): boolean {
  return ['1', 'true', 'yes', 'on'].includes((value || '').trim().toLowerCase());
}

function parseJsonBody(body: unknown): Record<string, unknown> | undefined {
  if (typeof body !== 'string') {
    return undefined;
  }

  try {
    const parsed = JSON.parse(body);
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch {
    return undefined;
  }
}

function createConfig(providers: ProviderConfig[]): GatewayConfig {
  return {
    providers,
    defaultTargetProvider: 'openai',
    defaultTargetProviders: ['openai'],
    openaiApiKey: 'openai-test-key',
    anthropicApiKey: 'anthropic-test-key',
    geminiApiKey: 'gemini-test-key',
    openaiBaseUrl: 'https://api.openai.com/v1',
    anthropicBaseUrl: 'https://api.anthropic.com',
    geminiBaseUrl: 'https://generativelanguage.googleapis.com',
    geminiApiVersion: 'v1beta',
    upstreamTimeoutMs: 30000,
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
