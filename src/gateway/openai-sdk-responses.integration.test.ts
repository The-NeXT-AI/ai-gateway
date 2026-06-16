import Fastify from 'fastify';
import OpenAI from 'openai';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { registerGatewayRoutes } from './routes';
import { createGatewayRuntime } from './runtime';
import type { GatewayConfig, ProviderConfig } from '../types';

describe('openai sdk responses integration', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns parsed output_text for non-stream responses.create when upstream sends chat SSE', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_sdk_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"Arrr! "}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"Semicolons be optional."}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}\n\n',
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
      const client = new OpenAI({
        apiKey: 'sk-test',
        baseURL: 'http://gateway.local/v1',
        defaultHeaders: {
          'x-target-provider': 'openai-main'
        },
        fetch: createInjectFetch(app)
      });

      const response = await client.responses.create({
        model: 'glm-5',
        instructions: 'You are a coding assistant that talks like a pirate.',
        input: 'Are semicolons optional in JavaScript?'
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(response.output_text).toContain('Semicolons be optional.');
      expect(response.usage?.total_tokens).toBe(18);
    } finally {
      await app.close();
    }
  });

  it('supports responses.stream text deltas when upstream sends chat SSE', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_sdk_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_stream_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
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
      const client = new OpenAI({
        apiKey: 'sk-test',
        baseURL: 'http://gateway.local/v1',
        defaultHeaders: {
          'x-target-provider': 'openai-main'
        },
        fetch: createInjectFetch(app)
      });

      const deltas: string[] = [];
      const stream = client.responses.stream({
        model: 'glm-5',
        input: 'Say hello world'
      });

      stream.on('response.output_text.delta', (event) => {
        deltas.push(event.delta);
      });

      const response = await stream.finalResponse();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(deltas.join('')).toBe('hello world');
      expect(response.output_text).toBe('hello world');
      expect(response.output[0]?.type).toBe('message');
    } finally {
      await app.close();
    }
  });

  it('supports responses.stream function call arguments when upstream sends chat tool_calls SSE', async () => {
    const fetchMock = vi.fn(async () => {
      return createSseResponse([
        'data: {"id":"chatcmpl_sdk_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Sh"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"arguments":"anghai\\"}"}}]}}]}\n\n',
        'data: {"id":"chatcmpl_sdk_tool_1","object":"chat.completion.chunk","model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\n',
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
      const client = new OpenAI({
        apiKey: 'sk-test',
        baseURL: 'http://gateway.local/v1',
        defaultHeaders: {
          'x-target-provider': 'openai-main'
        },
        fetch: createInjectFetch(app)
      });

      const argumentDeltas: string[] = [];
      const stream = client.responses.stream({
        model: 'glm-5',
        input: 'Call get_weather for Shanghai'
      });

      stream.on('response.function_call_arguments.delta', (event) => {
        argumentDeltas.push(event.delta);
      });

      const response = await stream.finalResponse();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(argumentDeltas.join('')).toBe('{"city":"Shanghai"}');
      expect(response.output_text).toBe('');
      expect(response.output[0]?.type).toBe('function_call');
      if (response.output[0]?.type === 'function_call') {
        expect(response.output[0].name).toBe('get_weather');
        expect(response.output[0].arguments).toBe('{"city":"Shanghai"}');
        expect(response.output[0].call_id).toBe('call_weather');
      }
    } finally {
      await app.close();
    }
  });
});

function createInjectFetch(app: ReturnType<typeof Fastify>) {
  return async function injectFetch(input: Parameters<typeof fetch>[0], init?: RequestInit): Promise<Response> {
    const request = new Request(input, init);
    const url = new URL(request.url);

    const headers: Record<string, string> = {};
    request.headers.forEach((value, key) => {
      headers[key] = value;
    });

    const bodyBuffer =
      request.method === 'GET' || request.method === 'HEAD'
        ? undefined
        : Buffer.from(await request.arrayBuffer());

    const response = await app.inject({
      method: request.method,
      url: `${url.pathname}${url.search}`,
      headers,
      payload: bodyBuffer
    });

    const responseHeaders = new Headers();
    for (const [key, value] of Object.entries(response.headers)) {
      if (Array.isArray(value)) {
        for (const entry of value) {
          responseHeaders.append(key, String(entry));
        }
        continue;
      }

      if (value !== undefined) {
        responseHeaders.set(key, String(value));
      }
    }

    return new Response(response.body, {
      status: response.statusCode,
      headers: responseHeaders
    });
  };
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
