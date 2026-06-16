import { createServer, type IncomingHttpHeaders, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import Fastify from 'fastify';
import { afterEach, describe, expect, it } from 'vitest';
import { WebSocket, WebSocketServer } from 'ws';
import { ProviderPluginRegistry } from '../adapters/registry';
import type { GatewayConfig } from '../types';
import { registerGatewayResponsesWebSocketRoute } from './websocket';

describe('gateway responses websocket relay', () => {
  const cleanupTasks: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanupTasks.length > 0) {
      const task = cleanupTasks.pop();
      if (!task) {
        continue;
      }
      await task();
    }
  });

  it('forwards response.completed before closing when upstream closes immediately', async () => {
    const upstream = await startUpstreamResponsesWebSocketServer({
      expectedPath: '/v1/responses'
    });
    cleanupTasks.push(async () => {
      await upstream.close();
    });

    const gateway = Fastify({ logger: false });
    const gatewayConfig = createWsGatewayTestConfig(`http://127.0.0.1:${upstream.port}/v1`);
    registerGatewayResponsesWebSocketRoute(gateway, gatewayConfig);
    await gateway.listen({ host: '127.0.0.1', port: 0 });
    cleanupTasks.push(async () => {
      await gateway.close();
    });

    const gatewayPort = (gateway.server.address() as AddressInfo).port;
    const receivedMessages: string[] = [];

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const socket = new WebSocket(`ws://127.0.0.1:${gatewayPort}/v1/responses`);
      const timeout = setTimeout(() => {
        settled = true;
        socket.terminate();
        reject(new Error('Timed out waiting for response.completed event.'));
      }, 8000);

      socket.on('open', () => {
        socket.send(
          JSON.stringify({
            type: 'response.create',
            model: 'gpt-5.4-mini',
            input: 'hello',
            stream: true
          })
        );
      });

      socket.on('message', (raw) => {
        const message = raw.toString();
        receivedMessages.push(message);
        if (!message.includes('"type":"response.completed"')) {
          return;
        }

        if (settled) {
          return;
        }

        settled = true;
        clearTimeout(timeout);
        socket.close(1000, 'test-done');
        resolve();
      });

      socket.on('close', () => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeout);
        reject(new Error('WebSocket closed before response.completed event was received.'));
      });

      socket.on('error', (error) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeout);
        reject(error);
      });
    });

    expect(receivedMessages.some((message) => message.includes('"type":"response.completed"'))).toBe(true);
    const completedMessage = receivedMessages.find((message) => message.includes('"type":"response.completed"'));
    const completedPayload = JSON.parse(completedMessage || '{}') as {
      response?: { usage?: Record<string, unknown> };
    };
    expect(completedPayload.response?.usage).toMatchObject({
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
  }, 12000);

  it('maps codex headers and normalizes response.create payload for codex backend', async () => {
    const upstream = await startUpstreamResponsesWebSocketServer({
      expectedPath: '/backend-api/codex/responses'
    });
    cleanupTasks.push(async () => {
      await upstream.close();
    });

    const gateway = Fastify({ logger: false });
    const gatewayConfig = createWsGatewayTestConfig(`http://127.0.0.1:${upstream.port}/backend-api/codex`);
    registerGatewayResponsesWebSocketRoute(gateway, gatewayConfig);
    await gateway.listen({ host: '127.0.0.1', port: 0 });
    cleanupTasks.push(async () => {
      await gateway.close();
    });

    const gatewayPort = (gateway.server.address() as AddressInfo).port;

    await waitForResponseCompleted(`ws://127.0.0.1:${gatewayPort}/v1/responses`, {
      'x-codex-access-token': 'atk-test-codex',
      'x-codex-account-id': 'acct-test-codex'
    }, {
      type: 'response.create',
      model: 'gpt-5.4-mini',
      input: 'hello codex ws'
    });

    expect(upstream.state.upgradeHeaders?.authorization).toBe('Bearer atk-test-codex');
    expect(upstream.state.upgradeHeaders?.['chatgpt-account-id']).toBe('acct-test-codex');

    const upstreamPayload = JSON.parse(upstream.state.receivedMessages[0] || '{}') as Record<string, unknown>;
    expect(upstreamPayload.type).toBe('response.create');
    expect(upstreamPayload.stream).toBe(true);
    expect(upstreamPayload.store).toBe(false);
    expect(upstreamPayload.instructions).toBe('You are a helpful assistant.');
  });

  it('prefers openai_responses provider base url for websocket upstream target', async () => {
    const upstream = await startUpstreamResponsesWebSocketServer({
      expectedPath: '/backend-api/codex/responses'
    });
    cleanupTasks.push(async () => {
      await upstream.close();
    });

    const gateway = Fastify({ logger: false });
    const gatewayConfig = createWsGatewayTestConfig(`http://127.0.0.1:${upstream.port}/v1`);
    gatewayConfig.providers = [
      {
        name: 'bigmodel',
        type: 'openai_chat_completions',
        apikey: 'bigmodel-key',
        baseurl: 'https://open.bigmodel.cn/api/coding/paas/v4',
        models: ['glm-5'],
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
      },
      {
        name: 'codex',
        type: 'openai_responses',
        baseurl: `http://127.0.0.1:${upstream.port}/backend-api/codex`,
        models: ['gpt-5.4'],
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
      }
    ] as any;
    registerGatewayResponsesWebSocketRoute(gateway, gatewayConfig);
    await gateway.listen({ host: '127.0.0.1', port: 0 });
    cleanupTasks.push(async () => {
      await gateway.close();
    });

    const gatewayPort = (gateway.server.address() as AddressInfo).port;
    await waitForResponseCompleted(`ws://127.0.0.1:${gatewayPort}/v1/responses`, {
      authorization: 'Bearer codex-access-token'
    }, {
      type: 'response.create',
      model: 'gpt-5.4-mini',
      input: 'hello target'
    });

    expect(upstream.state.upgradePath).toBe('/backend-api/codex/responses');
  });

  it('keeps single /responses suffix when base url already ends with /responses', async () => {
    const upstream = await startUpstreamResponsesWebSocketServer({
      expectedPath: '/backend-api/codex/responses'
    });
    cleanupTasks.push(async () => {
      await upstream.close();
    });

    const gateway = Fastify({ logger: false });
    const gatewayConfig = createWsGatewayTestConfig(
      `http://127.0.0.1:${upstream.port}/backend-api/codex/responses`
    );
    registerGatewayResponsesWebSocketRoute(gateway, gatewayConfig);
    await gateway.listen({ host: '127.0.0.1', port: 0 });
    cleanupTasks.push(async () => {
      await gateway.close();
    });

    const gatewayPort = (gateway.server.address() as AddressInfo).port;
    await waitForResponseCompleted(`ws://127.0.0.1:${gatewayPort}/v1/responses`, undefined, {
      type: 'response.create',
      model: 'gpt-5.4-mini',
      input: 'hello dedupe'
    });

    expect(upstream.state.upgradePath).toBe('/backend-api/codex/responses');
  });

  it('applies provider plugin auth headers for websocket upstream connection', async () => {
    const upstream = await startUpstreamResponsesWebSocketServer({
      expectedPath: '/backend-api/codex/responses'
    });
    cleanupTasks.push(async () => {
      await upstream.close();
    });

    const gateway = Fastify({ logger: false });
    const gatewayConfig = createWsGatewayTestConfig(`http://127.0.0.1:${upstream.port}/v1`);
    gatewayConfig.providers = [
      {
        name: 'codex',
        type: 'openai_responses',
        baseurl: `http://127.0.0.1:${upstream.port}/backend-api/codex`,
        models: ['gpt-5.4'],
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
      }
    ] as any;

    const providerPlugins = new ProviderPluginRegistry();
    providerPlugins.register({
      key: 'ws-plugin-auth',
      provider: 'openai',
      providerName: 'codex',
      authenticate: (input) => {
        return {
          ok: true,
          value: {
            ...input.upstreamRequest,
            headers: {
              ...input.upstreamRequest.headers,
              authorization: 'Bearer plugin-token',
              'chatgpt-account-id': 'acct-from-plugin'
            }
          }
        };
      }
    });

    registerGatewayResponsesWebSocketRoute(gateway, gatewayConfig, {
      providerPlugins
    } as any);
    await gateway.listen({ host: '127.0.0.1', port: 0 });
    cleanupTasks.push(async () => {
      await gateway.close();
    });

    const gatewayPort = (gateway.server.address() as AddressInfo).port;
    await waitForResponseCompleted(`ws://127.0.0.1:${gatewayPort}/v1/responses`, {
      authorization: 'Bearer gateway-token'
    }, {
      type: 'response.create',
      model: 'gpt-5.4-mini',
      input: 'hello plugin'
    });

    expect(upstream.state.upgradeHeaders?.authorization).toBe('Bearer plugin-token');
    expect(upstream.state.upgradeHeaders?.['chatgpt-account-id']).toBe('acct-from-plugin');
  });
});

async function startUpstreamResponsesWebSocketServer(options: {
  expectedPath?: string;
} = {}): Promise<{
  port: number;
  state: {
    upgradePath?: string;
    upgradeHeaders?: IncomingHttpHeaders;
    receivedMessages: string[];
  };
  close: () => Promise<void>;
}> {
  const expectedPath = options.expectedPath || '/v1/responses';
  const server = createServer();
  const websocketServer = new WebSocketServer({ noServer: true });
  const state: {
    upgradePath?: string;
    upgradeHeaders?: IncomingHttpHeaders;
    receivedMessages: string[];
  } = {
    receivedMessages: []
  };

  server.on('upgrade', (request, socket, head) => {
    const url = new URL(request.url || '/', 'http://localhost');
    state.upgradePath = url.pathname;
    state.upgradeHeaders = request.headers;
    if (url.pathname !== expectedPath) {
      socket.destroy();
      return;
    }

    websocketServer.handleUpgrade(request, socket, head, (ws) => {
      websocketServer.emit('connection', ws, request);
    });
  });

  websocketServer.on('connection', (socket) => {
    socket.on('message', (raw) => {
      state.receivedMessages.push(raw.toString());
      const completionPayload = JSON.stringify({
        type: 'response.completed',
        response: {
          id: 'resp_test_1',
          object: 'response',
          status: 'completed',
          output: [
            {
              id: 'msg_test_1',
              type: 'message',
              role: 'assistant',
              status: 'completed',
              content: [
                {
                  type: 'output_text',
                  text: 'x'.repeat(64 * 1024)
                }
              ]
            }
          ]
        }
      });

      socket.send(completionPayload);
      socket.close(1000, 'done');
    });
  });

  await listen(server);
  const port = (server.address() as AddressInfo).port;

  return {
    port,
    state,
    close: async () => {
      for (const client of websocketServer.clients) {
        client.terminate();
      }
      await closeWebSocketServer(websocketServer);
      await closeServer(server);
    }
  };
}

async function waitForResponseCompleted(
  websocketUrl: string,
  headers: Record<string, string> | undefined,
  payload: Record<string, unknown>
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const socket = new WebSocket(websocketUrl, {
      headers
    });
    const timeout = setTimeout(() => {
      settled = true;
      socket.terminate();
      reject(new Error('Timed out waiting for response.completed event.'));
    }, 8000);

    socket.on('open', () => {
      socket.send(JSON.stringify(payload));
    });

    socket.on('message', (raw) => {
      const message = raw.toString();
      if (!message.includes('"type":"response.completed"')) {
        return;
      }

      if (settled) {
        return;
      }

      settled = true;
      clearTimeout(timeout);
      socket.close(1000, 'test-done');
      resolve();
    });

    socket.on('close', () => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      reject(new Error('WebSocket closed before response.completed event was received.'));
    });

    socket.on('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      reject(error);
    });
  });
}

function createWsGatewayTestConfig(openaiBaseUrl: string): GatewayConfig {
  return {
    openaiBaseUrl,
    openaiApiKey: 'openai-test-key',
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
    }
  } as unknown as GatewayConfig;
}

async function listen(server: Server): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.listen(0, '127.0.0.1', (error?: Error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

async function closeWebSocketServer(server: WebSocketServer): Promise<void> {
  await new Promise<void>((resolve) => {
    server.close(() => resolve());
  });
}

async function closeServer(server: Server): Promise<void> {
  await new Promise<void>((resolve) => {
    server.close(() => resolve());
  });
}
