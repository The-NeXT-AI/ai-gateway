import { createServer, type Server } from 'node:http';
import { createServer as createHttp2Server, type Http2Server, type ServerHttp2Stream } from 'node:http2';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { WebSocketServer } from 'ws';
import { parseGatewayConfigFromRaw } from './config';
import {
  refreshGatewayConfigFromExternalSource,
  startGatewayExternalConfigPoller
} from './external-config';
import { encodeGrpcJsonMessage } from './grpc-json';

describe('gateway external config source', () => {
  const servers: Server[] = [];
  const http2Servers: Http2Server[] = [];
  const webSocketServers: WebSocketServer[] = [];

  afterEach(async () => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    await Promise.all(webSocketServers.splice(0).map((server) => closeWebSocketServer(server)));
    await Promise.all(http2Servers.splice(0).map((server) => closeHttp2Server(server)));
    await Promise.all(servers.splice(0).map((server) => closeHttpServer(server)));
  });

  it('refreshes gateway config from HTTP endpoint and triggers reload callback', async () => {
    const config = parseGatewayConfigFromRaw({
      Providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          models: ['gpt-4.1-mini'],
          apikey: 'openai-key'
        }
      ],
      configExternal: {
        enabled: true,
        endpoint: 'https://config.example.com/gateway',
        method: 'POST',
        apiKeyHeader: 'x-config-key',
        apiKey: 'config-secret',
        timeoutMs: 5000
      }
    });
    const onConfigReload = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          config: {
            Providers: [
              {
                name: 'anthropic-main',
                type: 'anthropic_messages',
                models: ['claude-3-7-sonnet'],
                apikey: 'anthropic-key'
              }
            ],
            billingWebhook: {
              enabled: true,
              endpoint: 'https://billing.example.com/usage'
            }
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

    const refreshed = await refreshGatewayConfigFromExternalSource({
      config,
      onConfigReload,
      reason: 'test_refresh'
    });

    expect(refreshed).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://config.example.com/gateway',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ type: 'gateway_config_request' })
      })
    );
    const requestInit = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(requestInit.headers).toMatchObject({
      accept: 'application/json',
      'x-config-key': 'config-secret'
    });
    expect(config.providers).toHaveLength(1);
    expect(config.providers[0]?.name).toBe('anthropic-main');
    expect(config.defaultTargetProviders).toEqual(['anthropic']);
    expect(config.billingWebhook.enabled).toBe(true);
    expect(config.billingWebhook.endpoint).toBe('https://billing.example.com/usage');
    expect(config.configExternal?.endpoint).toBe('https://config.example.com/gateway');
    expect(onConfigReload).toHaveBeenCalledWith(config, 'test_refresh');
  });

  it('does not fetch when external config source is disabled', async () => {
    const config = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: false,
        endpoint: 'https://config.example.com/gateway'
      }
    });
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const refreshed = await refreshGatewayConfigFromExternalSource({ config });

    expect(refreshed).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refreshes gateway config from WebSocket endpoint', async () => {
    const { server, webSocketServer, url, nextRequest } = await startConfigWebSocketServer({
      gatewayConfig: {
        Providers: [
          {
            name: 'gemini-ws',
            type: 'gemini_generate_content',
            models: ['gemini-2.0-flash'],
            apikey: 'gemini-key'
          }
        ]
      }
    });
    servers.push(server);
    webSocketServers.push(webSocketServer);
    const config = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        transport: 'websocket',
        endpoint: url,
        timeoutMs: 5000,
        apiKeyHeader: 'x-config-key',
        apiKey: 'ws-secret'
      }
    });

    const refreshed = await refreshGatewayConfigFromExternalSource({ config });

    expect(refreshed).toBe(true);
    const request = await nextRequest;
    expect(request.headers['x-config-key']).toBe('ws-secret');
    expect(request.payload).toEqual({ type: 'gateway_config_request' });
    expect(config.providers[0]?.name).toBe('gemini-ws');
    expect(config.configExternal?.transport).toBe('websocket');
  });

  it('refreshes gateway config from gRPC JSON endpoint', async () => {
    const { server, url, nextRequest } = await startConfigGrpcJsonServer({
      config: {
        Providers: [
          {
            name: 'openai-grpc',
            type: 'openai_responses',
            models: ['gpt-4.1-mini'],
            apikey: 'grpc-key'
          }
        ]
      }
    });
    http2Servers.push(server);
    const config = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        transport: 'grpc',
        endpoint: url,
        timeoutMs: 5000,
        apiKeyHeader: 'x-config-key',
        apiKey: 'grpc-secret'
      }
    });

    const refreshed = await refreshGatewayConfigFromExternalSource({ config });

    expect(refreshed).toBe(true);
    const request = await nextRequest;
    expect(request.path).toBe('/gateway.config.v1.ConfigService/GetConfig');
    expect(request.headers['x-config-key']).toBe('grpc-secret');
    expect(request.payload).toEqual({ type: 'gateway_config_request' });
    expect(config.providers[0]?.name).toBe('openai-grpc');
    expect(config.configExternal?.transport).toBe('grpc');
  });

  it('refreshes gateway config from stdio command', async () => {
    const config = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        transport: 'stdio',
        command: process.execPath,
        args: [
          '-e',
          'let input="";process.stdin.on("data",c=>input+=c);process.stdin.on("end",()=>process.stdout.write(JSON.stringify({config:{Providers:[{name:"openai-stdio",type:"openai_responses",models:["gpt-4.1-mini"],apikey:"stdio-key"}]}})));'
        ],
        timeoutMs: 5000
      }
    });

    const refreshed = await refreshGatewayConfigFromExternalSource({ config });

    expect(refreshed).toBe(true);
    expect(config.providers[0]?.name).toBe('openai-stdio');
    expect(config.configExternal?.transport).toBe('stdio');
    expect(config.configExternal?.command).toBe(process.execPath);
  });

  it('polls external config source when interval is configured', async () => {
    vi.useFakeTimers();
    const config = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        endpoint: 'https://config.example.com/gateway',
        intervalMs: 1000
      }
    });
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          gatewayConfig: {
            Providers: [
              {
                name: 'openai-poll',
                type: 'openai_responses',
                models: ['gpt-4.1-mini'],
                apikey: 'poll-key'
              }
            ]
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
    const onConfigReload = vi.fn();

    const poller = startGatewayExternalConfigPoller({
      config,
      onConfigReload
    });
    await vi.advanceTimersByTimeAsync(1000);

    expect(poller).toBeDefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(config.providers[0]?.name).toBe('openai-poll');
    expect(config.configExternal?.intervalMs).toBe(1000);
    expect(onConfigReload).toHaveBeenCalledWith(config, 'external_config_poll');

    poller?.close();
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

async function startConfigWebSocketServer(payload: unknown): Promise<{
  server: Server;
  webSocketServer: WebSocketServer;
  url: string;
  nextRequest: Promise<{ headers: Record<string, string | string[] | undefined>; payload: unknown }>;
}> {
  const server = createServer();
  const webSocketServer = new WebSocketServer({ server });
  let resolveRequest!: (value: {
    headers: Record<string, string | string[] | undefined>;
    payload: unknown;
  }) => void;
  const nextRequest = new Promise<{ headers: Record<string, string | string[] | undefined>; payload: unknown }>(
    (resolve) => {
      resolveRequest = resolve;
    }
  );
  webSocketServer.on('connection', (socket, request) => {
    socket.on('message', (data) => {
      resolveRequest({
        headers: request.headers,
        payload: JSON.parse(data.toString())
      });
      socket.send(JSON.stringify(payload));
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address() as AddressInfo;
  return {
    server,
    webSocketServer,
    url: `ws://127.0.0.1:${address.port}/config`,
    nextRequest
  };
}

async function startConfigGrpcJsonServer(payload: unknown): Promise<{
  server: Http2Server;
  url: string;
  nextRequest: Promise<{ path: string; headers: Record<string, string | string[] | undefined>; payload: unknown }>;
}> {
  const server = createHttp2Server();
  let resolveRequest!: (value: {
    path: string;
    headers: Record<string, string | string[] | undefined>;
    payload: unknown;
  }) => void;
  const nextRequest = new Promise<{ path: string; headers: Record<string, string | string[] | undefined>; payload: unknown }>(
    (resolve) => {
      resolveRequest = resolve;
    }
  );

  server.on('stream', (stream: ServerHttp2Stream, headers) => {
    const chunks: Buffer[] = [];
    stream.on('data', (chunk) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    stream.on('end', () => {
      resolveRequest({
        path: String(headers[':path'] || ''),
        headers: headers as Record<string, string | string[] | undefined>,
        payload: decodeGrpcJsonMessage(Buffer.concat(chunks))
      });
      stream.respond({
        ':status': 200,
        'content-type': 'application/grpc+json',
        'grpc-status': '0'
      });
      stream.end(encodeGrpcJsonMessage(payload));
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address() as AddressInfo;
  return {
    server,
    url: `grpc://127.0.0.1:${address.port}`,
    nextRequest
  };
}

function decodeGrpcJsonMessage(payload: Buffer): unknown {
  const length = payload.readUInt32BE(1);
  return JSON.parse(payload.subarray(5, 5 + length).toString('utf8'));
}

function closeWebSocketServer(server: WebSocketServer): Promise<void> {
  for (const client of server.clients) {
    client.close();
  }
  return new Promise((resolve) => server.close(() => resolve()));
}

function closeHttpServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function closeHttp2Server(server: Http2Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}
