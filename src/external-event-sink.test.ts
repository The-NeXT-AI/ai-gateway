import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer, type Server } from 'node:http';
import { createServer as createHttp2Server, type Http2Server, type ServerHttp2Stream } from 'node:http2';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { WebSocketServer } from 'ws';
import { publishJsonEventToExternalSink } from './external-event-sink';
import { encodeGrpcJsonMessage } from './grpc-json';

describe('external event sink', () => {
  const servers: Server[] = [];
  const http2Servers: Http2Server[] = [];
  const webSocketServers: WebSocketServer[] = [];
  const tempDirs: string[] = [];

  afterEach(async () => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    await Promise.all(webSocketServers.splice(0).map((server) => closeWebSocketServer(server)));
    await Promise.all(http2Servers.splice(0).map((server) => closeHttp2Server(server)));
    await Promise.all(servers.splice(0).map((server) => closeHttpServer(server)));
    for (const dir of tempDirs.splice(0)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('publishes JSON events through HTTP', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-1'
      },
      {
        transport: 'http',
        endpoint: 'https://sink.example.com/events',
        timeoutMs: 5000,
        headers: {
          authorization: 'Bearer sink-secret'
        }
      }
    );

    expect(delivered).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://sink.example.com/events',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ eventId: 'event-1' })
      })
    );
  });

  it('retries transient HTTP sink failures before succeeding', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response('temporary unavailable', { status: 503 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-retry'
      },
      {
        transport: 'http',
        endpoint: 'https://sink.example.com/events',
        timeoutMs: 5000,
        maxAttempts: 2,
        baseDelayMs: 1,
        maxDelayMs: 1,
        headers: {}
      }
    );

    expect(delivered).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('publishes JSON events through WebSocket', async () => {
    const { server, webSocketServer, url, nextMessage } = await startWebSocketSink();
    servers.push(server);
    webSocketServers.push(webSocketServer);

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-ws',
        type: 'billing'
      },
      {
        transport: 'websocket',
        endpoint: url,
        timeoutMs: 5000,
        headers: {
          'x-sink-key': 'sink-secret'
        }
      }
    );

    expect(delivered).toBe(true);
    const received = await nextMessage;
    expect(received.headers['x-sink-key']).toBe('sink-secret');
    expect(received.payload).toEqual({
      eventId: 'event-ws',
      type: 'billing'
    });
  });

  it('waits for WebSocket acknowledgement when required', async () => {
    const { server, webSocketServer, url, nextMessage } = await startWebSocketSink({ ackPayload: { ok: true } });
    servers.push(server);
    webSocketServers.push(webSocketServer);

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-ws-ack'
      },
      {
        transport: 'websocket',
        endpoint: url,
        timeoutMs: 5000,
        requireAck: true,
        headers: {}
      }
    );

    expect(delivered).toBe(true);
    await expect(nextMessage).resolves.toMatchObject({
      payload: {
        eventId: 'event-ws-ack'
      }
    });
  });

  it('publishes JSON events through gRPC JSON unary', async () => {
    const { server, url, nextRequest } = await startGrpcJsonSink();
    http2Servers.push(server);

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-grpc',
        type: 'billing'
      },
      {
        transport: 'grpc',
        endpoint: url,
        timeoutMs: 5000,
        headers: {
          'x-sink-key': 'sink-secret'
        }
      }
    );

    expect(delivered).toBe(true);
    const request = await nextRequest;
    expect(request.path).toBe('/gateway.events.v1.EventSink/Publish');
    expect(request.headers['x-sink-key']).toBe('sink-secret');
    expect(request.payload).toEqual({
      eventId: 'event-grpc',
      type: 'billing'
    });
  });

  it('publishes JSON events through stdio command stdin', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gateway-stdio-sink-'));
    tempDirs.push(dir);
    const outputPath = join(dir, 'event.jsonl');

    const delivered = await publishJsonEventToExternalSink(
      {
        eventId: 'event-stdio'
      },
      {
        transport: 'stdio',
        command: process.execPath,
        args: [
          '-e',
          'const fs=require("fs");let input="";process.stdin.on("data",c=>input+=c);process.stdin.on("end",()=>fs.writeFileSync(process.env.OUT,input));'
        ],
        env: {
          OUT: outputPath
        },
        timeoutMs: 5000,
        headers: {}
      }
    );

    expect(delivered).toBe(true);
    expect(JSON.parse(readFileSync(outputPath, 'utf8'))).toEqual({
      eventId: 'event-stdio'
    });
  });
});

async function startWebSocketSink(options?: { ackPayload?: unknown }): Promise<{
  server: Server;
  webSocketServer: WebSocketServer;
  url: string;
  nextMessage: Promise<{ headers: Record<string, string | string[] | undefined>; payload: unknown }>;
}> {
  const server = createServer();
  const webSocketServer = new WebSocketServer({ server });
  let resolveMessage!: (value: {
    headers: Record<string, string | string[] | undefined>;
    payload: unknown;
  }) => void;
  const nextMessage = new Promise<{ headers: Record<string, string | string[] | undefined>; payload: unknown }>(
    (resolve) => {
      resolveMessage = resolve;
    }
  );
  webSocketServer.on('connection', (socket, request) => {
    socket.on('message', (data) => {
      resolveMessage({
        payload: JSON.parse(data.toString()),
        headers: request.headers
      });
      if (options && Object.prototype.hasOwnProperty.call(options, 'ackPayload')) {
        socket.send(JSON.stringify(options.ackPayload));
      }
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address() as AddressInfo;
  return {
    server,
    webSocketServer,
    url: `ws://127.0.0.1:${address.port}/events`,
    nextMessage
  };
}

async function startGrpcJsonSink(): Promise<{
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
      stream.end(encodeGrpcJsonMessage({ ok: true }));
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
