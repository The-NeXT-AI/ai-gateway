import { createServer, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { callUpstream } from './client';

// Exercises the real undici dispatcher (client.test.ts mocks undici entirely).

const proxyEnvNames = [
  'CCR_UPSTREAM_PROXY_URL',
  'http_proxy',
  'HTTP_PROXY',
  'https_proxy',
  'HTTPS_PROXY',
  'all_proxy',
  'ALL_PROXY',
  'no_proxy',
  'NO_PROXY'
];
const originalProxyEnv = new Map(proxyEnvNames.map((name) => [name, process.env[name]]));

let server: Server | undefined;
const pendingResponses: ServerResponse[] = [];

async function startServer(
  delayHeadersMs: (requestNumber: number) => number
): Promise<{ url: string; requests: () => number }> {
  let requestCount = 0;
  server = createServer((request, response) => {
    requestCount += 1;
    const delayMs = delayHeadersMs(requestCount);
    request.resume();
    request.on('end', () => {
      pendingResponses.push(response);
      setTimeout(() => {
        if (response.destroyed) {
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ attempt: requestCount }));
      }, delayMs);
    });
  });
  await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}/v1/responses`, requests: () => requestCount };
}

describe('callUpstream headers timeout against a real server', () => {
  beforeEach(() => {
    for (const name of proxyEnvNames) {
      delete process.env[name];
    }
  });

  afterEach(async () => {
    for (const response of pendingResponses.splice(0)) {
      response.destroy();
    }
    if (server) {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server!.close(() => resolve()));
      server = undefined;
    }
    for (const [name, value] of originalProxyEnv) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  it('times out slow response headers and succeeds on the retry', async () => {
    const upstream = await startServer((requestNumber) => (requestNumber === 1 ? 2000 : 0));
    const startedAt = Date.now();

    const response = await callUpstream(
      upstream.url,
      { 'content-type': 'application/json' },
      { model: 'test-model', input: 'hello' },
      5000,
      undefined,
      undefined,
      undefined,
      { headersTimeoutMs: 300, bodyIdleTimeoutMs: 1000 }
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ attempt: 2 });
    expect(upstream.requests()).toBe(2);
    expect(Date.now() - startedAt).toBeLessThan(2000);
  });

  it('stops retrying at the overall deadline', async () => {
    const upstream = await startServer(() => 2000);

    const error = await callUpstream(
      upstream.url,
      { 'content-type': 'application/json' },
      { model: 'test-model', input: 'hello' },
      500,
      undefined,
      undefined,
      { maxAttempts: 5 },
      { headersTimeoutMs: 300, bodyIdleTimeoutMs: 1000 }
    ).catch((caught) => caught);

    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe('Upstream request timed out after 500ms.');
    expect(upstream.requests()).toBe(2);
  });
});
