import { Readable } from 'node:stream';
import Fastify from 'fastify';
import { afterEach, describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import type { GatewayConfig, GatewayRequestIdentity } from '../types';
import {
  createGatewayIdempotencyPreHandler,
  registerGatewayIdempotencyHooks,
  resetGatewayIdempotencyForTests
} from './idempotency';

describe('gateway idempotency', () => {
  afterEach(() => {
    resetGatewayIdempotencyForTests();
  });

  it('replays a cached successful JSON POST response without invoking the handler again', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/test', { preHandler }, async (_request, reply) => {
      calls += 1;
      return reply.header('x-upstream-call', String(calls)).send({ calls });
    });
    await app.ready();

    try {
      const first = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'retry-key'
        },
        payload: {
          prompt: 'hello'
        }
      });
      const second = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'retry-key'
        },
        payload: {
          prompt: 'hello'
        }
      });

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(JSON.parse(first.body)).toEqual({ calls: 1 });
      expect(JSON.parse(second.body)).toEqual({ calls: 1 });
      expect(first.headers['x-gateway-idempotency-status']).toBe('stored');
      expect(second.headers['x-gateway-idempotency-status']).toBe('replayed');
      expect(second.headers['x-upstream-call']).toBe('1');
      expect(calls).toBe(1);
    } finally {
      await app.close();
    }
  });

  it('hashes binary request bodies directly for replay and conflict detection', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    app.addContentTypeParser(
      'multipart/form-data',
      { parseAs: 'buffer' },
      (_request, body, done) => done(null, body)
    );
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/test', { preHandler }, async () => {
      calls += 1;
      return { calls };
    });
    await app.ready();

    const headers = {
      'content-type': 'multipart/form-data; boundary=binary-test',
      'idempotency-key': 'binary-key'
    };
    try {
      const first = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers,
        payload: Buffer.from('--binary-test\r\nfirst\r\n--binary-test--\r\n')
      });
      const replay = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers,
        payload: Buffer.from('--binary-test\r\nfirst\r\n--binary-test--\r\n')
      });
      const conflict = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers,
        payload: Buffer.from('--binary-test\r\nsecond\r\n--binary-test--\r\n')
      });

      expect(first.statusCode).toBe(200);
      expect(replay.headers['x-gateway-idempotency-status']).toBe('replayed');
      expect(conflict.statusCode).toBe(409);
      expect(conflict.headers['x-gateway-idempotency-status']).toBe('conflict');
      expect(calls).toBe(1);
    } finally {
      await app.close();
    }
  });

  it('rejects reuse of the same key with a different request fingerprint', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/test', { preHandler }, async () => {
      calls += 1;
      return { calls };
    });
    await app.ready();

    try {
      await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'conflict-key'
        },
        payload: {
          prompt: 'first'
        }
      });
      const conflict = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'conflict-key'
        },
        payload: {
          prompt: 'second'
        }
      });

      expect(conflict.statusCode).toBe(409);
      expect(conflict.headers['x-gateway-idempotency-status']).toBe('conflict');
      expect(JSON.parse(conflict.body).error.code).toBe('idempotency_key_conflict');
      expect(calls).toBe(1);
    } finally {
      await app.close();
    }
  });

  it('scopes cached responses by authenticated gateway identity', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post(
      '/v1/test',
      {
        preHandler: [
          async (request) => {
            const userId = String(request.headers['x-test-user'] || 'anonymous');
            request.gatewayIdentity = {
              source: 'trusted_header',
              billingSubjectKey: `user:${userId}`,
              userId
            } satisfies GatewayRequestIdentity;
          },
          preHandler
        ]
      },
      async () => {
        calls += 1;
        return { calls };
      }
    );
    await app.ready();

    try {
      const first = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'shared-key',
          'x-test-user': 'alice'
        },
        payload: {
          prompt: 'same'
        }
      });
      const second = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'shared-key',
          'x-test-user': 'bob'
        },
        payload: {
          prompt: 'same'
        }
      });
      const replay = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'shared-key',
          'x-test-user': 'alice'
        },
        payload: {
          prompt: 'same'
        }
      });

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(replay.statusCode).toBe(200);
      expect(JSON.parse(first.body)).toEqual({ calls: 1 });
      expect(JSON.parse(second.body)).toEqual({ calls: 2 });
      expect(JSON.parse(replay.body)).toEqual({ calls: 1 });
      expect(second.headers['x-gateway-idempotency-status']).toBe('stored');
      expect(replay.headers['x-gateway-idempotency-status']).toBe('replayed');
      expect(calls).toBe(2);
    } finally {
      await app.close();
    }
  });

  it('scopes cached responses by SDK-compatible auth headers when auth is disabled', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/test', { preHandler }, async () => {
      calls += 1;
      return { calls };
    });
    await app.ready();

    try {
      const first = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'sdk-key',
          'x-goog-api-key': 'token-a'
        },
        payload: {
          prompt: 'same'
        }
      });
      const second = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'sdk-key',
          'x-goog-api-key': 'token-b'
        },
        payload: {
          prompt: 'same'
        }
      });
      const replay = await app.inject({
        method: 'POST',
        url: '/v1/test',
        headers: {
          'idempotency-key': 'sdk-key',
          'x-goog-api-key': 'token-a'
        },
        payload: {
          prompt: 'same'
        }
      });

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(replay.statusCode).toBe(200);
      expect(JSON.parse(first.body)).toEqual({ calls: 1 });
      expect(JSON.parse(second.body)).toEqual({ calls: 2 });
      expect(JSON.parse(replay.body)).toEqual({ calls: 1 });
      expect(calls).toBe(2);
    } finally {
      await app.close();
    }
  });

  it('waits for an in-flight matching request and replays the completed response', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/test', { preHandler }, async () => {
      calls += 1;
      await delay(20);
      return { calls };
    });
    await app.ready();

    try {
      const request = {
        method: 'POST' as const,
        url: '/v1/test',
        headers: {
          'idempotency-key': 'pending-key'
        },
        payload: {
          prompt: 'same'
        }
      };
      const [first, second] = await Promise.all([app.inject(request), app.inject(request)]);

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(JSON.parse(first.body)).toEqual({ calls: 1 });
      expect(JSON.parse(second.body)).toEqual({ calls: 1 });
      expect(calls).toBe(1);
    } finally {
      await app.close();
    }
  });

  it('caches non-event-stream Readable responses after the stream completes', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/streamed-json', { preHandler }, async (_request, reply) => {
      calls += 1;
      return reply
        .header('content-type', 'application/json')
        .header('x-upstream-call', String(calls))
        .send(Readable.from([JSON.stringify({ calls })]));
    });
    await app.ready();

    try {
      const request = {
        method: 'POST' as const,
        url: '/v1/streamed-json',
        headers: {
          'idempotency-key': 'streamed-json-key'
        },
        payload: {
          prompt: 'same'
        }
      };
      const first = await app.inject(request);
      const second = await app.inject(request);

      expect(first.statusCode).toBe(200);
      expect(second.statusCode).toBe(200);
      expect(JSON.parse(first.body)).toEqual({ calls: 1 });
      expect(JSON.parse(second.body)).toEqual({ calls: 1 });
      expect(first.headers['x-gateway-idempotency-status']).toBe('stored');
      expect(second.headers['x-gateway-idempotency-status']).toBe('replayed');
      expect(second.headers['x-upstream-call']).toBe('1');
      expect(calls).toBe(1);
    } finally {
      await app.close();
    }
  });

  it('does not cache event-stream responses', async () => {
    const config = createConfig();
    const app = Fastify({ logger: false });
    registerGatewayIdempotencyHooks(app, config);
    const preHandler = createGatewayIdempotencyPreHandler(config);
    let calls = 0;
    app.post('/v1/stream', { preHandler }, async (_request, reply) => {
      calls += 1;
      return reply.header('content-type', 'text/event-stream').send(`data: ${calls}\n\n`);
    });
    await app.ready();

    try {
      const request = {
        method: 'POST' as const,
        url: '/v1/stream',
        headers: {
          'idempotency-key': 'stream-key'
        },
        payload: {
          stream: true
        }
      };
      const first = await app.inject(request);
      const second = await app.inject(request);

      expect(first.body).toBe('data: 1\n\n');
      expect(second.body).toBe('data: 2\n\n');
      expect(calls).toBe(2);
    } finally {
      await app.close();
    }
  });
});

function createConfig(): GatewayConfig {
  return parseGatewayConfigFromRaw({
    idempotency: {
      enabled: true,
      ttlMs: 60000,
      maxEntries: 100
    }
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
