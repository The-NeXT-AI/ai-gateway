import Fastify from 'fastify';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { registerLenientJsonParser } from './lenient-json-parser';

const hasZstdCli = spawnSync('zstd', ['--version'], { encoding: 'utf8' }).status === 0;

describe('registerLenientJsonParser', () => {
  it('accepts JSON payload when Content-Length is incorrect', async () => {
    const app = Fastify({ logger: false });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json',
          'content-length': '999'
        },
        payload: JSON.stringify({ hello: 'world' })
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        body: {
          hello: 'world'
        }
      });
    } finally {
      await app.close();
    }
  });

  it('still rejects invalid JSON payload', async () => {
    const app = Fastify({ logger: false });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json'
        },
        payload: '{invalid-json'
      });

      expect(response.statusCode).toBe(400);
      expect(JSON.parse(response.body)).toMatchObject({
        code: 'FST_ERR_CTP_INVALID_JSON_BODY'
      });
    } finally {
      await app.close();
    }
  });

  it('keeps Fastify bodyLimit enforcement', async () => {
    const app = Fastify({ logger: false, bodyLimit: 32 });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json'
        },
        payload: JSON.stringify({
          message: 'this payload is intentionally longer than 32 bytes'
        })
      });

      expect(response.statusCode).toBe(413);
      expect(JSON.parse(response.body)).toMatchObject({
        code: 'FST_ERR_CTP_BODY_TOO_LARGE'
      });
    } finally {
      await app.close();
    }
  });

  it('accepts gzip-compressed JSON body', async () => {
    const app = Fastify({ logger: false });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json',
          'content-encoding': 'gzip'
        },
        payload: gzipSync(Buffer.from(JSON.stringify({ hello: 'compressed' }), 'utf8'))
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        body: {
          hello: 'compressed'
        }
      });
    } finally {
      await app.close();
    }
  });

  it('rejects gzip-compressed JSON body when decompressed output exceeds bodyLimit', async () => {
    const app = Fastify({ logger: false, bodyLimit: 128 });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const source = Buffer.from(JSON.stringify({ message: 'a'.repeat(4096) }), 'utf8');
      const compressed = gzipSync(source);
      expect(compressed.length).toBeLessThan(128);

      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json',
          'content-encoding': 'gzip'
        },
        payload: compressed
      });

      expect(response.statusCode).toBe(413);
      expect(JSON.parse(response.body)).toMatchObject({
        code: 'FST_ERR_CTP_BODY_TOO_LARGE'
      });
    } finally {
      await app.close();
    }
  });

  (hasZstdCli ? it : it.skip)('accepts zstd-compressed JSON body', async () => {
    const app = Fastify({ logger: false });
    registerLenientJsonParser(app);
    app.post('/echo', async (request) => {
      return { body: request.body };
    });

    await app.ready();
    try {
      const source = Buffer.from(JSON.stringify({ hello: 'zstd' }), 'utf8');
      const compressed = spawnSync('zstd', ['--stdout', '--quiet'], { input: source });
      if (compressed.status !== 0 || !compressed.stdout) {
        throw new Error(`Failed to compress test payload with zstd: ${compressed.stderr?.toString('utf8') || ''}`);
      }

      const response = await app.inject({
        method: 'POST',
        url: '/echo',
        headers: {
          'content-type': 'application/json',
          'content-encoding': 'zstd'
        },
        payload: compressed.stdout
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.body)).toEqual({
        body: {
          hello: 'zstd'
        }
      });
    } finally {
      await app.close();
    }
  });
});
