import type { FastifyInstance, FastifyRequest } from 'fastify';
import { spawn } from 'node:child_process';
import type { Transform } from 'node:stream';
import { createBrotliDecompress, createGunzip, createInflate } from 'node:zlib';
import { cacheRawRequestBody } from '../raw-trace';

const fallbackBodyLimit = 1_048_576;
const zstdDecodeTimeoutMs = 5000;
const maxDecodeErrorOutput = 1024;

type RouteOptionsCarrier = FastifyRequest & {
  routeOptions?: {
    bodyLimit?: number;
  };
};
type DecodeBodyResult =
  | { ok: true; value: Buffer }
  | { ok: false; error: Error & { statusCode: number; code: string } };

export function registerLenientJsonParser(fastify: FastifyInstance): void {
  if (fastify.hasContentTypeParser('application/json')) {
    fastify.removeContentTypeParser('application/json');
  }

  const defaultBodyLimit = resolveDefaultBodyLimit(fastify);

  fastify.addContentTypeParser('application/json', (request, payload, done) => {
    const limit = resolveRouteBodyLimit(request, defaultBodyLimit);
    const chunks: Buffer[] = [];
    let receivedLength = 0;
    let completed = false;

    const finish = (error?: Error | null, value?: unknown) => {
      if (completed) {
        return;
      }

      completed = true;
      payload.removeListener('data', onData);
      payload.removeListener('end', onEnd);
      payload.removeListener('error', onError);
      done(error || null, value);
    };

    const onData = (chunk: Buffer | string) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      receivedLength += buffer.length;

      if (receivedLength > limit) {
        const error = Object.assign(new Error('Request body is too large.'), {
          statusCode: 413,
          code: 'FST_ERR_CTP_BODY_TOO_LARGE'
        });
        payload.destroy();
        finish(error);
        return;
      }

      chunks.push(buffer);
    };

    const onEnd = async () => {
      const decodedBodyResult = await decodeBodyByContentEncoding(
        Buffer.concat(chunks),
        request.headers['content-encoding'],
        limit
      );
      if (!decodedBodyResult.ok) {
        finish(decodedBodyResult.error);
        return;
      }

      const rawBody = stripUtf8Bom(decodedBodyResult.value.toString('utf8'));
      if (!rawBody.trim()) {
        const error = Object.assign(new Error('Body cannot be empty when content-type is application/json.'), {
          statusCode: 400,
          code: 'FST_ERR_CTP_EMPTY_JSON_BODY'
        });
        finish(error);
        return;
      }

      cacheRawRequestBody(request, rawBody);

      try {
        finish(null, JSON.parse(rawBody) as unknown);
      } catch (parseError) {
        request.log.warn(
          {
            method: request.method,
            url: request.url,
            contentType: request.headers['content-type'],
            contentEncoding: request.headers['content-encoding'],
            declaredContentLength: request.headers['content-length'],
            decodedBodyBytes: decodedBodyResult.value.length,
            jsonError: parseError instanceof Error ? parseError.message : String(parseError)
          },
          'Invalid JSON body received.'
        );
        const parseBodyError = Object.assign(new Error('Body is not valid JSON.'), {
          statusCode: 400,
          code: 'FST_ERR_CTP_INVALID_JSON_BODY'
        });
        finish(parseBodyError);
      }
    };

    const onError = (error: Error & { statusCode?: number }) => {
      if (typeof error.statusCode !== 'number') {
        error.statusCode = 400;
      }
      finish(error);
    };

    payload.on('data', onData);
    payload.on('end', onEnd);
    payload.on('error', onError);
  });
}

function resolveDefaultBodyLimit(fastify: FastifyInstance): number {
  const configured = fastify.initialConfig.bodyLimit;
  if (typeof configured === 'number' && Number.isFinite(configured) && configured > 0) {
    return configured;
  }

  return fallbackBodyLimit;
}

function resolveRouteBodyLimit(request: FastifyRequest, defaultBodyLimit: number): number {
  const routeBodyLimit = (request as RouteOptionsCarrier).routeOptions?.bodyLimit;
  if (typeof routeBodyLimit === 'number' && Number.isFinite(routeBodyLimit) && routeBodyLimit > 0) {
    return routeBodyLimit;
  }

  return defaultBodyLimit;
}

function stripUtf8Bom(value: string): string {
  if (value.charCodeAt(0) === 0xfeff) {
    return value.slice(1);
  }

  return value;
}

function decodeBodyByContentEncoding(
  body: Buffer,
  contentEncodingHeader: string | string[] | undefined,
  limit: number
): Promise<DecodeBodyResult> {
  const encodings = parseContentEncodings(contentEncodingHeader);
  if (encodings.length === 0) {
    return Promise.resolve({ ok: true, value: body });
  }

  return decodeBodyByEncodings(body, encodings, limit);
}

async function decodeBodyByEncodings(
  body: Buffer,
  encodings: string[],
  limit: number
): Promise<DecodeBodyResult> {
  let decoded = body;

  for (let index = encodings.length - 1; index >= 0; index -= 1) {
    const encoding = encodings[index];

    if (encoding === 'identity') {
      continue;
    }

    try {
      if (encoding === 'gzip' || encoding === 'x-gzip') {
        const result = await decodeZlibWithLimit(decoded, limit, encoding, () => createGunzip());
        if (!result.ok) {
          return result;
        }
        decoded = result.value;
      } else if (encoding === 'deflate') {
        const result = await decodeZlibWithLimit(decoded, limit, encoding, () => createInflate());
        if (!result.ok) {
          return result;
        }
        decoded = result.value;
      } else if (encoding === 'br') {
        const result = await decodeZlibWithLimit(decoded, limit, encoding, () => createBrotliDecompress());
        if (!result.ok) {
          return result;
        }
        decoded = result.value;
      } else if (encoding === 'zstd' || encoding === 'x-zstd') {
        const zstdDecoded = await decodeZstdWithCli(decoded, limit);
        if (!zstdDecoded.ok) {
          return zstdDecoded;
        }
        decoded = zstdDecoded.value;
      } else {
        return {
          ok: false,
          error: Object.assign(new Error(`Unsupported content-encoding: ${encoding}`), {
            statusCode: 415,
            code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
          })
        };
      }
    } catch {
      return {
        ok: false,
        error: Object.assign(new Error(`Body could not be decompressed using content-encoding: ${encoding}`), {
          statusCode: 400,
          code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
        })
      };
    }

    if (decoded.length > limit) {
      return {
        ok: false,
        error: Object.assign(new Error('Request body is too large.'), {
          statusCode: 413,
          code: 'FST_ERR_CTP_BODY_TOO_LARGE'
        })
      };
    }
  }

  return { ok: true, value: decoded };
}

function decodeZlibWithLimit(
  body: Buffer,
  limit: number,
  encoding: string,
  createDecoder: () => Transform
): Promise<DecodeBodyResult> {
  return new Promise((resolve) => {
    const decoder = createDecoder();
    const outputChunks: Buffer[] = [];
    let outputLength = 0;
    let settled = false;

    const finalize = (result: DecodeBodyResult) => {
      if (settled) {
        return;
      }

      settled = true;
      decoder.removeAllListeners();
      resolve(result);
    };

    decoder.on('data', (chunk: Buffer | string) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      outputLength += buffer.length;
      if (outputLength > limit) {
        decoder.destroy();
        finalize({
          ok: false,
          error: Object.assign(new Error('Request body is too large.'), {
            statusCode: 413,
            code: 'FST_ERR_CTP_BODY_TOO_LARGE'
          })
        });
        return;
      }

      outputChunks.push(buffer);
    });

    decoder.on('error', () => {
      finalize({
        ok: false,
        error: Object.assign(new Error(`Body could not be decompressed using content-encoding: ${encoding}`), {
          statusCode: 400,
          code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
        })
      });
    });

    decoder.on('end', () => {
      finalize({
        ok: true,
        value: Buffer.concat(outputChunks, outputLength)
      });
    });

    try {
      decoder.end(body);
    } catch {
      finalize({
        ok: false,
        error: Object.assign(new Error(`Body could not be decompressed using content-encoding: ${encoding}`), {
          statusCode: 400,
          code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
        })
      });
    }
  });
}

function decodeZstdWithCli(
  body: Buffer,
  limit: number
): Promise<DecodeBodyResult> {
  return new Promise((resolve) => {
    const child = spawn('zstd', ['-d', '--stdout', '--quiet'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    const outputChunks: Buffer[] = [];
    let outputLength = 0;
    let stderrText = '';
    let settled = false;
    let timedOut = false;

    const finalize = (
      result: { ok: true; value: Buffer } | { ok: false; error: Error & { statusCode: number; code: string } }
    ) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      resolve(result);
    };

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
      finalize({
        ok: false,
        error: Object.assign(new Error('Body decompression timed out for content-encoding: zstd'), {
          statusCode: 400,
          code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
        })
      });
    }, zstdDecodeTimeoutMs);

    child.on('error', (error) => {
      const code = (error as NodeJS.ErrnoException).code;
      const statusCode = code === 'ENOENT' ? 415 : 400;
      finalize({
        ok: false,
        error: Object.assign(new Error('Unsupported content-encoding: zstd'), {
          statusCode,
          code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
        })
      });
    });

    child.stdout.on('data', (chunk: Buffer) => {
      outputLength += chunk.length;
      if (outputLength > limit) {
        child.kill('SIGKILL');
        finalize({
          ok: false,
          error: Object.assign(new Error('Request body is too large.'), {
            statusCode: 413,
            code: 'FST_ERR_CTP_BODY_TOO_LARGE'
          })
        });
        return;
      }

      outputChunks.push(chunk);
    });

    child.stderr.on('data', (chunk: Buffer) => {
      if (stderrText.length >= maxDecodeErrorOutput) {
        return;
      }
      stderrText += chunk.toString('utf8').slice(0, maxDecodeErrorOutput - stderrText.length);
    });

    child.on('close', (code) => {
      if (timedOut || settled) {
        return;
      }
      if (code === 0) {
        finalize({ ok: true, value: Buffer.concat(outputChunks) });
        return;
      }

      finalize({
        ok: false,
        error: Object.assign(
          new Error(
            stderrText
              ? `Body could not be decompressed using content-encoding: zstd (${stderrText.trim()})`
              : 'Body could not be decompressed using content-encoding: zstd'
          ),
          {
            statusCode: 400,
            code: 'FST_ERR_CTP_INVALID_CONTENT_ENCODING'
          }
        )
      });
    });

    child.stdin.on('error', () => {
      // Ignore write-after-close when child exits early on invalid payload.
    });
    child.stdin.end(body);
  });
}

function parseContentEncodings(value: string | string[] | undefined): string[] {
  const joined = Array.isArray(value) ? value.join(',') : value;
  if (!joined) {
    return [];
  }

  return joined
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}
