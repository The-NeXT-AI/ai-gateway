import { createHash } from 'node:crypto';
import { PassThrough, Readable } from 'node:stream';
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import type { GatewayConfig, GatewayRequestIdentity } from '../types';
import { isObject, readHeader } from '../utils';

type CachedPayload = string | Buffer;
type CachedHeaderValue = string | number | string[];

interface CachedIdempotencyResponse {
  statusCode: number;
  headers: Record<string, CachedHeaderValue>;
  payload: CachedPayload;
}

interface PendingIdempotencyEntry {
  state: 'pending';
  requestHash: string;
  expiresAt: number;
  promise: Promise<CachedIdempotencyResponse | undefined>;
  resolve: (response: CachedIdempotencyResponse | undefined) => void;
}

interface CompletedIdempotencyEntry {
  state: 'completed';
  requestHash: string;
  expiresAt: number;
  response: CachedIdempotencyResponse;
}

type IdempotencyEntry = PendingIdempotencyEntry | CompletedIdempotencyEntry;

interface IdempotencyRequestContext {
  storeKey: string;
  requestHash: string;
  servedFromCache: boolean;
}

const idempotencyStore = new Map<string, IdempotencyEntry>();
const requestContexts = new WeakMap<FastifyRequest, IdempotencyRequestContext>();

const routeSensitiveHeaders = [
  'content-type',
  'authorization',
  'x-api-key',
  'api-key',
  'x-goog-api-key',
  'x-mcp-key',
  'x-codex-access-token',
  'x-codex-refresh-token',
  'x-codex-account-id',
  'x-target-provider',
  'x-target-providers',
  'x-target-model',
  'x-auth-user-id',
  'x-auth-tenant-id',
  'x-auth-sub',
  'x-auth-organization-id',
  'x-auth-plan',
  'openai-organization',
  'openai-project',
  'anthropic-version',
  'anthropic-beta'
];

const hopByHopHeaders = new Set([
  'connection',
  'content-encoding',
  'content-length',
  'date',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade'
]);

export function createGatewayIdempotencyPreHandler(config: GatewayConfig) {
  return async (request: FastifyRequest, reply: FastifyReply) => {
    if (!isIdempotencyEligible(request, config)) {
      return;
    }

    const key = readIdempotencyKey(request, config);
    if (!key) {
      return;
    }

    const storeKey = buildIdempotencyStoreKey(request, key);
    const requestHash = hashIdempotencyRequest(request, config);
    return handleMemoryIdempotencyPrecheck(request, reply, config, storeKey, requestHash);
  };
}

async function handleMemoryIdempotencyPrecheck(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  storeKey: string,
  requestHash: string
) {
  const now = Date.now();
  pruneIdempotencyStore(config, now);
  const existing = idempotencyStore.get(storeKey);

  if (existing && existing.expiresAt > now) {
    if (existing.requestHash !== requestHash) {
      return sendIdempotencyConflict(reply);
    }

    if (existing.state === 'completed') {
      requestContexts.set(request, {
        storeKey,
        requestHash,
        servedFromCache: true
      });
      return sendCachedResponse(reply, existing.response);
    }

    const response = await existing.promise;
    if (response) {
      requestContexts.set(request, {
        storeKey,
        requestHash,
        servedFromCache: true
      });
      return sendCachedResponse(reply, response);
    }

    return reply
      .code(409)
      .header('x-gateway-idempotency-status', 'not-cacheable')
      .send({
        error: {
          message: 'Original request did not produce a cacheable idempotency response.',
          code: 'idempotency_response_not_cacheable'
        }
      });
  }

  if (existing) {
    idempotencyStore.delete(storeKey);
  }

  const pending = createPendingEntry(requestHash, now + config.idempotency.ttlMs);
  idempotencyStore.set(storeKey, pending);
  requestContexts.set(request, {
    storeKey,
    requestHash,
    servedFromCache: false
  });
  pruneIdempotencyStore(config, now);
}

export function registerGatewayIdempotencyHooks(
  fastify: FastifyInstance,
  config: GatewayConfig
): void {
  fastify.addHook('onSend', async (request, reply, payload) => {
    const context = requestContexts.get(request);
    if (!context || context.servedFromCache) {
      return payload;
    }

    const entry = idempotencyStore.get(context.storeKey);
    if (!entry || entry.state !== 'pending' || entry.requestHash !== context.requestHash) {
      return payload;
    }

    const cached = buildCachedResponse(reply, payload, config);
    if (cached) {
      completePendingIdempotencyEntry(context, entry, cached);
      reply.header('x-gateway-idempotency-status', 'stored');
      return payload;
    }

    const cacheableStream = buildCacheableStreamResponse(reply, payload, context, entry, config);
    if (cacheableStream) {
      reply.header('x-gateway-idempotency-status', 'stored');
      return cacheableStream;
    }

    failPendingIdempotencyEntry(context, entry);
    return payload;
  });
}

function completePendingIdempotencyEntry(
  context: IdempotencyRequestContext,
  entry: PendingIdempotencyEntry,
  cached: CachedIdempotencyResponse
): void {
  const current = idempotencyStore.get(context.storeKey);
  if (current !== entry || current.state !== 'pending' || current.requestHash !== context.requestHash) {
    return;
  }

  const completed: CompletedIdempotencyEntry = {
    state: 'completed',
    requestHash: entry.requestHash,
    expiresAt: entry.expiresAt,
    response: cached
  };
  idempotencyStore.set(context.storeKey, completed);
  entry.resolve(cached);
}

function failPendingIdempotencyEntry(
  context: IdempotencyRequestContext,
  entry: PendingIdempotencyEntry
): void {
  const current = idempotencyStore.get(context.storeKey);
  if (current !== entry || current.state !== 'pending' || current.requestHash !== context.requestHash) {
    return;
  }

  idempotencyStore.delete(context.storeKey);
  entry.resolve(undefined);
}

function buildCacheableStreamResponse(
  reply: FastifyReply,
  payload: unknown,
  context: IdempotencyRequestContext,
  entry: PendingIdempotencyEntry,
  config: GatewayConfig
): Readable | undefined {
  if (!isCacheableStatus(reply.statusCode, config)) {
    return undefined;
  }

  if (isEventStreamResponse(reply)) {
    return undefined;
  }

  if (!isReadablePayload(payload)) {
    return undefined;
  }

  const stream = new PassThrough();
  const chunks: Buffer[] = [];
  const responseSnapshot = {
    statusCode: reply.statusCode,
    headers: sanitizeCachedHeaders(reply.getHeaders())
  };
  let settled = false;
  let streamCacheable = true;

  const settle = (cachedPayload: Buffer | undefined): void => {
    if (settled) {
      return;
    }
    settled = true;
    if (!cachedPayload) {
      failPendingIdempotencyEntry(context, entry);
      return;
    }

    const completed: CompletedIdempotencyEntry = {
      state: 'completed',
      requestHash: entry.requestHash,
      expiresAt: entry.expiresAt,
      response: {
        ...responseSnapshot,
        payload: cachedPayload
      }
    };
    const current = idempotencyStore.get(context.storeKey);
    if (current !== entry || current.state !== 'pending' || current.requestHash !== context.requestHash) {
      return;
    }

    idempotencyStore.set(context.storeKey, completed);
    entry.resolve(completed.response);
  };

  payload.on('data', (chunk) => {
    const buffer = normalizeStreamChunk(chunk);
    if (buffer) {
      chunks.push(buffer);
      return;
    }

    streamCacheable = false;
  });
  payload.once('end', () => {
    settle(streamCacheable ? Buffer.concat(chunks) : undefined);
  });
  payload.once('error', (error) => {
    settle(undefined);
    stream.destroy(error instanceof Error ? error : new Error(String(error)));
  });
  payload.once('close', () => {
    if (!settled && !payload.readableEnded) {
      settle(undefined);
    }
  });

  payload.pipe(stream);
  return stream;
}

export function resetGatewayIdempotencyForTests(): void {
  idempotencyStore.clear();
}

function isIdempotencyEligible(request: FastifyRequest, config: GatewayConfig): boolean {
  if (!config.idempotency?.enabled) {
    return false;
  }

  if (request.method.toUpperCase() !== 'POST') {
    return false;
  }

  const path = request.url.split('?')[0] || '';
  return path.startsWith('/v1/') || path.startsWith('/v1beta/');
}

function readIdempotencyKey(request: FastifyRequest, config: GatewayConfig): string | undefined {
  const headerName = config.idempotency.headerName.trim().toLowerCase();
  if (!headerName) {
    return undefined;
  }

  const value = readHeader(request.headers[headerName]);
  const normalized = value?.trim();
  return normalized || undefined;
}

function createPendingEntry(requestHash: string, expiresAt: number): PendingIdempotencyEntry {
  let resolve!: (response: CachedIdempotencyResponse | undefined) => void;
  const promise = new Promise<CachedIdempotencyResponse | undefined>((innerResolve) => {
    resolve = innerResolve;
  });

  return {
    state: 'pending',
    requestHash,
    expiresAt,
    promise,
    resolve
  };
}

function sendCachedResponse(reply: FastifyReply, response: CachedIdempotencyResponse) {
  for (const [name, value] of Object.entries(response.headers)) {
    reply.header(name, value);
  }

  return reply
    .code(response.statusCode)
    .header('x-gateway-idempotency-status', 'replayed')
    .send(Buffer.isBuffer(response.payload) ? Buffer.from(response.payload) : response.payload);
}

function sendIdempotencyConflict(reply: FastifyReply) {
  return reply
    .code(409)
    .header('x-gateway-idempotency-status', 'conflict')
    .send({
      error: {
        message: 'Idempotency key was reused with a different request.',
        code: 'idempotency_key_conflict'
      }
    });
}

function buildCachedResponse(
  reply: FastifyReply,
  payload: unknown,
  config: GatewayConfig
): CachedIdempotencyResponse | undefined {
  if (!isCacheableStatus(reply.statusCode, config)) {
    return undefined;
  }

  if (isEventStreamResponse(reply)) {
    return undefined;
  }

  if (typeof payload !== 'string' && !Buffer.isBuffer(payload)) {
    return undefined;
  }

  return {
    statusCode: reply.statusCode,
    headers: sanitizeCachedHeaders(reply.getHeaders()),
    payload: Buffer.isBuffer(payload) ? Buffer.from(payload) : payload
  };
}

function isReadablePayload(payload: unknown): payload is Readable {
  if (payload instanceof Readable) {
    return true;
  }

  if (!isObject(payload)) {
    return false;
  }

  const candidate = payload as {
    pipe?: unknown;
    on?: unknown;
    once?: unknown;
  };
  return (
    typeof candidate.pipe === 'function' &&
    typeof candidate.on === 'function' &&
    typeof candidate.once === 'function'
  );
}

function normalizeStreamChunk(chunk: unknown): Buffer | undefined {
  if (Buffer.isBuffer(chunk)) {
    return Buffer.from(chunk);
  }

  if (typeof chunk === 'string') {
    return Buffer.from(chunk, 'utf8');
  }

  if (chunk instanceof Uint8Array) {
    return Buffer.from(chunk);
  }

  return undefined;
}

function isCacheableStatus(statusCode: number, config: GatewayConfig): boolean {
  if (statusCode >= 200 && statusCode < 300) {
    return true;
  }

  return config.idempotency.cacheErrorResponses && statusCode >= 400 && statusCode < 500;
}

function isEventStreamResponse(reply: FastifyReply): boolean {
  const contentType = String(reply.getHeader('content-type') || '').toLowerCase();
  return contentType.includes('text/event-stream');
}

function sanitizeCachedHeaders(headers: Record<string, unknown>): Record<string, CachedHeaderValue> {
  const cached: Record<string, CachedHeaderValue> = {};

  for (const [rawName, rawValue] of Object.entries(headers)) {
    const name = rawName.toLowerCase();
    if (hopByHopHeaders.has(name) || name === 'x-gateway-idempotency-status') {
      continue;
    }

    if (typeof rawValue === 'string' || typeof rawValue === 'number') {
      cached[name] = rawValue;
    } else if (
      Array.isArray(rawValue) &&
      rawValue.every((item): item is string => typeof item === 'string')
    ) {
      cached[name] = rawValue;
    }
  }

  return cached;
}

function pruneIdempotencyStore(config: GatewayConfig, now: number): void {
  for (const [key, entry] of idempotencyStore) {
    if (entry.expiresAt <= now) {
      idempotencyStore.delete(key);
      if (entry.state === 'pending') {
        entry.resolve(undefined);
      }
    }
  }

  while (idempotencyStore.size > config.idempotency.maxEntries) {
    const oldestCompleted = Array.from(idempotencyStore.entries()).find(
      ([, entry]) => entry.state === 'completed'
    );
    const [key, entry] = oldestCompleted || idempotencyStore.entries().next().value || [];
    if (!key || !entry) {
      return;
    }

    idempotencyStore.delete(key);
    if (entry.state === 'pending') {
      entry.resolve(undefined);
    }
  }
}

function hashIdempotencyRequest(request: FastifyRequest, config: GatewayConfig): string {
  const hash = createHash('sha256');
  hash.update(request.method.toUpperCase());
  hash.update('\n');
  hash.update(request.url);
  hash.update('\n');
  hash.update(config.idempotency.headerName.trim().toLowerCase());
  hash.update('\n');
  hash.update(stableStringify(normalizeGatewayIdentityForFingerprint(request.gatewayIdentity)));
  hash.update('\n');
  hash.update(stableStringify(selectFingerprintHeaders(request)));
  hash.update('\n');
  updateIdempotencyBodyHash(hash, request.body);
  return hash.digest('hex');
}

function updateIdempotencyBodyHash(hash: ReturnType<typeof createHash>, body: unknown): void {
  if (Buffer.isBuffer(body)) {
    hash.update('Buffer\n');
    hash.update(body);
    return;
  }

  if (body instanceof ArrayBuffer) {
    hash.update('ArrayBuffer\n');
    hash.update(Buffer.from(body));
    return;
  }

  if (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView(body)) {
    hash.update(`${body.constructor.name || 'ArrayBufferView'}\n`);
    hash.update(Buffer.from(body.buffer, body.byteOffset, body.byteLength));
    return;
  }

  hash.update(stableStringify(body));
}

function buildIdempotencyStoreKey(request: FastifyRequest, idempotencyKey: string): string {
  const identity = request.gatewayIdentity;
  if (identity?.billingSubjectKey) {
    return `identity:${identity.source}:${hashStableValue(identity.billingSubjectKey)}:${idempotencyKey}`;
  }

  const fingerprintHeaders = selectFingerprintHeaders(request);
  if (Object.keys(fingerprintHeaders).length > 0) {
    return `headers:${hashStableValue(fingerprintHeaders)}:${idempotencyKey}`;
  }

  return `anonymous:${idempotencyKey}`;
}

function selectFingerprintHeaders(request: FastifyRequest): Record<string, string | string[]> {
  const selected: Record<string, string | string[]> = {};
  for (const headerName of routeSensitiveHeaders) {
    const value = request.headers[headerName];
    if (typeof value === 'string') {
      selected[headerName] = value;
    } else if (Array.isArray(value)) {
      selected[headerName] = value.map(String);
    }
  }

  return selected;
}

function normalizeGatewayIdentityForFingerprint(
  identity: GatewayRequestIdentity | undefined
): Record<string, string> | undefined {
  if (!identity) {
    return undefined;
  }

  const normalized: Record<string, string> = {
    source: identity.source,
    billingSubjectKey: identity.billingSubjectKey
  };

  for (const key of ['userId', 'tenantId', 'subject', 'organizationId', 'plan', 'apiKeyId'] as const) {
    const value = identity[key];
    if (typeof value === 'string') {
      normalized[key] = value;
    }
  }

  return normalized;
}

function hashStableValue(value: unknown): string {
  return createHash('sha256').update(stableStringify(value)).digest('hex');
}

function stableStringify(value: unknown): string {
  return JSON.stringify(sortStable(value)) ?? 'undefined';
}

function sortStable(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortStable);
  }

  if (!isObject(value)) {
    return value;
  }

  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = sortStable(value[key]);
  }
  return sorted;
}
