import { createHash } from 'node:crypto';
import net from 'node:net';
import tls from 'node:tls';
import type { FastifyRequest } from 'fastify';
import { calculateUsageBilling } from '../billing';
import type {
  BillingRate,
  GatewayConfig,
  GatewayPrecheckRuleBaseConfig,
  GatewayPrecheckRedisStorageConfig,
  GatewayPrecheckStorageConfig,
  GatewayPrecheckScope,
  GatewayPrecheckSubject,
  GatewayApiKeyRestrictions,
  GatewayRateLimitDimensionConfig,
  GatewayRateLimitMetric,
  GatewayRateLimitPrecheckConfig,
  GatewayRequestIdentity,
  Provider,
  ProviderConfig,
  StandardRequest,
  StandardRequestInputContent,
  StandardRequestInputMessage,
  StandardUsage
} from '../types';
import { findDefaultProviderConfig, isObject, readHeader } from '../utils';

type PrecheckKind = 'rate_limit' | 'quota' | 'budget';

export interface GatewayPrecheckInput {
  request: FastifyRequest;
  config: GatewayConfig;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  model?: string;
  standardRequest?: StandardRequest;
  requestBody?: unknown;
  imageCount?: number;
  videoSeconds?: number;
  videoSize?: string;
}

export interface GatewayPrecheckEstimate {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  imageCount: number;
  videoSeconds: number;
  estimatedCostUsd: number;
}

export interface GatewayPrecheckFailure {
  ok: false;
  kind: PrecheckKind;
  statusCode: number;
  code: string;
  message: string;
  details: {
    subject: string;
    scope: string;
    window_ms: number;
    limit: number;
    used: number;
    requested: number;
    metric: string;
    limit_name?: string;
    estimated?: GatewayPrecheckEstimate;
  };
}

export type GatewayPrecheckResult =
  | { ok: true; estimate?: GatewayPrecheckEstimate }
  | GatewayPrecheckFailure;

interface WindowCounter {
  windowStart: number;
  value: number;
}

interface PendingCheck {
  kind: PrecheckKind;
  key: string;
  subjectKey: string;
  scopeKey: string;
  metric: GatewayRateLimitMetric | 'cost_usd';
  limitName?: string;
  windowMs: number;
  limit: number;
  requested: number;
  windowStart: number;
}

interface ApiKeyRestrictionBudgetLimit extends GatewayPrecheckRuleBaseConfig {
  name: string;
  maxCostUsd: number;
}

type RedisReservationResult =
  | { ok: true }
  | { ok: false; failedIndex: number; used: number };

type RedisReservationExecutor = (
  storage: GatewayPrecheckRedisStorageConfig,
  checks: PendingCheck[]
) => Promise<RedisReservationResult>;

const counters = new Map<string, WindowCounter>();
const redisClients = new Map<string, RedisPrecheckClient>();
let redisReservationExecutorForTests: RedisReservationExecutor | undefined;

export async function evaluateGatewayPrecheck(input: GatewayPrecheckInput): Promise<GatewayPrecheckResult> {
  const precheck = input.config.precheck;
  const apiKeyRestrictionRateLimits = resolveApiKeyRestrictionRateLimits(input.request);
  const apiKeyRestrictionBudgetLimits = resolveApiKeyRestrictionBudgetLimits(input.request);
  if (
    !precheck?.enabled &&
    apiKeyRestrictionRateLimits.length === 0 &&
    apiKeyRestrictionBudgetLimits.length === 0
  ) {
    return { ok: true };
  }

  const staticPrecheckEnabled = precheck.enabled === true;
  const rateLimitRules = staticPrecheckEnabled && precheck.rateLimit.enabled
    ? [...resolveRateLimitRules(precheck.rateLimit), ...apiKeyRestrictionRateLimits]
    : apiKeyRestrictionRateLimits;
  const hasQuota = staticPrecheckEnabled && precheck.quota.enabled && precheck.quota.maxTokens > 0;
  const hasBudget = staticPrecheckEnabled && precheck.budget.enabled && precheck.budget.maxCostUsd > 0;
  const hasApiKeyBudget = apiKeyRestrictionBudgetLimits.length > 0;

  if (rateLimitRules.length === 0 && !hasQuota && !hasBudget && !hasApiKeyBudget) {
    return { ok: true };
  }

  const needsEstimate =
    hasQuota ||
    hasBudget ||
    hasApiKeyBudget ||
    rateLimitRules.some((limit) => limit.metric === 'tokens' || limit.metric === 'images');
  const estimate =
    needsEstimate
      ? estimateGatewayRequestUsage(input)
      : undefined;
  const now = Date.now();
  const checks: PendingCheck[] = [];

  for (const limit of rateLimitRules) {
    checks.push(
      buildRateLimitPendingCheck(
        'rate_limit',
        input,
        limit,
        resolveRateLimitRequestedValue(limit.metric, estimate),
        now
      )
    );
  }

  if (hasQuota && estimate) {
    checks.push(
      buildPendingCheck(
        'quota',
        input,
        precheck.quota,
        'tokens',
        undefined,
        precheck.quota.maxTokens,
        estimate.totalTokens,
        now
      )
    );
  }

  if (hasBudget && estimate) {
    checks.push(
      buildPendingCheck(
        'budget',
        input,
        precheck.budget,
        'cost_usd',
        undefined,
        precheck.budget.maxCostUsd,
        estimate.estimatedCostUsd,
        now
      )
    );
  }

  if (hasApiKeyBudget && estimate) {
    for (const limit of apiKeyRestrictionBudgetLimits) {
      checks.push(
        buildPendingCheck(
          'budget',
          input,
          limit,
          'cost_usd',
          limit.name,
          limit.maxCostUsd,
          estimate.estimatedCostUsd,
          now
        )
      );
    }
  }

  return reserveChecks(input.config.precheck.storage, checks, estimate);
}

async function reserveChecks(
  storage: GatewayPrecheckStorageConfig,
  checks: PendingCheck[],
  estimate: GatewayPrecheckEstimate | undefined
): Promise<GatewayPrecheckResult> {
  if (storage.type === 'redis') {
    return reserveRedisChecks(storage, checks, estimate);
  }

  return reserveMemoryChecks(checks, estimate);
}

function reserveMemoryChecks(
  checks: PendingCheck[],
  estimate: GatewayPrecheckEstimate | undefined
): GatewayPrecheckResult {
  for (const check of checks) {
    const counter = readWindowCounter(check.key, check.windowStart);
    const used = counter.value;
    if (used + check.requested > check.limit) {
      return buildPrecheckFailure(check, used, estimate);
    }
  }

  for (const check of checks) {
    const counter = readWindowCounter(check.key, check.windowStart);
    counter.value += check.requested;
  }

  return { ok: true, estimate };
}

async function reserveRedisChecks(
  storage: GatewayPrecheckRedisStorageConfig,
  checks: PendingCheck[],
  estimate: GatewayPrecheckEstimate | undefined
): Promise<GatewayPrecheckResult> {
  if (checks.length === 0) {
    return { ok: true, estimate };
  }

  try {
    const reservation = redisReservationExecutorForTests
      ? await redisReservationExecutorForTests(storage, checks)
      : await getRedisPrecheckClient(storage).reserve(checks);
    if (reservation.ok) {
      return { ok: true, estimate };
    }

    const failedCheck = checks[Math.max(0, reservation.failedIndex - 1)] || checks[0];
    return buildPrecheckFailure(failedCheck, reservation.used, estimate);
  } catch (error) {
    return buildPrecheckStoreFailure(checks[0], estimate, error);
  }
}

function getRedisPrecheckClient(storage: GatewayPrecheckRedisStorageConfig): RedisPrecheckClient {
  const cacheKey = [
    storage.url,
    storage.keyPrefix,
    storage.connectTimeoutMs,
    storage.commandTimeoutMs
  ].join('|');
  const existing = redisClients.get(cacheKey);
  if (existing) {
    return existing;
  }

  const created = new RedisPrecheckClient(storage);
  redisClients.set(cacheKey, created);
  return created;
}

export async function closeGatewayPrecheckStore(): Promise<void> {
  const clients = [...redisClients.values()];
  redisClients.clear();
  await Promise.allSettled(clients.map((client) => client.close()));
}

export function resetGatewayPrecheckStateForTests(): void {
  counters.clear();
  redisReservationExecutorForTests = undefined;
  for (const client of redisClients.values()) {
    void client.close();
  }
  redisClients.clear();
}

export function setGatewayPrecheckRedisReservationExecutorForTests(
  executor: RedisReservationExecutor | undefined
): void {
  redisReservationExecutorForTests = executor;
}

function resolveRateLimitRules(
  rateLimit: GatewayRateLimitPrecheckConfig
): GatewayRateLimitDimensionConfig[] {
  const configuredLimits = Array.isArray(rateLimit.limits)
    ? rateLimit.limits.filter((limit) => limit.enabled && limit.max > 0)
    : [];
  if (configuredLimits.length > 0 || rateLimit.maxRequests <= 0) {
    return configuredLimits;
  }

  return [
    {
      enabled: true,
      name: 'requests',
      metric: 'requests',
      windowMs: rateLimit.windowMs,
      max: rateLimit.maxRequests,
      subject: rateLimit.subject,
      scope: rateLimit.scope,
      headerName: rateLimit.headerName
    }
  ];
}

function resolveApiKeyRestrictionRateLimits(
  request: FastifyRequest
): GatewayRateLimitDimensionConfig[] {
  const restrictions = readRequestApiKeyRestrictions(request);
  if (!restrictions) {
    return [];
  }

  const limits: GatewayRateLimitDimensionConfig[] = [];
  const maxRequests = normalizePositiveInteger(
    restrictions.rateLimit ?? restrictions.requestsPerMinute ?? restrictions.rpm
  );
  if (maxRequests) {
    limits.push({
      enabled: true,
      name: 'api_key_restriction',
      metric: 'requests',
      windowMs: resolveRestrictionWindowMs(restrictions.rateLimitWindowSeconds),
      max: maxRequests,
      subject: 'api_key',
      scope: 'global',
    });
  }

  const maxTokens = normalizePositiveInteger(
    restrictions.tokensPerMinute ?? restrictions.tpm
  );
  if (maxTokens) {
    limits.push({
      enabled: true,
      name: 'api_key_tpm',
      metric: 'tokens',
      windowMs: resolveRestrictionWindowMs(restrictions.tokenLimitWindowSeconds),
      max: maxTokens,
      subject: 'api_key',
      scope: 'global',
    });
  }

  return limits;
}

function resolveApiKeyRestrictionBudgetLimits(
  request: FastifyRequest
): ApiKeyRestrictionBudgetLimit[] {
  const restrictions = readRequestApiKeyRestrictions(request);
  if (!restrictions) {
    return [];
  }

  const maxCostUsd = normalizePositiveNumber(
    restrictions.costLimitUsd ??
      restrictions.costLimit ??
      restrictions.maxCostUsd ??
      restrictions.costPerMinuteUsd
  );
  if (!maxCostUsd) {
    return [];
  }

  return [
    {
      enabled: true,
      name: 'api_key_cost',
      windowMs: resolveRestrictionWindowMs(restrictions.costLimitWindowSeconds),
      maxCostUsd,
      subject: 'api_key',
      scope: 'global',
    },
  ];
}

function readRequestApiKeyRestrictions(
  request: FastifyRequest
): GatewayApiKeyRestrictions | undefined {
  return (
    request as FastifyRequest & {
      gatewayApiKeyRestrictions?: GatewayApiKeyRestrictions;
    }
  ).gatewayApiKeyRestrictions;
}

function resolveRestrictionWindowMs(value: unknown): number {
  const windowSeconds = normalizePositiveInteger(value) || 60;
  return Math.min(Math.max(windowSeconds, 1), 3600) * 1000;
}

function normalizePositiveInteger(value: unknown): number | undefined {
  const numeric =
    typeof value === 'number'
      ? value
      : typeof value === 'string'
        ? Number(value)
        : NaN;
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return undefined;
  }

  return Math.floor(numeric);
}

function normalizePositiveNumber(value: unknown): number | undefined {
  const numeric =
    typeof value === 'number'
      ? value
      : typeof value === 'string'
        ? Number(value)
        : NaN;
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return undefined;
  }

  return numeric;
}

const redisReserveScript = `
local count = tonumber(ARGV[1])
for i = 1, count do
  local offset = 1 + ((i - 1) * 3)
  local requested = tonumber(ARGV[offset + 1])
  local limit = tonumber(ARGV[offset + 2])
  local used = tonumber(redis.call("GET", KEYS[i]) or "0")
  if used + requested > limit then
    return {0, i, used}
  end
end
for i = 1, count do
  local offset = 1 + ((i - 1) * 3)
  local requested = tonumber(ARGV[offset + 1])
  local ttl = tonumber(ARGV[offset + 3])
  redis.call("INCRBYFLOAT", KEYS[i], requested)
  redis.call("PEXPIRE", KEYS[i], ttl)
end
return {1, 0, 0}
`.trim();

class RedisPrecheckClient {
  private socket?: net.Socket | tls.TLSSocket;
  private connecting?: Promise<void>;
  private buffer = Buffer.alloc(0);
  private pending: Array<{
    resolve: (value: RedisReply) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }> = [];

  constructor(private readonly storage: GatewayPrecheckRedisStorageConfig) {}

  async reserve(checks: PendingCheck[]): Promise<RedisReservationResult> {
    const now = Date.now();
    const keys = checks.map((check) => this.buildRedisKey(check));
    const args = checks.flatMap((check) => [
      formatRedisNumber(check.requested),
      formatRedisNumber(check.limit),
      String(Math.max(check.windowStart + check.windowMs - now, 1) + 1000)
    ]);
    const reply = await this.command([
      'EVAL',
      redisReserveScript,
      String(keys.length),
      ...keys,
      String(keys.length),
      ...args
    ]);
    const parsed = parseRedisReservationReply(reply);
    if (!parsed) {
      throw new Error('Redis precheck reservation returned an invalid response.');
    }

    return parsed;
  }

  async close(): Promise<void> {
    const socket = this.socket;
    this.socket = undefined;
    this.connecting = undefined;
    if (!socket) {
      return;
    }

    socket.destroy();
  }

  private buildRedisKey(check: PendingCheck): string {
    const prefix = this.storage.keyPrefix.replace(/:+$/, '') || 'next-ai:gateway:precheck';
    const hash = createHash('sha256').update(check.key).digest('hex');
    return `${prefix}:${hash}:${check.windowStart}`;
  }

  private async command(args: string[]): Promise<RedisReply> {
    await this.ensureConnected();
    return this.rawCommand(args);
  }

  private async ensureConnected(): Promise<void> {
    if (this.socket && !this.socket.destroyed && !this.connecting) {
      return;
    }

    if (!this.connecting) {
      this.connecting = this.connect();
    }

    const connecting = this.connecting;
    try {
      await connecting;
    } finally {
      if (this.connecting === connecting) {
        this.connecting = undefined;
      }
    }
  }

  private async connect(): Promise<void> {
    const parsed = parseRedisUrl(this.storage.url);
    const socket = parsed.tls
      ? tls.connect({
          host: parsed.host,
          port: parsed.port,
          servername: parsed.host
        })
      : net.createConnection({
          host: parsed.host,
          port: parsed.port
        });
    this.socket = socket;
    socket.setNoDelay(true);
    socket.on('data', (chunk) => this.handleData(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    socket.on('error', (error) => this.handleSocketError(error));
    socket.on('close', () => this.handleSocketClose());

    await waitForSocketConnect(socket, this.storage.connectTimeoutMs, parsed.tls);

    if (parsed.password) {
      await this.rawCommand(
        parsed.username
          ? ['AUTH', parsed.username, parsed.password]
          : ['AUTH', parsed.password]
      );
    }

    if (parsed.db !== undefined && parsed.db > 0) {
      await this.rawCommand(['SELECT', String(parsed.db)]);
    }
  }

  private rawCommand(args: string[]): Promise<RedisReply> {
    const socket = this.socket;
    if (!socket || socket.destroyed) {
      return Promise.reject(new Error('Redis precheck socket is not connected.'));
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('Redis precheck command timed out.'));
        socket.destroy();
      }, this.storage.commandTimeoutMs);
      this.pending.push({ resolve, reject, timer });
      socket.write(serializeRedisCommand(args), (error) => {
        if (error) {
          clearTimeout(timer);
          this.removePending(resolve);
          reject(error);
        }
      });
    });
  }

  private removePending(resolve: (value: RedisReply) => void): void {
    const index = this.pending.findIndex((item) => item.resolve === resolve);
    if (index >= 0) {
      this.pending.splice(index, 1);
    }
  }

  private handleData(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.pending.length > 0) {
      const parsed = parseRedisReply(this.buffer, 0);
      if (!parsed) {
        return;
      }

      this.buffer = this.buffer.subarray(parsed.offset);
      const pending = this.pending.shift();
      if (!pending) {
        return;
      }

      clearTimeout(pending.timer);
      if (parsed.value instanceof Error) {
        pending.reject(parsed.value);
      } else {
        pending.resolve(parsed.value);
      }
    }
  }

  private handleSocketError(error: Error): void {
    this.rejectAll(error);
  }

  private handleSocketClose(): void {
    this.socket = undefined;
    this.connecting = undefined;
    this.rejectAll(new Error('Redis precheck socket closed.'));
  }

  private rejectAll(error: Error): void {
    const pending = this.pending.splice(0);
    for (const item of pending) {
      clearTimeout(item.timer);
      item.reject(error);
    }
  }
}

type RedisReply = string | number | null | RedisReply[];

interface ParsedRedisUrl {
  tls: boolean;
  host: string;
  port: number;
  username?: string;
  password?: string;
  db?: number;
}

function parseRedisUrl(value: string): ParsedRedisUrl {
  const parsed = new URL(value);
  if (parsed.protocol !== 'redis:' && parsed.protocol !== 'rediss:') {
    throw new Error('precheck.storage redis url must use redis:// or rediss://.');
  }

  const dbRaw = parsed.pathname.replace(/^\//, '').trim();
  const db = dbRaw ? Number(dbRaw) : undefined;
  if (db !== undefined && (!Number.isInteger(db) || db < 0)) {
    throw new Error('precheck.storage redis url has an invalid database index.');
  }

  return {
    tls: parsed.protocol === 'rediss:',
    host: parsed.hostname || '127.0.0.1',
    port: parsed.port ? Number(parsed.port) : 6379,
    username: parsed.username ? decodeURIComponent(parsed.username) : undefined,
    password: parsed.password ? decodeURIComponent(parsed.password) : undefined,
    db
  };
}

function waitForSocketConnect(
  socket: net.Socket | tls.TLSSocket,
  timeoutMs: number,
  secure: boolean
): Promise<void> {
  return new Promise((resolve, reject) => {
    const connectEvent = secure ? 'secureConnect' : 'connect';
    const timer = setTimeout(() => {
      cleanup();
      socket.destroy();
      reject(new Error('Redis precheck connection timed out.'));
    }, timeoutMs);
    const onConnect = () => {
      cleanup();
      resolve();
    };
    const onError = (error: Error) => {
      cleanup();
      reject(error);
    };
    const cleanup = () => {
      clearTimeout(timer);
      socket.off(connectEvent, onConnect);
      socket.off('error', onError);
    };

    socket.once(connectEvent, onConnect);
    socket.once('error', onError);
  });
}

function serializeRedisCommand(args: string[]): Buffer {
  const chunks: string[] = [`*${args.length}\r\n`];
  for (const arg of args) {
    const value = Buffer.from(arg);
    chunks.push(`$${value.length}\r\n`, arg, '\r\n');
  }

  return Buffer.from(chunks.join(''));
}

function parseRedisReply(
  buffer: Buffer,
  offset: number
): { value: RedisReply | Error; offset: number } | undefined {
  if (offset >= buffer.length) {
    return undefined;
  }

  const type = String.fromCharCode(buffer[offset]);
  if (type === '+' || type === '-' || type === ':') {
    const lineEnd = buffer.indexOf('\r\n', offset + 1);
    if (lineEnd < 0) {
      return undefined;
    }

    const line = buffer.toString('utf8', offset + 1, lineEnd);
    if (type === '+') {
      return { value: line, offset: lineEnd + 2 };
    }
    if (type === '-') {
      return { value: new Error(line), offset: lineEnd + 2 };
    }

    return { value: Number(line), offset: lineEnd + 2 };
  }

  if (type === '$') {
    const lineEnd = buffer.indexOf('\r\n', offset + 1);
    if (lineEnd < 0) {
      return undefined;
    }

    const length = Number(buffer.toString('utf8', offset + 1, lineEnd));
    if (length < 0) {
      return { value: null, offset: lineEnd + 2 };
    }

    const start = lineEnd + 2;
    const end = start + length;
    if (buffer.length < end + 2) {
      return undefined;
    }

    return {
      value: buffer.toString('utf8', start, end),
      offset: end + 2
    };
  }

  if (type === '*') {
    const lineEnd = buffer.indexOf('\r\n', offset + 1);
    if (lineEnd < 0) {
      return undefined;
    }

    const count = Number(buffer.toString('utf8', offset + 1, lineEnd));
    if (count < 0) {
      return { value: null, offset: lineEnd + 2 };
    }

    const values: RedisReply[] = [];
    let cursor = lineEnd + 2;
    for (let index = 0; index < count; index += 1) {
      const item = parseRedisReply(buffer, cursor);
      if (!item) {
        return undefined;
      }

      if (item.value instanceof Error) {
        return item;
      }

      values.push(item.value);
      cursor = item.offset;
    }

    return { value: values, offset: cursor };
  }

  return { value: new Error(`Unsupported Redis reply type: ${type}`), offset: buffer.length };
}

function parseRedisReservationReply(reply: RedisReply): RedisReservationResult | undefined {
  if (!Array.isArray(reply) || reply.length < 3) {
    return undefined;
  }

  const allowed = Number(reply[0]);
  if (allowed === 1) {
    return { ok: true };
  }

  return {
    ok: false,
    failedIndex: Math.max(Number(reply[1]) || 1, 1),
    used: Number(reply[2]) || 0
  };
}

function formatRedisNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(12)));
}

function buildRateLimitPendingCheck(
  kind: PrecheckKind,
  input: GatewayPrecheckInput,
  rule: GatewayRateLimitDimensionConfig,
  requested: number,
  now: number
): PendingCheck {
  return buildPendingCheck(
    kind,
    input,
    rule,
    rule.metric,
    rule.name,
    rule.max,
    requested,
    now
  );
}

function buildPendingCheck(
  kind: PrecheckKind,
  input: GatewayPrecheckInput,
  rule: GatewayPrecheckRuleBaseConfig,
  metric: GatewayRateLimitMetric | 'cost_usd',
  limitName: string | undefined,
  limit: number,
  requested: number,
  now: number
): PendingCheck {
  const subjectKey = resolveSubjectKey(input.request, rule.subject, rule.headerName);
  const scopeKey = resolveScopeKey(rule.scope, input.targetProvider, input.model);
  const key = [
    kind,
    metric,
    limitName || '',
    rule.windowMs,
    rule.subject,
    subjectKey,
    rule.scope,
    scopeKey
  ].join('|');

  return {
    kind,
    key,
    subjectKey,
    scopeKey,
    metric,
    limitName,
    windowMs: rule.windowMs,
    limit,
    requested,
    windowStart: calculateWindowStart(rule.windowMs, now)
  };
}

function readWindowCounter(key: string, windowStart: number): WindowCounter {
  const existing = counters.get(key);
  if (existing && existing.windowStart === windowStart) {
    return existing;
  }

  const fresh: WindowCounter = {
    windowStart,
    value: 0
  };
  counters.set(key, fresh);
  return fresh;
}

function calculateWindowStart(windowMs: number, now: number): number {
  return Math.floor(now / windowMs) * windowMs;
}

function buildPrecheckFailure(
  check: PendingCheck,
  used: number,
  estimate: GatewayPrecheckEstimate | undefined
): GatewayPrecheckFailure {
  const code =
    check.kind === 'rate_limit'
      ? 'rate_limit_exceeded'
      : check.kind === 'quota'
        ? 'quota_exceeded'
        : 'budget_exceeded';
  const statusCode = check.kind === 'budget' ? 402 : 429;
  const limitLabel =
    check.kind === 'rate_limit'
      ? `${check.limitName || check.metric} rate limit`
      : check.kind === 'quota'
        ? 'token quota'
        : 'budget';

  return {
    ok: false,
    kind: check.kind,
    statusCode,
    code,
    message: `Gateway ${limitLabel} precheck failed.`,
    details: {
      subject: check.subjectKey,
      scope: check.scopeKey,
      window_ms: check.windowMs,
      limit: check.limit,
      used,
      requested: check.requested,
      metric: check.metric,
      limit_name: check.limitName,
      estimated: estimate
    }
  };
}

function buildPrecheckStoreFailure(
  check: PendingCheck,
  estimate: GatewayPrecheckEstimate | undefined,
  error: unknown
): GatewayPrecheckFailure {
  void error;
  return {
    ok: false,
    kind: check.kind,
    statusCode: 503,
    code: 'precheck_store_unavailable',
    message: 'Gateway precheck store is unavailable.',
    details: {
      subject: check.subjectKey,
      scope: check.scopeKey,
      window_ms: check.windowMs,
      limit: check.limit,
      used: 0,
      requested: check.requested,
      metric: check.metric,
      limit_name: check.limitName,
      estimated: estimate
    }
  };
}

function resolveRateLimitRequestedValue(
  metric: GatewayRateLimitMetric,
  estimate: GatewayPrecheckEstimate | undefined
): number {
  if (metric === 'requests') {
    return 1;
  }

  if (metric === 'tokens') {
    return estimate?.totalTokens || 0;
  }

  return estimate?.imageCount || 0;
}

function estimateGatewayRequestUsage(input: GatewayPrecheckInput): GatewayPrecheckEstimate {
  const charsPerToken = Math.max(input.config.precheck.estimation.charsPerToken, 1);
  const inputCharacters = input.standardRequest
    ? countStandardRequestInputCharacters(input.standardRequest)
    : countUnknownCharacters(input.requestBody);
  const inputTokens = Math.ceil(inputCharacters / charsPerToken);
  const outputTokens = resolveMaxOutputTokens(input);
  const totalTokens = inputTokens + outputTokens;
  const usage: StandardUsage = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    video_seconds: input.videoSeconds,
    video_size: input.videoSize
  };
  const billing = calculateUsageBilling(
    input.targetProvider,
    usage,
    input.config.billing,
    resolveProviderBillingRate(input.config, input.targetProvider, input.model, input.targetProviderConfig)
  );

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    imageCount: input.imageCount ?? countImageInputs(input.requestBody),
    videoSeconds: input.videoSeconds ?? 0,
    estimatedCostUsd: billing.cost.total
  };
}

function countStandardRequestInputCharacters(request: StandardRequest): number {
  let count = 0;
  count += request.model?.length || 0;
  count += request.instructions?.length || 0;
  count += countStandardInputCharacters(request.input);
  count += countUnknownCharacters(request.tools);
  count += countUnknownCharacters(request.tool_choice);
  count += countUnknownCharacters(request.reasoning);
  count += countUnknownCharacters(request.thinking);
  count += countUnknownCharacters(request.output_config);
  count += countUnknownCharacters(request.text);
  return count;
}

function countStandardInputCharacters(input: StandardRequest['input']): number {
  if (typeof input === 'string') {
    return input.length;
  }

  return input.reduce((sum, message) => sum + countMessageCharacters(message), 0);
}

function countMessageCharacters(message: StandardRequestInputMessage): number {
  return (
    message.role.length +
    message.content.reduce((sum, item) => sum + countContentCharacters(item), 0)
  );
}

function countContentCharacters(item: StandardRequestInputContent): number {
  if (item.type === 'input_text') {
    return item.text.length;
  }

  if (item.type === 'tool_result') {
    return item.content.length;
  }

  if (item.type === 'reasoning') {
    return (
      (item.text?.length || 0) +
      (item.summary?.length || 0) +
      countUnknownCharacters(item.reasoning_details)
    );
  }

  if (item.type === 'tool_search_call') {
    return 'ToolSearch'.length + countUnknownCharacters(item.arguments);
  }

  if (item.type === 'tool_search_output') {
    return countUnknownCharacters(item.tools);
  }

  if (item.type === 'provider_native_item') {
    return countUnknownCharacters(item.raw_payload);
  }

  return item.name.length + countUnknownCharacters(item.input);
}

function countUnknownCharacters(value: unknown): number {
  if (value === undefined || value === null) {
    return 0;
  }

  if (typeof value === 'string') {
    return value.length;
  }

  if (isBinaryValue(value)) {
    return 0;
  }

  try {
    return JSON.stringify(value)?.length || 0;
  } catch {
    return String(value).length;
  }
}

function countImageInputs(value: unknown): number {
  if (Array.isArray(value)) {
    return value.reduce((sum, item) => sum + countImageInputs(item), 0);
  }

  if (isBinaryValue(value)) {
    return 0;
  }

  if (!isObject(value)) {
    return 0;
  }

  if (isImageBlock(value)) {
    return 1;
  }

  return Object.values(value).reduce<number>((sum, item) => sum + countImageInputs(item), 0);
}

function isBinaryValue(value: unknown): boolean {
  return (
    Buffer.isBuffer(value) ||
    value instanceof ArrayBuffer ||
    (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView(value))
  );
}

function isImageBlock(value: Record<string, unknown>): boolean {
  const type = readObjectString(value, 'type')?.toLowerCase();
  if (type === 'image_url' || type === 'input_image') {
    return true;
  }

  if (type === 'image') {
    return true;
  }

  if (value.image_url !== undefined || value.input_image !== undefined) {
    return true;
  }

  const inlineData = isObject(value.inlineData)
    ? value.inlineData
    : isObject(value.inline_data)
      ? value.inline_data
      : undefined;
  const inlineMimeType =
    readObjectString(inlineData, 'mimeType') || readObjectString(inlineData, 'mime_type');
  if (inlineMimeType?.toLowerCase().startsWith('image/')) {
    return true;
  }

  const source = isObject(value.source) ? value.source : undefined;
  const sourceMediaType = readObjectString(source, 'media_type');
  return sourceMediaType?.toLowerCase().startsWith('image/') === true;
}

function readObjectString(
  value: Record<string, unknown> | undefined,
  key: string
): string | undefined {
  const item = value?.[key];
  return typeof item === 'string' && item.trim() ? item.trim() : undefined;
}

function resolveMaxOutputTokens(input: GatewayPrecheckInput): number {
  const fromStandard = input.standardRequest?.max_output_tokens;
  if (typeof fromStandard === 'number' && Number.isFinite(fromStandard) && fromStandard >= 0) {
    return Math.max(0, Math.ceil(fromStandard));
  }

  if (isObject(input.requestBody)) {
    const raw =
      input.requestBody.max_output_tokens ??
      input.requestBody.max_tokens ??
      input.requestBody.max_completion_tokens;
    if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
      return Math.ceil(raw);
    }
  }

  return input.config.precheck.estimation.defaultMaxOutputTokens;
}

function resolveSubjectKey(
  request: FastifyRequest,
  subject: GatewayPrecheckSubject,
  headerName?: string
): string {
  const identity = (request as FastifyRequest & { gatewayIdentity?: GatewayRequestIdentity })
    .gatewayIdentity;

  if (subject === 'global') {
    return 'global';
  }

  if (subject === 'identity') {
    return (
      identity?.billingSubjectKey ||
      identity?.userId ||
      identity?.tenantId ||
      identity?.organizationId ||
      identity?.apiKeyId ||
      resolveClientIp(request)
    );
  }

  if (subject === 'user') {
    return identity?.userId || identity?.subject || resolveClientIp(request);
  }

  if (subject === 'tenant') {
    return identity?.tenantId || resolveClientIp(request);
  }

  if (subject === 'organization') {
    return identity?.organizationId || resolveClientIp(request);
  }

  if (subject === 'api_key') {
    return identity?.apiKeyId || resolveApiKeySubject(request) || resolveClientIp(request);
  }

  if (subject === 'header') {
    const headerValue = headerName ? readHeader(request.headers[headerName]) : undefined;
    return headerValue ? `header:${hashSubjectValue(headerValue)}` : resolveClientIp(request);
  }

  return resolveClientIp(request);
}

function resolveApiKeySubject(request: FastifyRequest): string | undefined {
  const value =
    readHeader(request.headers['x-api-key']) ||
    readHeader(request.headers['api-key']) ||
    readHeader(request.headers.authorization);
  return value ? `api_key:${hashSubjectValue(value)}` : undefined;
}

function resolveClientIp(request: FastifyRequest): string {
  return `ip:${request.ip || request.socket.remoteAddress || 'unknown'}`;
}

function resolveScopeKey(
  scope: GatewayPrecheckScope,
  provider: Provider,
  model: string | undefined
): string {
  if (scope === 'provider') {
    return `provider:${provider}`;
  }

  if (scope === 'model') {
    return `model:${model || 'unknown'}`;
  }

  if (scope === 'provider_model') {
    return `provider:${provider}:model:${model || 'unknown'}`;
  }

  return 'global';
}

function hashSubjectValue(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 24);
}

function resolveProviderBillingRate(
  config: GatewayConfig,
  provider: Provider,
  model: string | undefined,
  targetProviderConfig?: ProviderConfig
): BillingRate | undefined {
  const providerConfig = targetProviderConfig || findProviderConfigByType(config.providers, provider);
  if (!providerConfig) {
    return undefined;
  }

  if (model && providerConfig.billing.byModel[model]) {
    return providerConfig.billing.byModel[model];
  }

  return providerConfig.billing.default;
}

function findProviderConfigByType(
  providers: ProviderConfig[],
  provider: Provider
): ProviderConfig | undefined {
  return findDefaultProviderConfig(providers, provider);
}
