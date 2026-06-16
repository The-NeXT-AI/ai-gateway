import { createHash } from 'node:crypto';
import type { FastifyRequest } from 'fastify';
import { calculateUsageBilling } from '../billing';
import type {
  BillingRate,
  GatewayConfig,
  GatewayPrecheckRuleBaseConfig,
  GatewayPrecheckScope,
  GatewayPrecheckSubject,
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
import { isObject, providerFromProviderType, readHeader } from '../utils';

type PrecheckKind = 'rate_limit' | 'quota' | 'budget';

export interface GatewayPrecheckInput {
  request: FastifyRequest;
  config: GatewayConfig;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  model?: string;
  standardRequest?: StandardRequest;
  requestBody?: unknown;
}

export interface GatewayPrecheckEstimate {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  imageCount: number;
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

const counters = new Map<string, WindowCounter>();

export async function evaluateGatewayPrecheck(input: GatewayPrecheckInput): Promise<GatewayPrecheckResult> {
  const precheck = input.config.precheck;
  if (!precheck?.enabled) {
    return { ok: true };
  }

  const rateLimitRules = precheck.rateLimit.enabled
    ? resolveRateLimitRules(precheck.rateLimit)
    : [];
  const hasQuota = precheck.quota.enabled && precheck.quota.maxTokens > 0;
  const hasBudget = precheck.budget.enabled && precheck.budget.maxCostUsd > 0;

  if (rateLimitRules.length === 0 && !hasQuota && !hasBudget) {
    return { ok: true };
  }

  const needsEstimate =
    hasQuota ||
    hasBudget ||
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

export async function closeGatewayPrecheckStore(): Promise<void> {
  return;
}

export function resetGatewayPrecheckStateForTests(): void {
  counters.clear();
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
    total_tokens: totalTokens
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
    imageCount: countImageInputs(input.requestBody),
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

  return item.name.length + countUnknownCharacters(item.input);
}

function countUnknownCharacters(value: unknown): number {
  if (value === undefined || value === null) {
    return 0;
  }

  if (typeof value === 'string') {
    return value.length;
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

  if (!isObject(value)) {
    return 0;
  }

  if (isImageBlock(value)) {
    return 1;
  }

  return Object.values(value).reduce<number>((sum, item) => sum + countImageInputs(item), 0);
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
  return providers.find((item) => providerFromProviderType(item.type) === provider);
}
