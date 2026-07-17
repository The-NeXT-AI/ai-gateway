import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import type { FastifyRequest, preHandlerHookHandler } from 'fastify';
import { isInternalIp } from '../shared/ip';
import type {
  GatewayApiKeyRestrictions,
  GatewayAuthConfig,
  GatewayRequestIdentity,
  Provider,
  ProviderConfig
} from '../types';
import { readBearerToken, readHeader } from '../utils';
import { readOpenAIMultipartRequestMetadata } from './multipart';

const SDK_COMPATIBLE_TOKEN_HEADERS = [
  'authorization',
  'x-api-key',
  'api-key',
  'x-goog-api-key',
  'x-mcp-key',
  'x-codex-access-token'
] as const;
const agentInternalAuthHeader = 'x-gateway-agent-internal';
const agentInternalAuthHeaderValue = '1';

interface GatewayAuthResultOk {
  ok: true;
  identity?: GatewayRequestIdentity;
  apiKeyRestrictions?: GatewayApiKeyRestrictions;
}

interface GatewayAuthResultError {
  ok: false;
  statusCode: number;
  error: string;
}

export type GatewayAuthResult = GatewayAuthResultOk | GatewayAuthResultError;

declare module 'fastify' {
  interface FastifyRequest {
    gatewayIdentity?: GatewayRequestIdentity;
    gatewayApiKeyRestrictions?: GatewayApiKeyRestrictions;
    gatewayAuthModelCandidates?: string[];
  }
}

export function addGatewayAuthModelCandidate(
  request: FastifyRequest,
  model: string | undefined
): void {
  const normalized = normalizeString(model);
  if (!normalized) {
    return;
  }
  request.gatewayAuthModelCandidates ||= [];
  if (!request.gatewayAuthModelCandidates.includes(normalized)) {
    request.gatewayAuthModelCandidates.push(normalized);
  }
}

export function createGatewayAuthPreHandler(config: GatewayAuthConfig): preHandlerHookHandler {
  return async function gatewayAuthPreHandler(request, reply): Promise<void> {
    const result = await authenticateGatewayRequest(request, config);
    if (!result.ok) {
      reply.code(result.statusCode).send({
        error: {
          message: result.error
        }
      });
      return;
    }

    request.gatewayIdentity = result.identity;
    request.gatewayApiKeyRestrictions = result.apiKeyRestrictions;
  };
}

export function authenticateGatewayRequest(
  request: FastifyRequest,
  config: GatewayAuthConfig
): Promise<GatewayAuthResult> {
  if (config.mode === 'http_introspection') {
    return authenticateViaIntrospection(request, config);
  }

  if (config.mode === 'static_api_key') {
    return Promise.resolve(authenticateViaStaticApiKey(request, config));
  }

  return Promise.resolve(authenticateViaTrustedHeader(request, config));
}

function authenticateViaTrustedHeader(
  request: FastifyRequest,
  config: GatewayAuthConfig
): GatewayAuthResult {
  if (!config.enabled) {
    return { ok: true };
  }

  if (config.trustedCidrs.length > 0) {
    const clientIp = readClientIp(request);
    if (!isInternalIp(clientIp, config.trustedCidrs)) {
      return {
        ok: false,
        statusCode: 403,
        error: 'Request source is not in trusted CIDR ranges.'
      };
    }
  }

  const identity = readIdentityFromHeaders(request, config);
  const billingSubjectKey = resolveBillingSubjectKey(identity);

  if (!billingSubjectKey) {
    if (config.required) {
      return {
        ok: false,
        statusCode: 401,
        error: 'Missing authenticated user identity headers.'
      };
    }

    return { ok: true };
  }

  if (config.signature.enabled) {
    const signatureValidation = validateIdentitySignature(request, config, identity);
    if (!signatureValidation.ok) {
      return signatureValidation;
    }
  }

  return {
    ok: true,
    identity: {
      source: 'trusted_header',
      billingSubjectKey,
      ...identity
    }
  };
}

function authenticateViaStaticApiKey(
  request: FastifyRequest,
  config: GatewayAuthConfig
): GatewayAuthResult {
  if (!config.enabled) {
    return { ok: true };
  }

  const staticConfig = config.staticApiKeys;
  if (!staticConfig || staticConfig.keys.length === 0) {
    return {
      ok: false,
      statusCode: 500,
      error: 'Static API key auth is enabled but no API keys are configured.'
    };
  }

  const tokenLookup = resolveAuthToken(
    request.headers,
    staticConfig.keyHeader,
    staticConfig.keyBearerOnly
  );
  if (!tokenLookup.token) {
    if (config.required) {
      return {
        ok: false,
        statusCode: 401,
        error: tokenLookup.invalidFormat
          ? 'Invalid auth token format.'
          : `Missing auth token header: ${staticConfig.keyHeader}`
      };
    }

    return { ok: true };
  }

  if (!matchesAnyStaticApiKey(tokenLookup.token, staticConfig.keys)) {
    return {
      ok: false,
      statusCode: 401,
      error: 'Invalid API key.'
    };
  }

  const apiKeyId = buildStaticApiKeyId(tokenLookup.token);
  return {
    ok: true,
    identity: {
      source: 'static_api_key',
      billingSubjectKey: `api_key:${apiKeyId}`,
      subject: apiKeyId,
      apiKeyId
    }
  };
}

async function authenticateViaIntrospection(
  request: FastifyRequest,
  config: GatewayAuthConfig
): Promise<GatewayAuthResult> {
  if (!config.enabled) {
    return { ok: true };
  }

  if (allowInternalAgentRequestWithoutIntrospection(request, config)) {
    return buildInternalAgentAuthResult(request, config);
  }

  const endpoint = config.introspection.endpoint;
  if (!endpoint) {
    return {
      ok: false,
      statusCode: 500,
      error: 'Auth introspection endpoint is not configured.'
    };
  }

  const tokenLookup = resolveAuthToken(
    request.headers,
    config.introspection.tokenHeader,
    config.introspection.tokenBearerOnly
  );
  if (!tokenLookup.token) {
    if (config.required) {
      if (tokenLookup.invalidFormat) {
        return {
          ok: false,
          statusCode: 401,
          error: 'Invalid auth token format.'
        };
      }

      return {
        ok: false,
        statusCode: 401,
        error: `Missing auth token header: ${config.introspection.tokenHeader}`
      };
    }

    return { ok: true };
  }

  const token = tokenLookup.token;

  const modelCandidates = collectRequestedModelsFromRequest(request);
  const clientIp = resolveClientIp(request);
  const origin = readHeaderValue(request.headers, 'origin');
  const referer = readHeaderValue(request.headers, 'referer');

  const requestBody: Record<string, unknown> = {
    [config.introspection.requestTokenField]: token,
    method: request.method.toUpperCase(),
    path: sanitizePath(request.url)
  };
  if (clientIp) {
    requestBody.clientIp = clientIp;
  }
  if (origin) {
    requestBody.origin = origin;
  }
  if (referer) {
    requestBody.referer = referer;
  }
  if (modelCandidates.length > 0) {
    requestBody.model = modelCandidates[0];
    requestBody.models = modelCandidates;
  }

  const headers: Record<string, string> = {
    'content-type': 'application/json'
  };

  const credential = process.env[config.introspection.credentialEnv];
  if (credential) {
    headers[config.introspection.credentialHeader] = credential;
  }

  let response: Response;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.introspection.timeoutMs);
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });
  } catch (error) {
    const errorName = error instanceof Error ? error.name : '';
    if (errorName === 'AbortError') {
      return {
        ok: false,
        statusCode: 502,
        error: 'Auth introspection request timed out.'
      };
    }

    return {
      ok: false,
      statusCode: 502,
      error: 'Failed to reach auth introspection service.'
    };
  } finally {
    clearTimeout(timeout);
  }

  let payload: unknown = undefined;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }

  if (!response.ok) {
    return {
      ok: false,
      statusCode: response.status === 401 || response.status === 403 ? 401 : 502,
      error: 'Auth introspection service rejected the request.'
    };
  }

  const introspectionPayload = unwrapIntrospectionPayload(payload);
  const active =
    readBooleanFromPath(introspectionPayload, config.introspection.responseMap.active) ??
    readBooleanFromPath(payload, config.introspection.responseMap.active);
  if (active === false) {
    const reason =
      readStringFromPath(introspectionPayload, 'reason') ||
      readStringFromPath(payload, 'reason') ||
      readStringFromPath(introspectionPayload, 'message');
    return {
      ok: false,
      statusCode: 401,
      error: reason
        ? `Auth introspection token is inactive: ${reason}`
        : 'Auth introspection token is inactive.'
    };
  }

  const identity = {
    userId:
      readStringFromPath(introspectionPayload, config.introspection.responseMap.userId) ||
      readStringFromPath(payload, config.introspection.responseMap.userId),
    tenantId:
      readStringFromPath(introspectionPayload, config.introspection.responseMap.tenantId) ||
      readStringFromPath(payload, config.introspection.responseMap.tenantId),
    subject:
      readStringFromPath(introspectionPayload, config.introspection.responseMap.subject) ||
      readStringFromPath(payload, config.introspection.responseMap.subject),
    organizationId:
      readStringFromPath(introspectionPayload, config.introspection.responseMap.organizationId) ||
      readStringFromPath(payload, config.introspection.responseMap.organizationId),
    plan:
      readStringFromPath(introspectionPayload, config.introspection.responseMap.plan) ||
      readStringFromPath(payload, config.introspection.responseMap.plan),
    apiKeyId:
      (config.introspection.responseMap.apiKeyId
        ? readStringFromPath(introspectionPayload, config.introspection.responseMap.apiKeyId) ||
          readStringFromPath(payload, config.introspection.responseMap.apiKeyId)
        : undefined)
  };
  const apiKeyRestrictions = readApiKeyRestrictions(introspectionPayload) || readApiKeyRestrictions(payload);
  const restrictionResult = evaluateApiKeyRequestRestrictions(
    request,
    apiKeyRestrictions
  );
  if (!restrictionResult.ok) {
    return restrictionResult;
  }

  const billingSubjectKey = resolveBillingSubjectKey(identity);
  if (!billingSubjectKey) {
    if (config.required) {
      return {
        ok: false,
        statusCode: 401,
        error: 'Auth introspection response does not include a billing subject.'
      };
    }

    return {
      ok: true,
      apiKeyRestrictions
    };
  }

  return {
    ok: true,
    identity: {
      source: 'http_introspection',
      billingSubjectKey,
      ...identity
    },
    apiKeyRestrictions
  };
}

export function evaluateApiKeyModelRestriction(
  request: FastifyRequest,
  model: string | undefined,
  target?: {
    provider?: Provider;
    providerConfig?: ProviderConfig;
  }
): GatewayAuthResult {
  const restrictions = request.gatewayApiKeyRestrictions;
  const modelValue = normalizeString(model);
  if (!restrictions) {
    return { ok: true };
  }
  if (!modelValue) {
    const allowedModels = mergeRestrictionLists(
      restrictions.allowedModels,
      restrictions.modelWhitelist
    );
    return allowedModels.length > 0
      ? {
          ok: false,
          statusCode: 403,
          error: 'API key model restriction cannot be evaluated because the model is missing.'
        }
      : { ok: true };
  }

  return evaluateModelRestriction(
    buildModelRestrictionCandidates(modelValue, target),
    restrictions
  );
}

export function resolveBillingSubjectKey(identity: {
  userId?: string;
  tenantId?: string;
  subject?: string;
}): string | undefined {
  if (identity.tenantId && identity.userId) {
    return `${identity.tenantId}:${identity.userId}`;
  }

  if (identity.tenantId && identity.subject) {
    return `${identity.tenantId}:${identity.subject}`;
  }

  if (identity.userId) {
    return identity.userId;
  }

  return identity.subject;
}

function readIdentityFromHeaders(
  request: FastifyRequest,
  config: GatewayAuthConfig
): {
  userId?: string;
  tenantId?: string;
  subject?: string;
  organizationId?: string;
  plan?: string;
  apiKeyId?: string;
} {
  return {
    userId: readHeaderValue(request.headers, config.identityHeaders.userId),
    tenantId: readHeaderValue(request.headers, config.identityHeaders.tenantId),
    subject: readHeaderValue(request.headers, config.identityHeaders.subject),
    organizationId: readHeaderValue(request.headers, config.identityHeaders.organizationId),
    plan: readHeaderValue(request.headers, config.identityHeaders.plan),
    apiKeyId: config.identityHeaders.apiKeyId
      ? readHeaderValue(request.headers, config.identityHeaders.apiKeyId)
      : undefined
  };
}

function buildInternalAgentAuthResult(
  request: FastifyRequest,
  config: GatewayAuthConfig
): GatewayAuthResult {
  const identity = readIdentityFromHeaders(request, config);
  const billingSubjectKey = resolveBillingSubjectKey(identity);
  if (!billingSubjectKey) {
    return { ok: true };
  }

  return {
    ok: true,
    identity: {
      source: 'http_introspection',
      billingSubjectKey,
      ...identity
    }
  };
}

function validateIdentitySignature(
  request: FastifyRequest,
  config: GatewayAuthConfig,
  identity: {
    userId?: string;
    tenantId?: string;
    subject?: string;
    organizationId?: string;
    plan?: string;
  }
): GatewayAuthResult {
  const signatureValue = readHeaderValue(request.headers, config.signature.header);
  if (!signatureValue) {
    return {
      ok: false,
      statusCode: 401,
      error: `Missing signature header: ${config.signature.header}`
    };
  }

  const timestampValue = readHeaderValue(request.headers, config.signature.timestampHeader);
  if (!timestampValue) {
    return {
      ok: false,
      statusCode: 401,
      error: `Missing signature timestamp header: ${config.signature.timestampHeader}`
    };
  }

  const timestampMs = parseTimestampToMs(timestampValue);
  if (timestampMs === undefined) {
    return {
      ok: false,
      statusCode: 401,
      error: 'Invalid auth signature timestamp format.'
    };
  }

  const maxSkewMs = config.signature.maxSkewSec * 1000;
  if (Math.abs(Date.now() - timestampMs) > maxSkewMs) {
    return {
      ok: false,
      statusCode: 401,
      error: 'Auth signature timestamp is expired.'
    };
  }

  const secret = process.env[config.signature.secretEnv];
  if (!secret) {
    return {
      ok: false,
      statusCode: 500,
      error: `Auth signature secret env is missing: ${config.signature.secretEnv}`
    };
  }

  const payload = buildSignaturePayload(timestampValue, request, identity);
  const expected = createHmac('sha256', secret).update(payload).digest('hex');
  const provided = normalizeSignature(signatureValue);
  if (!isHexSha256(provided) || !timingSafeHexEqual(expected, provided)) {
    return {
      ok: false,
      statusCode: 401,
      error: 'Invalid auth signature.'
    };
  }

  return { ok: true };
}

function buildSignaturePayload(
  timestampValue: string,
  request: FastifyRequest,
  identity: {
    userId?: string;
    tenantId?: string;
    subject?: string;
    organizationId?: string;
    plan?: string;
  }
): string {
  const path = sanitizePath(request.url);
  return [
    timestampValue,
    request.method.toUpperCase(),
    path,
    identity.tenantId || '',
    identity.userId || '',
    identity.subject || '',
    identity.organizationId || '',
    identity.plan || ''
  ].join('\n');
}

function readHeaderValue(headers: FastifyRequest['headers'], headerName: string): string | undefined {
  const raw = headers[headerName.toLowerCase()];
  const value = readHeader(typeof raw === 'string' || Array.isArray(raw) ? raw : undefined);
  if (!value) {
    return undefined;
  }

  const normalized = value.trim();
  return normalized || undefined;
}

function unwrapIntrospectionPayload(payload: unknown): unknown {
  if (!isRecord(payload)) {
    return payload;
  }

  if (isRecord(payload.data)) {
    return payload.data;
  }

  return payload;
}

function resolveAuthToken(
  headers: FastifyRequest['headers'],
  primaryHeader: string,
  bearerOnly: boolean
): { token?: string; invalidFormat: boolean } {
  let invalidFormat = false;

  for (const headerName of buildTokenHeaderCandidates(primaryHeader)) {
    const tokenRaw = readHeaderValue(headers, headerName);
    if (!tokenRaw) {
      continue;
    }

    const token = normalizeAuthTokenValue(tokenRaw, headerName, primaryHeader, bearerOnly);
    if (token) {
      return { token, invalidFormat };
    }

    invalidFormat = true;
  }

  return { invalidFormat };
}

function buildTokenHeaderCandidates(primaryHeader: string): string[] {
  const candidates: string[] = [];
  const pushHeader = (headerName: string | undefined) => {
    if (!headerName) {
      return;
    }

    const normalized = headerName.trim().toLowerCase();
    if (!normalized || candidates.includes(normalized)) {
      return;
    }

    candidates.push(normalized);
  };

  pushHeader(primaryHeader);
  for (const headerName of SDK_COMPATIBLE_TOKEN_HEADERS) {
    pushHeader(headerName);
  }

  return candidates;
}

function normalizeAuthTokenValue(
  tokenRaw: string,
  headerName: string,
  primaryHeader: string,
  bearerOnly: boolean
): string | undefined {
  const normalizedPrimaryHeader = primaryHeader.toLowerCase();

  if (headerName === 'authorization') {
    const bearerToken = readBearerToken(tokenRaw);
    if (bearerToken) {
      return bearerToken;
    }

    if (headerName === normalizedPrimaryHeader && !bearerOnly) {
      return tokenRaw;
    }

    return undefined;
  }

  if (headerName === normalizedPrimaryHeader) {
    if (!bearerOnly) {
      return tokenRaw;
    }

    return readBearerToken(tokenRaw);
  }

  return tokenRaw;
}

function matchesAnyStaticApiKey(token: string, allowedKeys: string[]): boolean {
  return allowedKeys.some((allowedKey) => timingSafeTokenEqual(token, allowedKey));
}

function timingSafeTokenEqual(left: string, right: string): boolean {
  const leftDigest = createHash('sha256').update(left).digest();
  const rightDigest = createHash('sha256').update(right).digest();
  return timingSafeEqual(leftDigest, rightDigest);
}

function buildStaticApiKeyId(token: string): string {
  return createHash('sha256').update(token).digest('hex').slice(0, 16);
}

function readStringFromPath(value: unknown, path: string): string | undefined {
  const picked = readValueByPath(value, path);
  if (typeof picked !== 'string') {
    return undefined;
  }

  const normalized = picked.trim();
  return normalized || undefined;
}

function readBooleanFromPath(value: unknown, path: string): boolean | undefined {
  const picked = readValueByPath(value, path);
  if (typeof picked === 'boolean') {
    return picked;
  }

  if (typeof picked === 'string') {
    const normalized = picked.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'on') {
      return true;
    }

    if (normalized === 'false' || normalized === '0' || normalized === 'no' || normalized === 'off') {
      return false;
    }
  }

  if (typeof picked === 'number' && Number.isFinite(picked)) {
    return picked !== 0;
  }

  return undefined;
}

function readValueByPath(value: unknown, path: string): unknown {
  const normalizedPath = path.trim();
  if (!normalizedPath) {
    return undefined;
  }

  const segments = normalizedPath
    .split('.')
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length === 0) {
    return undefined;
  }

  let current: unknown = value;
  for (const segment of segments) {
    if (!isRecord(current)) {
      return undefined;
    }

    current = current[segment];
  }

  return current;
}

function readApiKeyRestrictions(value: unknown): GatewayApiKeyRestrictions | undefined {
  const raw = isRecord(value)
    ? isRecord(value.restrictions)
      ? value.restrictions
      : isRecord(value.apiKey) && isRecord(value.apiKey.restrictions)
        ? value.apiKey.restrictions
        : undefined
    : undefined;
  if (!raw) {
    return undefined;
  }

  const restrictions: GatewayApiKeyRestrictions = {};
  copyStringArray(raw, restrictions, 'ipWhitelist');
  copyStringArray(raw, restrictions, 'allowedIps');
  copyStringArray(raw, restrictions, 'allowedOrigins');
  copyStringArray(raw, restrictions, 'allowedDomains');
  copyStringArray(raw, restrictions, 'allowedModels');
  copyStringArray(raw, restrictions, 'modelWhitelist');
  copyNumber(raw, restrictions, 'rateLimit');
  copyNumber(raw, restrictions, 'requestsPerMinute');
  copyNumber(raw, restrictions, 'rpm');
  copyNumber(raw, restrictions, 'rateLimitWindowSeconds');
  copyNumber(raw, restrictions, 'tokensPerMinute');
  copyNumber(raw, restrictions, 'tpm');
  copyNumber(raw, restrictions, 'tokenLimitWindowSeconds');
  copyNumber(raw, restrictions, 'costLimitUsd');
  copyNumber(raw, restrictions, 'costLimit');
  copyNumber(raw, restrictions, 'maxCostUsd');
  copyNumber(raw, restrictions, 'costPerMinuteUsd');
  copyNumber(raw, restrictions, 'costLimitWindowSeconds');

  return Object.keys(restrictions).length > 0 ? restrictions : undefined;
}

function copyStringArray<K extends keyof GatewayApiKeyRestrictions>(
  source: Record<string, unknown>,
  target: GatewayApiKeyRestrictions,
  key: K
) {
  const value = source[key];
  if (!Array.isArray(value)) {
    return;
  }

  const strings = value
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.trim())
    .filter(Boolean);
  if (strings.length > 0) {
    target[key] = strings as GatewayApiKeyRestrictions[K];
  }
}

function copyNumber<K extends keyof GatewayApiKeyRestrictions>(
  source: Record<string, unknown>,
  target: GatewayApiKeyRestrictions,
  key: K
) {
  const value = source[key];
  const numeric = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
  if (Number.isFinite(numeric)) {
    target[key] = numeric as GatewayApiKeyRestrictions[K];
  }
}

function evaluateApiKeyRequestRestrictions(
  request: FastifyRequest,
  restrictions: GatewayApiKeyRestrictions | undefined
): GatewayAuthResult {
  if (!restrictions) {
    return { ok: true };
  }

  const allowedIps = mergeRestrictionLists(restrictions.ipWhitelist, restrictions.allowedIps);
  if (allowedIps.length > 0 && !matchesIpList(resolveClientIp(request), allowedIps)) {
    return {
      ok: false,
      statusCode: 403,
      error: 'API key is not allowed from this IP.'
    };
  }

  const allowedDomains = mergeRestrictionLists(restrictions.allowedOrigins, restrictions.allowedDomains);
  if (allowedDomains.length > 0 && !matchesRequestOrigin(request, allowedDomains)) {
    return {
      ok: false,
      statusCode: 403,
      error: 'API key is not allowed from this origin.'
    };
  }

  return { ok: true };
}

function evaluateModelRestriction(
  model: string | string[],
  restrictions: GatewayApiKeyRestrictions
): GatewayAuthResult {
  const allowedModels = mergeRestrictionLists(restrictions.allowedModels, restrictions.modelWhitelist);
  if (allowedModels.length === 0) {
    return { ok: true };
  }

  const candidates = Array.isArray(model) ? model : [model];
  if (allowedModels.some((pattern) => candidates.some((candidate) => matchesWildcard(pattern, candidate)))) {
    return { ok: true };
  }

  return {
    ok: false,
    statusCode: 403,
    error: `API key is not allowed to use this model: ${candidates[0]}`
  };
}

function buildModelRestrictionCandidates(
  model: string,
  target:
    | {
        provider?: Provider;
        providerConfig?: ProviderConfig;
      }
    | undefined
): string[] {
  const candidates: string[] = [];
  pushUniqueString(candidates, model);
  if (target?.provider) {
    pushUniqueString(candidates, `${target.provider}/${model}`);
  }
  if (target?.providerConfig?.name) {
    pushUniqueString(candidates, `${target.providerConfig.name}/${model}`);
  }
  if (target?.providerConfig?.type) {
    pushUniqueString(candidates, `${target.providerConfig.type}/${model}`);
  }
  return candidates;
}

function mergeRestrictionLists(...values: Array<string[] | undefined>): string[] {
  const merged: string[] = [];
  for (const value of values) {
    if (!Array.isArray(value)) {
      continue;
    }

    for (const item of value) {
      const normalized = item.trim();
      if (normalized && !merged.includes(normalized)) {
        merged.push(normalized);
      }
    }
  }

  return merged;
}

function collectRequestedModelsFromRequest(request: FastifyRequest): string[] {
  const models: string[] = [];
  for (const model of request.gatewayAuthModelCandidates || []) {
    pushUniqueString(models, normalizeString(model));
  }
  pushUniqueString(models, readHeaderValue(request.headers, 'x-target-model'));
  if (isRecord(request.body)) {
    pushUniqueString(models, normalizeString(request.body.model));
  }
  const multipartMetadata = readOpenAIMultipartRequestMetadata(request);
  if (multipartMetadata?.ok) {
    pushUniqueString(models, normalizeString(multipartMetadata.value.fields.model));
  }

  pushUniqueString(models, readModelFromPath(request.url));
  return models;
}

function readModelFromPath(url: string): string | undefined {
  const path = sanitizePath(url);
  const marker = '/models/';
  const markerIndex = path.indexOf(marker);
  if (markerIndex < 0) {
    return undefined;
  }

  const tail = decodeURIComponent(path.slice(markerIndex + marker.length));
  const model = tail.split(':')[0]?.trim();
  return model || undefined;
}

function pushUniqueString(values: string[], value: string | undefined) {
  if (value && !values.includes(value)) {
    values.push(value);
  }
}

function normalizeString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim();
  return normalized || undefined;
}

function resolveClientIp(request: FastifyRequest): string {
  return readClientIp(request) || request.socket?.remoteAddress || '';
}

function matchesIpList(ip: string, allowed: string[]): boolean {
  const normalizedIp = normalizeIpValue(ip);
  if (!normalizedIp) {
    return false;
  }

  return allowed.some((entry) => {
    const normalizedEntry = normalizeIpValue(entry);
    if (!normalizedEntry) {
      return false;
    }

    if (!normalizedEntry.includes('/')) {
      return normalizedEntry === normalizedIp;
    }

    return matchesIpv4Cidr(normalizedIp, normalizedEntry);
  });
}

function normalizeIpValue(value: string): string | undefined {
  const first = value.split(',')[0]?.trim();
  if (!first) {
    return undefined;
  }

  const withoutPort = stripIpPort(first);
  if (!withoutPort) {
    return undefined;
  }

  return withoutPort.startsWith('::ffff:') ? withoutPort.slice('::ffff:'.length) : withoutPort;
}

function stripIpPort(value: string): string {
  if (value.startsWith('[') && value.includes(']')) {
    return value.slice(1, value.indexOf(']'));
  }

  const colonCount = value.split(':').length - 1;
  if (colonCount === 1 && value.includes('.')) {
    return value.split(':')[0] || value;
  }

  return value;
}

function matchesIpv4Cidr(ip: string, cidr: string): boolean {
  const [base, bitsRaw] = cidr.split('/');
  const bits = Number(bitsRaw);
  if (!base || !Number.isInteger(bits) || bits < 0 || bits > 32) {
    return false;
  }

  const ipNumber = ipv4ToNumber(ip);
  const baseNumber = ipv4ToNumber(base);
  if (ipNumber === undefined || baseNumber === undefined) {
    return false;
  }

  const mask = bits === 0 ? 0 : (0xffffffff << (32 - bits)) >>> 0;
  return (ipNumber & mask) === (baseNumber & mask);
}

function ipv4ToNumber(ip: string): number | undefined {
  const parts = ip.split('.').map((part) => Number(part));
  if (
    parts.length !== 4 ||
    parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)
  ) {
    return undefined;
  }

  return (
    ((parts[0] << 24) >>> 0) +
    ((parts[1] << 16) >>> 0) +
    ((parts[2] << 8) >>> 0) +
    parts[3]
  ) >>> 0;
}

function matchesRequestOrigin(request: FastifyRequest, allowed: string[]): boolean {
  const origin = readHeaderValue(request.headers, 'origin');
  const referer = readHeaderValue(request.headers, 'referer');
  const candidates = [origin, referer].filter((value): value is string => Boolean(value));
  if (candidates.length === 0) {
    return false;
  }

  return candidates.some((candidate) =>
    allowed.some((allowedEntry) => matchesOriginOrDomain(candidate, allowedEntry))
  );
}

function matchesOriginOrDomain(requestOrigin: string, allowedEntry: string): boolean {
  if (allowedEntry === '*') {
    return true;
  }

  const requestParts = parseOriginParts(requestOrigin);
  const allowedParts = parseOriginParts(allowedEntry);
  if (!requestParts || !allowedParts) {
    return false;
  }

  if (allowedEntry.includes('://') && allowedParts.origin) {
    return requestParts.origin === allowedParts.origin;
  }

  return (
    matchesWildcard(allowedParts.hostname, requestParts.hostname) ||
    requestParts.hostname === allowedParts.hostname ||
    requestParts.hostname.endsWith(`.${allowedParts.hostname}`)
  );
}

function parseOriginParts(value: string): { origin?: string; hostname: string } | undefined {
  const trimmed = value.trim().toLowerCase().replace(/\/+$/, '');
  if (!trimmed) {
    return undefined;
  }

  try {
    const parsed = new URL(trimmed.includes('://') ? trimmed : `https://${trimmed}`);
    return {
      origin: parsed.origin,
      hostname: parsed.hostname
    };
  } catch {
    const hostname = trimmed.replace(/^https?:\/\//, '').split('/')[0]?.split(':')[0]?.trim();
    return hostname ? { hostname } : undefined;
  }
}

function matchesWildcard(pattern: string, value: string): boolean {
  if (pattern === '*') {
    return true;
  }

  if (!pattern.includes('*')) {
    return pattern === value;
  }

  const escaped = pattern
    .split('*')
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('.*');
  return new RegExp(`^${escaped}$`).test(value);
}

function allowInternalAgentRequestWithoutIntrospection(
  request: FastifyRequest,
  config: GatewayAuthConfig
): boolean {
  const internalMarker = readHeaderValue(request.headers, agentInternalAuthHeader);
  if (internalMarker !== agentInternalAuthHeaderValue) {
    return false;
  }

  if (!isLoopbackIp(request.ip)) {
    return false;
  }

  const expectedSecretRaw = process.env[config.introspection.credentialEnv];
  const expectedSecret = typeof expectedSecretRaw === 'string' ? expectedSecretRaw.trim() : '';
  if (!expectedSecret) {
    return false;
  }

  const providedSecret = readHeaderValue(request.headers, config.introspection.credentialHeader);
  if (!providedSecret) {
    return false;
  }

  return timingSafeStringEqual(providedSecret, expectedSecret);
}

function isLoopbackIp(ipValue: string | undefined): boolean {
  if (!ipValue) {
    return false;
  }

  const normalized = ipValue.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  if (normalized === '::1') {
    return true;
  }

  if (normalized === '::ffff:127.0.0.1') {
    return true;
  }

  return normalized.startsWith('127.');
}

function readClientIp(request: FastifyRequest): string {
  return request.ip;
}

function parseTimestampToMs(value: string): number | undefined {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    if (numeric > 1e12) {
      return Math.trunc(numeric);
    }

    if (numeric > 0) {
      return Math.trunc(numeric * 1000);
    }
  }

  const parsed = Date.parse(value);
  if (Number.isFinite(parsed)) {
    return parsed;
  }

  return undefined;
}

function sanitizePath(url: string): string {
  try {
    const parsed = new URL(url, 'http://gateway.local');
    return parsed.pathname;
  } catch {
    return url.split('?')[0] || url;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeSignature(value: string): string {
  const lower = value.trim().toLowerCase();
  if (lower.startsWith('sha256=')) {
    return lower.slice('sha256='.length).trim();
  }

  return lower;
}

function isHexSha256(value: string): boolean {
  return /^[0-9a-f]{64}$/i.test(value);
}

function timingSafeHexEqual(left: string, right: string): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const leftBuffer = Buffer.from(left, 'utf8');
  const rightBuffer = Buffer.from(right, 'utf8');
  return timingSafeEqual(leftBuffer, rightBuffer);
}

function timingSafeStringEqual(left: string, right: string): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return timingSafeEqual(Buffer.from(left, 'utf8'), Buffer.from(right, 'utf8'));
}
