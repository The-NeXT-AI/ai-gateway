import type { FastifyRequest } from 'fastify';
import type { GatewayConfig, HeaderBag, Result } from '../../types';
import { err, ok } from '../../types';
import { asNumber, isObject, readBearerToken, readHeader } from '../../utils';

const defaultAnthropicVersion = '2023-06-01';
const geminiPassthroughQueryParams = new Set(['alt', 'fields', 'prettyPrint', 'quotaUser', 'userIp']);
type OpenAIHeaderBuildConfig = Pick<GatewayConfig, 'openaiApiKey' | 'auth'> & {
  allowEnvApiKeyFallback?: boolean;
};

export function buildOpenAIHeaders(
  headers: HeaderBag,
  config: OpenAIHeaderBuildConfig
): Result<Record<string, string>> {
  const bearer = readBearerToken(readHeader(headers.authorization));
  const fromApiKeyHeader = readHeader(headers['x-api-key']) || readHeader(headers['api-key']);
  const managedApiKey =
    config.openaiApiKey || (config.allowEnvApiKeyFallback === false ? undefined : process.env.OPENAI_API_KEY);
  const shouldPreferManaged = shouldPreferManagedCredential(config);
  const apiKey = shouldPreferManaged
    ? managedApiKey || bearer || fromApiKeyHeader
    : bearer || fromApiKeyHeader || managedApiKey;
  if (!apiKey) {
    return err('OPENAI_API_KEY is missing.');
  }

  const mapped: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${apiKey}`
  };

  const organization = readHeader(headers['openai-organization']);
  if (organization) {
    mapped['openai-organization'] = organization;
  }

  const project = readHeader(headers['openai-project']);
  if (project) {
    mapped['openai-project'] = project;
  }

  return ok(mapped);
}

export function buildAnthropicHeaders(
  headers: HeaderBag,
  config: Pick<GatewayConfig, 'anthropicApiKey' | 'auth'>
): Result<Record<string, string>> {
  const fromHeader = readHeader(headers['x-api-key']);
  const fromBearer = readBearerToken(readHeader(headers.authorization));
  const managedApiKey = config.anthropicApiKey || process.env.ANTHROPIC_API_KEY;
  const shouldPreferManaged = shouldPreferManagedCredential(config);
  const apiKey = shouldPreferManaged
    ? managedApiKey || fromHeader || fromBearer
    : fromHeader || fromBearer || managedApiKey;
  if (!apiKey) {
    return err('ANTHROPIC_API_KEY is missing.');
  }

  const mapped: Record<string, string> = {
    'content-type': 'application/json',
    'x-api-key': apiKey,
    // OpenAI-compatible-hosted Anthropic endpoints (e.g. Ollama Cloud) reject
    // x-api-key and require the Bearer form. Anthropic's own API accepts both,
    // so send both.
    authorization: `Bearer ${apiKey}`,
    'anthropic-version':
      readHeader(headers['anthropic-version']) ||
      readHeader(headers['x-anthropic-version']) ||
      process.env.ANTHROPIC_VERSION ||
      defaultAnthropicVersion
  };

  const beta = readHeader(headers['anthropic-beta']);
  if (beta) {
    mapped['anthropic-beta'] = beta;
  }

  const userAgent = readHeader(headers['user-agent']);
  if (userAgent) {
    mapped['user-agent'] = userAgent;
  }

  return ok(mapped);
}

export function buildGeminiUrl(
  request: FastifyRequest,
  model: string,
  action: 'generateContent' | 'streamGenerateContent',
  apiVersion: string,
  config: GatewayConfig
): Result<string> {
  const incomingUrl = new URL(request.url, 'http://gateway.local');
  const incomingQuery = new URLSearchParams(incomingUrl.search);
  const query = new URLSearchParams();

  const keyFromQuery = incomingQuery.get('key');
  const key = keyFromQuery || config.geminiApiKey || process.env.GEMINI_API_KEY;
  if (!key) {
    return err('GEMINI_API_KEY is missing.');
  }

  for (const [name, value] of incomingQuery.entries()) {
    if (geminiPassthroughQueryParams.has(name)) {
      query.set(name, value);
    }
  }
  query.set('key', key);

  const path = `${config.geminiBaseUrl}/${apiVersion}/models/${encodeURIComponent(model)}:${action}`;
  const q = query.toString();
  return ok(q ? `${path}?${q}` : path);
}

export function buildGeminiInteractionsUrl(
  request: FastifyRequest,
  apiVersion: string,
  config: GatewayConfig
): Result<string> {
  const incomingUrl = new URL(request.url, 'http://gateway.local');
  const incomingQuery = new URLSearchParams(incomingUrl.search);
  const query = new URLSearchParams();

  const keyFromQuery = incomingQuery.get('key');
  const key = keyFromQuery || config.geminiApiKey || process.env.GEMINI_API_KEY;
  if (!key) {
    return err('GEMINI_API_KEY is missing.');
  }

  for (const [name, value] of incomingQuery.entries()) {
    if (geminiPassthroughQueryParams.has(name)) {
      query.set(name, value);
    }
  }
  query.set('key', key);

  const path = `${config.geminiBaseUrl}/${apiVersion}/interactions`;
  const q = query.toString();
  return ok(q ? `${path}?${q}` : path);
}

export function mapFinishReasonToOpenAI(reason?: string): string {
  if (!reason) {
    return 'stop';
  }

  const normalized = reason.toLowerCase();
  if (normalized.includes('max') || normalized.includes('length')) {
    return 'length';
  }

  if (normalized.includes('tool')) {
    return 'tool_calls';
  }

  return 'stop';
}

function shouldPreferManagedCredential(
  config: Pick<GatewayConfig, 'auth'>
): boolean {
  return Boolean(
    config.auth?.enabled &&
      (config.auth?.mode === 'http_introspection' || config.auth?.mode === 'static_api_key')
  );
}

export function mapFinishReasonToAnthropic(reason?: string): string {
  if (!reason) {
    return 'end_turn';
  }

  const normalized = reason.toLowerCase();
  if (normalized.includes('max') || normalized.includes('length')) {
    return 'max_tokens';
  }

  if (normalized.includes('tool')) {
    return 'tool_use';
  }

  return 'end_turn';
}

export function mapFinishReasonToGemini(reason?: string): string {
  if (!reason) {
    return 'STOP';
  }

  const normalized = reason.toLowerCase();
  if (normalized.includes('max') || normalized.includes('length')) {
    return 'MAX_TOKENS';
  }

  return 'STOP';
}

export function normalizeOpenAIResponsesUsage(usageRaw: unknown): Record<string, unknown> {
  const usage = isPlainRecord(usageRaw) ? usageRaw : {};
  const totalTokensRaw = asTokenCount(usage.total_tokens);
  const inputTokens = asTokenCount(usage.input_tokens) ?? asTokenCount(usage.prompt_tokens) ?? 0;
  const outputTokens =
    asTokenCount(usage.output_tokens) ??
    asTokenCount(usage.completion_tokens) ??
    (totalTokensRaw !== undefined ? Math.max(0, totalTokensRaw - inputTokens) : 0);
  const totalTokens = totalTokensRaw ?? inputTokens + outputTokens;
  const inputDetails = isPlainRecord(usage.input_tokens_details)
    ? usage.input_tokens_details
    : isPlainRecord(usage.prompt_tokens_details)
      ? usage.prompt_tokens_details
      : {};
  const outputDetails = isPlainRecord(usage.output_tokens_details)
    ? usage.output_tokens_details
    : isPlainRecord(usage.completion_tokens_details)
      ? usage.completion_tokens_details
      : {};
  const cacheCreationTokens =
    asTokenCount(inputDetails.cache_write_tokens) ??
    asTokenCount(inputDetails.cache_creation_tokens) ??
    asTokenCount(usage.cache_creation_input_tokens) ??
    asTokenCount(usage.cache_creation_tokens) ??
    asTokenCount(usage.cache_write_tokens);
  const serverToolUse = normalizeServerToolUse(usage.server_tool_use);

  return {
    input_tokens: inputTokens,
    input_tokens_details: {
      ...inputDetails,
      cached_tokens:
        asTokenCount(inputDetails.cached_tokens) ??
        asTokenCount(usage.cache_read_input_tokens) ??
        asTokenCount(usage.cache_read_tokens) ??
        0,
      ...(cacheCreationTokens !== undefined
        ? { cache_write_tokens: cacheCreationTokens, cache_creation_tokens: cacheCreationTokens }
        : {})
    },
    output_tokens: outputTokens,
    output_tokens_details: {
      ...outputDetails,
      reasoning_tokens:
        asTokenCount(outputDetails.reasoning_tokens) ??
        asTokenCount(usage.reasoning_tokens) ??
        0
    },
    total_tokens: totalTokens,
    ...(serverToolUse ? { server_tool_use: serverToolUse } : {})
  };
}

export function normalizeOpenAIResponsesCompletedResponse(
  response: Record<string, unknown>
): Record<string, unknown> {
  return {
    ...response,
    usage: normalizeOpenAIResponsesUsage(response.usage)
  };
}

export function normalizeOpenAIResponsesCompletedEventPayload(
  payload: Record<string, unknown>
): Record<string, unknown> {
  if (payload.type !== 'response.completed' || !isPlainRecord(payload.response)) {
    return payload;
  }

  return {
    ...payload,
    response: normalizeOpenAIResponsesCompletedResponse(payload.response)
  };
}

function asTokenCount(value: unknown): number | undefined {
  const numeric = asNumber(value);
  if (numeric === undefined) {
    return undefined;
  }

  return Math.max(0, Math.trunc(numeric));
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return isObject(value) && !Array.isArray(value);
}

function normalizeServerToolUse(value: unknown): Record<string, number> | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const serverToolUse: Record<string, number> = {};
  const webSearchRequests = asTokenCount(value.web_search_requests);
  if (webSearchRequests !== undefined) {
    serverToolUse.web_search_requests = webSearchRequests;
  }

  const webFetchRequests = asTokenCount(value.web_fetch_requests);
  if (webFetchRequests !== undefined) {
    serverToolUse.web_fetch_requests = webFetchRequests;
  }

  return Object.keys(serverToolUse).length > 0 ? serverToolUse : undefined;
}
