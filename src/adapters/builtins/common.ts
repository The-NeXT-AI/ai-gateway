import type { FastifyRequest } from 'fastify';
import type { GatewayConfig, HeaderBag, Result } from '../../types';
import { err, ok } from '../../types';
import { asNumber, isObject, readBearerToken, readHeader } from '../../utils';

const defaultAnthropicVersion = '2023-06-01';
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
  const query = new URLSearchParams(incomingUrl.search);

  const keyFromQuery = query.get('key');
  const key = keyFromQuery || config.geminiApiKey || process.env.GEMINI_API_KEY;
  if (!key) {
    return err('GEMINI_API_KEY is missing.');
  }

  query.set('key', key);
  query.delete('target_provider');

  const path = `${config.geminiBaseUrl}/${apiVersion}/models/${encodeURIComponent(model)}:${action}`;
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
  return Boolean(config.auth?.enabled && config.auth?.mode === 'http_introspection');
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

  return {
    input_tokens: inputTokens,
    input_tokens_details: {
      ...inputDetails,
      cached_tokens:
        asTokenCount(inputDetails.cached_tokens) ??
        asTokenCount(usage.cache_read_input_tokens) ??
        asTokenCount(usage.cache_read_tokens) ??
        0
    },
    output_tokens: outputTokens,
    output_tokens_details: {
      ...outputDetails,
      reasoning_tokens:
        asTokenCount(outputDetails.reasoning_tokens) ??
        asTokenCount(usage.reasoning_tokens) ??
        0
    },
    total_tokens: totalTokens
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
