import type { GatewayConfig, Provider, ProviderConfig } from '../types';
import { providerFromProviderType, trimTrailingSlash } from '../utils';
import { recordProviderHealthFailure, recordProviderHealthResponse } from './provider-health';

export interface ProviderHealthCheckResult {
  provider: Provider;
  providerName: string;
  ok: boolean;
  statusCode?: number;
  latencyMs: number;
  checkedAt: string;
  health: ProviderConfig['health'];
  endpoint?: string;
  error?: string;
}

interface HealthCheckRequest {
  url: string;
  headers: Record<string, string>;
}

interface HealthCheckOptions {
  timeoutMs?: number;
}

export async function checkProviderHealth(
  providerConfig: ProviderConfig,
  config: GatewayConfig,
  options: HealthCheckOptions = {}
): Promise<ProviderHealthCheckResult> {
  const provider = providerFromProviderType(providerConfig.type);
  const requestResult = buildHealthCheckRequest(providerConfig, config, provider);
  const startedAt = Date.now();

  if (!requestResult.ok) {
    recordProviderHealthFailure(providerConfig, 0);
    return {
      provider,
      providerName: providerConfig.name,
      ok: false,
      latencyMs: 0,
      checkedAt: providerConfig.health?.checkedAt || new Date().toISOString(),
      health: providerConfig.health,
      error: requestResult.error
    };
  }

  const timeoutMs = normalizeTimeoutMs(options.timeoutMs ?? config.upstreamTimeoutMs);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(requestResult.value.url, {
      method: 'GET',
      headers: requestResult.value.headers,
      signal: controller.signal
    });
    const latencyMs = Date.now() - startedAt;
    recordProviderHealthResponse(providerConfig, response.status, latencyMs);

    return {
      provider,
      providerName: providerConfig.name,
      ok: response.ok,
      statusCode: response.status,
      latencyMs: providerConfig.health?.latencyMs ?? latencyMs,
      checkedAt: providerConfig.health?.checkedAt || new Date().toISOString(),
      health: providerConfig.health,
      endpoint: requestResult.value.url
    };
  } catch (error) {
    const latencyMs = Date.now() - startedAt;
    recordProviderHealthFailure(providerConfig, latencyMs);
    return {
      provider,
      providerName: providerConfig.name,
      ok: false,
      latencyMs: providerConfig.health?.latencyMs ?? latencyMs,
      checkedAt: providerConfig.health?.checkedAt || new Date().toISOString(),
      health: providerConfig.health,
      endpoint: requestResult.value.url,
      error: error instanceof Error ? error.message : String(error)
    };
  } finally {
    clearTimeout(timeout);
  }
}

function buildHealthCheckRequest(
  providerConfig: ProviderConfig,
  config: GatewayConfig,
  provider: Provider
): { ok: true; value: HealthCheckRequest } | { ok: false; error: string } {
  if (provider === 'openai') {
    const apiKey = providerConfig.apikey || config.openaiApiKey || process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return { ok: false, error: 'OPENAI_API_KEY is missing.' };
    }

    return {
      ok: true,
      value: {
        url: `${trimTrailingSlash(providerConfig.baseurl || config.openaiBaseUrl)}/models`,
        headers: {
          authorization: `Bearer ${apiKey}`
        }
      }
    };
  }

  if (provider === 'anthropic') {
    const apiKey = providerConfig.apikey || config.anthropicApiKey || process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return { ok: false, error: 'ANTHROPIC_API_KEY is missing.' };
    }

    return {
      ok: true,
      value: {
        url: `${trimTrailingSlash(providerConfig.baseurl || config.anthropicBaseUrl)}/v1/models`,
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': process.env.ANTHROPIC_VERSION || '2023-06-01'
        }
      }
    };
  }

  if (provider === 'xai') {
    const apiKey = providerConfig.apikey || process.env.XAI_API_KEY;
    if (!apiKey) {
      return { ok: false, error: 'XAI_API_KEY is missing.' };
    }

    return {
      ok: true,
      value: {
        url: `${trimTrailingSlash(
          providerConfig.baseurl || process.env.XAI_BASE_URL || 'https://api.x.ai/v1'
        )}/models`,
        headers: {
          authorization: `Bearer ${apiKey}`
        }
      }
    };
  }

  const apiKey = providerConfig.apikey || config.geminiApiKey || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return { ok: false, error: 'GEMINI_API_KEY is missing.' };
  }

  const url = new URL(
    `${trimTrailingSlash(providerConfig.baseurl || config.geminiBaseUrl)}/${config.geminiApiVersion}/models`
  );
  url.searchParams.set('key', apiKey);
  return {
    ok: true,
    value: {
      url: url.toString(),
      headers: {}
    }
  };
}

function normalizeTimeoutMs(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 15000;
  }

  return Math.max(1, Math.min(Math.trunc(value), 30000));
}
