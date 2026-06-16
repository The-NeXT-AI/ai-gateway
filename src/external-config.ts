import { spawn } from 'node:child_process';
import WebSocket from 'ws';
import {
  applyGatewayConfigInPlace,
  parseGatewayConfigFromRaw
} from './config';
import type {
  GatewayConfig,
  GatewayConfigExternalSourceConfig
} from './types';
import { invokeGrpcJsonUnary } from './grpc-json';
import { readUpstreamPayload } from './upstream/client';
import { isObject } from './utils';

export interface GatewayExternalConfigLogger {
  info?: (...args: unknown[]) => void;
  warn?: (...args: unknown[]) => void;
}

export interface GatewayExternalConfigRefreshOptions {
  config: GatewayConfig;
  logger?: GatewayExternalConfigLogger;
  onConfigReload?: (config: GatewayConfig, reason: string) => Promise<void> | void;
  reason?: string;
}

export interface GatewayExternalConfigPollerOptions extends GatewayExternalConfigRefreshOptions {}

export interface GatewayExternalConfigPoller {
  close(): void;
}

const DEFAULT_TIMEOUT_MS = 5000;

export function isGatewayExternalConfigEnabled(config: GatewayConfig): boolean {
  return Boolean(config.configExternal?.enabled);
}

export async function refreshGatewayConfigFromExternalSource(
  options: GatewayExternalConfigRefreshOptions
): Promise<boolean> {
  const source = options.config.configExternal;
  if (!source?.enabled) {
    return false;
  }

  const payload = await fetchGatewayConfigPayload(source);
  const configPayload = extractGatewayConfigPayload(payload);
  const nextConfig = parseGatewayConfigFromRaw(configPayload);
  if (!hasGatewayConfigExternalSource(configPayload)) {
    nextConfig.configExternal = cloneGatewayConfigExternalSource(source);
  }
  applyGatewayConfigInPlace(options.config, nextConfig);
  await options.onConfigReload?.(options.config, options.reason || 'external_config_refresh');
  options.logger?.info?.(
    {
      endpoint: source.endpoint,
      command: source.command,
      transport: source.transport,
      method: source.method,
      reason: options.reason || 'external_config_refresh'
    },
    'Loaded gateway config from external endpoint.'
  );
  return true;
}

function cloneGatewayConfigExternalSource(
  source: GatewayConfigExternalSourceConfig
): GatewayConfigExternalSourceConfig {
  return {
    ...source,
    headers: {
      ...source.headers
    }
  };
}

export function startGatewayExternalConfigPoller(
  options: GatewayExternalConfigPollerOptions
): GatewayExternalConfigPoller | undefined {
  const intervalMs = normalizeIntervalMs(options.config.configExternal?.intervalMs);
  if (!options.config.configExternal?.enabled || intervalMs <= 0) {
    return undefined;
  }

  const timer = setInterval(() => {
    void refreshGatewayConfigFromExternalSource({
      ...options,
      reason: 'external_config_poll'
    }).catch((error) => {
      options.logger?.warn?.(
        {
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to refresh gateway config from external endpoint.'
      );
    });
  }, intervalMs);
  timer.unref?.();

  return {
    close: () => {
      clearInterval(timer);
    }
  };
}

async function fetchGatewayConfigPayload(source: GatewayConfigExternalSourceConfig): Promise<unknown> {
  if (source.transport === 'websocket') {
    return fetchGatewayConfigPayloadFromWebSocket(source);
  }
  if (source.transport === 'grpc') {
    return fetchGatewayConfigPayloadFromGrpc(source);
  }
  if (source.transport === 'stdio') {
    return fetchGatewayConfigPayloadFromStdio(source);
  }

  return fetchGatewayConfigPayloadFromHttp(source);
}

async function fetchGatewayConfigPayloadFromGrpc(source: GatewayConfigExternalSourceConfig): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error('configExternal.endpoint is required when configExternal.enabled=true.');
  }

  const headers: Record<string, string> = {
    ...source.headers
  };
  if (source.apiKey) {
    headers[source.apiKeyHeader] = source.apiKey;
  }

  const response = await invokeGrpcJsonUnary({
    endpoint,
    defaultPath: '/gateway.config.v1.ConfigService/GetConfig',
    payload: { type: 'gateway_config_request' },
    timeoutMs: source.timeoutMs,
    headers
  });

  return response.payload;
}

async function fetchGatewayConfigPayloadFromHttp(source: GatewayConfigExternalSourceConfig): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error('configExternal.endpoint is required when configExternal.enabled=true.');
  }

  const headers: Record<string, string> = {
    accept: 'application/json',
    ...source.headers
  };
  if (source.method === 'POST' && !hasHeader(headers, 'content-type')) {
    headers['content-type'] = 'application/json';
  }
  if (source.apiKey) {
    headers[source.apiKeyHeader] = source.apiKey;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, normalizeTimeoutMs(source.timeoutMs));

  try {
    const response = await fetch(endpoint, {
      method: source.method,
      headers,
      body: source.method === 'POST' ? JSON.stringify({ type: 'gateway_config_request' }) : undefined,
      signal: controller.signal
    });
    const payload = await readUpstreamPayload(response);

    if (!response.ok) {
      const details = summarizeErrorPayload(payload);
      throw new Error(
        details
          ? `External gateway config endpoint returned ${response.status}: ${details}`
          : `External gateway config endpoint returned ${response.status}.`
      );
    }

    return payload;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`External gateway config request timeout after ${normalizeTimeoutMs(source.timeoutMs)}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchGatewayConfigPayloadFromWebSocket(
  source: GatewayConfigExternalSourceConfig
): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error('configExternal.endpoint is required when configExternal.enabled=true.');
  }

  const headers: Record<string, string> = {
    accept: 'application/json',
    ...source.headers
  };
  if (source.apiKey) {
    headers[source.apiKeyHeader] = source.apiKey;
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeoutMs = normalizeTimeoutMs(source.timeoutMs);
    const socket = new WebSocket(endpoint, { headers });
    const timer = setTimeout(() => {
      finish(undefined, new Error(`External gateway config websocket timeout after ${timeoutMs}ms.`));
    }, timeoutMs);

    const finish = (payload?: unknown, error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      socket.removeAllListeners();
      if (
        socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close(1000, 'config received');
      }
      if (error) {
        reject(error);
        return;
      }
      resolve(payload);
    };

    socket.once('open', () => {
      socket.send(JSON.stringify({ type: 'gateway_config_request' }), (error) => {
        if (error) {
          finish(undefined, error instanceof Error ? error : new Error(String(error)));
        }
      });
    });
    socket.once('message', (data) => {
      try {
        finish(parseJsonPayload(data.toString(), 'External gateway config websocket payload'));
      } catch (error) {
        finish(undefined, error instanceof Error ? error : new Error(String(error)));
      }
    });
    socket.once('error', (error) => {
      finish(undefined, error instanceof Error ? error : new Error(String(error)));
    });
    socket.once('close', (code, reason) => {
      if (!settled) {
        const details = reason.length > 0 ? `: ${reason.toString()}` : '';
        finish(undefined, new Error(`External gateway config websocket closed before payload with code ${code}${details}`));
      }
    });
  });
}

async function fetchGatewayConfigPayloadFromStdio(
  source: GatewayConfigExternalSourceConfig
): Promise<unknown> {
  const command = normalizeString(source.command);
  if (!command) {
    throw new Error('configExternal.command is required when configExternal.transport=stdio.');
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let stdout = '';
    let stderr = '';
    const timeoutMs = normalizeTimeoutMs(source.timeoutMs);
    const child = spawn(command, source.args || [], {
      cwd: source.cwd,
      env: {
        ...process.env,
        ...(source.env || {})
      },
      stdio: ['pipe', 'pipe', 'pipe']
    });
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      finish(undefined, new Error(`External gateway config stdio timeout after ${timeoutMs}ms.`));
    }, timeoutMs);

    const finish = (payload?: unknown, error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (error) {
        reject(error);
        return;
      }
      resolve(payload);
    };

    child.stdout?.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr?.on('data', (chunk) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-500);
    });
    child.once('error', (error) => {
      finish(undefined, error instanceof Error ? error : new Error(String(error)));
    });
    child.once('exit', (code, signal) => {
      if (code === 0) {
        try {
          finish(parseJsonPayload(stdout, 'External gateway config stdio payload'));
        } catch (error) {
          finish(undefined, error instanceof Error ? error : new Error(String(error)));
        }
        return;
      }

      const suffix = stderr.trim() ? `: ${stderr.trim()}` : '';
      finish(undefined, new Error(`External gateway config stdio exited with code ${code ?? 'null'} signal ${signal ?? 'null'}${suffix}`));
    });

    child.stdin?.once('error', (error) => {
      finish(undefined, error instanceof Error ? error : new Error(String(error)));
    });
    child.stdin?.end(`${JSON.stringify({ type: 'gateway_config_request' })}\n`);
  });
}

function extractGatewayConfigPayload(payload: unknown): unknown {
  if (!isObject(payload)) {
    throw new Error('External gateway config endpoint payload must be a JSON object.');
  }

  const source = payload as Record<string, unknown>;
  if (isObject(source.config)) {
    return source.config;
  }
  if (isObject(source.gatewayConfig)) {
    return source.gatewayConfig;
  }

  return payload;
}

function hasGatewayConfigExternalSource(payload: unknown): boolean {
  if (!isObject(payload)) {
    return false;
  }

  const source = payload as Record<string, unknown>;
  return (
    Object.prototype.hasOwnProperty.call(source, 'configExternal') ||
    Object.prototype.hasOwnProperty.call(source, 'externalConfig')
  );
}

function summarizeErrorPayload(payload: unknown): string | undefined {
  if (typeof payload === 'string') {
    return payload.slice(0, 500);
  }

  if (isObject(payload)) {
    const source = payload as Record<string, unknown>;
    const message = source.error || source.message;
    if (typeof message === 'string') {
      return message.slice(0, 500);
    }

    try {
      return JSON.stringify(payload).slice(0, 500);
    } catch {
      return undefined;
    }
  }

  return undefined;
}

function parseJsonPayload(payload: string, label: string): unknown {
  const normalized = payload.trim();
  if (!normalized) {
    throw new Error(`${label} must not be empty.`);
  }

  try {
    return JSON.parse(normalized);
  } catch (error) {
    const details = error instanceof Error ? error.message : String(error);
    throw new Error(`${label} must be valid JSON: ${details}`);
  }
}

function hasHeader(headers: Record<string, string>, name: string): boolean {
  const normalized = name.toLowerCase();
  return Object.keys(headers).some((key) => key.toLowerCase() === normalized);
}

function normalizeString(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized || undefined;
}

function normalizeTimeoutMs(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : DEFAULT_TIMEOUT_MS;
}

function normalizeIntervalMs(value: number | undefined): number {
  return Number.isFinite(value) && value !== undefined && value > 0 ? Math.trunc(value) : 0;
}
