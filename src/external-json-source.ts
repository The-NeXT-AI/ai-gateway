import { spawn } from 'node:child_process';
import WebSocket from 'ws';
import { invokeGrpcJsonUnary } from './grpc-json';
import { readUpstreamPayload } from './upstream/client';
import type { GatewayExternalEventSinkTransport } from './types';

export interface ExternalJsonSourceConfig {
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  apiKeyHeader: string;
  apiKey?: string;
  headers: Record<string, string>;
}

export interface ExternalJsonRequestOptions {
  label: string;
  httpMethod?: 'GET' | 'POST';
  payload?: unknown;
  grpcDefaultPath: string;
}

export async function requestExternalJson(
  source: ExternalJsonSourceConfig,
  options: ExternalJsonRequestOptions
): Promise<unknown> {
  if (source.transport === 'websocket') {
    return requestExternalJsonFromWebSocket(source, options);
  }
  if (source.transport === 'grpc') {
    return requestExternalJsonFromGrpc(source, options);
  }
  if (source.transport === 'stdio') {
    return requestExternalJsonFromStdio(source, options);
  }

  return requestExternalJsonFromHttp(source, options);
}

async function requestExternalJsonFromHttp(
  source: ExternalJsonSourceConfig,
  options: ExternalJsonRequestOptions
): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error(`${options.label}.endpoint is required when ${options.label}.enabled=true.`);
  }

  const method = options.httpMethod || (options.payload === undefined ? 'GET' : 'POST');
  const headers = buildHeaders(source, method === 'GET' ? undefined : 'application/json');
  const controller = new AbortController();
  const timeoutMs = normalizeTimeoutMs(source.timeoutMs);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method,
      headers,
      body: method === 'GET' ? undefined : JSON.stringify(options.payload ?? {}),
      signal: controller.signal
    });
    const payload = await readUpstreamPayload(response);
    if (!response.ok) {
      const details = summarizePayload(payload);
      throw new Error(
        details
          ? `${options.label} endpoint returned ${response.status}: ${details}`
          : `${options.label} endpoint returned ${response.status}.`
      );
    }
    return payload;
  } catch (error) {
    if (isAbortError(error)) {
      throw new Error(`${options.label} endpoint request timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function requestExternalJsonFromWebSocket(
  source: ExternalJsonSourceConfig,
  options: ExternalJsonRequestOptions
): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error(`${options.label}.endpoint is required when ${options.label}.enabled=true.`);
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeoutMs = normalizeTimeoutMs(source.timeoutMs);
    const socket = new WebSocket(endpoint, {
      headers: buildHeaders(source)
    });
    const timer = setTimeout(() => {
      finish(undefined, new Error(`${options.label} websocket request timed out after ${timeoutMs}ms.`));
    }, timeoutMs);

    const finish = (payload?: unknown, error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      socket.removeAllListeners();
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close(1000, 'json response received');
      }
      if (error) {
        reject(error);
        return;
      }
      resolve(payload);
    };

    socket.once('open', () => {
      socket.send(JSON.stringify(options.payload ?? { type: `${options.label}_request` }), (error) => {
        if (error) {
          finish(undefined, error instanceof Error ? error : new Error(String(error)));
        }
      });
    });
    socket.once('message', (data) => {
      try {
        finish(parseJsonPayload(data.toString(), `${options.label} websocket payload`));
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
        finish(undefined, new Error(`${options.label} websocket closed before payload with code ${code}${details}`));
      }
    });
  });
}

async function requestExternalJsonFromGrpc(
  source: ExternalJsonSourceConfig,
  options: ExternalJsonRequestOptions
): Promise<unknown> {
  const endpoint = normalizeString(source.endpoint);
  if (!endpoint) {
    throw new Error(`${options.label}.endpoint is required when ${options.label}.enabled=true.`);
  }

  const response = await invokeGrpcJsonUnary({
    endpoint,
    defaultPath: options.grpcDefaultPath,
    payload: options.payload ?? { type: `${options.label}_request` },
    timeoutMs: source.timeoutMs,
    headers: buildHeaders(source)
  });
  return response.payload;
}

async function requestExternalJsonFromStdio(
  source: ExternalJsonSourceConfig,
  options: ExternalJsonRequestOptions
): Promise<unknown> {
  const command = normalizeString(source.command);
  if (!command) {
    throw new Error(`${options.label}.command is required when ${options.label}.transport=stdio.`);
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
      finish(undefined, new Error(`${options.label} stdio request timed out after ${timeoutMs}ms.`));
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
          finish(parseJsonPayload(stdout, `${options.label} stdio payload`));
        } catch (error) {
          finish(undefined, error instanceof Error ? error : new Error(String(error)));
        }
        return;
      }

      const suffix = stderr.trim() ? `: ${stderr.trim()}` : '';
      finish(undefined, new Error(`${options.label} stdio exited with code ${code ?? 'null'} signal ${signal ?? 'null'}${suffix}`));
    });

    child.stdin?.once('error', (error) => {
      finish(undefined, error instanceof Error ? error : new Error(String(error)));
    });
    child.stdin?.end(`${JSON.stringify(options.payload ?? { type: `${options.label}_request` })}\n`);
  });
}

function buildHeaders(source: ExternalJsonSourceConfig, contentType?: string): Record<string, string> {
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...source.headers
  };
  if (contentType) {
    headers['content-type'] = contentType;
  }
  if (source.apiKey) {
    headers[source.apiKeyHeader] = source.apiKey;
  }
  return headers;
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

function summarizePayload(payload: unknown): string | undefined {
  if (typeof payload === 'string') {
    return payload.slice(0, 500);
  }
  if (payload && typeof payload === 'object') {
    try {
      return JSON.stringify(payload).slice(0, 500);
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function normalizeString(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized || undefined;
}

function normalizeTimeoutMs(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : 5000;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}
