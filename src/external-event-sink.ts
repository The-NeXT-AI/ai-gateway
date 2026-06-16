import { spawn } from 'node:child_process';
import WebSocket from 'ws';
import { invokeGrpcJsonUnary } from './grpc-json';

export type ExternalEventSinkTransport = 'http' | 'websocket' | 'grpc' | 'stdio';

export interface ExternalJsonEventSinkConfig {
  transport: ExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  requireAck?: boolean;
  headers: Record<string, string>;
}

export async function publishJsonEventToExternalSink(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  if (!hasConfiguredSinkTarget(config)) {
    return false;
  }

  const maxAttempts = normalizeMaxAttempts(config.maxAttempts);
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await publishJsonEventToExternalSinkOnce(event, config);
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts) {
        break;
      }
      await sleep(calculateRetryDelayMs(config, attempt));
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

function hasConfiguredSinkTarget(config: ExternalJsonEventSinkConfig): boolean {
  if (config.transport === 'stdio') {
    return Boolean(config.command?.trim());
  }
  return Boolean(config.endpoint?.trim());
}

async function publishJsonEventToExternalSinkOnce(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  if (config.transport === 'websocket') {
    return publishJsonEventToWebSocket(event, config);
  }
  if (config.transport === 'grpc') {
    return publishJsonEventToGrpc(event, config);
  }
  if (config.transport === 'stdio') {
    return publishJsonEventToStdio(event, config);
  }

  return publishJsonEventToHttp(event, config);
}

async function publishJsonEventToGrpc(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  await invokeGrpcJsonUnary({
    endpoint: config.endpoint as string,
    defaultPath: '/gateway.events.v1.EventSink/Publish',
    payload: event,
    timeoutMs: config.timeoutMs,
    headers: config.headers
  });
  return true;
}

async function publishJsonEventToHttp(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), normalizeTimeoutMs(config.timeoutMs));

  try {
    const response = await fetch(config.endpoint as string, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...config.headers
      },
      body: JSON.stringify(event),
      signal: controller.signal
    });

    if (!response.ok) {
      const payload = await safeReadResponsePayload(response);
      throw new Error(
        `HTTP event sink request failed with status ${response.status}${payload ? `: ${payload}` : ''}`
      );
    }

    return true;
  } catch (error) {
    if (isAbortError(error)) {
      throw new Error(`HTTP event sink request timeout after ${normalizeTimeoutMs(config.timeoutMs)}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function publishJsonEventToWebSocket(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  const endpoint = config.endpoint as string;
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeoutMs = normalizeTimeoutMs(config.timeoutMs);
    const socket = new WebSocket(endpoint, {
      headers: config.headers
    });
    const timer = setTimeout(() => {
      finish(new Error(`WebSocket event sink request timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    const finish = (error?: Error) => {
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
        socket.close(1000, 'event delivered');
      }
      if (error) {
        reject(error);
        return;
      }
      resolve(true);
    };

    socket.once('open', () => {
      socket.send(JSON.stringify(event), (error) => {
        if (error) {
          finish(error instanceof Error ? error : new Error(String(error)));
          return;
        }
        if (!config.requireAck) {
          finish();
        }
      });
    });
    socket.once('message', (data) => {
      try {
        validateWebSocketAck(data.toString());
        finish();
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    });
    socket.once('error', (error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
    socket.once('close', (code, reason) => {
      if (!settled) {
        const details = reason.length > 0 ? `: ${reason.toString()}` : '';
        finish(new Error(`WebSocket event sink closed before delivery with code ${code}${details}`));
      }
    });
  });
}

function validateWebSocketAck(payload: string): void {
  const trimmed = payload.trim();
  if (!trimmed) {
    return;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return;
  }

  if (!parsed || typeof parsed !== 'object') {
    return;
  }

  const ack = parsed as { ok?: unknown; success?: unknown; error?: unknown; message?: unknown };
  if (ack.ok === false || ack.success === false) {
    const message =
      typeof ack.error === 'string'
        ? ack.error
        : typeof ack.message === 'string'
          ? ack.message
          : 'negative acknowledgement';
    throw new Error(`WebSocket event sink rejected event: ${message}`);
  }
}

async function publishJsonEventToStdio(
  event: unknown,
  config: ExternalJsonEventSinkConfig
): Promise<boolean> {
  const command = config.command?.trim();
  if (!command) {
    return false;
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let stderr = '';
    const timeoutMs = normalizeTimeoutMs(config.timeoutMs);
    const child = spawn(command, config.args || [], {
      cwd: config.cwd,
      env: {
        ...process.env,
        ...(config.env || {})
      },
      stdio: ['pipe', 'ignore', 'pipe']
    });
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      finish(new Error(`Stdio event sink request timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    const finish = (error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (error) {
        reject(error);
        return;
      }
      resolve(true);
    };

    child.stderr?.on('data', (chunk) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-500);
    });
    child.once('error', (error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
    child.once('exit', (code, signal) => {
      if (code === 0) {
        finish();
        return;
      }

      const suffix = stderr.trim() ? `: ${stderr.trim()}` : '';
      finish(new Error(`Stdio event sink exited with code ${code ?? 'null'} signal ${signal ?? 'null'}${suffix}`));
    });

    child.stdin?.once('error', (error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
    child.stdin?.end(`${JSON.stringify(event)}\n`);
  });
}

async function safeReadResponsePayload(response: Response): Promise<string | undefined> {
  try {
    const payload = await response.text();
    const normalized = payload.trim();
    if (!normalized) {
      return undefined;
    }

    if (normalized.length <= 300) {
      return normalized;
    }

    return `${normalized.slice(0, 300)}...`;
  } catch {
    return undefined;
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function normalizeTimeoutMs(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : 5000;
}

function normalizeMaxAttempts(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? Math.trunc(value) : 3;
}

function calculateRetryDelayMs(config: ExternalJsonEventSinkConfig, failedAttempt: number): number {
  const baseDelayMs = normalizeDelayMs(config.baseDelayMs, 200);
  const maxDelayMs = Math.max(baseDelayMs, normalizeDelayMs(config.maxDelayMs, 2000));
  return Math.min(maxDelayMs, baseDelayMs * 2 ** Math.max(0, failedAttempt - 1));
}

function normalizeDelayMs(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value >= 0 ? Math.trunc(value) : fallback;
}

async function sleep(delayMs: number): Promise<void> {
  if (delayMs <= 0) {
    return;
  }

  await new Promise<void>((resolve) => {
    setTimeout(resolve, delayMs);
  });
}
