import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const undiciMock = vi.hoisted(() => ({
  dispatch: vi.fn(() => true),
  fetch: vi.fn((input: string | URL | Request, init?: RequestInit) => globalThis.fetch(input, init)),
  proxyAgentInputs: [] as unknown[],
  proxyDispatch: vi.fn((_options: Record<string, unknown>, _handler: unknown) => true),
  getGlobalDispatcher: vi.fn()
}));

vi.mock('undici', () => ({
  Dispatcher: class MockDispatcher {
    dispatch(): boolean {
      throw new Error('dispatch not implemented');
    }

    close(callback?: () => void): Promise<void> | void {
      if (callback) {
        callback();
        return;
      }

      return Promise.resolve();
    }

    destroy(errorOrCallback?: Error | null | (() => void), callback?: () => void): Promise<void> | void {
      const done = typeof errorOrCallback === 'function' ? errorOrCallback : callback;
      if (done) {
        done();
        return;
      }

      return Promise.resolve();
    }
  },
  ProxyAgent: class MockProxyAgent {
    constructor(input?: unknown) {
      undiciMock.proxyAgentInputs.push(input);
    }

    dispatch(options: Record<string, unknown>, handler: unknown): boolean {
      return undiciMock.proxyDispatch(options, handler);
    }

    close(callback?: () => void): Promise<void> | void {
      if (callback) {
        callback();
        return;
      }

      return Promise.resolve();
    }

    destroy(errorOrCallback?: Error | null | (() => void), callback?: () => void): Promise<void> | void {
      const done = typeof errorOrCallback === 'function' ? errorOrCallback : callback;
      if (done) {
        done();
        return;
      }

      return Promise.resolve();
    }
  },
  fetch: undiciMock.fetch,
  getGlobalDispatcher: undiciMock.getGlobalDispatcher
}));

import {
  callUpstream,
  cancelResponseBodyOnAbort,
  readUpstreamPayload,
  sanitizeHeadersForLog,
  sanitizePayloadForLog
} from './client';

type FetchInitWithDispatcherForTest = RequestInit & { dispatcher?: unknown };
type TestDispatcher = {
  isMockActive?: unknown;
  dispatch(options: Record<string, unknown>, handler: unknown): boolean;
};

const proxyEnvNames = [
  'CCR_UPSTREAM_PROXY_URL',
  'http_proxy',
  'HTTP_PROXY',
  'https_proxy',
  'HTTPS_PROXY',
  'all_proxy',
  'ALL_PROXY',
  'no_proxy',
  'NO_PROXY'
];
const originalProxyEnv = new Map(proxyEnvNames.map((name) => [name, process.env[name]]));

describe('callUpstream', () => {
  beforeEach(() => {
    clearProxyEnv();
    undiciMock.dispatch.mockClear();
    undiciMock.fetch.mockClear();
    undiciMock.proxyAgentInputs.length = 0;
    undiciMock.proxyDispatch.mockClear();
    undiciMock.getGlobalDispatcher.mockReset();
    undiciMock.getGlobalDispatcher.mockReturnValue({ dispatch: undiciMock.dispatch });
  });

  afterEach(() => {
    restoreProxyEnv();
  });

  it('aborts model upstream requests based on timeoutMs', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;
    let capturedSignal: AbortSignal | undefined;

    try {
      const fetchMock = vi.fn((_url: string | URL | Request, init?: RequestInit) => {
        capturedSignal = init?.signal ?? undefined;
        return new Promise<Response>((_resolve, reject) => {
          capturedSignal?.addEventListener(
            'abort',
            () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
            { once: true }
          );
        });
      });
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        1
      ).catch((error) => error);

      await vi.advanceTimersByTimeAsync(1);
      const error = await pending;

      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Upstream request timed out after 1ms.');
      expect(capturedSignal?.aborted).toBe(true);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('leaves upstream timeout disabled when timeoutMs is zero', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;
    let capturedSignal: AbortSignal | undefined;

    try {
      global.fetch = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
        capturedSignal = init?.signal ?? undefined;
        await new Promise((resolve) => setTimeout(resolve, 10));
        return new Response('{}', { status: 200 });
      }) as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0
      );

      await vi.advanceTimersByTimeAsync(10);
      const response = await pending;

      expect(response.status).toBe(200);
      expect(capturedSignal?.aborted).toBe(false);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('uses undici fetch for dispatcher-carrying requests', async () => {
    const originalFetch = global.fetch;

    try {
      const globalFetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
      global.fetch = globalFetchMock as typeof fetch;
      undiciMock.fetch.mockResolvedValueOnce(new Response('{}', { status: 200 }));

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = undiciMock.fetch.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const sentHeaders = fetchInit?.headers as Record<string, string> | undefined;

      expect(response.status).toBe(200);
      expect(undiciMock.fetch).toHaveBeenCalledTimes(1);
      expect(globalFetchMock).not.toHaveBeenCalled();
      expect(fetchInit?.dispatcher).toBeTruthy();
      expect(sentHeaders?.['accept-encoding']).toBe('br, gzip, deflate');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('applies upstream timeouts through the current global dispatcher', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      const dispatchResult = (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(response.status).toBe(200);
      expect(dispatchResult).toBe(true);
      expect(undiciMock.getGlobalDispatcher).toHaveBeenCalledTimes(1);
      expect(undiciMock.dispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 600000,
          headersTimeout: 600000
        },
        handler
      );
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('applies separate headers and body idle timeouts to the dispatcher', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000,
        undefined,
        undefined,
        undefined,
        { headersTimeoutMs: 1000, bodyIdleTimeoutMs: 2000 }
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(undiciMock.dispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 2000,
          headersTimeout: 1000
        },
        handler
      );
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('falls back to the overall timeout when headers and body idle timeouts are 0', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000,
        undefined,
        undefined,
        undefined,
        { headersTimeoutMs: 0, bodyIdleTimeoutMs: 0 }
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(undiciMock.dispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 600000,
          headersTimeout: 600000
        },
        handler
      );
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('retries an attempt whose response headers miss the headers deadline', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;

    try {
      const firstSignals: AbortSignal[] = [];
      const fetchMock = vi.fn((_url: string | URL | Request, init?: RequestInit) => {
        if (fetchMock.mock.calls.length === 1) {
          const attemptSignal = init?.signal as AbortSignal;
          firstSignals.push(attemptSignal);
          return new Promise<Response>((_resolve, reject) => {
            attemptSignal.addEventListener(
              'abort',
              () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
              { once: true }
            );
          });
        }

        return Promise.resolve(new Response('{"ok":true}', { status: 200 }));
      });
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        10000,
        undefined,
        undefined,
        undefined,
        { headersTimeoutMs: 1000 }
      );

      await vi.advanceTimersByTimeAsync(1000);
      expect(firstSignals[0]?.aborted).toBe(true);
      await vi.advanceTimersByTimeAsync(150);
      const response = await pending;

      expect(response.status).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('does not let headers timeout retries outlive the overall deadline', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn((_url: string | URL | Request, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
            { once: true }
          );
        })
      );
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        1200,
        undefined,
        undefined,
        { maxAttempts: 5 },
        { headersTimeoutMs: 1000 }
      ).catch((error) => error);

      await vi.advanceTimersByTimeAsync(1200);
      const error = await pending;

      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Upstream request timed out after 1200ms.');
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('reports a headers timeout once retries are exhausted', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn((_url: string | URL | Request, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
            { once: true }
          );
        })
      );
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        60000,
        undefined,
        undefined,
        undefined,
        { headersTimeoutMs: 1000 }
      ).catch((error) => error);

      await vi.advanceTimersByTimeAsync(1000 + 150 + 1000);
      const error = await pending;

      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Upstream response headers timed out after 1000ms.');
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('classifies undici headers timeouts wrapped by fetch', async () => {
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn(async () => {
        throw new TypeError('fetch failed', {
          cause: Object.assign(new Error('Headers Timeout Error'), { code: 'UND_ERR_HEADERS_TIMEOUT' })
        });
      });
      global.fetch = fetchMock as typeof fetch;

      const error = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        60000,
        undefined,
        undefined,
        { enabled: false },
        { headersTimeoutMs: 1000 }
      ).catch((caught) => caught);

      expect(error.message).toBe('Upstream response headers timed out after 1000ms.');
      expect(error.cause).toBeInstanceOf(TypeError);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('applies upstream timeouts through a CCR upstream proxy dispatcher', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);
    process.env.CCR_UPSTREAM_PROXY_URL = 'http://127.0.0.1:8888';
    process.env.NO_PROXY = 'localhost,127.0.0.1';

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      const dispatchResult = (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(response.status).toBe(200);
      expect(dispatchResult).toBe(true);
      expect(undiciMock.proxyAgentInputs).toEqual(['http://127.0.0.1:8888']);
      expect(undiciMock.proxyDispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 600000,
          headersTimeout: 600000
        },
        handler
      );
      expect(undiciMock.dispatch).not.toHaveBeenCalled();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('applies the upstream proxy dispatcher when upstream timeout is disabled', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);
    process.env.CCR_UPSTREAM_PROXY_URL = 'http://127.0.0.1:7777';

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      const dispatchResult = (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(response.status).toBe(200);
      expect(dispatchResult).toBe(true);
      expect(undiciMock.proxyAgentInputs).toEqual(['http://127.0.0.1:7777']);
      expect(undiciMock.proxyDispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );
      expect(undiciMock.dispatch).not.toHaveBeenCalled();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('bypasses CCR upstream proxy for NO_PROXY matches while preserving timeouts', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);
    process.env.CCR_UPSTREAM_PROXY_URL = 'http://127.0.0.1:8888';
    process.env.NO_PROXY = '.test';

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      const dispatchResult = (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(response.status).toBe(200);
      expect(dispatchResult).toBe(true);
      expect(undiciMock.proxyAgentInputs).toEqual([]);
      expect(undiciMock.dispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 600000,
          headersTimeout: 600000
        },
        handler
      );
      expect(undiciMock.proxyDispatch).not.toHaveBeenCalled();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('keeps active MockAgent dispatchers ahead of environment proxies', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch, isMockActive: true };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);
    process.env.CCR_UPSTREAM_PROXY_URL = 'http://127.0.0.1:9999';

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const handler = {};
      const dispatchResult = (fetchInit?.dispatcher as TestDispatcher | undefined)?.dispatch(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST'
        },
        handler
      );

      expect(response.status).toBe(200);
      expect(dispatchResult).toBe(true);
      expect((fetchInit?.dispatcher as TestDispatcher | undefined)?.isMockActive).toBe(true);
      expect(undiciMock.proxyAgentInputs).toEqual([]);
      expect(undiciMock.dispatch).toHaveBeenCalledWith(
        {
          origin: 'https://example.test',
          path: '/v1/responses',
          method: 'POST',
          bodyTimeout: 600000,
          headersTimeout: 600000
        },
        handler
      );
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('reflects MockAgent state from the current global dispatcher', async () => {
    const originalFetch = global.fetch;
    const globalDispatcher = { dispatch: undiciMock.dispatch, isMockActive: true };
    undiciMock.getGlobalDispatcher.mockReturnValue(globalDispatcher);

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;

      expect(response.status).toBe(200);
      expect((fetchInit?.dispatcher as TestDispatcher | undefined)?.isMockActive).toBe(true);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('does not pass a dispatcher when timeoutMs disables the application timer', async () => {
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;

      expect(response.status).toBe(200);
      expect(fetchInit?.dispatcher).toBeUndefined();
      expect(undiciMock.fetch).not.toHaveBeenCalled();
      expect(undiciMock.getGlobalDispatcher).not.toHaveBeenCalled();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('sets a supported upstream accept-encoding header when callers do not provide one', async () => {
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const sentHeaders = fetchInit?.headers as Record<string, string> | undefined;

      expect(response.status).toBe(200);
      expect(sentHeaders?.['accept-encoding']).toBe('br, gzip, deflate');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('removes zstd from caller-provided upstream accept-encoding values', async () => {
    const originalFetch = global.fetch;
    const headers = {
      'content-type': 'application/json',
      'Accept-Encoding': 'gzip, zstd, br, x-zstd, *;q=0.5'
    };

    try {
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0], _init?: RequestInit) =>
        new Response('{}', { status: 200 })
      );
      global.fetch = fetchMock as typeof fetch;

      const response = await callUpstream(
        'https://example.test/v1/responses',
        headers,
        { model: 'test-model', input: 'hello' },
        0
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;
      const sentHeaders = fetchInit?.headers as Record<string, string> | undefined;

      expect(response.status).toBe(200);
      expect(sentHeaders?.['accept-encoding']).toBe('gzip, br');
      expect(sentHeaders?.['Accept-Encoding']).toBeUndefined();
      expect(headers['Accept-Encoding']).toBe('gzip, zstd, br, x-zstd, *;q=0.5');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('retries configured upstream response status codes', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify({ error: 'rate limited' }), { status: 429 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }));
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0,
        undefined,
        undefined,
        {
          enabled: true,
          maxAttempts: 2,
          baseDelayMs: 10,
          maxDelayMs: 10,
          retryStatusCodes: [429]
        }
      );

      await vi.advanceTimersByTimeAsync(10);
      const response = await pending;

      expect(response.status).toBe(200);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('stops retry backoff when the external signal aborts', async () => {
    vi.useFakeTimers();
    const originalFetch = global.fetch;
    const controller = new AbortController();

    try {
      const fetchMock = vi.fn(async () => {
        return new Response(JSON.stringify({ error: 'rate limited' }), { status: 429 });
      });
      global.fetch = fetchMock as typeof fetch;

      const pending = callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0,
        controller.signal,
        undefined,
        {
          enabled: true,
          maxAttempts: 2,
          baseDelayMs: 1000,
          maxDelayMs: 1000,
          retryStatusCodes: [429]
        }
      ).catch((error) => error);

      await Promise.resolve();
      await Promise.resolve();
      controller.abort(new Error('client disconnected'));
      const error = await pending;

      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('client disconnected');
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      global.fetch = originalFetch;
      vi.useRealTimers();
    }
  });

  it('does not retry connection errors when retry is disabled', async () => {
    const originalFetch = global.fetch;

    try {
      const fetchMock = vi.fn(async () => {
        throw new Error('network down');
      });
      global.fetch = fetchMock as typeof fetch;

      const error = await callUpstream(
        'https://example.test/v1/responses',
        { 'content-type': 'application/json' },
        { model: 'test-model', input: 'hello' },
        0,
        undefined,
        undefined,
        {
          enabled: false
        }
      ).catch((caught) => caught);

      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('network down');
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('does not clone or buffer binary responses when response body logging is disabled', async () => {
    const originalFetch = global.fetch;
    const response = new Response('mock-video-bytes', {
      headers: { 'content-type': 'video/mp4' }
    });
    const cloneSpy = vi.spyOn(response, 'clone');

    try {
      global.fetch = vi.fn(async () => response) as typeof fetch;
      const result = await callUpstream(
        'https://example.test/v1/videos/video-1/content',
        {},
        undefined,
        0,
        undefined,
        { logger: { info: vi.fn() } },
        undefined,
        { method: 'GET', bodyEncoding: 'none', skipResponseBodyLog: true }
      );

      expect(result).toBe(response);
      expect(cloneSpy).not.toHaveBeenCalled();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('does not clone or buffer event streams when response logging is enabled', async () => {
    const originalFetch = global.fetch;
    const response = new Response('data: partial\n\n', {
      headers: { 'content-type': 'text/event-stream; charset=utf-8' }
    });
    const cloneSpy = vi.spyOn(response, 'clone');

    try {
      global.fetch = vi.fn(async () => response) as typeof fetch;
      const result = await callUpstream(
        'https://example.test/v1/images/edits',
        { 'content-type': 'application/json' },
        { stream: true },
        0,
        undefined,
        { logger: { info: vi.fn() } }
      );

      expect(result).toBe(response);
      expect(cloneSpy).not.toHaveBeenCalled();
      expect(await result.text()).toBe('data: partial\n\n');
    } finally {
      global.fetch = originalFetch;
    }
  });
});

function clearProxyEnv(): void {
  for (const name of proxyEnvNames) {
    delete process.env[name];
  }
}

function restoreProxyEnv(): void {
  for (const name of proxyEnvNames) {
    const value = originalProxyEnv.get(name);
    if (value === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = value;
    }
  }
}

describe('upstream response abort handling', () => {
  it('aborts response body reads when the client abort signal fires', async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('{"partial":'));
      }
    });
    const response = new Response(body, {
      headers: {
        'content-type': 'application/json'
      }
    });

    const pending = readUpstreamPayload(response, controller.signal).catch((error) => error);
    await Promise.resolve();

    controller.abort(new Error('client disconnected'));
    const error = await pending;

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toBe('client disconnected');
  });

  it('cancels an upstream response body when the client abort signal fires', async () => {
    const controller = new AbortController();
    let cancelReason: unknown;
    const body = new ReadableStream<Uint8Array>({
      cancel(reason) {
        cancelReason = reason;
      }
    });
    const response = new Response(body);

    cancelResponseBodyOnAbort(response, controller.signal);
    const reason = new Error('client disconnected');
    controller.abort(reason);
    await Promise.resolve();

    expect(cancelReason).toBe(reason);
  });
});

describe('sanitizeHeadersForLog', () => {
  it('redacts gateway-compatible credential headers', () => {
    expect(
      sanitizeHeadersForLog({
        authorization: 'Bearer client-secret',
        'x-api-key': 'openai-secret',
        'api-key': 'anthropic-secret',
        'X-Goog-Api-Key': 'gemini-secret',
        'x-mcp-key': 'mcp-secret',
        'x-codex-access-token': 'codex-secret',
        'x-auth-signature': 'signature-secret',
        'x-manager-key': 'manager-secret',
        'content-type': 'application/json'
      })
    ).toEqual({
      authorization: '***',
      'x-api-key': '***',
      'api-key': '***',
      'X-Goog-Api-Key': '***',
      'x-mcp-key': '***',
      'x-codex-access-token': '***',
      'x-auth-signature': '***',
      'x-manager-key': '***',
      'content-type': 'application/json'
    });
  });
});

describe('sanitizePayloadForLog', () => {
  it('summarizes binary payloads without exposing or expanding their bytes', () => {
    const sanitized = sanitizePayloadForLog(Buffer.alloc(4 * 1024 * 1024, 7));

    expect(sanitized).toEqual({
      type: 'Buffer',
      byteLength: 4 * 1024 * 1024,
      omitted: true
    });
  });

  it('preserves usage token counters and details', () => {
    const sanitized = sanitizePayloadForLog({
      usage: {
        input_tokens: 120,
        output_tokens: 18,
        total_tokens: 138,
        input_tokens_details: {
          cached_tokens: 24,
          cache_creation_tokens: 12
        }
      },
      max_tokens: 300
    });

    expect(sanitized).toEqual({
      usage: {
        input_tokens: 120,
        output_tokens: 18,
        total_tokens: 138,
        input_tokens_details: {
          cached_tokens: 24,
          cache_creation_tokens: 12
        }
      },
      max_tokens: 300
    });
  });

  it('continues to redact credential token fields', () => {
    const sanitized = sanitizePayloadForLog({
      apiKey: 'camel-secret',
      api_key: 'snake-secret',
      access_token: 'access-secret',
      refreshToken: 'refresh-secret',
      nested: {
        token: 'opaque-secret',
        session_token: 'session-secret'
      },
      usage: {
        total_tokens: 66,
        promptTokenCount: 55,
        candidatesTokenCount: 11
      }
    });

    expect(sanitized).toEqual({
      apiKey: '***',
      api_key: '***',
      access_token: '***',
      refreshToken: '***',
      nested: {
        token: '***',
        session_token: '***'
      },
      usage: {
        total_tokens: 66,
        promptTokenCount: 55,
        candidatesTokenCount: 11
      }
    });
  });
});
