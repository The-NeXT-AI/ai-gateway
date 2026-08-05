import { afterEach, describe, expect, it, vi } from 'vitest';

const undiciMock = vi.hoisted(() => ({
  dispatch: vi.fn(() => true),
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
  getGlobalDispatcher: undiciMock.getGlobalDispatcher
}));

import {
  callUpstream,
  cancelResponseBodyOnAbort,
  readUpstreamPayload,
  sanitizePayloadForLog
} from './client';

type FetchInitWithDispatcherForTest = RequestInit & { dispatcher?: unknown };
type TestDispatcher = {
  isMockActive?: unknown;
  dispatch(options: Record<string, unknown>, handler: unknown): boolean;
};

describe('callUpstream', () => {
  afterEach(() => {
    undiciMock.dispatch.mockClear();
    undiciMock.getGlobalDispatcher.mockReset();
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

  it('passes timeout-aware dispatchers to upstream fetch', async () => {
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
        600000
      );

      const fetchInit = fetchMock.mock.calls[0]?.[1] as FetchInitWithDispatcherForTest | undefined;

      expect(response.status).toBe(200);
      expect(fetchInit?.dispatcher).toBeTruthy();
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
      expect(undiciMock.getGlobalDispatcher).not.toHaveBeenCalled();
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
