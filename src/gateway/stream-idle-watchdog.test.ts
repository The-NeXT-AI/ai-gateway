import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import { callUpstream } from '../upstream/client';
import {
  endConvertedStreamOnIdle,
  guardRelayedStreamIdle,
  UpstreamStreamIdleTimeoutError,
  withUpstreamStreamIdleTimeout
} from './stream-idle-watchdog';

const encoder = new TextEncoder();

// Hands out `chunks`, then stays open without producing anything more.
function createStalledEventStream(
  chunks: string[],
  contentType = 'text/event-stream'
): { response: Response; cancelled: () => boolean } {
  let index = 0;
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index += 1;
        return;
      }
      return new Promise<void>(() => undefined);
    },
    cancel() {
      cancelled = true;
    }
  });

  return {
    response: new Response(stream, { status: 200, headers: { 'content-type': contentType } }),
    cancelled: () => cancelled
  };
}

function createFiniteEventStream(chunks: string[]): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      }
    }),
    { status: 200, headers: { 'content-type': 'text/event-stream' } }
  );
}

async function* framesThenError(frames: string[], error: unknown): AsyncGenerator<string> {
  yield* frames;
  throw error;
}

async function collect(frames: AsyncIterable<string>): Promise<string> {
  let body = '';
  for await (const frame of frames) {
    body += frame;
  }
  return body;
}

const anthropicDelta =
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n';

describe('withUpstreamStreamIdleTimeout', () => {
  it('fails a stalled body with the idle error and cancels upstream', async () => {
    const upstream = createStalledEventStream([anthropicDelta]);
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 20);
    const reader = watched.body!.getReader();

    expect((await reader.read()).done).toBe(false);
    await expect(reader.read()).rejects.toBeInstanceOf(UpstreamStreamIdleTimeoutError);
    expect(upstream.cancelled()).toBe(true);
  });

  it('keeps status and headers', () => {
    const upstream = createStalledEventStream([]);
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 20);

    expect(watched.status).toBe(200);
    expect(watched.headers.get('content-type')).toBe('text/event-stream');
  });

  it('does not count time the consumer spends before asking for more', async () => {
    const watched = withUpstreamStreamIdleTimeout(createFiniteEventStream([anthropicDelta, anthropicDelta]), 20);
    const reader = watched.body!.getReader();

    await reader.read();
    await new Promise((resolve) => setTimeout(resolve, 60));

    expect((await reader.read()).done).toBe(false);
    expect((await reader.read()).done).toBe(true);
  });

  it('is disabled by a zero timeout', () => {
    const upstream = createStalledEventStream([]);

    expect(withUpstreamStreamIdleTimeout(upstream.response, 0)).toBe(upstream.response);
  });

  it('leaves non event-stream bodies alone', () => {
    const upstream = createStalledEventStream([], 'application/json');

    expect(withUpstreamStreamIdleTimeout(upstream.response, 20)).toBe(upstream.response);
  });

  it('releases a stalled upstream that a clone is still holding open', async () => {
    const upstream = createStalledEventStream([anthropicDelta]);
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 20);
    const unreadClone = watched.clone();

    await expect(watched.text()).rejects.toBeInstanceOf(UpstreamStreamIdleTimeoutError);
    expect(upstream.cancelled()).toBe(true);
    await expect(unreadClone.text()).rejects.toBeInstanceOf(UpstreamStreamIdleTimeoutError);
  });

  it('closes cleanly when upstream goes quiet after its terminal event', async () => {
    const stop = 'event: message_stop\ndata: {"type":"message_stop"}\n\n';
    const upstream = createStalledEventStream([anthropicDelta, stop]);
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 20);
    const billingClone = watched.clone();

    expect(await watched.text()).toBe(anthropicDelta + stop);
    expect(await billingClone.text()).toBe(anthropicDelta + stop);
    expect(upstream.cancelled()).toBe(true);
  });

  it('still fails when the terminal event was cut off before its blank line', async () => {
    const upstream = createStalledEventStream([anthropicDelta, 'event: message_stop\ndata: {"type":"message_stop"}\n']);
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 20);

    await expect(watched.text()).rejects.toBeInstanceOf(UpstreamStreamIdleTimeoutError);
  });
});

// A real socket, so undici's own bodyTimeout (set from the overall upstream
// timeout) can fire before the watchdog does.
describe('withUpstreamStreamIdleTimeout behind the undici body timeout', () => {
  let server: Server | undefined;

  afterEach(() => {
    server?.closeAllConnections();
    server?.close();
    server = undefined;
  });

  async function stallAfter(firstEvent: string): Promise<Response> {
    server = createServer((_request, response) => {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write(firstEvent);
    });
    await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    // undici checks its body timeout on a coarse clock, so it fires after
    // roughly 1-2s here; the watchdog is set well beyond that.
    const upstream = await callUpstream(`http://127.0.0.1:${port}/`, {}, { stream: true }, 1_000);
    return withUpstreamStreamIdleTimeout(upstream, 10_000);
  }

  it('fails with the idle error when undici gives up first', async () => {
    const response = await stallAfter('data: {"type":"message_start"}\n\n');

    const error = await response.text().then(
      () => undefined,
      (caught: unknown) => caught
    );

    expect(error).toBeInstanceOf(UpstreamStreamIdleTimeoutError);
    expect((error as UpstreamStreamIdleTimeoutError).statusCode).toBe(504);
    expect((error as Error).cause).toMatchObject({ cause: { code: 'UND_ERR_BODY_TIMEOUT' } });
  }, 8_000);

  it('closes cleanly when undici gives up after the terminal event', async () => {
    const terminal = 'event: message_stop\ndata: {"type":"message_stop"}\n\n';
    const response = await stallAfter(terminal);

    await expect(response.text()).resolves.toBe(terminal);
  }, 8_000);
});

describe('withUpstreamStreamIdleTimeout client abort', () => {
  it('cancels upstream when the client goes away mid-stream', async () => {
    const upstream = createStalledEventStream([anthropicDelta]);
    const client = new AbortController();
    const watched = withUpstreamStreamIdleTimeout(upstream.response, 10_000, client.signal);
    const reader = watched.body!.getReader();

    await reader.read();
    const pending = reader.read();
    client.abort(new Error('client gone'));

    expect(await pending).toEqual({ done: true, value: undefined });
    expect(upstream.cancelled()).toBe(true);
  });

  it('cancels upstream when the client left before the body was wrapped', async () => {
    const upstream = createStalledEventStream([anthropicDelta]);
    const client = new AbortController();
    client.abort(new Error('client gone'));

    withUpstreamStreamIdleTimeout(upstream.response, 10_000, client.signal);
    await new Promise((resolve) => setImmediate(resolve));

    expect(upstream.cancelled()).toBe(true);
  });
});

describe('guardRelayedStreamIdle', () => {
  it('ends an idle Anthropic stream with one error event', async () => {
    const upstream = createStalledEventStream([anthropicDelta]);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'anthropic_messages'
    });

    const body = await guarded.text();

    expect(body.startsWith(anthropicDelta)).toBe(true);
    expect(body.match(/^event: error$/gm)).toHaveLength(1);
    expect(body).toContain('"code":"upstream_stream_idle_timeout"');
    expect(body).toContain('Upstream stream sent no data for 20ms.');
  });

  it('ends an idle OpenAI chat stream with an error chunk and the done sentinel', async () => {
    const upstream = createStalledEventStream([
      'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
    ]);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'openai_chat'
    });

    const body = await guarded.text();

    expect(body).toContain('"code":"upstream_stream_idle_timeout"');
    expect(body.endsWith('data: [DONE]\n\n')).toBe(true);
  });

  it('closes the event a cut-off chunk left open before the error event', async () => {
    const upstream = createStalledEventStream(['event: content_block_delta\ndata: {"type":"content_block_delta"']);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'anthropic_messages'
    });

    const body = await guarded.text();

    expect(body).toMatch(/\n\nevent: error\n/);
  });

  it('closes quietly once the terminal event already went out', async () => {
    const upstream = createStalledEventStream(['event: message_stop\ndata: {"type":"message_stop"}\n\n']);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'anthropic_messages'
    });

    expect(await guarded.text()).toBe('event: message_stop\ndata: {"type":"message_stop"}\n\n');
  });

  it('fails the body when nothing was relayed, so the server can still send a status', async () => {
    const upstream = createStalledEventStream([]);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'anthropic_messages'
    });

    await expect(guarded.text()).rejects.toMatchObject({ statusCode: 504 });
  });

  it('passes other stream errors through unchanged', async () => {
    const error = new Error('terminated');
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(anthropicDelta));
          controller.error(error);
        }
      }),
      { headers: { 'content-type': 'text/event-stream' } }
    );
    const guarded = guardRelayedStreamIdle(response, { adapterKey: 'anthropic_messages' });

    await expect(guarded.text()).rejects.toBe(error);
  });

  it('fails as before for adapters without an in-band error event', async () => {
    const upstream = createStalledEventStream(['data: {"candidates":[]}\n\n']);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'gemini_generate_content'
    });

    await expect(guarded.text()).rejects.toBeInstanceOf(UpstreamStreamIdleTimeoutError);
  });

  it('does not frame the failure after the client aborted', async () => {
    const controller = new AbortController();
    const upstream = createStalledEventStream([anthropicDelta]);
    const guarded = guardRelayedStreamIdle(withUpstreamStreamIdleTimeout(upstream.response, 20), {
      adapterKey: 'anthropic_messages',
      signal: controller.signal
    });
    const reader = guarded.body!.getReader();

    await reader.read();
    controller.abort();

    await expect(reader.read()).rejects.toBeDefined();
  });
});

describe('endConvertedStreamOnIdle', () => {
  const idle = new UpstreamStreamIdleTimeoutError(20);

  it('appends the source protocol error frame after converted output', async () => {
    const body = await collect(
      endConvertedStreamOnIdle(framesThenError([anthropicDelta], idle), 'anthropic_messages')
    );

    expect(body.startsWith(anthropicDelta)).toBe(true);
    expect(body).toContain('"code":"upstream_stream_idle_timeout"');
  });

  it('ends an OpenAI Responses stream with a failed event and the done sentinel', async () => {
    const body = await collect(
      endConvertedStreamOnIdle(
        framesThenError(['event: response.created\ndata: {"type":"response.created"}\n\n'], idle),
        'openai_responses'
      )
    );

    expect(body.match(/^event: response\.failed$/gm)).toHaveLength(1);
    expect(body.endsWith('data: [DONE]\n\n')).toBe(true);
  });

  it('rethrows when no frame was produced yet', async () => {
    await expect(
      collect(endConvertedStreamOnIdle(framesThenError([], idle), 'anthropic_messages'))
    ).rejects.toBe(idle);
  });

  it('stops quietly when the converter already emitted its terminal event', async () => {
    const body = await collect(
      endConvertedStreamOnIdle(
        framesThenError(['event: message_stop\ndata: {"type":"message_stop"}\n\n'], idle),
        'anthropic_messages'
      )
    );

    expect(body).not.toContain('event: error');
  });

  it('rethrows errors other than the idle timeout', async () => {
    const error = new Error('boom');

    await expect(
      collect(endConvertedStreamOnIdle(framesThenError([anthropicDelta], error), 'anthropic_messages'))
    ).rejects.toBe(error);
  });
});
