import { describe, expect, it } from 'vitest';
import {
  buildStreamTerminalErrorFrames,
  createSseTerminalEventDetector,
  guardStreamTerminalError
} from './stream-terminal-error';
import { parseSseChunks } from '../sse';

const encoder = new TextEncoder();

function createEventStream(chunks: string[], errorAfter?: unknown): Response {
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index += 1;
        return;
      }
      if (errorAfter) {
        controller.error(errorAfter);
        return;
      }
      controller.close();
    }
  });

  return new Response(stream, {
    headers: { 'content-type': 'text/event-stream' },
    status: 200
  });
}

function createByteStream(chunks: Uint8Array[], errorAfter?: unknown): Response {
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(chunks[index]);
        index += 1;
        return;
      }
      if (errorAfter) {
        controller.error(errorAfter);
        return;
      }
      controller.close();
    }
  });

  return new Response(stream, {
    headers: { 'content-type': 'text/event-stream' },
    status: 200
  });
}

/**
 * A source that only hands out a chunk when the consumer's read actually asks for
 * one, so a read that happens ahead of demand is visible as an extra `pull`.
 */
function createDemandStream(): {
  response: Response;
  pullCount: () => number;
  cancelReason: () => unknown;
  grant: () => void;
} {
  let permits = 1;
  let pulls = 0;
  let cancelled = false;
  let cancelledWith: unknown;
  const waiters: Array<() => void> = [];

  const stream = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        pulls += 1;
        while (permits === 0 && !cancelled) {
          await new Promise<void>((resolve) => {
            waiters.push(resolve);
          });
        }
        if (cancelled || permits === 0) {
          return;
        }
        permits -= 1;
        controller.enqueue(encoder.encode(`data: {"n":${pulls}}\n\n`));
      },
      cancel(reason) {
        cancelled = true;
        cancelledWith = reason;
        for (const resolve of waiters.splice(0)) {
          resolve();
        }
      }
    },
    { highWaterMark: 0 }
  );

  return {
    response: new Response(stream, {
      headers: { 'content-type': 'text/event-stream' },
      status: 200
    }),
    pullCount: () => pulls,
    cancelReason: () => cancelledWith,
    grant: () => {
      permits += 1;
      for (const resolve of waiters.splice(0)) {
        resolve();
      }
    }
  };
}

/** Parses the guarded bytes with the gateway's own SSE consumer, not with the detector. */
async function dispatchedEvents(body: Uint8Array): Promise<Array<{ event?: string; data: string }>> {
  const events: Array<{ event?: string; data: string }> = [];
  for await (const parsed of parseSseChunks(new Response(body))) {
    events.push({ event: parsed.event, data: parsed.data });
  }
  return events;
}

function transportError(code: string, message = 'transport detail that must not leak'): Error {
  const error = new Error(message);
  (error as Error & { code?: string }).code = code;
  return error;
}

const truncatedThinkingChunk =
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"live"}}\n\n';

const anthropicTerminalFrame =
  'event: error\ndata: {"error":{"code":"upstream_stream_error","message":"Upstream stream failed.","type":"api_error"},"type":"error"}\n\n';

describe('buildStreamTerminalErrorFrames', () => {
  it('emits an Anthropic error event carrying a typed api error', () => {
    const frames = buildStreamTerminalErrorFrames({
      adapterKey: 'anthropic_messages',
      message: 'Upstream stream failed.'
    });

    expect(frames).toHaveLength(1);
    expect(frames[0]).toBe(anthropicTerminalFrame);
  });

  it('emits an OpenAI chat error chunk followed by the done sentinel', () => {
    const frames = buildStreamTerminalErrorFrames({
      adapterKey: 'openai_chat',
      message: 'Upstream stream failed.'
    });

    expect(frames).toHaveLength(2);
    expect(frames[0].startsWith('data: {')).toBe(true);
    expect(frames[0]).not.toContain('event:');
    expect(frames[1]).toBe('data: [DONE]\n\n');
  });

  it('emits a failed Responses event followed by the done sentinel', () => {
    const frames = buildStreamTerminalErrorFrames({
      adapterKey: 'openai_responses',
      message: 'Upstream stream failed.'
    });

    expect(frames).toHaveLength(2);
    expect(frames[0].startsWith('event: response.failed\n')).toBe(true);
    expect(frames[0]).toContain('"status":"failed"');
    expect(frames[1]).toBe('data: [DONE]\n\n');
  });

  it('returns no frame for adapters without an in-band error event', () => {
    expect(
      buildStreamTerminalErrorFrames({
        adapterKey: 'gemini_generate_content',
        message: 'Upstream stream failed.'
      })
    ).toEqual([]);
  });
});

describe('createSseTerminalEventDetector', () => {
  it('detects a terminal event split across chunks', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('event: message_'));
    expect(detector.hasTerminalEvent()).toBe(false);
    detector.push(encoder.encode('stop\ndata: {"type":"message_stop"}\n\n'));

    expect(detector.hasTerminalEvent()).toBe(true);
  });

  it('detects the OpenAI done sentinel', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('data: [DONE]\n\n'));

    expect(detector.hasTerminalEvent()).toBe(true);
  });

  it('ignores the word error inside streamed text', () => {
    const detector = createSseTerminalEventDetector();
    const delta = JSON.stringify({ choices: [{ delta: { content: '{"type":"error"}' } }] });
    detector.push(encoder.encode(`data: ${delta}\n\n`));

    expect(detector.hasTerminalEvent()).toBe(false);
  });

  it('reads a terminal payload assembled from several data lines', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('data: {"type":\ndata: "error"}\n\n'));

    expect(detector.hasTerminalEvent()).toBe(true);
  });

  it('accepts CRLF line endings', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n'));

    expect(detector.hasTerminalEvent()).toBe(true);
  });

  it('does not count a field line that no blank line delivered', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('event: message_stop\ndata: {"type":"message_stop"}\n'));

    expect(detector.hasTerminalEvent()).toBe(false);
  });

  it('does not count an event name that carries no payload yet', () => {
    const detector = createSseTerminalEventDetector();
    detector.push(encoder.encode('event: error\n'));
    expect(detector.hasTerminalEvent()).toBe(false);

    detector.push(encoder.encode('\n'));
    expect(detector.hasTerminalEvent()).toBe(true);
  });

  it('does not count a terminal word below the payload top level', () => {
    const nested = [
      { type: 'response.output_item.added', item: { type: 'function_call', nested: { type: 'error' } } },
      { type: 'content_block_start', content_block: { type: 'tool_use', input: { kind: 'error', type: 'error' } } },
      { type: 'content_block_delta', delta: { partial_json: '{"type":"error"}' } },
      { type: 'message_delta', delta: { stop_reason: null }, metadata: { type: 'error' } }
    ];

    for (const payload of nested) {
      const detector = createSseTerminalEventDetector();
      detector.push(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      expect(detector.hasTerminalEvent()).toBe(false);
    }
  });

  it('treats an OpenAI chat error chunk as terminal, with or without an inner type', () => {
    // The gateway's own fixtures send `{"error":{"message":"invalid key"}}` with
    // no `type` at either level, so the `error` key is the whole signal.
    for (const payload of [
      { error: { message: 'invalid key' } },
      { error: { message: 'x', type: 'api_error' } },
      { error: { message: 'x', type: 'error' } }
    ]) {
      const detector = createSseTerminalEventDetector();
      detector.push(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      expect(detector.hasTerminalEvent()).toBe(true);
    }
  });
});

describe('guardStreamTerminalError', () => {
  it('ends a truncated Anthropic stream with one terminal error frame', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(
        ['event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"live"}}\n\n'],
        new Error('terminated')
      ),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body).toContain('"type":"thinking_delta"');
    expect(body.endsWith(anthropicTerminalFrame)).toBe(true);
  });

  it('never writes a second terminal event', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(
        ['event: message_stop\ndata: {"type":"message_stop"}\n\n'],
        new Error('terminated')
      ),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    await expect(guarded.text()).rejects.toThrow('terminated');
  });

  it('stays out of the way of non event-stream responses', () => {
    const response = new Response('{"ok":true}', {
      headers: { 'content-type': 'application/json' }
    });

    expect(guardStreamTerminalError(response, { adapterKey: 'anthropic_messages' })).toBe(response);
  });

  it('stays out of the way of adapters without a terminal error frame', () => {
    const response = createEventStream(['data: {"candidates":[]}\n\n']);

    expect(guardStreamTerminalError(response, { adapterKey: 'gemini_generate_content' })).toBe(
      response
    );
  });

  it('does not mask the failure after the client aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const guarded = guardStreamTerminalError(
      createEventStream(['data: {"choices":[]}\n\n'], new Error('aborted')),
      { adapterKey: 'openai_chat', message: 'Upstream stream failed.', signal: controller.signal }
    );

    await expect(guarded.text()).rejects.toThrow('aborted');
  });
});

describe('guardStreamTerminalError transport identity', () => {
  it('names a recognised socket code without copying its message', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], transportError('ECONNRESET')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(
      'event: error\ndata: {"error":{"code":"upstream_stream_error","message":"Upstream stream failed. (transport: ECONNRESET)","type":"api_error"},"type":"error"}\n\n'
    )).toBe(true);
    expect(body).not.toContain('transport detail that must not leak');
  });

  it('finds the code nested in a cause chain', async () => {
    const inner = transportError('UND_ERR_SOCKET', 'socket closed while reading');
    const middle = new Error('fetch failed', { cause: inner });
    const outer = new Error('upstream request failed', { cause: middle });

    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], outer),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body).toContain('"message":"Upstream stream failed. (transport: UND_ERR_SOCKET)"');
    for (const leaked of ['socket closed while reading', 'fetch failed', 'upstream request failed']) {
      expect(body).not.toContain(leaked);
    }
  });

  it('keeps the OpenAI chat framing when the failure is identified', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(['data: {"choices":[]}\n\n'], transportError('ETIMEDOUT')),
      { adapterKey: 'openai_chat', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(
      'data: {"error":{"code":"upstream_stream_error","message":"Upstream stream failed. (transport: ETIMEDOUT)","type":"api_error"}}\n\ndata: [DONE]\n\n'
    )).toBe(true);
  });

  it('keeps the failed Responses framing when the failure is identified', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(['data: {"type":"response.created"}\n\n'], transportError('EPIPE')),
      { adapterKey: 'openai_responses', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(
      'event: response.failed\ndata: {"response":{"error":{"code":"upstream_stream_error","message":"Upstream stream failed. (transport: EPIPE)","type":"api_error"},"object":"response","status":"failed"},"type":"response.failed"}\n\ndata: [DONE]\n\n'
    )).toBe(true);
  });

  it('falls back to the default message plus the code when none is configured', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], transportError('EAI_AGAIN')),
      { adapterKey: 'anthropic_messages' }
    );

    const body = await guarded.text();

    expect(body).toContain(
      '"message":"Upstream stream failed before the response completed. (transport: EAI_AGAIN)"'
    );
  });

  it('refuses an unknown code and a private message', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(
        [truncatedThinkingChunk],
        transportError('ERR_SOMETHING_PRIVATE', 'secret upstream body at https://internal/route')
      ),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(anthropicTerminalFrame)).toBe(true);
    expect(body).not.toContain('ERR_SOMETHING_PRIVATE');
    expect(body).not.toContain('secret upstream body');
    expect(body).not.toContain('internal/route');
  });

  it('ignores a code carried in the message when the error has none', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], new Error('read ECONNRESET on socket')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(anthropicTerminalFrame)).toBe(true);
    expect(body).not.toContain('(transport:');
  });

  it('survives a circular cause chain without reporting a code', async () => {
    const first = new Error('first');
    const second = new Error('second', { cause: first });
    (first as Error & { cause?: unknown }).cause = second;

    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], first),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body).toContain('"message":"Upstream stream failed."');
    expect(body).not.toContain('(transport:');
  });

  it('accepts a recognised code on the thrown error itself when the cause is unrelated', async () => {
    const failure = transportError('ECONNREFUSED', 'connect ECONNREFUSED 192.0.2.1:443');

    const guarded = guardStreamTerminalError(
      createEventStream([truncatedThinkingChunk], failure),
      { adapterKey: 'anthropic_messages' }
    );

    const body = await guarded.text();

    expect(body).toContain('(transport: ECONNREFUSED)');
    expect(body).not.toContain('192.0.2.1');
  });
});

describe('guardStreamTerminalError frame accounting', () => {
  it('writes no extra frame once a terminal event already arrived', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(
        ['event: message_stop\ndata: {"type":"message_stop"}\n\n'],
        transportError('ECONNRESET')
      ),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    await expect(guarded.text()).rejects.toThrow('transport detail that must not leak');
  });

  it('writes no frame after the client aborted, even with a recognised code', async () => {
    const controller = new AbortController();
    controller.abort();
    const guarded = guardStreamTerminalError(
      createEventStream(['data: {"choices":[]}\n\n'], transportError('ECONNRESET')),
      { adapterKey: 'openai_chat', message: 'Upstream stream failed.', signal: controller.signal }
    );

    await expect(guarded.text()).rejects.toThrow('transport detail that must not leak');
  });

  it('keeps the identity passthrough for an unknown adapter', () => {
    const response = createEventStream(['data: {"candidates":[]}\n\n']);

    expect(guardStreamTerminalError(response, { adapterKey: 'gemini_generate_content' })).toBe(
      response
    );
  });
});

describe('guardStreamTerminalError event boundaries', () => {
  it('keeps a cut-off UTF-8 line out of the terminal frame', async () => {
    // A line the transport stopped inside: `data: {"t":"` plus the first two bytes
    // of a three-byte code point.
    const cutOff = Uint8Array.from([...encoder.encode('data: {"t":"'), 0xe2, 0x85]);

    const guarded = guardStreamTerminalError(
      createByteStream([cutOff], transportError('ECONNRESET')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = new Uint8Array(await guarded.arrayBuffer());
    const text = new TextDecoder().decode(body);

    // Not the old behaviour, which appended the frame to the partial line.
    expect(text).not.toContain('"t":event: error');
    expect(text).toContain('\n\nevent: error\n');

    const events = await dispatchedEvents(body);
    const last = events[events.length - 1];
    expect(last?.event).toBe('error');
    expect(JSON.parse(last?.data ?? '{}')).toMatchObject({
      type: 'error',
      error: { code: 'upstream_stream_error' }
    });
  });

  it('delivers the frame as its own event after a field line with no blank line', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(['data: {"type":"message_start"}\n'], transportError('ECONNRESET')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = new Uint8Array(await guarded.arrayBuffer());

    expect(new TextDecoder().decode(body)).toContain('\n\nevent: error\n');
    const events = await dispatchedEvents(body);
    expect(events[events.length - 1]?.event).toBe('error');
  });

  it('treats an undelivered terminal event as missing, not as an already-delivered one', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(
        ['event: message_stop\ndata: {"type":"message_stop"}\n'],
        transportError('ECONNRESET')
      ),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(
      'event: error\ndata: {"error":{"code":"upstream_stream_error","message":"Upstream stream failed. (transport: ECONNRESET)","type":"api_error"},"type":"error"}\n\n'
    )).toBe(true);
  });

  it('treats an event name with no data as undelivered', async () => {
    const guarded = guardStreamTerminalError(
      createEventStream(['event: error\n'], new Error('terminated')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body.endsWith(anthropicTerminalFrame)).toBe(true);
  });

  it('does not let a nested terminal word suppress the frame', async () => {
    const toolInput = JSON.stringify({
      type: 'content_block_start',
      index: 1,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'report', input: { kind: 'error', type: 'error' } }
    });

    const guarded = guardStreamTerminalError(
      createEventStream([`data: ${toolInput}\n\n`], transportError('ECONNRESET')),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = await guarded.text();

    expect(body).toContain(toolInput);
    expect(body.endsWith(
      'event: error\ndata: {"error":{"code":"upstream_stream_error","message":"Upstream stream failed. (transport: ECONNRESET)","type":"api_error"},"type":"error"}\n\n'
    )).toBe(true);
  });

  it('relays ordinary chunk bytes unchanged and in order', async () => {
    const chunks = [
      'event: message_start\ndata: {"type":"message_start"}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"h\u00e9llo \u4e16\u754c"}}\n\n'
    ];

    const guarded = guardStreamTerminalError(
      createByteStream(chunks.map((chunk) => encoder.encode(chunk))),
      { adapterKey: 'anthropic_messages', message: 'Upstream stream failed.' }
    );

    const body = new Uint8Array(await guarded.arrayBuffer());

    expect(body).toEqual(encoder.encode(chunks.join('')));
  });

  it('keeps the done sentinel last for both OpenAI protocols', async () => {
    for (const adapterKey of ['openai_chat', 'openai_responses']) {
      const guarded = guardStreamTerminalError(
        createEventStream(['data: {"partial":true}\n'], transportError('ECONNRESET')),
        { adapterKey, message: 'Upstream stream failed.' }
      );

      const body = await guarded.text();

      expect(body).toContain('\n\ndata: ');
      expect(body.endsWith('data: [DONE]\n\n')).toBe(true);
      const events = await dispatchedEvents(encoder.encode(body));
      expect(events[events.length - 1]?.data).toBe('[DONE]');
    }
  });
});

describe('guardStreamTerminalError backpressure', () => {
  it('leaves the upstream stream unread until the client asks', async () => {
    // A source with no gate at all: an eager pump drains it immediately, while a
    // pull-driven stream leaves every chunk where it is.
    const chunks = Array.from({ length: 32 }, (_, index) => `data: {"n":${index}}\n\n`);
    let served = 0;
    const upstream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (served < chunks.length) {
          controller.enqueue(encoder.encode(chunks[served]));
          served += 1;
          return;
        }
        controller.close();
      }
    });
    const guarded = guardStreamTerminalError(
      new Response(upstream, { headers: { 'content-type': 'text/event-stream' } }),
      { adapterKey: 'anthropic_messages' }
    );

    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    // One queued chunk is the transport's slack; the eager pump served all 32.
    expect(served).toBeLessThanOrEqual(2);
    await guarded.body!.cancel();
  });

  it('keeps upstream read-ahead bounded as the client reads', async () => {
    const source = createDemandStream();
    const guarded = guardStreamTerminalError(source.response, {
      adapterKey: 'anthropic_messages'
    });
    const reader = guarded.body!.getReader();

    // Read-ahead stays at one chunk; it must not grow with the number of reads.
    for (let read = 1; read <= 4; read += 1) {
      await reader.read();
      await Promise.resolve();
      await Promise.resolve();
      expect(source.pullCount()).toBe(read + 1);
      source.grant();
    }

    await reader.cancel();
  });

  it('cancels the upstream body when the client goes away', async () => {
    const source = createDemandStream();
    const guarded = guardStreamTerminalError(source.response, {
      adapterKey: 'anthropic_messages'
    });
    const reader = guarded.body!.getReader();

    await reader.read();
    await Promise.resolve();
    await reader.cancel(new Error('client gone'));
    await Promise.resolve();
    await Promise.resolve();

    expect(source.cancelReason()).toEqual(new Error('client gone'));
    // One served chunk plus the single queued one; no further reads after cancel.
    expect(source.pullCount()).toBe(2);
  });

  it('never buffers more than one chunk for a client that never asks', async () => {
    const source = createDemandStream();
    const guarded = guardStreamTerminalError(source.response, {
      adapterKey: 'anthropic_messages'
    });

    await Promise.resolve();
    await Promise.resolve();

    // At most the single queued chunk; never the whole stream.
    expect(source.pullCount()).toBeLessThanOrEqual(1);
    await guarded.body!.cancel();
  });
});
