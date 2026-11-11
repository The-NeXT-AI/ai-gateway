import {
  buildStreamTerminalErrorFrames,
  createSseTerminalEventDetector
} from './stream-terminal-error';
import { createSseEventTracker } from './stream-terminal-events';

const upstreamStreamIdleErrorCode = 'upstream_stream_idle_timeout';

// Raised when an upstream body stops producing bytes for longer than the idle
// limit. `statusCode` is what Fastify answers with when the relay fails before
// the first byte, so a stall before any output becomes a 504 instead of a
// connection the client has to time out on its own.
export class UpstreamStreamIdleTimeoutError extends Error {
  readonly code = 'UPSTREAM_STREAM_IDLE_TIMEOUT';
  readonly statusCode = 504;

  constructor(readonly idleTimeoutMs: number) {
    super(`Upstream stream sent no data for ${idleTimeoutMs}ms.`);
    this.name = 'UpstreamStreamIdleTimeoutError';
  }
}

export interface RelayedStreamIdleGuardInput {
  adapterKey: string;
  signal?: AbortSignal;
}

function isIdleTimeoutEnabled(idleTimeoutMs: number): boolean {
  return Number.isFinite(idleTimeoutMs) && idleTimeoutMs > 0;
}

function isEventStreamResponse(response: Response): boolean {
  return response.headers.get('content-type')?.toLowerCase().includes('text/event-stream') ?? false;
}

function buildIdleTerminalFrames(adapterKey: string, error: UpstreamStreamIdleTimeoutError): string[] {
  return buildStreamTerminalErrorFrames({
    adapterKey,
    code: upstreamStreamIdleErrorCode,
    message: error.message
  });
}

// Only the wait on upstream is timed: the read is issued when the consumer asks
// for data, so a slow client holding back demand never counts as upstream idle.
type UpstreamRead = Awaited<ReturnType<ReadableStreamDefaultReader<Uint8Array>['read']>>;

async function readWithinIdleTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  idleTimeoutMs: number
): Promise<UpstreamRead> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const idle = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new UpstreamStreamIdleTimeoutError(idleTimeoutMs)), idleTimeoutMs);
    timer.unref?.();
  });
  try {
    return await Promise.race([reader.read(), idle]);
  } finally {
    clearTimeout(timer);
  }
}

function cancelQuietly(reader: ReadableStreamDefaultReader<Uint8Array>, reason: unknown): void {
  reader.cancel(reason).catch(() => undefined);
}

/**
 * Fails an event-stream body with `UpstreamStreamIdleTimeoutError` once
 * upstream goes quiet for `idleTimeoutMs`, and cancels the upstream body so the
 * provider socket is released. Every reader downstream, including clones, sees
 * an ordinary stream error; other bodies are returned untouched.
 */
export function withUpstreamStreamIdleTimeout(
  response: Response,
  idleTimeoutMs: number,
  signal?: AbortSignal
): Response {
  if (!response.body || !isIdleTimeoutEnabled(idleTimeoutMs) || !isEventStreamResponse(response)) {
    return response;
  }

  const reader = response.body.getReader();
  const onAbort = () => cancelQuietly(reader, signal?.reason);
  if (signal?.aborted) {
    // An abort that already fired never reaches a new listener, and callUpstream
    // stopped forwarding the client signal to fetch once headers arrived, so the
    // client leaving in between would otherwise hold the provider socket open.
    onAbort();
  } else {
    signal?.addEventListener('abort', onAbort, { once: true });
  }
  const release = () => signal?.removeEventListener('abort', onAbort);

  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        let result: UpstreamRead;
        try {
          result = await readWithinIdleTimeout(reader, idleTimeoutMs);
        } catch (error) {
          release();
          if (error instanceof UpstreamStreamIdleTimeoutError) {
            cancelQuietly(reader, error);
          }
          controller.error(error);
          return;
        }
        if (result.done) {
          release();
          controller.close();
          return;
        }
        controller.enqueue(result.value);
      },
      cancel(reason) {
        release();
        return reader.cancel(reason);
      }
    },
    { highWaterMark: 0 }
  );

  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers
  });
}

/**
 * Ends a relayed (passthrough) body read through `withUpstreamStreamIdleTimeout`
 * the way the source protocol expects once the idle error arrives:
 * - nothing relayed yet: the body errors, so Fastify answers 504 while headers
 *   are still unsent;
 * - the stream already carried its terminal event: it just closes, since the
 *   upstream finished and only failed to hang up;
 * - otherwise: a protocol-native error frame is appended and the stream ends.
 * Protocols without an in-band error event fail as before; non-SSE bodies are
 * returned untouched.
 */
export function guardRelayedStreamIdle(response: Response, input: RelayedStreamIdleGuardInput): Response {
  if (!response.body || !isEventStreamResponse(response)) {
    return response;
  }

  const canFrame = buildStreamTerminalErrorFrames({ adapterKey: input.adapterKey }).length > 0;
  const reader = response.body.getReader();
  const tracker = createSseEventTracker();
  const detector = createSseTerminalEventDetector(tracker);
  const encoder = new TextEncoder();
  let relayedBytes = false;

  const endOnIdle = (controller: ReadableStreamDefaultController<Uint8Array>, error: unknown) => {
    if (!(error instanceof UpstreamStreamIdleTimeoutError) || input.signal?.aborted || !relayedBytes) {
      controller.error(error);
      return;
    }
    if (canFrame && detector.hasTerminalEvent()) {
      controller.close();
      return;
    }
    const frames = canFrame ? buildIdleTerminalFrames(input.adapterKey, error) : [];
    if (frames.length === 0) {
      controller.error(error);
      return;
    }
    const boundary = tracker.closeRetainedEvent();
    if (boundary.length > 0) {
      controller.enqueue(boundary);
    }
    for (const frame of frames) {
      controller.enqueue(encoder.encode(frame));
    }
    controller.close();
  };

  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            return;
          }
          if (canFrame) {
            detector.push(value);
          }
          relayedBytes = relayedBytes || value.byteLength > 0;
          controller.enqueue(value);
        } catch (error) {
          endOnIdle(controller, error);
        }
      },
      cancel(reason) {
        return reader.cancel(reason);
      }
    },
    { highWaterMark: 0 }
  );

  const headers = new Headers(response.headers);
  if (canFrame) {
    // Appending a frame changes the body length.
    headers.delete('content-length');
  }
  return new Response(body, { status: response.status, statusText: response.statusText, headers });
}

/**
 * Converted-stream counterpart of `guardRelayedStreamIdle`: the converter reads
 * an upstream body wrapped by `withUpstreamStreamIdleTimeout`, and this turns
 * the resulting idle error into the source protocol's error frame.
 */
export async function* endConvertedStreamOnIdle(
  frames: AsyncIterable<string>,
  adapterKey: string
): AsyncGenerator<string> {
  const detector = createSseTerminalEventDetector();
  const encoder = new TextEncoder();
  let relayedFrames = false;

  try {
    for await (const frame of frames) {
      detector.push(encoder.encode(frame));
      relayedFrames = relayedFrames || frame.length > 0;
      yield frame;
    }
  } catch (error) {
    if (!(error instanceof UpstreamStreamIdleTimeoutError) || !relayedFrames) {
      throw error;
    }
    if (detector.hasTerminalEvent()) {
      return;
    }
    const terminalFrames = buildIdleTerminalFrames(adapterKey, error);
    if (terminalFrames.length === 0) {
      throw error;
    }
    yield* terminalFrames;
  }
}
