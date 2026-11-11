import { createSseEventTracker, type SseEvent, type SseEventTracker } from './stream-terminal-events';
import { asString, isObject } from '../utils';

const defaultTerminalErrorMessage = 'Upstream stream failed before the response completed.';
const upstreamStreamErrorCode = 'upstream_stream_error';
const defaultTerminalErrorType = 'api_error';

// Transport failures worth naming back to the client: the client cannot discover
// these from a truncated body, and none of them carries request or credential
// material. The set is closed on purpose - anything else stays generic, so no
// provider message, URL or error object can reach the client through this path.
const knownTransportCodes: Record<string, true> = {
  EAI_AGAIN: true,
  ECONNABORTED: true,
  ECONNREFUSED: true,
  ECONNRESET: true,
  EHOSTUNREACH: true,
  ENETUNREACH: true,
  EPIPE: true,
  ETIMEDOUT: true,
  UND_ERR_BODY_TIMEOUT: true,
  UND_ERR_CLOSED: true,
  UND_ERR_CONNECT_TIMEOUT: true,
  UND_ERR_HEADERS_TIMEOUT: true,
  UND_ERR_SOCKET: true
};

const maxTransportCauseDepth = 4;

const terminalSseEventNames: Record<string, true> = {
  error: true,
  message_stop: true,
  'response.completed': true,
  'response.error': true,
  'response.failed': true,
  'response.incomplete': true
};

// A terminal `type` is only meaningful on the payload's own top level. Nested
// objects carry provider or tool data - an Anthropic tool input can hold
// `{"type":"error"}` quite legitimately - so a nested occurrence says nothing
// about the stream.
const terminalSsePayloadTypes: Record<string, true> = {
  error: true,
  message_stop: true,
  'response.completed': true,
  'response.error': true,
  'response.failed': true,
  'response.incomplete': true
};

export interface StreamTerminalErrorFrameInput {
  adapterKey: string;
  code?: string;
  errorType?: string;
  message?: string;
}

export interface SseTerminalEventDetector {
  hasTerminalEvent(): boolean;
  push(chunk: Uint8Array): void;
}

export interface StreamTerminalErrorGuardInput {
  adapterKey: string;
  message?: string;
  signal?: AbortSignal;
}

/**
 * Adapters that can express a failed stream in-band. Checked before building
 * anything so a caller can decide support without allocating a frame.
 */
function supportsTerminalErrorFrame(adapterKey: string): boolean {
  return (
    adapterKey === 'anthropic_messages' ||
    adapterKey === 'openai_chat' ||
    adapterKey === 'openai_responses'
  );
}

/**
 * Frames a downstream client can classify as a failed stream, in the client's own
 * protocol. An empty result means the adapter has no in-band error event, so the
 * caller must keep failing the stream the other way (socket close).
 */
export function buildStreamTerminalErrorFrames(input: StreamTerminalErrorFrameInput): string[] {
  const message = input.message ?? defaultTerminalErrorMessage;
  const code = input.code ?? upstreamStreamErrorCode;
  const errorType = input.errorType ?? defaultTerminalErrorType;
  const error = { code, message, type: errorType };

  switch (input.adapterKey) {
    case 'anthropic_messages':
      return [encodeSseFrame('error', { error, type: 'error' })];
    case 'openai_chat':
      return [encodeSseFrame(undefined, { error }), encodeSseFrame(undefined, '[DONE]')];
    case 'openai_responses':
      return [
        encodeSseFrame('response.failed', {
          response: { error, object: 'response', status: 'failed' },
          type: 'response.failed'
        }),
        encodeSseFrame(undefined, '[DONE]')
      ];
    default:
      return [];
  }
}

export function createSseTerminalEventDetector(
  tracker: SseEventTracker = createSseEventTracker()
): SseTerminalEventDetector {
  let terminalEventSeen = false;

  return {
    hasTerminalEvent: () => terminalEventSeen,
    push(chunk: Uint8Array): void {
      if (terminalEventSeen) {
        return;
      }

      // Only events a blank line dispatched can suppress a terminal frame: a
      // field line still being written, or the tail the transport cut off, was
      // never delivered, so the stream still needs its error.
      for (const event of tracker.push(chunk)) {
        if (isTerminalSseEvent(event)) {
          terminalEventSeen = true;
          return;
        }
      }
    }
  };
}

/**
 * Wraps an event-stream response so a body that dies mid-stream ends with one
 * terminal error frame instead of a bare EOF. The client cannot tell a truncated
 * frame from a completed answer otherwise, and never retries.
 */
export function guardStreamTerminalError(
  response: Response,
  input: StreamTerminalErrorGuardInput
): Response {
  const body = response.body;
  if (
    !body ||
    !isEventStreamContentType(response.headers.get('content-type')) ||
    // An adapter with no in-band error event has nothing to end the stream with,
    // so the response must pass through untouched.
    !supportsTerminalErrorFrame(input.adapterKey)
  ) {
    return response;
  }

  const reader = body.getReader();
  const tracker = createSseEventTracker();
  const state: TerminalErrorPumpState = {
    cancelled: false,
    relayedBytes: false,
    detector: createSseTerminalEventDetector(tracker),
    encoder: new TextEncoder(),
    // Built only when the stream actually fails, because only then is the
    // transport failure known - and only then can its identity be reported.
    buildFrames: (cause) =>
      buildStreamTerminalErrorFrames({
        adapterKey: input.adapterKey,
        message: terminalErrorMessageFor(input.message, cause)
      }),
    isClientAbort: () => Boolean(input.signal?.aborted)
  };
  const stream = createTerminalErrorBody(reader, tracker, state);

  const headers = new Headers(response.headers);
  // The terminal frame changes the body length, so the upstream value cannot stand.
  headers.delete('content-length');

  return new Response(stream, {
    headers,
    status: response.status,
    statusText: response.statusText
  });
}

interface TerminalErrorPumpState {
  cancelled: boolean;
  relayedBytes: boolean;
  detector: SseTerminalEventDetector;
  encoder: TextEncoder;
  buildFrames: (cause: unknown) => string[];
  isClientAbort: () => boolean;
}

/**
 * The configured message stays authoritative; a recognised transport code is
 * appended so the client learns why the body stopped without the server ever
 * copying a provider message into the frame.
 */
function terminalErrorMessageFor(configured: string | undefined, cause: unknown): string {
  const code = readTransportCode(cause);
  if (!code) {
    return configured ?? defaultTerminalErrorMessage;
  }

  const base = configured ?? defaultTerminalErrorMessage;
  // A configured message that already names the code is a caller's deliberate
  // wording; appending again would read as a doubled diagnostic.
  return base.includes(code) ? base : `${base} (transport: ${code})`;
}

/**
 * Walks a bounded `code`/`cause` chain and returns the first closed-set transport
 * code. Nothing else about the error is read: no message, no URL, no object.
 */
function readTransportCode(error: unknown): string | undefined {
  let current: unknown = error;

  for (let depth = 0; depth < maxTransportCauseDepth; depth += 1) {
    if (!current || typeof current !== 'object') {
      return undefined;
    }

    const record = current as { cause?: unknown; code?: unknown };
    const code = record.code;
    if (typeof code === 'string' && knownTransportCodes[code] === true) {
      return code;
    }

    // Only an explicitly closed set is ever returned, so an unknown code on this
    // record must not stop the walk - the real cause may be one level deeper.
    current = record.cause;
  }

  return undefined;
}

/**
 * The relayed body, read only as fast as the client asks for it. A pull-driven
 * stream keeps the upstream socket under the client's own backpressure: the
 * queue holds at most one chunk, so a slow consumer cannot make the gateway
 * buffer a fast provider in memory.
 *
 * That single chunk is required rather than zero: the relay hands the body to
 * `Readable.fromWeb`, which discards the bytes already read when the source
 * errors unless one chunk is queued, so a zero bound silently loses the last
 * event before a terminal frame.
 */
function createTerminalErrorBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  tracker: SseEventTracker,
  state: TerminalErrorPumpState
): ReadableStream<Uint8Array> {
  let sourceFinished = false;

  return new ReadableStream<Uint8Array>(
    {
      async pull(controller): Promise<void> {
        if (sourceFinished) {
          return;
        }

        let read: { done: false; value: Uint8Array } | { done: true };
        try {
          read = await reader.read();
        } catch (error) {
          sourceFinished = true;
          failWithTerminalError(controller, tracker, state, error);
          return;
        }

        // A consumer that cancelled while this read was in flight has nobody
        // left to receive the bytes, and the controller can no longer accept them.
        if (state.cancelled) {
          return;
        }

        if (read.done) {
          sourceFinished = true;
          controller.close();
          return;
        }

        state.detector.push(read.value);
        state.relayedBytes = state.relayedBytes || read.value.byteLength > 0;
        controller.enqueue(read.value);
      },
      async cancel(reason): Promise<void> {
        state.cancelled = true;
        sourceFinished = true;
        await reader.cancel(reason);
      }
    },
    // One chunk of slack is what `Readable.fromWeb` and Fastify need to flush
    // bytes that already arrived before the source errors; holding strictly
    // nothing makes them discard that in-flight chunk. Anything above one is
    // unbounded read-ahead again, so the bound stays here.
    { highWaterMark: 1 }
  );
}

function hasHttpStatusCode(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) {
    return false;
  }
  const statusCode = (error as { statusCode?: unknown }).statusCode;
  return typeof statusCode === 'number' && statusCode >= 400 && statusCode <= 599;
}

function failWithTerminalError(
  controller: ReadableStreamDefaultController<Uint8Array>,
  tracker: SseEventTracker,
  state: TerminalErrorPumpState,
  error: unknown
): void {
  if (state.cancelled) {
    return;
  }
  // A terminal event already reached the client: a second one would be a
  // protocol violation, so let the stream fail as before. The same holds once
  // the client is gone - there is nobody left to read a frame.
  if (state.isClientAbort() || state.detector.hasTerminalEvent()) {
    controller.error(error);
    return;
  }
  // Before the first byte the response headers are still unsent, so an error
  // that names its own HTTP status is better answered with that status.
  if (!state.relayedBytes && hasHttpStatusCode(error)) {
    controller.error(error);
    return;
  }
  const frames = state.buildFrames(error);
  if (frames.length === 0) {
    controller.error(error);
    return;
  }
  // Bytes the transport cut off were never delivered as an event; ending that
  // event first keeps the frame separate instead of appending it to the last
  // partial line the client is still holding.
  const boundary = tracker.closeRetainedEvent();
  if (boundary.length > 0) {
    controller.enqueue(boundary);
  }
  for (const frame of frames) {
    controller.enqueue(state.encoder.encode(frame));
  }
  controller.close();
}

/**
 * A dispatched event is terminal when the protocol itself says the stream is
 * over, read from the event name or from the payload's own top level. Nothing
 * below the top level counts: a tool input or provider metadata can repeat a
 * terminal word without the stream ending.
 */
function isTerminalSseEvent(event: SseEvent): boolean {
  const eventName = event.event?.trim().toLowerCase();
  if (eventName && terminalSseEventNames[eventName] === true) {
    return true;
  }

  const data = event.data.trim();
  if (data.length === 0) {
    return false;
  }
  if (data === '[DONE]') {
    return true;
  }

  const payload = parseSsePayload(data);
  if (!payload) {
    return false;
  }

  // An error chunk carries its signal in the `error` key itself - OpenAI chat
  // sends `{"error":{"message":"..."}}` with no `type` at all - so the key
  // presence is the evidence, not a value inside it.
  if (isObject(payload.error)) {
    return true;
  }

  const type = asString(payload.type)?.toLowerCase();
  return type !== undefined && terminalSsePayloadTypes[type] === true;
}

function parseSsePayload(data: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(data);
    return isObject(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function isEventStreamContentType(contentType: string | null): boolean {
  return Boolean(contentType && contentType.toLowerCase().includes('text/event-stream'));
}

function encodeSseFrame(eventName: string | undefined, data: unknown): string {
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  const dataLines = payload
    .split('\n')
    .map((line) => `data: ${line}`)
    .join('\n');
  return eventName ? `event: ${eventName}\n${dataLines}\n\n` : `${dataLines}\n\n`;
}
