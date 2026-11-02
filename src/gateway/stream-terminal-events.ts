/**
 * Byte-level SSE event boundaries.
 *
 * The gateway relays upstream event streams byte for byte, so the only trustworthy
 * evidence that an event reached the client is a blank line that ended it. This
 * module answers exactly that question without re-encoding or copying the relayed
 * bytes: it scans raw bytes, keeps only what is still undelivered, and reports the
 * events a client parser would actually dispatch.
 *
 * Splitting on `\n` / `\r` never cuts a UTF-8 sequence, because no multi-byte
 * sequence contains those bytes, so a completed line always decodes cleanly.
 */

export interface SseEvent {
  event?: string;
  /** Joined `data:` field values, newline separated, exactly as a parser would read them. */
  data: string;
}

export interface SseEventTracker {
  /**
   * Consumes raw bytes and returns the events dispatched by this chunk: only
   * events a blank line terminated. A trailing field line or a partially written
   * event is held back, never reported.
   */
  push(chunk: Uint8Array): SseEvent[];
  /**
   * Bytes held since the last dispatched event boundary; 0 means the client
   * currently holds nothing between events.
   */
  pendingBytes(): number;
  /**
   * The shortest bytes to append so that whatever is held becomes a dispatched
   * event of its own. Empty when nothing is held.
   */
  closeRetainedEvent(): Uint8Array;
}

const lfByte = 0x0a;
const crByte = 0x0d;
const spaceByte = 0x20;

export function createSseEventTracker(): SseEventTracker {
  const decoder = new TextDecoder();
  let lineBytes: number[] = [];
  let eventName: string | undefined;
  let dataLines: string[] = [];
  // Held bytes are counted, not retained: the relay already forwards them, so the
  // tracker only needs to know whether anything is still undelivered and whether
  // the last one was a line ending.
  let pendingBytes = 0;
  let lastPendingByte: number | undefined;
  // A chunk can end between `\r` and `\n`, where the pair is one line ending and
  // must not be read as a line ending followed by a blank line.
  let pendingCrLf = false;

  function dispatch(): SseEvent | undefined {
    const event =
      eventName === undefined && dataLines.length === 0
        ? undefined
        : { event: eventName, data: dataLines.join('\n') };
    eventName = undefined;
    dataLines = [];
    return event;
  }

  function endLine(out: SseEvent[]): void {
    if (lineBytes.length === 0) {
      // A blank line is the only event boundary; anything else is a field line.
      const event = dispatch();
      if (event) {
        out.push(event);
      }
      return;
    }

    const line = decoder.decode(Uint8Array.from(lineBytes));
    lineBytes = [];

    // A comment line is ignored, and a line with no colon is a field with an
    // empty value, which carries nothing this tracker needs.
    // `String#indexOf` needs the character, not the byte value: a numeric
    // argument never matches a string here.
    const separator = line.indexOf(':');
    if (separator <= 0) {
      return;
    }

    const field = line.slice(0, separator);
    const rawValue = line.slice(separator + 1);
    const value = rawValue.charCodeAt(0) === spaceByte ? rawValue.slice(1) : rawValue;

    if (field === 'event') {
      eventName = value.trim();
    } else if (field === 'data') {
      dataLines.push(value);
    }
  }

  return {
    push(chunk: Uint8Array): SseEvent[] {
      if (chunk.length === 0) {
        return [];
      }

      const events: SseEvent[] = [];

      for (let index = 0; index < chunk.length; index += 1) {
        const byte = chunk[index];

        if (pendingCrLf && byte === lfByte) {
          // Second half of a line ending split across chunks; the line was already
          // ended when the `\r` arrived.
          pendingCrLf = false;
          pendingBytes += 1;
          lastPendingByte = byte;
          continue;
        }
        pendingCrLf = false;

        if (byte === lfByte) {
          const before = events.length;
          endLine(events);
          // A line ending that dispatched an event ends the held region; anything
          // after it is new, and the blank line itself holds nothing.
          pendingBytes = events.length > before ? 0 : pendingBytes + 1;
          lastPendingByte = byte;
          continue;
        }

        if (byte === crByte) {
          const before = events.length;
          endLine(events);
          if (chunk[index + 1] === lfByte) {
            // `\r\n` is one line ending, not a line ending plus a blank line.
            index += 1;
            pendingBytes = events.length > before ? 0 : pendingBytes + 2;
            lastPendingByte = lfByte;
          } else {
            pendingCrLf = true;
            pendingBytes = events.length > before ? 0 : pendingBytes + 1;
            lastPendingByte = byte;
          }
          continue;
        }

        lineBytes.push(byte);
        pendingBytes += 1;
        lastPendingByte = byte;
      }

      return events;
    },

    pendingBytes: () => pendingBytes,

    closeRetainedEvent(): Uint8Array {
      if (pendingBytes === 0 && !pendingCrLf) {
        return new Uint8Array(0);
      }
      // The held bytes end mid-line (or on a `\r` whose `\n` never came), so one
      // line ending finishes them and one more makes the blank line that
      // dispatches them.
      return Uint8Array.from([lfByte, lfByte]);
    }
  };
}
