import { describe, expect, it } from 'vitest';
import { parseSseChunks } from './sse';

describe('parseSseChunks', () => {
  it('stops without throwing when the abort signal fires', async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('data: hello\n\n'));
      }
    });
    const iterator = parseSseChunks(new Response(body), controller.signal);

    await expect(iterator.next()).resolves.toMatchObject({
      done: false,
      value: {
        data: 'hello'
      }
    });

    const pending = iterator.next();
    await Promise.resolve();
    controller.abort(new Error('client disconnected'));

    await expect(pending).resolves.toMatchObject({
      done: true
    });
  });
});
