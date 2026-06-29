import { EventEmitter } from 'node:events';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { describe, expect, it } from 'vitest';
import { createClientDisconnectSignal } from './client-disconnect';

describe('createClientDisconnectSignal', () => {
  it('aborts when the response closes before finishing', () => {
    const requestRaw = new EventEmitter() as EventEmitter & { aborted: boolean };
    requestRaw.aborted = false;
    const replyRaw = new EventEmitter() as EventEmitter & {
      destroyed: boolean;
      writableEnded: boolean;
    };
    replyRaw.destroyed = false;
    replyRaw.writableEnded = false;

    const signal = createClientDisconnectSignal(
      { raw: requestRaw } as FastifyRequest,
      { raw: replyRaw } as FastifyReply
    );

    replyRaw.emit('close');

    expect(signal.aborted).toBe(true);
    expect(signal.reason).toBeInstanceOf(Error);
    expect((signal.reason as Error).message).toContain('Client connection closed');
  });

  it('does not abort after a normal response finish', () => {
    const requestRaw = new EventEmitter() as EventEmitter & { aborted: boolean };
    requestRaw.aborted = false;
    const replyRaw = new EventEmitter() as EventEmitter & {
      destroyed: boolean;
      writableEnded: boolean;
    };
    replyRaw.destroyed = false;
    replyRaw.writableEnded = true;

    const signal = createClientDisconnectSignal(
      { raw: requestRaw } as FastifyRequest,
      { raw: replyRaw } as FastifyReply
    );

    replyRaw.emit('finish');
    replyRaw.writableEnded = false;
    replyRaw.emit('close');

    expect(signal.aborted).toBe(false);
  });
});

