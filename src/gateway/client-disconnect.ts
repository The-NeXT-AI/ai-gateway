import type { FastifyReply, FastifyRequest } from 'fastify';

const clientDisconnectMessage = 'Client connection closed before the gateway response completed.';

export function createClientDisconnectSignal(
  request: FastifyRequest,
  reply: FastifyReply
): AbortSignal {
  const controller = new AbortController();
  let settled = false;

  const cleanup = () => {
    request.raw.off('aborted', handleRequestAborted);
    reply.raw.off('finish', handleResponseFinish);
    reply.raw.off('close', handleResponseClose);
  };

  const abort = () => {
    if (settled) {
      return;
    }
    settled = true;
    cleanup();
    controller.abort(new Error(clientDisconnectMessage));
  };

  const finish = () => {
    if (settled) {
      return;
    }
    settled = true;
    cleanup();
  };

  const handleRequestAborted = () => {
    abort();
  };

  const handleResponseFinish = () => {
    finish();
  };

  const handleResponseClose = () => {
    if (reply.raw.writableEnded) {
      finish();
      return;
    }
    abort();
  };

  request.raw.once('aborted', handleRequestAborted);
  reply.raw.once('finish', handleResponseFinish);
  reply.raw.once('close', handleResponseClose);

  if (request.raw.aborted || (reply.raw.destroyed && !reply.raw.writableEnded)) {
    abort();
  }

  return controller.signal;
}

