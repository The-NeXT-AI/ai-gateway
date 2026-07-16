import { createConnection } from 'node:net';

const LOOPBACK_HOST = '127.0.0.1';
const RETRY_DELAY_MS = 10;
const DEFAULT_TIMEOUT_MS = 1000;

export async function waitForLoopbackListener(
  port: number,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown;

  while (Date.now() < deadline) {
    try {
      await connectToLoopback(port, Math.max(1, deadline - Date.now()));
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => {
        setTimeout(resolve, Math.min(RETRY_DELAY_MS, Math.max(0, deadline - Date.now())));
      });
    }
  }

  throw new Error(`Loopback listener on port ${port} was not ready within ${timeoutMs}ms`, {
    cause: lastError
  });
}

function connectToLoopback(port: number, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const socket = createConnection({ host: LOOPBACK_HOST, port });

    socket.setTimeout(timeoutMs, () => {
      socket.destroy();
      reject(new Error(`Timed out connecting to loopback port ${port}`));
    });
    socket.once('error', (error) => {
      socket.destroy();
      reject(error);
    });
    socket.once('connect', () => {
      socket.setTimeout(0);
      socket.end();
      resolve();
    });
  });
}
