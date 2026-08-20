import { vi } from 'vitest';

vi.mock('undici', async (importOriginal) => {
  const undici = await importOriginal<typeof import('undici')>();
  return {
    ...undici,
    fetch: (...args: Parameters<typeof globalThis.fetch>) => globalThis.fetch(...args)
  };
});
