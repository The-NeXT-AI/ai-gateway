import { afterEach, describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import type { ProviderConfig } from '../types';
import {
  acquireProviderConcurrencySlot,
  resetProviderConcurrencyForTests
} from './upstream-concurrency';

describe('gateway upstream concurrency', () => {
  afterEach(() => {
    resetProviderConcurrencyForTests();
  });

  it('times out queued requests for the same provider when the provider slot is occupied', async () => {
    const config = parseGatewayConfigFromRaw({
      upstreamConcurrency: {
        enabled: true,
        maxInFlightPerProvider: 1,
        queueTimeoutMs: 1
      }
    });
    const provider = createProviderConfig('openai-main');
    const first = await acquireProviderConcurrencySlot(config, 'openai', provider);
    expect(first.ok).toBe(true);
    if (!first.ok) {
      return;
    }

    const second = await acquireProviderConcurrencySlot(config, 'openai', provider);

    expect(second).toMatchObject({
      ok: false,
      status: 429,
      message: 'Provider upstream concurrency limit exceeded.',
      details: {
        provider: 'openai',
        providerName: 'openai-main',
        maxInFlight: 1,
        queueTimeoutMs: 1
      }
    });
    first.release();
  });

  it('keeps named providers isolated from each other', async () => {
    const config = parseGatewayConfigFromRaw({
      upstreamConcurrency: {
        enabled: true,
        maxInFlightPerProvider: 1,
        queueTimeoutMs: 1
      }
    });
    const first = await acquireProviderConcurrencySlot(config, 'openai', createProviderConfig('openai-a'));
    const second = await acquireProviderConcurrencySlot(config, 'openai', createProviderConfig('openai-b'));

    expect(first.ok).toBe(true);
    expect(second.ok).toBe(true);
    if (first.ok) {
      first.release();
    }
    if (second.ok) {
      second.release();
    }
  });

  it('aborts queued requests when the client disconnect signal fires', async () => {
    const config = parseGatewayConfigFromRaw({
      upstreamConcurrency: {
        enabled: true,
        maxInFlightPerProvider: 1,
        queueTimeoutMs: 1000
      }
    });
    const provider = createProviderConfig('openai-main');
    const first = await acquireProviderConcurrencySlot(config, 'openai', provider);
    expect(first.ok).toBe(true);
    if (!first.ok) {
      return;
    }

    const controller = new AbortController();
    const queued = acquireProviderConcurrencySlot(config, 'openai', provider, controller.signal);
    controller.abort(new Error('client disconnected'));
    const result = await queued;

    expect(result).toMatchObject({
      ok: false,
      status: 499,
      aborted: true,
      message: 'Client connection closed before acquiring provider concurrency slot.',
      details: {
        provider: 'openai',
        providerName: 'openai-main',
        maxInFlight: 1,
        queueTimeoutMs: 1000
      }
    });

    first.release();
    const next = await acquireProviderConcurrencySlot(config, 'openai', provider);
    expect(next.ok).toBe(true);
    if (next.ok) {
      next.release();
    }
  });
});

function createProviderConfig(name: string): ProviderConfig {
  return {
    name,
    type: 'openai_responses',
    models: ['gpt-test'],
    extraHeaders: {
      default: {},
      byModel: {}
    },
    extraBody: {
      default: {},
      byModel: {}
    },
    billing: {
      byModel: {}
    }
  };
}
