import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import {
  closeProviderHealthScheduler,
  initializeProviderHealthScheduler,
  runScheduledProviderHealthChecks
} from './provider-health-scheduler';

describe('provider health scheduler', () => {
  afterEach(() => {
    closeProviderHealthScheduler();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('runs a scheduled health check batch and updates provider health', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const logger = {
      info: vi.fn()
    };
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-key',
          baseurl: 'https://openai.example/v1/',
          models: ['gpt-4.1-mini']
        }
      ],
      providerHealthCheck: {
        enabled: true,
        intervalMs: 10000,
        timeoutMs: 1000
      }
    });

    const results = await runScheduledProviderHealthChecks(config, logger);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(upstreamUrl).toBe('https://openai.example/v1/models');
    expect(upstreamInit.method).toBe('GET');
    expect(upstreamInit.headers).toMatchObject({
      authorization: 'Bearer provider-key'
    });
    expect(results[0]).toMatchObject({
      provider: 'openai',
      providerName: 'openai-main',
      ok: true,
      statusCode: 200
    });
    expect(config.providers[0]?.health).toMatchObject({
      status: 'healthy',
      available: true
    });
    expect(logger.info).toHaveBeenCalledWith(
      {
        checked: 1,
        healthy: 1,
        failed: 0
      },
      'Scheduled provider health check completed.'
    );
  });

  it('checks xAI providers through the bearer-authenticated models endpoint', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'xai-video',
          type: 'xai_video_generations',
          apikey: 'xai-provider-key',
          baseurl: 'https://api.x.ai/v1/',
          models: ['grok-imagine-video']
        }
      ],
      providerHealthCheck: {
        enabled: true,
        intervalMs: 10000,
        timeoutMs: 1000
      }
    });

    const results = await runScheduledProviderHealthChecks(config);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [upstreamUrl, upstreamInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(upstreamUrl).toBe('https://api.x.ai/v1/models');
    expect(upstreamInit.headers).toMatchObject({
      authorization: 'Bearer xai-provider-key'
    });
    expect(results[0]).toMatchObject({
      provider: 'xai',
      providerName: 'xai-video',
      ok: true,
      statusCode: 200
    });
  });

  it('does not schedule checks when disabled', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-key',
          baseurl: 'https://openai.example/v1/',
          models: ['gpt-4.1-mini']
        }
      ]
    });

    initializeProviderHealthScheduler(config);
    await vi.runOnlyPendingTimersAsync();

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('starts enabled scheduler after the configured initial delay', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-key',
          baseurl: 'https://openai.example/v1/',
          models: ['gpt-4.1-mini']
        }
      ],
      providerHealthCheck: {
        enabled: true,
        intervalMs: 10000,
        timeoutMs: 1000,
        initialDelayMs: 50
      }
    });

    initializeProviderHealthScheduler(config);
    await vi.advanceTimersByTimeAsync(49);
    expect(fetchMock).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
