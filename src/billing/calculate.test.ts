import { describe, expect, it } from 'vitest';

import {
  buildBillingHeaders,
  calculateUsageBilling,
  createProviderReportedCostBilling
} from './calculate';
import type { BillingConfig } from '../types';

const config: BillingConfig = {
  enabled: true,
  currency: 'USD',
  rates: {
    openai: {
      inputPerMillionUsd: 1,
      outputPerMillionUsd: 2,
      cacheReadPerMillionUsd: 0.1,
      cacheWritePerMillionUsd: 1.25
    },
    anthropic: {
      inputPerMillionUsd: 3,
      outputPerMillionUsd: 15,
      cacheReadPerMillionUsd: 0.3,
      cacheWritePerMillionUsd: 3.75
    },
    gemini: {
      inputPerMillionUsd: 1,
      outputPerMillionUsd: 2,
      cacheReadPerMillionUsd: 0.1,
      cacheWritePerMillionUsd: 1.25
    }
  }
};

describe('calculateUsageBilling total token normalization', () => {
  it('does not add OpenAI cache tokens to total when total_tokens is absent', () => {
    const result = calculateUsageBilling(
      'openai',
      {
        input_tokens: 100,
        output_tokens: 25,
        cache_read_tokens: 40,
        cache_write_tokens: 10
      },
      config
    );

    expect(result.usage.total_tokens).toBe(125);
  });

  it('adds Anthropic cache tokens to total when total_tokens is absent', () => {
    const result = calculateUsageBilling(
      'anthropic',
      {
        input_tokens: 100,
        output_tokens: 25,
        cache_read_tokens: 40,
        cache_write_tokens: 10
      },
      config
    );

    expect(result.usage.total_tokens).toBe(175);
  });

  it('uses zero rates for a custom provider without an explicit billing rate', () => {
    const result = calculateUsageBilling(
      'xai',
      {
        input_tokens: 100,
        output_tokens: 25
      },
      config
    );

    expect(result.rates).toEqual({
      input_per_million_usd: 0,
      output_per_million_usd: 0,
      cache_read_per_million_usd: 0,
      cache_write_per_million_usd: 0
    });
    expect(result.cost.total).toBe(0);
  });

  it('uses a size-specific video rate before the model fallback rate', () => {
    const result = calculateUsageBilling(
      'xai',
      { video_seconds: 8, video_size: ' 1024X1792 ' },
      config,
      {
        inputPerMillionUsd: 0,
        outputPerMillionUsd: 0,
        videoPerSecondUsd: 0.3,
        videoPerSecondUsdBySize: {
          '1024x1792': 0.5
        }
      }
    );

    expect(result.usage.video_seconds).toBe(8);
    expect(result.usage.video_size).toBe('1024x1792');
    expect(result.cost.media).toBe(4);
    expect(result.cost.total).toBe(4);
    expect(buildBillingHeaders(result)).toMatchObject({
      'x-gateway-billing-video-seconds': '8',
      'x-gateway-billing-video-size': '1024x1792',
      'x-gateway-billing-video-per-second-usd': '0.50000000',
      'x-gateway-billing-media-cost': '4.00000000',
      'x-gateway-billing-total-cost': '4.00000000'
    });
  });

  it('uses an exact provider-reported media cost when available', () => {
    const result = createProviderReportedCostBilling('xai', 0.125, config);
    expect(result.cost.media).toBe(0.125);
    expect(result.cost.total).toBe(0.125);
  });

  it('preserves provider-reported cost precision down to one xAI tick', () => {
    const result = createProviderReportedCostBilling('xai', 1 / 10_000_000_000, config);

    expect(result.cost.media).toBe(0.0000000001);
    expect(result.cost.total).toBe(0.0000000001);
    expect(buildBillingHeaders(result)).toMatchObject({
      'x-gateway-billing-media-cost': '0.0000000001',
      'x-gateway-billing-total-cost': '0.0000000001'
    });
  });
});
