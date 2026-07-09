import { describe, expect, it } from 'vitest';
import { TargetAdapterRegistry } from './registry';
import type { ProviderConfig, TargetAdapter } from '../types';

describe('TargetAdapterRegistry', () => {
  it('prefers provider type adapters over provider fallback adapters', () => {
    const registry = new TargetAdapterRegistry();
    const fallbackAdapter = createTargetAdapter('acme_fallback', {
      providerFallback: true
    });
    const preciseAdapter = createTargetAdapter('acme_chat', {
      providerTypes: ['acme_chat']
    });

    registry.register(fallbackAdapter);
    registry.register(preciseAdapter);

    expect(
      registry.get(
        'acme',
        {
          type: 'acme_chat'
        } as ProviderConfig
      )
    ).toBe(preciseAdapter);
  });
});

function createTargetAdapter(
  key: string,
  options: Pick<TargetAdapter, 'providerFallback' | 'providerTypes'> = {}
): TargetAdapter {
  return {
    key,
    provider: 'acme',
    ...options,
    buildRequestFromStandard() {
      return {
        ok: true,
        value: {
          url: 'https://api.acme.test/messages',
          headers: {},
          body: {}
        }
      };
    },
    toStandardResponse(payload) {
      return {
        ok: true,
        value: payload
      };
    }
  };
}
