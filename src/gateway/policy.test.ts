import type { FastifyRequest } from 'fastify';
import { describe, expect, it } from 'vitest';
import type { GatewayConfig, ProviderConfig } from '../types';
import { evaluateGatewayPolicy } from './policy';

describe('gateway routing policy', () => {
  it('applies matching tenant rules and lets deny override allow', () => {
    const config = createConfig();
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        allowProviders: ['openai']
      }),
      byTenant: {
        'tenant-a': createPolicyRuleConfig({
          allowProviderNames: ['openai-main'],
          denyModels: ['gpt-4-expensive']
        })
      }
    });

    const allowed = evaluateGatewayPolicy({
      request: createRequest({ tenantId: 'tenant-a', userId: 'user-1' }),
      config,
      targetProvider: 'openai',
      targetProviderConfig: createProviderConfig('openai-main'),
      model: 'gpt-4-mini'
    });
    expect(allowed.ok).toBe(true);

    const denied = evaluateGatewayPolicy({
      request: createRequest({ tenantId: 'tenant-a', userId: 'user-1' }),
      config,
      targetProvider: 'openai',
      targetProviderConfig: createProviderConfig('openai-main'),
      model: 'gpt-4-expensive'
    });
    expect(denied).toMatchObject({
      ok: false,
      statusCode: 403,
      code: 'gateway_policy_denied',
      details: {
        rule: 'tenant:tenant-a',
        model: 'gpt-4-expensive'
      }
    });
  });

  it('supports provider/model selectors with provider names and wildcards', () => {
    const config = createConfig();
    config.policy = createPolicyConfig({
      enabled: true,
      defaults: createPolicyRuleConfig({
        allowProviderModels: ['openai-main/gpt-4-*']
      })
    });

    expect(
      evaluateGatewayPolicy({
        request: createRequest(),
        config,
        targetProvider: 'openai',
        targetProviderConfig: createProviderConfig('openai-main'),
        model: 'gpt-4-mini'
      }).ok
    ).toBe(true);

    expect(
      evaluateGatewayPolicy({
        request: createRequest(),
        config,
        targetProvider: 'openai',
        targetProviderConfig: createProviderConfig('openai-main'),
        model: 'gpt-5-mini'
      })
    ).toMatchObject({
      ok: false,
      statusCode: 403,
      details: {
        rule: 'defaults'
      }
    });
  });
});

function createRequest(identity: Partial<FastifyRequest['gatewayIdentity']> = {}): FastifyRequest {
  return {
    gatewayIdentity:
      Object.keys(identity).length > 0
        ? {
            source: 'trusted_header',
            billingSubjectKey: identity.userId || identity.subject || 'subject',
            ...identity
          }
        : undefined
  } as FastifyRequest;
}

function createConfig(): GatewayConfig {
  return {
    providers: [],
    defaultTargetProviders: [],
    openaiBaseUrl: 'https://api.openai.com/v1'
  } as unknown as GatewayConfig;
}

function createProviderConfig(name: string): ProviderConfig {
  return {
    name,
    type: 'openai_responses',
    models: ['gpt-4-mini'],
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

function createPolicyConfig(
  overrides: Partial<GatewayConfig['policy']> = {}
): GatewayConfig['policy'] {
  return {
    enabled: false,
    defaults: createPolicyRuleConfig(),
    byUser: {},
    byTenant: {},
    byOrganization: {},
    bySubject: {},
    byPlan: {},
    byApiKey: {},
    ...overrides
  };
}

function createPolicyRuleConfig(
  overrides: Partial<GatewayConfig['policy']['defaults']> = {}
): GatewayConfig['policy']['defaults'] {
  return {
    allowProviders: [],
    denyProviders: [],
    allowProviderNames: [],
    denyProviderNames: [],
    allowModels: [],
    denyModels: [],
    allowProviderModels: [],
    denyProviderModels: [],
    ...overrides
  };
}
