import type { FastifyRequest } from 'fastify';
import type {
  GatewayConfig,
  GatewayPolicyRuleConfig,
  GatewayRequestIdentity,
  Provider,
  ProviderConfig
} from '../types';

export interface GatewayPolicyInput {
  request: FastifyRequest;
  config: GatewayConfig;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  model?: string;
  requireKnownModel?: boolean;
}

export type GatewayPolicyResult =
  | { ok: true }
  | {
      ok: false;
      statusCode: 403;
      code: 'gateway_policy_denied';
      message: string;
      details: {
        provider: Provider;
        providerName?: string;
        model?: string;
        rule: string;
      };
    };

interface EvaluatedPolicyRule {
  label: string;
  rule: GatewayPolicyRuleConfig;
}

export function evaluateGatewayPolicy(input: GatewayPolicyInput): GatewayPolicyResult {
  const policy = input.config.policy;
  if (!policy?.enabled) {
    return { ok: true };
  }

  const rules = resolvePolicyRules(input.request.gatewayIdentity, policy);
  for (const item of rules) {
    const result = evaluatePolicyRule(item, input);
    if (!result.ok) {
      return result;
    }
  }

  return { ok: true };
}

function resolvePolicyRules(
  identity: GatewayRequestIdentity | undefined,
  policy: GatewayConfig['policy']
): EvaluatedPolicyRule[] {
  const rules: EvaluatedPolicyRule[] = [
    {
      label: 'defaults',
      rule: policy.defaults
    }
  ];

  pushIdentityRule(rules, 'user', identity?.userId, policy.byUser);
  pushIdentityRule(rules, 'tenant', identity?.tenantId, policy.byTenant);
  pushIdentityRule(rules, 'organization', identity?.organizationId, policy.byOrganization);
  pushIdentityRule(rules, 'subject', identity?.subject, policy.bySubject);
  pushIdentityRule(rules, 'plan', identity?.plan, policy.byPlan);
  pushIdentityRule(rules, 'api_key', identity?.apiKeyId, policy.byApiKey);

  return rules;
}

function pushIdentityRule(
  rules: EvaluatedPolicyRule[],
  label: string,
  key: string | undefined,
  map: Record<string, GatewayPolicyRuleConfig>
): void {
  if (!key) {
    return;
  }

  const rule = map[key];
  if (!rule) {
    return;
  }

  rules.push({
    label: `${label}:${key}`,
    rule
  });
}

function evaluatePolicyRule(
  item: EvaluatedPolicyRule,
  input: GatewayPolicyInput
): GatewayPolicyResult {
  const rule = item.rule;
  const providerName = resolvePolicyProviderName(input.targetProviderConfig);
  const model = input.model;

  if (
    input.requireKnownModel &&
    !model &&
    (rule.denyModels.length > 0 || rule.denyProviderModels.length > 0)
  ) {
    return deny(
      input,
      item.label,
      'Model is missing, so model deny rules cannot be evaluated safely.'
    );
  }

  if (rule.denyProviders.includes(input.targetProvider)) {
    return deny(input, item.label, `Provider ${input.targetProvider} is denied by gateway policy.`);
  }

  if (providerName && includesNormalized(rule.denyProviderNames, providerName)) {
    return deny(input, item.label, `Provider ${providerName} is denied by gateway policy.`);
  }

  if (model && matchesAnyModelPattern(rule.denyModels, model)) {
    return deny(input, item.label, `Model ${model} is denied by gateway policy.`);
  }

  if (matchesAnyProviderModelSelector(rule.denyProviderModels, input.targetProvider, providerName, model)) {
    return deny(input, item.label, `Provider/model ${formatProviderModel(input)} is denied by gateway policy.`);
  }

  if (rule.allowProviders.length > 0 && !rule.allowProviders.includes(input.targetProvider)) {
    return deny(input, item.label, `Provider ${input.targetProvider} is not allowed by gateway policy.`);
  }

  if (rule.allowProviderNames.length > 0 && (!providerName || !includesNormalized(rule.allowProviderNames, providerName))) {
    return deny(input, item.label, `Provider ${providerName || input.targetProvider} is not allowed by gateway policy.`);
  }

  if (rule.allowModels.length > 0 && (!model || !matchesAnyModelPattern(rule.allowModels, model))) {
    return deny(input, item.label, `Model ${model || '<missing>'} is not allowed by gateway policy.`);
  }

  if (
    rule.allowProviderModels.length > 0 &&
    !matchesAnyProviderModelSelector(rule.allowProviderModels, input.targetProvider, providerName, model)
  ) {
    return deny(input, item.label, `Provider/model ${formatProviderModel(input)} is not allowed by gateway policy.`);
  }

  return { ok: true };
}

function deny(input: GatewayPolicyInput, rule: string, message: string): GatewayPolicyResult {
  return {
    ok: false,
    statusCode: 403,
    code: 'gateway_policy_denied',
    message,
    details: {
      provider: input.targetProvider,
      providerName: resolvePolicyProviderName(input.targetProviderConfig),
      model: input.model,
      rule
    }
  };
}

function includesNormalized(values: string[], value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return values.some((item) => item.trim().toLowerCase() === normalized);
}

function matchesAnyModelPattern(patterns: string[], model: string): boolean {
  return patterns.some((pattern) => matchesWildcard(pattern, model));
}

function matchesAnyProviderModelSelector(
  selectors: string[],
  provider: Provider,
  providerName: string | undefined,
  model: string | undefined
): boolean {
  if (!model) {
    return false;
  }

  return selectors.some((selector) => {
    const trimmed = selector.trim();
    if (!trimmed) {
      return false;
    }

    const slashIndex = trimmed.indexOf('/');
    if (slashIndex <= 0 || slashIndex >= trimmed.length - 1) {
      return matchesWildcard(trimmed, model);
    }

    const providerSelector = trimmed.slice(0, slashIndex).trim().toLowerCase();
    const modelSelector = trimmed.slice(slashIndex + 1).trim();
    if (!matchesProviderSelector(providerSelector, provider, providerName)) {
      return false;
    }

    return matchesWildcard(modelSelector, model);
  });
}

function matchesProviderSelector(
  selector: string,
  provider: Provider,
  providerName: string | undefined
): boolean {
  return selector === provider || selector === providerName?.trim().toLowerCase();
}

function matchesWildcard(pattern: string, value: string): boolean {
  const trimmed = pattern.trim();
  if (!trimmed) {
    return false;
  }

  if (trimmed === '*') {
    return true;
  }

  if (!trimmed.includes('*')) {
    return trimmed === value;
  }

  const escaped = trimmed
    .split('*')
    .map(escapeRegExp)
    .join('.*');
  return new RegExp(`^${escaped}$`).test(value);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function formatProviderModel(input: GatewayPolicyInput): string {
  const provider = resolvePolicyProviderName(input.targetProviderConfig) || input.targetProvider;
  return `${provider}/${input.model || '<missing>'}`;
}

function resolvePolicyProviderName(providerConfig: ProviderConfig | undefined): string | undefined {
  return providerConfig?.credentialSourceProviderName || providerConfig?.name;
}
