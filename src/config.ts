import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type {
  AgentConfig,
  AgentEventQueueConfig,
  AgentEventWebhookConfig,
  AgentExternalSourceConfig,
  AgentMcpStdioMessageMode,
  AgentRetryPolicyConfig,
  AgentRuntimeConfig,
  AgentStorageConfig,
  AgentMcpServerConfig,
  AgentMcpServerTransport,
  BillingRate,
  BillingTier,
  BillingQueueConfig,
  BillingWebhookConfig,
  GatewayAuthConfig,
  GatewayConfig,
  GatewayConfigExternalSourceConfig,
  GatewayConfigExternalSourceMethod,
  GatewayConfigExternalSourceTransport,
  GatewayCorsConfig,
  GatewayExternalEventSinkTransport,
  GatewayHealthAwareRoutingConfig,
  GatewayIdempotencyConfig,
  GatewayMetricsConfig,
  GatewayModelListConfig,
  GatewayPolicyConfig,
  GatewayPolicyRuleConfig,
  GatewayPrecheckConfig,
  GatewayPrecheckScope,
  GatewayPrecheckStorageType,
  GatewayPrecheckSubject,
  GatewayRateLimitDimensionConfig,
  GatewayRateLimitMetric,
  GatewayRoutingConfig,
  GatewayTransparentToolExecutionConfig,
  GatewayTransparentToolUnknownPolicy,
  GatewayUpstreamCircuitBreakerConfig,
  GatewayUpstreamConcurrencyConfig,
  GatewayUpstreamRetryConfig,
  McpGatewayConfig,
  McpGatewayGuardrailsConfig,
  McpGatewayOAuthConfig,
  McpGatewayPrincipalConfig,
  McpGatewayWebSocketAuthConfig,
  McpGatewayWebSocketConfig,
  McpServerExposure,
  ModelScopedBillingConfig,
  ModelScopedBodyConfig,
  ModelScopedHeadersConfig,
  Provider,
  ProviderPluginConfig,
  ProviderPluginCodexOAuthConfig,
  ProviderPluginDeepSeekThinkingConfig,
  ProviderPluginMutationConfig,
  ProviderPluginResponseMutationConfig,
  ProviderType,
  ProviderConfig,
  ProviderExternalSourceConfig,
  ProviderHealthCheckSchedulerConfig,
  ProviderHealthConfig,
  ProviderHealthStatus,
  VirtualModelProfileConfig,
  RawTraceCaptureMode,
  RawTraceConfig,
  RawTraceSyncConfig
} from './types';
import { parseProvider, parseProviderList, providerFromProviderType, trimTrailingSlash } from './utils';

const defaultConfigFileName = 'gateway.config.json';
const defaultCodexOauthTokenEndpoint = 'https://auth.openai.com/oauth/token';
const defaultCodexOauthClientId = 'app_EMoamEEZ73f0CkXaXp7hrann';
const defaultCodexOauthScope =
  'openid profile email offline_access api.connectors.read api.connectors.invoke';
const requiredCodexOauthScopes = ['api.connectors.read', 'api.connectors.invoke'];
const defaultBodyLimitBytes = 50 * 1024 * 1024;
const defaultCorsAllowedHeaders = [
  'Content-Type',
  'Authorization',
  'X-API-Key',
  'X-Codex-Access-Token',
  'Anthropic-Version',
  'Anthropic-Beta',
  'X-Gateway-Model-List-Format'
];
const defaultCorsAllowedMethods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'];
const defaultAgentLlmRetryConfig: AgentRetryPolicyConfig = {
  maxAttempts: 3,
  baseDelayMs: 200,
  maxDelayMs: 2000,
  backoffMultiplier: 2,
  jitterMs: 100
};
const defaultAgentToolRetryConfig: AgentRetryPolicyConfig = {
  maxAttempts: 2,
  baseDelayMs: 150,
  maxDelayMs: 1500,
  backoffMultiplier: 2,
  jitterMs: 50
};

interface BillingRateJsonConfig {
  inputPerMillionUsd?: unknown;
  outputPerMillionUsd?: unknown;
  cacheReadPerMillionUsd?: unknown;
  cacheWritePerMillionUsd?: unknown;
  tiers?: unknown;
}

interface BillingTierJsonConfig {
  upToTokens?: unknown;
  perMillionUsd?: unknown;
}

interface ProviderJsonConfig {
  name?: unknown;
  type?: unknown;
  provider?: unknown;
  apikey?: unknown;
  apiKey?: unknown;
  apikeyEnv?: unknown;
  apiKeyEnv?: unknown;
  baseurl?: unknown;
  baseUrl?: unknown;
  models?: unknown;
  openaiChatToolsFormat?: unknown;
  chatToolsFormat?: unknown;
  toolsFormat?: unknown;
  openaiChatStreamUsage?: unknown;
  chatStreamUsage?: unknown;
  streamUsage?: unknown;
  extraHeaders?: unknown;
  extraBody?: unknown;
  billing?: unknown;
  health?: unknown;
  status?: unknown;
  available?: unknown;
  priority?: unknown;
  latencyMs?: unknown;
  latency_ms?: unknown;
  checkedAt?: unknown;
  checked_at?: unknown;
}

interface ProviderHealthJsonConfig {
  status?: unknown;
  available?: unknown;
  priority?: unknown;
  latencyMs?: unknown;
  latency_ms?: unknown;
  checkedAt?: unknown;
  checked_at?: unknown;
}

interface GatewayPrecheckRuleJsonConfig {
  enabled?: unknown;
  name?: unknown;
  metric?: unknown;
  windowMs?: unknown;
  windowSeconds?: unknown;
  subject?: unknown;
  scope?: unknown;
  headerName?: unknown;
  max?: unknown;
  maxRequests?: unknown;
  maxTokens?: unknown;
  maxCostUsd?: unknown;
  rpm?: unknown;
  rpd?: unknown;
  tpm?: unknown;
  tpd?: unknown;
  ipm?: unknown;
  limits?: unknown;
}

interface GatewayPrecheckEstimationJsonConfig {
  charsPerToken?: unknown;
  defaultMaxOutputTokens?: unknown;
}

interface GatewayPrecheckStorageJsonConfig {
  type?: unknown;
  backend?: unknown;
}

interface GatewayPrecheckJsonConfig {
  enabled?: unknown;
  rateLimit?: unknown;
  quota?: unknown;
  budget?: unknown;
  estimation?: unknown;
  storage?: unknown;
}

interface GatewayHealthAwareRoutingJsonConfig {
  enabled?: unknown;
  skipUnavailable?: unknown;
  unhealthyStatuses?: unknown;
  preferHealthy?: unknown;
  preferLowerLatency?: unknown;
}

interface ProviderHealthCheckSchedulerJsonConfig {
  enabled?: unknown;
  intervalMs?: unknown;
  intervalSeconds?: unknown;
  timeoutMs?: unknown;
  timeoutSeconds?: unknown;
  initialDelayMs?: unknown;
  initialDelaySeconds?: unknown;
}

interface GatewayMetricsJsonConfig {
  enabled?: unknown;
  includeProviderHealth?: unknown;
}

interface GatewayCorsJsonConfig {
  enabled?: unknown;
  origin?: unknown;
  origins?: unknown;
  allowedHeaders?: unknown;
  allowedMethods?: unknown;
  allowCredentials?: unknown;
  maxAgeSeconds?: unknown;
  maxAge?: unknown;
}

interface GatewayIdempotencyJsonConfig {
  enabled?: unknown;
  headerName?: unknown;
  ttlMs?: unknown;
  ttlSeconds?: unknown;
  maxEntries?: unknown;
  cacheErrorResponses?: unknown;
}

interface GatewayUpstreamConcurrencyJsonConfig {
  enabled?: unknown;
  maxInFlightPerProvider?: unknown;
  queueTimeoutMs?: unknown;
  queueTimeoutSeconds?: unknown;
}

interface GatewayUpstreamCircuitBreakerJsonConfig {
  enabled?: unknown;
  failureThreshold?: unknown;
  cooldownMs?: unknown;
  cooldownSeconds?: unknown;
  failureStatusCodes?: unknown;
}

interface GatewayUpstreamRetryJsonConfig {
  enabled?: unknown;
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  baseDelaySeconds?: unknown;
  maxDelayMs?: unknown;
  maxDelaySeconds?: unknown;
  backoffMultiplier?: unknown;
  jitterMs?: unknown;
  jitterSeconds?: unknown;
  retryStatusCodes?: unknown;
}

interface GatewayTransparentToolExecutionJsonConfig {
  enabled?: unknown;
  maxTurns?: unknown;
  maxToolCalls?: unknown;
  requireClientDeclaration?: unknown;
  unknownToolPolicy?: unknown;
  allowTools?: unknown;
  denyTools?: unknown;
}

interface GatewayRoutingJsonConfig {
  healthAware?: unknown;
  policy?: unknown;
  preferSourceProviderForBareModels?: unknown;
  preferSourceProviderForBareModel?: unknown;
}

interface GatewayModelListJsonConfig {
  bareModelIds?: unknown;
  bareModelId?: unknown;
  bare_model_ids?: unknown;
}

interface GatewayPolicyRuleJsonConfig {
  allowProviders?: unknown;
  denyProviders?: unknown;
  allowProviderNames?: unknown;
  denyProviderNames?: unknown;
  allowModels?: unknown;
  denyModels?: unknown;
  allowProviderModels?: unknown;
  denyProviderModels?: unknown;
}

interface GatewayPolicyJsonConfig extends GatewayPolicyRuleJsonConfig {
  enabled?: unknown;
  defaults?: unknown;
  byUser?: unknown;
  byTenant?: unknown;
  byOrganization?: unknown;
  bySubject?: unknown;
  byPlan?: unknown;
  byApiKey?: unknown;
}

interface BillingQueueJsonConfig {
  enabled?: unknown;
  queueName?: unknown;
  jobName?: unknown;
  removeOnComplete?: unknown;
  removeOnFail?: unknown;
}

interface BillingWebhookJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  timeoutMs?: unknown;
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  baseDelaySeconds?: unknown;
  maxDelayMs?: unknown;
  maxDelaySeconds?: unknown;
  requireAck?: unknown;
  websocketRequireAck?: unknown;
  headers?: unknown;
}

interface AgentEventWebhookJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  timeoutMs?: unknown;
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  baseDelaySeconds?: unknown;
  maxDelayMs?: unknown;
  maxDelaySeconds?: unknown;
  requireAck?: unknown;
  websocketRequireAck?: unknown;
  headers?: unknown;
}

interface RawTraceSyncJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  timeoutMs?: unknown;
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  baseDelaySeconds?: unknown;
  maxDelayMs?: unknown;
  maxDelaySeconds?: unknown;
  requireAck?: unknown;
  websocketRequireAck?: unknown;
  apiKeyHeader?: unknown;
  apiKey?: unknown;
  authorization?: unknown;
  headers?: unknown;
}

interface RawTraceJsonConfig {
  enabled?: unknown;
  mode?: unknown;
  spoolDir?: unknown;
  maxPartBytes?: unknown;
  uploaderConcurrency?: unknown;
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  sync?: unknown;
}

interface GatewayAuthIdentityHeadersJsonConfig {
  userId?: unknown;
  tenantId?: unknown;
  subject?: unknown;
  organizationId?: unknown;
  plan?: unknown;
  apiKeyId?: unknown;
}

interface GatewayAuthSignatureJsonConfig {
  enabled?: unknown;
  header?: unknown;
  timestampHeader?: unknown;
  secretEnv?: unknown;
  maxSkewSec?: unknown;
}

interface GatewayAuthIntrospectionResponseMapJsonConfig {
  active?: unknown;
  userId?: unknown;
  tenantId?: unknown;
  subject?: unknown;
  organizationId?: unknown;
  plan?: unknown;
  apiKeyId?: unknown;
}

interface GatewayAuthIntrospectionJsonConfig {
  endpoint?: unknown;
  timeoutMs?: unknown;
  tokenHeader?: unknown;
  tokenBearerOnly?: unknown;
  requestTokenField?: unknown;
  credentialHeader?: unknown;
  credentialEnv?: unknown;
  responseMap?: unknown;
}

interface GatewayAuthStaticApiKeysJsonConfig {
  keys?: unknown;
  keyEnv?: unknown;
  keysEnv?: unknown;
  keyHeader?: unknown;
  tokenHeader?: unknown;
  keyBearerOnly?: unknown;
  tokenBearerOnly?: unknown;
}

interface GatewayAuthJsonConfig {
  enabled?: unknown;
  mode?: unknown;
  required?: unknown;
  trustedCidrs?: unknown;
  identityHeaders?: unknown;
  signature?: unknown;
  introspection?: unknown;
  staticApiKeys?: unknown;
  staticApiKey?: unknown;
}

interface AgentMcpServerJsonConfig {
  name?: unknown;
  transport?: unknown;
  stdioMessageMode?: unknown;
  command?: unknown;
  args?: unknown;
  env?: unknown;
  cwd?: unknown;
  url?: unknown;
  headers?: unknown;
  apiKey?: unknown;
  apiKeyEnv?: unknown;
  protocolVersion?: unknown;
  startupTimeoutMs?: unknown;
  requestTimeoutMs?: unknown;
}

interface AgentJsonConfig {
  mcpServers?: unknown;
  storageDir?: unknown;
  storage?: unknown;
  runtime?: unknown;
  external?: unknown;
  eventQueue?: unknown;
  eventWebhook?: unknown;
}

interface GatewayJsonConfig {
  host?: unknown;
  port?: unknown;
  providers?: unknown;
  Providers?: unknown;
  providerPlugins?: unknown;
  virtualModelProfiles?: unknown;
  providerExternal?: unknown;
  configExternal?: unknown;
  externalConfig?: unknown;
  provider?: unknown;
  defaultTargetProvider?: unknown;
  defaultTargetProviders?: unknown;
  modelList?: unknown;
  openaiApiKey?: unknown;
  anthropicApiKey?: unknown;
  geminiApiKey?: unknown;
  openaiBaseUrl?: unknown;
  anthropicBaseUrl?: unknown;
  geminiBaseUrl?: unknown;
  geminiApiVersion?: unknown;
  bodyLimitBytes?: unknown;
  bodyLimit?: unknown;
  upstreamTimeoutMs?: unknown;
  defaultOpenAIModel?: unknown;
  defaultAnthropicModel?: unknown;
  defaultGeminiModel?: unknown;
  auth?: unknown;
  policy?: unknown;
  precheck?: unknown;
  healthAwareRouting?: unknown;
  providerHealthCheck?: unknown;
  metrics?: unknown;
  cors?: unknown;
  idempotency?: unknown;
  upstreamConcurrency?: unknown;
  upstreamCircuitBreaker?: unknown;
  upstreamRetry?: unknown;
  transparentToolExecution?: unknown;
  routing?: unknown;
  billing?: {
    enabled?: unknown;
    rates?: {
      openai?: unknown;
      anthropic?: unknown;
      gemini?: unknown;
    };
  };
  billingQueue?: unknown;
  billingWebhook?: unknown;
  rawTrace?: unknown;
  agent?: unknown;
  mcpGateway?: unknown;
}

interface AgentRuntimeJsonConfig {
  sessionLockTimeoutMs?: unknown;
  eventWorkerConcurrency?: unknown;
  llmRetry?: unknown;
  toolRetry?: unknown;
}

interface AgentRetryPolicyJsonConfig {
  maxAttempts?: unknown;
  baseDelayMs?: unknown;
  maxDelayMs?: unknown;
  backoffMultiplier?: unknown;
  jitterMs?: unknown;
}

interface AgentExternalSourceJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  timeoutMs?: unknown;
  apiKeyHeader?: unknown;
  apiKey?: unknown;
  apiKeyEnv?: unknown;
  headers?: unknown;
}

interface ProviderExternalSourceJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  timeoutMs?: unknown;
  apiKeyHeader?: unknown;
  apiKey?: unknown;
  apiKeyEnv?: unknown;
  headers?: unknown;
}

interface GatewayConfigExternalSourceJsonConfig {
  enabled?: unknown;
  transport?: unknown;
  endpoint?: unknown;
  url?: unknown;
  command?: unknown;
  args?: unknown;
  cwd?: unknown;
  env?: unknown;
  method?: unknown;
  timeoutMs?: unknown;
  intervalMs?: unknown;
  intervalSeconds?: unknown;
  apiKeyHeader?: unknown;
  apiKey?: unknown;
  apiKeyEnv?: unknown;
  headers?: unknown;
}

interface ProviderJsonSettingsConfig {
  external?: unknown;
}

interface ProviderPluginMutationJsonConfig {
  strict?: unknown;
  headers?: unknown;
  query?: unknown;
  removeHeaders?: unknown;
  removeQuery?: unknown;
  bodySet?: unknown;
  bodyMerge?: unknown;
  bodyRemove?: unknown;
}

interface ProviderPluginResponseMutationJsonConfig {
  strict?: unknown;
  bodySet?: unknown;
  bodyMerge?: unknown;
  bodyRemove?: unknown;
}

interface ProviderPluginCodexOauthJsonConfig {
  enabled?: unknown;
  tokenEndpoint?: unknown;
  clientId?: unknown;
  scope?: unknown;
  accessToken?: unknown;
  refreshToken?: unknown;
  accountId?: unknown;
  account_id?: unknown;
  refreshIfMissingAccessToken?: unknown;
  forceRefresh?: unknown;
  required?: unknown;
  timeoutMs?: unknown;
  authHeader?: unknown;
  authScheme?: unknown;
}

interface ProviderPluginJsonConfig {
  key?: unknown;
  enabled?: unknown;
  provider?: unknown;
  providerName?: unknown;
  codexOauth?: unknown;
  deepseekThinking?: unknown;
  deepSeekThinking?: unknown;
  auth?: unknown;
  request?: unknown;
  response?: unknown;
}

interface AgentEventQueueJsonConfig {
  enabled?: unknown;
  queueName?: unknown;
  jobName?: unknown;
  removeOnComplete?: unknown;
  removeOnFail?: unknown;
}

interface McpGatewayJsonConfig {
  enabled?: unknown;
  endpoint?: unknown;
  websocket?: unknown;
  principals?: unknown;
  keys?: unknown;
  serverExposure?: unknown;
  internalCidrs?: unknown;
  oauth?: unknown;
  guardrails?: unknown;
}

interface McpGatewayPrincipalJsonConfig {
  key?: unknown;
  team?: unknown;
  organization?: unknown;
  allowServers?: unknown;
  allowTools?: unknown;
  denyTools?: unknown;
}

interface McpGatewayOAuthJsonConfig {
  enabled?: unknown;
  resource?: unknown;
  issuer?: unknown;
  authorizationEndpoint?: unknown;
  tokenEndpoint?: unknown;
  scopesSupported?: unknown;
  defaultPrincipalKey?: unknown;
  authorizationCodeTtlSec?: unknown;
  accessTokenTtlSec?: unknown;
  refreshTokenTtlSec?: unknown;
}

interface McpGatewayWebSocketAuthJsonConfig {
  allowQueryToken?: unknown;
  queryTokenParam?: unknown;
}

interface McpGatewayWebSocketJsonConfig {
  enabled?: unknown;
  endpoint?: unknown;
  auth?: unknown;
}

interface McpGatewayGuardrailsJsonConfig {
  enabled?: unknown;
  maxArgumentBytes?: unknown;
  blockedTools?: unknown;
  blockedArgumentKeys?: unknown;
  redactArgumentKeys?: unknown;
}

interface AgentStorageJsonConfig {
  type?: unknown;
  dir?: unknown;
  storageDir?: unknown;
}

export const config: GatewayConfig = loadGatewayConfigFromDisk();

export function resolveGatewayConfigPath(): string {
  const configuredPath = readString(process.env.GATEWAY_CONFIG_PATH);
  return configuredPath ? resolve(configuredPath) : resolve(process.cwd(), defaultConfigFileName);
}

export function loadGatewayConfigFromDisk(filePath = resolveGatewayConfigPath()): GatewayConfig {
  return buildGatewayConfig(loadJsonConfig(filePath));
}

export function parseGatewayConfigFromRaw(raw: unknown): GatewayConfig {
  if (!isPlainObject(raw)) {
    throw new Error('Gateway config payload must be a top-level JSON object.');
  }

  return buildGatewayConfig(raw as GatewayJsonConfig);
}

export function parseProvidersFromRaw(raw: unknown): ProviderConfig[] {
  return parseProvidersConfig(raw);
}

export function parseProviderPluginsFromRaw(raw: unknown): ProviderPluginConfig[] {
  return parseProviderPluginsConfig(raw);
}

export function parseVirtualModelProfilesFromRaw(raw: unknown): VirtualModelProfileConfig[] {
  return parseVirtualModelProfilesConfig(raw);
}

export function applyGatewayConfigInPlace(target: GatewayConfig, next: GatewayConfig): GatewayConfig {
  syncConfigObject(target as unknown as Record<string, unknown>, next as unknown as Record<string, unknown>);
  return target;
}

export function reloadGatewayConfig(target: GatewayConfig = config): GatewayConfig {
  const next = loadGatewayConfigFromDisk();
  return applyGatewayConfigInPlace(target, next);
}

function buildGatewayConfig(jsonConfig: GatewayJsonConfig): GatewayConfig {
  const providerSettings = isPlainObject(jsonConfig.provider)
    ? (jsonConfig.provider as ProviderJsonSettingsConfig)
    : undefined;
  const routingSettings = isPlainObject(jsonConfig.routing)
    ? (jsonConfig.routing as GatewayRoutingJsonConfig)
    : undefined;
  const healthAwareRoutingRaw = isPlainObject(jsonConfig.healthAwareRouting)
    ? (jsonConfig.healthAwareRouting as GatewayHealthAwareRoutingJsonConfig)
    : isPlainObject(routingSettings?.healthAware)
      ? (routingSettings.healthAware as GatewayHealthAwareRoutingJsonConfig)
      : undefined;
  const policyRaw = isPlainObject(jsonConfig.policy)
    ? (jsonConfig.policy as GatewayPolicyJsonConfig)
    : isPlainObject(routingSettings?.policy)
      ? (routingSettings.policy as GatewayPolicyJsonConfig)
      : undefined;
  const providerExternalRaw = isPlainObject(jsonConfig.providerExternal)
    ? (jsonConfig.providerExternal as ProviderExternalSourceJsonConfig)
    : isPlainObject(providerSettings?.external)
      ? (providerSettings.external as ProviderExternalSourceJsonConfig)
      : undefined;
  const configExternalRaw = isPlainObject(jsonConfig.configExternal)
    ? (jsonConfig.configExternal as GatewayConfigExternalSourceJsonConfig)
    : isPlainObject(jsonConfig.externalConfig)
      ? (jsonConfig.externalConfig as GatewayConfigExternalSourceJsonConfig)
      : undefined;
  const providers = parseProvidersConfig(jsonConfig.providers ?? jsonConfig.Providers);
  const openAIProviderConfig = findProviderConfigByType(providers, 'openai');
  const anthropicProviderConfig = findProviderConfigByType(providers, 'anthropic');
  const geminiProviderConfig = findProviderConfigByType(providers, 'gemini');
  const openAITopLevelBillingRate = parseBillingRate(jsonConfig.billing?.rates?.openai);
  const anthropicTopLevelBillingRate = parseBillingRate(jsonConfig.billing?.rates?.anthropic);
  const geminiTopLevelBillingRate = parseBillingRate(jsonConfig.billing?.rates?.gemini);

  return {
    host: readString(process.env.HOST) || readString(jsonConfig.host) || '0.0.0.0',
    port: readFiniteNumber(process.env.PORT) ?? readFiniteNumber(jsonConfig.port) ?? 3000,
    providers,
    providerPlugins: parseProviderPluginsConfig(jsonConfig.providerPlugins),
    virtualModelProfiles: parseVirtualModelProfilesConfig(jsonConfig.virtualModelProfiles),
    providerExternal: parseProviderExternalSourceConfig(providerExternalRaw),
    configExternal: parseGatewayConfigExternalSourceConfig(configExternalRaw),
    defaultTargetProvider:
      parseProvider(readString(process.env.DEFAULT_TARGET_PROVIDER)) ||
      parseProvider(readString(jsonConfig.defaultTargetProvider)) ||
      (providers[0] ? providerFromProviderType(providers[0].type) : undefined),
    defaultTargetProviders: resolveDefaultTargetProviders(
      process.env.DEFAULT_TARGET_PROVIDERS,
      providers
    ),
    routing: parseGatewayRoutingConfig(routingSettings),
    modelList: parseGatewayModelListConfig(jsonConfig.modelList),
    openaiApiKey:
      readString(process.env.OPENAI_API_KEY) ||
      openAIProviderConfig?.apikey ||
      readString(jsonConfig.openaiApiKey),
    anthropicApiKey:
      readString(process.env.ANTHROPIC_API_KEY) ||
      anthropicProviderConfig?.apikey ||
      readString(jsonConfig.anthropicApiKey),
    geminiApiKey:
      readString(process.env.GEMINI_API_KEY) ||
      geminiProviderConfig?.apikey ||
      readString(jsonConfig.geminiApiKey),
    openaiBaseUrl: trimTrailingSlash(
      readString(process.env.OPENAI_BASE_URL) ||
        openAIProviderConfig?.baseurl ||
        readString(jsonConfig.openaiBaseUrl) ||
        'https://api.openai.com/v1'
    ),
    anthropicBaseUrl: trimTrailingSlash(
      readString(process.env.ANTHROPIC_BASE_URL) ||
        anthropicProviderConfig?.baseurl ||
        readString(jsonConfig.anthropicBaseUrl) ||
        'https://api.anthropic.com'
    ),
    geminiBaseUrl: trimTrailingSlash(
      readString(process.env.GEMINI_BASE_URL) ||
        geminiProviderConfig?.baseurl ||
        readString(jsonConfig.geminiBaseUrl) ||
        'https://generativelanguage.googleapis.com'
    ),
    geminiApiVersion:
      readString(process.env.GEMINI_API_VERSION) || readString(jsonConfig.geminiApiVersion) || 'v1beta',
    bodyLimitBytes:
      resolveInteger(
        [
          process.env.GATEWAY_BODY_LIMIT_BYTES,
          process.env.BODY_LIMIT_BYTES,
          jsonConfig.bodyLimitBytes,
          jsonConfig.bodyLimit
        ],
        defaultBodyLimitBytes,
        1
      ),
    upstreamTimeoutMs:
      readFiniteNumber(process.env.UPSTREAM_TIMEOUT_MS) ??
      readFiniteNumber(jsonConfig.upstreamTimeoutMs) ??
      0,
    defaultOpenAIModel:
      readString(process.env.DEFAULT_OPENAI_MODEL) ||
      readString(jsonConfig.defaultOpenAIModel) ||
      openAIProviderConfig?.models[0],
    defaultAnthropicModel:
      readString(process.env.DEFAULT_ANTHROPIC_MODEL) ||
      readString(jsonConfig.defaultAnthropicModel) ||
      anthropicProviderConfig?.models[0],
    defaultGeminiModel:
      readString(process.env.DEFAULT_GEMINI_MODEL) ||
      readString(jsonConfig.defaultGeminiModel) ||
      geminiProviderConfig?.models[0],
    auth: parseGatewayAuthConfig(jsonConfig.auth),
    policy: parseGatewayPolicyConfig(policyRaw),
    precheck: parseGatewayPrecheckConfig(jsonConfig.precheck),
    healthAwareRouting: parseGatewayHealthAwareRoutingConfig(healthAwareRoutingRaw),
    providerHealthCheck: parseProviderHealthCheckSchedulerConfig(jsonConfig.providerHealthCheck),
    metrics: parseGatewayMetricsConfig(jsonConfig.metrics),
    cors: parseGatewayCorsConfig(jsonConfig.cors),
    idempotency: parseGatewayIdempotencyConfig(jsonConfig.idempotency),
    upstreamConcurrency: parseGatewayUpstreamConcurrencyConfig(jsonConfig.upstreamConcurrency),
    upstreamCircuitBreaker: parseGatewayUpstreamCircuitBreakerConfig(jsonConfig.upstreamCircuitBreaker),
    upstreamRetry: parseGatewayUpstreamRetryConfig(jsonConfig.upstreamRetry),
    transparentToolExecution: parseGatewayTransparentToolExecutionConfig(
      jsonConfig.transparentToolExecution
    ),
    billing: {
      enabled: resolveBoolean(process.env.BILLING_ENABLED, jsonConfig.billing?.enabled, true),
      currency: 'USD',
      rates: {
        openai: {
          inputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.OPENAI_INPUT_PRICE_PER_1M,
              openAITopLevelBillingRate?.inputPerMillionUsd,
              openAIProviderConfig?.billing.default?.inputPerMillionUsd
            ],
            0
          ),
          outputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.OPENAI_OUTPUT_PRICE_PER_1M,
              openAITopLevelBillingRate?.outputPerMillionUsd,
              openAIProviderConfig?.billing.default?.outputPerMillionUsd
            ],
            0
          ),
          cacheReadPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.OPENAI_CACHE_READ_PRICE_PER_1M,
              openAITopLevelBillingRate?.cacheReadPerMillionUsd,
              openAIProviderConfig?.billing.default?.cacheReadPerMillionUsd
            ],
            0
          ),
          cacheWritePerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.OPENAI_CACHE_WRITE_PRICE_PER_1M,
              openAITopLevelBillingRate?.cacheWritePerMillionUsd,
              openAIProviderConfig?.billing.default?.cacheWritePerMillionUsd
            ],
            0
          ),
          tiers: openAITopLevelBillingRate?.tiers || openAIProviderConfig?.billing.default?.tiers
        },
        anthropic: {
          inputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.ANTHROPIC_INPUT_PRICE_PER_1M,
              anthropicTopLevelBillingRate?.inputPerMillionUsd,
              anthropicProviderConfig?.billing.default?.inputPerMillionUsd
            ],
            0
          ),
          outputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.ANTHROPIC_OUTPUT_PRICE_PER_1M,
              anthropicTopLevelBillingRate?.outputPerMillionUsd,
              anthropicProviderConfig?.billing.default?.outputPerMillionUsd
            ],
            0
          ),
          cacheReadPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.ANTHROPIC_CACHE_READ_PRICE_PER_1M,
              anthropicTopLevelBillingRate?.cacheReadPerMillionUsd,
              anthropicProviderConfig?.billing.default?.cacheReadPerMillionUsd
            ],
            0
          ),
          cacheWritePerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.ANTHROPIC_CACHE_WRITE_PRICE_PER_1M,
              anthropicTopLevelBillingRate?.cacheWritePerMillionUsd,
              anthropicProviderConfig?.billing.default?.cacheWritePerMillionUsd
            ],
            0
          ),
          tiers: anthropicTopLevelBillingRate?.tiers || anthropicProviderConfig?.billing.default?.tiers
        },
        gemini: {
          inputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.GEMINI_INPUT_PRICE_PER_1M,
              geminiTopLevelBillingRate?.inputPerMillionUsd,
              geminiProviderConfig?.billing.default?.inputPerMillionUsd
            ],
            0
          ),
          outputPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.GEMINI_OUTPUT_PRICE_PER_1M,
              geminiTopLevelBillingRate?.outputPerMillionUsd,
              geminiProviderConfig?.billing.default?.outputPerMillionUsd
            ],
            0
          ),
          cacheReadPerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.GEMINI_CACHE_READ_PRICE_PER_1M,
              geminiTopLevelBillingRate?.cacheReadPerMillionUsd,
              geminiProviderConfig?.billing.default?.cacheReadPerMillionUsd
            ],
            0
          ),
          cacheWritePerMillionUsd: resolveNonNegativeNumber(
            [
              process.env.GEMINI_CACHE_WRITE_PRICE_PER_1M,
              geminiTopLevelBillingRate?.cacheWritePerMillionUsd,
              geminiProviderConfig?.billing.default?.cacheWritePerMillionUsd
            ],
            0
          ),
          tiers: geminiTopLevelBillingRate?.tiers || geminiProviderConfig?.billing.default?.tiers
        }
      }
    },
    billingQueue: parseBillingQueueConfig(jsonConfig.billingQueue),
    billingWebhook: parseBillingWebhookConfig(jsonConfig.billingWebhook),
    rawTrace: parseRawTraceConfig(
      isPlainObject(jsonConfig.rawTrace) ? (jsonConfig.rawTrace as RawTraceJsonConfig) : undefined
    ),
    agent: parseAgentConfig(jsonConfig.agent),
    mcpGateway: parseMcpGatewayConfig(jsonConfig.mcpGateway)
  };
}

function parseGatewayPrecheckConfig(value: unknown): GatewayPrecheckConfig {
  const raw = isPlainObject(value) ? (value as GatewayPrecheckJsonConfig) : undefined;
  const rateLimitRaw = isPlainObject(raw?.rateLimit)
    ? (raw?.rateLimit as GatewayPrecheckRuleJsonConfig)
    : undefined;
  const quotaRaw = isPlainObject(raw?.quota)
    ? (raw?.quota as GatewayPrecheckRuleJsonConfig)
    : undefined;
  const budgetRaw = isPlainObject(raw?.budget)
    ? (raw?.budget as GatewayPrecheckRuleJsonConfig)
    : undefined;
  const estimationRaw = isPlainObject(raw?.estimation)
    ? (raw?.estimation as GatewayPrecheckEstimationJsonConfig)
    : undefined;
  const storageRaw = isPlainObject(raw?.storage)
    ? (raw?.storage as GatewayPrecheckStorageJsonConfig)
    : undefined;
  const rateLimitDefaults = parseGatewayRateLimitDefaults(rateLimitRaw);

  return {
    enabled: resolveBoolean(process.env.PRECHECK_ENABLED, raw?.enabled, false),
    rateLimit: {
      enabled: resolveBoolean(process.env.RATE_LIMIT_ENABLED, rateLimitRaw?.enabled, false),
      windowMs: rateLimitDefaults.windowMs,
      maxRequests: rateLimitDefaults.maxRequests,
      subject: rateLimitDefaults.subject,
      scope: rateLimitDefaults.scope,
      headerName: rateLimitDefaults.headerName,
      rpm: rateLimitDefaults.rpm,
      rpd: rateLimitDefaults.rpd,
      tpm: rateLimitDefaults.tpm,
      tpd: rateLimitDefaults.tpd,
      ipm: rateLimitDefaults.ipm,
      limits: buildGatewayRateLimitDimensions(rateLimitRaw, rateLimitDefaults)
    },
    quota: {
      enabled: resolveBoolean(process.env.QUOTA_PRECHECK_ENABLED, quotaRaw?.enabled, false),
      windowMs: resolvePrecheckWindowMs(
        [process.env.QUOTA_WINDOW_MS, quotaRaw?.windowMs],
        [process.env.QUOTA_WINDOW_SECONDS, quotaRaw?.windowSeconds],
        86_400_000
      ),
      maxTokens: resolvePositiveNumber(
        [process.env.QUOTA_MAX_TOKENS, quotaRaw?.maxTokens],
        0
      ),
      subject: parseGatewayPrecheckSubject(
        readString(process.env.QUOTA_SUBJECT) || readString(quotaRaw?.subject),
        'identity'
      ),
      scope: parseGatewayPrecheckScope(
        readString(process.env.QUOTA_SCOPE) || readString(quotaRaw?.scope),
        'global'
      ),
      headerName: normalizeHeaderName(
        readString(process.env.QUOTA_HEADER) || readString(quotaRaw?.headerName),
        ''
      ) || undefined
    },
    budget: {
      enabled: resolveBoolean(process.env.BUDGET_PRECHECK_ENABLED, budgetRaw?.enabled, false),
      windowMs: resolvePrecheckWindowMs(
        [process.env.BUDGET_WINDOW_MS, budgetRaw?.windowMs],
        [process.env.BUDGET_WINDOW_SECONDS, budgetRaw?.windowSeconds],
        86_400_000
      ),
      maxCostUsd: resolvePositiveNumber(
        [process.env.BUDGET_MAX_COST_USD, budgetRaw?.maxCostUsd],
        0
      ),
      subject: parseGatewayPrecheckSubject(
        readString(process.env.BUDGET_SUBJECT) || readString(budgetRaw?.subject),
        'identity'
      ),
      scope: parseGatewayPrecheckScope(
        readString(process.env.BUDGET_SCOPE) || readString(budgetRaw?.scope),
        'global'
      ),
      headerName: normalizeHeaderName(
        readString(process.env.BUDGET_HEADER) || readString(budgetRaw?.headerName),
        ''
      ) || undefined
    },
    estimation: {
      charsPerToken: resolvePositiveNumber(
        [
          process.env.PRECHECK_CHARS_PER_TOKEN,
          estimationRaw?.charsPerToken
        ],
        4
      ),
      defaultMaxOutputTokens: resolveInteger(
        [
          process.env.PRECHECK_DEFAULT_MAX_OUTPUT_TOKENS,
          estimationRaw?.defaultMaxOutputTokens
        ],
        1024,
        0
      )
    },
    storage: parseGatewayPrecheckStorageConfig(storageRaw)
  };
}

function parseGatewayPrecheckStorageConfig(
  value: GatewayPrecheckStorageJsonConfig | undefined
): GatewayPrecheckConfig['storage'] {
  return {
    type: parseGatewayPrecheckStorageType(
      readString(process.env.PRECHECK_STORAGE_TYPE) ||
        readString(process.env.PRECHECK_STORAGE_BACKEND) ||
        readString(value?.type) ||
        readString(value?.backend),
      'memory'
    )
  };
}

function parseGatewayPrecheckStorageType(
  value: string | undefined,
  fallback: GatewayPrecheckStorageType
): GatewayPrecheckStorageType {
  const normalized = value?.trim().toLowerCase();
  if (normalized === 'memory' || normalized === 'in_memory' || normalized === 'in-memory') {
    return 'memory';
  }

  return fallback;
}

function parseGatewayRateLimitDefaults(rateLimitRaw: GatewayPrecheckRuleJsonConfig | undefined) {
  return {
    windowMs: resolvePrecheckWindowMs(
      [process.env.RATE_LIMIT_WINDOW_MS, rateLimitRaw?.windowMs],
      [process.env.RATE_LIMIT_WINDOW_SECONDS, rateLimitRaw?.windowSeconds],
      60_000
    ),
    maxRequests: resolvePositiveNumber(
      [process.env.RATE_LIMIT_MAX_REQUESTS, rateLimitRaw?.maxRequests],
      0
    ),
    subject: parseGatewayPrecheckSubject(
      readString(process.env.RATE_LIMIT_SUBJECT) || readString(rateLimitRaw?.subject),
      'identity'
    ),
    scope: parseGatewayPrecheckScope(
      readString(process.env.RATE_LIMIT_SCOPE) || readString(rateLimitRaw?.scope),
      'global'
    ),
    headerName:
      normalizeHeaderName(
        readString(process.env.RATE_LIMIT_HEADER) || readString(rateLimitRaw?.headerName),
        ''
      ) || undefined,
    rpm: resolvePositiveNumber([process.env.RATE_LIMIT_RPM, rateLimitRaw?.rpm], 0),
    rpd: resolvePositiveNumber([process.env.RATE_LIMIT_RPD, rateLimitRaw?.rpd], 0),
    tpm: resolvePositiveNumber([process.env.RATE_LIMIT_TPM, rateLimitRaw?.tpm], 0),
    tpd: resolvePositiveNumber([process.env.RATE_LIMIT_TPD, rateLimitRaw?.tpd], 0),
    ipm: resolvePositiveNumber([process.env.RATE_LIMIT_IPM, rateLimitRaw?.ipm], 0)
  };
}

function buildGatewayRateLimitDimensions(
  rateLimitRaw: GatewayPrecheckRuleJsonConfig | undefined,
  defaults: ReturnType<typeof parseGatewayRateLimitDefaults>
): GatewayRateLimitDimensionConfig[] {
  const limits: GatewayRateLimitDimensionConfig[] = [];
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'requests',
    metric: 'requests',
    windowMs: defaults.windowMs,
    max: defaults.maxRequests
  });
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'rpm',
    metric: 'requests',
    windowMs: 60_000,
    max: defaults.rpm
  });
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'rpd',
    metric: 'requests',
    windowMs: 86_400_000,
    max: defaults.rpd
  });
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'tpm',
    metric: 'tokens',
    windowMs: 60_000,
    max: defaults.tpm
  });
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'tpd',
    metric: 'tokens',
    windowMs: 86_400_000,
    max: defaults.tpd
  });
  addGatewayRateLimitDimension(limits, defaults, {
    name: 'ipm',
    metric: 'images',
    windowMs: 60_000,
    max: defaults.ipm
  });

  for (const limit of parseCustomRateLimitDimensions(rateLimitRaw?.limits, defaults)) {
    limits.push(limit);
  }

  return limits;
}

function addGatewayRateLimitDimension(
  limits: GatewayRateLimitDimensionConfig[],
  defaults: ReturnType<typeof parseGatewayRateLimitDefaults>,
  limit: {
    name: string;
    metric: GatewayRateLimitMetric;
    windowMs: number;
    max: number;
  }
): void {
  if (limit.max <= 0) {
    return;
  }

  limits.push({
    enabled: true,
    name: limit.name,
    metric: limit.metric,
    windowMs: limit.windowMs,
    max: limit.max,
    subject: defaults.subject,
    scope: defaults.scope,
    headerName: defaults.headerName
  });
}

function parseCustomRateLimitDimensions(
  value: unknown,
  defaults: ReturnType<typeof parseGatewayRateLimitDefaults>
): GatewayRateLimitDimensionConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: GatewayRateLimitDimensionConfig[] = [];
  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const raw = entry as GatewayPrecheckRuleJsonConfig;
    const enabled = readBoolean(raw.enabled) ?? true;
    const metric = parseGatewayRateLimitMetric(readString(raw.metric));
    const max = resolvePositiveNumber([raw.max, raw.maxRequests, raw.maxTokens], 0);
    if (!metric || max <= 0) {
      continue;
    }

    parsed.push({
      enabled,
      name: readString(raw.name) || `${metric}_${parsed.length + 1}`,
      metric,
      windowMs: resolvePrecheckWindowMs([raw.windowMs], [raw.windowSeconds], defaults.windowMs),
      max,
      subject: parseGatewayPrecheckSubject(readString(raw.subject), defaults.subject),
      scope: parseGatewayPrecheckScope(readString(raw.scope), defaults.scope),
      headerName: normalizeHeaderName(readString(raw.headerName), '') || defaults.headerName
    });
  }

  return parsed;
}

function parseGatewayRateLimitMetric(value: string | undefined): GatewayRateLimitMetric | undefined {
  const normalized = value?.trim().toLowerCase().replace(/-/g, '_');
  if (normalized === 'request' || normalized === 'requests') {
    return 'requests';
  }

  if (normalized === 'token' || normalized === 'tokens') {
    return 'tokens';
  }

  if (normalized === 'image' || normalized === 'images') {
    return 'images';
  }

  return undefined;
}

function parseGatewayHealthAwareRoutingConfig(
  value: GatewayHealthAwareRoutingJsonConfig | undefined
): GatewayHealthAwareRoutingConfig {
  return {
    enabled: resolveBoolean(process.env.HEALTH_AWARE_ROUTING_ENABLED, value?.enabled, false),
    skipUnavailable: resolveBoolean(
      process.env.HEALTH_AWARE_ROUTING_SKIP_UNAVAILABLE,
      value?.skipUnavailable,
      true
    ),
    unhealthyStatuses: parseProviderHealthStatusList(
      process.env.HEALTH_AWARE_ROUTING_UNHEALTHY_STATUSES,
      value?.unhealthyStatuses,
      ['down']
    ),
    preferHealthy: resolveBoolean(
      process.env.HEALTH_AWARE_ROUTING_PREFER_HEALTHY,
      value?.preferHealthy,
      true
    ),
    preferLowerLatency: resolveBoolean(
      process.env.HEALTH_AWARE_ROUTING_PREFER_LOWER_LATENCY,
      value?.preferLowerLatency,
      true
    )
  };
}

function parseGatewayRoutingConfig(value: GatewayRoutingJsonConfig | undefined): GatewayRoutingConfig {
  return {
    preferSourceProviderForBareModels: resolveBoolean(
      process.env.PREFER_SOURCE_PROVIDER_FOR_BARE_MODELS,
      value?.preferSourceProviderForBareModels ?? value?.preferSourceProviderForBareModel,
      false
    )
  };
}

function parseGatewayModelListConfig(value: unknown): GatewayModelListConfig {
  const raw = isPlainObject(value) ? (value as GatewayModelListJsonConfig) : undefined;
  return {
    bareModelIds:
      readBoolean(raw?.bareModelIds) ??
      readBoolean(raw?.bareModelId) ??
      readBoolean(raw?.bare_model_ids) ??
      false
  };
}

function parseProviderHealthCheckSchedulerConfig(value: unknown): ProviderHealthCheckSchedulerConfig {
  const raw = isPlainObject(value) ? (value as ProviderHealthCheckSchedulerJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.PROVIDER_HEALTH_CHECK_ENABLED, raw?.enabled, false),
    intervalMs: resolvePrecheckWindowMs(
      [process.env.PROVIDER_HEALTH_CHECK_INTERVAL_MS, raw?.intervalMs],
      [process.env.PROVIDER_HEALTH_CHECK_INTERVAL_SECONDS, raw?.intervalSeconds],
      60000
    ),
    timeoutMs: resolvePrecheckWindowMs(
      [process.env.PROVIDER_HEALTH_CHECK_TIMEOUT_MS, raw?.timeoutMs],
      [process.env.PROVIDER_HEALTH_CHECK_TIMEOUT_SECONDS, raw?.timeoutSeconds],
      5000
    ),
    initialDelayMs: resolvePrecheckWindowMs(
      [process.env.PROVIDER_HEALTH_CHECK_INITIAL_DELAY_MS, raw?.initialDelayMs],
      [process.env.PROVIDER_HEALTH_CHECK_INITIAL_DELAY_SECONDS, raw?.initialDelaySeconds],
      0
    )
  };
}

function parseGatewayMetricsConfig(value: unknown): GatewayMetricsConfig {
  const raw = isPlainObject(value) ? (value as GatewayMetricsJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.GATEWAY_METRICS_ENABLED, raw?.enabled, false),
    includeProviderHealth: resolveBoolean(
      process.env.GATEWAY_METRICS_INCLUDE_PROVIDER_HEALTH,
      raw?.includeProviderHealth,
      true
    )
  };
}

function parseGatewayCorsConfig(value: unknown): GatewayCorsConfig {
  const raw = isPlainObject(value) ? (value as GatewayCorsJsonConfig) : undefined;
  const origins = resolveStringList(
    process.env.GATEWAY_CORS_ORIGINS ?? process.env.GATEWAY_CORS_ORIGIN ?? process.env.CORS_ORIGIN,
    raw?.origins ?? raw?.origin,
    ['*']
  );

  return {
    enabled: resolveBoolean(process.env.GATEWAY_CORS_ENABLED, raw?.enabled, true),
    origins,
    allowedHeaders: resolveStringList(
      process.env.GATEWAY_CORS_ALLOWED_HEADERS,
      raw?.allowedHeaders,
      defaultCorsAllowedHeaders
    ),
    allowedMethods: resolveStringList(
      process.env.GATEWAY_CORS_ALLOWED_METHODS,
      raw?.allowedMethods,
      defaultCorsAllowedMethods
    ).map((method) => method.toUpperCase()),
    allowCredentials: resolveBoolean(
      process.env.GATEWAY_CORS_ALLOW_CREDENTIALS,
      raw?.allowCredentials,
      false
    ),
    maxAgeSeconds: resolveInteger(
      [process.env.GATEWAY_CORS_MAX_AGE_SECONDS, raw?.maxAgeSeconds, raw?.maxAge],
      86400,
      0
    )
  };
}

function parseGatewayIdempotencyConfig(value: unknown): GatewayIdempotencyConfig {
  const raw = isPlainObject(value) ? (value as GatewayIdempotencyJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.GATEWAY_IDEMPOTENCY_ENABLED, raw?.enabled, false),
    headerName:
      readString(process.env.GATEWAY_IDEMPOTENCY_HEADER) ||
      readString(raw?.headerName) ||
      'idempotency-key',
    ttlMs: resolvePrecheckWindowMs(
      [process.env.GATEWAY_IDEMPOTENCY_TTL_MS, raw?.ttlMs],
      [process.env.GATEWAY_IDEMPOTENCY_TTL_SECONDS, raw?.ttlSeconds],
      86400000
    ),
    maxEntries: resolveInteger(
      [process.env.GATEWAY_IDEMPOTENCY_MAX_ENTRIES, raw?.maxEntries],
      10000,
      1
    ),
    cacheErrorResponses: resolveBoolean(
      process.env.GATEWAY_IDEMPOTENCY_CACHE_ERROR_RESPONSES,
      raw?.cacheErrorResponses,
      false
    )
  };
}

function parseGatewayUpstreamConcurrencyConfig(value: unknown): GatewayUpstreamConcurrencyConfig {
  const raw = isPlainObject(value) ? (value as GatewayUpstreamConcurrencyJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.GATEWAY_UPSTREAM_CONCURRENCY_ENABLED, raw?.enabled, false),
    maxInFlightPerProvider: resolveInteger(
      [
        process.env.GATEWAY_UPSTREAM_MAX_IN_FLIGHT_PER_PROVIDER,
        raw?.maxInFlightPerProvider
      ],
      10,
      1
    ),
    queueTimeoutMs: resolvePrecheckWindowMs(
      [
        process.env.GATEWAY_UPSTREAM_CONCURRENCY_QUEUE_TIMEOUT_MS,
        raw?.queueTimeoutMs
      ],
      [
        process.env.GATEWAY_UPSTREAM_CONCURRENCY_QUEUE_TIMEOUT_SECONDS,
        raw?.queueTimeoutSeconds
      ],
      1000
    )
  };
}

function parseGatewayUpstreamCircuitBreakerConfig(value: unknown): GatewayUpstreamCircuitBreakerConfig {
  const raw = isPlainObject(value) ? (value as GatewayUpstreamCircuitBreakerJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_ENABLED, raw?.enabled, false),
    failureThreshold: resolveInteger(
      [
        process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        raw?.failureThreshold
      ],
      5,
      1
    ),
    cooldownMs: resolvePrecheckWindowMs(
      [
        process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_COOLDOWN_MS,
        raw?.cooldownMs
      ],
      [
        process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        raw?.cooldownSeconds
      ],
      30000
    ),
    failureStatusCodes: parseStatusCodeList(
      process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_STATUS_CODES,
      raw?.failureStatusCodes,
      [429, 500, 502, 503, 504]
    )
  };
}

function parseGatewayUpstreamRetryConfig(value: unknown): GatewayUpstreamRetryConfig {
  const raw = isPlainObject(value) ? (value as GatewayUpstreamRetryJsonConfig) : undefined;

  return {
    enabled: resolveBoolean(process.env.GATEWAY_UPSTREAM_RETRY_ENABLED, raw?.enabled, true),
    maxAttempts: resolveInteger(
      [process.env.GATEWAY_UPSTREAM_RETRY_MAX_ATTEMPTS, raw?.maxAttempts],
      2,
      1
    ),
    baseDelayMs: resolvePrecheckWindowMs(
      [process.env.GATEWAY_UPSTREAM_RETRY_BASE_DELAY_MS, raw?.baseDelayMs],
      [process.env.GATEWAY_UPSTREAM_RETRY_BASE_DELAY_SECONDS, raw?.baseDelaySeconds],
      150
    ),
    maxDelayMs: resolvePrecheckWindowMs(
      [process.env.GATEWAY_UPSTREAM_RETRY_MAX_DELAY_MS, raw?.maxDelayMs],
      [process.env.GATEWAY_UPSTREAM_RETRY_MAX_DELAY_SECONDS, raw?.maxDelaySeconds],
      150
    ),
    backoffMultiplier: resolvePositiveNumber(
      [process.env.GATEWAY_UPSTREAM_RETRY_BACKOFF_MULTIPLIER, raw?.backoffMultiplier],
      1
    ),
    jitterMs: resolvePrecheckWindowMs(
      [process.env.GATEWAY_UPSTREAM_RETRY_JITTER_MS, raw?.jitterMs],
      [process.env.GATEWAY_UPSTREAM_RETRY_JITTER_SECONDS, raw?.jitterSeconds],
      0
    ),
    retryStatusCodes: parseStatusCodeList(
      process.env.GATEWAY_UPSTREAM_RETRY_STATUS_CODES,
      raw?.retryStatusCodes,
      []
    )
  };
}

function parseGatewayTransparentToolExecutionConfig(
  value: unknown
): GatewayTransparentToolExecutionConfig {
  const raw = isPlainObject(value)
    ? (value as GatewayTransparentToolExecutionJsonConfig)
    : undefined;

  return {
    enabled: resolveBoolean(
      process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ENABLED,
      raw?.enabled,
      false
    ),
    maxTurns: resolveInteger(
      [process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TURNS, raw?.maxTurns],
      4,
      1
    ),
    maxToolCalls: resolveInteger(
      [
        process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TOOL_CALLS,
        raw?.maxToolCalls
      ],
      8,
      0
    ),
    requireClientDeclaration: resolveBoolean(
      process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_REQUIRE_CLIENT_DECLARATION,
      raw?.requireClientDeclaration,
      true
    ),
    unknownToolPolicy: parseGatewayTransparentToolUnknownPolicy(
      readString(process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_UNKNOWN_TOOL_POLICY) ||
        readString(raw?.unknownToolPolicy),
      'return_to_client'
    ),
    allowTools: resolveStringList(
      process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ALLOW_TOOLS,
      raw?.allowTools,
      []
    ),
    denyTools: resolveStringList(
      process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_DENY_TOOLS,
      raw?.denyTools,
      []
    )
  };
}

function parseGatewayTransparentToolUnknownPolicy(
  value: string | undefined,
  fallback: GatewayTransparentToolUnknownPolicy
): GatewayTransparentToolUnknownPolicy {
  const normalized = value?.trim().toLowerCase().replace(/-/g, '_');
  if (normalized === 'return_to_client' || normalized === 'return') {
    return 'return_to_client';
  }

  if (normalized === 'fail' || normalized === 'error') {
    return 'fail';
  }

  return fallback;
}

function parseGatewayPolicyConfig(
  value: GatewayPolicyJsonConfig | undefined
): GatewayPolicyConfig {
  const policy = isPlainObject(value) ? value : undefined;
  const defaultsRaw = mergeGatewayPolicyRuleJson(
    policy,
    isPlainObject(policy?.defaults)
      ? (policy?.defaults as GatewayPolicyRuleJsonConfig)
      : undefined
  );

  return {
    enabled: resolveBoolean(process.env.GATEWAY_POLICY_ENABLED, policy?.enabled, false),
    defaults: parseGatewayPolicyRule(defaultsRaw, {
      allowProviders: process.env.GATEWAY_POLICY_ALLOW_PROVIDERS,
      denyProviders: process.env.GATEWAY_POLICY_DENY_PROVIDERS,
      allowProviderNames: process.env.GATEWAY_POLICY_ALLOW_PROVIDER_NAMES,
      denyProviderNames: process.env.GATEWAY_POLICY_DENY_PROVIDER_NAMES,
      allowModels: process.env.GATEWAY_POLICY_ALLOW_MODELS,
      denyModels: process.env.GATEWAY_POLICY_DENY_MODELS,
      allowProviderModels: process.env.GATEWAY_POLICY_ALLOW_PROVIDER_MODELS,
      denyProviderModels: process.env.GATEWAY_POLICY_DENY_PROVIDER_MODELS
    }),
    byUser: parseGatewayPolicyRuleMap(policy?.byUser),
    byTenant: parseGatewayPolicyRuleMap(policy?.byTenant),
    byOrganization: parseGatewayPolicyRuleMap(policy?.byOrganization),
    bySubject: parseGatewayPolicyRuleMap(policy?.bySubject),
    byPlan: parseGatewayPolicyRuleMap(policy?.byPlan),
    byApiKey: parseGatewayPolicyRuleMap(policy?.byApiKey)
  };
}

function mergeGatewayPolicyRuleJson(
  base: GatewayPolicyRuleJsonConfig | undefined,
  override: GatewayPolicyRuleJsonConfig | undefined
): GatewayPolicyRuleJsonConfig | undefined {
  if (!base && !override) {
    return undefined;
  }

  return {
    allowProviders: override?.allowProviders ?? base?.allowProviders,
    denyProviders: override?.denyProviders ?? base?.denyProviders,
    allowProviderNames: override?.allowProviderNames ?? base?.allowProviderNames,
    denyProviderNames: override?.denyProviderNames ?? base?.denyProviderNames,
    allowModels: override?.allowModels ?? base?.allowModels,
    denyModels: override?.denyModels ?? base?.denyModels,
    allowProviderModels: override?.allowProviderModels ?? base?.allowProviderModels,
    denyProviderModels: override?.denyProviderModels ?? base?.denyProviderModels
  };
}

function parseGatewayPolicyRule(
  value: GatewayPolicyRuleJsonConfig | undefined,
  env: Partial<Record<keyof GatewayPolicyRuleConfig, string | undefined>> = {}
): GatewayPolicyRuleConfig {
  return {
    allowProviders: parseGatewayPolicyProviderList(env.allowProviders, value?.allowProviders),
    denyProviders: parseGatewayPolicyProviderList(env.denyProviders, value?.denyProviders),
    allowProviderNames: parseGatewayPolicyStringList(env.allowProviderNames, value?.allowProviderNames),
    denyProviderNames: parseGatewayPolicyStringList(env.denyProviderNames, value?.denyProviderNames),
    allowModels: parseGatewayPolicyStringList(env.allowModels, value?.allowModels),
    denyModels: parseGatewayPolicyStringList(env.denyModels, value?.denyModels),
    allowProviderModels: parseGatewayPolicyStringList(env.allowProviderModels, value?.allowProviderModels),
    denyProviderModels: parseGatewayPolicyStringList(env.denyProviderModels, value?.denyProviderModels)
  };
}

function parseGatewayPolicyRuleMap(value: unknown): Record<string, GatewayPolicyRuleConfig> {
  if (!isPlainObject(value)) {
    return {};
  }

  const parsed: Record<string, GatewayPolicyRuleConfig> = {};
  for (const [key, rawRule] of Object.entries(value)) {
    const normalizedKey = key.trim();
    if (!normalizedKey || !isPlainObject(rawRule)) {
      continue;
    }

    parsed[normalizedKey] = parseGatewayPolicyRule(rawRule as GatewayPolicyRuleJsonConfig);
  }

  return parsed;
}

function parseGatewayPolicyProviderList(
  envValue: string | undefined,
  fileValue: unknown
): Provider[] {
  const fromEnv = parseProviderList(envValue);
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  return parseModelList(fileValue)
    .map(parseProvider)
    .filter((item): item is Provider => Boolean(item));
}

function parseGatewayPolicyStringList(
  envValue: string | undefined,
  fileValue: unknown
): string[] {
  const fromEnv = parseModelList(envValue);
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  return parseModelList(fileValue);
}

function parseProviderHealthConfig(item: ProviderJsonConfig): ProviderHealthConfig | undefined {
  const rawHealth = isPlainObject(item.health)
    ? (item.health as ProviderHealthJsonConfig)
    : undefined;
  const raw = rawHealth || item;
  const status = parseProviderHealthStatus(readString(raw.status));
  const available = readBoolean(raw.available);
  const priority = readFiniteNumber(raw.priority);
  const latencyMs = readNonNegativeNumber(raw.latencyMs ?? raw.latency_ms);
  const checkedAt = readString(raw.checkedAt ?? raw.checked_at);

  if (
    status === undefined &&
    available === undefined &&
    priority === undefined &&
    latencyMs === undefined &&
    checkedAt === undefined
  ) {
    return undefined;
  }

  return {
    status: status || (available === false ? 'down' : available === true ? 'healthy' : 'unknown'),
    available,
    priority,
    latencyMs,
    checkedAt
  };
}

function resolvePrecheckWindowMs(
  millisecondValues: unknown[],
  secondValues: unknown[],
  fallback: number
): number {
  const fromMs = resolveInteger(millisecondValues, 0, 1);
  if (fromMs > 0) {
    return fromMs;
  }

  const fromSeconds = resolvePositiveNumber(secondValues, 0);
  if (fromSeconds > 0) {
    return Math.trunc(fromSeconds * 1000);
  }

  return fallback;
}

function parseGatewayPrecheckSubject(
  value: string | undefined,
  fallback: GatewayPrecheckSubject
): GatewayPrecheckSubject {
  const normalized = value?.trim().toLowerCase().replace(/-/g, '_');
  if (
    normalized === 'identity' ||
    normalized === 'user' ||
    normalized === 'tenant' ||
    normalized === 'organization' ||
    normalized === 'api_key' ||
    normalized === 'ip' ||
    normalized === 'header' ||
    normalized === 'global'
  ) {
    return normalized;
  }

  return fallback;
}

function parseGatewayPrecheckScope(
  value: string | undefined,
  fallback: GatewayPrecheckScope
): GatewayPrecheckScope {
  const normalized = value?.trim().toLowerCase().replace(/-/g, '_');
  if (
    normalized === 'global' ||
    normalized === 'provider' ||
    normalized === 'model' ||
    normalized === 'provider_model'
  ) {
    return normalized;
  }

  return fallback;
}

function parseProviderHealthStatus(value: string | undefined): ProviderHealthStatus | undefined {
  const normalized = value?.trim().toLowerCase().replace(/-/g, '_');
  if (
    normalized === 'healthy' ||
    normalized === 'degraded' ||
    normalized === 'unknown' ||
    normalized === 'down'
  ) {
    return normalized;
  }

  return undefined;
}

function parseProviderHealthStatusList(
  envValue: string | undefined,
  fileValue: unknown,
  fallback: ProviderHealthStatus[]
): ProviderHealthStatus[] {
  const fromEnv = parseModelList(envValue)
    .map(parseProviderHealthStatus)
    .filter((item): item is ProviderHealthStatus => Boolean(item));
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  const fromFile = parseModelList(fileValue)
    .map(parseProviderHealthStatus)
    .filter((item): item is ProviderHealthStatus => Boolean(item));
  if (fromFile.length > 0) {
    return fromFile;
  }

  return fallback;
}

function parseStatusCodeList(
  envValue: string | undefined,
  fileValue: unknown,
  fallback: number[]
): number[] {
  const fromEnv = parseStatusCodeTokens(envValue);
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  const fromFile = parseStatusCodeTokens(fileValue);
  if (fromFile.length > 0) {
    return fromFile;
  }

  return fallback;
}

function parseStatusCodeTokens(value: unknown): number[] {
  const rawItems =
    typeof value === 'string'
      ? value.split(',').map((item) => item.trim())
      : Array.isArray(value)
        ? value
        : [];
  const parsed: number[] = [];

  for (const item of rawItems) {
    const code =
      typeof item === 'number'
        ? item
        : typeof item === 'string'
          ? Number(item.trim())
          : NaN;
    if (!Number.isInteger(code) || code < 100 || code > 599) {
      continue;
    }

    if (!parsed.includes(code)) {
      parsed.push(code);
    }
  }

  return parsed;
}

function parseGatewayAuthConfig(value: unknown): GatewayAuthConfig {
  const auth = isPlainObject(value) ? (value as GatewayAuthJsonConfig) : undefined;
  const identityHeadersRaw = isPlainObject(auth?.identityHeaders)
    ? (auth?.identityHeaders as GatewayAuthIdentityHeadersJsonConfig)
    : undefined;
  const signatureRaw = isPlainObject(auth?.signature)
    ? (auth?.signature as GatewayAuthSignatureJsonConfig)
    : undefined;
  const introspectionRaw = isPlainObject(auth?.introspection)
    ? (auth?.introspection as GatewayAuthIntrospectionJsonConfig)
    : undefined;
  const staticApiKeysRaw = isPlainObject(auth?.staticApiKeys)
    ? (auth?.staticApiKeys as GatewayAuthStaticApiKeysJsonConfig)
    : isPlainObject(auth?.staticApiKey)
      ? (auth?.staticApiKey as GatewayAuthStaticApiKeysJsonConfig)
      : undefined;
  const introspectionResponseMapRaw = isPlainObject(introspectionRaw?.responseMap)
    ? (introspectionRaw?.responseMap as GatewayAuthIntrospectionResponseMapJsonConfig)
    : undefined;
  const trustedCidrsFromEnv = parseModelList(process.env.AUTH_TRUSTED_CIDRS);
  const trustedCidrsFromConfig = parseModelList(auth?.trustedCidrs);

  return {
    enabled: resolveBoolean(process.env.AUTH_ENABLED, auth?.enabled, false),
    mode: parseGatewayAuthMode(
      readString(process.env.AUTH_MODE) || readString(auth?.mode),
      'trusted_header'
    ),
    required: resolveBoolean(process.env.AUTH_REQUIRED, auth?.required, true),
    trustedCidrs: trustedCidrsFromEnv.length > 0 ? trustedCidrsFromEnv : trustedCidrsFromConfig,
    identityHeaders: {
      userId: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_USER_ID) || readString(identityHeadersRaw?.userId),
        'x-auth-user-id'
      ),
      tenantId: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_TENANT_ID) || readString(identityHeadersRaw?.tenantId),
        'x-auth-tenant-id'
      ),
      subject: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_SUBJECT) || readString(identityHeadersRaw?.subject),
        'x-auth-sub'
      ),
      organizationId: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_ORGANIZATION_ID) ||
          readString(identityHeadersRaw?.organizationId),
        'x-auth-organization-id'
      ),
      plan: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_PLAN) || readString(identityHeadersRaw?.plan),
        'x-auth-plan'
      ),
      apiKeyId: normalizeHeaderName(
        readString(process.env.AUTH_HEADER_API_KEY_ID) || readString(identityHeadersRaw?.apiKeyId),
        'x-auth-api-key-id'
      )
    },
    signature: {
      enabled: resolveBoolean(process.env.AUTH_SIGNATURE_ENABLED, signatureRaw?.enabled, false),
      header: normalizeHeaderName(
        readString(process.env.AUTH_SIGNATURE_HEADER) || readString(signatureRaw?.header),
        'x-auth-signature'
      ),
      timestampHeader: normalizeHeaderName(
        readString(process.env.AUTH_SIGNATURE_TIMESTAMP_HEADER) ||
          readString(signatureRaw?.timestampHeader),
        'x-auth-ts'
      ),
      secretEnv:
        readString(process.env.AUTH_SIGNATURE_SECRET_ENV) ||
        readString(signatureRaw?.secretEnv) ||
        'AUTH_HEADER_SIGNING_SECRET',
      maxSkewSec: resolveInteger([process.env.AUTH_SIGNATURE_MAX_SKEW_SEC, signatureRaw?.maxSkewSec], 120, 1)
    },
    introspection: {
      endpoint:
        readString(process.env.AUTH_INTROSPECTION_ENDPOINT) ||
        readString(introspectionRaw?.endpoint),
      timeoutMs: resolveInteger(
        [process.env.AUTH_INTROSPECTION_TIMEOUT_MS, introspectionRaw?.timeoutMs],
        3000,
        50
      ),
      tokenHeader: normalizeHeaderName(
        readString(process.env.AUTH_INTROSPECTION_TOKEN_HEADER) ||
          readString(introspectionRaw?.tokenHeader),
        'authorization'
      ),
      tokenBearerOnly: resolveBoolean(
        process.env.AUTH_INTROSPECTION_TOKEN_BEARER_ONLY,
        introspectionRaw?.tokenBearerOnly,
        true
      ),
      requestTokenField:
        readString(process.env.AUTH_INTROSPECTION_REQUEST_TOKEN_FIELD) ||
        readString(introspectionRaw?.requestTokenField) ||
        'token',
      credentialHeader: normalizeHeaderName(
        readString(process.env.AUTH_INTROSPECTION_CREDENTIAL_HEADER) ||
          readString(introspectionRaw?.credentialHeader),
        'x-gateway-auth'
      ),
      credentialEnv:
        readString(process.env.AUTH_INTROSPECTION_CREDENTIAL_ENV) ||
        readString(introspectionRaw?.credentialEnv) ||
        'AUTH_INTROSPECTION_SHARED_SECRET',
      responseMap: {
        active:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_ACTIVE_FIELD) ||
          readString(introspectionResponseMapRaw?.active) ||
          'active',
        userId:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_USER_ID_FIELD) ||
          readString(introspectionResponseMapRaw?.userId) ||
          'userId',
        tenantId:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_TENANT_ID_FIELD) ||
          readString(introspectionResponseMapRaw?.tenantId) ||
          'tenantId',
        subject:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_SUBJECT_FIELD) ||
          readString(introspectionResponseMapRaw?.subject) ||
          'sub',
        organizationId:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_ORGANIZATION_ID_FIELD) ||
          readString(introspectionResponseMapRaw?.organizationId) ||
          'organizationId',
        plan:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_PLAN_FIELD) ||
          readString(introspectionResponseMapRaw?.plan) ||
          'plan',
        apiKeyId:
          readString(process.env.AUTH_INTROSPECTION_RESPONSE_API_KEY_ID_FIELD) ||
          readString(introspectionResponseMapRaw?.apiKeyId) ||
          'apiKeyId'
      }
    },
    staticApiKeys: parseGatewayAuthStaticApiKeysConfig(staticApiKeysRaw)
  };
}

function parseGatewayAuthMode(value: string | undefined, fallback: GatewayAuthConfig['mode']): GatewayAuthConfig['mode'] {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }

  if (normalized === 'trusted_header' || normalized === 'trusted-header' || normalized === 'header') {
    return 'trusted_header';
  }

  if (
    normalized === 'http_introspection' ||
    normalized === 'http-introspection' ||
    normalized === 'introspection'
  ) {
    return 'http_introspection';
  }

  if (
    normalized === 'static_api_key' ||
    normalized === 'static-api-key' ||
    normalized === 'static_api_keys' ||
    normalized === 'static-api-keys' ||
    normalized === 'static' ||
    normalized === 'api_key' ||
    normalized === 'api-key'
  ) {
    return 'static_api_key';
  }

  return fallback;
}

function parseGatewayAuthStaticApiKeysConfig(
  raw: GatewayAuthStaticApiKeysJsonConfig | undefined
): GatewayAuthConfig['staticApiKeys'] {
  const keyEnv =
    readString(process.env.AUTH_STATIC_API_KEY_ENV) ||
    readString(process.env.AUTH_STATIC_API_KEYS_ENV) ||
    readString(raw?.keyEnv) ||
    readString(raw?.keysEnv);
  const keysFromDirectEnv = parseModelList(
    process.env.AUTH_STATIC_API_KEYS ?? process.env.AUTH_STATIC_API_KEY
  );
  const keysFromReferencedEnv = keyEnv ? parseModelList(process.env[keyEnv]) : [];
  const keysFromConfig = parseModelList(raw?.keys);

  return {
    keys: dedupeStrings([...keysFromDirectEnv, ...keysFromReferencedEnv, ...keysFromConfig]),
    keyEnv,
    keyHeader: normalizeHeaderName(
      readString(process.env.AUTH_STATIC_API_KEY_HEADER) ||
        readString(raw?.keyHeader) ||
        readString(raw?.tokenHeader),
      'authorization'
    ),
    keyBearerOnly: resolveBoolean(
      process.env.AUTH_STATIC_API_KEY_BEARER_ONLY,
      raw?.keyBearerOnly ?? raw?.tokenBearerOnly,
      true
    )
  };
}

function parseBillingQueueConfig(value: unknown): BillingQueueConfig {
  const raw = isPlainObject(value) ? (value as BillingQueueJsonConfig) : undefined;
  return {
    enabled: resolveBoolean(process.env.BILLING_QUEUE_ENABLED, raw?.enabled, false),
    queueName:
      readString(process.env.BILLING_QUEUE_NAME) || readString(raw?.queueName) || 'gateway-billing',
    jobName:
      readString(process.env.BILLING_QUEUE_JOB_NAME) || readString(raw?.jobName) || 'billing.usage',
    removeOnComplete: resolveInteger(
      [process.env.BILLING_QUEUE_REMOVE_ON_COMPLETE, raw?.removeOnComplete],
      1000,
      0
    ),
    removeOnFail: resolveInteger(
      [process.env.BILLING_QUEUE_REMOVE_ON_FAIL, raw?.removeOnFail],
      5000,
      0
    )
  };
}

function parseBillingWebhookConfig(value: unknown): BillingWebhookConfig {
  const raw = isPlainObject(value) ? (value as BillingWebhookJsonConfig) : undefined;
  const headers = parseHeaderMap(raw?.headers);
  const apiKeyHeader = normalizeHeaderName(readString(process.env.BILLING_WEBHOOK_API_KEY_HEADER), '');
  const apiKey =
    readString(process.env.BILLING_WEBHOOK_API_KEY) ||
    readString(process.env.PROVIDER_EXTERNAL_API_KEY) ||
    readString(process.env.GATEWAY_SYNC_API_KEY);
  if (apiKeyHeader && apiKey) {
    headers[apiKeyHeader] = apiKey;
  }

  const authorization = readString(process.env.BILLING_WEBHOOK_AUTHORIZATION);
  if (authorization) {
    headers.authorization = authorization;
  }

  const endpoint =
    readString(process.env.BILLING_WEBHOOK_ENDPOINT) ||
    readString(raw?.endpoint) ||
    readString(raw?.url);
  const command =
    readString(process.env.BILLING_WEBHOOK_STDIO_COMMAND) ||
    readString(raw?.command);

  return {
    enabled: resolveBoolean(process.env.BILLING_WEBHOOK_ENABLED, raw?.enabled, false),
    transport: parseExternalEventSinkTransport(
      readString(process.env.BILLING_WEBHOOK_TRANSPORT) || readString(raw?.transport),
      endpoint,
      command
    ),
    endpoint,
    command,
    args: resolveStringList(process.env.BILLING_WEBHOOK_STDIO_ARGS, raw?.args, []),
    cwd: readString(process.env.BILLING_WEBHOOK_STDIO_CWD) || readString(raw?.cwd),
    env: parseHeaderMap(raw?.env),
    timeoutMs: resolveInteger(
      [process.env.BILLING_WEBHOOK_TIMEOUT_MS, raw?.timeoutMs],
      5000,
      50
    ),
    ...parseExternalEventSinkRetryConfig(raw, 'BILLING_WEBHOOK'),
    requireAck: parseExternalEventSinkRequireAck(raw, 'BILLING_WEBHOOK'),
    headers
  };
}

function parseRawTraceConfig(value: RawTraceJsonConfig | undefined): RawTraceConfig {
  const sync = parseRawTraceSyncConfig(
    isPlainObject(value?.sync) ? (value?.sync as RawTraceSyncJsonConfig) : undefined
  );
  const mode = parseRawTraceCaptureMode(
    readString(process.env.RAW_TRACE_MODE) || readString(value?.mode),
    'body_full'
  );
  const enabled =
    resolveBoolean(process.env.RAW_TRACE_ENABLED, value?.enabled, false) &&
    mode !== 'disabled';

  return {
    enabled,
    mode,
    spoolDir:
      readString(process.env.RAW_TRACE_SPOOL_DIR) ||
      readString(value?.spoolDir) ||
      '.raw-trace-spool',
    maxPartBytes: resolveInteger(
      [process.env.RAW_TRACE_MAX_PART_BYTES, value?.maxPartBytes],
      50 * 1024 * 1024,
      1024,
    ),
    uploaderConcurrency: resolveInteger(
      [process.env.RAW_TRACE_UPLOADER_CONCURRENCY, value?.uploaderConcurrency],
      2,
      1,
    ),
    maxAttempts: resolveInteger(
      [process.env.RAW_TRACE_UPLOAD_MAX_ATTEMPTS, value?.maxAttempts],
      5,
      1,
    ),
    baseDelayMs: resolveInteger(
      [process.env.RAW_TRACE_UPLOAD_BASE_DELAY_MS, value?.baseDelayMs],
      1000,
      50,
    ),
    sync,
  };
}

function parseRawTraceSyncConfig(value: RawTraceSyncJsonConfig | undefined): RawTraceSyncConfig {
  const headers = parseHeaderMap(value?.headers);
  const apiKeyHeader = normalizeHeaderName(
    readString(process.env.RAW_TRACE_SYNC_API_KEY_HEADER) || readString(value?.apiKeyHeader),
    'x-gateway-sync-key',
  );
  const apiKey =
    readString(process.env.RAW_TRACE_SYNC_API_KEY) || readString(value?.apiKey) || readString(process.env.GATEWAY_SYNC_API_KEY);
  if (apiKeyHeader && apiKey) {
    headers[apiKeyHeader] = apiKey;
  }

  const authorization =
    readString(process.env.RAW_TRACE_SYNC_AUTHORIZATION) || readString(value?.authorization);
  if (authorization) {
    headers.authorization = authorization;
  }

  const endpoint =
    readString(process.env.RAW_TRACE_SYNC_ENDPOINT) ||
    readString(process.env.RAW_TRACE_SYNC_URL) ||
    readString(value?.endpoint) ||
    readString(value?.url);
  const command =
    readString(process.env.RAW_TRACE_SYNC_STDIO_COMMAND) ||
    readString(value?.command);

  return {
    enabled: resolveBoolean(process.env.RAW_TRACE_SYNC_ENABLED, value?.enabled, Boolean(endpoint || command)),
    transport: parseExternalEventSinkTransport(
      readString(process.env.RAW_TRACE_SYNC_TRANSPORT) || readString(value?.transport),
      endpoint,
      command
    ),
    endpoint,
    command,
    args: resolveStringList(process.env.RAW_TRACE_SYNC_STDIO_ARGS, value?.args, []),
    cwd: readString(process.env.RAW_TRACE_SYNC_STDIO_CWD) || readString(value?.cwd),
    env: parseHeaderMap(value?.env),
    timeoutMs: resolveInteger(
      [process.env.RAW_TRACE_SYNC_TIMEOUT_MS, value?.timeoutMs],
      5000,
      50,
    ),
    ...parseExternalEventSinkRetryConfig(value, 'RAW_TRACE_SYNC'),
    requireAck: parseExternalEventSinkRequireAck(value, 'RAW_TRACE_SYNC'),
    apiKeyHeader,
    apiKey,
    authorization,
    headers,
  };
}

function parseRawTraceCaptureMode(
  value: string | undefined,
  fallback: RawTraceCaptureMode,
): RawTraceCaptureMode {
  const normalized = value?.trim().toLowerCase();
  if (
    normalized === 'disabled' ||
    normalized === 'body_redacted' ||
    normalized === 'body_full' ||
    normalized === 'wire_raw'
  ) {
    return normalized;
  }

  return fallback;
}

function parseAgentConfig(value: unknown): AgentConfig {
  const raw = isPlainObject(value) ? (value as AgentJsonConfig) : undefined;
  return {
    mcpServers: parseMcpServerConfigList(raw?.mcpServers),
    storage: parseAgentStorageConfig(raw),
    runtime: parseAgentRuntimeConfig(raw),
    external: parseAgentExternalSourceConfig(raw),
    eventQueue: parseAgentEventQueueConfig(raw),
    eventWebhook: parseAgentEventWebhookConfig(raw)
  };
}

function parseAgentRuntimeConfig(value: AgentJsonConfig | undefined): AgentRuntimeConfig {
  const runtime = isPlainObject(value?.runtime)
    ? (value?.runtime as AgentRuntimeJsonConfig)
    : undefined;

  return {
    sessionLockTimeoutMs: resolveInteger(
      [process.env.AGENT_SESSION_LOCK_TIMEOUT_MS, runtime?.sessionLockTimeoutMs],
      15000,
      10
    ),
    eventWorkerConcurrency: resolveInteger(
      [process.env.AGENT_EVENT_WORKER_CONCURRENCY, runtime?.eventWorkerConcurrency],
      16,
      1
    ),
    llmRetry: parseAgentRetryPolicyConfig(runtime?.llmRetry, 'AGENT_LLM_RETRY_', defaultAgentLlmRetryConfig),
    toolRetry: parseAgentRetryPolicyConfig(
      runtime?.toolRetry,
      'AGENT_TOOL_RETRY_',
      defaultAgentToolRetryConfig
    )
  };
}

function parseAgentRetryPolicyConfig(
  value: unknown,
  envPrefix: string,
  fallback: AgentRetryPolicyConfig
): AgentRetryPolicyConfig {
  const retry = isPlainObject(value) ? (value as AgentRetryPolicyJsonConfig) : undefined;
  const maxAttempts = resolveInteger(
    [process.env[`${envPrefix}MAX_ATTEMPTS`], retry?.maxAttempts],
    fallback.maxAttempts,
    1
  );
  const baseDelayMs = resolveInteger(
    [process.env[`${envPrefix}BASE_DELAY_MS`], retry?.baseDelayMs],
    fallback.baseDelayMs,
    0
  );
  const maxDelayCandidate = resolveInteger(
    [process.env[`${envPrefix}MAX_DELAY_MS`], retry?.maxDelayMs],
    fallback.maxDelayMs,
    0
  );
  const maxDelayMs = Math.max(baseDelayMs, maxDelayCandidate);
  const backoffMultiplier = resolveInteger(
    [process.env[`${envPrefix}BACKOFF_MULTIPLIER`], retry?.backoffMultiplier],
    fallback.backoffMultiplier,
    1
  );
  const jitterMs = resolveInteger(
    [process.env[`${envPrefix}JITTER_MS`], retry?.jitterMs],
    fallback.jitterMs,
    0
  );

  return {
    maxAttempts,
    baseDelayMs,
    maxDelayMs,
    backoffMultiplier,
    jitterMs
  };
}

function parseMcpGatewayConfig(value: unknown): McpGatewayConfig {
  const raw = isPlainObject(value) ? (value as McpGatewayJsonConfig) : undefined;
  const principals = parseMcpGatewayPrincipalList(raw?.principals ?? raw?.keys);
  const inferredEnabled = principals.length > 0;
  const endpoint = normalizeHttpPath(
    readString(process.env.MCP_GATEWAY_ENDPOINT) || readString(raw?.endpoint) || '/mcp'
  );
  const internalCidrs = parseModelList(raw?.internalCidrs);

  return {
    enabled: resolveBoolean(process.env.MCP_GATEWAY_ENABLED, raw?.enabled, inferredEnabled),
    endpoint,
    websocket: parseMcpGatewayWebSocket(raw?.websocket),
    principals,
    serverExposure: parseMcpGatewayServerExposure(raw?.serverExposure),
    internalCidrs:
      internalCidrs.length > 0
        ? internalCidrs
        : ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '127.0.0.0/8', '::1/128', 'fc00::/7'],
    guardrails: parseMcpGatewayGuardrails(raw?.guardrails),
    oauth: parseMcpGatewayOauth(raw?.oauth)
  };
}

function parseMcpGatewayPrincipalList(value: unknown): McpGatewayPrincipalConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: McpGatewayPrincipalConfig[] = [];
  const usedKeys = new Set<string>();

  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const item = entry as McpGatewayPrincipalJsonConfig;
    const key = readString(item.key);
    if (!key || usedKeys.has(key)) {
      continue;
    }
    usedKeys.add(key);

    parsed.push({
      key,
      team: readString(item.team) || 'default',
      organization: readString(item.organization),
      allowServers: parseModelList(item.allowServers),
      allowTools: parseModelList(item.allowTools),
      denyTools: parseModelList(item.denyTools)
    });
  }

  return parsed;
}

function parseMcpGatewayServerExposure(value: unknown): Record<string, McpServerExposure> {
  const parsed: Record<string, McpServerExposure> = {};

  if (Array.isArray(value)) {
    for (const entry of value) {
      if (!isPlainObject(entry)) {
        continue;
      }

      const serverName = readString((entry as Record<string, unknown>).server)
        || readString((entry as Record<string, unknown>).name);
      const exposure = parseMcpServerExposureToken(
        (entry as Record<string, unknown>).exposure ?? (entry as Record<string, unknown>).access
      );

      if (!serverName || !exposure) {
        continue;
      }

      parsed[serverName] = exposure;
    }

    return parsed;
  }

  if (!isPlainObject(value)) {
    return parsed;
  }

  for (const [serverName, rawExposure] of Object.entries(value)) {
    const exposure = parseMcpServerExposureToken(rawExposure);
    if (!exposure) {
      continue;
    }

    parsed[serverName] = exposure;
  }

  return parsed;
}

function parseMcpServerExposureToken(value: unknown): McpServerExposure | undefined {
  if (typeof value === 'boolean') {
    return value ? 'public' : 'internal';
  }

  const normalized = readString(value)?.toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'public' || normalized === 'external' || normalized === 'internet') {
    return 'public';
  }

  if (normalized === 'internal' || normalized === 'private' || normalized === 'intranet') {
    return 'internal';
  }

  return undefined;
}

function parseMcpGatewayGuardrails(value: unknown): McpGatewayGuardrailsConfig {
  const guardrails = isPlainObject(value) ? (value as McpGatewayGuardrailsJsonConfig) : undefined;
  const redactArgumentKeys = parseModelList(guardrails?.redactArgumentKeys);

  return {
    enabled: resolveBoolean(process.env.MCP_GATEWAY_GUARDRAILS_ENABLED, guardrails?.enabled, true),
    maxArgumentBytes: resolveInteger(
      [process.env.MCP_GATEWAY_MAX_ARGUMENT_BYTES, guardrails?.maxArgumentBytes],
      64 * 1024,
      1
    ),
    blockedTools: parseModelList(guardrails?.blockedTools),
    blockedArgumentKeys: parseModelList(guardrails?.blockedArgumentKeys),
    redactArgumentKeys:
      redactArgumentKeys.length > 0
        ? redactArgumentKeys
        : ['authorization', 'api_key', 'apikey', 'token', 'access_token', 'password', 'secret']
  };
}

function parseMcpGatewayOauth(value: unknown): McpGatewayOAuthConfig {
  const oauth = isPlainObject(value) ? (value as McpGatewayOAuthJsonConfig) : undefined;
  const scopesSupported = parseModelList(oauth?.scopesSupported);

  return {
    enabled: resolveBoolean(process.env.MCP_GATEWAY_OAUTH_ENABLED, oauth?.enabled, false),
    resource: readString(process.env.MCP_GATEWAY_OAUTH_RESOURCE) || readString(oauth?.resource),
    issuer: readString(process.env.MCP_GATEWAY_OAUTH_ISSUER) || readString(oauth?.issuer),
    authorizationEndpoint:
      readString(process.env.MCP_GATEWAY_OAUTH_AUTHORIZATION_ENDPOINT) ||
      readString(oauth?.authorizationEndpoint),
    tokenEndpoint:
      readString(process.env.MCP_GATEWAY_OAUTH_TOKEN_ENDPOINT) || readString(oauth?.tokenEndpoint),
    defaultPrincipalKey:
      readString(process.env.MCP_GATEWAY_OAUTH_DEFAULT_PRINCIPAL_KEY) ||
      readString(oauth?.defaultPrincipalKey),
    authorizationCodeTtlSec: resolveInteger(
      [process.env.MCP_GATEWAY_OAUTH_AUTH_CODE_TTL_SEC, oauth?.authorizationCodeTtlSec],
      180,
      30
    ),
    accessTokenTtlSec: resolveInteger(
      [process.env.MCP_GATEWAY_OAUTH_ACCESS_TOKEN_TTL_SEC, oauth?.accessTokenTtlSec],
      3600,
      60
    ),
    refreshTokenTtlSec: resolveInteger(
      [process.env.MCP_GATEWAY_OAUTH_REFRESH_TOKEN_TTL_SEC, oauth?.refreshTokenTtlSec],
      2592000,
      300
    ),
    scopesSupported:
      scopesSupported.length > 0 ? scopesSupported : ['mcp:tools:list', 'mcp:tools:call']
  };
}

function parseMcpGatewayWebSocket(value: unknown): McpGatewayWebSocketConfig {
  const websocket = isPlainObject(value) ? (value as McpGatewayWebSocketJsonConfig) : undefined;
  const auth = isPlainObject(websocket?.auth)
    ? (websocket?.auth as McpGatewayWebSocketAuthJsonConfig)
    : undefined;
  const endpoint = normalizeHttpPath(
    readString(process.env.MCP_GATEWAY_WS_ENDPOINT) || readString(websocket?.endpoint) || '/mcp/ws'
  );

  return {
    enabled: resolveBoolean(process.env.MCP_GATEWAY_WS_ENABLED, websocket?.enabled, false),
    endpoint: ensureMcpPathPrefix(endpoint, 'mcpGateway.websocket.endpoint'),
    auth: parseMcpGatewayWebSocketAuth(auth)
  };
}

function parseMcpGatewayWebSocketAuth(
  value: McpGatewayWebSocketAuthJsonConfig | undefined
): McpGatewayWebSocketAuthConfig {
  return {
    allowQueryToken: resolveBoolean(
      process.env.MCP_GATEWAY_WS_ALLOW_QUERY_TOKEN,
      value?.allowQueryToken,
      true
    ),
    queryTokenParam:
      readString(process.env.MCP_GATEWAY_WS_QUERY_TOKEN_PARAM) ||
      readString(value?.queryTokenParam) ||
      'token'
  };
}

function parseAgentStorageConfig(value: AgentJsonConfig | undefined): AgentStorageConfig {
  const storageValue = isPlainObject(value?.storage) ? (value?.storage as AgentStorageJsonConfig) : undefined;
  const storageType =
    (readString(process.env.AGENT_STORAGE_TYPE) || readString(storageValue?.type) || 'memory')
      .trim()
      .toLowerCase();

  if (storageType === 'memory' || storageType === 'in_memory' || storageType === 'in-memory') {
    return {
      type: 'memory'
    };
  }

  if (storageType === 'filesystem' || storageType === 'file' || storageType === 'fs') {
    return {
      type: 'filesystem',
      dir:
        readString(process.env.AGENT_STORAGE_DIR) ||
        readString(storageValue?.dir) ||
        readString(storageValue?.storageDir) ||
        readString(value?.storageDir) ||
        resolve(process.cwd(), '.agent-data')
    };
  }

  if (storageType === 'http') {
    throw new Error('agent.storage.type=http has been removed. Use agent.external for external agent/session state.');
  }

  throw new Error('agent.storage.type currently supports only "memory" or "filesystem".');
}

function parseAgentExternalSourceConfig(
  value: AgentJsonConfig | undefined
): AgentExternalSourceConfig {
  const external = isPlainObject(value?.external)
    ? (value?.external as AgentExternalSourceJsonConfig)
    : undefined;
  const enabled = resolveBoolean(process.env.AGENT_EXTERNAL_ENABLED, external?.enabled, false);
  const endpoint =
    readString(process.env.AGENT_EXTERNAL_ENDPOINT) ||
    readString(process.env.AGENT_EXTERNAL_URL) ||
    readString(external?.endpoint) ||
    readString(external?.url);
  const command =
    readString(process.env.AGENT_EXTERNAL_STDIO_COMMAND) ||
    readString(external?.command);
  const transport = parseExternalEventSinkTransport(
    readString(process.env.AGENT_EXTERNAL_TRANSPORT) || readString(external?.transport),
    endpoint,
    command
  );
  if (enabled && transport !== 'stdio' && !endpoint) {
    throw new Error('agent.external.endpoint is required when agent.external.enabled=true.');
  }
  if (enabled && transport === 'stdio' && !command) {
    throw new Error('agent.external.command is required when agent.external.transport=stdio.');
  }

  const apiKeyEnvName =
    readString(process.env.AGENT_EXTERNAL_API_KEY_ENV) || readString(external?.apiKeyEnv);
  const apiKeyFromRefEnv = apiKeyEnvName ? readString(process.env[apiKeyEnvName]) : undefined;

  return {
    enabled,
    transport,
    endpoint,
    command,
    args: resolveStringList(process.env.AGENT_EXTERNAL_STDIO_ARGS, external?.args, []),
    cwd: readString(process.env.AGENT_EXTERNAL_STDIO_CWD) || readString(external?.cwd),
    env: parseHeaderMap(external?.env),
    timeoutMs: resolveInteger([process.env.AGENT_EXTERNAL_TIMEOUT_MS, external?.timeoutMs], 5000, 50),
    apiKeyHeader: normalizeHeaderName(
      readString(process.env.AGENT_EXTERNAL_API_KEY_HEADER) || readString(external?.apiKeyHeader),
      'x-agent-external-key'
    ),
    apiKey:
      readString(process.env.AGENT_EXTERNAL_API_KEY) ||
      readString(external?.apiKey) ||
      apiKeyFromRefEnv,
    headers: parseHeaderMap(external?.headers)
  };
}

function parseProviderExternalSourceConfig(
  value: ProviderExternalSourceJsonConfig | undefined
): ProviderExternalSourceConfig | undefined {
  const external = isPlainObject(value)
    ? (value as ProviderExternalSourceJsonConfig)
    : undefined;
  const enabled = resolveBoolean(process.env.PROVIDER_EXTERNAL_ENABLED, external?.enabled, false);
  const endpoint =
    readString(process.env.PROVIDER_EXTERNAL_ENDPOINT) ||
    readString(process.env.PROVIDER_EXTERNAL_URL) ||
    readString(external?.endpoint) ||
    readString(external?.url);
  const command =
    readString(process.env.PROVIDER_EXTERNAL_STDIO_COMMAND) ||
    readString(external?.command);
  const transport = parseExternalEventSinkTransport(
    readString(process.env.PROVIDER_EXTERNAL_TRANSPORT) || readString(external?.transport),
    endpoint,
    command
  );
  if (enabled && transport !== 'stdio' && !endpoint) {
    throw new Error('provider.external.endpoint is required when provider.external.enabled=true.');
  }
  if (enabled && transport === 'stdio' && !command) {
    throw new Error('provider.external.command is required when provider.external.transport=stdio.');
  }

  const apiKeyEnvName =
    readString(process.env.PROVIDER_EXTERNAL_API_KEY_ENV) || readString(external?.apiKeyEnv);
  const apiKeyFromRefEnv = apiKeyEnvName ? readString(process.env[apiKeyEnvName]) : undefined;

  return {
    enabled,
    transport,
    endpoint,
    command,
    args: resolveStringList(process.env.PROVIDER_EXTERNAL_STDIO_ARGS, external?.args, []),
    cwd: readString(process.env.PROVIDER_EXTERNAL_STDIO_CWD) || readString(external?.cwd),
    env: parseHeaderMap(external?.env),
    timeoutMs: resolveInteger([process.env.PROVIDER_EXTERNAL_TIMEOUT_MS, external?.timeoutMs], 5000, 50),
    apiKeyHeader: normalizeHeaderName(
      readString(process.env.PROVIDER_EXTERNAL_API_KEY_HEADER) || readString(external?.apiKeyHeader),
      'x-provider-external-key'
    ),
    apiKey:
      readString(process.env.PROVIDER_EXTERNAL_API_KEY) ||
      readString(external?.apiKey) ||
      apiKeyFromRefEnv,
    headers: parseHeaderMap(external?.headers)
  };
}

function parseGatewayConfigExternalSourceConfig(
  value: GatewayConfigExternalSourceJsonConfig | undefined
): GatewayConfigExternalSourceConfig | undefined {
  const external = isPlainObject(value)
    ? (value as GatewayConfigExternalSourceJsonConfig)
    : undefined;
  const enabled = resolveBoolean(process.env.GATEWAY_CONFIG_EXTERNAL_ENABLED, external?.enabled, false);
  const transportToken =
    readString(process.env.GATEWAY_CONFIG_EXTERNAL_TRANSPORT) || readString(external?.transport);
  const endpoint =
    readString(process.env.GATEWAY_CONFIG_EXTERNAL_ENDPOINT) ||
    readString(process.env.GATEWAY_CONFIG_EXTERNAL_URL) ||
    readString(external?.endpoint) ||
    readString(external?.url);
  const command =
    readString(process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_COMMAND) ||
    readString(external?.command);
  const transport = parseGatewayConfigExternalTransport(transportToken, endpoint, command) || 'http';
  if (enabled && transportToken && !parseGatewayConfigExternalTransport(transportToken)) {
    throw new Error('configExternal.transport currently supports only "http", "websocket", "grpc", or "stdio".');
  }

  if (enabled && transport !== 'stdio' && !endpoint) {
    throw new Error('configExternal.endpoint is required when configExternal.enabled=true.');
  }
  if (enabled && transport === 'stdio' && !command) {
    throw new Error('configExternal.command is required when configExternal.transport=stdio.');
  }

  const apiKeyEnvName =
    readString(process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY_ENV) || readString(external?.apiKeyEnv);
  const apiKeyFromRefEnv = apiKeyEnvName ? readString(process.env[apiKeyEnvName]) : undefined;

  return {
    enabled,
    transport,
    endpoint,
    command,
    args: resolveStringList(process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_ARGS, external?.args, []),
    cwd: readString(process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_CWD) || readString(external?.cwd),
    env: parseHeaderMap(external?.env),
    method: parseGatewayConfigExternalMethod(
      readString(process.env.GATEWAY_CONFIG_EXTERNAL_METHOD) || readString(external?.method),
      'GET'
    ),
    timeoutMs: resolveInteger([process.env.GATEWAY_CONFIG_EXTERNAL_TIMEOUT_MS, external?.timeoutMs], 5000, 50),
    intervalMs: resolveInteger(
      [process.env.GATEWAY_CONFIG_EXTERNAL_INTERVAL_MS, external?.intervalMs],
      resolveInteger([process.env.GATEWAY_CONFIG_EXTERNAL_INTERVAL_SECONDS, external?.intervalSeconds], 0, 0) *
        1000,
      0
    ),
    apiKeyHeader: normalizeHeaderName(
      readString(process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY_HEADER) || readString(external?.apiKeyHeader),
      'x-gateway-config-key'
    ),
    apiKey:
      readString(process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY) ||
      readString(external?.apiKey) ||
      apiKeyFromRefEnv,
    headers: parseHeaderMap(external?.headers)
  };
}

function parseGatewayConfigExternalTransport(
  value: string | undefined,
  endpoint?: string,
  command?: string
): GatewayConfigExternalSourceTransport | undefined {
  const normalized = value?.trim().toLowerCase();
  if (normalized === 'websocket' || normalized === 'ws' || normalized === 'wss') {
    return 'websocket';
  }
  if (normalized === 'grpc' || normalized === 'grpcs') {
    return 'grpc';
  }
  if (normalized === 'stdio' || normalized === 'process' || normalized === 'command') {
    return 'stdio';
  }
  if (!normalized || normalized === 'http' || normalized === 'https') {
    const endpointProtocol = endpoint?.trim().split(':', 1)[0]?.toLowerCase();
    if (!normalized && (endpointProtocol === 'ws' || endpointProtocol === 'wss')) {
      return 'websocket';
    }
    if (!normalized && (endpointProtocol === 'grpc' || endpointProtocol === 'grpcs')) {
      return 'grpc';
    }
    if (!normalized && command) {
      return 'stdio';
    }
    return 'http';
  }

  return undefined;
}

function parseGatewayConfigExternalMethod(
  value: string | undefined,
  fallback: GatewayConfigExternalSourceMethod
): GatewayConfigExternalSourceMethod {
  const normalized = value?.trim().toUpperCase();
  if (normalized === 'GET' || normalized === 'POST') {
    return normalized;
  }

  return fallback;
}

function parseAgentEventQueueConfig(
  value: AgentJsonConfig | undefined
): AgentEventQueueConfig {
  const eventQueue = isPlainObject(value?.eventQueue)
    ? (value?.eventQueue as AgentEventQueueJsonConfig)
    : undefined;

  return {
    enabled: resolveBoolean(process.env.AGENT_EVENT_QUEUE_ENABLED, eventQueue?.enabled, false),
    queueName:
      readString(process.env.AGENT_EVENT_QUEUE_NAME) ||
      readString(eventQueue?.queueName) ||
      'gateway-agent-events',
    jobName:
      readString(process.env.AGENT_EVENT_QUEUE_JOB_NAME) ||
      readString(eventQueue?.jobName) ||
      'agent.event',
    removeOnComplete: resolveInteger(
      [process.env.AGENT_EVENT_QUEUE_REMOVE_ON_COMPLETE, eventQueue?.removeOnComplete],
      1000,
      0
    ),
    removeOnFail: resolveInteger(
      [process.env.AGENT_EVENT_QUEUE_REMOVE_ON_FAIL, eventQueue?.removeOnFail],
      5000,
      0
    )
  };
}

function parseAgentEventWebhookConfig(
  value: AgentJsonConfig | undefined
): AgentEventWebhookConfig {
  const eventWebhook = isPlainObject(value?.eventWebhook)
    ? (value?.eventWebhook as AgentEventWebhookJsonConfig)
    : undefined;
  const headers = parseHeaderMap(eventWebhook?.headers);
  const apiKeyHeader = normalizeHeaderName(readString(process.env.AGENT_EVENT_WEBHOOK_API_KEY_HEADER), '');
  const apiKeyEnvName = readString(process.env.AGENT_EVENT_WEBHOOK_API_KEY_ENV);
  const apiKeyFromRefEnv = apiKeyEnvName ? readString(process.env[apiKeyEnvName]) : undefined;
  const apiKey = readString(process.env.AGENT_EVENT_WEBHOOK_API_KEY) || apiKeyFromRefEnv;
  if (apiKeyHeader && apiKey) {
    headers[apiKeyHeader] = apiKey;
  }

  const authorization = readString(process.env.AGENT_EVENT_WEBHOOK_AUTHORIZATION);
  if (authorization) {
    headers.authorization = authorization;
  }

  const endpoint =
    readString(process.env.AGENT_EVENT_WEBHOOK_ENDPOINT) ||
    readString(process.env.AGENT_EVENT_WEBHOOK_URL) ||
    readString(eventWebhook?.endpoint) ||
    readString(eventWebhook?.url);
  const command =
    readString(process.env.AGENT_EVENT_WEBHOOK_STDIO_COMMAND) ||
    readString(eventWebhook?.command);

  return {
    enabled: resolveBoolean(
      process.env.AGENT_EVENT_WEBHOOK_ENABLED,
      eventWebhook?.enabled,
      Boolean(endpoint || command)
    ),
    transport: parseExternalEventSinkTransport(
      readString(process.env.AGENT_EVENT_WEBHOOK_TRANSPORT) || readString(eventWebhook?.transport),
      endpoint,
      command
    ),
    endpoint,
    command,
    args: resolveStringList(process.env.AGENT_EVENT_WEBHOOK_STDIO_ARGS, eventWebhook?.args, []),
    cwd: readString(process.env.AGENT_EVENT_WEBHOOK_STDIO_CWD) || readString(eventWebhook?.cwd),
    env: parseHeaderMap(eventWebhook?.env),
    timeoutMs: resolveInteger(
      [process.env.AGENT_EVENT_WEBHOOK_TIMEOUT_MS, eventWebhook?.timeoutMs],
      5000,
      50
    ),
    ...parseExternalEventSinkRetryConfig(eventWebhook, 'AGENT_EVENT_WEBHOOK'),
    requireAck: parseExternalEventSinkRequireAck(eventWebhook, 'AGENT_EVENT_WEBHOOK'),
    headers
  };
}

function parseExternalEventSinkRetryConfig(
  value:
    | BillingWebhookJsonConfig
    | RawTraceSyncJsonConfig
    | AgentEventWebhookJsonConfig
    | undefined,
  envPrefix: string
): { maxAttempts: number; baseDelayMs: number; maxDelayMs: number } {
  const baseDelayMs = resolvePrecheckWindowMs(
    [process.env[`${envPrefix}_BASE_DELAY_MS`], value?.baseDelayMs],
    [process.env[`${envPrefix}_BASE_DELAY_SECONDS`], value?.baseDelaySeconds],
    200
  );
  const maxDelayCandidate = resolvePrecheckWindowMs(
    [process.env[`${envPrefix}_MAX_DELAY_MS`], value?.maxDelayMs],
    [process.env[`${envPrefix}_MAX_DELAY_SECONDS`], value?.maxDelaySeconds],
    2000
  );

  return {
    maxAttempts: resolveInteger(
      [process.env[`${envPrefix}_MAX_ATTEMPTS`], value?.maxAttempts],
      3,
      1
    ),
    baseDelayMs,
    maxDelayMs: Math.max(baseDelayMs, maxDelayCandidate)
  };
}

function parseExternalEventSinkRequireAck(
  value:
    | BillingWebhookJsonConfig
    | RawTraceSyncJsonConfig
    | AgentEventWebhookJsonConfig
    | undefined,
  envPrefix: string
): boolean {
  return resolveBoolean(
    process.env[`${envPrefix}_REQUIRE_ACK`] ||
      process.env[`${envPrefix}_WEBSOCKET_REQUIRE_ACK`],
    value?.requireAck,
    resolveBoolean(undefined, value?.websocketRequireAck, false)
  );
}

function parseExternalEventSinkTransport(
  value: string | undefined,
  endpoint?: string,
  command?: string
): GatewayExternalEventSinkTransport {
  const normalized = value?.trim().toLowerCase();
  if (normalized === 'stdio' || normalized === 'process' || normalized === 'command') {
    return 'stdio';
  }
  if (normalized === 'websocket' || normalized === 'ws' || normalized === 'wss') {
    return 'websocket';
  }
  if (normalized === 'grpc' || normalized === 'grpcs') {
    return 'grpc';
  }
  if (normalized === 'http' || normalized === 'https') {
    return 'http';
  }

  const endpointProtocol = endpoint?.trim().split(':', 1)[0]?.toLowerCase();
  if (endpointProtocol === 'ws' || endpointProtocol === 'wss') {
    return 'websocket';
  }
  if (endpointProtocol === 'grpc' || endpointProtocol === 'grpcs') {
    return 'grpc';
  }
  if (command) {
    return 'stdio';
  }

  return 'http';
}

function parseMcpServerTransportToken(value: unknown): AgentMcpServerTransport | undefined {
  const normalized = readString(value)?.toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'stdio' || normalized === 'process') {
    return 'stdio';
  }

  if (normalized === 'websocket' || normalized === 'ws' || normalized === 'wss') {
    return 'websocket';
  }

  return undefined;
}

function parseMcpStdioMessageModeToken(value: unknown): AgentMcpStdioMessageMode | undefined {
  const normalized = readString(value)?.toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'content-length' || normalized === 'content_length' || normalized === 'contentlength') {
    return 'content-length';
  }

  if (normalized === 'newline-json' || normalized === 'newline_json' || normalized === 'jsonl') {
    return 'newline-json';
  }

  return undefined;
}

function parseMcpServerConfigList(value: unknown): AgentMcpServerConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: AgentMcpServerConfig[] = [];
  const usedNames = new Set<string>();

  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const item = entry as AgentMcpServerJsonConfig;
    const transport = parseMcpServerTransportToken(item.transport) || 'stdio';
    const rawName =
      readString(item.name) ||
      (transport === 'websocket' ? readString(item.url) : readString(item.command));
    if (!rawName) {
      continue;
    }

    const name = uniqueProviderName(rawName, usedNames);
    const protocolVersion = readString(item.protocolVersion) || '2024-11-05';
    const startupTimeoutMs = resolveInteger([item.startupTimeoutMs], 10000, 100);
    const requestTimeoutMs = resolveInteger([item.requestTimeoutMs], 30000, 100);

    if (transport === 'websocket') {
      const url = readString(item.url);
      if (!url) {
        continue;
      }

      parsed.push({
        name,
        transport: 'websocket',
        url,
        headers: parseHeaderMap(item.headers),
        apiKey: readString(item.apiKey),
        apiKeyEnv: readString(item.apiKeyEnv),
        protocolVersion,
        startupTimeoutMs,
        requestTimeoutMs
      });
      continue;
    }

    const command = readString(item.command);
    if (!command) {
      continue;
    }

    const stdioMessageMode = parseMcpStdioMessageModeToken(item.stdioMessageMode) || 'content-length';
    const args = parseModelList(item.args);
    const env = parseHeaderMap(item.env);
    const cwd = readString(item.cwd) ? resolve(readString(item.cwd) as string) : undefined;

    parsed.push({
      name,
      transport: 'stdio',
      stdioMessageMode,
      command,
      args,
      env,
      cwd,
      protocolVersion,
      startupTimeoutMs,
      requestTimeoutMs
    });
  }

  return parsed;
}

function loadJsonConfig(configPath: string): GatewayJsonConfig {
  const configuredPath = readString(process.env.GATEWAY_CONFIG_PATH);

  if (!existsSync(configPath)) {
    if (configuredPath) {
      throw new Error(`JSON config file was not found: ${configPath}`);
    }

    return {};
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(configPath, 'utf8'));
  } catch (error) {
    const details = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to parse JSON config file "${configPath}": ${details}`);
  }

  if (!isPlainObject(parsed)) {
    throw new Error(`JSON config file "${configPath}" must contain a top-level object.`);
  }

  return parsed as GatewayJsonConfig;
}

function syncConfigObject(target: Record<string, unknown>, source: Record<string, unknown>): void {
  const targetKeys = Object.keys(target);
  for (const key of targetKeys) {
    if (!(key in source)) {
      delete target[key];
    }
  }

  for (const [key, value] of Object.entries(source)) {
    const current = target[key];
    target[key] = syncConfigValue(current, value);
  }
}

function syncConfigValue(current: unknown, next: unknown): unknown {
  if (Array.isArray(next)) {
    if (Array.isArray(current)) {
      current.length = 0;
      for (const item of next) {
        current.push(cloneConfigValue(item));
      }
      return current;
    }

    return next.map((item) => cloneConfigValue(item));
  }

  if (isPlainObject(next)) {
    const targetObject = isPlainObject(current) ? current : {};
    syncConfigObject(targetObject, next);
    return targetObject;
  }

  return next;
}

function cloneConfigValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => cloneConfigValue(item));
  }

  if (isPlainObject(value)) {
    const cloned: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      cloned[key] = cloneConfigValue(item);
    }
    return cloned;
  }

  return value;
}

function parseProvidersConfig(value: unknown): ProviderConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: ProviderConfig[] = [];
  const usedNames = new Set<string>();

  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const item = entry as ProviderJsonConfig;
    const typeToken = readString(item.type) || readString(item.provider);
    const type = parseProviderTypeToken(typeToken);
    if (!type) {
      continue;
    }

    const models = parseModelList(item.models);
    const name = uniqueProviderName(readString(item.name) || typeToken || type, usedNames);
    const apiKeyEnvName = readString(item.apiKeyEnv) || readString(item.apikeyEnv);
    const apiKeyFromEnv = apiKeyEnvName ? readString(process.env[apiKeyEnvName]) : undefined;

    const providerConfig: ProviderConfig = {
      name,
      type,
      apikey: readString(item.apikey) || readString(item.apiKey) || apiKeyFromEnv,
      apiKeyEnv: apiKeyEnvName,
      baseurl: normalizeBaseUrl(readString(item.baseurl) || readString(item.baseUrl)),
      models,
      openaiChatToolsFormat: parseOpenAIChatToolsFormatToken(
        readString(item.openaiChatToolsFormat) ||
          readString(item.chatToolsFormat) ||
          readString(item.toolsFormat)
      ),
      openaiChatStreamUsage: parseOpenAIChatStreamUsageToken(
        item.openaiChatStreamUsage ?? item.chatStreamUsage ?? item.streamUsage
      ),
      extraHeaders: parseModelScopedHeaders(item.extraHeaders, models),
      extraBody: parseModelScopedBody(item.extraBody, models),
      billing: parseModelScopedBilling(item.billing, models),
      health: parseProviderHealthConfig(item)
    };

    parsed.push(providerConfig);
  }

  return parsed;
}

function parseProviderPluginsConfig(value: unknown): ProviderPluginConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: ProviderPluginConfig[] = [];
  const usedKeys = new Set<string>();

  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const item = entry as ProviderPluginJsonConfig;
    const keyBase = readString(item.key);
    if (!keyBase) {
      continue;
    }

    const enabled = readBoolean(item.enabled);
    if (enabled === false) {
      continue;
    }

    const auth = parseProviderPluginMutation(item.auth);
    const request = parseProviderPluginMutation(item.request);
    const response = parseProviderPluginResponseMutation(item.response);
    const codexOauth = parseProviderPluginCodexOauth(item.codexOauth);
    const deepseekThinking = parseProviderPluginDeepSeekThinking(
      item.deepseekThinking ?? item.deepSeekThinking
    );
    if (!auth && !request && !response && !codexOauth && !deepseekThinking) {
      continue;
    }

    const provider = parseProvider(readString(item.provider));
    const providerName = readString(item.providerName);
    const key = uniqueProviderName(keyBase, usedKeys);

    parsed.push({
      key,
      enabled: true,
      provider,
      providerName,
      codexOauth,
      deepseekThinking,
      auth,
      request,
      response
    });
  }

  return parsed;
}

function parseVirtualModelProfilesConfig(value: unknown): VirtualModelProfileConfig[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: VirtualModelProfileConfig[] = [];
  const usedKeys = new Set<string>();

  for (const entry of value) {
    if (!isPlainObject(entry)) {
      continue;
    }

    const id = readString(entry.id);
    const keyBase = readString(entry.key);
    const displayName = readString(entry.displayName);
    if (!id || !keyBase || !displayName) {
      continue;
    }

    const enabled = readBoolean(entry.enabled);
    if (enabled === false) {
      continue;
    }

    const match = parseVirtualModelMatchConfig(entry.match);
    if (
      match.exactAliases.length === 0 &&
      match.prefixes.length === 0 &&
      match.suffixes.length === 0
    ) {
      continue;
    }

    const key = uniqueProviderName(keyBase, usedKeys);
    parsed.push({
      id,
      key,
      displayName,
      description: readString(entry.description),
      enabled: true,
      match,
      baseModel: parseVirtualModelBaseModelConfig(entry.baseModel),
      instructions: parseVirtualModelInstructionsConfig(entry.instructions),
      tools: parseVirtualModelToolsConfig(entry.tools),
      toolChoice: cloneUnknown(entry.toolChoice),
      execution: parseVirtualModelExecutionConfig(entry.execution),
      materialization: parseVirtualModelMaterializationConfig(entry.materialization),
      metadata: readRecord(entry.metadata)
    });
  }

  return parsed;
}

function parseVirtualModelMatchConfig(value: unknown): VirtualModelProfileConfig['match'] {
  const raw = isPlainObject(value) ? value : {};
  return {
    exactAliases: parseModelList((raw as Record<string, unknown>).exactAliases),
    prefixes: parseModelList((raw as Record<string, unknown>).prefixes),
    suffixes: parseModelList((raw as Record<string, unknown>).suffixes)
  };
}

function parseVirtualModelBaseModelConfig(
  value: unknown
): VirtualModelProfileConfig['baseModel'] | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value as Record<string, unknown>;
  const modeRaw = readString(raw.mode)?.toLowerCase();
  const mode =
    modeRaw === 'request' ||
    modeRaw === 'fixed' ||
    modeRaw === 'strip_prefix' ||
    modeRaw === 'strip_suffix'
      ? modeRaw
      : undefined;
  const fixedModel = readString(raw.fixedModel);
  if (!mode && !fixedModel) {
    return undefined;
  }

  return {
    mode,
    fixedModel
  };
}

function parseVirtualModelInstructionsConfig(
  value: unknown
): VirtualModelProfileConfig['instructions'] | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value as Record<string, unknown>;
  const prepend = readString(raw.prepend);
  const append = readString(raw.append);
  const replace = readString(raw.replace);
  if (!prepend && !append && !replace) {
    return undefined;
  }

  return {
    prepend,
    append,
    replace
  };
}

function parseVirtualModelToolsConfig(value: unknown): VirtualModelProfileConfig['tools'] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parsed: VirtualModelProfileConfig['tools'] = [];
  const usedNames = new Set<string>();

  for (const entry of value) {
    const raw = isPlainObject(entry) ? (entry as Record<string, unknown>) : undefined;
    const name = readString(raw?.name);
    if (!name || usedNames.has(name)) {
      continue;
    }

    usedNames.add(name);
    const visibilityRaw = readString(raw?.visibility)?.toLowerCase();
    parsed.push({
      name,
      description: readString(raw?.description),
      inputSchema: readRecord(raw?.inputSchema ?? raw?.input_schema ?? raw?.parameters),
      visibility: visibilityRaw === 'client' ? 'client' : 'internal'
    });
  }

  return parsed;
}

function parseVirtualModelExecutionConfig(
  value: unknown
): VirtualModelProfileConfig['execution'] {
  const raw = isPlainObject(value) ? (value as Record<string, unknown>) : {};
  const modeRaw = readString(raw.mode)?.toLowerCase();
  const policyRaw = readString(raw.clientToolsPolicy)?.toLowerCase();
  const streamModeRaw = readString(raw.streamMode ?? raw.stream_mode)?.toLowerCase();

  return {
    mode: modeRaw === 'decorate_only' ? 'decorate_only' : 'tool_loop',
    maxTurns: resolveInteger([raw.maxTurns], 6, 1),
    maxToolCalls: resolveInteger([raw.maxToolCalls], 8, 1),
    clientToolsPolicy: policyRaw === 'deny' ? 'deny' : 'allow',
    matchMultimodal: readBoolean(raw.matchMultimodal ?? raw.match_multimodal) ?? false,
    matchWebSearch:
      readBoolean(raw.matchWebSearch ?? raw.match_web_search ?? raw.matchWebsearch) ?? false,
    streamMode: streamModeRaw === 'optimistic' ? 'optimistic' : 'buffered'
  };
}

function parseVirtualModelMaterializationConfig(
  value: unknown
): VirtualModelProfileConfig['materialization'] {
  const raw = isPlainObject(value) ? (value as Record<string, unknown>) : {};
  return {
    enabled: readBoolean(raw.enabled) ?? true,
    includeInGatewayModels: readBoolean(raw.includeInGatewayModels) ?? true,
    displayNameTemplate: readString(raw.displayNameTemplate),
    descriptionTemplate: readString(raw.descriptionTemplate)
  };
}

function parseProviderPluginDeepSeekThinking(value: unknown): ProviderPluginDeepSeekThinkingConfig | undefined {
  if (value === true) {
    return { enabled: true };
  }

  if (!isPlainObject(value)) {
    return undefined;
  }

  const enabled = readBoolean((value as { enabled?: unknown }).enabled);
  if (enabled === false) {
    return undefined;
  }

  return { enabled: true };
}

function parseProviderPluginCodexOauth(value: unknown): ProviderPluginCodexOAuthConfig | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value as ProviderPluginCodexOauthJsonConfig;
  const enabled = readBoolean(raw.enabled);
  if (enabled === false) {
    return undefined;
  }

  const hasAccessToken = Object.prototype.hasOwnProperty.call(raw, 'accessToken');
  const hasRefreshToken = Object.prototype.hasOwnProperty.call(raw, 'refreshToken');
  const hasAccountId =
    Object.prototype.hasOwnProperty.call(raw, 'accountId') ||
    Object.prototype.hasOwnProperty.call(raw, 'account_id');
  const accessToken = hasAccessToken ? cloneUnknown(raw.accessToken) : undefined;
  const refreshToken = hasRefreshToken ? cloneUnknown(raw.refreshToken) : undefined;
  const accountId = hasAccountId ? cloneUnknown(raw.accountId ?? raw.account_id) : undefined;
  if (!hasAccessToken && !hasRefreshToken) {
    return undefined;
  }

  return {
    enabled: true,
    tokenEndpoint:
      readString(raw.tokenEndpoint) ||
      readString(process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE) ||
      defaultCodexOauthTokenEndpoint,
    clientId: readString(raw.clientId) || defaultCodexOauthClientId,
    scope: normalizeCodexOauthScope(readString(raw.scope)),
    accessToken,
    refreshToken,
    accountId,
    refreshIfMissingAccessToken: readBoolean(raw.refreshIfMissingAccessToken) ?? true,
    forceRefresh: readBoolean(raw.forceRefresh) ?? false,
    required: readBoolean(raw.required) ?? true,
    timeoutMs: resolveInteger([raw.timeoutMs], 8000, 1),
    authHeader: normalizeHeaderName(readString(raw.authHeader), 'authorization'),
    authScheme: readString(raw.authScheme) || 'Bearer'
  };
}

function normalizeCodexOauthScope(scope: string | undefined): string {
  const ordered: string[] = [];
  const seen = new Set<string>();

  const pushTokens = (value: string | undefined) => {
    if (!value) {
      return;
    }
    for (const token of value.split(/\s+/)) {
      const normalized = token.trim();
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      ordered.push(normalized);
    }
  };

  pushTokens(scope || defaultCodexOauthScope);
  for (const requiredScope of requiredCodexOauthScopes) {
    pushTokens(requiredScope);
  }

  return ordered.join(' ');
}

function parseProviderPluginMutation(value: unknown): ProviderPluginMutationConfig | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value as ProviderPluginMutationJsonConfig;
  const mutation: ProviderPluginMutationConfig = {
    strict: readBoolean(raw.strict) ?? false,
    headers: parsePluginValueMap(raw.headers),
    query: parsePluginValueMap(raw.query),
    removeHeaders: parseModelList(raw.removeHeaders),
    removeQuery: parseModelList(raw.removeQuery),
    bodySet: parsePluginValueMap(raw.bodySet),
    bodyMerge: parsePluginValueMap(raw.bodyMerge),
    bodyRemove: parseModelList(raw.bodyRemove)
  };

  if (
    Object.keys(mutation.headers).length === 0 &&
    Object.keys(mutation.query).length === 0 &&
    mutation.removeHeaders.length === 0 &&
    mutation.removeQuery.length === 0 &&
    Object.keys(mutation.bodySet).length === 0 &&
    Object.keys(mutation.bodyMerge).length === 0 &&
    mutation.bodyRemove.length === 0
  ) {
    return undefined;
  }

  return mutation;
}

function parseProviderPluginResponseMutation(
  value: unknown
): ProviderPluginResponseMutationConfig | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value as ProviderPluginResponseMutationJsonConfig;
  const mutation: ProviderPluginResponseMutationConfig = {
    strict: readBoolean(raw.strict) ?? false,
    bodySet: parsePluginValueMap(raw.bodySet),
    bodyMerge: parsePluginValueMap(raw.bodyMerge),
    bodyRemove: parseModelList(raw.bodyRemove)
  };

  if (
    Object.keys(mutation.bodySet).length === 0 &&
    Object.keys(mutation.bodyMerge).length === 0 &&
    mutation.bodyRemove.length === 0
  ) {
    return undefined;
  }

  return mutation;
}

function parsePluginValueMap(value: unknown): Record<string, unknown> {
  if (!isPlainObject(value)) {
    return {};
  }

  const mapped: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    const normalizedKey = key.trim();
    if (!normalizedKey) {
      continue;
    }

    mapped[normalizedKey] = cloneUnknown(item);
  }

  return mapped;
}

function parseModelList(value: unknown): string[] {
  if (typeof value === 'string') {
    return dedupeStrings(
      value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
    );
  }

  if (!Array.isArray(value)) {
    return [];
  }

  return dedupeStrings(value.filter((item): item is string => typeof item === 'string').map((item) => item.trim()));
}

function resolveStringList(envValue: unknown, fileValue: unknown, fallback: string[]): string[] {
  const fromEnv = parseModelList(envValue);
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  const fromFile = parseModelList(fileValue);
  if (fromFile.length > 0) {
    return fromFile;
  }

  return [...fallback];
}

function parseProviderTypeToken(value: string | undefined): ProviderType | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (
    normalized === 'openai_chat_completions' ||
    normalized === 'chat_completions' ||
    normalized === 'openai-chat-completions' ||
    normalized === 'openai_chat'
  ) {
    return 'openai_chat_completions';
  }

  if (
    normalized === 'openai_responses' ||
    normalized === 'responses' ||
    normalized === 'openai-responses' ||
    normalized === 'openai_response' ||
    normalized === 'openai'
  ) {
    return 'openai_responses';
  }

  if (
    normalized === 'anthropic_messages' ||
    normalized === 'anthropic-messages' ||
    normalized === 'anthropic_message' ||
    normalized === 'anthropic' ||
    normalized === 'claude'
  ) {
    return 'anthropic_messages';
  }

  if (
    normalized === 'gemini_generate_content' ||
    normalized === 'gemini-generate-content' ||
    normalized === 'generate_content' ||
    normalized === 'gemini' ||
    normalized === 'google'
  ) {
    return 'gemini_generate_content';
  }

  return undefined;
}

function parseOpenAIChatToolsFormatToken(value: string | undefined): ProviderConfig['openaiChatToolsFormat'] {
  const normalized = value?.trim().toLowerCase().replace(/[-\s]+/g, '_');
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'openai' || normalized === 'openai_function' || normalized === 'openai_functions') {
    return 'openai';
  }

  if (
    normalized === 'anthropic' ||
    normalized === 'anthropic_tool' ||
    normalized === 'anthropic_tools' ||
    normalized === 'input_schema'
  ) {
    return 'anthropic';
  }

  return undefined;
}

function parseOpenAIChatStreamUsageToken(value: unknown): ProviderConfig['openaiChatStreamUsage'] {
  if (typeof value === 'boolean') {
    return value ? 'include_usage' : 'disabled';
  }

  const normalized = readString(value)?.trim().toLowerCase().replace(/[-\s]+/g, '_');
  if (!normalized) {
    return undefined;
  }

  if (
    normalized === 'include_usage' ||
    normalized === 'includeusage' ||
    normalized === 'enabled' ||
    normalized === 'enable' ||
    normalized === 'true' ||
    normalized === 'on'
  ) {
    return 'include_usage';
  }

  if (
    normalized === 'disabled' ||
    normalized === 'disable' ||
    normalized === 'false' ||
    normalized === 'off' ||
    normalized === 'none'
  ) {
    return 'disabled';
  }

  return undefined;
}

function parseModelScopedHeaders(value: unknown, models: string[]): ModelScopedHeadersConfig {
  if (!isPlainObject(value)) {
    return {
      default: {},
      byModel: {}
    };
  }

  if (!hasExplicitModelScope(value, models)) {
    return {
      default: parseHeaderMap(value),
      byModel: {}
    };
  }

  const byModel: Record<string, Record<string, string>> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'default') {
      continue;
    }

    if (!isPlainObject(entry)) {
      continue;
    }

    const headers = parseHeaderMap(entry);
    if (Object.keys(headers).length > 0) {
      byModel[key] = headers;
    }
  }

  return {
    default: parseHeaderMap((value as Record<string, unknown>).default),
    byModel
  };
}

function parseModelScopedBody(value: unknown, models: string[]): ModelScopedBodyConfig {
  if (!isPlainObject(value)) {
    return {
      default: {},
      byModel: {}
    };
  }

  if (!hasExplicitModelScope(value, models)) {
    return {
      default: cloneObject(value),
      byModel: {}
    };
  }

  const byModel: Record<string, Record<string, unknown>> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'default') {
      continue;
    }

    if (!isPlainObject(entry)) {
      continue;
    }

    byModel[key] = cloneObject(entry);
  }

  return {
    default: cloneObject((value as Record<string, unknown>).default),
    byModel
  };
}

function parseModelScopedBilling(value: unknown, models: string[]): ModelScopedBillingConfig {
  const directRate = parseBillingRate(value);
  if (directRate) {
    return {
      default: directRate,
      byModel: {}
    };
  }

  if (!isPlainObject(value)) {
    return {
      byModel: {}
    };
  }

  const byModel: Record<string, BillingRate> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'default') {
      continue;
    }

    if (models.length > 0 && !models.includes(key)) {
      continue;
    }

    const rate = parseBillingRate(entry);
    if (rate) {
      byModel[key] = rate;
    }
  }

  return {
    default: parseBillingRate((value as Record<string, unknown>).default),
    byModel
  };
}

function parseBillingRate(value: unknown): BillingRate | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const input = readNonNegativeNumber((value as BillingRateJsonConfig).inputPerMillionUsd);
  const output = readNonNegativeNumber((value as BillingRateJsonConfig).outputPerMillionUsd);
  const cacheRead = readNonNegativeNumber((value as BillingRateJsonConfig).cacheReadPerMillionUsd);
  const cacheWrite = readNonNegativeNumber((value as BillingRateJsonConfig).cacheWritePerMillionUsd);
  const tiers = parseBillingTierSet((value as BillingRateJsonConfig).tiers);
  if (input === undefined && output === undefined && cacheRead === undefined && cacheWrite === undefined && !tiers) {
    return undefined;
  }

  const rate: BillingRate = {
    inputPerMillionUsd: input ?? 0,
    outputPerMillionUsd: output ?? 0
  };

  if (cacheRead !== undefined) {
    rate.cacheReadPerMillionUsd = cacheRead;
  }

  if (cacheWrite !== undefined) {
    rate.cacheWritePerMillionUsd = cacheWrite;
  }

  if (tiers) {
    rate.tiers = tiers;
  }

  return rate;
}

function parseBillingTierSet(value: unknown): BillingRate['tiers'] | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const input = parseBillingTierList((value as Record<string, unknown>).input);
  const output = parseBillingTierList((value as Record<string, unknown>).output);
  const cacheRead = parseBillingTierList((value as Record<string, unknown>).cacheRead);
  const cacheWrite = parseBillingTierList((value as Record<string, unknown>).cacheWrite);

  if (!input && !output && !cacheRead && !cacheWrite) {
    return undefined;
  }

  const parsed: NonNullable<BillingRate['tiers']> = {};
  if (input) {
    parsed.input = input;
  }

  if (output) {
    parsed.output = output;
  }

  if (cacheRead) {
    parsed.cacheRead = cacheRead;
  }

  if (cacheWrite) {
    parsed.cacheWrite = cacheWrite;
  }

  return parsed;
}

function parseBillingTierList(value: unknown): BillingTier[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const tiers: BillingTier[] = [];
  for (const item of value) {
    if (!isPlainObject(item)) {
      continue;
    }

    const tier = item as BillingTierJsonConfig;
    const perMillionUsd = readNonNegativeNumber(tier.perMillionUsd);
    if (perMillionUsd === undefined) {
      continue;
    }

    const upToRaw = readFiniteNumber(tier.upToTokens);
    const upToTokens = upToRaw !== undefined ? Math.max(0, Math.trunc(upToRaw)) : undefined;
    tiers.push({
      upToTokens: upToTokens && upToTokens > 0 ? upToTokens : undefined,
      perMillionUsd
    });
  }

  if (tiers.length === 0) {
    return undefined;
  }

  tiers.sort((a, b) => {
    const aUpper = a.upToTokens ?? Number.POSITIVE_INFINITY;
    const bUpper = b.upToTokens ?? Number.POSITIVE_INFINITY;
    return aUpper - bUpper;
  });

  return tiers;
}

function parseHeaderMap(value: unknown): Record<string, string> {
  if (!isPlainObject(value)) {
    return {};
  }

  const mapped: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === 'string') {
      mapped[key] = raw;
      continue;
    }

    if (typeof raw === 'number' || typeof raw === 'boolean') {
      mapped[key] = String(raw);
    }
  }

  return mapped;
}

function resolveDefaultTargetProviders(
  envValue: string | undefined,
  providerConfigs: ProviderConfig[]
): Provider[] {
  const fromEnv = parseProviderList(envValue);
  if (fromEnv.length > 0) {
    return fromEnv;
  }

  return dedupeProviderTypes(providerConfigs.map((item) => providerFromProviderType(item.type)));
}

function findProviderConfigByType(
  providers: ProviderConfig[],
  type: Provider
): ProviderConfig | undefined {
  return providers.find((item) => providerFromProviderType(item.type) === type);
}

function resolveNonNegativeNumber(values: unknown[], fallback: number): number {
  for (const value of values) {
    const parsed = readNonNegativeNumber(value);
    if (parsed !== undefined) {
      return parsed;
    }
  }

  return fallback;
}

function resolvePositiveNumber(values: unknown[], fallback: number): number {
  for (const value of values) {
    const parsed = readFiniteNumber(value);
    if (parsed !== undefined && parsed > 0) {
      return parsed;
    }
  }

  return fallback;
}

function resolveInteger(values: unknown[], fallback: number, minValue: number): number {
  for (const value of values) {
    const parsed = readFiniteNumber(value);
    if (parsed === undefined) {
      continue;
    }

    const normalized = Math.trunc(parsed);
    if (normalized < minValue) {
      continue;
    }

    return normalized;
  }

  return fallback;
}

function resolveBoolean(envValue: string | undefined, fileValue: unknown, fallback: boolean): boolean {
  const fromEnv = readBoolean(envValue);
  if (fromEnv !== undefined) {
    return fromEnv;
  }

  const fromFile = readBoolean(fileValue);
  if (fromFile !== undefined) {
    return fromFile;
  }

  return fallback;
}

function readString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

function readRecord(value: unknown): Record<string, unknown> | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function normalizeHeaderName(value: string | undefined, fallback: string): string {
  const normalized = value?.trim().toLowerCase();
  if (normalized) {
    return normalized;
  }

  return fallback;
}

function normalizeBaseUrl(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  return trimTrailingSlash(value);
}

function normalizeHttpPath(value: string): string {
  if (!value.startsWith('/')) {
    return `/${value}`;
  }

  return value;
}

function ensureMcpPathPrefix(path: string, fieldName: string): string {
  const normalized = path.trim();
  if (normalized === '/mcp' || normalized.startsWith('/mcp/')) {
    return normalized;
  }

  throw new Error(`${fieldName} must start with "/mcp" for path protection.`);
}

function readFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return undefined;
}

function readNonNegativeNumber(value: unknown): number | undefined {
  const parsed = readFiniteNumber(value);
  if (parsed === undefined || parsed < 0) {
    return undefined;
  }

  return parsed;
}

function readBoolean(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') {
    return true;
  }

  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') {
    return false;
  }

  return undefined;
}

function hasExplicitModelScope(value: Record<string, unknown>, models: string[]): boolean {
  if ('default' in value) {
    return true;
  }

  for (const model of models) {
    if (model in value) {
      return true;
    }
  }

  return false;
}

function cloneObject(value: unknown): Record<string, unknown> {
  if (!isPlainObject(value)) {
    return {};
  }

  return cloneUnknown(value) as Record<string, unknown>;
}

function cloneUnknown(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => cloneUnknown(item));
  }

  if (!isPlainObject(value)) {
    return value;
  }

  const cloned: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    cloned[key] = cloneUnknown(item);
  }

  return cloned;
}

function uniqueProviderName(baseName: string, usedNames: Set<string>): string {
  if (!usedNames.has(baseName)) {
    usedNames.add(baseName);
    return baseName;
  }

  let index = 2;
  while (usedNames.has(`${baseName}-${index}`)) {
    index += 1;
  }

  const nextName = `${baseName}-${index}`;
  usedNames.add(nextName);
  return nextName;
}

function dedupeStrings(values: string[]): string[] {
  const deduped: string[] = [];
  for (const value of values) {
    if (!value || deduped.includes(value)) {
      continue;
    }

    deduped.push(value);
  }

  return deduped;
}

function dedupeProviderTypes(values: Provider[]): Provider[] {
  const deduped: Provider[] = [];
  for (const provider of values) {
    if (!deduped.includes(provider)) {
      deduped.push(provider);
    }
  }

  return deduped;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
