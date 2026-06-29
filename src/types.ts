import type { FastifyRequest } from 'fastify';

export type Provider = 'openai' | 'anthropic' | 'gemini';
export type ProviderType =
  | 'openai_responses'
  | 'openai_chat_completions'
  | 'anthropic_messages'
  | 'gemini_generate_content';

export interface BillingTier {
  upToTokens?: number;
  perMillionUsd: number;
}

export interface BillingTierSet {
  input?: BillingTier[];
  output?: BillingTier[];
  cacheRead?: BillingTier[];
  cacheWrite?: BillingTier[];
}

export interface BillingRate {
  inputPerMillionUsd: number;
  outputPerMillionUsd: number;
  cacheReadPerMillionUsd?: number;
  cacheWritePerMillionUsd?: number;
  tiers?: BillingTierSet;
}

export interface ModelScopedHeadersConfig {
  default: Record<string, string>;
  byModel: Record<string, Record<string, string>>;
}

export interface ModelScopedBodyConfig {
  default: Record<string, unknown>;
  byModel: Record<string, Record<string, unknown>>;
}

export interface ModelScopedBillingConfig {
  default?: BillingRate;
  byModel: Record<string, BillingRate>;
}

export type ProviderHealthStatus = 'healthy' | 'degraded' | 'unknown' | 'down';

export interface ProviderHealthConfig {
  status: ProviderHealthStatus;
  available?: boolean;
  priority?: number;
  latencyMs?: number;
  checkedAt?: string;
}

export interface ProviderConfig {
  name: string;
  type: ProviderType;
  apikey?: string;
  apiKeyEnv?: string;
  baseurl?: string;
  models: string[];
  openaiChatToolsFormat?: 'openai' | 'anthropic';
  openaiChatStreamUsage?: 'include_usage' | 'disabled';
  extraHeaders: ModelScopedHeadersConfig;
  extraBody: ModelScopedBodyConfig;
  billing: ModelScopedBillingConfig;
  health?: ProviderHealthConfig;
}

export interface ProviderExternalSourceConfig {
  enabled: boolean;
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  apiKeyHeader: string;
  apiKey?: string;
  headers: Record<string, string>;
}

export type GatewayConfigExternalSourceTransport = 'http' | 'websocket' | 'grpc' | 'stdio';
export type GatewayConfigExternalSourceMethod = 'GET' | 'POST';

export interface GatewayConfigExternalSourceConfig {
  enabled: boolean;
  transport: GatewayConfigExternalSourceTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  method: GatewayConfigExternalSourceMethod;
  timeoutMs: number;
  intervalMs: number;
  apiKeyHeader: string;
  apiKey?: string;
  headers: Record<string, string>;
}

export interface GatewayPolicyRuleConfig {
  allowProviders: Provider[];
  denyProviders: Provider[];
  allowProviderNames: string[];
  denyProviderNames: string[];
  allowModels: string[];
  denyModels: string[];
  allowProviderModels: string[];
  denyProviderModels: string[];
}

export interface GatewayPolicyConfig {
  enabled: boolean;
  defaults: GatewayPolicyRuleConfig;
  byUser: Record<string, GatewayPolicyRuleConfig>;
  byTenant: Record<string, GatewayPolicyRuleConfig>;
  byOrganization: Record<string, GatewayPolicyRuleConfig>;
  bySubject: Record<string, GatewayPolicyRuleConfig>;
  byPlan: Record<string, GatewayPolicyRuleConfig>;
  byApiKey: Record<string, GatewayPolicyRuleConfig>;
}

export interface BillingConfig {
  enabled: boolean;
  currency: 'USD';
  rates: Record<Provider, BillingRate>;
}

export interface BillingQueueConfig {
  enabled: boolean;
  queueName: string;
  jobName: string;
  removeOnComplete: number;
  removeOnFail: number;
}

export type GatewayExternalEventSinkTransport = 'http' | 'websocket' | 'grpc' | 'stdio';

export interface BillingWebhookConfig {
  enabled: boolean;
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  requireAck: boolean;
  headers: Record<string, string>;
}

export type RawTraceCaptureMode = 'disabled' | 'body_redacted' | 'body_full' | 'wire_raw';
export type RawTracePartType =
  | 'client_request_metadata'
  | 'client_request'
  | 'upstream_request_metadata'
  | 'upstream_request'
  | 'upstream_response_metadata'
  | 'upstream_response'
  | 'response_stream';
export type RawTraceStorageBackend = 'local';

export interface RawTraceSyncConfig {
  enabled: boolean;
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  requireAck: boolean;
  apiKeyHeader: string;
  apiKey?: string;
  authorization?: string;
  headers: Record<string, string>;
}

export interface RawTraceConfig {
  enabled: boolean;
  mode: RawTraceCaptureMode;
  spoolDir: string;
  maxPartBytes: number;
  uploaderConcurrency: number;
  maxAttempts: number;
  baseDelayMs: number;
  sync: RawTraceSyncConfig;
}

export interface GatewayAuthIdentityHeadersConfig {
  userId: string;
  tenantId: string;
  subject: string;
  organizationId: string;
  plan: string;
  apiKeyId?: string;
}

export interface GatewayAuthSignatureConfig {
  enabled: boolean;
  header: string;
  timestampHeader: string;
  secretEnv: string;
  maxSkewSec: number;
}

export type GatewayAuthMode = 'trusted_header' | 'http_introspection' | 'static_api_key';

export interface GatewayAuthStaticApiKeysConfig {
  keys: string[];
  keyEnv?: string;
  keyHeader: string;
  keyBearerOnly: boolean;
}

export interface GatewayAuthIntrospectionResponseMapConfig {
  active: string;
  userId: string;
  tenantId: string;
  subject: string;
  organizationId: string;
  plan: string;
  apiKeyId?: string;
}

export interface GatewayAuthIntrospectionConfig {
  endpoint?: string;
  timeoutMs: number;
  tokenHeader: string;
  tokenBearerOnly: boolean;
  requestTokenField: string;
  credentialHeader: string;
  credentialEnv: string;
  responseMap: GatewayAuthIntrospectionResponseMapConfig;
}

export interface GatewayAuthConfig {
  enabled: boolean;
  mode: GatewayAuthMode;
  required: boolean;
  trustedCidrs: string[];
  identityHeaders: GatewayAuthIdentityHeadersConfig;
  signature: GatewayAuthSignatureConfig;
  introspection: GatewayAuthIntrospectionConfig;
  staticApiKeys?: GatewayAuthStaticApiKeysConfig;
}

export interface GatewayRequestIdentity {
  source: GatewayAuthMode;
  billingSubjectKey: string;
  userId?: string;
  tenantId?: string;
  subject?: string;
  organizationId?: string;
  plan?: string;
  apiKeyId?: string;
}

export type GatewayPrecheckSubject =
  | 'identity'
  | 'user'
  | 'tenant'
  | 'organization'
  | 'api_key'
  | 'ip'
  | 'header'
  | 'global';

export type GatewayPrecheckScope = 'global' | 'provider' | 'model' | 'provider_model';
export type GatewayRateLimitMetric = 'requests' | 'tokens' | 'images';

export interface GatewayPrecheckRuleBaseConfig {
  enabled: boolean;
  windowMs: number;
  subject: GatewayPrecheckSubject;
  scope: GatewayPrecheckScope;
  headerName?: string;
}

export interface GatewayRateLimitDimensionConfig extends GatewayPrecheckRuleBaseConfig {
  name: string;
  metric: GatewayRateLimitMetric;
  max: number;
}

export interface GatewayRateLimitPrecheckConfig extends GatewayPrecheckRuleBaseConfig {
  maxRequests: number;
  rpm: number;
  rpd: number;
  tpm: number;
  tpd: number;
  ipm: number;
  limits: GatewayRateLimitDimensionConfig[];
}

export interface GatewayQuotaPrecheckConfig extends GatewayPrecheckRuleBaseConfig {
  maxTokens: number;
}

export interface GatewayBudgetPrecheckConfig extends GatewayPrecheckRuleBaseConfig {
  maxCostUsd: number;
}

export interface GatewayPrecheckEstimationConfig {
  charsPerToken: number;
  defaultMaxOutputTokens: number;
}

export type GatewayPrecheckStorageType = 'memory';

export interface GatewayPrecheckStorageConfig {
  type: GatewayPrecheckStorageType;
}

export interface GatewayPrecheckConfig {
  enabled: boolean;
  rateLimit: GatewayRateLimitPrecheckConfig;
  quota: GatewayQuotaPrecheckConfig;
  budget: GatewayBudgetPrecheckConfig;
  estimation: GatewayPrecheckEstimationConfig;
  storage: GatewayPrecheckStorageConfig;
}

export interface GatewayHealthAwareRoutingConfig {
  enabled: boolean;
  skipUnavailable: boolean;
  unhealthyStatuses: ProviderHealthStatus[];
  preferHealthy: boolean;
  preferLowerLatency: boolean;
}

export interface GatewayRoutingConfig {
  preferSourceProviderForBareModels: boolean;
}

export interface GatewayModelListConfig {
  bareModelIds: boolean;
}

export interface ProviderHealthCheckSchedulerConfig {
  enabled: boolean;
  intervalMs: number;
  timeoutMs: number;
  initialDelayMs: number;
}

export interface GatewayMetricsConfig {
  enabled: boolean;
  includeProviderHealth: boolean;
}

export interface GatewayCorsConfig {
  enabled: boolean;
  origins: string[];
  allowedHeaders: string[];
  allowedMethods: string[];
  allowCredentials: boolean;
  maxAgeSeconds: number;
}

export interface GatewayIdempotencyConfig {
  enabled: boolean;
  headerName: string;
  ttlMs: number;
  maxEntries: number;
  cacheErrorResponses: boolean;
}

export interface GatewayUpstreamConcurrencyConfig {
  enabled: boolean;
  maxInFlightPerProvider: number;
  queueTimeoutMs: number;
}

export interface GatewayUpstreamCircuitBreakerConfig {
  enabled: boolean;
  failureThreshold: number;
  cooldownMs: number;
  failureStatusCodes: number[];
}

export interface GatewayUpstreamRetryConfig {
  enabled: boolean;
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
  jitterMs: number;
  retryStatusCodes: number[];
}

export type GatewayTransparentToolUnknownPolicy = 'return_to_client' | 'fail';

export interface GatewayTransparentToolExecutionConfig {
  enabled: boolean;
  maxTurns: number;
  maxToolCalls: number;
  requireClientDeclaration: boolean;
  unknownToolPolicy: GatewayTransparentToolUnknownPolicy;
  allowTools: string[];
  denyTools: string[];
}

export type AgentMcpServerTransport = 'stdio' | 'websocket';
export type AgentMcpStdioMessageMode = 'content-length' | 'newline-json';

export interface AgentMcpServerBaseConfig {
  name: string;
  transport: AgentMcpServerTransport;
  protocolVersion: string;
  startupTimeoutMs: number;
  requestTimeoutMs: number;
}

export interface AgentMcpStdioServerConfig extends AgentMcpServerBaseConfig {
  transport: 'stdio';
  stdioMessageMode: AgentMcpStdioMessageMode;
  command: string;
  args: string[];
  env: Record<string, string>;
  cwd?: string;
}

export interface AgentMcpWebSocketServerConfig extends AgentMcpServerBaseConfig {
  transport: 'websocket';
  url: string;
  headers: Record<string, string>;
  apiKey?: string;
  apiKeyEnv?: string;
}

export type AgentMcpServerConfig = AgentMcpStdioServerConfig | AgentMcpWebSocketServerConfig;

export interface AgentFilesystemStorageConfig {
  type: 'filesystem';
  dir: string;
}

export interface AgentMemoryStorageConfig {
  type: 'memory';
}

export type AgentStorageConfig =
  | AgentMemoryStorageConfig
  | AgentFilesystemStorageConfig;

export interface AgentExternalSourceConfig {
  enabled: boolean;
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  apiKeyHeader: string;
  apiKey?: string;
  headers: Record<string, string>;
}

export interface AgentEventQueueConfig {
  enabled: boolean;
  queueName: string;
  jobName: string;
  removeOnComplete: number;
  removeOnFail: number;
}

export interface AgentEventWebhookConfig {
  enabled: boolean;
  transport: GatewayExternalEventSinkTransport;
  endpoint?: string;
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs: number;
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  requireAck: boolean;
  headers: Record<string, string>;
}

export interface AgentRetryPolicyConfig {
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
  jitterMs: number;
}

export interface AgentRuntimeConfig {
  sessionLockTimeoutMs: number;
  eventWorkerConcurrency: number;
  llmRetry: AgentRetryPolicyConfig;
  toolRetry: AgentRetryPolicyConfig;
}

export interface AgentConfig {
  mcpServers: AgentMcpServerConfig[];
  storage: AgentStorageConfig;
  runtime: AgentRuntimeConfig;
  external?: AgentExternalSourceConfig;
  eventQueue?: AgentEventQueueConfig;
  eventWebhook?: AgentEventWebhookConfig;
}

export type McpServerExposure = 'internal' | 'public';

export interface McpGatewayPrincipalConfig {
  key: string;
  team: string;
  organization?: string;
  allowServers: string[];
  allowTools: string[];
  denyTools: string[];
}

export interface McpGatewayGuardrailsConfig {
  enabled: boolean;
  maxArgumentBytes: number;
  blockedTools: string[];
  blockedArgumentKeys: string[];
  redactArgumentKeys: string[];
}

export interface McpGatewayOAuthConfig {
  enabled: boolean;
  resource?: string;
  issuer?: string;
  authorizationEndpoint?: string;
  tokenEndpoint?: string;
  scopesSupported: string[];
  defaultPrincipalKey?: string;
  authorizationCodeTtlSec?: number;
  accessTokenTtlSec?: number;
  refreshTokenTtlSec?: number;
}

export interface McpGatewayWebSocketAuthConfig {
  allowQueryToken: boolean;
  queryTokenParam: string;
}

export interface McpGatewayWebSocketConfig {
  enabled: boolean;
  endpoint: string;
  auth: McpGatewayWebSocketAuthConfig;
}

export interface McpGatewayConfig {
  enabled: boolean;
  endpoint: string;
  websocket: McpGatewayWebSocketConfig;
  principals: McpGatewayPrincipalConfig[];
  serverExposure: Record<string, McpServerExposure>;
  internalCidrs: string[];
  guardrails: McpGatewayGuardrailsConfig;
  oauth: McpGatewayOAuthConfig;
}

export interface VirtualModelMatchConfig {
  exactAliases: string[];
  prefixes: string[];
  suffixes: string[];
}

export type VirtualModelBaseModelMode =
  | 'request'
  | 'fixed'
  | 'strip_prefix'
  | 'strip_suffix';

export interface VirtualModelBaseModelConfig {
  mode?: VirtualModelBaseModelMode;
  fixedModel?: string;
}

export interface VirtualModelInstructionsConfig {
  prepend?: string;
  append?: string;
  replace?: string;
}

export type VirtualModelToolVisibility = 'internal' | 'client';

export interface VirtualModelToolConfig {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  visibility: VirtualModelToolVisibility;
}

export type VirtualModelExecutionMode = 'decorate_only' | 'tool_loop';

export interface VirtualModelExecutionConfig {
  mode: VirtualModelExecutionMode;
  maxTurns: number;
  maxToolCalls: number;
  clientToolsPolicy: 'allow' | 'deny';
  matchMultimodal?: boolean;
  matchWebSearch?: boolean;
  streamMode: 'buffered' | 'optimistic';
}

export interface VirtualModelMaterializationConfig {
  enabled: boolean;
  includeInGatewayModels: boolean;
  displayNameTemplate?: string;
  descriptionTemplate?: string;
}

export interface VirtualModelProfileConfig {
  id: string;
  key: string;
  displayName: string;
  description?: string;
  enabled: boolean;
  match: VirtualModelMatchConfig;
  baseModel?: VirtualModelBaseModelConfig;
  instructions?: VirtualModelInstructionsConfig;
  tools: VirtualModelToolConfig[];
  toolChoice?: unknown;
  execution: VirtualModelExecutionConfig;
  materialization: VirtualModelMaterializationConfig;
  metadata?: Record<string, unknown>;
}

export interface GatewayConfig {
  host: string;
  port: number;
  providers: ProviderConfig[];
  providerPlugins?: ProviderPluginConfig[];
  virtualModelProfiles?: VirtualModelProfileConfig[];
  providerExternal?: ProviderExternalSourceConfig;
  configExternal?: GatewayConfigExternalSourceConfig;
  defaultTargetProvider?: Provider;
  defaultTargetProviders: Provider[];
  routing?: GatewayRoutingConfig;
  modelList?: GatewayModelListConfig;
  openaiApiKey?: string;
  anthropicApiKey?: string;
  geminiApiKey?: string;
  openaiBaseUrl: string;
  anthropicBaseUrl: string;
  geminiBaseUrl: string;
  geminiApiVersion: string;
  bodyLimitBytes: number;
  upstreamTimeoutMs: number;
  defaultOpenAIModel?: string;
  defaultAnthropicModel?: string;
  defaultGeminiModel?: string;
  auth: GatewayAuthConfig;
  policy: GatewayPolicyConfig;
  precheck: GatewayPrecheckConfig;
  healthAwareRouting: GatewayHealthAwareRoutingConfig;
  providerHealthCheck: ProviderHealthCheckSchedulerConfig;
  metrics: GatewayMetricsConfig;
  cors: GatewayCorsConfig;
  idempotency: GatewayIdempotencyConfig;
  upstreamConcurrency: GatewayUpstreamConcurrencyConfig;
  upstreamCircuitBreaker: GatewayUpstreamCircuitBreakerConfig;
  upstreamRetry: GatewayUpstreamRetryConfig;
  transparentToolExecution: GatewayTransparentToolExecutionConfig;
  billing: BillingConfig;
  billingQueue: BillingQueueConfig;
  billingWebhook: BillingWebhookConfig;
  rawTrace: RawTraceConfig;
  agent: AgentConfig;
  mcpGateway: McpGatewayConfig;
}

export interface GatewaySourceContext {
  adapterKey: string;
  metadata?: Record<string, string>;
}

export interface UpstreamRequest {
  url: string;
  headers: Record<string, string>;
  body: unknown;
}

export type HeaderBag = FastifyRequest['headers'];

export interface StandardResponseMessageContent {
  type: 'output_text';
  text: string;
  annotations: unknown[];
}

export interface StandardResponseMessage {
  id: string;
  type: 'message';
  role: 'assistant';
  status: 'completed';
  content: StandardResponseMessageContent[];
}

export interface StandardResponseFunctionCall {
  id: string;
  type: 'function_call';
  call_id: string;
  name: string;
  namespace?: string;
  arguments: string;
  status: 'completed';
}

export interface StandardResponseReasoningSummary {
  type: 'summary_text';
  text: string;
}

export interface StandardResponseReasoningContent {
  type: 'reasoning_text';
  text: string;
}

export interface StandardResponseReasoning {
  id: string;
  type: 'reasoning';
  status: 'completed';
  summary: StandardResponseReasoningSummary[];
  content?: StandardResponseReasoningContent[];
  encrypted_content?: string;
  reasoning_details?: unknown[];
}

export type StandardResponseOutputItem =
  | StandardResponseMessage
  | StandardResponseFunctionCall
  | StandardResponseReasoning;

export interface StandardUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  cache_duration_seconds?: number;
  cache_ttl_seconds?: number;
  cache_age_seconds?: number;
  server_tool_use?: {
    web_search_requests?: number;
    web_fetch_requests?: number;
  };
}

export interface StandardResponse {
  id: string;
  object: 'response';
  status: 'completed' | 'incomplete';
  model: string;
  output_text: string;
  output: StandardResponseOutputItem[];
  usage: StandardUsage;
  finish_reason?: string;
}

export type StandardRequestInputContent =
  | {
      type: 'input_text';
      text: string;
    }
  | {
      type: 'tool_use';
      id: string;
      name: string;
      input: unknown;
    }
  | {
      type: 'tool_result';
      tool_use_id: string;
      content: string;
      is_error?: boolean;
      result_format?: 'function' | 'web_search';
    }
  | {
      type: 'reasoning';
      text?: string;
      summary?: string;
      encrypted_content?: string;
      reasoning_details?: unknown[];
    };

export interface StandardRequestInputMessage {
  type: 'message';
  role: 'user' | 'assistant';
  content: StandardRequestInputContent[];
}

export interface GatewayRequestClientContext {
  agentId?: string;
  sessionId?: string;
  runId?: string;
  stepId?: string;
  workflow?: string;
  version?: string;
  promptVersion?: string;
  clientRequestId?: string;
  traceparent?: string;
  tracestate?: string;
  metadata?: Record<string, unknown>;
}

export interface GatewayRequestTraceSnapshot {
  headers?: Record<string, string>;
  body?: unknown;
}

export interface GatewayResponseTraceSnapshot {
  statusCode?: number;
  headers?: Record<string, string>;
  body?: unknown;
  outputText?: string;
  finishReason?: string;
}

export interface GatewayBillingTrace {
  request?: GatewayRequestTraceSnapshot;
  response?: GatewayResponseTraceSnapshot;
}

export interface StandardRequest {
  model?: string;
  instructions?: string;
  input: string | StandardRequestInputMessage[];
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: unknown[];
  tool_choice?: unknown;
  reasoning_split?: boolean;
  reasoning?: unknown;
  thinking?: unknown;
  output_config?: unknown;
}

export interface SourceAdapterRequestInput {
  request: FastifyRequest;
  body: Record<string, unknown>;
  source: GatewaySourceContext;
  config: GatewayConfig;
}

export interface SourceAdapterResponseInput {
  request: FastifyRequest;
  response: StandardResponse;
  standardRequest?: StandardRequest;
  source: GatewaySourceContext;
  config: GatewayConfig;
}

export interface SourceAdapter {
  key: string;
  provider: Provider;
  toStandardRequest(input: SourceAdapterRequestInput): Result<StandardRequest>;
  fromStandardResponse(input: SourceAdapterResponseInput): unknown;
  isStreamingRequest(input: SourceAdapterRequestInput): boolean;
  buildPassthroughRequest(input: SourceAdapterRequestInput): Result<UpstreamRequest>;
}

export interface TargetAdapterRequestInput {
  request: FastifyRequest;
  standardRequest: StandardRequest;
  config: GatewayConfig;
  targetProviderConfig?: ProviderConfig;
}

export interface TargetAdapter {
  provider: Provider;
  buildRequestFromStandard(input: TargetAdapterRequestInput): Result<UpstreamRequest>;
  toStandardResponse(payload: unknown): Result<StandardResponse>;
}

export interface ProviderPluginContext {
  request: FastifyRequest;
  config: GatewayConfig;
  source: GatewaySourceContext;
  sourceProvider: Provider;
  sourceAdapterKey: string;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  model?: string;
  passthrough: boolean;
  streaming: boolean;
  forceCodexOauthRefreshOnce?: boolean;
}

export interface ProviderPluginValueRef {
  from: string;
  default?: unknown;
}

export type ProviderPluginValue = unknown;

export interface ProviderPluginMutationConfig {
  strict: boolean;
  headers: Record<string, ProviderPluginValue>;
  query: Record<string, ProviderPluginValue>;
  removeHeaders: string[];
  removeQuery: string[];
  bodySet: Record<string, ProviderPluginValue>;
  bodyMerge: Record<string, ProviderPluginValue>;
  bodyRemove: string[];
}

export interface ProviderPluginResponseMutationConfig {
  strict: boolean;
  bodySet: Record<string, ProviderPluginValue>;
  bodyMerge: Record<string, ProviderPluginValue>;
  bodyRemove: string[];
}

export interface ProviderPluginCodexOAuthConfig {
  enabled: boolean;
  tokenEndpoint: string;
  clientId: string;
  scope: string;
  accessToken?: ProviderPluginValue;
  refreshToken?: ProviderPluginValue;
  accountId?: ProviderPluginValue;
  refreshIfMissingAccessToken: boolean;
  forceRefresh: boolean;
  required: boolean;
  timeoutMs: number;
  authHeader: string;
  authScheme: string;
}

export interface ProviderPluginDeepSeekThinkingConfig {
  enabled: boolean;
}

export interface ProviderPluginConfig {
  key: string;
  enabled: boolean;
  provider?: Provider;
  providerName?: string;
  codexOauth?: ProviderPluginCodexOAuthConfig;
  deepseekThinking?: ProviderPluginDeepSeekThinkingConfig;
  auth?: ProviderPluginMutationConfig;
  request?: ProviderPluginMutationConfig;
  response?: ProviderPluginResponseMutationConfig;
}

export interface ProviderPluginRequestInput extends ProviderPluginContext {
  upstreamRequest: UpstreamRequest;
  standardRequest?: StandardRequest;
}

export interface ProviderPluginResponseInput extends ProviderPluginContext {
  upstreamRequest: UpstreamRequest;
  upstreamResponse: Response;
  upstreamPayload: unknown;
  standardRequest?: StandardRequest;
}

export interface ProviderPlugin {
  key: string;
  provider?: Provider;
  providerName?: string;
  authenticate?(input: ProviderPluginRequestInput): Result<UpstreamRequest> | Promise<Result<UpstreamRequest>>;
  transformRequest?(input: ProviderPluginRequestInput): Result<UpstreamRequest> | Promise<Result<UpstreamRequest>>;
  transformResponse?(input: ProviderPluginResponseInput): Result<unknown> | Promise<Result<unknown>>;
}

export type Result<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

export function ok<T>(value: T): Result<T> {
  return { ok: true, value };
}

export function err<T>(error: string): Result<T> {
  return { ok: false, error };
}
