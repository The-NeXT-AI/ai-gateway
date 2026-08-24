import type {
  GatewayConfig,
  GatewayRequestIdentity,
  Provider,
  ProviderConfig
} from '../types';
import { parseSseChunks } from '../sse';
import { callUpstream, readUpstreamPayload, type UpstreamCallLogContext } from '../upstream/client';
import { asString, isObject } from '../utils';
import type { ProviderRoute } from './provider-router';
import type { AgentModelOutput, AgentModelStreamChunk, AgentRuntimeLogger, AgentToolDefinition } from './types';

export interface ProviderRequest {
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

export interface AgentGatewayTraceContext {
  agentId?: string;
  sessionId?: string;
  runId?: string;
  stepId?: string;
  workflow?: string;
  version?: string;
  promptVersion?: string;
}

export interface ProviderCallResult {
  ok: boolean;
  status: number;
  payload: unknown;
}

export type BuildProviderRequestResult =
  | { ok: true; request: ProviderRequest }
  | { ok: false; error: string };

const agentInternalAuthHeader = 'x-gateway-agent-internal';
const agentInternalAuthHeaderValue = '1';

// ---- Request Building ----

export function buildProviderRequest(
  route: ProviderRoute,
  systemPrompt: string,
  userPrompt: string,
  tools: AgentToolDefinition[],
  modelOverride: string | undefined,
  config: GatewayConfig,
  gatewayBaseUrl: string,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): BuildProviderRequestResult {
  if (route.provider === 'openai') {
    return buildOpenAIRequest(
      route,
      systemPrompt,
      userPrompt,
      tools,
      modelOverride,
      config,
      gatewayBaseUrl,
      requestIdentity,
      traceContext
    );
  }

  if (route.provider === 'anthropic') {
    return buildAnthropicRequest(
      route,
      systemPrompt,
      userPrompt,
      tools,
      modelOverride,
      config,
      gatewayBaseUrl,
      requestIdentity,
      traceContext
    );
  }

  return buildGeminiRequest(
    route,
    systemPrompt,
    userPrompt,
    tools,
    modelOverride,
    config,
    gatewayBaseUrl,
    requestIdentity,
    traceContext
  );
}

function buildOpenAIRequest(
  route: ProviderRoute,
  systemPrompt: string,
  userPrompt: string,
  tools: AgentToolDefinition[],
  modelOverride: string | undefined,
  config: GatewayConfig,
  gatewayBaseUrl: string,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): BuildProviderRequestResult {
  const providerConfig = route.providerConfig;
  const protocol = providerConfig?.type === 'openai_chat_completions' ? 'chat' : 'responses';
  const model = modelOverride || resolveModelForProvider('openai', config, providerConfig);
  if (!model) {
    return { ok: false, error: 'openai model is missing' };
  }

  const apiKey = providerConfig?.apikey || config.openaiApiKey || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return { ok: false, error: 'OPENAI_API_KEY is missing' };
  }

  const scopedHeaders = resolveScopedHeaders(providerConfig, model);
  const scopedBody = resolveScopedBody(providerConfig, model);
  const headers = buildGatewayRequestHeaders(
    route,
    config,
    {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
      ...scopedHeaders
    },
    requestIdentity,
    traceContext
  );

  if (protocol === 'chat') {
    const body: Record<string, unknown> = {
      model,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ],
      tools: tools.length > 0 ? tools.map((tool) => toOpenAIChatTool(tool)) : undefined,
      tool_choice: tools.length > 0 ? 'auto' : undefined,
      ...scopedBody
    };

    return {
      ok: true,
      request: {
        url: `${gatewayBaseUrl}/v1/chat/completions`,
        headers,
        body
      }
    };
  }

  const body: Record<string, unknown> = {
    model,
    instructions: systemPrompt,
    input: userPrompt,
    tools: tools.length > 0 ? tools.map((tool) => toOpenAIResponsesTool(tool)) : undefined,
    tool_choice: tools.length > 0 ? 'auto' : undefined,
    ...scopedBody
  };

  return {
    ok: true,
    request: {
      url: `${gatewayBaseUrl}/v1/responses`,
      headers,
      body
    }
  };
}

function buildAnthropicRequest(
  route: ProviderRoute,
  systemPrompt: string,
  userPrompt: string,
  tools: AgentToolDefinition[],
  modelOverride: string | undefined,
  config: GatewayConfig,
  gatewayBaseUrl: string,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): BuildProviderRequestResult {
  const providerConfig = route.providerConfig;
  const model = modelOverride || resolveModelForProvider('anthropic', config, providerConfig);
  if (!model) {
    return { ok: false, error: 'anthropic model is missing' };
  }

  const apiKey = providerConfig?.apikey || config.anthropicApiKey || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return { ok: false, error: 'ANTHROPIC_API_KEY is missing' };
  }

  const scopedHeaders = resolveScopedHeaders(providerConfig, model);
  const scopedBody = resolveScopedBody(providerConfig, model);
  const headers = buildGatewayRequestHeaders(
    route,
    config,
    {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      ...scopedHeaders
    },
    requestIdentity,
    traceContext
  );

  const body: Record<string, unknown> = {
    model,
    max_tokens: 1024,
    system: systemPrompt,
    messages: [
      {
        role: 'user',
        content: userPrompt
      }
    ],
    tools: tools.length > 0 ? tools.map((tool) => toAnthropicTool(tool)) : undefined,
    tool_choice: tools.length > 0 ? { type: 'auto' } : undefined,
    ...scopedBody
  };

  return {
    ok: true,
    request: {
      url: `${gatewayBaseUrl}/v1/messages`,
      headers,
      body
    }
  };
}

function buildGeminiRequest(
  route: ProviderRoute,
  systemPrompt: string,
  userPrompt: string,
  tools: AgentToolDefinition[],
  modelOverride: string | undefined,
  config: GatewayConfig,
  gatewayBaseUrl: string,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): BuildProviderRequestResult {
  const providerConfig = route.providerConfig;
  const model = modelOverride || resolveModelForProvider('gemini', config, providerConfig);
  if (!model) {
    return { ok: false, error: 'gemini model is missing' };
  }

  const apiKey = providerConfig?.apikey || config.geminiApiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    return { ok: false, error: 'GEMINI_API_KEY is missing' };
  }

  const scopedHeaders = resolveScopedHeaders(providerConfig, model);
  const scopedBody = resolveScopedBody(providerConfig, model);
  const url =
    `${gatewayBaseUrl}/${config.geminiApiVersion}/models/${encodeURIComponent(model)}:generateContent?key=` +
    encodeURIComponent(apiKey);

  const body: Record<string, unknown> = {
    systemInstruction: {
      parts: [{ text: systemPrompt }]
    },
    contents: [
      {
        role: 'user',
        parts: [{ text: userPrompt }]
      }
    ],
    ...scopedBody
  };

  if (tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: tools.map((tool) => toGeminiFunctionDeclaration(tool))
      }
    ];
    body.toolConfig = {
      functionCallingConfig: {
        mode: 'AUTO'
      }
    };
  }

  return {
    ok: true,
    request: {
      url,
      headers: buildGatewayRequestHeaders(
        route,
        config,
        {
          'content-type': 'application/json',
          ...scopedHeaders
        },
        requestIdentity,
        traceContext
      ),
      body
    }
  };
}

// ---- Scoped Config Resolution ----

export function resolveModelForProvider(
  provider: Provider,
  config: GatewayConfig,
  providerConfig?: ProviderConfig
): string | undefined {
  if (provider === 'openai') {
    return config.defaultOpenAIModel || providerConfig?.models[0];
  }

  if (provider === 'anthropic') {
    return config.defaultAnthropicModel || providerConfig?.models[0];
  }

  return config.defaultGeminiModel || providerConfig?.models[0];
}

function resolveScopedHeaders(providerConfig: ProviderConfig | undefined, model: string): Record<string, string> {
  if (!providerConfig) {
    return {};
  }

  const modelHeaders = providerConfig.extraHeaders.byModel[model];
  return {
    ...providerConfig.extraHeaders.default,
    ...(modelHeaders || {})
  };
}

function resolveScopedBody(providerConfig: ProviderConfig | undefined, model: string): Record<string, unknown> {
  if (!providerConfig) {
    return {};
  }

  const modelBody = providerConfig.extraBody.byModel[model];
  return {
    ...providerConfig.extraBody.default,
    ...(modelBody || {})
  };
}

function buildGatewayRequestHeaders(
  route: ProviderRoute,
  config: GatewayConfig,
  headers: Record<string, string>,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): Record<string, string> {
  const internalAuthHeaders = resolveAgentInternalAuthHeaders(config);
  const propagatedIdentityHeaders = resolvePropagatedIdentityHeaders(config, requestIdentity);
  const traceHeaders = resolveAgentTraceHeaders(traceContext);
  return {
    ...headers,
    ...internalAuthHeaders,
    ...propagatedIdentityHeaders,
    ...traceHeaders,
    'x-target-provider': route.providerConfig?.name || route.provider
  };
}

function resolveAgentInternalAuthHeaders(config: GatewayConfig): Record<string, string> {
  const authConfig = config.auth;
  if (!authConfig?.enabled || authConfig.mode !== 'http_introspection') {
    return {};
  }

  const credentialHeader = authConfig.introspection.credentialHeader;
  const credentialEnv = authConfig.introspection.credentialEnv;
  const credentialValueRaw = credentialEnv ? process.env[credentialEnv] : undefined;
  const credentialValue = typeof credentialValueRaw === 'string' ? credentialValueRaw.trim() : '';
  if (!credentialHeader || !credentialValue) {
    return {};
  }

  return {
    [agentInternalAuthHeader]: agentInternalAuthHeaderValue,
    [credentialHeader]: credentialValue
  };
}

function resolvePropagatedIdentityHeaders(
  config: GatewayConfig,
  identity?: GatewayRequestIdentity
): Record<string, string> {
  if (!config.auth?.enabled || !identity?.billingSubjectKey) {
    return {};
  }

  const result: Record<string, string> = {};
  const { identityHeaders } = config.auth;
  assignIdentityHeader(result, identityHeaders.userId, identity.userId);
  assignIdentityHeader(result, identityHeaders.tenantId, identity.tenantId);
  assignIdentityHeader(result, identityHeaders.subject, identity.subject);
  assignIdentityHeader(result, identityHeaders.organizationId, identity.organizationId);
  assignIdentityHeader(result, identityHeaders.plan, identity.plan);
  assignIdentityHeader(result, identityHeaders.apiKeyId, identity.apiKeyId);
  return result;
}

function assignIdentityHeader(
  target: Record<string, string>,
  headerName: string | undefined,
  value: string | undefined
): void {
  const normalizedHeader = typeof headerName === 'string' ? headerName.trim() : '';
  const normalizedValue = typeof value === 'string' ? value.trim() : '';
  if (!normalizedHeader || !normalizedValue) {
    return;
  }

  target[normalizedHeader] = normalizedValue;
}

function resolveAgentTraceHeaders(
  traceContext?: AgentGatewayTraceContext
): Record<string, string> {
  if (!traceContext) {
    return {};
  }

  const headers: Record<string, string> = {};
  assignTraceHeader(headers, 'x-agent-id', traceContext.agentId);
  assignTraceHeader(headers, 'x-agent-session-id', traceContext.sessionId);
  assignTraceHeader(headers, 'x-agent-run-id', traceContext.runId);
  assignTraceHeader(headers, 'x-agent-step-id', traceContext.stepId);
  assignTraceHeader(headers, 'x-agent-workflow', traceContext.workflow);
  assignTraceHeader(headers, 'x-agent-version', traceContext.version);
  assignTraceHeader(headers, 'x-agent-prompt-version', traceContext.promptVersion);
  return headers;
}

function assignTraceHeader(
  target: Record<string, string>,
  headerName: string,
  value: string | undefined
): void {
  const normalizedValue = typeof value === 'string' ? value.trim() : '';
  if (!normalizedValue) {
    return;
  }

  target[headerName] = normalizedValue;
}

// ---- Tool Conversion ----

function toOpenAIChatTool(tool: AgentToolDefinition): Record<string, unknown> {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: ensureJsonSchema(tool.inputSchema)
    }
  };
}

function toOpenAIResponsesTool(tool: AgentToolDefinition): Record<string, unknown> {
  return {
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: ensureJsonSchema(tool.inputSchema)
  };
}

function toAnthropicTool(tool: AgentToolDefinition): Record<string, unknown> {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: ensureJsonSchema(tool.inputSchema)
  };
}

function toGeminiFunctionDeclaration(tool: AgentToolDefinition): Record<string, unknown> {
  return {
    name: tool.name,
    description: tool.description,
    parameters: ensureJsonSchema(tool.inputSchema)
  };
}

function ensureJsonSchema(value: unknown): Record<string, unknown> {
  if (!isObject(value)) {
    return {
      type: 'object',
      properties: {},
      additionalProperties: true
    };
  }

  if (asString(value.type) !== 'object') {
    return {
      type: 'object',
      properties: {},
      additionalProperties: true
    };
  }

  return value;
}

// ---- Response Parsing ----

export function parseProviderOutput(provider: Provider, payload: unknown): AgentModelOutput | undefined {
  if (provider === 'openai') {
    return parseOpenAIOutput(payload);
  }

  if (provider === 'anthropic') {
    return parseAnthropicOutput(payload);
  }

  return parseGeminiOutput(payload);
}

function parseOpenAIOutput(payload: unknown): AgentModelOutput | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const toolDecisionFromChat = parseOpenAIChatToolCall(payload);
  if (toolDecisionFromChat) {
    return toolDecisionFromChat;
  }

  const toolDecisionFromResponses = parseOpenAIResponsesToolCall(payload);
  if (toolDecisionFromResponses) {
    return toolDecisionFromResponses;
  }

  const text =
    asString(payload.output_text) ||
    extractOpenAIChatText(payload) ||
    extractOpenAIResponsesText(payload);
  if (text) {
    return {
      type: 'reply',
      text
    };
  }

  return undefined;
}

function parseOpenAIChatToolCall(payload: Record<string, unknown>): AgentModelOutput | undefined {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const firstChoice = isObject(choices[0]) ? choices[0] : undefined;
  const message = isObject(firstChoice?.message) ? firstChoice.message : undefined;
  const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
  const firstToolCall = isObject(toolCalls[0]) ? toolCalls[0] : undefined;
  const functionPayload = isObject(firstToolCall?.function) ? firstToolCall.function : undefined;
  const toolName = asString(functionPayload?.name);
  if (!toolName) {
    return undefined;
  }

  return {
    type: 'tool_call',
    toolName,
    arguments: parseToolArguments(functionPayload?.arguments),
    reason: 'LLM requested function call.'
  };
}

function parseOpenAIResponsesToolCall(payload: Record<string, unknown>): AgentModelOutput | undefined {
  const output = Array.isArray(payload.output) ? payload.output : [];
  for (const item of output) {
    if (!isObject(item)) {
      continue;
    }

    const type = asString(item.type);
    if (type !== 'function_call') {
      continue;
    }

    const toolName = asString(item.name);
    if (!toolName) {
      continue;
    }

    return {
      type: 'tool_call',
      toolName,
      arguments: parseToolArguments(item.arguments),
      reason: 'LLM requested function call.'
    };
  }

  return undefined;
}

function parseAnthropicOutput(payload: unknown): AgentModelOutput | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const content = Array.isArray(payload.content) ? payload.content : [];
  let textParts: string[] = [];
  for (const block of content) {
    if (!isObject(block)) {
      continue;
    }

    const blockType = asString(block.type);
    if (blockType === 'tool_use') {
      const toolName = asString(block.name);
      if (toolName) {
        return {
          type: 'tool_call',
          toolName,
          arguments: parseToolArguments(block.input),
          reason: 'LLM requested tool_use.'
        };
      }
      continue;
    }

    if (blockType === 'text') {
      const text = asString(block.text);
      if (text) {
        textParts.push(text);
      }
    }
  }

  const text = textParts.join('\n').trim();
  if (!text) {
    return undefined;
  }

  return {
    type: 'reply',
    text
  };
}

function parseGeminiOutput(payload: unknown): AgentModelOutput | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const first = isObject(candidates[0]) ? candidates[0] : undefined;
  const content = isObject(first?.content) ? first.content : undefined;
  const parts = Array.isArray(content?.parts) ? content.parts : [];
  const textParts: string[] = [];

  for (const part of parts) {
    if (!isObject(part)) {
      continue;
    }

    const functionCall = isObject(part.functionCall)
      ? part.functionCall
      : isObject(part.function_call)
      ? part.function_call
      : undefined;
    if (functionCall) {
      const toolName = asString(functionCall.name);
      if (toolName) {
        return {
          type: 'tool_call',
          toolName,
          arguments: parseToolArguments(functionCall.args),
          reason: 'LLM requested functionCall.'
        };
      }
    }

    const text = asString(part.text);
    if (text) {
      textParts.push(text);
    }
  }

  const text = textParts.join('\n').trim();
  if (!text) {
    return undefined;
  }

  return {
    type: 'reply',
    text
  };
}

function extractOpenAIChatText(payload: Record<string, unknown>): string | undefined {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const firstChoice = isObject(choices[0]) ? choices[0] : undefined;
  const message = isObject(firstChoice?.message) ? firstChoice.message : undefined;
  const content = message?.content;

  if (typeof content === 'string') {
    const text = content.trim();
    return text || undefined;
  }

  if (!Array.isArray(content)) {
    return undefined;
  }

  const lines = content
    .map((part) => (isObject(part) ? asString(part.text) || asString(part.output_text) || '' : ''))
    .filter(Boolean);
  const text = lines.join('\n').trim();
  return text || undefined;
}

function extractOpenAIResponsesText(payload: Record<string, unknown>): string | undefined {
  const output = Array.isArray(payload.output) ? payload.output : [];
  const textParts: string[] = [];

  for (const item of output) {
    if (!isObject(item)) {
      continue;
    }

    if (asString(item.type) !== 'message') {
      continue;
    }

    const content = Array.isArray(item.content) ? item.content : [];
    for (const part of content) {
      if (!isObject(part)) {
        continue;
      }

      const text = asString(part.text) || asString(part.output_text);
      if (text) {
        textParts.push(text);
      }
    }
  }

  const text = textParts.join('\n').trim();
  return text || undefined;
}

function parseToolArguments(raw: unknown): Record<string, unknown> {
  if (isObject(raw)) {
    return raw;
  }

  if (typeof raw !== 'string') {
    return {};
  }

  const trimmed = raw.trim();
  if (!trimmed) {
    return {};
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

// ---- Upstream Call ----

export async function callProvider(
  request: ProviderRequest,
  timeoutMs: number,
  signal: AbortSignal | undefined,
  logger: AgentRuntimeLogger | undefined
): Promise<ProviderCallResult> {
  const targetProvider =
    (typeof request.headers['x-target-provider'] === 'string'
      ? request.headers['x-target-provider']
      : undefined) ||
    (typeof request.headers['X-Target-Provider'] === 'string'
      ? request.headers['X-Target-Provider']
      : undefined);
  const response = await callUpstream(request.url, request.headers, request.body, timeoutMs, signal, {
    logger,
    providerName: targetProvider,
    sourceAdapterKey: 'agent_model_client'
  });
  const payload = await readUpstreamPayload(response);
  return {
    ok: response.ok,
    status: response.status,
    payload
  };
}

// ---- Error Summarization ----

export function summarizeUpstreamErrorPayload(payload: unknown): string | undefined {
  if (payload === undefined || payload === null) {
    return undefined;
  }

  if (typeof payload === 'string') {
    const text = payload.trim();
    return text ? truncateText(text, 600) : undefined;
  }

  if (isObject(payload)) {
    const rawText = asString(payload.raw)?.trim();
    const directMessage =
      asString(payload.message) ||
      asString(payload.detail) ||
      asString(payload.error_description) ||
      asString(payload.reason);
    if (directMessage) {
      return truncateText(directMessage, 600);
    }

    const errorField = payload.error;
    if (typeof errorField === 'string') {
      const text = errorField.trim();
      if (text) {
        return truncateText(text, 600);
      }
    }

    if (isObject(errorField)) {
      const attemptsSummary = summarizeUpstreamFallbackAttempts(errorField.attempts);
      const messageWithAttempts =
        (asString(errorField.message) || asString(errorField.detail) || '').trim();
      if (attemptsSummary) {
        const base = messageWithAttempts || 'Upstream provider attempts failed';
        return truncateText(`${base}. Attempts: ${attemptsSummary}`, 600);
      }

      const segments = [
        asString(errorField.code),
        asString(errorField.type),
        asString(errorField.message),
        asString(errorField.detail),
        asString(errorField.reason)
      ].filter((item): item is string => Boolean(item && item.trim()));
      if (segments.length > 0) {
        return truncateText(segments.join(' | '), 600);
      }
    }

    if (rawText) {
      return truncateText(rawText, 600);
    }
  }

  try {
    const serialized = JSON.stringify(payload);
    if (!serialized) {
      return undefined;
    }
    return truncateText(serialized, 600);
  } catch {
    return undefined;
  }
}

function summarizeUpstreamFallbackAttempts(value: unknown): string | undefined {
  if (!Array.isArray(value) || value.length === 0) {
    return undefined;
  }

  const items = value
    .filter((item): item is Record<string, unknown> => isObject(item))
    .slice(0, 3)
    .map((attempt) => {
      const provider = asString(attempt.provider_name) || asString(attempt.provider) || 'unknown';
      const stage = asString(attempt.stage);
      const status = asString(attempt.status);
      const message = asString(attempt.message) || asString(attempt.details) || 'unknown failure';
      const parts = [
        provider,
        stage ? `stage=${stage}` : undefined,
        status ? `status=${status}` : undefined,
        message
      ].filter(Boolean);
      return parts.join(', ');
    })
    .filter(Boolean);

  if (items.length === 0) {
    return undefined;
  }

  const suffix = value.length > 3 ? ` (+${value.length - 3} more)` : '';
  return `${items.join(' || ')}${suffix}`;
}

function buildGeminiStreamUrl(originalUrl: string): string {
  try {
    const url = new URL(originalUrl);
    url.pathname = url.pathname.replace(':generateContent', ':streamGenerateContent');
    url.searchParams.set('alt', 'sse');
    return url.toString();
  } catch {
    return originalUrl
      .replace(':generateContent?', ':streamGenerateContent?alt=sse&')
      .replace(':streamGenerateContent?', ':streamGenerateContent?alt=sse&');
  }
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength)}...`;
}

// ---- Streaming Request Building ----

export function buildStreamingProviderRequest(
  route: ProviderRoute,
  systemPrompt: string,
  userPrompt: string,
  tools: AgentToolDefinition[],
  modelOverride: string | undefined,
  config: GatewayConfig,
  gatewayBaseUrl: string,
  requestIdentity?: GatewayRequestIdentity,
  traceContext?: AgentGatewayTraceContext
): BuildProviderRequestResult {
  const base = buildProviderRequest(
    route,
    systemPrompt,
    userPrompt,
    tools,
    modelOverride,
    config,
    gatewayBaseUrl,
    requestIdentity,
    traceContext
  );
  if (!base.ok) {
    return base;
  }

  if (route.provider === 'gemini') {
    const streamUrl = buildGeminiStreamUrl(base.request.url);
    return {
      ok: true,
      request: {
        ...base.request,
        url: streamUrl
      }
    };
  }

  // OpenAI and Anthropic: add stream: true to body
  return {
    ok: true,
    request: {
      ...base.request,
      body: {
        ...base.request.body,
        stream: true
      }
    }
  };
}

// ---- Streaming Upstream Call ----

export async function callProviderStreaming(
  request: ProviderRequest,
  timeoutMs: number,
  signal: AbortSignal | undefined,
  logger: AgentRuntimeLogger | undefined
): Promise<Response> {
  const targetProvider =
    (typeof request.headers['x-target-provider'] === 'string'
      ? request.headers['x-target-provider']
      : undefined) ||
    (typeof request.headers['X-Target-Provider'] === 'string'
      ? request.headers['X-Target-Provider']
      : undefined);
  return callUpstream(request.url, request.headers, request.body, timeoutMs, signal, {
    logger,
    providerName: targetProvider,
    sourceAdapterKey: 'agent_model_client_streaming'
  });
}

function parseSsePayload(data: string): unknown {
  const trimmed = data.trim();
  if (!trimmed || trimmed === '[DONE]') {
    return undefined;
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return undefined;
  }
}

// ---- Provider-Specific Stream Parsers ----

export async function* parseProviderStreamChunks(
  provider: Provider,
  response: Response
): AsyncGenerator<AgentModelStreamChunk> {
  if (provider === 'openai') {
    yield* parseOpenAIStreamChunks(response);
    return;
  }

  if (provider === 'anthropic') {
    yield* parseAnthropicStreamChunks(response);
    return;
  }

  yield* parseGeminiStreamChunks(response);
}

function parseOpenAIStreamType(body: Record<string, unknown>): 'chat' | 'responses' | 'unknown' {
  // Chat completions have `choices` array, Responses have `output` array or response object fields
  if (Array.isArray(body.choices)) {
    return 'chat';
  }

  if (Array.isArray(body.output) || asString(body.type)) {
    return 'responses';
  }

  return 'unknown';
}

async function* parseOpenAIStreamChunks(response: Response): AsyncGenerator<AgentModelStreamChunk> {
  let accumulatedText = '';
  let detectedFormat: 'chat' | 'responses' | 'unknown' = 'unknown';
  const pendingChatToolCalls = new Map<number, { toolName?: string; argumentsJson: string }>();

  function flushPendingChatToolCall(): AgentModelStreamChunk | undefined {
    const ordered = [...pendingChatToolCalls.entries()].sort((left, right) => left[0] - right[0]);
    for (const [, pending] of ordered) {
      if (!pending.toolName) {
        continue;
      }
      pendingChatToolCalls.clear();
      return {
        type: 'tool_call',
        toolName: pending.toolName,
        arguments: parseToolArguments(pending.argumentsJson),
        reason: 'LLM requested function call.'
      };
    }
    pendingChatToolCalls.clear();
    return undefined;
  }

  for await (const sse of parseSseChunks(response)) {
    const payload = parseSsePayload(sse.data);
    if (!isObject(payload)) {
      continue;
    }

    // Detect format from first meaningful chunk
    if (detectedFormat === 'unknown') {
      detectedFormat = parseOpenAIStreamType(payload as Record<string, unknown>);
    }

    if (detectedFormat === 'chat') {
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const firstChoice = isObject(choices[0]) ? choices[0] : undefined;
      const delta = isObject(firstChoice?.delta) ? firstChoice.delta : undefined;

      const toolCalls = Array.isArray(delta?.tool_calls) ? delta.tool_calls : [];
      for (const rawToolCall of toolCalls) {
        if (!isObject(rawToolCall)) {
          continue;
        }
        const index = typeof rawToolCall.index === 'number' && Number.isFinite(rawToolCall.index)
          ? rawToolCall.index
          : 0;
        const pending = pendingChatToolCalls.get(index) || { argumentsJson: '' };
        const functionPayload = isObject(rawToolCall.function) ? rawToolCall.function : undefined;
        const toolName = asString(functionPayload?.name);
        const argumentsChunk = asString(functionPayload?.arguments);
        if (toolName) {
          pending.toolName = toolName;
        }
        if (argumentsChunk) {
          pending.argumentsJson += argumentsChunk;
        }
        pendingChatToolCalls.set(index, pending);
      }

      const text = asString(delta?.content);
      if (text) {
        accumulatedText += text;
        yield { type: 'text_delta', text };
      }

      const finishReason = asString(firstChoice?.finish_reason);
      if (finishReason === 'tool_calls') {
        const chunk = flushPendingChatToolCall();
        if (chunk) {
          yield chunk;
          return;
        }
      }
      if (finishReason === 'stop' || finishReason === 'length') {
        yield { type: 'done', text: accumulatedText };
        return;
      }
    }

    if (detectedFormat === 'responses') {
      const eventType = asString(payload.type);

      // Response output_text delta
      if (eventType === 'response.output_text.delta') {
        const text = asString(payload.delta);
        if (text) {
          accumulatedText += text;
          yield { type: 'text_delta', text };
        }
        continue;
      }

      if (eventType === 'response.output_text.done') {
        yield { type: 'done', text: asString(payload.text) || accumulatedText };
        return;
      }

      // Function call in responses streaming
      if (eventType === 'response.function_call_arguments.done') {
        const toolName = asString(payload.name);
        if (toolName) {
          yield {
            type: 'tool_call',
            toolName,
            arguments: parseToolArguments(payload.arguments),
            reason: 'LLM requested function call.'
          };
          return;
        }
      }

      // Completed response
      if (eventType === 'response.completed') {
        if (accumulatedText) {
          yield { type: 'done', text: accumulatedText };
        }
        return;
      }
    }
  }

  // Stream ended without explicit done signal
  if (detectedFormat === 'chat') {
    const chunk = flushPendingChatToolCall();
    if (chunk) {
      yield chunk;
      return;
    }
  }
  if (accumulatedText) {
    yield { type: 'done', text: accumulatedText };
  }
}

async function* parseAnthropicStreamChunks(response: Response): AsyncGenerator<AgentModelStreamChunk> {
  let accumulatedText = '';
  let activeToolCall: { toolName: string; inputJsonBuffer: string } | undefined;

  function flushActiveToolCall(): AgentModelStreamChunk | undefined {
    if (!activeToolCall) {
      return undefined;
    }

    const toolName = activeToolCall.toolName;
    const rawJson = activeToolCall.inputJsonBuffer;
    activeToolCall = undefined;

    let args: Record<string, unknown>;
    try {
      const parsed = JSON.parse(rawJson);
      args = isObject(parsed) ? parsed as Record<string, unknown> : {};
    } catch {
      args = {};
    }

    return {
      type: 'tool_call',
      toolName,
      arguments: args,
      reason: 'LLM requested tool_use.'
    };
  }

  for await (const sse of parseSseChunks(response)) {
    const payload = parseSsePayload(sse.data);
    if (!isObject(payload)) {
      continue;
    }

    const eventType = asString(payload.type);

    if (eventType === 'content_block_delta') {
      const delta = isObject(payload.delta) ? payload.delta : undefined;
      const deltaType = asString(delta?.type);

      if (deltaType === 'text_delta') {
        const text = asString(delta?.text);
        if (text) {
          accumulatedText += text;
          yield { type: 'text_delta', text };
        }
      }

      if (deltaType === 'input_json_delta' && activeToolCall) {
        const partialJson = asString(delta?.partial_json);
        if (partialJson) {
          activeToolCall.inputJsonBuffer += partialJson;
        }
      }
      continue;
    }

    if (eventType === 'content_block_start') {
      const contentBlock = isObject(payload.content_block) ? payload.content_block : undefined;
      if (asString(contentBlock?.type) === 'tool_use') {
        const toolName = asString(contentBlock?.name);
        if (toolName) {
          activeToolCall = { toolName, inputJsonBuffer: '' };
        }
      }
      continue;
    }

    if (eventType === 'content_block_stop') {
      const chunk = flushActiveToolCall();
      if (chunk) {
        yield chunk;
        return;
      }
      continue;
    }

    if (eventType === 'message_delta') {
      const delta = isObject(payload.delta) ? payload.delta : undefined;
      const stopReason = asString(delta?.stop_reason);
      if (stopReason === 'end_turn' || stopReason === 'stop' || stopReason === 'max_tokens') {
        const pendingToolCall = flushActiveToolCall();
        if (pendingToolCall) {
          yield pendingToolCall;
          return;
        }
        if (accumulatedText) {
          yield { type: 'done', text: accumulatedText };
        }
        return;
      }
      continue;
    }

    if (eventType === 'message_stop') {
      const pendingToolCall = flushActiveToolCall();
      if (pendingToolCall) {
        yield pendingToolCall;
        return;
      }
      if (accumulatedText) {
        yield { type: 'done', text: accumulatedText };
      }
      return;
    }
  }

  const pendingToolCall = flushActiveToolCall();
  if (pendingToolCall) {
    yield pendingToolCall;
    return;
  }

  if (accumulatedText) {
    yield { type: 'done', text: accumulatedText };
  }
}

async function* parseGeminiStreamChunks(response: Response): AsyncGenerator<AgentModelStreamChunk> {
  let accumulatedText = '';

  for await (const sse of parseSseChunks(response)) {
    const payload = parseSsePayload(sse.data);
    if (!isObject(payload)) {
      continue;
    }

    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const first = isObject(candidates[0]) ? candidates[0] : undefined;
    const content = isObject(first?.content) ? first.content : undefined;
    const parts = Array.isArray(content?.parts) ? content.parts : [];

    for (const part of parts) {
      if (!isObject(part)) {
        continue;
      }

      const functionCall = isObject(part.functionCall)
        ? part.functionCall
        : isObject(part.function_call)
        ? part.function_call
        : undefined;
      if (functionCall) {
        const toolName = asString(functionCall.name);
        if (toolName) {
          yield {
            type: 'tool_call',
            toolName,
            arguments: parseToolArguments(functionCall.args),
            reason: 'LLM requested functionCall.'
          };
          return;
        }
      }

      const text = asString(part.text);
      if (text) {
        accumulatedText += text;
        yield { type: 'text_delta', text };
      }
    }

    // Check finish reason
    const finishReason = asString(first?.finishReason);
    if (finishReason === 'STOP' || finishReason === 'MAX_TOKENS') {
      yield { type: 'done', text: accumulatedText };
      return;
    }
  }

  if (accumulatedText) {
    yield { type: 'done', text: accumulatedText };
  }
}
