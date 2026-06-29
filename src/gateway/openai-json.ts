import type { FastifyReply, FastifyRequest } from 'fastify';
import { randomUUID } from 'node:crypto';
import { buildBillingHeaders, calculateUsageBilling, publishBillingEvent } from '../billing';
import { buildOpenAIHeaders } from '../adapters/builtins/common';
import type {
  BillingRate,
  GatewayConfig,
  Provider,
  ProviderConfig,
  ProviderPlugin,
  StandardRequest,
  StandardUsage,
  UpstreamRequest
} from '../types';
import { callUpstream, cancelResponseBodyOnAbort, readUpstreamPayload } from '../upstream/client';
import {
  asNumber,
  isObject,
  parseProvider,
  providerFromProviderType,
  readHeader
} from '../utils';
import { applyHealthAwareRouting } from './health-routing';
import { createClientDisconnectSignal } from './client-disconnect';
import { evaluateGatewayPolicy, type GatewayPolicyResult } from './policy';
import { recordProviderHealthFailure, recordProviderHealthResponse } from './provider-health';
import { evaluateGatewayPrecheck } from './precheck';
import type { GatewayRuntime } from './runtime';
import {
  checkProviderCircuitBreaker,
  recordProviderCircuitBreakerFailure,
  recordProviderCircuitBreakerResponse
} from './upstream-circuit-breaker';
import { acquireProviderConcurrencySlot } from './upstream-concurrency';

interface TargetProviderRoute {
  provider: Provider;
  providerConfig?: ProviderConfig;
}

interface ParsedModelReference {
  raw: string;
  model: string;
  provider?: Provider;
  providerConfig?: ProviderConfig;
}

interface OpenAIJsonAttemptFailure {
  provider: Provider;
  providerName?: string;
  stage: string;
  message: string;
  status?: number;
  details?: unknown;
}

type ProviderRequestPluginFailureStage = 'provider_auth' | 'provider_request_transform';

type ProviderRequestPluginResult =
  | { ok: true; value: UpstreamRequest }
  | { ok: false; stage: ProviderRequestPluginFailureStage; status: number; message: string };

type ProviderResponsePluginResult =
  | { ok: true; value: unknown }
  | { ok: false; stage: 'provider_response_transform'; status: number; message: string };

interface OpenAIJsonProviderPluginContext {
  request: FastifyRequest;
  config: GatewayConfig;
  endpoint: OpenAIJsonEndpointConfig;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  model?: string;
  clientAbortSignal?: AbortSignal;
  forceCodexOauthRefreshOnce?: boolean;
  standardRequest?: StandardRequest;
  plugins: ProviderPlugin[];
}

type OpenAIJsonUpstreamDispatchResult =
  | { ok: true; upstreamRequest: UpstreamRequest; upstreamResponse: Response }
  | {
      ok: false;
      stage:
        | 'provider_auth'
        | 'provider_request_transform'
        | 'upstream_connect'
        | 'upstream_concurrency'
        | 'upstream_circuit_open';
      status: number;
      message: string;
      details?: unknown;
      upstreamRequest?: UpstreamRequest;
    };

interface OpenAIJsonEndpointConfig {
  endpointPath: string;
  sourceAdapterKey: string;
  displayName: string;
  inputField: string;
  modelRequired: boolean;
  useDefaultOpenAIModel: boolean;
  billingUsageOptional: boolean;
}

const embeddingsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'embeddings',
  sourceAdapterKey: 'openai_embeddings',
  displayName: 'Embeddings',
  inputField: 'input',
  modelRequired: true,
  useDefaultOpenAIModel: true,
  billingUsageOptional: false
};

const moderationsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'moderations',
  sourceAdapterKey: 'openai_moderations',
  displayName: 'Moderations',
  inputField: 'input',
  modelRequired: true,
  useDefaultOpenAIModel: true,
  billingUsageOptional: true
};

const imageGenerationsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'images/generations',
  sourceAdapterKey: 'openai_image_generations',
  displayName: 'Image generations',
  inputField: 'prompt',
  modelRequired: false,
  useDefaultOpenAIModel: false,
  billingUsageOptional: true
};

const hopByHopResponseHeaders = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'content-encoding',
  'content-length',
  'host'
]);

export async function handleOpenAIEmbeddingsRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, embeddingsEndpoint);
}

export async function handleOpenAIModerationsRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, moderationsEndpoint);
}

export async function handleOpenAIImageGenerationsRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, imageGenerationsEndpoint);
}

async function handleOpenAIJsonRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime,
  endpoint: OpenAIJsonEndpointConfig
) {
  const clientAbortSignal = createClientDisconnectSignal(request, reply);
  const body = request.body;
  if (!isObject(body)) {
    return sendBadRequest(reply, 'Request body must be a JSON object.');
  }

  const requestedModel = readHeader(request.headers['x-target-model']) || readBodyModel(body);
  const targetProvidersResult = resolveTargetProviders(request, config, requestedModel);
  if (!targetProvidersResult.ok) {
    return sendBadRequest(reply, targetProvidersResult.error);
  }

  const targetProviders = applyHealthAwareRouting(targetProvidersResult.value, config);
  const attempts: OpenAIJsonAttemptFailure[] = [];
  let precheckApplied = false;

  for (const target of targetProviders) {
    if (clientAbortSignal.aborted) {
      return;
    }

    const targetProvider = target.provider;
    const targetProviderConfig = resolveProviderConfig(config, target);
    if (targetProvider !== 'openai') {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_provider',
        message: `${endpoint.displayName} currently support OpenAI-compatible targets only.`,
        status: 400
      });
      continue;
    }

    const modelResult = resolveTargetModel(request, target, readBodyModel(body), config, endpoint);
    if (!modelResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'model_resolution',
        message: modelResult.error,
        status: 400
      });
      continue;
    }

    const model = modelResult.value;
    if (!model && endpoint.modelRequired) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'model_resolution',
        message: `Model is required. Provide model in body, x-target-model header, or defaultOpenAIModel for ${endpoint.displayName}.`,
        status: 400
      });
      continue;
    }

    const policyResult = evaluateGatewayPolicy({
      request,
      config,
      targetProvider,
      targetProviderConfig,
      model
    });
    if (!policyResult.ok) {
      attempts.push(buildGatewayPolicyAttempt(targetProvider, targetProviderConfig, policyResult));
      continue;
    }

    const standardRequest = buildOpenAIJsonPrecheckStandardRequest(endpoint, model, body);
    if (!precheckApplied) {
      const precheckResult = await evaluateGatewayPrecheck({
        request,
        config,
        targetProvider,
        targetProviderConfig,
        model,
        standardRequest,
        requestBody: body
      });
      if (!precheckResult.ok) {
        return reply.code(precheckResult.statusCode).send({
          error: {
            message: precheckResult.message,
            code: precheckResult.code,
            details: precheckResult.details
          }
        });
      }
      precheckApplied = true;
    }

    const upstreamRequestResult = buildOpenAIJsonUpstreamRequest(
      endpoint,
      request,
      config,
      target,
      model,
      body
    );
    if (!upstreamRequestResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_request_build',
        message: upstreamRequestResult.error,
        status: 400
      });
      continue;
    }

    const pluginContext: OpenAIJsonProviderPluginContext = {
      request,
      config,
      endpoint,
      targetProvider,
      targetProviderConfig,
      model,
      clientAbortSignal,
      standardRequest,
      plugins: runtime.providerPlugins.resolve(targetProvider, targetProviderConfig?.name)
    };
    const dispatchResult = await dispatchOpenAIJsonUpstreamRequest(
      pluginContext,
      upstreamRequestResult.value,
      config.upstreamTimeoutMs
    );
    if (clientAbortSignal.aborted) {
      return;
    }
    if (!dispatchResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: dispatchResult.stage,
        message: dispatchResult.message,
        status: dispatchResult.status,
        details: dispatchResult.details
      });
      continue;
    }

    const { upstreamRequest, upstreamResponse } = dispatchResult;

    const upstreamPayload = await safeReadUpstreamPayload(
      endpoint,
      request,
      targetProvider,
      upstreamResponse,
      clientAbortSignal
    );
    if (clientAbortSignal.aborted) {
      return;
    }
    if (!upstreamResponse.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_response',
        message: 'Upstream request failed.',
        status: upstreamResponse.status,
        details: upstreamPayload
      });
      continue;
    }

    const responsePluginResult = await applyProviderResponsePlugins(
      pluginContext,
      upstreamRequest,
      upstreamResponse,
      upstreamPayload
    );
    if (!responsePluginResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: responsePluginResult.stage,
        message: responsePluginResult.message,
        status: responsePluginResult.status
      });
      continue;
    }

    const responsePayload = responsePluginResult.value;
    attachRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
    attachOpenAIJsonBillingHeaders(
      endpoint,
      request,
      reply,
      config,
      targetProvider,
      model,
      targetProviderConfig,
      responsePayload,
      attempts.length,
      upstreamResponse.status
    );
    return relayUpstreamResponseWithPayload(reply, upstreamResponse, responsePayload);
  }

  if (clientAbortSignal.aborted) {
    return;
  }

  const failure = buildFallbackErrorPayload(targetProviders, attempts);
  return reply.code(failure.status).send(failure.payload);
}

function buildOpenAIJsonPrecheckStandardRequest(
  endpoint: OpenAIJsonEndpointConfig,
  model: string | undefined,
  body: Record<string, unknown>
): StandardRequest {
  return {
    model,
    input: stringifyOpenAIJsonInput(body[endpoint.inputField]),
    max_output_tokens: 0
  };
}

function stringifyOpenAIJsonInput(input: unknown): string {
  if (typeof input === 'string') {
    return input;
  }

  try {
    return JSON.stringify(input ?? '');
  } catch {
    return String(input ?? '');
  }
}

function buildOpenAIJsonUpstreamRequest(
  endpoint: OpenAIJsonEndpointConfig,
  request: FastifyRequest,
  config: GatewayConfig,
  target: TargetProviderRoute,
  model: string | undefined,
  body: Record<string, unknown>
): { ok: true; value: UpstreamRequest } | { ok: false; error: string } {
  const providerConfig = resolveProviderConfig(config, target);
  const headersResult = buildOpenAIHeaders(request.headers, {
    ...config,
    openaiApiKey: providerConfig?.apikey || config.openaiApiKey
  });
  if (!headersResult.ok) {
    return headersResult;
  }

  const extraHeaders = resolveScopedHeaders(providerConfig, model);
  const extraBody = resolveScopedBody(providerConfig, model);
  let url = `${trimRightSlash(config.openaiBaseUrl)}/${endpoint.endpointPath}`;
  let headers = {
    ...headersResult.value,
    ...extraHeaders
  };
  if (target.providerConfig?.baseurl) {
    url = `${trimRightSlash(target.providerConfig.baseurl)}/${endpoint.endpointPath}`;
  }
  if (target.providerConfig?.apikey) {
    headers = {
      ...headers,
      authorization: `Bearer ${target.providerConfig.apikey}`
    };
  }

  return {
    ok: true,
    value: {
      url,
      headers,
      body: {
        ...body,
        ...extraBody,
        ...(model ? { model } : {})
      }
    }
  };
}

async function dispatchOpenAIJsonUpstreamRequest(
  context: OpenAIJsonProviderPluginContext,
  baseUpstreamRequest: UpstreamRequest,
  timeoutMs: number
): Promise<OpenAIJsonUpstreamDispatchResult> {
  const requestPluginResult = await applyProviderRequestPlugins(context, baseUpstreamRequest);
  if (!requestPluginResult.ok) {
    return requestPluginResult;
  }

  const initialUpstreamRequest = requestPluginResult.value;
  const initialUpstreamResponse = await callOpenAIJsonUpstream(context, initialUpstreamRequest, timeoutMs);
  if (!initialUpstreamResponse.ok) {
    return {
      ...initialUpstreamResponse,
      upstreamRequest: initialUpstreamRequest
    };
  }

  if (initialUpstreamResponse.value.status !== 401) {
    return {
      ok: true,
      upstreamRequest: initialUpstreamRequest,
      upstreamResponse: initialUpstreamResponse.value
    };
  }

  context.request.log.info(
    {
      provider: context.targetProvider,
      providerName: context.targetProviderConfig?.name,
      sourceAdapterKey: context.endpoint.sourceAdapterKey,
      status: 401
    },
    `${context.endpoint.displayName} upstream returned 401. Retrying once with forced provider auth refresh.`
  );

  const retryPluginResult = await applyProviderRequestPlugins(
    {
      ...context,
      forceCodexOauthRefreshOnce: true
    },
    baseUpstreamRequest
  );
  if (!retryPluginResult.ok) {
    context.request.log.warn(
      {
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.endpoint.sourceAdapterKey,
        details: retryPluginResult.message
      },
      `Forced ${context.endpoint.displayName.toLowerCase()} provider auth refresh failed after upstream 401. Returning original upstream response.`
    );
    return {
      ok: true,
      upstreamRequest: initialUpstreamRequest,
      upstreamResponse: initialUpstreamResponse.value
    };
  }

  const retryUpstreamRequest = retryPluginResult.value;
  const retryUpstreamResponse = await callOpenAIJsonUpstream(context, retryUpstreamRequest, timeoutMs);
  if (!retryUpstreamResponse.ok) {
    context.request.log.warn(
      {
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.endpoint.sourceAdapterKey,
        details: retryUpstreamResponse.details
      },
      `Retry ${context.endpoint.displayName.toLowerCase()} request failed after upstream 401 and forced provider auth refresh. Returning original upstream response.`
    );
    return {
      ok: true,
      upstreamRequest: initialUpstreamRequest,
      upstreamResponse: initialUpstreamResponse.value
    };
  }

  return {
    ok: true,
    upstreamRequest: retryUpstreamRequest,
    upstreamResponse: retryUpstreamResponse.value
  };
}

async function callOpenAIJsonUpstream(
  context: OpenAIJsonProviderPluginContext,
  upstreamRequest: UpstreamRequest,
  timeoutMs: number
): Promise<
  | { ok: true; value: Response }
  | {
      ok: false;
      stage: 'upstream_connect' | 'upstream_concurrency' | 'upstream_circuit_open';
      status: 502 | 429 | 503;
      message: string;
      details?: unknown;
    }
> {
  const circuit = checkProviderCircuitBreaker(
    context.config,
    context.targetProvider,
    context.targetProviderConfig
  );
  if (!circuit.ok) {
    return {
      ok: false,
      stage: 'upstream_circuit_open',
      status: circuit.status,
      message: circuit.message,
      details: circuit.details
    };
  }

  const slot = await acquireProviderConcurrencySlot(
    context.config,
    context.targetProvider,
    context.targetProviderConfig,
    context.clientAbortSignal
  );
  if (!slot.ok) {
    return {
      ok: false,
      stage: 'upstream_concurrency',
      status: slot.status,
      message: slot.message,
      details: slot.details
    };
  }

  const startedAt = Date.now();
  try {
    const response = await callUpstream(
      upstreamRequest.url,
      upstreamRequest.headers,
      upstreamRequest.body,
      timeoutMs,
      context.clientAbortSignal,
      {
        logger: context.request.log,
        requestId: context.request.id,
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.endpoint.sourceAdapterKey
      },
      context.config.upstreamRetry
    );
    cancelResponseBodyOnAbort(response, context.clientAbortSignal);
    recordProviderHealthResponse(
      context.targetProviderConfig,
      response.status,
      Date.now() - startedAt
    );
    recordProviderCircuitBreakerResponse(
      context.config,
      context.targetProvider,
      context.targetProviderConfig,
      response.status
    );
    return {
      ok: true,
      value: response
    };
  } catch (error) {
    if (!context.clientAbortSignal?.aborted) {
      recordProviderHealthFailure(context.targetProviderConfig, Date.now() - startedAt);
      recordProviderCircuitBreakerFailure(
        context.config,
        context.targetProvider,
        context.targetProviderConfig
      );
    }
    return {
      ok: false,
      stage: 'upstream_connect',
      status: 502,
      message: 'Failed to reach upstream provider.',
      details: error instanceof Error ? error.message : String(error)
    };
  } finally {
    slot.release();
  }
}

function resolveTargetProviders(
  request: FastifyRequest,
  config: GatewayConfig,
  requestModel: string | undefined
): { ok: true; value: TargetProviderRoute[] } | { ok: false; error: string } {
  const modelRefFromHeader = parseModelReference(readHeader(request.headers['x-target-model']), config.providers);
  const modelRefFromBody = parseModelReference(requestModel, config.providers);
  const providerRefFromModel = modelRefFromHeader?.provider
    ? modelRefFromHeader
    : modelRefFromBody?.provider
      ? modelRefFromBody
      : undefined;

  const fromHeaderListRaw = readHeader(request.headers['x-target-providers']);
  if (fromHeaderListRaw !== undefined) {
    const routes = parseProviderRouteList(fromHeaderListRaw, config.providers);
    if (routes.length === 0) {
      return { ok: false, error: 'x-target-providers must include at least one valid provider.' };
    }
    if (providerRefFromModel && !routes.some((route) => routeMatchesModelReference(route, providerRefFromModel))) {
      return { ok: false, error: `Model selector "${providerRefFromModel.raw}" conflicts with x-target-providers.` };
    }
    return { ok: true, value: routes };
  }

  const fromHeaderRaw = readHeader(request.headers['x-target-provider']);
  if (fromHeaderRaw !== undefined) {
    const route = parseProviderRoute(fromHeaderRaw, config.providers);
    if (!route) {
      return { ok: false, error: 'x-target-provider must be a configured provider type or provider name.' };
    }
    if (providerRefFromModel && !routeMatchesModelReference(route, providerRefFromModel)) {
      return { ok: false, error: `Model selector "${providerRefFromModel.raw}" conflicts with x-target-provider.` };
    }
    return { ok: true, value: [route] };
  }

  if (providerRefFromModel?.provider) {
    return { ok: true, value: [routeFromModelReference(providerRefFromModel)] };
  }

  if (config.defaultTargetProviders.length > 0) {
    return { ok: true, value: dedupeProviderRoutes(config.defaultTargetProviders.map((provider) => ({ provider }))) };
  }

  return { ok: true, value: [{ provider: config.defaultTargetProvider || 'openai' }] };
}

function resolveTargetModel(
  request: FastifyRequest,
  target: TargetProviderRoute,
  bodyModel: string | undefined,
  config: GatewayConfig,
  endpoint: OpenAIJsonEndpointConfig
): { ok: true; value: string | undefined } | { ok: false; error: string } {
  const fromHeader = parseModelReference(readHeader(request.headers['x-target-model']), config.providers);
  if (fromHeader) {
    if (fromHeader.provider && !routeMatchesModelReference(target, fromHeader)) {
      return {
        ok: false,
        error: `x-target-model "${fromHeader.raw}" conflicts with target provider ${formatTargetProviderLabel(target)}.`
      };
    }
    return validateModelForTarget(fromHeader.model, target, config);
  }

  const fromBody = parseModelReference(bodyModel, config.providers);
  if (fromBody) {
    if (fromBody.provider && !routeMatchesModelReference(target, fromBody)) {
      return {
        ok: false,
        error: `model "${fromBody.raw}" conflicts with target provider ${formatTargetProviderLabel(target)}.`
      };
    }
    return validateModelForTarget(fromBody.model, target, config);
  }

  return validateModelForTarget(
    endpoint.useDefaultOpenAIModel ? config.defaultOpenAIModel : undefined,
    target,
    config
  );
}

function validateModelForTarget(
  model: string | undefined,
  target: TargetProviderRoute,
  config: GatewayConfig
): { ok: true; value: string | undefined } | { ok: false; error: string } {
  const providerConfig = resolveProviderConfig(config, target);
  if (!model || !providerConfig || providerConfig.models.length === 0) {
    return { ok: true, value: model };
  }

  if (providerConfig.models.includes(model)) {
    return { ok: true, value: model };
  }

  const providerQualifiedModel = resolveProviderQualifiedModelForTarget(model, providerConfig);
  if (providerQualifiedModel && providerConfig.models.includes(providerQualifiedModel)) {
    return { ok: true, value: providerQualifiedModel };
  }

  return {
    ok: false,
    error: `Model "${model}" is not configured for target provider ${formatTargetProviderLabel(target)}. Allowed models: ${providerConfig.models.join(', ')}.`
  };
}

function resolveProviderQualifiedModelForTarget(
  model: string,
  providerConfig: ProviderConfig
): string | undefined {
  const slashIndex = model.indexOf('/');
  if (slashIndex <= 0 || slashIndex >= model.length - 1) {
    return undefined;
  }

  const providerHint = model.slice(0, slashIndex).trim();
  const targetModel = model.slice(slashIndex + 1).trim();
  if (!providerHint || !targetModel || !providerSelectorMatchesTarget(providerHint, providerConfig)) {
    return undefined;
  }

  return targetModel;
}

function providerSelectorMatchesTarget(providerHint: string, providerConfig: ProviderConfig): boolean {
  const normalizedHint = providerHint.trim().toLowerCase();
  if (!normalizedHint) {
    return false;
  }

  return providerConfigSelectorAliases(providerConfig).some((alias) => alias.toLowerCase() === normalizedHint);
}

function providerConfigSelectorAliases(providerConfig: ProviderConfig): string[] {
  const aliases = [providerConfig.name.trim()].filter(Boolean);
  const publicName = providerConfigPublicName(providerConfig);
  if (publicName && !aliases.some((alias) => alias.toLowerCase() === publicName.toLowerCase())) {
    aliases.push(publicName);
  }
  return aliases;
}

function providerConfigPublicName(providerConfig: ProviderConfig): string | undefined {
  const name = providerConfig.name.trim();
  const providerType = providerConfig.type.trim().toLowerCase();
  const segments = name.split('::').map((segment) => segment.trim());
  if (segments.length < 2 || !providerType) {
    return undefined;
  }

  for (let index = segments.length - 1; index > 0; index -= 1) {
    if (segments[index]?.toLowerCase() !== providerType) {
      continue;
    }
    const suffixes = segments.slice(index + 1);
    if (!suffixes.every(isProviderConfigPublicNameSuffix)) {
      continue;
    }
    const publicName = segments.slice(0, index).join('::').trim();
    return publicName || undefined;
  }

  return undefined;
}

function isProviderConfigPublicNameSuffix(segment: string): boolean {
  return segment.trim().toLowerCase().startsWith('cred:');
}

function findProviderConfigBySelectorAlias(
  providerConfigs: ProviderConfig[],
  selector: string
): ProviderConfig | undefined {
  const exactMatch = findProviderConfigByName(providerConfigs, selector);
  if (exactMatch) {
    return exactMatch;
  }

  const normalizedSelector = selector.trim().toLowerCase();
  if (!normalizedSelector) {
    return undefined;
  }

  return providerConfigs.find((providerConfig) =>
    providerConfigSelectorAliases(providerConfig).some(
      (alias) => alias.trim().toLowerCase() === normalizedSelector
    )
  );
}

function parseProviderRouteList(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): TargetProviderRoute[] {
  if (!value) {
    return [];
  }

  return dedupeProviderRoutes(
    value
      .split(',')
      .map((item) => parseProviderRoute(item, providerConfigs))
      .filter((item): item is TargetProviderRoute => Boolean(item))
  );
}

function parseProviderRoute(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): TargetProviderRoute | undefined {
  const normalized = value?.trim();
  if (!normalized) {
    return undefined;
  }

  const byName = findProviderConfigBySelectorAlias(providerConfigs, normalized);
  if (byName) {
    return {
      provider: providerFromProviderType(byName.type),
      providerConfig: byName
    };
  }

  const provider = parseProvider(normalized);
  return provider ? { provider } : undefined;
}

function parseModelReference(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): ParsedModelReference | undefined {
  const raw = value?.trim();
  if (!raw) {
    return undefined;
  }

  const slashIndex = raw.indexOf('/');
  if (slashIndex <= 0 || slashIndex >= raw.length - 1) {
    return { raw, model: raw };
  }

  const providerHint = raw.slice(0, slashIndex).trim();
  const model = raw.slice(slashIndex + 1).trim();
  const providerConfig = findProviderConfigBySelectorAlias(providerConfigs, providerHint);
  if (providerConfig) {
    return {
      raw,
      model,
      provider: providerFromProviderType(providerConfig.type),
      providerConfig
    };
  }

  const provider = parseProvider(providerHint);
  if (provider) {
    return { raw, model, provider };
  }

  return { raw, model: raw };
}

function dedupeProviderRoutes(routes: TargetProviderRoute[]): TargetProviderRoute[] {
  const used = new Set<string>();
  const deduped: TargetProviderRoute[] = [];
  for (const route of routes) {
    const key = route.providerConfig ? `name:${route.providerConfig.name}` : `type:${route.provider}`;
    if (used.has(key)) {
      continue;
    }
    used.add(key);
    deduped.push(route);
  }
  return deduped;
}

function routeMatchesModelReference(route: TargetProviderRoute, reference: ParsedModelReference): boolean {
  if (!reference.provider) {
    return true;
  }

  if (reference.providerConfig) {
    return route.providerConfig?.name === reference.providerConfig.name;
  }

  return route.provider === reference.provider;
}

function routeFromModelReference(reference: ParsedModelReference): TargetProviderRoute {
  return {
    provider: reference.provider as Provider,
    providerConfig: reference.providerConfig
  };
}

async function applyProviderRequestPlugins(
  context: OpenAIJsonProviderPluginContext,
  baseUpstreamRequest: UpstreamRequest
): Promise<ProviderRequestPluginResult> {
  let upstreamRequest = baseUpstreamRequest;
  for (const plugin of context.plugins) {
    if (plugin.authenticate) {
      try {
        const result = await plugin.authenticate({
          request: context.request,
          config: context.config,
          source: { adapterKey: context.endpoint.sourceAdapterKey },
          sourceProvider: 'openai',
          sourceAdapterKey: context.endpoint.sourceAdapterKey,
          targetProvider: context.targetProvider,
          targetProviderConfig: context.targetProviderConfig,
          model: context.model,
          passthrough: true,
          streaming: false,
          forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
          upstreamRequest,
          standardRequest: context.standardRequest
        });
        if (!result.ok) {
          return {
            ok: false,
            stage: 'provider_auth',
            status: 400,
            message: `Provider plugin "${plugin.key}" auth failed: ${result.error}`
          };
        }
        upstreamRequest = result.value;
      } catch (error) {
        return {
          ok: false,
          stage: 'provider_auth',
          status: 400,
          message: `Provider plugin "${plugin.key}" auth failed: ${formatPluginExecutionError(error)}`
        };
      }
    }

    if (plugin.transformRequest) {
      try {
        const result = await plugin.transformRequest({
          request: context.request,
          config: context.config,
          source: { adapterKey: context.endpoint.sourceAdapterKey },
          sourceProvider: 'openai',
          sourceAdapterKey: context.endpoint.sourceAdapterKey,
          targetProvider: context.targetProvider,
          targetProviderConfig: context.targetProviderConfig,
          model: context.model,
          passthrough: true,
          streaming: false,
          forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
          upstreamRequest,
          standardRequest: context.standardRequest
        });
        if (!result.ok) {
          return {
            ok: false,
            stage: 'provider_request_transform',
            status: 400,
            message: `Provider plugin "${plugin.key}" request transform failed: ${result.error}`
          };
        }
        upstreamRequest = result.value;
      } catch (error) {
        return {
          ok: false,
          stage: 'provider_request_transform',
          status: 400,
          message: `Provider plugin "${plugin.key}" request transform failed: ${formatPluginExecutionError(error)}`
        };
      }
    }
  }

  return { ok: true, value: upstreamRequest };
}

async function applyProviderResponsePlugins(
  context: OpenAIJsonProviderPluginContext,
  upstreamRequest: UpstreamRequest,
  upstreamResponse: Response,
  basePayload: unknown
): Promise<ProviderResponsePluginResult> {
  let payload = basePayload;
  for (const plugin of context.plugins) {
    if (!plugin.transformResponse) {
      continue;
    }

    try {
      const result = await plugin.transformResponse({
        request: context.request,
        config: context.config,
        source: { adapterKey: context.endpoint.sourceAdapterKey },
        sourceProvider: 'openai',
        sourceAdapterKey: context.endpoint.sourceAdapterKey,
        targetProvider: context.targetProvider,
        targetProviderConfig: context.targetProviderConfig,
        model: context.model,
        passthrough: true,
        streaming: false,
        forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
        upstreamRequest,
        upstreamResponse,
        upstreamPayload: payload,
        standardRequest: context.standardRequest
      });
      if (!result.ok) {
        return {
          ok: false,
          stage: 'provider_response_transform',
          status: 502,
          message: `Provider plugin "${plugin.key}" response transform failed: ${result.error}`
        };
      }
      payload = result.value;
    } catch (error) {
      return {
        ok: false,
        stage: 'provider_response_transform',
        status: 502,
        message: `Provider plugin "${plugin.key}" response transform failed: ${formatPluginExecutionError(error)}`
      };
    }
  }

  return { ok: true, value: payload };
}

function attachOpenAIJsonBillingHeaders(
  endpoint: OpenAIJsonEndpointConfig,
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  targetProvider: Provider,
  model: string | undefined,
  targetProviderConfig: ProviderConfig | undefined,
  payload: unknown,
  fallbackAttempts: number,
  responseStatusCode: number
): void {
  if (!config.billing.enabled) {
    return;
  }

  const usage = extractOpenAIJsonUsage(payload);
  if (!usage) {
    if (endpoint.billingUsageOptional) {
      return;
    }

    request.log.warn(
      { provider: targetProvider, model },
      `Failed to parse ${endpoint.displayName.toLowerCase()} usage for billing.`
    );
    return;
  }

  const billing = calculateUsageBilling(
    targetProvider,
    usage,
    config.billing,
    resolveProviderBillingRate(config, targetProvider, model, targetProviderConfig)
  );
  for (const [key, value] of Object.entries(buildBillingHeaders(billing))) {
    reply.header(key, value);
  }

  void publishBillingEvent({
    eventId: randomUUID(),
    emittedAt: new Date().toISOString(),
    requestId: request.id,
    route: {
      method: request.method,
      url: request.url
    },
    source: {
      provider: 'openai',
      adapterKey: endpoint.sourceAdapterKey
    },
    target: {
      provider: targetProvider,
      providerName: targetProviderConfig?.name,
      model
    },
    fallback: {
      used: fallbackAttempts > 0,
      attempts: fallbackAttempts
    },
    identity: request.gatewayIdentity,
    outcome: {
      status: 'success',
      statusCode: responseStatusCode
    },
    billing
  }).catch((error) => {
    request.log.warn(
      { details: error instanceof Error ? error.message : String(error) },
      `Failed to publish ${endpoint.displayName.toLowerCase()} billing event.`
    );
  });
}

function extractOpenAIJsonUsage(payload: unknown): StandardUsage | undefined {
  if (!isObject(payload) || !isObject(payload.usage)) {
    return undefined;
  }

  const usageRaw = payload.usage;
  const inputTokens = asNumber(usageRaw.input_tokens) ?? asNumber(usageRaw.prompt_tokens);
  const totalTokens = asNumber(usageRaw.total_tokens) ?? inputTokens;
  if (inputTokens === undefined && totalTokens === undefined) {
    return undefined;
  }

  return {
    input_tokens: inputTokens ?? totalTokens,
    output_tokens: asNumber(usageRaw.output_tokens) ?? 0,
    total_tokens: totalTokens
  };
}

function attachRoutingHeaders(
  reply: FastifyReply,
  provider: Provider,
  providerName: string | undefined,
  fallbackAttempts: number
): void {
  reply.header('x-gateway-target-provider', provider);
  if (providerName) {
    reply.header('x-gateway-target-provider-name', providerName);
  }
  if (fallbackAttempts > 0) {
    reply.header('x-gateway-fallback-used', 'true');
    reply.header('x-gateway-fallback-count', String(fallbackAttempts));
  }
}

function relayUpstreamResponseWithPayload(
  reply: FastifyReply,
  upstreamResponse: Response,
  payload: unknown
) {
  reply.code(upstreamResponse.status);
  upstreamResponse.headers.forEach((value, key) => {
    if (!hopByHopResponseHeaders.has(key.toLowerCase())) {
      reply.header(key, value);
    }
  });
  if (isObject(payload) || Array.isArray(payload)) {
    reply.header('content-type', 'application/json');
  }
  return reply.send(payload);
}

async function safeReadUpstreamPayload(
  endpoint: OpenAIJsonEndpointConfig,
  request: FastifyRequest,
  provider: Provider,
  upstreamResponse: Response,
  clientAbortSignal?: AbortSignal
): Promise<unknown> {
  try {
    return await readUpstreamPayload(upstreamResponse, clientAbortSignal);
  } catch (error) {
    if (clientAbortSignal?.aborted) {
      return { read_error: error instanceof Error ? error.message : String(error) };
    }
    const details = error instanceof Error ? error.message : String(error);
    request.log.warn({ provider, details }, `Failed to parse ${endpoint.displayName.toLowerCase()} upstream payload.`);
    return { read_error: details };
  }
}

function buildGatewayPolicyAttempt(
  provider: Provider,
  providerConfig: ProviderConfig | undefined,
  result: Extract<GatewayPolicyResult, { ok: false }>
): OpenAIJsonAttemptFailure {
  return {
    provider,
    providerName: providerConfig?.name,
    stage: 'gateway_policy',
    message: result.message,
    status: result.statusCode,
    details: {
      code: result.code,
      ...result.details
    }
  };
}

function buildFallbackErrorPayload(
  providers: TargetProviderRoute[],
  attempts: OpenAIJsonAttemptFailure[]
) {
  const last = attempts[attempts.length - 1];
  const status =
    last && typeof last.status === 'number' && last.status >= 100 && last.status <= 599
      ? last.status
      : 502;

  return {
    status,
    payload: {
      error: {
        message: 'All target providers failed.',
        target_providers: providers.map((item) => item.provider),
        target_provider_names: providers
          .map((item) => item.providerConfig?.name)
          .filter(Boolean),
        attempts: attempts.map((attempt) => ({
          provider: attempt.provider,
          provider_name: attempt.providerName,
          stage: attempt.stage,
          message: attempt.message,
          status: attempt.status,
          details: attempt.details
        }))
      }
    }
  };
}

function resolveProviderConfig(
  config: GatewayConfig,
  target: TargetProviderRoute
): ProviderConfig | undefined {
  return target.providerConfig || findProviderConfigByType(config.providers, target.provider);
}

function findProviderConfigByType(
  providers: ProviderConfig[],
  provider: Provider
): ProviderConfig | undefined {
  return providers.find((item) => providerFromProviderType(item.type) === provider);
}

function findProviderConfigByName(
  providers: ProviderConfig[],
  name: string
): ProviderConfig | undefined {
  const normalized = name.trim().toLowerCase();
  return providers.find((item) => item.name.trim().toLowerCase() === normalized);
}

function resolveScopedHeaders(
  providerConfig: ProviderConfig | undefined,
  model: string | undefined
): Record<string, string> {
  if (!providerConfig) {
    return {};
  }

  const modelHeaders = model ? providerConfig.extraHeaders.byModel[model] : undefined;
  return {
    ...providerConfig.extraHeaders.default,
    ...(modelHeaders || {})
  };
}

function resolveScopedBody(
  providerConfig: ProviderConfig | undefined,
  model: string | undefined
): Record<string, unknown> {
  if (!providerConfig) {
    return {};
  }

  const modelBody = model ? providerConfig.extraBody.byModel[model] : undefined;
  return {
    ...providerConfig.extraBody.default,
    ...(modelBody || {})
  };
}

function resolveProviderBillingRate(
  config: GatewayConfig,
  provider: Provider,
  model: string | undefined,
  targetProviderConfig?: ProviderConfig
): BillingRate | undefined {
  const providerConfig = targetProviderConfig || findProviderConfigByType(config.providers, provider);
  return (model ? providerConfig?.billing.byModel[model] : undefined) || providerConfig?.billing.default;
}

function readBodyModel(body: Record<string, unknown>): string | undefined {
  return typeof body.model === 'string' && body.model.trim() ? body.model.trim() : undefined;
}

function formatTargetProviderLabel(route: TargetProviderRoute): string {
  return route.providerConfig?.name || route.provider;
}

function sendBadRequest(reply: FastifyReply, message: string) {
  return reply.code(400).send({ error: { message } });
}

function trimRightSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

function formatPluginExecutionError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
