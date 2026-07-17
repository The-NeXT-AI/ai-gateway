import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { createHash, randomUUID } from 'node:crypto';
import {
  buildBillingHeaders,
  calculateUsageBilling,
  createProviderReportedCostBilling,
  publishBillingEvent,
  resolveVideoPerSecondUsd
} from '../billing';
import { buildOpenAIHeaders } from '../adapters/builtins/common';
import type {
  BillingRate,
  GatewayConfig,
  Provider,
  ProviderConfig,
  ProviderType,
  ProviderPlugin,
  StandardRequest,
  StandardUsage,
  UpstreamRequest
} from '../types';
import {
  callUpstream,
  cancelResponseBody,
  cancelResponseBodyOnAbort,
  readUpstreamPayload,
  relayUpstreamResponse
} from '../upstream/client';
import {
  asNumber,
  findDefaultProviderConfig,
  formatErrorWithCause,
  isObject,
  parseProvider,
  providerFromProviderType,
  readHeader
} from '../utils';
import { applyHealthAwareRouting } from './health-routing';
import { createClientDisconnectSignal } from './client-disconnect';
import { evaluateApiKeyModelRestriction } from './auth';
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
import { resolveGatewayClientIp } from './client-ip';
import {
  parseOpenAIMultipartMetadata,
  readOpenAIMultipartRequestMetadata,
  type OpenAIMultipartMetadata
} from './multipart';
import {
  applyGatewayScheduling,
  attachGatewaySchedulingHeaders,
  recordGatewaySchedulingResponse,
  recordGatewaySchedulingUsage,
  resolveGatewayScheduledCredential,
  setGatewaySchedulingRequestEstimate
} from './scheduler';
import {
  convertVideoCreateBody,
  claimVideoBillingEvent,
  completeVideoBillingEvent,
  decodeGatewayVideoId,
  encodeGatewayVideoId,
  isGatewayVideoId,
  readVideoCreateMetadata,
  releaseVideoBillingEvent,
  validateVideoCreateConversion,
  videoProviderKey,
  videoOwnerKey,
  videoProtocolForTarget,
  type GatewayVideoReference,
  type VideoApiProtocol
} from './video-compat';

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
        | 'gateway_precheck'
        | 'upstream_connect'
        | 'upstream_concurrency'
        | 'upstream_circuit_open';
      status: number;
      code?: string;
      message: string;
      details?: unknown;
      upstreamRequest?: UpstreamRequest;
    };

type OpenAIJsonPreflightResult =
  | { ok: true }
  | {
      ok: false;
      status: number;
      code: string;
      message: string;
      details?: unknown;
    };

interface OpenAIJsonGovernanceRequest {
  body: Record<string, unknown>;
  model?: string;
  standardRequest: StandardRequest;
  imageCount?: number;
  videoSeconds?: number;
  videoSize?: string;
}

interface OpenAIJsonEndpointConfig {
  endpointPath: string;
  sourceAdapterKey: string;
  displayName: string;
  inputField: string;
  modelRequired: boolean;
  useDefaultOpenAIModel: boolean;
  billingUsageOptional: boolean;
  bodyMode?: 'json' | 'json-or-multipart' | 'none';
  method?: 'GET' | 'POST';
  precheck?: boolean;
  targetProviderTypes?: ProviderType[];
  sourceProvider?: Provider;
  binaryResponse?: boolean;
  fixedTarget?: TargetProviderRoute;
  nonIdempotentCreate?: boolean;
  video?: {
    operation: 'create' | 'status';
    sourceProtocol: VideoApiProtocol;
    publicRequestId?: string;
    reference?: GatewayVideoReference;
  };
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
  billingUsageOptional: true,
  nonIdempotentCreate: true,
  targetProviderTypes: ['openai_image_generations']
};

const imageEditsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'images/edits',
  sourceAdapterKey: 'openai_image_generations',
  displayName: 'Image edits',
  inputField: 'prompt',
  modelRequired: false,
  useDefaultOpenAIModel: false,
  billingUsageOptional: true,
  nonIdempotentCreate: true,
  bodyMode: 'json-or-multipart',
  targetProviderTypes: ['openai_image_generations']
};

const openAIVideoGenerationsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'videos',
  sourceAdapterKey: 'openai_video_generations',
  displayName: 'OpenAI video generations',
  inputField: 'prompt',
  modelRequired: false,
  useDefaultOpenAIModel: false,
  billingUsageOptional: true,
  nonIdempotentCreate: true,
  bodyMode: 'json-or-multipart',
  targetProviderTypes: ['openai_video_generations', 'xai_video_generations'],
  video: {
    operation: 'create',
    sourceProtocol: 'openai'
  }
};

const xaiVideoGenerationsEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'videos/generations',
  sourceAdapterKey: 'xai_video_generations',
  displayName: 'xAI video generations',
  inputField: 'prompt',
  modelRequired: false,
  useDefaultOpenAIModel: false,
  billingUsageOptional: true,
  nonIdempotentCreate: true,
  targetProviderTypes: ['openai_video_generations', 'xai_video_generations'],
  sourceProvider: 'xai',
  video: {
    operation: 'create',
    sourceProtocol: 'xai'
  }
};

const videoStatusEndpoint: OpenAIJsonEndpointConfig = {
  endpointPath: 'videos',
  sourceAdapterKey: 'openai_video_generations',
  displayName: 'Video status',
  inputField: 'id',
  modelRequired: false,
  useDefaultOpenAIModel: false,
  billingUsageOptional: true,
  bodyMode: 'none',
  method: 'GET',
  precheck: false,
  targetProviderTypes: ['openai_video_generations', 'xai_video_generations'],
  video: {
    operation: 'status',
    sourceProtocol: 'openai'
  }
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

const binaryDownloadRequestHeaders = [
  'accept',
  'range',
  'if-range',
  'if-none-match',
  'if-modified-since'
] as const;
const maxOpenAIJsonUsageEventTailChars = 256 * 1024;

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

export async function handleOpenAIImageEditsRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, imageEditsEndpoint);
}

export async function handleOpenAIVideoGenerationRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, openAIVideoGenerationsEndpoint);
}

export async function handleXAIVideoGenerationRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  return handleOpenAIJsonRequest(request, reply, config, runtime, xaiVideoGenerationsEndpoint);
}

export async function handleOpenAIVideoStatusRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime,
  requestId: string
) {
  const normalizedRequestId = requestId.trim();
  if (!normalizedRequestId) {
    return sendBadRequest(reply, 'Video request id is required.');
  }
  let reference = decodeGatewayVideoId(normalizedRequestId, videoIdCodecOptions(config));
  if (!reference && isGatewayVideoId(normalizedRequestId)) {
    return sendBadRequest(reply, 'Video request id is invalid, expired, or has a bad signature.');
  }
  if (!reference && config.auth.enabled) {
    return sendForbidden(
      reply,
      'Raw upstream video ids are not accepted when gateway authentication is enabled.'
    );
  }
  if (!isVideoReferenceOwner(reference, request, config.auth.enabled)) {
    return sendForbidden(reply, 'This video request belongs to a different identity.');
  }
  const upstreamRequestId = reference?.upstreamId || normalizedRequestId;
  const referencedProviderConfig = reference
    ? resolveVideoReferenceProviderConfig(reference, config.providers)
    : undefined;
  if ((reference?.targetProviderName || reference?.targetProviderKey) && !referencedProviderConfig) {
    return sendBadRequest(
      reply,
      'Video request target provider is no longer configured.'
    );
  }
  reference = enrichVideoReference(reference, referencedProviderConfig);
  const sourceProtocol =
    reference?.sourceProtocol || inferUnwrappedVideoProtocol(request, config);
  const fixedTarget = reference
    ? { provider: reference.targetProvider, providerConfig: referencedProviderConfig }
    : undefined;
  return handleOpenAIJsonRequest(
    request,
    reply,
    config,
    runtime,
    {
      ...videoStatusEndpoint,
      endpointPath: `videos/${encodeURIComponent(upstreamRequestId)}`,
      sourceAdapterKey:
        sourceProtocol === 'xai'
          ? 'xai_video_generations'
          : 'openai_video_generations',
      sourceProvider: sourceProtocol,
      fixedTarget,
      video: {
        operation: 'status',
        sourceProtocol,
        publicRequestId: normalizedRequestId,
        reference
      }
    }
  );
}

export async function handleOpenAIVideoContentRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  runtime: GatewayRuntime,
  requestId: string
) {
  const normalizedRequestId = requestId.trim();
  if (!normalizedRequestId) {
    return sendBadRequest(reply, 'Video request id is required.');
  }

  let reference = decodeGatewayVideoId(normalizedRequestId, videoIdCodecOptions(config));
  if (!reference && isGatewayVideoId(normalizedRequestId)) {
    return sendBadRequest(reply, 'Video request id is invalid, expired, or has a bad signature.');
  }
  if (!reference && config.auth.enabled) {
    return sendForbidden(
      reply,
      'Raw upstream video ids are not accepted when gateway authentication is enabled.'
    );
  }
  if (!isVideoReferenceOwner(reference, request, config.auth.enabled)) {
    return sendForbidden(reply, 'This video request belongs to a different identity.');
  }
  const referencedProviderConfig = reference
    ? resolveVideoReferenceProviderConfig(reference, config.providers)
    : undefined;
  if ((reference?.targetProviderName || reference?.targetProviderKey) && !referencedProviderConfig) {
    return sendBadRequest(
      reply,
      'Video request target provider is no longer configured.'
    );
  }
  reference = enrichVideoReference(reference, referencedProviderConfig);
  const sourceProtocol =
    reference?.sourceProtocol || inferUnwrappedVideoProtocol(request, config);

  const upstreamRequestId = reference?.upstreamId || normalizedRequestId;
  const fixedTarget = reference
    ? { provider: reference.targetProvider, providerConfig: referencedProviderConfig }
    : undefined;
  const variantResult = readVideoContentVariant(request.url);
  if (!variantResult.ok) {
    return sendBadRequest(reply, variantResult.error);
  }
  const variant = variantResult.value;
  const endpoint: OpenAIJsonEndpointConfig = {
    ...videoStatusEndpoint,
    endpointPath: `videos/${encodeURIComponent(upstreamRequestId)}`,
    displayName: 'Video content',
    sourceAdapterKey:
      sourceProtocol === 'xai'
        ? 'xai_video_generations'
        : 'openai_video_generations',
    fixedTarget,
    video: {
      operation: 'status',
      sourceProtocol,
      publicRequestId: normalizedRequestId,
      reference
    }
  };
  const targetProvidersResult = resolveTargetProviders(
    request,
    config,
    reference?.model,
    endpoint
  );
  if (!targetProvidersResult.ok) {
    return sendBadRequest(reply, targetProvidersResult.error);
  }

  const clientAbortSignal = createClientDisconnectSignal(request, reply);
  const scheduledTargetProviders = applyGatewayScheduling(targetProvidersResult.value, {
    config,
    request,
    requestModel: reference?.model
  });
  const targetProviders = applyHealthAwareRouting(scheduledTargetProviders, config);
  const attempts: OpenAIJsonAttemptFailure[] = [];
  for (const target of targetProviders) {
    const targetProvider = target.provider;
    const targetProviderConfig = resolveProviderConfig(config, target);
    if (!isSupportedEndpointTarget(endpoint, targetProvider, targetProviderConfig)) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_provider',
        message: 'Video content supports OpenAI and xAI video targets only.',
        status: 400
      });
      continue;
    }

    const targetProtocol = videoProtocolForTarget(
      targetProvider,
      targetProviderConfig?.type,
      reference?.targetProtocol || 'openai'
    );
    if (targetProtocol === 'xai' && variant && variant !== 'video') {
      return sendBadRequest(
        reply,
        `xAI video targets do not support the OpenAI content variant "${variant}".`
      );
    }
    const model = reference?.model;
    const modelRestriction = evaluateApiKeyModelRestriction(request, model, {
      provider: targetProvider,
      providerConfig: targetProviderConfig
    });
    if (!modelRestriction.ok) {
      attempts.push(
        buildApiKeyModelRestrictionAttempt(targetProvider, targetProviderConfig, modelRestriction)
      );
      continue;
    }

    const policyResult = evaluateGatewayPolicy({
      request,
      config,
      targetProvider,
      targetProviderConfig,
      model,
      requireKnownModel: true
    });
    if (!policyResult.ok) {
      attempts.push(buildGatewayPolicyAttempt(targetProvider, targetProviderConfig, policyResult));
      continue;
    }

    const targetEndpoint: OpenAIJsonEndpointConfig = {
      ...endpoint,
      endpointPath:
        targetProtocol === 'openai'
          ? `videos/${encodeURIComponent(upstreamRequestId)}/content${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`
          : `videos/${encodeURIComponent(upstreamRequestId)}`,
      sourceProvider: sourceProtocol,
      binaryResponse: targetProtocol === 'openai'
    };
    const upstreamRequestResult = buildOpenAIJsonUpstreamRequest(
      targetEndpoint,
      request,
      config,
      target,
      model,
      undefined,
      targetProtocol
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

    const standardRequest: StandardRequest = {
      model,
      input: upstreamRequestId,
      max_output_tokens: 0
    };
    const pluginContext: OpenAIJsonProviderPluginContext = {
      request,
      config,
      endpoint: targetEndpoint,
      targetProvider,
      targetProviderConfig,
      model,
      clientAbortSignal,
      standardRequest,
      plugins: runtime.providerPlugins.resolve(
        targetProvider,
        targetProviderConfig?.credentialSourceProviderName || targetProviderConfig?.name
      )
    };
    const dispatchResult = await dispatchOpenAIJsonUpstreamRequest(
      pluginContext,
      upstreamRequestResult.value,
      config.upstreamTimeoutMs
    );
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
    if (targetProtocol === 'openai') {
      if (!upstreamResponse.ok && !shouldRelayVideoContentResponse(upstreamResponse.status)) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'upstream_response',
          message: 'Upstream video content request failed.',
          status: upstreamResponse.status,
          details: await safeReadUpstreamPayload(
            targetEndpoint,
            request,
            targetProvider,
            upstreamResponse,
            clientAbortSignal
          )
        });
        continue;
      }
      attachRoutingHeaders(
        reply,
        targetProvider,
        targetProviderConfig?.name,
        attempts.length,
        targetProviderConfig
      );
      return relayUpstreamResponse(reply, upstreamResponse, clientAbortSignal);
    }

    const upstreamPayload = await safeReadUpstreamPayload(
      targetEndpoint,
      request,
      targetProvider,
      upstreamResponse,
      clientAbortSignal
    );
    if (!upstreamResponse.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_response',
        message: 'Upstream video status request failed.',
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

    const transformedPayload = responsePluginResult.value;
    const videoUrl = readXaiVideoUrl(transformedPayload);
    if (!videoUrl) {
      const videoStatus = isObject(transformedPayload)
        ? readRecordString(transformedPayload, 'status')?.toLowerCase()
        : undefined;
      if (videoStatus === 'failed' || videoStatus === 'expired') {
        const statusCode = videoStatus === 'expired' ? 410 : 422;
        const errorMessage =
          videoStatus === 'expired'
            ? 'The xAI video request has expired.'
            : 'The xAI video generation failed.';
        attachRoutingHeaders(
          reply,
          targetProvider,
          targetProviderConfig?.name,
          attempts.length,
          targetProviderConfig
        );
        attachOpenAIJsonBillingHeaders(
          endpoint,
          request,
          reply,
          config,
          targetProvider,
          model,
          targetProviderConfig,
          transformedPayload,
          upstreamPayload,
          upstreamRequest,
          attempts.length,
          upstreamResponse.status,
          {
            outcome: {
              status: 'error',
              statusCode,
              errorMessage
            }
          }
        );
        return reply.code(statusCode).send({
          error: {
            message: errorMessage,
            code: videoStatus === 'expired' ? 'video_expired' : 'video_generation_failed',
            ...(isObject(transformedPayload) && transformedPayload.error !== undefined
              ? { details: transformedPayload.error }
              : {})
          }
        });
      }
      return reply.code(409).send({
        error: {
          message: 'Video content is not available until the xAI generation is complete.',
          code: 'video_not_ready'
        }
      });
    }
    attachRoutingHeaders(
      reply,
      targetProvider,
      targetProviderConfig?.name,
      attempts.length,
      targetProviderConfig
    );
    attachOpenAIJsonBillingHeaders(
      endpoint,
      request,
      reply,
      config,
      targetProvider,
      model,
      targetProviderConfig,
      responsePluginResult.value,
      upstreamPayload,
      upstreamRequest,
      attempts.length,
      upstreamResponse.status
    );
    return reply.redirect(videoUrl);
  }

  const failure = buildFallbackErrorPayload(targetProviders, attempts);
  return reply.code(failure.status).send(failure.payload);
}

export function registerOpenAIMediaBodyParsers(
  fastify: FastifyInstance,
  bodyLimitBytes: number
): void {
  if (fastify.hasContentTypeParser('multipart/form-data')) {
    fastify.removeContentTypeParser('multipart/form-data');
  }
  fastify.addContentTypeParser(
    'multipart/form-data',
    { parseAs: 'buffer', bodyLimit: bodyLimitBytes },
    (_request, body, done) => done(null, body)
  );
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
  const bodyMode = endpoint.bodyMode ?? 'json';
  const multipartBody = Buffer.isBuffer(body) ? body : undefined;
  const jsonBody = !multipartBody && isJsonObject(body) ? body : undefined;
  const contentType = readHeader(request.headers['content-type']);
  const multipartContentType = isMultipartFormDataContentType(contentType);
  if (bodyMode === 'json' && (multipartContentType || !jsonBody)) {
    return sendBadRequest(reply, 'Request body must be a JSON object.');
  }
  if (
    bodyMode === 'json-or-multipart' &&
    ((!multipartContentType && !jsonBody) || (multipartContentType && !multipartBody))
  ) {
    return sendBadRequest(reply, 'Request body must be JSON or multipart/form-data.');
  }

  let multipartMetadata: OpenAIMultipartMetadata | undefined;
  if (multipartBody) {
    const metadataResult =
      readOpenAIMultipartRequestMetadata(request) ||
      parseOpenAIMultipartMetadata(multipartBody, contentType || '');
    if (!metadataResult.ok) {
      return sendBadRequest(reply, metadataResult.error);
    }
    multipartMetadata = metadataResult.value;
  }

  if (
    endpoint.video?.operation === 'create' &&
    config.auth.enabled &&
    !request.gatewayIdentity?.billingSubjectKey
  ) {
    return sendForbidden(
      reply,
      'An authenticated identity is required to create an owned video request.'
    );
  }

  const requestBodyForGovernance = jsonBody ?? multipartMetadata?.fields ?? {};
  const endpointImageCount =
    multipartMetadata?.imageCount ??
    countEndpointImageInputs(endpoint, requestBodyForGovernance);
  if (multipartMetadata || endpointImageCount !== undefined) {
    setGatewaySchedulingRequestEstimate(request, {
      body: requestBodyForGovernance,
      imageCount: endpointImageCount
    });
  }
  const requestBodyModel =
    readBodyModel(requestBodyForGovernance) || endpoint.video?.reference?.model;
  const requestedModel = readHeader(request.headers['x-target-model']) || requestBodyModel;
  const targetProvidersResult = resolveTargetProviders(request, config, requestedModel, endpoint);
  if (!targetProvidersResult.ok) {
    return sendBadRequest(reply, targetProvidersResult.error);
  }

  const scheduledTargetProviders = applyGatewayScheduling(targetProvidersResult.value, {
    config,
    request,
    requestModel: requestBodyModel
  });
  const targetProviders = applyHealthAwareRouting(scheduledTargetProviders, config);
  const attempts: OpenAIJsonAttemptFailure[] = [];
  let precheckApplied = false;

  for (const target of targetProviders) {
    if (clientAbortSignal.aborted) {
      return;
    }

    const targetProvider = target.provider;
    const targetProviderConfig = resolveProviderConfig(config, target);
    const targetVideoProtocol = endpoint.video
      ? videoProtocolForTarget(
          targetProvider,
          targetProviderConfig?.type,
          endpoint.video.sourceProtocol
        )
      : undefined;
    if (!isSupportedEndpointTarget(endpoint, targetProvider, targetProviderConfig)) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_provider',
        message: `${endpoint.displayName} currently support OpenAI-compatible targets only.`,
        status: 400
      });
      continue;
    }
    const videoContentUrlError = validateVideoContentUrlConversion(
      endpoint,
      targetVideoProtocol,
      config
    );
    if (videoContentUrlError) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_request_build',
        message: videoContentUrlError,
        status: 400
      });
      continue;
    }

    const forwardedMultipartModel = requestBodyModel;
    const modelResult = resolveTargetModel(
      request,
      target,
      forwardedMultipartModel,
      config,
      endpoint
    );
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

    if (
      multipartBody &&
      (!endpoint.video || targetVideoProtocol === 'openai') &&
      model !== undefined &&
      forwardedMultipartModel !== model
    ) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'model_resolution',
        message:
          'Multipart model must resolve to the unqualified upstream model and match x-target-model because multipart bytes are forwarded unchanged.',
        status: 400
      });
      continue;
    }

    const apiKeyModelRestriction = evaluateApiKeyModelRestriction(request, model, {
      provider: targetProvider,
      providerConfig: targetProviderConfig
    });
    if (!apiKeyModelRestriction.ok) {
      attempts.push(
        buildApiKeyModelRestrictionAttempt(targetProvider, targetProviderConfig, apiKeyModelRestriction)
      );
      continue;
    }

    const policyResult = evaluateGatewayPolicy({
      request,
      config,
      targetProvider,
      targetProviderConfig,
      model,
      requireKnownModel: !endpoint.modelRequired
    });
    if (!policyResult.ok) {
      attempts.push(buildGatewayPolicyAttempt(targetProvider, targetProviderConfig, policyResult));
      continue;
    }

    const preparedBodyResult = prepareEndpointUpstreamBody(
      endpoint,
      targetVideoProtocol,
      bodyMode === 'none' ? undefined : multipartBody ?? jsonBody ?? {},
      multipartMetadata
    );
    if (!preparedBodyResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_request_build',
        message: preparedBodyResult.error,
        status: 400
      });
      continue;
    }

    const upstreamRequestResult = buildOpenAIJsonUpstreamRequest(
      endpoint,
      request,
      config,
      target,
      model,
      preparedBodyResult.value,
      targetVideoProtocol
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

    const standardRequest = buildOpenAIJsonPrecheckStandardRequest(
      endpoint,
      model,
      requestBodyForGovernance
    );
    const runPreflight =
      endpoint.precheck === false
        ? undefined
        : async (finalUpstreamRequest: UpstreamRequest): Promise<OpenAIJsonPreflightResult> => {
            if (precheckApplied) {
              return { ok: true };
            }
            const governanceResult = buildOpenAIJsonGovernanceRequest(
              endpoint,
              finalUpstreamRequest,
              targetVideoProtocol,
              model
            );
            if (!governanceResult.ok) {
              return {
                ok: false,
                status: 400,
                code: 'invalid_upstream_request',
                message: governanceResult.error
              };
            }
            const governance = governanceResult.value;
            const providerRate = resolveProviderBillingRate(
              config,
              targetProvider,
              governance.model,
              targetProviderConfig
            );
            if (
              endpoint.video?.operation === 'create' &&
              governance.videoSeconds === undefined &&
              (resolveVideoPerSecondUsd(providerRate, governance.videoSize) || 0) > 0 &&
              hasCostBudgetPrecheck(request, config)
            ) {
              return {
                ok: false,
                status: 400,
                code: 'video_duration_required',
                message: 'Video duration must be explicit when cost budget precheck is enabled.'
              };
            }
            const precheckResult = await evaluateGatewayPrecheck({
              request,
              config,
              targetProvider,
              targetProviderConfig,
              model: governance.model,
              standardRequest: governance.standardRequest,
              requestBody: governance.body,
              imageCount: governance.imageCount,
              videoSeconds: governance.videoSeconds,
              videoSize: governance.videoSize
            });
            if (!precheckResult.ok) {
              return {
                ok: false,
                status: precheckResult.statusCode,
                code: precheckResult.code,
                message: precheckResult.message,
                details: precheckResult.details
              };
            }
            precheckApplied = true;
            return { ok: true };
          };

    const pluginContext: OpenAIJsonProviderPluginContext = {
      request,
      config,
      endpoint,
      targetProvider,
      targetProviderConfig,
      model,
      clientAbortSignal,
      standardRequest,
      plugins: runtime.providerPlugins.resolve(
        targetProvider,
        targetProviderConfig?.credentialSourceProviderName || targetProviderConfig?.name
      )
    };
    const dispatchResult = await dispatchOpenAIJsonUpstreamRequest(
      pluginContext,
      upstreamRequestResult.value,
      config.upstreamTimeoutMs,
      runPreflight
    );
    if (clientAbortSignal.aborted) {
      return;
    }
    if (!dispatchResult.ok) {
      if (dispatchResult.stage === 'gateway_precheck') {
        return reply.code(dispatchResult.status).send({
          error: {
            message: dispatchResult.message,
            code: dispatchResult.code,
            details: dispatchResult.details
          }
        });
      }
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: dispatchResult.stage,
        message: dispatchResult.message,
        status: dispatchResult.status,
        details: dispatchResult.details
      });
      if (endpoint.nonIdempotentCreate && dispatchResult.stage === 'upstream_connect') {
        break;
      }
      continue;
    }

    const { upstreamRequest, upstreamResponse } = dispatchResult;

    if (upstreamResponse.ok && isEventStreamResponse(upstreamResponse)) {
      const billingResponse =
        config.billing.enabled || config.scheduling?.enabled
          ? upstreamResponse.clone()
          : undefined;
      attachRoutingHeaders(
        reply,
        targetProvider,
        targetProviderConfig?.name,
        attempts.length,
        targetProviderConfig
      );
      if (billingResponse) {
        void processOpenAIJsonEventStreamUsage({
          endpoint,
          request,
          reply,
          config,
          targetProvider,
          model,
          targetProviderConfig,
          response: billingResponse,
          upstreamRequest,
          fallbackAttempts: attempts.length,
          responseStatusCode: upstreamResponse.status
        });
      }
      return relayUpstreamResponse(reply, upstreamResponse, clientAbortSignal);
    }

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
      if (endpoint.nonIdempotentCreate) {
        attachRoutingHeaders(
          reply,
          targetProvider,
          targetProviderConfig?.name,
          attempts.length,
          targetProviderConfig
        );
        return relayUpstreamResponseWithPayload(reply, upstreamResponse, upstreamPayload);
      }
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
    let providerResponsePayload: unknown;
    if (!responsePluginResult.ok) {
      if (!endpoint.nonIdempotentCreate) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: responsePluginResult.stage,
          message: responsePluginResult.message,
          status: responsePluginResult.status
        });
        continue;
      }
      request.log.warn(
        {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          details: responsePluginResult.message
        },
        `Returning the committed ${endpoint.displayName.toLowerCase()} upstream response after a provider response plugin failed.`
      );
      reply.header(
        'x-gateway-provider-response-transform',
        'bypassed-after-upstream-commit'
      );
      providerResponsePayload = upstreamPayload;
    } else {
      providerResponsePayload = responsePluginResult.value;
    }

    const responsePayload = transformEndpointResponse(
      endpoint,
      request,
      config,
      targetProvider,
      targetProviderConfig,
      targetVideoProtocol,
      model,
      upstreamRequest,
      providerResponsePayload
    );
    attachRoutingHeaders(
      reply,
      targetProvider,
      targetProviderConfig?.name,
      attempts.length,
      targetProviderConfig
    );
    attachOpenAIJsonBillingHeaders(
      endpoint,
      request,
      reply,
      config,
      targetProvider,
      model,
      targetProviderConfig,
      responsePayload,
      upstreamPayload,
      upstreamRequest,
      attempts.length,
      upstreamResponse.status,
      resolveVideoStatusBillingOptions(endpoint, responsePayload)
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

function buildOpenAIJsonGovernanceRequest(
  endpoint: OpenAIJsonEndpointConfig,
  upstreamRequest: UpstreamRequest,
  targetVideoProtocol: VideoApiProtocol | undefined,
  fallbackModel: string | undefined
):
  | { ok: true; value: OpenAIJsonGovernanceRequest }
  | { ok: false; error: string } {
  let body: Record<string, unknown> = {};
  let multipartMetadata: OpenAIMultipartMetadata | undefined;
  if (Buffer.isBuffer(upstreamRequest.body)) {
    const contentType = upstreamRequest.headers['content-type'] || '';
    const metadataResult = parseOpenAIMultipartMetadata(upstreamRequest.body, contentType);
    if (!metadataResult.ok) {
      return { ok: false, error: metadataResult.error };
    }
    multipartMetadata = metadataResult.value;
    body = multipartMetadata.fields;
  } else if (isObject(upstreamRequest.body)) {
    body = upstreamRequest.body;
  }

  const model = readBodyModel(body) || fallbackModel;
  const videoMetadata =
    endpoint.video?.operation === 'create'
      ? readVideoCreateMetadata(
          body,
          targetVideoProtocol || endpoint.video.sourceProtocol
        )
      : undefined;
  return {
    ok: true,
    value: {
      body,
      model,
      standardRequest: buildOpenAIJsonPrecheckStandardRequest(endpoint, model, body),
      imageCount:
        multipartMetadata?.imageCount ??
        countEndpointImageInputs(endpoint, body, targetVideoProtocol),
      videoSeconds: videoMetadata?.duration,
      videoSize: videoMetadata?.size
    }
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

function countVideoCreateImageInputs(
  endpoint: OpenAIJsonEndpointConfig,
  body: Record<string, unknown>,
  protocol: VideoApiProtocol | undefined = endpoint.video?.sourceProtocol
): number | undefined {
  if (endpoint.video?.operation !== 'create') {
    return undefined;
  }
  if (protocol === 'openai') {
    return isObject(body.input_reference) ? 1 : 0;
  }

  const imageCount = isObject(body.image) ? 1 : 0;
  const referenceImageCount = Array.isArray(body.reference_images)
    ? body.reference_images.length
    : 0;
  return imageCount + referenceImageCount;
}

function countEndpointImageInputs(
  endpoint: OpenAIJsonEndpointConfig,
  body: Record<string, unknown>,
  videoProtocol?: VideoApiProtocol
): number | undefined {
  const videoImageCount = countVideoCreateImageInputs(endpoint, body, videoProtocol);
  if (videoImageCount !== undefined) {
    return videoImageCount;
  }
  if (endpoint.endpointPath !== imageEditsEndpoint.endpointPath) {
    return undefined;
  }

  const imageCount =
    countImageEditReferences(body.images) + countImageEditReferences(body.image);
  const maskCount = isImageEditReference(body.mask) ? 1 : 0;
  return imageCount + maskCount;
}

function countImageEditReferences(value: unknown): number {
  if (Array.isArray(value)) {
    return value.reduce(
      (sum, reference) => sum + (isImageEditReference(reference) ? 1 : 0),
      0
    );
  }
  return isImageEditReference(value) ? 1 : 0;
}

function isImageEditReference(value: unknown): boolean {
  if (!isObject(value)) {
    return false;
  }
  return Boolean(
    readRecordString(value, 'image_url') ||
      readRecordString(value, 'file_id') ||
      readRecordString(value, 'url')
  );
}

function isSupportedEndpointTarget(
  endpoint: OpenAIJsonEndpointConfig,
  provider: Provider,
  providerConfig: ProviderConfig | undefined
): boolean {
  if (!endpoint.video) {
    return provider === 'openai';
  }

  return (
    provider === 'openai' ||
    provider === 'xai' ||
    providerConfig?.type === 'openai_video_generations' ||
    providerConfig?.type === 'xai_video_generations'
  );
}

function prepareEndpointUpstreamBody(
  endpoint: OpenAIJsonEndpointConfig,
  targetVideoProtocol: VideoApiProtocol | undefined,
  body: unknown,
  multipartMetadata: OpenAIMultipartMetadata | undefined
): { ok: true; value: unknown } | { ok: false; error: string } {
  if (!endpoint.video || endpoint.video.operation !== 'create' || !targetVideoProtocol) {
    return { ok: true, value: body };
  }

  const sourceProtocol = endpoint.video.sourceProtocol;
  if (sourceProtocol === targetVideoProtocol) {
    return { ok: true, value: body };
  }

  if (Buffer.isBuffer(body)) {
    if (sourceProtocol !== 'openai' || targetVideoProtocol !== 'xai') {
      return { ok: false, error: 'This multipart video protocol conversion is not supported.' };
    }
    if (multipartMetadata?.imageCount) {
      return {
        ok: false,
        error:
          'OpenAI multipart input_reference uploads cannot be converted to xAI. Use a JSON input_reference.image_url instead.'
      };
    }
    const unsupportedParts = [
      ...(multipartMetadata?.unsupportedFields || []),
      ...(multipartMetadata?.fileFields || [])
    ];
    if (unsupportedParts.length > 0) {
      return {
        ok: false,
        error: `OpenAI multipart fields cannot be converted to xAI without data loss: ${unsupportedParts.join(', ')}.`
      };
    }
    const fields = multipartMetadata?.fields || {};
    const conversionError = validateVideoCreateConversion(
      fields,
      sourceProtocol,
      targetVideoProtocol
    );
    if (conversionError) {
      return { ok: false, error: conversionError };
    }
    return {
      ok: true,
      value: convertVideoCreateBody(fields, 'openai', 'xai')
    };
  }

  if (!isJsonObject(body)) {
    return { ok: false, error: 'Video creation body must be a JSON object.' };
  }
  const conversionError = validateVideoCreateConversion(
    body,
    sourceProtocol,
    targetVideoProtocol
  );
  if (conversionError) {
    return { ok: false, error: conversionError };
  }
  return {
    ok: true,
    value: convertVideoCreateBody(body, sourceProtocol, targetVideoProtocol)
  };
}

function resolveEndpointPath(
  endpoint: OpenAIJsonEndpointConfig,
  targetVideoProtocol: VideoApiProtocol | undefined
): string {
  if (endpoint.video?.operation !== 'create') {
    return endpoint.endpointPath;
  }
  return (targetVideoProtocol || endpoint.video.sourceProtocol) === 'xai'
    ? 'videos/generations'
    : 'videos';
}

function transformEndpointResponse(
  endpoint: OpenAIJsonEndpointConfig,
  request: FastifyRequest,
  config: GatewayConfig,
  targetProvider: Provider,
  targetProviderConfig: ProviderConfig | undefined,
  targetVideoProtocol: VideoApiProtocol | undefined,
  model: string | undefined,
  upstreamRequest: UpstreamRequest,
  payload: unknown
): unknown {
  if (!endpoint.video || !targetVideoProtocol || !isObject(payload)) {
    return payload;
  }

  const sourceProtocol = endpoint.video.sourceProtocol;
  if (endpoint.video.operation === 'create') {
    const upstreamId = readVideoResponseId(payload, targetVideoProtocol);
    if (!upstreamId) {
      return payload;
    }
    const governanceResult = buildOpenAIJsonGovernanceRequest(
      endpoint,
      upstreamRequest,
      targetVideoProtocol,
      model
    );
    const governance = governanceResult.ok ? governanceResult.value : undefined;
    const metadata = {
      duration: governance?.videoSeconds,
      size: governance?.videoSize
    };
    const effectiveModel = governance?.model || model || readRecordString(payload, 'model');
    const reference: GatewayVideoReference = {
      version: 2,
      upstreamId,
      sourceProtocol,
      targetProtocol: targetVideoProtocol,
      targetProvider,
      targetProviderName:
        targetProviderConfig?.credentialSourceProviderName || targetProviderConfig?.name,
      targetCredentialId: targetProviderConfig?.credentialId,
      model: effectiveModel,
      duration: metadata.duration,
      size: metadata.size,
      createdAt: Math.floor(Date.now() / 1000),
      ownerKey: videoOwnerKey(request.gatewayIdentity?.billingSubjectKey)
    };
    const publicId = encodeGatewayVideoId(reference, videoIdCodecOptions(config));
    if (sourceProtocol === 'xai') {
      return sourceProtocol === targetVideoProtocol
        ? { ...payload, request_id: publicId }
        : { request_id: publicId };
    }
    if (sourceProtocol === targetVideoProtocol) {
      return { ...payload, id: publicId };
    }
    return compactObject({
      id: publicId,
      object: 'video',
      created_at: reference.createdAt,
      status: 'queued',
      progress: 0,
      model: effectiveModel,
      seconds: metadata.duration === undefined ? undefined : String(metadata.duration),
      size: metadata.size
    });
  }

  const reference = endpoint.video.reference;
  const publicId = endpoint.video.publicRequestId;
  if (!reference || !publicId) {
    return payload;
  }

  if (sourceProtocol === targetVideoProtocol) {
    return sourceProtocol === 'openai' ? { ...payload, id: publicId } : payload;
  }

  if (sourceProtocol === 'openai') {
    return convertXAIStatusResponseToOpenAI(payload, reference, publicId);
  }
  return convertOpenAIStatusResponseToXAI(payload, reference, publicId, config);
}

function convertXAIStatusResponseToOpenAI(
  payload: Record<string, unknown>,
  reference: GatewayVideoReference,
  publicId: string
): Record<string, unknown> {
  const video = isObject(payload.video) ? payload.video : undefined;
  const rawStatus = readRecordString(payload, 'status')?.toLowerCase();
  const status =
    rawStatus === 'done'
      ? 'completed'
      : rawStatus === 'failed' || rawStatus === 'expired'
        ? 'failed'
        : 'in_progress';
  return compactObject({
    id: publicId,
    object: 'video',
    created_at: reference.createdAt,
    status,
    progress: asNumber(payload.progress) ?? (status === 'completed' ? 100 : 0),
    model: readRecordString(payload, 'model') || reference.model,
    seconds:
      asNumber(video?.duration) === undefined
        ? reference.duration === undefined
          ? undefined
          : String(reference.duration)
        : String(asNumber(video?.duration)),
    size: reference.size,
    error:
      status === 'failed'
        ? payload.error || { message: rawStatus === 'expired' ? 'Video request expired.' : 'Video generation failed.' }
        : undefined
  });
}

function convertOpenAIStatusResponseToXAI(
  payload: Record<string, unknown>,
  reference: GatewayVideoReference,
  publicId: string,
  config: GatewayConfig
): Record<string, unknown> {
  const rawStatus = readRecordString(payload, 'status')?.toLowerCase();
  const status =
    rawStatus === 'completed'
      ? 'done'
      : rawStatus === 'failed' || rawStatus === 'cancelled' || rawStatus === 'expired'
        ? 'failed'
        : 'pending';
  const duration = readFlexibleNumber(payload.seconds) ?? reference.duration;
  return compactObject({
    status,
    model: readRecordString(payload, 'model') || reference.model,
    progress: asNumber(payload.progress) ?? (status === 'done' ? 100 : 0),
    video:
      status === 'done'
        ? compactObject({
            url: buildGatewayVideoContentUrl(config, publicId),
            duration
          })
        : undefined,
    error: status === 'failed' ? payload.error : undefined
  });
}

function readVideoResponseId(
  payload: Record<string, unknown>,
  protocol: VideoApiProtocol
): string | undefined {
  return readRecordString(payload, protocol === 'xai' ? 'request_id' : 'id');
}

function readRecordString(
  value: Record<string, unknown> | undefined,
  key: string
): string | undefined {
  const item = value?.[key];
  return typeof item === 'string' && item.trim() ? item.trim() : undefined;
}

function readFlexibleNumber(value: unknown): number | undefined {
  const numeric =
    asNumber(value) ??
    (typeof value === 'string' && value.trim() ? Number(value.trim()) : undefined);
  return numeric !== undefined && Number.isFinite(numeric) ? numeric : undefined;
}

function compactObject(value: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined));
}

function buildGatewayVideoContentUrl(config: GatewayConfig, publicId: string): string {
  const path = `/v1/videos/${encodeURIComponent(publicId)}/content`;
  const publicBaseUrl = resolveMediaPublicBaseUrl(config);
  if (!publicBaseUrl) {
    throw new Error(
      'media.publicBaseUrl must be an absolute HTTP(S) URL without query parameters or fragments for cross-protocol video content URLs.'
    );
  }
  return `${publicBaseUrl}${path}`;
}

function validateVideoContentUrlConversion(
  endpoint: OpenAIJsonEndpointConfig,
  targetVideoProtocol: VideoApiProtocol | undefined,
  config: GatewayConfig
): string | undefined {
  if (
    endpoint.video?.sourceProtocol !== 'xai' ||
    targetVideoProtocol !== 'openai' ||
    resolveMediaPublicBaseUrl(config)
  ) {
    return undefined;
  }
  return 'media.publicBaseUrl must be an absolute HTTP(S) URL without query parameters or fragments when routing the xAI video API to an OpenAI video provider.';
}

function resolveMediaPublicBaseUrl(config: GatewayConfig): string | undefined {
  const value = config.media?.publicBaseUrl?.trim();
  if (!value) {
    return undefined;
  }
  try {
    const url = new URL(value);
    return (url.protocol === 'http:' || url.protocol === 'https:') &&
      !url.search &&
      !url.hash &&
      !/[?#]/.test(value)
      ? trimRightSlash(url.toString())
      : undefined;
  } catch {
    return undefined;
  }
}

function readVideoContentVariant(
  requestUrl: string
): { ok: true; value?: 'video' | 'thumbnail' | 'spritesheet' } | { ok: false; error: string } {
  const variants = new URL(requestUrl, 'http://gateway.local').searchParams.getAll('variant');
  if (variants.length === 0) {
    return { ok: true };
  }
  if (variants.length !== 1) {
    return { ok: false, error: 'Video content variant must be provided at most once.' };
  }
  const variant = variants[0]?.trim();
  return variant === 'video' || variant === 'thumbnail' || variant === 'spritesheet'
    ? { ok: true, value: variant }
    : {
        ok: false,
        error: 'Video content variant must be video, thumbnail, or spritesheet.'
      };
}

function readXaiVideoUrl(payload: unknown): string | undefined {
  if (!isObject(payload) || !isObject(payload.video)) {
    return undefined;
  }
  const value = readRecordString(payload.video, 'url');
  if (!value) {
    return undefined;
  }
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.toString() : undefined;
  } catch {
    return undefined;
  }
}

function buildOpenAIJsonUpstreamRequest(
  endpoint: OpenAIJsonEndpointConfig,
  request: FastifyRequest,
  config: GatewayConfig,
  target: TargetProviderRoute,
  model: string | undefined,
  body: unknown,
  targetVideoProtocol?: VideoApiProtocol
): { ok: true; value: UpstreamRequest } | { ok: false; error: string } {
  const providerConfig = resolveProviderConfig(config, target);
  const isXaiTarget = targetVideoProtocol === 'xai';
  const xaiApiKey = isXaiTarget
    ? providerConfig?.apikey || process.env.XAI_API_KEY
    : undefined;
  const gatewayTokenAuthenticatesRequest =
    config.auth.enabled &&
    (config.auth.mode === 'http_introspection' || config.auth.mode === 'static_api_key');
  if (isXaiTarget && !xaiApiKey && gatewayTokenAuthenticatesRequest) {
    return { ok: false, error: 'XAI_API_KEY is missing.' };
  }
  const managedApiKey = isXaiTarget
    ? xaiApiKey
    : providerConfig?.apikey || config.openaiApiKey;
  const headersResult = buildOpenAIHeaders(request.headers, {
    ...config,
    openaiApiKey: managedApiKey,
    allowEnvApiKeyFallback: !isXaiTarget
  });
  if (!headersResult.ok) {
    return isXaiTarget
      ? { ok: false, error: 'XAI_API_KEY is missing.' }
      : headersResult;
  }

  const extraHeaders = resolveScopedHeaders(providerConfig, model);
  const extraBody = resolveScopedBody(providerConfig, model);
  const endpointPath = resolveEndpointPath(endpoint, targetVideoProtocol);
  const baseUrl =
    providerConfig?.baseurl ||
    (isXaiTarget
      ? process.env.XAI_BASE_URL || 'https://api.x.ai/v1'
      : config.openaiBaseUrl);
  const url = `${trimRightSlash(baseUrl)}/${endpointPath}`;
  const protocolHeaders = { ...headersResult.value };
  if (isXaiTarget) {
    delete protocolHeaders['openai-organization'];
    delete protocolHeaders['openai-project'];
  }
  let headers: Record<string, string> = {
    ...protocolHeaders,
    ...extraHeaders
  };
  if (providerConfig?.apikey || xaiApiKey) {
    headers = {
      ...headers,
      authorization: `Bearer ${providerConfig?.apikey || xaiApiKey}`
    };
  }

  const isMultipart = Buffer.isBuffer(body);
  if (isMultipart) {
    const contentType = readHeader(request.headers['content-type']);
    if (!contentType?.toLowerCase().startsWith('multipart/form-data')) {
      return { ok: false, error: 'Multipart media requests require a multipart/form-data content type.' };
    }
    headers['content-type'] = contentType;
  }
  if ((endpoint.bodyMode ?? 'json') === 'none') {
    delete headers['content-type'];
  }
  if (endpoint.binaryResponse) {
    for (const headerName of binaryDownloadRequestHeaders) {
      const value = readHeader(request.headers[headerName]);
      if (value) {
        headers[headerName] = value;
      }
    }
  }

  const jsonBody = isObject(body) ? body : undefined;

  return {
    ok: true,
    value: {
      method: endpoint.method ?? 'POST',
      url,
      headers,
      body: isMultipart
        ? body
        : (endpoint.bodyMode ?? 'json') === 'none'
          ? undefined
          : {
              ...(jsonBody ?? {}),
              ...extraBody,
              ...(model ? { model } : {})
            },
      bodyEncoding: isMultipart
        ? 'bytes'
        : (endpoint.bodyMode ?? 'json') === 'none'
          ? 'none'
          : 'json',
      skipResponseBodyLog: endpoint.binaryResponse
    }
  };
}

async function dispatchOpenAIJsonUpstreamRequest(
  context: OpenAIJsonProviderPluginContext,
  baseUpstreamRequest: UpstreamRequest,
  timeoutMs: number,
  runPreflight?: (upstreamRequest: UpstreamRequest) => Promise<OpenAIJsonPreflightResult>
): Promise<OpenAIJsonUpstreamDispatchResult> {
  const requestPluginResult = await applyProviderRequestPlugins(context, baseUpstreamRequest);
  if (!requestPluginResult.ok) {
    return requestPluginResult;
  }

  const initialUpstreamRequest = requestPluginResult.value;
  const initialUpstreamResponse = await callOpenAIJsonUpstream(
    context,
    initialUpstreamRequest,
    timeoutMs,
    runPreflight
  );
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
  const retryUpstreamResponse = await callOpenAIJsonUpstream(
    context,
    retryUpstreamRequest,
    timeoutMs,
    runPreflight
  );
  if (!retryUpstreamResponse.ok) {
    context.request.log.warn(
      {
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.endpoint.sourceAdapterKey,
        details: retryUpstreamResponse.details
      },
      `Retry ${context.endpoint.displayName.toLowerCase()} request failed after upstream 401 and forced provider auth refresh.`
    );
    if (context.endpoint.nonIdempotentCreate) {
      await cancelResponseBody(initialUpstreamResponse.value);
      return {
        ...retryUpstreamResponse,
        upstreamRequest: retryUpstreamRequest
      };
    }
    return {
      ok: true,
      upstreamRequest: initialUpstreamRequest,
      upstreamResponse: initialUpstreamResponse.value
    };
  }

  await cancelResponseBody(initialUpstreamResponse.value);
  return {
    ok: true,
    upstreamRequest: retryUpstreamRequest,
    upstreamResponse: retryUpstreamResponse.value
  };
}

async function callOpenAIJsonUpstream(
  context: OpenAIJsonProviderPluginContext,
  upstreamRequest: UpstreamRequest,
  timeoutMs: number,
  runPreflight?: (upstreamRequest: UpstreamRequest) => Promise<OpenAIJsonPreflightResult>
): Promise<
  | { ok: true; value: Response }
  | {
      ok: false;
      stage:
        | 'gateway_precheck'
        | 'upstream_connect'
        | 'upstream_concurrency'
        | 'upstream_circuit_open';
      status: number;
      code?: string;
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
    if (runPreflight) {
      const preflightResult = await runPreflight(upstreamRequest);
      if (!preflightResult.ok) {
        return {
          ok: false,
          stage: 'gateway_precheck',
          status: preflightResult.status,
          code: preflightResult.code,
          message: preflightResult.message,
          details: preflightResult.details
        };
      }
    }
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
      context.endpoint.nonIdempotentCreate
        ? { ...context.config.upstreamRetry, enabled: false, maxAttempts: 1 }
        : context.config.upstreamRetry,
      {
        method: upstreamRequest.method,
        bodyEncoding: upstreamRequest.bodyEncoding,
        skipResponseBodyLog: upstreamRequest.skipResponseBodyLog
      }
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
    recordGatewaySchedulingResponse({
      config: context.config,
      request: context.request,
      providerConfig: context.targetProviderConfig,
      model: context.model,
      statusCode: response.status
    });
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
      recordGatewaySchedulingResponse({
        config: context.config,
        request: context.request,
        providerConfig: context.targetProviderConfig,
        model: context.model,
        error: true
      });
      context.request.log.warn(
        {
          requestId: context.request.id,
          provider: context.targetProvider,
          providerName: context.targetProviderConfig?.name,
          details: formatErrorWithCause(error)
        },
        'Failed to reach upstream provider.'
      );
    }
    return {
      ok: false,
      stage: 'upstream_connect',
      status: 502,
      message: 'Failed to reach upstream provider.'
    };
  } finally {
    slot.release();
  }
}

function resolveTargetProviders(
  request: FastifyRequest,
  config: GatewayConfig,
  requestModel: string | undefined,
  endpoint: OpenAIJsonEndpointConfig
): { ok: true; value: TargetProviderRoute[] } | { ok: false; error: string } {
  if (endpoint.fixedTarget) {
    return { ok: true, value: [endpoint.fixedTarget] };
  }

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
    return {
      ok: true,
      value: expandEndpointProviderRoutes(routes, config.providers, endpoint, requestModel)
    };
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
    return {
      ok: true,
      value: expandEndpointProviderRoutes([route], config.providers, endpoint, requestModel)
    };
  }

  if (providerRefFromModel?.provider) {
    return {
      ok: true,
      value: expandEndpointProviderRoutes(
        [routeFromModelReference(providerRefFromModel)],
        config.providers,
        endpoint,
        requestModel
      )
    };
  }

  const configuredEndpointRoutes = resolveConfiguredEndpointRoutes(
    config.providers,
    endpoint,
    requestModel
  );
  if (configuredEndpointRoutes.length > 0) {
    return {
      ok: true,
      value: prioritizeConfiguredEndpointRoutes(
        configuredEndpointRoutes,
        config.defaultTargetProviders
      )
    };
  }
  if (config.defaultTargetProviders.length > 0) {
    const routes = dedupeProviderRoutes(
      config.defaultTargetProviders.map((provider) => ({ provider }))
    );
    const expandedRoutes = expandEndpointProviderRoutes(
      routes,
      config.providers,
      endpoint,
      requestModel
    );
    return {
      ok: true,
      value: expandedRoutes
    };
  }

  return {
    ok: true,
    value: expandEndpointProviderRoutes(
      [{ provider: config.defaultTargetProvider || 'openai' }],
      config.providers,
      endpoint,
      requestModel
    )
  };
}

function prioritizeConfiguredEndpointRoutes(
  routes: TargetProviderRoute[],
  defaultProviders: Provider[]
): TargetProviderRoute[] {
  const providerPriority = new Map(
    defaultProviders.map((provider, index) => [provider, index] as const)
  );
  return routes
    .map((route, index) => ({ route, index }))
    .sort((left, right) => {
      const leftPriority = providerPriority.get(left.route.provider) ?? Number.MAX_SAFE_INTEGER;
      const rightPriority = providerPriority.get(right.route.provider) ?? Number.MAX_SAFE_INTEGER;
      return leftPriority - rightPriority || left.index - right.index;
    })
    .map(({ route }) => route);
}

function resolveConfiguredEndpointRoutes(
  providerConfigs: ProviderConfig[],
  endpoint: OpenAIJsonEndpointConfig,
  requestModel: string | undefined
): TargetProviderRoute[] {
  if (!endpoint.targetProviderTypes?.length) {
    return [];
  }

  const preferredTypes = new Set(endpoint.targetProviderTypes);
  const preferredConfigs = providerConfigs.filter((providerConfig) =>
    preferredTypes.has(providerConfig.type)
  );
  const model = parseModelReference(requestModel, providerConfigs)?.model;
  const candidates = model
    ? preferredConfigs.filter(
        (providerConfig) =>
          providerConfig.models.length === 0 || providerConfig.models.includes(model)
      )
    : preferredConfigs;
  return candidates.map((providerConfig) => ({
    provider: providerFromProviderType(providerConfig.type),
    providerConfig
  }));
}

function expandEndpointProviderRoutes(
  routes: TargetProviderRoute[],
  providerConfigs: ProviderConfig[],
  endpoint: OpenAIJsonEndpointConfig,
  requestModel: string | undefined
): TargetProviderRoute[] {
  if (!endpoint.targetProviderTypes?.length) {
    return routes;
  }

  const parsedModel = parseModelReference(requestModel, providerConfigs);
  const model = parsedModel?.model;
  const expanded = routes.flatMap((route) => {
    if (route.providerConfig) {
      return [route];
    }

    const candidates = resolveEndpointProviderConfigs(
      providerConfigs,
      endpoint,
      model,
      route.provider
    );
    return candidates.length > 0
      ? candidates.map((providerConfig) => ({
          provider: providerFromProviderType(providerConfig.type),
          providerConfig
        }))
      : [route];
  });
  return dedupeProviderRoutes(expanded);
}

function resolveEndpointProviderConfigs(
  providerConfigs: ProviderConfig[],
  endpoint: OpenAIJsonEndpointConfig,
  model: string | undefined,
  provider: Provider
): ProviderConfig[] {
  const providerFamilyConfigs = providerConfigs.filter(
    (providerConfig) => providerFromProviderType(providerConfig.type) === provider
  );
  const preferredTypes = new Set(endpoint.targetProviderTypes || []);
  const preferredConfigs = providerFamilyConfigs.filter((providerConfig) =>
    preferredTypes.has(providerConfig.type)
  );
  if (endpoint.video) {
    if (!model) {
      if (preferredConfigs.length > 0) {
        return preferredConfigs;
      }
      return providerFamilyConfigs;
    }

    const preferredFamilyMatching = preferredConfigs.filter(
      (providerConfig) =>
        providerConfig.models.length === 0 || providerConfig.models.includes(model)
    );
    if (preferredFamilyMatching.length > 0) {
      return preferredFamilyMatching;
    }
  }
  if (!model) {
    return preferredConfigs.length > 0 ? preferredConfigs : providerFamilyConfigs;
  }

  const preferredMatching = preferredConfigs.filter(
    (providerConfig) =>
      providerConfig.models.length === 0 || providerConfig.models.includes(model)
  );
  if (preferredMatching.length > 0) {
    return preferredMatching;
  }

  const matching = providerFamilyConfigs.filter(
    (providerConfig) =>
      providerConfig.models.length === 0 || providerConfig.models.includes(model)
  );
  if (matching.length > 0) {
    return matching;
  }

  return preferredConfigs.length > 0 ? preferredConfigs : providerFamilyConfigs;
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
    const referenceModel = endpoint.video?.reference?.model;
    if (referenceModel && fromHeader.model !== referenceModel) {
      return {
        ok: false,
        error: `x-target-model "${fromHeader.raw}" conflicts with the signed video model "${referenceModel}".`
      };
    }
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

  const providerConfig = resolveProviderConfig(config, target);
  const inferredMediaModel =
    endpoint.video?.operation === 'create' && providerConfig?.models.length === 1
      ? providerConfig.models[0]
      : undefined;
  return validateModelForTarget(
    endpoint.useDefaultOpenAIModel
      ? config.defaultOpenAIModel
      : inferredMediaModel,
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

  const byType = providerConfigs.find(
    (providerConfig) => providerConfig.type.trim().toLowerCase() === normalized.toLowerCase()
  );
  if (byType) {
    return { provider: providerFromProviderType(byType.type) };
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

  const provider = parseConfiguredModelReferenceProvider(providerHint, providerConfigs);
  if (provider) {
    return { raw, model, provider };
  }

  return { raw, model: raw };
}

function parseConfiguredModelReferenceProvider(
  providerHint: string,
  providerConfigs: ProviderConfig[]
): Provider | undefined {
  const provider = parseProvider(providerHint);
  if (!provider) {
    return undefined;
  }

  if (provider === 'openai' || provider === 'anthropic' || provider === 'gemini') {
    return provider;
  }

  return providerConfigs.some((providerConfig) => providerFromProviderType(providerConfig.type) === provider)
    ? provider
    : undefined;
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
    const routeProviderName =
      route.providerConfig?.credentialSourceProviderName || route.providerConfig?.name;
    return routeProviderName === reference.providerConfig.name;
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
          sourceProvider: context.endpoint.sourceProvider || 'openai',
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
          sourceProvider: context.endpoint.sourceProvider || 'openai',
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
        sourceProvider: context.endpoint.sourceProvider || 'openai',
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

async function processOpenAIJsonEventStreamUsage(input: {
  endpoint: OpenAIJsonEndpointConfig;
  request: FastifyRequest;
  reply: FastifyReply;
  config: GatewayConfig;
  targetProvider: Provider;
  model: string | undefined;
  targetProviderConfig: ProviderConfig | undefined;
  response: Response;
  upstreamRequest: UpstreamRequest;
  fallbackAttempts: number;
  responseStatusCode: number;
}): Promise<void> {
  try {
    const usagePayload = await readOpenAIJsonEventStreamUsagePayload(input.response);
    attachOpenAIJsonBillingHeaders(
      input.endpoint,
      input.request,
      input.reply,
      input.config,
      input.targetProvider,
      input.model,
      input.targetProviderConfig,
      usagePayload,
      usagePayload,
      input.upstreamRequest,
      input.fallbackAttempts,
      input.responseStatusCode,
      { attachHeaders: false }
    );
  } catch (error) {
    input.request.log.warn(
      {
        provider: input.targetProvider,
        model: input.model,
        details: error instanceof Error ? error.message : String(error)
      },
      `Failed to parse ${input.endpoint.displayName.toLowerCase()} event-stream usage for billing.`
    );
  }
}

async function readOpenAIJsonEventStreamUsagePayload(response: Response): Promise<unknown> {
  if (!response.body) {
    return undefined;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  let oversizedEvent = false;
  let usagePayload: unknown;
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    while (true) {
      const separator = findOpenAIJsonEventSeparator(pending);
      if (!separator) {
        break;
      }
      const eventEnd = separator.index;
      const eventStart = Math.max(0, eventEnd - maxOpenAIJsonUsageEventTailChars);
      const event = pending.slice(eventStart, eventEnd);
      if (!oversizedEvent || event.includes('"usage"') || event.includes('cost_in_usd_ticks')) {
        usagePayload = extractOpenAIJsonEventUsagePayload(event) ?? usagePayload;
      }
      pending = pending.slice(eventEnd + separator.length);
      oversizedEvent = false;
    }
    if (pending.length > maxOpenAIJsonUsageEventTailChars) {
      pending = pending.slice(-maxOpenAIJsonUsageEventTailChars);
      oversizedEvent = true;
    }
    if (done) {
      break;
    }
  }
  if (!oversizedEvent || pending.includes('"usage"') || pending.includes('cost_in_usd_ticks')) {
    usagePayload = extractOpenAIJsonEventUsagePayload(pending) ?? usagePayload;
  }
  return usagePayload;
}

function extractOpenAIJsonEventUsagePayload(event: string): unknown {
  const dataLines = event
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart());
  const data = dataLines.length > 0
    ? dataLines.join('\n')
    : event.includes('"usage"') || event.includes('cost_in_usd_ticks')
      ? event
      : '';
  if (!data || data === '[DONE]') {
    return undefined;
  }
  const usage = extractJsonObjectProperty(data, 'usage');
  if (usage) {
    return { usage };
  }
  if (!data.includes('cost_in_usd_ticks')) {
    return undefined;
  }
  try {
    const payload = JSON.parse(data) as unknown;
    return extractOpenAIJsonUsage(payload) ||
      extractProviderReportedCostUsd(payload) !== undefined
      ? payload
      : undefined;
  } catch {
    // Ignore non-JSON SSE events; only completed media events carry billable usage.
    return undefined;
  }
}

function findOpenAIJsonEventSeparator(
  value: string
): { index: number; length: number } | undefined {
  const match = /\r?\n\r?\n/.exec(value);
  return match?.index === undefined
    ? undefined
    : { index: match.index, length: match[0].length };
}

function extractJsonObjectProperty(
  json: string,
  propertyName: string
): Record<string, unknown> | undefined {
  const marker = JSON.stringify(propertyName);
  let searchFrom = 0;
  while (searchFrom < json.length) {
    const markerIndex = json.indexOf(marker, searchFrom);
    if (markerIndex < 0) {
      return undefined;
    }
    let cursor = markerIndex + marker.length;
    while (/\s/.test(json[cursor] || '')) {
      cursor += 1;
    }
    if (json[cursor] !== ':') {
      searchFrom = markerIndex + marker.length;
      continue;
    }
    cursor += 1;
    while (/\s/.test(json[cursor] || '')) {
      cursor += 1;
    }
    if (json[cursor] !== '{') {
      searchFrom = markerIndex + marker.length;
      continue;
    }
    const objectText = readBalancedJsonObject(json, cursor);
    if (!objectText) {
      return undefined;
    }
    try {
      const parsed = JSON.parse(objectText) as unknown;
      return isObject(parsed) ? parsed : undefined;
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function readBalancedJsonObject(json: string, start: number): string | undefined {
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < json.length; index += 1) {
    const character = json[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && character === '\\') {
      escaped = true;
      continue;
    }
    if (character === '"') {
      quoted = !quoted;
      continue;
    }
    if (quoted) {
      continue;
    }
    if (character === '{') {
      depth += 1;
    } else if (character === '}') {
      depth -= 1;
      if (depth === 0) {
        return json.slice(start, index + 1);
      }
    }
  }
  return undefined;
}

function resolveVideoStatusBillingOptions(
  endpoint: OpenAIJsonEndpointConfig,
  payload: unknown
):
  | {
      outcome: {
        status: 'error';
        statusCode: number;
        errorMessage: string;
      };
    }
  | undefined {
  if (endpoint.video?.operation !== 'status' || !isObject(payload)) {
    return undefined;
  }

  const status = readRecordString(payload, 'status')?.toLowerCase();
  if (status !== 'failed' && status !== 'cancelled' && status !== 'expired') {
    return undefined;
  }

  const error = isObject(payload.error) ? payload.error : undefined;
  const errorMessage =
    readRecordString(error, 'message') ||
    (typeof payload.error === 'string' && payload.error.trim()
      ? payload.error.trim()
      : status === 'expired'
        ? 'The video request has expired.'
        : status === 'cancelled'
          ? 'The video request was cancelled.'
          : 'The video generation failed.');
  return {
    outcome: {
      status: 'error',
      statusCode: status === 'expired' ? 410 : 422,
      errorMessage
    }
  };
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
  providerPayload: unknown,
  upstreamRequest: UpstreamRequest,
  fallbackAttempts: number,
  responseStatusCode: number,
  options: {
    attachHeaders?: boolean;
    outcome?: {
      status: 'success' | 'error' | 'timeout' | 'rate-limited';
      statusCode?: number;
      errorMessage?: string;
    };
  } = {}
): void {
  const reportedCostUsd = extractProviderReportedCostUsd(providerPayload);
  const targetVideoProtocol = endpoint.video
    ? videoProtocolForTarget(
        targetProvider,
        targetProviderConfig?.type,
        endpoint.video.sourceProtocol
      )
    : undefined;
  const governanceResult = buildOpenAIJsonGovernanceRequest(
    endpoint,
    upstreamRequest,
    targetVideoProtocol,
    model
  );
  const governance = governanceResult.ok ? governanceResult.value : undefined;
  const billingModel = governance?.model || model;
  const rate = resolveProviderBillingRate(
    config,
    targetProvider,
    billingModel,
    targetProviderConfig
  );
  const videoRate = resolveVideoPerSecondUsd(rate, governance?.videoSize);
  const usage =
    extractOpenAIJsonUsage(payload) ||
    (endpoint.video?.operation === 'create' && videoRate !== undefined
      ? {
          video_seconds: governance?.videoSeconds,
          video_size: governance?.videoSize
        }
      : undefined);
  recordGatewaySchedulingUsage({
    config,
    request,
    providerConfig: targetProviderConfig,
    model: billingModel,
    usage
  });
  if (!config.billing.enabled) {
    return;
  }
  if (!usage && reportedCostUsd === undefined) {
    if (endpoint.billingUsageOptional) {
      return;
    }

    request.log.warn(
      { provider: targetProvider, model: billingModel },
      `Failed to parse ${endpoint.displayName.toLowerCase()} usage for billing.`
    );
    return;
  }

  const billing =
    reportedCostUsd !== undefined
      ? createProviderReportedCostBilling(targetProvider, reportedCostUsd, config.billing)
      : calculateUsageBilling(targetProvider, usage || {}, config.billing, rate);
  if (options.attachHeaders !== false) {
    for (const [key, value] of Object.entries(buildBillingHeaders(billing))) {
      reply.header(key, value);
    }
  }

  if (
    endpoint.video?.operation === 'create' &&
    targetVideoProtocol === 'xai' &&
    reportedCostUsd === undefined
  ) {
    return;
  }

  const videoBillingRequestId =
    reportedCostUsd !== undefined && endpoint.video
      ? endpoint.video.operation === 'status'
        ? endpoint.video.publicRequestId
        : isObject(payload)
          ? readVideoResponseId(payload, endpoint.video.sourceProtocol)
          : undefined
      : undefined;
  if (
    videoBillingRequestId &&
    !claimVideoBillingEvent(videoBillingRequestId, config.media?.videoIdTtlMs)
  ) {
    return;
  }

  void publishBillingEvent({
    eventId: videoBillingRequestId
      ? buildVideoBillingEventId(targetProviderConfig?.name || targetProvider, videoBillingRequestId)
      : randomUUID(),
    emittedAt: new Date().toISOString(),
    requestId: request.id,
    clientIp: resolveGatewayClientIp(request, config),
    route: {
      method: request.method,
      url: request.url
    },
    source: {
      provider: endpoint.sourceProvider || 'openai',
      adapterKey: endpoint.sourceAdapterKey
    },
    target: {
      provider: targetProvider,
      providerName: targetProviderConfig?.name,
      model: billingModel
    },
    fallback: {
      used: fallbackAttempts > 0,
      attempts: fallbackAttempts
    },
    identity: request.gatewayIdentity,
    outcome: options.outcome || {
      status: 'success',
      statusCode: responseStatusCode
    },
    billing
  })
    .then((delivered) => {
      if (!videoBillingRequestId) {
        return;
      }
      if (delivered) {
        completeVideoBillingEvent(videoBillingRequestId);
      } else {
        releaseVideoBillingEvent(videoBillingRequestId);
      }
    })
    .catch((error) => {
      if (videoBillingRequestId) {
        releaseVideoBillingEvent(videoBillingRequestId);
      }
      request.log.warn(
        { details: error instanceof Error ? error.message : String(error) },
        `Failed to publish ${endpoint.displayName.toLowerCase()} billing event.`
      );
    });
}

function extractProviderReportedCostUsd(payload: unknown): number | undefined {
  if (!isObject(payload) || !isObject(payload.usage)) {
    return undefined;
  }
  const ticks = asNumber(payload.usage.cost_in_usd_ticks);
  return ticks !== undefined && Number.isFinite(ticks) && ticks >= 0
    ? ticks / 10_000_000_000
    : undefined;
}

function buildVideoBillingEventId(providerKey: string, publicRequestId: string): string {
  return `video_${createHash('sha256')
    .update(`${providerKey}:${publicRequestId}`)
    .digest('hex')}`;
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
  fallbackAttempts: number,
  providerConfig?: ProviderConfig
): void {
  reply.header('x-gateway-target-provider', provider);
  if (providerName) {
    reply.header('x-gateway-target-provider-name', providerName);
  }
  attachGatewaySchedulingHeaders(reply, providerConfig);
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

function isEventStreamResponse(response: Response): boolean {
  return response.headers.get('content-type')?.toLowerCase().includes('text/event-stream') ?? false;
}

function shouldRelayVideoContentResponse(status: number): boolean {
  return status === 304 || status === 416;
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

function buildApiKeyModelRestrictionAttempt(
  provider: Provider,
  providerConfig: ProviderConfig | undefined,
  result: Extract<ReturnType<typeof evaluateApiKeyModelRestriction>, { ok: false }>
): OpenAIJsonAttemptFailure {
  return {
    provider,
    providerName: providerConfig?.name,
    stage: 'api_key_model_restriction',
    message: result.error,
    status: result.statusCode
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
  return findDefaultProviderConfig(providers, provider);
}

function findProviderConfigByName(
  providers: ProviderConfig[],
  name: string
): ProviderConfig | undefined {
  const normalized = name.trim().toLowerCase();
  return providers.find((item) => item.name.trim().toLowerCase() === normalized);
}

function inferUnwrappedVideoProtocol(
  request: FastifyRequest,
  config: GatewayConfig
): VideoApiProtocol {
  const targets = resolveTargetProviders(request, config, undefined, videoStatusEndpoint);
  if (!targets.ok || targets.value.length === 0) {
    return 'openai';
  }
  const target = targets.value[0];
  const providerConfig = resolveProviderConfig(config, target);
  return videoProtocolForTarget(target.provider, providerConfig?.type, 'openai');
}

function resolveVideoReferenceProviderConfig(
  reference: GatewayVideoReference,
  providers: ProviderConfig[]
): ProviderConfig | undefined {
  let providerConfig: ProviderConfig | undefined;
  if (reference.targetProviderName) {
    providerConfig = findProviderConfigByName(providers, reference.targetProviderName);
  } else if (reference.targetProviderKey) {
    providerConfig = providers.find(
      (providerConfig) => videoProviderKey(providerConfig.name) === reference.targetProviderKey
    );
  } else {
    providerConfig = providers.find(
      (item) => providerFromProviderType(item.type) === reference.targetProvider
    );
  }

  return providerConfig
    ? resolveGatewayScheduledCredential(providerConfig, reference.targetCredentialId)
    : undefined;
}

function enrichVideoReference(
  reference: GatewayVideoReference | undefined,
  providerConfig: ProviderConfig | undefined
): GatewayVideoReference | undefined {
  if (!reference || reference.model || providerConfig?.models.length !== 1) {
    return reference;
  }
  return {
    ...reference,
    model: providerConfig.models[0]
  };
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
  return (
    (model ? providerConfig?.billing.byModel[model] : undefined) ||
    providerConfig?.billing.default ||
    config.billing.rates[provider]
  );
}

function hasCostBudgetPrecheck(request: FastifyRequest, config: GatewayConfig): boolean {
  if (
    config.precheck.enabled &&
    config.precheck.budget.enabled &&
    config.precheck.budget.maxCostUsd > 0
  ) {
    return true;
  }

  const restrictions = request.gatewayApiKeyRestrictions;
  const limit =
    restrictions?.costLimitUsd ??
    restrictions?.costLimit ??
    restrictions?.maxCostUsd ??
    restrictions?.costPerMinuteUsd;
  return typeof limit === 'number' && Number.isFinite(limit) && limit > 0;
}

function readBodyModel(body: Record<string, unknown> | undefined): string | undefined {
  return typeof body?.model === 'string' && body.model.trim() ? body.model.trim() : undefined;
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return isObject(value) && !Array.isArray(value) && !Buffer.isBuffer(value);
}

function isMultipartFormDataContentType(value: string | undefined): boolean {
  return /^multipart\/form-data(?:\s*;|$)/i.test(value?.trim() || '');
}

function formatTargetProviderLabel(route: TargetProviderRoute): string {
  return route.providerConfig?.name || route.provider;
}

function sendBadRequest(reply: FastifyReply, message: string) {
  return reply.code(400).send({ error: { message } });
}

function sendForbidden(reply: FastifyReply, message: string) {
  return reply.code(403).send({ error: { message } });
}

function videoIdCodecOptions(config: GatewayConfig) {
  return {
    signingSecret: config.media?.videoIdSigningSecret,
    ttlMs: config.media?.videoIdTtlMs
  };
}

function isVideoReferenceOwner(
  reference: GatewayVideoReference | undefined,
  request: FastifyRequest,
  authEnabled: boolean
): boolean {
  if (!reference) {
    return true;
  }
  if (!reference.ownerKey) {
    return !authEnabled;
  }
  return reference.ownerKey === videoOwnerKey(request.gatewayIdentity?.billingSubjectKey);
}

function trimRightSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

function formatPluginExecutionError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
