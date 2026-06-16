import { randomUUID } from 'node:crypto';
import type { FastifyReply, FastifyRequest } from 'fastify';
import type { BillingResult } from '../billing';
import { buildBillingHeaders, calculateUsageBilling, publishBillingEvent } from '../billing';
import type { AgentToolDefinition } from '../agent/types';
import type {
  GatewayBillingTrace,
  GatewayConfig,
  GatewayRequestClientContext,
  GatewaySourceContext,
  Provider,
  ProviderPlugin,
  ProviderConfig,
  SourceAdapter,
  StandardRequest,
  StandardRequestInputContent,
  StandardRequestInputMessage,
  StandardResponse,
  StandardResponseFunctionCall,
  StandardUsage,
  TargetAdapter,
  UpstreamRequest,
  VirtualModelProfileConfig
} from '../types';
import {
  callUpstream,
  forceEventStreamHeaders,
  readUpstreamPayload,
  relayUpstreamResponse,
  sanitizeHeadersForLog,
  sanitizePayloadForLog
} from '../upstream/client';
import {
  asNumber,
  asString,
  collectStandardInputMessages,
  isObject,
  normalizeMessageRole,
  parseProvider,
  providerFromProviderType,
  readHeader
} from '../utils';
import {
  collectAnthropicNonStreamPayloadFromEventStream,
  collectOpenAINonStreamPayloadFromEventStream,
  relayConvertedStreamFromStandardResponse,
  relayConvertedStreamFromUpstreamResponse
} from './streaming-conversion';
import { addNamespaceFieldsToStandardResponse } from '../adapters/builtins/target/tools';
import type { GatewayRuntime } from './runtime';
import { applyHealthAwareRouting } from './health-routing';
import { evaluateGatewayPolicy, type GatewayPolicyResult } from './policy';
import { recordProviderHealthFailure, recordProviderHealthResponse } from './provider-health';
import { evaluateGatewayPrecheck, type GatewayPrecheckResult } from './precheck';
import {
  checkProviderCircuitBreaker,
  recordProviderCircuitBreakerFailure,
  recordProviderCircuitBreakerResponse
} from './upstream-circuit-breaker';
import { acquireProviderConcurrencySlot } from './upstream-concurrency';
import {
  recordGatewayStreamConversion,
  recordGatewayToolExecution
} from './metrics';
import {
  enqueueRawTraceCapture,
  markRawTraceCaptureSubmitted,
  readRawRequestBody,
} from '../raw-trace';
import { matchesAnyPattern } from '../shared/pattern';

interface ProviderAttemptFailure {
  provider: Provider;
  providerName?: string;
  stage: string;
  message: string;
  status?: number;
  details?: unknown;
  upstreamRequest?: UpstreamRequest;
  upstreamResponseBody?: unknown;
}

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

type ProviderRequestPluginFailureStage = 'provider_auth' | 'provider_request_transform';

type ProviderRequestPluginResult =
  | { ok: true; value: UpstreamRequest }
  | { ok: false; stage: ProviderRequestPluginFailureStage; status: number; message: string };

type ProviderResponsePluginResult =
  | { ok: true; value: unknown }
  | { ok: false; stage: 'provider_response_transform'; status: number; message: string };

interface ProviderPluginExecutionContext {
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
  plugins: ProviderPlugin[];
}

interface VirtualModelResolution {
  profile: VirtualModelProfileConfig;
  requestedModel: string;
  requestedModelReference: ParsedModelReference;
  matchedBy: 'exact' | 'prefix' | 'suffix';
  matchedToken: string;
  targetModelSelector: string;
  targetModelReference: ParsedModelReference;
}

interface VirtualToolOwner {
  visibility: 'internal' | 'client';
  source: 'profile' | 'client';
  runtimeToolName?: string;
}

interface TransparentToolBinding {
  runtimeToolName: string;
}

type TransparentToolResolution =
  | {
      ok: true;
      executableCalls: StandardResponseFunctionCall[];
      bindings: Map<string, TransparentToolBinding>;
    }
  | {
      ok: false;
      stage: 'transparent_tool_resolution';
      status: number;
      message: string;
    }
  | {
      ok: true;
      executableCalls: [];
      bindings: Map<string, TransparentToolBinding>;
      returnToClient: true;
    };

type TransparentToolLoopResult =
  | {
      ok: true;
      standardRequest: StandardRequest;
      standardResponse: StandardResponse;
      upstreamRequest: UpstreamRequest;
      attemptSequence: number;
      upstreamAttemptSequence: number;
    }
  | {
      ok: false;
      upstreamAttemptSequence: number;
    };

export interface VirtualMultimodalReference {
  id: string;
  kind: 'image' | 'file' | 'document' | 'media';
  sourceType: 'url' | 'base64';
  value: string;
  mimeType?: string;
  filename?: string;
}

export interface VirtualMultimodalRewrite {
  request: StandardRequest;
  references: VirtualMultimodalReference[];
}

interface BillingResponseSnapshot {
  model?: string;
  usage: StandardUsage;
  recovered: boolean;
}

type GatewayUsageOutcomeStatus =
  | 'success'
  | 'error'
  | 'timeout'
  | 'rate-limited';

interface GatewayBillingAttempt {
  kind: 'upstream_attempt';
  sequence: number;
}

interface GatewayRawTraceCapture {
  upstreamRequest?: UpstreamRequest;
  upstreamResponseBody?: unknown;
  upstreamResponseHeaders?: Record<string, string>;
  upstreamResponseStatus?: number;
  upstreamResponseStream?: unknown;
  upstreamResponseStreamContentType?: string;
}

export async function handleGatewayRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  source: GatewaySourceContext,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  const sourceAdapter = runtime.sourceAdapters.get(source.adapterKey);
  if (!sourceAdapter) {
    return reply.code(500).send({
      error: {
        message: `Source adapter is not registered: ${source.adapterKey}`
      }
    });
  }

  const body = request.body;
  if (!isObject(body)) {
    return sendBadRequest(reply, 'Request body must be a JSON object.');
  }

  const adapterInput = {
    request,
    body,
    source,
    config
  };
  const requestedModelSelector =
    readHeader(request.headers['x-target-model']) || resolvePassthroughModel(body, source);
  const virtualModelResolution = requestedModelSelector
    ? resolveVirtualModelRequest(config, requestedModelSelector)
    : undefined;

  const targetProvidersResult = resolveTargetProviders(
    request,
    sourceAdapter.provider,
    config,
    virtualModelResolution?.targetModelSelector || resolvePassthroughModel(body, source)
  );
  if (!targetProvidersResult.ok) {
    return sendBadRequest(reply, targetProvidersResult.error);
  }

  const targetProviders = applyHealthAwareRouting(targetProvidersResult.value, config);
  const isStreaming = sourceAdapter.isStreamingRequest(adapterInput);

  if (virtualModelResolution) {
    return handleVirtualModelRequest(
      request,
      reply,
      source,
      config,
      runtime,
      sourceAdapter,
      targetProviders,
      adapterInput,
      isStreaming,
      virtualModelResolution
    );
  }

  const attempts: ProviderAttemptFailure[] = [];
  let upstreamAttemptSequence = 0;
  let baseStandardRequest: StandardRequest | undefined;
  let precheckApplied = false;
  const runPrecheckOnce = async (
    targetProvider: Provider,
    targetProviderConfig: ProviderConfig | undefined,
    model: string | undefined,
    standardRequest: StandardRequest | undefined,
    requestBody: unknown
  ): Promise<GatewayPrecheckResult> => {
    if (precheckApplied) {
      return { ok: true };
    }

    const result = await evaluateGatewayPrecheck({
      request,
      config,
      targetProvider,
      targetProviderConfig,
      model,
      standardRequest,
      requestBody
    });
    if (result.ok) {
      precheckApplied = true;
    }
    return result;
  };

  for (const target of targetProviders) {
    const targetProvider = target.provider;
    const targetProviderConfig = resolveProviderConfig(config, target);
    const providerPlugins = runtime.providerPlugins.resolve(targetProvider, targetProviderConfig?.name);
    const targetProviderLabel = formatTargetProviderLabel(target);
    const targetAdapter = runtime.targetAdapters.get(targetProvider);
    if (!targetAdapter) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_adapter_lookup',
        message: `Target adapter is not registered for provider: ${targetProvider}`,
        status: 400
      });
      continue;
    }

    if (
      canPassthroughWithoutProtocolConversion(
        source.adapterKey,
        sourceAdapter.provider,
        targetProvider,
        targetProviderConfig
      ) &&
      !shouldUseTransparentToolExecutionPath(config, isStreaming)
    ) {
      const passthroughResult = sourceAdapter.buildPassthroughRequest(adapterInput);
      if (!passthroughResult.ok) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'passthrough_build',
          message: passthroughResult.error,
          status: 400
        });
        continue;
      }

      const passthroughModelResult = resolveTargetModel(
        request,
        target,
        resolvePassthroughModel(body, source),
        config
      );
      if (!passthroughModelResult.ok) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'model_resolution',
          message: passthroughModelResult.error,
          status: 400
        });
        continue;
      }

      const passthroughModel = passthroughModelResult.value;
      const policyResult = evaluateGatewayPolicy({
        request,
        config,
        targetProvider,
        targetProviderConfig,
        model: passthroughModel
      });
      if (!policyResult.ok) {
        attempts.push(buildGatewayPolicyAttempt(targetProvider, targetProviderConfig, policyResult));
        continue;
      }

      const passthroughRequest = applyProviderRequestOverrides(
        config,
        target,
        passthroughModel,
        passthroughResult.value
      );
      const baseUpstreamRequest = applyPassthroughModelOverride(
        passthroughRequest,
        targetProvider,
        passthroughModel
      );
      const providerPluginContext: ProviderPluginExecutionContext = {
        request,
        config,
        source,
        sourceProvider: sourceAdapter.provider,
        sourceAdapterKey: source.adapterKey,
        targetProvider,
        targetProviderConfig,
        model: passthroughModel,
        passthrough: true,
        streaming: isStreaming,
        plugins: providerPlugins
      };

      const precheckResult = await runPrecheckOnce(
        targetProvider,
        targetProviderConfig,
        passthroughModel,
        undefined,
        body
      );
      if (!precheckResult.ok) {
        return sendGatewayPrecheckFailure(reply, precheckResult);
      }

      const upstreamDispatchResult = await dispatchUpstreamRequest(
        providerPluginContext,
        baseUpstreamRequest,
        config.upstreamTimeoutMs
      );
      if (!upstreamDispatchResult.ok) {
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: upstreamDispatchResult.stage,
          message: upstreamDispatchResult.message,
          status: upstreamDispatchResult.status,
          details: upstreamDispatchResult.details,
          upstreamRequest: upstreamDispatchResult.upstreamRequest,
          upstreamResponseBody:
            upstreamDispatchResult.details !== undefined
              ? {
                  error: {
                    message: upstreamDispatchResult.message,
                    details: upstreamDispatchResult.details,
                  },
                }
              : undefined,
        };
        attempts.push(attempt);
        if (upstreamDispatchResult.upstreamRequest) {
          upstreamAttemptSequence += 1;
          publishFailedAttemptEventSafe(
            request,
            reply,
            config,
            sourceAdapter.provider,
            source.adapterKey,
            attempt,
            passthroughModel,
            upstreamAttemptSequence,
            attempts,
            targetProviderConfig,
          );
        }
        continue;
      }

      const { upstreamRequest, upstreamResponse } = upstreamDispatchResult;
      upstreamAttemptSequence += 1;
      const currentAttemptSequence = upstreamAttemptSequence;

      if (!upstreamResponse.ok) {
        const details = await safeReadUpstreamPayload(request, targetProvider, upstreamResponse);
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'upstream_response',
          message: 'Upstream request failed.',
          status: upstreamResponse.status,
          details,
          upstreamRequest,
          upstreamResponseBody: details,
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          passthroughModel,
          currentAttemptSequence,
          attempts,
          targetProviderConfig,
        );
        continue;
      }

      if (!isStreaming) {
        const hasResponseTransformPlugin = providerPlugins.some((plugin) => Boolean(plugin.transformResponse));
        let transformedPayload: unknown | undefined;
        if (
          !isEventStreamResponse(upstreamResponse) &&
          (hasResponseTransformPlugin ||
            config.billing.enabled ||
            config.rawTrace.enabled)
        ) {
          const upstreamPayload = await safeReadUpstreamPayload(request, targetProvider, upstreamResponse.clone());
          let billingPayload = upstreamPayload;

          if (hasResponseTransformPlugin) {
            const responsePluginResult = await applyProviderResponsePlugins(
              providerPluginContext,
              upstreamRequest,
              upstreamResponse,
              upstreamPayload
            );
            if (!responsePluginResult.ok) {
              const attempt: ProviderAttemptFailure = {
                provider: targetProvider,
                providerName: targetProviderConfig?.name,
                stage: responsePluginResult.stage,
                message: responsePluginResult.message,
                status: responsePluginResult.status,
                upstreamRequest,
                upstreamResponseBody: upstreamPayload,
              };
              attempts.push(attempt);
              publishFailedAttemptEventSafe(
                request,
                reply,
                config,
                sourceAdapter.provider,
                source.adapterKey,
                attempt,
                passthroughModel,
                currentAttemptSequence,
                attempts,
                targetProviderConfig,
              );
              continue;
            }

            transformedPayload = responsePluginResult.value;
            billingPayload = transformedPayload;
          }

          if (config.billing.enabled || config.rawTrace.enabled) {
            tryAttachBillingHeadersFromUpstreamPayload(
              request,
              reply,
              config,
              targetProvider,
              sourceAdapter.provider,
              source.adapterKey,
              attempts.length,
              currentAttemptSequence,
              targetAdapter,
              upstreamRequest,
              billingPayload,
              passthroughModel,
              targetProviderConfig,
              transformedPayload ?? upstreamPayload,
              upstreamResponse.status
            );
          }
        } else {
          await tryAttachBillingHeadersFromUpstreamResponse(
            request,
            reply,
            config,
            targetProvider,
            sourceAdapter.provider,
            source.adapterKey,
            attempts.length,
            currentAttemptSequence,
            targetAdapter,
            upstreamRequest,
            upstreamResponse,
            passthroughModel,
            targetProviderConfig
          );
        }

        attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
        if (transformedPayload !== undefined) {
          return relayUpstreamResponseWithPayload(reply, upstreamResponse, transformedPayload);
        }
        if (shouldForceEventStreamHeaders(source, isStreaming)) {
          forceEventStreamHeaders(reply);
        }
        return relayUpstreamResponse(reply, upstreamResponse);
      } else {
        const rawTraceStreamResponse = cloneResponseForRawStreamTrace(config, upstreamResponse);
        void tryPublishStreamingBillingEventFromUpstreamResponse(
          request,
          reply,
          config,
          targetProvider,
          sourceAdapter.provider,
          source.adapterKey,
          attempts.length,
          currentAttemptSequence,
          targetAdapter,
          upstreamRequest,
          upstreamResponse.clone(),
          passthroughModel,
          targetProviderConfig,
          rawTraceStreamResponse
        );
      }

      attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
      if (isStreaming) {
        recordGatewayStreamConversion({
          sourceAdapter: source.adapterKey,
          targetProvider,
          targetProviderName: targetProviderConfig?.name,
          mode: 'passthrough'
        });
      }
      if (shouldForceEventStreamHeaders(source, isStreaming)) {
        forceEventStreamHeaders(reply);
      }
      return relayUpstreamResponse(reply, upstreamResponse);
    }

    if (isStreaming) {
      if (!baseStandardRequest) {
        const standardRequestResult = sourceAdapter.toStandardRequest(adapterInput);
        if (!standardRequestResult.ok) {
          return sendBadRequest(reply, standardRequestResult.error);
        }

        baseStandardRequest = standardRequestResult.value;
      }

      const modelResult = resolveTargetModel(request, target, baseStandardRequest.model, config);
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
      if (!model) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'model_resolution',
          message: `Model is required. Provide model in body, x-target-model header, or default model env for ${targetProviderLabel}.`,
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

      const supportsLiveStreamConversion = canRelayLiveConvertedStream(source, targetProvider, targetProviderConfig);
      const standardRequest: StandardRequest = {
        ...baseStandardRequest,
        model,
        stream: supportsLiveStreamConversion
      };

      const targetRequestResult = targetAdapter.buildRequestFromStandard({
        request,
        standardRequest,
        config,
        targetProviderConfig
      });
      if (!targetRequestResult.ok) {
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'target_request_build',
          message: targetRequestResult.error,
          status: 400
        });
        continue;
      }

      const baseUpstreamRequest = applyProviderRequestOverrides(
        config,
        target,
        model,
        targetRequestResult.value
      );
      const providerPluginContext: ProviderPluginExecutionContext = {
        request,
        config,
        source,
        sourceProvider: sourceAdapter.provider,
        sourceAdapterKey: source.adapterKey,
        targetProvider,
        targetProviderConfig,
        model,
        passthrough: false,
        streaming: true,
        plugins: providerPlugins
      };

      const precheckResult = await runPrecheckOnce(
        targetProvider,
        targetProviderConfig,
        model,
        standardRequest,
        body
      );
      if (!precheckResult.ok) {
        return sendGatewayPrecheckFailure(reply, precheckResult);
      }

      const upstreamDispatchResult = await dispatchUpstreamRequest(
        providerPluginContext,
        baseUpstreamRequest,
        config.upstreamTimeoutMs,
        standardRequest
      );
      if (!upstreamDispatchResult.ok) {
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: upstreamDispatchResult.stage,
          message: upstreamDispatchResult.message,
          status: upstreamDispatchResult.status,
          details: upstreamDispatchResult.details,
          upstreamRequest: upstreamDispatchResult.upstreamRequest,
          upstreamResponseBody:
            upstreamDispatchResult.details !== undefined
              ? {
                  error: {
                    message: upstreamDispatchResult.message,
                    details: upstreamDispatchResult.details,
                  },
                }
              : undefined,
        };
        attempts.push(attempt);
        if (upstreamDispatchResult.upstreamRequest) {
          upstreamAttemptSequence += 1;
          publishFailedAttemptEventSafe(
            request,
            reply,
            config,
            sourceAdapter.provider,
            source.adapterKey,
            attempt,
            model,
            upstreamAttemptSequence,
            attempts,
            targetProviderConfig,
          );
        }
        continue;
      }

      const { upstreamRequest, upstreamResponse } = upstreamDispatchResult;
      upstreamAttemptSequence += 1;
      const currentAttemptSequence = upstreamAttemptSequence;

      if (!upstreamResponse.ok) {
        const upstreamPayload = await safeReadUpstreamPayload(request, targetProvider, upstreamResponse);
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'upstream_response',
          message: 'Upstream request failed.',
          status: upstreamResponse.status,
          details: upstreamPayload,
          upstreamRequest,
          upstreamResponseBody: upstreamPayload,
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          currentAttemptSequence,
          attempts,
          targetProviderConfig,
        );
        continue;
      }

      if (canTreatAsLiveStreamResponse(source, targetProvider, upstreamResponse)) {
        recordGatewayStreamConversion({
          sourceAdapter: source.adapterKey,
          targetProvider,
          targetProviderName: targetProviderConfig?.name,
          mode: 'live'
        });
        const rawTraceStreamResponse = cloneResponseForRawStreamTrace(config, upstreamResponse);
        void tryPublishStreamingBillingEventFromUpstreamResponse(
          request,
          reply,
          config,
          targetProvider,
          sourceAdapter.provider,
          source.adapterKey,
          attempts.length,
          currentAttemptSequence,
          targetAdapter,
          upstreamRequest,
          upstreamResponse.clone(),
          model,
          targetProviderConfig,
          rawTraceStreamResponse
        );
        attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
        return relayConvertedStreamFromUpstreamResponse(reply, source, upstreamResponse, standardRequest);
      }

      recordGatewayStreamConversion({
        sourceAdapter: source.adapterKey,
        targetProvider,
        targetProviderName: targetProviderConfig?.name,
        mode: 'buffered'
      });
      const upstreamPayload = await safeReadUpstreamPayload(request, targetProvider, upstreamResponse);
      const responsePluginResult = await applyProviderResponsePlugins(
        providerPluginContext,
        upstreamRequest,
        upstreamResponse,
        upstreamPayload,
        standardRequest
      );
      if (!responsePluginResult.ok) {
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: responsePluginResult.stage,
          message: responsePluginResult.message,
          status: responsePluginResult.status,
          upstreamRequest,
          upstreamResponseBody: upstreamPayload,
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          currentAttemptSequence,
          attempts,
          targetProviderConfig,
        );
        continue;
      }

      const transformedPayload = responsePluginResult.value;
      const standardPayload = await normalizeOpenAIPayloadForResponseParseRecovery(
        request,
        targetProvider,
        transformedPayload
      );
      const standardResponseResult = targetAdapter.toStandardResponse(standardPayload);
      if (!standardResponseResult.ok) {
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'response_parse',
          message: standardResponseResult.error,
          status: 502,
          details: standardPayload,
          upstreamRequest,
          upstreamResponseBody: standardPayload,
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          currentAttemptSequence,
          attempts,
          targetProviderConfig,
        );
        continue;
      }

      const sourceStandardResponse = prepareStandardResponseForSource(
        source,
        standardResponseResult.value,
        standardRequest
      );
      attachBillingHeaders(
        request,
        reply,
        sourceAdapter.provider,
        source.adapterKey,
        targetProvider,
        standardResponseResult.value.usage,
        config,
        attempts.length,
        currentAttemptSequence,
        resolveBillingModel(standardResponseResult.value.model, model),
        targetProviderConfig,
        buildGatewayBillingTraceSnapshot(request, reply, {
          responseBody: sourceStandardResponse,
          standardResponse: sourceStandardResponse
        }),
        {
          upstreamRequest,
          upstreamResponseBody: sourceStandardResponse
        }
      );
      attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
      return relayConvertedStreamFromStandardResponse(reply, source, sourceStandardResponse);
    }

    if (!baseStandardRequest) {
      const standardRequestResult = sourceAdapter.toStandardRequest(adapterInput);
      if (!standardRequestResult.ok) {
        return sendBadRequest(reply, standardRequestResult.error);
      }

      baseStandardRequest = standardRequestResult.value;
    }

    const modelResult = resolveTargetModel(request, target, baseStandardRequest.model, config);
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
      if (!model) {
        attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'model_resolution',
        message: `Model is required. Provide model in body, x-target-model header, or default model env for ${targetProviderLabel}.`,
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

    const standardRequest: StandardRequest = {
      ...baseStandardRequest,
      model
    };

    const targetRequestResult = targetAdapter.buildRequestFromStandard({
      request,
      standardRequest,
      config,
      targetProviderConfig
    });
    if (!targetRequestResult.ok) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_request_build',
        message: targetRequestResult.error,
        status: 400
      });
      continue;
    }

    const baseUpstreamRequest = applyProviderRequestOverrides(
      config,
      target,
      model,
      targetRequestResult.value
    );
    const providerPluginContext: ProviderPluginExecutionContext = {
      request,
      config,
      source,
      sourceProvider: sourceAdapter.provider,
      sourceAdapterKey: source.adapterKey,
      targetProvider,
      targetProviderConfig,
      model,
      passthrough: false,
      streaming: false,
      plugins: providerPlugins
    };

    const precheckResult = await runPrecheckOnce(
      targetProvider,
      targetProviderConfig,
      model,
      standardRequest,
      body
    );
    if (!precheckResult.ok) {
      return sendGatewayPrecheckFailure(reply, precheckResult);
    }

    const upstreamDispatchResult = await dispatchUpstreamRequest(
      providerPluginContext,
      baseUpstreamRequest,
      config.upstreamTimeoutMs,
      standardRequest
    );
    if (!upstreamDispatchResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: upstreamDispatchResult.stage,
        message: upstreamDispatchResult.message,
        status: upstreamDispatchResult.status,
        details: upstreamDispatchResult.details,
        upstreamRequest: upstreamDispatchResult.upstreamRequest,
        upstreamResponseBody:
          upstreamDispatchResult.details !== undefined
            ? {
                error: {
                  message: upstreamDispatchResult.message,
                  details: upstreamDispatchResult.details,
                },
              }
            : undefined,
      };
      attempts.push(attempt);
      if (upstreamDispatchResult.upstreamRequest) {
        upstreamAttemptSequence += 1;
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          upstreamAttemptSequence,
          attempts,
          targetProviderConfig,
        );
      }
      continue;
    }

    const { upstreamRequest, upstreamResponse } = upstreamDispatchResult;
    upstreamAttemptSequence += 1;
    const currentAttemptSequence = upstreamAttemptSequence;

    const upstreamPayload =
      isEventStreamResponse(upstreamResponse) && targetProvider === 'openai'
        ? await collectOpenAINonStreamPayloadFromEventStream(upstreamResponse)
        : await safeReadUpstreamPayload(request, targetProvider, upstreamResponse);
    if (!upstreamResponse.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'upstream_response',
        message: 'Upstream request failed.',
        status: upstreamResponse.status,
        details: upstreamPayload,
        upstreamRequest,
        upstreamResponseBody: upstreamPayload,
      };
      attempts.push(attempt);
      publishFailedAttemptEventSafe(
        request,
        reply,
        config,
        sourceAdapter.provider,
        source.adapterKey,
        attempt,
        model,
        currentAttemptSequence,
        attempts,
        targetProviderConfig,
      );
      continue;
    }

    const responsePluginResult = await applyProviderResponsePlugins(
      providerPluginContext,
      upstreamRequest,
      upstreamResponse,
      upstreamPayload,
      standardRequest
    );
    if (!responsePluginResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: responsePluginResult.stage,
        message: responsePluginResult.message,
        status: responsePluginResult.status,
        upstreamRequest,
        upstreamResponseBody: upstreamPayload,
      };
      attempts.push(attempt);
      publishFailedAttemptEventSafe(
        request,
        reply,
        config,
        sourceAdapter.provider,
        source.adapterKey,
        attempt,
        model,
        currentAttemptSequence,
        attempts,
        targetProviderConfig,
      );
      continue;
    }

    const transformedPayload = responsePluginResult.value;
    const standardPayload = await normalizeOpenAIPayloadForResponseParseRecovery(
      request,
      targetProvider,
      transformedPayload
    );
    const standardResponseResult = targetAdapter.toStandardResponse(standardPayload);
    if (!standardResponseResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'response_parse',
        message: standardResponseResult.error,
        status: 502,
        details: standardPayload,
        upstreamRequest,
        upstreamResponseBody: standardPayload,
      };
      attempts.push(attempt);
      publishFailedAttemptEventSafe(
        request,
        reply,
        config,
        sourceAdapter.provider,
        source.adapterKey,
        attempt,
        model,
        currentAttemptSequence,
        attempts,
        targetProviderConfig,
      );
      continue;
    }

    const transparentToolExecutionResult = await runTransparentToolExecutionLoop({
      request,
      reply,
      source,
      config,
      runtime,
      target,
      targetProvider,
      targetProviderConfig,
      targetAdapter,
      providerPluginContext,
      model,
      initialStandardRequest: standardRequest,
      initialStandardResponse: standardResponseResult.value,
      initialUpstreamRequest: upstreamRequest,
      initialUpstreamResponseBody: transformedPayload,
      initialAttemptSequence: currentAttemptSequence,
      state: {
        attempts,
        upstreamAttemptSequence
      }
    });
    upstreamAttemptSequence = transparentToolExecutionResult.upstreamAttemptSequence;
    if (!transparentToolExecutionResult.ok) {
      continue;
    }

    const sourceStandardResponse = prepareStandardResponseForSource(
      source,
      transparentToolExecutionResult.standardResponse,
      transparentToolExecutionResult.standardRequest
    );
    const sourcePayload = sourceAdapter.fromStandardResponse({
      request,
      response: sourceStandardResponse,
      standardRequest: transparentToolExecutionResult.standardRequest,
      source,
      config
    });

    attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, attempts.length);
    attachBillingHeaders(
      request,
      reply,
      sourceAdapter.provider,
      source.adapterKey,
      targetProvider,
      transparentToolExecutionResult.standardResponse.usage,
      config,
      attempts.length,
      transparentToolExecutionResult.attemptSequence,
      resolveBillingModel(transparentToolExecutionResult.standardResponse.model, model),
      targetProviderConfig,
      buildGatewayBillingTraceSnapshot(request, reply, {
        responseBody: sourcePayload,
        standardResponse: sourceStandardResponse
      }),
      {
        upstreamRequest: transparentToolExecutionResult.upstreamRequest,
        upstreamResponseBody: sourcePayload
      }
    );

    return reply.code(200).send(sourcePayload);
  }

  const fallbackFailure = buildFallbackErrorPayload(targetProviders, attempts);
  reply.code(fallbackFailure.status);
  const lastTraceableAttempt = [...attempts]
    .reverse()
    .find(
      (attempt) =>
        attempt.upstreamRequest !== undefined ||
        attempt.upstreamResponseBody !== undefined,
    );
  const failureTarget =
    attempts[attempts.length - 1]
      ? {
          provider: attempts[attempts.length - 1]!.provider,
          providerConfig: attempts[attempts.length - 1]!.providerName
            ? findProviderConfigByName(
                config.providers,
                attempts[attempts.length - 1]!.providerName!,
              ) ||
              findProviderConfigByType(
                config.providers,
                attempts[attempts.length - 1]!.provider,
              )
            : findProviderConfigByType(
                config.providers,
                attempts[attempts.length - 1]!.provider,
              ),
        }
      : targetProviders[targetProviders.length - 1];
  if (failureTarget) {
    publishRequestFailureEventSafe(
      request,
      reply,
      config,
      sourceAdapter.provider,
      source.adapterKey,
      failureTarget.provider,
      baseStandardRequest?.model || resolvePassthroughModel(body, source),
      attempts.length,
      attempts,
      attempts[attempts.length - 1]?.message || fallbackFailure.payload.error.message,
      failureTarget.providerConfig,
      buildGatewayBillingTraceSnapshot(request, reply, {
        responseBody: fallbackFailure.payload,
      }),
      resolveAttemptRawTraceCapture(lastTraceableAttempt),
    );
  }

  return reply.send(fallbackFailure.payload);
}

async function runTransparentToolExecutionLoop(input: {
  request: FastifyRequest;
  reply: FastifyReply;
  source: GatewaySourceContext;
  config: GatewayConfig;
  runtime: GatewayRuntime;
  target: TargetProviderRoute;
  targetProvider: Provider;
  targetProviderConfig?: ProviderConfig;
  targetAdapter: TargetAdapter;
  providerPluginContext: ProviderPluginExecutionContext;
  model: string;
  initialStandardRequest: StandardRequest;
  initialStandardResponse: StandardResponse;
  initialUpstreamRequest: UpstreamRequest;
  initialUpstreamResponseBody: unknown;
  initialAttemptSequence: number;
  state: {
    attempts: ProviderAttemptFailure[];
    upstreamAttemptSequence: number;
  };
}): Promise<TransparentToolLoopResult> {
  const executionConfig = input.config.transparentToolExecution;
  let upstreamAttemptSequence = input.state.upstreamAttemptSequence;
  let workingRequest = input.initialStandardRequest;
  let lastResponse = input.initialStandardResponse;
  let lastUpstreamRequest = input.initialUpstreamRequest;
  let lastAttemptSequence = input.initialAttemptSequence;
  let lastUpstreamResponseBody = input.initialUpstreamResponseBody;
  let aggregatedUsage = lastResponse.usage;

  if (!executionConfig?.enabled) {
    return {
      ok: true,
      standardRequest: workingRequest,
      standardResponse: lastResponse,
      upstreamRequest: lastUpstreamRequest,
      attemptSequence: lastAttemptSequence,
      upstreamAttemptSequence
    };
  }

  let completedModelTurns = 1;
  let executedToolCalls = 0;

  while (true) {
    const functionCalls = extractFunctionCallsFromStandardResponse(lastResponse);
    if (functionCalls.length === 0) {
      return {
        ok: true,
        standardRequest: workingRequest,
        standardResponse: {
          ...lastResponse,
          usage: aggregatedUsage
        },
        upstreamRequest: lastUpstreamRequest,
        attemptSequence: lastAttemptSequence,
        upstreamAttemptSequence
      };
    }

    if (completedModelTurns >= executionConfig.maxTurns) {
      pushTransparentToolFailureAttempt(input, {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: 'tool_loop_limit',
        message: `Transparent tool execution exceeded maxTurns=${executionConfig.maxTurns}.`,
        status: 400,
        upstreamRequest: lastUpstreamRequest,
        upstreamResponseBody: lastUpstreamResponseBody
      }, lastAttemptSequence);
      return { ok: false, upstreamAttemptSequence };
    }

    const toolResolution = await resolveTransparentToolCalls(
      input.runtime,
      input.config,
      workingRequest,
      functionCalls
    );
    if (!toolResolution.ok) {
      pushTransparentToolFailureAttempt(input, {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: toolResolution.stage,
        message: toolResolution.message,
        status: toolResolution.status,
        upstreamRequest: lastUpstreamRequest,
        upstreamResponseBody: lastUpstreamResponseBody
      }, lastAttemptSequence);
      return { ok: false, upstreamAttemptSequence };
    }

    if ('returnToClient' in toolResolution) {
      return {
        ok: true,
        standardRequest: workingRequest,
        standardResponse: {
          ...lastResponse,
          usage: aggregatedUsage
        },
        upstreamRequest: lastUpstreamRequest,
        attemptSequence: lastAttemptSequence,
        upstreamAttemptSequence
      };
    }

    executedToolCalls += toolResolution.executableCalls.length;
    if (executedToolCalls > executionConfig.maxToolCalls) {
      pushTransparentToolFailureAttempt(input, {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: 'tool_loop_limit',
        message: `Transparent tool execution exceeded maxToolCalls=${executionConfig.maxToolCalls}.`,
        status: 400,
        upstreamRequest: lastUpstreamRequest,
        upstreamResponseBody: lastUpstreamResponseBody
      }, lastAttemptSequence);
      return { ok: false, upstreamAttemptSequence };
    }

    const toolResults = await executeTransparentToolCalls(
      input.runtime,
      input.source,
      input.targetProvider,
      input.targetProviderConfig,
      toolResolution.executableCalls,
      toolResolution.bindings
    );
    workingRequest = appendVirtualToolResultsToRequest(
      workingRequest,
      lastResponse,
      toolResolution.executableCalls,
      toolResults
    );

    const targetRequestResult = input.targetAdapter.buildRequestFromStandard({
      request: input.request,
      standardRequest: workingRequest,
      config: input.config,
      targetProviderConfig: input.targetProviderConfig
    });
    if (!targetRequestResult.ok) {
      input.state.attempts.push({
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: 'target_request_build',
        message: targetRequestResult.error,
        status: 400
      });
      return { ok: false, upstreamAttemptSequence };
    }

    const baseUpstreamRequest = applyProviderRequestOverrides(
      input.config,
      input.target,
      input.model,
      targetRequestResult.value
    );
    const upstreamDispatchResult = await dispatchUpstreamRequest(
      input.providerPluginContext,
      baseUpstreamRequest,
      input.config.upstreamTimeoutMs,
      workingRequest
    );
    if (!upstreamDispatchResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: upstreamDispatchResult.stage,
        message: upstreamDispatchResult.message,
        status: upstreamDispatchResult.status,
        details: upstreamDispatchResult.details,
        upstreamRequest: upstreamDispatchResult.upstreamRequest,
        upstreamResponseBody:
          upstreamDispatchResult.details !== undefined
            ? {
                error: {
                  message: upstreamDispatchResult.message,
                  details: upstreamDispatchResult.details
                }
              }
            : undefined
      };
      input.state.attempts.push(attempt);
      if (upstreamDispatchResult.upstreamRequest) {
        upstreamAttemptSequence += 1;
        publishFailedAttemptEventSafe(
          input.request,
          input.reply,
          input.config,
          input.providerPluginContext.sourceProvider,
          input.source.adapterKey,
          attempt,
          input.model,
          upstreamAttemptSequence,
          input.state.attempts,
          input.targetProviderConfig
        );
      }
      return { ok: false, upstreamAttemptSequence };
    }

    const { upstreamRequest, upstreamResponse } = upstreamDispatchResult;
    lastUpstreamRequest = upstreamRequest;
    upstreamAttemptSequence += 1;
    lastAttemptSequence = upstreamAttemptSequence;
    completedModelTurns += 1;

    const upstreamPayload =
      isEventStreamResponse(upstreamResponse) && input.targetProvider === 'openai'
        ? await collectOpenAINonStreamPayloadFromEventStream(upstreamResponse)
        : await safeReadUpstreamPayload(input.request, input.targetProvider, upstreamResponse);
    lastUpstreamResponseBody = upstreamPayload;
    if (!upstreamResponse.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: 'upstream_response',
        message: 'Upstream request failed.',
        status: upstreamResponse.status,
        details: upstreamPayload,
        upstreamRequest,
        upstreamResponseBody: upstreamPayload
      };
      input.state.attempts.push(attempt);
      publishFailedAttemptEventSafe(
        input.request,
        input.reply,
        input.config,
        input.providerPluginContext.sourceProvider,
        input.source.adapterKey,
        attempt,
        input.model,
        lastAttemptSequence,
        input.state.attempts,
        input.targetProviderConfig
      );
      return { ok: false, upstreamAttemptSequence };
    }

    const responsePluginResult = await applyProviderResponsePlugins(
      input.providerPluginContext,
      upstreamRequest,
      upstreamResponse,
      upstreamPayload,
      workingRequest
    );
    if (!responsePluginResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: responsePluginResult.stage,
        message: responsePluginResult.message,
        status: responsePluginResult.status,
        upstreamRequest,
        upstreamResponseBody: upstreamPayload
      };
      input.state.attempts.push(attempt);
      publishFailedAttemptEventSafe(
        input.request,
        input.reply,
        input.config,
        input.providerPluginContext.sourceProvider,
        input.source.adapterKey,
        attempt,
        input.model,
        lastAttemptSequence,
        input.state.attempts,
        input.targetProviderConfig
      );
      return { ok: false, upstreamAttemptSequence };
    }

    const transformedPayload = responsePluginResult.value;
    lastUpstreamResponseBody = transformedPayload;
    const standardPayload = await normalizeOpenAIPayloadForResponseParseRecovery(
      input.request,
      input.targetProvider,
      transformedPayload
    );
    const standardResponseResult = input.targetAdapter.toStandardResponse(standardPayload);
    if (!standardResponseResult.ok) {
      const attempt: ProviderAttemptFailure = {
        provider: input.targetProvider,
        providerName: input.targetProviderConfig?.name,
        stage: 'response_parse',
        message: standardResponseResult.error,
        status: 502,
        details: standardPayload,
        upstreamRequest,
        upstreamResponseBody: standardPayload
      };
      input.state.attempts.push(attempt);
      publishFailedAttemptEventSafe(
        input.request,
        input.reply,
        input.config,
        input.providerPluginContext.sourceProvider,
        input.source.adapterKey,
        attempt,
        input.model,
        lastAttemptSequence,
        input.state.attempts,
        input.targetProviderConfig
      );
      return { ok: false, upstreamAttemptSequence };
    }

    lastResponse = standardResponseResult.value;
    aggregatedUsage = mergeStandardUsage(aggregatedUsage, lastResponse.usage);
  }
}

function pushTransparentToolFailureAttempt(
  input: {
    request: FastifyRequest;
    reply: FastifyReply;
    source: GatewaySourceContext;
    config: GatewayConfig;
    providerPluginContext: ProviderPluginExecutionContext;
    targetProvider: Provider;
    targetProviderConfig?: ProviderConfig;
    model: string;
    state: {
      attempts: ProviderAttemptFailure[];
    };
  },
  attempt: ProviderAttemptFailure,
  attemptSequence: number
): void {
  input.state.attempts.push(attempt);
  if (!attempt.upstreamRequest) {
    return;
  }

  publishFailedAttemptEventSafe(
    input.request,
    input.reply,
    input.config,
    input.providerPluginContext.sourceProvider,
    input.source.adapterKey,
    attempt,
    input.model,
    attemptSequence,
    input.state.attempts,
    input.targetProviderConfig
  );
}

async function handleVirtualModelRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  source: GatewaySourceContext,
  config: GatewayConfig,
  runtime: GatewayRuntime,
  sourceAdapter: SourceAdapter,
  targetProviders: TargetProviderRoute[],
  adapterInput: {
    request: FastifyRequest;
    body: Record<string, unknown>;
    source: GatewaySourceContext;
    config: GatewayConfig;
  },
  isStreaming: boolean,
  virtualModel: VirtualModelResolution
) {
  const virtualTargetModel = resolveVirtualRuntimeModelName(virtualModel);
  const standardRequestResult = sourceAdapter.toStandardRequest(adapterInput);
  if (!standardRequestResult.ok) {
    return sendCountedBadRequest(
      request,
      reply,
      config,
      sourceAdapter.provider,
      source.adapterKey,
      targetProviders[0],
      virtualModel.targetModelReference.model,
      standardRequestResult.error
    );
  }

  const multimodalRewrite = virtualModel.profile.execution.matchMultimodal
    ? rewriteVirtualModelMultimodalInput(
        standardRequestResult.value,
        adapterInput.body,
        source.adapterKey
      )
    : { request: standardRequestResult.value, references: [] };
  const baseStandardRequest = multimodalRewrite.request;
  const mergedToolingResult = await buildVirtualTooling(
    baseStandardRequest.tools,
    virtualModel.profile,
    runtime
  );
  if (!mergedToolingResult.ok) {
    return sendCountedBadRequest(
      request,
      reply,
      config,
      sourceAdapter.provider,
      source.adapterKey,
      targetProviders[0],
      virtualTargetModel,
      mergedToolingResult.error
    );
  }

  if (
    virtualModel.profile.execution.mode === 'decorate_only' &&
    mergedToolingResult.hasInternalTools
  ) {
    return sendCountedBadRequest(
      request,
      reply,
      config,
      sourceAdapter.provider,
      source.adapterKey,
      targetProviders[0],
      virtualTargetModel,
      'decorate_only mode cannot be used with internal virtual model tools.'
    );
  }

  const rewrittenBaseRequest: StandardRequest = {
    ...baseStandardRequest,
    instructions: mergeVirtualModelInstructions(
      baseStandardRequest.instructions,
      virtualModel.profile.instructions
    ),
    tools: mergedToolingResult.tools,
    tool_choice: resolveVirtualToolChoice(
      baseStandardRequest.tool_choice,
      virtualModel.profile.toolChoice,
      mergedToolingResult.tools.length
    )
  };

  const attempts: ProviderAttemptFailure[] = [];
  let upstreamAttemptSequence = 0;
  let precheckApplied = false;
  const runPrecheckOnce = async (
    targetProvider: Provider,
    targetProviderConfig: ProviderConfig | undefined,
    model: string | undefined,
    standardRequest: StandardRequest
  ): Promise<GatewayPrecheckResult> => {
    if (precheckApplied) {
      return { ok: true };
    }

    const result = await evaluateGatewayPrecheck({
      request,
      config,
      targetProvider,
      targetProviderConfig,
      model,
      standardRequest,
      requestBody: adapterInput.body
    });
    if (result.ok) {
      precheckApplied = true;
    }
    return result;
  };

  for (const target of targetProviders) {
    const targetProvider = target.provider;
    const targetProviderConfig = resolveProviderConfig(config, target);
    const targetProviderLabel = formatTargetProviderLabel(target);
    const targetAdapter = runtime.targetAdapters.get(targetProvider);
    if (!targetAdapter) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'target_adapter_lookup',
        message: `Target adapter is not registered for provider: ${targetProvider}`,
        status: 400
      });
      continue;
    }

    const modelResult = validateModelForTarget(
      virtualTargetModel,
      target,
      config
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
    if (!model) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'model_resolution',
        message: `Model is required for virtual model profile ${virtualModel.profile.key} on target provider ${targetProviderLabel}.`,
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

    const providerPlugins = runtime.providerPlugins.resolve(targetProvider, targetProviderConfig?.name);
    const providerPluginContext: ProviderPluginExecutionContext = {
      request,
      config,
      source,
      sourceProvider: sourceAdapter.provider,
      sourceAdapterKey: source.adapterKey,
      targetProvider,
      targetProviderConfig,
      model,
      passthrough: false,
      streaming: false,
      plugins: providerPlugins
    };

    let workingRequest: StandardRequest = {
      ...rewrittenBaseRequest,
      model,
      stream: false
    };
    let aggregatedUsage: StandardUsage = {};
    let internalToolCalls = 0;
    let lastUpstreamRequest: UpstreamRequest | undefined;
    let lastResponse: StandardResponse | undefined;
    let lastAttemptSequence = 0;
    let loopExhausted = true;

    const precheckResult = await runPrecheckOnce(
      targetProvider,
      targetProviderConfig,
      model,
      workingRequest
    );
    if (!precheckResult.ok) {
      return sendGatewayPrecheckFailure(reply, precheckResult);
    }

    for (let turn = 0; turn < virtualModel.profile.execution.maxTurns; turn += 1) {
      const targetRequestResult = targetAdapter.buildRequestFromStandard({
        request,
        standardRequest: workingRequest,
        config,
        targetProviderConfig
      });
      if (!targetRequestResult.ok) {
        loopExhausted = false;
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'target_request_build',
          message: targetRequestResult.error,
          status: 400
        });
        break;
      }

      const baseUpstreamRequest = applyProviderRequestOverrides(
        config,
        target,
        model,
        targetRequestResult.value
      );
      const upstreamDispatchResult = await dispatchUpstreamRequest(
        providerPluginContext,
        baseUpstreamRequest,
        config.upstreamTimeoutMs,
        workingRequest
      );
      if (!upstreamDispatchResult.ok) {
        loopExhausted = false;
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: upstreamDispatchResult.stage,
          message: upstreamDispatchResult.message,
          status: upstreamDispatchResult.status,
          details: upstreamDispatchResult.details,
          upstreamRequest: upstreamDispatchResult.upstreamRequest,
          upstreamResponseBody:
            upstreamDispatchResult.details !== undefined
              ? {
                  error: {
                    message: upstreamDispatchResult.message,
                    details: upstreamDispatchResult.details
                  }
                }
              : undefined
        };
        attempts.push(attempt);
        if (upstreamDispatchResult.upstreamRequest) {
          upstreamAttemptSequence += 1;
          publishFailedAttemptEventSafe(
            request,
            reply,
            config,
            sourceAdapter.provider,
            source.adapterKey,
            attempt,
            model,
            upstreamAttemptSequence,
            attempts,
            targetProviderConfig
          );
        }
        break;
      }

      const { upstreamRequest, upstreamResponse } = upstreamDispatchResult;
      lastUpstreamRequest = upstreamRequest;
      upstreamAttemptSequence += 1;
      lastAttemptSequence = upstreamAttemptSequence;

      const upstreamPayload =
        isEventStreamResponse(upstreamResponse) && targetProvider === 'openai'
          ? await collectOpenAINonStreamPayloadFromEventStream(upstreamResponse)
          : await safeReadUpstreamPayload(request, targetProvider, upstreamResponse);
      if (!upstreamResponse.ok) {
        loopExhausted = false;
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'upstream_response',
          message: 'Upstream request failed.',
          status: upstreamResponse.status,
          details: upstreamPayload,
          upstreamRequest,
          upstreamResponseBody: upstreamPayload
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          lastAttemptSequence,
          attempts,
          targetProviderConfig
        );
        break;
      }

      const responsePluginResult = await applyProviderResponsePlugins(
        providerPluginContext,
        upstreamRequest,
        upstreamResponse,
        upstreamPayload,
        workingRequest
      );
      if (!responsePluginResult.ok) {
        loopExhausted = false;
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: responsePluginResult.stage,
          message: responsePluginResult.message,
          status: responsePluginResult.status,
          upstreamRequest,
          upstreamResponseBody: upstreamPayload
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          lastAttemptSequence,
          attempts,
          targetProviderConfig
        );
        break;
      }

      const transformedPayload = responsePluginResult.value;
      const standardPayload = await normalizeOpenAIPayloadForResponseParseRecovery(
        request,
        targetProvider,
        transformedPayload
      );
      const standardResponseResult = targetAdapter.toStandardResponse(standardPayload);
      if (!standardResponseResult.ok) {
        loopExhausted = false;
        const attempt: ProviderAttemptFailure = {
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'response_parse',
          message: standardResponseResult.error,
          status: 502,
          details: standardPayload,
          upstreamRequest,
          upstreamResponseBody: standardPayload
        };
        attempts.push(attempt);
        publishFailedAttemptEventSafe(
          request,
          reply,
          config,
          sourceAdapter.provider,
          source.adapterKey,
          attempt,
          model,
          lastAttemptSequence,
          attempts,
          targetProviderConfig
        );
        break;
      }

      lastResponse = standardResponseResult.value;
      aggregatedUsage = mergeStandardUsage(aggregatedUsage, lastResponse.usage);
      const functionCalls = extractFunctionCallsFromStandardResponse(lastResponse);
      const callPartition = partitionVirtualFunctionCalls(
        functionCalls,
        mergedToolingResult.toolOwners
      );

      if (callPartition.internal.length === 0) {
        return sendVirtualModelResponse(
          reply,
          request,
          source,
          config,
          sourceAdapter,
          targetProvider,
          targetProviderConfig,
          attempts.length,
          lastAttemptSequence,
        filterInternalToolCallsFromStandardResponse(
          lastResponse,
          mergedToolingResult.toolOwners,
          aggregatedUsage
        ),
        isStreaming,
        model,
        rewrittenBaseRequest,
        lastUpstreamRequest
      );
      }

      if (callPartition.client.length > 0 || callPartition.unknown.length > 0) {
        return sendVirtualModelResponse(
          reply,
          request,
          source,
          config,
          sourceAdapter,
          targetProvider,
          targetProviderConfig,
          attempts.length,
          lastAttemptSequence,
        filterInternalToolCallsFromStandardResponse(
          lastResponse,
          mergedToolingResult.toolOwners,
          aggregatedUsage
        ),
        isStreaming,
        model,
        rewrittenBaseRequest,
        lastUpstreamRequest
      );
      }

      internalToolCalls += callPartition.internal.length;
      if (internalToolCalls > virtualModel.profile.execution.maxToolCalls) {
        loopExhausted = false;
        attempts.push({
          provider: targetProvider,
          providerName: targetProviderConfig?.name,
          stage: 'tool_loop_limit',
          message: `Virtual model tool loop exceeded maxToolCalls=${virtualModel.profile.execution.maxToolCalls}.`,
          status: 400,
          upstreamRequest,
          upstreamResponseBody: transformedPayload
        });
        break;
      }

      const toolResults = await executeInternalVirtualToolCalls(
        runtime,
        virtualModel.profile,
        callPartition.internal,
        mergedToolingResult.toolOwners,
        multimodalRewrite.references
      );
      workingRequest = appendVirtualToolResultsToRequest(
        workingRequest,
        lastResponse,
        callPartition.internal,
        toolResults
      );
    }

    if (loopExhausted) {
      attempts.push({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        stage: 'tool_loop_limit',
        message: `Virtual model tool loop exceeded maxTurns=${virtualModel.profile.execution.maxTurns}.`,
        status: 400,
        upstreamRequest: lastUpstreamRequest,
        upstreamResponseBody: lastResponse
      });
    }
  }

  const fallbackFailure = buildFallbackErrorPayload(targetProviders, attempts);
  reply.code(fallbackFailure.status);
  const lastTraceableAttempt = [...attempts]
    .reverse()
    .find(
      (attempt) =>
        attempt.upstreamRequest !== undefined ||
        attempt.upstreamResponseBody !== undefined,
    );
  const failureTarget =
    attempts[attempts.length - 1]
      ? {
          provider: attempts[attempts.length - 1]!.provider,
          providerConfig: attempts[attempts.length - 1]!.providerName
            ? findProviderConfigByName(
                config.providers,
                attempts[attempts.length - 1]!.providerName!,
              ) ||
              findProviderConfigByType(
                config.providers,
                attempts[attempts.length - 1]!.provider,
              )
            : findProviderConfigByType(
                config.providers,
                attempts[attempts.length - 1]!.provider,
              ),
        }
      : targetProviders[targetProviders.length - 1];
  if (failureTarget) {
    publishRequestFailureEventSafe(
      request,
      reply,
      config,
      sourceAdapter.provider,
      source.adapterKey,
      failureTarget.provider,
      virtualTargetModel,
      attempts.length,
      attempts,
      attempts[attempts.length - 1]?.message || fallbackFailure.payload.error.message,
      failureTarget.providerConfig,
      buildGatewayBillingTraceSnapshot(request, reply, {
        responseBody: fallbackFailure.payload,
      }),
      resolveAttemptRawTraceCapture(lastTraceableAttempt),
    );
  }
  return reply.send(fallbackFailure.payload);
}

function resolveVirtualRuntimeModelName(
  virtualModel: VirtualModelResolution
): string | undefined {
  if (virtualModel.profile.baseModel?.fixedModel) {
    return virtualModel.targetModelReference.model;
  }

  const requestedModel = virtualModel.requestedModelReference.model;
  if (
    virtualModel.matchedBy === 'suffix' &&
    requestedModel.endsWith(virtualModel.matchedToken) &&
    requestedModel.length > virtualModel.matchedToken.length
  ) {
    return requestedModel.slice(0, -virtualModel.matchedToken.length);
  }

  if (
    virtualModel.matchedBy === 'prefix' &&
    requestedModel.startsWith(virtualModel.matchedToken) &&
    requestedModel.length > virtualModel.matchedToken.length
  ) {
    return requestedModel.slice(virtualModel.matchedToken.length);
  }

  if (virtualModel.profile.baseModel?.mode === 'request') {
    return requestedModel;
  }

  return virtualModel.targetModelReference.model;
}

async function buildVirtualTooling(
  clientTools: unknown[] | undefined,
  profile: VirtualModelProfileConfig,
  runtime: GatewayRuntime
): Promise<
  | {
      ok: true;
      tools: unknown[];
      toolOwners: Map<string, VirtualToolOwner>;
      hasInternalTools: boolean;
    }
  | { ok: false; error: string }
> {
  const normalizedClientTools = Array.isArray(clientTools) ? clientTools : [];
  if (profile.execution.clientToolsPolicy === 'deny' && normalizedClientTools.length > 0) {
    return {
      ok: false,
      error: `Virtual model profile ${profile.key} does not allow client supplied tools.`
    };
  }

  const toolOwners = new Map<string, VirtualToolOwner>();
  const mergedTools: unknown[] = [];
  let hasInternalTools = false;

  let toolCatalog = new Map<string, { description?: string; inputSchema?: Record<string, unknown> }>();
  let toolDefinitions: AgentToolDefinition[] = [];
  if (profile.tools.length > 0) {
    if (!runtime.toolProvider) {
      return {
        ok: false,
        error: 'Virtual model tools require MCP tool provider to be configured on gateway.'
      };
    }

    toolDefinitions = await runtime.toolProvider.listDefinitions();
    toolCatalog = new Map(
      toolDefinitions.map((tool) => [
        tool.name,
        {
          description: tool.description,
          inputSchema: tool.inputSchema
        }
      ])
    );
  }

  for (const tool of profile.tools) {
    const resolvedTool = resolveVirtualProfileToolDefinition(tool.name, toolDefinitions);
    if (!resolvedTool) {
      return {
        ok: false,
        error: `Virtual model tool is not available from MCP provider: ${tool.name}`
      };
    }
    if ('error' in resolvedTool) {
      return {
        ok: false,
        error: resolvedTool.error
      };
    }

    const catalogEntry = toolCatalog.get(resolvedTool.runtimeToolName);
    if (!catalogEntry) {
      return {
        ok: false,
        error: `Virtual model tool is not available from MCP provider: ${tool.name}`
      };
    }

    if (toolOwners.has(tool.name)) {
      return {
        ok: false,
        error: `Duplicate virtual model tool name: ${tool.name}`
      };
    }

    toolOwners.set(tool.name, {
      visibility: tool.visibility,
      source: 'profile',
      runtimeToolName: resolvedTool.runtimeToolName
    });
    if (tool.visibility === 'internal') {
      hasInternalTools = true;
    }
    mergedTools.push({
      type: 'function',
      name: tool.name,
      description: tool.description || catalogEntry.description,
      input_schema: tool.inputSchema || catalogEntry.inputSchema || { type: 'object', properties: {} }
    });
  }

  for (const tool of normalizedClientTools) {
    const toolName = extractStandardToolName(tool);
    if (!toolName) {
      mergedTools.push(tool);
      continue;
    }

    if (toolOwners.has(toolName)) {
      return {
        ok: false,
        error: `Client tool "${toolName}" conflicts with virtual model tool.`
      };
    }

    toolOwners.set(toolName, {
      visibility: 'client',
      source: 'client'
    });
    mergedTools.push(tool);
  }

  return {
    ok: true,
    tools: mergedTools,
    toolOwners,
    hasInternalTools
  };
}

async function executeInternalVirtualToolCalls(
  runtime: GatewayRuntime,
  profile: VirtualModelProfileConfig,
  toolCalls: StandardResponseFunctionCall[],
  toolOwners: Map<string, VirtualToolOwner>,
  multimodalReferences: VirtualMultimodalReference[] = []
): Promise<StandardRequestInputContent[]> {
  if (!runtime.toolProvider) {
    throw new Error(`Virtual model profile ${profile.key} requires gateway tool provider.`);
  }

  const results: StandardRequestInputContent[] = [];
  for (const toolCall of toolCalls) {
    const owner = toolOwners.get(toolCall.name);
    if (!owner || owner.visibility !== 'internal') {
      continue;
    }

    const args = parseVirtualToolArguments(toolCall.arguments);
    const runtimeToolName = owner.runtimeToolName || toolCall.name;
    const timestamp = new Date().toISOString();
    try {
      const output = await runtime.toolProvider.execute(runtimeToolName, {
        args,
        mcpMeta: buildVirtualMultimodalMcpMeta(multimodalReferences),
        session: {
          sessionId: `virtual-model:${profile.key}`,
          agentId: `virtual-model:${profile.id}`,
          systemPrompt: profile.instructions?.replace || '',
          model: profile.baseModel?.fixedModel,
          allowedTools: [runtimeToolName],
          allowedToolsConfigured: true,
          memoryRefs: [],
          messages: [],
          pendingToolCalls: {},
          taskState: {
            id: `virtual-model:${profile.key}`,
            goal: profile.displayName,
            activeStep: null,
            constraints: [],
            done: [],
            todo: [],
            status: 'running'
          },
          transcriptWindow: { items: [] },
          guards: {
            doNotRepeat: [],
            doNotForget: [],
            doNotViolate: []
          },
          lastEventOffset: 0,
          updatedAt: timestamp
        } as any,
        event: {
          id: randomUUID(),
          type: 'TOOL_CALL_REQUESTED',
          sessionId: `virtual-model:${profile.key}`,
          timestamp,
          correlationId: randomUUID(),
          payload: {
            toolCallId: toolCall.call_id,
            toolName: toolCall.name,
            arguments: args
          }
        } as any
      });

      results.push({
        type: 'tool_result',
        tool_use_id: toolCall.call_id,
        content: stringifyVirtualToolOutput(output),
        result_format:
          profile.execution.matchWebSearch && isVirtualWebSearchToolName(toolCall.name)
            ? 'web_search'
            : 'function'
      });
    } catch (error) {
      results.push({
        type: 'tool_result',
        tool_use_id: toolCall.call_id,
        content: describeVirtualToolError(error),
        is_error: true,
        result_format:
          profile.execution.matchWebSearch && isVirtualWebSearchToolName(toolCall.name)
            ? 'web_search'
            : 'function'
      });
    }
  }

  return results;
}

export function rewriteVirtualModelMultimodalInput(
  request: StandardRequest,
  body: Record<string, unknown>,
  adapterKey: string
): VirtualMultimodalRewrite {
  const rewriteState = createVirtualMultimodalRewriteState();
  const multimodalMessages = extractVirtualMultimodalMessages(body, adapterKey, rewriteState);
  multimodalMessages.push(
    ...extractVirtualStandardToolResultMultimodalMessages(request, rewriteState)
  );
  const requestWithMediaRefs = rewriteVirtualStandardRequestMediaReferences(
    request,
    rewriteState.references
  );
  if (multimodalMessages.length === 0) {
    return {
      request: requestWithMediaRefs,
      references: rewriteState.references
    };
  }

  const messages = collectStandardInputMessages(requestWithMediaRefs.input);
  const descriptionsByMessage = groupVirtualMultimodalDescriptions(multimodalMessages);
  if (messages.length === 0) {
    return {
      request: {
        ...requestWithMediaRefs,
        input: multimodalMessages.map((text) => ({
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: text.text }]
        }))
      },
      references: rewriteState.references
    };
  }

  const rewritten = messages.map((message, index) => {
    const extraText = descriptionsByMessage.get(index);
    if (!extraText) {
      return message;
    }

    return {
      ...message,
      content: [
        ...message.content,
        {
          type: 'input_text' as const,
          text: extraText
        }
      ]
    };
  });

  for (const [index, text] of [...descriptionsByMessage.entries()].sort((a, b) => a[0] - b[0])) {
    if (index < messages.length) {
      continue;
    }

    rewritten.push({
      type: 'message',
      role: 'user',
      content: [
        {
          type: 'input_text',
          text
        }
      ]
    });
  }

  return {
    request: {
      ...requestWithMediaRefs,
      input: rewritten
    },
    references: rewriteState.references
  };
}

function extractVirtualMultimodalMessages(
  body: Record<string, unknown>,
  adapterKey: string,
  rewriteState: VirtualMultimodalRewriteState
): VirtualMultimodalMessageDescription[] {
  const descriptions: VirtualMultimodalMessageDescription[] = [];
  if (adapterKey === 'openai_chat' || adapterKey === 'openai_chat_completions') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    let messageIndex = 0;
    for (const message of messages) {
      if (!isObject(message)) {
        continue;
      }

      if (normalizeMessageRole(message.role) === 'system') {
        continue;
      }

      addVirtualMultimodalDescription(descriptions, messageIndex, message.content, rewriteState);
      messageIndex += 1;
    }
    return descriptions;
  }

  if (adapterKey === 'openai_responses') {
    const input = body.input;
    if (typeof input === 'string') {
      return [];
    }
    const items = Array.isArray(input) ? input : [input];
    let messageIndex = 0;
    for (const item of items) {
      if (!isObject(item)) {
        continue;
      }

      addVirtualMultimodalDescription(
        descriptions,
        messageIndex,
        getOpenAIResponsesMultimodalScanValue(item),
        rewriteState
      );
      messageIndex += 1;
    }
    return descriptions;
  }

  if (adapterKey === 'anthropic_messages') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    let messageIndex = 0;
    for (const message of messages) {
      if (!isObject(message)) {
        continue;
      }

      addVirtualMultimodalDescription(descriptions, messageIndex, message.content, rewriteState);
      messageIndex += 1;
    }
    return descriptions;
  }

  if (
    adapterKey === 'gemini_generate' ||
    adapterKey === 'gemini_stream' ||
    adapterKey === 'gemini_generate_content' ||
    adapterKey === 'gemini_stream_generate_content'
  ) {
    const contents = Array.isArray(body.contents) ? body.contents : [];
    let messageIndex = 0;
    for (const content of contents) {
      if (!isObject(content)) {
        continue;
      }

      addVirtualMultimodalDescription(descriptions, messageIndex, content.parts, rewriteState);
      messageIndex += 1;
    }
    return descriptions;
  }

  return descriptions;
}

interface VirtualMultimodalRewriteState {
  references: VirtualMultimodalReference[];
}

interface VirtualMultimodalMessageDescription {
  messageIndex: number;
  text: string;
}

function createVirtualMultimodalRewriteState(): VirtualMultimodalRewriteState {
  return { references: [] };
}

function describeVirtualMultimodalBlocks(
  value: unknown,
  rewriteState: VirtualMultimodalRewriteState
): string {
  const descriptions: string[] = [];
  collectVirtualMultimodalBlockDescriptions(value, rewriteState, descriptions);
  const uniqueDescriptions = [...new Set(descriptions)];

  return uniqueDescriptions.length > 0
    ? `Multimodal inputs available to tools. Use the media_ref value when calling tools:\n${uniqueDescriptions.join('\n')}`
    : '';
}

function addVirtualMultimodalDescription(
  descriptions: VirtualMultimodalMessageDescription[],
  messageIndex: number,
  value: unknown,
  rewriteState: VirtualMultimodalRewriteState
) {
  const text = describeVirtualMultimodalBlocks(value, rewriteState);
  if (text) {
    descriptions.push({ messageIndex, text });
  }
}

function groupVirtualMultimodalDescriptions(
  descriptions: VirtualMultimodalMessageDescription[]
): Map<number, string> {
  const grouped = new Map<number, string[]>();
  for (const description of descriptions) {
    const group = grouped.get(description.messageIndex) || [];
    if (!group.includes(description.text)) {
      group.push(description.text);
    }
    grouped.set(description.messageIndex, group);
  }

  return new Map([...grouped.entries()].map(([index, values]) => [index, values.join('\n')]));
}

function getOpenAIResponsesMultimodalScanValue(item: Record<string, unknown>): unknown {
  const type = asString(item.type);
  if (type === 'function_call_output') {
    return item.output ?? item.content ?? item.result;
  }

  if (type === 'message' || item.role !== undefined || item.content !== undefined) {
    return item.content;
  }

  return item.content ?? item;
}

function extractVirtualStandardToolResultMultimodalMessages(
  request: StandardRequest,
  rewriteState: VirtualMultimodalRewriteState
): VirtualMultimodalMessageDescription[] {
  const descriptions: VirtualMultimodalMessageDescription[] = [];
  const messages = collectStandardInputMessages(request.input);
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex += 1) {
    const message = messages[messageIndex]!;
    for (const item of message.content) {
      if (item.type !== 'tool_result') {
        continue;
      }

      addVirtualMultimodalDescription(
        descriptions,
        messageIndex,
        item.content,
        rewriteState
      );
    }
  }
  return descriptions;
}

function collectVirtualMultimodalBlockDescriptions(
  value: unknown,
  rewriteState: VirtualMultimodalRewriteState,
  descriptions: string[]
) {
  if (typeof value === 'string') {
    const parsed = parseVirtualMultimodalJsonString(value);
    if (parsed !== undefined) {
      collectVirtualMultimodalBlockDescriptions(parsed, rewriteState, descriptions);
    }
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectVirtualMultimodalBlockDescriptions(item, rewriteState, descriptions);
    }
    return;
  }

  if (!isObject(value)) {
    return;
  }

  const description = describeVirtualMultimodalBlock(value, rewriteState);
  if (description) {
    descriptions.push(description);
    return;
  }

  for (const child of Object.values(value)) {
    collectVirtualMultimodalBlockDescriptions(child, rewriteState, descriptions);
  }
}

function parseVirtualMultimodalJsonString(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed || (trimmed[0] !== '{' && trimmed[0] !== '[')) {
    return undefined;
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return undefined;
  }
}

function rewriteVirtualStandardRequestMediaReferences(
  request: StandardRequest,
  references: VirtualMultimodalReference[]
): StandardRequest {
  if (references.length === 0) {
    return request;
  }

  if (typeof request.input === 'string') {
    return {
      ...request,
      input: replaceVirtualMultimodalReferenceString(request.input, references)
    };
  }

  return {
    ...request,
    input: request.input.map((message) => ({
      ...message,
      content: message.content.map((item) => rewriteVirtualStandardInputContentMediaReferences(item, references))
    }))
  };
}

function rewriteVirtualStandardInputContentMediaReferences(
  item: StandardRequestInputContent,
  references: VirtualMultimodalReference[]
): StandardRequestInputContent {
  if (item.type === 'input_text') {
    return {
      ...item,
      text: replaceVirtualMultimodalReferenceString(item.text, references)
    };
  }

  if (item.type === 'tool_result') {
    return {
      ...item,
      content: replaceVirtualMultimodalReferenceString(item.content, references)
    };
  }

  if (item.type === 'tool_use') {
    return {
      ...item,
      input: replaceVirtualMultimodalReferenceValue(item.input, references)
    };
  }

  return item;
}

function replaceVirtualMultimodalReferenceValue(
  value: unknown,
  references: VirtualMultimodalReference[]
): unknown {
  if (typeof value === 'string') {
    return replaceVirtualMultimodalReferenceString(value, references);
  }

  if (Array.isArray(value)) {
    return value.map((item) => replaceVirtualMultimodalReferenceValue(item, references));
  }

  if (!isObject(value)) {
    return value;
  }

  const rewritten: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    rewritten[key] = replaceVirtualMultimodalReferenceValue(child, references);
  }
  return rewritten;
}

function replaceVirtualMultimodalReferenceString(
  value: string,
  references: VirtualMultimodalReference[]
): string {
  let rewritten = value;
  const sortedReferences = [...references].sort((a, b) => b.value.length - a.value.length);
  for (const reference of sortedReferences) {
    rewritten = rewritten.split(reference.value).join(`[media_ref:${reference.id}]`);
  }
  return rewritten;
}

function describeVirtualMultimodalBlock(
  block: unknown,
  rewriteState: VirtualMultimodalRewriteState
): string | undefined {
  if (!isObject(block)) {
    return undefined;
  }

  const type = asString(block.type);
  const imageUrl = readVirtualNestedString(block.image_url, 'url') || asString(block.image_url);
  if (type === 'image_url' || imageUrl) {
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'image',
        value: imageUrl,
        mimeType: readVirtualNestedString(block.image_url, 'mime_type')
      })
    ) || '- image: inline image';
  }

  if (type === 'input_image') {
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'image',
        value: asString(block.image_url) || asString(block.file_data)
      })
    ) || `- image: ${asString(block.file_id) || 'inline image'}`;
  }

  if (type === 'input_file' || type === 'file') {
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'file',
        value:
          asString(block.file_data) ||
          asString(block.file_url) ||
          readVirtualNestedString(block.file, 'file_data') ||
          readVirtualNestedString(block.file, 'file_url'),
        filename: asString(block.filename) || readVirtualNestedString(block.file, 'filename')
      })
    ) || `- file: ${asString(block.filename) || asString(block.file_id) || 'attached file'}`;
  }

  if (type === 'image') {
    const source = isObject(block.source) ? block.source : undefined;
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'image',
        value: asString(source?.url) || asString(source?.data),
        mimeType: asString(source?.media_type)
      })
    ) || `- image: ${asString(source?.media_type) || 'inline image'}`;
  }

  if (type === 'document') {
    const source = isObject(block.source) ? block.source : undefined;
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'document',
        value: asString(source?.url) || asString(source?.data),
        mimeType: asString(source?.media_type),
        filename: asString(block.title) || asString(block.name)
      })
    ) || `- document: ${asString(block.title) || asString(block.name) || 'attached document'}`;
  }

  const inlineData = isObject(block.inlineData)
    ? block.inlineData
    : isObject(block.inline_data)
      ? block.inline_data
      : undefined;
  if (inlineData) {
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'media',
        value: asString(inlineData.data),
        mimeType: asString(inlineData.mimeType) || asString(inlineData.mime_type)
      })
    ) || `- inline media: ${asString(inlineData.mimeType) || asString(inlineData.mime_type) || 'unknown mime type'}`;
  }

  const fileData = isObject(block.fileData)
    ? block.fileData
    : isObject(block.file_data)
      ? block.file_data
      : undefined;
  if (fileData) {
    return describeVirtualMultimodalReference(
      registerVirtualMultimodalReference(rewriteState, {
        kind: 'file',
        value: asString(fileData.fileUri) || asString(fileData.file_uri),
        mimeType: asString(fileData.mimeType) || asString(fileData.mime_type)
      })
    ) || '- file: attached file';
  }

  return undefined;
}

function registerVirtualMultimodalReference(
  rewriteState: VirtualMultimodalRewriteState,
  input: {
    kind: VirtualMultimodalReference['kind'];
    value?: string;
    mimeType?: string;
    filename?: string;
  }
): VirtualMultimodalReference | undefined {
  const value = input.value?.trim();
  if (!value || !isVirtualMultimodalReferenceValue(value)) {
    return undefined;
  }

  const existing = rewriteState.references.find((reference) => reference.value === value);
  if (existing) {
    return existing;
  }

  const reference: VirtualMultimodalReference = {
    id: `mm_${randomUUID().replace(/-/g, '').slice(0, 10)}`,
    kind: input.kind,
    sourceType: isVirtualUrl(value) ? 'url' : 'base64',
    value
  };
  if (input.mimeType) {
    reference.mimeType = input.mimeType;
  }
  if (input.filename) {
    reference.filename = input.filename;
  }
  rewriteState.references.push(reference);
  return reference;
}

function describeVirtualMultimodalReference(
  reference: VirtualMultimodalReference | undefined
): string | undefined {
  if (!reference) {
    return undefined;
  }

  const details = [
    reference.sourceType,
    reference.mimeType,
    reference.filename ? `name=${reference.filename}` : undefined
  ].filter(Boolean);
  return `- ${reference.kind}: [media_ref:${reference.id}] (${details.join(', ')})`;
}

function isVirtualMultimodalReferenceValue(value: string): boolean {
  return isVirtualUrl(value) || isVirtualDataUrl(value) || isLikelyBase64Payload(value);
}

function isVirtualUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function isVirtualDataUrl(value: string): boolean {
  return /^data:[^,]+;base64,/i.test(value);
}

function isLikelyBase64Payload(value: string): boolean {
  if (value.length < 32 || /\s/.test(value)) {
    return false;
  }

  return /^[A-Za-z0-9+/]+={0,2}$/.test(value);
}

export function hydrateVirtualMultimodalReferences(
  value: unknown,
  references: VirtualMultimodalReference[]
): unknown {
  if (references.length === 0) {
    return value;
  }

  if (typeof value === 'string') {
    return hydrateVirtualMultimodalReferenceString(value, references);
  }

  if (Array.isArray(value)) {
    return value.map((item) => hydrateVirtualMultimodalReferences(item, references));
  }

  if (!isObject(value)) {
    return value;
  }

  const hydrated: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    hydrated[key] = hydrateVirtualMultimodalReferences(child, references);
  }
  return hydrated;
}

function buildVirtualMultimodalMcpMeta(
  references: VirtualMultimodalReference[]
): Record<string, unknown> | undefined {
  if (references.length === 0) {
    return undefined;
  }

  return {
    virtualMultimodalReferences: references
  };
}

function hydrateVirtualMultimodalReferenceString(
  value: string,
  references: VirtualMultimodalReference[]
): string {
  const trimmed = value.trim();
  for (const reference of references) {
    if (
      trimmed === reference.id ||
      trimmed === `media_ref:${reference.id}` ||
      trimmed === `[media_ref:${reference.id}]`
    ) {
      return reference.value;
    }
  }

  let hydrated = value;
  for (const reference of references) {
    hydrated = hydrated
      .split(`[media_ref:${reference.id}]`)
      .join(reference.value)
      .split(`media_ref:${reference.id}`)
      .join(reference.value);
  }
  return hydrated;
}

function readVirtualNestedString(value: unknown, key: string): string | undefined {
  return isObject(value) ? asString(value[key]) : undefined;
}

function isVirtualWebSearchToolName(name: string): boolean {
  const normalized = name.toLowerCase().replace(/[-.]/g, '_');
  return normalized === 'web_search' || normalized.endsWith('_web_search') || normalized.includes('search_web');
}

function appendVirtualToolResultsToRequest(
  request: StandardRequest,
  response: StandardResponse,
  toolCalls: StandardResponseFunctionCall[],
  toolResults: StandardRequestInputContent[]
): StandardRequest {
  const messages = collectStandardInputMessages(request.input);
  const assistantContent: StandardRequestInputContent[] = [];

  if (response.output_text) {
    assistantContent.push({
      type: 'input_text',
      text: response.output_text
    });
  }

  for (const toolCall of toolCalls) {
    assistantContent.push({
      type: 'tool_use',
      id: toolCall.call_id,
      name: toolCall.name,
      input: parseVirtualToolArguments(toolCall.arguments)
    });
  }

  return {
    ...request,
    input: [
      ...messages,
      {
        type: 'message',
        role: 'assistant',
        content: assistantContent
      },
      {
        type: 'message',
        role: 'user',
        content: toolResults
      }
    ]
  };
}

function sendVirtualModelResponse(
  reply: FastifyReply,
  request: FastifyRequest,
  source: GatewaySourceContext,
  config: GatewayConfig,
  sourceAdapter: SourceAdapter,
  targetProvider: Provider,
  targetProviderConfig: ProviderConfig | undefined,
  fallbackAttempts: number,
  attemptSequence: number,
  standardResponse: StandardResponse,
  streaming: boolean,
  model: string,
  standardRequest?: StandardRequest,
  upstreamRequest?: UpstreamRequest
) {
  attachTargetRoutingHeaders(reply, targetProvider, targetProviderConfig?.name, fallbackAttempts);

  const sourceStandardResponse = prepareStandardResponseForSource(source, standardResponse, standardRequest);

  if (!streaming) {
    const sourcePayload = sourceAdapter.fromStandardResponse({
      request,
      response: sourceStandardResponse,
      standardRequest,
      source,
      config
    });
    attachBillingHeaders(
      request,
      reply,
      sourceAdapter.provider,
      source.adapterKey,
      targetProvider,
      standardResponse.usage,
      config,
      fallbackAttempts,
      attemptSequence,
      resolveBillingModel(standardResponse.model, model),
      targetProviderConfig,
      buildGatewayBillingTraceSnapshot(request, reply, {
        responseBody: sourcePayload,
        standardResponse: sourceStandardResponse
      }),
      {
        upstreamRequest,
        upstreamResponseBody: sourcePayload
      }
    );
    return reply.code(200).send(sourcePayload);
  }

  attachBillingHeaders(
    request,
    reply,
    sourceAdapter.provider,
    source.adapterKey,
    targetProvider,
    standardResponse.usage,
    config,
    fallbackAttempts,
    attemptSequence,
    resolveBillingModel(standardResponse.model, model),
    targetProviderConfig,
    buildGatewayBillingTraceSnapshot(request, reply, {
      responseBody: sourceStandardResponse,
      standardResponse: sourceStandardResponse
    }),
    {
      upstreamRequest,
      upstreamResponseBody: sourceStandardResponse
    }
  );
  return relayConvertedStreamFromStandardResponse(reply, source, sourceStandardResponse);
}

function prepareStandardResponseForSource(
  source: GatewaySourceContext,
  standardResponse: StandardResponse,
  standardRequest?: StandardRequest
): StandardResponse {
  if (source.adapterKey !== 'openai_responses') {
    return standardResponse;
  }

  return addNamespaceFieldsToStandardResponse(standardResponse, standardRequest?.tools);
}

function shouldUseTransparentToolExecutionPath(
  config: GatewayConfig,
  streaming: boolean
): boolean {
  return config.transparentToolExecution?.enabled === true && !streaming;
}

function resolveVirtualModelRequest(
  config: GatewayConfig,
  requestedModel: string
): VirtualModelResolution | undefined {
  const profiles = config.virtualModelProfiles || [];
  const requestRef = parseModelReference(requestedModel, config.providers);
  if (!requestRef) {
    return undefined;
  }

  const exactMatches = profiles.flatMap((profile) =>
    profile.enabled !== false
      ? profile.match.exactAliases
          .filter((alias) => alias === requestRef.raw || alias === requestRef.model)
          .map((alias) => ({ profile, matchedBy: 'exact' as const, matchedToken: alias }))
      : []
  );
  if (exactMatches.length > 0) {
    const selected = exactMatches[0];
    const targetModelSelector = resolveVirtualTargetModelSelector(requestRef, selected);
    if (!targetModelSelector) {
      return undefined;
    }
    return {
      profile: selected.profile,
      requestedModel,
      requestedModelReference: requestRef,
      matchedBy: selected.matchedBy,
      matchedToken: selected.matchedToken,
      targetModelSelector,
      targetModelReference: parseModelReference(targetModelSelector, config.providers) || {
        raw: targetModelSelector,
        model: targetModelSelector
      }
    };
  }

  const suffixMatches = profiles.flatMap((profile) =>
    profile.enabled !== false
      ? profile.match.suffixes
          .filter(
            (suffix) =>
              requestRef.model.endsWith(suffix) && requestRef.model.length > suffix.length
          )
          .map((suffix) => ({ profile, matchedBy: 'suffix' as const, matchedToken: suffix }))
      : []
  );
  suffixMatches.sort((a, b) => b.matchedToken.length - a.matchedToken.length);
  if (suffixMatches.length > 0) {
    const selected = suffixMatches[0];
    const targetModelSelector = resolveVirtualTargetModelSelector(requestRef, selected);
    if (!targetModelSelector) {
      return undefined;
    }
    return {
      profile: selected.profile,
      requestedModel,
      requestedModelReference: requestRef,
      matchedBy: selected.matchedBy,
      matchedToken: selected.matchedToken,
      targetModelSelector,
      targetModelReference: parseModelReference(targetModelSelector, config.providers) || {
        raw: targetModelSelector,
        model: targetModelSelector
      }
    };
  }

  const prefixMatches = profiles.flatMap((profile) =>
    profile.enabled !== false
      ? profile.match.prefixes
          .filter(
            (prefix) =>
              requestRef.model.startsWith(prefix) && requestRef.model.length > prefix.length
          )
          .map((prefix) => ({ profile, matchedBy: 'prefix' as const, matchedToken: prefix }))
      : []
  );
  prefixMatches.sort((a, b) => b.matchedToken.length - a.matchedToken.length);
  if (prefixMatches.length === 0) {
    return undefined;
  }

  const selected = prefixMatches[0];
  const targetModelSelector = resolveVirtualTargetModelSelector(requestRef, selected);
  if (!targetModelSelector) {
    return undefined;
  }

  return {
    profile: selected.profile,
    requestedModel,
    requestedModelReference: requestRef,
    matchedBy: selected.matchedBy,
    matchedToken: selected.matchedToken,
    targetModelSelector,
    targetModelReference: parseModelReference(targetModelSelector, config.providers) || {
      raw: targetModelSelector,
      model: targetModelSelector
    }
  };
}

function resolveVirtualTargetModelSelector(
  requestRef: ParsedModelReference,
  match: {
    profile: VirtualModelProfileConfig;
    matchedBy: 'exact' | 'prefix' | 'suffix';
    matchedToken: string;
  }
): string | undefined {
  if (match.profile.baseModel?.mode === 'fixed' || match.profile.baseModel?.fixedModel) {
    return match.profile.baseModel?.fixedModel;
  }

  if (match.matchedBy === 'suffix') {
    return composeVirtualTargetModelSelector(
      requestRef,
      requestRef.model.slice(0, -match.matchedToken.length)
    );
  }

  if (match.matchedBy === 'prefix') {
    return composeVirtualTargetModelSelector(
      requestRef,
      requestRef.model.slice(match.matchedToken.length)
    );
  }

  if (match.profile.baseModel?.mode === 'request') {
    return requestRef.raw;
  }

  return undefined;
}

function composeVirtualTargetModelSelector(
  requestRef: ParsedModelReference,
  nextModel: string
): string | undefined {
  if (!nextModel) {
    return undefined;
  }

  if (requestRef.providerConfig?.name) {
    return `${requestRef.providerConfig.name}/${nextModel}`;
  }

  if (requestRef.provider) {
    return `${requestRef.provider}/${nextModel}`;
  }

  return nextModel;
}

function mergeVirtualModelInstructions(
  baseInstructions: string | undefined,
  profileInstructions: VirtualModelProfileConfig['instructions']
): string | undefined {
  if (!profileInstructions) {
    return baseInstructions;
  }

  if (profileInstructions.replace) {
    return profileInstructions.replace;
  }

  const parts = [profileInstructions.prepend, baseInstructions, profileInstructions.append]
    .filter((item): item is string => Boolean(item && item.trim()))
    .map((item) => item.trim());

  if (parts.length === 0) {
    return undefined;
  }

  return parts.join('\n\n');
}

function resolveVirtualToolChoice(
  baseToolChoice: unknown,
  profileToolChoice: unknown,
  toolCount: number
): unknown {
  if (profileToolChoice !== undefined) {
    return profileToolChoice;
  }

  if (toolCount === 0) {
    return baseToolChoice;
  }

  return baseToolChoice !== undefined ? baseToolChoice : 'auto';
}

function extractStandardToolName(tool: unknown): string | undefined {
  if (!isObject(tool)) {
    return undefined;
  }

  const functionPayload = isObject(tool.function) ? tool.function : undefined;
  return asString(tool.name) || asString(functionPayload?.name);
}

function extractFunctionCallsFromStandardResponse(
  response: StandardResponse
): StandardResponseFunctionCall[] {
  return response.output.filter(
    (item): item is StandardResponseFunctionCall => item.type === 'function_call'
  );
}

function partitionVirtualFunctionCalls(
  toolCalls: StandardResponseFunctionCall[],
  toolOwners: Map<string, VirtualToolOwner>
) {
  const internal: StandardResponseFunctionCall[] = [];
  const client: StandardResponseFunctionCall[] = [];
  const unknown: StandardResponseFunctionCall[] = [];

  for (const toolCall of toolCalls) {
    const owner = toolOwners.get(toolCall.name);
    if (!owner) {
      unknown.push(toolCall);
      continue;
    }

    if (owner.visibility === 'internal') {
      internal.push(toolCall);
      continue;
    }

    client.push(toolCall);
  }

  return { internal, client, unknown };
}

function filterInternalToolCallsFromStandardResponse(
  response: StandardResponse,
  toolOwners: Map<string, VirtualToolOwner>,
  usage?: StandardUsage
): StandardResponse {
  return {
    ...response,
    output: response.output.filter((item) => {
      if (item.type !== 'function_call') {
        return true;
      }

      const owner = toolOwners.get(item.name);
      return owner?.visibility !== 'internal';
    }),
    usage: usage || response.usage
  };
}

function resolveVirtualProfileToolDefinition(
  requestedToolName: string,
  definitions: AgentToolDefinition[]
):
  | { runtimeToolName: string }
  | { error: string }
  | undefined {
  const resolved = resolveRuntimeToolDefinition(requestedToolName, definitions);
  if (!resolved || !('error' in resolved)) {
    return resolved;
  }

  return {
    error:
      `Virtual model tool "${requestedToolName}" is ambiguous across MCP providers. ` +
      `Use the canonical tool name instead. Candidates: ${resolved.candidates.join(', ')}`
  };
}

function resolveRuntimeToolDefinition(
  requestedToolName: string,
  definitions: AgentToolDefinition[]
):
  | { runtimeToolName: string }
  | { error: 'ambiguous'; candidates: string[] }
  | undefined {
  if (!requestedToolName) {
    return undefined;
  }

  const exact = definitions.find((tool) => tool.name === requestedToolName);
  if (exact) {
    return { runtimeToolName: exact.name };
  }

  const suffixMatches = definitions.filter((tool) => tool.name.endsWith(`.${requestedToolName}`));
  if (suffixMatches.length === 1) {
    return { runtimeToolName: suffixMatches[0].name };
  }

  if (suffixMatches.length > 1) {
    return {
      error: 'ambiguous',
      candidates: suffixMatches.map((tool) => tool.name)
    };
  }

  return undefined;
}

async function resolveTransparentToolCalls(
  runtime: GatewayRuntime,
  config: GatewayConfig,
  standardRequest: StandardRequest,
  toolCalls: StandardResponseFunctionCall[]
): Promise<TransparentToolResolution> {
  const executionConfig = config.transparentToolExecution;
  if (!runtime.toolProvider) {
    return handleTransparentUnknownTool(
      executionConfig?.unknownToolPolicy || 'return_to_client',
      'Transparent tool execution requires gateway MCP tool provider to be configured.'
    );
  }

  const declaredToolNames = new Set(
    Array.isArray(standardRequest.tools)
      ? standardRequest.tools
          .map((tool) => extractStandardToolName(tool))
          .filter((toolName): toolName is string => Boolean(toolName))
      : []
  );
  const definitions = await runtime.toolProvider.listDefinitions();
  const bindings = new Map<string, TransparentToolBinding>();
  const executableCalls: StandardResponseFunctionCall[] = [];

  for (const toolCall of toolCalls) {
    if (
      executionConfig?.requireClientDeclaration !== false &&
      !declaredToolNames.has(toolCall.name)
    ) {
      return handleTransparentUnknownTool(
        executionConfig?.unknownToolPolicy || 'return_to_client',
        `Transparent tool execution cannot run undeclared tool: ${toolCall.name}`
      );
    }

    const resolvedTool = resolveRuntimeToolDefinition(toolCall.name, definitions);
    if (!resolvedTool) {
      return handleTransparentUnknownTool(
        executionConfig?.unknownToolPolicy || 'return_to_client',
        `Transparent tool is not available from MCP provider: ${toolCall.name}`
      );
    }

    if ('error' in resolvedTool) {
      return {
        ok: false,
        stage: 'transparent_tool_resolution',
        status: 400,
        message:
          `Transparent tool "${toolCall.name}" is ambiguous across MCP providers. ` +
          `Use the canonical tool name instead. Candidates: ${resolvedTool.candidates.join(', ')}`
      };
    }

    if (
      !isTransparentToolAllowed(
        toolCall.name,
        resolvedTool.runtimeToolName,
        executionConfig?.allowTools || [],
        executionConfig?.denyTools || []
      )
    ) {
      return {
        ok: false,
        stage: 'transparent_tool_resolution',
        status: 403,
        message: `Transparent tool execution is not allowed for tool: ${toolCall.name}`
      };
    }

    bindings.set(toolCall.name, {
      runtimeToolName: resolvedTool.runtimeToolName
    });
    executableCalls.push(toolCall);
  }

  return {
    ok: true,
    executableCalls,
    bindings
  };
}

function handleTransparentUnknownTool(
  policy: GatewayConfig['transparentToolExecution']['unknownToolPolicy'],
  message: string
): TransparentToolResolution {
  if (policy === 'fail') {
    return {
      ok: false,
      stage: 'transparent_tool_resolution',
      status: 400,
      message
    };
  }

  return {
    ok: true,
    executableCalls: [],
    bindings: new Map(),
    returnToClient: true
  };
}

function isTransparentToolAllowed(
  requestedToolName: string,
  runtimeToolName: string,
  allowTools: string[],
  denyTools: string[]
): boolean {
  if (
    matchesAnyPattern(requestedToolName, denyTools) ||
    matchesAnyPattern(runtimeToolName, denyTools)
  ) {
    return false;
  }

  if (allowTools.length === 0) {
    return true;
  }

  return (
    matchesAnyPattern(requestedToolName, allowTools) ||
    matchesAnyPattern(runtimeToolName, allowTools)
  );
}

async function executeTransparentToolCalls(
  runtime: GatewayRuntime,
  source: GatewaySourceContext,
  targetProvider: Provider,
  targetProviderConfig: ProviderConfig | undefined,
  toolCalls: StandardResponseFunctionCall[],
  bindings: Map<string, TransparentToolBinding>
): Promise<StandardRequestInputContent[]> {
  if (!runtime.toolProvider) {
    throw new Error('Transparent tool execution requires gateway MCP tool provider.');
  }

  const results: StandardRequestInputContent[] = [];
  for (const toolCall of toolCalls) {
    const binding = bindings.get(toolCall.name);
    if (!binding) {
      continue;
    }

    const args = parseVirtualToolArguments(toolCall.arguments);
    const timestamp = new Date().toISOString();
    try {
      const output = await runtime.toolProvider.execute(binding.runtimeToolName, {
        args,
        session: {
          sessionId: 'gateway-transparent-tool-execution',
          agentId: 'gateway',
          systemPrompt: '',
          model: undefined,
          allowedTools: [binding.runtimeToolName],
          allowedToolsConfigured: true,
          memoryRefs: [],
          messages: [],
          pendingToolCalls: {},
          taskState: {
            id: 'gateway-transparent-tool-execution',
            goal: 'Transparent gateway tool execution',
            activeStep: null,
            constraints: [],
            done: [],
            todo: [],
            status: 'running'
          },
          transcriptWindow: { items: [] },
          guards: {
            doNotRepeat: [],
            doNotForget: [],
            doNotViolate: []
          },
          lastEventOffset: 0,
          updatedAt: timestamp
        } as any,
        event: {
          id: randomUUID(),
          type: 'TOOL_CALL_REQUESTED',
          sessionId: 'gateway-transparent-tool-execution',
          timestamp,
          correlationId: randomUUID(),
          payload: {
            toolCallId: toolCall.call_id,
            toolName: toolCall.name,
            arguments: args
          }
        } as any
      });

      recordGatewayToolExecution({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        sourceAdapter: source.adapterKey,
        outcome: 'success'
      });
      results.push({
        type: 'tool_result',
        tool_use_id: toolCall.call_id,
        content: stringifyVirtualToolOutput(output),
        result_format: 'function'
      });
    } catch (error) {
      recordGatewayToolExecution({
        provider: targetProvider,
        providerName: targetProviderConfig?.name,
        sourceAdapter: source.adapterKey,
        outcome: 'error'
      });
      results.push({
        type: 'tool_result',
        tool_use_id: toolCall.call_id,
        content: describeVirtualToolError(error),
        is_error: true,
        result_format: 'function'
      });
    }
  }

  return results;
}

function parseVirtualToolArguments(argumentsText: string): Record<string, unknown> {
  if (!argumentsText.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(argumentsText);
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function stringifyVirtualToolOutput(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (value === undefined) {
    return '';
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function describeVirtualToolError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mergeStandardUsage(base: StandardUsage, next: StandardUsage): StandardUsage {
  return {
    input_tokens: (base.input_tokens || 0) + (next.input_tokens || 0) || undefined,
    output_tokens: (base.output_tokens || 0) + (next.output_tokens || 0) || undefined,
    total_tokens: (base.total_tokens || 0) + (next.total_tokens || 0) || undefined,
    cache_read_tokens:
      (base.cache_read_tokens || 0) + (next.cache_read_tokens || 0) || undefined,
    cache_write_tokens:
      (base.cache_write_tokens || 0) + (next.cache_write_tokens || 0) || undefined,
    cache_duration_seconds:
      (base.cache_duration_seconds || 0) + (next.cache_duration_seconds || 0) || undefined,
    cache_ttl_seconds: next.cache_ttl_seconds ?? base.cache_ttl_seconds,
    cache_age_seconds: next.cache_age_seconds ?? base.cache_age_seconds
  };
}

export function parseGeminiTail(tail: string): { model: string; action: 'generateContent' | 'streamGenerateContent' } | null {
  const separator = tail.lastIndexOf(':');
  if (separator <= 0) {
    return null;
  }

  const model = decodeURIComponent(tail.slice(0, separator));
  const action = tail.slice(separator + 1);
  if (!model) {
    return null;
  }

  if (action !== 'generateContent' && action !== 'streamGenerateContent') {
    return null;
  }

  return {
    model,
    action
  };
}

function resolveTargetProviders(
  request: FastifyRequest,
  sourceProvider: Provider,
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
    const fromHeaderList = parseProviderRouteList(fromHeaderListRaw, config.providers);
    if (fromHeaderList.length === 0) {
      return {
        ok: false,
        error: `x-target-providers must include at least one valid provider: ${formatAllowedProviderValues(config.providers)}`
      };
    }

    if (providerRefFromModel && !fromHeaderList.some((route) => routeMatchesModelReference(route, providerRefFromModel))) {
      return {
        ok: false,
        error: `Model selector "${providerRefFromModel.raw}" conflicts with x-target-providers.`
      };
    }

    return { ok: true, value: fromHeaderList };
  }

  const fromHeaderRaw = readHeader(request.headers['x-target-provider']);
  if (fromHeaderRaw !== undefined) {
    const fromHeader = parseProviderRoute(fromHeaderRaw, config.providers);
    if (!fromHeader) {
      return {
        ok: false,
        error: `x-target-provider must be one of: ${formatAllowedProviderValues(config.providers)}`
      };
    }

    if (providerRefFromModel && !routeMatchesModelReference(fromHeader, providerRefFromModel)) {
      return {
        ok: false,
        error: `Model selector "${providerRefFromModel.raw}" conflicts with x-target-provider.`
      };
    }

    return {
      ok: true,
      value: [fromHeader]
    };
  }

  if (providerRefFromModel?.provider) {
    return {
      ok: true,
      value: [routeFromModelReference(providerRefFromModel)]
    };
  }

  if (config.defaultTargetProviders.length > 0) {
    return {
      ok: true,
      value: dedupeProviderRoutes(config.defaultTargetProviders.map((provider) => ({ provider })))
    };
  }

  if (config.defaultTargetProvider) {
    return {
      ok: true,
      value: [{ provider: config.defaultTargetProvider }]
    };
  }

  return {
    ok: true,
    value: [{ provider: sourceProvider }]
  };
}

function resolveTargetModel(
  request: FastifyRequest,
  target: TargetProviderRoute,
  bodyModel: string | undefined,
  config: GatewayConfig
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

  if (target.provider === 'openai') {
    return validateModelForTarget(config.defaultOpenAIModel, target, config);
  }

  if (target.provider === 'anthropic') {
    return validateModelForTarget(config.defaultAnthropicModel, target, config);
  }

  return validateModelForTarget(config.defaultGeminiModel, target, config);
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

  return {
    ok: false,
    error: `Model "${model}" is not configured for target provider ${formatTargetProviderLabel(target)}. Allowed models: ${providerConfig.models.join(', ')}.`
  };
}

function parseProviderRouteList(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): TargetProviderRoute[] {
  if (!value) {
    return [];
  }

  const routes = value
    .split(',')
    .map((item) => parseProviderRoute(item, providerConfigs))
    .filter((item): item is TargetProviderRoute => Boolean(item));

  return dedupeProviderRoutes(routes);
}

function parseProviderRoute(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): TargetProviderRoute | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim();
  if (!normalized) {
    return undefined;
  }

  const byName = findProviderConfigByName(providerConfigs, normalized);
  if (byName) {
    return {
      provider: providerFromProviderType(byName.type),
      providerConfig: byName
    };
  }

  const byType = parseProvider(normalized);
  if (!byType) {
    return undefined;
  }

  return {
    provider: byType
  };
}

function dedupeProviderRoutes(routes: TargetProviderRoute[]): TargetProviderRoute[] {
  const deduped: TargetProviderRoute[] = [];
  const usedKeys = new Set<string>();

  for (const route of routes) {
    const key = route.providerConfig ? `name:${route.providerConfig.name}` : `type:${route.provider}`;
    if (usedKeys.has(key)) {
      continue;
    }

    usedKeys.add(key);
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

function formatTargetProviderLabel(route: TargetProviderRoute): string {
  return route.providerConfig?.name || route.provider;
}

function parseModelReference(
  value: string | undefined,
  providerConfigs: ProviderConfig[]
): ParsedModelReference | undefined {
  if (!value) {
    return undefined;
  }

  const raw = value.trim();
  if (!raw) {
    return undefined;
  }

  const slashIndex = raw.indexOf('/');
  if (slashIndex <= 0 || slashIndex >= raw.length - 1) {
    return {
      raw,
      model: raw
    };
  }

  const providerHint = raw.slice(0, slashIndex).trim();
  const model = raw.slice(slashIndex + 1).trim();
  if (!providerHint || !model) {
    return {
      raw,
      model: raw
    };
  }

  const providerConfig = findProviderConfigByName(providerConfigs, providerHint);
  if (providerConfig) {
    return {
      raw,
      model,
      provider: providerFromProviderType(providerConfig.type),
      providerConfig
    };
  }

  const providerType = parseProvider(providerHint);
  if (providerType) {
    return {
      raw,
      model,
      provider: providerType
    };
  }

  return {
    raw,
    model: raw
  };
}

function formatAllowedProviderValues(providerConfigs: ProviderConfig[]): string {
  const providerTypes: Provider[] = ['openai', 'anthropic', 'gemini'];
  const providerNames = dedupeProviderRoutes(
    providerConfigs.map((providerConfig) => ({
      provider: providerFromProviderType(providerConfig.type),
      providerConfig
    }))
  )
    .map((route) => route.providerConfig?.name)
    .filter((item): item is string => Boolean(item));

  return [...providerTypes, ...providerNames].join(', ');
}

function sendBadRequest(reply: FastifyReply, message: string) {
  return reply.code(400).send({ error: { message } });
}

function sendGatewayPrecheckFailure(reply: FastifyReply, result: GatewayPrecheckResult) {
  if (result.ok) {
    return undefined;
  }

  return reply.code(result.statusCode).send({
    error: {
      message: result.message,
      code: result.code,
      details: result.details
    }
  });
}

function sendCountedBadRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  target: TargetProviderRoute | undefined,
  model: string | undefined,
  message: string,
) {
  if (!target) {
    return sendBadRequest(reply, message);
  }

  const payload = {
    error: {
      message,
    },
  };
  reply.code(400);
  publishRequestFailureEventSafe(
    request,
    reply,
    config,
    sourceProvider,
    sourceAdapterKey,
    target.provider,
    model,
    0,
    [],
    message,
    resolveProviderConfig(config, target),
    buildGatewayBillingTraceSnapshot(request, reply, {
      responseBody: payload,
    }),
  );
  return reply.send(payload);
}

function buildGatewayPolicyAttempt(
  provider: Provider,
  providerConfig: ProviderConfig | undefined,
  result: Extract<GatewayPolicyResult, { ok: false }>
): ProviderAttemptFailure {
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
  attempts: ProviderAttemptFailure[],
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
          details: attempt.details,
        })),
      },
    },
  };
}

function attachTargetRoutingHeaders(
  reply: FastifyReply,
  provider: Provider,
  providerName: string | undefined,
  fallbackAttempts: number
) {
  reply.header('x-gateway-target-provider', provider);
  if (providerName) {
    reply.header('x-gateway-target-provider-name', providerName);
  }

  if (fallbackAttempts > 0) {
    reply.header('x-gateway-fallback-used', 'true');
    reply.header('x-gateway-fallback-count', String(fallbackAttempts));
  }
}

function attachBillingHeaders(
  request: FastifyRequest,
  reply: FastifyReply,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  provider: Provider,
  usage: StandardUsage,
  config: GatewayConfig,
  fallbackAttempts: number,
  attemptSequence: number,
  model?: string,
  targetProviderConfig?: ProviderConfig,
  trace?: GatewayBillingTrace,
  rawTraceCapture?: GatewayRawTraceCapture
) {
  if (!config.billing.enabled) {
    if (config.rawTrace.enabled) {
      publishRawTraceCaptureSafe(
        request,
        config,
        provider,
        model,
        targetProviderConfig,
        rawTraceCapture
      );
    }
    return;
  }

  const billing = calculateUsageBilling(
    provider,
    usage,
    config.billing,
    resolveProviderBillingRate(config, provider, model, targetProviderConfig)
  );
  const headers = buildBillingHeaders(billing);
  for (const [key, value] of Object.entries(headers)) {
    reply.header(key, value);
  }

  request.log.info(
    {
      provider: billing.provider,
      usage: billing.usage,
      rates: billing.rates,
      cost: billing.cost
    },
    'Usage billing computed.'
  );

  publishBillingEventSafe(
    request,
    reply,
    config,
    sourceProvider,
    sourceAdapterKey,
    provider,
    model,
    fallbackAttempts,
    undefined,
    billing,
    targetProviderConfig,
    trace,
    rawTraceCapture
  );
}

async function tryAttachBillingHeadersFromUpstreamResponse(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  targetProvider: Provider,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  fallbackAttempts: number,
  attemptSequence: number,
  targetAdapter: TargetAdapter,
  upstreamRequest: UpstreamRequest,
  upstreamResponse: Response,
  fallbackModel?: string,
  targetProviderConfig?: ProviderConfig
) {
  if ((!config.billing.enabled && !config.rawTrace.enabled) || !upstreamResponse.ok) {
    return;
  }

  let upstreamPayload: unknown;
  try {
    upstreamPayload = await collectBillingPayloadFromUpstreamResponse(targetProvider, upstreamResponse.clone());
  } catch (error) {
    request.log.warn(
      {
        provider: targetProvider,
        details: error instanceof Error ? error.message : String(error)
      },
      'Failed to read passthrough response for billing.'
    );
    return;
  }

  if (!config.billing.enabled) {
    publishRawTraceCaptureSafe(
      request,
      config,
      targetProvider,
      fallbackModel,
      targetProviderConfig,
      buildUpstreamRawTraceCapture(upstreamRequest, upstreamResponse, upstreamPayload)
    );
    return;
  }

  const billingResponseResult = resolveBillingResponseSnapshot(
    targetProvider,
    targetAdapter,
    upstreamPayload
  );
  if (!billingResponseResult.ok) {
    request.log.warn(
      {
        provider: targetProvider,
        error: billingResponseResult.error
      },
      'Failed to parse passthrough response usage for billing.'
    );
    return;
  }

  if (billingResponseResult.value.recovered) {
    request.log.debug(
      {
        provider: targetProvider
      },
      'Recovered passthrough response usage from provider payload for billing.'
    );
  }

  attachBillingHeaders(
    request,
    reply,
    sourceProvider,
    sourceAdapterKey,
    targetProvider,
    billingResponseResult.value.usage,
    config,
    fallbackAttempts,
    attemptSequence,
    resolveBillingModel(billingResponseResult.value.model, fallbackModel),
    targetProviderConfig,
    buildGatewayBillingTraceSnapshot(request, reply, {
      responseBody: upstreamPayload,
      responseStatusCode: upstreamResponse.status
    }),
    buildUpstreamRawTraceCapture(upstreamRequest, upstreamResponse, upstreamPayload)
  );
}

function tryAttachBillingHeadersFromUpstreamPayload(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  targetProvider: Provider,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  fallbackAttempts: number,
  attemptSequence: number,
  targetAdapter: TargetAdapter,
  upstreamRequest: UpstreamRequest,
  upstreamPayload: unknown,
  fallbackModel?: string,
  targetProviderConfig?: ProviderConfig,
  responsePayload?: unknown,
  responseStatusCode?: number
) {
  if (!config.billing.enabled && !config.rawTrace.enabled) {
    return;
  }

  if (!config.billing.enabled) {
    publishRawTraceCaptureSafe(
      request,
      config,
      targetProvider,
      fallbackModel,
      targetProviderConfig,
      {
        upstreamRequest,
        upstreamResponseBody: responsePayload ?? upstreamPayload,
        upstreamResponseStatus: responseStatusCode
      }
    );
    return;
  }

  const billingResponseResult = resolveBillingResponseSnapshot(
    targetProvider,
    targetAdapter,
    upstreamPayload
  );
  if (!billingResponseResult.ok) {
    request.log.warn(
      {
        provider: targetProvider,
        error: billingResponseResult.error
      },
      'Failed to parse passthrough payload usage for billing.'
    );
    return;
  }

  if (billingResponseResult.value.recovered) {
    request.log.debug(
      {
        provider: targetProvider
      },
      'Recovered passthrough payload usage from provider payload for billing.'
    );
  }

  attachBillingHeaders(
    request,
    reply,
    sourceProvider,
    sourceAdapterKey,
    targetProvider,
    billingResponseResult.value.usage,
    config,
    fallbackAttempts,
    attemptSequence,
    resolveBillingModel(billingResponseResult.value.model, fallbackModel),
    targetProviderConfig,
    buildGatewayBillingTraceSnapshot(request, reply, {
      responseBody: responsePayload ?? upstreamPayload,
      responseStatusCode
    }),
    {
      upstreamRequest,
      upstreamResponseBody: responsePayload ?? upstreamPayload,
      upstreamResponseStatus: responseStatusCode
    }
  );
}

async function tryPublishStreamingBillingEventFromUpstreamResponse(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  targetProvider: Provider,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  fallbackAttempts: number,
  attemptSequence: number,
  targetAdapter: TargetAdapter,
  upstreamRequest: UpstreamRequest,
  upstreamResponse: Response,
  fallbackModel?: string,
  targetProviderConfig?: ProviderConfig,
  rawTraceStreamResponse?: Response
) {
  if ((!config.billing.enabled && !config.rawTrace.enabled) || !upstreamResponse.ok) {
    return;
  }

  const rawStreamCapturePromise = collectRawTraceResponseStreamSafe(
    request,
    config,
    rawTraceStreamResponse,
  );
  let upstreamPayload: unknown;
  try {
    upstreamPayload = await collectBillingPayloadFromUpstreamResponse(targetProvider, upstreamResponse);
  } catch (error) {
    const rawStreamCapture = await rawStreamCapturePromise;
    if (config.rawTrace.enabled) {
      publishRawTraceCaptureSafe(
        request,
        config,
        targetProvider,
        fallbackModel,
        targetProviderConfig,
        buildUpstreamRawTraceCapture(
          upstreamRequest,
          upstreamResponse,
          {
            read_error: error instanceof Error ? error.message : String(error),
          },
          rawStreamCapture,
        ),
      );
    }
    request.log.warn(
      {
        provider: targetProvider,
        details: error instanceof Error ? error.message : String(error)
      },
      'Failed to collect streaming response payload for billing.'
    );
    return;
  }
  const rawStreamCapture = await rawStreamCapturePromise;
  const rawTraceCapture = buildUpstreamRawTraceCapture(
    upstreamRequest,
    upstreamResponse,
    upstreamPayload,
    rawStreamCapture,
  );

  if (!config.billing.enabled) {
    publishRawTraceCaptureSafe(
      request,
      config,
      targetProvider,
      fallbackModel,
      targetProviderConfig,
      rawTraceCapture
    );
    return;
  }

  const billingResponseResult = resolveBillingResponseSnapshot(
    targetProvider,
    targetAdapter,
    upstreamPayload
  );
  if (!billingResponseResult.ok) {
    request.log.warn(
      {
        provider: targetProvider,
        details: billingResponseResult.error
      },
      'Failed to parse streaming response usage for billing.'
    );
    return;
  }

  if (billingResponseResult.value.recovered) {
    request.log.debug(
      {
        provider: targetProvider
      },
      'Recovered streaming response usage from provider payload for billing.'
    );
  }

  const billingModel = resolveBillingModel(billingResponseResult.value.model, fallbackModel);
  const billing = calculateUsageBilling(
    targetProvider,
    billingResponseResult.value.usage,
    config.billing,
    resolveProviderBillingRate(config, targetProvider, billingModel, targetProviderConfig)
  );

  request.log.info(
    {
      provider: billing.provider,
      usage: billing.usage,
      rates: billing.rates,
      cost: billing.cost
    },
    'Streaming usage billing computed.'
  );

  publishBillingEventSafe(
    request,
    reply,
    config,
    sourceProvider,
    sourceAdapterKey,
    targetProvider,
    billingModel,
    fallbackAttempts,
    undefined,
    billing,
    targetProviderConfig,
    buildGatewayBillingTraceSnapshot(request, reply, {
      responseBody: upstreamPayload,
      responseStatusCode: upstreamResponse.status
    }),
    rawTraceCapture
  );
}

export function resolveBillingResponseSnapshot(
  targetProvider: Provider,
  targetAdapter: TargetAdapter,
  upstreamPayload: unknown
): { ok: true; value: BillingResponseSnapshot } | { ok: false; error: string } {
  const standardResponseResult = targetAdapter.toStandardResponse(upstreamPayload);
  if (standardResponseResult.ok) {
    return {
      ok: true,
      value: {
        model: standardResponseResult.value.model,
        usage: normalizeBillingSnapshotUsage(
          targetProvider,
          standardResponseResult.value.usage,
        ),
        recovered: false
      }
    };
  }

  const recovered = extractBillingResponseSnapshotFromProviderPayload(targetProvider, upstreamPayload);
  if (recovered) {
    return {
      ok: true,
      value: {
        ...recovered,
        usage: normalizeBillingSnapshotUsage(targetProvider, recovered.usage),
        recovered: true
      }
    };
  }

  return {
    ok: false,
    error: standardResponseResult.error
  };
}

function normalizeBillingSnapshotUsage(
  provider: Provider,
  usage: StandardUsage,
): StandardUsage {
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;
  const cacheReadTokens = usage.cache_read_tokens;
  const cacheWriteTokens = usage.cache_write_tokens;

  if (provider !== 'anthropic') {
    return usage;
  }

  const anthropicTotal = sumOptionalUsageTokens(
    inputTokens,
    outputTokens,
    cacheReadTokens,
    cacheWriteTokens,
  );
  if (anthropicTotal === undefined) {
    return usage;
  }

  return {
    ...usage,
    total_tokens: Math.max(usage.total_tokens || 0, anthropicTotal)
  };
}

function extractBillingResponseSnapshotFromProviderPayload(
  targetProvider: Provider,
  upstreamPayload: unknown
): Omit<BillingResponseSnapshot, 'recovered'> | undefined {
  if (targetProvider === 'openai') {
    return extractOpenAIBillingResponseSnapshot(upstreamPayload);
  }

  if (targetProvider === 'anthropic') {
    return extractAnthropicBillingResponseSnapshot(upstreamPayload);
  }

  return extractGeminiBillingResponseSnapshot(upstreamPayload);
}

function extractOpenAIBillingResponseSnapshot(
  payload: unknown
): Omit<BillingResponseSnapshot, 'recovered'> | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const responsePayload = isObject(payload.response) ? payload.response : payload;
  const usageRaw = isObject(responsePayload.usage) ? responsePayload.usage : undefined;
  if (!usageRaw) {
    return recoverZeroOpenAIBillingResponseSnapshot(responsePayload);
  }

  const inputDetails = isObject(usageRaw.input_tokens_details)
    ? usageRaw.input_tokens_details
    : isObject(usageRaw.prompt_tokens_details)
      ? usageRaw.prompt_tokens_details
      : undefined;
  const usage: StandardUsage = {
    input_tokens: asNumber(usageRaw.input_tokens) ?? asNumber(usageRaw.prompt_tokens),
    output_tokens: asNumber(usageRaw.output_tokens) ?? asNumber(usageRaw.completion_tokens),
    total_tokens: asNumber(usageRaw.total_tokens),
    cache_read_tokens: asNumber(inputDetails?.cached_tokens) ?? asNumber(usageRaw.cache_read_tokens),
    cache_write_tokens:
      asNumber(inputDetails?.cache_creation_tokens) ??
      asNumber(usageRaw.cache_creation_tokens) ??
      asNumber(usageRaw.cache_write_tokens),
    cache_duration_seconds: extractBillingCacheDurationSeconds(usageRaw, inputDetails)
  };
  if (!hasUsageData(usage)) {
    return recoverZeroOpenAIBillingResponseSnapshot(responsePayload);
  }

  return {
    model: asString(responsePayload.model) || undefined,
    usage
  };
}

function recoverZeroOpenAIBillingResponseSnapshot(
  responsePayload: Record<string, unknown>
): Omit<BillingResponseSnapshot, 'recovered'> | undefined {
  if (!isRecoverableOpenAIBillingPayload(responsePayload)) {
    return undefined;
  }

  return {
    model: asString(responsePayload.model) || undefined,
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      cache_read_tokens: 0
    }
  };
}

function isRecoverableOpenAIBillingPayload(responsePayload: Record<string, unknown>): boolean {
  const object = asString(responsePayload.object);
  if (object === 'response' || object === 'chat.completion') {
    return true;
  }

  const id = asString(responsePayload.id);
  if (id?.startsWith('resp_') || id?.startsWith('chatcmpl_')) {
    return true;
  }

  return Array.isArray(responsePayload.output) || Array.isArray(responsePayload.choices);
}

function extractAnthropicBillingResponseSnapshot(
  payload: unknown
): Omit<BillingResponseSnapshot, 'recovered'> | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const usageRaw = isObject(payload.usage) ? payload.usage : undefined;
  if (!usageRaw) {
    return undefined;
  }

  const inputTokens = asNumber(usageRaw.input_tokens);
  const outputTokens = asNumber(usageRaw.output_tokens);
  const cacheReadTokens =
    asNumber(usageRaw.cache_read_input_tokens) ?? asNumber(usageRaw.cache_read_tokens);
  const cacheWriteTokens =
    asNumber(usageRaw.cache_creation_input_tokens) ??
    asNumber(usageRaw.cache_creation_tokens) ??
    asNumber(usageRaw.cache_write_tokens);
  const usage: StandardUsage = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens:
      asNumber(usageRaw.total_tokens) ??
      sumOptionalUsageTokens(
        inputTokens,
        outputTokens,
        cacheReadTokens,
        cacheWriteTokens
      ),
    cache_read_tokens: cacheReadTokens,
    cache_write_tokens: cacheWriteTokens,
    cache_duration_seconds: extractBillingCacheDurationSeconds(usageRaw)
  };
  if (!hasUsageData(usage)) {
    return undefined;
  }

  return {
    model: asString(payload.model) || undefined,
    usage
  };
}

function extractGeminiBillingResponseSnapshot(
  payload: unknown
): Omit<BillingResponseSnapshot, 'recovered'> | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const usageRaw = isObject(payload.usageMetadata) ? payload.usageMetadata : undefined;
  if (!usageRaw) {
    return undefined;
  }

  const usage: StandardUsage = {
    input_tokens: asNumber(usageRaw.promptTokenCount),
    output_tokens: asNumber(usageRaw.candidatesTokenCount),
    total_tokens: asNumber(usageRaw.totalTokenCount),
    cache_read_tokens: asNumber(usageRaw.cachedContentTokenCount),
    cache_duration_seconds: extractBillingCacheDurationSeconds(usageRaw)
  };
  if (!hasUsageData(usage)) {
    return undefined;
  }

  return {
    model: asString(payload.modelVersion) || undefined,
    usage
  };
}

function hasUsageData(usage: StandardUsage): boolean {
  return (
    usage.input_tokens !== undefined ||
    usage.output_tokens !== undefined ||
    usage.total_tokens !== undefined ||
    usage.cache_read_tokens !== undefined ||
    usage.cache_write_tokens !== undefined ||
    usage.cache_duration_seconds !== undefined ||
    usage.cache_ttl_seconds !== undefined ||
    usage.cache_age_seconds !== undefined
  );
}

function sumOptionalUsageTokens(...values: Array<number | undefined>): number | undefined {
  if (values.every((value) => value === undefined)) {
    return undefined;
  }

  return values.reduce<number>((sum, value) => sum + (value || 0), 0);
}

function extractBillingCacheDurationSeconds(
  usageRaw?: Record<string, unknown>,
  detailsRaw?: Record<string, unknown>
): number | undefined {
  if (!usageRaw && !detailsRaw) {
    return undefined;
  }

  const fromSeconds =
    asNumber(detailsRaw?.cache_duration_seconds) ??
    asNumber(detailsRaw?.cache_ttl_seconds) ??
    asNumber(usageRaw?.cache_duration_seconds) ??
    asNumber(usageRaw?.cache_ttl_seconds) ??
    asNumber(usageRaw?.cache_age_seconds);
  if (fromSeconds !== undefined) {
    return normalizeBillingDurationSeconds(fromSeconds);
  }

  const fromMillis =
    asNumber(detailsRaw?.cache_duration_ms) ??
    asNumber(detailsRaw?.cache_ttl_ms) ??
    asNumber(usageRaw?.cache_duration_ms) ??
    asNumber(usageRaw?.cache_ttl_ms);
  if (fromMillis !== undefined) {
    return normalizeBillingDurationSeconds(fromMillis / 1000);
  }

  return undefined;
}

function normalizeBillingDurationSeconds(value: number): number | undefined {
  if (!Number.isFinite(value) || value <= 0) {
    return undefined;
  }

  return Math.trunc(value);
}

async function collectBillingPayloadFromUpstreamResponse(
  targetProvider: Provider,
  upstreamResponse: Response
): Promise<unknown> {
  if (!isEventStreamResponse(upstreamResponse)) {
    return await readUpstreamPayload(upstreamResponse);
  }

  if (targetProvider === 'openai') {
    return await collectOpenAINonStreamPayloadFromEventStream(upstreamResponse);
  }

  if (targetProvider === 'anthropic') {
    return await collectAnthropicNonStreamPayloadFromEventStream(upstreamResponse);
  }

  return await readUpstreamPayload(upstreamResponse);
}

type UpstreamDispatchFailureStage =
  | ProviderRequestPluginFailureStage
  | 'upstream_connect'
  | 'upstream_concurrency'
  | 'upstream_circuit_open';

type UpstreamDispatchResult =
  | { ok: true; upstreamRequest: UpstreamRequest; upstreamResponse: Response }
  | {
      ok: false;
      stage: UpstreamDispatchFailureStage;
      status: number;
      message: string;
      details?: unknown;
      upstreamRequest?: UpstreamRequest;
    };

async function dispatchUpstreamRequest(
  context: ProviderPluginExecutionContext,
  baseUpstreamRequest: UpstreamRequest,
  timeoutMs: number,
  standardRequest?: StandardRequest
): Promise<UpstreamDispatchResult> {
  const requestPluginResult = await applyProviderRequestPlugins(
    context,
    baseUpstreamRequest,
    standardRequest
  );
  if (!requestPluginResult.ok) {
    return requestPluginResult;
  }

  const initialUpstreamRequest = requestPluginResult.value;
  const initialUpstreamResponse = await callUpstreamWithFailureCapture(
    context,
    initialUpstreamRequest,
    timeoutMs
  );
  if (!initialUpstreamResponse.ok) {
    return {
      ...initialUpstreamResponse,
      upstreamRequest: initialUpstreamRequest,
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
      sourceAdapterKey: context.sourceAdapterKey,
      status: 401
    },
    'Upstream returned 401. Retrying once with forced provider auth refresh.'
  );

  const retryPluginResult = await applyProviderRequestPlugins(
    {
      ...context,
      forceCodexOauthRefreshOnce: true
    },
    baseUpstreamRequest,
    standardRequest
  );
  if (!retryPluginResult.ok) {
    context.request.log.warn(
      {
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.sourceAdapterKey,
        details: retryPluginResult.message
      },
      'Forced provider auth refresh failed after upstream 401. Returning original upstream response.'
    );
    return {
      ok: true,
      upstreamRequest: initialUpstreamRequest,
      upstreamResponse: initialUpstreamResponse.value
    };
  }

  const retryUpstreamRequest = retryPluginResult.value;
  const retryUpstreamResponse = await callUpstreamWithFailureCapture(
    context,
    retryUpstreamRequest,
    timeoutMs
  );
  if (!retryUpstreamResponse.ok) {
    context.request.log.warn(
      {
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.sourceAdapterKey,
        details: retryUpstreamResponse.details
      },
      'Retry request failed after upstream 401 and forced provider auth refresh. Returning original upstream response.'
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

async function callUpstreamWithFailureCapture(
  context: ProviderPluginExecutionContext,
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
    context.targetProviderConfig
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
      undefined,
      {
        logger: context.request.log,
        requestId: context.request.id,
        provider: context.targetProvider,
        providerName: context.targetProviderConfig?.name,
        sourceAdapterKey: context.sourceAdapterKey
      },
      context.config.upstreamRetry
    );
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
    recordProviderHealthFailure(
      context.targetProviderConfig,
      Date.now() - startedAt
    );
    recordProviderCircuitBreakerFailure(
      context.config,
      context.targetProvider,
      context.targetProviderConfig
    );
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

async function safeReadUpstreamPayload(
  request: FastifyRequest,
  provider: Provider,
  upstreamResponse: Response
): Promise<unknown> {
  try {
    return await readUpstreamPayload(upstreamResponse);
  } catch (error) {
    const details = error instanceof Error ? error.message : String(error);
    request.log.warn(
      {
        provider,
        details
      },
      'Failed to parse upstream payload.'
    );

    return {
      read_error: details
    };
  }
}

async function applyProviderRequestPlugins(
  context: ProviderPluginExecutionContext,
  baseUpstreamRequest: UpstreamRequest,
  standardRequest?: StandardRequest
): Promise<ProviderRequestPluginResult> {
  let upstreamRequest = baseUpstreamRequest;

  for (const plugin of context.plugins) {
    if (plugin.authenticate) {
      try {
        const result = await plugin.authenticate({
          request: context.request,
          config: context.config,
          source: context.source,
          sourceProvider: context.sourceProvider,
          sourceAdapterKey: context.sourceAdapterKey,
          targetProvider: context.targetProvider,
          targetProviderConfig: context.targetProviderConfig,
          model: context.model,
          passthrough: context.passthrough,
          streaming: context.streaming,
          forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
          upstreamRequest,
          standardRequest
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
          source: context.source,
          sourceProvider: context.sourceProvider,
          sourceAdapterKey: context.sourceAdapterKey,
          targetProvider: context.targetProvider,
          targetProviderConfig: context.targetProviderConfig,
          model: context.model,
          passthrough: context.passthrough,
          streaming: context.streaming,
          forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
          upstreamRequest,
          standardRequest
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

  return {
    ok: true,
    value: upstreamRequest
  };
}

async function applyProviderResponsePlugins(
  context: ProviderPluginExecutionContext,
  upstreamRequest: UpstreamRequest,
  upstreamResponse: Response,
  basePayload: unknown,
  standardRequest?: StandardRequest
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
        source: context.source,
        sourceProvider: context.sourceProvider,
        sourceAdapterKey: context.sourceAdapterKey,
        targetProvider: context.targetProvider,
        targetProviderConfig: context.targetProviderConfig,
        model: context.model,
        passthrough: context.passthrough,
        streaming: context.streaming,
        forceCodexOauthRefreshOnce: context.forceCodexOauthRefreshOnce,
        upstreamRequest,
        upstreamResponse,
        upstreamPayload: payload,
        standardRequest
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

  return {
    ok: true,
    value: payload
  };
}

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

function relayUpstreamResponseWithPayload(
  reply: FastifyReply,
  upstreamResponse: Response,
  payload: unknown
) {
  reply.code(upstreamResponse.status);

  upstreamResponse.headers.forEach((value, key) => {
    const normalized = key.toLowerCase();
    if (!hopByHopResponseHeaders.has(normalized) && normalized !== 'content-length') {
      reply.header(key, value);
    }
  });

  if ((isPlainObject(payload) || Array.isArray(payload)) && !isJsonContentType(upstreamResponse.headers.get('content-type'))) {
    reply.header('content-type', 'application/json');
  }

  return reply.send(payload);
}

function isJsonContentType(value: string | null): boolean {
  if (!value) {
    return false;
  }

  return value.toLowerCase().includes('application/json');
}

function formatPluginExecutionError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function resolvePassthroughModel(body: Record<string, unknown>, source: GatewaySourceContext): string | undefined {
  const modelFromMetadata = source.metadata?.model;
  if (typeof modelFromMetadata === 'string' && modelFromMetadata.trim()) {
    return modelFromMetadata.trim();
  }

  const model = body.model;
  return typeof model === 'string' && model.trim() ? model.trim() : undefined;
}

function applyPassthroughModelOverride(
  request: { url: string; headers: Record<string, string>; body: unknown },
  provider: Provider,
  model: string | undefined
): { url: string; headers: Record<string, string>; body: unknown } {
  if (!model) {
    return request;
  }

  if (provider === 'gemini') {
    return {
      ...request,
      url: replaceGeminiModelInUrl(request.url, model)
    };
  }

  if (!isPlainObject(request.body)) {
    return request;
  }

  return {
    ...request,
    body: {
      ...request.body,
      model
    }
  };
}

function replaceGeminiModelInUrl(url: string, model: string): string {
  try {
    const parsed = new URL(url);
    parsed.pathname = parsed.pathname.replace(
      /\/models\/[^/]+:(generateContent|streamGenerateContent)$/,
      `/models/${encodeURIComponent(model)}:$1`
    );
    return parsed.toString();
  } catch {
    return url;
  }
}

function applyProviderRequestOverrides(
  config: GatewayConfig,
  target: TargetProviderRoute,
  model: string | undefined,
  request: { url: string; headers: Record<string, string>; body: unknown }
) {
  const providerConfig = resolveProviderConfig(config, target);
  if (!providerConfig) {
    return request;
  }

  const extraHeaders = resolveScopedHeaders(providerConfig, model);
  const extraBody = resolveScopedBody(providerConfig, model);
  const hasHeadersOverride = Object.keys(extraHeaders).length > 0;
  const hasBodyOverride = Object.keys(extraBody).length > 0;
  const hasExplicitProviderConfig = Boolean(target.providerConfig);

  let url = request.url;
  let headers = hasHeadersOverride ? { ...request.headers, ...extraHeaders } : request.headers;
  if (hasExplicitProviderConfig && providerConfig.baseurl) {
    url = overrideUpstreamBaseUrl(url, providerConfig.baseurl, target.provider, config);
  }

  if (hasExplicitProviderConfig && providerConfig.apikey) {
    const credentialOverride = applyProviderCredentials(url, headers, target.provider, providerConfig.apikey);
    url = credentialOverride.url;
    headers = credentialOverride.headers;
  }

  let body = request.body;
  if (hasBodyOverride && isPlainObject(body)) {
    body = mergeJsonObjects(body, extraBody);
  }

  return {
    url,
    headers,
    body
  };
}

function resolveProviderBillingRate(
  config: GatewayConfig,
  provider: Provider,
  model: string | undefined,
  targetProviderConfig?: ProviderConfig
) {
  const providerConfig = targetProviderConfig || findProviderConfigByType(config.providers, provider);
  if (!providerConfig) {
    return undefined;
  }

  if (model && providerConfig.billing.byModel[model]) {
    return providerConfig.billing.byModel[model];
  }

  return providerConfig.billing.default;
}

function resolveScopedHeaders(providerConfig: ProviderConfig, model: string | undefined): Record<string, string> {
  if (!model) {
    return providerConfig.extraHeaders.default;
  }

  const modelHeaders = providerConfig.extraHeaders.byModel[model];
  if (!modelHeaders) {
    return providerConfig.extraHeaders.default;
  }

  return {
    ...providerConfig.extraHeaders.default,
    ...modelHeaders
  };
}

function resolveScopedBody(providerConfig: ProviderConfig, model: string | undefined): Record<string, unknown> {
  if (!model) {
    return providerConfig.extraBody.default;
  }

  const modelBody = providerConfig.extraBody.byModel[model];
  if (!modelBody) {
    return providerConfig.extraBody.default;
  }

  return mergeJsonObjects(providerConfig.extraBody.default, modelBody);
}

function resolveProviderConfig(config: GatewayConfig, target: TargetProviderRoute): ProviderConfig | undefined {
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
  if (!normalized) {
    return undefined;
  }

  return providers.find((item) => item.name.trim().toLowerCase() === normalized);
}

function overrideUpstreamBaseUrl(
  url: string,
  overriddenBaseUrl: string,
  provider: Provider,
  config: GatewayConfig
): string {
  const defaultBaseUrl = providerBaseUrlForType(provider, config);
  if (url.startsWith(defaultBaseUrl)) {
    return `${overriddenBaseUrl}${url.slice(defaultBaseUrl.length)}`;
  }

  try {
    const parsedUrl = new URL(url);
    const parsedOverrideBase = new URL(overriddenBaseUrl);
    const overridePath = trimRightSlash(parsedOverrideBase.pathname);
    parsedOverrideBase.pathname = `${overridePath}${parsedUrl.pathname}`;
    parsedOverrideBase.search = parsedUrl.search;
    return parsedOverrideBase.toString();
  } catch {
    return url;
  }
}

function providerBaseUrlForType(provider: Provider, config: GatewayConfig): string {
  if (provider === 'openai') {
    return config.openaiBaseUrl;
  }

  if (provider === 'anthropic') {
    return config.anthropicBaseUrl;
  }

  return config.geminiBaseUrl;
}

function applyProviderCredentials(
  url: string,
  headers: Record<string, string>,
  provider: Provider,
  apiKey: string
): { url: string; headers: Record<string, string> } {
  if (provider === 'openai') {
    return {
      url,
      headers: {
        ...headers,
        authorization: `Bearer ${apiKey}`
      }
    };
  }

  if (provider === 'anthropic') {
    return {
      url,
      headers: {
        ...headers,
        'x-api-key': apiKey
      }
    };
  }

  try {
    const parsed = new URL(url);
    parsed.searchParams.set('key', apiKey);
    return {
      url: parsed.toString(),
      headers
    };
  } catch {
    return {
      url,
      headers
    };
  }
}

function resolveBillingModel(primary?: string, fallback?: string): string | undefined {
  if (primary && primary !== 'unknown') {
    return primary;
  }

  return fallback;
}

export function extractGatewayRequestClientContext(
  request: FastifyRequest,
  body: Record<string, unknown>
): GatewayRequestClientContext | undefined {
  const metadata = isPlainObject(body.metadata) ? body.metadata : undefined;
  const detectedContext = detectGatewayAgentClientContext(request, body, metadata);
  const contextMetadata = mergeGatewayClientContextMetadata(
    metadata,
    detectedContext?.metadata
  );
  const clientContext: GatewayRequestClientContext = {
    agentId:
      readClientContextValue(request, metadata, 'x-agent-id', ['agentId', 'agent_id']) ||
      detectedContext?.agentId,
    sessionId:
      readClientContextValue(request, metadata, 'x-agent-session-id', [
        'sessionId',
        'session_id'
      ]) || detectedContext?.sessionId,
    runId: readClientContextValue(request, metadata, 'x-agent-run-id', ['runId', 'run_id']),
    stepId: readClientContextValue(request, metadata, 'x-agent-step-id', ['stepId', 'step_id']),
    workflow:
      readClientContextValue(request, metadata, 'x-agent-workflow', ['workflow']) ||
      detectedContext?.workflow,
    version:
      readClientContextValue(request, metadata, 'x-agent-version', ['version']) ||
      detectedContext?.version,
    promptVersion: readClientContextValue(request, metadata, 'x-agent-prompt-version', [
      'promptVersion',
      'prompt_version'
    ]),
    clientRequestId:
      readClientContextValue(request, metadata, 'x-client-request-id', [
        'clientRequestId',
        'client_request_id'
      ]) || detectedContext?.clientRequestId,
    traceparent: readClientContextValue(request, metadata, 'traceparent', ['traceparent']),
    tracestate: readClientContextValue(request, metadata, 'tracestate', ['tracestate']),
    metadata: contextMetadata
  };

  return hasGatewayRequestClientContext(clientContext) ? clientContext : undefined;
}

export function buildGatewayBillingTraceSnapshot(
  request: FastifyRequest,
  reply: FastifyReply,
  options: {
    responseBody?: unknown;
    responseStatusCode?: number;
    standardResponse?: StandardResponse;
  } = {}
): GatewayBillingTrace | undefined {
  const requestHeaders = sanitizeHeadersForLog(
    normalizeHeaderBagForTrace(request.headers),
  );
  const responseHeaders = sanitizeHeadersForLog(
    normalizeHeaderBagForTrace(reply.getHeaders()),
  );
  const requestBody = request.body;
  const responseBody =
    options.responseBody !== undefined
      ? sanitizePayloadForLog(options.responseBody)
      : undefined;
  const outputText = sanitizeTraceText(options.standardResponse?.output_text);
  const finishReason = sanitizeTraceText(options.standardResponse?.finish_reason);
  const statusCode =
    typeof options.responseStatusCode === 'number' &&
    Number.isFinite(options.responseStatusCode)
      ? options.responseStatusCode
      : typeof reply.statusCode === 'number' && Number.isFinite(reply.statusCode)
      ? reply.statusCode
      : undefined;

  if (
    Object.keys(requestHeaders).length === 0 &&
    Object.keys(responseHeaders).length === 0 &&
    requestBody === undefined &&
    responseBody === undefined &&
    outputText === undefined &&
    finishReason === undefined &&
    statusCode === undefined
  ) {
    return undefined;
  }

  return {
    request:
      requestBody !== undefined || Object.keys(requestHeaders).length > 0
        ? {
            headers: requestHeaders,
            body: requestBody
          }
        : undefined,
    response:
      responseBody !== undefined ||
      outputText !== undefined ||
      finishReason !== undefined ||
      statusCode !== undefined
        ? {
            statusCode,
            headers: responseHeaders,
            body: responseBody,
            outputText,
            finishReason
          }
        : undefined
  };
}

function normalizeHeaderBagForTrace(headers: Record<string, unknown>): Record<string, string> {
  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (typeof value === 'string') {
      normalized[key] = value;
    } else if (Array.isArray(value)) {
      normalized[key] = value.map((item) => String(item)).join(', ');
    } else if (value !== undefined) {
      normalized[key] = String(value);
    }
  }
  return normalized;
}

function sanitizeTraceText(value?: string): string | undefined {
  const sanitized = sanitizePayloadForLog(value);
  return typeof sanitized === 'string' && sanitized.trim() ? sanitized : undefined;
}

function resolveRawTraceContent(
  mode: GatewayConfig['rawTrace']['mode'],
  value: unknown,
): unknown {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (
    isObject(value) &&
    Object.keys(value).length === 1 &&
    typeof value.raw === 'string'
  ) {
    value = value.raw;
  }

  if (mode === 'body_redacted') {
    if (typeof value === 'string') {
      try {
        return sanitizePayloadForLog(JSON.parse(value));
      } catch {
        return sanitizePayloadForLog(value);
      }
    }
    return sanitizePayloadForLog(value);
  }

  return value;
}

interface RawTraceResponseStreamCapture {
  content: unknown;
  contentType?: string;
}

function cloneResponseForRawStreamTrace(
  config: GatewayConfig,
  response: Response,
): Response | undefined {
  if (
    !config.rawTrace.enabled ||
    config.rawTrace.mode === 'body_redacted' ||
    !isEventStreamResponse(response)
  ) {
    return undefined;
  }

  try {
    return response.clone();
  } catch {
    return undefined;
  }
}

async function collectRawTraceResponseStreamSafe(
  request: FastifyRequest,
  config: GatewayConfig,
  response?: Response,
): Promise<RawTraceResponseStreamCapture | undefined> {
  if (!response || !config.rawTrace.enabled || config.rawTrace.mode === 'body_redacted') {
    return undefined;
  }

  try {
    const content = await response.text();
    if (!content) {
      return undefined;
    }
    return {
      content,
      contentType: response.headers.get('content-type') || 'text/event-stream; charset=utf-8',
    };
  } catch (error) {
    request.log.warn(
      {
        details: error instanceof Error ? error.message : String(error),
      },
      'Failed to collect raw streaming response for raw trace.',
    );
    return undefined;
  }
}

function buildUpstreamRawTraceCapture(
  upstreamRequest: UpstreamRequest,
  upstreamResponse: Response,
  upstreamResponseBody: unknown,
  rawStreamCapture?: RawTraceResponseStreamCapture,
): GatewayRawTraceCapture {
  return {
    upstreamRequest,
    upstreamResponseBody,
    upstreamResponseHeaders: normalizeHeadersForRawTrace(upstreamResponse.headers),
    upstreamResponseStatus: upstreamResponse.status,
    upstreamResponseStream: rawStreamCapture?.content,
    upstreamResponseStreamContentType: rawStreamCapture?.contentType,
  };
}

function normalizeHeadersForRawTrace(
  headers: Headers | Record<string, unknown>,
): Record<string, string> {
  const normalized: Record<string, string> = {};
  const entries = headers instanceof Headers ? Array.from(headers.entries()) : Object.entries(headers);
  for (const [key, value] of entries) {
    if (typeof value === 'string') {
      normalized[key] = value;
    } else if (Array.isArray(value)) {
      normalized[key] = value.map((item) => String(item)).join(', ');
    } else if (value !== undefined) {
      normalized[key] = String(value);
    }
  }
  return normalized;
}

function publishRawTraceCaptureSafe(
  request: FastifyRequest,
  config: GatewayConfig,
  targetProvider: Provider,
  model: string | undefined,
  targetProviderConfig?: ProviderConfig,
  rawTraceCapture?: GatewayRawTraceCapture
): void {
  if (!config.rawTrace.enabled || !rawTraceCapture) {
    return;
  }

  if (!markRawTraceCaptureSubmitted(request)) {
    return;
  }

  const redactionPolicy =
    config.rawTrace.mode === 'body_redacted' ? 'body_redacted' : 'none';
  const traceUrl =
    config.rawTrace.mode === 'wire_raw'
      ? request.url
      : sanitizeRequestUrlForEvent(request.url);
  const upstreamUrl = rawTraceCapture.upstreamRequest?.url;
  const upstreamTraceUrl =
    config.rawTrace.mode === 'wire_raw' || !upstreamUrl
      ? upstreamUrl
      : sanitizeRequestUrlForEvent(upstreamUrl);
  const parts: Parameters<typeof enqueueRawTraceCapture>[0]['parts'] = [
    {
      partType: 'client_request_metadata' as const,
      content: resolveRawTraceContent(config.rawTrace.mode, {
        method: request.method,
        url: traceUrl,
        headers: normalizeHeadersForRawTrace(request.headers as Record<string, unknown>),
      }),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
    {
      partType: 'client_request' as const,
      content: resolveRawTraceContent(
        config.rawTrace.mode,
        readRawRequestBody(request) ?? request.body,
      ),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
    {
      partType: 'upstream_request_metadata' as const,
      content: resolveRawTraceContent(config.rawTrace.mode, {
        method: 'POST',
        url: upstreamTraceUrl,
        headers: rawTraceCapture.upstreamRequest?.headers,
      }),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
    {
      partType: 'upstream_request' as const,
      content: resolveRawTraceContent(
        config.rawTrace.mode,
        rawTraceCapture.upstreamRequest?.body,
      ),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
    {
      partType: 'upstream_response_metadata' as const,
      content: resolveRawTraceContent(config.rawTrace.mode, {
        statusCode: rawTraceCapture.upstreamResponseStatus,
        headers: rawTraceCapture.upstreamResponseHeaders,
      }),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
    {
      partType: 'upstream_response' as const,
      content: resolveRawTraceContent(
        config.rawTrace.mode,
        rawTraceCapture.upstreamResponseBody,
      ),
      contentType: 'application/json; charset=utf-8',
      redactionPolicy,
    },
  ];

  if (
    rawTraceCapture.upstreamResponseStream !== undefined &&
    rawTraceCapture.upstreamResponseStream !== null
  ) {
    parts.push({
      partType: 'response_stream' as const,
      content: rawTraceCapture.upstreamResponseStream,
      contentType:
        rawTraceCapture.upstreamResponseStreamContentType ||
        'text/event-stream; charset=utf-8',
      redactionPolicy: 'none',
    });
  }

  enqueueRawTraceCapture({
    requestId: request.id,
    method: request.method,
    url: traceUrl,
    identity: request.gatewayIdentity,
    clientContext: isObject(request.body)
      ? extractGatewayRequestClientContext(
          request,
          request.body as Record<string, unknown>,
        )
      : undefined,
    target: {
      provider: targetProvider,
      providerName: targetProviderConfig?.name,
      model,
    },
    parts,
  });
}

function resolveGatewayUsageOutcomeStatus(
  statusCode?: number,
): GatewayUsageOutcomeStatus {
  if (statusCode === 429) {
    return 'rate-limited';
  }

  if (statusCode === 408 || statusCode === 504) {
    return 'timeout';
  }

  if (typeof statusCode === 'number' && statusCode >= 200 && statusCode < 400) {
    return 'success';
  }

  return 'error';
}

function resolveGatewayBillingOutcome(
  reply: FastifyReply,
  trace?: GatewayBillingTrace,
): {
  status: GatewayUsageOutcomeStatus;
  statusCode: number;
} | undefined {
  const traceStatusCode =
    trace?.response &&
    typeof trace.response.statusCode === 'number' &&
    Number.isFinite(trace.response.statusCode)
      ? trace.response.statusCode
      : undefined;
  const replyStatusCode =
    typeof reply.statusCode === 'number' && Number.isFinite(reply.statusCode)
      ? reply.statusCode
      : undefined;
  const statusCode = traceStatusCode ?? replyStatusCode;

  if (statusCode === undefined) {
    return undefined;
  }

  return {
    status: resolveGatewayUsageOutcomeStatus(statusCode),
    statusCode,
  };
}

function createZeroBillingResult(provider: Provider): BillingResult {
  return {
    provider,
    currency: 'USD',
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      total_tokens: 0,
      cache_duration_seconds: 0,
    },
    rates: {
      input_per_million_usd: 0,
      output_per_million_usd: 0,
      cache_read_per_million_usd: 0,
      cache_write_per_million_usd: 0,
    },
    cost: {
      input: 0,
      output: 0,
      cache_read: 0,
      cache_write: 0,
      tiered: 0,
      total: 0,
    },
    breakdown: {
      input: [],
      output: [],
      cache_read: [],
      cache_write: [],
    },
  };
}

function sanitizeAttemptFailures(attempts: ProviderAttemptFailure[]) {
  return attempts.map((attempt) => ({
    provider: attempt.provider,
    providerName: attempt.providerName,
    stage: attempt.stage,
    message: attempt.message,
    status:
      typeof attempt.status === 'number' && Number.isFinite(attempt.status)
        ? attempt.status
        : undefined,
    details:
      attempt.details !== undefined
        ? sanitizePayloadForLog(attempt.details)
        : undefined,
  }));
}

function resolveAttemptRawTraceCapture(attempt?: ProviderAttemptFailure): GatewayRawTraceCapture | undefined {
  if (
    !attempt ||
    (attempt.upstreamRequest === undefined &&
      attempt.upstreamResponseBody === undefined)
  ) {
    return undefined;
  }

  return {
    upstreamRequest: attempt.upstreamRequest,
    upstreamResponseBody: attempt.upstreamResponseBody,
    upstreamResponseStatus: attempt.status,
  };
}

function publishBillingEventSafe(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  targetProvider: Provider,
  model: string | undefined,
  fallbackAttempts: number,
  attempt: GatewayBillingAttempt | undefined,
  billing: BillingResult,
  targetProviderConfig?: ProviderConfig,
  trace?: GatewayBillingTrace,
  rawTraceCapture?: GatewayRawTraceCapture
) {
  if (!config.billingQueue.enabled && !config.billingWebhook.enabled) {
    if (config.rawTrace.enabled) {
      publishRawTraceCaptureSafe(
        request,
        config,
        targetProvider,
        model,
        targetProviderConfig,
        rawTraceCapture
      );
    }
    return;
  }

  const resolvedTargetProviderConfig =
    targetProviderConfig || findProviderConfigByType(config.providers, targetProvider);
  const outcome = resolveGatewayBillingOutcome(reply, trace);
  const event = {
    eventId: randomUUID(),
    emittedAt: new Date().toISOString(),
    requestId: request.id,
    attempt,
    route: {
      method: request.method,
      url: sanitizeRequestUrlForEvent(request.url)
    },
    source: {
      provider: sourceProvider,
      adapterKey: sourceAdapterKey
    },
    target: {
      provider: targetProvider,
      model,
      providerName: resolvedTargetProviderConfig?.name
    },
    fallback: {
      used: fallbackAttempts > 0,
      attempts: fallbackAttempts
    },
    performance: {
      latency_ms: resolveLatencyMs(reply)
    },
    identity: request.gatewayIdentity,
    clientContext: extractGatewayRequestClientContext(request, request.body as Record<string, unknown>),
    trace,
    outcome,
    billing
  };

  void publishBillingEvent(event)
    .then((published) => {
      if (published) {
        request.log.debug(
          {
            requestId: event.requestId,
            eventId: event.eventId,
            queueEnabled: config.billingQueue.enabled,
            webhookEnabled: config.billingWebhook.enabled
          },
          'Billing event delivered.'
        );
      }
    })
    .catch((error) => {
      request.log.warn(
        {
          requestId: request.id,
          provider: targetProvider,
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to deliver billing event.'
      );
    });

  publishRawTraceCaptureSafe(
    request,
    config,
    targetProvider,
    model,
    targetProviderConfig,
    rawTraceCapture
  );
}

function publishRequestFailureEventSafe(
  request: FastifyRequest,
  reply: FastifyReply,
  config: GatewayConfig,
  sourceProvider: Provider,
  sourceAdapterKey: string,
  targetProvider: Provider,
  model: string | undefined,
  fallbackAttempts: number,
  attempts: ProviderAttemptFailure[],
  errorMessage: string,
  targetProviderConfig?: ProviderConfig,
  trace?: GatewayBillingTrace,
  rawTraceCapture?: GatewayRawTraceCapture
) {
  if (!config.billingQueue.enabled && !config.billingWebhook.enabled) {
    if (config.rawTrace.enabled) {
      publishRawTraceCaptureSafe(
        request,
        config,
        targetProvider,
        model,
        targetProviderConfig,
        rawTraceCapture
      );
    }
    return;
  }

  const outcome = resolveGatewayBillingOutcome(reply, trace);
  const event = {
    eventId: randomUUID(),
    emittedAt: new Date().toISOString(),
    requestId: request.id,
    route: {
      method: request.method,
      url: sanitizeRequestUrlForEvent(request.url)
    },
    source: {
      provider: sourceProvider,
      adapterKey: sourceAdapterKey
    },
    target: {
      provider: targetProvider,
      model,
      providerName: targetProviderConfig?.name
    },
    fallback: {
      used: fallbackAttempts > 0,
      attempts: fallbackAttempts
    },
    performance: {
      latency_ms: resolveLatencyMs(reply)
    },
    identity: request.gatewayIdentity,
    clientContext: extractGatewayRequestClientContext(request, request.body as Record<string, unknown>),
    trace,
    outcome: {
      status: outcome?.status || 'error',
      statusCode: outcome?.statusCode,
      errorMessage
    },
    ...(attempts.length > 0
      ? {
          attempts: sanitizeAttemptFailures(attempts)
        }
      : {}),
    billing: createZeroBillingResult(targetProvider)
  };

  void publishBillingEvent(event)
    .then((published) => {
      if (published) {
        request.log.debug(
          {
            requestId: event.requestId,
            eventId: event.eventId,
            queueEnabled: config.billingQueue.enabled,
            webhookEnabled: config.billingWebhook.enabled,
            status: event.outcome.status
          },
          'Failed request billing event delivered.'
        );
      }
    })
    .catch((error) => {
      request.log.warn(
        {
          requestId: request.id,
          provider: targetProvider,
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to deliver failed request billing event.'
      );
    });

  publishRawTraceCaptureSafe(
    request,
    config,
    targetProvider,
    model,
    targetProviderConfig,
    rawTraceCapture
  );
}

function publishFailedAttemptEventSafe(
  request: FastifyRequest,
  _reply: FastifyReply,
  config: GatewayConfig,
  _sourceProvider: Provider,
  _sourceAdapterKey: string,
  attempt: ProviderAttemptFailure,
  model: string | undefined,
  _attemptSequence: number,
  _attempts: ProviderAttemptFailure[],
  targetProviderConfig?: ProviderConfig,
) {
  if (!config.rawTrace.enabled) {
    return;
  }

  const rawTraceCapture = resolveAttemptRawTraceCapture(attempt);
  if (!rawTraceCapture) {
    return;
  }

  publishRawTraceCaptureSafe(
    request,
    config,
    attempt.provider,
    model,
    targetProviderConfig,
    rawTraceCapture,
  );
}

function resolveLatencyMs(reply: FastifyReply): number | undefined {
  const value = typeof reply.elapsedTime === 'number' ? reply.elapsedTime : undefined;
  if (value === undefined || !Number.isFinite(value) || value < 0) {
    return undefined;
  }

  return Math.round(value);
}

function sanitizeRequestUrlForEvent(url: string): string {
  try {
    const parsed = new URL(url, 'http://gateway.local');
    for (const key of ['key', 'api_key', 'apikey', 'token', 'access_token']) {
      if (parsed.searchParams.has(key)) {
        parsed.searchParams.set(key, '***');
      }
    }

    const query = parsed.searchParams.toString();
    return query ? `${parsed.pathname}?${query}` : parsed.pathname;
  } catch {
    return url;
  }
}

function mergeJsonObjects(
  base: Record<string, unknown>,
  override: Record<string, unknown>
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...base };
  for (const [key, overrideValue] of Object.entries(override)) {
    const baseValue = merged[key];
    if (isPlainObject(baseValue) && isPlainObject(overrideValue)) {
      merged[key] = mergeJsonObjects(baseValue, overrideValue);
      continue;
    }

    merged[key] = cloneUnknown(overrideValue);
  }

  return merged;
}

function cloneUnknown(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => cloneUnknown(item));
  }

  if (isPlainObject(value)) {
    const cloned: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      cloned[key] = cloneUnknown(item);
    }
    return cloned;
  }

  return value;
}

function trimRightSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readClientContextValue(
  request: FastifyRequest,
  metadata: Record<string, unknown> | undefined,
  headerName: string | string[],
  metadataKeys: string[]
): string | undefined {
  const headerNames = Array.isArray(headerName) ? headerName : [headerName];
  for (const name of headerNames) {
    const headerValue = readHeader(request.headers[name]);
    if (headerValue && headerValue.trim()) {
      return headerValue.trim();
    }
  }

  if (!metadata) {
    return undefined;
  }

  for (const key of metadataKeys) {
    const value = asString(metadata[key]);
    if (value && value.trim()) {
      return value.trim();
    }
  }

  return undefined;
}

function detectGatewayAgentClientContext(
  request: FastifyRequest,
  body: Record<string, unknown>,
  metadata?: Record<string, unknown>
): GatewayRequestClientContext | undefined {
  const userAgent = readHeader(request.headers['user-agent']);
  const agentHint = [
    readHeader(request.headers['x-agent-type']),
    readHeader(request.headers['x-agent-name']),
    readHeader(request.headers['x-client-agent']),
    readHeader(request.headers['x-client-name']),
    readHeader(request.headers['anthropic-client-name']),
    readKnownAgentMetadataValue(metadata)
  ]
    .filter((value): value is string => Boolean(value && value.trim()))
    .join(' ');
  const detectionInput = [
    userAgent,
    agentHint,
    typeof request.url === 'string' ? request.url : undefined
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  const hasCodexHeader = Boolean(
    readHeader(request.headers['x-codex-access-token']) ||
      readHeader(request.headers['x-codex-refresh-token']) ||
      readHeader(request.headers['x-codex-account-id'])
  );
  const hasClaudeCodeHeader = Boolean(
    readHeader(request.headers['x-claude-code-session-id']) ||
      readHeader(request.headers['x-claude-session-id'])
  );

  const detectedAgent =
    hasCodexHeader || /\bcodex\b/.test(detectionInput)
      ? 'codex'
      : hasClaudeCodeHeader ||
          /claude[-_\s]?code/.test(detectionInput) ||
          /claude[-_\s]?cli/.test(detectionInput)
        ? 'claude-code'
        : undefined;

  if (!detectedAgent) {
    return undefined;
  }

  const sessionId = readDetectedAgentSessionId(request, body, metadata, detectedAgent);
  const version = readDetectedAgentVersion(userAgent, detectedAgent);
  const clientRequestId = readClientContextValue(
    request,
    metadata,
    ['x-request-id', 'request-id', 'x-client-request-id'],
    ['clientRequestId', 'client_request_id', 'requestId', 'request_id']
  );
  const detectionMetadata: Record<string, unknown> = {
    agent: detectedAgent,
    source: resolveDetectedAgentSource(request, agentHint, userAgent, detectedAgent)
  };

  if (userAgent?.trim()) {
    detectionMetadata.userAgent = userAgent.trim();
  }
  if (sessionId) {
    detectionMetadata.sessionId = sessionId;
  }
  if (version) {
    detectionMetadata.version = version;
  }

  return {
    agentId: detectedAgent,
    sessionId,
    workflow: detectedAgent,
    version,
    clientRequestId,
    metadata: {
      agentDetection: detectionMetadata
    }
  };
}

function readKnownAgentMetadataValue(
  metadata?: Record<string, unknown>
): string | undefined {
  if (!metadata) {
    return undefined;
  }

  for (const key of [
    'agent',
    'agentName',
    'agent_name',
    'agentType',
    'agent_type',
    'client',
    'clientName',
    'client_name',
    'app',
    'application',
    'source'
  ]) {
    const value = asString(metadata[key]);
    if (value && value.trim()) {
      return value.trim();
    }
  }

  return undefined;
}

function readDetectedAgentSessionId(
  request: FastifyRequest,
  body: Record<string, unknown>,
  metadata: Record<string, unknown> | undefined,
  agentId: string
): string | undefined {
  const explicitSessionId = readClientContextValue(
    request,
    metadata,
    [
      'x-agent-session-id',
      'x-session-id',
      'x-conversation-id',
      'x-thread-id',
      'x-codex-session-id',
      'x-codex-conversation-id',
      'x-claude-code-session-id',
      'x-claude-session-id'
    ],
    ['sessionId', 'session_id', 'conversationId', 'conversation_id', 'threadId', 'thread_id']
  );
  if (explicitSessionId) {
    return explicitSessionId;
  }

  const bodySessionId =
    readStringAtObjectPath(body, ['sessionId']) ||
    readStringAtObjectPath(body, ['session_id']) ||
    readStringAtObjectPath(body, ['conversationId']) ||
    readStringAtObjectPath(body, ['conversation_id']) ||
    readStringAtObjectPath(body, ['threadId']) ||
    readStringAtObjectPath(body, ['thread_id']) ||
    readBodyConversationId(body);
  if (bodySessionId) {
    return bodySessionId;
  }

  if (agentId === 'codex') {
    return readStringAtObjectPath(body, ['previous_response_id']);
  }

  return undefined;
}

function readBodyConversationId(body: Record<string, unknown>): string | undefined {
  const conversation = body.conversation;
  if (typeof conversation === 'string' && conversation.trim()) {
    return conversation.trim();
  }

  if (isPlainObject(conversation)) {
    return (
      readStringAtObjectPath(conversation, ['id']) ||
      readStringAtObjectPath(conversation, ['conversationId']) ||
      readStringAtObjectPath(conversation, ['conversation_id'])
    );
  }

  return undefined;
}

function readStringAtObjectPath(
  source: Record<string, unknown> | undefined,
  path: string[]
): string | undefined {
  let current: unknown = source;
  for (const key of path) {
    if (!isPlainObject(current)) {
      return undefined;
    }
    current = current[key];
  }

  return typeof current === 'string' && current.trim() ? current.trim() : undefined;
}

function readDetectedAgentVersion(
  userAgent: string | undefined,
  agentId: string
): string | undefined {
  if (!userAgent) {
    return undefined;
  }

  const pattern =
    agentId === 'codex'
      ? /(?:codex(?:[-_/ ]?(?:cli|rs|code))*|openai[-_/ ]?codex)\/?v?([0-9][^\s;)]+)/i
      : /(?:claude[-_/ ]?code|claude[-_/ ]?cli)\/?v?([0-9][^\s;)]+)/i;
  const match = userAgent.match(pattern);
  return match?.[1]?.trim();
}

function resolveDetectedAgentSource(
  request: FastifyRequest,
  agentHint: string,
  userAgent: string | undefined,
  agentId: string
): string {
  if (
    agentId === 'codex' &&
    (readHeader(request.headers['x-codex-access-token']) ||
      readHeader(request.headers['x-codex-refresh-token']) ||
      readHeader(request.headers['x-codex-account-id']))
  ) {
    return 'codex_headers';
  }

  if (
    agentId === 'claude-code' &&
    (readHeader(request.headers['x-claude-code-session-id']) ||
      readHeader(request.headers['x-claude-session-id']))
  ) {
    return 'claude_code_headers';
  }

  if (agentHint.trim()) {
    return 'agent_hint';
  }

  return userAgent?.trim() ? 'user_agent' : 'request';
}

function mergeGatewayClientContextMetadata(
  metadata: Record<string, unknown> | undefined,
  detectedMetadata?: Record<string, unknown>
): Record<string, unknown> | undefined {
  if (!detectedMetadata || Object.keys(detectedMetadata).length === 0) {
    return metadata;
  }

  return {
    ...(metadata || {}),
    ...detectedMetadata
  };
}

function hasGatewayRequestClientContext(value: GatewayRequestClientContext): boolean {
  return Object.values(value).some((item) => {
    if (typeof item === 'string') {
      return item.trim().length > 0;
    }

    return isPlainObject(item) && Object.keys(item).length > 0;
  });
}

function canRelayLiveConvertedStream(
  source: GatewaySourceContext,
  targetProvider: Provider,
  targetProviderConfig?: ProviderConfig
): boolean {
  if (source.adapterKey === 'openai_chat' && targetProvider === 'anthropic') {
    return true;
  }

  if (source.adapterKey === 'anthropic_messages' && targetProvider === 'openai') {
    return true;
  }

  if (
    source.adapterKey === 'openai_responses' &&
    targetProvider === 'openai' &&
    targetProviderConfig?.type === 'openai_chat_completions'
  ) {
    return true;
  }

  if (
    source.adapterKey === 'openai_chat' &&
    targetProvider === 'openai' &&
    targetProviderConfig?.type === 'openai_responses'
  ) {
    return true;
  }

  if (source.adapterKey === 'gemini_stream' && targetProvider === 'openai') {
    return true;
  }

  return false;
}

function canPassthroughWithoutProtocolConversion(
  sourceAdapterKey: string,
  sourceProvider: Provider,
  targetProvider: Provider,
  targetProviderConfig?: ProviderConfig
): boolean {
  if (sourceProvider !== targetProvider) {
    return false;
  }

  if (sourceProvider !== 'openai') {
    return true;
  }

  const targetProtocol = targetProviderConfig?.type;
  if (!targetProtocol) {
    return true;
  }

  if (sourceAdapterKey === 'openai_responses') {
    return targetProtocol === 'openai_responses';
  }

  if (sourceAdapterKey === 'openai_chat') {
    return targetProtocol === 'openai_chat_completions';
  }

  return true;
}

function isEventStreamResponse(response: Response): boolean {
  const contentType = response.headers.get('content-type');
  return Boolean(contentType && contentType.toLowerCase().includes('text/event-stream'));
}

function canTreatAsLiveStreamResponse(
  source: GatewaySourceContext,
  targetProvider: Provider,
  response: Response
): boolean {
  if (isEventStreamResponse(response)) {
    return true;
  }

  const contentType = (response.headers.get('content-type') || '').toLowerCase();
  const isOpenAIStreamSource =
    source.adapterKey === 'openai_responses' || source.adapterKey === 'openai_chat';
  if (
    isOpenAIStreamSource &&
    targetProvider === 'openai' &&
    contentType.length > 0 &&
    !contentType.includes('application/json')
  ) {
    return true;
  }

  if (isOpenAIStreamSource && targetProvider === 'openai' && contentType.length === 0) {
    return true;
  }

  return false;
}

function shouldForceEventStreamHeaders(source: GatewaySourceContext, streaming: boolean): boolean {
  if (!streaming) {
    return false;
  }

  return source.adapterKey === 'openai_responses' || source.adapterKey === 'openai_chat';
}

async function normalizeOpenAIPayloadForResponseParseRecovery(
  request: FastifyRequest,
  provider: Provider,
  payload: unknown
): Promise<unknown> {
  if (provider !== 'openai' || !isObject(payload)) {
    return payload;
  }

  const raw = typeof payload.raw === 'string' ? payload.raw : undefined;
  if (!raw || !looksLikeOpenAIEventStreamRawPayload(raw)) {
    return payload;
  }

  try {
    const recovered = await collectOpenAINonStreamPayloadFromEventStream(
      new Response(raw, {
        status: 200,
        headers: {
          'content-type': 'text/event-stream; charset=utf-8'
        }
      })
    );

    request.log.info(
      {
        provider,
        recovered: true
      },
      'Recovered OpenAI non-stream payload from raw SSE text.'
    );
    return recovered;
  } catch (error) {
    request.log.warn(
      {
        provider,
        details: error instanceof Error ? error.message : String(error)
      },
      'Failed to recover OpenAI payload from raw SSE text.'
    );
    return payload;
  }
}

function looksLikeOpenAIEventStreamRawPayload(raw: string): boolean {
  const trimmed = raw.trim();
  if (!trimmed) {
    return false;
  }

  return /(^|\n)\s*(event|data)\s*:/.test(trimmed);
}
