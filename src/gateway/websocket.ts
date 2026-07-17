import type { IncomingHttpHeaders, IncomingMessage } from 'node:http';
import type { Socket } from 'node:net';
import { URL } from 'node:url';
import type { FastifyInstance, FastifyRequest } from 'fastify';
import { WebSocket, WebSocketServer, type RawData } from 'ws';
import {
  buildOpenAIHeaders,
  normalizeOpenAIResponsesCompletedEventPayload
} from '../adapters/builtins/common';
import type {
  GatewayConfig,
  GatewaySourceContext,
  HeaderBag,
  Provider,
  ProviderConfig,
  ProviderPlugin,
  UpstreamRequest
} from '../types';
import { err, ok, type Result } from '../types';
import { findDefaultProviderConfig, parseProvider, providerFromProviderType } from '../utils';
import { authenticateGatewayRequest, evaluateApiKeyModelRestriction } from './auth';
import {
  parseGatewayCodexWsSourceAdapterKey,
  transformClientMessageToCodexRequest,
  type GatewayCodexWsSourceAdapterKey
} from './codex-websocket-conversion';
import { evaluateGatewayPrecheck } from './precheck';
import type { GatewayRuntime } from './runtime';

interface GatewaySocketContext {
  headers: IncomingHttpHeaders;
  requestUrl: string;
  request: FastifyRequest;
  sourceAdapterHint?: GatewayCodexWsSourceAdapterKey;
  targetProviderConfig?: ProviderConfig;
}

const blockedForwardHeaderSet = new Set([
  'host',
  'connection',
  'upgrade',
  'keep-alive',
  'proxy-connection',
  'transfer-encoding',
  'content-length',
  'content-type',
  'authorization',
  'proxy-authorization',
  'sec-websocket-key',
  'sec-websocket-version',
  'sec-websocket-extensions',
  'sec-websocket-protocol',
  'sec-websocket-accept'
]);

const internalGatewayQueryParamSet = new Set(['source_adapter', 'source']);
const codexDefaultInstructions = 'You are a helpful assistant.';
const pendingDownstreamMessageLimit = 16;
type WebSocketPayload = RawData | string;

export function registerGatewayResponsesWebSocketRoute(
  fastify: FastifyInstance,
  config: GatewayConfig,
  runtime?: Pick<GatewayRuntime, 'providerPlugins'>
): void {
  const maxPayload = resolveGatewayWebSocketMaxPayloadBytes(config);
  const websocketServer = new WebSocketServer({ noServer: true, maxPayload });
  const socketContext = new WeakMap<WebSocket, GatewaySocketContext>();

  const onUpgrade = (request: IncomingMessage, socket: Socket, head: Buffer): void => {
    const requestUrl = safeParseRequestUrl(request);
    if (!requestUrl || !isResponsesWebSocketPath(requestUrl.pathname)) {
      return;
    }

    void authorizeAndUpgrade(request, socket, head, requestUrl);
  };

  async function authorizeAndUpgrade(
    request: IncomingMessage,
    socket: Socket,
    head: Buffer,
    requestUrl: URL
  ): Promise<void> {
    let authResult;
    try {
      authResult = await authenticateGatewayRequest(
        {
          headers: request.headers,
          method: request.method || 'GET',
          url: request.url || requestUrl.pathname,
          ip: request.socket.remoteAddress || ''
        } as FastifyRequest,
        config.auth
      );
    } catch (error) {
      fastify.log.warn(
        {
          details: error instanceof Error ? error.message : String(error)
        },
        'Gateway websocket auth check failed unexpectedly.'
      );
      rejectUpgrade(socket, 500, 'Gateway websocket auth check failed.');
      return;
    }

    if (!authResult.ok) {
      rejectUpgrade(socket, authResult.statusCode || 401, authResult.error || 'Unauthorized');
      return;
    }

    websocketServer.handleUpgrade(request, socket, head, (ws) => {
      const pluginCompatibleRequest = createWebSocketPluginCompatibleRequest(request, requestUrl, fastify);
      pluginCompatibleRequest.gatewayIdentity = authResult.ok ? authResult.identity : undefined;
      pluginCompatibleRequest.gatewayApiKeyRestrictions = authResult.ok
        ? authResult.apiKeyRestrictions
        : undefined;
      socketContext.set(ws, {
        headers: request.headers,
        requestUrl: requestUrl.toString(),
        request: pluginCompatibleRequest,
        sourceAdapterHint: readSourceAdapterHintFromRequestUrl(requestUrl)
      });
      websocketServer.emit('connection', ws, request);
    });
  }

  websocketServer.on('connection', (downstreamSocket) => {
    void initializeWebSocketRelay(downstreamSocket);
  });

  async function initializeWebSocketRelay(downstreamSocket: WebSocket): Promise<void> {
    const context = socketContext.get(downstreamSocket);
    if (!context) {
      downstreamSocket.close(1008, 'Unauthorized');
      return;
    }

    let upstreamSocket: WebSocket | undefined;
    try {
      const upstreamTarget = resolveResponsesWebSocketTarget(config, context);
      context.targetProviderConfig = upstreamTarget.providerConfig;
      const upstreamUrl = buildResponsesUpstreamUrl(upstreamTarget.baseUrl, context.requestUrl);
      const upstreamHeaders = buildUpstreamHeaders(context.headers, {
        openaiApiKey: upstreamTarget.apiKey,
        auth: config.auth,
        allowEnvApiKeyFallback: !looksLikeCodexBaseUrl(upstreamTarget.baseUrl)
      });
      const pluginContext: WebSocketProviderPluginContext = {
        request: context.request,
        config,
        source: {
          adapterKey: context.sourceAdapterHint || 'openai_responses'
        },
        sourceProvider: 'openai',
        sourceAdapterKey: context.sourceAdapterHint || 'openai_responses',
        targetProvider: 'openai',
        targetProviderConfig: upstreamTarget.providerConfig,
        model: undefined,
        passthrough: true,
        streaming: true,
        plugins:
          runtime?.providerPlugins.resolve('openai', upstreamTarget.providerConfig?.name) || []
      };
      const upstreamRequestResult = await applyWebSocketProviderRequestPlugins(
        pluginContext,
        {
          url: upstreamUrl,
          headers: upstreamHeaders,
          body: {}
        }
      );
      if (!upstreamRequestResult.ok) {
        fastify.log.warn(
          {
            details: upstreamRequestResult.error,
            providerName: upstreamTarget.providerConfig?.name
          },
          'Gateway responses websocket provider plugin auth failed.'
        );
        downstreamSocket.close(1011, `Failed to init upstream websocket: ${upstreamRequestResult.error}`);
        return;
      }

      const normalizedWebSocketUrl = normalizeUrlForWebSocket(upstreamRequestResult.value.url);
      upstreamSocket = new WebSocket(normalizedWebSocketUrl, {
        headers: upstreamRequestResult.value.headers,
        maxPayload: resolveGatewayWebSocketMaxPayloadBytes(config)
      });
    } catch (error) {
      downstreamSocket.close(1011, `Failed to init upstream websocket: ${toErrorMessage(error)}`);
      return;
    }

    bindSocketRelay(downstreamSocket, upstreamSocket, fastify, context, config);
  }

  fastify.server.on('upgrade', onUpgrade);
  fastify.addHook('onClose', async () => {
    fastify.server.off('upgrade', onUpgrade);
    for (const client of websocketServer.clients) {
      client.terminate();
    }
    await new Promise<void>((resolve) => {
      websocketServer.close(() => resolve());
    });
  });
}

function bindSocketRelay(
  downstreamSocket: WebSocket,
  upstreamSocket: WebSocket,
  fastify: FastifyInstance,
  context: GatewaySocketContext,
  config: GatewayConfig
): void {
  const pendingDownstreamMessages: Array<{
    payload: WebSocketPayload;
    binary: boolean;
    byteLength: number;
  }> = [];
  const pendingDownstreamByteLimit = resolveGatewayWebSocketMaxPayloadBytes(config);
  let pendingDownstreamBytes = 0;
  let pendingUpstreamToDownstreamSends = 0;
  let upstreamCloseForceTimer: NodeJS.Timeout | undefined;
  let downstreamMessageQueue = Promise.resolve();
  let upstreamClosePending:
    | {
        code: number;
        reason: Buffer | string;
      }
    | undefined;

  const closePeer = (peer: WebSocket, code: number, reason: Buffer | string): void => {
    const normalizedCode = normalizeCloseCode(code);
    const reasonText = typeof reason === 'string' ? reason : reason.toString('utf8');
    if (peer.readyState === WebSocket.OPEN || peer.readyState === WebSocket.CONNECTING) {
      try {
        peer.close(normalizedCode, reasonText);
      } catch {
        peer.terminate();
      }
    }
  };

  const flushUpstreamCloseToDownstream = (): void => {
    if (!upstreamClosePending) {
      return;
    }

    if (pendingUpstreamToDownstreamSends > 0) {
      return;
    }

    const { code, reason } = upstreamClosePending;
    upstreamClosePending = undefined;
    if (upstreamCloseForceTimer) {
      clearTimeout(upstreamCloseForceTimer);
      upstreamCloseForceTimer = undefined;
    }
    closePeer(downstreamSocket, code, reason);
  };

  const sendMessageToUpstream = (payload: WebSocketPayload, binary: boolean): void => {
    upstreamSocket.send(payload, { binary }, (error) => {
      if (error) {
        closePeer(downstreamSocket, 1011, 'Upstream send failed.');
      }
    });
  };

  const flushPendingDownstreamMessages = (): void => {
    if (upstreamSocket.readyState !== WebSocket.OPEN) {
      return;
    }

    while (pendingDownstreamMessages.length > 0) {
      const next = pendingDownstreamMessages.shift();
      if (!next) {
        return;
      }

      pendingDownstreamBytes = Math.max(0, pendingDownstreamBytes - next.byteLength);
      sendMessageToUpstream(next.payload, next.binary);
    }
  };

  downstreamSocket.on('message', (raw, isBinary) => {
    downstreamMessageQueue = downstreamMessageQueue
      .then(() => handleDownstreamMessage(raw, isBinary))
      .catch((error) => {
        fastify.log.warn(
          {
            details: toErrorMessage(error)
          },
          'Gateway responses websocket downstream message handling failed.'
        );
        closePeer(downstreamSocket, 1011, 'Gateway websocket message handling failed.');
      });
  });

  async function handleDownstreamMessage(raw: RawData, isBinary: boolean): Promise<void> {
    const messageForUpstream = await buildMessageForUpstream(
      raw,
      isBinary,
      context,
      downstreamSocket,
      fastify,
      config
    );
    if (!messageForUpstream) {
      return;
    }

    if (upstreamSocket.readyState === WebSocket.CONNECTING) {
      const byteLength = webSocketPayloadByteLength(messageForUpstream.payload);
      if (
        pendingDownstreamMessages.length >= pendingDownstreamMessageLimit ||
        pendingDownstreamBytes + byteLength > pendingDownstreamByteLimit
      ) {
        closePeer(downstreamSocket, 1013, 'Gateway websocket upstream connection is not ready.');
        closePeer(upstreamSocket, 1013, 'Gateway websocket upstream connection is not ready.');
        return;
      }

      pendingDownstreamMessages.push({
        ...messageForUpstream,
        byteLength
      });
      pendingDownstreamBytes += byteLength;
      return;
    }

    if (upstreamSocket.readyState !== WebSocket.OPEN) {
      return;
    }

    sendMessageToUpstream(messageForUpstream.payload, messageForUpstream.binary);
  }

  upstreamSocket.on('message', (raw, isBinary) => {
    if (downstreamSocket.readyState !== WebSocket.OPEN) {
      return;
    }

    const messageForDownstream = buildMessageForDownstream(raw, isBinary);
    pendingUpstreamToDownstreamSends += 1;
    downstreamSocket.send(messageForDownstream.payload, { binary: messageForDownstream.binary }, (error) => {
      pendingUpstreamToDownstreamSends = Math.max(0, pendingUpstreamToDownstreamSends - 1);
      if (error) {
        closePeer(upstreamSocket, 1011, 'Downstream send failed.');
      }

      flushUpstreamCloseToDownstream();
    });
  });

  downstreamSocket.on('close', (code, reason) => {
    if (upstreamCloseForceTimer) {
      clearTimeout(upstreamCloseForceTimer);
      upstreamCloseForceTimer = undefined;
    }
    closePeer(upstreamSocket, code, reason);
  });

  upstreamSocket.on('close', (code, reason) => {
    upstreamClosePending = { code, reason };
    if (pendingUpstreamToDownstreamSends > 0 && !upstreamCloseForceTimer) {
      upstreamCloseForceTimer = setTimeout(() => {
        upstreamCloseForceTimer = undefined;
        if (!upstreamClosePending) {
          return;
        }

        const pendingClose = upstreamClosePending;
        upstreamClosePending = undefined;
        closePeer(downstreamSocket, pendingClose.code, pendingClose.reason);
      }, 500);
      upstreamCloseForceTimer.unref?.();
    }
    flushUpstreamCloseToDownstream();
  });

  upstreamSocket.on('open', () => {
    flushPendingDownstreamMessages();
  });

  downstreamSocket.on('error', (error) => {
    fastify.log.warn(
      {
        details: toErrorMessage(error)
      },
      'Gateway responses websocket downstream error.'
    );
    upstreamSocket.terminate();
  });

  upstreamSocket.on('error', (error) => {
    fastify.log.warn(
      {
        details: toErrorMessage(error)
      },
      'Gateway responses websocket upstream error.'
    );
    if (downstreamSocket.readyState === WebSocket.OPEN) {
      downstreamSocket.close(1011, 'Gateway upstream websocket error.');
    } else if (downstreamSocket.readyState === WebSocket.CONNECTING) {
      downstreamSocket.terminate();
    }
  });
}

function buildMessageForDownstream(
  raw: RawData,
  isBinary: boolean
): { payload: WebSocketPayload; binary: boolean } {
  if (isBinary) {
    return {
      payload: raw,
      binary: true
    };
  }

  return {
    payload: normalizeResponseCompletedTextPayload(rawDataToUtf8String(raw)),
    binary: false
  };
}

function normalizeResponseCompletedTextPayload(payload: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return payload;
  }

  if (!isRecord(parsed)) {
    return payload;
  }

  const normalized = normalizeOpenAIResponsesCompletedEventPayload(parsed);
  return normalized === parsed ? payload : JSON.stringify(normalized);
}

async function buildMessageForUpstream(
  raw: RawData,
  isBinary: boolean,
  context: GatewaySocketContext,
  downstreamSocket: WebSocket,
  fastify: FastifyInstance,
  config: GatewayConfig
): Promise<{ payload: WebSocketPayload; binary: boolean } | undefined> {
  if (isBinary) {
    return {
      payload: raw,
      binary: true
    };
  }

  const textPayload = rawDataToUtf8String(raw);
  const transformed = transformClientMessageToCodexRequest(textPayload, {
    sourceAdapterHint: context.sourceAdapterHint
  });
  if (transformed.kind === 'error') {
    sendInvalidRequestEvent(downstreamSocket, transformed.message);
    return undefined;
  }

  if (transformed.kind === 'converted') {
    fastify.log.debug(
      {
        sourceAdapterKey: transformed.sourceAdapterKey
      },
      'Gateway websocket request converted to Codex response.create.'
    );
  }

  const normalizedPayload = maybeNormalizeCodexResponseCreatePayload(
    transformed.payload,
    config.openaiBaseUrl
  );
  const guardResult = await evaluateWebSocketResponseCreateGuards(
    context,
    normalizedPayload,
    config
  );
  if (!guardResult.ok) {
    sendWebSocketErrorEvent(
      downstreamSocket,
      guardResult.statusCode,
      guardResult.message
    );
    return undefined;
  }

  return {
    payload: normalizedPayload,
    binary: false
  };
}

async function evaluateWebSocketResponseCreateGuards(
  context: GatewaySocketContext,
  payload: string,
  config: GatewayConfig
): Promise<{ ok: true } | { ok: false; statusCode: number; message: string }> {
  const responseCreatePayload = parseResponseCreatePayload(payload);
  if (!responseCreatePayload) {
    return { ok: true };
  }

  const model = readStringField(responseCreatePayload.model);
  const modelRestriction = evaluateApiKeyModelRestriction(context.request, model, {
    provider: 'openai',
    providerConfig: context.targetProviderConfig
  });
  if (!modelRestriction.ok) {
    return {
      ok: false,
      statusCode: modelRestriction.statusCode,
      message: modelRestriction.error
    };
  }

  const precheck = await evaluateGatewayPrecheck({
    request: context.request,
    config,
    targetProvider: 'openai',
    targetProviderConfig: context.targetProviderConfig,
    model,
    requestBody: responseCreatePayload
  });
  if (!precheck.ok) {
    return {
      ok: false,
      statusCode: precheck.statusCode,
      message: precheck.message
    };
  }

  return { ok: true };
}

function parseResponseCreatePayload(payload: string): Record<string, unknown> | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return undefined;
  }

  return isRecord(parsed) && parsed.type === 'response.create'
    ? parsed
    : undefined;
}

function readStringField(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim();
  return normalized || undefined;
}

function buildUpstreamHeaders(
  incomingHeaders: IncomingHttpHeaders,
  config: Pick<GatewayConfig, 'openaiApiKey' | 'auth'> & {
    allowEnvApiKeyFallback?: boolean;
  }
): Record<string, string> {
  const authHeaders = buildOpenAIHeaders(
    withCodexAuthorizationOverride(incomingHeaders) as HeaderBag,
    config
  );
  if (!authHeaders.ok) {
    throw new Error(authHeaders.error);
  }

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(incomingHeaders)) {
    const normalizedKey = key.toLowerCase();
    if (blockedForwardHeaderSet.has(normalizedKey)) {
      continue;
    }

    const headerValue = normalizeHeaderValue(value);
    if (!headerValue) {
      continue;
    }

    headers[normalizedKey] = headerValue;
  }

  headers.authorization = authHeaders.value.authorization;
  const organization = authHeaders.value['openai-organization'];
  if (organization) {
    headers['openai-organization'] = organization;
  }
  const project = authHeaders.value['openai-project'];
  if (project) {
    headers['openai-project'] = project;
  }
  const codexAccountId =
    normalizeHeaderValue(incomingHeaders['chatgpt-account-id']) ||
    normalizeHeaderValue(incomingHeaders['x-codex-account-id']);
  if (codexAccountId) {
    headers['chatgpt-account-id'] = codexAccountId;
  }

  return headers;
}

function buildResponsesUpstreamUrl(openAIBaseUrl: string, requestUrl: string): string {
  const parsedBase = new URL(openAIBaseUrl);
  const normalizedPath = parsedBase.pathname.replace(/\/+$/, '');
  const upstreamPath = normalizedPath.endsWith('/responses')
    ? normalizedPath
    : `${normalizedPath}/responses`;
  parsedBase.pathname = upstreamPath;

  const incoming = new URL(requestUrl, 'http://gateway.local');
  const mergedParams = new URLSearchParams(parsedBase.search);
  for (const [key, value] of incoming.searchParams.entries()) {
    if (internalGatewayQueryParamSet.has(key.toLowerCase())) {
      continue;
    }

    mergedParams.set(key, value);
  }
  parsedBase.search = mergedParams.toString();

  switch (parsedBase.protocol) {
    case 'http:':
      parsedBase.protocol = 'ws:';
      break;
    case 'https:':
      parsedBase.protocol = 'wss:';
      break;
    default:
      break;
  }

  return parsedBase.toString();
}

function normalizeUrlForWebSocket(url: string): string {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }

  if (parsed.protocol === 'http:') {
    parsed.protocol = 'ws:';
    return parsed.toString();
  }

  if (parsed.protocol === 'https:') {
    parsed.protocol = 'wss:';
    return parsed.toString();
  }

  return parsed.toString();
}

function withCodexAuthorizationOverride(headers: IncomingHttpHeaders): IncomingHttpHeaders {
  const cloned: IncomingHttpHeaders = { ...headers };
  const codexAccessToken = normalizeHeaderValue(cloned['x-codex-access-token']);
  if (!codexAccessToken) {
    return cloned;
  }

  cloned.authorization = `Bearer ${codexAccessToken}`;
  return cloned;
}

function maybeNormalizeCodexResponseCreatePayload(payload: string, openAIBaseUrl: string): string {
  if (!looksLikeCodexBaseUrl(openAIBaseUrl)) {
    return payload;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return payload;
  }

  if (!isRecord(parsed) || parsed.type !== 'response.create') {
    return payload;
  }

  const normalized: Record<string, unknown> = {
    ...parsed
  };
  if (normalized.stream !== true) {
    normalized.stream = true;
  }
  if (normalized.store !== false) {
    normalized.store = false;
  }
  const instructions = typeof normalized.instructions === 'string' ? normalized.instructions.trim() : '';
  if (!instructions) {
    normalized.instructions = codexDefaultInstructions;
  }

  return JSON.stringify(normalized);
}

function looksLikeCodexBaseUrl(baseUrl: string): boolean {
  try {
    const parsed = new URL(baseUrl);
    return parsed.hostname === 'chatgpt.com' || parsed.pathname.includes('/backend-api/codex');
  } catch {
    return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function resolveResponsesWebSocketTarget(
  config: Pick<GatewayConfig, 'openaiBaseUrl' | 'openaiApiKey' | 'providers'>,
  context: GatewaySocketContext
): {
  baseUrl: string;
  apiKey?: string;
  providerConfig?: ProviderConfig;
} {
  const providers = Array.isArray(config.providers) ? config.providers : [];
  const hint = readTargetProviderHint(context);
  const selectedByHint = hint ? findOpenAIProviderByHint(providers, hint) : undefined;
  if (selectedByHint) {
    return {
      baseUrl: selectedByHint.baseurl || config.openaiBaseUrl,
      apiKey: selectedByHint.apikey,
      providerConfig: selectedByHint
    };
  }

  const preferredResponsesProvider = providers.find(
    (item) => item.type === 'openai_responses'
  );
  if (preferredResponsesProvider) {
    return {
      baseUrl: preferredResponsesProvider.baseurl || config.openaiBaseUrl,
      apiKey: preferredResponsesProvider.apikey,
      providerConfig: preferredResponsesProvider
    };
  }

  const anyOpenAIProvider = findDefaultProviderConfig(providers, 'openai');
  if (anyOpenAIProvider) {
    return {
      baseUrl: anyOpenAIProvider.baseurl || config.openaiBaseUrl,
      apiKey: anyOpenAIProvider.apikey || config.openaiApiKey,
      providerConfig: anyOpenAIProvider
    };
  }

  return {
    baseUrl: config.openaiBaseUrl,
    apiKey: config.openaiApiKey
  };
}

interface WebSocketProviderPluginContext {
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

async function applyWebSocketProviderRequestPlugins(
  context: WebSocketProviderPluginContext,
  baseUpstreamRequest: UpstreamRequest
): Promise<Result<UpstreamRequest>> {
  let upstreamRequest = baseUpstreamRequest;

  for (const plugin of context.plugins) {
    if (plugin.authenticate) {
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
        standardRequest: undefined
      });
      if (!result.ok) {
        return err(`Provider plugin "${plugin.key}" auth failed: ${result.error}`);
      }

      upstreamRequest = result.value;
    }

    if (plugin.transformRequest) {
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
        standardRequest: undefined
      });
      if (!result.ok) {
        return err(`Provider plugin "${plugin.key}" request transform failed: ${result.error}`);
      }

      upstreamRequest = result.value;
    }
  }

  return ok(upstreamRequest);
}

function readTargetProviderHint(context: GatewaySocketContext): string | undefined {
  const fromHeader = normalizeHeaderValue(context.headers['x-target-provider']);
  if (fromHeader) {
    return fromHeader;
  }

  try {
    const url = new URL(context.requestUrl);
    const fromQuery =
      url.searchParams.get('target_provider') ||
      url.searchParams.get('target-provider');
    const normalizedQuery = fromQuery?.trim();
    if (normalizedQuery) {
      return normalizedQuery;
    }
  } catch {
    return undefined;
  }

  return undefined;
}

function findOpenAIProviderByHint(
  providers: ProviderConfig[],
  hintRaw: string
): ProviderConfig | undefined {
  const hint = hintRaw.trim();
  if (!hint) {
    return undefined;
  }

  const byName = providers.find((item) => item.name.toLowerCase() === hint.toLowerCase());
  if (byName && providerFromProviderType(byName.type) === 'openai') {
    return byName;
  }

  const parsedProviderType = parseProvider(hint);
  if (!parsedProviderType || parsedProviderType !== 'openai') {
    return undefined;
  }

  return (
    providers.find((item) => item.type === 'openai_responses') ||
    findDefaultProviderConfig(providers, 'openai')
  );
}

function readSourceAdapterHintFromRequestUrl(url: URL): GatewayCodexWsSourceAdapterKey | undefined {
  const fromSourceAdapter = parseGatewayCodexWsSourceAdapterKey(url.searchParams.get('source_adapter') || undefined);
  if (fromSourceAdapter) {
    return fromSourceAdapter;
  }

  return parseGatewayCodexWsSourceAdapterKey(url.searchParams.get('source') || undefined);
}

function resolveGatewayWebSocketMaxPayloadBytes(config: GatewayConfig): number {
  if (Number.isFinite(config.bodyLimitBytes) && config.bodyLimitBytes > 0) {
    return Math.max(1024, Math.floor(config.bodyLimitBytes));
  }

  return 1024 * 1024;
}

function webSocketPayloadByteLength(payload: WebSocketPayload): number {
  if (typeof payload === 'string') {
    return Buffer.byteLength(payload, 'utf8');
  }

  if (Buffer.isBuffer(payload)) {
    return payload.byteLength;
  }

  if (Array.isArray(payload)) {
    return payload.reduce((total, item) => total + webSocketPayloadByteLength(item), 0);
  }

  if (payload instanceof ArrayBuffer) {
    return payload.byteLength;
  }

  return 0;
}

function rawDataToUtf8String(rawData: RawData): string {
  if (typeof rawData === 'string') {
    return rawData;
  }

  if (Buffer.isBuffer(rawData)) {
    return rawData.toString('utf8');
  }

  if (Array.isArray(rawData)) {
    return Buffer.concat(rawData).toString('utf8');
  }

  return Buffer.from(rawData).toString('utf8');
}

function sendInvalidRequestEvent(socket: WebSocket, message: string): void {
  sendWebSocketErrorEvent(socket, 400, message);
}

function sendWebSocketErrorEvent(socket: WebSocket, status: number, message: string): void {
  if (socket.readyState !== WebSocket.OPEN) {
    return;
  }

  socket.send(
    JSON.stringify({
      type: 'error',
      status,
      error: {
        type: status === 429 ? 'rate_limit_error' : 'invalid_request_error',
        message
      }
    })
  );
}

function normalizeHeaderValue(value: string | string[] | undefined): string | undefined {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed || undefined;
  }

  if (Array.isArray(value) && value.length > 0) {
    const first = value[0]?.trim();
    return first || undefined;
  }

  return undefined;
}

function normalizeCloseCode(code: number): number {
  if (code >= 1000 && code <= 4999) {
    return code;
  }

  return 1011;
}

function isResponsesWebSocketPath(pathname: string): boolean {
  const normalized = pathname.replace(/\/+$/, '');
  return normalized === '/v1/responses';
}

function safeParseRequestUrl(request: IncomingMessage): URL | undefined {
  const host = request.headers.host || 'localhost';
  const path = request.url || '/';
  try {
    return new URL(path, `http://${host}`);
  } catch {
    return undefined;
  }
}

function createWebSocketPluginCompatibleRequest(
  request: IncomingMessage,
  requestUrl: URL,
  fastify: FastifyInstance
): FastifyRequest {
  const query: Record<string, string | string[]> = {};
  for (const [key, value] of requestUrl.searchParams.entries()) {
    const current = query[key];
    if (current === undefined) {
      query[key] = value;
      continue;
    }

    if (Array.isArray(current)) {
      current.push(value);
      continue;
    }

    query[key] = [current, value];
  }

  return {
    id: `ws-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    headers: request.headers,
    method: request.method || 'GET',
    url: request.url || requestUrl.pathname,
    query,
    body: undefined,
    log: fastify.log
  } as FastifyRequest;
}

function rejectUpgrade(socket: Socket, statusCode: number, message: string): void {
  const body = JSON.stringify({
    error: message
  });
  const response =
    `HTTP/1.1 ${statusCode} ${resolveStatusMessage(statusCode)}\r\n` +
    'Connection: close\r\n' +
    'Content-Type: application/json\r\n' +
    `Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n` +
    '\r\n' +
    body;

  socket.write(response);
  socket.destroy();
}

function resolveStatusMessage(statusCode: number): string {
  if (statusCode === 401) {
    return 'Unauthorized';
  }

  if (statusCode === 403) {
    return 'Forbidden';
  }

  if (statusCode === 404) {
    return 'Not Found';
  }

  if (statusCode === 500) {
    return 'Internal Server Error';
  }

  return 'Bad Request';
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
