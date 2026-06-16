import { createDecipheriv } from 'node:crypto';
import type { FastifyRequest } from 'fastify';
import type { ProviderPluginRegistry } from '../adapters/registry';
import { createDeepSeekThinkingProviderPlugin } from './deepseek-thinking';
import { err, ok } from '../types';
import { readBearerToken } from '../utils';
import type {
  GatewayConfig,
  ProviderPlugin,
  ProviderPluginConfig,
  ProviderPluginCodexOAuthConfig,
  ProviderPluginMutationConfig,
  ProviderPluginResponseMutationConfig,
  Result,
  StandardRequest,
  UpstreamRequest
} from '../types';

const configuredPluginKeyPrefix = 'config:';
const configuredPluginKeysStore = new WeakMap<ProviderPluginRegistry, Set<string>>();
const codexOauthRefreshGrantType = 'refresh_token';
const codexOauthTokenRefreshSkewSeconds = 60;
const codexOauthUpstreamBaseUrl = 'https://chatgpt.com/backend-api/codex';
const codexOauthRequiredScopeExpression = 'api.connectors.read api.connectors.invoke';
const codexOauthAccountHeader = 'ChatGPT-Account-ID';
const codexOauthDefaultInstructions = 'You are a helpful assistant.';
const codexCredentialEncryptionAlgorithm = 'aes-256-gcm';
const codexCredentialEncryptedPrefix = 'enc:v1:';

interface DistributedCredentialEncryptionConfig {
  keyBytes: Buffer;
  keyVersion?: string;
}

let distributedCredentialEncryptionConfig: DistributedCredentialEncryptionConfig | undefined;

interface PluginValueResolveContext {
  config: GatewayConfig;
  request: FastifyRequest;
  sourceProvider: string;
  sourceAdapterKey: string;
  targetProvider: string;
  targetProviderName?: string;
  model?: string;
  forceCodexOauthRefreshOnce?: boolean;
  standardRequest?: StandardRequest;
  upstreamRequest: UpstreamRequest;
  upstreamPayload?: unknown;
}

interface ResolvedValue {
  found: boolean;
  value?: unknown;
  missingRef?: string;
}

interface CredentialEncryptionKeyResolveResult {
  keyBytes: Buffer;
  keyVersion?: string;
}

interface EncryptedCredentialEnvelope {
  iv: string;
  tag: string;
  data: string;
  keyVersion?: string;
}

interface CodexOauthStoredState {
  accessToken?: string;
  refreshToken?: string;
  accountId?: string;
  updatedAt?: string;
}

interface CodexOauthResolvedState {
  accessToken?: string;
  refreshToken?: string;
  accountId?: string;
}

interface CodexOauthTokenRefreshResult {
  idToken?: string;
  accessToken?: string;
  refreshToken?: string;
  accountId?: string;
  scope?: string;
}

const codexOauthStateStore = new Map<string, CodexOauthStoredState>();

export function syncProviderPluginsFromConfig(
  registry: ProviderPluginRegistry,
  config: GatewayConfig
): void {
  const previouslyConfigured = configuredPluginKeysStore.get(registry) || new Set<string>();
  for (const key of previouslyConfigured) {
    registry.unregister(key);
  }

  const configuredNow = new Set<string>();
  for (const pluginConfig of config.providerPlugins || []) {
    if (!pluginConfig.enabled) {
      continue;
    }

    const plugin = buildConfiguredProviderPlugin(pluginConfig);
    registry.register(plugin, { overwrite: true });
    configuredNow.add(plugin.key);
  }

  configuredPluginKeysStore.set(registry, configuredNow);
}

export function updateDistributedCredentialEncryption(input?: {
  key?: string;
  keyVersion?: string;
  algorithm?: string;
}): void {
  if (!input) {
    distributedCredentialEncryptionConfig = undefined;
    return;
  }

  const algorithm = normalizeNonEmptyString(input.algorithm);
  if (algorithm && algorithm !== codexCredentialEncryptionAlgorithm) {
    distributedCredentialEncryptionConfig = undefined;
    return;
  }

  const keyRaw = normalizeNonEmptyString(input.key);
  if (!keyRaw) {
    distributedCredentialEncryptionConfig = undefined;
    return;
  }

  const keyBytes = decodeCredentialEncryptionKey(keyRaw);
  if (!keyBytes) {
    distributedCredentialEncryptionConfig = undefined;
    return;
  }

  distributedCredentialEncryptionConfig = {
    keyBytes,
    keyVersion: normalizeNonEmptyString(input.keyVersion)
  };
}

export async function closeCodexOauthStateStore(): Promise<void> {
  codexOauthStateStore.clear();
}

function buildConfiguredProviderPlugin(config: ProviderPluginConfig): ProviderPlugin {
  const key = `${configuredPluginKeyPrefix}${config.key}`;
  const codexOauthConfig = config.codexOauth?.enabled ? config.codexOauth : undefined;
  const deepseekThinkingPlugin = config.deepseekThinking?.enabled
    ? createDeepSeekThinkingProviderPlugin()
    : undefined;

  return {
    key,
    provider: config.provider,
    providerName: config.providerName,
    authenticate:
      codexOauthConfig || config.auth
        ? async (input) => {
            let upstreamRequest = input.upstreamRequest;
            let context: PluginValueResolveContext = {
              config: input.config,
              request: input.request,
              sourceProvider: input.sourceProvider,
              sourceAdapterKey: input.sourceAdapterKey,
              targetProvider: input.targetProvider,
              targetProviderName: input.targetProviderConfig?.name,
              model: input.model,
              forceCodexOauthRefreshOnce: input.forceCodexOauthRefreshOnce,
              standardRequest: input.standardRequest,
              upstreamRequest
            };

            if (codexOauthConfig) {
              const codexOauthResult = await applyCodexOauthAuthentication(
                `providerPlugins[${config.key}].codexOauth`,
                codexOauthConfig,
                context
              );
              if (!codexOauthResult.ok) {
                return codexOauthResult;
              }

              upstreamRequest = codexOauthResult.value;
              context = {
                ...context,
                upstreamRequest
              };
            }

            if (config.auth) {
              const mutationResult = applyRequestMutation(
                `providerPlugins[${config.key}].auth`,
                config.auth as ProviderPluginMutationConfig,
                context
              );
              if (!mutationResult.ok) {
                return mutationResult;
              }

              upstreamRequest = mutationResult.value;
            }

            return ok(upstreamRequest);
          }
        : undefined,
    transformRequest: config.request || deepseekThinkingPlugin
      ? async (input) => {
          let upstreamRequest = input.upstreamRequest;
          if (config.request) {
            const mutationResult = applyRequestMutation(
              `providerPlugins[${config.key}].request`,
              config.request as ProviderPluginMutationConfig,
              {
                config: input.config,
                request: input.request,
                sourceProvider: input.sourceProvider,
                sourceAdapterKey: input.sourceAdapterKey,
                targetProvider: input.targetProvider,
                targetProviderName: input.targetProviderConfig?.name,
                model: input.model,
                forceCodexOauthRefreshOnce: input.forceCodexOauthRefreshOnce,
                standardRequest: input.standardRequest,
                upstreamRequest
              }
            );
            if (!mutationResult.ok) {
              return mutationResult;
            }
            upstreamRequest = mutationResult.value;
          }

          if (deepseekThinkingPlugin?.transformRequest) {
            return deepseekThinkingPlugin.transformRequest({
              ...input,
              upstreamRequest
            });
          }

          return ok(upstreamRequest);
        }
      : undefined,
    transformResponse: config.response
      ? (input) =>
          applyResponseMutation(
            `providerPlugins[${config.key}].response`,
            config.response as ProviderPluginResponseMutationConfig,
            {
              config: input.config,
              request: input.request,
              sourceProvider: input.sourceProvider,
              sourceAdapterKey: input.sourceAdapterKey,
              targetProvider: input.targetProvider,
              targetProviderName: input.targetProviderConfig?.name,
              model: input.model,
              forceCodexOauthRefreshOnce: input.forceCodexOauthRefreshOnce,
              standardRequest: input.standardRequest,
              upstreamRequest: input.upstreamRequest,
              upstreamPayload: input.upstreamPayload
            }
          )
      : undefined
  };
}

async function applyCodexOauthAuthentication(
  section: string,
  codexOauth: ProviderPluginCodexOAuthConfig,
  context: PluginValueResolveContext
): Promise<Result<UpstreamRequest>> {
  const rewrittenUpstreamRequest = rewriteCodexOauthUpstreamUrl(section, context.upstreamRequest, context);
  const normalizedUpstreamRequest = normalizeCodexOauthUpstreamRequestBody(
    section,
    rewrittenUpstreamRequest,
    context
  );
  const resolveContext =
    normalizedUpstreamRequest === context.upstreamRequest
      ? context
      : {
          ...context,
          upstreamRequest: normalizedUpstreamRequest
        };

  const accessResolution =
    codexOauth.accessToken === undefined
      ? undefined
      : resolvePluginValue(codexOauth.accessToken, resolveContext);
  const accessTokenResolve = resolveCodexOauthTokenValue(
    section,
    'accessToken',
    accessResolution,
    resolveContext
  );
  if (!accessTokenResolve.ok) {
    return accessTokenResolve;
  }
  let accessToken = accessTokenResolve.value;

  const refreshResolution =
    codexOauth.refreshToken === undefined
      ? undefined
      : resolvePluginValue(codexOauth.refreshToken, resolveContext);
  const refreshTokenResolve = resolveCodexOauthTokenValue(
    section,
    'refreshToken',
    refreshResolution,
    resolveContext
  );
  if (!refreshTokenResolve.ok) {
    return refreshTokenResolve;
  }
  let refreshToken = refreshTokenResolve.value;

  const accountResolution =
    codexOauth.accountId === undefined
      ? undefined
      : resolvePluginValue(codexOauth.accountId, resolveContext);
  let accountId = accountResolution?.found
    ? normalizeNonEmptyString(accountResolution.value)
    : undefined;
  if (accountResolution?.found && !accountId) {
    return err(`${section}.accountId must resolve to a non-empty string.`);
  }
  if (accountResolution && !accountResolution.found && accountResolution.missingRef) {
    return err(`${section}.accountId references missing value: ${accountResolution.missingRef}`);
  }

  const storedState = readCodexOauthState(section, resolveContext);
  if (storedState) {
    accessToken = storedState.accessToken || accessToken;
    refreshToken = storedState.refreshToken || refreshToken;
    accountId = storedState.accountId || accountId;
  }

  const tokenAnalysis = accessToken
    ? analyzeCodexAccessToken(accessToken, codexOauthRequiredScopeExpression)
    : undefined;
  let refreshResponseScope: string | undefined;
  const refreshReasons: string[] = [];
  if (codexOauth.forceRefresh) {
    refreshReasons.push('force_refresh');
  }
  if (context.forceCodexOauthRefreshOnce) {
    refreshReasons.push('upstream_401_retry');
  }
  if (!accessToken && codexOauth.refreshIfMissingAccessToken) {
    refreshReasons.push('missing_access_token');
  }
  if (tokenAnalysis?.isExpired) {
    refreshReasons.push('access_token_expired');
  }
  if (tokenAnalysis && tokenAnalysis.missingScopes.length > 0) {
    refreshReasons.push(`missing_scopes:${tokenAnalysis.missingScopes.join(',')}`);
  }
  const shouldRefresh = refreshReasons.length > 0;
  logCodexOauthEvent(context, 'info', 'Codex OAuth auth started.', {
    section,
    target_provider: context.targetProvider,
    target_provider_name: context.targetProviderName,
    model: context.model,
    has_access_token: Boolean(accessToken),
    has_refresh_token: Boolean(refreshToken),
    should_refresh: shouldRefresh,
    refresh_reasons: refreshReasons,
    token_scope_inspectable: tokenAnalysis?.inspectable,
    token_scopes: tokenAnalysis?.scopes,
    token_missing_scopes: tokenAnalysis?.missingScopes,
    token_expires_at_ms: tokenAnalysis?.expiresAtMs,
    token_is_expired: tokenAnalysis?.isExpired,
    required: codexOauth.required,
    token_endpoint: codexOauth.tokenEndpoint,
    scope: codexOauth.scope,
    required_scope: codexOauthRequiredScopeExpression,
    loaded_from_cache: Boolean(storedState)
  });

  if (shouldRefresh && refreshToken) {
    const refreshedTokens = await requestCodexOauthTokenRefresh(section, codexOauth, refreshToken, context);
    if (refreshedTokens.ok) {
      accessToken = refreshedTokens.value.accessToken || accessToken;
      refreshToken = refreshedTokens.value.refreshToken || refreshToken;
      accountId = refreshedTokens.value.accountId || accountId;
      refreshResponseScope = refreshedTokens.value.scope;
      logCodexOauthEvent(context, 'info', 'Codex OAuth token refresh succeeded.', {
        section,
        response_scope: refreshedTokens.value.scope,
        returned_access_token: Boolean(refreshedTokens.value.accessToken),
        returned_refresh_token: Boolean(refreshedTokens.value.refreshToken),
        returned_account_id: Boolean(refreshedTokens.value.accountId),
        access_token_length: refreshedTokens.value.accessToken?.length,
        refresh_token_length: refreshedTokens.value.refreshToken?.length
      });
      persistCodexOauthState(section, resolveContext, {
        accessToken,
        refreshToken,
        accountId
      });
    } else if (!accessToken && codexOauth.required) {
      logCodexOauthEvent(context, 'warn', 'Codex OAuth token refresh failed and access token is required.', {
        section,
        reason: refreshedTokens.error
      });
      return refreshedTokens;
    }
  }

  if (!accessToken) {
    logCodexOauthEvent(context, codexOauth.required ? 'warn' : 'info', 'Codex OAuth access token is missing after refresh logic.', {
      section,
      required: codexOauth.required,
      has_refresh_token: Boolean(refreshToken),
      should_refresh: shouldRefresh,
      access_resolution_found: accessResolution?.found,
      refresh_resolution_found: refreshResolution?.found
    });
    if (!codexOauth.required) {
      return ok(normalizedUpstreamRequest);
    }

    if (accessResolution && !accessResolution.found && accessResolution.missingRef) {
      return err(`${section}.accessToken references missing value: ${accessResolution.missingRef}`);
    }

    if (shouldRefresh && refreshResolution && !refreshResolution.found && refreshResolution.missingRef) {
      return err(`${section}.refreshToken references missing value: ${refreshResolution.missingRef}`);
    }

    if (shouldRefresh && !refreshToken) {
      return err(`${section}.refreshToken is required when token refresh is enabled.`);
    }

    return err(`${section}.accessToken is required but missing.`);
  }

  const finalTokenAnalysis = analyzeCodexAccessToken(accessToken, codexOauthRequiredScopeExpression);
  const finalScopeEvaluation = evaluateCodexScopeRequirements(
    codexOauthRequiredScopeExpression,
    finalTokenAnalysis,
    refreshResponseScope
  );
  if (!accountId) {
    accountId = extractCodexAccountIdFromToken(accessToken);
  }

  if (finalScopeEvaluation.known && finalScopeEvaluation.missingScopes.length > 0) {
    logCodexOauthEvent(
      context,
      codexOauth.required ? 'warn' : 'info',
      'Codex OAuth token is missing required scopes after auth flow.',
      {
        section,
        required_scope: codexOauthRequiredScopeExpression,
        requested_scope: codexOauth.scope,
        scope_source: finalScopeEvaluation.source,
        granted_scopes: finalScopeEvaluation.scopes,
        missing_scopes: finalScopeEvaluation.missingScopes
      }
    );
    if (codexOauth.required) {
      return err(
        `${section} resolved token is missing required scopes: ${finalScopeEvaluation.missingScopes.join(
          ', '
        )}.`
      );
    }
  }

  const authHeader = codexOauth.authHeader.trim() || 'authorization';
  const authScheme = codexOauth.authScheme.trim();
  const headerValue = authScheme ? `${authScheme} ${accessToken}` : accessToken;
  const headers = { ...normalizedUpstreamRequest.headers };
  setHeaderValue(headers, authHeader, headerValue);
  if (accountId) {
    setHeaderValue(headers, codexOauthAccountHeader, accountId);
  }
  logCodexOauthEvent(context, 'info', 'Codex OAuth credential injected into upstream request.', {
    section,
    auth_header: authHeader,
    auth_scheme: authScheme || undefined,
    access_token_length: accessToken.length,
    account_header: accountId ? codexOauthAccountHeader : undefined
  });

  persistCodexOauthState(section, resolveContext, {
    accessToken,
    refreshToken,
    accountId
  });

  return ok({
    ...normalizedUpstreamRequest,
    headers
  });
}

async function requestCodexOauthTokenRefresh(
  section: string,
  codexOauth: ProviderPluginCodexOAuthConfig,
  refreshToken: string,
  context: PluginValueResolveContext
): Promise<Result<CodexOauthTokenRefreshResult>> {
  const abortController = new AbortController();
  const timeout = setTimeout(() => {
    abortController.abort();
  }, codexOauth.timeoutMs);

  try {
    logCodexOauthEvent(context, 'info', 'Requesting Codex OAuth token refresh.', {
      section,
      token_endpoint: codexOauth.tokenEndpoint,
      client_id: codexOauth.clientId,
      scope: codexOauth.scope,
      refresh_token_length: refreshToken.length
    });

    const response = await fetch(codexOauth.tokenEndpoint, {
      method: 'POST',
      headers: {
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        client_id: codexOauth.clientId,
        scope: codexOauth.scope,
        grant_type: codexOauthRefreshGrantType,
        refresh_token: refreshToken
      }),
      signal: abortController.signal
    });

    const responseText = await response.text();
    const responsePayload = parseJsonIfPossible(responseText);
    if (!response.ok) {
      const details = extractCodexOauthErrorDetails(responsePayload, responseText);
      logCodexOauthEvent(context, 'warn', 'Codex OAuth token refresh HTTP failure.', {
        section,
        status: response.status,
        details,
        response_scope: readStringProperty(responsePayload, 'scope')
      });
      return err(`${section} token refresh failed with status ${response.status}: ${details}`);
    }

    const responseScope =
      readStringProperty(responsePayload, 'scope') ||
      readStringProperty(responsePayload, 'scopes');
    logCodexOauthEvent(context, 'info', 'Codex OAuth token refresh HTTP success.', {
      section,
      status: response.status,
      response_scope: responseScope
    });

    return ok({
      idToken:
        readStringProperty(responsePayload, 'id_token') ||
        readStringProperty(responsePayload, 'idToken'),
      accessToken:
        readStringProperty(responsePayload, 'access_token') ||
        readStringProperty(responsePayload, 'accessToken'),
      refreshToken:
        readStringProperty(responsePayload, 'refresh_token') ||
        readStringProperty(responsePayload, 'refreshToken'),
      accountId:
        readStringProperty(responsePayload, 'account_id') ||
        readStringProperty(responsePayload, 'accountId'),
      scope: responseScope
    });
  } catch (error) {
    if (isAbortError(error)) {
      logCodexOauthEvent(context, 'warn', 'Codex OAuth token refresh request timed out.', {
        section,
        timeout_ms: codexOauth.timeoutMs
      });
      return err(`${section} token refresh timed out after ${codexOauth.timeoutMs}ms.`);
    }

    logCodexOauthEvent(context, 'warn', 'Codex OAuth token refresh request failed.', {
      section,
      error: error instanceof Error ? error.message : String(error)
    });
    return err(
      `${section} token refresh request failed: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  } finally {
    clearTimeout(timeout);
  }
}

function readCodexOauthState(
  section: string,
  context: PluginValueResolveContext
): CodexOauthResolvedState | undefined {
  const stored = codexOauthStateStore.get(buildCodexOauthStateKey(section, context));
  if (!stored) {
    return undefined;
  }

  return {
    accessToken: normalizeTokenValue(stored.accessToken),
    refreshToken: normalizeTokenValue(stored.refreshToken),
    accountId: normalizeNonEmptyString(stored.accountId)
  };
}

function persistCodexOauthState(
  section: string,
  context: PluginValueResolveContext,
  state: CodexOauthResolvedState
): void {
  if (!state.accessToken && !state.refreshToken && !state.accountId) {
    return;
  }

  codexOauthStateStore.set(buildCodexOauthStateKey(section, context), {
    accessToken: state.accessToken,
    refreshToken: state.refreshToken,
    accountId: state.accountId,
    updatedAt: new Date().toISOString()
  });
}

function buildCodexOauthStateKey(
  section: string,
  context: PluginValueResolveContext
): string {
  const pluginName = extractPluginNameFromSection(section);
  const providerName = normalizeNonEmptyString(context.targetProviderName) || context.targetProvider;
  return [
    'codex_oauth',
    sanitizeStateKeySegment(providerName),
    sanitizeStateKeySegment(pluginName)
  ].join(':');
}

function extractPluginNameFromSection(section: string): string {
  const match = section.match(/^providerPlugins\[(.+?)\]\.codexOauth$/);
  if (!match) {
    return section;
  }
  return match[1] || section;
}

function sanitizeStateKeySegment(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/[^a-z0-9:_-]+/g, '_');
  return normalized || 'default';
}

function rewriteCodexOauthUpstreamUrl(
  section: string,
  upstreamRequest: UpstreamRequest,
  context: PluginValueResolveContext
): UpstreamRequest {
  let currentUrl: URL;
  try {
    currentUrl = new URL(upstreamRequest.url);
  } catch {
    logCodexOauthEvent(context, 'warn', 'Codex OAuth upstream URL is invalid and cannot be rewritten.', {
      section,
      upstream_url: upstreamRequest.url
    });
    return upstreamRequest;
  }

  const targetBase = new URL(codexOauthUpstreamBaseUrl);
  const suffixPath = normalizeCodexUpstreamPath(currentUrl.pathname);
  const targetPath = joinUrlPath(targetBase.pathname, suffixPath);
  const rewritten = new URL(codexOauthUpstreamBaseUrl);
  rewritten.pathname = targetPath;
  rewritten.search = currentUrl.search;

  const rewrittenUrl = rewritten.toString();
  if (rewrittenUrl === upstreamRequest.url) {
    return upstreamRequest;
  }

  logCodexOauthEvent(context, 'info', 'Codex OAuth upstream URL rewritten to ChatGPT backend.', {
    section,
    upstream_url_before: upstreamRequest.url,
    upstream_url_after: rewrittenUrl
  });

  return {
    ...upstreamRequest,
    url: rewrittenUrl
  };
}

function normalizeCodexOauthUpstreamRequestBody(
  section: string,
  upstreamRequest: UpstreamRequest,
  context: PluginValueResolveContext
): UpstreamRequest {
  if (!isCodexResponsesUrl(upstreamRequest.url)) {
    return upstreamRequest;
  }

  if (!isPlainObject(upstreamRequest.body)) {
    return upstreamRequest;
  }

  const body = upstreamRequest.body as Record<string, unknown>;
  let changed = false;
  const nextBody: Record<string, unknown> = {
    ...body
  };

  if (nextBody.store !== false) {
    nextBody.store = false;
    changed = true;
  }

  if (nextBody.stream !== true) {
    nextBody.stream = true;
    changed = true;
  }

  const instructions =
    typeof nextBody.instructions === 'string' ? nextBody.instructions.trim() : '';
  if (!instructions) {
    nextBody.instructions = codexOauthDefaultInstructions;
    changed = true;
  }

  if (!changed) {
    if (nextBody.stream === true) {
      const headers = { ...upstreamRequest.headers };
      if (!readHeaderValueCaseInsensitive(headers, 'accept')) {
        setHeaderValue(headers, 'accept', 'text/event-stream');
        return {
          ...upstreamRequest,
          headers
        };
      }
    }
    return upstreamRequest;
  }

  logCodexOauthEvent(
    context,
    'info',
    'Codex OAuth request body normalized for ChatGPT backend compatibility.',
    {
      section,
      store_forced_false: nextBody.store === false,
      stream_forced_true: nextBody.stream === true,
      instructions_injected: nextBody.instructions === codexOauthDefaultInstructions
    }
  );

  const headers = { ...upstreamRequest.headers };
  if (nextBody.stream === true && !readHeaderValueCaseInsensitive(headers, 'accept')) {
    setHeaderValue(headers, 'accept', 'text/event-stream');
  }

  return {
    ...upstreamRequest,
    headers,
    body: nextBody
  };
}

function readHeaderValueCaseInsensitive(
  headers: Record<string, string>,
  name: string
): string | undefined {
  const target = name.trim().toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === target) {
      return value;
    }
  }
  return undefined;
}

function isCodexResponsesUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.pathname.endsWith('/responses');
  } catch {
    return false;
  }
}

function normalizeCodexUpstreamPath(pathname: string): string {
  const normalized = pathname.trim() || '/';
  const withLeadingSlash = normalized.startsWith('/') ? normalized : `/${normalized}`;
  const withoutV1 = stripPathPrefix(withLeadingSlash, '/v1') ?? withLeadingSlash;
  const withoutCodexPrefix =
    stripPathPrefix(withoutV1, '/backend-api/codex') ?? withoutV1;
  return withoutCodexPrefix || '/';
}

function stripPathPrefix(pathname: string, prefix: string): string | undefined {
  if (pathname === prefix) {
    return '/';
  }

  if (pathname.startsWith(`${prefix}/`)) {
    return pathname.slice(prefix.length);
  }

  return undefined;
}

function joinUrlPath(basePath: string, suffixPath: string): string {
  const normalizedBase = basePath.endsWith('/') ? basePath.slice(0, -1) : basePath;
  const normalizedSuffix = suffixPath.startsWith('/') ? suffixPath : `/${suffixPath}`;
  if (normalizedSuffix === '/') {
    return normalizedBase || '/';
  }
  return `${normalizedBase}${normalizedSuffix}`;
}

function parseJsonIfPossible(text: string): unknown {
  const normalized = text.trim();
  if (!normalized) {
    return undefined;
  }

  try {
    return JSON.parse(normalized);
  } catch {
    return undefined;
  }
}

function extractCodexOauthErrorDetails(payload: unknown, fallback: string): string {
  if (isPlainObject(payload)) {
    const error = payload.error;
    if (typeof error === 'string') {
      const code = readStringProperty(payload, 'code');
      const message = readStringProperty(payload, 'message');
      const details = [code, message].filter(Boolean).join(', ');
      return details ? `${error} (${details})` : error;
    }

    if (isPlainObject(error)) {
      const code = readStringProperty(error, 'code');
      const message = readStringProperty(error, 'message');
      const details = [code, message].filter(Boolean).join(', ');
      if (details) {
        return details;
      }
    }

    const code = readStringProperty(payload, 'code');
    const message = readStringProperty(payload, 'message');
    const details = [code, message].filter(Boolean).join(', ');
    if (details) {
      return details;
    }
  }

  const normalizedFallback = fallback.trim();
  return normalizedFallback || 'unknown error';
}

function readStringProperty(value: unknown, key: string): string | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const raw = value[key];
  if (typeof raw !== 'string') {
    return undefined;
  }

  const normalized = raw.trim();
  return normalized || undefined;
}

function normalizeTokenValue(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim();
  if (!normalized) {
    return undefined;
  }

  return readBearerToken(normalized) || normalized;
}

function resolveCodexOauthTokenValue(
  section: string,
  fieldName: 'accessToken' | 'refreshToken',
  resolution: ResolvedValue | undefined,
  context: PluginValueResolveContext
): Result<string | undefined> {
  if (!resolution) {
    return ok(undefined as string | undefined);
  }

  if (!resolution.found) {
    // Missing references are validated later with refresh/required semantics.
    return ok(undefined as string | undefined);
  }

  const token = normalizeTokenValue(resolution.value);
  if (!token) {
    return err(`${section}.${fieldName} must resolve to a non-empty string.`);
  }

  if (!isCodexEncryptedToken(token)) {
    return ok(token);
  }

  return decryptCodexEncryptedToken(section, fieldName, token, context);
}

function isCodexEncryptedToken(value: string): boolean {
  return value.startsWith(codexCredentialEncryptedPrefix);
}

function decryptCodexEncryptedToken(
  section: string,
  fieldName: string,
  encryptedToken: string,
  context: PluginValueResolveContext
): Result<string> {
  const encodedEnvelope = encryptedToken.slice(codexCredentialEncryptedPrefix.length);
  if (!encodedEnvelope) {
    return err(`${section}.${fieldName} encrypted token payload is empty.`);
  }

  let envelope: EncryptedCredentialEnvelope;
  try {
    const decoded = Buffer.from(encodedEnvelope, 'base64url').toString('utf8');
    const parsed = JSON.parse(decoded);
    if (!isPlainObject(parsed)) {
      return err(`${section}.${fieldName} encrypted token payload is invalid.`);
    }

    const iv = normalizeNonEmptyString(parsed.iv);
    const tag = normalizeNonEmptyString(parsed.tag);
    const data = normalizeNonEmptyString(parsed.data);
    if (!iv || !tag || !data) {
      return err(`${section}.${fieldName} encrypted token payload is incomplete.`);
    }

    envelope = {
      iv,
      tag,
      data,
      keyVersion: normalizeNonEmptyString(parsed.keyVersion)
    };
  } catch {
    return err(`${section}.${fieldName} encrypted token payload is malformed.`);
  }

  const keyResolution = resolveCredentialEncryptionKeyBytes();
  if (!keyResolution.ok) {
    return err(`${section}.${fieldName} is encrypted but ${keyResolution.error}`);
  }

  if (
    envelope.keyVersion &&
    keyResolution.value.keyVersion &&
    envelope.keyVersion !== keyResolution.value.keyVersion
  ) {
    return err(
      `${section}.${fieldName} encrypted token key version mismatch: expected ${keyResolution.value.keyVersion}, got ${envelope.keyVersion}.`
    );
  }

  const ivBuffer = decodeBase64CredentialPart(envelope.iv);
  if (!ivBuffer) {
    return err(`${section}.${fieldName} encrypted token iv is invalid.`);
  }
  if (ivBuffer.length !== 12) {
    return err(`${section}.${fieldName} encrypted token iv must be 12 bytes.`);
  }

  const tagBuffer = decodeBase64CredentialPart(envelope.tag);
  if (!tagBuffer) {
    return err(`${section}.${fieldName} encrypted token auth tag is invalid.`);
  }

  const encryptedBuffer = decodeBase64CredentialPart(envelope.data);
  if (!encryptedBuffer) {
    return err(`${section}.${fieldName} encrypted token ciphertext is invalid.`);
  }

  try {
    const decipher = createDecipheriv(
      codexCredentialEncryptionAlgorithm,
      keyResolution.value.keyBytes,
      ivBuffer
    );
    decipher.setAuthTag(tagBuffer);
    const decrypted = Buffer.concat([decipher.update(encryptedBuffer), decipher.final()]).toString('utf8');
    const token = normalizeTokenValue(decrypted);
    if (!token) {
      return err(`${section}.${fieldName} decrypted token is empty.`);
    }

    logCodexOauthEvent(context, 'info', 'Codex OAuth encrypted token decrypted.', {
      section,
      field: fieldName,
      key_version: envelope.keyVersion || keyResolution.value.keyVersion
    });
    return ok(token);
  } catch {
    return err(`${section}.${fieldName} encrypted token decryption failed.`);
  }
}

function decodeBase64CredentialPart(value: string): Buffer | undefined {
  try {
    const decoded = Buffer.from(value, 'base64');
    if (decoded.length === 0) {
      return undefined;
    }

    const normalizedInput = value.replace(/=+$/g, '');
    const normalizedDecoded = decoded.toString('base64').replace(/=+$/g, '');
    if (normalizedDecoded !== normalizedInput) {
      return undefined;
    }

    return decoded;
  } catch {
    return undefined;
  }
}

function resolveCredentialEncryptionKeyBytes(): Result<CredentialEncryptionKeyResolveResult> {
  if (distributedCredentialEncryptionConfig?.keyBytes) {
    return ok({
      keyBytes: distributedCredentialEncryptionConfig.keyBytes,
      keyVersion: distributedCredentialEncryptionConfig.keyVersion
    } satisfies CredentialEncryptionKeyResolveResult);
  }

  const fallbackKeyRaw = normalizeNonEmptyString(process.env.GATEWAY_CREDENTIAL_ENCRYPTION_KEY);
  if (!fallbackKeyRaw) {
    return err('no credential encryption key is configured.');
  }

  const keyBytes = decodeCredentialEncryptionKey(fallbackKeyRaw);
  if (!keyBytes) {
    return err('credential encryption key format is invalid.');
  }

  return ok({
    keyBytes,
    keyVersion: normalizeNonEmptyString(process.env.GATEWAY_CREDENTIAL_ENCRYPTION_KEY_VERSION)
  } satisfies CredentialEncryptionKeyResolveResult);
}

function decodeCredentialEncryptionKey(raw: string): Buffer | undefined {
  const key = normalizeNonEmptyString(raw);
  if (!key) {
    return undefined;
  }

  const asBase64 = tryDecodeCredentialKeyByEncoding(key, 'base64');
  if (asBase64) {
    return asBase64;
  }

  const asHex = tryDecodeCredentialKeyByEncoding(key, 'hex');
  if (asHex) {
    return asHex;
  }

  const asUtf8 = Buffer.from(key, 'utf8');
  if (asUtf8.length === 32) {
    return asUtf8;
  }

  return undefined;
}

function tryDecodeCredentialKeyByEncoding(value: string, encoding: 'base64' | 'hex'): Buffer | undefined {
  try {
    const decoded = Buffer.from(value, encoding);
    if (decoded.length !== 32) {
      return undefined;
    }
    if (encoding === 'base64' && decoded.toString('base64').replace(/=+$/g, '') !== value.replace(/=+$/g, '')) {
      return undefined;
    }
    if (encoding === 'hex' && decoded.toString('hex') !== value.toLowerCase()) {
      return undefined;
    }
    return decoded;
  } catch {
    return undefined;
  }
}

function normalizeNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim();
  return normalized || undefined;
}

interface CodexAccessTokenAnalysis {
  inspectable: boolean;
  scopes: string[];
  missingScopes: string[];
  expiresAtMs?: number;
  isExpired?: boolean;
}

interface CodexScopeEvaluation {
  known: boolean;
  source: 'token' | 'response_scope' | 'unknown';
  scopes: string[];
  missingScopes: string[];
}

function analyzeCodexAccessToken(token: string, requiredScopeExpression: string): CodexAccessTokenAnalysis {
  const payload = decodeJwtPayload(token);
  const requiredScopes = parseScopeExpression(requiredScopeExpression);
  if (!payload) {
    return {
      inspectable: false,
      scopes: [],
      missingScopes: []
    };
  }

  const scopes = parseScopeClaims(payload);
  const missingScopes = requiredScopes.filter((scope) => !scopes.includes(scope));
  const expRaw = payload.exp;
  const expiresAtMs =
    typeof expRaw === 'number' && Number.isFinite(expRaw) ? Math.floor(expRaw * 1000) : undefined;
  const isExpired =
    expiresAtMs !== undefined
      ? expiresAtMs <= Date.now() + codexOauthTokenRefreshSkewSeconds * 1000
      : undefined;

  return {
    inspectable: true,
    scopes,
    missingScopes,
    expiresAtMs,
    isExpired
  };
}

function evaluateCodexScopeRequirements(
  requiredScopeExpression: string,
  tokenAnalysis: CodexAccessTokenAnalysis,
  refreshResponseScope?: string
): CodexScopeEvaluation {
  const requiredScopes = parseScopeExpression(requiredScopeExpression);
  if (requiredScopes.length === 0) {
    return {
      known: true,
      source: 'token',
      scopes: tokenAnalysis.scopes,
      missingScopes: []
    };
  }

  if (tokenAnalysis.inspectable) {
    return {
      known: true,
      source: 'token',
      scopes: tokenAnalysis.scopes,
      missingScopes: requiredScopes.filter((scope) => !tokenAnalysis.scopes.includes(scope))
    };
  }

  const responseScopes = parseScopeExpression(refreshResponseScope || '');
  if (responseScopes.length > 0) {
    return {
      known: true,
      source: 'response_scope',
      scopes: responseScopes,
      missingScopes: requiredScopes.filter((scope) => !responseScopes.includes(scope))
    };
  }

  return {
    known: false,
    source: 'unknown',
    scopes: [],
    missingScopes: []
  };
}

function parseScopeExpression(value: string): string[] {
  const ordered: string[] = [];
  const seen = new Set<string>();
  for (const token of value.split(/\s+/)) {
    const normalized = token.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ordered.push(normalized);
  }
  return ordered;
}

function parseScopeClaims(payload: Record<string, unknown>): string[] {
  const ordered: string[] = [];
  const seen = new Set<string>();
  const appendToken = (value: string) => {
    for (const part of value.split(/\s+/)) {
      const normalized = part.trim();
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      ordered.push(normalized);
    }
  };

  const appendUnknown = (value: unknown) => {
    if (typeof value === 'string') {
      appendToken(value);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === 'string') {
          appendToken(item);
        }
      }
    }
  };

  appendUnknown(payload.scope);
  appendUnknown(payload.scopes);
  appendUnknown(payload.scp);

  return ordered;
}

function decodeJwtPayload(token: string): Record<string, unknown> | undefined {
  const parts = token.split('.');
  if (parts.length < 2) {
    return undefined;
  }

  const payloadSegment = parts[1];
  if (!payloadSegment) {
    return undefined;
  }

  try {
    const normalized = payloadSegment.replace(/-/g, '+').replace(/_/g, '/');
    const remainder = normalized.length % 4;
    const padded = remainder === 0 ? normalized : `${normalized}${'='.repeat(4 - remainder)}`;
    const decoded = Buffer.from(padded, 'base64').toString('utf8');
    const parsed = JSON.parse(decoded);
    return isPlainObject(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function extractCodexAccountIdFromToken(token: string): string | undefined {
  const payload = decodeJwtPayload(token);
  if (!payload) {
    return undefined;
  }

  return readStringProperty(payload, 'account_id') || readStringProperty(payload, 'accountId');
}

function logCodexOauthEvent(
  context: PluginValueResolveContext,
  level: 'info' | 'warn',
  message: string,
  details: Record<string, unknown>
): void {
  const logger = context.request.log as
    | {
        info?: (obj: Record<string, unknown>, msg: string) => void;
        warn?: (obj: Record<string, unknown>, msg: string) => void;
      }
    | undefined;
  if (!logger) {
    return;
  }

  const logFn = level === 'warn' ? logger.warn : logger.info;
  if (typeof logFn !== 'function') {
    return;
  }

  logFn.call(logger, details, message);
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function applyRequestMutation(
  section: string,
  mutation: ProviderPluginMutationConfig,
  context: PluginValueResolveContext
): Result<UpstreamRequest> {
  let upstreamRequest = context.upstreamRequest;
  let nextContext = context;

  if (Object.keys(mutation.headers).length > 0 || mutation.removeHeaders.length > 0) {
    const headers = { ...upstreamRequest.headers };
    for (const [headerName, valueSpec] of Object.entries(mutation.headers)) {
      const resolved = resolvePluginValue(valueSpec, nextContext);
      if (!resolved.found) {
        if (mutation.strict) {
          return err(`${section}.headers.${headerName} references missing value: ${resolved.missingRef}`);
        }
        continue;
      }

      setHeaderValue(headers, headerName, serializeValueForHeader(resolved.value));
    }

    for (const headerName of mutation.removeHeaders) {
      removeHeaderValue(headers, headerName);
    }

    upstreamRequest = {
      ...upstreamRequest,
      headers
    };
    nextContext = {
      ...nextContext,
      upstreamRequest
    };
  }

  if (Object.keys(mutation.query).length > 0 || mutation.removeQuery.length > 0) {
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(upstreamRequest.url);
    } catch {
      if (mutation.strict) {
        return err(`${section}.query requires absolute upstreamRequest.url, got: ${upstreamRequest.url}`);
      }

      return ok(upstreamRequest);
    }

    for (const [queryKey, valueSpec] of Object.entries(mutation.query)) {
      const resolved = resolvePluginValue(valueSpec, nextContext);
      if (!resolved.found) {
        if (mutation.strict) {
          return err(`${section}.query.${queryKey} references missing value: ${resolved.missingRef}`);
        }
        continue;
      }

      parsedUrl.searchParams.set(queryKey, serializeValueForQuery(resolved.value));
    }

    for (const queryKey of mutation.removeQuery) {
      parsedUrl.searchParams.delete(queryKey);
    }

    upstreamRequest = {
      ...upstreamRequest,
      url: parsedUrl.toString()
    };
    nextContext = {
      ...nextContext,
      upstreamRequest
    };
  }

  if (
    Object.keys(mutation.bodySet).length > 0 ||
    Object.keys(mutation.bodyMerge).length > 0 ||
    mutation.bodyRemove.length > 0
  ) {
    const bodyResult = applyPayloadMutation(
      section,
      {
        strict: mutation.strict,
        bodySet: mutation.bodySet,
        bodyMerge: mutation.bodyMerge,
        bodyRemove: mutation.bodyRemove
      },
      nextContext,
      upstreamRequest.body
    );
    if (!bodyResult.ok) {
      return bodyResult;
    }

    upstreamRequest = {
      ...upstreamRequest,
      body: bodyResult.value
    };
  }

  return ok(upstreamRequest);
}

function applyResponseMutation(
  section: string,
  mutation: ProviderPluginResponseMutationConfig,
  context: PluginValueResolveContext
): Result<unknown> {
  const payloadResult = applyPayloadMutation(
    section,
    mutation,
    context,
    context.upstreamPayload
  );
  if (!payloadResult.ok) {
    return payloadResult;
  }

  return ok(payloadResult.value);
}

function applyPayloadMutation(
  section: string,
  mutation: Pick<ProviderPluginResponseMutationConfig, 'strict' | 'bodySet' | 'bodyMerge' | 'bodyRemove'>,
  context: PluginValueResolveContext,
  payload: unknown
): Result<Record<string, unknown>> {
  const mutablePayload = (isPlainObject(payload) ? cloneUnknown(payload) : {}) as Record<string, unknown>;
  let mutableContext = {
    ...context,
    upstreamPayload: mutablePayload
  };

  for (const [path, valueSpec] of Object.entries(mutation.bodySet)) {
    const resolved = resolvePluginValue(valueSpec, mutableContext);
    if (!resolved.found) {
      if (mutation.strict) {
        return err(`${section}.bodySet.${path} references missing value: ${resolved.missingRef}`);
      }
      continue;
    }

    setValueAtPath(mutablePayload, path, cloneUnknown(resolved.value));
    mutableContext = {
      ...mutableContext,
      upstreamPayload: mutablePayload
    };
  }

  for (const [path, valueSpec] of Object.entries(mutation.bodyMerge)) {
    const resolved = resolvePluginValue(valueSpec, mutableContext);
    if (!resolved.found) {
      if (mutation.strict) {
        return err(`${section}.bodyMerge.${path} references missing value: ${resolved.missingRef}`);
      }
      continue;
    }

    if (!isPlainObject(resolved.value)) {
      if (mutation.strict) {
        return err(`${section}.bodyMerge.${path} requires object value.`);
      }
      continue;
    }

    if (!path.trim()) {
      const mergedRoot = mergeJsonObjects(mutablePayload, resolved.value);
      for (const key of Object.keys(mutablePayload)) {
        delete mutablePayload[key];
      }
      for (const [key, item] of Object.entries(mergedRoot)) {
        mutablePayload[key] = item;
      }
      continue;
    }

    const current = readValueByPath(mutablePayload, path);
    const nextValue = isPlainObject(current)
      ? mergeJsonObjects(current, resolved.value)
      : cloneUnknown(resolved.value);
    setValueAtPath(mutablePayload, path, nextValue);
    mutableContext = {
      ...mutableContext,
      upstreamPayload: mutablePayload
    };
  }

  for (const path of mutation.bodyRemove) {
    removeValueAtPath(mutablePayload, path);
  }

  return ok(mutablePayload);
}

function resolvePluginValue(value: unknown, context: PluginValueResolveContext): ResolvedValue {
  if (isReferenceObject(value)) {
    const resolved = resolveReference(value.from, context);
    if (resolved !== undefined) {
      return {
        found: true,
        value: cloneUnknown(resolved)
      };
    }

    if (Object.prototype.hasOwnProperty.call(value, 'default')) {
      return {
        found: true,
        value: cloneUnknown(value.default)
      };
    }

    return {
      found: false,
      missingRef: value.from
    };
  }

  if (typeof value === 'string') {
    const reference = parseReferenceTemplate(value);
    if (!reference) {
      return {
        found: true,
        value
      };
    }

    const resolved = resolveReference(reference, context);
    if (resolved === undefined) {
      return {
        found: false,
        missingRef: reference
      };
    }

    return {
      found: true,
      value: cloneUnknown(resolved)
    };
  }

  if (Array.isArray(value)) {
    const resolvedArray: unknown[] = [];
    for (const item of value) {
      const resolved = resolvePluginValue(item, context);
      if (!resolved.found) {
        return resolved;
      }

      resolvedArray.push(resolved.value);
    }

    return {
      found: true,
      value: resolvedArray
    };
  }

  if (isPlainObject(value)) {
    const mapped: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      const resolved = resolvePluginValue(item, context);
      if (!resolved.found) {
        return resolved;
      }

      mapped[key] = resolved.value;
    }

    return {
      found: true,
      value: mapped
    };
  }

  return {
    found: true,
    value
  };
}

function parseReferenceTemplate(value: string): string | undefined {
  const match = value.match(/^\{\{\s*([^\s}].*?)\s*\}\}$/);
  if (!match) {
    return undefined;
  }

  const reference = match[1]?.trim();
  return reference || undefined;
}

function resolveReference(reference: string, context: PluginValueResolveContext): unknown {
  const normalized = reference.trim();
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'model') {
    return context.model;
  }

  if (normalized === 'source.provider') {
    return context.sourceProvider;
  }

  if (normalized === 'source.adapterKey') {
    return context.sourceAdapterKey;
  }

  if (normalized === 'target.provider') {
    return context.targetProvider;
  }

  if (normalized === 'target.providerName') {
    return context.targetProviderName;
  }

  if (normalized.startsWith('env.')) {
    const key = normalized.slice('env.'.length).trim();
    return key ? process.env[key] : undefined;
  }

  if (normalized.startsWith('request.headers.')) {
    const headerName = normalized.slice('request.headers.'.length).trim();
    return readRequestHeader(context.request.headers, headerName);
  }

  if (normalized.startsWith('request.query.')) {
    const path = normalized.slice('request.query.'.length).trim();
    return readValueByPath(context.request.query, path);
  }

  if (normalized === 'request.body') {
    return context.request.body;
  }

  if (normalized.startsWith('request.body.')) {
    const path = normalized.slice('request.body.'.length).trim();
    return readValueByPath(context.request.body, path);
  }

  if (normalized === 'standardRequest') {
    return context.standardRequest;
  }

  if (normalized.startsWith('standardRequest.')) {
    const path = normalized.slice('standardRequest.'.length).trim();
    return readValueByPath(context.standardRequest, path);
  }

  if (normalized.startsWith('upstreamRequest.headers.')) {
    const headerName = normalized.slice('upstreamRequest.headers.'.length).trim();
    return readHeaderValue(context.upstreamRequest.headers, headerName);
  }

  if (normalized.startsWith('upstreamRequest.query.')) {
    const queryKey = normalized.slice('upstreamRequest.query.'.length).trim();
    if (!queryKey) {
      return undefined;
    }

    try {
      const parsed = new URL(context.upstreamRequest.url);
      return parsed.searchParams.get(queryKey) ?? undefined;
    } catch {
      return undefined;
    }
  }

  if (normalized === 'upstreamRequest.body') {
    return context.upstreamRequest.body;
  }

  if (normalized.startsWith('upstreamRequest.body.')) {
    const path = normalized.slice('upstreamRequest.body.'.length).trim();
    return readValueByPath(context.upstreamRequest.body, path);
  }

  if (normalized === 'upstreamPayload') {
    return context.upstreamPayload;
  }

  if (normalized.startsWith('upstreamPayload.')) {
    const path = normalized.slice('upstreamPayload.'.length).trim();
    return readValueByPath(context.upstreamPayload, path);
  }

  return undefined;
}

function readRequestHeader(
  headers: FastifyRequest['headers'],
  headerName: string
): string | undefined {
  const value = (headers as Record<string, unknown>)[headerName.toLowerCase()];
  if (typeof value === 'string') {
    const normalized = value.trim();
    return normalized || undefined;
  }

  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string') {
    const normalized = value[0].trim();
    return normalized || undefined;
  }

  return undefined;
}

function readHeaderValue(headers: Record<string, string>, headerName: string): string | undefined {
  const target = headerName.trim().toLowerCase();
  if (!target) {
    return undefined;
  }

  for (const [key, value] of Object.entries(headers)) {
    if (key.trim().toLowerCase() === target) {
      return value;
    }
  }

  return undefined;
}

function setHeaderValue(headers: Record<string, string>, headerName: string, value: string): void {
  removeHeaderValue(headers, headerName);
  headers[headerName] = value;
}

function removeHeaderValue(headers: Record<string, string>, headerName: string): void {
  const target = headerName.trim().toLowerCase();
  if (!target) {
    return;
  }

  for (const key of Object.keys(headers)) {
    if (key.trim().toLowerCase() === target) {
      delete headers[key];
    }
  }
}

function serializeValueForHeader(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return JSON.stringify(value ?? null);
}

function serializeValueForQuery(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return JSON.stringify(value ?? null);
}

function setValueAtPath(target: Record<string, unknown>, path: string, value: unknown): void {
  const segments = splitPath(path);
  if (segments.length === 0) {
    if (isPlainObject(value)) {
      for (const key of Object.keys(target)) {
        delete target[key];
      }

      for (const [key, item] of Object.entries(value)) {
        target[key] = cloneUnknown(item);
      }
    }
    return;
  }

  let current: Record<string, unknown> = target;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const segment = segments[i];
    const next = current[segment];
    if (!isPlainObject(next)) {
      current[segment] = {};
    }

    current = current[segment] as Record<string, unknown>;
  }

  current[segments[segments.length - 1]] = value;
}

function removeValueAtPath(target: Record<string, unknown>, path: string): void {
  const segments = splitPath(path);
  if (segments.length === 0) {
    return;
  }

  let current: Record<string, unknown> = target;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const segment = segments[i];
    const next = current[segment];
    if (!isPlainObject(next)) {
      return;
    }

    current = next;
  }

  delete current[segments[segments.length - 1]];
}

function readValueByPath(value: unknown, path: string): unknown {
  const segments = splitPath(path);
  if (segments.length === 0) {
    return value;
  }

  let current: unknown = value;
  for (const segment of segments) {
    if (!isPlainObject(current)) {
      return undefined;
    }

    current = current[segment];
  }

  return current;
}

function splitPath(path: string): string[] {
  return path
    .split('.')
    .map((segment) => segment.trim())
    .filter(Boolean);
}

function mergeJsonObjects(
  base: Record<string, unknown>,
  override: Record<string, unknown>
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...base };
  for (const [key, overrideValue] of Object.entries(override)) {
    const currentValue = merged[key];
    if (isPlainObject(currentValue) && isPlainObject(overrideValue)) {
      merged[key] = mergeJsonObjects(currentValue, overrideValue);
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

  if (!isPlainObject(value)) {
    return value;
  }

  const cloned: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    cloned[key] = cloneUnknown(item);
  }
  return cloned;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isReferenceObject(value: unknown): value is { from: string; default?: unknown } {
  if (!isPlainObject(value)) {
    return false;
  }

  if (typeof value.from !== 'string') {
    return false;
  }

  return Object.keys(value).every((key) => key === 'from' || key === 'default');
}
