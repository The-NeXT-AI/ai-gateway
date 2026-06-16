import { createHash, randomBytes, randomUUID } from 'node:crypto';
import type { IncomingHttpHeaders } from 'node:http';
import type { FastifyRequest } from 'fastify';
import { createMcpAgentToolProvider, type AgentToolProvider } from '../agent/tools';
import { createInitialGuards } from '../agent/guards';
import { createInitialTaskState } from '../agent/task-state';
import { createTranscriptWindow } from '../agent/transcript-window';
import type {
  AgentEvent,
  AgentSessionState,
  AgentToolDefinition,
  ToolCallRequestedPayload
} from '../agent/types';
import type { AgentMcpServerConfig, McpGatewayConfig, McpGatewayPrincipalConfig } from '../types';
import { isObject, readBearerToken, readHeader } from '../utils';
import {
  containsBlockedArgumentKeys,
  filterToolsByPolicy,
  isInternalIp,
  isToolAllowed,
  matchesAnyPattern,
  redactSensitiveArguments,
  toLowerSet
} from './policy';

export interface McpGatewayLogger {
  info?(context: unknown, message?: string): void;
  warn?(context: unknown, message?: string): void;
  error?(context: unknown, message?: string): void;
}

export interface CreateMcpGatewayRuntimeOptions {
  config: McpGatewayConfig;
  servers: AgentMcpServerConfig[];
  logger?: McpGatewayLogger;
  toolProvider?: AgentToolProvider;
}

export interface McpGatewayPrincipalContext {
  key: string;
  team: string;
  organization?: string;
  principal: McpGatewayPrincipalConfig;
  clientIp: string;
  isInternalCaller: boolean;
}

export interface McpGatewayAuthResult {
  ok: boolean;
  context?: McpGatewayPrincipalContext;
  statusCode?: number;
  error?: string;
}

export interface McpGatewayOAuthRequestContext {
  origin: string;
  endpointPath: string;
}

export interface McpGatewayOAuthAuthorizeParams {
  clientId: string;
  redirectUri: string;
  responseType?: string;
  state?: string;
  scope?: string;
  resource?: string;
  codeChallenge?: string;
  codeChallengeMethod?: string;
}

export interface McpGatewayOAuthTokenRequest {
  grantType: string;
  clientId?: string;
  code?: string;
  redirectUri?: string;
  codeVerifier?: string;
  refreshToken?: string;
}

export interface McpGatewayOAuthTokenResponse {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
  refresh_token: string;
  scope?: string;
}

export class McpGatewayOAuthError extends Error {
  constructor(
    public readonly error: string,
    message: string,
    public readonly statusCode = 400
  ) {
    super(message);
    this.name = 'McpGatewayOAuthError';
  }
}

type OAuthCodeChallengeMethod = 'plain' | 'S256';

interface McpGatewayOAuthAuthorizationCodeRecord {
  code: string;
  clientId: string;
  principalKey: string;
  redirectUri: string;
  scope: string[];
  resource?: string;
  codeChallenge?: string;
  codeChallengeMethod?: OAuthCodeChallengeMethod;
  expiresAt: number;
}

interface McpGatewayOAuthAccessTokenRecord {
  accessToken: string;
  refreshToken: string;
  clientId: string;
  principalKey: string;
  scope: string[];
  resource?: string;
  expiresAt: number;
}

interface McpGatewayOAuthRefreshTokenRecord {
  refreshToken: string;
  clientId: string;
  principalKey: string;
  scope: string[];
  resource?: string;
  expiresAt: number;
}

interface ResolvedOAuthAccessToken {
  record: McpGatewayOAuthAccessTokenRecord;
  principal: McpGatewayPrincipalConfig;
  auditKey: string;
}

export class McpGatewayError extends Error {
  constructor(
    public readonly code: number,
    message: string,
    public readonly statusCode = 400,
    public readonly data?: unknown
  ) {
    super(message);
    this.name = 'McpGatewayError';
  }
}

export class McpGatewayRuntime {
  private readonly blockedArgumentKeySet: Set<string>;
  private readonly redactArgumentKeySet: Set<string>;
  private readonly oauthAuthorizationCodes = new Map<string, McpGatewayOAuthAuthorizationCodeRecord>();
  private readonly oauthAccessTokens = new Map<string, McpGatewayOAuthAccessTokenRecord>();
  private readonly oauthRefreshTokens = new Map<string, McpGatewayOAuthRefreshTokenRecord>();

  constructor(private readonly options: CreateMcpGatewayRuntimeOptions) {
    this.blockedArgumentKeySet = toLowerSet(options.config.guardrails.blockedArgumentKeys);
    this.redactArgumentKeySet = toLowerSet(options.config.guardrails.redactArgumentKeys);
  }

  async close(): Promise<void> {
    if (this.options.toolProvider) {
      await this.options.toolProvider.close();
    }
  }

  authenticate(request: FastifyRequest): McpGatewayAuthResult {
    return this.authenticateWithHeaders(request.headers, request.ip);
  }

  authenticateSocket(headers: IncomingHttpHeaders, clientIp?: string): McpGatewayAuthResult {
    return this.authenticateWithHeaders(headers, clientIp);
  }

  private authenticateWithHeaders(
    headers: McpGatewayHeaderBag,
    fallbackClientIp?: string
  ): McpGatewayAuthResult {
    if (!this.options.config.enabled) {
      return {
        ok: false,
        statusCode: 404,
        error: 'MCP Gateway is disabled.'
      };
    }

    const keyFromHeader = readApiKeyHeader(headers);
    const bearerToken = readBearerToken(readHeader(headers.authorization))?.trim();
    const credential = keyFromHeader || bearerToken;
    if (!credential) {
      return {
        ok: false,
        statusCode: 401,
        error: 'Missing API key. Use Authorization: Bearer <key> or x-api-key.'
      };
    }

    const principalFromApiKey = this.options.config.principals.find((item) => item.key === credential);
    const oauthAccess = principalFromApiKey
      ? undefined
      : bearerToken
        ? this.resolveOAuthAccessToken(bearerToken)
        : undefined;
    const principal = principalFromApiKey || oauthAccess?.principal;
    if (!principal) {
      return {
        ok: false,
        statusCode: bearerToken && this.oauthEnabled ? 401 : 403,
        error: bearerToken && this.oauthEnabled
          ? 'Access token is invalid or expired for MCP Gateway.'
          : 'API key is not allowed for MCP Gateway.'
      };
    }

    const clientIp = readClientIp(headers, fallbackClientIp);
    const isInternalCaller = isInternalIp(clientIp, this.options.config.internalCidrs);
    const resolvedKey = principalFromApiKey ? credential : oauthAccess?.auditKey || credential;

    return {
      ok: true,
      context: {
        key: resolvedKey,
        team: principal.team,
        organization: principal.organization,
        principal,
        clientIp,
        isInternalCaller
      }
    };
  }

  async listTools(context: McpGatewayPrincipalContext): Promise<AgentToolDefinition[]> {
    const availableTools = await this.toolProvider.listDefinitions();
    const filteredTools = filterToolsByPolicy(availableTools, context, this.options.config)
      .map((tool) => ({ ...tool }))
      .sort((a, b) => a.name.localeCompare(b.name));

    this.options.logger?.info?.(
      {
        principal: buildPrincipalAuditInfo(context),
        availableToolCount: availableTools.length,
        exposedToolCount: filteredTools.length
      },
      'MCP tools listed through gateway.'
    );

    return filteredTools;
  }

  buildInitializeResponse(): Record<string, unknown> {
    return {
      protocolVersion: '2024-11-05',
      capabilities: {
        tools: {
          listChanged: false
        }
      },
      serverInfo: {
        name: 'next-ai-gateway-mcp',
        version: '1.0.0'
      },
      instructions:
        'Use tools/list for discovery and tools/call for invocation. Tool names are canonical: <server>.<tool>.'
    };
  }

  async callTool(
    context: McpGatewayPrincipalContext,
    canonicalToolName: string,
    args: Record<string, unknown>,
    meta?: Record<string, unknown>
  ): Promise<unknown> {
    if (!isToolAllowed(canonicalToolName, context, this.options.config)) {
      throw new McpGatewayError(1001, `Tool is not allowed: ${canonicalToolName}`, 403);
    }

    const hasTool = await this.toolProvider.has(canonicalToolName);
    if (!hasTool) {
      throw new McpGatewayError(1002, `Tool is not available: ${canonicalToolName}`, 404);
    }

    this.enforceGuardrails(canonicalToolName, args);

    const redactedArgs = redactSensitiveArguments(args, this.redactArgumentKeySet);

    this.options.logger?.info?.(
      {
        principal: buildPrincipalAuditInfo(context),
        toolName: canonicalToolName,
        arguments: redactedArgs
      },
      'MCP tool call accepted.'
    );

    try {
      const hydratedArgs = hydrateVirtualMultimodalReferencesFromMeta(args, meta);
      const result = await this.toolProvider.execute(
        canonicalToolName,
        buildExecutionInput(canonicalToolName, hydratedArgs)
      );

      this.options.logger?.info?.(
        {
          principal: buildPrincipalAuditInfo(context),
          toolName: canonicalToolName
        },
        'MCP tool call finished.'
      );

      return result;
    } catch (error) {
      this.options.logger?.warn?.(
        {
          principal: buildPrincipalAuditInfo(context),
          toolName: canonicalToolName,
          details: toErrorMessage(error)
        },
        'MCP tool call failed.'
      );

      throw new McpGatewayError(2001, toErrorMessage(error), 502);
    }
  }

  buildProtectedResourceMetadata(requestContext: McpGatewayOAuthRequestContext): Record<string, unknown> {
    const resource =
      this.options.config.oauth.resource || `${requestContext.origin}${requestContext.endpointPath}`;

    return {
      resource,
      authorization_servers: this.options.config.oauth.issuer
        ? [this.options.config.oauth.issuer]
        : undefined,
      bearer_methods_supported: ['header'],
      scopes_supported: this.options.config.oauth.scopesSupported
    };
  }

  buildAuthorizationServerMetadata(
    requestContext: McpGatewayOAuthRequestContext
  ): Record<string, unknown> {
    const issuer = this.options.config.oauth.issuer || requestContext.origin;
    const authorizationEndpoint = this.resolveOAuthAuthorizationEndpoint(requestContext);
    const tokenEndpoint = this.resolveOAuthTokenEndpoint(requestContext);

    return {
      issuer,
      authorization_endpoint: authorizationEndpoint,
      token_endpoint: tokenEndpoint,
      response_types_supported: ['code'],
      grant_types_supported: ['authorization_code', 'refresh_token'],
      token_endpoint_auth_methods_supported: ['none', 'client_secret_post', 'client_secret_basic'],
      code_challenge_methods_supported: ['plain', 'S256'],
      scopes_supported: this.options.config.oauth.scopesSupported
    };
  }

  get oauthEnabled(): boolean {
    return this.options.config.oauth.enabled;
  }

  get oauthAuthorizationCodeTtlSec(): number {
    return this.options.config.oauth.authorizationCodeTtlSec || 180;
  }

  get oauthAccessTokenTtlSec(): number {
    return this.options.config.oauth.accessTokenTtlSec || 3600;
  }

  get oauthRefreshTokenTtlSec(): number {
    return this.options.config.oauth.refreshTokenTtlSec || 2592000;
  }

  get oauthAuthorizationServerDiscoveryPaths(): string[] {
    return buildOauthDiscoveryPaths(this.endpointPath, 'oauth-authorization-server');
  }

  get oauthProtectedResourceDiscoveryPaths(): string[] {
    return buildOauthDiscoveryPaths(this.endpointPath, 'oauth-protected-resource');
  }

  buildOAuthWwwAuthenticateHeader(
    requestContext: McpGatewayOAuthRequestContext,
    error?: string,
    errorDescription?: string
  ): string {
    const resourceMetadataUrl = `${requestContext.origin}${this.oauthProtectedResourceDiscoveryPaths[0]}`;
    const params: string[] = ['realm="mcp"', `resource_metadata="${resourceMetadataUrl}"`];
    if (error) {
      params.push(`error="${escapeQuoted(error)}"`);
    }
    if (errorDescription) {
      params.push(`error_description="${escapeQuoted(errorDescription)}"`);
    }
    return `Bearer ${params.join(', ')}`;
  }

  buildOAuthAuthorizeRedirectUrl(params: McpGatewayOAuthAuthorizeParams): string {
    if (!this.oauthEnabled) {
      throw new McpGatewayOAuthError('access_denied', 'MCP OAuth is disabled.', 403);
    }

    const responseType = (params.responseType || 'code').trim();
    if (responseType !== 'code') {
      throw new McpGatewayOAuthError(
        'unsupported_response_type',
        `Unsupported response_type: ${responseType}`
      );
    }

    const redirectUri = safeParseAbsoluteUrl(params.redirectUri, 'redirect_uri');
    const principalKey = this.resolveOAuthPrincipalKey(params.clientId);
    if (!principalKey) {
      throw new McpGatewayOAuthError(
        'unauthorized_client',
        'No principal is available for OAuth token exchange.',
        401
      );
    }

    const scope = this.resolveAndValidateOAuthScope(params.scope);
    const codeChallenge = normalizeOptionalString(params.codeChallenge);
    const codeChallengeMethod = normalizeCodeChallengeMethod(params.codeChallengeMethod, codeChallenge);

    const codeRecord: McpGatewayOAuthAuthorizationCodeRecord = {
      code: this.generateOAuthToken('code'),
      clientId: params.clientId,
      principalKey,
      redirectUri: redirectUri.toString(),
      scope,
      resource: normalizeOptionalString(params.resource),
      codeChallenge,
      codeChallengeMethod,
      expiresAt: Date.now() + this.oauthAuthorizationCodeTtlSec * 1000
    };
    this.cleanupExpiredOAuthRecords();
    this.oauthAuthorizationCodes.set(codeRecord.code, codeRecord);

    redirectUri.searchParams.set('code', codeRecord.code);
    const state = normalizeOptionalString(params.state);
    if (state) {
      redirectUri.searchParams.set('state', state);
    }
    return redirectUri.toString();
  }

  exchangeOAuthToken(request: McpGatewayOAuthTokenRequest): McpGatewayOAuthTokenResponse {
    if (!this.oauthEnabled) {
      throw new McpGatewayOAuthError('access_denied', 'MCP OAuth is disabled.', 403);
    }

    const grantType = normalizeOptionalString(request.grantType);
    if (!grantType) {
      throw new McpGatewayOAuthError('invalid_request', 'Missing grant_type.');
    }

    if (grantType === 'authorization_code') {
      return this.exchangeAuthorizationCode(request);
    }

    if (grantType === 'refresh_token') {
      return this.exchangeRefreshToken(request);
    }

    throw new McpGatewayOAuthError('unsupported_grant_type', `Unsupported grant_type: ${grantType}`);
  }

  get enabled(): boolean {
    return this.options.config.enabled;
  }

  get endpointPath(): string {
    return this.options.config.endpoint;
  }

  get websocketEnabled(): boolean {
    return this.options.config.websocket.enabled;
  }

  get websocketEndpointPath(): string {
    return this.options.config.websocket.endpoint;
  }

  get websocketAllowQueryToken(): boolean {
    return this.options.config.websocket.auth.allowQueryToken;
  }

  get websocketQueryTokenParam(): string {
    return this.options.config.websocket.auth.queryTokenParam;
  }

  private resolveOAuthAuthorizationEndpoint(requestContext: McpGatewayOAuthRequestContext): string {
    return this.options.config.oauth.authorizationEndpoint || `${requestContext.origin}/oauth/authorize`;
  }

  private resolveOAuthTokenEndpoint(requestContext: McpGatewayOAuthRequestContext): string {
    return this.options.config.oauth.tokenEndpoint || `${requestContext.origin}/oauth/token`;
  }

  private resolveOAuthPrincipalKey(clientId: string): string | undefined {
    const configuredDefault = normalizeOptionalString(this.options.config.oauth.defaultPrincipalKey);
    if (configuredDefault) {
      const principal = this.options.config.principals.find((item) => item.key === configuredDefault);
      if (principal) {
        return principal.key;
      }
      this.options.logger?.warn?.(
        {
          clientId,
          configuredDefaultPrincipalKey: configuredDefault
        },
        'MCP OAuth default principal key is not defined in principals config.'
      );
    }

    return this.options.config.principals[0]?.key;
  }

  private resolveAndValidateOAuthScope(rawScope: string | undefined): string[] {
    const parsed = parseScopeList(rawScope);
    if (parsed.length === 0) {
      return this.options.config.oauth.scopesSupported;
    }

    const supported = new Set(this.options.config.oauth.scopesSupported);
    for (const scope of parsed) {
      if (!supported.has(scope)) {
        throw new McpGatewayOAuthError('invalid_scope', `Unsupported scope requested: ${scope}`);
      }
    }

    return parsed;
  }

  private exchangeAuthorizationCode(request: McpGatewayOAuthTokenRequest): McpGatewayOAuthTokenResponse {
    this.cleanupExpiredOAuthRecords();
    const code = normalizeOptionalString(request.code);
    if (!code) {
      throw new McpGatewayOAuthError('invalid_request', 'Missing code.');
    }

    const redirectUri = normalizeOptionalString(request.redirectUri);
    if (!redirectUri) {
      throw new McpGatewayOAuthError('invalid_request', 'Missing redirect_uri.');
    }

    const record = this.oauthAuthorizationCodes.get(code);
    if (!record) {
      throw new McpGatewayOAuthError('invalid_grant', 'Authorization code is invalid or expired.');
    }

    const requestClientId = normalizeOptionalString(request.clientId);
    if (requestClientId && requestClientId !== record.clientId) {
      throw new McpGatewayOAuthError('invalid_grant', 'Authorization code does not match client.');
    }

    if (redirectUri !== record.redirectUri) {
      throw new McpGatewayOAuthError('invalid_grant', 'redirect_uri does not match authorization code.');
    }

    this.validateCodeVerifier(record, normalizeOptionalString(request.codeVerifier));
    this.oauthAuthorizationCodes.delete(code);

    return this.issueOAuthTokenSet({
      clientId: record.clientId,
      principalKey: record.principalKey,
      scope: record.scope,
      resource: record.resource
    });
  }

  private exchangeRefreshToken(request: McpGatewayOAuthTokenRequest): McpGatewayOAuthTokenResponse {
    this.cleanupExpiredOAuthRecords();
    const refreshToken = normalizeOptionalString(request.refreshToken);
    if (!refreshToken) {
      throw new McpGatewayOAuthError('invalid_request', 'Missing refresh_token.');
    }

    const record = this.oauthRefreshTokens.get(refreshToken);
    if (!record) {
      throw new McpGatewayOAuthError('invalid_grant', 'Refresh token is invalid or expired.');
    }

    const requestClientId = normalizeOptionalString(request.clientId);
    if (requestClientId && requestClientId !== record.clientId) {
      throw new McpGatewayOAuthError('invalid_grant', 'Refresh token does not match client.');
    }

    this.oauthRefreshTokens.delete(refreshToken);
    return this.issueOAuthTokenSet({
      clientId: record.clientId,
      principalKey: record.principalKey,
      scope: record.scope,
      resource: record.resource
    });
  }

  private issueOAuthTokenSet(input: {
    clientId: string;
    principalKey: string;
    scope: string[];
    resource?: string;
  }): McpGatewayOAuthTokenResponse {
    const accessToken = this.generateOAuthToken('atk');
    const refreshToken = this.generateOAuthToken('rtk');
    const now = Date.now();
    const accessExpiresAt = now + this.oauthAccessTokenTtlSec * 1000;
    const refreshExpiresAt = now + this.oauthRefreshTokenTtlSec * 1000;

    this.oauthAccessTokens.set(accessToken, {
      accessToken,
      refreshToken,
      clientId: input.clientId,
      principalKey: input.principalKey,
      scope: input.scope,
      resource: input.resource,
      expiresAt: accessExpiresAt
    });
    this.oauthRefreshTokens.set(refreshToken, {
      refreshToken,
      clientId: input.clientId,
      principalKey: input.principalKey,
      scope: input.scope,
      resource: input.resource,
      expiresAt: refreshExpiresAt
    });

    return {
      access_token: accessToken,
      token_type: 'bearer',
      expires_in: this.oauthAccessTokenTtlSec,
      refresh_token: refreshToken,
      scope: input.scope.length > 0 ? input.scope.join(' ') : undefined
    };
  }

  private validateCodeVerifier(
    record: McpGatewayOAuthAuthorizationCodeRecord,
    codeVerifier: string | undefined
  ): void {
    if (!record.codeChallenge) {
      return;
    }

    if (!codeVerifier) {
      throw new McpGatewayOAuthError('invalid_grant', 'Missing code_verifier.');
    }

    if ((record.codeChallengeMethod || 'plain') === 'plain') {
      if (codeVerifier !== record.codeChallenge) {
        throw new McpGatewayOAuthError('invalid_grant', 'Invalid code_verifier.');
      }
      return;
    }

    const digest = createHash('sha256').update(codeVerifier, 'utf8').digest('base64url');
    if (digest !== record.codeChallenge) {
      throw new McpGatewayOAuthError('invalid_grant', 'Invalid code_verifier.');
    }
  }

  private resolveOAuthAccessToken(token: string): ResolvedOAuthAccessToken | undefined {
    this.cleanupExpiredOAuthRecords();
    const record = this.oauthAccessTokens.get(token);
    if (!record) {
      return undefined;
    }

    const principal = this.options.config.principals.find((item) => item.key === record.principalKey);
    if (!principal) {
      return undefined;
    }

    return {
      record,
      principal,
      auditKey: `oauth:${record.clientId}`
    };
  }

  private cleanupExpiredOAuthRecords(now = Date.now()): void {
    for (const [code, record] of this.oauthAuthorizationCodes.entries()) {
      if (record.expiresAt <= now) {
        this.oauthAuthorizationCodes.delete(code);
      }
    }

    for (const [token, record] of this.oauthAccessTokens.entries()) {
      if (record.expiresAt <= now) {
        this.oauthAccessTokens.delete(token);
      }
    }

    for (const [token, record] of this.oauthRefreshTokens.entries()) {
      if (record.expiresAt <= now) {
        this.oauthRefreshTokens.delete(token);
      }
    }
  }

  private generateOAuthToken(prefix: string): string {
    return `${prefix}_${randomUUID().replace(/-/g, '')}${randomBytes(16).toString('hex')}`;
  }

  private enforceGuardrails(toolName: string, args: Record<string, unknown>): void {
    if (!this.options.config.guardrails.enabled) {
      return;
    }

    if (matchesAnyPattern(toolName, this.options.config.guardrails.blockedTools)) {
      throw new McpGatewayError(1101, `Tool blocked by guardrail: ${toolName}`, 403);
    }

    const encoded = Buffer.from(JSON.stringify(args), 'utf8');
    if (encoded.byteLength > this.options.config.guardrails.maxArgumentBytes) {
      throw new McpGatewayError(
        1102,
        `Tool arguments exceed max size (${this.options.config.guardrails.maxArgumentBytes} bytes).`,
        400
      );
    }

    if (containsBlockedArgumentKeys(args, this.blockedArgumentKeySet)) {
      throw new McpGatewayError(1103, 'Tool arguments contain blocked keys.', 403);
    }
  }

  private get toolProvider(): AgentToolProvider {
    if (this.options.toolProvider) {
      return this.options.toolProvider;
    }

    this.options.toolProvider = createMcpAgentToolProvider({
      servers: this.options.servers,
      exposureMode: 'canonical',
      logger: this.options.logger
    });

    return this.options.toolProvider;
  }
}

export function createMcpGatewayRuntime(options: CreateMcpGatewayRuntimeOptions): McpGatewayRuntime {
  return new McpGatewayRuntime(options);
}

type McpGatewayHeaderBag = Record<string, string | string[] | undefined>;

function readApiKeyHeader(headers: McpGatewayHeaderBag): string | undefined {
  const fromApiKeyHeader = readHeader(headers['x-api-key']);
  if (fromApiKeyHeader) {
    return fromApiKeyHeader.trim();
  }

  const fromMcpKeyHeader = readHeader(headers['x-mcp-key']);
  if (fromMcpKeyHeader) {
    return fromMcpKeyHeader.trim();
  }

  return undefined;
}

function readClientIp(headers: McpGatewayHeaderBag, fallbackIp?: string): string {
  const forwardedFor = readHeader(headers['x-forwarded-for']);
  if (forwardedFor) {
    const first = forwardedFor.split(',')[0]?.trim();
    if (first) {
      return first;
    }
  }

  const normalizedFallback = fallbackIp?.trim();
  if (normalizedFallback) {
    return normalizedFallback;
  }

  return '0.0.0.0';
}

function buildExecutionInput(
  toolName: string,
  args: Record<string, unknown>
): {
  args: Record<string, unknown>;
  session: AgentSessionState;
  event: AgentEvent<ToolCallRequestedPayload>;
} {
  const now = new Date().toISOString();
  const sessionId = `mcp-gateway-${randomUUID()}`;
  const toolCallId = randomUUID();

  return {
    args,
      session: {
        sessionId,
        agentId: 'mcp-gateway',
        systemPrompt: 'MCP gateway runtime session',
        allowedTools: [toolName],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        taskState: createInitialTaskState(sessionId),
        transcriptWindow: createTranscriptWindow(),
        guards: createInitialGuards(),
        lastEventOffset: 0,
        updatedAt: now
      },
    event: {
      id: randomUUID(),
      type: 'TOOL_CALL_REQUESTED',
      sessionId,
      timestamp: now,
      correlationId: randomUUID(),
      payload: {
        toolCallId,
        toolName,
        arguments: args,
        reason: 'MCP gateway invocation'
      }
    }
  };
}

function buildPrincipalAuditInfo(context: McpGatewayPrincipalContext): Record<string, unknown> {
  return {
    key: maskKey(context.key),
    team: context.team,
    organization: context.organization,
    clientIp: context.clientIp,
    callerType: context.isInternalCaller ? 'internal' : 'external'
  };
}

interface VirtualMultimodalReference {
  id: string;
  value: string;
}

function hydrateVirtualMultimodalReferencesFromMeta(
  args: Record<string, unknown>,
  meta: Record<string, unknown> | undefined
): Record<string, unknown> {
  if (!meta) {
    return args;
  }

  const references = readVirtualMultimodalReferences(meta.virtualMultimodalReferences);
  if (references.length === 0) {
    return args;
  }

  return hydrateVirtualMultimodalReferences(args, references) as Record<string, unknown>;
}

function readVirtualMultimodalReferences(value: unknown): VirtualMultimodalReference[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const references: VirtualMultimodalReference[] = [];
  for (const item of value) {
    if (!isObject(item) || typeof item.id !== 'string' || typeof item.value !== 'string') {
      continue;
    }

    references.push({
      id: item.id,
      value: item.value
    });
  }

  return references;
}

function hydrateVirtualMultimodalReferences(
  value: unknown,
  references: VirtualMultimodalReference[]
): unknown {
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

function maskKey(key: string): string {
  if (key.length <= 6) {
    return `${key.slice(0, 2)}***`;
  }

  return `${key.slice(0, 4)}***${key.slice(-2)}`;
}

function toErrorMessage(error: unknown): string {
  if (error instanceof McpGatewayError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (isObject(error) && typeof error.message === 'string') {
    return error.message;
  }

  return String(error);
}

function parseScopeList(rawScope: string | undefined): string[] {
  const scopeValue = normalizeOptionalString(rawScope);
  if (!scopeValue) {
    return [];
  }

  const deduped = new Set<string>();
  for (const scope of scopeValue.split(/\s+/)) {
    const normalized = scope.trim();
    if (normalized) {
      deduped.add(normalized);
    }
  }
  return Array.from(deduped);
}

function normalizeOptionalString(value: string | undefined): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function normalizeCodeChallengeMethod(
  methodValue: string | undefined,
  codeChallenge: string | undefined
): OAuthCodeChallengeMethod | undefined {
  if (!codeChallenge) {
    return undefined;
  }

  const method = (methodValue || 'plain').trim();
  if (method === 'S256' || method === 'plain') {
    return method;
  }

  throw new McpGatewayOAuthError(
    'invalid_request',
    `Unsupported code_challenge_method: ${method}`
  );
}

function safeParseAbsoluteUrl(value: string, fieldName: string): URL {
  try {
    return new URL(value);
  } catch {
    throw new McpGatewayOAuthError('invalid_request', `Invalid ${fieldName}.`);
  }
}

function buildOauthDiscoveryPaths(endpointPath: string, wellKnownKey: string): string[] {
  const normalizedEndpoint = endpointPath.trim();
  const canonical = `/.well-known/${wellKnownKey}`;
  const trimmed = normalizedEndpoint.replace(/^\/+/, '').replace(/\/+$/, '');
  if (!trimmed) {
    return [canonical];
  }

  const preferred = `${canonical}/${trimmed}`;
  const secondary = `/${trimmed}/.well-known/${wellKnownKey}`;
  return Array.from(new Set([preferred, secondary, canonical]));
}

function escapeQuoted(value: string): string {
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}
