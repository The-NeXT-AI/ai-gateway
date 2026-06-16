import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { readHeader } from '../utils';
import {
  buildJsonRpcErrorResponse,
  handleMcpJsonRpcMethod,
  parseJsonRpcRequest
} from './jsonrpc';
import { McpGatewayError, McpGatewayOAuthError, type McpGatewayRuntime } from './runtime';

export function registerMcpGatewayRoutes(fastify: FastifyInstance, runtime: McpGatewayRuntime): void {
  if (!runtime.enabled || !runtime.endpointPath) {
    return;
  }

  ensureFormBodyParser(fastify);

  fastify.post<{ Body: unknown }>(runtime.endpointPath, async (request, reply) => {
    const parsed = parseJsonRpcRequest(request.body);
    if (!parsed.ok) {
      return reply
        .code(400)
        .send(buildJsonRpcErrorResponse(null, -32600, parsed.error || 'Invalid JSON-RPC request.'));
    }

    const authResult = runtime.authenticate(request);
    if (!authResult.ok || !authResult.context) {
      if (runtime.oauthEnabled && (authResult.statusCode || 401) === 401) {
        reply.header(
          'www-authenticate',
          runtime.buildOAuthWwwAuthenticateHeader(
            buildOauthRequestContext(request, runtime.endpointPath),
            'invalid_token',
            authResult.error || 'Unauthorized'
          )
        );
      }
      return reply
        .code(authResult.statusCode || 401)
        .send(buildJsonRpcErrorResponse(parsed.request.id, -32001, authResult.error || 'Unauthorized'));
    }

    try {
      const result = await handleMcpJsonRpcMethod(
        runtime,
        authResult.context,
        parsed.request.method,
        parsed.request.params
      );

      return reply.send({
        jsonrpc: '2.0',
        id: parsed.request.id,
        result
      });
    } catch (error) {
      return sendJsonRpcMethodError(reply, parsed.request.id, error);
    }
  });

  if (!runtime.oauthEnabled) {
    return;
  }

  for (const path of runtime.oauthProtectedResourceDiscoveryPaths) {
    fastify.get(path, async (request) => {
      return runtime.buildProtectedResourceMetadata(buildOauthRequestContext(request, runtime.endpointPath));
    });
  }

  for (const path of runtime.oauthAuthorizationServerDiscoveryPaths) {
    fastify.get(path, async (request) => {
      return runtime.buildAuthorizationServerMetadata(buildOauthRequestContext(request, runtime.endpointPath));
    });
  }

  fastify.get('/oauth/authorize', async (request, reply) => {
    const query = isRecord(request.query) ? request.query : {};
    const redirectUri = readObjectParam(query.redirect_uri);
    const state = readObjectParam(query.state);

    try {
      const redirect = runtime.buildOAuthAuthorizeRedirectUrl({
        clientId: readRequiredObjectParam(query.client_id, 'client_id'),
        redirectUri: readRequiredObjectParam(query.redirect_uri, 'redirect_uri'),
        responseType: readObjectParam(query.response_type),
        state,
        scope: readObjectParam(query.scope),
        resource: readObjectParam(query.resource),
        codeChallenge: readObjectParam(query.code_challenge),
        codeChallengeMethod: readObjectParam(query.code_challenge_method)
      });
      return reply.redirect(redirect, 302);
    } catch (error) {
      if (error instanceof McpGatewayOAuthError && redirectUri) {
        const errorRedirect = buildOAuthErrorRedirect(redirectUri, error.error, error.message, state);
        if (errorRedirect) {
          return reply.redirect(errorRedirect, 302);
        }
      }

      return sendOAuthError(reply, error);
    }
  });

  fastify.post('/oauth/token', async (request, reply) => {
    try {
      const params = parseOAuthTokenRequest(request);
      const token = runtime.exchangeOAuthToken(params);
      return reply.send(token);
    } catch (error) {
      return sendOAuthError(reply, error);
    }
  });
}

function sendJsonRpcMethodError(
  reply: FastifyReply,
  id: string | number | null,
  error: unknown
): FastifyReply {
  if (error instanceof McpGatewayError) {
    const code = Number.isInteger(error.code) ? error.code : -32000;
    return reply
      .code(error.statusCode || 400)
      .send(buildJsonRpcErrorResponse(id, code, error.message, error.data));
  }

  const message = error instanceof Error ? error.message : String(error);
  return reply.code(500).send(buildJsonRpcErrorResponse(id, -32603, message));
}

function buildOauthRequestContext(request: FastifyRequest, endpointPath: string): {
  origin: string;
  endpointPath: string;
} {
  const protocol = readHeader(request.headers['x-forwarded-proto']) || request.protocol || 'http';
  const host =
    readHeader(request.headers['x-forwarded-host']) || readHeader(request.headers.host) || 'localhost';

  return {
    origin: `${protocol}://${host}`,
    endpointPath
  };
}

function ensureFormBodyParser(fastify: FastifyInstance): void {
  if (fastify.hasContentTypeParser('application/x-www-form-urlencoded')) {
    return;
  }

  fastify.addContentTypeParser(
    'application/x-www-form-urlencoded',
    { parseAs: 'string' },
    (_request, body, done) => {
      done(null, parseFormBody(body));
    }
  );
}

function parseFormBody(body: string | Buffer): Record<string, string | string[]> {
  const params = new URLSearchParams(Buffer.isBuffer(body) ? body.toString('utf8') : body);
  const parsed: Record<string, string | string[]> = {};
  for (const [key, value] of params.entries()) {
    const existing = parsed[key];
    if (existing === undefined) {
      parsed[key] = value;
      continue;
    }

    if (Array.isArray(existing)) {
      existing.push(value);
      continue;
    }

    parsed[key] = [existing, value];
  }
  return parsed;
}

function parseOAuthTokenRequest(request: FastifyRequest): {
  grantType: string;
  clientId?: string;
  code?: string;
  redirectUri?: string;
  codeVerifier?: string;
  refreshToken?: string;
} {
  const body = isRecord(request.body) ? request.body : {};
  const basicAuth = readBasicAuthClientId(request.headers.authorization);

  return {
    grantType: readRequiredObjectParam(body.grant_type, 'grant_type'),
    clientId: readObjectParam(body.client_id) || basicAuth,
    code: readObjectParam(body.code),
    redirectUri: readObjectParam(body.redirect_uri),
    codeVerifier: readObjectParam(body.code_verifier),
    refreshToken: readObjectParam(body.refresh_token)
  };
}

function readBasicAuthClientId(authorizationHeader: unknown): string | undefined {
  const authorization = Array.isArray(authorizationHeader) || typeof authorizationHeader === 'string'
    ? readHeader(authorizationHeader)
    : undefined;
  if (!authorization) {
    return undefined;
  }

  const prefix = 'Basic ';
  if (!authorization.startsWith(prefix)) {
    return undefined;
  }

  try {
    const decoded = Buffer.from(authorization.slice(prefix.length).trim(), 'base64').toString('utf8');
    const separatorIndex = decoded.indexOf(':');
    if (separatorIndex < 0) {
      return undefined;
    }
    const clientId = decoded.slice(0, separatorIndex).trim();
    return clientId || undefined;
  } catch {
    return undefined;
  }
}

function sendOAuthError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof McpGatewayOAuthError) {
    return reply.code(error.statusCode).send({
      error: error.error,
      error_description: error.message
    });
  }

  return reply.code(500).send({
    error: 'server_error',
    error_description: error instanceof Error ? error.message : String(error)
  });
}

function buildOAuthErrorRedirect(
  redirectUri: string,
  error: string,
  errorDescription: string,
  state?: string
): string | undefined {
  try {
    const redirect = new URL(redirectUri);
    redirect.searchParams.set('error', error);
    redirect.searchParams.set('error_description', errorDescription);
    if (state) {
      redirect.searchParams.set('state', state);
    }
    return redirect.toString();
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function readObjectParam(value: unknown): string | undefined {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed || undefined;
  }

  if (Array.isArray(value) && typeof value[0] === 'string') {
    const trimmed = value[0].trim();
    return trimmed || undefined;
  }

  return undefined;
}

function readRequiredObjectParam(value: unknown, fieldName: string): string {
  const parsed = readObjectParam(value);
  if (!parsed) {
    throw new McpGatewayOAuthError('invalid_request', `Missing ${fieldName}.`);
  }
  return parsed;
}
