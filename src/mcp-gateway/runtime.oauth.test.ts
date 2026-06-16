import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import type { McpGatewayConfig } from '../types';
import { createMcpGatewayRuntime, McpGatewayOAuthError } from './runtime';

function createConfig(): McpGatewayConfig {
  return {
    enabled: true,
    endpoint: '/mcp',
    websocket: {
      enabled: true,
      endpoint: '/mcp/ws',
      auth: {
        allowQueryToken: true,
        queryTokenParam: 'token'
      }
    },
    principals: [
      {
        key: 'mcp-test-key',
        team: 'eng',
        allowServers: ['*'],
        allowTools: ['*'],
        denyTools: []
      }
    ],
    serverExposure: {},
    internalCidrs: [],
    guardrails: {
      enabled: false,
      maxArgumentBytes: 1024 * 1024,
      blockedTools: [],
      blockedArgumentKeys: [],
      redactArgumentKeys: []
    },
    oauth: {
      enabled: true,
      scopesSupported: ['mcp:tools:list', 'mcp:tools:call'],
      defaultPrincipalKey: 'mcp-test-key',
      authorizationCodeTtlSec: 120,
      accessTokenTtlSec: 600,
      refreshTokenTtlSec: 3600
    }
  };
}

function parseCode(redirectUrl: string): string {
  const parsed = new URL(redirectUrl);
  const code = parsed.searchParams.get('code');
  if (!code) {
    throw new Error('missing code');
  }
  return code;
}

describe('mcp gateway oauth runtime', () => {
  it('issues authorization code, exchanges token, and authenticates bearer token', () => {
    const runtime = createMcpGatewayRuntime({
      config: createConfig(),
      servers: []
    });

    const redirect = runtime.buildOAuthAuthorizeRedirectUrl({
      clientId: 'codex-client',
      redirectUri: 'http://127.0.0.1:9900/callback',
      state: 'state-1',
      scope: 'mcp:tools:list'
    });
    const code = parseCode(redirect);

    const token = runtime.exchangeOAuthToken({
      grantType: 'authorization_code',
      clientId: 'codex-client',
      code,
      redirectUri: 'http://127.0.0.1:9900/callback'
    });

    expect(token.access_token).toContain('atk_');
    expect(token.refresh_token).toContain('rtk_');
    expect(token.scope).toBe('mcp:tools:list');

    const auth = runtime.authenticateSocket({
      authorization: `Bearer ${token.access_token}`
    });
    expect(auth.ok).toBe(true);
    expect(auth.context?.key).toBe('oauth:codex-client');
    expect(auth.context?.principal.key).toBe('mcp-test-key');
  });

  it('validates pkce code_verifier when code_challenge is provided', () => {
    const runtime = createMcpGatewayRuntime({
      config: createConfig(),
      servers: []
    });

    const verifier = 'pkce-verifier';
    const challenge = createHash('sha256').update(verifier, 'utf8').digest('base64url');
    const redirect = runtime.buildOAuthAuthorizeRedirectUrl({
      clientId: 'codex-client',
      redirectUri: 'http://127.0.0.1:9901/callback',
      codeChallenge: challenge,
      codeChallengeMethod: 'S256'
    });
    const code = parseCode(redirect);

    expect(() =>
      runtime.exchangeOAuthToken({
        grantType: 'authorization_code',
        clientId: 'codex-client',
        code,
        redirectUri: 'http://127.0.0.1:9901/callback',
        codeVerifier: 'wrong-verifier'
      })
    ).toThrowError(McpGatewayOAuthError);

    const secondRedirect = runtime.buildOAuthAuthorizeRedirectUrl({
      clientId: 'codex-client',
      redirectUri: 'http://127.0.0.1:9901/callback',
      codeChallenge: challenge,
      codeChallengeMethod: 'S256'
    });
    const secondCode = parseCode(secondRedirect);
    const token = runtime.exchangeOAuthToken({
      grantType: 'authorization_code',
      clientId: 'codex-client',
      code: secondCode,
      redirectUri: 'http://127.0.0.1:9901/callback',
      codeVerifier: verifier
    });
    expect(token.access_token).toContain('atk_');
  });

  it('supports refresh token grant and rotates refresh token', () => {
    const runtime = createMcpGatewayRuntime({
      config: createConfig(),
      servers: []
    });

    const redirect = runtime.buildOAuthAuthorizeRedirectUrl({
      clientId: 'codex-client',
      redirectUri: 'http://127.0.0.1:9902/callback'
    });
    const code = parseCode(redirect);
    const initial = runtime.exchangeOAuthToken({
      grantType: 'authorization_code',
      clientId: 'codex-client',
      code,
      redirectUri: 'http://127.0.0.1:9902/callback'
    });

    const refreshed = runtime.exchangeOAuthToken({
      grantType: 'refresh_token',
      clientId: 'codex-client',
      refreshToken: initial.refresh_token
    });

    expect(refreshed.access_token).not.toBe(initial.access_token);
    expect(refreshed.refresh_token).not.toBe(initial.refresh_token);

    expect(() =>
      runtime.exchangeOAuthToken({
        grantType: 'refresh_token',
        clientId: 'codex-client',
        refreshToken: initial.refresh_token
      })
    ).toThrowError(McpGatewayOAuthError);
  });

  it('exposes codex-compatible oauth discovery paths', () => {
    const runtime = createMcpGatewayRuntime({
      config: createConfig(),
      servers: []
    });

    expect(runtime.oauthAuthorizationServerDiscoveryPaths).toEqual([
      '/.well-known/oauth-authorization-server/mcp',
      '/mcp/.well-known/oauth-authorization-server',
      '/.well-known/oauth-authorization-server'
    ]);
    expect(runtime.oauthProtectedResourceDiscoveryPaths).toEqual([
      '/.well-known/oauth-protected-resource/mcp',
      '/mcp/.well-known/oauth-protected-resource',
      '/.well-known/oauth-protected-resource'
    ]);
  });
});
