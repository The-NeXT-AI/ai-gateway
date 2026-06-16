import { createHmac } from 'node:crypto';
import type { FastifyRequest } from 'fastify';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GatewayAuthConfig } from '../types';
import { authenticateGatewayRequest } from './auth';

const baseConfig: GatewayAuthConfig = {
  enabled: true,
  mode: 'trusted_header',
  required: true,
  trustedCidrs: [],
  identityHeaders: {
    userId: 'x-auth-user-id',
    tenantId: 'x-auth-tenant-id',
    subject: 'x-auth-sub',
    organizationId: 'x-auth-organization-id',
    plan: 'x-auth-plan'
  },
  signature: {
    enabled: false,
    header: 'x-auth-signature',
    timestampHeader: 'x-auth-ts',
    secretEnv: 'AUTH_HEADER_SIGNING_SECRET',
    maxSkewSec: 120
  },
  introspection: {
    endpoint: 'http://auth.local/introspect',
    timeoutMs: 3000,
    tokenHeader: 'authorization',
    tokenBearerOnly: true,
    requestTokenField: 'token',
    credentialHeader: 'x-gateway-auth',
    credentialEnv: 'AUTH_INTROSPECTION_SHARED_SECRET',
    responseMap: {
      active: 'active',
      userId: 'userId',
      tenantId: 'tenantId',
      subject: 'sub',
      organizationId: 'organizationId',
      plan: 'plan'
    }
  }
};

describe('gateway auth', () => {
  const originalSecret = process.env.AUTH_HEADER_SIGNING_SECRET;
  const originalIntrospectionSecret = process.env.AUTH_INTROSPECTION_SHARED_SECRET;

  afterEach(() => {
    if (originalSecret === undefined) {
      delete process.env.AUTH_HEADER_SIGNING_SECRET;
      return;
    }

    process.env.AUTH_HEADER_SIGNING_SECRET = originalSecret;

    if (originalIntrospectionSecret === undefined) {
      delete process.env.AUTH_INTROSPECTION_SHARED_SECRET;
    } else {
      process.env.AUTH_INTROSPECTION_SHARED_SECRET = originalIntrospectionSecret;
    }

    vi.restoreAllMocks();
  });

  it('allows requests when auth is disabled', async () => {
    const request = createRequest();
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      enabled: false
    });

    expect(result.ok).toBe(true);
  });

  it('rejects requests without identity when auth is required', async () => {
    const request = createRequest();
    const result = await authenticateGatewayRequest(request, baseConfig);

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.statusCode).toBe(401);
  });

  it('extracts billing identity from trusted headers', async () => {
    const request = createRequest({
      'x-auth-user-id': 'user-1',
      'x-auth-tenant-id': 'tenant-a'
    });
    const result = await authenticateGatewayRequest(request, baseConfig);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity?.billingSubjectKey).toBe('tenant-a:user-1');
    expect(result.identity?.userId).toBe('user-1');
    expect(result.identity?.tenantId).toBe('tenant-a');
    expect(result.identity?.source).toBe('trusted_header');
  });

  it('rejects requests from untrusted cidr when trusted list is configured', async () => {
    const request = createRequest(
      {
        'x-auth-user-id': 'user-1'
      },
      {
        ip: '8.8.8.8'
      }
    );
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      trustedCidrs: ['10.0.0.0/8']
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.statusCode).toBe(403);
  });

  it('validates signature when enabled', async () => {
    process.env.AUTH_HEADER_SIGNING_SECRET = 'test-secret';

    const timestamp = String(Math.floor(Date.now() / 1000));
    const request = createRequest({
      'x-auth-user-id': 'user-1',
      'x-auth-tenant-id': 'tenant-a',
      'x-auth-sub': 'sub-1',
      'x-auth-plan': 'pro'
    });
    const signature = createSignature(timestamp, request, {
      tenantId: 'tenant-a',
      userId: 'user-1',
      subject: 'sub-1',
      plan: 'pro'
    });
    request.headers['x-auth-ts'] = timestamp;
    request.headers['x-auth-signature'] = signature;

    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      signature: {
        ...baseConfig.signature,
        enabled: true
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity?.billingSubjectKey).toBe('tenant-a:user-1');
  });

  it('rejects expired signature', async () => {
    process.env.AUTH_HEADER_SIGNING_SECRET = 'test-secret';

    const timestamp = String(Math.floor(Date.now() / 1000) - 600);
    const request = createRequest({
      'x-auth-user-id': 'user-1',
      'x-auth-tenant-id': 'tenant-a'
    });
    const signature = createSignature(timestamp, request, {
      tenantId: 'tenant-a',
      userId: 'user-1'
    });
    request.headers['x-auth-ts'] = timestamp;
    request.headers['x-auth-signature'] = signature;

    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      signature: {
        ...baseConfig.signature,
        enabled: true,
        maxSkewSec: 30
      }
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.statusCode).toBe(401);
  });

  it('authenticates with http introspection and maps identity', async () => {
    process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'introspection-secret';
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          active: true,
          userId: 'user-2',
          tenantId: 'tenant-b',
          sub: 'sub-2',
          organizationId: 'org-2',
          plan: 'enterprise'
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer token-123'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection'
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity?.source).toBe('http_introspection');
    expect(result.identity?.billingSubjectKey).toBe('tenant-b:user-2');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reads identity fields from envelope data payload', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          code: 200,
          message: 'Success',
          data: {
            active: true,
            userId: 'env-user',
            tenantId: 'env-tenant'
          }
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer token-envelope'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection'
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity?.billingSubjectKey).toBe('env-tenant:env-user');
  });

  it('treats envelope data.active=false as inactive token', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          code: 200,
          data: {
            active: false
          }
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer inactive-token'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection'
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.statusCode).toBe(401);
    expect(result.error).toContain('inactive');
  });

  it('supports Authorization bearer fallback when token header is x-api-key', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          active: true,
          userId: 'fallback-user'
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer fallback-token'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      introspection: {
        ...baseConfig.introspection,
        tokenHeader: 'x-api-key',
        tokenBearerOnly: false
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    const fetchOptions = (fetchMock.mock.calls[0] as unknown as [string, RequestInit] | undefined)?.[1];
    const requestBody = fetchOptions?.body ? JSON.parse(fetchOptions.body as string) : {};
    expect(requestBody.token).toBe('fallback-token');
  });

  it('supports x-codex-access-token fallback when token header is x-api-key', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          active: true,
          userId: 'codex-fallback-user'
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      'x-codex-access-token': 'codex-token-123'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      introspection: {
        ...baseConfig.introspection,
        tokenHeader: 'x-api-key',
        tokenBearerOnly: false
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    const fetchOptions = (fetchMock.mock.calls[0] as unknown as [string, RequestInit] | undefined)?.[1];
    const requestBody = fetchOptions?.body ? JSON.parse(fetchOptions.body as string) : {};
    expect(requestBody.token).toBe('codex-token-123');
  });

  it('allows loopback internal agent requests with shared secret without introspection', async () => {
    process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'introspection-secret';
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest(
      {
        'x-gateway-agent-internal': '1',
        'x-gateway-auth': 'introspection-secret'
      },
      {
        ip: '127.0.0.1'
      }
    );
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection'
    });

    expect(result.ok).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('preserves propagated identity for loopback internal agent requests', async () => {
    process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'introspection-secret';
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest(
      {
        'x-gateway-agent-internal': '1',
        'x-gateway-auth': 'introspection-secret',
        'x-auth-user-id': 'user-1',
        'x-auth-tenant-id': 'tenant-a',
        'x-auth-sub': 'user-1',
        'x-auth-organization-id': 'org-1',
        'x-auth-plan': 'project',
        'x-auth-api-key-id': 'key-1'
      },
      {
        ip: '127.0.0.1'
      }
    );
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      identityHeaders: {
        ...baseConfig.identityHeaders,
        apiKeyId: 'x-auth-api-key-id'
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity).toMatchObject({
      source: 'http_introspection',
      billingSubjectKey: 'tenant-a:user-1',
      userId: 'user-1',
      tenantId: 'tenant-a',
      subject: 'user-1',
      organizationId: 'org-1',
      plan: 'project',
      apiKeyId: 'key-1'
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not bypass introspection for non-loopback internal marker requests', async () => {
    process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'introspection-secret';
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          active: false
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      'x-gateway-agent-internal': '1',
      'x-gateway-auth': 'introspection-secret',
      'x-api-key': 'provider-key'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      introspection: {
        ...baseConfig.introspection,
        tokenHeader: 'x-api-key',
        tokenBearerOnly: false
      }
    });

    expect(result.ok).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('rejects inactive token from introspection response', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          active: false
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer token-123'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection'
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.statusCode).toBe(401);
  });

  it('supports nested response map paths in introspection mode', async () => {
    const fetchMock = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          auth: { active: true },
          user: { id: 'nested-user' },
          tenant: { id: 'nested-tenant' }
        })
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest({
      authorization: 'Bearer token-xyz'
    });
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      introspection: {
        ...baseConfig.introspection,
        responseMap: {
          ...baseConfig.introspection.responseMap,
          active: 'auth.active',
          userId: 'user.id',
          tenantId: 'tenant.id'
        }
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.identity?.billingSubjectKey).toBe('nested-tenant:nested-user');
  });

  it('allows missing token when introspection is optional', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const request = createRequest();
    const result = await authenticateGatewayRequest(request, {
      ...baseConfig,
      mode: 'http_introspection',
      required: false
    });

    expect(result.ok).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

function createRequest(
  headers: Record<string, string> = {},
  overrides: {
    ip?: string;
    method?: string;
    url?: string;
  } = {}
): FastifyRequest {
  return {
    headers,
    ip: overrides.ip || '10.0.0.10',
    method: overrides.method || 'POST',
    url: overrides.url || '/v1/responses'
  } as FastifyRequest;
}

function createSignature(
  timestamp: string,
  request: FastifyRequest,
  identity: {
    tenantId?: string;
    userId?: string;
    subject?: string;
    organizationId?: string;
    plan?: string;
  }
): string {
  const path = new URL(request.url, 'http://gateway.local').pathname;
  const payload = [
    timestamp,
    request.method.toUpperCase(),
    path,
    identity.tenantId || '',
    identity.userId || '',
    identity.subject || '',
    identity.organizationId || '',
    identity.plan || ''
  ].join('\n');

  const secret = process.env.AUTH_HEADER_SIGNING_SECRET as string;
  return createHmac('sha256', secret).update(payload).digest('hex');
}
