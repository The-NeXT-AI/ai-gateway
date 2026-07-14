import type { FastifyRequest } from 'fastify';
import { describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import { resolveGatewayClientIp } from './client-ip';

describe('resolveGatewayClientIp', () => {
  it('uses the direct request IP when no forwarded headers are trusted', () => {
    const config = parseGatewayConfigFromRaw({});
    const request = createRequest({
      ip: '198.51.100.10',
      remoteAddress: '198.51.100.10',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('198.51.100.10');
  });

  it('uses x-forwarded-for when the direct peer is a configured trusted proxy', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['198.51.100.0/24']
    });
    const request = createRequest({
      ip: '198.51.100.10',
      remoteAddress: '198.51.100.10',
      headers: {
        'x-forwarded-for': '203.0.113.42, 198.51.100.10'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('uses forwarded headers from loopback reverse proxies', () => {
    const config = parseGatewayConfigFromRaw({ trustedProxyHeader: 'x-real-ip' });
    const request = createRequest({
      ip: '127.0.0.1',
      remoteAddress: '127.0.0.1',
      headers: {
        'x-real-ip': '203.0.113.99'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.99');
  });

  it('parses the standard Forwarded header from trusted proxies', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/8'],
      trustedProxyHeader: 'forwarded'
    });
    const request = createRequest({
      ip: '10.0.0.8',
      remoteAddress: '10.0.0.8',
      headers: {
        forwarded: 'for="[2001:db8:cafe::17]:4711";proto=https'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('2001:db8:cafe::17');
  });

  it('matches IPv6 trusted proxy CIDRs', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['2001:db8::/32']
    });
    const request = createRequest({
      ip: '2001:db8:abcd::1',
      remoteAddress: '2001:db8:abcd::1',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('matches equivalent textual forms of exact IPv6 trusted proxy addresses', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['2001:0DB8:0000:0000:0000:0000:0000:0001']
    });
    const request = createRequest({
      ip: '2001:db8::1',
      remoteAddress: '2001:db8::1',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('recognizes expanded IPv6 loopback addresses', () => {
    const config = parseGatewayConfigFromRaw({});
    const request = createRequest({
      ip: '0:0:0:0:0:0:0:1',
      remoteAddress: '0:0:0:0:0:0:0:1',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('does not treat an empty CIDR prefix as a wildcard', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/']
    });
    const request = createRequest({
      ip: '198.51.100.10',
      remoteAddress: '198.51.100.10',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('198.51.100.10');
  });

  it('does not bypass gateway trust settings with Fastify-resolved IPs', () => {
    const config = parseGatewayConfigFromRaw({});
    const request = createRequest({
      ip: '203.0.113.42',
      remoteAddress: '198.51.100.10',
      ips: ['203.0.113.42', '198.51.100.10'],
      headers: {
        'x-forwarded-for': '203.0.113.42, 198.51.100.10'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('198.51.100.10');
  });

  it('walks appended proxy chains from right to left to reject spoofed prefixes', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/8']
    });
    const request = createRequest({
      ip: '10.0.0.8',
      remoteAddress: '10.0.0.8',
      headers: {
        'x-forwarded-for': '192.0.2.123, 203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('walks through multiple trusted proxy hops', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/8']
    });
    const request = createRequest({
      ip: '10.0.0.9',
      remoteAddress: '10.0.0.9',
      headers: {
        'x-forwarded-for': '203.0.113.42, 10.0.0.8'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('falls back to the direct peer when a forwarding chain contains an invalid hop', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/8']
    });
    const request = createRequest({
      ip: '10.0.0.8',
      remoteAddress: '10.0.0.8',
      headers: {
        'x-forwarded-for': '192.0.2.123, unknown'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('10.0.0.8');
  });

  it('ignores non-authoritative forwarding headers', () => {
    const config = parseGatewayConfigFromRaw({
      trustedProxyCidrs: ['10.0.0.0/8']
    });
    const request = createRequest({
      ip: '10.0.0.8',
      remoteAddress: '10.0.0.8',
      headers: {
        forwarded: 'for=192.0.2.123',
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('203.0.113.42');
  });

  it('supports legacy configs without trusted proxy fields', () => {
    const config = parseGatewayConfigFromRaw({});
    delete config.trustedProxyCidrs;
    delete config.trustedProxyHeader;
    const request = createRequest({
      ip: '198.51.100.10',
      remoteAddress: '198.51.100.10',
      headers: {
        'x-forwarded-for': '203.0.113.42'
      }
    });

    expect(resolveGatewayClientIp(request, config)).toBe('198.51.100.10');
  });
});

function createRequest(input: {
  ip: string;
  remoteAddress: string;
  headers?: Record<string, string>;
  ips?: string[];
}): FastifyRequest {
  return {
    headers: input.headers || {},
    ip: input.ip,
    ips: input.ips,
    socket: {
      remoteAddress: input.remoteAddress
    }
  } as unknown as FastifyRequest;
}
