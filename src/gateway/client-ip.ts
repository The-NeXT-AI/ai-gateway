import type { FastifyRequest } from 'fastify';
import type { GatewayConfig, GatewayTrustedProxyHeader } from '../types';
import { readHeader } from '../utils';

export function resolveGatewayClientIp(request: FastifyRequest, config: GatewayConfig): string {
  const fastifyIp = normalizeIp(request.ip);
  const directIp = normalizeIp(request.socket?.remoteAddress) || fastifyIp;
  if (directIp && isTrustedForwardedPeer(directIp, config)) {
    const forwardedIp = readForwardedClientIp(request, config, directIp);
    if (forwardedIp) {
      return forwardedIp;
    }
  }

  return directIp || fastifyIp || 'unknown';
}

function isTrustedForwardedPeer(peerIp: string | undefined, config: GatewayConfig): boolean {
  if (!peerIp) {
    return false;
  }

  if (isLoopbackIp(peerIp)) {
    return true;
  }

  return (config.trustedProxyCidrs ?? []).some((cidr) => matchesIpOrCidr(peerIp, cidr));
}

function matchesIpOrCidr(ip: string, value: string): boolean {
  const raw = value.trim();
  if (!raw) {
    return false;
  }

  if (raw.includes('/')) {
    return matchesCidr(ip, raw);
  }

  const normalized = normalizeIp(raw);
  return normalized !== undefined && areEquivalentIps(ip, normalized);
}

function matchesCidr(ip: string, cidr: string): boolean {
  const slash = cidr.indexOf('/');
  if (slash <= 0) {
    return false;
  }

  const base = normalizeIp(cidr.slice(0, slash));
  const prefixRaw = cidr.slice(slash + 1).trim();
  if (!base || !/^\d+$/.test(prefixRaw)) {
    return false;
  }
  const prefix = Number(prefixRaw);

  const ipV4 = ipv4ToUint32(ip);
  const baseV4 = ipv4ToUint32(base);
  if (ipV4 !== undefined && baseV4 !== undefined) {
    return matchesIpv4Cidr(ipV4, baseV4, prefix);
  }

  const ipV6 = ipv6ToBigInt(ip);
  const baseV6 = ipv6ToBigInt(base);
  if (ipV6 !== undefined && baseV6 !== undefined) {
    return matchesIpv6Cidr(ipV6, baseV6, prefix);
  }

  return false;
}

function readForwardedClientIp(
  request: FastifyRequest,
  config: GatewayConfig,
  directIp: string
): string | undefined {
  const header = config.trustedProxyHeader ?? 'x-forwarded-for';
  const forwardedIps = readForwardedIps(request, header);
  if (!forwardedIps || forwardedIps.length === 0) {
    return undefined;
  }

  return resolveForwardedChainClientIp(directIp, forwardedIps, config);
}

function readForwardedIps(
  request: FastifyRequest,
  header: GatewayTrustedProxyHeader
): string[] | undefined {
  if (header === 'forwarded') {
    return readForwardedHeaderIps(readHeader(request.headers.forwarded));
  }

  if (header === 'x-forwarded-for') {
    return readForwardedListIps(readHeader(request.headers['x-forwarded-for']));
  }

  const ip = normalizeIp(readHeader(request.headers[header]));
  return ip ? [ip] : [];
}

function readForwardedHeaderIps(value: string | undefined): string[] | undefined {
  if (!value) {
    return [];
  }

  const ips: string[] = [];
  for (const forwardedElement of value.split(',')) {
    let elementIp: string | undefined;
    for (const part of forwardedElement.split(';')) {
      const separator = part.indexOf('=');
      if (separator <= 0) {
        continue;
      }

      const key = part.slice(0, separator).trim().toLowerCase();
      if (key !== 'for') {
        continue;
      }

      const ip = normalizeIp(part.slice(separator + 1));
      if (!ip || elementIp) {
        return undefined;
      }
      elementIp = ip;
    }

    if (!elementIp) {
      return undefined;
    }
    ips.push(elementIp);
  }

  return ips;
}

function readForwardedListIps(value: string | undefined): string[] | undefined {
  if (!value) {
    return [];
  }

  const ips: string[] = [];
  for (const part of value.split(',')) {
    const ip = normalizeIp(part);
    if (!ip) {
      return undefined;
    }
    ips.push(ip);
  }

  return ips;
}

function resolveForwardedChainClientIp(
  directIp: string,
  forwardedIps: string[],
  config: GatewayConfig
): string {
  let candidate = directIp;
  for (let index = forwardedIps.length - 1; index >= 0; index -= 1) {
    if (!isTrustedForwardedPeer(candidate, config)) {
      return candidate;
    }
    candidate = forwardedIps[index];
  }

  return candidate;
}

function normalizeIp(value: string | undefined): string | undefined {
  const raw = value?.trim();
  if (!raw) {
    return undefined;
  }

  let normalized = raw;
  if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith("'") && normalized.endsWith("'"))
  ) {
    normalized = normalized.slice(1, -1).trim();
  }

  if (!normalized || normalized.toLowerCase() === 'unknown') {
    return undefined;
  }

  normalized = stripIpPort(normalized);
  if (normalized.toLowerCase().startsWith('::ffff:')) {
    normalized = normalized.slice('::ffff:'.length);
  }

  if (!isValidIp(normalized)) {
    return undefined;
  }

  return normalized;
}

function stripIpPort(value: string): string {
  if (value.startsWith('[')) {
    const end = value.indexOf(']');
    if (end > 0) {
      return value.slice(1, end);
    }
  }

  const colonCount = value.split(':').length - 1;
  if (colonCount === 1 && value.includes('.')) {
    return value.slice(0, value.indexOf(':'));
  }

  return value;
}

function isValidIp(value: string): boolean {
  return ipv4ToUint32(value) !== undefined || ipv6ToBigInt(value) !== undefined;
}

function areEquivalentIps(left: string, right: string): boolean {
  const leftV4 = ipv4ToUint32(left);
  const rightV4 = ipv4ToUint32(right);
  if (leftV4 !== undefined || rightV4 !== undefined) {
    return leftV4 !== undefined && rightV4 !== undefined && leftV4 === rightV4;
  }

  const leftV6 = ipv6ToBigInt(left);
  const rightV6 = ipv6ToBigInt(right);
  return leftV6 !== undefined && rightV6 !== undefined && leftV6 === rightV6;
}

function ipv4ToUint32(value: string): number | undefined {
  const parts = value.split('.');
  if (parts.length !== 4) {
    return undefined;
  }

  let parsed = 0;
  for (const part of parts) {
    if (!/^\d+$/.test(part)) {
      return undefined;
    }

    const value = Number(part);
    if (!Number.isInteger(value) || value < 0 || value > 255) {
      return undefined;
    }

    parsed = (parsed << 8) + value;
  }

  return parsed >>> 0;
}

function matchesIpv4Cidr(ip: number, base: number, prefix: number): boolean {
  if (prefix < 0 || prefix > 32) {
    return false;
  }

  if (prefix === 0) {
    return true;
  }

  const mask = (0xffffffff << (32 - prefix)) >>> 0;
  return (ip & mask) === (base & mask);
}

function ipv6ToBigInt(value: string): bigint | undefined {
  const normalized = value.trim().toLowerCase();
  if (!normalized.includes(':') || normalized.includes(':::')) {
    return undefined;
  }

  const expandedIpv4 = expandEmbeddedIpv4(normalized);
  if (!expandedIpv4) {
    return undefined;
  }

  const doubleColonParts = expandedIpv4.split('::');
  if (doubleColonParts.length > 2) {
    return undefined;
  }

  const left = parseIpv6Hextets(doubleColonParts[0]);
  const right = doubleColonParts.length === 2 ? parseIpv6Hextets(doubleColonParts[1]) : [];
  if (!left || !right) {
    return undefined;
  }

  const missing = 8 - left.length - right.length;
  if (doubleColonParts.length === 1 && missing !== 0) {
    return undefined;
  }

  if (doubleColonParts.length === 2 && missing < 0) {
    return undefined;
  }

  const hextets = [...left, ...new Array<number>(missing).fill(0), ...right];
  if (hextets.length !== 8) {
    return undefined;
  }

  let parsed = 0n;
  for (const hextet of hextets) {
    parsed = (parsed << 16n) + BigInt(hextet);
  }

  return parsed;
}

function expandEmbeddedIpv4(value: string): string | undefined {
  const lastColon = value.lastIndexOf(':');
  const tail = lastColon >= 0 ? value.slice(lastColon + 1) : value;
  if (!tail.includes('.')) {
    return value;
  }

  const ipv4 = ipv4ToUint32(tail);
  if (ipv4 === undefined) {
    return undefined;
  }

  const high = ((ipv4 >>> 16) & 0xffff).toString(16);
  const low = (ipv4 & 0xffff).toString(16);
  return `${value.slice(0, lastColon + 1)}${high}:${low}`;
}

function parseIpv6Hextets(value: string): number[] | undefined {
  if (!value) {
    return [];
  }

  const parts = value.split(':');
  const hextets: number[] = [];
  for (const part of parts) {
    if (!/^[0-9a-f]{1,4}$/i.test(part)) {
      return undefined;
    }

    hextets.push(Number.parseInt(part, 16));
  }

  return hextets;
}

function matchesIpv6Cidr(ip: bigint, base: bigint, prefix: number): boolean {
  if (prefix < 0 || prefix > 128) {
    return false;
  }

  if (prefix === 0) {
    return true;
  }

  const shift = 128n - BigInt(prefix);
  const mask = ((1n << 128n) - 1n) ^ ((1n << shift) - 1n);
  return (ip & mask) === (base & mask);
}

function isLoopbackIp(value: string): boolean {
  const ipv4 = ipv4ToUint32(value);
  if (ipv4 !== undefined) {
    return (ipv4 >>> 24) === 127;
  }

  return ipv6ToBigInt(value) === 1n;
}
