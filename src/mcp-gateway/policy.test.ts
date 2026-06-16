import { describe, expect, it } from 'vitest';
import type { AgentToolDefinition } from '../agent/types';
import type { McpGatewayConfig, McpGatewayPrincipalConfig } from '../types';
import {
  containsBlockedArgumentKeys,
  filterToolsByPolicy,
  isInternalIp,
  matchesPattern,
  redactSensitiveArguments,
  toLowerSet
} from './policy';

describe('mcp gateway policy', () => {
  const tools: AgentToolDefinition[] = [
    { name: 'filesystem.read_file', description: 'read file' },
    { name: 'filesystem.write_file', description: 'write file' },
    { name: 'slack.send_message', description: 'send message' }
  ];

  const principal: McpGatewayPrincipalConfig = {
    key: 'k-1',
    team: 'eng',
    organization: 'next-ai',
    allowServers: ['filesystem'],
    allowTools: ['filesystem.*'],
    denyTools: ['filesystem.write_*']
  };

  const config: McpGatewayConfig = {
    enabled: true,
    endpoint: '/mcp',
    websocket: {
      enabled: false,
      endpoint: '/mcp/ws',
      auth: {
        allowQueryToken: true,
        queryTokenParam: 'token'
      }
    },
    principals: [principal],
    serverExposure: {
      filesystem: 'internal',
      slack: 'public'
    },
    internalCidrs: ['10.0.0.0/8'],
    guardrails: {
      enabled: true,
      maxArgumentBytes: 1024,
      blockedTools: [],
      blockedArgumentKeys: [],
      redactArgumentKeys: []
    },
    oauth: {
      enabled: false,
      scopesSupported: ['mcp:tools:list', 'mcp:tools:call']
    }
  };

  it('filters tools by principal allow/deny rules', () => {
    const filtered = filterToolsByPolicy(
      tools,
      {
        principal,
        isInternalCaller: true
      },
      config
    );

    expect(filtered.map((tool) => tool.name)).toEqual(['filesystem.read_file']);
  });

  it('blocks internal-only servers for external callers', () => {
    const externalPrincipal: McpGatewayPrincipalConfig = {
      ...principal,
      allowServers: ['*'],
      allowTools: ['*'],
      denyTools: []
    };

    const filtered = filterToolsByPolicy(
      tools,
      {
        principal: externalPrincipal,
        isInternalCaller: false
      },
      config
    );

    expect(filtered.map((tool) => tool.name)).toEqual(['slack.send_message']);
  });

  it('supports wildcard pattern matching', () => {
    expect(matchesPattern('filesystem.read_file', 'filesystem.*')).toBe(true);
    expect(matchesPattern('filesystem.read_file', '*.read_file')).toBe(true);
    expect(matchesPattern('filesystem.read_file', 'filesystem.write_*')).toBe(false);
  });

  it('detects internal ip ranges', () => {
    expect(isInternalIp('10.1.2.3', [])).toBe(true);
    expect(isInternalIp('192.168.10.99', [])).toBe(true);
    expect(isInternalIp('8.8.8.8', ['8.8.8.0/24'])).toBe(true);
    expect(isInternalIp('1.1.1.1', [])).toBe(false);
  });

  it('blocks and redacts sensitive argument keys recursively', () => {
    const args = {
      query: 'hello',
      token: 'secret-token',
      nested: {
        password: 'p@ss'
      }
    };

    const blocked = toLowerSet(['password']);
    expect(containsBlockedArgumentKeys(args, blocked)).toBe(true);

    const redacted = redactSensitiveArguments(args, toLowerSet(['token', 'password'])) as Record<string, unknown>;
    expect(redacted.token).toBe('[REDACTED]');
    expect((redacted.nested as Record<string, unknown>).password).toBe('[REDACTED]');
  });
});
