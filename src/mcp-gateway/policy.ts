import type { AgentToolDefinition } from '../agent/types';
import { matchesAnyPattern } from '../shared/pattern';
import type { McpGatewayConfig, McpGatewayPrincipalConfig } from '../types';
import { isObject } from '../utils';

export { isInternalIp } from '../shared/ip';
export { matchesAnyPattern, matchesPattern, toLowerSet } from '../shared/pattern';

export interface McpGatewayAccessContext {
  principal: McpGatewayPrincipalConfig;
  isInternalCaller: boolean;
}

export function filterToolsByPolicy(
  tools: AgentToolDefinition[],
  context: McpGatewayAccessContext,
  config: McpGatewayConfig
): AgentToolDefinition[] {
  return tools.filter((tool) => isToolAllowed(tool.name, context, config));
}

export function isToolAllowed(
  canonicalToolName: string,
  context: McpGatewayAccessContext,
  config: McpGatewayConfig
): boolean {
  const serverName = getServerNameFromCanonicalTool(canonicalToolName);
  if (!serverName) {
    return false;
  }

  if (!context.isInternalCaller && !isServerPublic(serverName, config)) {
    return false;
  }

  const allowServers = context.principal.allowServers;
  if (allowServers.length > 0 && !matchesAnyPattern(serverName, allowServers)) {
    return false;
  }

  const allowTools = context.principal.allowTools;
  if (allowTools.length > 0 && !matchesAnyPattern(canonicalToolName, allowTools)) {
    return false;
  }

  if (matchesAnyPattern(canonicalToolName, context.principal.denyTools)) {
    return false;
  }

  return true;
}

export function getServerNameFromCanonicalTool(toolName: string): string | undefined {
  const separator = toolName.indexOf('.');
  if (separator <= 0) {
    return undefined;
  }

  const serverName = toolName.slice(0, separator).trim();
  return serverName || undefined;
}

export function isServerPublic(serverName: string, config: McpGatewayConfig): boolean {
  return config.serverExposure[serverName] === 'public';
}

export function containsBlockedArgumentKeys(value: unknown, blockedKeys: Set<string>): boolean {
  if (blockedKeys.size === 0) {
    return false;
  }

  if (Array.isArray(value)) {
    return value.some((item) => containsBlockedArgumentKeys(item, blockedKeys));
  }

  if (!isObject(value)) {
    return false;
  }

  for (const [key, item] of Object.entries(value)) {
    if (blockedKeys.has(key.toLowerCase())) {
      return true;
    }

    if (containsBlockedArgumentKeys(item, blockedKeys)) {
      return true;
    }
  }

  return false;
}

export function redactSensitiveArguments(value: unknown, redactKeys: Set<string>): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => redactSensitiveArguments(item, redactKeys));
  }

  if (!isObject(value)) {
    return value;
  }

  const redacted: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    if (redactKeys.has(key.toLowerCase())) {
      redacted[key] = '[REDACTED]';
      continue;
    }

    redacted[key] = redactSensitiveArguments(item, redactKeys);
  }

  return redacted;
}
