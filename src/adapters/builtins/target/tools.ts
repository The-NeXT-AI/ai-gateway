import { createHash } from 'node:crypto';
import type {
  StandardRequestInputContent,
  StandardResponse,
  StandardResponseFunctionCall
} from '../../../types';
import { asBoolean, asString, isObject } from '../../../utils';

const maxTargetToolNameLength = 64;
export const anthropicWebSearchToolType = 'web_search_20250305';

export interface FlattenedStandardTool {
  name: string;
  targetName: string;
  unqualifiedName: string;
  namespace?: string;
  description?: string;
  parameters: Record<string, unknown>;
  strict?: boolean;
  deferLoading?: boolean;
}

type StandardToolSearchOutput = Extract<
  StandardRequestInputContent,
  { type: 'tool_search_output' }
>;

export function serializeStandardToolSearchOutput(item: StandardToolSearchOutput): string {
  return JSON.stringify({
    type: 'tool_search_output',
    execution: item.execution,
    ...(item.status ? { status: item.status } : {}),
    tools: item.tools
  });
}

export function appendToolReferencesToResultContent(
  content: string,
  toolReferences: string[] | undefined,
  tools?: unknown[]
): string {
  if (!toolReferences || toolReferences.length === 0) {
    return content;
  }

  const mappedNames = [
    ...new Set(toolReferences.map((name) => mapStandardToolNameToTargetName(name, tools)))
  ];
  const serializedReferences = JSON.stringify(
    mappedNames.map((toolName) => ({
      type: 'tool_reference',
      tool_name: toolName
    }))
  );
  return content ? `${content}\n${serializedReferences}` : serializedReferences;
}

export function flattenStandardTools(tools: unknown[] | undefined): FlattenedStandardTool[] {
  if (!tools || tools.length === 0) {
    return [];
  }

  const flattened: Omit<FlattenedStandardTool, 'targetName'>[] = [];
  for (const tool of tools) {
    flattened.push(...flattenStandardTool(tool));
  }

  return addTargetToolNames(flattened);
}

export function mapToolChoiceFunctionName(
  toolChoice: unknown,
  tools?: unknown[]
): string | undefined {
  const rawName = readToolChoiceFunctionName(toolChoice);
  return rawName ? mapStandardToolNameToTargetName(rawName, tools) : undefined;
}

export function readToolChoiceFunctionName(toolChoice: unknown): string | undefined {
  if (!isObject(toolChoice)) {
    return undefined;
  }

  const functionPayload = isObject(toolChoice.function) ? toolChoice.function : undefined;
  return normalizeNamespacedToolName(
    asString(toolChoice.name) || asString(functionPayload?.name),
    asString(toolChoice.namespace) || asString(functionPayload?.namespace)
  );
}

export function mapStandardToolNameToTargetName(
  name: string,
  tools: unknown[] | undefined
): string {
  const flattened = flattenStandardTools(tools);
  const match = flattened.find((tool) => tool.name === name);
  return match?.targetName ?? normalizeTargetToolName(name);
}

export function splitNamespacedToolCallName(
  name: string,
  tools: unknown[] | undefined
): { name: string; namespace?: string } {
  const mappedTool = flattenStandardTools(tools).find(
    (tool) => tool.targetName === name || tool.name === name
  );
  if (mappedTool) {
    return mappedTool.namespace
      ? {
          namespace: mappedTool.namespace,
          name: mappedTool.unqualifiedName
        }
      : { name: mappedTool.name };
  }

  const namespace = findMatchingNamespace(name, collectNamespaceToolNames(tools));
  if (!namespace) {
    return { name };
  }

  return {
    namespace,
    name: name.slice(namespace.length + 1)
  };
}

export function addNamespaceFieldsToStandardResponse(
  response: StandardResponse,
  tools: unknown[] | undefined
): StandardResponse {
  const namespaces = collectNamespaceToolNames(tools);
  if (namespaces.length === 0) {
    return response;
  }

  let changed = false;
  const output = response.output.map((item) => {
    if (item.type !== 'function_call') {
      return item;
    }

    const splitName = splitNamespacedToolCallName(item.name, tools);
    if (splitName.name === item.name && splitName.namespace === item.namespace) {
      return item;
    }

    changed = true;
    const { namespace: _namespace, ...itemWithoutNamespace } = item;
    return {
      ...itemWithoutNamespace,
      name: splitName.name,
      ...(splitName.namespace ? { namespace: splitName.namespace } : {})
    } satisfies StandardResponseFunctionCall;
  });

  return changed
    ? {
        ...response,
        output
      }
    : response;
}

function collectNamespaceToolNames(tools: unknown[] | undefined): string[] {
  if (!tools || tools.length === 0) {
    return [];
  }

  const namespaces = tools
    .map((tool) => (isObject(tool) && asString(tool.type) === 'namespace' ? asString(tool.name)?.trim() : undefined))
    .filter((name): name is string => Boolean(name));

  return [...new Set(namespaces)].sort((left, right) => right.length - left.length);
}

function findMatchingNamespace(name: string, namespaces: string[]): string | undefined {
  return namespaces.find((namespace) => name.startsWith(`${namespace}.`) && name.length > namespace.length + 1);
}

export function normalizeNamespacedToolName(
  name: string | undefined,
  namespace?: string
): string | undefined {
  if (!name) {
    return undefined;
  }

  const trimmedName = name.trim();
  const trimmedNamespace = namespace?.trim();
  if (!trimmedName) {
    return undefined;
  }

  if (!trimmedNamespace || trimmedName.startsWith(`${trimmedNamespace}.`)) {
    return trimmedName;
  }

  return `${trimmedNamespace}.${trimmedName}`;
}

export function isOpenAIWebSearchTool(tool: unknown): boolean {
  if (!isObject(tool)) {
    return false;
  }

  const type = asString(tool.type);
  return type === 'web_search' || type === 'web_search_preview';
}

export function isAnthropicWebSearchTool(tool: unknown): boolean {
  if (!isObject(tool)) {
    return false;
  }

  const type = asString(tool.type);
  return Boolean(type && /^web_search_\d{8}$/.test(type));
}

export function isHostedWebSearchTool(tool: unknown): boolean {
  return isOpenAIWebSearchTool(tool) || isAnthropicWebSearchTool(tool);
}

export function normalizeTargetToolName(name: string): string {
  const fallback = 'tool';
  const sanitized = name
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  const base = sanitized || fallback;
  if (base.length <= maxTargetToolNameLength) {
    return base;
  }

  const hash = createHash('sha256').update(name).digest('hex').slice(0, 8);
  return `${base.slice(0, maxTargetToolNameLength - hash.length - 1)}_${hash}`;
}

function addTargetToolNames(
  tools: Array<Omit<FlattenedStandardTool, 'targetName'>>
): FlattenedStandardTool[] {
  const used = new Set<string>();
  return tools.map((tool) => {
    const targetName = normalizeUniqueTargetToolName(tool.name, used);
    used.add(targetName);
    return {
      ...tool,
      targetName
    };
  });
}

function normalizeUniqueTargetToolName(name: string, used: Set<string>): string {
  const targetName = normalizeTargetToolName(name);
  if (!used.has(targetName)) {
    return targetName;
  }

  const hash = createHash('sha256').update(name).digest('hex').slice(0, 8);
  const suffix = `_${hash}`;
  const prefix = targetName.slice(0, maxTargetToolNameLength - suffix.length);
  let candidate = `${prefix}${suffix}`;
  let counter = 1;
  while (used.has(candidate)) {
    const counterSuffix = `_${hash}_${counter}`;
    candidate = `${targetName.slice(0, maxTargetToolNameLength - counterSuffix.length)}${counterSuffix}`;
    counter += 1;
  }

  return candidate;
}

function flattenStandardTool(tool: unknown): Array<Omit<FlattenedStandardTool, 'targetName'>> {
  if (!isObject(tool)) {
    return [];
  }

  if (isHostedWebSearchTool(tool)) {
    return [];
  }

  if (asString(tool.type) === 'namespace') {
    return flattenNamespaceTool(tool);
  }

  const mapped = mapFunctionLikeTool(tool);
  return mapped ? [mapped] : [];
}

function flattenNamespaceTool(
  namespaceTool: Record<string, unknown>
): Array<Omit<FlattenedStandardTool, 'targetName'>> {
  const namespaceName = asString(namespaceTool.name);
  const namespaceTools = Array.isArray(namespaceTool.tools) ? namespaceTool.tools : [];
  if (!namespaceName || namespaceTools.length === 0) {
    return [];
  }

  const flattened: Array<Omit<FlattenedStandardTool, 'targetName'>> = [];
  for (const tool of namespaceTools) {
    if (!isObject(tool)) {
      continue;
    }

    const mapped = mapFunctionLikeTool(tool, namespaceName);
    if (mapped) {
      flattened.push(mapped);
    }
  }

  return flattened;
}

function mapFunctionLikeTool(
  tool: Record<string, unknown>,
  namespaceName?: string
): Omit<FlattenedStandardTool, 'targetName'> | null {
  const functionPayload = isObject(tool.function) ? tool.function : undefined;
  const rawName = asString(tool.name) || asString(functionPayload?.name);
  if (!rawName) {
    return null;
  }

  const name = namespaceName ? `${namespaceName}.${rawName}` : rawName;
  return {
    name,
    unqualifiedName: rawName,
    ...(namespaceName ? { namespace: namespaceName } : {}),
    description: asString(tool.description) || asString(functionPayload?.description),
    parameters: ensureJsonSchema(tool.parameters ?? tool.input_schema ?? functionPayload?.parameters),
    strict: asBoolean(tool.strict) ?? asBoolean(functionPayload?.strict),
    deferLoading: asBoolean(tool.defer_loading) ?? asBoolean(functionPayload?.defer_loading)
  };
}

export function ensureJsonSchema(value: unknown): Record<string, unknown> {
  if (isObject(value)) {
    return value;
  }

  return {
    type: 'object',
    properties: {}
  };
}
