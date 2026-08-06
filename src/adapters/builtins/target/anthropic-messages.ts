import type { StandardRequestInputContent, StandardRequestInputMessage, TargetAdapter } from '../../../types';
import { err, ok } from '../../../types';
import { asString, collectStandardInputMessages, isObject } from '../../../utils';
import { buildAnthropicHeaders } from '../common';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  isProviderNativePayloadStructurallyValid,
  normalizeAnthropicThinkingMode
} from '../reasoning-envelope';
import { parseAnthropicToStandardResponse } from './shared';
import {
  anthropicWebSearchToolType,
  appendToolReferencesToResultContent,
  ensureJsonSchema,
  flattenStandardTools,
  isAnthropicWebSearchTool,
  isOpenAIWebSearchTool,
  mapStandardToolNameToTargetName,
  mapToolChoiceFunctionName,
  serializeStandardToolSearchOutput
} from './tools';

const defaultAnthropicMaxTokens = 1024;

type StandardToolSearchOutput = Extract<
  StandardRequestInputContent,
  { type: 'tool_search_output' }
>;

interface AnthropicToolContext {
  tools: unknown[] | undefined;
  functionTools: unknown[] | undefined;
  deferredTargetNames: Set<string>;
  toolSearchReferences: Map<StandardToolSearchOutput, string[]>;
}

interface DeferredToolDefinition {
  signature: string;
  deferLoading: boolean;
}

export const anthropicMessagesTargetAdapter: TargetAdapter = {
  provider: 'anthropic',
  providerTypes: ['anthropic_messages'],
  providerFallback: true,
  buildRequestFromStandard(input) {
    if (input.standardRequest.native_state_error) {
      return err(input.standardRequest.native_state_error);
    }
    const headersResult = buildAnthropicHeaders(input.request.headers, {
      ...input.config,
      anthropicApiKey: input.targetProviderConfig?.apikey || input.config.anthropicApiKey
    });
    if (!headersResult.ok) {
      return headersResult;
    }

    const toolsDisabled = isAnthropicToolChoiceNone(input.standardRequest.tool_choice);
    const activeNativeToolState = activeAnthropicNativeToolState(
      input.standardRequest.input
    );
    const requestedThinkingMode = normalizeAnthropicThinkingMode(
      input.standardRequest.thinking
    );
    if (
      activeNativeToolState.active &&
      (
        isAnthropicThinkingDisabled(input.standardRequest.thinking) ||
        [...activeNativeToolState.providerModes].some(
          (providerMode) => providerMode !== requestedThinkingMode
        )
      )
    ) {
      return err('incompatible_anthropic_thinking_mode');
    }
    const toolContext = buildAnthropicToolContext(
      input.standardRequest.input,
      input.standardRequest.tools,
      !toolsDisabled
    );
    const messages = standardInputToAnthropicMessages(
      input.standardRequest.input,
      toolContext
    );
    const body: Record<string, unknown> = {
      model: input.standardRequest.model,
      messages: messages.length > 0 ? messages : [{ role: 'user', content: [{ type: 'text', text: '' }] }],
      max_tokens: input.standardRequest.max_output_tokens ?? defaultAnthropicMaxTokens
    };

    if (input.standardRequest.instructions) {
      body.system = input.standardRequest.instructions;
    }

    if (input.standardRequest.temperature !== undefined) {
      body.temperature = input.standardRequest.temperature;
    }

    if (input.standardRequest.top_p !== undefined) {
      body.top_p = input.standardRequest.top_p;
    }

    if (input.standardRequest.stop !== undefined) {
      body.stop_sequences = Array.isArray(input.standardRequest.stop)
        ? input.standardRequest.stop
        : [input.standardRequest.stop];
    }

    if (input.standardRequest.stream !== undefined) {
      body.stream = input.standardRequest.stream;
    }

    if (input.standardRequest.thinking !== undefined) {
      body.thinking = input.standardRequest.thinking;
    }

    const tools = toolsDisabled ? undefined : mapStandardToolsToAnthropicTools(toolContext.tools);
    if (tools) {
      body.tools = tools;
    }

    const toolChoice = toolsDisabled
      ? undefined
      : mapStandardToolChoiceToAnthropicToolChoice(
          input.standardRequest.tool_choice,
          toolContext.functionTools
        );
    if (toolChoice !== undefined) {
      body.tool_choice = toolChoice;
    }

    return ok({
      url: `${input.targetProviderConfig?.baseurl || input.config.anthropicBaseUrl}/v1/messages`,
      headers: headersResult.value,
      body
    });
  },
  toStandardResponse(payload, input) {
    return parseAnthropicToStandardResponse(payload, {
      ...(input
        ? { providerMode: normalizeAnthropicThinkingMode(input.standardRequest.thinking) }
        : {})
    });
  }
};

function buildAnthropicToolContext(
  input: string | StandardRequestInputMessage[],
  configuredTools: unknown[] | undefined,
  nativeReferencesEnabled: boolean
): AnthropicToolContext {
  const effectiveTools = [...(configuredTools ?? [])];
  const reservedTargetNames = anthropicSpecialToolTargetNames(configuredTools);
  const definitions = new Map<string, DeferredToolDefinition | null>();
  for (const tool of flattenAnthropicFunctionTools(configuredTools)) {
    if (definitions.has(tool.name)) {
      definitions.set(tool.name, null);
      continue;
    }
    definitions.set(tool.name, {
      signature: deferredToolDefinitionSignature(tool),
      deferLoading: tool.deferLoading === true
    });
  }

  const pendingReferences = new Map<StandardToolSearchOutput, string[]>();
  if (nativeReferencesEnabled) {
    for (const message of collectStandardInputMessages(input)) {
      for (const item of message.content) {
        if (
          item.type !== 'tool_search_output' ||
          (item.status !== undefined && item.status !== 'completed')
        ) {
          continue;
        }

        const candidates = new Map<
          string,
          { tool: Record<string, unknown>; definition: DeferredToolDefinition }
        >();
        let valid = item.tools.length > 0;
        for (const rawTool of item.tools) {
          if (!isObject(rawTool) || isAnthropicSpecialTool(rawTool)) {
            valid = false;
            break;
          }

          const flattened = flattenAnthropicFunctionTools([rawTool]);
          if (flattened.length !== 1) {
            valid = false;
            break;
          }

          const flattenedTool = flattened[0];
          if (reservedTargetNames.has(flattenedTool.targetName)) {
            valid = false;
            break;
          }
          const definition = {
            signature: deferredToolDefinitionSignature(flattenedTool),
            deferLoading: true
          };
          const duplicate = candidates.get(flattenedTool.name);
          if (duplicate && duplicate.definition.signature !== definition.signature) {
            valid = false;
            break;
          }
          candidates.set(flattenedTool.name, { tool: rawTool, definition });
        }

        if (!valid) {
          continue;
        }

        for (const [name, candidate] of candidates) {
          if (!definitions.has(name)) {
            continue;
          }
          const existing = definitions.get(name);
          if (
            !existing ||
            existing.signature !== candidate.definition.signature ||
            !existing.deferLoading
          ) {
            valid = false;
            break;
          }
        }

        if (!valid) {
          continue;
        }

        for (const [name, candidate] of candidates) {
          if (!definitions.has(name)) {
            effectiveTools.push({ ...candidate.tool, defer_loading: true });
            definitions.set(name, candidate.definition);
          }
        }
        pendingReferences.set(item, [...candidates.keys()]);
      }
    }
  }

  const functionTools = effectiveTools.filter((tool) => !isAnthropicSpecialTool(tool));
  const flattenedTools = flattenStandardTools(functionTools);
  const targetNameByName = new Map<string, string>();
  const deferredTargetNames = new Set<string>();
  for (const tool of flattenedTools) {
    if (!targetNameByName.has(tool.name)) {
      targetNameByName.set(tool.name, tool.targetName);
    }
    if (tool.deferLoading === true) {
      deferredTargetNames.add(tool.targetName);
    }
  }
  for (const tool of effectiveTools) {
    if (!isObject(tool) || tool.defer_loading !== true) {
      continue;
    }
    const mappedWebSearchTool = mapStandardWebSearchToolToAnthropicTool(tool);
    const mappedName = mappedWebSearchTool && asString(mappedWebSearchTool.name);
    if (mappedName) {
      deferredTargetNames.add(mappedName);
    }
  }

  const toolSearchReferences = new Map<StandardToolSearchOutput, string[]>();
  for (const [item, names] of pendingReferences) {
    const targetNames = names.map((name) => targetNameByName.get(name));
    if (
      targetNames.every(
        (name): name is string => Boolean(name && deferredTargetNames.has(name))
      )
    ) {
      toolSearchReferences.set(item, targetNames);
    }
  }

  return {
    tools: effectiveTools.length > 0 ? effectiveTools : undefined,
    functionTools: functionTools.length > 0 ? functionTools : undefined,
    deferredTargetNames,
    toolSearchReferences
  };
}

function deferredToolDefinitionSignature(tool: {
  parameters: Record<string, unknown>;
  strict?: boolean;
}): string {
  return JSON.stringify({
    parameters: tool.parameters,
    strict: tool.strict ?? null
  });
}

function isAnthropicSpecialTool(tool: unknown): boolean {
  return (
    isStandardToolSearchDeclaration(tool) ||
    isAnthropicWebSearchTool(tool) ||
    isOpenAIWebSearchTool(tool)
  );
}

function anthropicSpecialToolTargetNames(tools: unknown[] | undefined): Set<string> {
  const names = new Set<string>();
  for (const tool of tools ?? []) {
    const mapped =
      mapStandardToolSearchToAnthropicTool(tool) ??
      mapStandardWebSearchToolToAnthropicTool(tool);
    const name = mapped && asString(mapped.name);
    if (name) {
      names.add(name);
    }
  }
  return names;
}

function flattenAnthropicFunctionTools(tools: unknown[] | undefined) {
  return flattenStandardTools((tools ?? []).filter((tool) => !isAnthropicSpecialTool(tool)));
}

function standardInputToAnthropicMessages(
  input: string | StandardRequestInputMessage[],
  toolContext: AnthropicToolContext
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  for (const message of collectStandardInputMessages(input)) {
    const content = standardContentToAnthropicBlocks(message.content, toolContext);
    if (content.length === 0) {
      continue;
    }

    messages.push({
      role: message.role === 'assistant' ? 'assistant' : 'user',
      content
    });
  }

  return messages;
}

function standardContentToAnthropicBlocks(
  content: StandardRequestInputContent[],
  toolContext: AnthropicToolContext
): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  for (const item of content) {
    if (item.type === 'provider_native_item') {
      if (
        item.source_format === ANTHROPIC_CLAUDE_REASONING_FORMAT &&
        isProviderNativePayloadStructurallyValid(item, ANTHROPIC_CLAUDE_REASONING_FORMAT)
      ) {
        blocks.push({ ...item.raw_payload });
      }
      continue;
    }
    const nativeItem = 'native_item' in item ? item.native_item : undefined;
    if (
      nativeItem?.source_format === ANTHROPIC_CLAUDE_REASONING_FORMAT &&
      isProviderNativePayloadStructurallyValid(nativeItem, ANTHROPIC_CLAUDE_REASONING_FORMAT)
    ) {
      blocks.push({ ...nativeItem.raw_payload });
      continue;
    }
    if (item.type === 'input_text') {
      const text = item.text.trim();
      if (!text) {
        continue;
      }

      blocks.push({
        type: 'text',
        text
      });
      continue;
    }

    if (item.type === 'tool_use') {
      blocks.push({
        type: 'tool_use',
        id: item.id,
        name: mapStandardToolNameToTargetName(item.name, toolContext.functionTools),
        input: isObject(item.input) ? item.input : {}
      });
      continue;
    }

    if (item.type === 'tool_search_call') {
      blocks.push({
        type: 'tool_use',
        id: item.call_id,
        name: 'ToolSearch',
        input: normalizeAnthropicToolInput(item.arguments)
      });
      continue;
    }

    if (item.type === 'tool_search_output') {
      const referenceNames = toolContext.toolSearchReferences.get(item) ?? [];
      const references = referenceNames.map((toolName) => ({
        type: 'tool_reference',
        tool_name: toolName
      }));
      blocks.push({
        type: 'tool_result',
        tool_use_id: item.call_id,
        content: references.length > 0 ? references : serializeStandardToolSearchOutput(item)
      });
      continue;
    }

    if (item.type === 'reasoning') {
      blocks.push(...standardReasoningToAnthropicBlocks(item));
      continue;
    }

    if (item.type !== 'tool_result') {
      continue;
    }

    const referenceNames = [
      ...new Set(
        (item.tool_references ?? []).map((name) =>
          mapStandardToolNameToTargetName(name, toolContext.functionTools)
        )
      )
    ];
    const canUseNativeReferences =
      referenceNames.length > 0 &&
      referenceNames.every((name) => toolContext.deferredTargetNames.has(name));
    const referenceBlocks = canUseNativeReferences
      ? referenceNames.map((toolName) => ({
          type: 'tool_reference',
          tool_name: toolName
        }))
      : [];
    const toolResultBlock: Record<string, unknown> = {
      type: 'tool_result',
      tool_use_id: item.tool_use_id,
      content:
        referenceBlocks.length > 0
          ? [
              ...(item.content ? [{ type: 'text', text: item.content }] : []),
              ...referenceBlocks
            ]
          : appendToolReferencesToResultContent(
              item.content,
              item.tool_references,
              toolContext.functionTools
            )
    };
    if (item.is_error !== undefined) {
      toolResultBlock.is_error = item.is_error;
    }
    blocks.push(toolResultBlock);
  }

  return blocks;
}

function normalizeAnthropicToolInput(value: unknown): Record<string, unknown> {
  if (isObject(value)) {
    return value;
  }

  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (isObject(parsed)) {
        return parsed;
      }
    } catch {
      return {};
    }
  }

  return {};
}

function standardReasoningToAnthropicBlocks(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }>
): Array<Record<string, unknown>> {
  if (standardReasoningFormat(item) !== ANTHROPIC_CLAUDE_REASONING_FORMAT) {
    return [];
  }

  const blocks = anthropicBlocksFromReasoningDetails(
    item.reasoning_details,
    ANTHROPIC_CLAUDE_REASONING_FORMAT
  );
  if (blocks.length > 0) {
    return blocks;
  }

  const thinking = [item.text, item.summary].filter(Boolean).join('\n').trim();
  if (thinking) {
    blocks.push({
      type: 'thinking',
      thinking
    });
  }

  if (
    item.encrypted_content &&
    item.source_format === ANTHROPIC_CLAUDE_REASONING_FORMAT
  ) {
    blocks.push({
      type: 'redacted_thinking',
      data: item.encrypted_content
    });
  }

  return blocks;
}

function anthropicBlocksFromReasoningDetails(
  value: unknown[] | undefined,
  expectedFormat: string
): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) {
    return [];
  }

  const blocks: Array<Record<string, unknown>> = [];
  for (const detail of value) {
    if (!isObject(detail)) {
      continue;
    }

    const format = asString(detail.format);
    if (format !== expectedFormat) {
      continue;
    }
    const type = asString(detail.type);
    const thinking =
      asString(detail.thinking) ||
      asString(detail.text) ||
      asString(detail.reasoning) ||
      asString(detail.summary);
    const data = asString(detail.data) || asString(detail.encrypted_content);
    const signature = asString(detail.signature);

    if (type === 'reasoning.encrypted' || type === 'redacted_thinking' || (!thinking && data)) {
      if (data) {
        blocks.push({
          type: 'redacted_thinking',
          data
        });
      }
      continue;
    }

    if (!thinking && !signature) {
      continue;
    }

    const block: Record<string, unknown> = {
      type: 'thinking',
      thinking: thinking || ''
    };
    if (signature) {
      block.signature = signature;
    }
    blocks.push(block);
  }

  return blocks;
}

function standardReasoningFormat(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }>
): string | undefined {
  if (item.source_format) {
    return item.source_format;
  }

  for (const detail of item.reasoning_details || []) {
    if (!isObject(detail)) {
      continue;
    }
    const format = asString(detail.format);
    if (format) {
      return format;
    }
  }

  return undefined;
}

function mapStandardToolsToAnthropicTools(tools: unknown[] | undefined): Record<string, unknown>[] | undefined {
  const mapped: Record<string, unknown>[] = [];
  const flattenedTools = flattenAnthropicFunctionTools(tools);
  let flattenedOffset = 0;
  for (const tool of tools || []) {
    const toolSearchTool = mapStandardToolSearchToAnthropicTool(tool);
    if (toolSearchTool) {
      mapped.push(toolSearchTool);
      continue;
    }

    const webSearchTool = mapStandardWebSearchToolToAnthropicTool(tool);
    if (webSearchTool) {
      mapped.push(webSearchTool);
      continue;
    }

    const flattenedCount = flattenAnthropicFunctionTools([tool]).length;
    for (const flattenedTool of flattenedTools.slice(
      flattenedOffset,
      flattenedOffset + flattenedCount
    )) {
      const mappedTool: Record<string, unknown> = {
        name: flattenedTool.targetName,
        input_schema: flattenedTool.parameters
      };
      if (flattenedTool.description) {
        mappedTool.description = flattenedTool.description;
      }
      if (flattenedTool.deferLoading === true) {
        mappedTool.defer_loading = true;
      }

      mapped.push(mappedTool);
    }
    flattenedOffset += flattenedCount;
  }
  return mapped.length > 0 ? mapped : undefined;
}

function isStandardToolSearchDeclaration(tool: unknown): tool is Record<string, unknown> {
  return (
    isObject(tool) &&
    asString(tool.type) === 'tool_search' &&
    (tool.execution === undefined || tool.execution === 'client')
  );
}

function mapStandardToolSearchToAnthropicTool(tool: unknown): Record<string, unknown> | null {
  if (!isStandardToolSearchDeclaration(tool)) {
    return null;
  }

  const mapped: Record<string, unknown> = {
    name: 'ToolSearch',
    input_schema: ensureJsonSchema(tool.parameters ?? tool.input_schema)
  };
  const description = asString(tool.description);
  if (description) {
    mapped.description = description;
  }
  return mapped;
}

function mapStandardWebSearchToolToAnthropicTool(tool: unknown): Record<string, unknown> | null {
  if (!isObject(tool)) {
    return null;
  }

  if (isAnthropicWebSearchTool(tool)) {
    const mapped: Record<string, unknown> = {
      type: asString(tool.type) || anthropicWebSearchToolType,
      name: asString(tool.name) || 'web_search'
    };
    copyDefinedToolFields(tool, mapped, [
      'max_uses',
      'allowed_domains',
      'blocked_domains',
      'user_location',
      'cache_control',
      'defer_loading'
    ]);
    return mapped;
  }

  if (!isOpenAIWebSearchTool(tool)) {
    return null;
  }

  const mapped: Record<string, unknown> = {
    type: anthropicWebSearchToolType,
    name: 'web_search'
  };
  const filters = isObject(tool.filters) ? tool.filters : undefined;
  const allowedDomains = readStringArray(filters?.allowed_domains);
  const blockedDomains = readStringArray(filters?.blocked_domains);
  if (allowedDomains) {
    mapped.allowed_domains = allowedDomains;
  }
  if (blockedDomains) {
    mapped.blocked_domains = blockedDomains;
  }
  if (tool.user_location !== undefined) {
    mapped.user_location = tool.user_location;
  }
  if (tool.defer_loading !== undefined) {
    mapped.defer_loading = tool.defer_loading;
  }

  return mapped;
}

function copyDefinedToolFields(
  source: Record<string, unknown>,
  target: Record<string, unknown>,
  keys: string[]
): void {
  for (const key of keys) {
    if (source[key] !== undefined) {
      target[key] = source[key];
    }
  }
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const values = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);
  return values.length > 0 ? values : undefined;
}

function mapStandardToolChoiceToAnthropicToolChoice(
  toolChoice: unknown,
  tools?: unknown[]
): unknown {
  if (toolChoice === undefined) {
    return undefined;
  }

  if (typeof toolChoice === 'string') {
    if (toolChoice === 'auto') {
      return { type: 'auto' };
    }

    if (toolChoice === 'required') {
      return { type: 'any' };
    }

    return undefined;
  }

  if (!isObject(toolChoice)) {
    return undefined;
  }

  const type = asString(toolChoice.type);
  if (type === 'auto') {
    return { type: 'auto' };
  }

  if (type === 'any' || type === 'required') {
    return { type: 'any' };
  }

  const name = mapToolChoiceFunctionName(toolChoice, tools);
  if (name) {
    return {
      type: 'tool',
      name
    };
  }

  return undefined;
}

function isAnthropicToolChoiceNone(toolChoice: unknown): boolean {
  if (toolChoice === 'none') {
    return true;
  }

  if (!isObject(toolChoice)) {
    return false;
  }

  return asString(toolChoice.type) === 'none';
}

function isAnthropicThinkingDisabled(value: unknown): boolean {
  if (!isObject(value)) {
    return false;
  }
  const type = asString(value.type)?.toLowerCase();
  return type === 'disabled' || type === 'off' || type === 'none';
}

function activeAnthropicNativeToolState(
  input: string | StandardRequestInputMessage[]
): { active: boolean; providerModes: Set<string> } {
  if (typeof input === 'string') {
    return { active: false, providerModes: new Set() };
  }
  const results = new Set<string>();
  for (const message of input) {
    for (const item of message.content) {
      if (item.type === 'tool_result') {
        results.add(item.tool_use_id);
      }
    }
  }
  const providerModes = new Set<string>();
  let active = false;
  for (const message of input) {
    if (message.role !== 'assistant') {
      continue;
    }
    for (const item of message.content) {
      if (
        item.type !== 'tool_use' ||
        item.native_item?.source_format !== ANTHROPIC_CLAUDE_REASONING_FORMAT ||
        results.has(item.id)
      ) {
        continue;
      }
      active = true;
      if (item.native_item.provider_mode) {
        providerModes.add(item.native_item.provider_mode);
      }
    }
  }
  return { active, providerModes };
}
