import { randomUUID } from 'node:crypto';
import type {
  ProviderConfig,
  StandardRequest,
  StandardRequestInputContent,
  StandardRequestInputMessage,
  TargetAdapter
} from '../../../types';
import { ok } from '../../../types';
import { asString, collectStandardInputMessages, isObject } from '../../../utils';
import { buildOpenAIHeaders } from '../common';
import { applyOpenAIChatStreamUsageOption, parseOpenAIToStandardResponse } from './shared';
import {
  isAnthropicWebSearchTool,
  isOpenAIWebSearchTool,
  ensureJsonSchema,
  flattenStandardTools,
  mapStandardToolNameToTargetName,
  mapToolChoiceFunctionName,
  readToolChoiceFunctionName,
  splitNamespacedToolCallName
} from './tools';

export const openAIResponsesTargetAdapter: TargetAdapter = {
  provider: 'openai',
  buildRequestFromStandard(input) {
    const headersResult = buildOpenAIHeaders(input.request.headers, {
      ...input.config,
      openaiApiKey: input.targetProviderConfig?.apikey || input.config.openaiApiKey
    });
    if (!headersResult.ok) {
      return headersResult;
    }

    const protocol = resolveOpenAITargetProtocol(input.targetProviderConfig);
    if (protocol === 'openai_chat_completions') {
      const messages = standardInputToOpenAIChatMessages(
        input.standardRequest.input,
        input.standardRequest.instructions,
        input.standardRequest.tools
      );
      const body: Record<string, unknown> = {
        model: input.standardRequest.model,
        messages: messages.length > 0 ? messages : [{ role: 'user', content: '' }],
        temperature: input.standardRequest.temperature,
        top_p: input.standardRequest.top_p,
        max_tokens: input.standardRequest.max_output_tokens,
        stop: input.standardRequest.stop
      };

      const tools = mapStandardToolsToOpenAIChatTools(
        input.standardRequest.tools,
        input.targetProviderConfig?.openaiChatToolsFormat
      );
      if (tools) {
        body.tools = tools;
      }

      const toolChoice = mapStandardToolChoiceToOpenAIChatToolChoice(
        input.standardRequest.tool_choice,
        input.standardRequest.tools
      );
      if (toolChoice !== undefined) {
        body.tool_choice = toolChoice;
      }

      if (input.standardRequest.stream === true) {
        body.stream = true;
      }
      body.reasoning_split = true;
      applyOpenAIChatReasoningOptions(body, input.standardRequest);

      return ok({
        url: `${input.config.openaiBaseUrl}/chat/completions`,
        headers: headersResult.value,
        body: applyOpenAIChatStreamUsageOption(body, input.targetProviderConfig)
      });
    }

    const body = buildOpenAIResponsesBodyFromStandardRequest(input.standardRequest);

    return ok({
      url: `${input.config.openaiBaseUrl}/responses`,
      headers: headersResult.value,
      body
    });
  },
  toStandardResponse(payload) {
    return parseOpenAIToStandardResponse(payload);
  }
};

export function buildOpenAIResponsesBodyFromStandardRequest(
  standardRequest: StandardRequest
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: standardRequest.model,
    instructions: standardRequest.instructions,
    input: standardInputToOpenAIResponsesInput(standardRequest.input, standardRequest.tools),
    temperature: standardRequest.temperature,
    top_p: standardRequest.top_p,
    max_output_tokens: standardRequest.max_output_tokens,
    stop: standardRequest.stop
  };

  const tools = mapStandardToolsToOpenAIResponsesTools(standardRequest.tools);
  if (tools) {
    body.tools = tools;
  }

  const toolChoice = mapStandardToolChoiceToOpenAIResponsesToolChoice(
    standardRequest.tool_choice,
    standardRequest.tools
  );
  if (toolChoice !== undefined) {
    body.tool_choice = toolChoice;
  }

  if (standardRequest.stream === true) {
    body.stream = true;
  }

  applyOpenAIResponsesReasoningOptions(body, standardRequest);

  return body;
}

function resolveOpenAITargetProtocol(providerConfig: ProviderConfig | undefined): 'openai_responses' | 'openai_chat_completions' {
  if (providerConfig?.type === 'openai_chat_completions') {
    return 'openai_chat_completions';
  }

  return 'openai_responses';
}

function applyOpenAIChatReasoningOptions(body: Record<string, unknown>, standardRequest: StandardRequest): void {
  const thinking = standardRequest.thinking ?? thinkingFromResponsesReasoning(standardRequest.reasoning);
  if (thinking !== undefined) {
    body.thinking = thinking;
  }

  const outputConfig =
    standardRequest.output_config ?? outputConfigFromResponsesReasoning(standardRequest.reasoning);
  if (outputConfig !== undefined) {
    body.output_config = outputConfig;
  }
}

function applyOpenAIResponsesReasoningOptions(body: Record<string, unknown>, standardRequest: StandardRequest): void {
  if (standardRequest.reasoning !== undefined) {
    body.reasoning = standardRequest.reasoning;
  }
  if (standardRequest.thinking !== undefined) {
    body.thinking = standardRequest.thinking;
  }
  if (standardRequest.output_config !== undefined) {
    body.output_config = standardRequest.output_config;
  }
}

function thinkingFromResponsesReasoning(reasoning: unknown): Record<string, string> | undefined {
  return readResponsesReasoningEffort(reasoning) ? { type: 'enabled' } : undefined;
}

function outputConfigFromResponsesReasoning(reasoning: unknown): Record<string, string> | undefined {
  const effort = readResponsesReasoningEffort(reasoning);
  return effort ? { effort } : undefined;
}

function readResponsesReasoningEffort(reasoning: unknown): string | undefined {
  return isObject(reasoning) ? asString(reasoning.effort) : undefined;
}

function standardInputToOpenAIChatMessages(
  input: string | StandardRequestInputMessage[],
  instructions?: string,
  tools?: unknown[]
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  if (instructions) {
    messages.push({
      role: 'system',
      content: instructions
    });
  }

  for (const message of collectStandardInputMessages(input)) {
    const text = extractStandardInputTextContent(message.content);
    if (message.role === 'assistant') {
      const toolCalls = collectAssistantToolCalls(message.content, tools);
      const reasoning = collectAssistantReasoning(message.content);
      if (!text && toolCalls.length === 0 && !reasoning) {
        continue;
      }

      const assistantMessage: Record<string, unknown> = {
        role: 'assistant'
      };
      if (text) {
        assistantMessage.content = text;
      }
      if (toolCalls.length > 0) {
        assistantMessage.tool_calls = toolCalls;
        if (!text) {
          assistantMessage.content = '';
        }
      }
      if (reasoning) {
        if (reasoning.text) {
          assistantMessage.reasoning_content = reasoning.text;
        }
        if (reasoning.reasoning_details && reasoning.reasoning_details.length > 0) {
          assistantMessage.reasoning_details = reasoning.reasoning_details;
        }
        if (assistantMessage.content === undefined) {
          assistantMessage.content = '';
        }
      }

      messages.push(assistantMessage);
      continue;
    }

    const toolResults = collectUserToolResults(message.content);
    for (const toolResult of toolResults) {
      messages.push({
        role: 'tool',
        tool_call_id: toolResult.tool_call_id,
        content:
          toolResult.result_format === 'web_search'
            ? formatWebSearchResultText(toolResult.content)
            : toolResult.content
      });
    }
    if (text) {
      messages.push({
        role: 'user',
        content: text
      });
    }
  }

  return messages;
}

function standardInputToOpenAIResponsesInput(
  input: string | StandardRequestInputMessage[],
  tools?: unknown[]
): string | Array<Record<string, unknown>> {
  if (typeof input === 'string') {
    return input;
  }

  const items: Array<Record<string, unknown>> = [];
  for (const message of collectStandardInputMessages(input)) {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    const text = extractStandardInputTextContent(message.content);
    if (text) {
      items.push({
        type: 'message',
        role,
        content: [
          {
            type: 'input_text',
            text
          }
        ]
      });
    }

    if (message.role === 'assistant') {
      const reasoning = collectAssistantReasoning(message.content);
      if (reasoning) {
        items.push({
          type: 'reasoning',
          id: `rs_${randomUUID().replace(/-/g, '')}`,
          status: 'completed',
          summary: reasoning.summary
            ? [
                {
                  type: 'summary_text',
                  text: reasoning.summary
                }
              ]
            : [],
          ...(reasoning.text
            ? {
                content: [
                  {
                    type: 'reasoning_text',
                    text: reasoning.text
                  }
                ]
              }
            : {}),
          ...(reasoning.encrypted_content ? { encrypted_content: reasoning.encrypted_content } : {})
        });
      }

      for (const item of message.content) {
        if (item.type !== 'tool_use') {
          continue;
        }

        const splitName = splitNamespacedToolCallName(item.name, tools);
        items.push({
          type: 'function_call',
          call_id: item.id,
          name: splitName.name,
          ...(splitName.namespace ? { namespace: splitName.namespace } : {}),
          arguments: normalizeFunctionArguments(item.input)
        });
      }
      continue;
    }

    const toolResults = collectUserToolResults(message.content);
    for (const toolResult of toolResults) {
      if (toolResult.result_format === 'web_search') {
        items.push({
          type: 'web_search_call',
          id: toolResult.tool_call_id,
          status: 'completed',
          action: normalizeWebSearchAction(toolResult.content)
        });
        items.push({
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: formatWebSearchResultText(toolResult.content)
            }
          ]
        });
        continue;
      }

      items.push({
        type: 'function_call_output',
        call_id: toolResult.tool_call_id,
        output: toolResult.content
      });
    }
  }

  return items;
}

function extractStandardInputTextContent(content: StandardRequestInputContent[]): string {
  return content
    .map((item) => (item.type === 'input_text' ? item.text : ''))
    .filter(Boolean)
    .join('\n')
    .trim();
}

function collectAssistantToolCalls(
  content: StandardRequestInputContent[],
  tools?: unknown[]
): Array<Record<string, unknown>> {
  const toolCalls: Array<Record<string, unknown>> = [];
  for (const item of content) {
    if (item.type !== 'tool_use') {
      continue;
    }

    toolCalls.push({
      id: item.id,
      type: 'function',
      function: {
        name: mapStandardToolNameToTargetName(item.name, tools),
        arguments: normalizeFunctionArguments(item.input)
      }
    });
  }

  return toolCalls;
}

function collectAssistantReasoning(content: StandardRequestInputContent[]):
  | {
      text?: string;
      summary?: string;
      encrypted_content?: string;
      reasoning_details?: unknown[];
    }
  | undefined {
  const reasoningItems = content.filter((item) => item.type === 'reasoning');
  if (reasoningItems.length === 0) {
    return undefined;
  }

  const text = reasoningItems
    .map((item) => item.text)
    .filter(Boolean)
    .join('\n')
    .trim();
  const summary = reasoningItems
    .map((item) => item.summary)
    .filter(Boolean)
    .join('\n')
    .trim();
  const encryptedContent = reasoningItems.find((item) => item.encrypted_content)?.encrypted_content;
  const reasoningDetails = reasoningItems.flatMap((item) => item.reasoning_details || []);

  return {
    ...(text ? { text } : {}),
    ...(summary ? { summary } : {}),
    ...(encryptedContent ? { encrypted_content: encryptedContent } : {}),
    ...(reasoningDetails.length > 0 ? { reasoning_details: reasoningDetails } : {})
  };
}

function collectUserToolResults(
  content: StandardRequestInputContent[]
): Array<{ tool_call_id: string; content: string; result_format?: 'function' | 'web_search' }> {
  const toolResults: Array<{ tool_call_id: string; content: string; result_format?: 'function' | 'web_search' }> = [];
  for (const item of content) {
    if (item.type !== 'tool_result') {
      continue;
    }

    toolResults.push({
      tool_call_id: item.tool_use_id,
      content: item.content,
      result_format: item.result_format
    });
  }

  return toolResults;
}

function normalizeFunctionArguments(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

function formatWebSearchResultText(content: string): string {
  return `web_search result:\n${content}`;
}

function normalizeWebSearchAction(content: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(content) as unknown;
    if (isObject(parsed)) {
      const query =
        asString(parsed.query) ||
        asString(parsed.search_query) ||
        asString(parsed.q) ||
        asString(parsed.title);
      return query
        ? { type: 'search', query, queries: [query] }
        : { type: 'search', queries: [] };
    }
  } catch {
    // Keep plain-text search output visible in the adjacent message item.
  }

  return { type: 'search', queries: [] };
}

function mapStandardToolsToOpenAIChatTools(
  tools: unknown[] | undefined,
  format: ProviderConfig['openaiChatToolsFormat'] = 'openai'
): Record<string, unknown>[] | undefined {
  const mapped = flattenStandardTools(tools).map((tool) => {
    if (format === 'anthropic') {
      const mappedTool: Record<string, unknown> = {
        name: tool.targetName,
        input_schema: tool.parameters
      };
      if (tool.description) {
        mappedTool.description = tool.description;
      }
      return mappedTool;
    }

    const functionObject: Record<string, unknown> = {
      name: tool.targetName,
      parameters: tool.parameters
    };
    if (tool.description) {
      functionObject.description = tool.description;
    }
    if (tool.strict !== undefined) {
      functionObject.strict = tool.strict;
    }

    return {
      type: 'function',
      function: functionObject
    };
  });
  return mapped.length > 0 ? mapped : undefined;
}

function mapStandardToolsToOpenAIResponsesTools(tools: unknown[] | undefined): Record<string, unknown>[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  const mapped = tools
    .map((tool) => mapStandardToolToOpenAIResponsesTool(tool))
    .filter((tool): tool is Record<string, unknown> => Boolean(tool));
  return mapped.length > 0 ? mapped : undefined;
}

function mapStandardToolToOpenAIResponsesTool(tool: unknown): Record<string, unknown> | null {
  if (!isObject(tool)) {
    return null;
  }

  if (isOpenAIWebSearchTool(tool)) {
    return mapOpenAIWebSearchToolToOpenAIResponsesTool(tool);
  }

  if (isAnthropicWebSearchTool(tool)) {
    return mapAnthropicWebSearchToolToOpenAIResponsesTool(tool);
  }

  if (asString(tool.type) === 'namespace') {
    return mapStandardNamespaceToolToOpenAIResponsesTool(tool);
  }

  const functionPayload = isObject(tool.function) ? tool.function : undefined;
  const name = asString(tool.name) || asString(functionPayload?.name);
  if (!name) {
    return null;
  }

  const description = asString(tool.description) || asString(functionPayload?.description);
  const parameters = ensureJsonSchema(tool.parameters ?? tool.input_schema ?? functionPayload?.parameters);
  const mapped: Record<string, unknown> = {
    type: 'function',
    name,
    parameters
  };
  if (description) {
    mapped.description = description;
  }
  if (typeof tool.strict === 'boolean') {
    mapped.strict = tool.strict;
  } else if (typeof functionPayload?.strict === 'boolean') {
    mapped.strict = functionPayload.strict;
  }

  return mapped;
}

function mapOpenAIWebSearchToolToOpenAIResponsesTool(
  tool: Record<string, unknown>
): Record<string, unknown> | null {
  const type = asString(tool.type);
  if (type !== 'web_search' && type !== 'web_search_preview') {
    return null;
  }

  const mapped: Record<string, unknown> = { type };
  copyDefinedToolFields(tool, mapped, [
    'search_context_size',
    'user_location',
    'filters',
    'external_web_access',
    'return_token_budget',
    'search_content_types',
    'image_settings'
  ]);
  return mapped;
}

function mapAnthropicWebSearchToolToOpenAIResponsesTool(
  tool: Record<string, unknown>
): Record<string, unknown> {
  const mapped: Record<string, unknown> = {
    type: 'web_search'
  };
  const filters = mapAnthropicWebSearchFiltersToOpenAI(tool);
  if (filters) {
    mapped.filters = filters;
  }
  return mapped;
}

function mapAnthropicWebSearchFiltersToOpenAI(
  tool: Record<string, unknown>
): Record<string, unknown> | undefined {
  const filters: Record<string, unknown> = {};
  const allowedDomains = readStringArray(tool.allowed_domains);
  const blockedDomains = readStringArray(tool.blocked_domains);

  if (allowedDomains) {
    filters.allowed_domains = allowedDomains;
  }
  if (blockedDomains) {
    filters.blocked_domains = blockedDomains;
  }

  return Object.keys(filters).length > 0 ? filters : undefined;
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

function mapStandardNamespaceToolToOpenAIResponsesTool(tool: Record<string, unknown>): Record<string, unknown> | null {
  const name = asString(tool.name);
  const nestedTools = Array.isArray(tool.tools) ? tool.tools : [];
  if (!name || nestedTools.length === 0) {
    return null;
  }

  const mappedTools = nestedTools
    .map((nestedTool) => mapStandardToolToOpenAIResponsesTool(nestedTool))
    .filter((nestedTool): nestedTool is Record<string, unknown> => Boolean(nestedTool));
  if (mappedTools.length === 0) {
    return null;
  }

  const mapped: Record<string, unknown> = {
    type: 'namespace',
    name,
    tools: mappedTools
  };
  const description = asString(tool.description);
  if (description) {
    mapped.description = description;
  }

  return mapped;
}

function mapStandardToolChoiceToOpenAIChatToolChoice(
  toolChoice: unknown,
  tools?: unknown[]
): unknown {
  if (toolChoice === undefined) {
    return undefined;
  }

  if (typeof toolChoice === 'string') {
    return toolChoice;
  }

  if (!isObject(toolChoice)) {
    return undefined;
  }

  const type = asString(toolChoice.type);
  if (type === 'auto' || type === 'none') {
    return type;
  }

  if (type === 'any' || type === 'required') {
    return 'required';
  }

  const name = mapToolChoiceFunctionName(toolChoice, tools);
  if (name) {
    return {
      type: 'function',
      function: {
        name
      }
    };
  }

  return toolChoice;
}

function mapStandardToolChoiceToOpenAIResponsesToolChoice(
  toolChoice: unknown,
  tools?: unknown[]
): unknown {
  if (toolChoice === undefined) {
    return undefined;
  }

  if (typeof toolChoice === 'string') {
    return toolChoice;
  }

  if (!isObject(toolChoice)) {
    return undefined;
  }

  const type = asString(toolChoice.type);
  if (type === 'auto' || type === 'none') {
    return type;
  }

  if (type === 'any' || type === 'required') {
    return 'required';
  }

  const rawName = readToolChoiceFunctionName(toolChoice);
  const splitName = rawName ? splitNamespacedToolCallName(rawName, tools) : undefined;
  const name = splitName?.name;
  if (name) {
    return {
      type: 'function',
      name,
      ...(splitName?.namespace ? { namespace: splitName.namespace } : {})
    };
  }

  return toolChoice;
}
