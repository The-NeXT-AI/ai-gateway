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
import {
  applyOpenAIChatStreamUsageOption,
  parseOpenAIToStandardResponse,
  rewriteOpenAIChatCompatibleRequest
} from './shared';
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

const openAIResponsesReasoningEffortOrder = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max'
] as const;

type OpenAIResponsesReasoningEffort = (typeof openAIResponsesReasoningEffortOrder)[number];

interface OpenAIModelReasoningCapability {
  pattern: RegExp;
  supportedEfforts: readonly OpenAIResponsesReasoningEffort[];
}

// Keep this table aligned with the effort values documented on OpenAI's model pages.
// Unknown models intentionally pass valid effort values through instead of inheriting
// capabilities from a different model family.
const knownOpenAIModelReasoningCapabilities = [
  {
    pattern: /^gpt-5\.6(?:-(?:sol|terra|luna))?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max']
  },
  {
    pattern: /^gpt-5\.5-pro(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.5(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['none', 'low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.4-pro(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.4(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['none', 'low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.3-codex(?:-spark)?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.2-codex(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.2(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['none', 'low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.1-codex-max(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high', 'xhigh']
  },
  {
    pattern: /^gpt-5\.1-codex(?:-mini)?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high']
  },
  {
    pattern: /^gpt-5\.1(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['none', 'low', 'medium', 'high']
  },
  {
    pattern: /^gpt-5-pro(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['high']
  },
  {
    pattern: /^gpt-5-codex(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high']
  },
  {
    pattern: /^gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['minimal', 'low', 'medium', 'high']
  },
  {
    pattern: /^o3(?:-mini)?(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high']
  },
  {
    pattern: /^o4-mini(?:-\d{4}-\d{2}-\d{2})?$/,
    supportedEfforts: ['low', 'medium', 'high']
  }
] satisfies readonly OpenAIModelReasoningCapability[];

export const openAIResponsesTargetAdapter: TargetAdapter = {
  provider: 'openai',
  providerTypes: ['openai_responses', 'openai_chat_completions'],
  providerFallback: true,
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
        input.standardRequest.tools,
        true
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
      const rewrittenBody = rewriteOpenAIChatCompatibleRequest(
        body,
        input.targetProviderConfig,
        {
          generatedReasoningMessageFields: true,
          standardRequest: input.standardRequest
        }
      );

      return ok({
        url: `${input.config.openaiBaseUrl}/chat/completions`,
        headers: headersResult.value,
        body: applyOpenAIChatStreamUsageOption(rewrittenBody, input.targetProviderConfig)
      });
    }

    const body = buildOpenAIResponsesBodyFromStandardRequest(
      input.standardRequest,
      input.targetProviderConfig
    );

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
  standardRequest: StandardRequest,
  targetProviderConfig?: ProviderConfig
): Record<string, unknown> {
  // The Responses API does not accept `stop`; Chat Completions handles it above.
  const deferredToolSearch = buildDeferredToolSearchPlan(
    standardRequest.input,
    standardRequest.tools,
    standardRequest.tool_choice
  );
  const body: Record<string, unknown> = {
    model: standardRequest.model,
    instructions: standardRequest.instructions,
    input: standardInputToOpenAIResponsesInput(
      standardRequest.input,
      standardRequest.tools,
      deferredToolSearch
    ),
    temperature: standardRequest.temperature,
    top_p: standardRequest.top_p,
    max_output_tokens: standardRequest.max_output_tokens,
    text: standardRequest.text
  };

  const tools = mapStandardToolsToOpenAIResponsesTools(
    standardRequest.tools,
    deferredToolSearch
  );
  if (tools) {
    body.tools = tools;
  }

  const toolChoice = mapStandardToolChoiceToOpenAIResponsesToolChoice(
    standardRequest.tool_choice,
    standardRequest.tools,
    deferredToolSearch
  );
  if (toolChoice !== undefined) {
    body.tool_choice = toolChoice;
  }

  if (standardRequest.stream === true) {
    body.stream = true;
  }

  applyOpenAIResponsesReasoningOptions(body, standardRequest, targetProviderConfig);
  applyOpenAIResponsesTextOptions(body, standardRequest);

  return body;
}

function resolveOpenAITargetProtocol(providerConfig: ProviderConfig | undefined): 'openai_responses' | 'openai_chat_completions' {
  if (providerConfig?.type === 'openai_chat_completions') {
    return 'openai_chat_completions';
  }

  return 'openai_responses';
}

function applyOpenAIResponsesReasoningOptions(
  body: Record<string, unknown>,
  standardRequest: StandardRequest,
  targetProviderConfig?: ProviderConfig
): void {
  if (standardRequest.reasoning !== undefined) {
    if (!isObject(standardRequest.reasoning) || Array.isArray(standardRequest.reasoning)) {
      body.reasoning = standardRequest.reasoning;
      return;
    }

    const effort = asString(standardRequest.reasoning.effort)?.trim();
    const translatedEffort = openAIResponsesReasoningEffort(
      standardRequest.output_config,
      standardRequest.model,
      targetProviderConfig
    );
    body.reasoning =
      effort || !translatedEffort
        ? standardRequest.reasoning
        : {
            ...standardRequest.reasoning,
            effort: translatedEffort
          };
    return;
  }

  const effort = openAIResponsesReasoningEffort(
    standardRequest.output_config,
    standardRequest.model,
    targetProviderConfig
  );
  if (effort) {
    body.reasoning = { effort };
  }
}

function openAIResponsesReasoningEffort(
  outputConfig: unknown,
  model: string | undefined,
  targetProviderConfig: ProviderConfig | undefined
): OpenAIResponsesReasoningEffort | undefined {
  const requestedEffort = readOpenAIResponsesReasoningEffort(outputConfig);
  if (!requestedEffort) {
    return undefined;
  }

  const supportedEfforts = providerModelReasoningEfforts(targetProviderConfig, model);
  if (!supportedEfforts) {
    return requestedEffort;
  }

  if (supportedEfforts.has(requestedEffort)) {
    return requestedEffort;
  }

  const requestedIndex = openAIResponsesReasoningEffortOrder.indexOf(requestedEffort);
  for (let index = requestedIndex - 1; index >= 0; index -= 1) {
    const effort = openAIResponsesReasoningEffortOrder[index];
    if (supportedEfforts.has(effort)) {
      return effort;
    }
  }

  for (let index = requestedIndex + 1; index < openAIResponsesReasoningEffortOrder.length; index += 1) {
    const effort = openAIResponsesReasoningEffortOrder[index];
    if (supportedEfforts.has(effort)) {
      return effort;
    }
  }

  return undefined;
}

function providerModelReasoningEfforts(
  targetProviderConfig: ProviderConfig | undefined,
  model: string | undefined
): ReadonlySet<OpenAIResponsesReasoningEffort> | undefined {
  const normalizedModel = model?.trim().toLowerCase();
  if (!normalizedModel) {
    return undefined;
  }

  if (targetProviderConfig?.modelMetadata) {
    const matchingMetadata = Object.entries(targetProviderConfig.modelMetadata).find(
      ([configuredModel]) => configuredModel.trim().toLowerCase() === normalizedModel
    )?.[1];
    if (matchingMetadata?.supportedReasoningLevels !== undefined) {
      const supportedEfforts = new Set<OpenAIResponsesReasoningEffort>();
      for (const level of matchingMetadata.supportedReasoningLevels) {
        const effort = readOpenAIResponsesReasoningEffort(level);
        if (effort) {
          supportedEfforts.add(effort);
        }
      }
      return supportedEfforts;
    }
  }

  const knownCapability = knownOpenAIModelReasoningCapabilities.find(({ pattern }) =>
    pattern.test(normalizedModel)
  );
  return knownCapability ? new Set(knownCapability.supportedEfforts) : undefined;
}

function readOpenAIResponsesReasoningEffort(
  value: unknown
): OpenAIResponsesReasoningEffort | undefined {
  if (!isObject(value) || Array.isArray(value)) {
    return undefined;
  }

  const effort = asString(value.effort)?.trim().toLowerCase();
  return openAIResponsesReasoningEffortOrder.find((candidate) => candidate === effort);
}

function applyOpenAIResponsesTextOptions(
  body: Record<string, unknown>,
  standardRequest: StandardRequest
): void {
  if (!isObject(standardRequest.output_config) || Array.isArray(standardRequest.output_config)) {
    return;
  }

  const verbosity = asString(standardRequest.output_config.verbosity);
  if (!verbosity) {
    return;
  }

  body.text = {
    verbosity,
    ...(isObject(body.text) && !Array.isArray(body.text) ? body.text : {})
  };
}

function standardInputToOpenAIChatMessages(
  input: string | StandardRequestInputMessage[],
  instructions?: string,
  tools?: unknown[],
  includeReasoningFields = false
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
      const reasoning = includeReasoningFields ? collectAssistantReasoning(message.content) : undefined;
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
  tools?: unknown[],
  deferredToolSearch?: DeferredToolSearchPlan
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
            type: role === 'assistant' ? 'output_text' : 'input_text',
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
        if (item.type === 'tool_search_call') {
          items.push({
            type: 'tool_search_call',
            execution: item.execution,
            call_id: item.call_id,
            ...(item.status ? { status: item.status } : {}),
            arguments: item.arguments
          });
          continue;
        }
        if (item.type !== 'tool_use') {
          continue;
        }

        const discovery =
          item.name === 'ToolSearch'
            ? deferredToolSearch?.discoveries.get(item.id)
            : undefined;
        if (discovery) {
          items.push({
            type: 'tool_search_call',
            execution: 'client',
            call_id: item.id,
            status: 'completed',
            arguments: normalizeToolSearchArguments(item.input)
          });
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

    for (const item of message.content) {
      if (item.type !== 'tool_search_output') {
        continue;
      }
      items.push({
        type: 'tool_search_output',
        execution: item.execution,
        call_id: item.call_id,
        ...(item.status ? { status: item.status } : {}),
        tools: item.tools
      });
    }

    const toolResults = collectUserToolResults(message.content);
    for (const toolResult of toolResults) {
      const discovery = deferredToolSearch?.discoveries.get(toolResult.tool_call_id);
      if (discovery && deferredToolSearch) {
        items.push({
          type: 'tool_search_output',
          execution: 'client',
          call_id: toolResult.tool_call_id,
          status: 'completed',
          tools: discovery.toolNames
            .map((name) => deferredToolSearch.deferredToolsByName.get(name))
            .map((tool) => mapStandardToolToOpenAIResponsesTool(tool, true))
            .filter((tool): tool is Record<string, unknown> => Boolean(tool))
        });
        if (toolResult.content) {
          items.push({
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: toolResult.content }]
          });
        }
        continue;
      }

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
): Array<{
  tool_call_id: string;
  content: string;
  is_error?: boolean;
  result_format?: 'function' | 'web_search';
  tool_references?: string[];
}> {
  const toolResults: Array<{
    tool_call_id: string;
    content: string;
    is_error?: boolean;
    result_format?: 'function' | 'web_search';
    tool_references?: string[];
  }> = [];
  for (const item of content) {
    if (item.type !== 'tool_result') {
      continue;
    }

    toolResults.push({
      tool_call_id: item.tool_use_id,
      content: item.content,
      is_error: item.is_error,
      result_format: item.result_format,
      tool_references: item.tool_references
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

function normalizeToolSearchArguments(value: unknown): Record<string, unknown> {
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

function mapStandardToolsToOpenAIResponsesTools(
  tools: unknown[] | undefined,
  deferredToolSearch?: DeferredToolSearchPlan
): Record<string, unknown>[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  const toolSearchPlan = deferredToolSearch;
  const mapped = tools
    .map((tool) => {
      if (toolSearchPlan && toolSearchPlan.searchTool === tool) {
        return mapStandardToolSearchToOpenAIResponsesTool(toolSearchPlan.searchTool);
      }
      if (toolSearchPlan && isObject(tool) && toolSearchPlan.deferredTools.has(tool)) {
        return null;
      }
      return mapStandardToolToOpenAIResponsesTool(tool);
    })
    .filter((tool): tool is Record<string, unknown> => Boolean(tool));
  return mapped.length > 0 ? mapped : undefined;
}

function mapStandardToolToOpenAIResponsesTool(
  tool: unknown,
  preserveDeferredLoading = false
): Record<string, unknown> | null {
  if (!isObject(tool)) {
    return null;
  }

  if (asString(tool.type) === 'tool_search') {
    return { ...tool };
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
  if (preserveDeferredLoading && tool.defer_loading === true) {
    mapped.defer_loading = true;
  }

  return mapped;
}

function mapStandardToolSearchToOpenAIResponsesTool(
  tool: Record<string, unknown>
): Record<string, unknown> {
  const functionPayload = isObject(tool.function) ? tool.function : undefined;
  const mapped: Record<string, unknown> = {
    type: 'tool_search',
    execution: 'client',
    parameters: ensureJsonSchema(
      tool.parameters ?? tool.input_schema ?? functionPayload?.parameters
    )
  };
  const description = asString(tool.description) || asString(functionPayload?.description);
  if (description) {
    mapped.description = description;
  }
  return mapped;
}

interface DeferredToolSearchPlan {
  searchTool: Record<string, unknown>;
  deferredTools: Set<Record<string, unknown>>;
  deferredToolsByName: Map<string, Record<string, unknown>>;
  discoveries: Map<string, { toolNames: string[] }>;
}

function buildDeferredToolSearchPlan(
  input: string | StandardRequestInputMessage[],
  tools: unknown[] | undefined,
  toolChoice: unknown
): DeferredToolSearchPlan | undefined {
  if (typeof input === 'string' || !tools) {
    return undefined;
  }

  let searchTool: Record<string, unknown> | undefined;
  const deferredTools = new Set<Record<string, unknown>>();
  const deferredToolsByName = new Map<string, Record<string, unknown>>();
  const discoverableToolNames = new Set<string>();
  const seenToolNames = new Set<string>();
  for (const tool of tools) {
    if (!isObject(tool)) {
      continue;
    }

    const functionPayload = isObject(tool.function) ? tool.function : undefined;
    const name = asString(tool.name) || asString(functionPayload?.name);
    const mappedTool = mapStandardToolToOpenAIResponsesTool(tool);
    const isFunctionTool = asString(mappedTool?.type) === 'function';
    if (name) {
      if (seenToolNames.has(name)) {
        return undefined;
      }
      seenToolNames.add(name);
    }
    if (name === 'ToolSearch') {
      if (searchTool || !isFunctionTool) {
        return undefined;
      }
      searchTool = tool;
      continue;
    }
    if (name && tool.defer_loading === true) {
      discoverableToolNames.add(name);
      if (isFunctionTool) {
        deferredTools.add(tool);
        deferredToolsByName.set(name, tool);
      }
    }
  }

  if (!searchTool || discoverableToolNames.size === 0) {
    return undefined;
  }

  const forcedToolName = readToolChoiceFunctionName(toolChoice);
  if (
    forcedToolName &&
    forcedToolName !== 'ToolSearch' &&
    discoverableToolNames.has(forcedToolName)
  ) {
    return undefined;
  }

  const callCounts = new Map<string, number>();
  const searchCallCounts = new Map<string, number>();
  const results = new Map<string, StandardRequestInputContent & { type: 'tool_result' }>();
  const resultCounts = new Map<string, number>();
  const callPositions = new Map<string, number>();
  const resultPositions = new Map<string, number>();
  let position = 0;
  for (const message of collectStandardInputMessages(input)) {
    for (const item of message.content) {
      position += 1;
      if (item.type === 'tool_use') {
        callCounts.set(item.id, (callCounts.get(item.id) ?? 0) + 1);
        callPositions.set(item.id, position);
        if (item.name === 'ToolSearch') {
          searchCallCounts.set(item.id, (searchCallCounts.get(item.id) ?? 0) + 1);
        }
      } else if (item.type === 'tool_result') {
        results.set(item.tool_use_id, item);
        resultCounts.set(
          item.tool_use_id,
          (resultCounts.get(item.tool_use_id) ?? 0) + 1
        );
        resultPositions.set(item.tool_use_id, position);
      }
    }
  }

  const discoveries = new Map<string, { toolNames: string[] }>();
  for (const [callId, searchCallCount] of searchCallCounts) {
    const result = results.get(callId);
    if (
      searchCallCount !== 1 ||
      callCounts.get(callId) !== 1 ||
      resultCounts.get(callId) !== 1 ||
      !result ||
      (callPositions.get(callId) ?? Number.POSITIVE_INFINITY) >=
        (resultPositions.get(callId) ?? Number.NEGATIVE_INFINITY) ||
      result.is_error === true
    ) {
      return undefined;
    }

    const toolNames = result.tool_references ?? [];
    if (toolNames.some((name) => !discoverableToolNames.has(name))) {
      return undefined;
    }
    discoveries.set(callId, { toolNames });
  }

  return {
    searchTool,
    deferredTools,
    deferredToolsByName,
    discoveries
  };
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
  tools?: unknown[],
  deferredToolSearch?: DeferredToolSearchPlan
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
  if (deferredToolSearch && rawName === 'ToolSearch') {
    return { type: 'tool_search' };
  }
  const hostedToolChoice = rawName
    ? mapForcedHostedToolChoiceToOpenAIResponses(rawName, tools)
    : undefined;
  if (hostedToolChoice) {
    return hostedToolChoice;
  }
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

function mapForcedHostedToolChoiceToOpenAIResponses(
  name: string,
  tools: unknown[] | undefined
): Record<string, unknown> | undefined {
  if (!tools) {
    return undefined;
  }

  for (const tool of tools) {
    if (!isObject(tool)) {
      continue;
    }

    const functionPayload = isObject(tool.function) ? tool.function : undefined;
    const toolName = asString(tool.name) || asString(functionPayload?.name);
    if (
      toolName !== name ||
      (!isOpenAIWebSearchTool(tool) && !isAnthropicWebSearchTool(tool))
    ) {
      continue;
    }

    const mappedTool = mapStandardToolToOpenAIResponsesTool(tool);
    const type = asString(mappedTool?.type);
    if (!type) {
      return undefined;
    }

    return {
      type: 'allowed_tools',
      mode: 'required',
      tools: [{ type }]
    };
  }

  return undefined;
}
