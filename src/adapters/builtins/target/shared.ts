import { randomUUID } from 'node:crypto';
import type {
  Result,
  StandardResponse,
  StandardResponseFunctionCall,
  StandardResponseReasoning,
  StandardResponseOutputItem,
  StandardUsage
} from '../../../types';
import { err, ok } from '../../../types';
import { asNumber, asString, extractTextFromPart, isObject } from '../../../utils';

export function parseOpenAIToStandardResponse(payload: unknown): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid OpenAI response payload.');
  }

  const outputText =
    asString(payload.output_text) ||
    extractOpenAIResponseOutputText(payload.output) ||
    extractOpenAIChatText(payload.choices);
  const toolCalls = extractOpenAIFunctionCalls(payload);
  const reasoningItems = extractOpenAIReasoningItems(payload);

  if (!outputText && toolCalls.length === 0 && reasoningItems.length === 0) {
    return err('OpenAI response does not contain text output, reasoning output, or tool calls.');
  }

  const usageRaw = isObject(payload.usage) ? payload.usage : undefined;
  const inputDetails = isObject(usageRaw?.input_tokens_details)
    ? usageRaw.input_tokens_details
    : isObject(usageRaw?.prompt_tokens_details)
      ? usageRaw.prompt_tokens_details
      : undefined;
  const usage: StandardUsage = {
    input_tokens: asNumber(usageRaw?.input_tokens) ?? asNumber(usageRaw?.prompt_tokens),
    output_tokens: asNumber(usageRaw?.output_tokens) ?? asNumber(usageRaw?.completion_tokens),
    total_tokens: asNumber(usageRaw?.total_tokens),
    cache_read_tokens: asNumber(inputDetails?.cached_tokens) ?? asNumber(usageRaw?.cache_read_tokens),
    cache_write_tokens:
      asNumber(inputDetails?.cache_creation_tokens) ??
      asNumber(usageRaw?.cache_creation_tokens) ??
      asNumber(usageRaw?.cache_write_tokens),
    cache_duration_seconds: extractCacheDurationSeconds(usageRaw, inputDetails)
  };

  return ok(createStandardResponse({
    id: asString(payload.id) || `resp_${randomUUID()}`,
    model: asString(payload.model) || 'unknown',
    outputText,
    outputItems: buildStandardResponseOutputItems(outputText, toolCalls, reasoningItems),
    usage,
    finishReason: extractOpenAIFinishReason(payload.choices)
  }));
}

export function parseAnthropicToStandardResponse(payload: unknown): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid Anthropic response payload.');
  }

  const text = extractAnthropicText(payload.content);
  const toolCalls = extractAnthropicFunctionCalls(payload.content);
  if (!text && toolCalls.length === 0) {
    return err('Anthropic response does not contain text output or tool calls.');
  }

  const usageRaw = isObject(payload.usage) ? payload.usage : undefined;
  const inputTokens = asNumber(usageRaw?.input_tokens);
  const outputTokens = asNumber(usageRaw?.output_tokens);
  const cacheReadTokens = asNumber(usageRaw?.cache_read_input_tokens) ?? asNumber(usageRaw?.cache_read_tokens);
  const cacheWriteTokens =
    asNumber(usageRaw?.cache_creation_input_tokens) ?? asNumber(usageRaw?.cache_write_tokens);
  const serverToolUse = isObject(usageRaw?.server_tool_use) ? usageRaw.server_tool_use : undefined;
  const usage: StandardUsage = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: sumOptional(inputTokens, outputTokens),
    cache_read_tokens: cacheReadTokens,
    cache_write_tokens: cacheWriteTokens,
    cache_duration_seconds: extractCacheDurationSeconds(usageRaw),
    server_tool_use: normalizeServerToolUse(serverToolUse)
  };

  return ok(createStandardResponse({
    id: asString(payload.id) || `msg_${randomUUID()}`,
    model: asString(payload.model) || 'unknown',
    outputText: text,
    outputItems: buildStandardResponseOutputItems(text, toolCalls),
    usage,
    finishReason: asString(payload.stop_reason)
  }));
}

export function parseGeminiToStandardResponse(payload: unknown): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid Gemini response payload.');
  }

  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const first = isObject(candidates[0]) ? candidates[0] : undefined;
  const content = isObject(first?.content) ? first.content : undefined;
  const parts = Array.isArray(content?.parts) ? content.parts : [];
  const text = parts.map(extractTextFromPart).filter(Boolean).join('\n').trim();

  if (!text) {
    return err('Gemini response does not contain text output.');
  }

  const usageRaw = isObject(payload.usageMetadata) ? payload.usageMetadata : undefined;
  const usage: StandardUsage = {
    input_tokens: asNumber(usageRaw?.promptTokenCount),
    output_tokens: asNumber(usageRaw?.candidatesTokenCount),
    total_tokens: asNumber(usageRaw?.totalTokenCount),
    cache_read_tokens: asNumber(usageRaw?.cachedContentTokenCount),
    cache_duration_seconds: extractCacheDurationSeconds(usageRaw)
  };

  return ok(createStandardResponse({
    id: `gem_${randomUUID()}`,
    model: asString(payload.modelVersion) || 'unknown',
    outputText: text,
    usage,
    finishReason: asString(first?.finishReason)
  }));
}

function createStandardResponse(args: {
  id: string;
  model: string;
  outputText: string;
  outputItems?: StandardResponseOutputItem[];
  usage: StandardUsage;
  finishReason?: string;
}): StandardResponse {
  const output = args.outputItems && args.outputItems.length > 0
    ? args.outputItems
    : buildStandardResponseOutputItems(args.outputText);

  return {
    id: args.id,
    object: 'response',
    status: 'completed',
    model: args.model,
    output_text: args.outputText,
    output,
    usage: args.usage,
    finish_reason: args.finishReason
  };
}

function normalizeServerToolUse(value: Record<string, unknown> | undefined): StandardUsage['server_tool_use'] {
  if (!value) {
    return undefined;
  }

  const serverToolUse = {
    web_search_requests: asNumber(value.web_search_requests),
    web_fetch_requests: asNumber(value.web_fetch_requests)
  };

  return Object.values(serverToolUse).some((count) => count !== undefined)
    ? serverToolUse
    : undefined;
}

function buildStandardResponseOutputItems(
  outputText: string,
  toolCalls: StandardResponseFunctionCall[] = [],
  reasoningItems: StandardResponseReasoning[] = []
): StandardResponseOutputItem[] {
  const output: StandardResponseOutputItem[] = [...reasoningItems];

  if (outputText) {
    output.push({
      id: `msg_${randomUUID()}`,
      type: 'message',
      role: 'assistant',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: outputText,
          annotations: []
        }
      ]
    });
  }

  output.push(...toolCalls);
  return output;
}

function extractOpenAIResponseOutputText(output: unknown): string {
  if (!Array.isArray(output)) {
    return '';
  }

  const chunks: string[] = [];
  for (const item of output) {
    if (!isObject(item)) {
      continue;
    }

    if (item.type === 'message' && Array.isArray(item.content)) {
      for (const content of item.content) {
        const text = extractTextFromPart(content);
        if (text) {
          chunks.push(text);
        }
      }

      continue;
    }

    const text = extractTextFromPart(item);
    if (text) {
      chunks.push(text);
    }
  }

  return chunks.join('\n').trim();
}

function extractOpenAIChatText(choices: unknown): string {
  if (!Array.isArray(choices) || choices.length === 0) {
    return '';
  }

  const first = choices[0];
  if (!isObject(first)) {
    return '';
  }

  const message = isObject(first.message) ? first.message : undefined;
  const content = message?.content;

  if (typeof content === 'string') {
    const text = content.trim();
    return text || '';
  }

  if (!Array.isArray(content)) {
    return '';
  }

  return content.map(extractTextFromPart).filter(Boolean).join('\n').trim();
}

function extractOpenAIReasoningItems(payload: Record<string, unknown>): StandardResponseReasoning[] {
  return [
    ...extractOpenAIResponsesReasoningItems(payload.output),
    ...extractOpenAIChatReasoningItems(payload.choices)
  ];
}

function extractOpenAIResponsesReasoningItems(output: unknown): StandardResponseReasoning[] {
  if (!Array.isArray(output)) {
    return [];
  }

  const items: StandardResponseReasoning[] = [];
  for (const item of output) {
    const reasoningItem = normalizeOpenAIResponsesReasoningItem(item);
    if (reasoningItem) {
      items.push(reasoningItem);
    }
  }

  return items;
}

function normalizeOpenAIResponsesReasoningItem(item: unknown): StandardResponseReasoning | null {
  if (!isObject(item) || asString(item.type) !== 'reasoning') {
    return null;
  }

  const summary = normalizeReasoningSummaryParts(item.summary);
  const content = normalizeReasoningContentParts(item.content);
  const encryptedContent = asString(item.encrypted_content);

  const reasoning: StandardResponseReasoning = {
    id: asString(item.id) || `rs_${randomUUID().replace(/-/g, '')}`,
    type: 'reasoning',
    status: 'completed',
    summary
  };

  if (content.length > 0) {
    reasoning.content = content;
  }
  if (encryptedContent) {
    reasoning.encrypted_content = encryptedContent;
  }

  return reasoning;
}

function extractOpenAIChatReasoningItems(choices: unknown): StandardResponseReasoning[] {
  if (!Array.isArray(choices) || choices.length === 0) {
    return [];
  }

  const first = choices[0];
  if (!isObject(first)) {
    return [];
  }

  const message = isObject(first.message) ? first.message : undefined;
  if (!message) {
    return [];
  }

  const details = normalizeChatReasoningDetails(message.reasoning_details);
  const reasoningText =
    asString(message.reasoning_content) ||
    asString(message.reasoning) ||
    asString(message.thinking);

  if (reasoningText) {
    appendReasoningContentIfDistinct(details.content, reasoningText);
  }

  if (
    details.content.length === 0 &&
    details.summary.length === 0 &&
    !details.encryptedContent &&
    details.rawDetails.length === 0
  ) {
    return [];
  }

  const reasoning: StandardResponseReasoning = {
    id: details.id || `rs_${randomUUID().replace(/-/g, '')}`,
    type: 'reasoning',
    status: 'completed',
    summary: details.summary
  };

  if (details.content.length > 0) {
    reasoning.content = details.content;
  }
  if (details.encryptedContent) {
    reasoning.encrypted_content = details.encryptedContent;
  }
  if (details.rawDetails.length > 0) {
    reasoning.reasoning_details = details.rawDetails;
  }

  return [reasoning];
}

function normalizeReasoningSummaryParts(value: unknown): StandardResponseReasoning['summary'] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parts: StandardResponseReasoning['summary'] = [];
  for (const item of value) {
    if (typeof item === 'string') {
      const text = item.trim();
      if (text) {
        parts.push({ type: 'summary_text', text });
      }
      continue;
    }

    if (!isObject(item)) {
      continue;
    }

    const text = asString(item.text) || asString(item.summary);
    if (text) {
      parts.push({ type: 'summary_text', text });
    }
  }

  return parts;
}

function normalizeReasoningContentParts(value: unknown): NonNullable<StandardResponseReasoning['content']> {
  if (!Array.isArray(value)) {
    return [];
  }

  const parts: NonNullable<StandardResponseReasoning['content']> = [];
  for (const item of value) {
    if (typeof item === 'string') {
      const text = item.trim();
      if (text) {
        parts.push({ type: 'reasoning_text', text });
      }
      continue;
    }

    if (!isObject(item)) {
      continue;
    }

    const text = asString(item.text) || asString(item.reasoning) || asString(item.thinking);
    if (text) {
      parts.push({ type: 'reasoning_text', text });
    }
  }

  return parts;
}

function normalizeChatReasoningDetails(value: unknown): {
  id?: string;
  content: NonNullable<StandardResponseReasoning['content']>;
  summary: StandardResponseReasoning['summary'];
  encryptedContent?: string;
  rawDetails: unknown[];
} {
  const normalized: {
    id?: string;
    content: NonNullable<StandardResponseReasoning['content']>;
    summary: StandardResponseReasoning['summary'];
    encryptedContent?: string;
    rawDetails: unknown[];
  } = {
    content: [],
    summary: [],
    rawDetails: []
  };

  if (!Array.isArray(value)) {
    return normalized;
  }

  for (const detail of value) {
    if (typeof detail === 'string') {
      const text = detail.trim();
      if (text) {
        normalized.content.push({ type: 'reasoning_text', text });
        normalized.rawDetails.push(detail);
      }
      continue;
    }

    if (!isObject(detail)) {
      continue;
    }

    normalized.rawDetails.push(detail);
    const id = asString(detail.id);
    if (id && !normalized.id) {
      normalized.id = id;
    }

    const type = asString(detail.type);
    const summary = asString(detail.summary);
    const text = asString(detail.text) || asString(detail.reasoning) || asString(detail.thinking);
    const encryptedContent = asString(detail.encrypted_content) || asString(detail.data);

    if (type === 'reasoning.summary' || (summary && !text)) {
      const summaryText = summary || text;
      if (!summaryText) {
        continue;
      }
      normalized.summary.push({
        type: 'summary_text',
        text: summaryText
      });
      continue;
    }

    if (text) {
      normalized.content.push({
        type: 'reasoning_text',
        text
      });
    }

    if (encryptedContent && !normalized.encryptedContent) {
      normalized.encryptedContent = encryptedContent;
    }
  }

  return normalized;
}

function appendReasoningContentIfDistinct(
  parts: NonNullable<StandardResponseReasoning['content']>,
  value: string
): void {
  const text = value.trim();
  if (!text) {
    return;
  }

  const existingText = parts
    .map((part) => part.text)
    .filter(Boolean)
    .join('\n')
    .trim();
  if (existingText === text || parts.some((part) => part.text.trim() === text)) {
    return;
  }

  parts.push({
    type: 'reasoning_text',
    text
  });
}

function extractOpenAIFunctionCalls(payload: Record<string, unknown>): StandardResponseFunctionCall[] {
  return [...extractOpenAIResponsesFunctionCalls(payload.output), ...extractOpenAIChatFunctionCalls(payload.choices)];
}

function extractOpenAIResponsesFunctionCalls(output: unknown): StandardResponseFunctionCall[] {
  if (!Array.isArray(output)) {
    return [];
  }

  const toolCalls: StandardResponseFunctionCall[] = [];
  for (const item of output) {
    if (!isObject(item)) {
      continue;
    }

    const type = asString(item.type);
    if (type !== 'function_call' && type !== 'tool_call') {
      continue;
    }

    const functionPayload = isObject(item.function) ? item.function : undefined;
    const name = asString(item.name) || asString(functionPayload?.name);
    if (!name) {
      continue;
    }

    const id = asString(item.id) || `fc_${randomUUID().replace(/-/g, '')}`;
    toolCalls.push({
      id,
      type: 'function_call',
      call_id: asString(item.call_id) || id,
      name,
      arguments: normalizeFunctionCallArguments(item.arguments ?? functionPayload?.arguments ?? item.input),
      status: 'completed'
    });
  }

  return toolCalls;
}

function extractOpenAIChatFunctionCalls(choices: unknown): StandardResponseFunctionCall[] {
  if (!Array.isArray(choices) || choices.length === 0) {
    return [];
  }

  const first = choices[0];
  if (!isObject(first)) {
    return [];
  }

  const message = isObject(first.message) ? first.message : undefined;
  const toolCallsRaw = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
  const toolCalls: StandardResponseFunctionCall[] = [];

  for (const toolCall of toolCallsRaw) {
    if (!isObject(toolCall)) {
      continue;
    }

    const functionPayload = isObject(toolCall.function) ? toolCall.function : undefined;
    const name = asString(functionPayload?.name) || asString(toolCall.name);
    if (!name) {
      continue;
    }

    const id = asString(toolCall.id) || `chat_call_${randomUUID().replace(/-/g, '')}`;
    toolCalls.push({
      id,
      type: 'function_call',
      call_id: id,
      name,
      arguments: normalizeFunctionCallArguments(functionPayload?.arguments ?? toolCall.arguments ?? toolCall.input),
      status: 'completed'
    });
  }

  return toolCalls;
}

function extractOpenAIFinishReason(choices: unknown): string | undefined {
  if (!Array.isArray(choices) || choices.length === 0) {
    return undefined;
  }

  const first = choices[0];
  if (!isObject(first)) {
    return undefined;
  }

  return asString(first.finish_reason);
}

function extractAnthropicText(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim();
  }

  if (!Array.isArray(content)) {
    return '';
  }

  return content.map(extractTextFromPart).filter(Boolean).join('\n').trim();
}

function extractAnthropicFunctionCalls(content: unknown): StandardResponseFunctionCall[] {
  if (!Array.isArray(content)) {
    return [];
  }

  const toolCalls: StandardResponseFunctionCall[] = [];
  for (const block of content) {
    if (!isObject(block) || asString(block.type) !== 'tool_use') {
      continue;
    }

    const name = asString(block.name);
    if (!name) {
      continue;
    }

    const id = asString(block.id) || `toolu_${randomUUID().replace(/-/g, '')}`;
    toolCalls.push({
      id,
      type: 'function_call',
      call_id: id,
      name,
      arguments: normalizeFunctionCallArguments(block.input),
      status: 'completed'
    });
  }

  return toolCalls;
}

function normalizeFunctionCallArguments(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (value === undefined) {
    return '{}';
  }

  try {
    return JSON.stringify(value);
  } catch {
    return '{}';
  }
}

function sumOptional(a?: number, b?: number): number | undefined {
  if (a === undefined && b === undefined) {
    return undefined;
  }

  return (a || 0) + (b || 0);
}

function extractCacheDurationSeconds(
  usageRaw?: Record<string, unknown>,
  detailsRaw?: Record<string, unknown>
): number | undefined {
  if (!usageRaw && !detailsRaw) {
    return undefined;
  }

  const fromSeconds =
    asNumber(detailsRaw?.cache_duration_seconds) ??
    asNumber(detailsRaw?.cache_ttl_seconds) ??
    asNumber(usageRaw?.cache_duration_seconds) ??
    asNumber(usageRaw?.cache_ttl_seconds) ??
    asNumber(usageRaw?.cache_age_seconds);
  if (fromSeconds !== undefined) {
    return normalizeDurationSeconds(fromSeconds);
  }

  const fromMillis =
    asNumber(detailsRaw?.cache_duration_ms) ??
    asNumber(detailsRaw?.cache_ttl_ms) ??
    asNumber(usageRaw?.cache_duration_ms) ??
    asNumber(usageRaw?.cache_ttl_ms);
  if (fromMillis !== undefined) {
    return normalizeDurationSeconds(fromMillis / 1000);
  }

  return undefined;
}

function normalizeDurationSeconds(value: number): number | undefined {
  if (!Number.isFinite(value) || value <= 0) {
    return undefined;
  }

  return Math.trunc(value);
}
