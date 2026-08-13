import { randomUUID } from 'node:crypto';
import type {
  ProviderNativeItem,
  Result,
  ProviderConfig,
  StandardRequest,
  StandardResponse,
  StandardResponseFunctionCall,
  StandardResponseMessageContent,
  StandardResponseReasoning,
  StandardResponseOutputItem,
  StandardUsage
} from '../../../types';
import { err, ok } from '../../../types';
import { asBoolean, asNumber, asString, extractTextFromPart, isObject } from '../../../utils';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import { resolveOpenAIChatProviderThinkingAdapter } from './openai-chat-compat';

interface OpenAIChatCompatibleRewriteOptions {
  generatedReasoningMessageFields?: boolean;
  standardRequest?: StandardRequest;
}

export function rewriteOpenAIChatCompatibleRequest(
  body: unknown,
  targetProviderConfig?: ProviderConfig,
  options: OpenAIChatCompatibleRewriteOptions = {}
): unknown {
  if (!isObject(body) || targetProviderConfig?.type !== 'openai_chat_completions') {
    return body;
  }

  let nextBody: Record<string, unknown> = { ...body };
  const model = options.standardRequest?.model || asString(nextBody.model);
  const adapter = resolveOpenAIChatProviderThinkingAdapter(targetProviderConfig, model);

  if (adapter) {
    adapter.rewriteRequest({
      body: nextBody,
      standardRequest: options.standardRequest,
      providerConfig: targetProviderConfig,
      model
    });
  } else {
    nextBody = rewriteGenericOpenAIChatCompatibleRequest(nextBody, targetProviderConfig, options);
  }

  if (shouldStripOpenAIChatThinkingOptions(targetProviderConfig)) {
    nextBody = omitOpenAIChatThinkingOptionsFields(nextBody);
  }

  if (shouldStripOpenAIChatReasoningSplit(targetProviderConfig)) {
    nextBody = omitOpenAIChatReasoningSplitFields(nextBody);
  }

  return nextBody;
}

function rewriteGenericOpenAIChatCompatibleRequest(
  body: Record<string, unknown>,
  targetProviderConfig: ProviderConfig,
  options: OpenAIChatCompatibleRewriteOptions
): Record<string, unknown> {
  let nextBody = body;
  const reasoningSplitMode = targetProviderConfig.openaiChatReasoningSplit ?? 'auto';
  const thinkingOptionsMode = targetProviderConfig.openaiChatThinkingOptions ?? 'auto';
  const requestedReasoningSplit = Boolean(
    options.standardRequest?.reasoning_split ?? readOpenAIChatReasoningSplitOption(nextBody)
  );

  if (reasoningSplitMode === 'enabled' || requestedReasoningSplit) {
    nextBody = {
      ...nextBody,
      reasoning_split: true
    };
    nextBody = omitOpenAIChatReasoningSplitAliases(nextBody);
  } else if (options.generatedReasoningMessageFields) {
    nextBody = omitOpenAIChatMessageReasoningFieldsFromBody(nextBody);
  } else {
    nextBody = omitOpenAIChatReasoningSplitAliases(nextBody);
  }

  if (thinkingOptionsMode === 'enabled' && options.standardRequest) {
    nextBody = applyGenericOpenAIChatThinkingOptions(nextBody, options.standardRequest);
  }

  return nextBody;
}

function shouldStripOpenAIChatReasoningSplit(targetProviderConfig?: ProviderConfig): boolean {
  return (
    targetProviderConfig?.type === 'openai_chat_completions' &&
    targetProviderConfig.openaiChatReasoningSplit === 'disabled'
  );
}

function shouldStripOpenAIChatThinkingOptions(targetProviderConfig?: ProviderConfig): boolean {
  return (
    targetProviderConfig?.type === 'openai_chat_completions' &&
    targetProviderConfig.openaiChatThinkingOptions === 'disabled'
  );
}

function omitOpenAIChatReasoningSplitFields(body: Record<string, unknown>): Record<string, unknown> {
  const strippedMessages = omitOpenAIChatMessageReasoningFields(body.messages);
  if (
    !Object.prototype.hasOwnProperty.call(body, 'reasoning_split') &&
    !Object.prototype.hasOwnProperty.call(body, 'interleaved_thinking') &&
    !Object.prototype.hasOwnProperty.call(body, 'interleavedThinking') &&
    strippedMessages === body.messages
  ) {
    return body;
  }

  const rest = { ...body };
  delete rest.reasoning_split;
  delete rest.interleaved_thinking;
  delete rest.interleavedThinking;
  if (strippedMessages !== body.messages) {
    rest.messages = strippedMessages;
  }
  return rest;
}

function readOpenAIChatReasoningSplitOption(body: Record<string, unknown>): boolean | undefined {
  return (
    asBoolean(body.reasoning_split) ??
    asBoolean(body.interleaved_thinking) ??
    asBoolean(body.interleavedThinking)
  );
}

function omitOpenAIChatReasoningSplitAliases(body: Record<string, unknown>): Record<string, unknown> {
  if (
    !Object.prototype.hasOwnProperty.call(body, 'interleaved_thinking') &&
    !Object.prototype.hasOwnProperty.call(body, 'interleavedThinking')
  ) {
    return body;
  }

  const rest = { ...body };
  delete rest.interleaved_thinking;
  delete rest.interleavedThinking;
  return rest;
}

function omitOpenAIChatMessageReasoningFieldsFromBody(
  body: Record<string, unknown>
): Record<string, unknown> {
  const strippedMessages = omitOpenAIChatMessageReasoningFields(body.messages);
  if (strippedMessages === body.messages) {
    return body;
  }

  return {
    ...body,
    messages: strippedMessages
  };
}

function omitOpenAIChatMessageReasoningFields(messages: unknown): unknown {
  if (!Array.isArray(messages)) {
    return messages;
  }

  let changed = false;
  const nextMessages = messages.map((message) => {
    if (
      !isObject(message) ||
      (
        !Object.prototype.hasOwnProperty.call(message, 'reasoning_content') &&
        !Object.prototype.hasOwnProperty.call(message, 'reasoning_details') &&
        !Object.prototype.hasOwnProperty.call(message, 'reasoning') &&
        !Object.prototype.hasOwnProperty.call(message, 'thinking')
      )
    ) {
      return message;
    }

    changed = true;
    const nextMessage = { ...message };
    delete nextMessage.reasoning_content;
    delete nextMessage.reasoning_details;
    delete nextMessage.reasoning;
    delete nextMessage.thinking;
    return nextMessage;
  });

  return changed ? nextMessages.filter((message) => !isEmptyOpenAIChatAssistantMessage(message)) : messages;
}

function isEmptyOpenAIChatAssistantMessage(message: unknown): boolean {
  if (!isObject(message) || message.role !== 'assistant') {
    return false;
  }

  if (
    Object.prototype.hasOwnProperty.call(message, 'tool_calls') ||
    Object.prototype.hasOwnProperty.call(message, 'function_call')
  ) {
    return false;
  }

  const content = message.content;
  return content === undefined || content === '' || (Array.isArray(content) && content.length === 0);
}

function omitOpenAIChatThinkingOptionsFields(body: Record<string, unknown>): Record<string, unknown> {
  if (
    !Object.prototype.hasOwnProperty.call(body, 'thinking') &&
    !Object.prototype.hasOwnProperty.call(body, 'output_config') &&
    !Object.prototype.hasOwnProperty.call(body, 'reasoning_effort')
  ) {
    return body;
  }

  const rest = { ...body };
  delete rest.thinking;
  delete rest.output_config;
  delete rest.reasoning_effort;
  return rest;
}

function applyGenericOpenAIChatThinkingOptions(
  body: Record<string, unknown>,
  standardRequest: StandardRequest
): Record<string, unknown> {
  let nextBody = body;
  const thinking = standardRequest.thinking ?? thinkingFromResponsesReasoning(standardRequest.reasoning);
  if (thinking !== undefined) {
    nextBody = {
      ...nextBody,
      thinking
    };
  }

  const outputConfig =
    standardRequest.output_config ?? outputConfigFromResponsesReasoning(standardRequest.reasoning);
  if (outputConfig !== undefined) {
    nextBody = {
      ...nextBody,
      output_config: outputConfig
    };
  }

  return nextBody;
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

export function applyOpenAIChatStreamUsageOption(
  body: unknown,
  targetProviderConfig?: ProviderConfig
): unknown {
  if (!shouldIncludeOpenAIChatStreamUsage(targetProviderConfig) || !isObject(body)) {
    return body;
  }

  if (asBoolean(body.stream) !== true) {
    return body;
  }

  const streamOptions = isObject(body.stream_options) ? body.stream_options : {};
  return {
    ...body,
    stream_options: {
      ...streamOptions,
      include_usage: true
    }
  };
}

export function shouldIncludeOpenAIChatStreamUsage(
  targetProviderConfig?: ProviderConfig
): boolean {
  return (
    targetProviderConfig?.type === 'openai_chat_completions' &&
    targetProviderConfig.openaiChatStreamUsage !== 'disabled'
  );
}

export function parseOpenAIToStandardResponse(
  payload: unknown,
  options: { compactionMode?: ProviderNativeItem['compaction_mode'] } = {}
): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid OpenAI response payload.');
  }

  const outputText =
    asString(payload.output_text) ||
    extractOpenAIResponseOutputText(payload.output) ||
    extractOpenAIChatText(payload.choices);
  const toolCalls = extractOpenAIFunctionCalls(payload);
  const reasoningItems = extractOpenAIReasoningItems(payload);
  const responseIncomplete = asString(payload.status) === 'incomplete';
  const orderedResponsesOutput = Array.isArray(payload.output)
    ? buildOrderedOpenAIResponsesOutputItems(
        payload.output,
        options.compactionMode
      )
    : undefined;

  if (
    !responseIncomplete &&
    !outputText &&
    toolCalls.length === 0 &&
    reasoningItems.length === 0 &&
    (!orderedResponsesOutput || orderedResponsesOutput.length === 0)
  ) {
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
      asNumber(inputDetails?.cache_write_tokens) ??
      asNumber(inputDetails?.cache_creation_tokens) ??
      asNumber(usageRaw?.cache_creation_tokens) ??
      asNumber(usageRaw?.cache_write_tokens),
    cache_duration_seconds: extractCacheDurationSeconds(usageRaw, inputDetails),
    server_tool_use: normalizeServerToolUse(
      isObject(usageRaw?.server_tool_use) ? usageRaw.server_tool_use : undefined
    )
  };

  return ok(createStandardResponse({
    id: asString(payload.id) || `resp_${randomUUID()}`,
    model: asString(payload.model) || 'unknown',
    outputText,
    outputItems: orderedResponsesOutput || buildStandardResponseOutputItems(outputText, toolCalls, reasoningItems),
    usage,
    finishReason: responseIncomplete
      ? asString(isObject(payload.incomplete_details) ? payload.incomplete_details.reason : undefined) ||
        'max_tokens'
      : extractOpenAIFinishReason(payload.choices),
    status: responseIncomplete ? 'incomplete' : 'completed'
  }));
}

export function parseAnthropicToStandardResponse(
  payload: unknown,
  options: { providerMode?: string } = {}
): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid Anthropic response payload.');
  }

  const text = extractAnthropicText(payload.content);
  const toolCalls = extractAnthropicFunctionCalls(payload.content);
  const reasoningItems = extractAnthropicReasoningItems(payload.content);
  const orderedContent = Array.isArray(payload.content)
    ? buildOrderedAnthropicOutputItems(payload.content, options.providerMode)
    : undefined;
  if (!text && toolCalls.length === 0 && reasoningItems.length === 0) {
    return err('Anthropic response does not contain text output, reasoning output, or tool calls.');
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
    outputItems: orderedContent || buildStandardResponseOutputItems(text, toolCalls, reasoningItems),
    usage,
    finishReason: asString(payload.stop_reason)
  }));
}

export function parseGeminiToStandardResponse(payload: unknown): Result<StandardResponse> {
  if (!isObject(payload)) {
    return err('Invalid Gemini response payload.');
  }

  if (payload.object === 'interaction' || Array.isArray(payload.steps)) {
    return parseGeminiInteractionToStandardResponse(payload);
  }

  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const first = isObject(candidates[0]) ? candidates[0] : undefined;
  const content = isObject(first?.content) ? first.content : undefined;
  const parts = Array.isArray(content?.parts) ? content.parts : [];
  const text = extractGeminiText(parts);
  const toolCalls = extractGeminiFunctionCalls(parts);
  const reasoningItems = extractGeminiReasoningItems(parts);
  const orderedParts = buildOrderedGeminiGenerateContentOutputItems(parts);

  if (!text && toolCalls.length === 0 && reasoningItems.length === 0) {
    return err('Gemini response does not contain text output, reasoning output, or tool calls.');
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
    outputItems: orderedParts.length > 0
      ? orderedParts
      : buildStandardResponseOutputItems(text, toolCalls, reasoningItems),
    usage,
    finishReason: toolCalls.length > 0 ? 'tool_use' : asString(first?.finishReason)
  }));
}

function parseGeminiInteractionToStandardResponse(payload: Record<string, unknown>): Result<StandardResponse> {
  const steps = Array.isArray(payload.steps) ? payload.steps : [];
  const outputText = extractGeminiInteractionOutputText(steps);
  const toolCalls = extractGeminiInteractionFunctionCalls(steps);
  const reasoningItems = extractGeminiInteractionReasoningItems(steps);
  const orderedSteps = buildOrderedGeminiInteractionOutputItems(steps);

  if (!outputText && toolCalls.length === 0 && reasoningItems.length === 0) {
    return err('Gemini interaction response does not contain text output, reasoning output, or tool calls.');
  }

  const usageRaw = isObject(payload.usage) ? payload.usage : undefined;
  const usage: StandardUsage = {
    input_tokens: asNumber(usageRaw?.total_input_tokens),
    output_tokens: asNumber(usageRaw?.total_output_tokens),
    total_tokens: asNumber(usageRaw?.total_tokens),
    cache_read_tokens: asNumber(usageRaw?.total_cached_tokens)
  };

  const status = asString(payload.status);
  return ok(createStandardResponse({
    id: asString(payload.id) || `gemini_interaction_${randomUUID().replace(/-/g, '')}`,
    model: asString(payload.model) || asString(payload.agent) || 'unknown',
    outputText,
    outputItems: orderedSteps.length > 0
      ? orderedSteps
      : buildStandardResponseOutputItems(outputText, toolCalls, reasoningItems),
    usage,
    finishReason: toolCalls.length > 0 ? 'tool_use' : mapGeminiInteractionStatusToFinishReason(status)
  }));
}

function buildOrderedGeminiGenerateContentOutputItems(
  parts: unknown[]
): StandardResponseOutputItem[] {
  const output: StandardResponseOutputItem[] = [];
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    if (!isObject(part)) {
      continue;
    }
    const functionCall = isObject(part.functionCall)
      ? part.functionCall
      : isObject(part.function_call)
        ? part.function_call
        : undefined;
    const functionResponse = isObject(part.functionResponse)
      ? part.functionResponse
      : isObject(part.function_response)
        ? part.function_response
        : undefined;
    const signature = readGeminiThoughtSignature(part, functionCall);
    const isThought = asBoolean(part.thought) === true;
    const itemType = functionCall
      ? 'function_call'
      : functionResponse
        ? 'function_response'
        : isThought
          ? 'thought'
          : signature
            ? 'part'
            : 'part';
    const callId = functionCall
      ? asString(functionCall.id) || asString(functionCall.call_id)
      : functionResponse
        ? asString(functionResponse.id) || asString(functionResponse.call_id)
        : undefined;
    const text = extractTextFromPart(part);
    const nativeItem: ProviderNativeItem = {
      type: 'provider_native_item',
      item_type: itemType,
      ...(callId ? { native_id: callId, group_id: callId, call_id: callId, pair_id: callId } : {}),
      raw_payload: part,
      provider_schema_version: 'gemini-generate-content-v1beta',
      item_origin: 'native',
      source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      source_origin: { provider: 'gemini', endpoint: 'pending' },
      position: { turn: 0, step: 0, item: index },
      capture_state: 'complete',
      ...(isThought && text ? { readable_text: text } : {})
    };

    if (functionCall) {
      const name = asString(functionCall.name);
      if (!name) {
        output.push(nativeItem);
        continue;
      }
      const id = callId || `gemini_call_${randomUUID().replace(/-/g, '')}`;
      output.push({
        id,
        type: 'function_call',
        call_id: id,
        name,
        arguments: normalizeFunctionCallArguments(functionCall.args ?? functionCall.arguments),
        ...(signature ? {
          thought_signature: signature,
          thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT
        } : {}),
        status: 'completed',
        native_item: { ...nativeItem, native_id: id, group_id: id, call_id: id, pair_id: id }
      });
      continue;
    }

    if (isThought) {
      const details: unknown[] = [
        ...(text ? [{
          type: 'reasoning.text',
          text,
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index
        }] : []),
        ...(signature ? [{
          type: 'reasoning.encrypted',
          data: signature,
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index
        }] : [])
      ];
      output.push({
        id: asString(part.id) || `rs_${randomUUID().replace(/-/g, '')}`,
        type: 'reasoning',
        status: 'completed',
        summary: [],
        ...(text ? { content: [{ type: 'reasoning_text', text }] } : {}),
        ...(signature ? { encrypted_content: signature } : {}),
        reasoning_details: details,
        source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        native_item: nativeItem
      });
      continue;
    }

    if (!functionResponse && (typeof part.text === 'string' || text)) {
      const rawText = typeof part.text === 'string' ? part.text : text;
      output.push({
        id: asString(part.id) || `msg_${randomUUID().replace(/-/g, '')}`,
        type: 'message',
        role: 'assistant',
        status: 'completed',
        native_item: { ...nativeItem, ...(rawText ? { readable_text: rawText } : {}) },
        content: [{ type: 'output_text', text: rawText, annotations: [] }]
      });
      continue;
    }

    output.push(nativeItem);
  }
  return output;
}

function buildOrderedGeminiInteractionOutputItems(
  steps: unknown[]
): StandardResponseOutputItem[] {
  const output: StandardResponseOutputItem[] = [];
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    if (!isObject(step)) {
      continue;
    }
    const type = asString(step.type) || 'unknown';
    const signedNative = type === 'thought' || isGeminiInteractionsSignedBuiltInStepType(type);
    const signature = typeof step.signature === 'string' ? step.signature : undefined;
    const nativeItem: ProviderNativeItem | undefined = signedNative
      ? {
          type: 'provider_native_item',
          item_type: type,
          ...(asString(step.id) ? { native_id: asString(step.id) } : {}),
          raw_payload: step,
          provider_schema_version: 'gemini-interactions-v1beta',
          item_origin: 'native',
          source_format: GEMINI_INTERACTIONS_REASONING_FORMAT,
          source_origin: { provider: 'gemini', endpoint: 'pending' },
          position: { turn: 0, step: index, item: 0 },
          ...(asString(step.call_id) ? {
            group_id: asString(step.call_id),
            call_id: asString(step.call_id),
            pair_id: asString(step.call_id)
          } : {}),
          capture_state: 'complete'
        }
      : undefined;

    if (type === 'thought') {
      const summary = extractGeminiInteractionThoughtSummary(step.summary) ||
        extractGeminiInteractionThoughtSummary(step.thought_summary);
      const text = asString(step.text) || asString(step.thought);
      output.push({
        id: asString(step.id) || `rs_${randomUUID().replace(/-/g, '')}`,
        type: 'reasoning',
        status: 'completed',
        summary: summary ? [{ type: 'summary_text', text: summary }] : [],
        ...(text ? { content: [{ type: 'reasoning_text', text }] } : {}),
        ...(signature ? { encrypted_content: signature } : {}),
        reasoning_details: [
          ...(summary ? [{ type: 'reasoning.summary', summary, format: GEMINI_INTERACTIONS_REASONING_FORMAT, index }] : []),
          ...(text ? [{ type: 'reasoning.text', text, format: GEMINI_INTERACTIONS_REASONING_FORMAT, index }] : []),
          ...(signature ? [{ type: 'reasoning.encrypted', data: signature, format: GEMINI_INTERACTIONS_REASONING_FORMAT, index }] : [])
        ],
        source_format: GEMINI_INTERACTIONS_REASONING_FORMAT,
        ...(nativeItem ? { native_item: { ...nativeItem, ...(text ? { readable_text: text } : {}), ...(summary ? { readable_summary: summary } : {}) } } : {})
      });
      continue;
    }

    if (type === 'model_output') {
      const content = Array.isArray(step.content) ? step.content : [];
      const text = content.map(extractTextFromPart).filter(Boolean).join('\n').trim();
      if (text) {
        output.push({
          id: asString(step.id) || `msg_${randomUUID().replace(/-/g, '')}`,
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text, annotations: [] }]
        });
      }
      continue;
    }

    if (type === 'function_call') {
      const name = asString(step.name);
      if (name) {
        const id = asString(step.id) || asString(step.call_id) || `gemini_call_${randomUUID().replace(/-/g, '')}`;
        output.push({
          id,
          type: 'function_call',
          call_id: id,
          name,
          arguments: normalizeFunctionCallArguments(step.arguments),
          status: 'completed'
        });
      }
      continue;
    }

    if (nativeItem) {
      output.push(nativeItem);
    }
  }
  return output;
}

function isGeminiInteractionsSignedBuiltInStepType(type: string): boolean {
  return /^(?:code_execution|file_search|google_maps|google_search|retrieval)_(?:call|result)$/.test(type);
}

function createStandardResponse(args: {
  id: string;
  model: string;
  outputText: string;
  outputItems?: StandardResponseOutputItem[];
  usage: StandardUsage;
  finishReason?: string;
  status?: 'completed' | 'incomplete';
}): StandardResponse {
  const output = args.outputItems && args.outputItems.length > 0
    ? args.outputItems
    : buildStandardResponseOutputItems(args.outputText);

  return {
    id: args.id,
    object: 'response',
    status: args.status ?? 'completed',
    model: args.model,
    output_text: args.outputText,
    output,
    usage: args.usage,
    finish_reason: args.finishReason
  };
}

function extractGeminiInteractionOutputText(steps: unknown[]): string {
  const chunks: string[] = [];
  for (const step of steps) {
    if (!isObject(step) || asString(step.type) !== 'model_output') {
      continue;
    }

    const content = Array.isArray(step.content) ? step.content : [];
    for (const item of content) {
      const text = extractTextFromPart(item);
      if (text) {
        chunks.push(text);
      }
    }
  }

  return chunks.join('\n').trim();
}

function extractGeminiInteractionFunctionCalls(steps: unknown[]): StandardResponseFunctionCall[] {
  const toolCalls: StandardResponseFunctionCall[] = [];
  for (const step of steps) {
    if (!isObject(step) || asString(step.type) !== 'function_call') {
      continue;
    }

    const name = asString(step.name);
    if (!name) {
      continue;
    }

    const id = asString(step.id) || asString(step.call_id) || `gemini_call_${randomUUID().replace(/-/g, '')}`;
    toolCalls.push({
      id,
      type: 'function_call',
      call_id: id,
      name,
      arguments: normalizeFunctionCallArguments(step.arguments),
      status: 'completed'
    });
  }

  return toolCalls;
}

function extractGeminiInteractionReasoningItems(steps: unknown[]): StandardResponseReasoning[] {
  const reasoningItems: StandardResponseReasoning[] = [];

  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    if (!isObject(step) || asString(step.type) !== 'thought') {
      continue;
    }

    const reasoningContent: NonNullable<StandardResponseReasoning['content']> = [];
    const reasoningSummary: NonNullable<StandardResponseReasoning['summary']> = [];
    const reasoningDetails: unknown[] = [];
    const summary =
      extractGeminiInteractionThoughtSummary(step.summary) ||
      extractGeminiInteractionThoughtSummary(step.thought_summary);
    if (summary) {
      reasoningSummary.push({
        type: 'summary_text',
        text: summary
      });
      reasoningDetails.push({
        type: 'reasoning.summary',
        summary,
        format: GEMINI_INTERACTIONS_REASONING_FORMAT,
        index
      });
    }

    const text = asString(step.text) || asString(step.thought);
    if (text) {
      reasoningContent.push({
        type: 'reasoning_text',
        text
      });
      reasoningDetails.push({
        type: 'reasoning.text',
        text,
        format: GEMINI_INTERACTIONS_REASONING_FORMAT,
        index
      });
    }

    const signature = asString(step.signature);
    if (signature) {
      reasoningDetails.push({
        type: 'reasoning.encrypted',
        data: signature,
        format: GEMINI_INTERACTIONS_REASONING_FORMAT,
        index
      });
    }

    if (reasoningDetails.length === 0) {
      continue;
    }

    reasoningItems.push({
      id: asString(step.id) || `rs_${randomUUID().replace(/-/g, '')}`,
      type: 'reasoning',
      status: 'completed',
      summary: reasoningSummary,
      ...(reasoningContent.length > 0 ? { content: reasoningContent } : {}),
      ...(signature ? { encrypted_content: signature } : {}),
      reasoning_details: reasoningDetails,
      source_format: GEMINI_INTERACTIONS_REASONING_FORMAT
    });
  }

  return reasoningItems;
}

function extractGeminiInteractionThoughtSummary(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }

  const items = Array.isArray(value) ? value : value !== undefined ? [value] : [];
  return items
    .map((item) => extractTextFromPart(item))
    .filter(Boolean)
    .join('\n')
    .trim();
}

function mapGeminiInteractionStatusToFinishReason(status?: string): string | undefined {
  if (!status) {
    return undefined;
  }

  if (status === 'incomplete' || status === 'budget_exceeded') {
    return 'length';
  }

  if (status === 'requires_action') {
    return 'tool_use';
  }

  return status;
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

function buildOrderedOpenAIResponsesOutputItems(
  rawOutput: unknown[],
  compactionMode: ProviderNativeItem['compaction_mode'] | undefined
): StandardResponseOutputItem[] {
  const output: StandardResponseOutputItem[] = [];
  for (let index = 0; index < rawOutput.length; index += 1) {
    const rawItem = rawOutput[index];
    if (!isObject(rawItem)) {
      continue;
    }

    const nativeItem = captureOpenAIResponsesOutputItem(
      rawItem,
      index,
      compactionMode
    );
    const type = asString(rawItem.type);
    if (type === 'reasoning') {
      const reasoning = normalizeOpenAIResponsesReasoningItem(rawItem);
      if (reasoning) {
        output.push({
          ...reasoning,
          status: asString(rawItem.status) === 'incomplete' ? 'incomplete' : 'completed',
          native_item: nativeItem
        });
      } else {
        output.push(nativeItem);
      }
      continue;
    }

    if (type === 'message') {
      const content = normalizeOpenAIResponsesOutputMessageContent(rawItem.content);
      output.push({
        id: asString(rawItem.id) || `msg_${randomUUID().replace(/-/g, '')}`,
        type: 'message',
        role: 'assistant',
        status: asString(rawItem.status) === 'incomplete' ? 'incomplete' : 'completed',
        ...(asString(rawItem.phase) ? { phase: asString(rawItem.phase) } : {}),
        native_item: nativeItem,
        content
      });
      continue;
    }

    const isCompletedClientToolSearch = type === 'tool_search_call' &&
      asString(rawItem.execution) === 'client' &&
      asString(rawItem.status) === 'completed' &&
      Boolean(asString(rawItem.call_id));
    if (type === 'function_call' || type === 'tool_call' || isCompletedClientToolSearch) {
      const functionPayload = isObject(rawItem.function) ? rawItem.function : undefined;
      const name = isCompletedClientToolSearch
        ? 'ToolSearch'
        : asString(rawItem.name) || asString(functionPayload?.name);
      if (!name) {
        output.push(nativeItem);
        continue;
      }
      const id = asString(rawItem.id) ||
        (isCompletedClientToolSearch ? asString(rawItem.call_id) : undefined) ||
        `fc_${randomUUID().replace(/-/g, '')}`;
      output.push({
        id,
        type: 'function_call',
        call_id: asString(rawItem.call_id) || id,
        name,
        arguments: normalizeFunctionCallArguments(
          rawItem.arguments ?? functionPayload?.arguments ?? rawItem.input
        ),
        status: asString(rawItem.status) === 'incomplete' ? 'incomplete' : 'completed',
        ...(isObject(rawItem.caller) ? { caller: rawItem.caller } : {}),
        native_item: nativeItem
      });
      continue;
    }

    // Program, Program Output, Compaction, function-call outputs, and unknown
    // future items stay lossless internally. Their protocol projector decides
    // whether the exact raw item is safe to replay.
    output.push(nativeItem);
  }
  return output;
}

function normalizeOpenAIResponsesOutputMessageContent(
  value: unknown
): StandardResponseMessageContent[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const content: StandardResponseMessageContent[] = [];
  for (const part of value) {
    if (!isObject(part)) {
      continue;
    }
    const text = asString(part.text);
    if (asString(part.type) !== 'output_text' || text === undefined) {
      continue;
    }
    content.push({
      type: 'output_text',
      text,
      annotations: Array.isArray(part.annotations) ? part.annotations : []
    });
  }
  return content;
}

function captureOpenAIResponsesOutputItem(
  rawItem: Record<string, unknown>,
  itemIndex: number,
  compactionMode: ProviderNativeItem['compaction_mode'] | undefined
): ProviderNativeItem {
  const itemType = asString(rawItem.type) || 'unknown';
  const callId = asString(rawItem.call_id) || asString(rawItem.program_id);
  const caller = isObject(rawItem.caller) ? rawItem.caller : undefined;
  const callerId = asString(caller?.id) || asString(caller?.caller_id);
  const rawDependsOn = Array.isArray(rawItem.depends_on)
    ? rawItem.depends_on.map(asString).filter((value): value is string => Boolean(value))
    : [];
  const dependsOn = [
    ...rawDependsOn,
    ...(asString(rawItem.program_id) ? [asString(rawItem.program_id)!] : [])
  ];
  const readableText = itemType === 'message'
    ? extractOpenAIResponseOutputText([rawItem])
    : normalizeReasoningContentParts(rawItem.content).map((part) => part.text).join('\n').trim();
  const readableSummary = normalizeReasoningSummaryParts(rawItem.summary)
    .map((part) => part.text)
    .join('\n')
    .trim();

  return {
    type: 'provider_native_item',
    item_type: itemType,
    ...(asString(rawItem.id) ? { native_id: asString(rawItem.id) } : {}),
    raw_payload: rawItem,
    provider_schema_version: 'openai-responses-v1',
    item_origin: 'native',
    source_format: OPENAI_RESPONSES_REASONING_FORMAT,
    source_origin: {
      provider: 'openai',
      endpoint: 'pending',
      ...(asString(rawItem.model) ? { model: asString(rawItem.model) } : {})
    },
    position: { turn: 0, step: 0, item: itemIndex },
    ...(callerId || callId ? { group_id: callerId || callId } : {}),
    ...(callId ? { call_id: callId, pair_id: callId } : {}),
    ...(dependsOn.length > 0 ? { depends_on: [...new Set(dependsOn)] } : {}),
    capture_state: 'complete',
    ...(asString(rawItem.status) ? { provider_status: asString(rawItem.status) } : {}),
    ...(readableText ? { readable_text: readableText } : {}),
    ...(readableSummary ? { readable_summary: readableSummary } : {}),
    ...(itemType === 'compaction'
      ? { compaction_mode: compactionMode || 'server_side' }
      : {})
  };
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
    summary,
    source_format: OPENAI_RESPONSES_REASONING_FORMAT
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
  if (details.sourceFormat) {
    reasoning.source_format = details.sourceFormat;
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
  sourceFormat?: string;
  content: NonNullable<StandardResponseReasoning['content']>;
  summary: StandardResponseReasoning['summary'];
  encryptedContent?: string;
  rawDetails: unknown[];
} {
  const normalized: {
    id?: string;
    sourceFormat?: string;
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
    const format = asString(detail.format);
    if (format && !normalized.sourceFormat) {
      normalized.sourceFormat = format;
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
  return [
    ...extractOpenAIResponsesFunctionCalls(payload.output, asString(payload.status)),
    ...extractOpenAIChatFunctionCalls(payload.choices)
  ];
}

function extractOpenAIResponsesFunctionCalls(
  output: unknown,
  responseStatus: string | undefined
): StandardResponseFunctionCall[] {
  if (!Array.isArray(output)) {
    return [];
  }
  if (responseStatus === 'incomplete') {
    return [];
  }

  const toolCalls: StandardResponseFunctionCall[] = [];
  for (const item of output) {
    if (!isObject(item)) {
      continue;
    }

    const type = asString(item.type);
    const isClientToolSearch =
      type === 'tool_search_call' &&
      asString(item.execution) === 'client' &&
      asString(item.status) === 'completed' &&
      Boolean(asString(item.call_id));
    if (asString(item.status) === 'incomplete') {
      continue;
    }
    if (type !== 'function_call' && type !== 'tool_call' && !isClientToolSearch) {
      continue;
    }

    const functionPayload = isObject(item.function) ? item.function : undefined;
    const name = isClientToolSearch
      ? 'ToolSearch'
      : asString(item.name) || asString(functionPayload?.name);
    if (!name) {
      continue;
    }

    const id =
      asString(item.id) ||
      (isClientToolSearch ? asString(item.call_id) : undefined) ||
      `fc_${randomUUID().replace(/-/g, '')}`;
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

function buildOrderedAnthropicOutputItems(
  content: unknown[],
  providerMode: string | undefined
): StandardResponseOutputItem[] {
  const output: StandardResponseOutputItem[] = [];
  for (let index = 0; index < content.length; index += 1) {
    const block = content[index];
    if (!isObject(block)) {
      continue;
    }
    const type = asString(block.type) || 'unknown';
    const nextToolId = findNextAnthropicToolUseId(content, index + 1);
    const toolId = type === 'tool_use' ? asString(block.id) : undefined;
    const groupId = toolId || ((type === 'thinking' || type === 'redacted_thinking') ? nextToolId : undefined);
    const nativeItem: ProviderNativeItem = {
      type: 'provider_native_item',
      item_type: type,
      ...(asString(block.id) ? { native_id: asString(block.id) } : {}),
      raw_payload: block,
      provider_schema_version: 'anthropic-messages-2023-06-01',
      item_origin: 'native',
      source_format: ANTHROPIC_CLAUDE_REASONING_FORMAT,
      source_origin: { provider: 'anthropic', endpoint: 'pending' },
      position: { turn: 0, step: 0, item: index },
      ...(groupId ? { group_id: groupId, call_id: groupId } : {}),
      capture_state: 'complete',
      ...(providerMode ? { provider_mode: providerMode } : {})
    };

    if (type === 'thinking' || type === 'redacted_thinking') {
      const thinking = typeof block.thinking === 'string' ? block.thinking : undefined;
      const signature = typeof block.signature === 'string' ? block.signature : undefined;
      const data = typeof block.data === 'string' ? block.data : undefined;
      const details = type === 'thinking'
        ? [{
            type: 'thinking',
            thinking: thinking ?? '',
            ...(signature !== undefined ? { signature } : {}),
            format: ANTHROPIC_CLAUDE_REASONING_FORMAT
          }]
        : [{
            type: 'redacted_thinking',
            ...(data !== undefined ? { data } : {}),
            format: ANTHROPIC_CLAUDE_REASONING_FORMAT
          }];
      output.push({
        id: asString(block.id) || `rs_${randomUUID().replace(/-/g, '')}`,
        type: 'reasoning',
        status: 'completed',
        summary: [],
        ...(thinking ? { content: [{ type: 'reasoning_text', text: thinking }] } : {}),
        ...(data ? { encrypted_content: data } : {}),
        reasoning_details: details,
        source_format: ANTHROPIC_CLAUDE_REASONING_FORMAT,
        native_item: {
          ...nativeItem,
          ...(thinking !== undefined ? { readable_text: thinking } : {})
        }
      });
      continue;
    }

    if (type === 'text') {
      const text = typeof block.text === 'string' ? block.text : '';
      output.push({
        id: asString(block.id) || `msg_${randomUUID().replace(/-/g, '')}`,
        type: 'message',
        role: 'assistant',
        status: 'completed',
        native_item: { ...nativeItem, ...(text ? { readable_text: text } : {}) },
        content: [{ type: 'output_text', text, annotations: [] }]
      });
      continue;
    }

    if (type === 'tool_use') {
      const name = asString(block.name);
      if (name && toolId) {
        output.push({
          id: toolId,
          type: 'function_call',
          call_id: toolId,
          name,
          arguments: normalizeFunctionCallArguments(block.input),
          status: 'completed',
          native_item: nativeItem
        });
      } else {
        output.push(nativeItem);
      }
      continue;
    }

    output.push(nativeItem);
  }
  return output;
}

function findNextAnthropicToolUseId(content: unknown[], startIndex: number): string | undefined {
  for (let index = startIndex; index < content.length; index += 1) {
    const block = content[index];
    if (!isObject(block)) {
      continue;
    }
    const type = asString(block.type);
    if (type === 'tool_use') {
      return asString(block.id);
    }
    if (type === 'text') {
      return undefined;
    }
  }
  return undefined;
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

function extractAnthropicReasoningItems(content: unknown): StandardResponseReasoning[] {
  if (!Array.isArray(content)) {
    return [];
  }

  const reasoningContent: NonNullable<StandardResponseReasoning['content']> = [];
  const reasoningDetails: unknown[] = [];
  let encryptedContent: string | undefined;

  for (const block of content) {
    if (!isObject(block)) {
      continue;
    }

    const type = asString(block.type);
    if (type === 'thinking') {
      const thinking = asString(block.thinking);
      const signature = asString(block.signature);
      if (!thinking && !signature) {
        continue;
      }

      if (thinking) {
        reasoningContent.push({
          type: 'reasoning_text',
          text: thinking
        });
      }
      reasoningDetails.push({
        type: 'thinking',
        thinking: thinking || '',
        ...(signature ? { signature } : {}),
        format: ANTHROPIC_CLAUDE_REASONING_FORMAT
      });
      continue;
    }

    if (type === 'redacted_thinking') {
      const data = asString(block.data);
      if (!data) {
        continue;
      }

      encryptedContent = encryptedContent || data;
      reasoningDetails.push({
        type: 'redacted_thinking',
        data,
        format: ANTHROPIC_CLAUDE_REASONING_FORMAT
      });
    }
  }

  if (reasoningContent.length === 0 && !encryptedContent && reasoningDetails.length === 0) {
    return [];
  }

  const reasoning: StandardResponseReasoning = {
    id: `rs_${randomUUID().replace(/-/g, '')}`,
    type: 'reasoning',
    status: 'completed',
    summary: [],
    source_format: ANTHROPIC_CLAUDE_REASONING_FORMAT
  };

  if (reasoningContent.length > 0) {
    reasoning.content = reasoningContent;
  }
  if (encryptedContent) {
    reasoning.encrypted_content = encryptedContent;
  }
  if (reasoningDetails.length > 0) {
    reasoning.reasoning_details = reasoningDetails;
  }

  return [reasoning];
}

function extractGeminiText(parts: unknown[]): string {
  return parts
    .map((part) => {
      if (!isObject(part) || asBoolean(part.thought) === true) {
        return '';
      }

      return extractTextFromPart(part);
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

function extractGeminiFunctionCalls(parts: unknown[]): StandardResponseFunctionCall[] {
  const toolCalls: StandardResponseFunctionCall[] = [];
  for (const part of parts) {
    if (!isObject(part)) {
      continue;
    }

    const functionCall = isObject(part.functionCall)
      ? part.functionCall
      : isObject(part.function_call)
        ? part.function_call
        : undefined;
    if (!functionCall) {
      continue;
    }

    const name = asString(functionCall.name);
    if (!name) {
      continue;
    }

    const id = asString(functionCall.id) || `gemini_call_${randomUUID().replace(/-/g, '')}`;
    const thoughtSignature = readGeminiThoughtSignature(part, functionCall);
    toolCalls.push({
      id,
      type: 'function_call',
      call_id: id,
      name,
      arguments: normalizeFunctionCallArguments(functionCall.args ?? functionCall.arguments),
      ...(thoughtSignature ? { thought_signature: thoughtSignature } : {}),
      ...(thoughtSignature
        ? { thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT }
        : {}),
      status: 'completed'
    });
  }

  return toolCalls;
}

function readGeminiThoughtSignature(
  part: Record<string, unknown>,
  functionCall?: Record<string, unknown>
): string | undefined {
  return (
    asString(part.thoughtSignature) ||
    asString(part.thought_signature) ||
    (functionCall
      ? asString(functionCall.thoughtSignature) || asString(functionCall.thought_signature)
      : undefined)
  );
}

function extractGeminiReasoningItems(parts: unknown[]): StandardResponseReasoning[] {
  const reasoningItems: StandardResponseReasoning[] = [];

  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    if (!isObject(part)) {
      continue;
    }

    const functionCall = isObject(part.functionCall) || isObject(part.function_call);
    const isThought = asBoolean(part.thought) === true;
    const text = isThought ? extractTextFromPart(part) : '';
    const thoughtSignature = functionCall
      ? undefined
      : readGeminiThoughtSignature(part);
    if (!text && !thoughtSignature) {
      continue;
    }

    const reasoningContent: NonNullable<StandardResponseReasoning['content']> = [];
    const reasoningDetails: unknown[] = [];
    if (text) {
      reasoningContent.push({
        type: 'reasoning_text',
        text
      });
      reasoningDetails.push({
        type: 'reasoning.text',
        text,
        format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        index
      });
    }
    if (thoughtSignature) {
      reasoningDetails.push({
        type: 'reasoning.encrypted',
        data: thoughtSignature,
        format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        index
      });
    }

    reasoningItems.push({
      id: `rs_${randomUUID().replace(/-/g, '')}`,
      type: 'reasoning',
      status: 'completed',
      summary: [],
      ...(reasoningContent.length > 0 ? { content: reasoningContent } : {}),
      ...(thoughtSignature ? { encrypted_content: thoughtSignature } : {}),
      reasoning_details: reasoningDetails,
      source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT
    });
  }

  return reasoningItems;
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
