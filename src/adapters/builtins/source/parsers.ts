import type {
  ProviderNativeItem,
  Result,
  SourceAdapterRequestInput,
  StandardRequest,
  StandardRequestInputContent,
  StandardRequestInputMessage
} from '../../../types';
import { err, ok } from '../../../types';
import {
  asBoolean,
  asNumber,
  asStop,
  asString,
  extractTextFromPart,
  isObject,
  normalizeConversationRole,
  normalizeMessageRole
} from '../../../utils';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  decodeGeminiThoughtSignatureToolCallId,
  decodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  normalizeAnthropicThinkingMode,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import { normalizeNamespacedToolName } from '../target/tools';

export function parseOpenAIResponsesRequest(
  body: Record<string, unknown>,
  operation: 'create' | 'compact' = 'create'
): Result<StandardRequest> {
  const inputResult = normalizeResponsesInput(body.input, operation);
  if (!inputResult.ok) {
    return inputResult;
  }

  const instructions = asString(body.instructions);
  const input = ensureInputWithInstructions(inputResult.value, instructions);
  if (!input) {
    return err('OpenAI responses request requires non-empty input.');
  }

  return ok({
    model: asString(body.model),
    instructions,
    input,
    temperature: asNumber(body.temperature),
    top_p: asNumber(body.top_p),
    max_output_tokens: asNumber(body.max_output_tokens),
    stop: asStop(body.stop),
    stream: asBoolean(body.stream),
    tools: readTools(body.tools),
    tool_choice: readToolChoice(body.tool_choice),
    reasoning_split: readReasoningSplitOption(body),
    reasoning: readReasoningOption(body),
    thinking: readOptionalRequestOption(body.thinking),
    output_config: readOptionalRequestOption(body.output_config),
    text: readOptionalRequestOption(body.text),
    openai_responses: {
      operation,
      ...(asString(body.previous_response_id)
        ? { previous_response_id: asString(body.previous_response_id) }
        : {}),
      ...(asBoolean(body.store) !== undefined ? { store: asBoolean(body.store) } : {}),
      ...(body.conversation !== undefined ? { conversation: body.conversation } : {}),
      ...(body.context_management !== undefined
        ? { context_management: body.context_management }
        : {}),
      ...(body.include !== undefined ? { include: body.include } : {})
    }
  });
}

export function parseOpenAIChatCompletionsRequest(body: Record<string, unknown>): Result<StandardRequest> {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  if (rawMessages.length === 0) {
    return err('OpenAI chat request requires non-empty messages array.');
  }

  const instructions: string[] = [];
  const inputMessages: StandardRequestInputMessage[] = [];

  for (const rawMessage of rawMessages) {
    if (!isObject(rawMessage)) {
      continue;
    }

    const role = normalizeMessageRole(rawMessage.role);

    if (role === 'system') {
      const text = extractMessageText(rawMessage.content);
      if (!text) {
        continue;
      }
      instructions.push(text);
      continue;
    }

    const content = extractOpenAIChatMessageContent(rawMessage);
    if (content.length === 0) {
      continue;
    }

    inputMessages.push({
      type: 'message',
      role: role === 'assistant' ? 'assistant' : 'user',
      content
    });
  }

  const mergedInstructions = instructions.join('\n').trim() || undefined;
  const input = ensureInputWithInstructions(inputMessages, mergedInstructions);
  if (!input) {
    return err('OpenAI chat request contains no valid text message.');
  }

  return ok({
    model: asString(body.model),
    instructions: mergedInstructions,
    input,
    temperature: asNumber(body.temperature),
    top_p: asNumber(body.top_p),
    max_output_tokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stop: asStop(body.stop),
    stream: asBoolean(body.stream),
    tools: readTools(body.tools),
    tool_choice: readToolChoice(body.tool_choice),
    reasoning_split: readReasoningSplitOption(body),
    reasoning: readReasoningOption(body),
    thinking: readOptionalRequestOption(body.thinking),
    output_config: readOptionalRequestOption(body.output_config)
  });
}

export function parseAnthropicMessagesRequest(body: Record<string, unknown>): Result<StandardRequest> {
  const inputMessages: StandardRequestInputMessage[] = [];
  const instructions = extractAnthropicSystem(body.system);
  const sourceModel = asString(body.model);
  const providerMode = normalizeAnthropicThinkingMode(body.thinking);

  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (let messageIndex = 0; messageIndex < rawMessages.length; messageIndex += 1) {
    const rawMessage = rawMessages[messageIndex];
    if (!isObject(rawMessage)) {
      continue;
    }

    const role = normalizeConversationRole(rawMessage.role);
    const content = extractAnthropicMessageContent(
      role,
      rawMessage.content,
      messageIndex,
      sourceModel,
      providerMode
    );
    if (content.length === 0) {
      continue;
    }

    inputMessages.push({
      type: 'message',
      role,
      content,
      ...collectNativeItemsFromInputContent(content)
    });
  }

  linkAnthropicNativeToolGroups(inputMessages);

  const input = ensureInputWithInstructions(inputMessages, instructions);
  if (!input) {
    return err('Anthropic request requires non-empty messages or system prompt.');
  }

  return ok({
    model: sourceModel,
    instructions,
    input,
    temperature: asNumber(body.temperature),
    top_p: asNumber(body.top_p),
    max_output_tokens: asNumber(body.max_tokens),
    stop: asStop(body.stop_sequences),
    stream: asBoolean(body.stream),
    tools: readTools(body.tools),
    tool_choice: readToolChoice(body.tool_choice),
    reasoning_split: readReasoningSplitOption(body),
    reasoning: readReasoningOption(body),
    thinking: readOptionalRequestOption(body.thinking),
    output_config: readOptionalRequestOption(body.output_config)
  });
}

export function parseGeminiGenerateContentRequest(
  body: Record<string, unknown>,
  modelFromPath?: string
): Result<StandardRequest> {
  const inputMessages: StandardRequestInputMessage[] = [];
  const instructions = extractGeminiSystemInstruction(body.systemInstruction);
  const geminiToolState = createGeminiToolCallState();
  const sourceModel = modelFromPath || asString(body.model);

  const contents = Array.isArray(body.contents) ? body.contents : [];
  for (let itemIndex = 0; itemIndex < contents.length; itemIndex += 1) {
    const item = contents[itemIndex];
    if (!isObject(item)) {
      continue;
    }

    const role = normalizeConversationRole(item.role === 'model' ? 'assistant' : item.role);
    const parts = Array.isArray(item.parts) ? item.parts : [];
    const content = extractGeminiMessageContent(
      role,
      parts,
      geminiToolState,
      itemIndex,
      sourceModel
    );
    if (content.length === 0) {
      continue;
    }

    inputMessages.push({
      type: 'message',
      role,
      content,
      ...collectNativeItemsFromInputContent(content)
    });
  }

  const input = ensureInputWithInstructions(inputMessages, instructions);
  if (!input) {
    return err('Gemini request requires non-empty contents or systemInstruction.');
  }

  const generationConfig = isObject(body.generationConfig) ? body.generationConfig : undefined;
  const toolConfig = isObject(body.toolConfig)
    ? body.toolConfig
    : isObject(body.tool_config)
      ? body.tool_config
      : undefined;
  const tools = readGeminiTools(body.tools) || readTools(body.tools);
  const toolChoice = readGeminiToolChoice(toolConfig) ?? readToolChoice(body.tool_choice);
  const thinking = readGeminiThinkingOption(generationConfig?.thinkingConfig ?? generationConfig?.thinking_config);

  return ok({
    model: sourceModel,
    instructions,
    input,
    temperature: asNumber(generationConfig?.temperature),
    top_p: asNumber(generationConfig?.topP),
    max_output_tokens: asNumber(generationConfig?.maxOutputTokens),
    stop: asStop(generationConfig?.stopSequences),
    stream: asBoolean(body.stream),
    tools,
    tool_choice: toolChoice,
    thinking
  });
}

export function parseGeminiInteractionsRequest(body: Record<string, unknown>): Result<StandardRequest> {
  const inputResult = normalizeGeminiInteractionsInput(body.input);
  if (!inputResult.ok) {
    return inputResult;
  }

  const generationConfig = readRecordOption(body.generation_config ?? body.generationConfig);
  const instructions = asString(body.system_instruction) || asString(body.systemInstruction);
  const input = ensureInputWithInstructions(inputResult.value, instructions);
  if (!input) {
    return err('Gemini interactions request requires non-empty input or system_instruction.');
  }

  const agent = asString(body.agent);
  const model = asString(body.model) || agent;
  const toolChoice = readGeminiInteractionsToolChoice(body.tool_choice) ?? readToolChoice(body.tool_choice);

  return ok({
    model,
    instructions,
    input,
    temperature: asNumber(generationConfig?.temperature),
    top_p: asNumber(generationConfig?.top_p) ?? asNumber(generationConfig?.topP),
    max_output_tokens: asNumber(generationConfig?.max_output_tokens) ?? asNumber(generationConfig?.maxOutputTokens),
    stop: asStop(generationConfig?.stop_sequences ?? generationConfig?.stopSequences),
    stream: asBoolean(body.stream),
    tools: readTools(body.tools),
    tool_choice: toolChoice,
    gemini_interactions: {
      ...(agent ? { agent } : {}),
      ...(asString(body.previous_interaction_id) ? { previous_interaction_id: asString(body.previous_interaction_id) } : {}),
      ...(asBoolean(body.store) !== undefined ? { store: asBoolean(body.store) } : {}),
      ...(asBoolean(body.background) !== undefined ? { background: asBoolean(body.background) } : {}),
      ...(body.response_format !== undefined ? { response_format: body.response_format } : {}),
      ...(generationConfig ? { generation_config: generationConfig } : {}),
      ...(body.agent_config !== undefined ? { agent_config: body.agent_config } : {}),
      ...(body.response_modalities !== undefined ? { response_modalities: body.response_modalities } : {}),
      ...(asString(body.service_tier) ? { service_tier: asString(body.service_tier) } : {}),
      ...(body.environment !== undefined ? { environment: body.environment } : {}),
      ...(asString(body.cached_content) ? { cached_content: asString(body.cached_content) } : {}),
      ...(body.webhook_config !== undefined ? { webhook_config: body.webhook_config } : {})
    }
  });
}

export function readGeminiMetadata(
  input: SourceAdapterRequestInput,
  defaultAction: 'generateContent' | 'streamGenerateContent'
): Result<{ model: string; action: 'generateContent' | 'streamGenerateContent'; apiVersion: string }> {
  const model = input.source.metadata?.model;
  if (!model) {
    return err('Gemini model is missing in route path.');
  }

  const actionRaw = input.source.metadata?.action as 'generateContent' | 'streamGenerateContent' | undefined;
  const action = actionRaw || defaultAction;
  if (action !== 'generateContent' && action !== 'streamGenerateContent') {
    return err('Invalid Gemini action.');
  }

  const apiVersion = input.source.metadata?.apiVersion || input.config.geminiApiVersion;
  return ok({ model, action, apiVersion });
}

export function readGeminiInteractionsMetadata(
  input: SourceAdapterRequestInput
): Result<{ apiVersion: string }> {
  return ok({
    apiVersion: input.source.metadata?.apiVersion || input.config.geminiApiVersion
  });
}

interface OpenAIResponsesRawInputProjection {
  rawPayload: Record<string, unknown>;
  position: ProviderNativeItem['position'];
}

type OpenAIResponsesRawInputProjectionMap = Map<
  StandardRequestInputContent,
  OpenAIResponsesRawInputProjection
>;

function normalizeResponsesInput(
  input: unknown,
  operation: 'create' | 'compact' = 'create'
): Result<string | StandardRequestInputMessage[]> {
  if (typeof input === 'string') {
    return ok(input.trim());
  }

  if (!Array.isArray(input)) {
    if (isObject(input)) {
      const rawItemsByContent: OpenAIResponsesRawInputProjectionMap = new Map();
      const asMessage = normalizeResponsesInputItem(
        input,
        0,
        operation,
        rawItemsByContent
      );
      if (!asMessage) {
        return err('OpenAI responses request contains invalid input item.');
      }

      const messages = [asMessage];
      linkOpenAIResponsesNativeToolGroups(messages, rawItemsByContent);
      return ok(messages);
    }

    return err('OpenAI responses request requires input.');
  }

  const messages: StandardRequestInputMessage[] = [];
  const rawItemsByContent: OpenAIResponsesRawInputProjectionMap = new Map();
  for (let itemIndex = 0; itemIndex < input.length; itemIndex += 1) {
    const normalized = normalizeResponsesInputItem(
      input[itemIndex],
      itemIndex,
      operation,
      rawItemsByContent
    );
    if (normalized) {
      messages.push(normalized);
    }
  }

  const coalesced = coalesceResponsesInputMessages(messages);
  linkOpenAIResponsesNativeToolGroups(coalesced, rawItemsByContent);
  return ok(coalesced);
}

function normalizeGeminiInteractionsInput(input: unknown): Result<string | StandardRequestInputMessage[]> {
  if (typeof input === 'string') {
    return ok(input.trim());
  }

  if (!Array.isArray(input)) {
    if (isObject(input)) {
      const message = normalizeGeminiInteractionInputItem(input, new Map(), 0);
      if (!message) {
        return err('Gemini interactions request contains invalid input item.');
      }
      return ok([message]);
    }

    return err('Gemini interactions request requires input.');
  }

  const messages: StandardRequestInputMessage[] = [];
  const toolNamesById = new Map<string, string>();
  for (let itemIndex = 0; itemIndex < input.length; itemIndex += 1) {
    const message = normalizeGeminiInteractionInputItem(
      input[itemIndex],
      toolNamesById,
      itemIndex
    );
    if (message) {
      messages.push(message);
    }
  }

  return ok(coalesceResponsesInputMessages(messages));
}

function normalizeGeminiInteractionInputItem(
  item: unknown,
  toolNamesById = new Map<string, string>(),
  itemIndex = 0
): StandardRequestInputMessage | null {
  if (typeof item === 'string') {
    const text = item.trim();
    return text
      ? {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text }]
        }
      : null;
  }

  if (!isObject(item)) {
    return null;
  }

  const type = asString(item.type);
  if (type === 'function_call') {
    const name = asString(item.name);
    if (!name) {
      return null;
    }
    const id = asString(item.id) || asString(item.call_id) || `gemini_interaction_call_${name}`;
    toolNamesById.set(id, name);
    return {
      type: 'message',
      role: 'assistant',
      content: [
        {
          type: 'tool_use',
          id,
          name,
          input: normalizeFunctionArgumentsInput(item.arguments)
        }
      ]
    };
  }

  if (type === 'function_result') {
    const callId = asString(item.call_id) || asString(item.id);
    if (!callId) {
      return null;
    }
    const name = asString(item.name) || toolNamesById.get(callId);
    const toolResult: StandardRequestInputContent = {
      type: 'tool_result',
      tool_use_id: callId,
      ...(name ? { name } : {}),
      content: normalizeGeminiInteractionResultContent(item.result)
    };
    return {
      type: 'message',
      role: 'user',
      content: [toolResult]
    };
  }

  if (type === 'thought') {
    const text = asString(item.text) || asString(item.thought);
    const summary =
      normalizeGeminiInteractionThoughtSummary(item.summary) ||
      normalizeGeminiInteractionThoughtSummary(item.thought_summary);
    const signature = asString(item.signature);
    const envelope = signature
      ? decodeReasoningTransportEnvelope(signature)
      : undefined;
    const encryptedContent = envelope?.data || signature;
    const sourceFormat = envelope?.format || GEMINI_INTERACTIONS_REASONING_FORMAT;
    if (!text && !summary && !signature) {
      return null;
    }
    const nativeItem = captureGeminiInteractionSignedStep(item, itemIndex, envelope);
    return {
      type: 'message',
      role: 'assistant',
      content: [
        {
          type: 'reasoning',
          ...(envelope?.id ? { id: envelope.id } : {}),
          source_format: sourceFormat,
          ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
          ...(text ? { text } : {}),
          ...(summary ? { summary } : {}),
          ...(encryptedContent ? { encrypted_content: encryptedContent } : {}),
          reasoning_details: [
            ...(summary
              ? [
                  {
                    type: 'reasoning.summary',
                    summary,
                    format: sourceFormat
                  }
                ]
              : []),
            ...(text
              ? [
                  {
                    type: 'reasoning.text',
                    text,
                    format: sourceFormat
                  }
                ]
              : []),
            ...(encryptedContent
              ? [
                  envelope?.kind === 'signature'
                    ? {
                        type: 'reasoning.text',
                        signature: encryptedContent,
                        ...(envelope?.id ? { id: envelope.id } : {}),
                        format: sourceFormat
                      }
                    : {
                        type: 'reasoning.encrypted',
                        data: encryptedContent,
                        ...(envelope?.id ? { id: envelope.id } : {}),
                        format: sourceFormat
                      }
                ]
              : [])
          ],
          ...(nativeItem ? { native_item: nativeItem } : {})
        }
      ],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  if (type && isGeminiInteractionsSignedBuiltInStepType(type)) {
    const signature = asString(item.signature);
    const envelope = signature ? decodeReasoningTransportEnvelope(signature) : undefined;
    const nativeItem = captureGeminiInteractionSignedStep(item, itemIndex, envelope);
    if (!nativeItem) {
      return null;
    }
    return {
      type: 'message',
      role: type.endsWith('_result') ? 'user' : 'assistant',
      content: [nativeItem],
      native_items: [nativeItem]
    };
  }

  if (type === 'model_output') {
    const content = normalizeGeminiInteractionContent(item.content);
    return content.length > 0
      ? {
          type: 'message',
          role: 'assistant',
          content
        }
      : null;
  }

  if (type === 'user_input') {
    const content = normalizeGeminiInteractionContent(item.content);
    return content.length > 0
      ? {
          type: 'message',
          role: 'user',
          content
        }
      : null;
  }

  if (type === 'text' || type === 'input_text' || type === 'output_text') {
    const text = asString(item.text);
    return text
      ? {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text }]
        }
      : null;
  }

  if (item.content !== undefined || item.role !== undefined) {
    const role = normalizeConversationRole(item.role);
    const content = normalizeGeminiInteractionContent(item.content);
    return content.length > 0
      ? {
          type: 'message',
          role,
          content
        }
      : null;
  }

  const serialized = stringifyUnknownInputItem(item);
  return serialized
    ? {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: serialized }]
      }
    : null;
}

function captureGeminiInteractionSignedStep(
  step: Record<string, unknown>,
  stepIndex: number,
  envelope: ReturnType<typeof decodeReasoningTransportEnvelope>
): ProviderNativeItem | undefined {
  if (envelope?.nativeItem) {
    return envelope.nativeItem;
  }
  const itemType = asString(step.type) || 'unknown';
  const signature = asString(step.signature);
  if (itemType === 'thought' && !signature) {
    return undefined;
  }
  const callId = asString(step.call_id) || asString(step.id);
  return {
    type: 'provider_native_item',
    item_type: itemType,
    ...(asString(step.id) ? { native_id: asString(step.id) } : {}),
    raw_payload: unwrapReasoningTransportCarriers(step),
    provider_schema_version: 'gemini-interactions-v1beta',
    item_origin: 'native',
    source_format: envelope?.format || GEMINI_INTERACTIONS_REASONING_FORMAT,
    source_origin: envelope?.origin || { provider: 'gemini', endpoint: 'unverified' },
    position: { turn: 0, step: stepIndex, item: 0 },
    ...(callId ? { group_id: callId, call_id: callId, pair_id: callId } : {}),
    capture_state:
      envelope?.carrierVersion === 2 && envelope.origin ? 'complete' : 'partial',
    ...(asString(step.text) ? { readable_text: asString(step.text) } : {}),
    ...(normalizeGeminiInteractionThoughtSummary(step.summary)
      ? { readable_summary: normalizeGeminiInteractionThoughtSummary(step.summary) }
      : {})
  };
}

function isGeminiInteractionsSignedBuiltInStepType(type: string): boolean {
  return /^(?:code_execution|file_search|google_maps|google_search|retrieval)_(?:call|result)$/.test(type);
}

function normalizeGeminiInteractionContent(content: unknown): StandardRequestInputContent[] {
  if (typeof content === 'string') {
    const text = content.trim();
    return text ? [{ type: 'input_text', text }] : [];
  }

  const items = Array.isArray(content) ? content : content !== undefined ? [content] : [];
  const normalized: StandardRequestInputContent[] = [];
  for (const item of items) {
    const text = extractTextFromPart(item);
    if (text) {
      normalized.push({
        type: 'input_text',
        text
      });
      continue;
    }

    if (isObject(item)) {
      const serialized = stringifyUnknownInputItem(item);
      if (serialized) {
        normalized.push({
          type: 'input_text',
          text: serialized
        });
      }
    }
  }

  return normalized;
}

function normalizeGeminiInteractionThoughtSummary(value: unknown): string {
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

function normalizeGeminiInteractionResultContent(result: unknown): string {
  if (typeof result === 'string') {
    return result;
  }

  if (Array.isArray(result)) {
    const text = result.map(extractTextFromPart).filter(Boolean).join('\n').trim();
    if (text) {
      return text;
    }
  }

  if (isObject(result)) {
    const text = extractTextFromPart(result);
    if (text) {
      return text;
    }
  }

  return normalizeToolResultContent(result);
}

function coalesceResponsesInputMessages(messages: StandardRequestInputMessage[]): StandardRequestInputMessage[] {
  const coalesced: StandardRequestInputMessage[] = [];
  let pendingAssistant: StandardRequestInputMessage | undefined;

  const flushPendingAssistant = () => {
    if (pendingAssistant) {
      coalesced.push(pendingAssistant);
      pendingAssistant = undefined;
    }
  };

  for (const message of messages) {
    if (message.role !== 'assistant') {
      flushPendingAssistant();
      coalesced.push(message);
      continue;
    }

    if (!pendingAssistant) {
      pendingAssistant = {
        ...message,
        content: [...message.content],
        ...(message.native_items ? { native_items: [...message.native_items] } : {})
      };
      continue;
    }

    if (shouldCoalesceResponsesAssistantMessages(pendingAssistant.content, message.content)) {
      pendingAssistant.content.push(...message.content);
      if (message.native_items) {
        pendingAssistant.native_items = [
          ...(pendingAssistant.native_items || []),
          ...message.native_items
        ];
      }
      continue;
    }

    flushPendingAssistant();
    pendingAssistant = {
      ...message,
      content: [...message.content],
      ...(message.native_items ? { native_items: [...message.native_items] } : {})
    };
  }

  flushPendingAssistant();
  return coalesced;
}

function shouldCoalesceResponsesAssistantMessages(
  left: StandardRequestInputContent[],
  right: StandardRequestInputContent[]
): boolean {
  return hasToolUseContent(left) || hasToolUseContent(right);
}

function hasToolUseContent(content: StandardRequestInputContent[]): boolean {
  return content.some(
    (item) => item.type === 'tool_use' || item.type === 'tool_search_call'
  );
}

function normalizeResponsesInputItem(
  item: unknown,
  itemIndex = 0,
  operation: 'create' | 'compact' = 'create',
  rawItemsByContent?: OpenAIResponsesRawInputProjectionMap
): StandardRequestInputMessage | null {
  if (!isObject(item)) {
    return null;
  }

  const type = asString(item.type);
  const nativeItem = captureOpenAIResponsesInputNativeItem(item, itemIndex, operation);

  const reasoningContent = normalizeOpenAIResponsesReasoningItem(item);
  if (reasoningContent) {
    if (nativeItem && reasoningContent.type === 'reasoning') {
      reasoningContent.native_item = nativeItem;
    }
    return {
      type: 'message',
      role: 'assistant',
      content: [reasoningContent],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }
  if (type === 'reasoning') {
    return null;
  }

  const toolSearchCallContent = normalizeOpenAIResponsesToolSearchCallItem(item);
  if (toolSearchCallContent) {
    rememberOpenAIResponsesRawInputProjection(
      rawItemsByContent,
      toolSearchCallContent,
      item,
      { turn: itemIndex, step: 0, item: itemIndex }
    );
    if (nativeItem && toolSearchCallContent.type === 'tool_search_call') {
      toolSearchCallContent.native_item = nativeItem;
    }
    return {
      type: 'message',
      role: 'assistant',
      content: [toolSearchCallContent],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  const toolSearchOutputContent = normalizeOpenAIResponsesToolSearchOutputItem(item);
  if (toolSearchOutputContent) {
    rememberOpenAIResponsesRawInputProjection(
      rawItemsByContent,
      toolSearchOutputContent,
      item,
      { turn: itemIndex, step: 0, item: itemIndex }
    );
    if (nativeItem && toolSearchOutputContent.type === 'tool_search_output') {
      toolSearchOutputContent.native_item = nativeItem;
    }
    return {
      type: 'message',
      role: 'user',
      content: [toolSearchOutputContent],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  const functionCallContent = normalizeOpenAIResponsesFunctionCallItem(item);
  if (functionCallContent) {
    rememberOpenAIResponsesRawInputProjection(
      rawItemsByContent,
      functionCallContent,
      item,
      { turn: itemIndex, step: 0, item: itemIndex }
    );
    if (nativeItem && functionCallContent.type === 'tool_use') {
      functionCallContent.native_item = nativeItem;
    }
    return {
      type: 'message',
      role: 'assistant',
      content: [functionCallContent],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  const functionCallOutputContent = normalizeOpenAIResponsesFunctionCallOutputItem(item);
  if (functionCallOutputContent) {
    rememberOpenAIResponsesRawInputProjection(
      rawItemsByContent,
      functionCallOutputContent,
      item,
      { turn: itemIndex, step: 0, item: itemIndex }
    );
    if (nativeItem && functionCallOutputContent.type === 'tool_result') {
      functionCallOutputContent.native_item = nativeItem;
    }
    return {
      type: 'message',
      role: 'user',
      content: [functionCallOutputContent],
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  if (type === 'message' || (!type && (item.role !== undefined || item.content !== undefined))) {
    const role = normalizeConversationRole(item.role);
    const content = normalizeOpenAIResponsesMessageContent(
      role,
      item.content,
      rawItemsByContent,
      itemIndex
    );
    if (content.length === 0) {
      return null;
    }

    return {
      type: 'message',
      ...(asString(item.id) ? { id: asString(item.id) } : {}),
      role,
      content,
      ...(asString(item.phase) ? { phase: asString(item.phase) } : {}),
      ...(asString(item.status) ? { status: asString(item.status) } : {}),
      ...(nativeItem ? { native_items: [nativeItem] } : {})
    };
  }

  if (type === 'input_text' || type === 'text') {
    const text = asString(item.text);
    if (!text) {
      return null;
    }

    return {
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text }]
    };
  }

  if (type === 'output_text') {
    const text = asString(item.text);
    if (!text) {
      return null;
    }

    return {
      type: 'message',
      role: 'assistant',
      content: [{ type: 'input_text', text }]
    };
  }

  if (nativeItem && type) {
    return {
      type: 'message',
      role: isOpenAIResponsesResultItemType(type) ? 'user' : 'assistant',
      content: [nativeItem],
      native_items: [nativeItem]
    };
  }

  const fallbackText = extractTextFromPart(item);
  if (fallbackText) {
    return {
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text: fallbackText }]
    };
  }

  const serialized = stringifyUnknownInputItem(item);
  if (!serialized) {
    return null;
  }

  return {
    type: 'message',
    role: 'user',
    content: [{ type: 'input_text', text: serialized }]
  };
}

function captureOpenAIResponsesInputNativeItem(
  item: Record<string, unknown>,
  itemIndex: number,
  operation: 'create' | 'compact'
): ProviderNativeItem | undefined {
  const itemType = asString(item.type);
  if (!itemType || itemType === 'input_text' || itemType === 'output_text' || itemType === 'text') {
    return undefined;
  }

  const envelope = findReasoningTransportEnvelope(item);
  if (envelope?.nativeItem) {
    return {
      ...envelope.nativeItem,
      item_origin: envelope.nativeItem.item_origin || 'native'
    };
  }

  const hasNativeDependency = Boolean(
    item.caller !== undefined ||
    item.fingerprint !== undefined ||
    item.program_id !== undefined ||
    item.depends_on !== undefined
  );
  const hasUnverifiedOpaqueReasoning = itemType === 'reasoning' && Boolean(asString(item.encrypted_content));
  const inherentlyNative = itemType === 'compaction' || itemType === 'program' ||
    itemType === 'program_output' || !new Set([
      'reasoning', 'message', 'function_call', 'function_call_output',
      'tool_search_call', 'tool_search_output'
    ]).has(itemType);
  if (!envelope && !hasNativeDependency && !hasUnverifiedOpaqueReasoning && !inherentlyNative) {
    return undefined;
  }

  const callId = asString(item.call_id) || asString(item.program_id);
  const caller = isObject(item.caller) ? item.caller : undefined;
  const callerId = asString(caller?.id) || asString(caller?.caller_id);
  const rawDependsOn = Array.isArray(item.depends_on)
    ? item.depends_on.map(asString).filter((value): value is string => Boolean(value))
    : [];
  const dependsOn = [
    ...rawDependsOn,
    ...(asString(item.program_id) ? [asString(item.program_id)!] : [])
  ];
  const readableText = itemType === 'message'
    ? normalizeOpenAIResponsesMessageContent('assistant', item.content)
        .map((content) => content.type === 'input_text' ? content.text : '')
        .filter(Boolean)
        .join('\n')
        .trim()
    : normalizeReasoningContentText(item.content) || asString(item.text);
  const readableSummary = normalizeReasoningSummaryText(item.summary);
  const sourceOrigin = envelope?.origin || {
    provider: 'openai',
    endpoint: 'unverified'
  };

  return {
    type: 'provider_native_item',
    item_type: itemType,
    ...(envelope?.id || asString(item.id)
      ? { native_id: envelope?.id || asString(item.id) }
      : {}),
    raw_payload: unwrapReasoningTransportCarriers(item),
    provider_schema_version: 'openai-responses-v1',
    item_origin: 'native',
    source_format: envelope?.format || OPENAI_RESPONSES_REASONING_FORMAT,
    source_origin: sourceOrigin,
    position: { turn: itemIndex, step: 0, item: itemIndex },
    ...(callerId || callId ? { group_id: callerId || callId } : {}),
    ...(callId ? { call_id: callId, pair_id: callId } : {}),
    ...(dependsOn.length > 0 ? { depends_on: [...new Set(dependsOn)] } : {}),
    capture_state: envelope?.carrierVersion === 2 && envelope.origin ? 'complete' : 'partial',
    ...(asString(item.status) ? { provider_status: asString(item.status) } : {}),
    ...(readableText ? { readable_text: readableText } : {}),
    ...(readableSummary ? { readable_summary: readableSummary } : {}),
    ...(itemType === 'compaction'
      ? { compaction_mode: operation === 'compact' ? 'standalone' : 'server_side' }
      : {})
  };
}

function findReasoningTransportEnvelope(
  value: unknown
): ReturnType<typeof decodeReasoningTransportEnvelope> {
  if (typeof value === 'string') {
    return decodeReasoningTransportEnvelope(value);
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      const envelope = findReasoningTransportEnvelope(entry);
      if (envelope) {
        return envelope;
      }
    }
    return undefined;
  }
  if (!isObject(value)) {
    return undefined;
  }
  for (const entry of Object.values(value)) {
    const envelope = findReasoningTransportEnvelope(entry);
    if (envelope) {
      return envelope;
    }
  }
  return undefined;
}

function unwrapReasoningTransportCarriers(value: unknown): Record<string, unknown> {
  const unwrap = (candidate: unknown): unknown => {
    if (typeof candidate === 'string') {
      return decodeReasoningTransportEnvelope(candidate)?.data || candidate;
    }
    if (Array.isArray(candidate)) {
      return candidate.map(unwrap);
    }
    if (!isObject(candidate)) {
      return candidate;
    }
    return Object.fromEntries(Object.entries(candidate).map(([key, entry]) => [key, unwrap(entry)]));
  };
  return unwrap(value) as Record<string, unknown>;
}

function isOpenAIResponsesResultItemType(type: string): boolean {
  return type.endsWith('_output') || type.endsWith('_result') || type === 'function_call_output';
}

function normalizeOpenAIResponsesToolSearchCallItem(
  item: Record<string, unknown>
): StandardRequestInputContent | null {
  if (
    asString(item.type) !== 'tool_search_call' ||
    (item.execution !== undefined && item.execution !== 'client')
  ) {
    return null;
  }

  const callId = asString(item.call_id);
  if (!callId) {
    return null;
  }

  return {
    type: 'tool_search_call',
    execution: 'client',
    call_id: callId,
    status: asString(item.status),
    arguments: normalizeFunctionArgumentsInput(item.arguments)
  };
}

function normalizeOpenAIResponsesToolSearchOutputItem(
  item: Record<string, unknown>
): StandardRequestInputContent | null {
  if (
    asString(item.type) !== 'tool_search_output' ||
    (item.execution !== undefined && item.execution !== 'client')
  ) {
    return null;
  }

  const callId = asString(item.call_id);
  if (!callId || !Array.isArray(item.tools)) {
    return null;
  }

  return {
    type: 'tool_search_output',
    execution: 'client',
    call_id: callId,
    status: asString(item.status),
    tools: item.tools
  };
}

function normalizeOpenAIResponsesMessageContent(
  role: 'user' | 'assistant',
  content: unknown,
  rawItemsByContent?: OpenAIResponsesRawInputProjectionMap,
  turnIndex = 0
): StandardRequestInputContent[] {
  const normalized: StandardRequestInputContent[] = [];

  if (typeof content === 'string') {
    const text = content.trim();
    if (text) {
      normalized.push({ type: 'input_text', text });
    }
    return normalized;
  }

  const blocks = Array.isArray(content) ? content : [content];
  for (let blockIndex = 0; blockIndex < blocks.length; blockIndex += 1) {
    const block = blocks[blockIndex];
    if (typeof block === 'string') {
      const text = block.trim();
      if (text) {
        normalized.push({ type: 'input_text', text });
      }
      continue;
    }

    if (!isObject(block)) {
      continue;
    }

    if (role === 'assistant') {
      const reasoningContent = normalizeOpenAIResponsesReasoningItem(block);
      if (reasoningContent) {
        normalized.push(reasoningContent);
        continue;
      }

      const functionCallContent = normalizeOpenAIResponsesFunctionCallItem(block);
      if (functionCallContent) {
        rememberOpenAIResponsesRawInputProjection(
          rawItemsByContent,
          functionCallContent,
          block,
          { turn: turnIndex, step: 0, item: blockIndex }
        );
        normalized.push(functionCallContent);
        continue;
      }
    }

    if (role === 'user') {
      const functionCallOutputContent = normalizeOpenAIResponsesFunctionCallOutputItem(block);
      if (functionCallOutputContent) {
        rememberOpenAIResponsesRawInputProjection(
          rawItemsByContent,
          functionCallOutputContent,
          block,
          { turn: turnIndex, step: 0, item: blockIndex }
        );
        normalized.push(functionCallOutputContent);
        continue;
      }
    }

    const text = extractTextFromPart(block);
    if (text) {
      normalized.push({ type: 'input_text', text });
    }
  }

  return normalized;
}

function normalizeOpenAIResponsesFunctionCallItem(item: Record<string, unknown>): StandardRequestInputContent | null {
  const type = asString(item.type);
  if (type && type !== 'function_call') {
    return null;
  }

  const name = normalizeNamespacedToolName(asString(item.name), asString(item.namespace));
  if (!name) {
    return null;
  }

  const id = asString(item.call_id) || asString(item.id) || `openai_call_${name}`;
  const input = normalizeFunctionArgumentsInput(item.arguments ?? item.input);

  return {
    type: 'tool_use',
    id,
    ...(asString(item.id) ? { native_id: asString(item.id) } : {}),
    name,
    input,
    ...(asString(item.status) ? { status: asString(item.status) } : {}),
    ...(isObject(item.caller) ? { caller: item.caller } : {})
  };
}

function normalizeOpenAIResponsesReasoningItem(item: Record<string, unknown>): StandardRequestInputContent | null {
  if (asString(item.type) !== 'reasoning') {
    return null;
  }

  const summary = normalizeReasoningSummaryText(item.summary);
  const text = normalizeReasoningContentText(item.content) || asString(item.text);
  const rawEncryptedContent = asString(item.encrypted_content);
  const envelope = rawEncryptedContent
    ? decodeReasoningTransportEnvelope(rawEncryptedContent)
    : undefined;
  const encryptedContent = envelope?.data || rawEncryptedContent;
  const id = envelope?.id || asString(item.id);
  const reasoning: StandardRequestInputContent = {
    type: 'reasoning',
    source_format: envelope?.format || OPENAI_RESPONSES_REASONING_FORMAT,
    ...(asString(item.status) ? { status: asString(item.status) } : {}),
    ...(envelope?.origin ? { source_origin: envelope.origin } : {})
  };

  if (id) {
    reasoning.id = id;
  }
  if (text) {
    reasoning.text = text;
  }
  if (summary) {
    reasoning.summary = summary;
  }
  if (encryptedContent) {
    reasoning.encrypted_content = encryptedContent;
  }

  if (!reasoning.text && !reasoning.summary && !reasoning.encrypted_content) {
    return null;
  }

  reasoning.reasoning_details =
    envelope?.kind === 'signature'
      ? [
          {
            type: 'reasoning.text',
            ...(text ? { text } : {}),
            signature: encryptedContent,
            ...(id ? { id } : {}),
            format: reasoning.source_format
          }
        ]
      : buildReasoningDetailsForChat(reasoning);
  return reasoning;
}

function normalizeReasoningSummaryText(value: unknown): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const parts = value
    .map((item) => {
      if (typeof item === 'string') {
        return item.trim();
      }
      return isObject(item) ? asString(item.text) || asString(item.summary) || '' : '';
    })
    .filter(Boolean);

  return parts.join('\n').trim() || undefined;
}

function normalizeReasoningContentText(value: unknown): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const parts = value
    .map((item) => {
      if (typeof item === 'string') {
        return item.trim();
      }
      return isObject(item)
        ? asString(item.text) || asString(item.reasoning) || asString(item.thinking) || ''
        : '';
    })
    .filter(Boolean);

  return parts.join('\n').trim() || undefined;
}

function buildReasoningDetailsForChat(reasoning: StandardRequestInputContent & { type: 'reasoning' }): unknown[] {
  const details: unknown[] = [];
  const metadata = {
    format: reasoning.source_format || OPENAI_RESPONSES_REASONING_FORMAT
  };
  if (reasoning.summary) {
    details.push({
      type: 'reasoning.summary',
      summary: reasoning.summary,
      ...metadata,
      index: details.length
    });
  }
  if (reasoning.text) {
    details.push({
      type: 'reasoning.text',
      text: reasoning.text,
      ...metadata,
      index: details.length
    });
  }
  if (reasoning.encrypted_content) {
    details.push({
      type: 'reasoning.encrypted',
      data: reasoning.encrypted_content,
      ...metadata,
      ...(reasoning.id ? { id: reasoning.id } : {}),
      index: details.length
    });
  }
  return details;
}

function normalizeOpenAIResponsesFunctionCallOutputItem(item: Record<string, unknown>): StandardRequestInputContent | null {
  const type = asString(item.type);
  if (type && type !== 'function_call_output') {
    return null;
  }

  const toolUseId = asString(item.call_id) || asString(item.tool_call_id) || asString(item.id);
  if (!toolUseId) {
    return null;
  }

  const toolResult: StandardRequestInputContent = {
    type: 'tool_result',
    tool_use_id: toolUseId,
    ...(asString(item.id) ? { native_id: asString(item.id) } : {}),
    content: normalizeToolResultContent(item.output ?? item.content ?? item.result),
    ...(asString(item.status) ? { status: asString(item.status) } : {}),
    ...(isObject(item.caller) ? { caller: item.caller } : {})
  };
  const isError = asBoolean(item.is_error);
  if (isError !== undefined) {
    toolResult.is_error = isError;
  }

  return toolResult;
}

function normalizeFunctionArgumentsInput(value: unknown): unknown {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return {};
    }

    try {
      return JSON.parse(trimmed) as unknown;
    } catch {
      return value;
    }
  }

  if (value === undefined) {
    return {};
  }

  return value;
}

function normalizeToolResultContent(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (isObject(value) || Array.isArray(value)) {
    try {
      return JSON.stringify(value);
    } catch {
      return '';
    }
  }

  return '';
}

function stringifyUnknownInputItem(item: Record<string, unknown>): string | undefined {
  try {
    const serialized = JSON.stringify(item);
    if (!serialized || serialized === '{}' || serialized === '[]') {
      return undefined;
    }
    return serialized;
  } catch {
    return undefined;
  }
}

function extractMessageText(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim();
  }

  if (isObject(content) && typeof content.text === 'string') {
    return content.text.trim();
  }

  if (!Array.isArray(content)) {
    return '';
  }

  return content.map(extractTextFromPart).filter(Boolean).join('\n').trim();
}

function extractOpenAIChatMessageContent(message: Record<string, unknown>): StandardRequestInputContent[] {
  const rawRole = asString(message.role)?.trim().toLowerCase();
  if (rawRole === 'tool') {
    return normalizeOpenAIChatToolResultMessage(message);
  }

  const normalized: StandardRequestInputContent[] = [];
  const text = extractMessageText(message.content);
  if (normalizeMessageRole(message.role) !== 'assistant') {
    if (text) {
      normalized.push({ type: 'input_text', text });
    }
    return normalized;
  }

  normalized.push(...normalizeOpenAIChatAssistantReasoning(message));
  if (text) {
    normalized.push({ type: 'input_text', text });
  }
  normalized.push(...normalizeOpenAIChatAssistantToolCalls(message.tool_calls));

  const legacyFunctionCall = normalizeOpenAIChatAssistantFunctionCall(message.function_call);
  if (legacyFunctionCall) {
    normalized.push(legacyFunctionCall);
  }

  return normalized;
}

function normalizeOpenAIChatAssistantReasoning(
  message: Record<string, unknown>
): StandardRequestInputContent[] {
  const text =
    asString(message.reasoning_content) ||
    asString(message.reasoning) ||
    asString(message.thinking);
  const indexedDetailGroups = groupOpenAIChatReasoningDetailsByIndex(
    message.reasoning_details
  );
  if (indexedDetailGroups) {
    const reasoningItems = indexedDetailGroups
      .map((group) =>
        buildOpenAIChatAssistantReasoning(
          undefined,
          normalizeOpenAIChatReasoningDetails(group)
        )
      )
      .filter((item): item is StandardRequestInputContent => Boolean(item));
    if (reasoningItems.length > 0) {
      return reasoningItems;
    }
  }

  const details = normalizeOpenAIChatReasoningDetails(message.reasoning_details);
  const reasoning = buildOpenAIChatAssistantReasoning(text, details);
  return reasoning ? [reasoning] : [];
}

interface NormalizedOpenAIChatReasoningDetails {
  id?: string;
  sourceFormat?: string;
  sourceOrigin?: NonNullable<Extract<StandardRequestInputContent, { type: 'reasoning' }>['source_origin']>;
  text?: string;
  summary?: string;
  encryptedContent?: string;
  rawDetails: unknown[];
}

function buildOpenAIChatAssistantReasoning(
  text: string | undefined,
  details: NormalizedOpenAIChatReasoningDetails
): StandardRequestInputContent | null {
  if (!text && !details.text && !details.summary && !details.encryptedContent && details.rawDetails.length === 0) {
    return null;
  }

  const reasoning: StandardRequestInputContent = {
    type: 'reasoning'
  };
  if (details.id) {
    reasoning.id = details.id;
  }
  if (details.sourceFormat) {
    reasoning.source_format = details.sourceFormat;
  }
  if (details.sourceOrigin) {
    reasoning.source_origin = details.sourceOrigin;
  }
  const mergedText = mergeDistinctReasoningText(details.text, text);
  if (mergedText) {
    reasoning.text = mergedText;
  }
  if (details.summary) {
    reasoning.summary = details.summary;
  }
  if (details.encryptedContent) {
    reasoning.encrypted_content = details.encryptedContent;
  }
  reasoning.reasoning_details =
    details.rawDetails.length > 0
      ? details.rawDetails
      : buildReasoningDetailsForChat(reasoning as StandardRequestInputContent & { type: 'reasoning' });

  return reasoning;
}

function groupOpenAIChatReasoningDetailsByIndex(value: unknown): unknown[][] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const groups = new Map<number, unknown[]>();
  for (const detail of value) {
    if (!isObject(detail)) {
      return undefined;
    }

    const index = asNumber(detail.index);
    if (index === undefined) {
      return undefined;
    }

    const group = groups.get(index);
    if (group) {
      group.push(detail);
    } else {
      groups.set(index, [detail]);
    }
  }

  return groups.size > 1 ? [...groups.values()] : undefined;
}

function normalizeOpenAIChatReasoningDetails(
  value: unknown
): NormalizedOpenAIChatReasoningDetails {
  const normalized: NormalizedOpenAIChatReasoningDetails = {
    rawDetails: []
  };

  if (!Array.isArray(value)) {
    return normalized;
  }

  const textParts: string[] = [];
  const summaryParts: string[] = [];
  for (const detail of value) {
    if (typeof detail === 'string') {
      normalized.rawDetails.push(detail);
      if (detail) {
        textParts.push(detail);
      }
      continue;
    }

    if (!isObject(detail)) {
      normalized.rawDetails.push(detail);
      continue;
    }

    const type = asString(detail.type);
    const rawSignature =
      asString(detail.signature) ||
      asString(detail.thoughtSignature) ||
      asString(detail.thought_signature);
    const rawEncryptedContent = asString(detail.encrypted_content) || asString(detail.data);
    const envelope = rawSignature || rawEncryptedContent
      ? decodeReasoningTransportEnvelope(rawSignature || rawEncryptedContent || '')
      : undefined;
    const opaqueContent = envelope?.data || rawSignature || rawEncryptedContent;
    const id = envelope?.id || asString(detail.id);
    const format = envelope?.format || asString(detail.format);
    const summary = asString(detail.summary);
    const text = asString(detail.text) || asString(detail.reasoning) || asString(detail.thinking);
    const normalizedDetail: Record<string, unknown> = {
      ...detail,
      ...(format ? { format } : {}),
      ...(id ? { id } : {})
    };
    if (envelope) {
      delete normalizedDetail.thoughtSignature;
      delete normalizedDetail.thought_signature;
      delete normalizedDetail.encrypted_content;
      if (envelope.kind === 'signature' || rawSignature) {
        delete normalizedDetail.data;
        normalizedDetail.signature = envelope.data;
      } else {
        delete normalizedDetail.signature;
        normalizedDetail.data = envelope.data;
      }
    }
    normalized.rawDetails.push(normalizedDetail);
    if (id && !normalized.id) {
      normalized.id = id;
    }
    if (format && !normalized.sourceFormat) {
      normalized.sourceFormat = format;
    }
    if (envelope?.origin && !normalized.sourceOrigin) {
      normalized.sourceOrigin = envelope.origin;
    }

    if (type === 'reasoning.summary' || (summary && !text)) {
      if (summary || text) {
        summaryParts.push(summary || text || '');
      }
      continue;
    }

    if (text) {
      textParts.push(text);
    }
    if (opaqueContent && !normalized.encryptedContent) {
      normalized.encryptedContent = opaqueContent;
    }
  }

  normalized.text = textParts.join('\n').trim() || undefined;
  normalized.summary = summaryParts.join('\n').trim() || undefined;
  return normalized;
}

function mergeDistinctReasoningText(...values: Array<string | undefined>): string | undefined {
  const parts: string[] = [];
  for (const value of values) {
    const text = value?.trim();
    if (!text) {
      continue;
    }

    if (parts.some((part) => part === text)) {
      continue;
    }

    parts.push(text);
  }

  return parts.join('\n').trim() || undefined;
}

function normalizeOpenAIChatToolResultMessage(message: Record<string, unknown>): StandardRequestInputContent[] {
  const rawToolUseId = asString(message.tool_call_id) || asString(message.id);
  if (!rawToolUseId) {
    return [];
  }
  const toolUseId = decodeGeminiThoughtSignatureToolCallId(rawToolUseId)?.toolCallId || rawToolUseId;

  const content = normalizeToolResultContent(message.content);
  if (!content) {
    return [];
  }

  const toolResult: StandardRequestInputContent = {
    type: 'tool_result',
    tool_use_id: toolUseId,
    content
  };
  const isError = asBoolean(message.is_error);
  if (isError !== undefined) {
    toolResult.is_error = isError;
  }

  return [toolResult];
}

function normalizeOpenAIChatAssistantToolCalls(toolCalls: unknown): StandardRequestInputContent[] {
  if (!Array.isArray(toolCalls)) {
    return [];
  }

  const normalized: StandardRequestInputContent[] = [];
  for (const toolCall of toolCalls) {
    if (!isObject(toolCall)) {
      continue;
    }

    const functionPayload = isObject(toolCall.function) ? toolCall.function : undefined;
    const name = asString(functionPayload?.name) || asString(toolCall.name);
    if (!name) {
      continue;
    }

    const extraContent = isObject(toolCall.extra_content) ? toolCall.extra_content : undefined;
    const googleExtra = isObject(extraContent?.google) ? extraContent.google : undefined;
    const rawThoughtSignature =
      asString(googleExtra?.thought_signature) ||
      asString(googleExtra?.thoughtSignature);
    const rawToolCallId = asString(toolCall.id) || `chatcmpl_call_${name}`;
    const toolCallIdCarrier = decodeGeminiThoughtSignatureToolCallId(rawToolCallId);
    const extensionEnvelope = rawThoughtSignature
      ? decodeReasoningTransportEnvelope(rawThoughtSignature)
      : undefined;
    const envelope =
      extensionEnvelope ||
      (!rawThoughtSignature || rawThoughtSignature === toolCallIdCarrier?.envelope.data
        ? toolCallIdCarrier?.envelope
        : undefined);
    const thoughtSignature = envelope?.data || rawThoughtSignature;
    const thoughtSignatureFormat =
      envelope?.format || (thoughtSignature ? GEMINI_GENERATE_CONTENT_REASONING_FORMAT : undefined);
    if (
      thoughtSignature &&
      thoughtSignatureFormat &&
      thoughtSignatureFormat !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT
    ) {
      normalized.push({
        type: 'reasoning',
        ...(envelope?.id ? { id: envelope.id } : {}),
        source_format: thoughtSignatureFormat,
        ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
        encrypted_content: thoughtSignature,
        reasoning_details: [
          envelope?.kind === 'signature'
            ? {
                type: 'reasoning.text',
                signature: thoughtSignature,
                ...(envelope?.id ? { id: envelope.id } : {}),
                format: thoughtSignatureFormat
              }
            : {
                type: 'reasoning.encrypted',
                data: thoughtSignature,
                ...(envelope?.id ? { id: envelope.id } : {}),
                format: thoughtSignatureFormat
              }
        ]
      });
    }

    normalized.push({
      type: 'tool_use',
      id: toolCallIdCarrier?.toolCallId || rawToolCallId,
      name,
      input: normalizeFunctionArgumentsInput(functionPayload?.arguments ?? toolCall.arguments ?? toolCall.input),
      ...(thoughtSignature && thoughtSignatureFormat === GEMINI_GENERATE_CONTENT_REASONING_FORMAT
        ? {
            thought_signature: thoughtSignature,
            thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
            ...(envelope?.origin ? { thought_signature_origin: envelope.origin } : {})
          }
        : {})
    });
  }

  return normalized;
}

function normalizeOpenAIChatAssistantFunctionCall(value: unknown): StandardRequestInputContent | null {
  if (!isObject(value)) {
    return null;
  }

  const name = asString(value.name);
  if (!name) {
    return null;
  }

  return {
    type: 'tool_use',
    id: asString(value.call_id) || asString(value.id) || `chatcmpl_call_${name}`,
    name,
    input: normalizeFunctionArgumentsInput(value.arguments ?? value.input)
  };
}

function extractAnthropicSystem(system: unknown): string | undefined {
  if (typeof system === 'string') {
    return system.trim() || undefined;
  }

  if (!Array.isArray(system)) {
    return undefined;
  }

  const value = system.map(extractTextFromPart).filter(Boolean).join('\n').trim();
  return value || undefined;
}

function extractAnthropicMessageContent(
  role: 'user' | 'assistant',
  content: unknown,
  messageIndex: number,
  sourceModel: string | undefined,
  providerMode: string
): StandardRequestInputContent[] {
  const normalized: StandardRequestInputContent[] = [];
  let pendingForeignNativeCall: ProviderNativeItem | undefined;

  if (typeof content === 'string') {
    const text = content.trim();
    if (text) {
      normalized.push({ type: 'input_text', text });
    }

    return normalized;
  }

  const blocks = Array.isArray(content) ? content : [content];
  for (const block of blocks) {
    if (typeof block === 'string') {
      const text = block.trim();
      if (text) {
        normalized.push({ type: 'input_text', text });
      }
      continue;
    }

    if (!isObject(block)) {
      continue;
    }

    const blockType = asString(block.type);
    const reasoning = normalizeAnthropicThinkingBlock(
      block,
      role,
      normalized.length,
      messageIndex,
      sourceModel,
      providerMode
    );
    if (reasoning) {
      if (
        reasoning.type === 'reasoning' &&
        reasoning.native_item &&
        reasoning.native_item.source_format !== ANTHROPIC_CLAUDE_REASONING_FORMAT &&
        reasoning.native_item.item_type === 'function_call'
      ) {
        pendingForeignNativeCall = reasoning.native_item;
        delete reasoning.native_item;
      }
      normalized.push(reasoning);
      continue;
    }

    if (blockType === 'tool_use' && role === 'assistant') {
      const id = asString(block.id);
      const name = asString(block.name);
      if (!id || !name) {
        continue;
      }

      const toolUse: StandardRequestInputContent = {
        type: 'tool_use',
        id,
        name,
        input: block.input ?? {},
        ...(pendingForeignNativeCall ? { native_item: pendingForeignNativeCall } : {})
      };
      pendingForeignNativeCall = undefined;
      const thoughtSignature = readThoughtSignature(block);
      if (thoughtSignature) {
        toolUse.thought_signature = thoughtSignature;
      }
      normalized.push(toolUse);
      continue;
    }

    if (blockType === 'tool_result' && role === 'user') {
      const toolUseId = asString(block.tool_use_id) || asString(block.tool_call_id) || asString(block.id);
      if (!toolUseId) {
        continue;
      }

      const toolResult: StandardRequestInputContent = {
        type: 'tool_result',
        tool_use_id: toolUseId,
        content: normalizeAnthropicToolResultContent(block.content)
      };
      const toolReferences = extractAnthropicToolReferences(block.content);
      if (toolReferences.length > 0) {
        toolResult.tool_references = toolReferences;
      }
      const isError = asBoolean(block.is_error);
      if (isError !== undefined) {
        toolResult.is_error = isError;
      }
      normalized.push(toolResult);
      continue;
    }

    const text = extractTextFromPart(block);
    if (text) {
      normalized.push({ type: 'input_text', text });
    }
  }

  return normalized;
}

function normalizeAnthropicThinkingBlock(
  block: Record<string, unknown>,
  role: 'user' | 'assistant',
  index: number,
  messageIndex: number,
  sourceModel: string | undefined,
  providerMode: string
): StandardRequestInputContent | null {
  if (role !== 'assistant') {
    return null;
  }

  const blockType = asString(block.type);
  if (blockType === 'thinking') {
    const thinking = asString(block.thinking) || asString(block.text);
    const rawSignature = asString(block.signature);
    const envelope = rawSignature
      ? decodeReasoningTransportEnvelope(rawSignature)
      : undefined;
    const signature = envelope?.data || rawSignature;
    const sourceFormat = envelope?.format || ANTHROPIC_CLAUDE_REASONING_FORMAT;
    if (!thinking && !rawSignature) {
      return null;
    }

    const detail: Record<string, unknown> = {
      type: 'reasoning.text',
      format: sourceFormat,
      index
    };
    if (thinking) {
      detail.text = thinking;
    }
    if (signature) {
      detail.signature = signature;
    }

    const reasoning: StandardRequestInputContent = {
      type: 'reasoning',
      ...(envelope?.id ? { id: envelope.id } : {}),
      source_format: sourceFormat,
      ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
      reasoning_details: [detail]
    };
    const nativeItem = envelope?.nativeItem ||
      (shouldCaptureAnthropicNativeBlock(envelope)
        ? captureAnthropicNativeThinkingBlock({
            block,
            itemType: 'thinking',
            messageIndex,
            itemIndex: index,
            sourceModel,
            providerMode,
            envelope
          })
        : undefined);
    if (nativeItem) {
      reasoning.native_item = nativeItem;
    }
    if (thinking) {
      reasoning.text = thinking;
    }
    return reasoning;
  }

  if (blockType === 'redacted_thinking') {
    const data = asString(block.data);
    if (!data) {
      return null;
    }

    const envelope = decodeReasoningTransportEnvelope(data);
    const encryptedContent = envelope?.data || data;
    const sourceFormat = envelope?.format || ANTHROPIC_CLAUDE_REASONING_FORMAT;
    const nativeItem = envelope?.nativeItem ||
      (shouldCaptureAnthropicNativeBlock(envelope)
        ? captureAnthropicNativeThinkingBlock({
            block,
            itemType: 'redacted_thinking',
            messageIndex,
            itemIndex: index,
            sourceModel,
            providerMode,
            envelope
          })
        : undefined);
    return {
      type: 'reasoning',
      ...(envelope?.id ? { id: envelope.id } : {}),
      source_format: sourceFormat,
      ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
      encrypted_content: encryptedContent,
      reasoning_details: [
        {
          type: 'reasoning.encrypted',
          data: encryptedContent,
          ...(envelope?.id ? { id: envelope.id } : {}),
          format: sourceFormat,
          index
        }
      ],
      ...(nativeItem ? { native_item: nativeItem } : {})
    };
  }

  return null;
}

function shouldCaptureAnthropicNativeBlock(
  envelope: ReturnType<typeof decodeReasoningTransportEnvelope>
): boolean {
  return !envelope || envelope.format === ANTHROPIC_CLAUDE_REASONING_FORMAT;
}

function captureAnthropicNativeThinkingBlock(options: {
  block: Record<string, unknown>;
  itemType: 'thinking' | 'redacted_thinking';
  messageIndex: number;
  itemIndex: number;
  sourceModel: string | undefined;
  providerMode: string;
  envelope: ReturnType<typeof decodeReasoningTransportEnvelope>;
}): ProviderNativeItem {
  const sourceOrigin = options.envelope?.origin || {
    provider: 'anthropic',
    endpoint: 'unverified',
    ...(options.sourceModel ? { model: options.sourceModel } : {})
  };
  const rawPayload = unwrapReasoningTransportCarriers(options.block);
  const readableText = typeof rawPayload.thinking === 'string'
    ? rawPayload.thinking
    : undefined;
  return {
    type: 'provider_native_item',
    item_type: options.itemType,
    ...(asString(rawPayload.id) ? { native_id: asString(rawPayload.id) } : {}),
    raw_payload: rawPayload,
    provider_schema_version: 'anthropic-messages-2023-06-01',
    item_origin: 'native',
    source_format: options.envelope?.format || ANTHROPIC_CLAUDE_REASONING_FORMAT,
    source_origin: sourceOrigin,
    position: { turn: options.messageIndex, step: 0, item: options.itemIndex },
    capture_state:
      options.envelope?.carrierVersion === 2 && options.envelope.origin
        ? 'complete'
        : 'partial',
    provider_mode: options.providerMode,
    ...(readableText !== undefined ? { readable_text: readableText } : {})
  };
}

function normalizeAnthropicToolResultContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    const residualContent = content.filter(
      (item) => !isObject(item) || asString(item.type) !== 'tool_reference'
    );
    const text = residualContent.map(extractTextFromPart).filter(Boolean).join('\n').trim();
    if (text) {
      return text;
    }

    if (residualContent.length === 0) {
      return '';
    }

    try {
      return JSON.stringify(residualContent);
    } catch {
      return '';
    }
  }

  if (isObject(content)) {
    const text = extractTextFromPart(content);
    if (text) {
      return text;
    }

    try {
      return JSON.stringify(content);
    } catch {
      return '';
    }
  }

  if (typeof content === 'number' || typeof content === 'boolean') {
    return String(content);
  }

  return '';
}

function collectNativeItemsFromInputContent(
  content: StandardRequestInputContent[]
): Pick<StandardRequestInputMessage, 'native_items'> | Record<string, never> {
  const nativeItems = content.flatMap((item) => {
    if (item.type === 'provider_native_item') {
      return [item];
    }
    return 'native_item' in item && item.native_item ? [item.native_item] : [];
  });
  return nativeItems.length > 0 ? { native_items: [...new Set(nativeItems)] } : {};
}

function linkOpenAIResponsesNativeToolGroups(
  messages: StandardRequestInputMessage[],
  rawItemsByContent: OpenAIResponsesRawInputProjectionMap
): void {
  const nativeCallsById = new Map<string, ProviderNativeItem>();

  for (let messageIndex = 0; messageIndex < messages.length; messageIndex += 1) {
    const message = messages[messageIndex]!;
    attachDetachedOpenAIResponsesNativeItems(message);

    if (message.role === 'assistant') {
      const reasoningItems = collectOpenAIResponsesReasoningNativeItems(message);
      const primaryReasoning = reasoningItems[0];
      const linkedReasoningItems = primaryReasoning
        ? reasoningItems.filter((item) =>
            haveSameReasoningStateOrigin(item, primaryReasoning)
          )
        : [];

      for (let reasoningIndex = 0; reasoningIndex < linkedReasoningItems.length; reasoningIndex += 1) {
        const reasoning = linkedReasoningItems[reasoningIndex]!;
        reasoning.native_id ||= `responses-reasoning-${reasoning.position.turn}-${reasoning.position.step}-${reasoning.position.item}-${reasoningIndex}`;
      }

      const existingCall = message.content
        .map(readOpenAIResponsesNativeItemFromContent)
        .find((item) =>
          item?.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
          (!primaryReasoning || haveSameReasoningStateOrigin(item, primaryReasoning)) &&
          isOpenAIResponsesCallNativeItem(item)
        );
      const sharedGroupId = primaryReasoning
        ? primaryReasoning.group_id || existingCall?.group_id ||
          `responses-turn-${primaryReasoning.native_id}`
        : undefined;

      if (sharedGroupId) {
        for (const reasoning of linkedReasoningItems) {
          reasoning.group_id = sharedGroupId;
        }
      }

      const reasoningDependencies = linkedReasoningItems
        .map((item) => item.native_id)
        .filter((id): id is string => Boolean(id));
      const reasoningCaptureState = combineProviderNativeCaptureStates(
        linkedReasoningItems.map((item) => item.capture_state)
      );

      for (let contentIndex = 0; contentIndex < message.content.length; contentIndex += 1) {
        const content = message.content[contentIndex]!;
        if (content.type !== 'tool_use' && content.type !== 'tool_search_call') {
          continue;
        }

        const callId = content.type === 'tool_use' ? content.id : content.call_id;
        let nativeItem = content.native_item;
        if (nativeItem && nativeItem.source_format !== OPENAI_RESPONSES_REASONING_FORMAT) {
          continue;
        }

        const canLinkReasoning = Boolean(
          primaryReasoning &&
          (!nativeItem || haveSameReasoningStateOrigin(nativeItem, primaryReasoning))
        );
        if (!nativeItem && !canLinkReasoning) {
          continue;
        }

        const projection = rawItemsByContent.get(content);
        const rawPayload = projection?.rawPayload ||
          buildFallbackOpenAIResponsesCallPayload(content);
        const itemType = asString(rawPayload.type) ||
          (content.type === 'tool_use' ? 'function_call' : 'tool_search_call');
        const groupId = canLinkReasoning
          ? sharedGroupId!
          : nativeItem?.group_id || callId;
        const dependencies = [
          ...(nativeItem?.depends_on || []),
          ...(canLinkReasoning ? reasoningDependencies : [])
        ];
        const captureState = combineProviderNativeCaptureStates([
          ...(nativeItem ? [nativeItem.capture_state] : []),
          ...(canLinkReasoning ? [reasoningCaptureState] : [])
        ]);

        if (!nativeItem) {
          nativeItem = {
            type: 'provider_native_item',
            item_type: itemType,
            native_id: asString(rawPayload.id) ||
              `responses-${itemType}-${callId}`,
            raw_payload: rawPayload,
            provider_schema_version: primaryReasoning!.provider_schema_version,
            item_origin: 'native',
            source_format: OPENAI_RESPONSES_REASONING_FORMAT,
            source_origin: primaryReasoning!.source_origin,
            position: projection?.position || {
              turn: messageIndex,
              step: 0,
              item: contentIndex
            },
            group_id: groupId,
            call_id: callId,
            pair_id: callId,
            ...(dependencies.length > 0
              ? { depends_on: [...new Set(dependencies)] }
              : {}),
            capture_state: captureState,
            ...(content.status ? { provider_status: content.status } : {})
          };
          content.native_item = nativeItem;
        } else {
          nativeItem.item_origin ||= 'native';
          nativeItem.native_id ||= asString(nativeItem.raw_payload.id) ||
            `responses-${nativeItem.item_type}-${callId}`;
          nativeItem.group_id = groupId;
          nativeItem.call_id = callId;
          nativeItem.pair_id = callId;
          nativeItem.capture_state = captureState;
          nativeItem.depends_on = dependencies.length > 0
            ? [...new Set(dependencies)]
            : undefined;
        }
        nativeCallsById.set(callId, nativeItem);
      }
    } else {
      for (let contentIndex = 0; contentIndex < message.content.length; contentIndex += 1) {
        const content = message.content[contentIndex]!;
        if (content.type !== 'tool_result' && content.type !== 'tool_search_output') {
          continue;
        }

        const callId = content.type === 'tool_result'
          ? content.tool_use_id
          : content.call_id;
        const nativeCall = nativeCallsById.get(callId);
        if (!nativeCall) {
          continue;
        }

        let nativeItem = content.native_item;
        if (nativeItem && nativeItem.source_format !== OPENAI_RESPONSES_REASONING_FORMAT) {
          continue;
        }
        if (
          nativeItem &&
          !haveSameReasoningStateOrigin(nativeItem, nativeCall)
        ) {
          continue;
        }

        const projection = rawItemsByContent.get(content);
        const rawPayload = projection?.rawPayload ||
          buildFallbackOpenAIResponsesResultPayload(content);
        const itemType = asString(rawPayload.type) ||
          (content.type === 'tool_result'
            ? 'function_call_output'
            : 'tool_search_output');
        const callNativeId = nativeCall.native_id || callId;
        const dependencies = [
          ...(nativeItem?.depends_on || []),
          callNativeId
        ];
        const captureState = combineProviderNativeCaptureStates([
          nativeCall.capture_state,
          ...(nativeItem ? [nativeItem.capture_state] : [])
        ]);

        if (!nativeItem) {
          nativeItem = {
            type: 'provider_native_item',
            item_type: itemType,
            native_id: asString(rawPayload.id) ||
              `responses-${itemType}-${callId}`,
            raw_payload: rawPayload,
            provider_schema_version: nativeCall.provider_schema_version,
            item_origin: 'native',
            source_format: OPENAI_RESPONSES_REASONING_FORMAT,
            source_origin: nativeCall.source_origin,
            position: projection?.position || {
              turn: messageIndex,
              step: 0,
              item: contentIndex
            },
            group_id: nativeCall.group_id || callId,
            call_id: callId,
            pair_id: callId,
            depends_on: [...new Set(dependencies)],
            capture_state: captureState,
            ...(content.status ? { provider_status: content.status } : {})
          };
          content.native_item = nativeItem;
        } else {
          nativeItem.item_origin ||= 'native';
          nativeItem.native_id ||= asString(nativeItem.raw_payload.id) ||
            `responses-${nativeItem.item_type}-${callId}`;
          nativeItem.group_id = nativeCall.group_id || callId;
          nativeItem.call_id = callId;
          nativeItem.pair_id = callId;
          nativeItem.depends_on = [...new Set(dependencies)];
          nativeItem.capture_state = captureState;
        }
      }
    }

    const nativeItems = message.content.flatMap((content) => {
      const nativeItem = readOpenAIResponsesNativeItemFromContent(content);
      return nativeItem ? [nativeItem] : [];
    });
    const mergedNativeItems = [...new Set([
      ...(message.native_items || []),
      ...nativeItems
    ])];
    message.native_items = mergedNativeItems.length > 0
      ? mergedNativeItems
      : undefined;
  }
}

function attachDetachedOpenAIResponsesNativeItems(
  message: StandardRequestInputMessage
): void {
  const detached = message.native_items || [];
  const claimed = new Set(
    message.content
      .map(readOpenAIResponsesNativeItemFromContent)
      .filter((item): item is ProviderNativeItem => Boolean(item))
  );

  for (const content of message.content) {
    if (content.type === 'provider_native_item' || content.native_item) {
      continue;
    }
    const matching = detached.find((nativeItem) =>
      !claimed.has(nativeItem) &&
      nativeItem.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
      doesOpenAIResponsesNativeItemMatchContent(nativeItem, content)
    );
    if (matching) {
      content.native_item = matching;
      claimed.add(matching);
    }
  }
}

function collectOpenAIResponsesReasoningNativeItems(
  message: StandardRequestInputMessage
): ProviderNativeItem[] {
  const nativeItems = [
    ...(message.native_items || []),
    ...message.content.flatMap((content) => {
      const nativeItem = readOpenAIResponsesNativeItemFromContent(content);
      return nativeItem ? [nativeItem] : [];
    })
  ];
  return [...new Set(nativeItems)].filter((item) =>
    item.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
    item.item_type.toLowerCase() === 'reasoning'
  );
}

function readOpenAIResponsesNativeItemFromContent(
  content: StandardRequestInputContent
): ProviderNativeItem | undefined {
  if (content.type === 'provider_native_item') {
    return content;
  }
  return content.native_item;
}

function doesOpenAIResponsesNativeItemMatchContent(
  nativeItem: ProviderNativeItem,
  content: Exclude<StandardRequestInputContent, ProviderNativeItem>
): boolean {
  const itemType = nativeItem.item_type.toLowerCase();
  const callId = nativeItem.call_id || nativeItem.pair_id ||
    asString(nativeItem.raw_payload.call_id) ||
    asString(nativeItem.raw_payload.tool_use_id);
  if (content.type === 'reasoning') {
    return itemType === 'reasoning' &&
      (!content.id || !nativeItem.native_id || content.id === nativeItem.native_id);
  }
  if (content.type === 'tool_use') {
    return (itemType === 'function_call' || itemType === 'tool_call') &&
      callId === content.id;
  }
  if (content.type === 'tool_search_call') {
    return itemType === 'tool_search_call' && callId === content.call_id;
  }
  if (content.type === 'tool_result') {
    return itemType === 'function_call_output' && callId === content.tool_use_id;
  }
  if (content.type === 'tool_search_output') {
    return itemType === 'tool_search_output' && callId === content.call_id;
  }
  return false;
}

function isOpenAIResponsesCallNativeItem(item: ProviderNativeItem): boolean {
  const type = item.item_type.toLowerCase();
  return type === 'function_call' || type === 'tool_call' || type === 'tool_search_call';
}

function haveSameReasoningStateOrigin(
  left: ProviderNativeItem,
  right: ProviderNativeItem
): boolean {
  return left.source_format === right.source_format &&
    left.source_origin.provider === right.source_origin.provider &&
    left.source_origin.endpoint === right.source_origin.endpoint &&
    left.source_origin.model === right.source_origin.model &&
    left.source_origin.credentialScope === right.source_origin.credentialScope;
}

function combineProviderNativeCaptureStates(
  states: ProviderNativeItem['capture_state'][]
): ProviderNativeItem['capture_state'] {
  if (states.some((state) => state === 'interrupted')) {
    return 'interrupted';
  }
  return states.length > 0 && states.every((state) => state === 'complete')
    ? 'complete'
    : 'partial';
}

function buildFallbackOpenAIResponsesCallPayload(
  content: Extract<StandardRequestInputContent, { type: 'tool_use' | 'tool_search_call' }>
): Record<string, unknown> {
  if (content.type === 'tool_search_call') {
    return {
      type: 'tool_search_call',
      execution: 'client',
      call_id: content.call_id,
      arguments: content.arguments,
      ...(content.status ? { status: content.status } : {})
    };
  }
  return {
    type: 'function_call',
    ...(content.native_id ? { id: content.native_id } : {}),
    call_id: content.id,
    name: content.name,
    arguments: serializeOpenAIResponsesFunctionArguments(content.input),
    ...(content.status ? { status: content.status } : {}),
    ...(content.caller ? { caller: content.caller } : {})
  };
}

function buildFallbackOpenAIResponsesResultPayload(
  content: Extract<StandardRequestInputContent, { type: 'tool_result' | 'tool_search_output' }>
): Record<string, unknown> {
  if (content.type === 'tool_search_output') {
    return {
      type: 'tool_search_output',
      execution: 'client',
      call_id: content.call_id,
      tools: content.tools,
      ...(content.status ? { status: content.status } : {})
    };
  }
  return {
    type: 'function_call_output',
    ...(content.native_id ? { id: content.native_id } : {}),
    call_id: content.tool_use_id,
    output: content.content,
    ...(content.status ? { status: content.status } : {}),
    ...(content.caller ? { caller: content.caller } : {}),
    ...(content.is_error !== undefined ? { is_error: content.is_error } : {})
  };
}

function serializeOpenAIResponsesFunctionArguments(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value ?? {}) || '{}';
  } catch {
    return '{}';
  }
}

function linkAnthropicNativeToolGroups(messages: StandardRequestInputMessage[]): void {
  const groupsByCallId = new Map<string, ProviderNativeItem>();
  for (const message of messages) {
    if (message.role === 'assistant') {
      let pendingNative: ProviderNativeItem | undefined;
      for (const item of message.content) {
        if (
          item.type === 'reasoning' &&
          item.native_item?.source_format === ANTHROPIC_CLAUDE_REASONING_FORMAT
        ) {
          pendingNative = item.native_item;
          continue;
        }
        if (item.type !== 'tool_use' || !pendingNative) {
          continue;
        }
        const callId = item.id;
        if (pendingNative.group_id && pendingNative.group_id !== callId) {
          pendingNative = undefined;
          continue;
        }
        pendingNative.group_id = callId;
        pendingNative.call_id = callId;
        pendingNative.native_id ||= `anthropic-thinking-${pendingNative.position.turn}-${pendingNative.position.item}`;
        const nativeToolUse: ProviderNativeItem = {
          type: 'provider_native_item',
          item_type: 'tool_use',
          native_id: callId,
          raw_payload: {
            type: 'tool_use',
            id: callId,
            name: item.name,
            input: item.input
          },
          provider_schema_version: pendingNative.provider_schema_version,
          item_origin: 'native',
          source_format: pendingNative.source_format,
          source_origin: pendingNative.source_origin,
          position: {
            ...pendingNative.position,
            item: pendingNative.position.item + 1
          },
          group_id: callId,
          call_id: callId,
          pair_id: callId,
          depends_on: [pendingNative.native_id],
          capture_state: pendingNative.capture_state,
          ...(pendingNative.provider_mode
            ? { provider_mode: pendingNative.provider_mode }
            : {})
        };
        item.native_item = nativeToolUse;
        groupsByCallId.set(callId, nativeToolUse);
        pendingNative = undefined;
      }
    } else {
      for (const item of message.content) {
        if (item.type !== 'tool_result') {
          continue;
        }
        const call = groupsByCallId.get(item.tool_use_id);
        if (!call) {
          continue;
        }
        item.native_item = {
          type: 'provider_native_item',
          item_type: 'tool_result',
          raw_payload: {
            type: 'tool_result',
            tool_use_id: item.tool_use_id,
            content: item.content,
            ...(item.is_error !== undefined ? { is_error: item.is_error } : {})
          },
          provider_schema_version: call.provider_schema_version,
          item_origin: 'native',
          source_format: call.source_format,
          source_origin: call.source_origin,
          position: {
            turn: call.position.turn + 1,
            step: call.position.step,
            item: call.position.item + 1
          },
          group_id: item.tool_use_id,
          call_id: item.tool_use_id,
          pair_id: item.tool_use_id,
          depends_on: [call.native_id || item.tool_use_id],
          capture_state: call.capture_state,
          ...(call.provider_mode ? { provider_mode: call.provider_mode } : {})
        };
      }
    }
    const native = collectNativeItemsFromInputContent(message.content);
    message.native_items = 'native_items' in native ? native.native_items : undefined;
  }
}

function extractAnthropicToolReferences(content: unknown): string[] {
  if (!Array.isArray(content)) {
    return [];
  }

  const names = content
    .map((item) =>
      isObject(item) && asString(item.type) === 'tool_reference'
        ? asString(item.tool_name)
        : undefined
    )
    .filter((name): name is string => Boolean(name));
  return [...new Set(names)];
}

function extractGeminiSystemInstruction(systemInstruction: unknown): string | undefined {
  if (typeof systemInstruction === 'string') {
    return systemInstruction.trim() || undefined;
  }

  if (!isObject(systemInstruction)) {
    return undefined;
  }

  const parts = Array.isArray(systemInstruction.parts) ? systemInstruction.parts : [];
  const value = parts.map(extractTextFromPart).filter(Boolean).join('\n').trim();
  return value || undefined;
}

interface GeminiToolCallState {
  toolUseIdsByName: Map<string, string[]>;
  nativeToolGroupsById: Map<string, ProviderNativeItem>;
}

function createGeminiToolCallState(): GeminiToolCallState {
  return {
    toolUseIdsByName: new Map(),
    nativeToolGroupsById: new Map()
  };
}

function extractGeminiMessageContent(
  role: 'user' | 'assistant',
  parts: unknown[],
  state: GeminiToolCallState,
  messageIndex: number,
  sourceModel?: string
): StandardRequestInputContent[] {
  const normalized: StandardRequestInputContent[] = [];

  for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
    const part = parts[partIndex];
    if (!isObject(part)) {
      const text = extractTextFromPart(part);
      if (text) {
        normalized.push({
          type: 'input_text',
          text
        });
      }
      continue;
    }

    const functionCall = readGeminiFunctionCall(part);
    if (functionCall && role === 'assistant') {
      const name = asString(functionCall.name);
      if (!name) {
        continue;
      }

      const id =
        asString(functionCall.id) ||
        asString(functionCall.call_id) ||
        asString(functionCall.callId) ||
        `gemini_tool_${messageIndex}_${partIndex}`;
      const toolUse: StandardRequestInputContent = {
        type: 'tool_use',
        id,
        name,
        input: normalizeGeminiFunctionCallArguments(functionCall.args ?? functionCall.arguments)
      };
      const rawThoughtSignature = readGeminiThoughtSignature(part, functionCall);
      const envelope = rawThoughtSignature
        ? decodeReasoningTransportEnvelope(rawThoughtSignature)
        : undefined;
      const thoughtSignature = envelope?.data || rawThoughtSignature;
      const thoughtSignatureFormat =
        envelope?.format || GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
      const nativeItem = captureGeminiGenerateContentPartNativeItem({
        part,
        messageIndex,
        partIndex,
        itemType: 'function_call',
        callId: id,
        envelope,
        requireSignature: isGemini3ModelName(sourceModel),
        sourceModel
      });
      if (
        thoughtSignature &&
        thoughtSignatureFormat !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT
      ) {
        normalized.push({
          type: 'reasoning',
          ...(envelope?.id ? { id: envelope.id } : {}),
          source_format: thoughtSignatureFormat,
          ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
          encrypted_content: thoughtSignature,
          reasoning_details: [
            envelope?.kind === 'signature'
              ? {
                  type: 'reasoning.text',
                  signature: thoughtSignature,
                  ...(envelope?.id ? { id: envelope.id } : {}),
                  format: thoughtSignatureFormat,
                  index: partIndex
                }
              : {
                  type: 'reasoning.encrypted',
                  data: thoughtSignature,
                  ...(envelope?.id ? { id: envelope.id } : {}),
                  format: thoughtSignatureFormat,
                  index: partIndex
                }
          ]
        });
      } else if (thoughtSignature) {
        toolUse.thought_signature = thoughtSignature;
        toolUse.thought_signature_format = GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
        if (envelope?.origin) {
          toolUse.thought_signature_origin = envelope.origin;
        }
      }
      if (nativeItem) {
        toolUse.native_item = nativeItem;
        state.nativeToolGroupsById.set(id, nativeItem);
      }
      normalized.push(toolUse);
      trackGeminiToolUseId(state, name, id);
      continue;
    }

    const functionResponse = readGeminiFunctionResponse(part);
    if (functionResponse && role === 'user') {
      const responsePayload = isObject(functionResponse.response) ? functionResponse.response : undefined;
      const name = asString(functionResponse.name) || asString(responsePayload?.name);
      const explicitToolUseId =
        asString(functionResponse.id) ||
        asString(functionResponse.call_id) ||
        asString(functionResponse.callId) ||
        asString(responsePayload?.id) ||
        asString(responsePayload?.call_id) ||
        asString(responsePayload?.callId);
      const toolUseId =
        explicitToolUseId ||
        (name ? consumeGeminiToolUseId(state, name) : undefined) ||
        `gemini_tool_${messageIndex}_${partIndex}`;
      const rawResponseContent =
        responsePayload && responsePayload.content !== undefined
          ? responsePayload.content
          : functionResponse.response ?? functionResponse.output ?? functionResponse.result ?? functionResponse.content;
      const toolResult: StandardRequestInputContent = {
        type: 'tool_result',
        tool_use_id: toolUseId,
        content: normalizeGeminiFunctionResponseContent(rawResponseContent)
      };
      const nativeCall = state.nativeToolGroupsById.get(toolUseId);
      if (nativeCall) {
        toolResult.native_item = {
          type: 'provider_native_item',
          item_type: 'function_response',
          raw_payload: part,
          provider_schema_version: nativeCall.provider_schema_version,
          item_origin: 'native',
          source_format: nativeCall.source_format,
          source_origin: nativeCall.source_origin,
          position: { turn: messageIndex, step: 0, item: partIndex },
          group_id: toolUseId,
          call_id: toolUseId,
          pair_id: toolUseId,
          depends_on: [nativeCall.native_id || toolUseId],
          capture_state: nativeCall.capture_state
        };
      }
      const isError =
        asBoolean(functionResponse.is_error) ??
        asBoolean(functionResponse.error) ??
        asBoolean(responsePayload?.is_error) ??
        asBoolean(responsePayload?.error);
      if (isError !== undefined) {
        toolResult.is_error = isError;
      }

      normalized.push(toolResult);
      continue;
    }

    const thought = normalizeGeminiThoughtPart(part, role, partIndex);
    if (thought) {
      const signature = readThoughtSignature(part);
      const envelope = signature ? decodeReasoningTransportEnvelope(signature) : undefined;
      const nativeItem = captureGeminiGenerateContentPartNativeItem({
        part,
        messageIndex,
        partIndex,
        itemType: 'thought',
        envelope,
        sourceModel
      });
      if (nativeItem && thought.type === 'reasoning') {
        thought.native_item = nativeItem;
      }
      normalized.push(thought);
      continue;
    }

    const text = extractTextFromPart(part);
    const rawPartSignature = readThoughtSignature(part);
    if (text || (typeof part.text === 'string' && rawPartSignature)) {
      const envelope = rawPartSignature
        ? decodeReasoningTransportEnvelope(rawPartSignature)
        : undefined;
      const nativeItem = captureGeminiGenerateContentPartNativeItem({
        part,
        messageIndex,
        partIndex,
        itemType: 'part',
        envelope,
        sourceModel
      });
      normalized.push({
        type: 'input_text',
        text: typeof part.text === 'string' ? part.text : text,
        ...(nativeItem ? { native_item: nativeItem } : {})
      });
    }
  }

  return normalized;
}

function rememberOpenAIResponsesRawInputProjection(
  rawItemsByContent: OpenAIResponsesRawInputProjectionMap | undefined,
  content: StandardRequestInputContent,
  rawPayload: Record<string, unknown>,
  position: ProviderNativeItem['position']
): void {
  rawItemsByContent?.set(content, {
    rawPayload: unwrapReasoningTransportCarriers(rawPayload),
    position
  });
}

function captureGeminiGenerateContentPartNativeItem(options: {
  part: Record<string, unknown>;
  messageIndex: number;
  partIndex: number;
  itemType: 'part' | 'thought' | 'function_call';
  callId?: string;
  envelope?: ReturnType<typeof decodeReasoningTransportEnvelope>;
  requireSignature?: boolean;
  sourceModel?: string;
}): ProviderNativeItem | undefined {
  if (options.envelope?.nativeItem) {
    return options.envelope.nativeItem;
  }
  const rawSignature = readThoughtSignature(options.part) ||
    (isObject(options.part.functionCall)
      ? readThoughtSignature(options.part.functionCall)
      : undefined);
  if (!rawSignature && !options.requireSignature) {
    return undefined;
  }
  const readableText = typeof options.part.text === 'string' ? options.part.text : undefined;
  return {
    type: 'provider_native_item',
    item_type: options.itemType,
    ...(asString(options.part.id) || options.callId
      ? { native_id: asString(options.part.id) || options.callId }
      : {}),
    raw_payload: unwrapReasoningTransportCarriers(options.part),
    provider_schema_version: 'gemini-generate-content-v1beta',
    item_origin: 'native',
    source_format: options.envelope?.format || GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
    source_origin: options.envelope?.origin || {
      provider: 'gemini',
      endpoint: 'unverified',
      ...(options.sourceModel ? { model: options.sourceModel } : {})
    },
    position: { turn: options.messageIndex, step: 0, item: options.partIndex },
    ...(options.callId ? {
      group_id: options.callId,
      call_id: options.callId,
      pair_id: options.callId
    } : {}),
    capture_state:
      options.envelope?.carrierVersion === 2 && options.envelope.origin
        ? 'complete'
        : 'partial',
    ...(readableText !== undefined ? { readable_text: readableText } : {})
  };
}

function isGemini3ModelName(model: string | undefined): boolean {
  return /^gemini-3(?:[.-]|$)/i.test(model?.trim() || '');
}

function normalizeGeminiThoughtPart(
  part: Record<string, unknown>,
  role: 'user' | 'assistant',
  index: number
): StandardRequestInputContent | null {
  if (role !== 'assistant' || asBoolean(part.thought) !== true) {
    return null;
  }

  const text = asString(part.text) || asString(part.thoughtText) || asString(part.thought_text);
  const thoughtSignature = readThoughtSignature(part);
  if (!text && !thoughtSignature) {
    return null;
  }
  const envelope = thoughtSignature
    ? decodeReasoningTransportEnvelope(thoughtSignature)
    : undefined;
  const encryptedContent = envelope?.data || thoughtSignature;
  const sourceFormat = envelope?.format || GEMINI_GENERATE_CONTENT_REASONING_FORMAT;

  const reasoning: StandardRequestInputContent = {
    type: 'reasoning',
    ...(envelope?.id ? { id: envelope.id } : {}),
    source_format: sourceFormat,
    ...(envelope?.origin ? { source_origin: envelope.origin } : {}),
    reasoning_details: [
      ...(text
        ? [
            {
              type: 'reasoning.text',
              text,
              format: sourceFormat,
              index
            }
          ]
        : []),
      ...(encryptedContent
        ? [
            envelope?.kind === 'signature'
              ? {
                  type: 'reasoning.text',
                  signature: encryptedContent,
                  ...(envelope?.id ? { id: envelope.id } : {}),
                  format: sourceFormat,
                  index
                }
              : {
                  type: 'reasoning.encrypted',
                  data: encryptedContent,
                  ...(envelope?.id ? { id: envelope.id } : {}),
                  format: sourceFormat,
                  index
                }
          ]
        : [])
    ]
  };

  if (text) {
    reasoning.text = text;
  }
  if (encryptedContent) {
    reasoning.encrypted_content = encryptedContent;
  }

  return reasoning;
}

function readGeminiFunctionCall(part: Record<string, unknown>): Record<string, unknown> | undefined {
  if (isObject(part.functionCall)) {
    return part.functionCall;
  }

  if (isObject(part.function_call)) {
    return part.function_call;
  }

  return undefined;
}

function readGeminiFunctionResponse(part: Record<string, unknown>): Record<string, unknown> | undefined {
  if (isObject(part.functionResponse)) {
    return part.functionResponse;
  }

  if (isObject(part.function_response)) {
    return part.function_response;
  }

  return undefined;
}

function readGeminiThoughtSignature(
  part: Record<string, unknown>,
  functionCall?: Record<string, unknown>
): string | undefined {
  return (
    readThoughtSignature(part) ||
    (functionCall ? readThoughtSignature(functionCall) : undefined)
  );
}

function readGeminiThinkingOption(value: unknown): Record<string, string> | undefined {
  if (!isObject(value)) {
    return undefined;
  }

  const budget = asNumber(value.thinkingBudget) ?? asNumber(value.thinking_budget);
  if (budget === 0 || asBoolean(value.enabled) === false) {
    return { type: 'disabled' };
  }

  return { type: 'enabled' };
}

function readThoughtSignature(value: Record<string, unknown>): string | undefined {
  return asString(value.thoughtSignature) || asString(value.thought_signature);
}

function normalizeGeminiFunctionCallArguments(value: unknown): unknown {
  if (isObject(value)) {
    return value;
  }

  if (typeof value !== 'string') {
    return {};
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return {};
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function normalizeGeminiFunctionResponseContent(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (!isObject(value) && !Array.isArray(value)) {
    return '';
  }

  try {
    return JSON.stringify(value);
  } catch {
    return '';
  }
}

function trackGeminiToolUseId(state: GeminiToolCallState, name: string, id: string) {
  const queue = state.toolUseIdsByName.get(name) || [];
  queue.push(id);
  state.toolUseIdsByName.set(name, queue);
}

function consumeGeminiToolUseId(state: GeminiToolCallState, name: string): string | undefined {
  const queue = state.toolUseIdsByName.get(name);
  if (!queue || queue.length === 0) {
    return undefined;
  }

  const id = queue.shift();
  if (queue.length === 0) {
    state.toolUseIdsByName.delete(name);
  } else {
    state.toolUseIdsByName.set(name, queue);
  }

  return id;
}

function readGeminiTools(value: unknown): unknown[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const normalized: Record<string, unknown>[] = [];
  for (const item of value) {
    if (!isObject(item)) {
      continue;
    }

    const declarations = readGeminiFunctionDeclarations(item);
    for (const declaration of declarations) {
      const mappedTool = mapGeminiFunctionDeclaration(declaration);
      if (mappedTool) {
        normalized.push(mappedTool);
      }
    }
  }

  return normalized.length > 0 ? normalized : undefined;
}

function readGeminiFunctionDeclarations(item: Record<string, unknown>): Record<string, unknown>[] {
  const declarationsRaw = Array.isArray(item.functionDeclarations)
    ? item.functionDeclarations
    : Array.isArray(item.function_declarations)
      ? item.function_declarations
      : [];

  return declarationsRaw.filter((entry): entry is Record<string, unknown> => isObject(entry));
}

function mapGeminiFunctionDeclaration(declaration: Record<string, unknown>): Record<string, unknown> | null {
  const name = asString(declaration.name);
  if (!name) {
    return null;
  }

  const parameters = ensureGeminiFunctionParameters(
    declaration.parameters ?? declaration.parametersJsonSchema ?? declaration.parameters_json_schema
  );
  const functionObject: Record<string, unknown> = {
    name,
    parameters
  };
  const description = asString(declaration.description);
  if (description) {
    functionObject.description = description;
  }

  return {
    type: 'function',
    function: functionObject
  };
}

function ensureGeminiFunctionParameters(value: unknown): Record<string, unknown> {
  if (isObject(value)) {
    return value;
  }

  return {
    type: 'object',
    properties: {}
  };
}

function readGeminiToolChoice(value: unknown): unknown {
  if (!isObject(value)) {
    return undefined;
  }

  const functionCallingConfig = isObject(value.functionCallingConfig)
    ? value.functionCallingConfig
    : isObject(value.function_calling_config)
      ? value.function_calling_config
      : undefined;
  if (!functionCallingConfig) {
    return undefined;
  }

  const mode = asString(functionCallingConfig.mode)?.trim().toUpperCase();
  const allowedFunctionNames = readGeminiAllowedFunctionNames(functionCallingConfig);
  if (mode === 'NONE') {
    return 'none';
  }

  if (mode === 'AUTO') {
    return 'auto';
  }

  if (mode === 'ANY') {
    if (allowedFunctionNames.length === 1) {
      return {
        type: 'function',
        function: {
          name: allowedFunctionNames[0]
        }
      };
    }

    return 'required';
  }

  if (allowedFunctionNames.length === 1) {
    return {
      type: 'function',
      function: {
        name: allowedFunctionNames[0]
      }
    };
  }

  return undefined;
}

function readGeminiInteractionsToolChoice(value: unknown): unknown {
  if (typeof value === 'string') {
    if (value === 'any') {
      return 'required';
    }
    if (value === 'none' || value === 'auto' || value === 'validated') {
      return value;
    }
    return undefined;
  }

  if (!isObject(value)) {
    return undefined;
  }

  const allowedTools = isObject(value.allowed_tools)
    ? value.allowed_tools
    : isObject(value.allowedTools)
      ? value.allowedTools
      : undefined;
  if (!allowedTools) {
    return undefined;
  }

  const mode = asString(allowedTools.mode);
  const tools = Array.isArray(allowedTools.tools)
    ? allowedTools.tools.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];
  if (tools.length === 1) {
    return {
      type: 'function',
      function: {
        name: tools[0]
      }
    };
  }
  if (mode === 'any') {
    return 'required';
  }
  if (mode === 'auto' || mode === 'none' || mode === 'validated') {
    return mode;
  }

  return undefined;
}

function readGeminiAllowedFunctionNames(functionCallingConfig: Record<string, unknown>): string[] {
  const rawNames = Array.isArray(functionCallingConfig.allowedFunctionNames)
    ? functionCallingConfig.allowedFunctionNames
    : Array.isArray(functionCallingConfig.allowed_function_names)
      ? functionCallingConfig.allowed_function_names
      : [];
  const names = rawNames
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);

  return [...new Set(names)];
}

function ensureInputWithInstructions(
  input: string | StandardRequestInputMessage[],
  instructions?: string
): string | StandardRequestInputMessage[] | null {
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (trimmed) {
      return trimmed;
    }

    return instructions ? '' : null;
  }

  if (input.length > 0) {
    return input;
  }

  return instructions ? '' : null;
}

function readTools(value: unknown): unknown[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const tools = value.filter((item) => isObject(item));
  return tools.length > 0 ? tools : undefined;
}

function readReasoningSplitOption(body: Record<string, unknown>): boolean | undefined {
  return (
    asBoolean(body.reasoning_split) ??
    asBoolean(body.interleaved_thinking) ??
    asBoolean(body.interleavedThinking)
  );
}

function readOptionalRequestOption(value: unknown): unknown | undefined {
  return value === undefined ? undefined : value;
}

function readReasoningOption(body: Record<string, unknown>): unknown {
  const reasoning = readOptionalRequestOption(body.reasoning);
  const effort = asString(body.reasoning_effort);
  if (effort === undefined) {
    return reasoning;
  }

  if (isObject(reasoning) && !Object.prototype.hasOwnProperty.call(reasoning, 'effort')) {
    return {
      ...reasoning,
      effort
    };
  }

  return reasoning ?? { effort };
}

function readRecordOption(value: unknown): Record<string, unknown> | undefined {
  return isObject(value) ? value : undefined;
}

function readToolChoice(value: unknown): unknown {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value === 'string' || isObject(value)) {
    return value;
  }

  return undefined;
}
