import type {
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
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import { normalizeNamespacedToolName } from '../target/tools';

export function parseOpenAIResponsesRequest(body: Record<string, unknown>): Result<StandardRequest> {
  const inputResult = normalizeResponsesInput(body.input);
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
    text: readOptionalRequestOption(body.text)
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

  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const rawMessage of rawMessages) {
    if (!isObject(rawMessage)) {
      continue;
    }

    const role = normalizeConversationRole(rawMessage.role);
    const content = extractAnthropicMessageContent(role, rawMessage.content);
    if (content.length === 0) {
      continue;
    }

    inputMessages.push({
      type: 'message',
      role,
      content
    });
  }

  const input = ensureInputWithInstructions(inputMessages, instructions);
  if (!input) {
    return err('Anthropic request requires non-empty messages or system prompt.');
  }

  return ok({
    model: asString(body.model),
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

  const contents = Array.isArray(body.contents) ? body.contents : [];
  for (let itemIndex = 0; itemIndex < contents.length; itemIndex += 1) {
    const item = contents[itemIndex];
    if (!isObject(item)) {
      continue;
    }

    const role = normalizeConversationRole(item.role === 'model' ? 'assistant' : item.role);
    const parts = Array.isArray(item.parts) ? item.parts : [];
    const content = extractGeminiMessageContent(role, parts, geminiToolState, itemIndex);
    if (content.length === 0) {
      continue;
    }

    inputMessages.push({
      type: 'message',
      role,
      content
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
    model: modelFromPath || asString(body.model),
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

function normalizeResponsesInput(input: unknown): Result<string | StandardRequestInputMessage[]> {
  if (typeof input === 'string') {
    return ok(input.trim());
  }

  if (!Array.isArray(input)) {
    if (isObject(input)) {
      const asMessage = normalizeResponsesInputItem(input);
      if (!asMessage) {
        return err('OpenAI responses request contains invalid input item.');
      }

      return ok([asMessage]);
    }

    return err('OpenAI responses request requires input.');
  }

  const messages: StandardRequestInputMessage[] = [];
  for (const item of input) {
    const normalized = normalizeResponsesInputItem(item);
    if (normalized) {
      messages.push(normalized);
    }
  }

  return ok(coalesceResponsesInputMessages(messages));
}

function normalizeGeminiInteractionsInput(input: unknown): Result<string | StandardRequestInputMessage[]> {
  if (typeof input === 'string') {
    return ok(input.trim());
  }

  if (!Array.isArray(input)) {
    if (isObject(input)) {
      const message = normalizeGeminiInteractionInputItem(input);
      if (!message) {
        return err('Gemini interactions request contains invalid input item.');
      }
      return ok([message]);
    }

    return err('Gemini interactions request requires input.');
  }

  const messages: StandardRequestInputMessage[] = [];
  const toolNamesById = new Map<string, string>();
  for (const item of input) {
    const message = normalizeGeminiInteractionInputItem(item, toolNamesById);
    if (message) {
      messages.push(message);
    }
  }

  return ok(coalesceResponsesInputMessages(messages));
}

function normalizeGeminiInteractionInputItem(
  item: unknown,
  toolNamesById = new Map<string, string>()
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
          ]
        }
      ]
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
        content: [...message.content]
      };
      continue;
    }

    if (shouldCoalesceResponsesAssistantMessages(pendingAssistant.content, message.content)) {
      pendingAssistant.content.push(...message.content);
      continue;
    }

    flushPendingAssistant();
    pendingAssistant = {
      ...message,
      content: [...message.content]
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

function normalizeResponsesInputItem(item: unknown): StandardRequestInputMessage | null {
  if (!isObject(item)) {
    return null;
  }

  const type = asString(item.type);

  const reasoningContent = normalizeOpenAIResponsesReasoningItem(item);
  if (reasoningContent) {
    return {
      type: 'message',
      role: 'assistant',
      content: [reasoningContent]
    };
  }
  if (type === 'reasoning') {
    return null;
  }

  const toolSearchCallContent = normalizeOpenAIResponsesToolSearchCallItem(item);
  if (toolSearchCallContent) {
    return {
      type: 'message',
      role: 'assistant',
      content: [toolSearchCallContent]
    };
  }

  const toolSearchOutputContent = normalizeOpenAIResponsesToolSearchOutputItem(item);
  if (toolSearchOutputContent) {
    return {
      type: 'message',
      role: 'user',
      content: [toolSearchOutputContent]
    };
  }

  const functionCallContent = normalizeOpenAIResponsesFunctionCallItem(item);
  if (functionCallContent) {
    return {
      type: 'message',
      role: 'assistant',
      content: [functionCallContent]
    };
  }

  const functionCallOutputContent = normalizeOpenAIResponsesFunctionCallOutputItem(item);
  if (functionCallOutputContent) {
    return {
      type: 'message',
      role: 'user',
      content: [functionCallOutputContent]
    };
  }

  if (type === 'message' || item.role !== undefined || item.content !== undefined) {
    const role = normalizeConversationRole(item.role);
    const content = normalizeOpenAIResponsesMessageContent(role, item.content);
    if (content.length === 0) {
      return null;
    }

    return {
      type: 'message',
      role,
      content
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
  content: unknown
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

    if (role === 'assistant') {
      const reasoningContent = normalizeOpenAIResponsesReasoningItem(block);
      if (reasoningContent) {
        normalized.push(reasoningContent);
        continue;
      }

      const functionCallContent = normalizeOpenAIResponsesFunctionCallItem(block);
      if (functionCallContent) {
        normalized.push(functionCallContent);
        continue;
      }
    }

    if (role === 'user') {
      const functionCallOutputContent = normalizeOpenAIResponsesFunctionCallOutputItem(block);
      if (functionCallOutputContent) {
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
    name,
    input
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
    content: normalizeToolResultContent(item.output ?? item.content ?? item.result)
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
  content: unknown
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
    const reasoning = normalizeAnthropicThinkingBlock(block, role, normalized.length);
    if (reasoning) {
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
        input: block.input ?? {}
      };
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
  index: number
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
      ]
    };
  }

  return null;
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
}

function createGeminiToolCallState(): GeminiToolCallState {
  return {
    toolUseIdsByName: new Map()
  };
}

function extractGeminiMessageContent(
  role: 'user' | 'assistant',
  parts: unknown[],
  state: GeminiToolCallState,
  messageIndex: number
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
      normalized.push(thought);
      continue;
    }

    const text = extractTextFromPart(part);
    if (text) {
      normalized.push({
        type: 'input_text',
        text
      });
    }
  }

  return normalized;
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
