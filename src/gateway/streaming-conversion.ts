import { randomUUID } from 'node:crypto';
import { Readable } from 'node:stream';
import type { FastifyReply } from 'fastify';
import {
  mapFinishReasonToAnthropic,
  mapFinishReasonToGemini,
  mapFinishReasonToOpenAI,
  normalizeOpenAIResponsesCompletedEventPayload,
  normalizeOpenAIResponsesCompletedResponse,
  normalizeOpenAIResponsesUsage
} from '../adapters/builtins/common';
import {
  formatAnthropicMessagesResponse,
  formatGeminiGenerateContentResponse
} from '../adapters/builtins/source/formatters';
import { splitNamespacedToolCallName } from '../adapters/builtins/target/tools';
import { parseSseChunks } from '../sse';
import { bindAbortSignalToReadable } from '../upstream/client';
import type {
  GatewaySourceContext,
  StandardRequest,
  StandardResponse,
  StandardResponseReasoning,
  StandardUsage
} from '../types';
import { asNumber, asString, isObject } from '../utils';

interface OpenAIResponsesRelayState {
  started: boolean;
  finished: boolean;
  responseId: string;
  model: string;
  outputText: string;
  reasoningItemId: string;
  reasoningOutputIndex?: number;
  reasoningText: string;
  reasoningSummaryText: string;
  reasoningEncryptedContent?: string;
  reasoningItemStarted: boolean;
  reasoningSummaryStarted: boolean;
  messageItemId: string;
  messageOutputIndex?: number;
  messageItemStarted: boolean;
  messageContentStarted: boolean;
  pendingToolCalls: Map<number, PendingOpenAIResponsesToolCall>;
  usedOutputIndices: Set<number>;
  nextOutputIndex: number;
  finishReason?: string;
  usage: Record<string, unknown>;
}

interface PendingOpenAIResponsesToolCall {
  index: number;
  outputIndex: number;
  id: string;
  callId: string;
  name: string;
  namespace?: string;
  argumentsJson: string;
  emittedArgumentsLength: number;
  added: boolean;
  done: boolean;
}

interface GeminiRelayState {
  model: string;
  outputText: string;
  finishReason?: string;
  usage: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
    cachedContentTokenCount?: number;
  };
  emittedAnyDelta: boolean;
  emittedFinal: boolean;
  pendingToolCalls: Map<number, PendingGeminiToolCall>;
}

interface PendingGeminiToolCall {
  index: number;
  name: string;
  argumentsJson: string;
}

interface AnthropicRelayState {
  started: boolean;
  finished: boolean;
  messageId: string;
  model: string;
  inputTokens?: number;
  outputTokens: number;
  cacheReadInputTokens?: number;
  cacheCreationInputTokens?: number;
  serverToolUse?: {
    web_search_requests?: number;
    web_fetch_requests?: number;
  };
  finishReason?: string;
  activeBlockType?: AnthropicContentBlockType;
  activeBlockIndex?: number;
  nextBlockIndex: number;
  pendingToolCalls: Map<number, PendingAnthropicToolCall>;
}

type AnthropicContentBlockType = 'text' | 'thinking';

interface PendingAnthropicToolCall {
  index: number;
  blockIndex: number;
  id: string;
  name: string;
  argumentsJson: string;
  emittedArgumentsLength: number;
  started: boolean;
  closed: boolean;
}

interface OpenAIChatRelayState {
  started: boolean;
  finished: boolean;
  id: string;
  model: string;
  created: number;
  emittedTextDelta: boolean;
  nextToolCallIndex: number;
  activeAnthropicToolCall?: PendingOpenAIChatAnthropicToolCall;
  finishReason?: string;
  usage: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
    cachedPromptTokens?: number;
    cacheCreationPromptTokens?: number;
  };
}

interface PendingOpenAIChatAnthropicToolCall {
  blockIndex: number;
  toolIndex: number;
  id: string;
  name: string;
  started: boolean;
}

interface OpenAIStreamToolCallAccumulator {
  id?: string;
  type?: string;
  name?: string;
  argumentsJson: string;
}

interface OpenAIReasoningAccumulator {
  id?: string;
  text: string;
  summary: string;
  encryptedContent?: string;
  rawDetails: unknown[];
}

interface OpenAINonStreamCollectionState {
  id: string;
  model: string;
  outputText: string;
  finishReason?: string;
  usage: Record<string, unknown>;
  completedResponse?: Record<string, unknown>;
  outputItems: Record<string, unknown>[];
  toolCalls: Map<number, OpenAIStreamToolCallAccumulator>;
  reasoning: OpenAIReasoningAccumulator;
}

interface AnthropicStreamToolUseAccumulator {
  id: string;
  name: string;
  inputJson: string;
}

interface AnthropicStreamThinkingAccumulator {
  type: 'thinking' | 'redacted_thinking';
  thinking: string;
  data?: string;
  signature?: string;
}

export interface OptimisticOpenAIChatStreamTurnResult {
  upstreamPayload?: Record<string, unknown>;
}

type OptimisticOpenAIChatRelay =
  | {
      sourceAdapterKey: 'openai_responses';
      state: OpenAIResponsesRelayState;
      tools?: unknown[];
    }
  | {
      sourceAdapterKey: 'anthropic_messages';
      state: AnthropicRelayState;
    };

export function relayConvertedStreamFromStandardResponse(
  reply: FastifyReply,
  source: GatewaySourceContext,
  standardResponse: StandardResponse
) {
  const frames = buildConvertedStreamFrames(source, standardResponse);

  reply.code(200);
  reply.header('content-type', 'text/event-stream; charset=utf-8');
  reply.header('cache-control', 'no-cache, no-transform');
  reply.header('connection', 'keep-alive');
  reply.header('x-accel-buffering', 'no');

  return reply.send(Readable.from(frames));
}

export function relayConvertedStreamFromUpstreamResponse(
  reply: FastifyReply,
  source: GatewaySourceContext,
  upstreamResponse: Response,
  standardRequest?: StandardRequest,
  abortSignal?: AbortSignal
) {
  reply.code(200);
  reply.header('content-type', 'text/event-stream; charset=utf-8');
  reply.header('cache-control', 'no-cache, no-transform');
  reply.header('connection', 'keep-alive');
  reply.header('x-accel-buffering', 'no');

  if (!upstreamResponse.body) {
    return reply.send('');
  }

  let stream: Readable;
  if (source.adapterKey === 'anthropic_messages') {
    stream = Readable.from(relayAnthropicMessagesFromOpenAIStream(upstreamResponse));
  } else if (source.adapterKey === 'openai_responses') {
    stream = Readable.from(relayOpenAIResponsesFromOpenAIStream(upstreamResponse, standardRequest?.tools));
  } else if (source.adapterKey === 'gemini_stream') {
    stream = Readable.from(relayGeminiStreamFromOpenAIStream(upstreamResponse));
  } else if (source.adapterKey === 'openai_chat') {
    stream = Readable.from(relayOpenAIChatFromUpstreamStream(upstreamResponse));
  } else {
    stream = Readable.fromWeb(upstreamResponse.body as unknown as ReadableStream<Uint8Array>);
  }

  bindAbortSignalToReadable(stream, abortSignal, () => {
    upstreamResponse.body?.cancel(abortSignal?.reason).catch(() => undefined);
  });
  return reply.send(stream);
}

export function canRelayOptimisticOpenAIChatStream(source: GatewaySourceContext): boolean {
  return source.adapterKey === 'openai_responses' || source.adapterKey === 'anthropic_messages';
}

export function createOptimisticOpenAIChatStreamRelay(
  source: GatewaySourceContext,
  standardRequest?: StandardRequest
): OptimisticOpenAIChatRelay | undefined {
  if (source.adapterKey === 'openai_responses') {
    return {
      sourceAdapterKey: 'openai_responses',
      tools: standardRequest?.tools,
      state: {
        started: false,
        finished: false,
        responseId: `resp_${randomUUID()}`,
        model: 'unknown',
        outputText: '',
        reasoningItemId: `rs_${randomUUID().replace(/-/g, '')}`,
        reasoningText: '',
        reasoningSummaryText: '',
        reasoningItemStarted: false,
        reasoningSummaryStarted: false,
        messageItemId: `msg_${randomUUID()}`,
        messageOutputIndex: undefined,
        messageItemStarted: false,
        messageContentStarted: false,
        pendingToolCalls: new Map(),
        usedOutputIndices: new Set(),
        nextOutputIndex: 0,
        usage: {}
      }
    };
  }

  if (source.adapterKey === 'anthropic_messages') {
    return {
      sourceAdapterKey: 'anthropic_messages',
      state: {
        started: false,
        finished: false,
        messageId: `msg_${randomUUID()}`,
        model: 'unknown',
        outputTokens: 0,
        nextBlockIndex: 0,
        pendingToolCalls: new Map()
      }
    };
  }

  return undefined;
}

export async function* relayOptimisticOpenAIChatStreamTurn(
  upstreamResponse: Response,
  relay: OptimisticOpenAIChatRelay,
  result: OptimisticOpenAIChatStreamTurnResult,
  abortSignal?: AbortSignal
): AsyncGenerator<string> {
  const collectionState = createOpenAINonStreamCollectionState();

  for await (const chunk of parseSseChunks(upstreamResponse, abortSignal)) {
    const data = chunk.data.trim();
    if (!data || data === '[DONE]') {
      continue;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload) || isOpenAIResponsesStreamEvent(payload)) {
      continue;
    }

    collectOpenAINonStreamStateFromChatChunk(collectionState, payload);
    const sanitizedPayload = stripOpenAIChatChunkToolCallsAndFinish(payload);
    if (!sanitizedPayload) {
      continue;
    }

    const frames =
      relay.sourceAdapterKey === 'openai_responses'
        ? emitOpenAIResponsesFramesFromChatChunk(relay.state, sanitizedPayload, relay.tools)
        : emitAnthropicFramesFromOpenAIChatChunk(relay.state, sanitizedPayload);
    for (const frame of frames) {
      yield frame;
    }
  }

  result.upstreamPayload = buildOpenAINonStreamPayloadFromCollectionState(collectionState);
}

export function finalizeOptimisticOpenAIChatStreamRelay(
  relay: OptimisticOpenAIChatRelay,
  finalPayload: Record<string, unknown>,
  usage?: StandardUsage
): string[] {
  const firstChoice = Array.isArray(finalPayload.choices) && isObject(finalPayload.choices[0])
    ? finalPayload.choices[0]
    : undefined;
  const finishReason = asString(firstChoice?.finish_reason);

  if (relay.sourceAdapterKey === 'openai_responses') {
    if (usage) {
      relay.state.usage = buildOpenAIResponsesUsageFromStandardUsage(usage);
    }
    if (finishReason) {
      relay.state.finishReason = finishReason;
    }
    return [...finalizeOpenAIResponsesRelay(relay.state), 'data: [DONE]\n\n'];
  }

  if (usage) {
    applyStandardUsageToAnthropicRelayState(relay.state, usage);
  }
  if (finishReason) {
    relay.state.finishReason = finishReason;
  }
  return [...flushPendingAnthropicToolCalls(relay.state), ...finalizeAnthropicRelay(relay.state)];
}

function buildOpenAIResponsesUsageFromStandardUsage(usage: StandardUsage): Record<string, unknown> {
  const inputDetails: Record<string, unknown> = {};
  if (usage.cache_read_tokens !== undefined) {
    inputDetails.cached_tokens = usage.cache_read_tokens;
  }
  if (usage.cache_write_tokens !== undefined) {
    inputDetails.cache_creation_tokens = usage.cache_write_tokens;
  }

  return {
    input_tokens: usage.input_tokens,
    output_tokens: usage.output_tokens,
    total_tokens: usage.total_tokens,
    ...(Object.keys(inputDetails).length > 0 ? { input_tokens_details: inputDetails } : {}),
    ...(usage.server_tool_use ? { server_tool_use: usage.server_tool_use } : {})
  };
}

function applyStandardUsageToAnthropicRelayState(
  state: AnthropicRelayState,
  usage: StandardUsage
): void {
  state.inputTokens = usage.input_tokens;
  state.outputTokens = usage.output_tokens ?? state.outputTokens;
  state.cacheReadInputTokens = usage.cache_read_tokens;
  state.cacheCreationInputTokens = usage.cache_write_tokens;
  state.serverToolUse = usage.server_tool_use;
}

function stripOpenAIChatChunkToolCallsAndFinish(
  payload: Record<string, unknown>
): Record<string, unknown> | undefined {
  const choicesRaw = Array.isArray(payload.choices) ? payload.choices : [];
  const choices: unknown[] = [];
  let keptChoiceData = false;

  for (const choiceRaw of choicesRaw) {
    if (!isObject(choiceRaw)) {
      continue;
    }

    const choice: Record<string, unknown> = {
      ...choiceRaw
    };
    delete choice.finish_reason;

    const delta = isObject(choiceRaw.delta) ? { ...choiceRaw.delta } : undefined;
    if (delta) {
      delete delta.tool_calls;
      if (Object.keys(delta).length > 0) {
        choice.delta = delta;
        keptChoiceData = true;
      } else {
        delete choice.delta;
      }
    }

    const message = isObject(choiceRaw.message) ? { ...choiceRaw.message } : undefined;
    if (message) {
      delete message.tool_calls;
      if (Object.keys(message).length > 0) {
        choice.message = message;
        keptChoiceData = true;
      } else {
        delete choice.message;
      }
    }

    const hasNonEmptyChoice = Object.keys(choice).some((key) => {
      if (key === 'index') {
        return false;
      }
      return choice[key] !== undefined;
    });
    if (hasNonEmptyChoice) {
      choices.push(choice);
    }
  }

  const usage = isObject(payload.usage) ? payload.usage : undefined;
  if (!keptChoiceData && !usage) {
    return undefined;
  }

  return {
    ...payload,
    choices,
    ...(usage ? { usage } : {})
  };
}

export async function collectOpenAINonStreamPayloadFromEventStream(
  upstreamResponse: Response,
  abortSignal?: AbortSignal
): Promise<Record<string, unknown>> {
  const state = createOpenAINonStreamCollectionState();

  for await (const chunk of parseSseChunks(upstreamResponse, abortSignal)) {
    const data = chunk.data.trim();
    if (!data || data === '[DONE]') {
      continue;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    if (isOpenAIResponsesStreamEvent(payload)) {
      collectOpenAINonStreamStateFromResponsesEvent(state, payload);
      continue;
    }

    collectOpenAINonStreamStateFromChatChunk(state, payload);
  }

  return buildOpenAINonStreamPayloadFromCollectionState(state);
}

function createOpenAINonStreamCollectionState(): OpenAINonStreamCollectionState {
  return {
    id: `chatcmpl_${randomUUID()}`,
    model: 'unknown',
    outputText: '',
    usage: {},
    outputItems: [],
    toolCalls: new Map(),
    reasoning: {
      text: '',
      summary: '',
      rawDetails: []
    }
  };
}

function buildOpenAINonStreamPayloadFromCollectionState(
  state: OpenAINonStreamCollectionState
): Record<string, unknown> {
  if (state.completedResponse) {
    const completedResponse = { ...state.completedResponse };
    const output = Array.isArray(completedResponse.output) ? completedResponse.output : [];
    if (output.length === 0 && state.outputItems.length > 0) {
      completedResponse.output = state.outputItems;
    }
    if (!asString(completedResponse.output_text) && state.outputText) {
      completedResponse.output_text = state.outputText;
    }
    const usage = isObject(completedResponse.usage) ? completedResponse.usage : undefined;
    completedResponse.usage = usage ? { ...state.usage, ...usage } : state.usage;
    return normalizeOpenAIResponsesCompletedResponse(completedResponse);
  }

  return {
    id: state.id,
    object: 'chat.completion',
    model: state.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: state.outputText,
          ...(state.reasoning.text
            ? {
                reasoning_content: state.reasoning.text
              }
            : {}),
          ...(state.reasoning.rawDetails.length > 0
            ? {
                reasoning_details: state.reasoning.rawDetails
              }
            : state.reasoning.summary || state.reasoning.encryptedContent
              ? {
                  reasoning_details: buildChatReasoningDetailsFromAccumulator(state.reasoning)
                }
              : {}),
          ...(state.toolCalls.size > 0
            ? {
                tool_calls: buildOpenAIStreamToolCalls(state.toolCalls)
              }
            : {})
        },
        finish_reason: state.finishReason
      }
    ],
    usage: state.usage
  };
}

export async function collectAnthropicNonStreamPayloadFromEventStream(
  upstreamResponse: Response,
  abortSignal?: AbortSignal
): Promise<Record<string, unknown>> {
  const state: {
    id: string;
    model: string;
    outputText: string;
    stopReason?: string;
    usage: Record<string, unknown>;
    toolBlocks: Map<number, AnthropicStreamToolUseAccumulator>;
    thinkingBlocks: Map<number, AnthropicStreamThinkingAccumulator>;
    activeToolBlockIndex?: number;
    activeThinkingBlockIndex?: number;
  } = {
    id: `msg_${randomUUID()}`,
    model: 'unknown',
    outputText: '',
    usage: {},
    toolBlocks: new Map(),
    thinkingBlocks: new Map()
  };

  for await (const chunk of parseSseChunks(upstreamResponse, abortSignal)) {
    const data = chunk.data.trim();
    if (!data || data === '[DONE]') {
      continue;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    const eventType = asString(payload.type) || chunk.event || '';
    if (eventType === 'message_start') {
      const message = isObject(payload.message) ? payload.message : undefined;
      const id = asString(message?.id);
      if (id) {
        state.id = id;
      }

      const model = asString(message?.model);
      if (model) {
        state.model = model;
      }

      mergeAnthropicUsageSnapshot(state.usage, isObject(message?.usage) ? message.usage : undefined);
      continue;
    }

    if (eventType === 'content_block_start') {
      const blockIndex = asNumber(payload.index);
      const block = isObject(payload.content_block) ? payload.content_block : undefined;
      if (asString(block?.type) === 'text') {
        const text = asString(block?.text);
        if (text) {
          state.outputText += text;
        }
      } else if (asString(block?.type) === 'tool_use' && blockIndex !== undefined) {
        const name = asString(block?.name);
        if (name) {
          state.toolBlocks.set(blockIndex, {
            id: asString(block?.id) || `toolu_${randomUUID().replace(/-/g, '')}`,
            name,
            inputJson: normalizeAnthropicToolStartInput(block?.input)
          });
          state.activeToolBlockIndex = blockIndex;
        }
      } else if (asString(block?.type) === 'thinking' && blockIndex !== undefined) {
        state.thinkingBlocks.set(blockIndex, {
          type: 'thinking',
          thinking: asString(block?.thinking) || '',
          signature: asString(block?.signature)
        });
        state.activeThinkingBlockIndex = blockIndex;
      } else if (asString(block?.type) === 'redacted_thinking' && blockIndex !== undefined) {
        state.thinkingBlocks.set(blockIndex, {
          type: 'redacted_thinking',
          thinking: '',
          data: asString(block?.data)
        });
        state.activeThinkingBlockIndex = blockIndex;
      }
      continue;
    }

    if (eventType === 'content_block_delta') {
      const delta = isObject(payload.delta) ? payload.delta : undefined;
      if (asString(delta?.type) === 'text_delta') {
        const text = asString(delta?.text);
        if (text) {
          state.outputText += text;
        }
      } else if (asString(delta?.type) === 'thinking_delta') {
        const blockIndex = asNumber(payload.index) ?? state.activeThinkingBlockIndex;
        const thinking = asString(delta?.thinking);
        if (blockIndex !== undefined && thinking) {
          const thinkingBlock = state.thinkingBlocks.get(blockIndex);
          if (thinkingBlock) {
            thinkingBlock.thinking += thinking;
          }
        }
      } else if (asString(delta?.type) === 'signature_delta') {
        const blockIndex = asNumber(payload.index) ?? state.activeThinkingBlockIndex;
        const signature = asString(delta?.signature);
        if (blockIndex !== undefined && signature) {
          const thinkingBlock = state.thinkingBlocks.get(blockIndex);
          if (thinkingBlock) {
            thinkingBlock.signature = signature;
          }
        }
      } else if (asString(delta?.type) === 'input_json_delta') {
        const blockIndex = asNumber(payload.index) ?? state.activeToolBlockIndex;
        const partialJson = asString(delta?.partial_json);
        if (blockIndex !== undefined && partialJson) {
          const toolBlock = state.toolBlocks.get(blockIndex);
          if (toolBlock) {
            toolBlock.inputJson += partialJson;
          }
        }
      }
      continue;
    }

    if (eventType === 'message_delta') {
      const delta = isObject(payload.delta) ? payload.delta : undefined;
      const stopReason = asString(delta?.stop_reason);
      if (stopReason) {
        state.stopReason = stopReason;
      }

      mergeAnthropicUsageSnapshot(state.usage, isObject(payload.usage) ? payload.usage : undefined);
    }
  }

  const content: Array<Record<string, unknown>> = [];
  for (const thinkingBlock of [...state.thinkingBlocks.entries()].sort((a, b) => a[0] - b[0])) {
    const block = thinkingBlock[1];
    if (block.type === 'redacted_thinking') {
      if (block.data) {
        content.push({
          type: 'redacted_thinking',
          data: block.data
        });
      }
      continue;
    }

    if (!block.thinking) {
      continue;
    }
    content.push({
      type: 'thinking',
      thinking: block.thinking,
      ...(block.signature ? { signature: block.signature } : {})
    });
  }
  if (state.outputText) {
    content.push({
      type: 'text',
      text: state.outputText
    });
  }
  for (const toolBlock of [...state.toolBlocks.values()]) {
    content.push({
      type: 'tool_use',
      id: toolBlock.id,
      name: toolBlock.name,
      input: parseStreamToolArguments(toolBlock.inputJson)
    });
  }

  return {
    id: state.id,
    type: 'message',
    role: 'assistant',
    model: state.model,
    content,
    stop_reason: state.stopReason,
    usage: state.usage
  };
}

function buildConvertedStreamFrames(source: GatewaySourceContext, standardResponse: StandardResponse): string[] {
  if (source.adapterKey === 'openai_chat') {
    return buildOpenAIChatStreamFrames(standardResponse);
  }

  if (source.adapterKey === 'openai_responses') {
    return buildOpenAIResponsesStreamFrames(standardResponse);
  }

  if (source.adapterKey === 'anthropic_messages') {
    return buildAnthropicMessagesStreamFrames(standardResponse);
  }

  if (source.adapterKey === 'gemini_stream') {
    return buildGeminiStreamFrames(standardResponse);
  }

  return buildOpenAIChatStreamFrames(standardResponse);
}

function mergeAnthropicUsageSnapshot(
  target: Record<string, unknown>,
  usage: Record<string, unknown> | undefined
): void {
  if (!usage) {
    return;
  }

  const inputTokens = asNumber(usage.input_tokens) ?? asNumber(usage.prompt_tokens);
  if (inputTokens !== undefined) {
    target.input_tokens = inputTokens;
  }

  const outputTokens = asNumber(usage.output_tokens) ?? asNumber(usage.completion_tokens);
  if (outputTokens !== undefined) {
    target.output_tokens = outputTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    target.total_tokens = totalTokens;
  }

  const cacheReadTokens =
    asNumber(usage.cache_read_input_tokens) ??
    asNumber(usage.cache_read_tokens) ??
    asNumber(isObject(usage.input_tokens_details) ? usage.input_tokens_details.cached_tokens : undefined);
  if (cacheReadTokens !== undefined) {
    target.cache_read_input_tokens = cacheReadTokens;
  }

  const cacheWriteTokens =
    asNumber(usage.cache_creation_input_tokens) ??
    asNumber(usage.cache_creation_tokens) ??
    asNumber(usage.cache_write_tokens) ??
    asNumber(isObject(usage.input_tokens_details) ? usage.input_tokens_details.cache_creation_tokens : undefined);
  if (cacheWriteTokens !== undefined) {
    target.cache_creation_input_tokens = cacheWriteTokens;
  }

  const serverToolUse = extractServerToolUse(usage.server_tool_use);
  if (serverToolUse) {
    target.server_tool_use = serverToolUse;
  }
}

function buildOpenAIChatUsage(usage: StandardResponse['usage']): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    prompt_tokens: usage.input_tokens,
    completion_tokens: usage.output_tokens,
    total_tokens: usage.total_tokens
  };

  const promptTokenDetails: Record<string, unknown> = {};
  if (usage.cache_read_tokens !== undefined) {
    promptTokenDetails.cached_tokens = usage.cache_read_tokens;
  }
  if (usage.cache_write_tokens !== undefined) {
    promptTokenDetails.cache_creation_tokens = usage.cache_write_tokens;
  }
  if (Object.keys(promptTokenDetails).length > 0) {
    payload.prompt_tokens_details = promptTokenDetails;
  }

  return payload;
}

function buildAnthropicMessageStartUsage(usage: StandardResponse['usage']): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    input_tokens: usage.input_tokens ?? 0,
    output_tokens: 0
  };
  addStandardCacheUsageToAnthropicUsage(payload, usage);
  addServerToolUseToAnthropicUsage(payload, usage.server_tool_use);
  return payload;
}

function buildAnthropicMessageDeltaUsageFromStandard(usage: StandardResponse['usage']): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    output_tokens: usage.output_tokens ?? 0
  };
  if (usage.input_tokens !== undefined) {
    payload.input_tokens = usage.input_tokens;
  }
  addStandardCacheUsageToAnthropicUsage(payload, usage);
  addServerToolUseToAnthropicUsage(payload, usage.server_tool_use);
  return payload;
}

function addStandardCacheUsageToAnthropicUsage(
  payload: Record<string, unknown>,
  usage: StandardResponse['usage']
): void {
  if (usage.cache_read_tokens !== undefined) {
    payload.cache_read_input_tokens = usage.cache_read_tokens;
  }
  if (usage.cache_write_tokens !== undefined) {
    payload.cache_creation_input_tokens = usage.cache_write_tokens;
  }
}

function addServerToolUseToAnthropicUsage(
  payload: Record<string, unknown>,
  serverToolUse: StandardResponse['usage']['server_tool_use']
): void {
  const normalized = extractServerToolUse(serverToolUse);
  if (normalized) {
    payload.server_tool_use = normalized;
  }
}

function extractServerToolUse(value: unknown): StandardResponse['usage']['server_tool_use'] {
  if (!isObject(value)) {
    return undefined;
  }

  const serverToolUse: NonNullable<StandardResponse['usage']['server_tool_use']> = {};
  const webSearchRequests = asNumber(value.web_search_requests);
  if (webSearchRequests !== undefined) {
    serverToolUse.web_search_requests = Math.max(0, Math.trunc(webSearchRequests));
  }

  const webFetchRequests = asNumber(value.web_fetch_requests);
  if (webFetchRequests !== undefined) {
    serverToolUse.web_fetch_requests = Math.max(0, Math.trunc(webFetchRequests));
  }

  return Object.keys(serverToolUse).length > 0 ? serverToolUse : undefined;
}

function buildOpenAIChatStreamFrames(standardResponse: StandardResponse): string[] {
  const created = Math.floor(Date.now() / 1000);
  const frames: string[] = [];
  frames.push(
    encodeSseData({
      id: standardResponse.id,
      object: 'chat.completion.chunk',
      created,
      model: standardResponse.model,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant'
          }
        }
      ]
    })
  );

  const reasoningText = collectStandardResponseReasoningText(standardResponse);
  if (reasoningText) {
    frames.push(
      encodeSseData({
        id: standardResponse.id,
        object: 'chat.completion.chunk',
        created,
        model: standardResponse.model,
        choices: [
          {
            index: 0,
            delta: {
              reasoning_content: reasoningText
            }
          }
        ]
      })
    );
  }

  if (standardResponse.output_text) {
    frames.push(
      encodeSseData({
        id: standardResponse.id,
        object: 'chat.completion.chunk',
        created,
        model: standardResponse.model,
        choices: [
          {
            index: 0,
            delta: {
              content: standardResponse.output_text
            }
          }
        ]
      })
    );
  }

  const toolCalls = collectStandardResponseToolCallsForOpenAIChat(standardResponse);
  for (const toolCall of toolCalls) {
    frames.push(
      encodeSseData({
        id: standardResponse.id,
        object: 'chat.completion.chunk',
        created,
        model: standardResponse.model,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: toolCall.index,
                  id: toolCall.id,
                  type: 'function',
                  function: {
                    name: toolCall.name,
                    arguments: toolCall.argumentsJson
                  }
                }
              ]
            }
          }
        ]
      })
    );
  }

  const usage = buildOpenAIChatUsage(standardResponse.usage);

  frames.push(
    encodeSseData({
      id: standardResponse.id,
      object: 'chat.completion.chunk',
      created,
      model: standardResponse.model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: mapFinishReasonToOpenAI(standardResponse.finish_reason)
        }
      ],
      usage
    })
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

function buildOpenAIResponsesStreamFrames(standardResponse: StandardResponse): string[] {
  const frames: string[] = [];
  frames.push(
    encodeSseData({
      type: 'response.created',
      response: {
        id: standardResponse.id,
        object: standardResponse.object,
        status: 'in_progress',
        model: standardResponse.model,
        output: []
      }
    })
  );

  for (let outputIndex = 0; outputIndex < standardResponse.output.length; outputIndex += 1) {
    const item = standardResponse.output[outputIndex];
    if (!item) {
      continue;
    }

    if (item.type === 'message') {
      frames.push(...buildOpenAIResponsesMessageStreamFrames(item, outputIndex));
      continue;
    }

    if (item.type === 'reasoning') {
      frames.push(...buildOpenAIResponsesReasoningStreamFrames(item, outputIndex));
      continue;
    }

    frames.push(...buildOpenAIResponsesFunctionCallStreamFrames(item, outputIndex));
  }

  frames.push(
    encodeSseData({
      type: 'response.completed',
      response: normalizeOpenAIResponsesCompletedResponse({ ...standardResponse })
    })
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

function buildOpenAIResponsesMessageStreamFrames(
  item: StandardResponse['output'][number] & { type: 'message' },
  outputIndex: number
): string[] {
  const frames: string[] = [];
  const text = item.content
    .map((content) => (content.type === 'output_text' ? content.text : ''))
    .filter(Boolean)
    .join('\n');

  frames.push(
    encodeSseData({
      type: 'response.output_item.added',
      output_index: outputIndex,
      item: {
        id: item.id,
        type: 'message',
        role: item.role,
        status: 'in_progress',
        content: []
      }
    })
  );

  frames.push(
    encodeSseData({
      type: 'response.content_part.added',
      output_index: outputIndex,
      item_id: item.id,
      content_index: 0,
      part: {
        type: 'output_text',
        text: '',
        annotations: []
      }
    })
  );

  if (text) {
    frames.push(
      encodeSseData({
        type: 'response.output_text.delta',
        delta: text,
        output_index: outputIndex,
        content_index: 0,
        item_id: item.id
      })
    );
  }

  frames.push(
    encodeSseData({
      type: 'response.output_text.done',
      text,
      output_index: outputIndex,
      content_index: 0,
      item_id: item.id
    })
  );
  frames.push(
    encodeSseData({
      type: 'response.content_part.done',
      output_index: outputIndex,
      item_id: item.id,
      content_index: 0,
      part: {
        type: 'output_text',
        text,
        annotations: []
      }
    })
  );
  frames.push(
    encodeSseData({
      type: 'response.output_item.done',
      output_index: outputIndex,
      item
    })
  );

  return frames;
}

function buildOpenAIResponsesReasoningStreamFrames(
  item: StandardResponseReasoning,
  outputIndex: number
): string[] {
  const frames: string[] = [];
  frames.push(
    encodeSseData({
      type: 'response.output_item.added',
      output_index: outputIndex,
      item: buildOpenAIResponsesReasoningItem(item, 'in_progress')
    })
  );

  for (let summaryIndex = 0; summaryIndex < item.summary.length; summaryIndex += 1) {
    const summary = item.summary[summaryIndex];
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_part.added',
        item_id: item.id,
        output_index: outputIndex,
        summary_index: summaryIndex,
        part: {
          type: 'summary_text',
          text: ''
        }
      })
    );
    if (summary.text) {
      frames.push(
        encodeSseData({
          type: 'response.reasoning_summary_text.delta',
          item_id: item.id,
          output_index: outputIndex,
          summary_index: summaryIndex,
          delta: summary.text
        })
      );
    }
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_text.done',
        item_id: item.id,
        output_index: outputIndex,
        summary_index: summaryIndex,
        text: summary.text
      })
    );
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_part.done',
        item_id: item.id,
        output_index: outputIndex,
        summary_index: summaryIndex,
        part: summary
      })
    );
  }

  const content = item.content || [];
  for (let contentIndex = 0; contentIndex < content.length; contentIndex += 1) {
    const part = content[contentIndex];
    if (part.text) {
      frames.push(
        encodeSseData({
          type: 'response.reasoning_text.delta',
          item_id: item.id,
          output_index: outputIndex,
          content_index: contentIndex,
          delta: part.text
        })
      );
    }
    frames.push(
      encodeSseData({
        type: 'response.reasoning_text.done',
        item_id: item.id,
        output_index: outputIndex,
        content_index: contentIndex,
        text: part.text
      })
    );
  }

  frames.push(
    encodeSseData({
      type: 'response.output_item.done',
      output_index: outputIndex,
      item
    })
  );
  return frames;
}

function buildOpenAIResponsesReasoningItem(
  item: StandardResponseReasoning,
  status: 'in_progress' | 'completed'
): Record<string, unknown> {
  return {
    id: item.id,
    type: 'reasoning',
    summary: status === 'in_progress' ? [] : item.summary,
    ...(status === 'completed' && item.content ? { content: item.content } : {}),
    ...(item.encrypted_content ? { encrypted_content: item.encrypted_content } : {}),
    status
  };
}

function buildOpenAIResponsesFunctionCallStreamFrames(
  item: StandardResponse['output'][number] & { type: 'function_call' },
  outputIndex: number
): string[] {
  const frames: string[] = [];
  const inProgressItem = {
    ...item,
    arguments: '',
    status: 'in_progress'
  };

  frames.push(
    encodeSseData({
      type: 'response.output_item.added',
      output_index: outputIndex,
      item: inProgressItem
    })
  );

  if (item.arguments) {
    frames.push(
      encodeSseData({
        type: 'response.function_call_arguments.delta',
        output_index: outputIndex,
        item_id: item.id,
        delta: item.arguments
      })
    );
  }

  frames.push(
    encodeSseData({
      type: 'response.function_call_arguments.done',
      output_index: outputIndex,
      item_id: item.id,
      name: item.name,
      ...(item.namespace ? { namespace: item.namespace } : {}),
      arguments: item.arguments
    })
  );
  frames.push(
    encodeSseData({
      type: 'response.output_item.done',
      output_index: outputIndex,
      item
    })
  );

  return frames;
}

function collectStandardResponseToolCallsForOpenAIChat(
  standardResponse: StandardResponse
): Array<{ index: number; id: string; name: string; argumentsJson: string }> {
  const toolCalls: Array<{ index: number; id: string; name: string; argumentsJson: string }> = [];
  let index = 0;
  for (const item of standardResponse.output) {
    if (item.type !== 'function_call') {
      continue;
    }

    toolCalls.push({
      index,
      id: item.call_id || item.id,
      name: item.name,
      argumentsJson: item.arguments
    });
    index += 1;
  }

  return toolCalls;
}

function collectStandardResponseReasoningText(standardResponse: StandardResponse): string {
  return standardResponse.output
    .filter((item): item is StandardResponseReasoning => item.type === 'reasoning')
    .flatMap((item) => item.content || [])
    .map((content) => content.text)
    .filter(Boolean)
    .join('\n')
    .trim();
}

function buildAnthropicMessagesStreamFrames(standardResponse: StandardResponse): string[] {
  const frames: string[] = [];
  const anthropicPayload = formatAnthropicMessagesResponse(standardResponse);

  frames.push(
    encodeSseEvent('message_start', {
      type: 'message_start',
      message: {
        id: standardResponse.id,
        type: 'message',
        role: 'assistant',
        model: standardResponse.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: buildAnthropicMessageStartUsage(standardResponse.usage)
      }
    })
  );

  const contentBlocks = Array.isArray(anthropicPayload.content) ? anthropicPayload.content : [];
  for (let index = 0; index < contentBlocks.length; index += 1) {
    frames.push(...buildAnthropicStreamContentBlockFrames(index, contentBlocks[index]));
  }

  frames.push(
    encodeSseEvent('message_delta', {
      type: 'message_delta',
      delta: {
        stop_reason: mapFinishReasonToAnthropic(standardResponse.finish_reason),
        stop_sequence: null
      },
      usage: buildAnthropicMessageDeltaUsageFromStandard(standardResponse.usage)
    })
  );

  frames.push(
    encodeSseEvent('message_stop', {
      type: 'message_stop'
    })
  );
  return frames;
}

function buildAnthropicStreamContentBlockFrames(index: number, block: unknown): string[] {
  if (!isObject(block)) {
    return [];
  }

  const type = asString(block.type);
  if (type === 'thinking') {
    const thinking = asString(block.thinking) || '';
    const signature = asString(block.signature);
    const frames = [
      encodeSseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'thinking',
          thinking: ''
        }
      })
    ];
    if (thinking) {
      frames.push(
        encodeSseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: {
            type: 'thinking_delta',
            thinking
          }
        })
      );
    }
    if (signature) {
      frames.push(
        encodeSseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: {
            type: 'signature_delta',
            signature
          }
        })
      );
    }
    frames.push(buildAnthropicContentBlockStopFrame(index));
    return frames;
  }

  if (type === 'redacted_thinking') {
    return [
      encodeSseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'redacted_thinking',
          data: asString(block.data) || ''
        }
      }),
      buildAnthropicContentBlockStopFrame(index)
    ];
  }

  if (type === 'tool_use') {
    const inputJson = normalizeToolArguments(block.input);
    const frames = [
      encodeSseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: asString(block.id) || `toolu_${randomUUID().replace(/-/g, '')}`,
          name: asString(block.name) || 'tool',
          input: {}
        }
      })
    ];
    if (inputJson && inputJson !== '{}') {
      frames.push(
        encodeSseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: {
            type: 'input_json_delta',
            partial_json: inputJson
          }
        })
      );
    }
    frames.push(buildAnthropicContentBlockStopFrame(index));
    return frames;
  }

  const text = asString(block.text) || '';
  return [
    encodeSseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: {
        type: 'text',
        text: ''
      }
    }),
    ...(text
      ? [
          encodeSseEvent('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: {
              type: 'text_delta',
              text
            }
          })
        ]
      : []),
    buildAnthropicContentBlockStopFrame(index)
  ];
}

function buildGeminiStreamFrames(standardResponse: StandardResponse): string[] {
  return [encodeSseData(formatGeminiGenerateContentResponse(standardResponse))];
}

async function* relayAnthropicMessagesFromOpenAIStream(upstreamResponse: Response): AsyncGenerator<string> {
  const state: AnthropicRelayState = {
    started: false,
    finished: false,
    messageId: `msg_${randomUUID()}`,
    model: 'unknown',
    outputTokens: 0,
    nextBlockIndex: 0,
    pendingToolCalls: new Map()
  };

  for await (const chunk of parseSseChunks(upstreamResponse)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }

    if (data === '[DONE]') {
      yield* flushPendingAnthropicToolCalls(state);
      yield* finalizeAnthropicRelay(state);
      return;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    const emittedFrames = isOpenAIResponsesStreamEvent(payload)
      ? emitAnthropicFramesFromOpenAIResponsesEvent(state, payload)
      : emitAnthropicFramesFromOpenAIChatChunk(state, payload);

    for (const frame of emittedFrames) {
      yield frame;
    }

    if (state.finished) {
      return;
    }
  }

  if (!state.finished) {
    yield* flushPendingAnthropicToolCalls(state);
    yield* finalizeAnthropicRelay(state);
  }
}

async function* relayOpenAIResponsesFromOpenAIStream(
  upstreamResponse: Response,
  tools?: unknown[]
): AsyncGenerator<string> {
  const state: OpenAIResponsesRelayState = {
    started: false,
    finished: false,
    responseId: `resp_${randomUUID()}`,
    model: 'unknown',
    outputText: '',
    reasoningItemId: `rs_${randomUUID().replace(/-/g, '')}`,
    reasoningText: '',
    reasoningSummaryText: '',
    reasoningItemStarted: false,
    reasoningSummaryStarted: false,
    messageItemId: `msg_${randomUUID()}`,
    messageOutputIndex: undefined,
    messageItemStarted: false,
    messageContentStarted: false,
    pendingToolCalls: new Map(),
    usedOutputIndices: new Set(),
    nextOutputIndex: 0,
    usage: {}
  };

  for await (const chunk of parseSseChunks(upstreamResponse)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }

    if (data === '[DONE]') {
      if (!state.finished) {
        yield* finalizeOpenAIResponsesRelay(state);
      }
      yield 'data: [DONE]\n\n';
      return;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    const emittedFrames = isOpenAIResponsesStreamEvent(payload)
      ? emitOpenAIResponsesFramesFromResponsesEvent(state, payload)
      : emitOpenAIResponsesFramesFromChatChunk(state, payload, tools);
    for (const frame of emittedFrames) {
      yield frame;
    }
  }

  if (!state.finished) {
    yield* finalizeOpenAIResponsesRelay(state);
    yield 'data: [DONE]\n\n';
  }
}

async function* relayGeminiStreamFromOpenAIStream(upstreamResponse: Response): AsyncGenerator<string> {
  const state: GeminiRelayState = {
    model: 'unknown',
    outputText: '',
    usage: {},
    emittedAnyDelta: false,
    emittedFinal: false,
    pendingToolCalls: new Map()
  };

  for await (const chunk of parseSseChunks(upstreamResponse)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }

    if (data === '[DONE]') {
      if (!state.emittedFinal) {
        yield* flushPendingGeminiToolCalls(state);
        const frame = buildGeminiFinalFrame(state);
        if (frame) {
          yield frame;
        }
      }
      return;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    const emittedFrames = isOpenAIResponsesStreamEvent(payload)
      ? emitGeminiFramesFromOpenAIResponsesEvent(state, payload)
      : emitGeminiFramesFromOpenAIChatChunk(state, payload);
    for (const frame of emittedFrames) {
      yield frame;
    }
  }

  if (!state.emittedFinal) {
    yield* flushPendingGeminiToolCalls(state);
    const frame = buildGeminiFinalFrame(state);
    if (frame) {
      yield frame;
    }
  }
}

async function* relayOpenAIChatFromUpstreamStream(upstreamResponse: Response): AsyncGenerator<string> {
  const state: OpenAIChatRelayState = {
    started: false,
    finished: false,
    id: `chatcmpl_${randomUUID()}`,
    model: 'unknown',
    created: Math.floor(Date.now() / 1000),
    emittedTextDelta: false,
    nextToolCallIndex: 0,
    usage: {}
  };

  for await (const chunk of parseSseChunks(upstreamResponse)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }

    if (data === '[DONE]') {
      if (!state.finished) {
        yield* finalizeOpenAIChatRelay(state);
      }
      yield 'data: [DONE]\n\n';
      return;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }

    if (!isObject(payload)) {
      continue;
    }

    const emittedFrames = isOpenAIResponsesStreamEvent(payload)
      ? emitOpenAIChatFramesFromOpenAIResponsesEvent(state, payload)
      : emitOpenAIChatFramesFromAnthropicEvent(state, payload);
    for (const frame of emittedFrames) {
      yield frame;
    }

    if (state.finished) {
      yield 'data: [DONE]\n\n';
      return;
    }
  }

  if (!state.finished) {
    yield* finalizeOpenAIChatRelay(state);
    yield 'data: [DONE]\n\n';
  }
}

function emitOpenAIChatFramesFromAnthropicEvent(
  state: OpenAIChatRelayState,
  payload: Record<string, unknown>
): string[] {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return [];
  }

  if (eventType === 'message_start') {
    const message = isObject(payload.message) ? payload.message : undefined;
    const id = asString(message?.id);
    if (id) {
      state.id = id;
    }

    const model = asString(message?.model);
    if (model) {
      state.model = model;
    }

    updateOpenAIChatRelayUsageFromAnthropic(state, isObject(message?.usage) ? message.usage : undefined);
    return ensureOpenAIChatRelayStarted(state);
  }

  if (eventType === 'content_block_delta') {
    const delta = isObject(payload.delta) ? payload.delta : undefined;
    const deltaType = asString(delta?.type);
    if (deltaType === 'text_delta') {
      const text = asString(delta?.text) || '';
      if (!text) {
        return [];
      }

      const frames = ensureOpenAIChatRelayStarted(state);
      frames.push(buildOpenAIChatRelayDeltaFrame(state, { content: text }));
      return frames;
    }

    if (deltaType === 'input_json_delta') {
      const partialJson = asString(delta?.partial_json) || '';
      if (!partialJson || !state.activeAnthropicToolCall) {
        return [];
      }

      const frames = ensureOpenAIChatRelayStarted(state);
      frames.push(buildOpenAIChatAnthropicToolDeltaFrame(state, state.activeAnthropicToolCall, partialJson));
      return frames;
    }

    if (deltaType === 'thinking_delta') {
      const thinking = asString(delta?.thinking) || '';
      if (!thinking) {
        return [];
      }

      const frames = ensureOpenAIChatRelayStarted(state);
      frames.push(buildOpenAIChatRelayDeltaFrame(state, { reasoning_content: thinking }));
      return frames;
    }

    return [];
  }

  if (eventType === 'content_block_start') {
    const contentBlock = isObject(payload.content_block) ? payload.content_block : undefined;
    if (asString(contentBlock?.type) !== 'tool_use') {
      return [];
    }

    const name = asString(contentBlock?.name);
    if (!name) {
      return [];
    }

    const blockIndex = asNumber(payload.index);
    if (blockIndex === undefined) {
      return [];
    }

    const toolCall: PendingOpenAIChatAnthropicToolCall = {
      blockIndex,
      toolIndex: state.nextToolCallIndex,
      id: asString(contentBlock?.id) || `call_${randomUUID().replace(/-/g, '')}`,
      name,
      started: true
    };
    state.nextToolCallIndex += 1;
    state.activeAnthropicToolCall = toolCall;

    const frames = ensureOpenAIChatRelayStarted(state);
    frames.push(buildOpenAIChatAnthropicToolDeltaFrame(state, toolCall, ''));
    return frames;
  }

  if (eventType === 'content_block_stop') {
    const blockIndex = asNumber(payload.index);
    if (
      blockIndex !== undefined &&
      state.activeAnthropicToolCall &&
      state.activeAnthropicToolCall.blockIndex === blockIndex
    ) {
      state.activeAnthropicToolCall = undefined;
    }
    return [];
  }

  if (eventType === 'message_delta') {
    updateOpenAIChatRelayUsageFromAnthropic(state, isObject(payload.usage) ? payload.usage : undefined);
    const delta = isObject(payload.delta) ? payload.delta : undefined;
    const stopReason = asString(delta?.stop_reason);
    if (stopReason) {
      state.finishReason = stopReason;
      return finalizeOpenAIChatRelay(state);
    }

    return [];
  }

  if (eventType === 'message_stop') {
    return finalizeOpenAIChatRelay(state);
  }

  return [];
}

function emitOpenAIChatFramesFromOpenAIResponsesEvent(
  state: OpenAIChatRelayState,
  payload: Record<string, unknown>
): string[] {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return [];
  }

  if (eventType === 'response.created') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateOpenAIChatRelayIdentityFromOpenAIResponses(state, response);
    updateOpenAIChatRelayUsageFromOpenAIResponses(state, response);
    return ensureOpenAIChatRelayStarted(state);
  }

  if (eventType === 'response.output_text.delta') {
    const deltaText = asString(payload.delta) || '';
    if (!deltaText) {
      return [];
    }

    state.emittedTextDelta = true;
    const frames = ensureOpenAIChatRelayStarted(state);
    frames.push(buildOpenAIChatRelayDeltaFrame(state, { content: deltaText }));
    return frames;
  }

  if (eventType === 'response.output_text.done') {
    if (state.emittedTextDelta) {
      return [];
    }

    const doneText = asString(payload.text) || '';
    if (!doneText) {
      return [];
    }

    state.emittedTextDelta = true;
    const frames = ensureOpenAIChatRelayStarted(state);
    frames.push(buildOpenAIChatRelayDeltaFrame(state, { content: doneText }));
    return frames;
  }

  if (eventType === 'response.completed') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateOpenAIChatRelayIdentityFromOpenAIResponses(state, response);
    updateOpenAIChatRelayUsageFromOpenAIResponses(state, response);
    state.finishReason = asString(response?.finish_reason) || extractResponsesFinishReason(response);

    const frames = ensureOpenAIChatRelayStarted(state);
    if (!state.emittedTextDelta) {
      const outputText = asString(response?.output_text) || extractOpenAIResponsesOutputText(response?.output);
      if (outputText) {
        state.emittedTextDelta = true;
        frames.push(buildOpenAIChatRelayDeltaFrame(state, { content: outputText }));
      }
    }

    frames.push(...finalizeOpenAIChatRelay(state));
    return frames;
  }

  return [];
}

function ensureOpenAIChatRelayStarted(state: OpenAIChatRelayState): string[] {
  if (state.started) {
    return [];
  }

  state.started = true;
  return [
    encodeSseData({
      id: state.id,
      object: 'chat.completion.chunk',
      created: state.created,
      model: state.model,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant'
          }
        }
      ]
    })
  ];
}

function buildOpenAIChatRelayDeltaFrame(
  state: OpenAIChatRelayState,
  delta: Record<string, unknown>
): string {
  return encodeSseData({
    id: state.id,
    object: 'chat.completion.chunk',
    created: state.created,
    model: state.model,
    choices: [
      {
        index: 0,
        delta
      }
    ]
  });
}

function buildOpenAIChatAnthropicToolDeltaFrame(
  state: OpenAIChatRelayState,
  toolCall: PendingOpenAIChatAnthropicToolCall,
  argumentsChunk: string
): string {
  return buildOpenAIChatRelayDeltaFrame(state, {
    tool_calls: [
      {
        index: toolCall.toolIndex,
        id: toolCall.id,
        type: 'function',
        function: {
          name: toolCall.name,
          arguments: argumentsChunk
        }
      }
    ]
  });
}

function finalizeOpenAIChatRelay(state: OpenAIChatRelayState): string[] {
  if (state.finished) {
    return [];
  }

  const frames = ensureOpenAIChatRelayStarted(state);
  const usage: Record<string, unknown> = {};
  if (state.usage.promptTokens !== undefined) {
    usage.prompt_tokens = state.usage.promptTokens;
  }
  if (state.usage.completionTokens !== undefined) {
    usage.completion_tokens = state.usage.completionTokens;
  }

  const totalTokens =
    state.usage.totalTokens !== undefined
      ? state.usage.totalTokens
      : state.usage.promptTokens !== undefined && state.usage.completionTokens !== undefined
        ? state.usage.promptTokens + state.usage.completionTokens
        : undefined;
  if (totalTokens !== undefined) {
    usage.total_tokens = totalTokens;
  }

  if (
    state.usage.cachedPromptTokens !== undefined ||
    state.usage.cacheCreationPromptTokens !== undefined
  ) {
    const promptTokenDetails: Record<string, unknown> = {};
    if (state.usage.cachedPromptTokens !== undefined) {
      promptTokenDetails.cached_tokens = state.usage.cachedPromptTokens;
    }
    if (state.usage.cacheCreationPromptTokens !== undefined) {
      promptTokenDetails.cache_creation_tokens = state.usage.cacheCreationPromptTokens;
    }
    usage.prompt_tokens_details = promptTokenDetails;
  }

  const finalChunk: Record<string, unknown> = {
    id: state.id,
    object: 'chat.completion.chunk',
    created: state.created,
    model: state.model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: mapFinishReasonToOpenAI(state.finishReason)
      }
    ]
  };

  if (Object.keys(usage).length > 0) {
    finalChunk.usage = usage;
  }

  frames.push(encodeSseData(finalChunk));
  state.finished = true;
  return frames;
}

function updateOpenAIChatRelayUsageFromAnthropic(
  state: OpenAIChatRelayState,
  usage: Record<string, unknown> | undefined
) {
  if (!usage) {
    return;
  }

  const promptTokens = asNumber(usage.input_tokens) ?? asNumber(usage.prompt_tokens);
  if (promptTokens !== undefined) {
    state.usage.promptTokens = promptTokens;
  }

  const completionTokens = asNumber(usage.output_tokens) ?? asNumber(usage.completion_tokens);
  if (completionTokens !== undefined) {
    state.usage.completionTokens = completionTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    state.usage.totalTokens = totalTokens;
  }

  const cachedPromptTokens =
    asNumber(usage.cache_read_input_tokens) ??
    asNumber(usage.cache_read_tokens) ??
    asNumber(isObject(usage.input_tokens_details) ? usage.input_tokens_details.cached_tokens : undefined);
  if (cachedPromptTokens !== undefined) {
    state.usage.cachedPromptTokens = cachedPromptTokens;
  }

  const cacheCreationPromptTokens =
    asNumber(usage.cache_creation_input_tokens) ??
    asNumber(usage.cache_creation_tokens) ??
    asNumber(usage.cache_write_tokens) ??
    asNumber(isObject(usage.input_tokens_details) ? usage.input_tokens_details.cache_creation_tokens : undefined);
  if (cacheCreationPromptTokens !== undefined) {
    state.usage.cacheCreationPromptTokens = cacheCreationPromptTokens;
  }
}

function updateOpenAIChatRelayIdentityFromOpenAIResponses(
  state: OpenAIChatRelayState,
  response: Record<string, unknown> | undefined
) {
  if (!response) {
    return;
  }

  const id = asString(response.id);
  if (id) {
    state.id = id;
  }

  const model = asString(response.model);
  if (model) {
    state.model = model;
  }
}

function updateOpenAIChatRelayUsageFromOpenAIResponses(
  state: OpenAIChatRelayState,
  response: Record<string, unknown> | undefined
) {
  const usage = isObject(response?.usage) ? response.usage : undefined;
  if (!usage) {
    return;
  }

  const promptTokens = asNumber(usage.input_tokens) ?? asNumber(usage.prompt_tokens);
  if (promptTokens !== undefined) {
    state.usage.promptTokens = promptTokens;
  }

  const completionTokens = asNumber(usage.output_tokens) ?? asNumber(usage.completion_tokens);
  if (completionTokens !== undefined) {
    state.usage.completionTokens = completionTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    state.usage.totalTokens = totalTokens;
  }

  const inputDetails = isObject(usage.input_tokens_details)
    ? usage.input_tokens_details
    : isObject(usage.prompt_tokens_details)
      ? usage.prompt_tokens_details
      : undefined;
  const cachedPromptTokens =
    asNumber(inputDetails?.cached_tokens) ??
    asNumber(usage.cache_read_input_tokens) ??
    asNumber(usage.cache_read_tokens);
  if (cachedPromptTokens !== undefined) {
    state.usage.cachedPromptTokens = cachedPromptTokens;
  }

  const cacheCreationPromptTokens =
    asNumber(inputDetails?.cache_creation_tokens) ??
    asNumber(usage.cache_creation_input_tokens) ??
    asNumber(usage.cache_creation_tokens) ??
    asNumber(usage.cache_write_tokens);
  if (cacheCreationPromptTokens !== undefined) {
    state.usage.cacheCreationPromptTokens = cacheCreationPromptTokens;
  }
}

function emitGeminiFramesFromOpenAIChatChunk(
  state: GeminiRelayState,
  payload: Record<string, unknown>
): string[] {
  const model = asString(payload.model);
  if (model) {
    state.model = model;
  }

  updateGeminiRelayUsageFromOpenAIChat(state, isObject(payload.usage) ? payload.usage : undefined);

  const firstChoice = Array.isArray(payload.choices) && isObject(payload.choices[0]) ? payload.choices[0] : undefined;
  const delta = isObject(firstChoice?.delta) ? firstChoice.delta : undefined;
  const deltaText = asString(delta?.content) || asString(delta?.reasoning_content) || '';
  const finishReason = asString(firstChoice?.finish_reason);
  if (finishReason) {
    state.finishReason = finishReason;
  }

  const frames: string[] = [];
  if (deltaText) {
    state.outputText += deltaText;
    state.emittedAnyDelta = true;
    frames.push(buildGeminiDeltaFrame(state.model, deltaText));
  }

  collectOpenAIChatToolCallsForGemini(state, delta?.tool_calls);

  if (finishReason && !state.emittedFinal) {
    frames.push(...flushPendingGeminiToolCalls(state));
    const finalFrame = buildGeminiFinalFrame(state);
    if (finalFrame) {
      frames.push(finalFrame);
    }
  }

  return frames;
}

function emitGeminiFramesFromOpenAIResponsesEvent(
  state: GeminiRelayState,
  payload: Record<string, unknown>
): string[] {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return [];
  }

  if (eventType === 'response.created') {
    const response = isObject(payload.response) ? payload.response : undefined;
    const model = asString(response?.model);
    if (model) {
      state.model = model;
    }
    return [];
  }

  if (eventType === 'response.output_text.delta') {
    const deltaText = asString(payload.delta) || '';
    if (!deltaText) {
      return [];
    }

    state.outputText += deltaText;
    state.emittedAnyDelta = true;
    return [buildGeminiDeltaFrame(state.model, deltaText)];
  }

  if (eventType === 'response.output_text.done') {
    const text = asString(payload.text);
    if (text) {
      state.outputText = text;
    }
    return [];
  }

  if (eventType === 'response.completed') {
    const response = isObject(payload.response) ? payload.response : undefined;
    const model = asString(response?.model);
    if (model) {
      state.model = model;
    }
    const outputText = asString(response?.output_text);
    if (outputText && !state.emittedAnyDelta) {
      state.outputText = outputText;
      state.emittedAnyDelta = true;
    }
    state.finishReason =
      asString(response?.finish_reason) ||
      extractResponsesFinishReason(response) ||
      state.finishReason;
    updateGeminiRelayUsageFromOpenAIResponses(state, response);
    collectOpenAIResponsesToolCallsForGemini(state, response);

    if (state.emittedFinal) {
      return [];
    }

    const frames = flushPendingGeminiToolCalls(state);
    const finalFrame = buildGeminiFinalFrame(state);
    if (finalFrame) {
      frames.push(finalFrame);
    }

    return frames;
  }

  return [];
}

function buildGeminiDeltaFrame(model: string, text: string): string {
  return encodeSseData({
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: [{ text }]
        }
      }
    ],
    modelVersion: model
  });
}

function collectOpenAIChatToolCallsForGemini(state: GeminiRelayState, rawToolCalls: unknown) {
  if (!Array.isArray(rawToolCalls)) {
    return;
  }

  for (let position = 0; position < rawToolCalls.length; position += 1) {
    const rawToolCall = rawToolCalls[position];
    if (!isObject(rawToolCall)) {
      continue;
    }

    const indexValue = asNumber(rawToolCall.index);
    const toolIndex = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : position;
    const functionPayload = isObject(rawToolCall.function) ? rawToolCall.function : undefined;
    const name = asString(functionPayload?.name) || asString(rawToolCall.name);
    const argumentsPatch = readOpenAIChatToolArgumentsPatch(functionPayload, rawToolCall);

    mergePendingGeminiToolCall(
      state,
      toolIndex,
      {
        name,
        argumentsJson: argumentsPatch?.value
      },
      argumentsPatch?.append ?? true
    );
  }
}

function collectOpenAIResponsesToolCallsForGemini(
  state: GeminiRelayState,
  response: Record<string, unknown> | undefined
) {
  if (!response || !Array.isArray(response.output)) {
    return;
  }

  let fallbackIndex = state.pendingToolCalls.size;
  for (const outputItem of response.output) {
    if (!isObject(outputItem)) {
      continue;
    }

    const outputType = asString(outputItem.type);
    if (outputType !== 'function_call' && outputType !== 'tool_call') {
      continue;
    }

    const indexValue = asNumber(outputItem.index);
    const toolIndex = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : fallbackIndex++;
    const functionPayload = isObject(outputItem.function) ? outputItem.function : undefined;
    const name = asString(outputItem.name) || asString(functionPayload?.name);
    const argumentsJson = normalizeToolArguments(
      outputItem.arguments ?? functionPayload?.arguments ?? outputItem.input
    );

    mergePendingGeminiToolCall(
      state,
      toolIndex,
      {
        name,
        argumentsJson
      },
      false
    );
  }
}

function mergePendingGeminiToolCall(
  state: GeminiRelayState,
  index: number,
  patch: { name?: string; argumentsJson?: string },
  appendArguments: boolean
): PendingGeminiToolCall {
  const existing = state.pendingToolCalls.get(index);
  const pending: PendingGeminiToolCall = existing || {
    index,
    name: '',
    argumentsJson: ''
  };

  if (patch.name) {
    pending.name = patch.name;
  }

  if (patch.argumentsJson !== undefined) {
    pending.argumentsJson = appendArguments ? pending.argumentsJson + patch.argumentsJson : patch.argumentsJson;
  }

  state.pendingToolCalls.set(index, pending);
  return pending;
}

function flushPendingGeminiToolCalls(state: GeminiRelayState): string[] {
  if (state.pendingToolCalls.size === 0) {
    return [];
  }

  const frames: string[] = [];
  const toolCalls = [...state.pendingToolCalls.values()].sort((a, b) => a.index - b.index);
  for (const toolCall of toolCalls) {
    if (!toolCall.name) {
      continue;
    }

    frames.push(
      buildGeminiFunctionCallFrame(
        state.model,
        toolCall.name,
        parseGeminiFunctionArguments(toolCall.argumentsJson)
      )
    );
  }

  state.pendingToolCalls.clear();
  if (frames.length > 0) {
    state.emittedAnyDelta = true;
  }

  return frames;
}

function buildGeminiFunctionCallFrame(model: string, name: string, args: Record<string, unknown>): string {
  return encodeSseData({
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              functionCall: {
                name,
                args
              }
            }
          ]
        }
      }
    ],
    modelVersion: model
  });
}

function parseGeminiFunctionArguments(argumentsJson: string): Record<string, unknown> {
  const trimmed = argumentsJson.trim();
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

function buildGeminiFinalFrame(state: GeminiRelayState): string | undefined {
  if (state.emittedFinal) {
    return undefined;
  }

  state.emittedFinal = true;
  const candidate: Record<string, unknown> = {
    index: 0,
    content: {
      role: 'model',
      parts: []
    }
  };

  if (state.finishReason) {
    candidate.finishReason = mapFinishReasonToGemini(state.finishReason);
  }

  const payload: Record<string, unknown> = {
    candidates: [candidate],
    modelVersion: state.model
  };

  const usageMetadata = buildGeminiUsageMetadata(state.usage);
  if (usageMetadata) {
    payload.usageMetadata = usageMetadata;
  }

  return encodeSseData(payload);
}

function updateGeminiRelayUsageFromOpenAIChat(
  state: GeminiRelayState,
  usage: Record<string, unknown> | undefined
) {
  if (!usage) {
    return;
  }

  const promptTokens = asNumber(usage.prompt_tokens);
  if (promptTokens !== undefined) {
    state.usage.promptTokenCount = promptTokens;
  }

  const completionTokens = asNumber(usage.completion_tokens);
  if (completionTokens !== undefined) {
    state.usage.candidatesTokenCount = completionTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    state.usage.totalTokenCount = totalTokens;
  }

  const promptDetails = isObject(usage.prompt_tokens_details) ? usage.prompt_tokens_details : undefined;
  const cachedTokens = asNumber(promptDetails?.cached_tokens);
  if (cachedTokens !== undefined) {
    state.usage.cachedContentTokenCount = cachedTokens;
  }
}

function updateGeminiRelayUsageFromOpenAIResponses(
  state: GeminiRelayState,
  response: Record<string, unknown> | undefined
) {
  const usage = isObject(response?.usage) ? response.usage : undefined;
  if (!usage) {
    return;
  }

  const inputTokens = asNumber(usage.input_tokens) ?? asNumber(usage.prompt_tokens);
  if (inputTokens !== undefined) {
    state.usage.promptTokenCount = inputTokens;
  }

  const outputTokens = asNumber(usage.output_tokens) ?? asNumber(usage.completion_tokens);
  if (outputTokens !== undefined) {
    state.usage.candidatesTokenCount = outputTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    state.usage.totalTokenCount = totalTokens;
  }

  const inputDetails = isObject(usage.input_tokens_details)
    ? usage.input_tokens_details
    : isObject(usage.prompt_tokens_details)
      ? usage.prompt_tokens_details
      : undefined;
  const cachedTokens =
    asNumber(inputDetails?.cached_tokens) ??
    asNumber(usage.cache_read_tokens) ??
    asNumber(usage.cache_read_input_tokens);
  if (cachedTokens !== undefined) {
    state.usage.cachedContentTokenCount = cachedTokens;
  }
}

function buildGeminiUsageMetadata(
  usage: GeminiRelayState['usage']
): Record<string, unknown> | undefined {
  const metadata: Record<string, unknown> = {};
  if (usage.promptTokenCount !== undefined) {
    metadata.promptTokenCount = usage.promptTokenCount;
  }
  if (usage.candidatesTokenCount !== undefined) {
    metadata.candidatesTokenCount = usage.candidatesTokenCount;
  }
  if (usage.totalTokenCount !== undefined) {
    metadata.totalTokenCount = usage.totalTokenCount;
  }
  if (usage.cachedContentTokenCount !== undefined) {
    metadata.cachedContentTokenCount = usage.cachedContentTokenCount;
  }

  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

function emitOpenAIResponsesFramesFromResponsesEvent(
  state: OpenAIResponsesRelayState,
  payload: Record<string, unknown>
): string[] {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return [];
  }

  if (eventType === 'response.created') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateOpenAIResponsesRelayIdentity(state, response);
    state.started = true;
  } else if (eventType === 'response.output_text.delta') {
    const delta = asString(payload.delta) || '';
    if (delta) {
      state.outputText += delta;
    }
  } else if (eventType === 'response.output_text.done') {
    const text = asString(payload.text);
    if (text) {
      state.outputText = text;
    }
  } else if (eventType === 'response.completed') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateOpenAIResponsesRelayIdentity(state, response);
    updateOpenAIResponsesRelayUsageFromResponse(state, response);
    const outputText = asString(response?.output_text);
    if (outputText) {
      state.outputText = outputText;
    }
    state.finishReason = asString(response?.finish_reason) || state.finishReason;
    state.finished = true;
  }

  return [encodeSseData(normalizeOpenAIResponsesCompletedEventPayload(payload))];
}

function emitOpenAIResponsesFramesFromChatChunk(
  state: OpenAIResponsesRelayState,
  payload: Record<string, unknown>,
  tools?: unknown[]
): string[] {
  if (state.finished) {
    return [];
  }

  const messageId = asString(payload.id);
  if (messageId && !state.started) {
    state.responseId = messageId;
  }

  const model = asString(payload.model);
  if (model) {
    state.model = model;
  }

  const usage = isObject(payload.usage) ? payload.usage : undefined;
  updateOpenAIResponsesRelayUsageFromChat(state, usage);

  const firstChoice = Array.isArray(payload.choices) && isObject(payload.choices[0]) ? payload.choices[0] : undefined;
  const delta = isObject(firstChoice?.delta) ? firstChoice.delta : undefined;
  const deltaText = asString(delta?.content) || '';
  const reasoningDeltas = collectOpenAIChatReasoningDeltas(delta);
  const finishReason = asString(firstChoice?.finish_reason);
  if (finishReason) {
    state.finishReason = finishReason;
  }

  const frames = ensureOpenAIResponsesRelayStarted(state);

  for (const summaryDelta of reasoningDeltas.summaryDeltas) {
    frames.push(...emitOpenAIResponsesReasoningSummaryDelta(state, summaryDelta));
  }
  for (const textDelta of reasoningDeltas.textDeltas) {
    frames.push(...emitOpenAIResponsesReasoningTextDelta(state, textDelta));
  }
  if (reasoningDeltas.encryptedContent && !state.reasoningEncryptedContent) {
    state.reasoningEncryptedContent = reasoningDeltas.encryptedContent;
    frames.push(...ensureOpenAIResponsesReasoningOutputStarted(state));
  }

  if (deltaText) {
    frames.push(...ensureOpenAIResponsesMessageOutputStarted(state));
    state.outputText += deltaText;
    if (state.messageOutputIndex !== undefined) {
      frames.push(
        encodeSseData({
          type: 'response.output_text.delta',
          delta: deltaText,
          output_index: state.messageOutputIndex,
          content_index: 0,
          item_id: state.messageItemId
        })
      );
    }
  }

  frames.push(...collectOpenAIChatToolCallsForOpenAIResponses(state, delta?.tool_calls, tools));

  if (state.finishReason && usage) {
    frames.push(...finalizeOpenAIResponsesRelay(state));
  }

  return frames;
}

function collectOpenAIChatReasoningDeltas(delta: Record<string, unknown> | undefined): {
  textDeltas: string[];
  summaryDeltas: string[];
  encryptedContent?: string;
} {
  const collected: {
    textDeltas: string[];
    summaryDeltas: string[];
    encryptedContent?: string;
  } = {
    textDeltas: [],
    summaryDeltas: []
  };

  if (!delta) {
    return collected;
  }

  const reasoningDetails = Array.isArray(delta.reasoning_details) ? delta.reasoning_details : [];
  for (const detail of reasoningDetails) {
    if (typeof detail === 'string') {
      if (detail) {
        appendReasoningDeltaIfDistinct(collected.textDeltas, detail);
      }
      continue;
    }

    if (!isObject(detail)) {
      continue;
    }

    const type = asString(detail.type);
    const summary = asString(detail.summary);
    const text = asString(detail.text) || asString(detail.reasoning) || asString(detail.thinking);
    const encryptedContent = asString(detail.encrypted_content) || asString(detail.data);

    if (type === 'reasoning.summary' || (summary && !text)) {
      if (summary || text) {
        collected.summaryDeltas.push(summary || text || '');
      }
      continue;
    }

    if (text) {
      appendReasoningDeltaIfDistinct(collected.textDeltas, text);
    }

    if (encryptedContent && !collected.encryptedContent) {
      collected.encryptedContent = encryptedContent;
    }
  }

  const reasoningText =
    asString(delta.reasoning_content) ||
    asString(delta.reasoning) ||
    asString(delta.thinking);
  if (reasoningText) {
    appendReasoningDeltaIfDistinct(collected.textDeltas, reasoningText);
  }

  return collected;
}

function appendReasoningDeltaIfDistinct(parts: string[], value: string): void {
  if (!value) {
    return;
  }

  const existingText = parts.join('').trim();
  const text = value.trim();
  if (text && (existingText === text || parts.some((part) => part.trim() === text))) {
    return;
  }

  parts.push(value);
}

function ensureOpenAIResponsesReasoningOutputStarted(state: OpenAIResponsesRelayState): string[] {
  const frames = ensureOpenAIResponsesRelayStarted(state);
  if (state.reasoningItemStarted) {
    return frames;
  }

  const outputIndex = allocateOpenAIResponsesOutputIndex(state, 0);
  state.reasoningOutputIndex = outputIndex;
  state.reasoningItemStarted = true;
  frames.push(
    encodeSseData({
      type: 'response.output_item.added',
      output_index: outputIndex,
      item: {
        id: state.reasoningItemId,
        type: 'reasoning',
        summary: [],
        content: [],
        status: 'in_progress'
      }
    })
  );

  return frames;
}

function emitOpenAIResponsesReasoningTextDelta(
  state: OpenAIResponsesRelayState,
  delta: string
): string[] {
  if (!delta) {
    return [];
  }

  const frames = ensureOpenAIResponsesReasoningOutputStarted(state);
  state.reasoningText += delta;
  if (state.reasoningOutputIndex !== undefined) {
    frames.push(
      encodeSseData({
        type: 'response.reasoning_text.delta',
        item_id: state.reasoningItemId,
        output_index: state.reasoningOutputIndex,
        content_index: 0,
        delta
      })
    );
  }

  return frames;
}

function emitOpenAIResponsesReasoningSummaryDelta(
  state: OpenAIResponsesRelayState,
  delta: string
): string[] {
  if (!delta) {
    return [];
  }

  const frames = ensureOpenAIResponsesReasoningOutputStarted(state);
  if (state.reasoningOutputIndex === undefined) {
    return frames;
  }

  if (!state.reasoningSummaryStarted) {
    state.reasoningSummaryStarted = true;
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_part.added',
        item_id: state.reasoningItemId,
        output_index: state.reasoningOutputIndex,
        summary_index: 0,
        part: {
          type: 'summary_text',
          text: ''
        }
      })
    );
  }

  state.reasoningSummaryText += delta;
  frames.push(
    encodeSseData({
      type: 'response.reasoning_summary_text.delta',
      item_id: state.reasoningItemId,
      output_index: state.reasoningOutputIndex,
      summary_index: 0,
      delta
    })
  );

  return frames;
}

function ensureOpenAIResponsesMessageOutputStarted(state: OpenAIResponsesRelayState): string[] {
  const frames = ensureOpenAIResponsesRelayStarted(state);
  if (state.messageItemStarted) {
    return frames;
  }

  const outputIndex = allocateOpenAIResponsesOutputIndex(state, 0);
  state.messageOutputIndex = outputIndex;
  state.messageItemStarted = true;
  state.messageContentStarted = true;

  frames.push(
    encodeSseData({
      type: 'response.output_item.added',
      output_index: outputIndex,
      item: {
        id: state.messageItemId,
        type: 'message',
        role: 'assistant',
        status: 'in_progress',
        content: []
      }
    })
  );
  frames.push(
    encodeSseData({
      type: 'response.content_part.added',
      output_index: outputIndex,
      item_id: state.messageItemId,
      content_index: 0,
      part: {
        type: 'output_text',
        text: '',
        annotations: []
      }
    })
  );

  return frames;
}

function collectOpenAIChatToolCallsForOpenAIResponses(
  state: OpenAIResponsesRelayState,
  rawToolCalls: unknown,
  tools?: unknown[]
): string[] {
  if (!Array.isArray(rawToolCalls)) {
    return [];
  }

  const frames = ensureOpenAIResponsesRelayStarted(state);
  for (let position = 0; position < rawToolCalls.length; position += 1) {
    const rawToolCall = rawToolCalls[position];
    if (!isObject(rawToolCall)) {
      continue;
    }

    const indexValue = asNumber(rawToolCall.index);
    const toolIndex = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : position;
    const functionPayload = isObject(rawToolCall.function) ? rawToolCall.function : undefined;
    const id = asString(rawToolCall.id);
    const name = asString(functionPayload?.name) || asString(rawToolCall.name);
    const argumentsPatch = readOpenAIChatToolArgumentsPatch(functionPayload, rawToolCall);
    const argumentsChunk = argumentsPatch?.value || '';
    const splitName = name ? splitNamespacedToolCallName(name, tools) : undefined;

    const toolCall = mergePendingOpenAIResponsesToolCall(
      state,
      toolIndex,
      {
        id,
        callId: id,
        name: splitName?.name,
        namespace: splitName?.namespace,
        argumentsJson: argumentsChunk
      },
      argumentsPatch?.append ?? true
    );

    if (!toolCall.added) {
      frames.push(
        encodeSseData({
          type: 'response.output_item.added',
          output_index: toolCall.outputIndex,
          item: buildOpenAIResponsesFunctionCallItem(toolCall, 'in_progress')
        })
      );
      toolCall.added = true;
      toolCall.done = false;
    }

    if (argumentsChunk) {
      frames.push(
        encodeSseData({
          type: 'response.function_call_arguments.delta',
          output_index: toolCall.outputIndex,
          item_id: toolCall.id,
          delta: argumentsChunk
        })
      );
      toolCall.emittedArgumentsLength = toolCall.argumentsJson.length;
    }
  }

  return frames;
}

function mergePendingOpenAIResponsesToolCall(
  state: OpenAIResponsesRelayState,
  index: number,
  patch: { id?: string; callId?: string; name?: string; namespace?: string; argumentsJson?: string },
  appendArguments: boolean
): PendingOpenAIResponsesToolCall {
  const existing = state.pendingToolCalls.get(index);
  const pending: PendingOpenAIResponsesToolCall = existing || {
    index,
    outputIndex: allocateOpenAIResponsesOutputIndex(state, index),
    id: `fc_${randomUUID().replace(/-/g, '')}`,
    callId: `call_${randomUUID().replace(/-/g, '')}`,
    name: '',
    argumentsJson: '',
    emittedArgumentsLength: 0,
    added: false,
    done: false
  };

  if (patch.id) {
    pending.id = patch.id;
  }
  if (patch.callId) {
    pending.callId = patch.callId;
  } else if (!pending.callId) {
    pending.callId = pending.id;
  }
  if (patch.name) {
    pending.name = patch.name;
  }
  if (patch.namespace) {
    pending.namespace = patch.namespace;
  }

  if (patch.argumentsJson !== undefined) {
    pending.argumentsJson = appendArguments ? pending.argumentsJson + patch.argumentsJson : patch.argumentsJson;
  }

  state.pendingToolCalls.set(index, pending);
  return pending;
}

function allocateOpenAIResponsesOutputIndex(state: OpenAIResponsesRelayState, preferredIndex: number): number {
  if (!state.usedOutputIndices.has(preferredIndex)) {
    state.usedOutputIndices.add(preferredIndex);
    state.nextOutputIndex = Math.max(state.nextOutputIndex, preferredIndex + 1);
    return preferredIndex;
  }

  let index = state.nextOutputIndex;
  while (state.usedOutputIndices.has(index)) {
    index += 1;
  }

  state.usedOutputIndices.add(index);
  state.nextOutputIndex = index + 1;
  return index;
}

function flushPendingOpenAIResponsesToolCalls(state: OpenAIResponsesRelayState): string[] {
  if (state.pendingToolCalls.size === 0) {
    return [];
  }

  const frames: string[] = [];
  const toolCalls = [...state.pendingToolCalls.values()].sort((a, b) => a.outputIndex - b.outputIndex);
  for (const toolCall of toolCalls) {
    if (!toolCall.added) {
      frames.push(
        encodeSseData({
          type: 'response.output_item.added',
          output_index: toolCall.outputIndex,
          item: buildOpenAIResponsesFunctionCallItem(toolCall, 'in_progress')
        })
      );
      toolCall.added = true;
    }

    if (toolCall.argumentsJson.length > toolCall.emittedArgumentsLength) {
      const remainingArguments = toolCall.argumentsJson.slice(toolCall.emittedArgumentsLength);
      frames.push(
        encodeSseData({
          type: 'response.function_call_arguments.delta',
          output_index: toolCall.outputIndex,
          item_id: toolCall.id,
          delta: remainingArguments
        })
      );
      toolCall.emittedArgumentsLength = toolCall.argumentsJson.length;
    }

    if (!toolCall.done) {
      frames.push(
        encodeSseData({
          type: 'response.function_call_arguments.done',
          output_index: toolCall.outputIndex,
          item_id: toolCall.id,
          name: toolCall.name,
          ...(toolCall.namespace ? { namespace: toolCall.namespace } : {}),
          arguments: toolCall.argumentsJson
        })
      );
      frames.push(
        encodeSseData({
          type: 'response.output_item.done',
          output_index: toolCall.outputIndex,
          item: buildOpenAIResponsesFunctionCallItem(toolCall, 'completed')
        })
      );
      toolCall.done = true;
    }
  }

  return frames;
}

function buildOpenAIResponsesFunctionCallItem(
  toolCall: PendingOpenAIResponsesToolCall,
  status: 'in_progress' | 'completed'
): Record<string, unknown> {
  return {
    id: toolCall.id,
    type: 'function_call',
    call_id: toolCall.callId || toolCall.id,
    name: toolCall.name || 'tool',
    ...(toolCall.namespace ? { namespace: toolCall.namespace } : {}),
    arguments: status === 'in_progress' ? '' : toolCall.argumentsJson,
    status
  };
}

function ensureOpenAIResponsesRelayStarted(state: OpenAIResponsesRelayState): string[] {
  if (state.started) {
    return [];
  }

  state.started = true;
  return [
    encodeSseData({
      type: 'response.created',
      response: {
        id: state.responseId,
        object: 'response',
        status: 'in_progress',
        model: state.model,
        output: []
      }
    })
  ];
}

function finalizeOpenAIResponsesRelay(state: OpenAIResponsesRelayState): string[] {
  if (state.finished) {
    return [];
  }

  const frames = ensureOpenAIResponsesRelayStarted(state);
  frames.push(...finalizeOpenAIResponsesReasoningOutput(state));
  if (state.messageItemStarted && state.messageOutputIndex !== undefined) {
    if (state.outputText) {
      frames.push(
        encodeSseData({
          type: 'response.output_text.done',
          text: state.outputText,
          output_index: state.messageOutputIndex,
          content_index: 0,
          item_id: state.messageItemId
        })
      );
    }
    if (state.messageContentStarted) {
      frames.push(
        encodeSseData({
          type: 'response.content_part.done',
          output_index: state.messageOutputIndex,
          item_id: state.messageItemId,
          content_index: 0,
          part: {
            type: 'output_text',
            text: state.outputText,
            annotations: []
          }
        })
      );
    }
    frames.push(
      encodeSseData({
        type: 'response.output_item.done',
        output_index: state.messageOutputIndex,
        item: {
          id: state.messageItemId,
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            {
              type: 'output_text',
              text: state.outputText,
              annotations: []
            }
          ]
        }
      })
    );
  }
  frames.push(...flushPendingOpenAIResponsesToolCalls(state));

  frames.push(
    encodeSseData({
      type: 'response.completed',
      response: buildOpenAIResponsesCompletedPayload(state)
    })
  );
  state.finished = true;
  return frames;
}

function finalizeOpenAIResponsesReasoningOutput(state: OpenAIResponsesRelayState): string[] {
  if (!state.reasoningItemStarted || state.reasoningOutputIndex === undefined) {
    return [];
  }

  const frames: string[] = [];
  if (state.reasoningSummaryStarted) {
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_text.done',
        item_id: state.reasoningItemId,
        output_index: state.reasoningOutputIndex,
        summary_index: 0,
        text: state.reasoningSummaryText
      })
    );
    frames.push(
      encodeSseData({
        type: 'response.reasoning_summary_part.done',
        item_id: state.reasoningItemId,
        output_index: state.reasoningOutputIndex,
        summary_index: 0,
        part: {
          type: 'summary_text',
          text: state.reasoningSummaryText
        }
      })
    );
  }

  if (state.reasoningText) {
    frames.push(
      encodeSseData({
        type: 'response.reasoning_text.done',
        item_id: state.reasoningItemId,
        output_index: state.reasoningOutputIndex,
        content_index: 0,
        text: state.reasoningText
      })
    );
  }

  frames.push(
    encodeSseData({
      type: 'response.output_item.done',
      output_index: state.reasoningOutputIndex,
      item: buildOpenAIResponsesReasoningItemFromState(state)
    })
  );

  return frames;
}

function buildOpenAIResponsesCompletedPayload(state: OpenAIResponsesRelayState): Record<string, unknown> {
  const outputItems: Array<{ outputIndex: number; item: Record<string, unknown> }> = [];
  if (state.reasoningItemStarted && state.reasoningOutputIndex !== undefined) {
    outputItems.push({
      outputIndex: state.reasoningOutputIndex,
      item: buildOpenAIResponsesReasoningItemFromState(state)
    });
  }

  if (state.messageItemStarted && state.messageOutputIndex !== undefined) {
    outputItems.push({
      outputIndex: state.messageOutputIndex,
      item: {
        id: state.messageItemId,
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: state.outputText,
            annotations: []
          }
        ]
      }
    });
  }

  const sortedToolCalls = [...state.pendingToolCalls.values()].sort((a, b) => a.outputIndex - b.outputIndex);
  for (const toolCall of sortedToolCalls) {
    outputItems.push({
      outputIndex: toolCall.outputIndex,
      item: buildOpenAIResponsesFunctionCallItem(toolCall, 'completed')
    });
  }

  const response: Record<string, unknown> = {
    id: state.responseId,
    object: 'response',
    status: 'completed',
    model: state.model,
    output_text: state.outputText,
    output: outputItems.sort((a, b) => a.outputIndex - b.outputIndex).map((entry) => entry.item),
    usage: normalizeOpenAIResponsesUsage(state.usage)
  };

  if (state.finishReason) {
    response.finish_reason = state.finishReason;
  }

  return response;
}

function buildOpenAIResponsesReasoningItemFromState(state: OpenAIResponsesRelayState): Record<string, unknown> {
  return {
    id: state.reasoningItemId,
    type: 'reasoning',
    status: 'completed',
    summary: state.reasoningSummaryText
      ? [
          {
            type: 'summary_text',
            text: state.reasoningSummaryText
          }
        ]
      : [],
    ...(state.reasoningText
      ? {
          content: [
            {
              type: 'reasoning_text',
              text: state.reasoningText
            }
          ]
        }
      : {}),
    ...(state.reasoningEncryptedContent ? { encrypted_content: state.reasoningEncryptedContent } : {})
  };
}

function updateOpenAIResponsesRelayIdentity(
  state: OpenAIResponsesRelayState,
  response: Record<string, unknown> | undefined
) {
  const id = asString(response?.id);
  if (id) {
    state.responseId = id;
  }

  const model = asString(response?.model);
  if (model) {
    state.model = model;
  }
}

function updateOpenAIResponsesRelayUsageFromResponse(
  state: OpenAIResponsesRelayState,
  response: Record<string, unknown> | undefined
) {
  const usage = isObject(response?.usage) ? response.usage : undefined;
  if (!usage) {
    return;
  }

  state.usage = {
    ...state.usage,
    ...usage
  };
}

function updateOpenAIResponsesRelayUsageFromChat(
  state: { usage: Record<string, unknown> },
  usage: Record<string, unknown> | undefined
) {
  if (!usage) {
    return;
  }

  const mappedUsage: Record<string, unknown> = {
    ...state.usage
  };

  const inputTokens = asNumber(usage.prompt_tokens);
  if (inputTokens !== undefined) {
    mappedUsage.input_tokens = inputTokens;
  }

  const outputTokens = asNumber(usage.completion_tokens);
  if (outputTokens !== undefined) {
    mappedUsage.output_tokens = outputTokens;
  }

  const totalTokens = asNumber(usage.total_tokens);
  if (totalTokens !== undefined) {
    mappedUsage.total_tokens = totalTokens;
  }

  const promptDetails = isObject(usage.prompt_tokens_details) ? usage.prompt_tokens_details : undefined;
  const cachedTokens = asNumber(promptDetails?.cached_tokens);
  if (cachedTokens !== undefined) {
    mappedUsage.input_tokens_details = {
      ...(isObject(mappedUsage.input_tokens_details) ? mappedUsage.input_tokens_details : {}),
      cached_tokens: cachedTokens
    };
  }

  const cacheCreationTokens =
    asNumber(promptDetails?.cache_creation_tokens) ??
    asNumber(usage.cache_creation_input_tokens) ??
    asNumber(usage.cache_creation_tokens) ??
    asNumber(usage.cache_write_tokens);
  if (cacheCreationTokens !== undefined) {
    mappedUsage.input_tokens_details = {
      ...(isObject(mappedUsage.input_tokens_details) ? mappedUsage.input_tokens_details : {}),
      cache_creation_tokens: cacheCreationTokens
    };
  }

  state.usage = mappedUsage;
}

function emitAnthropicFramesFromOpenAIChatChunk(
  state: AnthropicRelayState,
  payload: Record<string, unknown>
): string[] {
  const messageId = asString(payload.id);
  if (messageId && !state.started) {
    state.messageId = messageId;
  }

  const model = asString(payload.model);
  if (model) {
    state.model = model;
  }

  const usage = isObject(payload.usage) ? payload.usage : undefined;
  updateAnthropicRelayUsage(state, usage);

  const firstChoice = Array.isArray(payload.choices) && isObject(payload.choices[0]) ? payload.choices[0] : undefined;
  const delta = isObject(firstChoice?.delta) ? firstChoice.delta : undefined;
  const deltaText = asString(delta?.content) || '';
  const reasoningDeltas = collectOpenAIChatReasoningDeltas(delta);
  const finishReason = asString(firstChoice?.finish_reason);
  if (finishReason) {
    state.finishReason = finishReason;
  }

  const frames = ensureAnthropicRelayStarted(state);

  // OpenAI-compatible providers may stream chain-of-thought tokens in reasoning_content or reasoning_details.
  for (const thinkingDelta of [...reasoningDeltas.summaryDeltas, ...reasoningDeltas.textDeltas]) {
    frames.push(...emitAnthropicContentDelta(state, 'thinking', thinkingDelta));
  }
  frames.push(...emitAnthropicContentDelta(state, 'text', deltaText));
  frames.push(...collectOpenAIChatToolCalls(state, delta?.tool_calls));

  if (finishReason) {
    frames.push(...flushPendingAnthropicToolCalls(state));
    frames.push(...finalizeAnthropicRelay(state));
  }

  return frames;
}

function emitAnthropicFramesFromOpenAIResponsesEvent(
  state: AnthropicRelayState,
  payload: Record<string, unknown>
): string[] {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return [];
  }

  if (eventType === 'response.created') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateAnthropicRelayIdentity(state, response);
    updateAnthropicRelayUsage(state, isObject(response?.usage) ? response.usage : undefined);
    return ensureAnthropicRelayStarted(state);
  }

  if (eventType === 'response.output_text.delta') {
    const deltaText = asString(payload.delta) || '';
    if (!deltaText) {
      return [];
    }

    return emitAnthropicContentDelta(state, 'text', deltaText);
  }

  if (eventType === 'response.completed') {
    const response = isObject(payload.response) ? payload.response : undefined;
    updateAnthropicRelayIdentity(state, response);
    updateAnthropicRelayUsage(state, isObject(response?.usage) ? response.usage : undefined);
    collectOpenAIResponsesToolCalls(state, response);

    state.finishReason = asString(response?.finish_reason) || extractResponsesFinishReason(response);
    return [...flushPendingAnthropicToolCalls(state), ...finalizeAnthropicRelay(state)];
  }

  return [];
}

function updateAnthropicRelayIdentity(state: AnthropicRelayState, response: Record<string, unknown> | undefined) {
  const id = asString(response?.id);
  if (id) {
    state.messageId = id;
  }

  const model = asString(response?.model);
  if (model) {
    state.model = model;
  }
}

function extractResponsesFinishReason(response: Record<string, unknown> | undefined): string | undefined {
  if (!response) {
    return undefined;
  }

  if (Array.isArray(response.output)) {
    for (const item of response.output) {
      if (!isObject(item)) {
        continue;
      }

      const finishReason = asString(item.finish_reason) || asString(item.stop_reason);
      if (finishReason) {
        return finishReason;
      }
    }
  }

  if (asString(response.status) === 'incomplete') {
    const incompleteDetails = isObject(response.incomplete_details) ? response.incomplete_details : undefined;
    return asString(incompleteDetails?.reason) || 'max_tokens';
  }

  return undefined;
}

function collectOpenAIChatToolCalls(state: AnthropicRelayState, rawToolCalls: unknown): string[] {
  if (!Array.isArray(rawToolCalls)) {
    return [];
  }

  const frames = ensureAnthropicRelayStarted(state);
  for (let position = 0; position < rawToolCalls.length; position += 1) {
    const rawToolCall = rawToolCalls[position];
    if (!isObject(rawToolCall)) {
      continue;
    }

    const indexValue = asNumber(rawToolCall.index);
    const toolIndex = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : position;
    const functionPayload = isObject(rawToolCall.function) ? rawToolCall.function : undefined;
    const id = asString(rawToolCall.id);
    const name = asString(functionPayload?.name) || asString(rawToolCall.name);
    const argumentsPatch = readOpenAIChatToolArgumentsPatch(functionPayload, rawToolCall);
    const argumentsChunk = argumentsPatch?.value || '';

    frames.push(...closeActiveAnthropicTextBlock(state));
    const toolCall = mergePendingAnthropicToolCall(
      state,
      toolIndex,
      {
        id,
        name,
        argumentsJson: argumentsChunk
      },
      argumentsPatch?.append ?? true
    );
    if (!toolCall.started) {
      frames.push(buildAnthropicToolUseStartFrame(toolCall));
      toolCall.started = true;
      toolCall.closed = false;
    }

    if (argumentsChunk) {
      frames.push(buildAnthropicToolUseDeltaFrame(toolCall.blockIndex, argumentsChunk));
      toolCall.emittedArgumentsLength = toolCall.argumentsJson.length;
    }
  }

  return frames;
}

function collectOpenAIResponsesToolCalls(state: AnthropicRelayState, response: Record<string, unknown> | undefined) {
  if (!response || !Array.isArray(response.output)) {
    return;
  }

  let fallbackIndex = state.pendingToolCalls.size;
  for (const outputItem of response.output) {
    if (!isObject(outputItem)) {
      continue;
    }

    const outputType = asString(outputItem.type);
    if (outputType !== 'function_call' && outputType !== 'tool_call') {
      continue;
    }

    const indexValue = asNumber(outputItem.index);
    const toolIndex = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : fallbackIndex++;
    const functionPayload = isObject(outputItem.function) ? outputItem.function : undefined;
    const id = asString(outputItem.call_id) || asString(outputItem.id);
    const name = asString(outputItem.name) || asString(functionPayload?.name);
    const argumentsJson = normalizeToolArguments(
      outputItem.arguments ?? functionPayload?.arguments ?? outputItem.input
    );

    mergePendingAnthropicToolCall(
      state,
      toolIndex,
      {
        id,
        name,
        argumentsJson
      },
      false
    );
  }
}

function mergePendingAnthropicToolCall(
  state: AnthropicRelayState,
  index: number,
  patch: { id?: string; name?: string; argumentsJson?: string },
  appendArguments: boolean
): PendingAnthropicToolCall {
  const existing = state.pendingToolCalls.get(index);
  const pending: PendingAnthropicToolCall = existing || {
    index,
    blockIndex: state.nextBlockIndex++,
    id: `toolu_${randomUUID().replace(/-/g, '')}`,
    name: 'tool',
    argumentsJson: '',
    emittedArgumentsLength: 0,
    started: false,
    closed: false
  };

  if (existing && existing.closed) {
    pending.blockIndex = state.nextBlockIndex++;
    pending.argumentsJson = '';
    pending.emittedArgumentsLength = 0;
    pending.started = false;
    pending.closed = false;
  }

  if (patch.id) {
    pending.id = patch.id;
  }
  if (patch.name) {
    pending.name = patch.name;
  }

  if (patch.argumentsJson) {
    pending.argumentsJson = appendArguments ? pending.argumentsJson + patch.argumentsJson : patch.argumentsJson;
  }

  state.pendingToolCalls.set(index, pending);
  return pending;
}

function normalizeToolArguments(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value) || isObject(value)) {
    try {
      return JSON.stringify(value);
    } catch {
      return '';
    }
  }

  return '';
}

function flushPendingAnthropicToolCalls(state: AnthropicRelayState): string[] {
  if (state.pendingToolCalls.size === 0) {
    return [];
  }

  const frames = ensureAnthropicRelayStarted(state);
  frames.push(...closeActiveAnthropicTextBlock(state));

  const toolCalls = [...state.pendingToolCalls.values()].sort((a, b) => a.index - b.index);
  for (const toolCall of toolCalls) {
    if (!toolCall.started) {
      frames.push(buildAnthropicToolUseStartFrame(toolCall));
      toolCall.started = true;
    }

    if (toolCall.argumentsJson.length > toolCall.emittedArgumentsLength) {
      const remainingArguments = toolCall.argumentsJson.slice(toolCall.emittedArgumentsLength);
      frames.push(buildAnthropicToolUseDeltaFrame(toolCall.blockIndex, remainingArguments));
      toolCall.emittedArgumentsLength = toolCall.argumentsJson.length;
    }

    if (!toolCall.closed) {
      frames.push(buildAnthropicContentBlockStopFrame(toolCall.blockIndex));
      toolCall.closed = true;
    }
  }

  return frames;
}

function closeActiveAnthropicTextBlock(state: AnthropicRelayState): string[] {
  if (state.activeBlockIndex === undefined) {
    return [];
  }

  const frames = [buildAnthropicContentBlockStopFrame(state.activeBlockIndex)];
  state.activeBlockType = undefined;
  state.activeBlockIndex = undefined;
  return frames;
}

function closePendingAnthropicToolCalls(state: AnthropicRelayState): string[] {
  if (state.pendingToolCalls.size === 0) {
    return [];
  }

  const frames: string[] = [];
  const toolCalls = [...state.pendingToolCalls.values()].sort((a, b) => a.index - b.index);
  for (const toolCall of toolCalls) {
    if (!toolCall.started || toolCall.closed) {
      continue;
    }

    frames.push(buildAnthropicContentBlockStopFrame(toolCall.blockIndex));
    toolCall.closed = true;
  }

  return frames;
}

function ensureAnthropicRelayStarted(state: AnthropicRelayState): string[] {
  if (state.started) {
    return [];
  }

  state.started = true;
  return [buildAnthropicMessageStartFrame(state)];
}

function emitAnthropicContentDelta(
  state: AnthropicRelayState,
  blockType: AnthropicContentBlockType,
  content: string
): string[] {
  if (!content) {
    return [];
  }

  const frames = closePendingAnthropicToolCalls(state);
  frames.push(...ensureAnthropicRelayBlock(state, blockType));
  if (state.activeBlockIndex === undefined) {
    return frames;
  }

  frames.push(
    encodeSseEvent('content_block_delta', {
      type: 'content_block_delta',
      index: state.activeBlockIndex,
      delta:
        blockType === 'thinking'
          ? {
              type: 'thinking_delta',
              thinking: content
            }
          : {
              type: 'text_delta',
              text: content
            }
    })
  );

  return frames;
}

function ensureAnthropicRelayBlock(
  state: AnthropicRelayState,
  blockType: AnthropicContentBlockType
): string[] {
  const frames = ensureAnthropicRelayStarted(state);
  if (state.activeBlockType === blockType && state.activeBlockIndex !== undefined) {
    return frames;
  }

  if (state.activeBlockIndex !== undefined) {
    frames.push(buildAnthropicContentBlockStopFrame(state.activeBlockIndex));
  }

  const nextBlockIndex = state.nextBlockIndex;
  state.nextBlockIndex += 1;
  state.activeBlockType = blockType;
  state.activeBlockIndex = nextBlockIndex;

  frames.push(buildAnthropicContentBlockStartFrame(nextBlockIndex, blockType));
  return frames;
}

function finalizeAnthropicRelay(state: AnthropicRelayState): string[] {
  if (state.finished) {
    return [];
  }

  const frames = ensureAnthropicRelayStarted(state);
  if (state.activeBlockIndex !== undefined) {
    frames.push(buildAnthropicContentBlockStopFrame(state.activeBlockIndex));
    state.activeBlockType = undefined;
    state.activeBlockIndex = undefined;
  }
  frames.push(...closePendingAnthropicToolCalls(state));

  frames.push(
    encodeSseEvent('message_delta', {
      type: 'message_delta',
      delta: {
        stop_reason: mapFinishReasonToAnthropic(state.finishReason),
        stop_sequence: null
      },
      usage: buildAnthropicMessageDeltaUsage(state)
    })
  );
  frames.push(
    encodeSseEvent('message_stop', {
      type: 'message_stop'
    })
  );

  state.finished = true;
  return frames;
}

function buildAnthropicMessageStartFrame(state: AnthropicRelayState): string {
  const usage: Record<string, unknown> = {
    input_tokens: state.inputTokens ?? 0,
    output_tokens: 0
  };
  if (state.cacheCreationInputTokens !== undefined) {
    usage.cache_creation_input_tokens = state.cacheCreationInputTokens;
  }
  if (state.cacheReadInputTokens !== undefined) {
    usage.cache_read_input_tokens = state.cacheReadInputTokens;
  }
  addServerToolUseToAnthropicUsage(usage, state.serverToolUse);

  return encodeSseEvent('message_start', {
    type: 'message_start',
    message: {
      id: state.messageId,
      type: 'message',
      role: 'assistant',
      model: state.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage
    }
  });
}

function buildAnthropicMessageDeltaUsage(state: AnthropicRelayState): Record<string, unknown> {
  const usage: Record<string, unknown> = {
    output_tokens: state.outputTokens
  };
  if (state.inputTokens !== undefined) {
    usage.input_tokens = state.inputTokens;
  }
  if (state.cacheCreationInputTokens !== undefined) {
    usage.cache_creation_input_tokens = state.cacheCreationInputTokens;
  }
  if (state.cacheReadInputTokens !== undefined) {
    usage.cache_read_input_tokens = state.cacheReadInputTokens;
  }
  addServerToolUseToAnthropicUsage(usage, state.serverToolUse);

  return usage;
}

function updateAnthropicRelayUsage(
  state: AnthropicRelayState,
  usage: Record<string, unknown> | undefined
) {
  if (!usage) {
    return;
  }

  const inputDetails = isObject(usage.input_tokens_details)
    ? usage.input_tokens_details
    : isObject(usage.prompt_tokens_details)
      ? usage.prompt_tokens_details
      : undefined;

  const inputTokens = asNumber(usage.input_tokens) ?? asNumber(usage.prompt_tokens);
  if (inputTokens !== undefined) {
    state.inputTokens = inputTokens;
  }

  const outputTokens = asNumber(usage.output_tokens) ?? asNumber(usage.completion_tokens);
  if (outputTokens !== undefined) {
    state.outputTokens = outputTokens;
  }

  const cacheReadInputTokens =
    asNumber(inputDetails?.cached_tokens) ??
    asNumber(usage.cache_read_input_tokens) ??
    asNumber(usage.cache_read_tokens);
  if (cacheReadInputTokens !== undefined) {
    state.cacheReadInputTokens = cacheReadInputTokens;
  }

  const cacheCreationInputTokens =
    asNumber(inputDetails?.cache_creation_tokens) ??
    asNumber(usage.cache_creation_input_tokens) ??
    asNumber(usage.cache_creation_tokens) ??
    asNumber(usage.cache_write_tokens);
  if (cacheCreationInputTokens !== undefined) {
    state.cacheCreationInputTokens = cacheCreationInputTokens;
  }

  const serverToolUse = extractServerToolUse(usage.server_tool_use);
  if (serverToolUse) {
    state.serverToolUse = serverToolUse;
  }
}

function buildAnthropicContentBlockStartFrame(index: number, blockType: AnthropicContentBlockType): string {
  return encodeSseEvent('content_block_start', {
    type: 'content_block_start',
    index,
    content_block:
      blockType === 'thinking'
        ? {
            type: 'thinking',
            thinking: ''
          }
        : {
            type: 'text',
            text: ''
          }
  });
}

function buildAnthropicContentBlockStopFrame(index: number): string {
  return encodeSseEvent('content_block_stop', {
    type: 'content_block_stop',
    index
  });
}

function buildAnthropicToolUseStartFrame(toolCall: PendingAnthropicToolCall): string {
  return encodeSseEvent('content_block_start', {
    type: 'content_block_start',
    index: toolCall.blockIndex,
    content_block: {
      type: 'tool_use',
      id: toolCall.id,
      name: toolCall.name,
      input: {}
    }
  });
}

function buildAnthropicToolUseDeltaFrame(blockIndex: number, partialJson: string): string {
  return encodeSseEvent('content_block_delta', {
    type: 'content_block_delta',
    index: blockIndex,
    delta: {
      type: 'input_json_delta',
      partial_json: partialJson
    }
  });
}

function isOpenAIResponsesStreamEvent(payload: Record<string, unknown>): boolean {
  const eventType = asString(payload.type);
  return typeof eventType === 'string' && eventType.startsWith('response.');
}

function collectOpenAINonStreamStateFromResponsesEvent(
  state: {
    id: string;
    model: string;
    outputText: string;
    finishReason?: string;
    usage: Record<string, unknown>;
    completedResponse?: Record<string, unknown>;
    outputItems: Record<string, unknown>[];
    toolCalls: Map<number, OpenAIStreamToolCallAccumulator>;
    reasoning: OpenAIReasoningAccumulator;
  },
  payload: Record<string, unknown>
) {
  const eventType = asString(payload.type) || '';
  if (!eventType) {
    return;
  }

  if (eventType === 'response.created') {
    const response = isObject(payload.response) ? payload.response : undefined;
    const id = asString(response?.id);
    const model = asString(response?.model);
    if (id) {
      state.id = id;
    }
    if (model) {
      state.model = model;
    }
    return;
  }

  if (eventType === 'response.output_text.delta') {
    const delta = asString(payload.delta);
    if (delta) {
      state.outputText += delta;
    }
    return;
  }

  if (eventType === 'response.output_text.done') {
    const text = asString(payload.text);
    if (text) {
      state.outputText = text;
    }
    return;
  }

  if (eventType === 'response.output_item.done') {
    const item = isObject(payload.item) ? payload.item : undefined;
    if (item) {
      state.outputItems.push(item);
    }
    return;
  }

  if (eventType === 'response.completed') {
    const response = isObject(payload.response) ? payload.response : undefined;
    if (!response) {
      return;
    }

    const id = asString(response.id);
    const model = asString(response.model);
    if (id) {
      state.id = id;
    }
    if (model) {
      state.model = model;
    }

    const outputText = asString(response.output_text) || extractOpenAIResponsesOutputText(response.output);
    if (outputText) {
      state.outputText = outputText;
    }

    const finishReason = asString(response.finish_reason) || extractResponsesFinishReason(response);
    if (finishReason) {
      state.finishReason = finishReason;
    }

    const usage = isObject(response.usage) ? response.usage : undefined;
    if (usage) {
      state.usage = {
        ...state.usage,
        ...usage
      };
    }

    state.completedResponse = response;
  }
}

function collectOpenAINonStreamStateFromChatChunk(
  state: {
    id: string;
    model: string;
    outputText: string;
    finishReason?: string;
    usage: Record<string, unknown>;
    completedResponse?: Record<string, unknown>;
    toolCalls: Map<number, OpenAIStreamToolCallAccumulator>;
    reasoning: OpenAIReasoningAccumulator;
  },
  payload: Record<string, unknown>
) {
  const id = asString(payload.id);
  const model = asString(payload.model);
  if (id) {
    state.id = id;
  }
  if (model) {
    state.model = model;
  }

  updateOpenAIResponsesRelayUsageFromChat(state, isObject(payload.usage) ? payload.usage : undefined);

  const firstChoice = Array.isArray(payload.choices) && isObject(payload.choices[0]) ? payload.choices[0] : undefined;
  const delta = isObject(firstChoice?.delta) ? firstChoice.delta : undefined;
  const deltaText = asString(delta?.content) || '';
  if (deltaText) {
    state.outputText += deltaText;
  }
  collectOpenAIReasoningAccumulator(state.reasoning, delta);
  collectOpenAIStreamToolCalls(state.toolCalls, delta?.tool_calls);

  const fullMessage = isObject(firstChoice?.message) ? firstChoice.message : undefined;
  const fullText = asString(fullMessage?.content);
  if (fullText) {
    state.outputText = fullText;
  }
  collectOpenAIReasoningAccumulator(state.reasoning, fullMessage, true);
  collectOpenAIStreamToolCalls(state.toolCalls, fullMessage?.tool_calls);

  const finishReason = asString(firstChoice?.finish_reason);
  if (finishReason) {
    state.finishReason = finishReason;
  }
}

function collectOpenAIReasoningAccumulator(
  accumulator: OpenAIReasoningAccumulator,
  value: Record<string, unknown> | undefined,
  replace = false
) {
  if (!value) {
    return;
  }

  const reasoningText =
    asString(value.reasoning_content) ||
    asString(value.reasoning) ||
    asString(value.thinking);

  const reasoningDetails = Array.isArray(value.reasoning_details) ? value.reasoning_details : [];
  if (replace && reasoningDetails.length > 0) {
    accumulator.rawDetails = [];
    accumulator.summary = '';
    accumulator.encryptedContent = undefined;
    accumulator.text = '';
  }

  if (reasoningText && (replace && reasoningDetails.length === 0)) {
    accumulator.rawDetails = [];
    accumulator.summary = '';
    accumulator.encryptedContent = undefined;
    accumulator.text = reasoningText;
  }

  for (const detail of reasoningDetails) {
    accumulator.rawDetails.push(detail);
    if (typeof detail === 'string') {
      appendReasoningAccumulatorText(accumulator, detail);
      continue;
    }

    if (!isObject(detail)) {
      continue;
    }

    const id = asString(detail.id);
    if (id && !accumulator.id) {
      accumulator.id = id;
    }

    const type = asString(detail.type);
    const summary = asString(detail.summary);
    const text = asString(detail.text) || asString(detail.reasoning) || asString(detail.thinking);
    const encryptedContent = asString(detail.encrypted_content) || asString(detail.data);

    if (type === 'reasoning.summary' || (summary && !text)) {
      accumulator.summary += summary || text || '';
      continue;
    }

    if (text) {
      appendReasoningAccumulatorText(accumulator, text);
    }

    if (encryptedContent && !accumulator.encryptedContent) {
      accumulator.encryptedContent = encryptedContent;
    }
  }

  if (reasoningText && !(replace && reasoningDetails.length === 0)) {
    appendReasoningAccumulatorText(accumulator, reasoningText);
  }
}

function appendReasoningAccumulatorText(accumulator: OpenAIReasoningAccumulator, value: string): void {
  if (!value) {
    return;
  }

  const text = value.trim();
  if (
    text &&
    (accumulator.text.trim() === text ||
      accumulator.text
        .split('\n')
        .some((part) => part.trim() === text))
  ) {
    return;
  }

  accumulator.text += value;
}

function buildChatReasoningDetailsFromAccumulator(accumulator: OpenAIReasoningAccumulator): unknown[] {
  const details: unknown[] = [];
  if (accumulator.summary) {
    details.push({
      type: 'reasoning.summary',
      summary: accumulator.summary,
      id: accumulator.id || null,
      format: 'openai-responses-v1',
      index: details.length
    });
  }
  if (accumulator.text) {
    details.push({
      type: 'reasoning.text',
      text: accumulator.text,
      id: accumulator.id || null,
      format: 'openai-responses-v1',
      index: details.length
    });
  }
  if (accumulator.encryptedContent) {
    details.push({
      type: 'reasoning.encrypted',
      data: accumulator.encryptedContent,
      id: accumulator.id || null,
      format: 'openai-responses-v1',
      index: details.length
    });
  }

  return details;
}

function collectOpenAIStreamToolCalls(
  toolCalls: Map<number, OpenAIStreamToolCallAccumulator>,
  rawToolCalls: unknown
) {
  if (!Array.isArray(rawToolCalls)) {
    return;
  }

  for (let position = 0; position < rawToolCalls.length; position += 1) {
    const rawToolCall = rawToolCalls[position];
    if (!isObject(rawToolCall)) {
      continue;
    }

    const indexValue = asNumber(rawToolCall.index);
    const index = indexValue !== undefined ? Math.max(0, Math.trunc(indexValue)) : position;
    const functionPayload = isObject(rawToolCall.function) ? rawToolCall.function : undefined;
    const current = toolCalls.get(index) || {
      argumentsJson: ''
    };
    const id = asString(rawToolCall.id);
    const type = asString(rawToolCall.type);
    const name = asString(functionPayload?.name) || asString(rawToolCall.name);
    const argumentsPatch = readOpenAIChatToolArgumentsPatch(functionPayload, rawToolCall);

    if (id) {
      current.id = id;
    }
    if (type) {
      current.type = type;
    }
    if (name) {
      current.name = name;
    }
    if (argumentsPatch?.value) {
      current.argumentsJson = argumentsPatch.append
        ? current.argumentsJson + argumentsPatch.value
        : argumentsPatch.value;
    }

    toolCalls.set(index, current);
  }
}

function readOpenAIChatToolArgumentsPatch(
  functionPayload: Record<string, unknown> | undefined,
  rawToolCall: Record<string, unknown>
): { value: string; append: boolean } | undefined {
  const hasFunctionArguments = Boolean(
    functionPayload && Object.prototype.hasOwnProperty.call(functionPayload, 'arguments')
  );
  const hasTopLevelArguments = Object.prototype.hasOwnProperty.call(rawToolCall, 'arguments');
  if (!hasFunctionArguments && !hasTopLevelArguments) {
    return undefined;
  }

  const rawValue = hasFunctionArguments ? functionPayload?.arguments : rawToolCall.arguments;
  const value = normalizeStreamToolArguments(rawValue);
  if (!value) {
    return undefined;
  }

  return {
    value,
    append: typeof rawValue === 'string'
  };
}

function buildOpenAIStreamToolCalls(
  toolCalls: Map<number, OpenAIStreamToolCallAccumulator>
) {
  return [...toolCalls.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([index, toolCall]) => ({
      id: toolCall.id || `call_${index}`,
      type: toolCall.type || 'function',
      function: {
        name: toolCall.name || '',
        arguments: toolCall.argumentsJson || ''
      }
    }));
}

function normalizeStreamToolArguments(value: unknown) {
  if (value === undefined || value === null) {
    return '';
  }
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeAnthropicToolStartInput(value: unknown) {
  if (isObject(value) && Object.keys(value).length === 0) {
    return '';
  }

  return normalizeStreamToolArguments(value);
}

function parseStreamToolArguments(value: string): unknown {
  if (!value.trim()) {
    return {};
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function extractOpenAIResponsesOutputText(output: unknown): string {
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
        if (!isObject(content)) {
          continue;
        }
        const text = asString(content.text) || asString(content.output_text) || asString(content.input_text);
        if (text) {
          chunks.push(text);
        }
      }
      continue;
    }

    const text = asString(item.text) || asString(item.output_text) || asString(item.input_text);
    if (text) {
      chunks.push(text);
    }
  }

  return chunks.join('\n').trim();
}

function encodeSseEvent(eventName: string, data: unknown): string {
  return `event: ${eventName}\n${encodeSseDataLines(data)}\n\n`;
}

function encodeSseData(data: unknown): string {
  return `${encodeSseDataLines(data)}\n\n`;
}

function encodeSseDataLines(data: unknown): string {
  const serialized = JSON.stringify(data ?? null);
  const lines = serialized.split('\n');
  return lines.map((line) => `data: ${line}`).join('\n');
}
