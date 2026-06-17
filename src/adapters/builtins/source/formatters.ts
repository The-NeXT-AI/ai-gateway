import type { StandardResponse, StandardResponseReasoning } from '../../../types';
import {
  mapFinishReasonToAnthropic,
  mapFinishReasonToGemini,
  mapFinishReasonToOpenAI
} from '../common';

export function formatOpenAIChatCompletionsResponse(response: StandardResponse): Record<string, unknown> {
  const usage: Record<string, unknown> = {
    prompt_tokens: response.usage.input_tokens,
    completion_tokens: response.usage.output_tokens,
    total_tokens: response.usage.total_tokens
  };

  if (response.usage.cache_read_tokens !== undefined) {
    usage.prompt_tokens_details = {
      cached_tokens: response.usage.cache_read_tokens
    };
  }

  const toolCalls = collectOpenAIChatToolCalls(response);
  const message: Record<string, unknown> = {
    role: 'assistant'
  };
  if (response.output_text) {
    message.content = response.output_text;
  }
  const reasoning = collectOpenAIChatReasoning(response);
  if (reasoning.reasoningContent) {
    message.reasoning_content = reasoning.reasoningContent;
  }
  if (reasoning.reasoningDetails.length > 0) {
    message.reasoning_details = reasoning.reasoningDetails;
  }
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
    if (message.content === undefined) {
      message.content = '';
    }
  } else if (message.content === undefined) {
    message.content = '';
  }

  return {
    id: response.id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: mapFinishReasonToOpenAI(response.finish_reason)
      }
    ],
    usage
  };
}

export function formatAnthropicMessagesResponse(response: StandardResponse): Record<string, unknown> {
  const usage: Record<string, unknown> = {
    input_tokens: response.usage.input_tokens,
    output_tokens: response.usage.output_tokens
  };

  if (response.usage.cache_read_tokens !== undefined) {
    usage.cache_read_input_tokens = response.usage.cache_read_tokens;
  }

  if (response.usage.cache_write_tokens !== undefined) {
    usage.cache_creation_input_tokens = response.usage.cache_write_tokens;
  }

  if (response.usage.server_tool_use) {
    usage.server_tool_use = {
      ...response.usage.server_tool_use
    };
  }

  return {
    id: response.id,
    type: 'message',
    role: 'assistant',
    model: response.model,
    content: collectAnthropicContentBlocks(response),
    stop_reason: mapFinishReasonToAnthropic(response.finish_reason),
    stop_sequence: null,
    usage
  };
}

export function formatGeminiGenerateContentResponse(response: StandardResponse): Record<string, unknown> {
  const usageMetadata: Record<string, unknown> = {
    promptTokenCount: response.usage.input_tokens,
    candidatesTokenCount: response.usage.output_tokens,
    totalTokenCount: response.usage.total_tokens
  };

  if (response.usage.cache_read_tokens !== undefined) {
    usageMetadata.cachedContentTokenCount = response.usage.cache_read_tokens;
  }

  return {
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: collectGeminiParts(response)
        },
        finishReason: mapFinishReasonToGemini(response.finish_reason)
      }
    ],
    usageMetadata,
    modelVersion: response.model
  };
}

function collectOpenAIChatToolCalls(response: StandardResponse): Array<Record<string, unknown>> {
  const toolCalls: Array<Record<string, unknown>> = [];
  for (const item of response.output) {
    if (item.type !== 'function_call') {
      continue;
    }

    toolCalls.push({
      id: item.call_id || item.id,
      type: 'function',
      function: {
        name: item.name,
        arguments: item.arguments
      }
    });
  }

  return toolCalls;
}

function collectAnthropicContentBlocks(response: StandardResponse): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  for (const item of response.output) {
    if (item.type === 'message') {
      for (const content of item.content) {
        if (content.type !== 'output_text' || !content.text) {
          continue;
        }

        blocks.push({
          type: 'text',
          text: content.text
        });
      }
      continue;
    }

    if (item.type === 'reasoning') {
      blocks.push(...formatAnthropicThinkingBlocks(item));
      continue;
    }

    blocks.push({
      type: 'tool_use',
      id: item.call_id || item.id,
      name: item.name,
      input: parseFunctionArguments(item.arguments)
    });
  }

  return blocks.length > 0 ? blocks : [{ type: 'text', text: '' }];
}

function collectGeminiParts(response: StandardResponse): Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = [];
  for (const item of response.output) {
    if (item.type === 'message') {
      for (const content of item.content) {
        if (content.type !== 'output_text' || !content.text) {
          continue;
        }

        parts.push({
          text: content.text
        });
      }
      continue;
    }

    if (item.type === 'reasoning') {
      continue;
    }

    parts.push({
      functionCall: {
        name: item.name,
        args: parseFunctionArguments(item.arguments)
      }
    });
  }

  return parts;
}

function formatAnthropicThinkingBlocks(item: StandardResponseReasoning): Array<Record<string, unknown>> {
  const blocks = anthropicBlocksFromReasoningDetails(item.reasoning_details);
  if (blocks.length > 0) {
    return blocks;
  }

  const thinking = collectReasoningText(item);
  if (thinking) {
    blocks.push({
      type: 'thinking',
      thinking
    });
  }

  if (item.encrypted_content) {
    blocks.push({
      type: 'redacted_thinking',
      data: item.encrypted_content
    });
  }

  return blocks;
}

function anthropicBlocksFromReasoningDetails(value: unknown[] | undefined): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) {
    return [];
  }

  const blocks: Array<Record<string, unknown>> = [];
  for (const detail of value) {
    if (typeof detail === 'string') {
      const thinking = detail.trim();
      if (thinking) {
        blocks.push({
          type: 'thinking',
          thinking
        });
      }
      continue;
    }

    if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) {
      continue;
    }

    const record = detail as Record<string, unknown>;
    const type = typeof record.type === 'string' ? record.type : undefined;
    const thinking =
      asOptionalString(record.thinking) ||
      asOptionalString(record.text) ||
      asOptionalString(record.reasoning) ||
      asOptionalString(record.summary);
    const data = asOptionalString(record.data) || asOptionalString(record.encrypted_content);

    if (type === 'reasoning.encrypted' || type === 'redacted_thinking' || (!thinking && data)) {
      if (data) {
        blocks.push({
          type: 'redacted_thinking',
          data
        });
      }
      continue;
    }

    if (!thinking) {
      continue;
    }

    const block: Record<string, unknown> = {
      type: 'thinking',
      thinking
    };
    const signature = asOptionalString(record.signature);
    if (signature) {
      block.signature = signature;
    }
    blocks.push(block);
  }

  return blocks;
}

function asOptionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function collectOpenAIChatReasoning(response: StandardResponse): {
  reasoningContent?: string;
  reasoningDetails: unknown[];
} {
  const reasoningItems = response.output.filter(
    (item): item is StandardResponseReasoning => item.type === 'reasoning'
  );
  const reasoningDetails: unknown[] = [];
  const reasoningContentParts: string[] = [];

  for (let index = 0; index < reasoningItems.length; index += 1) {
    const item = reasoningItems[index];
    const reasoningContent = collectReasoningText(item);
    if (reasoningContent) {
      reasoningContentParts.push(reasoningContent);
    }

    if (Array.isArray(item.reasoning_details) && item.reasoning_details.length > 0) {
      reasoningDetails.push(...item.reasoning_details);
      continue;
    }

    for (const summary of item.summary) {
      if (!summary.text) {
        continue;
      }

      reasoningDetails.push({
        type: 'reasoning.summary',
        summary: summary.text,
        id: item.id,
        format: 'openai-responses-v1',
        index
      });
    }

    if (reasoningContent) {
      reasoningDetails.push({
        type: 'reasoning.text',
        text: reasoningContent,
        id: item.id,
        format: 'openai-responses-v1',
        index
      });
    }

    if (item.encrypted_content) {
      reasoningDetails.push({
        type: 'reasoning.encrypted',
        data: item.encrypted_content,
        id: item.id,
        format: 'openai-responses-v1',
        index
      });
    }
  }

  return {
    reasoningContent: reasoningContentParts.join('\n').trim() || undefined,
    reasoningDetails
  };
}

function collectReasoningText(item: StandardResponseReasoning): string {
  return (item.content || [])
    .map((content) => content.text)
    .filter(Boolean)
    .join('\n')
    .trim();
}

function parseFunctionArguments(argumentsJson: string): Record<string, unknown> {
  const trimmed = argumentsJson.trim();
  if (!trimmed) {
    return {};
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}
