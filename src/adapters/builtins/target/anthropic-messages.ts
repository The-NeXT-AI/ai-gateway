import type { StandardRequestInputContent, StandardRequestInputMessage, TargetAdapter } from '../../../types';
import { ok } from '../../../types';
import { asString, collectStandardInputMessages, isObject } from '../../../utils';
import { buildAnthropicHeaders } from '../common';
import { parseAnthropicToStandardResponse } from './shared';
import { flattenStandardTools, mapStandardToolNameToTargetName, mapToolChoiceFunctionName } from './tools';

const defaultAnthropicMaxTokens = 1024;

export const anthropicMessagesTargetAdapter: TargetAdapter = {
  provider: 'anthropic',
  buildRequestFromStandard(input) {
    const headersResult = buildAnthropicHeaders(input.request.headers, input.config);
    if (!headersResult.ok) {
      return headersResult;
    }

    const messages = standardInputToAnthropicMessages(input.standardRequest.input, input.standardRequest.tools);
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

    const toolsDisabled = isAnthropicToolChoiceNone(input.standardRequest.tool_choice);
    const tools = toolsDisabled ? undefined : mapStandardToolsToAnthropicTools(input.standardRequest.tools);
    if (tools) {
      body.tools = tools;
    }

    const toolChoice = toolsDisabled
      ? undefined
      : mapStandardToolChoiceToAnthropicToolChoice(
          input.standardRequest.tool_choice,
          input.standardRequest.tools
        );
    if (toolChoice !== undefined) {
      body.tool_choice = toolChoice;
    }

    return ok({
      url: `${input.config.anthropicBaseUrl}/v1/messages`,
      headers: headersResult.value,
      body
    });
  },
  toStandardResponse(payload) {
    return parseAnthropicToStandardResponse(payload);
  }
};

function standardInputToAnthropicMessages(
  input: string | StandardRequestInputMessage[],
  tools?: unknown[]
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  for (const message of collectStandardInputMessages(input)) {
    const content = standardContentToAnthropicBlocks(message.content, tools);
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
  tools?: unknown[]
): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  for (const item of content) {
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
        name: mapStandardToolNameToTargetName(item.name, tools),
        input: isObject(item.input) ? item.input : {}
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

    const toolResultBlock: Record<string, unknown> = {
      type: 'tool_result',
      tool_use_id: item.tool_use_id,
      content: item.content
    };
    if (item.is_error !== undefined) {
      toolResultBlock.is_error = item.is_error;
    }
    blocks.push(toolResultBlock);
  }

  return blocks;
}

function standardReasoningToAnthropicBlocks(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }>
): Array<Record<string, unknown>> {
  const blocks = anthropicBlocksFromReasoningDetails(item.reasoning_details);
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

    if (!isObject(detail)) {
      continue;
    }

    const type = asString(detail.type);
    const thinking =
      asString(detail.thinking) ||
      asString(detail.text) ||
      asString(detail.reasoning) ||
      asString(detail.summary);
    const data = asString(detail.data) || asString(detail.encrypted_content);

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
    const signature = asString(detail.signature);
    if (signature) {
      block.signature = signature;
    }
    blocks.push(block);
  }

  return blocks;
}

function mapStandardToolsToAnthropicTools(tools: unknown[] | undefined): Record<string, unknown>[] | undefined {
  const mapped = flattenStandardTools(tools).map((tool) => {
    const mappedTool: Record<string, unknown> = {
      name: tool.targetName,
      input_schema: tool.parameters
    };
    if (tool.description) {
      mappedTool.description = tool.description;
    }

    return mappedTool;
  });
  return mapped.length > 0 ? mapped : undefined;
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
