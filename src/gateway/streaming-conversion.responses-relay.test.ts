import type { Readable } from 'node:stream';
import type { FastifyReply } from 'fastify';
import { describe, expect, it } from 'vitest';
import { relayConvertedStreamFromUpstreamResponse } from './streaming-conversion';

type ToolDelta = {
  index: number;
  id?: string;
  type?: string;
  function: { name?: string; arguments: string };
};

type ChatFrame = {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string; reasoning_content?: string; tool_calls?: ToolDelta[] };
    finish_reason?: string;
  }>;
  usage?: Record<string, unknown>;
};

const created = {
  type: 'response.created',
  response: { id: 'resp_test', model: 'acme/loud-model' }
};
const usage = { input_tokens: 12, output_tokens: 7, total_tokens: 19 };

function completed(output: unknown[], extra: Record<string, unknown> = {}) {
  return {
    type: 'response.completed',
    response: { id: 'resp_test', model: 'acme/loud-model', status: 'completed', output, usage, ...extra }
  };
}

function toolCall(id: string, args: string) {
  return { type: 'function_call', id: `item_${id}`, call_id: id, name: 'run_cmd', arguments: args };
}

function expectedToolDelta(index: number, id: string, args: string): ToolDelta {
  return { index, id, type: 'function', function: { name: 'run_cmd', arguments: args } };
}

function expectedToolArguments(index: number, args: string): ToolDelta {
  return { index, function: { arguments: args } };
}

async function relay(events: unknown[]): Promise<ChatFrame[]> {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
  const upstream = new Response(body, { headers: { 'content-type': 'text/event-stream' } });
  const reply = {
    code() { return this; },
    header() { return this; },
    send(stream: Readable) { return stream; }
  };
  const stream = relayConvertedStreamFromUpstreamResponse(
    reply as unknown as FastifyReply,
    { adapterKey: 'openai_chat' },
    upstream
  ) as unknown as Readable;

  const parts: string[] = [];
  for await (const part of stream) {
    parts.push(String(part));
  }
  const blocks = parts.join('').trim().split('\n\n');
  expect(blocks.pop()).toBe('data: [DONE]');
  return blocks.map((block) => {
    expect(block.startsWith('data: ')).toBe(true);
    return JSON.parse(block.slice(6)) as ChatFrame;
  });
}

function toolDeltas(frames: ChatFrame[]): ToolDelta[] {
  return frames.flatMap((frame) => frame.choices[0].delta.tool_calls || []);
}

describe('Responses to chat streaming relay', () => {
  it('preserves text-only frames, identity, finish reason, and usage', async () => {
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 0, item: { type: 'message' } },
      { type: 'response.output_text.delta', delta: 'Hello' },
      { type: 'response.output_text.delta', delta: ' world' },
      { type: 'response.output_text.done', text: 'Hello world' },
      completed([{ type: 'message' }], { output_text: 'Hello world' })
    ]);

    expect(frames.map((frame) => frame.choices[0].delta)).toEqual([
      { role: 'assistant' },
      { content: 'Hello' },
      { content: ' world' },
      {}
    ]);
    expect(frames.every((frame) => frame.id === 'resp_test' && frame.model === 'acme/loud-model')).toBe(true);
    expect(frames[3].choices[0].finish_reason).toBe('stop');
    expect(frames[3].usage).toEqual({ prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 });
  });

  it.each([0, -1, undefined])('omits cache details for nonpositive or absent cached tokens (%s)', async (cachedTokens) => {
    const frames = await relay([
      created,
      completed([], { usage: { ...usage, input_tokens_details: { cached_tokens: cachedTokens, cache_write_tokens: 5 } } })
    ]);

    expect(frames[frames.length - 1].usage).toEqual({ prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 });
  });

  it('includes positive cached-token details from Responses usage', async () => {
    const frames = await relay([
      created,
      completed([], { usage: { ...usage, input_tokens_details: { cached_tokens: 3, cache_write_tokens: 5 } } })
    ]);

    expect(frames[frames.length - 1].usage).toEqual({
      prompt_tokens: 12,
      completion_tokens: 7,
      total_tokens: 19,
      prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 5, cache_creation_tokens: 5 }
    });
  });

  it('streams a tool call and does not repeat completed arguments', async () => {
    const item = toolCall('call_test1', '{"cmd":"ls"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '' } },
      { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"cmd":' },
      { type: 'response.function_call_arguments.delta', output_index: 0, delta: '"ls"}' },
      { type: 'response.function_call_arguments.done', output_index: 0, arguments: item.arguments },
      { type: 'response.output_item.done', output_index: 0, item },
      completed([item])
    ]);

    expect(frames[0].choices[0].delta).toEqual({ role: 'assistant' });
    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolArguments(0, '{"cmd":'),
      expectedToolArguments(0, '"ls"}')
    ]);
    expect(frames[frames.length - 1].choices[0].finish_reason).toBe('tool_calls');
    expect(frames[frames.length - 1].usage).toEqual({ prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 });
  });

  it('routes interleaved parallel arguments using output indices rather than tool indices', async () => {
    const first = toolCall('call_test1', '{"x":1}');
    const second = toolCall('call_test2', '{"y":2}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 1, item: first },
      { type: 'response.output_item.added', output_index: 2, item: second },
      { type: 'response.function_call_arguments.delta', output_index: 2, delta: '{"y":' },
      { type: 'response.function_call_arguments.delta', output_index: 1, delta: '{"x":1}' },
      { type: 'response.function_call_arguments.delta', output_index: 2, delta: '2}' },
      { type: 'response.function_call_arguments.done', output_index: 1, arguments: first.arguments },
      { type: 'response.output_item.done', output_index: 2, item: second },
      completed([{ type: 'message' }, first, second])
    ]);

    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolDelta(1, 'call_test2', ''),
      expectedToolArguments(1, '{"y":'),
      expectedToolArguments(0, '{"x":1}'),
      expectedToolArguments(1, '2}')
    ]);
    expect(frames[frames.length - 1].choices[0].finish_reason).toBe('tool_calls');
  });

  it('emits full arguments exactly once when only arguments.done carries them', async () => {
    const item = toolCall('call_test1', '{"cmd":"pwd"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '' } },
      { type: 'response.function_call_arguments.done', output_index: 0, arguments: item.arguments },
      { type: 'response.output_item.done', output_index: 0, item },
      completed([item])
    ]);

    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolArguments(0, item.arguments)
    ]);
  });

  it('emits full arguments from output_item.done when no argument events arrived', async () => {
    const item = toolCall('call_test1', '{"cmd":"pwd"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '' } },
      { type: 'response.output_item.done', output_index: 0, item },
      completed([item])
    ]);

    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolArguments(0, item.arguments)
    ]);
  });

  it('keeps exactly one tool introduction when no argument or done events arrive', async () => {
    const item = toolCall('call_test1', '{"cmd":"pwd"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '' } },
      completed([item])
    ]);

    expect(toolDeltas(frames)).toEqual([expectedToolDelta(0, 'call_test1', '')]);
    expect(frames[frames.length - 1].choices[0].finish_reason).toBe('tool_calls');
  });

  it('registers a call first seen in output_item.done without replaying it on completion', async () => {
    const item = toolCall('call_test1', '{"cmd":"pwd"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.done', output_index: 0, item },
      completed([item])
    ]);

    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolArguments(0, item.arguments)
    ]);
    expect(frames[frames.length - 1].choices[0].finish_reason).toBe('tool_calls');
  });

  it('recovers calls only present in the completed output without replaying known calls', async () => {
    const known = toolCall('call_test1', '{"cmd":"pwd"}');
    const missed = toolCall('call_test2', '{"cmd":"ls"}');
    const frames = await relay([
      created,
      { type: 'response.output_item.added', output_index: 1, item: { ...known, arguments: '' } },
      { type: 'response.function_call_arguments.delta', output_index: 1, delta: '{"cmd":' },
      completed([{ type: 'message' }, known, missed])
    ]);

    expect(toolDeltas(frames)).toEqual([
      expectedToolDelta(0, 'call_test1', ''),
      expectedToolArguments(0, '{"cmd":'),
      expectedToolDelta(1, 'call_test2', ''),
      expectedToolArguments(1, missed.arguments)
    ]);
    expect(frames[frames.length - 1].choices[0].finish_reason).toBe('tool_calls');
  });

  it('forwards reasoning and reasoning summary deltas before text', async () => {
    const frames = await relay([
      { type: 'response.reasoning_text.delta', delta: 'think ' },
      { type: 'response.reasoning_text.delta', delta: 'first' },
      { type: 'response.reasoning_summary_text.delta', delta: 'summary ' },
      { type: 'response.reasoning_summary_text.delta', delta: 'next' },
      { type: 'response.output_text.delta', delta: 'answer' },
      completed([], { output_text: 'answer' })
    ]);

    expect(frames.map((frame) => frame.choices[0].delta)).toEqual([
      { role: 'assistant' },
      { reasoning_content: 'think ' },
      { reasoning_content: 'first' },
      { reasoning_content: 'summary ' },
      { reasoning_content: 'next' },
      { content: 'answer' },
      {}
    ]);
  });

  it('uses item id or a generated call id and ignores unknown argument events', async () => {
    const first = { type: 'function_call', id: 'item_test1', arguments: '{}' };
    const second = { type: 'function_call', arguments: '{}' };
    const frames = await relay([
      { type: 'response.function_call_arguments.delta', output_index: 9, delta: 'ignored' },
      { type: 'response.function_call_arguments.done', output_index: 9, arguments: 'ignored' },
      { type: 'response.output_item.done', output_index: 0, item: first },
      { type: 'response.output_item.done', output_index: 1, item: second },
      completed([first, second])
    ]);
    const calls = toolDeltas(frames);

    expect(calls).toHaveLength(4);
    expect(calls[0]).toEqual({ index: 0, id: 'item_test1', type: 'function', function: { name: '', arguments: '' } });
    expect(calls[1]).toEqual(expectedToolArguments(0, '{}'));
    expect(calls[2].id).toMatch(/^call_[0-9a-f]{32}$/);
    expect(calls[2].function).toEqual({ name: '', arguments: '' });
    expect(calls[3]).toEqual(expectedToolArguments(1, '{}'));
  });
});
