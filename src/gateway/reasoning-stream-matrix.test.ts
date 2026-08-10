import { describe, expect, it } from 'vitest';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  decodeOpenAIResponsesReasoningEnvelope,
  decodeReasoningTransportEnvelope
} from '../adapters/builtins/reasoning-envelope';
import { relayConvertedStreamFromUpstreamResponse } from './streaming-conversion';

type StreamSourceKey =
  | 'openai_responses'
  | 'openai_chat'
  | 'anthropic_messages'
  | 'gemini_stream'
  | 'gemini_interactions';

type StreamTargetProtocol =
  | 'openai_responses'
  | 'openai_chat'
  | 'anthropic_messages'
  | 'gemini_interactions';

interface LiveConversionScenario {
  name: string;
  source: StreamSourceKey;
  target: StreamTargetProtocol;
}

const reasoningMarker = 'reasoning-live-marker';
const answerMarker = 'answer-live-marker';
const responsesOpaqueMarker = 'responses-opaque-live-marker';
const anthropicOpaqueMarker = 'anthropic-opaque-live-marker';
const interactionsOpaqueMarker = 'interactions-opaque-live-marker';

const liveConversionScenarios: LiveConversionScenario[] = [
  { name: 'OpenAI Chat ← Anthropic', source: 'openai_chat', target: 'anthropic_messages' },
  { name: 'Anthropic ← OpenAI Responses', source: 'anthropic_messages', target: 'openai_responses' },
  { name: 'Anthropic ← OpenAI Chat', source: 'anthropic_messages', target: 'openai_chat' },
  { name: 'OpenAI Responses ← OpenAI Chat', source: 'openai_responses', target: 'openai_chat' },
  { name: 'OpenAI Chat ← OpenAI Responses', source: 'openai_chat', target: 'openai_responses' },
  { name: 'Gemini stream ← OpenAI Responses', source: 'gemini_stream', target: 'openai_responses' },
  { name: 'Gemini stream ← OpenAI Chat', source: 'gemini_stream', target: 'openai_chat' },
  { name: 'OpenAI Responses ← Gemini Interactions', source: 'openai_responses', target: 'gemini_interactions' },
  { name: 'OpenAI Chat ← Gemini Interactions', source: 'openai_chat', target: 'gemini_interactions' },
  { name: 'Anthropic ← Gemini Interactions', source: 'anthropic_messages', target: 'gemini_interactions' },
  { name: 'Gemini stream ← Gemini Interactions', source: 'gemini_stream', target: 'gemini_interactions' },
  { name: 'Gemini Interactions ← OpenAI Responses', source: 'gemini_interactions', target: 'openai_responses' },
  { name: 'Gemini Interactions ← OpenAI Chat', source: 'gemini_interactions', target: 'openai_chat' },
  { name: 'Gemini Interactions ← Anthropic', source: 'gemini_interactions', target: 'anthropic_messages' }
];

describe('live reasoning conversion matrix', () => {
  it('contains every live converted stream path supported by the gateway', () => {
    expect(liveConversionScenarios).toHaveLength(14);
  });

  it.each(liveConversionScenarios)('$name', async ({ source, target }) => {
    const stream = relayConvertedStreamFromUpstreamResponse(
      createReply(),
      {
        adapterKey: source
      } as never,
      createUpstreamStream(target)
    ) as unknown as AsyncIterable<string | Buffer>;

    let body = '';
    for await (const chunk of stream) {
      body += chunk.toString();
    }

    const opaqueMarker = expectedOpaqueMarker(source, target);
    expect(
      body.includes(reasoningMarker) ||
      (opaqueMarker !== undefined && containsMarkerInSse(body, opaqueMarker))
    ).toBe(true);
    expect(body).toContain(answerMarker);
    expectSourceCompletion(body, source);

    if (opaqueMarker) {
      expect(containsMarkerInSse(body, opaqueMarker)).toBe(true);
    }

    if (source === 'gemini_interactions' && target === 'anthropic_messages') {
      const events = parseSseJsonFrames(body);
      const signatureEvent = events.find(
        (event) =>
          (event.delta as Record<string, unknown> | undefined)?.type === 'thought_signature'
      );
      expect(
        decodeReasoningTransportEnvelope(
          String(
            (signatureEvent?.delta as Record<string, unknown> | undefined)
              ?.signature
          )
        )
      ).toMatchObject({
        format: ANTHROPIC_CLAUDE_REASONING_FORMAT,
        data: anthropicOpaqueMarker,
        kind: 'signature'
      });
    }
  });
});

describe('reasoning stream regressions', () => {
  it('forwards a converted event before the upstream stream closes', async () => {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
    const upstream = new Response(
      new ReadableStream<Uint8Array>({
        start(nextController) {
          controller = nextController;
          nextController.enqueue(new TextEncoder().encode(
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_live","model":"claude-live","usage":{}}}\n\n'
          ));
        }
      }),
      { headers: { 'content-type': 'text/event-stream' } }
    );
    const stream = relayConvertedStreamFromUpstreamResponse(
      createReply(),
      { adapterKey: 'openai_chat' } as never,
      upstream
    ) as unknown as AsyncIterable<string | Buffer>;
    const iterator = stream[Symbol.asyncIterator]();
    const nextChunk = iterator.next();
    const timedOut = Symbol('timed-out');
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const first = await Promise.race([
      nextChunk,
      new Promise<typeof timedOut>((resolve) => {
        timeoutHandle = setTimeout(() => resolve(timedOut), 1000);
      })
    ]);
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }
    controller?.close();

    if (first === timedOut) {
      await nextChunk;
    }
    await iterator.return?.();
    expect(first).not.toBe(timedOut);
    if (first !== timedOut) {
      expect(String(first.value)).toContain('chat.completion.chunk');
    }
  });

  it('keeps every indexed encrypted reasoning detail from a Chat stream', async () => {
    const stream = relayConvertedStreamFromUpstreamResponse(
      createReply(),
      {
        adapterKey: 'openai_responses'
      } as never,
      createSseResponse([
        {
          id: 'chatcmpl_multi_reasoning',
          object: 'chat.completion.chunk',
          model: 'chat-model',
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                reasoning_details: [
                  {
                    type: 'reasoning.encrypted',
                    data: 'enc_one',
                    id: 'rs_one',
                    format: 'openai-responses-v1',
                    index: 0
                  },
                  {
                    type: 'reasoning.encrypted',
                    data: 'enc_two',
                    id: 'rs_two',
                    format: 'openai-responses-v1',
                    index: 1
                  }
                ]
              },
              finish_reason: null
            }
          ]
        },
        {
          id: 'chatcmpl_multi_reasoning',
          object: 'chat.completion.chunk',
          model: 'chat-model',
          choices: [
            {
              index: 0,
              delta: {
                content: 'answer'
              },
              finish_reason: 'stop'
            }
          ],
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2
          }
        },
        '[DONE]'
      ])
    ) as unknown as AsyncIterable<string | Buffer>;

    let body = '';
    for await (const chunk of stream) {
      body += chunk.toString();
    }

    const completed = parseSseJsonFrames(body).find(
      (event) => event.type === 'response.completed'
    );
    const response = completed?.response as
      | { output?: Array<Record<string, unknown>> }
      | undefined;
    expect(
      response?.output
        ?.filter((item) => item.type === 'reasoning')
        .map((item) => [item.id, item.encrypted_content])
    ).toEqual([
      ['rs_one', 'enc_one'],
      ['rs_two', 'enc_two']
    ]);
  });

  it('keeps a Gemini thought summary and its signature in one sequential Interactions step', async () => {
    const stream = relayConvertedStreamFromUpstreamResponse(
      createReply(),
      {
        adapterKey: 'gemini_interactions'
      } as never,
      createSseResponse([
        {
          candidates: [
            {
              content: {
                role: 'model',
                parts: [
                  {
                    text: 'summary',
                    thought: true,
                    thoughtSignature: 'gemini-signature'
                  },
                  {
                    text: 'answer'
                  }
                ]
              },
              finishReason: 'STOP'
            }
          ],
          modelVersion: 'gemini-3.5-flash'
        },
        '[DONE]'
      ])
    ) as unknown as AsyncIterable<string | Buffer>;

    let body = '';
    for await (const chunk of stream) {
      body += chunk.toString();
    }

    const events = parseSseJsonFrames(body);
    const summary = events.find(
      (event) =>
        (event.delta as Record<string, unknown> | undefined)?.type ===
        'thought_summary'
    );
    const signature = events.find(
      (event) =>
        (event.delta as Record<string, unknown> | undefined)?.type ===
        'thought_signature'
    );
    expect(summary).toBeDefined();
    expect(signature).toBeDefined();
    if (!summary || !signature) {
      return;
    }

    const thoughtStopPosition = events.findIndex(
      (event) =>
        event.event_type === 'step.stop' &&
        event.index === summary.index
    );
    const modelStartPosition = events.findIndex(
      (event) =>
        event.event_type === 'step.start' &&
        (event.step as Record<string, unknown> | undefined)?.type ===
          'model_output'
    );
    expect(summary.index).toBe(signature.index);
    expect(events.indexOf(summary)).toBeLessThan(events.indexOf(signature));
    expect(thoughtStopPosition).toBeGreaterThanOrEqual(0);
    expect(modelStartPosition).toBeGreaterThanOrEqual(0);
    expect(thoughtStopPosition).toBeLessThan(modelStartPosition);
  });
});

function createReply() {
  return {
    code() {
      return this;
    },
    header() {
      return this;
    },
    send(payload: unknown) {
      return payload;
    }
  } as never;
}

function createUpstreamStream(target: StreamTargetProtocol): Response {
  if (target === 'openai_responses') {
    return createSseResponse([
      {
        type: 'response.created',
        response: {
          id: 'resp_live_matrix',
          model: 'responses-model'
        }
      },
      {
        type: 'response.reasoning_text.delta',
        delta: reasoningMarker,
        item_id: 'rs_live_matrix',
        output_index: 0,
        content_index: 0
      },
      {
        type: 'response.output_text.delta',
        delta: answerMarker,
        item_id: 'msg_live_matrix',
        output_index: 1,
        content_index: 0
      },
      {
        type: 'response.completed',
        response: {
          id: 'resp_live_matrix',
          model: 'responses-model',
          status: 'completed',
          output_text: answerMarker,
          output: [
            {
              id: 'rs_live_matrix',
              type: 'reasoning',
              status: 'completed',
              summary: [],
              content: [{ type: 'reasoning_text', text: reasoningMarker }],
              encrypted_content: responsesOpaqueMarker
            },
            {
              id: 'msg_live_matrix',
              type: 'message',
              role: 'assistant',
              status: 'completed',
              content: [
                {
                  type: 'output_text',
                  text: answerMarker,
                  annotations: []
                }
              ]
            }
          ],
          usage: {
            input_tokens: 4,
            output_tokens: 3,
            total_tokens: 7
          }
        }
      },
      '[DONE]'
    ]);
  }

  if (target === 'openai_chat') {
    return createSseResponse([
      {
        id: 'chatcmpl_live_matrix',
        object: 'chat.completion.chunk',
        model: 'chat-model',
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              reasoning_content: reasoningMarker
            },
            finish_reason: null
          }
        ]
      },
      {
        id: 'chatcmpl_live_matrix',
        object: 'chat.completion.chunk',
        model: 'chat-model',
        choices: [
          {
            index: 0,
            delta: {
              content: answerMarker
            },
            finish_reason: null
          }
        ]
      },
      {
        id: 'chatcmpl_live_matrix',
        object: 'chat.completion.chunk',
        model: 'chat-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 4,
          completion_tokens: 3,
          total_tokens: 7
        }
      },
      '[DONE]'
    ]);
  }

  if (target === 'anthropic_messages') {
    return createSseResponse([
      {
        type: 'message_start',
        message: {
          id: 'msg_live_matrix',
          type: 'message',
          role: 'assistant',
          model: 'anthropic-model',
          content: [],
          usage: {
            input_tokens: 4,
            output_tokens: 0
          }
        }
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: {
          type: 'thinking',
          thinking: ''
        }
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: {
          type: 'thinking_delta',
          thinking: reasoningMarker
        }
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: {
          type: 'signature_delta',
          signature: anthropicOpaqueMarker
        }
      },
      {
        type: 'content_block_stop',
        index: 0
      },
      {
        type: 'content_block_start',
        index: 1,
        content_block: {
          type: 'text',
          text: ''
        }
      },
      {
        type: 'content_block_delta',
        index: 1,
        delta: {
          type: 'text_delta',
          text: answerMarker
        }
      },
      {
        type: 'content_block_stop',
        index: 1
      },
      {
        type: 'message_delta',
        delta: {
          stop_reason: 'end_turn'
        },
        usage: {
          output_tokens: 3
        }
      },
      {
        type: 'message_stop'
      }
    ]);
  }

  return createSseResponse([
    {
      event_type: 'interaction.created',
      interaction: {
        id: 'int_live_matrix',
        object: 'interaction',
        status: 'in_progress',
        model: 'interactions-model'
      }
    },
    {
      event_type: 'step.start',
      index: 0,
      step: {
        type: 'thought'
      }
    },
    {
      event_type: 'step.delta',
      index: 0,
      delta: {
        type: 'thought_summary',
        content: {
          type: 'text',
          text: reasoningMarker
        }
      }
    },
    {
      event_type: 'step.delta',
      index: 0,
      delta: {
        type: 'thought_signature',
        signature: interactionsOpaqueMarker
      }
    },
    {
      event_type: 'step.stop',
      index: 0
    },
    {
      event_type: 'step.start',
      index: 1,
      step: {
        type: 'model_output'
      }
    },
    {
      event_type: 'step.delta',
      index: 1,
      delta: {
        type: 'text',
        text: answerMarker
      }
    },
    {
      event_type: 'step.stop',
      index: 1
    },
    {
      event_type: 'interaction.completed',
      interaction: {
        id: 'int_live_matrix',
        object: 'interaction',
        status: 'completed',
        model: 'interactions-model',
        usage: {
          total_input_tokens: 4,
          total_output_tokens: 3,
          total_tokens: 7
        }
      }
    },
    '[DONE]'
  ]);
}

function createSseResponse(payloads: Array<Record<string, unknown> | '[DONE]'>): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const payload of payloads) {
        const data = payload === '[DONE]' ? payload : JSON.stringify(payload);
        controller.enqueue(new TextEncoder().encode(`data: ${data}\n\n`));
      }
      controller.close();
    }
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream; charset=utf-8'
    }
  });
}

function expectSourceCompletion(body: string, source: StreamSourceKey): void {
  switch (source) {
    case 'openai_responses':
      expect(body).toContain('"type":"response.completed"');
      break;
    case 'openai_chat':
      expect(body).toContain('data: [DONE]');
      break;
    case 'anthropic_messages':
      expect(body).toContain('"type":"message_stop"');
      break;
    case 'gemini_interactions':
      expect(body).toContain('event: interaction.completed');
      expect(body).toContain('event: done');
      break;
    default:
      expect(body).toContain('"candidates"');
      expect(body).toContain('"usageMetadata"');
  }
}

function expectedOpaqueMarker(
  _source: StreamSourceKey,
  target: StreamTargetProtocol
): string | undefined {
  if (target === 'openai_responses') {
    return responsesOpaqueMarker;
  }
  if (target === 'anthropic_messages') {
    return anthropicOpaqueMarker;
  }
  if (target === 'gemini_interactions') {
    return interactionsOpaqueMarker;
  }
  return undefined;
}

function containsMarkerInSse(body: string, marker: string): boolean {
  if (body.includes(marker)) {
    return true;
  }
  return parseSseJsonFrames(body).some((event) => containsMarker(event, marker));
}

function containsMarker(value: unknown, marker: string): boolean {
  if (typeof value === 'string') {
    return (
      decodeReasoningTransportEnvelope(value)?.data === marker ||
      decodeOpenAIResponsesReasoningEnvelope(value)?.encryptedContent === marker
    );
  }
  if (Array.isArray(value)) {
    return value.some((item) => containsMarker(item, marker));
  }
  if (!value || typeof value !== 'object') {
    return false;
  }
  return Object.values(value as Record<string, unknown>).some((item) =>
    containsMarker(item, marker)
  );
}

function parseSseJsonFrames(body: string): Array<Record<string, unknown>> {
  const frames: Array<Record<string, unknown>> = [];
  for (const line of body.split(/\r?\n/)) {
    if (!line.startsWith('data: ')) {
      continue;
    }
    const data = line.slice('data: '.length);
    if (!data || data === '[DONE]') {
      continue;
    }
    try {
      frames.push(JSON.parse(data) as Record<string, unknown>);
    } catch {
      continue;
    }
  }
  return frames;
}
