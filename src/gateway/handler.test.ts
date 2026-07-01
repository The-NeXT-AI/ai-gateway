import { describe, expect, it } from 'vitest';
import { anthropicMessagesTargetAdapter } from '../adapters/builtins/target/anthropic-messages';
import { openAIResponsesTargetAdapter } from '../adapters/builtins/target/openai-responses';
import {
  buildGatewayBillingTraceSnapshot,
  extractGatewayRequestClientContext,
  hydrateVirtualMultimodalReferences,
  rewriteVirtualModelMultimodalInput,
  resolveBillingResponseSnapshot
} from './handler';
import {
  collectAnthropicNonStreamPayloadFromEventStream,
  collectOpenAINonStreamPayloadFromEventStream,
  relayConvertedStreamFromStandardResponse
} from './streaming-conversion';

describe('resolveBillingResponseSnapshot', () => {
  it('recovers anthropic usage when the response contains no text output', () => {
    const result = resolveBillingResponseSnapshot('anthropic', anthropicMessagesTargetAdapter, {
      id: 'msg_123',
      model: 'claude-3-5-sonnet-latest',
      content: [
        {
          type: 'tool_use',
          id: 'toolu_123',
          name: 'search_docs',
          input: {
            query: 'gateway auth'
          }
        }
      ],
      usage: {
        input_tokens: 120,
        output_tokens: 18,
        cache_read_input_tokens: 24,
        cache_creation_input_tokens: 12
      }
    });

    expect(result).toEqual({
      ok: true,
      value: {
        model: 'claude-3-5-sonnet-latest',
        recovered: false,
        usage: {
          input_tokens: 120,
          output_tokens: 18,
          total_tokens: 174,
          cache_read_tokens: 24,
          cache_write_tokens: 12,
          cache_duration_seconds: undefined
        }
      }
    });
  });

  it('recovers openai usage when the response only exposes usage fields', () => {
    const result = resolveBillingResponseSnapshot('openai', openAIResponsesTargetAdapter, {
      id: 'resp_123',
      model: 'gpt-5.4',
      output: [],
      usage: {
        input_tokens: 55,
        output_tokens: 11,
        total_tokens: 66,
        input_tokens_details: {
          cached_tokens: 9,
          cache_creation_tokens: 7,
          cache_duration_ms: 2500
        }
      }
    });

    expect(result).toEqual({
      ok: true,
      value: {
        model: 'gpt-5.4',
        recovered: true,
        usage: {
          input_tokens: 55,
          output_tokens: 11,
          total_tokens: 66,
          cache_read_tokens: 9,
          cache_write_tokens: 7,
          cache_duration_seconds: 2
        }
      }
    });
  });

  it('recovers zero openai usage from completed stream events without output', async () => {
    const upstreamPayload = await collectOpenAINonStreamPayloadFromEventStream(
      createSseResponse([
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_empty_stream","object":"response","model":"gpt-5.4"}}\n\n',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_empty_stream","object":"response","model":"gpt-5.4","status":"completed","output":[]}}\n\n',
        'data: [DONE]\n\n'
      ])
    );

    const result = resolveBillingResponseSnapshot('openai', openAIResponsesTargetAdapter, upstreamPayload);

    expect(result).toEqual({
      ok: true,
      value: {
        model: 'gpt-5.4',
        recovered: true,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: undefined,
          cache_duration_seconds: undefined
        }
      }
    });
  });

  it('preserves openai output items when completed stream response has empty output', async () => {
    const upstreamPayload = await collectOpenAINonStreamPayloadFromEventStream(
      createSseResponse([
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_item_stream","object":"response","model":"gpt-5.4"}}\n\n',
        'event: response.output_text.done\ndata: {"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_item_stream","text":"hello from item"}\n\n',
        'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_item_stream","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello from item","annotations":[]}]}}\n\n',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_item_stream","object":"response","model":"gpt-5.4","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n',
        'data: [DONE]\n\n'
      ])
    );

    const result = openAIResponsesTargetAdapter.toStandardResponse(upstreamPayload);

    expect(upstreamPayload).toMatchObject({
      id: 'resp_item_stream',
      model: 'gpt-5.4',
      output_text: 'hello from item',
      output: [
        {
          id: 'msg_item_stream',
          type: 'message',
          content: [
            {
              type: 'output_text',
              text: 'hello from item'
            }
          ]
        }
      ]
    });
    expect(result).toMatchObject({
      ok: true,
      value: {
        id: 'resp_item_stream',
        model: 'gpt-5.4',
        output_text: 'hello from item',
        usage: {
          input_tokens: 3,
          output_tokens: 2,
          total_tokens: 5
        }
      }
    });
  });

  it('recovers zero openai usage from empty chat completion stream payloads', async () => {
    const upstreamPayload = await collectOpenAINonStreamPayloadFromEventStream(
      createSseResponse([
        'data: {"id":"chatcmpl_empty_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n'
      ])
    );

    const result = resolveBillingResponseSnapshot('openai', openAIResponsesTargetAdapter, upstreamPayload);

    expect(result).toEqual({
      ok: true,
      value: {
        model: 'gpt-5.4',
        recovered: true,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: undefined,
          cache_duration_seconds: undefined
        }
      }
    });
  });

  it('recovers anthropic usage from streaming event payloads without text output', async () => {
    const upstreamPayload = await collectAnthropicNonStreamPayloadFromEventStream(
      createSseResponse([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream_123","type":"message","role":"assistant","model":"claude-3-5-sonnet-latest","content":[],"usage":{"input_tokens":120,"output_tokens":0,"cache_read_input_tokens":24}}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":18,"cache_creation_input_tokens":12,"server_tool_use":{"web_search_requests":1}}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ])
    );

    const result = resolveBillingResponseSnapshot('anthropic', anthropicMessagesTargetAdapter, upstreamPayload);

    expect(result).toEqual({
      ok: true,
      value: {
        model: 'claude-3-5-sonnet-latest',
        recovered: true,
        usage: {
          input_tokens: 120,
          output_tokens: 18,
          total_tokens: 174,
          cache_read_tokens: 24,
          cache_write_tokens: 12,
          cache_duration_seconds: undefined,
          server_tool_use: {
            web_search_requests: 1,
            web_fetch_requests: undefined
          }
        }
      }
    });
  });

  it('includes input cache and server tool usage in converted anthropic stream deltas', async () => {
    const stream = relayConvertedStreamFromStandardResponse(
      {
        code() {
          return this;
        },
        header() {
          return this;
        },
        send(payload: unknown) {
          return payload;
        }
      } as never,
      {
        adapterKey: 'anthropic_messages'
      } as never,
      {
        id: 'msg_converted_stream_123',
        object: 'response',
        status: 'completed',
        model: 'claude-3-5-sonnet-latest',
        output_text: 'done',
        output: [
          {
            id: 'msg_item_123',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'done',
                annotations: []
              }
            ]
          }
        ],
        usage: {
          input_tokens: 120,
          output_tokens: 18,
          total_tokens: 138,
          cache_read_tokens: 24,
          cache_write_tokens: 12,
          server_tool_use: {
            web_search_requests: 1
          }
        },
        finish_reason: 'stop'
      }
    ) as unknown as AsyncIterable<string | Buffer>;

    let body = '';
    for await (const chunk of stream) {
      body += chunk.toString();
    }

    const messageStartLine = body
      .split('\n')
      .find((line) => line.startsWith('data: ') && line.includes('"type":"message_start"'));
    expect(messageStartLine).toBeDefined();
    const messageStart = JSON.parse(String(messageStartLine).slice('data: '.length));
    expect(messageStart.message.usage).toEqual({
      input_tokens: 120,
      output_tokens: 0,
      cache_read_input_tokens: 24,
      cache_creation_input_tokens: 12,
      server_tool_use: {
        web_search_requests: 1
      }
    });

    const messageDeltaLine = body
      .split('\n')
      .find((line) => line.startsWith('data: ') && line.includes('"type":"message_delta"'));
    expect(messageDeltaLine).toBeDefined();
    const messageDelta = JSON.parse(String(messageDeltaLine).slice('data: '.length));
    expect(messageDelta.usage).toEqual({
      output_tokens: 18,
      input_tokens: 120,
      cache_read_input_tokens: 24,
      cache_creation_input_tokens: 12,
      server_tool_use: {
        web_search_requests: 1
      }
    });
  });

  it('collects anthropic tool input deltas without prefixing empty start input', async () => {
    const upstreamPayload = await collectAnthropicNonStreamPayloadFromEventStream(
      createSseResponse([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_tool_stream","type":"message","role":"assistant","model":"glm-5.2","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_bash","name":"Bash","input":{}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\":\\"pwd\\"}"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":11}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ])
    );

    expect(upstreamPayload).toMatchObject({
      content: [
        {
          type: 'tool_use',
          id: 'toolu_bash',
          name: 'Bash',
          input: {
            command: 'pwd'
          }
        }
      ]
    });

    const result = anthropicMessagesTargetAdapter.toStandardResponse(upstreamPayload);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.output[0]).toMatchObject({
        type: 'function_call',
        name: 'Bash',
        arguments: '{"command":"pwd"}'
      });
    }
  });

  it('preserves anthropic thinking blocks when collecting streaming payloads', async () => {
    const upstreamPayload = await collectAnthropicNonStreamPayloadFromEventStream(
      createSseResponse([
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_thinking_stream","type":"message","role":"assistant","model":"claude-3-5-sonnet-latest","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think "}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_123"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ])
    );

    const result = anthropicMessagesTargetAdapter.toStandardResponse(upstreamPayload);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.output[0]).toMatchObject({
      type: 'reasoning',
      content: [
        {
          type: 'reasoning_text',
          text: 'think first'
        }
      ],
      reasoning_details: [
        {
          type: 'thinking',
          thinking: 'think first',
          signature: 'sig_123'
        }
      ]
    });
    expect(result.value.output_text).toBe('answer');
  });

  it('streams standard reasoning as anthropic thinking deltas', async () => {
    const stream = relayConvertedStreamFromStandardResponse(
      {
        code() {
          return this;
        },
        header() {
          return this;
        },
        send(payload: unknown) {
          return payload;
        }
      } as never,
      {
        adapterKey: 'anthropic_messages'
      } as never,
      {
        id: 'msg_reasoning_stream',
        object: 'response',
        status: 'completed',
        model: 'claude-3-5-sonnet-latest',
        output_text: 'answer',
        output: [
          {
            id: 'rs_123',
            type: 'reasoning',
            status: 'completed',
            summary: [],
            content: [
              {
                type: 'reasoning_text',
                text: 'think first'
              }
            ],
            reasoning_details: [
              {
                type: 'thinking',
                thinking: 'think first',
                signature: 'sig_123'
              }
            ]
          },
          {
            id: 'msg_item_123',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'answer',
                annotations: []
              }
            ]
          }
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15
        },
        finish_reason: 'stop'
      }
    ) as unknown as AsyncIterable<string | Buffer>;

    let body = '';
    for await (const chunk of stream) {
      body += chunk.toString();
    }

    expect(body).toContain('"type":"thinking_delta","thinking":"think first"');
    expect(body).toContain('"type":"signature_delta","signature":"sig_123"');
    expect(body).toContain('"type":"text_delta","text":"answer"');
  });
});

describe('buildGatewayBillingTraceSnapshot', () => {
  it('uses an explicit response status over a later mutable reply status', () => {
    const trace = buildGatewayBillingTraceSnapshot(
      {
        headers: {},
        body: {
          model: 'mimo-v2.5-pro'
        }
      } as any,
      {
        statusCode: 500,
        getHeaders: () => ({})
      } as any,
      {
        responseStatusCode: 200,
        responseBody: {
          id: 'chatcmpl_123'
        }
      }
    );

    expect(trace?.response?.statusCode).toBe(200);
  });
});

describe('extractGatewayRequestClientContext', () => {
  it('reads explicit agent context and keeps metadata fallbacks', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {
          'x-agent-id': '92d91b78-639f-4d44-b87f-b9b0ca6f38f4',
          'x-agent-run-id': 'run-123',
          traceparent: '00-abc-xyz-01'
        }
      } as any,
      {
        metadata: {
          agent_id: 'metadata-agent',
          session_id: 'session-456',
          workflow: 'lead-qualify',
          version: '2026-04-15',
          prompt_version: 'prompt-v3'
        }
      }
    );

    expect(context).toEqual({
      agentId: '92d91b78-639f-4d44-b87f-b9b0ca6f38f4',
      sessionId: 'session-456',
      runId: 'run-123',
      workflow: 'lead-qualify',
      version: '2026-04-15',
      promptVersion: 'prompt-v3',
      traceparent: '00-abc-xyz-01',
      metadata: {
        agent_id: 'metadata-agent',
        session_id: 'session-456',
        workflow: 'lead-qualify',
        version: '2026-04-15',
        prompt_version: 'prompt-v3'
      }
    });
  });

  it('accepts named x-agent-id values and metadata agentId fallback', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {
          'x-agent-id': 'support-agent'
        }
      } as any,
      {
        metadata: {
          agentId: 'metadata-agent',
          session_id: 'session-456'
        }
      }
    );

    expect(context).toEqual({
      agentId: 'support-agent',
      sessionId: 'session-456',
      metadata: {
        agentId: 'metadata-agent',
        session_id: 'session-456'
      }
    });
  });

  it('falls back to metadata agentId when x-agent-id is absent', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {}
      } as any,
      {
        metadata: {
          agentId: 'metadata-agent',
          session_id: 'session-789'
        }
      }
    );

    expect(context).toEqual({
      agentId: 'metadata-agent',
      sessionId: 'session-789',
      metadata: {
        agentId: 'metadata-agent',
        session_id: 'session-789'
      }
    });
  });

  it('detects Codex agent context from Codex headers and user agent', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {
          'user-agent': 'codex_cli_rs/0.42.0',
          'x-codex-account-id': 'acct-test-001',
          'x-request-id': 'req-codex-1'
        },
        url: '/v1/responses'
      } as any,
      {
        model: 'gpt-5',
        conversation: 'conv-codex-1',
        input: 'hello'
      }
    );

    expect(context).toEqual({
      agentId: 'codex',
      sessionId: 'conv-codex-1',
      workflow: 'codex',
      version: '0.42.0',
      clientRequestId: 'req-codex-1',
      metadata: {
        agentDetection: {
          agent: 'codex',
          source: 'codex_headers',
          userAgent: 'codex_cli_rs/0.42.0',
          sessionId: 'conv-codex-1',
          version: '0.42.0'
        }
      }
    });
  });

  it('detects Claude Code agent context from headers and user agent', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {
          'user-agent': 'Claude-Code/1.0.71',
          'x-claude-code-session-id': 'claude-session-1'
        }
      } as any,
      {
        model: 'claude-sonnet-4-5',
        messages: [{ role: 'user', content: 'hello' }]
      }
    );

    expect(context).toEqual({
      agentId: 'claude-code',
      sessionId: 'claude-session-1',
      workflow: 'claude-code',
      version: '1.0.71',
      metadata: {
        agentDetection: {
          agent: 'claude-code',
          source: 'claude_code_headers',
          userAgent: 'Claude-Code/1.0.71',
          sessionId: 'claude-session-1',
          version: '1.0.71'
        }
      }
    });
  });

  it('returns undefined when no client context is present', () => {
    const context = extractGatewayRequestClientContext(
      {
        headers: {}
      } as any,
      {}
    );

    expect(context).toBeUndefined();
  });
});

describe('virtual model multimodal reference rewriting', () => {
  it('replaces OpenAI chat image URLs with short media references', () => {
    const imageUrl = 'https://example.com/private/image.png?token=secret';
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gpt-5:vision_tool',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: 'inspect this image' }]
          }
        ]
      },
      {
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'inspect this image' },
              { type: 'image_url', image_url: { url: imageUrl } }
            ]
          }
        ]
      },
      'openai_chat'
    );

    expect(rewrite.references).toHaveLength(1);
    expect(rewrite.references[0]).toMatchObject({
      kind: 'image',
      sourceType: 'url',
      value: imageUrl
    });

    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(imageUrl);
  });

  it('does not treat object storage URIs as gateway media URLs', () => {
    const imageUrl = 's3://private-bucket/image.png';
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gpt-5:vision_tool',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: 'inspect this image' }]
          }
        ]
      },
      {
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'inspect this image' },
              { type: 'image_url', image_url: { url: imageUrl } }
            ]
          }
        ]
      },
      'openai_chat'
    );

    expect(rewrite.references).toHaveLength(0);
    expect(JSON.stringify(rewrite.request.input)).not.toContain('[media_ref:');
  });

  it('replaces Gemini inline base64 media with short media references', () => {
    const base64 = 'a'.repeat(64);
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gemini-2.5-pro:vision_tool',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: 'read it' }]
          }
        ]
      },
      {
        contents: [
          {
            role: 'user',
            parts: [{ inlineData: { mimeType: 'image/png', data: base64 } }]
          }
        ]
      },
      'gemini_generate'
    );

    expect(rewrite.references).toHaveLength(1);
    expect(rewrite.references[0]).toMatchObject({
      kind: 'media',
      sourceType: 'base64',
      value: base64,
      mimeType: 'image/png'
    });

    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(base64);
  });

  it('rewrites OpenAI Responses function_call_output multimodal tool output', () => {
    const imageUrl = `data:image/png;base64,${'b'.repeat(64)}`;
    const toolOutput = [
      { type: 'input_text', text: 'Computer Use state' },
      { type: 'input_image', detail: 'high', image_url: imageUrl }
    ];
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gpt-5:cua',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_cua',
                content: JSON.stringify(toolOutput)
              }
            ]
          }
        ]
      },
      {
        input: [
          {
            type: 'function_call_output',
            call_id: 'call_cua',
            output: toolOutput
          }
        ]
      },
      'openai_responses'
    );

    expect(rewrite.references).toHaveLength(1);
    expect(rewrite.references[0]).toMatchObject({
      kind: 'image',
      sourceType: 'base64',
      value: imageUrl
    });

    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(imageUrl);
  });

  it('rewrites multimodal data discovered only in standardized tool_result content', () => {
    const imageUrl = `data:image/png;base64,${'f'.repeat(64)}`;
    const toolOutput = [
      { type: 'input_text', text: 'Computer Use state' },
      { type: 'input_image', image_url: imageUrl }
    ];
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gpt-5:cua',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_cua',
                content: JSON.stringify(toolOutput)
              }
            ]
          }
        ]
      },
      {
        input: [
          {
            type: 'function_call_output',
            call_id: 'call_cua',
            output: 'plain text output'
          }
        ]
      },
      'openai_responses'
    );

    expect(rewrite.references).toHaveLength(1);
    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(imageUrl);
  });

  it('rewrites OpenAI chat tool message multimodal output', () => {
    const imageUrl = `data:image/png;base64,${'c'.repeat(64)}`;
    const toolOutput = [
      { type: 'input_text', text: 'Computer Use state' },
      { type: 'input_image', image_url: imageUrl }
    ];
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gpt-5:cua',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'call_cua',
                name: 'computer_use',
                input: {}
              }
            ]
          },
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_cua',
                content: JSON.stringify(toolOutput)
              }
            ]
          }
        ]
      },
      {
        messages: [
          {
            role: 'assistant',
            tool_calls: [
              {
                id: 'call_cua',
                type: 'function',
                function: { name: 'computer_use', arguments: '{}' }
              }
            ]
          },
          {
            role: 'tool',
            tool_call_id: 'call_cua',
            content: toolOutput
          }
        ]
      },
      'openai_chat'
    );

    expect(rewrite.references).toHaveLength(1);
    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(imageUrl);
  });

  it('rewrites Anthropic tool_result multimodal output', () => {
    const base64 = 'd'.repeat(64);
    const toolContent = [
      { type: 'text', text: 'Computer Use state' },
      {
        type: 'image',
        source: {
          type: 'base64',
          media_type: 'image/png',
          data: base64
        }
      }
    ];
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'claude-3-5-sonnet:cua',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'toolu_cua',
                name: 'computer_use',
                input: {}
              }
            ]
          },
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_cua',
                content: JSON.stringify(toolContent)
              }
            ]
          }
        ]
      },
      {
        messages: [
          {
            role: 'assistant',
            content: [{ type: 'tool_use', id: 'toolu_cua', name: 'computer_use', input: {} }]
          },
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_cua',
                content: toolContent
              }
            ]
          }
        ]
      },
      'anthropic_messages'
    );

    expect(rewrite.references).toHaveLength(1);
    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(base64);
  });

  it('rewrites Gemini functionResponse multimodal output', () => {
    const base64 = 'e'.repeat(64);
    const responseContent = [
      { text: 'Computer Use state' },
      {
        inlineData: {
          mimeType: 'image/png',
          data: base64
        }
      }
    ];
    const rewrite = rewriteVirtualModelMultimodalInput(
      {
        model: 'gemini-2.5-pro:cua',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'gemini_tool_0_0',
                name: 'computer_use',
                input: {}
              }
            ]
          },
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'gemini_tool_0_0',
                content: JSON.stringify(responseContent)
              }
            ]
          }
        ]
      },
      {
        contents: [
          {
            role: 'model',
            parts: [{ functionCall: { name: 'computer_use', args: {} } }]
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  name: 'computer_use',
                  response: {
                    content: responseContent
                  }
                }
              }
            ]
          }
        ]
      },
      'gemini_generate'
    );

    expect(rewrite.references).toHaveLength(1);
    const serializedInput = JSON.stringify(rewrite.request.input);
    expect(serializedInput).toContain(`[media_ref:${rewrite.references[0]!.id}]`);
    expect(serializedInput).not.toContain(base64);
  });

  it('hydrates nested tool arguments back to original multimodal payloads', () => {
    const hydrated = hydrateVirtualMultimodalReferences(
      {
        image: 'mm_abc123',
        files: [{ url: '[media_ref:mm_def456]' }],
        prompt: 'compare media_ref:mm_abc123 with media_ref:mm_def456'
      },
      [
        {
          id: 'mm_abc123',
          kind: 'image',
          sourceType: 'url',
          value: 'https://example.com/image.png'
        },
        {
          id: 'mm_def456',
          kind: 'file',
          sourceType: 'base64',
          value: 'data:application/pdf;base64,QUJDRA=='
        }
      ]
    );

    expect(hydrated).toEqual({
      image: 'https://example.com/image.png',
      files: [{ url: 'data:application/pdf;base64,QUJDRA==' }],
      prompt:
        'compare https://example.com/image.png with data:application/pdf;base64,QUJDRA=='
    });
  });
});

function createSseResponse(chunks: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(new TextEncoder().encode(chunk));
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
