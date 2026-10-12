import { describe, expect, it } from 'vitest';
import { geminiInteractionsSourceAdapter } from './gemini-interactions';

describe('geminiInteractionsSourceAdapter', () => {
  it('parses Gemini Interactions requests with options, generation config, and tool choice', () => {
    const parsed = geminiInteractionsSourceAdapter.toStandardRequest({
      body: {
        agent: 'agents/weather-agent',
        input: 'Weather in Shanghai?',
        system_instruction: 'Answer briefly.',
        generation_config: {
          temperature: 0.3,
          top_p: 0.8,
          max_output_tokens: 64,
          stop_sequences: ['END']
        },
        previous_interaction_id: 'int_prev',
        store: true,
        background: false,
        response_format: {
          type: 'json_schema'
        },
        service_tier: 'default',
        stream: true,
        tools: [
          {
            type: 'function',
            name: 'get_weather',
            parameters: {
              type: 'object',
              properties: {
                city: { type: 'string' }
              },
              required: ['city']
            }
          }
        ],
        tool_choice: {
          allowed_tools: {
            mode: 'any',
            tools: ['get_weather']
          }
        }
      },
      request: {
        url: '/v1beta/interactions'
      } as never,
      source: {
        adapterKey: 'gemini_interactions'
      },
      config: {} as never
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value).toMatchObject({
      model: 'agents/weather-agent',
      instructions: 'Answer briefly.',
      input: 'Weather in Shanghai?',
      temperature: 0.3,
      top_p: 0.8,
      max_output_tokens: 64,
      stop: ['END'],
      stream: true,
      tool_choice: {
        type: 'function',
        function: {
          name: 'get_weather'
        }
      },
      gemini_interactions: {
        agent: 'agents/weather-agent',
        previous_interaction_id: 'int_prev',
        store: true,
        background: false,
        response_format: {
          type: 'json_schema'
        },
        generation_config: {
          temperature: 0.3,
          top_p: 0.8,
          max_output_tokens: 64,
          stop_sequences: ['END']
        },
        service_tier: 'default'
      }
    });
    expect(parsed.value.tools).toEqual([
      {
        type: 'function',
        name: 'get_weather',
        parameters: {
          type: 'object',
          properties: {
            city: { type: 'string' }
          },
          required: ['city']
        }
      }
    ]);
  });

  it('parses Interactions step history with function result names preserved', () => {
    const parsed = geminiInteractionsSourceAdapter.toStandardRequest({
      body: {
        model: 'gemini-2.5-flash',
        input: [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'Need weather.' }]
          },
          {
            type: 'thought',
            summary: [{ type: 'text', text: 'Use a weather tool.' }],
            text: 'Need current conditions.',
            signature: 'sig_123'
          },
          {
            type: 'function_call',
            id: 'call_weather',
            name: 'get_weather',
            arguments: {
              city: 'Shanghai'
            }
          },
          {
            type: 'function_result',
            call_id: 'call_weather',
            name: 'get_weather',
            result: [{ type: 'text', text: '{"temperature":22}' }]
          }
        ]
      },
      request: {
        url: '/v1beta/interactions'
      } as never,
      source: {
        adapterKey: 'gemini_interactions'
      },
      config: {} as never
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Need weather.' }]
      },
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            text: 'Need current conditions.',
            summary: 'Use a weather tool.',
            encrypted_content: 'sig_123',
            reasoning_details: [
              {
                type: 'reasoning.summary',
                summary: 'Use a weather tool.',
                format: 'google-interactions-v1'
              },
              {
                type: 'reasoning.text',
                text: 'Need current conditions.',
                format: 'google-interactions-v1'
              },
              {
                type: 'reasoning.encrypted',
                data: 'sig_123',
                format: 'google-interactions-v1'
              }
            ]
          },
          {
            type: 'tool_use',
            id: 'call_weather',
            name: 'get_weather',
            input: {
              city: 'Shanghai'
            }
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_weather',
            name: 'get_weather',
            content: '{"temperature":22}'
          }
        ]
      }
    ]);
  });

  // The native multimodal function result puts image subcontent next to the
  // text one; replaying that history must not fold it back into the text.
  it('recovers image subcontent from a function result', () => {
    const parsed = geminiInteractionsSourceAdapter.toStandardRequest({
      body: {
        model: 'gemini-3-pro-preview',
        input: [
          {
            type: 'function_call',
            id: 'call_shot',
            name: 'screenshot',
            arguments: {}
          },
          {
            type: 'function_result',
            call_id: 'call_shot',
            name: 'screenshot',
            result: [
              { type: 'text', text: 'desktop' },
              { type: 'image', mime_type: 'image/png', data: 'c2hvdA==' },
              { type: 'image', uri: 'https://example.test/full.png' }
            ]
          }
        ]
      },
      request: {
        url: '/v1beta/interactions'
      } as never,
      source: {
        adapterKey: 'gemini_interactions'
      },
      config: {} as never
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok || typeof parsed.value.input === 'string') {
      return;
    }

    const toolResult = parsed.value.input
      .flatMap((message) => (Array.isArray(message.content) ? message.content : []))
      .find((item) => item.type === 'tool_result');
    expect(toolResult).toEqual({
      type: 'tool_result',
      tool_use_id: 'call_shot',
      name: 'screenshot',
      content: 'desktop',
      images: [
        'data:image/png;base64,c2hvdA==',
        'https://example.test/full.png'
      ]
    });
  });

  it('formats standard responses as Gemini Interaction objects', () => {
    const formatted = geminiInteractionsSourceAdapter.fromStandardResponse({
      response: {
        id: 'resp_123',
        object: 'response',
        status: 'completed',
        model: 'gemini-2.5-flash',
        output_text: 'It is sunny.',
        output: [
          {
            id: 'rs_123',
            type: 'reasoning',
            status: 'completed',
            summary: [{ type: 'summary_text', text: 'Need weather.' }],
            content: [{ type: 'reasoning_text', text: 'Use tool result.' }],
            encrypted_content: 'sig_123'
          },
          {
            id: 'msg_123',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'It is sunny.',
                annotations: []
              }
            ]
          },
          {
            id: 'fc_123',
            type: 'function_call',
            call_id: 'call_weather',
            name: 'get_weather',
            arguments: '{"city":"Shanghai"}',
            status: 'completed'
          }
        ],
        usage: {
          input_tokens: 5,
          output_tokens: 3,
          total_tokens: 8,
          cache_read_tokens: 1
        },
        finish_reason: 'tool_use'
      }
    } as never);

    expect(formatted).toMatchObject({
      id: 'resp_123',
      object: 'interaction',
      model: 'gemini-2.5-flash',
      status: 'requires_action',
      steps: [
        {
          type: 'thought',
          summary: [{ type: 'text', text: 'Need weather.\nUse tool result.' }],
          signature: 'sig_123'
        },
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'It is sunny.' }]
        },
        {
          type: 'function_call',
          id: 'call_weather',
          name: 'get_weather',
          arguments: {
            city: 'Shanghai'
          }
        }
      ],
      usage: {
        total_input_tokens: 5,
        total_output_tokens: 3,
        total_tokens: 8,
        total_cached_tokens: 1
      }
    });
    expect(typeof (formatted as Record<string, unknown>).created).toBe('string');
    expect(typeof (formatted as Record<string, unknown>).updated).toBe('string');
  });

  it('builds passthrough Interactions upstream requests', () => {
    const body = {
      model: 'gemini-2.5-flash',
      input: 'hello'
    };
    const built = geminiInteractionsSourceAdapter.buildPassthroughRequest({
      body,
      request: {
        url: '/v1/interactions?fields=steps&ignored=true'
      } as never,
      source: {
        adapterKey: 'gemini_interactions',
        metadata: {
          apiVersion: 'v1'
        }
      },
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect(built.value).toEqual({
      url: 'https://mock.local/v1/interactions?fields=steps&key=sk-test',
      headers: {
        'content-type': 'application/json'
      },
      body
    });
  });
});
