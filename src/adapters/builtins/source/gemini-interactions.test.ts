import { describe, expect, it } from 'vitest';
import {
  decodeOpenAIResponsesReasoningEnvelope,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
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

    expect(parsed.value.input).toMatchObject([
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
            source_format: GEMINI_INTERACTIONS_REASONING_FORMAT,
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
            source_format: GEMINI_INTERACTIONS_REASONING_FORMAT,
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

  it('preserves Responses reasoning IDs across Gemini Interactions history', () => {
    const formatted = geminiInteractionsSourceAdapter.fromStandardResponse({
      response: {
        id: 'resp_roundtrip',
        object: 'response',
        status: 'completed',
        model: 'responses-model',
        output_text: 'First answer.',
        output: [
          {
            id: 'rs_roundtrip',
            type: 'reasoning',
            status: 'completed',
            summary: [],
            content: [],
            encrypted_content: 'encrypted-roundtrip',
            source_format: OPENAI_RESPONSES_REASONING_FORMAT
          },
          {
            id: 'msg_roundtrip',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'First answer.',
                annotations: []
              }
            ]
          }
        ],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          total_tokens: 2,
          cache_read_tokens: 0
        }
      }
    } as never) as Record<string, unknown>;

    const steps = formatted.steps as Array<Record<string, unknown>>;
    expect(decodeOpenAIResponsesReasoningEnvelope(String(steps[0]?.signature))).toEqual({
      id: 'rs_roundtrip',
      encryptedContent: 'encrypted-roundtrip'
    });

    const parsed = geminiInteractionsSourceAdapter.toStandardRequest({
      body: {
        model: 'responses-model',
        input: [...steps, { type: 'user_input', content: [{ type: 'text', text: 'Continue.' }] }]
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

    const input = parsed.value.input as Array<{
      role: string;
      content: Array<Record<string, unknown>>;
    }>;
    expect(input[0]?.content[0]).toMatchObject({
      type: 'reasoning',
      id: 'rs_roundtrip',
      source_format: OPENAI_RESPONSES_REASONING_FORMAT,
      encrypted_content: 'encrypted-roundtrip'
    });
    expect(input[0]?.content[0]?.reasoning_details).toEqual([
      {
        type: 'reasoning.encrypted',
        data: 'encrypted-roundtrip',
        id: 'rs_roundtrip',
        format: OPENAI_RESPONSES_REASONING_FORMAT
      }
    ]);
  });

  it('keeps malformed reasoning envelopes as native Gemini signatures', () => {
    const malformedEnvelope = 'ccr-openai-responses-reasoning-v1:not-valid-base64';
    const parsed = geminiInteractionsSourceAdapter.toStandardRequest({
      body: {
        model: 'gemini-model',
        input: [
          {
            type: 'thought',
            signature: malformedEnvelope
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

    const reasoning = (parsed.value.input as Array<{
      content: Array<Record<string, unknown>>;
    }>)[0]?.content[0];
    expect(reasoning).toMatchObject({
      type: 'reasoning',
      encrypted_content: malformedEnvelope,
      reasoning_details: [
        {
          type: 'reasoning.encrypted',
          data: malformedEnvelope,
          format: 'google-interactions-v1'
        }
      ]
    });
    expect(reasoning).not.toHaveProperty('id');
    expect(reasoning).toHaveProperty(
      'source_format',
      GEMINI_INTERACTIONS_REASONING_FORMAT
    );
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
