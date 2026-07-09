import { describe, expect, it } from 'vitest';
import type { StandardRequest } from '../../../types';
import { formatAnthropicMessagesResponse } from '../source/formatters';
import { parseAnthropicMessagesRequest, parseOpenAIResponsesRequest } from '../source/parsers';
import { geminiGenerateContentTargetAdapter } from './gemini-generate-content';

describe('geminiGenerateContentTargetAdapter', () => {
  it('flattens OpenAI Responses namespace tools when targeting Gemini', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-pro',
      input: 'Run JavaScript',
      tools: [
        {
          name: 'mcp__node_repl__',
          type: 'namespace',
          tools: [
            {
              name: 'js',
              type: 'function',
              parameters: {
                type: 'object',
                required: ['code'],
                properties: {
                  code: {
                    type: 'string'
                  }
                },
                additionalProperties: false
              },
              description: 'Run JavaScript.'
            }
          ],
          description: 'Node REPL tools.'
        }
      ],
      tool_choice: {
        type: 'function',
        name: 'mcp__node_repl__.js'
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest: parsed.value,
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

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: 'mcp__node_repl___js',
            description: 'Run JavaScript.',
            parameters: {
              type: 'object',
              required: ['code'],
              properties: {
                code: {
                  type: 'string'
                }
              }
            }
          }
        ]
      }
    ]);
    expect(body.toolConfig).toEqual({
      functionCallingConfig: {
        mode: 'ANY',
        allowedFunctionNames: ['mcp__node_repl___js']
      }
    });
  });

  it('builds Gemini Interactions requests from standard requests', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-flash',
      instructions: 'Be terse.',
      input: 'What is the weather?',
      temperature: 0.2,
      top_p: 0.9,
      max_output_tokens: 128,
      stop: ['END'],
      reasoning: {
        effort: 'high'
      },
      stream: true,
      tools: [
        {
          type: 'function',
          name: 'get_weather',
          description: 'Get weather.',
          parameters: {
            type: 'object',
            properties: {
              city: { type: 'string' }
            },
            required: ['city'],
            additionalProperties: false
          }
        }
      ],
      tool_choice: {
        type: 'function',
        name: 'get_weather'
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const standardRequest: StandardRequest = {
      ...parsed.value,
      gemini_interactions: {
        previous_interaction_id: 'int_prev',
        store: true,
        background: false,
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'weather'
          }
        },
        generation_config: {
          candidate_count: 2
        },
        service_tier: 'default'
      }
    };

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1/responses?fields=steps'
      } as never,
      standardRequest,
      targetProviderConfig: {
        type: 'gemini_interactions'
      } as never,
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

    expect(built.value.url).toBe('https://mock.local/v1beta/interactions?fields=steps&key=sk-test');
    const body = built.value.body as Record<string, unknown>;
    expect(body).toMatchObject({
      model: 'gemini-2.5-flash',
      input: 'What is the weather?',
      system_instruction: 'Be terse.',
      stream: true,
      previous_interaction_id: 'int_prev',
      store: true,
      background: false,
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'weather'
        }
      },
      service_tier: 'default'
    });
    expect(body.generation_config).toEqual({
      candidate_count: 2,
      temperature: 0.2,
      top_p: 0.9,
      max_output_tokens: 128,
      stop_sequences: ['END'],
      thinking_level: 'high'
    });
    expect(body.tools).toEqual([
      {
        type: 'function',
        name: 'get_weather',
        description: 'Get weather.',
        parameters: {
          type: 'object',
          properties: {
            city: { type: 'string' }
          },
          required: ['city']
        }
      }
    ]);
    expect(body.tool_choice).toEqual({
      allowed_tools: {
        mode: 'any',
        tools: ['get_weather']
      }
    });
  });

  it('maps tool call history into Gemini Interactions steps', () => {
    const standardRequest: StandardRequest = {
      model: 'gemini-2.5-flash',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'Use a tool.' }]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              summary: 'Need weather data.'
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
      ]
    };

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1/responses'
      } as never,
      standardRequest,
      targetProviderConfig: {
        type: 'gemini_interactions'
      } as never,
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

    expect((built.value.body as Record<string, unknown>).input).toEqual([
      {
        type: 'user_input',
        content: [{ type: 'text', text: 'Use a tool.' }]
      },
      {
        type: 'thought',
        summary: [{ type: 'text', text: 'Need weather data.' }]
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
    ]);
  });

  it('sanitizes tool schemas to the Gemini schema subset', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-pro',
      input: 'Run tool',
      tools: [
        {
          name: 'complex_tool',
          type: 'function',
          parameters: {
            $schema: 'https://json-schema.org/draft/2020-12/schema',
            type: 'object',
            additionalProperties: false,
            propertyNames: { pattern: '^[a-z]+$' },
            required: ['items', 'mode'],
            properties: {
              items: {
                type: 'array',
                additionalProperties: false,
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    count: {
                      type: 'integer',
                      exclusiveMinimum: 0,
                      minimum: 0
                    }
                  }
                }
              },
              mode: {
                anyOf: [
                  { type: 'string', enum: ['fast'] },
                  { const: 'safe' }
                ]
              },
              maybe: {
                type: ['string', 'null']
              }
            }
          }
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent?beta=tools-2024-04-04'
      } as never,
      standardRequest: parsed.value,
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

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: 'complex_tool',
            parameters: {
              type: 'object',
              required: ['items', 'mode'],
              properties: {
                items: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      count: {
                        type: 'integer',
                        minimum: 0
                      }
                    }
                  }
                },
                mode: {
                  anyOf: [
                    { type: 'string', enum: ['fast'] },
                    { enum: ['safe'] }
                  ]
                },
                maybe: {
                  nullable: true,
                  type: 'string'
                }
              }
            }
          }
        ]
      }
    ]);
    expect(built.value.url).toBe('https://mock.local/v1beta/models/gemini-2.5-pro:generateContent?key=sk-test');
  });

  it('maps Anthropic thinking and tool calls into Gemini content parts', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'gemini-2.5-pro',
      max_tokens: 1024,
      tools: [
        {
          name: 'get_weather',
          description: 'Get weather.',
          input_schema: {
            type: 'object',
            required: ['city'],
            properties: {
              city: {
                type: 'string'
              }
            }
          }
        }
      ],
      messages: [
        {
          role: 'user',
          content: 'Weather in Shanghai?'
        },
        {
          role: 'assistant',
          content: [
            {
              type: 'thinking',
              thinking: 'Need to call the weather tool.',
              signature: 'anthropic-signature'
            },
            {
              type: 'tool_use',
              id: 'toolu_1',
              name: 'get_weather',
              thought_signature: 'gemini-function-signature',
              input: {
                city: 'Shanghai'
              }
            }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_1',
              content: 'Sunny, 28 C.'
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest: parsed.value,
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

    const body = built.value.body as Record<string, unknown>;
    expect(body.contents).toEqual([
      {
        role: 'user',
        parts: [{ text: 'Weather in Shanghai?' }]
      },
      {
        role: 'model',
        parts: [
          {
            text: 'Need to call the weather tool.',
            thought: true
          },
          {
            thoughtSignature: 'gemini-function-signature',
            functionCall: {
              id: 'toolu_1',
              name: 'get_weather',
              args: {
                city: 'Shanghai'
              }
            }
          }
        ]
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: 'toolu_1',
              name: 'get_weather',
              response: {
                content: 'Sunny, 28 C.'
              }
            }
          }
        ]
      }
    ]);
    expect(body.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: 'get_weather',
            description: 'Get weather.',
            parameters: {
              type: 'object',
              required: ['city'],
              properties: {
                city: {
                  type: 'string'
                }
              }
            }
          }
        ]
      }
    ]);
  });

  it('parses Gemini Interaction responses into standard output, reasoning, tool calls, and usage', () => {
    const parsed = geminiGenerateContentTargetAdapter.toStandardResponse({
      id: 'int_123',
      object: 'interaction',
      status: 'requires_action',
      model: 'gemini-2.5-flash',
      steps: [
        {
          type: 'thought',
          summary: [{ type: 'text', text: 'Need current weather.' }],
          text: 'Call the tool.',
          signature: 'sig_123'
        },
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'Checking weather.' }]
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
        total_input_tokens: 7,
        total_output_tokens: 4,
        total_tokens: 11,
        total_cached_tokens: 2
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value).toMatchObject({
      id: 'int_123',
      model: 'gemini-2.5-flash',
      output_text: 'Checking weather.',
      finish_reason: 'tool_use',
      usage: {
        input_tokens: 7,
        output_tokens: 4,
        total_tokens: 11,
        cache_read_tokens: 2
      }
    });
    expect(parsed.value.output[0]).toMatchObject({
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: 'Need current weather.' }],
      content: [{ type: 'reasoning_text', text: 'Call the tool.' }],
      encrypted_content: 'sig_123'
    });
    expect(parsed.value.output[2]).toMatchObject({
      type: 'function_call',
      id: 'call_weather',
      call_id: 'call_weather',
      name: 'get_weather',
      arguments: '{"city":"Shanghai"}'
    });
  });

  it('maps Gemini thought and function calls back into Anthropic tool_use responses', () => {
    const parsed = geminiGenerateContentTargetAdapter.toStandardResponse({
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                text: 'Need to call the weather tool.',
                thought: true
              },
              {
                thoughtSignature: 'gemini-function-signature',
                functionCall: {
                  id: 'toolu_sig',
                  name: 'get_weather',
                  args: {
                    city: 'Shanghai'
                  }
                }
              }
            ]
          },
          finishReason: 'STOP'
        }
      ],
      usageMetadata: {
        promptTokenCount: 12,
        candidatesTokenCount: 8,
        totalTokenCount: 20
      },
      modelVersion: 'gemini-2.5-pro'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.output_text).toBe('');
    expect(parsed.value.output).toEqual([
      expect.objectContaining({
        type: 'reasoning',
        content: [
          {
            type: 'reasoning_text',
            text: 'Need to call the weather tool.'
          }
        ]
      }),
      expect.objectContaining({
        type: 'function_call',
        id: 'toolu_sig',
        call_id: 'toolu_sig',
        name: 'get_weather',
        thought_signature: 'gemini-function-signature',
        arguments: '{"city":"Shanghai"}'
      })
    ]);

    const anthropic = formatAnthropicMessagesResponse(parsed.value);
    expect(anthropic.stop_reason).toBe('tool_use');
    expect(anthropic.content).toEqual([
      {
        type: 'thinking',
        thinking: 'Need to call the weather tool.',
        signature: 'gemini-function-signature'
      },
      {
        type: 'tool_use',
        id: 'toolu_sig',
        name: 'get_weather',
        input: {
          city: 'Shanghai'
        }
      }
    ]);
  });

  it('replays cached Gemini thought signatures when Anthropic follow-up omits the extension field', () => {
    const request = {
      headers: {
        'x-api-key': 'profile-a'
      },
      url: '/v1beta/models/gemini-3.5-flash:generateContent'
    } as never;
    const config = {
      geminiApiKey: 'sk-test',
      geminiBaseUrl: 'https://mock.local',
      geminiApiVersion: 'v1beta'
    } as never;
    const targetProviderConfig = {
      name: 'google-main'
    } as never;
    const firstTurnRequest: StandardRequest = {
      model: 'gemini-3.5-flash',
      input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Weather in Paris?' }] }]
    };

    const firstTurn = geminiGenerateContentTargetAdapter.toStandardResponse(
      {
        candidates: [
          {
            content: {
              role: 'model',
              parts: [
                {
                  thoughtSignature: 'cached-gemini-function-signature',
                  functionCall: {
                    id: 'cached_toolu_weather',
                    name: 'get_weather',
                    args: {
                      city: 'Paris'
                    }
                  }
                }
              ]
            },
            finishReason: 'STOP'
          }
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15
        },
        modelVersion: 'gemini-3.5-flash'
      },
      {
        request,
        standardRequest: firstTurnRequest,
        config,
        targetProviderConfig
      }
    );

    expect(firstTurn.ok).toBe(true);
    if (!firstTurn.ok) {
      return;
    }

    const standardRequest: StandardRequest = {
      model: 'gemini-3.5-flash',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'Weather in Paris?' }]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'cached_toolu_weather',
              name: 'get_weather',
              input: {
                city: 'Paris'
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
              tool_use_id: 'cached_toolu_weather',
              content: '18 C, partly cloudy'
            }
          ]
        }
      ]
    };

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request,
      standardRequest,
      config,
      targetProviderConfig
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.contents).toEqual([
      {
        role: 'user',
        parts: [{ text: 'Weather in Paris?' }]
      },
      {
        role: 'model',
        parts: [
          {
            thoughtSignature: 'cached-gemini-function-signature',
            functionCall: {
              id: 'cached_toolu_weather',
              name: 'get_weather',
              args: {
                city: 'Paris'
              }
            }
          }
        ]
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: 'cached_toolu_weather',
              name: 'get_weather',
              response: {
                content: '18 C, partly cloudy'
              }
            }
          }
        ]
      }
    ]);
  });
});
