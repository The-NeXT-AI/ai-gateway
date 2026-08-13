import { describe, expect, it } from 'vitest';
import type { ProviderNativeItem, StandardRequest, StandardResponse } from '../../../types';
import {
  decodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import {
  formatAnthropicMessagesResponse,
  formatGeminiGenerateContentResponse,
  formatOpenAIChatCompletionsResponse
} from '../source/formatters';
import {
  parseAnthropicMessagesRequest,
  parseGeminiGenerateContentRequest,
  parseOpenAIChatCompletionsRequest,
  parseOpenAIResponsesRequest
} from '../source/parsers';
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
              source_format: GEMINI_INTERACTIONS_REASONING_FORMAT,
              summary: 'Need weather data.',
              encrypted_content: 'interaction-thinking-signature'
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
        summary: [{ type: 'text', text: 'Need weather data.' }],
        signature: 'interaction-thinking-signature'
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

  it('maps native Responses tool-search history into both Gemini request formats', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-flash',
      input: [
        {
          type: 'tool_search_call',
          execution: 'client',
          call_id: 'search_123',
          status: 'completed',
          arguments: { query: 'calendar' }
        },
        {
          type: 'tool_search_output',
          execution: 'client',
          call_id: 'search_123',
          status: 'completed',
          tools: [
            {
              type: 'function',
              name: 'calendar_create',
              defer_loading: true,
              parameters: { type: 'object', properties: {} }
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const config = {
      geminiApiKey: 'sk-test',
      geminiBaseUrl: 'https://mock.local',
      geminiApiVersion: 'v1beta'
    } as never;
    const generated = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-flash:generateContent'
      } as never,
      standardRequest: parsed.value,
      config
    });
    const interactions = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1/responses'
      } as never,
      standardRequest: parsed.value,
      targetProviderConfig: {
        type: 'gemini_interactions'
      } as never,
      config
    });

    expect(generated.ok).toBe(true);
    expect(interactions.ok).toBe(true);
    if (!generated.ok || !interactions.ok) {
      return;
    }

    const serializedOutput =
      '{"type":"tool_search_output","execution":"client","status":"completed","tools":[{"type":"function","name":"calendar_create","defer_loading":true,"parameters":{"type":"object","properties":{}}}]}';
    expect((generated.value.body as Record<string, unknown>).contents).toEqual([
      {
        role: 'model',
        parts: [
          {
            functionCall: {
              id: 'search_123',
              name: 'ToolSearch',
              args: { query: 'calendar' }
            }
          }
        ]
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: 'search_123',
              name: 'ToolSearch',
              response: { content: serializedOutput }
            }
          }
        ]
      }
    ]);
    expect((interactions.value.body as Record<string, unknown>).input).toEqual([
      {
        type: 'function_call',
        id: 'search_123',
        name: 'ToolSearch',
        arguments: { query: 'calendar' }
      },
      {
        type: 'function_result',
        call_id: 'search_123',
        name: 'ToolSearch',
        result: [{ type: 'text', text: serializedOutput }]
      }
    ]);
  });

  it('preserves Anthropic tool references in both Gemini result formats', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_search',
              name: 'ToolSearch',
              input: { query: 'calendar' }
            }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_search',
              content: [
                { type: 'text', text: 'matched one tool' },
                { type: 'tool_reference', tool_name: 'calendar_create' }
              ]
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const config = {
      geminiApiKey: 'sk-test',
      geminiBaseUrl: 'https://mock.local',
      geminiApiVersion: 'v1beta'
    } as never;
    const generated = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-flash:generateContent'
      } as never,
      standardRequest: parsed.value,
      config
    });
    const interactions = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1/responses'
      } as never,
      standardRequest: parsed.value,
      targetProviderConfig: {
        type: 'gemini_interactions'
      } as never,
      config
    });

    expect(generated.ok).toBe(true);
    expect(interactions.ok).toBe(true);
    if (!generated.ok || !interactions.ok) {
      return;
    }

    const resultContent =
      'matched one tool\n[{"type":"tool_reference","tool_name":"calendar_create"}]';
    expect((generated.value.body as Record<string, unknown>).contents).toEqual([
      {
        role: 'model',
        parts: [
          {
            functionCall: {
              id: 'toolu_search',
              name: 'ToolSearch',
              args: { query: 'calendar' }
            }
          }
        ]
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: 'toolu_search',
              name: 'ToolSearch',
              response: { content: resultContent }
            }
          }
        ]
      }
    ]);
    expect((interactions.value.body as Record<string, unknown>).input).toEqual([
      {
        type: 'function_call',
        id: 'toolu_search',
        name: 'ToolSearch',
        arguments: { query: 'calendar' }
      },
      {
        type: 'function_result',
        call_id: 'toolu_search',
        name: 'ToolSearch',
        result: [{ type: 'text', text: resultContent }]
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

  it('filters non-string enum values from tool schemas', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-pro',
      input: 'Run tool',
      tools: [
        {
          name: 'enum_tool',
          type: 'function',
          parameters: {
            type: 'object',
            properties: {
              flag: {
                type: 'boolean',
                description: 'Set to true to clear.',
                enum: [true]
              },
              mode: {
                type: 'string',
                enum: ['fast', 'safe', 42]
              },
              tags: {
                type: 'array',
                items: {
                  type: 'string'
                }
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
            name: 'enum_tool',
            parameters: {
              type: 'object',
              properties: {
                flag: {
                  type: 'boolean',
                  description: 'Set to true to clear.'
                },
                mode: {
                  type: 'string',
                  enum: ['fast', 'safe']
                },
                tags: {
                  type: 'array',
                  items: {
                    type: 'string'
                  }
                }
              }
            }
          }
        ]
      }
    ]);
  });

  it('adds a default items field to array tool parameters without one', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-pro',
      input: 'Run tool',
      tools: [
        {
          name: 'array_tool',
          type: 'function',
          parameters: {
            type: 'object',
            properties: {
              ids: {
                type: 'array',
                description: 'List of ids.'
              },
              nested: {
                type: 'array',
                items: {
                  type: 'array'
                }
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
            name: 'array_tool',
            parameters: {
              type: 'object',
              properties: {
                ids: {
                  type: 'array',
                  description: 'List of ids.',
                  items: {}
                },
                nested: {
                  type: 'array',
                  items: {
                    type: 'array',
                    items: {}
                  }
                }
              }
            }
          }
        ]
      }
    ]);
  });

  it('maps request-level thinking controls into Gemini thinkingConfig', () => {
    const anthropicParsed = parseAnthropicMessagesRequest({
      model: 'gemini-2.5-pro',
      max_tokens: 128,
      thinking: {
        type: 'enabled',
        budget_tokens: 2048
      },
      messages: [
        {
          role: 'user',
          content: 'Think before answering.'
        }
      ]
    });

    expect(anthropicParsed.ok).toBe(true);
    if (!anthropicParsed.ok) {
      return;
    }

    const anthropicBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest: anthropicParsed.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(anthropicBuilt.ok).toBe(true);
    if (!anthropicBuilt.ok) {
      return;
    }

    expect((anthropicBuilt.value.body as Record<string, unknown>).generationConfig).toMatchObject({
      maxOutputTokens: 128,
      thinkingConfig: {
        thinkingBudget: 2048,
        includeThoughts: true
      }
    });

    const openAIParsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-flash',
      input: 'Think before answering.',
      reasoning: {
        effort: 'high'
      }
    });

    expect(openAIParsed.ok).toBe(true);
    if (!openAIParsed.ok) {
      return;
    }

    const openAIBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-flash:generateContent'
      } as never,
      standardRequest: openAIParsed.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(openAIBuilt.ok).toBe(true);
    if (!openAIBuilt.ok) {
      return;
    }

    expect((openAIBuilt.value.body as Record<string, unknown>).generationConfig).toEqual({
      thinkingConfig: {
        includeThoughts: true
      }
    });

    const disabledParsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-flash',
      input: 'Answer directly.',
      reasoning: {
        effort: 'high'
      },
      thinking: {
        type: 'disabled'
      }
    });

    expect(disabledParsed.ok).toBe(true);
    if (!disabledParsed.ok) {
      return;
    }

    const disabledBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-flash:generateContent'
      } as never,
      standardRequest: disabledParsed.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(disabledBuilt.ok).toBe(true);
    if (!disabledBuilt.ok) {
      return;
    }

    expect((disabledBuilt.value.body as Record<string, unknown>).generationConfig).toEqual({
      thinkingConfig: {
        thinkingBudget: 0
      }
    });
  });

  it('keeps Anthropic opaque reasoning out of Gemini content parts', () => {
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

  it('does not map foreign encrypted reasoning into Gemini thought signatures', () => {
    const openAIParsed = parseOpenAIResponsesRequest({
      model: 'gemini-2.5-pro',
      input: [
        {
          type: 'reasoning',
          encrypted_content: 'openai-encrypted-thinking'
        },
        {
          type: 'function_call',
          call_id: 'call_lookup',
          name: 'lookup_value',
          arguments: '{"key":"live"}'
        }
      ]
    });

    expect(openAIParsed.ok).toBe(true);
    if (!openAIParsed.ok) {
      return;
    }

    const openAIBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest: openAIParsed.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(openAIBuilt.ok).toBe(true);
    if (!openAIBuilt.ok) {
      return;
    }

    expect((openAIBuilt.value.body as Record<string, unknown>).contents).toEqual([
      {
        role: 'model',
        parts: [
          {
            functionCall: {
              id: 'call_lookup',
              name: 'lookup_value',
              args: {
                key: 'live'
              }
            }
          }
        ]
      }
    ]);

    const anthropicParsed = parseAnthropicMessagesRequest({
      model: 'gemini-2.5-pro',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'redacted_thinking',
              data: 'anthropic-redacted-thinking'
            },
            {
              type: 'tool_use',
              id: 'toolu_lookup',
              name: 'lookup_value',
              input: {
                key: 'live'
              }
            }
          ]
        }
      ]
    });

    expect(anthropicParsed.ok).toBe(true);
    if (!anthropicParsed.ok) {
      return;
    }

    const anthropicBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest: anthropicParsed.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(anthropicBuilt.ok).toBe(true);
    if (!anthropicBuilt.ok) {
      return;
    }

    expect((anthropicBuilt.value.body as Record<string, unknown>).contents).toEqual([
      {
        role: 'model',
        parts: [
          {
            functionCall: {
              id: 'toolu_lookup',
              name: 'lookup_value',
              args: {
                key: 'live'
              }
            }
          }
        ]
      }
    ]);
  });

  it('keeps Responses encrypted reasoning out of Gemini requests', () => {
    const standardRequest: StandardRequest = {
      model: 'gemini-2.5-pro',
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              id: 'rs_original',
              source_format: OPENAI_RESPONSES_REASONING_FORMAT,
              encrypted_content: 'encrypted-original'
            },
            {
              type: 'tool_use',
              id: 'call_lookup',
              name: 'lookup_value',
              input: {
                key: 'live'
              }
            }
          ]
        }
      ]
    };

    const generateBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-2.5-pro:generateContent'
      } as never,
      standardRequest,
      config: {
        geminiApiKey: 'test-key',
        geminiBaseUrl: 'https://provider.example',
        geminiApiVersion: 'v1beta'
      } as never
    });

    expect(generateBuilt.ok).toBe(true);
    if (!generateBuilt.ok) {
      return;
    }

    const generateContents = (generateBuilt.value.body as Record<string, unknown>).contents as Array<{
      parts: Array<Record<string, unknown>>;
    }>;
    const generateParts = generateContents[0]?.parts || [];
    expect(generateParts).toEqual([{
      functionCall: {
        id: 'call_lookup',
        name: 'lookup_value',
        args: {
          key: 'live'
        }
      }
    }]);

    const interactionsBuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/interactions'
      } as never,
      standardRequest,
      config: {
        geminiApiKey: 'test-key',
        geminiBaseUrl: 'https://provider.example',
        geminiApiVersion: 'v1beta'
      } as never,
      targetProviderConfig: {
        type: 'gemini_interactions'
      } as never
    });

    expect(interactionsBuilt.ok).toBe(true);
    if (!interactionsBuilt.ok) {
      return;
    }

    const interactionInput = (interactionsBuilt.value.body as Record<string, unknown>).input as Array<
      Record<string, unknown>
    >;
    expect(interactionInput).toHaveLength(1);
    expect(interactionInput[0]).toMatchObject({
      type: 'function_call',
      id: 'call_lookup',
      name: 'lookup_value'
    });
  });

  it('formats standard reasoning as Gemini thought parts', () => {
    const response: StandardResponse = {
      id: 'resp_reasoning',
      object: 'response',
      status: 'completed',
      model: 'gemini-2.5-pro',
      output_text: 'Visible answer.',
      output: [
        {
          id: 'rs_1',
          type: 'reasoning',
          status: 'completed',
          source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          summary: [{ type: 'summary_text', text: 'Need a lookup.' }],
          content: [{ type: 'reasoning_text', text: 'Call lookup before answering.' }],
          encrypted_content: 'gemini-response-thinking-signature'
        },
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'Visible answer.', annotations: [] }]
        },
        {
          id: 'call_lookup',
          type: 'function_call',
          call_id: 'call_lookup',
          name: 'lookup_value',
          arguments: '{"key":"live"}',
          status: 'completed'
        }
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15
      },
      finish_reason: 'tool_use'
    };

    const payload = formatGeminiGenerateContentResponse(response);

    expect(payload).toMatchObject({
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                text: 'Need a lookup.\nCall lookup before answering.',
                thought: true
              },
              {
                text: 'Visible answer.'
              },
              {
                thoughtSignature: 'gemini-response-thinking-signature',
                functionCall: {
                  id: 'call_lookup',
                  name: 'lookup_value',
                  args: {
                    key: 'live'
                  }
                }
              }
            ]
          }
        }
      ]
    });
  });

  it('keeps multiple signed Gemini thought parts separate when formatting a response', () => {
    const parsed = geminiGenerateContentTargetAdapter.toStandardResponse({
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                text: 'First thought summary.',
                thought: true,
                thoughtSignature: 'gemini-thought-signature-1'
              },
              {
                text: 'Second thought summary.',
                thought: true,
                thoughtSignature: 'gemini-thought-signature-2'
              },
              {
                text: 'Visible answer.'
              }
            ]
          },
          finishReason: 'STOP'
        }
      ],
      usageMetadata: {
        promptTokenCount: 8,
        candidatesTokenCount: 6,
        totalTokenCount: 14
      },
      modelVersion: 'gemini-3.5-flash'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const reasoningItems = parsed.value.output.filter((item) => item.type === 'reasoning');
    expect(reasoningItems).toHaveLength(2);
    expect(reasoningItems[0]).toMatchObject({
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: 'First thought summary.' }],
      encrypted_content: 'gemini-thought-signature-1',
      reasoning_details: [
        {
          type: 'reasoning.text',
          text: 'First thought summary.',
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index: 0
        },
        {
          type: 'reasoning.encrypted',
          data: 'gemini-thought-signature-1',
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index: 0
        }
      ]
    });
    expect(reasoningItems[1]).toMatchObject({
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: 'Second thought summary.' }],
      encrypted_content: 'gemini-thought-signature-2',
      reasoning_details: [
        {
          type: 'reasoning.text',
          text: 'Second thought summary.',
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index: 1
        },
        {
          type: 'reasoning.encrypted',
          data: 'gemini-thought-signature-2',
          format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          index: 1
        }
      ]
    });

    const formatted = formatGeminiGenerateContentResponse(parsed.value) as Record<string, any>;
    expect(formatted.candidates[0].content.parts).toEqual([
      {
        text: 'First thought summary.',
        thought: true,
        thoughtSignature: 'gemini-thought-signature-1'
      },
      {
        text: 'Second thought summary.',
        thought: true,
        thoughtSignature: 'gemini-thought-signature-2'
      },
      {
        text: 'Visible answer.'
      }
    ]);

    const chatResponse = formatOpenAIChatCompletionsResponse(parsed.value) as Record<
      string,
      any
    >;
    const chatRequest = parseOpenAIChatCompletionsRequest({
      model: 'gemini-3.5-flash',
      messages: [
        chatResponse.choices[0].message,
        {
          role: 'user',
          content: 'Continue.'
        }
      ]
    });
    expect(chatRequest.ok).toBe(true);
    if (!chatRequest.ok) {
      return;
    }

    const rebuilt = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-3.5-flash:generateContent'
      } as never,
      standardRequest: chatRequest.value,
      config: {
        geminiApiKey: 'sk-test',
        geminiBaseUrl: 'https://mock.local',
        geminiApiVersion: 'v1beta'
      } as never
    });
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) {
      return;
    }

    const rebuiltContents = (rebuilt.value.body as Record<string, unknown>).contents as Array<{
      parts: Array<Record<string, unknown>>;
    }>;
    expect(rebuiltContents[0]?.parts).toEqual([
      {
        text: 'First thought summary.',
        thought: true,
        thoughtSignature: 'gemini-thought-signature-1'
      },
      {
        text: 'Second thought summary.',
        thought: true,
        thoughtSignature: 'gemini-thought-signature-2'
      },
      {
        text: 'Visible answer.'
      }
    ]);
  });

  it('keeps multiple signed Gemini thought parts separate when rebuilding a request', () => {
    const parsed = parseGeminiGenerateContentRequest(
      {
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Solve this.' }]
          },
          {
            role: 'model',
            parts: [
              {
                text: 'First thought summary.',
                thought: true,
                thoughtSignature: 'gemini-thought-signature-1'
              },
              {
                text: 'Second thought summary.',
                thought: true,
                thoughtSignature: 'gemini-thought-signature-2'
              },
              {
                text: 'Visible answer.'
              }
            ]
          },
          {
            role: 'user',
            parts: [{ text: 'Continue.' }]
          }
        ]
      },
      'gemini-3.5-flash'
    );

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-3.5-flash:generateContent'
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

    expect((built.value.body as Record<string, unknown>).contents).toEqual([
      {
        role: 'user',
        parts: [{ text: 'Solve this.' }]
      },
      {
        role: 'model',
        parts: [
          {
            text: 'First thought summary.',
            thought: true,
            thoughtSignature: 'gemini-thought-signature-1'
          },
          {
            text: 'Second thought summary.',
            thought: true,
            thoughtSignature: 'gemini-thought-signature-2'
          },
          {
            text: 'Visible answer.'
          }
        ]
      },
      {
        role: 'user',
        parts: [{ text: 'Continue.' }]
      }
    ]);
  });

  it('keeps Responses reasoning envelopes separate from Gemini function calls', () => {
    const response: StandardResponse = {
      id: 'resp_responses_reasoning',
      object: 'response',
      status: 'completed',
      model: 'gpt-5.6-sol',
      output_text: '',
      output: [
        {
          id: 'rs_responses_reasoning_1',
          type: 'reasoning',
          status: 'completed',
          summary: [],
          source_format: 'openai-responses-v1',
          encrypted_content: 'encrypted-responses-reasoning'
        },
        {
          id: 'call_lookup',
          type: 'function_call',
          call_id: 'call_lookup',
          name: 'lookup_value',
          arguments: '{"key":"live"}',
          status: 'completed'
        }
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15
      },
      finish_reason: 'tool_use'
    };

    const payload = formatGeminiGenerateContentResponse(response) as Record<string, any>;
    const parts = payload.candidates[0].content.parts as Array<Record<string, unknown>>;

    expect(parts[0]).toMatchObject({
      thought: true,
      thoughtSignature: expect.stringMatching(/^ccr-openai-responses-reasoning-v1:/)
    });
    expect(parts[0]?.thoughtSignature).not.toBe('encrypted-responses-reasoning');
    expect(parts[1]).toEqual({
      functionCall: {
        id: 'call_lookup',
        name: 'lookup_value',
        args: {
          key: 'live'
        }
      }
    });
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
        thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        arguments: '{"city":"Shanghai"}'
      })
    ]);

    const anthropic = formatAnthropicMessagesResponse(parsed.value);
    expect(anthropic.stop_reason).toBe('tool_use');
    const anthropicContent = anthropic.content as Array<Record<string, unknown>>;
    expect(anthropicContent).toEqual([
      {
        type: 'thinking',
        thinking: 'Need to call the weather tool.',
        signature: expect.any(String)
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
    const thinkingBlock = anthropicContent[0]!;
    expect(
      decodeReasoningTransportEnvelope(String(thinkingBlock.signature))
    ).toMatchObject({
      format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      data: 'gemini-function-signature',
      kind: 'signature'
    });
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

  it('preserves a signature that is present on an empty Gemini text Part', () => {
    const rawParts = [
      {
        text: '',
        thoughtSignature: 'empty-text-part-signature'
      },
      {
        text: 'Visible answer.'
      }
    ];
    const parsed = geminiGenerateContentTargetAdapter.toStandardResponse({
      candidates: [
        {
          content: {
            role: 'model',
            parts: rawParts
          },
          finishReason: 'STOP'
        }
      ],
      usageMetadata: {
        promptTokenCount: 6,
        candidatesTokenCount: 3,
        totalTokenCount: 9
      },
      modelVersion: 'gemini-3.5-flash'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.output[0]).toMatchObject({
      type: 'message',
      content: [{ type: 'output_text', text: '' }],
      native_item: {
        item_type: 'part',
        raw_payload: rawParts[0],
        capture_state: 'complete',
        position: { item: 0 }
      }
    });
    const formatted = formatGeminiGenerateContentResponse(parsed.value) as Record<string, any>;
    expect(formatted.candidates[0].content.parts).toEqual(rawParts);
  });

  it('preserves text and native Part order when replaying Gemini history', () => {
    const nativePart: ProviderNativeItem = {
      type: 'provider_native_item',
      item_type: 'thought',
      native_id: 'gemini_native_thought',
      raw_payload: {
        text: 'native thought',
        thought: true,
        thoughtSignature: 'native-signature'
      },
      provider_schema_version: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      item_origin: 'native',
      source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      source_origin: {
        provider: 'gemini',
        endpoint: 'gemini-endpoint',
        model: 'gemini-3.5-flash',
        credentialScope: 'gemini-account'
      },
      position: { turn: 0, step: 0, item: 1 },
      capture_state: 'complete'
    };
    const projectedNativePart: ProviderNativeItem = {
      ...nativePart,
      item_type: 'function_call',
      native_id: 'gemini_native_call',
      raw_payload: {
        thoughtSignature: 'call-signature',
        functionCall: {
          id: 'call_native',
          name: 'lookup',
          args: { query: 'weather' }
        }
      },
      position: { turn: 0, step: 0, item: 3 }
    };
    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-3.5-flash:generateContent'
      } as never,
      standardRequest: {
        model: 'gemini-3.5-flash',
        input: [{
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'input_text', text: 'before native' },
            nativePart,
            { type: 'input_text', text: 'between native parts' },
            {
              type: 'tool_use',
              id: 'call_native',
              name: 'lookup',
              input: { query: 'weather' },
              native_item: projectedNativePart
            },
            { type: 'input_text', text: 'after native' }
          ]
        }]
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
    const contents = (built.value.body as Record<string, unknown>).contents as Array<{
      parts: Array<Record<string, unknown>>;
    }>;
    expect(contents[0]?.parts).toEqual([
      { text: 'before native' },
      nativePart.raw_payload,
      { text: 'between native parts' },
      projectedNativePart.raw_payload,
      { text: 'after native' }
    ]);
  });

  it('keeps parallel Gemini calls and results in FC1, FC2, FR1, FR2 order', () => {
    const parsed = parseGeminiGenerateContentRequest(
      {
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Look up both cities.' }]
          },
          {
            role: 'model',
            parts: [
              {
                thoughtSignature: 'signature-fc-1',
                functionCall: {
                  id: 'call_fc_1',
                  name: 'get_weather',
                  args: { city: 'Wuhu' }
                }
              },
              {
                thoughtSignature: 'signature-fc-2',
                functionCall: {
                  id: 'call_fc_2',
                  name: 'get_weather',
                  args: { city: 'Hefei' }
                }
              }
            ]
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call_fc_1',
                  name: 'get_weather',
                  response: { content: '22 C' }
                }
              },
              {
                functionResponse: {
                  id: 'call_fc_2',
                  name: 'get_weather',
                  response: { content: '24 C' }
                }
              }
            ]
          }
        ]
      },
      'gemini-3.5-flash'
    );

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = geminiGenerateContentTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1beta/models/gemini-3.5-flash:generateContent'
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

    const contents = (built.value.body as Record<string, unknown>).contents as Array<{
      role: string;
      parts: Array<Record<string, unknown>>;
    }>;
    expect(contents[1]?.parts.map((part) =>
      (part.functionCall as Record<string, unknown> | undefined)?.id
    )).toEqual(['call_fc_1', 'call_fc_2']);
    expect(contents[2]?.parts.map((part) =>
      (part.functionResponse as Record<string, unknown> | undefined)?.id
    )).toEqual(['call_fc_1', 'call_fc_2']);
    expect([
      ...contents[1]!.parts.map((part) =>
        `FC${(part.functionCall as Record<string, unknown>).id === 'call_fc_1' ? '1' : '2'}`
      ),
      ...contents[2]!.parts.map((part) =>
        `FR${(part.functionResponse as Record<string, unknown>).id === 'call_fc_1' ? '1' : '2'}`
      )
    ]).toEqual(['FC1', 'FC2', 'FR1', 'FR2']);
  });
});
