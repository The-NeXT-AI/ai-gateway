import { describe, expect, it } from 'vitest';
import {
  parseAnthropicMessagesRequest,
  parseOpenAIChatCompletionsRequest,
  parseOpenAIResponsesRequest
} from '../source/parsers';
import { anthropicMessagesTargetAdapter } from './anthropic-messages';

describe('anthropicMessagesTargetAdapter', () => {
  it('sets a default max_tokens when converted request does not provide one', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'glm-5',
      messages: [{ role: 'user', content: 'what is your knowledge cutoff date' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.max_tokens).toBe(1024);
  });

  it('keeps max_tokens from chat/completions request when provided', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'glm-5',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'hello' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.max_tokens).toBe(256);
  });

  it('passes through stream=true for anthropic upstream streaming', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'glm-5',
      stream: true,
      messages: [{ role: 'user', content: 'hello' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.stream).toBe(true);
  });

  it('maps tools, tool_choice, assistant tool_calls, and tool messages into anthropic messages', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'glm-5',
      tools: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get current weather.',
            parameters: {
              type: 'object',
              properties: {
                city: { type: 'string' }
              },
              required: ['city']
            }
          }
        }
      ],
      tool_choice: {
        type: 'function',
        function: {
          name: 'get_weather'
        }
      },
      messages: [
        { role: 'user', content: 'What is the weather in Shanghai?' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"city":"Shanghai"}'
              }
            }
          ]
        },
        {
          role: 'tool',
          tool_call_id: 'call_weather',
          content: '{"temperature":22}'
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        name: 'get_weather',
        description: 'Get current weather.',
        input_schema: {
          type: 'object',
          properties: {
            city: { type: 'string' }
          },
          required: ['city']
        }
      }
    ]);
    expect(body.tool_choice).toEqual({
      type: 'tool',
      name: 'get_weather'
    });
    expect(body.messages).toEqual([
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'What is the weather in Shanghai?'
          }
        ]
      },
      {
        role: 'assistant',
        content: [
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
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_weather',
            content: '{"temperature":22}'
          }
        ]
      }
    ]);
  });

  it('maps native Responses tool-search history into Anthropic tool history', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      tools: [
        {
          type: 'tool_search',
          execution: 'client',
          description: 'Find tools.',
          parameters: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query']
          }
        }
      ],
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
              description: 'Create a calendar event.',
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

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).tools).toEqual([
      {
        name: 'ToolSearch',
        description: 'Find tools.',
        input_schema: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query']
        }
      },
      {
        name: 'calendar_create',
        description: 'Create a calendar event.',
        input_schema: { type: 'object', properties: {} },
        defer_loading: true
      }
    ]);
    expect((built.value.body as Record<string, unknown>).messages).toEqual([
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'search_123',
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
            tool_use_id: 'search_123',
            content: [{ type: 'tool_reference', tool_name: 'calendar_create' }]
          }
        ]
      }
    ]);
  });

  it('keeps deferred tool declarations and references aligned after name collisions', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      tools: [
        {
          type: 'tool_search',
          execution: 'client',
          parameters: { type: 'object', properties: {} }
        }
      ],
      input: [
        {
          type: 'tool_search_call',
          execution: 'client',
          call_id: 'search_collision',
          status: 'completed',
          arguments: { query: 'colliding names' }
        },
        {
          type: 'tool_search_output',
          execution: 'client',
          call_id: 'search_collision',
          status: 'completed',
          tools: [
            {
              type: 'function',
              name: 'foo.bar',
              defer_loading: true,
              parameters: { type: 'object', properties: {} }
            },
            {
              type: 'function',
              name: 'foo_bar',
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

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    const declarations = (body.tools as Array<Record<string, unknown>>).filter(
      (tool) => tool.name !== 'ToolSearch'
    );
    const declarationNames = declarations.map((tool) => tool.name);
    const messages = body.messages as Array<Record<string, unknown>>;
    const resultBlocks = messages[1]?.content as Array<Record<string, unknown>>;
    const references = resultBlocks[0]?.content as Array<Record<string, unknown>>;
    const referenceNames = references.map((reference) => reference.tool_name);

    expect(declarationNames).toHaveLength(2);
    expect(new Set(declarationNames).size).toBe(2);
    expect(declarations.every((tool) => tool.defer_loading === true)).toBe(true);
    expect(referenceNames).toEqual(declarationNames);
  });

  it.each([
    {
      name: 'a collision with the ToolSearch declaration',
      configuredTools: [],
      outputStatus: 'completed',
      outputTool: {
        type: 'function',
        name: 'ToolSearch',
        parameters: { type: 'object', properties: {} }
      },
      expectedToolNames: ['ToolSearch']
    },
    {
      name: 'a conflicting configured definition',
      configuredTools: [
        {
          type: 'function',
          name: 'calendar_create',
          defer_loading: true,
          parameters: {
            type: 'object',
            properties: { date: { type: 'string' } }
          }
        }
      ],
      outputStatus: 'completed',
      outputTool: {
        type: 'function',
        name: 'calendar_create',
        parameters: {
          type: 'object',
          properties: { event_id: { type: 'number' } }
        }
      },
      expectedToolNames: ['ToolSearch', 'calendar_create']
    },
    {
      name: 'an incomplete discovery result',
      configuredTools: [],
      outputStatus: 'incomplete',
      outputTool: {
        type: 'function',
        name: 'calendar_create',
        parameters: { type: 'object', properties: {} }
      },
      expectedToolNames: ['ToolSearch']
    }
  ])('serializes tool-search output for $name', (scenario) => {
    const callId = `search_${scenario.outputTool.name}_${scenario.outputStatus}`;
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      tools: [
        {
          type: 'tool_search',
          execution: 'client',
          parameters: { type: 'object', properties: {} }
        },
        ...scenario.configuredTools
      ],
      input: [
        {
          type: 'tool_search_call',
          execution: 'client',
          call_id: callId,
          status: 'completed',
          arguments: { query: 'calendar' }
        },
        {
          type: 'tool_search_output',
          execution: 'client',
          call_id: callId,
          status: scenario.outputStatus,
          tools: [scenario.outputTool]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    const toolNames = (body.tools as Array<Record<string, unknown>>).map(
      (tool) => tool.name
    );
    const messages = body.messages as Array<Record<string, unknown>>;
    const resultBlocks = messages[1]?.content as Array<Record<string, unknown>>;
    const resultContent = resultBlocks[0]?.content;

    expect(toolNames).toEqual(scenario.expectedToolNames);
    expect(resultContent).toBe(
      JSON.stringify({
        type: 'tool_search_output',
        execution: 'client',
        status: scenario.outputStatus,
        tools: [scenario.outputTool]
      })
    );
  });

  it('round-trips Anthropic tool references without an empty result body', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      tools: [
        {
          name: 'ToolSearch',
          description: 'Find tools.',
          input_schema: {
            type: 'object',
            properties: { query: { type: 'string' } }
          }
        },
        {
          name: 'calendar_create',
          description: 'Create a calendar event.',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_search',
              name: 'ToolSearch',
              input: { query: 'calendar create' }
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
                { type: 'tool_reference', tool_name: 'calendar_create' },
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

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).tools).toEqual([
      {
        name: 'ToolSearch',
        description: 'Find tools.',
        input_schema: {
          type: 'object',
          properties: { query: { type: 'string' } }
        }
      },
      {
        name: 'calendar_create',
        description: 'Create a calendar event.',
        input_schema: { type: 'object', properties: {} },
        defer_loading: true
      }
    ]);
    expect((built.value.body as Record<string, unknown>).messages).toEqual([
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'toolu_search',
            name: 'ToolSearch',
            input: { query: 'calendar create' }
          }
        ]
      },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'toolu_search',
            content: [{ type: 'tool_reference', tool_name: 'calendar_create' }]
          }
        ]
      }
    ]);
  });

  it('maps chat reasoning_details into Anthropic thinking blocks before tool_use', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'MiniMax-M2.7',
      messages: [
        { role: 'user', content: 'Use a tool' },
        {
          role: 'assistant',
          content: '',
          reasoning_content: 'interleaved thinking',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'interleaved thinking',
              format: 'anthropic-claude-v1',
              signature: 'sig_123',
              index: 0
            }
          ],
          tool_calls: [
            {
              id: 'call_weather',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"city":"Shanghai"}'
              }
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.messages).toEqual([
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'Use a tool'
          }
        ]
      },
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: 'interleaved thinking',
            signature: 'sig_123'
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
      }
    ]);
  });

  it('flattens OpenAI Responses namespace tools when targeting Anthropic', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.4',
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
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        name: 'mcp__node_repl___js',
        description: 'Run JavaScript.',
        input_schema: {
          type: 'object',
          required: ['code'],
          properties: {
            code: {
              type: 'string'
            }
          },
          additionalProperties: false
        }
      }
    ]);
  });

  it('does not add web search tools when the client did not declare one', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.4',
      input: 'What happened today?'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body).not.toHaveProperty('tools');
  });

  it('maps explicit OpenAI Responses web_search tools to Anthropic server tools', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.4',
      input: 'What happened today?',
      tools: [
        {
          type: 'web_search',
          filters: {
            allowed_domains: ['openai.com'],
            blocked_domains: ['example.com']
          }
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: 'web_search_20250305',
        name: 'web_search',
        allowed_domains: ['openai.com'],
        blocked_domains: ['example.com']
      }
    ]);
  });

  it('preserves explicit Anthropic web_search server tools', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.4',
      input: 'What happened today?',
      tools: [
        {
          type: 'web_search_20250305',
          name: 'web_search',
          max_uses: 3,
          allowed_domains: ['docs.anthropic.com']
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: 'web_search_20250305',
        name: 'web_search',
        max_uses: 3,
        allowed_domains: ['docs.anthropic.com']
      }
    ]);
  });

  it('keeps a custom function named web_search as a client tool', () => {
    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'glm-5',
      tools: [
        {
          type: 'function',
          function: {
            name: 'web_search',
            description: 'Custom search function.',
            parameters: {
              type: 'object',
              properties: {
                query: { type: 'string' }
              },
              required: ['query']
            }
          }
        }
      ],
      messages: [{ role: 'user', content: 'Use my custom search function' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = anthropicMessagesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        anthropicApiKey: 'sk-test',
        anthropicBaseUrl: 'https://mock.local'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        name: 'web_search',
        description: 'Custom search function.',
        input_schema: {
          type: 'object',
          properties: {
            query: { type: 'string' }
          },
          required: ['query']
        }
      }
    ]);
  });
});
