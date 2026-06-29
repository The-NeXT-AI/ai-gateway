import { describe, expect, it } from 'vitest';
import { parseAnthropicMessagesRequest, parseOpenAIResponsesRequest } from '../source/parsers';
import { openAIResponsesTargetAdapter } from './openai-responses';

describe('openAIResponsesTargetAdapter', () => {
  it('preserves OpenAI server tool usage counters in standard responses', () => {
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_server_tools',
      model: 'gpt-5.1',
      output_text: 'searched',
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14,
        server_tool_use: {
          web_search_requests: 2,
          web_fetch_requests: 1
        }
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.usage.server_tool_use).toEqual({
      web_search_requests: 2,
      web_fetch_requests: 1
    });
  });

  it('converts anthropic tool_use/tool_result history into OpenAI chat tool messages', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-3-5-sonnet-latest',
      stream: true,
      max_tokens: 64,
      messages: [
        { role: 'user', content: '先调用工具' },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_abc',
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
              tool_use_id: 'toolu_abc',
              content: '{"temperature":22}'
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
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
        content: '先调用工具'
      },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'toolu_abc',
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
        tool_call_id: 'toolu_abc',
        content: '{"temperature":22}'
      }
    ]);
  });

  it('keeps restored tool results adjacent to assistant tool_calls before user text', () => {
    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: {
        model: 'deepseek-chat',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'call_weather',
                name: 'get_weather',
                input: {
                  city: 'Shanghai'
                }
              },
              {
                type: 'tool_use',
                id: 'call_time',
                name: 'get_time',
                input: {
                  timezone: 'Asia/Shanghai'
                }
              }
            ]
          },
          {
            type: 'message',
            role: 'user',
            content: [
              {
                type: 'input_text',
                text: 'continue'
              },
              {
                type: 'tool_result',
                tool_use_id: 'call_weather',
                content: '{"temperature":22}',
                result_format: 'function'
              },
              {
                type: 'tool_result',
                tool_use_id: 'call_time',
                content: '{"local_time":"10:00"}',
                result_format: 'function'
              }
            ]
          }
        ]
      },
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.messages).toEqual([
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
          },
          {
            id: 'call_time',
            type: 'function',
            function: {
              name: 'get_time',
              arguments: '{"timezone":"Asia/Shanghai"}'
            }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 'call_weather',
        content: '{"temperature":22}'
      },
      {
        role: 'tool',
        tool_call_id: 'call_time',
        content: '{"local_time":"10:00"}'
      },
      {
        role: 'user',
        content: 'continue'
      }
    ]);
  });

  it('converts Responses reasoning input into OpenAI chat reasoning fields', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'MiniMax-M2.7',
      input: [
        {
          type: 'reasoning',
          id: 'rs_123',
          status: 'completed',
          content: [
            {
              type: 'reasoning_text',
              text: 'previous reasoning'
            }
          ]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'output_text',
              text: 'previous answer'
            }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: 'next turn'
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.messages).toEqual([
      {
        role: 'assistant',
        content: '',
        reasoning_content: 'previous reasoning',
        reasoning_details: [
          {
            type: 'reasoning.text',
            text: 'previous reasoning',
            format: 'openai-responses-v1',
            index: 0
          }
        ]
      },
      {
        role: 'assistant',
        content: 'previous answer'
      },
      {
        role: 'user',
        content: 'next turn'
      }
    ]);
  });

  it('passes explicit Responses thinking options into OpenAI chat targets', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'deepseek-v4-pro',
      reasoning: {
        effort: 'max'
      },
      thinking: {
        type: 'enabled'
      },
      output_config: {
        effort: 'low'
      },
      input: 'hello'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.thinking).toEqual({
      type: 'enabled'
    });
    expect(body.output_config).toEqual({
      effort: 'low'
    });
  });

  it('maps Responses reasoning effort into OpenAI chat thinking options', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'deepseek-v4-pro',
      reasoning: {
        effort: 'max'
      },
      input: 'hello'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.thinking).toEqual({
      type: 'enabled'
    });
    expect(body.output_config).toEqual({
      effort: 'max'
    });
  });

  it('keeps Responses reasoning on assistant tool call messages when targeting OpenAI chat', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'deepseek-v4-pro',
      input: [
        {
          type: 'reasoning',
          id: 'rs_123',
          status: 'completed',
          content: [
            {
              type: 'reasoning_text',
              text: 'need a tool'
            }
          ]
        },
        {
          type: 'function_call',
          call_id: 'call_weather',
          name: 'get_weather',
          arguments: '{"city":"Shanghai"}'
        },
        {
          type: 'function_call_output',
          call_id: 'call_weather',
          output: '{"temperature":22}'
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: 'continue'
            }
          ]
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.messages).toEqual([
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
        ],
        reasoning_content: 'need a tool',
        reasoning_details: [
          {
            type: 'reasoning.text',
            text: 'need a tool',
            format: 'openai-responses-v1',
            index: 0
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 'call_weather',
        content: '{"temperature":22}'
      },
      {
        role: 'user',
        content: 'continue'
      }
    ]);
  });

  it('enables reasoning_split automatically when targeting OpenAI chat/completions', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'MiniMax-M2.7',
      input: 'hello'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(parsed.value.reasoning_split).toBeUndefined();
    expect(body.reasoning_split).toBe(true);
  });

  it('requests usage in OpenAI chat/completions streams when targeting chat from Responses', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'glm-5',
      input: 'hello',
      stream: true
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).stream_options).toEqual({
      include_usage: true
    });
  });

  it('can disable usage requests in OpenAI chat/completions streams for incompatible targets', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'legacy-chat',
      input: 'hello',
      stream: true
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions',
        openaiChatStreamUsage: 'disabled'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).stream_options).toBeUndefined();
  });

  it('passes reasoning_split when targeting OpenAI chat/completions', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'MiniMax-M2.7',
      reasoning_split: true,
      input: 'hello'
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.reasoning_split).toBe(true);
  });

  it('flattens OpenAI Responses namespace tools when targeting OpenAI chat', () => {
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
              strict: false,
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
            },
            {
              name: 'js_reset',
              type: 'function',
              parameters: {
                type: 'object',
                properties: {},
                additionalProperties: false
              },
              description: 'Reset JavaScript state.'
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

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'mcp__node_repl___js',
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
          description: 'Run JavaScript.',
          strict: false
        }
      },
      {
        type: 'function',
        function: {
          name: 'mcp__node_repl___js_reset',
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false
          },
          description: 'Reset JavaScript state.'
        }
      }
    ]);
    expect(body.tool_choice).toEqual({
      type: 'function',
      function: {
        name: 'mcp__node_repl___js'
      }
    });
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

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body).not.toHaveProperty('tools');
  });

  it('passes explicit OpenAI Responses web_search tools through as hosted tools', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.4',
      input: 'What happened today?',
      tools: [
        {
          type: 'web_search',
          search_context_size: 'low',
          filters: {
            allowed_domains: ['openai.com']
          }
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: 'web_search',
        search_context_size: 'low',
        filters: {
          allowed_domains: ['openai.com']
        }
      }
    ]);
  });

  it('maps explicit Anthropic web_search server tools to OpenAI Responses web_search', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 256,
      tools: [
        {
          type: 'web_search_20250305',
          name: 'web_search',
          allowed_domains: ['docs.anthropic.com'],
          blocked_domains: ['example.com']
        }
      ],
      messages: [{ role: 'user', content: 'Search the docs' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: 'web_search',
        filters: {
          allowed_domains: ['docs.anthropic.com'],
          blocked_domains: ['example.com']
        }
      }
    ]);
  });

  it('does not expose hosted web_search as an OpenAI chat/completions function tool', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 256,
      tools: [
        {
          type: 'web_search_20250305',
          name: 'web_search'
        }
      ],
      messages: [{ role: 'user', content: 'Search the docs' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body).not.toHaveProperty('tools');
  });

  it('can emit Anthropic-style tools for OpenAI chat/completions compatibility providers', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'glm-5.1',
      max_tokens: 256,
      tools: [
        {
          name: 'web_search',
          input_schema: {
            type: 'object',
            properties: {
              prompt: {
                type: 'string'
              }
            },
            required: ['prompt']
          },
          description: 'Search the web.'
        }
      ],
      messages: [{ role: 'user', content: 'Search the docs' }]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {}
      } as never,
      standardRequest: parsed.value,
      config: {
        openaiApiKey: 'sk-test',
        openaiBaseUrl: 'https://mock.local/v1'
      } as never,
      targetProviderConfig: {
        type: 'openai_chat_completions',
        openaiChatToolsFormat: 'anthropic'
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
        input_schema: {
          type: 'object',
          properties: {
            prompt: {
              type: 'string'
            }
          },
          required: ['prompt']
        },
        description: 'Search the web.'
      }
    ]);
  });
});
