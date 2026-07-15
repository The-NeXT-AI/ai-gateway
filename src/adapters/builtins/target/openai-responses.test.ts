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

  it('parses GPT-5.6 cache write counters from OpenAI usage details', () => {
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_cache_write',
      model: 'gpt-5.6',
      output_text: 'cached',
      usage: {
        input_tokens: 120,
        output_tokens: 8,
        total_tokens: 128,
        input_tokens_details: {
          cached_tokens: 32,
          cache_write_tokens: 16
        }
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.usage.cache_read_tokens).toBe(32);
    expect(parsed.value.usage.cache_write_tokens).toBe(16);
  });

  it('parses GPT-5.6 cache write counters from chat usage details', () => {
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'chatcmpl_cache_write',
      object: 'chat.completion',
      model: 'gpt-5.6',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'cached'
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: 120,
        completion_tokens: 8,
        total_tokens: 128,
        prompt_tokens_details: {
          cached_tokens: 32,
          cache_write_tokens: 16
        }
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.usage.cache_read_tokens).toBe(32);
    expect(parsed.value.usage.cache_write_tokens).toBe(16);
  });

  it('maps Messages output effort to Responses reasoning effort', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'source-model',
      max_tokens: 64,
      output_config: {
        effort: 'high'
      },
      messages: [{ role: 'user', content: 'hello' }]
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

    expect((built.value.body as Record<string, unknown>).reasoning).toEqual({
      effort: 'high'
    });
  });

  it('omits Messages output configuration from Responses requests', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'source-model',
      max_tokens: 64,
      output_config: {
        effort: 'high',
        task_budget: 512
      },
      messages: [{ role: 'user', content: 'hello' }]
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

    expect(built.value.body).not.toHaveProperty('output_config');
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
    expect(body.reasoning_split).toBeUndefined();
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
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
        type: 'openai_chat_completions',
        openaiChatReasoningSplit: 'enabled'
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

  it('does not send Responses reasoning input as OpenAI chat message fields to generic targets by default', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'strict-chat',
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
        name: 'strict-chat',
        models: ['strict-chat'],
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
        type: 'openai_chat_completions',
        openaiChatThinkingOptions: 'enabled'
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
        type: 'openai_chat_completions',
        openaiChatThinkingOptions: 'enabled'
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

  it('does not pass OpenAI chat thinking options to generic targets by default', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'strict-chat',
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
        name: 'strict-chat',
        models: ['strict-chat'],
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
  });

  it('passes OpenAI chat thinking options automatically for Zhipu targets', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'glm-5.2',
      reasoning: {
        effort: 'high'
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
        name: 'Zhipu AI (China) - Coding Plan',
        models: ['glm-5.2'],
        type: 'openai_chat_completions',
        baseurl: 'https://open.bigmodel.cn/api/paas/v4'
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
    expect(body.reasoning_effort).toBe('high');
    expect(body.output_config).toBeUndefined();
  });

  it('passes OpenAI chat thinking options automatically for Zhipu domains', () => {
    for (const baseurl of [
      'https://api.z.ai/api/paas/v4',
      'https://open.bigmodel.cn/api/paas/v4',
      'https://api.zhipuai.cn/api/paas/v4'
    ]) {
      const parsed = parseOpenAIResponsesRequest({
        model: 'glm-5.2',
        reasoning: {
          effort: 'high'
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
          name: 'generic-openai-compatible',
          models: ['glm-5.2'],
          type: 'openai_chat_completions',
          baseurl
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
      expect(body.reasoning_effort).toBe('high');
      expect(body.output_config).toBeUndefined();
      expect(body.reasoning_split).toBeUndefined();
    }
  });

  it('does not match OpenAI chat thinking options on lookalike Zhipu hosts', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'glm-5.2',
      reasoning: {
        effort: 'high'
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
        name: 'generic-openai-compatible',
        models: ['glm-5.2'],
        type: 'openai_chat_completions',
        baseurl: 'https://api.z.ai.evil.test/v1'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
    expect(body.reasoning_effort).toBeUndefined();
  });

  it('passes DeepSeek thinking options as reasoning_effort for DeepSeek domains', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'v4-pro',
      reasoning: {
        effort: 'xhigh'
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
        name: 'generic-openai-compatible',
        models: ['v4-pro'],
        type: 'openai_chat_completions',
        baseurl: 'https://api.deepseek.com'
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
    expect(body.reasoning_effort).toBe('max');
    expect(body.output_config).toBeUndefined();
    expect(body.reasoning_split).toBeUndefined();
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
        type: 'openai_chat_completions',
        baseurl: 'https://api.deepseek.com'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(body.reasoning_split).toBeUndefined();
    expect(body.thinking).toBeUndefined();
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

  it('keeps Responses reasoning for DeepSeek OpenAI chat domains', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'v4-pro',
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
        name: 'generic-openai-compatible',
        models: ['v4-pro'],
        type: 'openai_chat_completions',
        baseurl: 'https://api.deepseek.com'
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
        reasoning_content: 'need a tool',
        reasoning_details: [
          {
            type: 'reasoning.text',
            text: 'need a tool',
            format: 'openai-responses-v1',
            index: 0
          }
        ],
        content: ''
      },
      {
        role: 'user',
        content: 'continue'
      }
    ]);
  });

  it('keeps interleaved thinking on DeepSeek tool-call history', () => {
    const body = buildInterleavedThinkingOpenAIChatBody({
      name: 'generic-openai-compatible',
      models: ['v4-pro'],
      type: 'openai_chat_completions',
      baseurl: 'https://api.deepseek.com'
    });

    expect(body.reasoning_split).toBeUndefined();
    expect(body.thinking).toEqual({
      type: 'enabled'
    });
    expect(body.reasoning_effort).toBe('high');
    expect(body.output_config).toBeUndefined();
    expect(body.messages).toEqual(expectedInterleavedThinkingToolMessages(true));
  });

  it('keeps interleaved thinking on Xiaomi MiMo tool-call history', () => {
    for (const baseurl of [
      'https://api.xiaomimimo.com/v1',
      'https://token-plan-cn.xiaomimimo.com/v1'
    ]) {
      const body = buildInterleavedThinkingOpenAIChatBody({
        name: 'generic-openai-compatible',
        models: ['mimo-v2.5-pro'],
        type: 'openai_chat_completions',
        baseurl
      });

      expect(body.reasoning_split).toBeUndefined();
      expect(body.thinking).toEqual({
        type: 'enabled'
      });
      expect(body.reasoning_effort).toBeUndefined();
      expect(body.output_config).toBeUndefined();
      expect(body.messages).toEqual(expectedXiaomiMimoInterleavedThinkingToolMessages());
    }
  });

  it('strips interleaved thinking message fields for Zhipu while keeping tool-call history valid', () => {
    const body = buildInterleavedThinkingOpenAIChatBody({
      name: 'generic-openai-compatible',
      models: ['glm-5.2'],
      type: 'openai_chat_completions',
      baseurl: 'https://open.bigmodel.cn/api/paas/v4'
    });

    expect(body.reasoning_split).toBeUndefined();
    expect(body.thinking).toEqual({
      type: 'enabled'
    });
    expect(body.reasoning_effort).toBe('medium');
    expect(body.output_config).toBeUndefined();
    expect(body.messages).toEqual(expectedInterleavedThinkingToolMessages(false));
  });

  it('converts Anthropic thinking/tool_use history per OpenAI chat provider', () => {
    const cases = [
      {
        baseurl: 'https://api.deepseek.com',
        expectedReasoningSplit: undefined,
        expectedThinking: { type: 'enabled' },
        expectedReasoningEffort: 'high',
        expectedMessages: expectedAnthropicInterleavedThinkingToolMessages(true)
      },
      {
        baseurl: 'https://api.xiaomimimo.com/v1',
        expectedReasoningSplit: undefined,
        expectedThinking: { type: 'enabled' },
        expectedReasoningEffort: undefined,
        expectedMessages: expectedXiaomiMimoAnthropicInterleavedThinkingToolMessages()
      },
      {
        baseurl: 'https://open.bigmodel.cn/api/paas/v4',
        expectedReasoningSplit: undefined,
        expectedThinking: { type: 'enabled' },
        expectedReasoningEffort: 'medium',
        expectedMessages: expectedAnthropicInterleavedThinkingToolMessages(false)
      },
      {
        baseurl: 'https://api.minimax.io/v1',
        expectedReasoningSplit: true,
        expectedThinking: undefined,
        expectedReasoningEffort: undefined,
        expectedMessages: expectedAnthropicInterleavedThinkingToolMessages(true)
      }
    ];

    for (const testCase of cases) {
      const body = buildAnthropicInterleavedThinkingOpenAIChatBody({
        name: 'generic-openai-compatible',
        models: ['interleaved-thinking-model'],
        type: 'openai_chat_completions',
        baseurl: testCase.baseurl
      });

      expect(body.reasoning_split).toBe(testCase.expectedReasoningSplit);
      expect(body.interleaved_thinking).toBeUndefined();
      expect(body.interleavedThinking).toBeUndefined();
      expect(body.thinking).toEqual(testCase.expectedThinking);
      expect(body.reasoning_effort).toBe(testCase.expectedReasoningEffort);
      expect(body.output_config).toBeUndefined();
      expect(body.messages).toEqual(testCase.expectedMessages);
    }
  });

  it('enables reasoning_split automatically for Minimax OpenAI chat/completions targets', () => {
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
        name: 'Minimax',
        models: ['MiniMax-M2.7'],
        type: 'openai_chat_completions',
        baseurl: 'https://api.minimax.io/v1'
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

  it('keeps interleaved thinking as reasoning_details for Minimax tool-call history', () => {
    const body = buildInterleavedThinkingOpenAIChatBody({
      name: 'generic-openai-compatible',
      models: ['m2'],
      type: 'openai_chat_completions',
      baseurl: 'https://api.minimax.io/v1'
    });

    expect(body.reasoning_split).toBe(true);
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.messages).toEqual(expectedInterleavedThinkingToolMessages(true));
  });

  it('enables reasoning_split automatically for Minimax OpenAI chat domains', () => {
    for (const baseurl of [
      'https://api.minimax.io/v1',
      'https://api.minimax.chat/v1',
      'https://api.minimaxi.com/v1'
    ]) {
      const parsed = parseOpenAIResponsesRequest({
        model: 'm2',
        reasoning: {
          effort: 'high'
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
          name: 'generic-openai-compatible',
          models: ['m2'],
          type: 'openai_chat_completions',
          baseurl
        } as never
      });

      expect(built.ok).toBe(true);
      if (!built.ok) {
        return;
      }

      const body = built.value.body as Record<string, unknown>;
      expect(parsed.value.reasoning_split).toBeUndefined();
      expect(body.reasoning_split).toBe(true);
      expect(body.thinking).toBeUndefined();
      expect(body.output_config).toBeUndefined();
      expect(body.reasoning_effort).toBeUndefined();
    }
  });

  it('does not enable reasoning_split automatically for generic OpenAI chat/completions targets', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'glm-5',
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
        name: 'generic-openai-compatible',
        models: ['glm-5'],
        type: 'openai_chat_completions'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).reasoning_split).toBeUndefined();
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

  it('normalizes interleaved_thinking aliases when targeting OpenAI chat/completions', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'MiniMax-M2.7',
      interleaved_thinking: true,
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
    expect(parsed.value.reasoning_split).toBe(true);
    expect(body.reasoning_split).toBe(true);
    expect(body.interleaved_thinking).toBeUndefined();
    expect(body.interleavedThinking).toBeUndefined();
  });

  it('can disable reasoning_split for incompatible OpenAI chat/completions targets', () => {
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
        name: 'strict-chat',
        models: ['MiniMax-M2.7'],
        type: 'openai_chat_completions',
        openaiChatReasoningSplit: 'disabled'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect((built.value.body as Record<string, unknown>).reasoning_split).toBeUndefined();
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

function buildInterleavedThinkingOpenAIChatBody(
  targetProviderConfig: Record<string, unknown>
): Record<string, unknown> {
  const parsed = parseOpenAIResponsesRequest({
    model: 'interleaved-thinking-model',
    reasoning: {
      effort: 'medium'
    },
    input: [
      {
        type: 'reasoning',
        id: 'rs_interleaved',
        status: 'completed',
        content: [
          {
            type: 'reasoning_text',
            text: 'Need to call the weather tool before answering.'
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
    throw new Error(parsed.error);
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
    targetProviderConfig: targetProviderConfig as never
  });

  expect(built.ok).toBe(true);
  if (!built.ok) {
    throw new Error(built.error);
  }

  return built.value.body as Record<string, unknown>;
}

function buildAnthropicInterleavedThinkingOpenAIChatBody(
  targetProviderConfig: Record<string, unknown>
): Record<string, unknown> {
  const parsed = parseAnthropicMessagesRequest({
    model: 'interleaved-thinking-model',
    max_tokens: 128,
    thinking: {
      type: 'enabled'
    },
    output_config: {
      effort: 'medium'
    },
    messages: [
      {
        role: 'user',
        content: 'Use a tool'
      },
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: 'Need to call the weather tool before answering.',
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
      },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_weather',
            content: '{"temperature":22}'
          },
          {
            type: 'text',
            text: 'continue'
          }
        ]
      }
    ]
  });

  expect(parsed.ok).toBe(true);
  if (!parsed.ok) {
    throw new Error(parsed.error);
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
    targetProviderConfig: targetProviderConfig as never
  });

  expect(built.ok).toBe(true);
  if (!built.ok) {
    throw new Error(built.error);
  }

  return built.value.body as Record<string, unknown>;
}

function expectedInterleavedThinkingToolMessages(includeReasoning: boolean): Array<Record<string, unknown>> {
  const assistantMessage: Record<string, unknown> = {
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
  };

  if (includeReasoning) {
    assistantMessage.reasoning_content = 'Need to call the weather tool before answering.';
    assistantMessage.reasoning_details = [
      {
        type: 'reasoning.text',
        text: 'Need to call the weather tool before answering.',
        format: 'openai-responses-v1',
        index: 0
      }
    ];
  }

  return [
    assistantMessage,
    {
      role: 'tool',
      tool_call_id: 'call_weather',
      content: '{"temperature":22}'
    },
    {
      role: 'user',
      content: 'continue'
    }
  ];
}

function expectedXiaomiMimoInterleavedThinkingToolMessages(): Array<Record<string, unknown>> {
  const messages = expectedInterleavedThinkingToolMessages(true);
  delete messages[0].reasoning_details;
  return messages;
}

function expectedAnthropicInterleavedThinkingToolMessages(includeReasoning: boolean): Array<Record<string, unknown>> {
  const assistantMessage: Record<string, unknown> = {
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
  };

  if (includeReasoning) {
    assistantMessage.reasoning_content = 'Need to call the weather tool before answering.';
    assistantMessage.reasoning_details = [
      {
        type: 'reasoning.text',
        text: 'Need to call the weather tool before answering.',
        format: 'anthropic-claude-v1',
        index: 0,
        signature: 'sig_123'
      }
    ];
  }

  return [
    {
      role: 'user',
      content: 'Use a tool'
    },
    assistantMessage,
    {
      role: 'tool',
      tool_call_id: 'call_weather',
      content: '{"temperature":22}'
    },
    {
      role: 'user',
      content: 'continue'
    }
  ];
}

function expectedXiaomiMimoAnthropicInterleavedThinkingToolMessages(): Array<Record<string, unknown>> {
  const messages = expectedAnthropicInterleavedThinkingToolMessages(true);
  delete messages[1].reasoning_details;
  return messages;
}
