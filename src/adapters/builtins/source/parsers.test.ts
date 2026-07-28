import { describe, expect, it } from 'vitest';
import {
  parseAnthropicMessagesRequest,
  parseGeminiGenerateContentRequest,
  parseOpenAIChatCompletionsRequest,
  parseOpenAIResponsesRequest
} from './parsers';

describe('parseOpenAIResponsesRequest', () => {
  it('parses function_call_output as tool_result content', () => {
    const result = parseOpenAIResponsesRequest({
      input: {
        type: 'function_call_output',
        call_id: 'call_123',
        output: {
          weather: 'sunny',
          temperature: 28
        }
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_123',
            content: '{"weather":"sunny","temperature":28}'
          }
        ]
      }
    ]);
  });

  it('parses function_call as tool_use content', () => {
    const result = parseOpenAIResponsesRequest({
      input: {
        type: 'function_call',
        call_id: 'call_456',
        name: 'get_weather',
        arguments: '{"city":"Shanghai"}'
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'call_456',
            name: 'get_weather',
            input: {
              city: 'Shanghai'
            }
          }
        ]
      }
    ]);
  });

  it('coalesces reasoning output items with following function calls', () => {
    const result = parseOpenAIResponsesRequest({
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
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            id: 'rs_123',
            source_format: 'openai-responses-v1',
            text: 'need a tool',
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
            content: '{"temperature":22}'
          }
        ]
      }
    ]);
  });

  it('parses reasoning output items without serializing them as user text', () => {
    const result = parseOpenAIResponsesRequest({
      input: {
        type: 'reasoning',
        id: 'rs_123',
        status: 'completed',
        summary: [
          {
            type: 'summary_text',
            text: 'short reasoning summary'
          }
        ],
        content: [
          {
            type: 'reasoning_text',
            text: 'private reasoning text'
          }
        ]
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            id: 'rs_123',
            source_format: 'openai-responses-v1',
            text: 'private reasoning text',
            summary: 'short reasoning summary',
            reasoning_details: [
              {
                type: 'reasoning.summary',
                summary: 'short reasoning summary',
                format: 'openai-responses-v1',
                index: 0
              },
              {
                type: 'reasoning.text',
                text: 'private reasoning text',
                format: 'openai-responses-v1',
                index: 1
              }
            ]
          }
        ]
      }
    ]);
  });

  it('falls back to serializing unknown object input instead of rejecting', () => {
    const result = parseOpenAIResponsesRequest({
      input: {
        foo: 'bar',
        nested: {
          value: 1
        }
      }
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: '{"foo":"bar","nested":{"value":1}}'
          }
        ]
      }
    ]);
  });

  it('preserves tool search input when execution is omitted but rejects explicit server execution', () => {
    const result = parseOpenAIResponsesRequest({
      input: [
        {
          type: 'tool_search_call',
          call_id: 'search_123',
          status: 'completed',
          arguments: { query: 'calendar' }
        },
        {
          type: 'tool_search_output',
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
        },
        {
          type: 'tool_search_call',
          execution: 'server',
          call_id: 'search_server',
          status: 'completed',
          arguments: { query: 'weather' }
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'tool_search_call',
            execution: 'client',
            call_id: 'search_123',
            status: 'completed',
            arguments: { query: 'calendar' }
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [
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
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: '{"type":"tool_search_call","execution":"server","call_id":"search_server","status":"completed","arguments":{"query":"weather"}}'
          }
        ]
      }
    ]);
  });
});

describe('parseAnthropicMessagesRequest', () => {
  it('parses thinking blocks into standard reasoning content', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'thinking',
              thinking: 'anthropic thinking',
              signature: 'sig_123'
            },
            {
              type: 'tool_use',
              id: 'toolu_weather',
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
              tool_use_id: 'toolu_weather',
              content: '{"temperature":22}'
            }
          ]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            source_format: 'anthropic-claude-v1',
            text: 'anthropic thinking',
            reasoning_details: [
              {
                type: 'reasoning.text',
                text: 'anthropic thinking',
                format: 'anthropic-claude-v1',
                index: 0,
                signature: 'sig_123'
              }
            ]
          },
          {
            type: 'tool_use',
            id: 'toolu_weather',
            name: 'get_weather',
            thought_signature: 'gemini-function-signature',
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
            tool_use_id: 'toolu_weather',
            content: '{"temperature":22}'
          }
        ]
      }
    ]);
  });

  it('keeps top-level thinking controls for protocol conversion', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
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
          content: 'hello'
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.thinking).toEqual({
      type: 'enabled'
    });
    expect(result.value.output_config).toEqual({
      effort: 'medium'
    });
  });

  it('preserves deferred tool references separately from residual result text', () => {
    const result = parseAnthropicMessagesRequest({
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
                { type: 'text', text: 'matched one tool' },
                { type: 'tool_reference', tool_name: 'calendar_create' },
                { type: 'text', text: 'ready to call' }
              ]
            }
          ]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }

    expect(result.value.input[1]?.content).toEqual([
      {
        type: 'tool_result',
        tool_use_id: 'toolu_search',
        content: 'matched one tool\nready to call',
        tool_references: ['calendar_create']
      }
    ]);
  });
});

describe('parseOpenAIChatCompletionsRequest', () => {
  it('keeps reasoning_split and de-duplicates equivalent chat reasoning fields', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'MiniMax-M2.7',
      reasoning_split: true,
      messages: [
        {
          role: 'assistant',
          reasoning_content: 'interleaved thinking',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'interleaved thinking',
              id: 'reasoning-text-1',
              format: 'anthropic-claude-v1',
              index: 0
            }
          ],
          content: 'visible answer'
        },
        {
          role: 'user',
          content: 'continue'
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.reasoning_split).toBe(true);
    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            id: 'reasoning-text-1',
            source_format: 'anthropic-claude-v1',
            text: 'interleaved thinking',
            reasoning_details: [
              {
                type: 'reasoning.text',
                text: 'interleaved thinking',
                id: 'reasoning-text-1',
                format: 'anthropic-claude-v1',
                index: 0
              }
            ]
          },
          {
            type: 'input_text',
            text: 'visible answer'
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
    ]);
  });

  it('restores Responses reasoning IDs from chat reasoning_details', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'gpt-5.6-sol',
      messages: [
        {
          role: 'assistant',
          content: 'first answer',
          reasoning_details: [
            {
              type: 'reasoning.encrypted',
              data: 'encrypted-chat-reasoning',
              id: 'rs_chat_reasoning_1',
              format: 'openai-responses-v1',
              index: 0
            }
          ]
        },
        {
          role: 'user',
          content: 'continue'
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }

    expect(result.value.input[0]?.content).toContainEqual({
      type: 'reasoning',
      id: 'rs_chat_reasoning_1',
      source_format: 'openai-responses-v1',
      encrypted_content: 'encrypted-chat-reasoning',
      reasoning_details: [
        {
          type: 'reasoning.encrypted',
          data: 'encrypted-chat-reasoning',
          id: 'rs_chat_reasoning_1',
          format: 'openai-responses-v1',
          index: 0
        }
      ]
    });
  });

  it('keeps OpenAI-compatible reasoning_effort as standard reasoning effort', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'glm-5.2',
      reasoning_effort: 'high',
      messages: [
        {
          role: 'user',
          content: 'hello'
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.reasoning).toEqual({
      effort: 'high'
    });
  });

  it('parses tools, assistant tool_calls, and tool role messages into standard input', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'gpt-5.4',
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
      tool_choice: 'required',
      messages: [
        { role: 'system', content: 'You are a tool-calling assistant.' },
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

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.instructions).toBe('You are a tool-calling assistant.');
    expect(result.value.tools).toEqual([
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
    ]);
    expect(result.value.tool_choice).toBe('required');
    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What is the weather in Shanghai?'
          }
        ]
      },
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
            content: '{"temperature":22}'
          }
        ]
      }
    ]);
  });
});

describe('parseGeminiGenerateContentRequest', () => {
  it('keeps thinking config and maps thought parts into standard reasoning content', () => {
    const result = parseGeminiGenerateContentRequest(
      {
        generationConfig: {
          thinkingConfig: {
            thinkingBudget: 1024
          }
        },
        contents: [
          {
            role: 'model',
            parts: [
              {
                text: 'gemini interleaved thinking',
                thought: true,
                thoughtSignature: 'gemini-thought-signature'
              },
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
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call_lookup',
                  name: 'lookup_value',
                  response: {
                    content: '{"value":"live-ok"}'
                  }
                }
              },
              {
                text: 'continue'
              }
            ]
          }
        ]
      },
      'gemini-test'
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.thinking).toEqual({
      type: 'enabled'
    });
    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            source_format: 'google-generate-content-v1',
            text: 'gemini interleaved thinking',
            encrypted_content: 'gemini-thought-signature',
            reasoning_details: [
              {
                type: 'reasoning.text',
                text: 'gemini interleaved thinking',
                format: 'google-generate-content-v1',
                index: 0
              },
              {
                type: 'reasoning.encrypted',
                data: 'gemini-thought-signature',
                format: 'google-generate-content-v1',
                index: 0
              }
            ]
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
      },
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_lookup',
            content: '{"value":"live-ok"}'
          },
          {
            type: 'input_text',
            text: 'continue'
          }
        ]
      }
    ]);
  });

  it('restores Responses reasoning IDs from Gemini thought signatures', () => {
    const envelope =
      'ccr-openai-responses-reasoning-v1:' +
      Buffer.from(
        JSON.stringify({
          id: 'rs_gemini_reasoning_1',
          encrypted_content: 'encrypted-gemini-reasoning'
        }),
        'utf8'
      ).toString('base64url');
    const result = parseGeminiGenerateContentRequest(
      {
        contents: [
          {
            role: 'model',
            parts: [
              {
                thought: true,
                thoughtSignature: envelope
              },
              {
                text: 'first answer'
              }
            ]
          },
          {
            role: 'user',
            parts: [{ text: 'continue' }]
          }
        ]
      },
      'gpt-5.6-sol'
    );

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }

    expect(result.value.input[0]?.content).toContainEqual({
      type: 'reasoning',
      id: 'rs_gemini_reasoning_1',
      source_format: 'openai-responses-v1',
      encrypted_content: 'encrypted-gemini-reasoning',
      reasoning_details: [
        {
          type: 'reasoning.encrypted',
          data: 'encrypted-gemini-reasoning',
          id: 'rs_gemini_reasoning_1',
          format: 'openai-responses-v1',
          index: 0
        }
      ]
    });
  });
});
