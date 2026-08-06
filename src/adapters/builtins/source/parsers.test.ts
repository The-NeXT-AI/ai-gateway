import { describe, expect, it } from 'vitest';
import {
  parseAnthropicMessagesRequest,
  parseGeminiGenerateContentRequest,
  parseOpenAIChatCompletionsRequest,
  parseOpenAIResponsesRequest
} from './parsers';
import {
  encodeReasoningTransportEnvelope,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../reasoning-envelope';
import {
  deriveProviderNativeGroups,
  prepareReasoningStateForTargetResult
} from '../../../gateway/reasoning-state';

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

    expect(result.value.input).toMatchObject([
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

    expect(result.value.input).toMatchObject([
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
    expect((result.value.input as any[])[0].content[0].native_item).toBeUndefined();
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

    expect(result.value.input).toMatchObject([
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
    const messages = result.value.input as any[];
    expect(messages[0].content[0].native_item).toBeUndefined();
    expect(messages[0].content[1].native_item).toBeUndefined();
    expect(messages[1].content[0].native_item).toBeUndefined();
  });

  it('reconstructs one complete native dependency group from a v3 reasoning carrier and exact tool items', () => {
    const origin = {
      provider: 'openai',
      endpoint: 'responses-endpoint',
      model: 'gpt-5.6-sol',
      credentialScope: 'responses-account'
    };
    const encryptedContent = encodeReasoningTransportEnvelope(
      OPENAI_RESPONSES_REASONING_FORMAT,
      'opaque-reasoning',
      'rs_native',
      'encrypted',
      origin,
      {
        nativeItem: {
          item_type: 'reasoning',
          native_id: 'rs_native',
          raw_payload: {
            type: 'reasoning',
            id: 'rs_native',
            status: 'completed',
            summary: [],
            encrypted_content: 'opaque-reasoning'
          },
          provider_schema_version: 'openai-responses-v1',
          item_origin: 'native',
          position: { turn: 0, step: 0, item: 0 },
          capture_state: 'complete'
        }
      }
    );
    const result = parseOpenAIResponsesRequest({
      input: [
        {
          type: 'reasoning',
          id: 'rs_native',
          status: 'completed',
          summary: [],
          encrypted_content: encryptedContent
        },
        {
          type: 'function_call',
          id: 'fc_weather',
          call_id: 'call_weather',
          name: 'get_weather',
          arguments: '{ "city": "Shanghai" }',
          status: 'completed'
        },
        {
          type: 'function_call_output',
          id: 'fco_weather',
          call_id: 'call_weather',
          output: '{"temperature":22}',
          status: 'completed'
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    const messages = result.value.input as any[];
    const reasoning = messages[0].content[0].native_item;
    const call = messages[0].content[1].native_item;
    const output = messages[1].content[0].native_item;
    expect(reasoning).toMatchObject({
      item_type: 'reasoning',
      native_id: 'rs_native',
      capture_state: 'complete'
    });
    expect(call).toMatchObject({
      item_type: 'function_call',
      native_id: 'fc_weather',
      call_id: 'call_weather',
      pair_id: 'call_weather',
      depends_on: ['rs_native'],
      capture_state: 'complete',
      raw_payload: {
        type: 'function_call',
        id: 'fc_weather',
        call_id: 'call_weather',
        arguments: '{ "city": "Shanghai" }'
      }
    });
    expect(output).toMatchObject({
      item_type: 'function_call_output',
      native_id: 'fco_weather',
      call_id: 'call_weather',
      pair_id: 'call_weather',
      depends_on: ['fc_weather'],
      capture_state: 'complete'
    });
    expect(call.group_id).toBe(reasoning.group_id);
    expect(output.group_id).toBe(reasoning.group_id);
    expect(deriveProviderNativeGroups(result.value)).toMatchObject([
      {
        state: 'active_waiting_model',
        active: true,
        items: expect.arrayContaining([reasoning, call, output])
      }
    ]);
  });

  it('rejects an active reconstructed Responses tool group when its native route changes', () => {
    const origin = {
      provider: 'openai',
      endpoint: 'responses-endpoint',
      model: 'gpt-5.6-sol',
      credentialScope: 'responses-account'
    };
    const encryptedContent = encodeReasoningTransportEnvelope(
      OPENAI_RESPONSES_REASONING_FORMAT,
      'opaque-reasoning',
      'rs_active',
      'encrypted',
      origin,
      {
        nativeItem: {
          item_type: 'reasoning',
          native_id: 'rs_active',
          raw_payload: {
            type: 'reasoning',
            id: 'rs_active',
            status: 'completed',
            summary: [],
            encrypted_content: 'opaque-reasoning'
          },
          provider_schema_version: 'openai-responses-v1',
          item_origin: 'native',
          position: { turn: 0, step: 0, item: 0 },
          capture_state: 'complete'
        }
      }
    );
    const parsed = parseOpenAIResponsesRequest({
      input: [
        {
          type: 'reasoning',
          id: 'rs_active',
          encrypted_content: encryptedContent
        },
        {
          type: 'function_call',
          id: 'fc_active',
          call_id: 'call_active',
          name: 'lookup',
          arguments: '{}'
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }
    expect(deriveProviderNativeGroups(parsed.value)).toMatchObject([
      { state: 'active_waiting_tool', active: true }
    ]);
    expect(
      prepareReasoningStateForTargetResult(
        parsed.value,
        OPENAI_RESPONSES_REASONING_FORMAT,
        {
          ...origin,
          model: 'gpt-5.6-terra'
        },
        { historyPolicy: 'native' }
      )
    ).toMatchObject({
      ok: false,
      error: expect.stringContaining('incompatible_active_native_group')
    });
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

    expect(result.value.input).toMatchObject([
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

    expect(result.value.input).toMatchObject([
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

    expect(result.value.input).toMatchObject([
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
    const messages = result.value.input as any[];
    expect(messages[0].content[0].native_item).toMatchObject({
      item_type: 'thinking',
      capture_state: 'partial',
      provider_mode: 'default',
      raw_payload: {
        type: 'thinking',
        thinking: 'anthropic thinking',
        signature: 'sig_123'
      }
    });
    expect(messages[0].content[1].native_item).toMatchObject({
      item_type: 'tool_use',
      group_id: 'toolu_weather',
      provider_mode: 'default',
      capture_state: 'partial'
    });
    expect(messages[1].content[0].native_item).toMatchObject({
      item_type: 'tool_result',
      group_id: 'toolu_weather',
      provider_mode: 'default',
      capture_state: 'partial'
    });
  });

  it('reanalyzes an origin-bearing v2 thinking carrier into a complete native tool group', () => {
    const origin = {
      provider: 'anthropic',
      endpoint: 'anthropic-endpoint',
      model: 'claude-sonnet-4-5',
      credentialScope: 'anthropic-account'
    };
    const signature = `ccr-reasoning-transport-v2:${Buffer.from(JSON.stringify({
      format: 'anthropic-claude-v1',
      data: 'v2-signature',
      kind: 'signature',
      origin
    }), 'utf8').toString('base64url')}`;
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      thinking: { type: 'adaptive' },
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: '', signature },
            {
              type: 'tool_use',
              id: 'toolu_v2',
              name: 'lookup',
              input: { key: 'value' }
            }
          ]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }
    const [reasoning, toolUse] = result.value.input[0]!.content as any[];
    expect(reasoning.native_item).toMatchObject({
      item_type: 'thinking',
      raw_payload: {
        type: 'thinking',
        thinking: '',
        signature: 'v2-signature'
      },
      source_origin: origin,
      group_id: 'toolu_v2',
      capture_state: 'complete',
      provider_mode: 'adaptive'
    });
    expect(toolUse.native_item).toMatchObject({
      item_type: 'tool_use',
      group_id: 'toolu_v2',
      capture_state: 'complete',
      provider_mode: 'adaptive'
    });
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

    expect(result.value.input[0]?.content).toContainEqual(expect.objectContaining({
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
    }));
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
    expect(result.value.input).toMatchObject([
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

    expect(result.value.input[0]?.content).toContainEqual(expect.objectContaining({
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
    }));
  });
});
