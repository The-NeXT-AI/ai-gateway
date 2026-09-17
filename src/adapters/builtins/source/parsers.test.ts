import { describe, expect, it } from 'vitest';
import {
  parseAnthropicMessagesRequest,
  parseGeminiGenerateContentRequest,
  parseGeminiInteractionsRequest,
  parseOpenAIChatCompletionsRequest,
  parseOpenAIResponsesRequest
} from './parsers';

describe('parseOpenAIResponsesRequest', () => {
  it('lifts developer and system messages into instructions', () => {
    const result = parseOpenAIResponsesRequest({
      model: 'gpt-5.5',
      instructions: 'You are Codex, a coding agent.',
      input: [
        {
          type: 'message',
          role: 'developer',
          content: [
            {
              type: 'input_text',
              text: 'Send progress updates before calling tools.'
            },
            {
              type: 'input_text',
              text: 'Finish with a concise user-facing answer.'
            }
          ]
        },
        {
          type: 'message',
          role: 'system',
          content: 'Keep policy instructions high priority.'
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: 'Analyze the local logs.'
            }
          ]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.instructions).toBe(
      [
        'You are Codex, a coding agent.',
        'Send progress updates before calling tools.',
        'Finish with a concise user-facing answer.',
        'Keep policy instructions high priority.'
      ].join('\n')
    );
    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'Analyze the local logs.'
          }
        ]
      }
    ]);
  });

  it('preserves image content and its order within user messages', () => {
    const result = parseOpenAIResponsesRequest({
      model: 'gpt-4.1-mini',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Compare these images:' },
            {
              type: 'input_image',
              image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
            },
            { type: 'input_text', text: 'and this one:' },
            {
              type: 'input_image',
              image_url: 'https://example.test/pixel.jpg'
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
        role: 'user',
        content: [
          { type: 'input_text', text: 'Compare these images:' },
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
          },
          { type: 'input_text', text: 'and this one:' },
          {
            type: 'input_image',
            image_url: 'https://example.test/pixel.jpg'
          }
        ]
      }
    ]);
  });

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
  it('converts anthropic image blocks into standard input_image content', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'What color is this image?'
            },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: 'iVBORw0KGgoAAAANSUhEUg=='
              }
            },
            {
              type: 'image',
              source: {
                type: 'url',
                url: 'https://example.test/pixel.png'
              }
            },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: null
              }
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
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'What color is this image?'
          },
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
          },
          {
            type: 'input_image',
            image_url: 'https://example.test/pixel.png'
          }
        ]
      }
    ]);
  });

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

  it('extracts image blocks from image-only tool_result content instead of JSON-stringifying them', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_read', name: 'read_image', input: { path: 'x.png' } }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_read',
              content: [
                {
                  type: 'image',
                  source: { type: 'base64', media_type: 'image/png', data: 'aGVsbG8=' }
                }
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
        tool_use_id: 'toolu_read',
        content: '',
        images: ['data:image/png;base64,aGVsbG8=']
      }
    ]);
  });

  it('keeps text and images from a mixed tool_result content array in their fields', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_read', name: 'read_image', input: { path: 'x.png' } }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_read',
              content: [
                { type: 'text', text: 'rendered at 2x' },
                {
                  type: 'image',
                  source: { type: 'base64', media_type: 'image/jpeg', data: 'anM=' }
                }
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
        tool_use_id: 'toolu_read',
        content: 'rendered at 2x',
        images: ['data:image/jpeg;base64,anM=']
      }
    ]);
  });

  // Documents keep the JSON fallback on purpose: the standard tool_result
  // shape only carries images today, so a document's base64 still serializes
  // into the text (same context cost as before this fix). Promoting documents
  // to a first-class standard field is a separate change.
  it('does not mistake anthropic document blocks in tool_result content for images', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_pdf', name: 'read_pdf', input: { path: 'x.pdf' } }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_pdf',
              content: [
                {
                  type: 'document',
                  source: { type: 'base64', media_type: 'application/pdf', data: 'JVBERi0=' }
                }
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
        tool_use_id: 'toolu_pdf',
        content: '[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]'
      }
    ]);
  });

  // An `image_url` business field is common in ordinary tool output and does
  // not make the object a protocol content part. Classifying it as one would
  // swallow the object whole and drop every other field on it.
  it('does not mistake untyped tool_result objects carrying an image_url field for images', () => {
    const result = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 128,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_search', name: 'search', input: { query: 'cats' } }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_search',
              content: [{ title: 'search result', image_url: 'https://example.test/thumbnail.png' }]
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
        content: '[{"title":"search result","image_url":"https://example.test/thumbnail.png"}]'
      }
    ]);
  });
});

describe('parseOpenAIChatCompletionsRequest', () => {
  it('preserves image_url content and its order within user messages', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'gpt-4.1-mini',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Compare these images:' },
            {
              type: 'image_url',
              image_url: {
                url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
              }
            },
            { type: 'text', text: 'and this one:' },
            {
              type: 'image_url',
              image_url: 'https://example.test/pixel.jpg'
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
        role: 'user',
        content: [
          { type: 'input_text', text: 'Compare these images:' },
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
          },
          { type: 'input_text', text: 'and this one:' },
          {
            type: 'input_image',
            image_url: 'https://example.test/pixel.jpg'
          }
        ]
      }
    ]);
  });

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
            type: 'input_text',
            text: 'visible answer'
          },
          {
            type: 'reasoning',
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

  it('extracts image parts from tool-message content arrays instead of JSON-stringifying them', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'GLM-5.3-Flash',
      messages: [
        {
          role: 'tool',
          tool_call_id: 'call_shot',
          content: [
            { type: 'text', text: 'screenshot captured' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,c2hvdA==' } }
          ]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_shot',
            content: 'screenshot captured',
            images: ['data:image/png;base64,c2hvdA==']
          }
        ]
      }
    ]);
  });

  it('extracts text parts from image-less tool-message content arrays', () => {
    const result = parseOpenAIChatCompletionsRequest({
      model: 'GLM-5.3-Flash',
      messages: [
        {
          role: 'tool',
          tool_call_id: 'call_text',
          content: [{ type: 'text', text: 'plain result' }]
        }
      ]
    });

    expect(result.ok).toBe(true);
    if (!result.ok || typeof result.value.input === 'string') {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_text',
            content: 'plain result'
          }
        ]
      }
    ]);
  });

});

describe('parseGeminiGenerateContentRequest', () => {
  it('preserves inline and URI image parts in their original order', () => {
    const result = parseGeminiGenerateContentRequest(
      {
        contents: [
          {
            role: 'user',
            parts: [
              { text: 'Compare these images:' },
              {
                inlineData: {
                  mimeType: 'image/png',
                  data: 'iVBORw0KGgoAAAANSUhEUg=='
                }
              },
              { text: 'and this one:' },
              {
                file_data: {
                  mime_type: 'image/jpeg',
                  file_uri: 'https://example.test/pixel.jpg'
                }
              }
            ]
          }
        ]
      },
      'gemini-2.5-flash'
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          { type: 'input_text', text: 'Compare these images:' },
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
          },
          { type: 'input_text', text: 'and this one:' },
          {
            type: 'input_image',
            image_url: 'https://example.test/pixel.jpg'
          }
        ]
      }
    ]);
  });

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
});

describe('parseGeminiInteractionsRequest', () => {
  it('preserves inline and URI image content in their original order', () => {
    const result = parseGeminiInteractionsRequest({
      model: 'gemini-3.7-flash',
      input: [
        { type: 'text', text: 'Compare these images:' },
        {
          type: 'image',
          mime_type: 'image/png',
          data: 'iVBORw0KGgoAAAANSUhEUg=='
        },
        { type: 'text', text: 'and this one:' },
        {
          type: 'image',
          uri: 'https://example.test/pixel.jpg',
          mime_type: 'image/jpeg'
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
        role: 'user',
        content: [
          { type: 'input_text', text: 'Compare these images:' },
          {
            type: 'input_image',
            image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='
          },
          { type: 'input_text', text: 'and this one:' },
          {
            type: 'input_image',
            image_url: 'https://example.test/pixel.jpg'
          }
        ]
      }
    ]);
  });
});
