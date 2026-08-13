import { describe, expect, it } from 'vitest';
import { formatAnthropicMessagesResponse } from '../source/formatters';
import { parseAnthropicMessagesRequest, parseOpenAIResponsesRequest } from '../source/parsers';
import {
  buildOpenAIResponsesBodyFromStandardRequest,
  openAIResponsesTargetAdapter
} from './openai-responses';

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

  it('translates Anthropic output_config effort for OpenAI Responses targets', () => {
    const body = buildAnthropicOpenAITargetBody(
      {
        model: 'gpt-reasoning',
        max_tokens: 128,
        thinking: {
          type: 'enabled',
          budget_tokens: 1024
        },
        output_config: {
          effort: 'xhigh'
        },
        messages: [{ role: 'user', content: 'Think carefully' }]
      },
      {
        name: 'openai-main',
        type: 'openai_responses',
        models: ['gpt-reasoning'],
        modelMetadata: {
          'GPT-REASONING': {
            supportedReasoningLevels: [
              { effort: 'low' },
              { effort: 'high' },
              { effort: 'xhigh' }
            ]
          }
        }
      }
    );

    expect(body.reasoning).toEqual({ effort: 'xhigh' });
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
  });

  it('omits Anthropic stop sequences for OpenAI Responses targets', () => {
    const body = buildAnthropicOpenAITargetBody(
      {
        model: 'gpt-5.6-sol',
        max_tokens: 2112,
        stop_sequences: ['</block>'],
        messages: [{ role: 'user', content: 'Classify this tool call.' }]
      },
      {
        name: 'openai-main',
        type: 'openai_responses',
        models: ['gpt-5.6-sol']
      }
    );

    expect(body).not.toHaveProperty('stop');
  });

  it('preserves Anthropic stop sequences for OpenAI Chat Completions targets', () => {
    const body = buildAnthropicOpenAITargetBody(
      {
        model: 'chat-model',
        max_tokens: 128,
        stop_sequences: ['</block>'],
        messages: [{ role: 'user', content: 'Classify this tool call.' }]
      },
      {
        name: 'openai-chat',
        type: 'openai_chat_completions',
        models: ['chat-model']
      }
    );

    expect(body.stop).toEqual(['</block>']);
  });

  it('encodes assistant history as output_text for OpenAI Responses targets', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'gpt-5.6-sol',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: '你好' }]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [{ type: 'input_text', text: '你好！很高兴见到你。' }]
        },
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'hello' }]
        }
      ]
    });

    expect(body.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: '你好' }]
      },
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: '你好！很高兴见到你。' }]
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'hello' }]
      }
    ]);
  });

  it('drops legacy encrypted assistant reasoning that has no original Responses item ID', () => {
    const body = buildAnthropicOpenAITargetBody(
      {
        model: 'gpt-5.6-sol',
        max_tokens: 8192,
        messages: [
          {
            role: 'user',
            content: 'first turn'
          },
          {
            role: 'assistant',
            content: [
              {
                type: 'redacted_thinking',
                data: 'encrypted-reasoning'
              },
              {
                type: 'text',
                text: 'done'
              }
            ]
          },
          {
            role: 'user',
            content: 'second turn'
          }
        ]
      },
      {
        name: 'codex-api',
        type: 'openai_responses',
        models: ['gpt-5.6-sol'],
        openaiResponsesReasoningHistoryPolicy: 'encrypted'
      }
    );

    const input = body.input as Array<Record<string, unknown>>;
    const reasoning = input.find((item) => item.type === 'reasoning');
    expect(reasoning).toBeUndefined();
    expect(input).toContainEqual({
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: 'done' }]
    });
  });

  it('preserves encrypted Responses reasoning IDs across Anthropic history round trips', () => {
    const parsedResponse = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_first_turn',
      object: 'response',
      status: 'completed',
      model: 'gpt-5.6-sol',
      output_text: 'done',
      output: [
        {
          type: 'reasoning',
          id: 'rs_original_reasoning_item',
          status: 'completed',
          summary: [],
          encrypted_content: 'encrypted-reasoning'
        },
        {
          type: 'message',
          id: 'msg_first_turn',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'done', annotations: [] }]
        }
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14
      }
    });

    expect(parsedResponse.ok).toBe(true);
    if (!parsedResponse.ok) {
      return;
    }

    const anthropicResponse = formatAnthropicMessagesResponse(parsedResponse.value);
    const responseContent = anthropicResponse.content as Array<Record<string, unknown>>;
    expect(responseContent[0]).toMatchObject({
      type: 'redacted_thinking'
    });
    expect(responseContent[0]?.data).not.toBe('encrypted-reasoning');

    const body = buildAnthropicOpenAITargetBody(
      {
        model: 'gpt-5.6-sol',
        max_tokens: 8192,
        messages: [
          {
            role: 'user',
            content: 'first turn'
          },
          {
            role: 'assistant',
            content: responseContent
          },
          {
            role: 'user',
            content: 'second turn'
          }
        ]
      },
      {
        name: 'codex-api',
        type: 'openai_responses',
        models: ['gpt-5.6-sol'],
        openaiResponsesReasoningHistoryPolicy: 'encrypted'
      }
    );

    const input = body.input as Array<Record<string, unknown>>;
    expect(input[1]).toEqual({
      type: 'reasoning',
      id: 'rs_original_reasoning_item',
      summary: [],
      encrypted_content: 'encrypted-reasoning'
    });
    expect(input[1]).not.toHaveProperty('status');
    expect(input[2]).toEqual({
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: 'done' }]
    });
  });

  it('replays every OpenAI Responses reasoning item from one assistant message', () => {
    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: {
        headers: {},
        url: '/v1/responses'
      } as never,
      standardRequest: {
        model: 'gpt-5.6-sol',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'reasoning',
                id: 'rs_one',
                source_format: 'openai-responses-v1',
                encrypted_content: 'enc_one'
              },
              {
                type: 'reasoning',
                id: 'rs_two',
                source_format: 'openai-responses-v1',
                encrypted_content: 'enc_two'
              }
            ]
          }
        ]
      },
      config: {
        openaiApiKey: 'test',
        openaiBaseUrl: 'https://example.test/v1'
      } as never,
      targetProviderConfig: {
        name: 'test-responses',
        type: 'openai_responses',
        models: ['gpt-5.6-sol'],
        openaiResponsesReasoningHistoryPolicy: 'encrypted'
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const input = (built.value.body as { input: Array<Record<string, unknown>> }).input;
    expect(input.filter((item) => item.type === 'reasoning')).toEqual([
      {
        type: 'reasoning',
        id: 'rs_one',
        summary: [],
        encrypted_content: 'enc_one'
      },
      {
        type: 'reasoning',
        id: 'rs_two',
        summary: [],
        encrypted_content: 'enc_two'
      }
    ]);
  });

  it('replays only complete encrypted state and never sends readable content to encrypted targets', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest(
      {
        model: 'gpt-5.6-sol',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'reasoning',
                id: 'rs_valid',
                text: 'must not be sent',
                summary: 'safe attached summary',
                source_format: 'openai-responses-v1',
                encrypted_content: 'enc_valid'
              },
              {
                type: 'reasoning',
                text: 'missing id',
                source_format: 'openai-responses-v1',
                encrypted_content: 'enc_missing_id'
              },
              {
                type: 'reasoning',
                id: 'rs_foreign',
                text: 'foreign text',
                source_format: 'anthropic-claude-v1',
                encrypted_content: 'foreign_ciphertext'
              }
            ]
          }
        ]
      },
      {
        name: 'official-openai',
        type: 'openai_responses',
        models: ['gpt-5.6-sol'],
        openaiResponsesReasoningHistoryPolicy: 'encrypted'
      } as never
    );

    expect(body.input).toEqual([
      {
        type: 'reasoning',
        id: 'rs_valid',
        summary: [{ type: 'summary_text', text: 'safe attached summary' }],
        encrypted_content: 'enc_valid'
      }
    ]);
    expect(JSON.stringify(body.input)).not.toContain('must not be sent');
    expect(JSON.stringify(body.input)).not.toContain('enc_missing_id');
    expect(JSON.stringify(body.input)).not.toContain('foreign_ciphertext');
  });

  it('infers encrypted policy for official endpoints and plaintext policy otherwise', () => {
    const input = [
      {
        type: 'message' as const,
        role: 'assistant' as const,
        content: [
          {
            type: 'reasoning' as const,
            id: 'rs_previous',
            text: 'readable reasoning',
            source_format: 'openai-responses-v1',
            encrypted_content: 'encrypted reasoning'
          },
          { type: 'input_text' as const, text: 'answer' }
        ]
      }
    ];
    const provider = {
      name: 'responses-auto',
      type: 'openai_responses',
      models: ['reasoning-model'],
      openaiResponsesReasoningHistoryPolicy: 'auto'
    } as never;

    const official = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://api.openai.com/v1'
    );
    const codex = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://chatgpt.com/backend-api/codex'
    );
    const deepSeek = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://api.deepseek.com/v1'
    );
    const mimo = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://api.xiaomimimo.com/v1'
    );
    const mimoTokenPlan = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://token-plan-cn.xiaomimimo.com/v1'
    );
    const unknown = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider,
      'https://responses.example.test/v1'
    );
    const missingBaseUrl = buildOpenAIResponsesBodyFromStandardRequest(
      { model: 'reasoning-model', input },
      provider
    );

    for (const body of [official, codex]) {
      expect((body.input as Array<Record<string, unknown>>)[0]).toEqual({
        type: 'reasoning',
        id: 'rs_previous',
        summary: [],
        encrypted_content: 'encrypted reasoning'
      });
    }
    for (const body of [deepSeek, mimo, mimoTokenPlan, unknown, missingBaseUrl]) {
      expect((body.input as Array<Record<string, unknown>>)[0]).toEqual({
        type: 'reasoning',
        id: 'rs_previous',
        content: [{ type: 'reasoning_text', text: 'readable reasoning' }]
      });
    }
  });

  it('lets a model override the provider policy and optionally converts summaries to plaintext content', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest(
      {
        model: 'deepseek-reasoner',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'reasoning',
                summary: 'reasoning summary',
                source_format: 'gemini-generate-content-v1',
                encrypted_content: 'foreign opaque state'
              }
            ]
          }
        ]
      },
      {
        name: 'mixed-responses',
        type: 'openai_responses',
        models: ['deepseek-reasoner'],
        openaiResponsesReasoningHistoryPolicy: 'encrypted',
        openaiResponsesReasoningSummaryPolicy: 'drop',
        modelMetadata: {
          'DEEPSEEK-REASONER': {
            openaiResponsesReasoningHistoryPolicy: 'plaintext',
            openaiResponsesReasoningSummaryPolicy: 'as_content'
          }
        }
      } as never
    );

    const reasoning = (body.input as Array<Record<string, unknown>>)[0];
    expect(reasoning).toMatchObject({
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: 'reasoning summary' }]
    });
    expect(reasoning).toHaveProperty('id');
    expect(reasoning).not.toHaveProperty('summary');
    expect(reasoning).not.toHaveProperty('encrypted_content');
    expect(JSON.stringify(reasoning)).not.toContain('foreign opaque state');
  });

  it('drops summary-only plaintext history unless summary conversion is enabled', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest(
      {
        model: 'compatible-model',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'reasoning', summary: 'summary only' }]
          }
        ]
      },
      {
        name: 'compatible-responses',
        type: 'openai_responses',
        models: ['compatible-model'],
        openaiResponsesReasoningHistoryPolicy: 'plaintext',
        openaiResponsesReasoningSummaryPolicy: 'drop'
      } as never
    );

    expect(body.input).toEqual([]);
  });

  it('builds identical second-turn reasoning history for streaming and non-streaming requests', () => {
    const request = {
      model: 'gpt-5.6-sol',
      input: [
        {
          type: 'message' as const,
          role: 'assistant' as const,
          content: [
            {
              type: 'reasoning' as const,
              id: 'rs_stream_parity',
              source_format: 'openai-responses-v1',
              encrypted_content: 'enc_stream_parity'
            },
            { type: 'tool_use' as const, id: 'call_1', name: 'lookup', input: {} }
          ]
        },
        {
          type: 'message' as const,
          role: 'user' as const,
          content: [{ type: 'tool_result' as const, tool_use_id: 'call_1', content: 'ok' }]
        }
      ]
    };
    const provider = {
      name: 'official-openai',
      type: 'openai_responses',
      models: ['gpt-5.6-sol'],
      openaiResponsesReasoningHistoryPolicy: 'encrypted'
    } as never;
    const nonStreaming = buildOpenAIResponsesBodyFromStandardRequest(request, provider);
    const streaming = buildOpenAIResponsesBodyFromStandardRequest(
      { ...request, stream: true },
      provider
    );

    expect(streaming.input).toEqual(nonStreaming.input);
    expect(streaming.stream).toBe(true);
    expect(nonStreaming.stream).toBeUndefined();
  });

  it('keeps supported reasoning efforts and selects the closest supported fallback', () => {
    const allLevelsProvider = {
      modelMetadata: {
        'gpt-all': {
          supportedReasoningLevels: [
            { effort: 'low' },
            { effort: 'medium' },
            { effort: 'high' },
            { effort: 'xhigh' },
            { effort: 'max' }
          ]
        }
      }
    } as never;

    for (const effort of ['low', 'medium', 'high', 'xhigh', 'max']) {
      const body = buildOpenAIResponsesBodyFromStandardRequest(
        {
          model: 'gpt-all',
          input: 'hello',
          output_config: { effort }
        },
        allLevelsProvider
      );
      expect(body.reasoning).toEqual({ effort });
    }

    const sparseProvider = {
      modelMetadata: {
        'GPT-SPARSE': {
          supportedReasoningLevels: [{ effort: 'low' }, { effort: 'high' }]
        }
      }
    } as never;

    expect(
      buildOpenAIResponsesBodyFromStandardRequest(
        {
          model: 'gpt-sparse',
          input: 'hello',
          output_config: { effort: 'xhigh' }
        },
        sparseProvider
      ).reasoning
    ).toEqual({ effort: 'high' });
    expect(
      buildOpenAIResponsesBodyFromStandardRequest(
        {
          model: 'gpt-sparse',
          input: 'hello',
          output_config: { effort: 'medium' }
        },
        sparseProvider
      ).reasoning
    ).toEqual({ effort: 'low' });
    expect(
      buildOpenAIResponsesBodyFromStandardRequest(
        {
          model: 'gpt-sparse',
          input: 'hello',
          output_config: { effort: 'minimal' }
        },
        sparseProvider
      ).reasoning
    ).toEqual({ effort: 'low' });
  });

  it('uses the built-in OpenAI model reasoning capability table', () => {
    const cases = [
      ['gpt-5.6', 'max', 'max'],
      ['gpt-5.6-sol', 'max', 'max'],
      ['gpt-5.6-terra', 'max', 'max'],
      ['gpt-5.6-luna', 'max', 'max'],
      ['gpt-5.5', 'max', 'xhigh'],
      ['gpt-5.5-pro', 'low', 'medium'],
      ['gpt-5.4', 'max', 'xhigh'],
      ['gpt-5.4-mini', 'max', 'xhigh'],
      ['gpt-5.4-nano', 'max', 'xhigh'],
      ['gpt-5.4-pro', 'low', 'medium'],
      ['gpt-5.3-codex', 'max', 'xhigh'],
      ['gpt-5.3-codex-spark', 'none', 'low'],
      ['gpt-5.2', 'max', 'xhigh'],
      ['gpt-5.2-codex', 'none', 'low'],
      ['gpt-5.1', 'max', 'high'],
      ['gpt-5.1-codex-max', 'max', 'xhigh'],
      ['gpt-5.1-codex', 'max', 'high'],
      ['gpt-5.1-codex-mini', 'none', 'low'],
      ['gpt-5', 'none', 'minimal'],
      ['gpt-5-mini', 'none', 'minimal'],
      ['gpt-5-nano', 'max', 'high'],
      ['gpt-5-pro', 'low', 'high'],
      ['gpt-5-codex', 'max', 'high'],
      ['o3', 'minimal', 'low'],
      ['o3-mini', 'max', 'high'],
      ['o4-mini', 'none', 'low']
    ] as const;

    for (const [model, requestedEffort, expectedEffort] of cases) {
      const body = buildOpenAIResponsesBodyFromStandardRequest({
        model,
        input: 'hello',
        output_config: { effort: requestedEffort }
      });
      expect(body.reasoning, model).toEqual({ effort: expectedEffort });
    }
  });

  it('matches dated model snapshots without matching longer model-name prefixes', () => {
    const snapshotBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'gpt-5.4-pro-2026-03-05',
      input: 'hello',
      output_config: { effort: 'low' }
    });
    expect(snapshotBody.reasoning).toEqual({ effort: 'medium' });

    const longerUnknownModelBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'gpt-5.4-proxy',
      input: 'hello',
      output_config: { effort: 'low' }
    });
    expect(longerUnknownModelBody.reasoning).toEqual({ effort: 'low' });
  });

  it('lets provider metadata override built-in OpenAI model capabilities', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest(
      {
        model: 'gpt-5.4',
        input: 'hello',
        output_config: { effort: 'xhigh' }
      },
      {
        modelMetadata: {
          'GPT-5.4': {
            supportedReasoningLevels: [{ effort: 'low' }, { effort: 'high' }]
          }
        }
      } as never
    );

    expect(body.reasoning).toEqual({ effort: 'high' });
  });

  it('passes valid efforts through for unknown models and ignores invalid efforts', () => {
    const maxBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'future-reasoning-model',
      input: 'hello',
      output_config: { effort: 'max' }
    });
    expect(maxBody.reasoning).toEqual({ effort: 'max' });
    expect(maxBody.output_config).toBeUndefined();

    const unknownBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'future-reasoning-model',
      input: 'hello',
      output_config: { effort: 'extreme' }
    });
    expect(unknownBody.reasoning).toBeUndefined();
    expect(unknownBody.output_config).toBeUndefined();
  });

  it('respects explicit Responses reasoning and can fill a missing effort', () => {
    const explicitBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'gpt-5-pro',
      input: 'hello',
      reasoning: {
        effort: 'low',
        summary: 'auto'
      },
      output_config: { effort: 'high' }
    });
    expect(explicitBody.reasoning).toEqual({
      effort: 'low',
      summary: 'auto'
    });

    const supplementedBody = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'gpt-5.4-pro',
      input: 'hello',
      reasoning: {
        summary: 'auto'
      },
      output_config: { effort: 'low' }
    });
    expect(supplementedBody.reasoning).toEqual({
      effort: 'medium',
      summary: 'auto'
    });
  });

  it('does not add reasoning when model metadata explicitly supports no effort levels', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest(
      {
        model: 'gpt-5.4',
        input: 'hello',
        output_config: { effort: 'high' }
      },
      {
        modelMetadata: {
          'gpt-5.4': {
            supportedReasoningLevels: []
          }
        }
      } as never
    );

    expect(body.reasoning).toBeUndefined();
    expect(body.output_config).toBeUndefined();
  });

  it('maps source verbosity while preserving explicit Responses reasoning', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'source-model',
      max_tokens: 64,
      output_config: {
        verbosity: 'low'
      },
      reasoning: {
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

    expect(built.value.body).not.toHaveProperty('output_config');
    expect((built.value.body as Record<string, unknown>).reasoning).toEqual({
      effort: 'high'
    });
    expect((built.value.body as Record<string, unknown>).text).toEqual({
      verbosity: 'low'
    });
  });

  it('preserves native Responses text options over source aliases', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'target-model',
      text: {
        format: {
          type: 'text'
        },
        verbosity: 'high'
      },
      output_config: {
        verbosity: 'low'
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
      } as never
    });

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    expect(built.value.body).not.toHaveProperty('output_config');
    expect((built.value.body as Record<string, unknown>).text).toEqual({
      format: {
        type: 'text'
      },
      verbosity: 'high'
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

  it('translates native client tool search history when targeting OpenAI chat', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      tools: [
        {
          type: 'tool_search',
          execution: 'client',
          parameters: {
            type: 'object',
            properties: { query: { type: 'string' } }
          }
        },
        {
          type: 'function',
          name: 'calendar_create',
          defer_loading: true,
          parameters: { type: 'object', properties: {} }
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

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
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
            id: 'search_123',
            type: 'function',
            function: {
              name: 'ToolSearch',
              arguments: '{"query":"calendar"}'
            }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 'search_123',
        content:
          '{"type":"tool_search_output","execution":"client","status":"completed","tools":[{"type":"function","name":"calendar_create","defer_loading":true,"parameters":{"type":"object","properties":{}}}]}'
      }
    ]);
  });

  it('preserves Anthropic tool references when targeting OpenAI chat', () => {
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
              content: [{ type: 'tool_reference', tool_name: 'calendar_create' }]
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
      request: { headers: {} } as never,
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

    expect((built.value.body as Record<string, unknown>).messages).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'toolu_search',
            type: 'function',
            function: {
              name: 'ToolSearch',
              arguments: '{"query":"calendar"}'
            }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 'toolu_search',
        content: '[{"type":"tool_reference","tool_name":"calendar_create"}]'
      }
    ]);
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

  it('translates Claude deferred tool discovery into client-executed tool search', () => {
    const parsed = parseAnthropicMessagesRequest({
      model: 'claude-sonnet-4-5',
      max_tokens: 256,
      tools: [
        {
          name: 'ToolSearch',
          description: 'Find tools needed for the current task.',
          input_schema: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
            additionalProperties: false
          }
        },
        {
          name: 'calendar_create',
          description: 'Create a calendar event.',
          defer_loading: true,
          input_schema: {
            type: 'object',
            properties: { title: { type: 'string' } },
            required: ['title'],
            additionalProperties: false
          }
        },
        {
          name: 'get_time',
          description: 'Get the current time.',
          input_schema: {
            type: 'object',
            properties: {}
          }
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
                { type: 'text', text: 'tool loaded' }
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

    const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
      request: { headers: {} } as never,
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
        type: 'tool_search',
        execution: 'client',
        description: 'Find tools needed for the current task.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
          additionalProperties: false
        }
      },
      {
        type: 'function',
        name: 'get_time',
        description: 'Get the current time.',
        parameters: {
          type: 'object',
          properties: {}
        }
      }
    ]);
    expect(body.input).toEqual([
      {
        type: 'tool_search_call',
        execution: 'client',
        call_id: 'toolu_search',
        status: 'completed',
        arguments: { query: 'calendar create' }
      },
      {
        type: 'tool_search_output',
        execution: 'client',
        call_id: 'toolu_search',
        status: 'completed',
        tools: [
          {
            type: 'function',
            name: 'calendar_create',
            description: 'Create a calendar event.',
            defer_loading: true,
            parameters: {
              type: 'object',
              properties: { title: { type: 'string' } },
              required: ['title'],
              additionalProperties: false
            }
          }
        ]
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'tool loaded' }]
      }
    ]);
  });

  it('falls back atomically when deferred discovery call IDs are ambiguous', () => {
    const standardRequest = {
      model: 'claude-sonnet-4-5',
      tools: [
        {
          name: 'ToolSearch',
          input_schema: { type: 'object', properties: {} }
        },
        {
          name: 'calendar_create',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      input: [
        {
          type: 'message' as const,
          role: 'assistant' as const,
          content: [
            {
              type: 'tool_use' as const,
              id: 'toolu_duplicate',
              name: 'ToolSearch',
              input: { query: 'calendar' }
            },
            {
              type: 'tool_use' as const,
              id: 'toolu_duplicate',
              name: 'ToolSearch',
              input: { query: 'calendar create' }
            }
          ]
        },
        {
          type: 'message' as const,
          role: 'user' as const,
          content: [
            {
              type: 'tool_result' as const,
              tool_use_id: 'toolu_duplicate',
              content: '',
              tool_references: ['calendar_create']
            }
          ]
        }
      ]
    };

    const body = buildOpenAIResponsesBodyFromStandardRequest(standardRequest);

    expect(body.tools).toEqual([
      {
        type: 'function',
        name: 'ToolSearch',
        parameters: { type: 'object', properties: {} }
      },
      {
        type: 'function',
        name: 'calendar_create',
        parameters: { type: 'object', properties: {} }
      }
    ]);
    expect(JSON.stringify(body.input)).not.toContain('tool_search_call');
    expect(JSON.stringify(body.input)).not.toContain('tool_search_output');
    expect(JSON.stringify(body.input)).toContain(
      '[{\\"type\\":\\"tool_reference\\",\\"tool_name\\":\\"calendar_create\\"}]'
    );
  });

  it('falls back atomically when deferred tool definitions are ambiguous', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        {
          name: 'ToolSearch',
          input_schema: { type: 'object', properties: {} }
        },
        {
          name: 'calendar_create',
          description: 'First definition.',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        },
        {
          name: 'calendar_create',
          description: 'Conflicting definition.',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]
    });

    expect(Array.isArray(body.tools)).toBe(true);
    expect(body.tools).toHaveLength(3);
    expect(JSON.stringify(body.tools)).not.toContain('tool_search');
  });

  it('keeps native search choices coherent with the advertised tools', () => {
    const tools = [
      {
        name: 'ToolSearch',
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'calendar_create',
        defer_loading: true,
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'get_time',
        input_schema: { type: 'object', properties: {} }
      }
    ];
    const input = [
      {
        type: 'message' as const,
        role: 'user' as const,
        content: [{ type: 'input_text' as const, text: 'hello' }]
      }
    ];

    const forcedSearch = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools,
      tool_choice: { type: 'tool', name: 'ToolSearch' },
      input
    });
    expect(forcedSearch.tool_choice).toEqual({ type: 'tool_search' });
    expect(forcedSearch.tools).toEqual([
      { type: 'tool_search', execution: 'client', parameters: { type: 'object', properties: {} } },
      { type: 'function', name: 'get_time', parameters: { type: 'object', properties: {} } }
    ]);

    const forcedDeferred = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools,
      tool_choice: { type: 'tool', name: 'calendar_create' },
      input
    });
    expect(forcedDeferred.tool_choice).toEqual({ type: 'function', name: 'calendar_create' });
    expect(forcedDeferred.tools).toHaveLength(3);
    expect(JSON.stringify(forcedDeferred.tools)).not.toContain('"type":"tool_search"');

    const forcedEager = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools,
      tool_choice: { type: 'tool', name: 'get_time' },
      input
    });
    expect(forcedEager.tool_choice).toEqual({ type: 'function', name: 'get_time' });
    expect(JSON.stringify(forcedEager.tools)).toContain('"type":"tool_search"');
  });

  it('falls back when a search call ID is shared with an ordinary tool call', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        {
          name: 'calendar_create',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        },
        { name: 'lookup', input_schema: { type: 'object', properties: {} } }
      ],
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'shared_id', name: 'ToolSearch', input: { query: 'calendar' } },
            { type: 'tool_use', id: 'shared_id', name: 'lookup', input: {} }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'shared_id',
              content: '',
              tool_references: ['calendar_create']
            }
          ]
        }
      ]
    });

    expect(body.tools).toHaveLength(3);
    expect(JSON.stringify(body.input)).not.toContain('tool_search_call');
    expect(JSON.stringify(body.input)).not.toContain('tool_search_output');
  });

  it('falls back when a search result precedes its call', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        {
          name: 'calendar_create',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      input: [
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'search_1',
              content: '',
              tool_references: ['calendar_create']
            }
          ]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'search_1', name: 'ToolSearch', input: { query: 'calendar' } }
          ]
        }
      ]
    });

    expect(body.tools).toHaveLength(2);
    expect(JSON.stringify(body.input)).not.toContain('tool_search_call');
    expect(JSON.stringify(body.input)).not.toContain('tool_search_output');
  });

  it('falls back when a deferred function call precedes its discovery output', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        {
          name: 'calendar_create',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'calendar_before_discovery',
              name: 'calendar_create',
              input: { title: 'Planning' }
            }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'calendar_before_discovery',
              content: 'created'
            }
          ]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'search_after_call',
              name: 'ToolSearch',
              input: { query: 'calendar' }
            }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'search_after_call',
              content: '',
              tool_references: ['calendar_create']
            }
          ]
        }
      ]
    });

    expect(body.tools).toEqual([
      { type: 'function', name: 'ToolSearch', parameters: { type: 'object', properties: {} } },
      {
        type: 'function',
        name: 'calendar_create',
        parameters: { type: 'object', properties: {} }
      }
    ]);
    expect(body.input).toEqual([
      {
        type: 'function_call',
        call_id: 'calendar_before_discovery',
        name: 'calendar_create',
        arguments: '{"title":"Planning"}'
      },
      {
        type: 'function_call_output',
        call_id: 'calendar_before_discovery',
        output: 'created'
      },
      {
        type: 'function_call',
        call_id: 'search_after_call',
        name: 'ToolSearch',
        arguments: '{"query":"calendar"}'
      },
      {
        type: 'function_call_output',
        call_id: 'search_after_call',
        output: '[{"type":"tool_reference","tool_name":"calendar_create"}]'
      }
    ]);
  });

  it('keeps native tool search when deferred function calls follow discovery', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        {
          name: 'calendar_create',
          description: 'Create a calendar event.',
          defer_loading: true,
          input_schema: { type: 'object', properties: {} }
        }
      ],
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'search_before_call',
              name: 'ToolSearch',
              input: { query: 'calendar' }
            }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'search_before_call',
              content: '',
              tool_references: ['calendar_create']
            }
          ]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'calendar_after_discovery',
              name: 'calendar_create',
              input: { title: 'Planning' }
            }
          ]
        }
      ]
    });

    expect(body.tools).toEqual([
      {
        type: 'tool_search',
        execution: 'client',
        parameters: { type: 'object', properties: {} }
      }
    ]);
    expect(body.input).toEqual([
      {
        type: 'tool_search_call',
        execution: 'client',
        call_id: 'search_before_call',
        status: 'completed',
        arguments: { query: 'calendar' }
      },
      {
        type: 'tool_search_output',
        execution: 'client',
        call_id: 'search_before_call',
        status: 'completed',
        tools: [
          {
            type: 'function',
            name: 'calendar_create',
            description: 'Create a calendar event.',
            parameters: { type: 'object', properties: {} },
            defer_loading: true
          }
        ]
      },
      {
        type: 'function_call',
        call_id: 'calendar_after_discovery',
        name: 'calendar_create',
        arguments: '{"title":"Planning"}'
      }
    ]);
  });

  it('falls back when eager and deferred tools share a name', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        { name: 'calendar_create', input_schema: { type: 'object', properties: { eager: { type: 'boolean' } } } },
        {
          name: 'calendar_create',
          defer_loading: true,
          input_schema: { type: 'object', properties: { deferred: { type: 'boolean' } } }
        }
      ],
      input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]
    });

    expect(body.tools).toHaveLength(3);
    expect(JSON.stringify(body.tools)).not.toContain('"type":"tool_search"');
  });

  it('keeps deferred hosted tools eager while translating their search history', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        { type: 'web_search_20250305', name: 'web_search', defer_loading: true }
      ],
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'search_web', name: 'ToolSearch', input: { query: 'news' } }
          ]
        },
        {
          type: 'message',
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'search_web',
              content: '',
              tool_references: ['web_search']
            }
          ]
        }
      ]
    });

    expect(body.tools).toEqual([
      { type: 'tool_search', execution: 'client', parameters: { type: 'object', properties: {} } },
      { type: 'web_search' }
    ]);
    expect(body.input).toEqual([
      {
        type: 'tool_search_call',
        execution: 'client',
        call_id: 'search_web',
        status: 'completed',
        arguments: { query: 'news' }
      },
      {
        type: 'tool_search_output',
        execution: 'client',
        call_id: 'search_web',
        status: 'completed',
        tools: []
      }
    ]);
  });

  it('forces deferred hosted tools through a compatible advertised choice', () => {
    const body = buildOpenAIResponsesBodyFromStandardRequest({
      model: 'claude-sonnet-4-5',
      tools: [
        { name: 'ToolSearch', input_schema: { type: 'object', properties: {} } },
        { type: 'web_search_20250305', name: 'web_search', defer_loading: true }
      ],
      tool_choice: { type: 'tool', name: 'web_search' },
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'find current news' }]
        }
      ]
    });

    expect(body.tools).toEqual([
      { type: 'function', name: 'ToolSearch', parameters: { type: 'object', properties: {} } },
      { type: 'web_search' }
    ]);
    expect(body.tool_choice).toEqual({
      type: 'allowed_tools',
      mode: 'required',
      tools: [{ type: 'web_search' }]
    });
    expect(body.tool_choice).not.toEqual({ type: 'function', name: 'web_search' });
  });

  it('preserves native client tool search during Responses-to-Responses routing', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      tools: [
        {
          type: 'tool_search',
          execution: 'client',
          description: 'Find a matching tool.',
          parameters: {
            type: 'object',
            properties: { query: { type: 'string' } }
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
              description: 'Create an event.',
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

    const body = buildOpenAIResponsesBodyFromStandardRequest(parsed.value);
    expect(body.tools).toEqual([
      {
        type: 'tool_search',
        execution: 'client',
        description: 'Find a matching tool.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } }
        }
      }
    ]);
    expect(body.input).toEqual([
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
            description: 'Create an event.',
            defer_loading: true,
            parameters: { type: 'object', properties: {} }
          }
        ]
      }
    ]);
  });

  it('preserves deferred flags for native Responses tool search declarations', () => {
    const parsed = parseOpenAIResponsesRequest({
      model: 'gpt-5.6',
      input: 'Find calendar tools',
      tools: [
        {
          type: 'function',
          name: 'calendar_create',
          defer_loading: true,
          parameters: { type: 'object', properties: {} }
        },
        {
          type: 'namespace',
          name: 'calendar',
          tools: [
            {
              type: 'function',
              name: 'delete',
              defer_loading: true,
              parameters: { type: 'object', properties: {} }
            },
            {
              type: 'function',
              name: 'list',
              parameters: { type: 'object', properties: {} }
            }
          ]
        },
        {
          type: 'tool_search',
          execution: 'client',
          parameters: {
            type: 'object',
            properties: { query: { type: 'string' } }
          }
        }
      ]
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const body = buildOpenAIResponsesBodyFromStandardRequest(parsed.value);
    expect(body.tools).toEqual([
      {
        type: 'function',
        name: 'calendar_create',
        parameters: { type: 'object', properties: {} },
        defer_loading: true
      },
      {
        type: 'namespace',
        name: 'calendar',
        tools: [
          {
            type: 'function',
            name: 'delete',
            parameters: { type: 'object', properties: {} },
            defer_loading: true
          },
          {
            type: 'function',
            name: 'list',
            parameters: { type: 'object', properties: {} }
          }
        ]
      },
      {
        type: 'tool_search',
        execution: 'client',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } }
        }
      }
    ]);
  });

  it('parses client tool search calls as Claude ToolSearch requests', () => {
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_tool_search',
      model: 'gpt-5.6',
      output: [
        {
          type: 'tool_search_call',
          execution: 'client',
          call_id: 'toolu_search',
          status: 'completed',
          arguments: { query: 'calendar create' }
        }
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.output).toMatchObject([
      {
        type: 'function_call',
        id: 'toolu_search',
        call_id: 'toolu_search',
        name: 'ToolSearch',
        arguments: '{"query":"calendar create"}',
        status: 'completed'
      }
    ]);
  });

  it('does not promote incomplete client tool search calls to executable calls', () => {
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_tool_search_incomplete',
      model: 'gpt-5.6',
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [
        {
          type: 'tool_search_call',
          execution: 'client',
          call_id: 'toolu_search_incomplete',
          status: 'incomplete',
          arguments: { query: 'calendar' }
        }
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14
      }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }
    expect(parsed.value.status).toBe('incomplete');
    expect(parsed.value.finish_reason).toBe('max_output_tokens');
    expect(parsed.value.output).toMatchObject([
      {
        type: 'provider_native_item',
        item_type: 'tool_search_call',
        provider_status: 'incomplete',
        capture_state: 'complete'
      }
    ]);
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

  it('captures the complete ordered Responses output including phase, PTC, and compaction state', () => {
    const rawOutput = [
      {
        type: 'reasoning',
        id: 'rs_ordered',
        status: 'completed',
        summary: [{ type: 'summary_text', text: 'Plan the lookup.' }],
        content: [{ type: 'reasoning_text', text: 'Use the program first.' }],
        encrypted_content: 'encrypted-reasoning'
      },
      {
        type: 'message',
        id: 'msg_ordered',
        role: 'assistant',
        phase: 'commentary',
        status: 'completed',
        content: [{ type: 'output_text', text: 'Running the program.', annotations: [] }]
      },
      {
        type: 'program',
        id: 'prog_ordered',
        status: 'completed',
        code: 'const result = await lookup({ city: "Wuhu" });',
        caller: { type: 'code_interpreter', id: 'caller_ordered' },
        fingerprint: 'program-fingerprint'
      },
      {
        type: 'program_output',
        id: 'prog_output_ordered',
        program_id: 'prog_ordered',
        status: 'completed',
        output: '22 C'
      },
      {
        type: 'function_call',
        id: 'fc_ordered',
        call_id: 'call_ordered',
        name: 'lookup',
        arguments: '{"city":"Wuhu"}',
        caller: { type: 'program', id: 'prog_ordered' },
        status: 'completed'
      },
      {
        type: 'function_call_output',
        id: 'fco_ordered',
        call_id: 'call_ordered',
        output: '22 C',
        status: 'completed'
      },
      {
        type: 'compaction',
        id: 'cmp_ordered',
        encrypted_content: 'opaque-compacted-window',
        status: 'completed'
      }
    ];
    const parsed = openAIResponsesTargetAdapter.toStandardResponse({
      id: 'resp_ordered_native_state',
      model: 'gpt-5.6-sol',
      status: 'completed',
      output: rawOutput,
      usage: { input_tokens: 12, output_tokens: 8, total_tokens: 20 }
    });

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    expect(parsed.value.output.map((item) => item.type)).toEqual([
      'reasoning',
      'message',
      'provider_native_item',
      'provider_native_item',
      'function_call',
      'provider_native_item',
      'provider_native_item'
    ]);
    expect(parsed.value.output[1]).toMatchObject({
      type: 'message',
      phase: 'commentary'
    });
    const nativeItems = parsed.value.output.map((item) =>
      item.type === 'provider_native_item'
        ? item
        : 'native_item' in item
          ? item.native_item
          : undefined
    );
    expect(nativeItems.map((item) => item?.raw_payload)).toEqual(rawOutput);
    expect(nativeItems.map((item) => item?.position.item)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(nativeItems.every((item) => item?.capture_state === 'complete')).toBe(true);
    expect(nativeItems[6]).toMatchObject({
      item_type: 'compaction',
      compaction_mode: 'server_side'
    });

    const compactParsed = openAIResponsesTargetAdapter.toStandardResponse(
      {
        id: 'resp_standalone_compaction',
        model: 'gpt-5.6-sol',
        status: 'completed',
        output: [rawOutput[6]]
      },
      {
        standardRequest: {
          model: 'gpt-5.6-sol',
          input: 'compact this window',
          openai_responses: { operation: 'compact' }
        }
      } as never
    );
    expect(compactParsed.ok).toBe(true);
    if (compactParsed.ok) {
      expect(compactParsed.value.output[0]).toMatchObject({
        type: 'provider_native_item',
        item_type: 'compaction',
        compaction_mode: 'standalone',
        capture_state: 'complete'
      });
    }
  });

  it('prunes only stateless server-side history before the latest compaction', () => {
    const compaction = {
      type: 'provider_native_item',
      item_type: 'compaction',
      native_id: 'cmp_latest',
      raw_payload: {
        type: 'compaction',
        id: 'cmp_latest',
        encrypted_content: 'opaque-window'
      },
      provider_schema_version: 'openai-responses-v1',
      source_format: 'openai-responses-v1',
      source_origin: {
        provider: 'openai',
        endpoint: 'openai-endpoint',
        model: 'gpt-5.6-sol',
        credentialScope: 'account-scope'
      },
      position: { turn: 1, step: 0, item: 0 },
      capture_state: 'complete',
      compaction_mode: 'server_side'
    };
    const request = {
      model: 'gpt-5.6-sol',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'history before compaction' }]
        },
        {
          type: 'message',
          role: 'assistant',
          content: [compaction],
          native_items: [compaction]
        },
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'new delta' }]
        }
      ],
      openai_responses: { operation: 'create' }
    };

    const serverSide = buildOpenAIResponsesBodyFromStandardRequest(request as never);
    expect(serverSide.input).toEqual([
      compaction.raw_payload,
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'new delta' }]
      }
    ]);

    const standalone = buildOpenAIResponsesBodyFromStandardRequest({
      ...request,
      openai_responses: { operation: 'compact' }
    } as never);
    expect(standalone.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'history before compaction' }]
      },
      compaction.raw_payload,
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'new delta' }]
      }
    ]);
  });
});

function buildAnthropicOpenAITargetBody(
  requestBody: Record<string, unknown>,
  targetProviderConfig: Record<string, unknown>
): Record<string, unknown> {
  const parsed = parseAnthropicMessagesRequest(requestBody);
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
        index: 0
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
