import { Readable } from 'node:stream';
import { describe, expect, it } from 'vitest';
import type { FastifyReply } from 'fastify';
import type {
  GatewayConfig,
  ProviderConfig,
  ReasoningStateOrigin,
  StandardRequest,
  StandardResponse
} from '../types';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  appendGeminiThoughtSignatureToToolCallId,
  containsReasoningTransportCarrier,
  decodeGeminiThoughtSignatureToolCallId,
  decodeReasoningTransportEnvelope,
  encodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT
} from '../adapters/builtins/reasoning-envelope';
import { formatOpenAIChatCompletionsResponse } from '../adapters/builtins/source/formatters';
import { parseOpenAIChatCompletionsRequest } from '../adapters/builtins/source/parsers';
import { relayConvertedStreamFromStandardResponse } from './streaming-conversion';
import {
  buildReasoningStateOrigin,
  canReplayReasoningState,
  createReasoningAwarePassthroughSseStream,
  normalizeReasoningEndpoint,
  prepareReasoningStateForTarget,
  wrapPassthroughReasoningPayload
} from './reasoning-state';

const gatewayConfig = {
  openaiBaseUrl: 'https://api.openai.com/v1',
  anthropicBaseUrl: 'https://api.anthropic.com',
  geminiBaseUrl: 'https://generativelanguage.googleapis.com'
} as GatewayConfig;

const openAIProvider = (baseurl: string): ProviderConfig => ({
  name: 'test-openai',
  type: 'openai_responses',
  baseurl,
  models: [],
  extraHeaders: { default: {}, byModel: {} },
  extraBody: { default: {}, byModel: {} },
  billing: { byModel: {} }
});

describe('reasoning state origin', () => {
  it('normalizes endpoints without credentials, query, fragments, default ports, or trailing slashes', () => {
    expect(
      normalizeReasoningEndpoint('HTTPS://user:pass@EXAMPLE.com:443/v1///?account=private#fragment')
    ).toBe('https://example.com/v1');
  });

  it('uses provider family and normalized service endpoint, not provider name or model', () => {
    const first = buildReasoningStateOrigin(
      'openai',
      { ...openAIProvider('https://EXAMPLE.com:443/v1/'), name: 'first' },
      gatewayConfig,
      'gpt-5.6-sol'
    );
    const second = buildReasoningStateOrigin(
      'openai',
      { ...openAIProvider('https://example.com/v1'), name: 'second' },
      gatewayConfig,
      'gpt-5.6-luna'
    );
    const otherEndpoint = buildReasoningStateOrigin(
      'openai',
      openAIProvider('https://other.example/v1'),
      gatewayConfig,
      'gpt-5.6-sol'
    );

    expect(first.endpoint).toBe(second.endpoint);
    expect(first.endpoint).not.toBe(otherEndpoint.endpoint);
    expect(first.endpoint).not.toContain('example.com');
    expect(first.model).toBe('gpt-5.6-sol');
  });

  it('requires the exact same model for every opaque reasoning format', () => {
    const endpoint = 'endpoint-fingerprint';
    const openAISol: ReasoningStateOrigin = { provider: 'openai', endpoint, model: 'gpt-5.6-sol' };
    const openAILuna: ReasoningStateOrigin = { provider: 'openai', endpoint, model: 'gpt-5.6-luna' };
    expect(
      canReplayReasoningState(
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAISol,
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAILuna
      )
    ).toBe(false);
    expect(
      canReplayReasoningState(
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAISol,
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAISol
      )
    ).toBe(true);

    const claudeA: ReasoningStateOrigin = { provider: 'anthropic', endpoint, model: 'claude-a' };
    const claudeB: ReasoningStateOrigin = { provider: 'anthropic', endpoint, model: 'claude-b' };
    expect(
      canReplayReasoningState(
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        claudeA,
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        claudeB
      )
    ).toBe(false);
    expect(
      canReplayReasoningState(
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        claudeA,
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        claudeA
      )
    ).toBe(true);

    const geminiA: ReasoningStateOrigin = { provider: 'gemini', endpoint, model: 'gemini-a' };
    const geminiB: ReasoningStateOrigin = { provider: 'gemini', endpoint, model: 'gemini-b' };
    for (const format of [
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      GEMINI_INTERACTIONS_REASONING_FORMAT
    ]) {
      expect(canReplayReasoningState(format, geminiA, format, geminiB)).toBe(false);
      expect(canReplayReasoningState(format, geminiA, format, geminiA)).toBe(true);
    }
  });

  it('rejects old origin-less state and state from another endpoint', () => {
    const target: ReasoningStateOrigin = { provider: 'gemini', endpoint: 'endpoint-a' };
    expect(
      canReplayReasoningState(
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        undefined,
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        target
      )
    ).toBe(false);
    expect(
      canReplayReasoningState(
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        { provider: 'gemini', endpoint: 'endpoint-b' },
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        target
      )
    ).toBe(false);
  });
});

describe('reasoning transport envelope v2', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'gemini',
    endpoint: 'endpoint-fingerprint',
    model: 'gemini-3-pro'
  };

  it('round-trips format, opaque state, id, kind, and origin while retaining v1 compatibility', () => {
    const encoded = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'encrypted-signature',
      'reasoning-id',
      'signature',
      origin
    );
    expect(encoded).toMatch(/^ccr-reasoning-transport-v2:/);
    expect(decodeReasoningTransportEnvelope(encoded)).toEqual({
      format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      data: 'encrypted-signature',
      id: 'reasoning-id',
      kind: 'signature',
      origin
    });

    const legacy = encodeReasoningTransportEnvelope(
      OPENAI_RESPONSES_REASONING_FORMAT,
      'legacy-encrypted',
      'rs_legacy',
      'encrypted'
    );
    expect(decodeReasoningTransportEnvelope(legacy)?.origin).toBeUndefined();
  });

  it('adds and removes a Gemini tool-call ID carrier only for a valid v2 Gemini signature', () => {
    const encoded = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'gemini-signature',
      undefined,
      'signature',
      origin
    );
    const carrierId = appendGeminiThoughtSignatureToToolCallId('call_123', encoded);
    expect(carrierId).toContain('__thought__ccr-reasoning-transport-v2:');
    expect(decodeGeminiThoughtSignatureToolCallId(carrierId)).toMatchObject({
      toolCallId: 'call_123',
      envelope: { data: 'gemini-signature', origin }
    });
    expect(containsReasoningTransportCarrier({ tool_call_id: carrierId })).toBe(true);
    expect(decodeGeminiThoughtSignatureToolCallId('normal__thought__not-an-envelope')).toBeUndefined();
  });
});

describe('target reasoning-state filtering', () => {
  const sourceOrigin: ReasoningStateOrigin = {
    provider: 'openai',
    endpoint: 'endpoint-a',
    model: 'gpt-a'
  };
  const request: StandardRequest = {
    model: 'gpt-b',
    input: [
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            id: 'rs_1',
            text: 'readable summary',
            encrypted_content: 'encrypted-openai-state',
            source_format: OPENAI_RESPONSES_REASONING_FORMAT,
            source_origin: sourceOrigin,
            reasoning_details: [
              { type: 'reasoning.text', text: 'readable summary', format: OPENAI_RESPONSES_REASONING_FORMAT },
              { type: 'reasoning.encrypted', data: 'encrypted-openai-state', format: OPENAI_RESPONSES_REASONING_FORMAT }
            ]
          },
          {
            type: 'tool_use',
            id: 'call_1',
            name: 'lookup',
            input: {},
            thought_signature: 'gemini-signature',
            thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
            thought_signature_origin: { provider: 'gemini', endpoint: 'endpoint-gemini' }
          }
        ]
      }
    ]
  };

  it('drops same-endpoint OpenAI state after a model switch', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      OPENAI_RESPONSES_REASONING_FORMAT,
      { ...sourceOrigin, model: 'gpt-b' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    expect(content.some((item) => item.type === 'reasoning')).toBe(false);
    expect(content.find((item) => item.type === 'tool_use')).toEqual({
      type: 'tool_use',
      id: 'call_1',
      name: 'lookup',
      input: {}
    });
  });

  it('keeps OpenAI state only when the endpoint and model both match', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      OPENAI_RESPONSES_REASONING_FORMAT,
      sourceOrigin
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    expect(content.find((item) => item.type === 'reasoning')).toEqual(
      typeof request.input === 'string' ? undefined : request.input[0]?.content[0]
    );
  });

  it('drops foreign encrypted state before a strict target while preserving the tool call', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      { provider: 'gemini', endpoint: 'other-gemini' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    expect(content.some((item) => item.type === 'reasoning')).toBe(false);
    expect(content).toContainEqual({ type: 'tool_use', id: 'call_1', name: 'lookup', input: {} });
  });

  it('keeps readable reasoning for an OpenAI Chat-compatible target but strips all opaque fields', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      undefined,
      { provider: 'openai', endpoint: 'chat-endpoint' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    const reasoning = content.find((item) => item.type === 'reasoning');
    expect(reasoning).toMatchObject({ type: 'reasoning', text: 'readable summary' });
    expect(reasoning).not.toHaveProperty('encrypted_content');
    expect(JSON.stringify(reasoning)).not.toContain('encrypted-openai-state');
  });
});

describe('OpenAI Chat Gemini signature field-preservation fallback', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'gemini',
    endpoint: 'gemini-endpoint',
    model: 'gemini-3-pro'
  };

  const response: StandardResponse = {
    id: 'response_1',
    object: 'response',
    status: 'completed',
    model: 'gemini-3-pro',
    output_text: '',
    output: [
      {
        id: 'fc_1',
        type: 'function_call',
        call_id: 'call_1',
        name: 'lookup',
        arguments: '{"city":"Wuhu"}',
        thought_signature: 'gemini-signature',
        thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        thought_signature_origin: origin,
        status: 'completed'
      },
      {
        id: 'fc_2',
        type: 'function_call',
        call_id: 'call_2',
        name: 'other',
        arguments: '{}',
        status: 'completed'
      }
    ],
    usage: {}
  };

  it('emits both the extension field and ID carrier, then restores clean IDs when the extension is lost', () => {
    const formatted = formatOpenAIChatCompletionsResponse(response) as any;
    const [signedCall, unsignedCall] = formatted.choices[0].message.tool_calls;
    expect(signedCall.extra_content.google.thought_signature).toMatch(/^ccr-reasoning-transport-v2:/);
    expect(signedCall.id).toContain('__thought__ccr-reasoning-transport-v2:');
    expect(unsignedCall.id).toBe('call_2');

    const parsed = parseOpenAIChatCompletionsRequest({
      model: 'chat-client',
      messages: [
        {
          role: 'assistant',
          tool_calls: [
            {
              id: signedCall.id,
              type: 'function',
              function: signedCall.function
            }
          ]
        },
        {
          role: 'tool',
          tool_call_id: signedCall.id,
          content: 'sunny'
        }
      ]
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok || typeof parsed.value.input === 'string') {
      return;
    }
    expect(parsed.value.input[0]?.content).toContainEqual({
      type: 'tool_use',
      id: 'call_1',
      name: 'lookup',
      input: { city: 'Wuhu' },
      thought_signature: 'gemini-signature',
      thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      thought_signature_origin: origin
    });
    expect(parsed.value.input[1]?.content).toContainEqual({
      type: 'tool_result',
      tool_use_id: 'call_1',
      content: 'sunny'
    });
  });

  it('puts the carrier in the initial streaming tool-call delta', async () => {
    let body: Readable | undefined;
    const reply = {
      code: () => reply,
      header: () => reply,
      send: (value: Readable) => {
        body = value;
        return value;
      }
    } as unknown as FastifyReply;
    relayConvertedStreamFromStandardResponse(
      reply,
      { adapterKey: 'openai_chat' },
      response
    );
    expect(body).toBeDefined();
    const chunks: string[] = [];
    for await (const chunk of body!) {
      chunks.push(String(chunk));
    }
    const wire = chunks.join('');
    expect(wire).toContain('call_1__thought__ccr-reasoning-transport-v2:');
    expect(wire).toContain('"id":"call_2"');
  });
});

describe('same-protocol passthrough wrapping', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'anthropic',
    endpoint: 'anthropic-endpoint',
    model: 'claude-test'
  };

  it('wraps native non-stream opaque fields without changing readable content', () => {
    const payload = {
      content: [
        { type: 'thinking', thinking: 'summary', signature: 'sig' },
        { type: 'text', text: 'answer' }
      ]
    };
    const wrapped = wrapPassthroughReasoningPayload(payload, 'anthropic_messages', origin);
    expect(wrapped.changed).toBe(true);
    expect(payload.content[0]?.thinking).toBe('summary');
    expect(decodeReasoningTransportEnvelope(payload.content[0]?.signature || '')).toMatchObject({
      data: 'sig',
      origin
    });
  });

  it('combines Anthropic signature fragments before emitting one v2 carrier', async () => {
    const upstream = new Response(
      [
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"part"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ].join(''),
      { headers: { 'content-type': 'text/event-stream' } }
    );
    const stream = createReasoningAwarePassthroughSseStream(
      upstream,
      'anthropic_messages',
      origin
    );
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(String(chunk));
    }
    const wire = chunks.join('');
    expect(wire.match(/signature_delta/g)).toHaveLength(1);
    const signatureMatch = wire.match(/"signature":"([^"]+)"/);
    expect(signatureMatch).not.toBeNull();
    expect(decodeReasoningTransportEnvelope(signatureMatch?.[1] || '')).toMatchObject({
      data: 'sig-part',
      origin
    });
    expect(wire.indexOf('signature_delta')).toBeLessThan(wire.indexOf('content_block_stop'));
  });
});
