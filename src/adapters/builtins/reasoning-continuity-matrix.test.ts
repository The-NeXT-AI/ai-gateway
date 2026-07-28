import { describe, expect, it } from 'vitest';
import type {
  SourceAdapter,
  StandardRequest,
  StandardRequestInputContent,
  StandardResponse,
  TargetAdapter
} from '../../types';
import {
  decodeReasoningTransportEnvelope,
  decodeOpenAIResponsesReasoningEnvelope,
  OPENAI_RESPONSES_REASONING_FORMAT
} from './reasoning-envelope';
import { anthropicMessagesSourceAdapter } from './source/anthropic-messages';
import { geminiGenerateContentSourceAdapter } from './source/gemini-generate-content';
import { geminiInteractionsSourceAdapter } from './source/gemini-interactions';
import { geminiStreamGenerateContentSourceAdapter } from './source/gemini-stream-generate-content';
import { openAIChatCompletionsSourceAdapter } from './source/openai-chat-completions';
import { openAIResponsesSourceAdapter } from './source/openai-responses';
import { anthropicMessagesTargetAdapter } from './target/anthropic-messages';
import { geminiGenerateContentTargetAdapter } from './target/gemini-generate-content';
import { openAIResponsesTargetAdapter } from './target/openai-responses';

type WireObject = Record<string, any>;

interface SourceScenario {
  name: string;
  protocol: string;
  adapter: SourceAdapter;
  source: Record<string, unknown>;
  passthroughBody: Record<string, unknown>;
  buildHistory(response: WireObject): Record<string, unknown>;
}

interface TargetScenario {
  name: string;
  protocol: string;
  adapter: TargetAdapter;
  marker: string;
  targetProviderConfig: Record<string, unknown>;
  rawResponse: Record<string, unknown>;
}

const gatewayConfig = {
  openaiApiKey: 'test-api-key',
  openaiBaseUrl: 'https://provider.example/v1',
  anthropicApiKey: 'test-api-key',
  anthropicBaseUrl: 'https://provider.example',
  geminiApiKey: 'test-api-key',
  geminiBaseUrl: 'https://provider.example',
  geminiApiVersion: 'v1beta'
};

const sourceScenarios: SourceScenario[] = [
  {
    name: 'OpenAI Responses',
    protocol: 'openai_responses',
    adapter: openAIResponsesSourceAdapter,
    source: { adapterKey: 'openai_responses' },
    passthroughBody: {
      model: 'gpt-5.6-sol',
      input: 'Hello'
    },
    buildHistory(response) {
      return {
        model: 'gpt-5.6-sol',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: 'First turn' }]
          },
          ...response.output,
          {
            type: 'message',
            role: 'user',
            content: [{ type: 'input_text', text: 'Second turn' }]
          }
        ]
      };
    }
  },
  {
    name: 'OpenAI Chat Completions',
    protocol: 'openai_chat_completions',
    adapter: openAIChatCompletionsSourceAdapter,
    source: { adapterKey: 'openai_chat' },
    passthroughBody: {
      model: 'gpt-5.6-sol',
      messages: [{ role: 'user', content: 'Hello' }]
    },
    buildHistory(response) {
      return {
        model: 'gpt-5.6-sol',
        messages: [
          { role: 'user', content: 'First turn' },
          response.choices[0].message,
          { role: 'user', content: 'Second turn' }
        ]
      };
    }
  },
  {
    name: 'Anthropic Messages',
    protocol: 'anthropic_messages',
    adapter: anthropicMessagesSourceAdapter,
    source: { adapterKey: 'anthropic_messages' },
    passthroughBody: {
      model: 'gpt-5.6-sol',
      max_tokens: 128,
      messages: [{ role: 'user', content: 'Hello' }]
    },
    buildHistory(response) {
      return {
        model: 'gpt-5.6-sol',
        max_tokens: 128,
        messages: [
          { role: 'user', content: 'First turn' },
          { role: 'assistant', content: response.content },
          { role: 'user', content: 'Second turn' }
        ]
      };
    }
  },
  {
    name: 'Gemini generateContent',
    protocol: 'gemini_generate_content',
    adapter: geminiGenerateContentSourceAdapter,
    source: {
      adapterKey: 'gemini_generate',
      metadata: {
        model: 'gpt-5.6-sol',
        action: 'generateContent',
        apiVersion: 'v1beta'
      }
    },
    passthroughBody: {
      contents: [{ role: 'user', parts: [{ text: 'Hello' }] }]
    },
    buildHistory(response) {
      return {
        contents: [
          { role: 'user', parts: [{ text: 'First turn' }] },
          response.candidates[0].content,
          { role: 'user', parts: [{ text: 'Second turn' }] }
        ]
      };
    }
  },
  {
    name: 'Gemini streamGenerateContent',
    protocol: 'gemini_generate_content',
    adapter: geminiStreamGenerateContentSourceAdapter,
    source: {
      adapterKey: 'gemini_stream',
      metadata: {
        model: 'gpt-5.6-sol',
        action: 'streamGenerateContent',
        apiVersion: 'v1beta'
      }
    },
    passthroughBody: {
      contents: [{ role: 'user', parts: [{ text: 'Hello' }] }]
    },
    buildHistory(response) {
      return {
        contents: [
          { role: 'user', parts: [{ text: 'First turn' }] },
          response.candidates[0].content,
          { role: 'user', parts: [{ text: 'Second turn' }] }
        ]
      };
    }
  },
  {
    name: 'Gemini Interactions',
    protocol: 'gemini_interactions',
    adapter: geminiInteractionsSourceAdapter,
    source: {
      adapterKey: 'gemini_interactions',
      metadata: {
        apiVersion: 'v1beta'
      }
    },
    passthroughBody: {
      model: 'gpt-5.6-sol',
      input: 'Hello'
    },
    buildHistory(response) {
      return {
        model: 'gpt-5.6-sol',
        input: [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'First turn' }]
          },
          ...response.steps,
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'Second turn' }]
          }
        ]
      };
    }
  }
];

const targetScenarios: TargetScenario[] = [
  {
    name: 'OpenAI Responses',
    protocol: 'openai_responses',
    adapter: openAIResponsesTargetAdapter,
    marker: 'encrypted-openai-responses-reasoning',
    targetProviderConfig: {
      name: 'responses-target',
      type: 'openai_responses',
      models: ['gpt-5.6-sol']
    },
    rawResponse: {
      id: 'resp_openai_responses_reasoning',
      object: 'response',
      status: 'completed',
      model: 'gpt-5.6-sol',
      output_text: 'First answer.',
      output: [
        {
          id: 'rs_openai_responses_reasoning',
          type: 'reasoning',
          status: 'completed',
          summary: [],
          encrypted_content: 'encrypted-openai-responses-reasoning'
        },
        {
          id: 'msg_openai_responses_reasoning',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            {
              type: 'output_text',
              text: 'First answer.',
              annotations: []
            }
          ]
        }
      ],
      usage: {
        input_tokens: 4,
        output_tokens: 3,
        total_tokens: 7
      }
    }
  },
  {
    name: 'OpenAI Chat Completions',
    protocol: 'openai_chat_completions',
    adapter: openAIResponsesTargetAdapter,
    marker: 'encrypted-openai-chat-reasoning',
    targetProviderConfig: {
      name: 'chat-target',
      type: 'openai_chat_completions',
      baseurl: 'https://api.minimax.io/v1',
      models: ['gpt-5.6-sol']
    },
    rawResponse: {
      id: 'chatcmpl_reasoning',
      object: 'chat.completion',
      model: 'gpt-5.6-sol',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'First answer.',
            reasoning_details: [
              {
                type: 'reasoning.encrypted',
                data: 'encrypted-openai-chat-reasoning',
                format: 'openai-chat-compat-v1'
              }
            ]
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: 4,
        completion_tokens: 3,
        total_tokens: 7
      }
    }
  },
  {
    name: 'Anthropic Messages',
    protocol: 'anthropic_messages',
    adapter: anthropicMessagesTargetAdapter,
    marker: 'encrypted-anthropic-reasoning',
    targetProviderConfig: {
      name: 'anthropic-target',
      type: 'anthropic_messages',
      models: ['gpt-5.6-sol']
    },
    rawResponse: {
      id: 'msg_anthropic_reasoning',
      type: 'message',
      role: 'assistant',
      model: 'gpt-5.6-sol',
      content: [
        {
          type: 'redacted_thinking',
          data: 'encrypted-anthropic-reasoning'
        },
        {
          type: 'text',
          text: 'First answer.'
        }
      ],
      stop_reason: 'end_turn',
      usage: {
        input_tokens: 4,
        output_tokens: 3
      }
    }
  },
  {
    name: 'Gemini generateContent',
    protocol: 'gemini_generate_content',
    adapter: geminiGenerateContentTargetAdapter,
    marker: 'encrypted-gemini-generate-reasoning',
    targetProviderConfig: {
      name: 'gemini-generate-target',
      type: 'gemini_generate_content',
      models: ['gpt-5.6-sol']
    },
    rawResponse: {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                text: 'Need a lookup.',
                thought: true
              },
              {
                thoughtSignature: 'encrypted-gemini-generate-reasoning',
                functionCall: {
                  id: 'call_gemini_generate_reasoning',
                  name: 'lookup_value',
                  args: {
                    key: 'first'
                  }
                }
              },
              {
                text: 'First answer.'
              }
            ]
          },
          finishReason: 'STOP'
        }
      ],
      usageMetadata: {
        promptTokenCount: 4,
        candidatesTokenCount: 3,
        totalTokenCount: 7
      },
      modelVersion: 'gpt-5.6-sol'
    }
  },
  {
    name: 'Gemini Interactions',
    protocol: 'gemini_interactions',
    adapter: geminiGenerateContentTargetAdapter,
    marker: 'encrypted-gemini-interactions-reasoning',
    targetProviderConfig: {
      name: 'gemini-interactions-target',
      type: 'gemini_interactions',
      models: ['gpt-5.6-sol']
    },
    rawResponse: {
      id: 'int_gemini_interactions_reasoning',
      object: 'interaction',
      status: 'completed',
      model: 'gpt-5.6-sol',
      steps: [
        {
          type: 'thought',
          signature: 'encrypted-gemini-interactions-reasoning'
        },
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'First answer.' }]
        }
      ],
      usage: {
        total_input_tokens: 4,
        total_output_tokens: 3,
        total_tokens: 7
      }
    }
  }
];

const strictReasoningTargets = targetScenarios.filter(
  (target) => target.protocol !== 'openai_chat_completions'
);

const conversionScenarios = sourceScenarios.flatMap((source) =>
  strictReasoningTargets
    .map((target) => ({
      name: `${source.name} → ${target.name}`,
      source,
      target
    }))
);

const sourceCarrierScenarios = sourceScenarios.flatMap((source) =>
  targetScenarios.map((origin) => ({
    name: `${source.name} carries ${origin.name}`,
    source,
    origin
  }))
);

const targetAcceptanceScenarios = targetScenarios.flatMap((origin) =>
  targetScenarios.map((target) => ({
    name: `${target.name} accepts ${origin.name}`,
    origin,
    target
  }))
);

describe('reasoning continuity conversion matrix', () => {
  it('contains all 24 converted source-to-target paths', () => {
    expect(conversionScenarios).toHaveLength(24);
  });

  it.each(conversionScenarios)('preserves provider-native reasoning through $name', ({ source, target }) => {
    const upstreamParsed = parseTargetResponse(target);
    expect(upstreamParsed.ok).toBe(true);
    if (!upstreamParsed.ok) {
      return;
    }
    expect(containsReasoningMarker(upstreamParsed.value, target.marker)).toBe(true);

    const formatted = source.adapter.fromStandardResponse({
      request: {} as never,
      response: upstreamParsed.value,
      source: source.source,
      config: {}
    } as never) as WireObject;
    const parsed = source.adapter.toStandardRequest({
      request: {} as never,
      body: source.buildHistory(formatted),
      source: source.source,
      config: {
        geminiApiVersion: 'v1beta'
      }
    } as never);

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const built = buildTargetRequest(target, parsed.value);
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(containsReasoningMarker(body, target.marker)).toBe(true);
    expect(JSON.stringify(body)).toContain('First answer.');

    if (target.protocol === 'openai_responses') {
      expect(findObject(body, isResponsesReasoningItem)).toMatchObject({
        type: 'reasoning',
        id: 'rs_openai_responses_reasoning',
        encrypted_content: target.marker
      });
    }

  });

  it('contains all 30 source carrier contracts', () => {
    expect(sourceCarrierScenarios).toHaveLength(30);
  });

  it.each(sourceCarrierScenarios)('$name', ({ source, origin }) => {
    const upstreamParsed = parseTargetResponse(origin);
    expect(upstreamParsed.ok).toBe(true);
    if (!upstreamParsed.ok) {
      return;
    }

    const formatted = source.adapter.fromStandardResponse({
      request: {} as never,
      response: upstreamParsed.value,
      source: source.source,
      config: {}
    } as never) as WireObject;
    const reparsed = source.adapter.toStandardRequest({
      request: {} as never,
      body: source.buildHistory(formatted),
      source: source.source,
      config: {
        geminiApiVersion: 'v1beta'
      }
    } as never);

    expect(reparsed.ok).toBe(true);
    if (!reparsed.ok) {
      return;
    }
    expect(containsReasoningMarker(reparsed.value, origin.marker)).toBe(true);

    if (origin.protocol === 'openai_responses') {
      expect(findObject(reparsed.value, isResponsesReasoningItem)).toMatchObject({
        type: 'reasoning',
        id: 'rs_openai_responses_reasoning',
        source_format: OPENAI_RESPONSES_REASONING_FORMAT,
        encrypted_content: origin.marker
      });
    }
  });

  it('contains all 25 target acceptance contracts', () => {
    expect(targetAcceptanceScenarios).toHaveLength(25);
  });

  it.each(targetAcceptanceScenarios)('$name', ({ origin, target }) => {
    const upstreamParsed = parseTargetResponse(origin);
    expect(upstreamParsed.ok).toBe(true);
    if (!upstreamParsed.ok) {
      return;
    }

    const built = buildTargetRequest(
      target,
      createSecondTurnStandardRequest(upstreamParsed.value)
    );
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }

    const body = built.value.body as Record<string, unknown>;
    expect(containsReasoningMarker(body, origin.marker)).toBe(
      targetCanAcceptOrigin(target, origin)
    );
    expect(JSON.stringify(body)).toContain('First answer.');

    if (
      target.protocol === 'openai_responses' &&
      origin.protocol === 'openai_responses'
    ) {
      expect(findObject(body, isResponsesReasoningItem)).toMatchObject({
        type: 'reasoning',
        id: 'rs_openai_responses_reasoning',
        encrypted_content: origin.marker
      });
    }

  });

  it('keeps unsupported generated reasoning out of generic Chat requests', () => {
    const origin = targetScenarios[0]!;
    const upstreamParsed = parseTargetResponse(origin);
    expect(upstreamParsed.ok).toBe(true);
    if (!upstreamParsed.ok) {
      return;
    }

    const genericChat: TargetScenario = {
      ...targetScenarios[1]!,
      targetProviderConfig: {
        name: 'generic-chat-target',
        type: 'openai_chat_completions',
        baseurl: 'https://provider.example/v1',
        models: ['gpt-5.6-sol']
      }
    };
    const built = buildTargetRequest(
      genericChat,
      createSecondTurnStandardRequest(upstreamParsed.value)
    );
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    expect(containsReasoningMarker(built.value.body, origin.marker)).toBe(false);
  });

  it.each(sourceScenarios)('keeps $name same-protocol requests as passthrough', (source) => {
    const built = source.adapter.buildPassthroughRequest({
      request: {
        headers: {},
        url: passthroughUrl(source)
      } as never,
      body: source.passthroughBody,
      source: source.source,
      config: gatewayConfig
    } as never);

    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    expect(built.value.body).toEqual(source.passthroughBody);
  });
});

function parseTargetResponse(target: TargetScenario) {
  return target.adapter.toStandardResponse(target.rawResponse, {
    request: {
      headers: {}
    } as never,
    standardRequest: {
      model: 'gpt-5.6-sol',
      input: 'First turn'
    },
    config: gatewayConfig,
    targetProviderConfig: target.targetProviderConfig
  } as never);
}

function buildTargetRequest(target: TargetScenario, standardRequest: StandardRequest) {
  return target.adapter.buildRequestFromStandard({
    request: {
      headers: {}
    } as never,
    standardRequest,
    config: gatewayConfig,
    targetProviderConfig: target.targetProviderConfig
  } as never);
}

function createSecondTurnStandardRequest(response: StandardResponse): StandardRequest {
  const assistantContent: StandardRequestInputContent[] = [];
  for (const item of response.output) {
    if (item.type === 'reasoning') {
      const reasoningText = item.content
        ?.filter((part) => part.type === 'reasoning_text')
        .map((part) => part.text)
        .join('\n');
      const reasoningDetails =
        item.reasoning_details ||
        (
          item.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
          item.encrypted_content
            ? [
                {
                  type: 'reasoning.encrypted',
                  data: item.encrypted_content,
                  id: item.id,
                  format: OPENAI_RESPONSES_REASONING_FORMAT
                }
              ]
            : undefined
        );
      assistantContent.push({
        type: 'reasoning',
        id: item.id,
        ...(item.source_format ? { source_format: item.source_format } : {}),
        ...(item.encrypted_content
          ? { encrypted_content: item.encrypted_content }
          : {}),
        ...(reasoningDetails
          ? { reasoning_details: reasoningDetails }
          : {}),
        ...(reasoningText ? { text: reasoningText } : {}),
        ...(item.summary.length
          ? { summary: item.summary.map((entry) => entry.text).join('\n') }
          : {})
      });
      continue;
    }

    if (item.type === 'message') {
      for (const content of item.content) {
        if (content.type === 'output_text') {
          assistantContent.push({
            type: 'input_text',
            text: content.text
          });
        }
      }
      continue;
    }

    if (item.type === 'function_call') {
      assistantContent.push({
        type: 'tool_use',
        id: item.call_id || item.id,
        name: item.name,
        input: parseJsonObject(item.arguments),
        ...(item.thought_signature
          ? {
              thought_signature: item.thought_signature,
              ...(item.thought_signature_format
                ? {
                    thought_signature_format:
                      item.thought_signature_format
                  }
                : {})
            }
          : {})
      });
    }
  }

  return {
    model: 'gpt-5.6-sol',
    input: [
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'First turn' }]
      },
      {
        type: 'message',
        role: 'assistant',
        content: assistantContent
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Second turn' }]
      }
    ]
  };
}

function parseJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
}

function containsReasoningMarker(value: unknown, marker: string): boolean {
  if (typeof value === 'string') {
    if (value === marker || value.includes(marker)) {
      return true;
    }
    return (
      decodeReasoningTransportEnvelope(value)?.data === marker ||
      decodeOpenAIResponsesReasoningEnvelope(value)?.encryptedContent === marker
    );
  }
  if (Array.isArray(value)) {
    return value.some((item) => containsReasoningMarker(item, marker));
  }
  if (!value || typeof value !== 'object') {
    return false;
  }
  return Object.values(value as Record<string, unknown>).some((item) =>
    containsReasoningMarker(item, marker)
  );
}

function findObject(
  value: unknown,
  predicate: (record: Record<string, unknown>) => boolean
): Record<string, unknown> | undefined {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findObject(item, predicate);
      if (found) {
        return found;
      }
    }
    return undefined;
  }
  if (!value || typeof value !== 'object') {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (predicate(record)) {
    return record;
  }
  for (const child of Object.values(record)) {
    const found = findObject(child, predicate);
    if (found) {
      return found;
    }
  }
  return undefined;
}

function isResponsesReasoningItem(record: Record<string, unknown>): boolean {
  return (
    record.type === 'reasoning' &&
    record.source_format === OPENAI_RESPONSES_REASONING_FORMAT
  ) || (
    record.type === 'reasoning' &&
    typeof record.id === 'string' &&
    record.id.startsWith('rs_') &&
    typeof record.encrypted_content === 'string'
  );
}

function passthroughUrl(source: SourceScenario): string {
  switch (source.adapter.key) {
    case 'openai_responses':
      return '/v1/responses';
    case 'openai_chat':
      return '/v1/chat/completions';
    case 'anthropic_messages':
      return '/v1/messages';
    case 'gemini_interactions':
      return '/v1beta/interactions';
    case 'gemini_stream':
      return '/v1beta/models/gpt-5.6-sol:streamGenerateContent';
    default:
      return '/v1beta/models/gpt-5.6-sol:generateContent';
  }
}

function targetCanAcceptOrigin(target: TargetScenario, origin: TargetScenario): boolean {
  return (
    target.protocol !== 'openai_chat_completions' &&
    target.protocol === origin.protocol
  );
}
