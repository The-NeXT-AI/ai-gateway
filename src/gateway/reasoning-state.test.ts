import { Readable } from 'node:stream';
import { describe, expect, it } from 'vitest';
import type { FastifyReply, FastifyRequest } from 'fastify';
import type {
  GatewayConfig,
  ProviderNativeItem,
  ProviderConfig,
  ProviderPlugin,
  ReasoningStateOrigin,
  StandardRequest,
  StandardResponse
} from '../types';
import { ProviderPluginRegistry } from '../adapters/registry';
import { parseProviderPluginsFromRaw } from '../config';
import { syncProviderPluginsFromConfig } from '../provider/plugins';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  appendGeminiThoughtSignatureToToolCallId,
  containsReasoningTransportCarrier,
  decodeGeminiThoughtSignatureToolCallId,
  decodeReasoningTransportEnvelope,
  encodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  OPENAI_RESPONSES_REASONING_FORMAT,
  REASONING_TRANSPORT_ENVELOPE_V3_PREFIX,
  validateReasoningTransportCarriers
} from '../adapters/builtins/reasoning-envelope';
import { formatOpenAIChatCompletionsResponse } from '../adapters/builtins/source/formatters';
import { parseOpenAIChatCompletionsRequest } from '../adapters/builtins/source/parsers';
import { relayConvertedStreamFromStandardResponse } from './streaming-conversion';
import {
  attachReasoningStateOrigin,
  buildReasoningStateOrigin,
  canReplayReasoningState,
  clearStatefulNativeRouteCacheForTests,
  createReasoningAwarePassthroughSseStream,
  decideProviderNativeItemReplay,
  deriveProviderNativeGroups,
  normalizeReasoningEndpoint,
  prepareReasoningStateForTarget,
  prepareReasoningStateForTargetResult,
  type ReasoningStateCredentialContext,
  wrapPassthroughReasoningPayload
} from './reasoning-state';

const testCredentialScope = 'test-credential-scope';

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

function credentialContext(
  config: GatewayConfig,
  providerConfig: ProviderConfig,
  plugins: ProviderPlugin[] = [],
  request: Partial<FastifyRequest> = {}
): ReasoningStateCredentialContext {
  return {
    request: {
      headers: {},
      query: {},
      ...request
    } as FastifyRequest,
    config,
    source: { adapterKey: 'anthropic_messages' },
    sourceProvider: 'anthropic',
    sourceAdapterKey: 'anthropic_messages',
    targetProvider: 'openai',
    targetProviderConfig: providerConfig,
    model: 'gpt-5.6-sol',
    passthrough: false,
    streaming: false,
    plugins
  };
}

function configuredPlugins(config: GatewayConfig, providerName: string): ProviderPlugin[] {
  const registry = new ProviderPluginRegistry();
  syncProviderPluginsFromConfig(registry, config);
  return registry.resolve('openai', providerName);
}

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

  it('requires exact models except for Gemini Interactions within the same route scope', () => {
    const endpoint = 'endpoint-fingerprint';
    const openAISol: ReasoningStateOrigin = {
      provider: 'openai',
      endpoint,
      model: 'gpt-5.6-sol',
      credentialScope: testCredentialScope
    };
    const openAILuna: ReasoningStateOrigin = {
      provider: 'openai',
      endpoint,
      model: 'gpt-5.6-luna',
      credentialScope: testCredentialScope
    };
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

    const claudeA: ReasoningStateOrigin = {
      provider: 'anthropic',
      endpoint,
      model: 'claude-a',
      credentialScope: testCredentialScope
    };
    const claudeB: ReasoningStateOrigin = {
      provider: 'anthropic',
      endpoint,
      model: 'claude-b',
      credentialScope: testCredentialScope
    };
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

    const geminiA: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint,
      model: 'gemini-a',
      credentialScope: testCredentialScope
    };
    const geminiB: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint,
      model: 'gemini-b',
      credentialScope: testCredentialScope
    };
    expect(
      canReplayReasoningState(
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        geminiA,
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        geminiB
      )
    ).toBe(false);
    expect(
      canReplayReasoningState(
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        geminiA,
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        geminiA
      )
    ).toBe(true);
    expect(
      canReplayReasoningState(
        GEMINI_INTERACTIONS_REASONING_FORMAT,
        geminiA,
        GEMINI_INTERACTIONS_REASONING_FORMAT,
        geminiB
      )
    ).toBe(true);
  });

  it('requires the exact same credential scope for every opaque reasoning format', () => {
    const first: ReasoningStateOrigin = {
      provider: 'openai',
      endpoint: 'endpoint-fingerprint',
      model: 'model-a',
      credentialScope: 'credential-a'
    };
    const second: ReasoningStateOrigin = {
      ...first,
      credentialScope: 'credential-b'
    };
    for (const format of [
      OPENAI_RESPONSES_REASONING_FORMAT,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      GEMINI_INTERACTIONS_REASONING_FORMAT
    ]) {
      expect(canReplayReasoningState(format, first, format, second)).toBe(false);
      expect(canReplayReasoningState(format, first, format, first)).toBe(true);
    }
  });

  it('rejects old origin-less state and state from another endpoint', () => {
    const target: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'endpoint-a',
      model: 'gemini-a',
      credentialScope: testCredentialScope
    };
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
        {
          provider: 'gemini',
          endpoint: 'endpoint-b',
          model: 'gemini-a',
          credentialScope: testCredentialScope
        },
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        target
      )
    ).toBe(false);
  });

  it('separates static API keys and credential-pool entries without exposing their values', () => {
    const provider = openAIProvider('https://api.example.com/v1');
    const first = buildReasoningStateOrigin(
      'openai',
      { ...provider, apikey: 'test-key-alpha', credentialId: 'account-a' },
      gatewayConfig,
      'gpt-5.6-sol'
    );
    const same = buildReasoningStateOrigin(
      'openai',
      { ...provider, apikey: 'test-key-alpha', credentialId: 'account-a' },
      gatewayConfig,
      'gpt-5.6-sol'
    );
    const changedKey = buildReasoningStateOrigin(
      'openai',
      { ...provider, apikey: 'test-key-beta', credentialId: 'account-a' },
      gatewayConfig,
      'gpt-5.6-sol'
    );
    const changedCredential = buildReasoningStateOrigin(
      'openai',
      { ...provider, apikey: 'test-key-alpha', credentialId: 'account-b' },
      gatewayConfig,
      'gpt-5.6-sol'
    );

    expect(first.credentialScope).toBe(same.credentialScope);
    expect(first.credentialScope).not.toBe(changedKey.credentialScope);
    expect(first.credentialScope).not.toBe(changedCredential.credentialScope);
    expect(JSON.stringify(first)).not.toContain('test-key-alpha');
    expect(JSON.stringify(first)).not.toContain('account-a');
  });

  it('uses the actual request-side credential when unmanaged OpenAI auth is forwarded', () => {
    const provider = openAIProvider('https://api.example.com/v1');
    const first = buildReasoningStateOrigin(
      'openai',
      provider,
      gatewayConfig,
      'gpt-5.6-sol',
      credentialContext(gatewayConfig, provider, [], {
        headers: { authorization: 'Bearer request-key-a' }
      })
    );
    const second = buildReasoningStateOrigin(
      'openai',
      provider,
      gatewayConfig,
      'gpt-5.6-sol',
      credentialContext(gatewayConfig, provider, [], {
        headers: { authorization: 'Bearer request-key-b' }
      })
    );

    expect(first.credentialScope).toBeTruthy();
    expect(first.credentialScope).not.toBe(second.credentialScope);
    expect(JSON.stringify(first)).not.toContain('request-key-a');
  });

  it('separates Anthropic and Gemini credentials with their protocol-specific inputs', () => {
    const anthropicProvider: ProviderConfig = {
      ...openAIProvider('https://api.anthropic.com'),
      name: 'test-anthropic',
      type: 'anthropic_messages'
    };
    const anthropicA = buildReasoningStateOrigin(
      'anthropic',
      anthropicProvider,
      gatewayConfig,
      'claude-test',
      {
        ...credentialContext(gatewayConfig, anthropicProvider, [], {
          headers: { 'x-api-key': 'anthropic-key-a' }
        }),
        targetProvider: 'anthropic',
        model: 'claude-test'
      }
    );
    const anthropicB = buildReasoningStateOrigin(
      'anthropic',
      anthropicProvider,
      gatewayConfig,
      'claude-test',
      {
        ...credentialContext(gatewayConfig, anthropicProvider, [], {
          headers: { 'x-api-key': 'anthropic-key-b' }
        }),
        targetProvider: 'anthropic',
        model: 'claude-test'
      }
    );

    const geminiProvider: ProviderConfig = {
      ...openAIProvider('https://generativelanguage.googleapis.com'),
      name: 'test-gemini',
      type: 'gemini_generate_content'
    };
    const geminiA = buildReasoningStateOrigin(
      'gemini',
      geminiProvider,
      gatewayConfig,
      'gemini-test',
      {
        ...credentialContext(gatewayConfig, geminiProvider, [], { query: { key: 'gemini-key-a' } }),
        targetProvider: 'gemini',
        model: 'gemini-test'
      }
    );
    const geminiB = buildReasoningStateOrigin(
      'gemini',
      geminiProvider,
      gatewayConfig,
      'gemini-test',
      {
        ...credentialContext(gatewayConfig, geminiProvider, [], { query: { key: 'gemini-key-b' } }),
        targetProvider: 'gemini',
        model: 'gemini-test'
      }
    );

    expect(anthropicA.credentialScope).not.toBe(anthropicB.credentialScope);
    expect(geminiA.credentialScope).not.toBe(geminiB.credentialScope);
    expect(JSON.stringify({ anthropicA, geminiA })).not.toContain('key-a');
  });

  it('keeps Codex OAuth scope stable across access-token refreshes and separates accounts', () => {
    const provider = {
      ...openAIProvider('https://chatgpt.com/backend-api/codex'),
      apikey: 'local-agent-placeholder'
    };
    const originFor = (accountId: string, accessToken: string) => {
      const config = {
        ...gatewayConfig,
        providerPlugins: parseProviderPluginsFromRaw([
          {
            key: 'codex-oauth',
            provider: 'openai',
            providerName: provider.name,
            codexOauth: {
              accessToken,
              refreshToken: `refresh-${accountId}`,
              accountId
            }
          }
        ])
      } as GatewayConfig;
      return buildReasoningStateOrigin(
        'openai',
        provider,
        config,
        'gpt-5.6-sol',
        credentialContext(config, provider, configuredPlugins(config, provider.name))
      );
    };

    const first = originFor('test-account-a', 'access-token-old');
    const refreshed = originFor('test-account-a', 'access-token-new');
    const otherAccount = originFor('test-account-b', 'access-token-other');
    expect(first.credentialScope).toBe(refreshed.credentialScope);
    expect(first.credentialScope).not.toBe(otherAccount.credentialScope);
    expect(JSON.stringify(first)).not.toContain('test-account-a');
    expect(JSON.stringify(first)).not.toContain('access-token-old');
  });

  it('extracts a stable Codex account from rotating JWT access tokens', () => {
    const provider = {
      ...openAIProvider('https://chatgpt.com/backend-api/codex'),
      apikey: 'local-agent-placeholder'
    };
    const tokenFor = (marker: string) =>
      `header.${Buffer.from(JSON.stringify({ account_id: 'jwt-account', marker }), 'utf8').toString('base64url')}.signature`;
    const originFor = (accessToken: string) => {
      const config = {
        ...gatewayConfig,
        providerPlugins: parseProviderPluginsFromRaw([
          {
            key: 'codex-oauth',
            provider: 'openai',
            providerName: provider.name,
            codexOauth: { accessToken }
          }
        ])
      } as GatewayConfig;
      return buildReasoningStateOrigin(
        'openai',
        provider,
        config,
        'gpt-5.6-sol',
        credentialContext(config, provider, configuredPlugins(config, provider.name))
      );
    };

    expect(originFor(tokenFor('old')).credentialScope).toBe(
      originFor(tokenFor('new')).credentialScope
    );
  });

  it('uses the Codex refresh token only when no stable account ID is available', () => {
    const provider = {
      ...openAIProvider('https://chatgpt.com/backend-api/codex'),
      apikey: 'local-agent-placeholder'
    };
    const originFor = (refreshToken: string) => {
      const config = {
        ...gatewayConfig,
        providerPlugins: parseProviderPluginsFromRaw([
          {
            key: 'codex-oauth',
            provider: 'openai',
            providerName: provider.name,
            codexOauth: { refreshToken }
          }
        ])
      } as GatewayConfig;
      return buildReasoningStateOrigin(
        'openai',
        provider,
        config,
        'gpt-5.6-sol',
        credentialContext(config, provider, configuredPlugins(config, provider.name))
      );
    };

    expect(originFor('refresh-token-a').credentialScope).toBe(
      originFor('refresh-token-a').credentialScope
    );
    expect(originFor('refresh-token-a').credentialScope).not.toBe(
      originFor('refresh-token-b').credentialScope
    );
    expect(JSON.stringify(originFor('refresh-token-a'))).not.toContain('refresh-token-a');
  });

  it('fails closed when an authenticating plugin cannot provide a stable scope', () => {
    const provider = {
      ...openAIProvider('https://api.example.com/v1'),
      apikey: 'placeholder-key'
    };
    const plugin: ProviderPlugin = {
      key: 'custom-auth',
      authenticate: ({ upstreamRequest }) => ({ ok: true, value: upstreamRequest })
    };
    const origin = buildReasoningStateOrigin(
      'openai',
      provider,
      gatewayConfig,
      'gpt-5.6-sol',
      credentialContext(gatewayConfig, provider, [plugin])
    );

    expect(origin.credentialScope).toBeUndefined();
  });

  it('supports an explicit stable scope for custom authentication plugins', () => {
    const provider = {
      ...openAIProvider('https://custom.example/v1'),
      apikey: 'placeholder-key'
    };
    const config = {
      ...gatewayConfig,
      providerPlugins: parseProviderPluginsFromRaw([
        {
          key: 'custom-scope',
          provider: 'openai',
          providerName: provider.name,
          credentialScope: '{{ request.headers.x-upstream-account }}'
        }
      ])
    } as GatewayConfig;
    const plugins = configuredPlugins(config, provider.name);
    const originFor = (account: string) =>
      buildReasoningStateOrigin(
        'openai',
        provider,
        config,
        'gpt-5.6-sol',
        credentialContext(config, provider, plugins, {
          headers: { 'x-upstream-account': account }
        })
      );

    expect(originFor('custom-account-a').credentialScope).not.toBe(
      originFor('custom-account-b').credentialScope
    );
    expect(JSON.stringify(originFor('custom-account-a'))).not.toContain('custom-account-a');
  });
});

describe('reasoning transport envelope v3', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'gemini',
    endpoint: 'endpoint-fingerprint',
    model: 'gemini-3-pro',
    credentialScope: testCredentialScope
  };

  it('round-trips format, opaque state, id, kind, and origin while retaining v1 compatibility', () => {
    const encoded = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'encrypted-signature',
      'reasoning-id',
      'signature',
      origin
    );
    expect(encoded).toMatch(/^ccr-reasoning-transport-v3:/);
    expect(decodeReasoningTransportEnvelope(encoded)).toMatchObject({
      format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      data: 'encrypted-signature',
      id: 'reasoning-id',
      kind: 'signature',
      origin,
      carrierVersion: 3,
      nativeItem: {
        type: 'provider_native_item',
        item_type: 'thought_signature',
        raw_payload: { data: 'encrypted-signature' },
        capture_state: 'partial'
      }
    });

    const legacy = encodeReasoningTransportEnvelope(
      OPENAI_RESPONSES_REASONING_FORMAT,
      'legacy-encrypted',
      'rs_legacy',
      'encrypted'
    );
    expect(decodeReasoningTransportEnvelope(legacy)?.origin).toBeUndefined();
  });

  it('decodes an older v2 origin but does not treat its missing credential scope as replayable', () => {
    const oldOrigin: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'endpoint-fingerprint',
      model: 'gemini-3-pro'
    };
    const encoded = `ccr-reasoning-transport-v2:${Buffer.from(JSON.stringify({
      format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      data: 'old-v2-signature',
      kind: 'signature',
      origin: oldOrigin
    }), 'utf8').toString('base64url')}`;
    const decoded = decodeReasoningTransportEnvelope(encoded);
    expect(decoded?.origin).toEqual(oldOrigin);
    expect(
      canReplayReasoningState(
        decoded?.format,
        decoded?.origin,
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        origin
      )
    ).toBe(false);
  });

  it('adds and removes a Gemini tool-call ID carrier only for a valid v3 Gemini signature', () => {
    const encoded = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'gemini-signature',
      undefined,
      'signature',
      origin
    );
    const carrierId = appendGeminiThoughtSignatureToToolCallId('call_123', encoded);
    expect(carrierId).toContain('__thought__ccr-reasoning-transport-v3:');
    expect(decodeGeminiThoughtSignatureToolCallId(carrierId)).toMatchObject({
      toolCallId: 'call_123',
      envelope: { data: 'gemini-signature', origin }
    });
    expect(containsReasoningTransportCarrier({ tool_call_id: carrierId })).toBe(true);
    expect(decodeGeminiThoughtSignatureToolCallId('normal__thought__not-an-envelope')).toBeUndefined();
  });

  it('does not treat ordinary IDs containing __thought__ as reasoning carriers', () => {
    const ordinaryIds = {
      id: 'record__thought__literal',
      tool_calls: [{ id: 'call__thought__literal' }],
      tool_call_id: 'result__thought__literal'
    };
    expect(containsReasoningTransportCarrier(ordinaryIds)).toBe(false);
    expect(validateReasoningTransportCarriers(ordinaryIds, 1024 * 1024))
      .toMatchObject({ ok: true, itemCount: 0 });

    const malformedCarrier = {
      tool_calls: [{
        id: `call__thought__${REASONING_TRANSPORT_ENVELOPE_V3_PREFIX}not-valid-base64-json`
      }]
    };
    expect(containsReasoningTransportCarrier(malformedCarrier)).toBe(true);
    expect(validateReasoningTransportCarriers(malformedCarrier, 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'invalid_tool_call_id_carrier' });
  });
});

describe('target reasoning-state filtering', () => {
  const sourceOrigin: ReasoningStateOrigin = {
    provider: 'openai',
    endpoint: 'endpoint-a',
    model: 'gpt-a',
    credentialScope: testCredentialScope
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

  it('strips opaque OpenAI state after a model switch but keeps readable reasoning for policy evaluation', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      OPENAI_RESPONSES_REASONING_FORMAT,
      { ...sourceOrigin, model: 'gpt-b' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    const reasoning = content.find((item) => item.type === 'reasoning');
    expect(reasoning).toMatchObject({ type: 'reasoning', text: 'readable summary' });
    expect(reasoning).not.toHaveProperty('encrypted_content');
    expect(reasoning).not.toHaveProperty('source_origin');
    expect(JSON.stringify(reasoning)).not.toContain('encrypted-openai-state');
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

  it('strips opaque OpenAI state when only the credential scope changes', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      OPENAI_RESPONSES_REASONING_FORMAT,
      { ...sourceOrigin, credentialScope: 'other-credential-scope' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    const reasoning = content.find((item) => item.type === 'reasoning');
    expect(reasoning).toMatchObject({ type: 'reasoning', text: 'readable summary' });
    expect(reasoning).not.toHaveProperty('encrypted_content');
    expect(reasoning).not.toHaveProperty('source_origin');
    expect(content).toContainEqual({ type: 'tool_use', id: 'call_1', name: 'lookup', input: {} });
  });

  it('drops an origin-mismatched Responses item when it has no readable reasoning', () => {
    const encryptedOnly: StandardRequest = {
      model: 'gpt-b',
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'reasoning',
              id: 'rs_encrypted_only',
              source_format: OPENAI_RESPONSES_REASONING_FORMAT,
              source_origin: sourceOrigin,
              encrypted_content: 'encrypted-only-state'
            }
          ]
        }
      ]
    };
    const prepared = prepareReasoningStateForTarget(
      encryptedOnly,
      OPENAI_RESPONSES_REASONING_FORMAT,
      { ...sourceOrigin, model: 'gpt-b' }
    );
    const content = typeof prepared.input === 'string' ? [] : prepared.input[0]?.content || [];
    expect(content).toEqual([]);
  });

  it('applies the same credential check to a Gemini tool-call signature carrier', () => {
    const geminiOrigin: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'gemini-endpoint',
      model: 'gemini-3-pro',
      credentialScope: 'gemini-account-a'
    };
    const geminiRequest: StandardRequest = {
      model: 'gemini-3-pro',
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'call_gemini',
              name: 'lookup',
              input: {},
              thought_signature: 'gemini-signature',
              thought_signature_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
              thought_signature_origin: geminiOrigin
            }
          ]
        }
      ]
    };

    const retained = prepareReasoningStateForTarget(
      geminiRequest,
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      geminiOrigin
    );
    const stripped = prepareReasoningStateForTarget(
      geminiRequest,
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      { ...geminiOrigin, credentialScope: 'gemini-account-b' }
    );
    const retainedContent = typeof retained.input === 'string' ? [] : retained.input[0]?.content || [];
    const strippedContent = typeof stripped.input === 'string' ? [] : stripped.input[0]?.content || [];
    expect(retainedContent[0]).toHaveProperty('thought_signature', 'gemini-signature');
    expect(strippedContent[0]).not.toHaveProperty('thought_signature');
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

  it('does not preserve foreign readable reasoning for an Anthropic strict target', () => {
    const prepared = prepareReasoningStateForTarget(
      request,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      { provider: 'anthropic', endpoint: 'anthropic-endpoint' }
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
    model: 'gemini-3-pro',
    credentialScope: testCredentialScope
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
    expect(signedCall.extra_content.google.thought_signature).toMatch(/^ccr-reasoning-transport-v3:/);
    expect(signedCall.id).toContain('__thought__ccr-reasoning-transport-v3:');
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
    expect(wire).toContain('call_1__thought__ccr-reasoning-transport-v3:');
    expect(wire).toContain('"id":"call_2"');
  });
});

describe('same-protocol passthrough wrapping', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'anthropic',
    endpoint: 'anthropic-endpoint',
    model: 'claude-test',
    credentialScope: testCredentialScope
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

  it('combines Anthropic signature fragments before emitting one complete v3 carrier', async () => {
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
      origin,
      carrierVersion: 3,
      nativeItem: {
        item_type: 'thinking',
        raw_payload: {
          type: 'thinking',
          thinking: '',
          signature: 'sig-part'
        },
        capture_state: 'complete'
      }
    });
    expect(wire.indexOf('signature_delta')).toBeLessThan(wire.indexOf('content_block_stop'));
  });

  it('forwards a complete passthrough event before the upstream stream closes', async () => {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
    const upstream = new Response(
      new ReadableStream<Uint8Array>({
        start(nextController) {
          controller = nextController;
          nextController.enqueue(new TextEncoder().encode(
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_live"}}\n\n'
          ));
        }
      }),
      { headers: { 'content-type': 'text/event-stream' } }
    );
    const stream = createReasoningAwarePassthroughSseStream(
      upstream,
      'anthropic_messages',
      origin
    );
    const iterator = stream[Symbol.asyncIterator]();
    const nextChunk = iterator.next();
    const timedOut = Symbol('timed-out');
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const first = await Promise.race([
      nextChunk,
      new Promise<typeof timedOut>((resolve) => {
        timeoutHandle = setTimeout(() => resolve(timedOut), 1000);
      })
    ]);
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }
    controller?.close();

    if (first === timedOut) {
      await nextChunk;
    }
    await iterator.return?.();
    expect(first).not.toBe(timedOut);
    if (first !== timedOut) {
      expect(String(first.value)).toContain('message_start');
    }
  });
});

describe('provider-native group decisions', () => {
  const openAIOrigin: ReasoningStateOrigin = {
    provider: 'openai',
    endpoint: 'openai-endpoint',
    model: 'gpt-5.6-sol',
    credentialScope: 'openai-account'
  };
  const anthropicOrigin: ReasoningStateOrigin = {
    provider: 'anthropic',
    endpoint: 'anthropic-endpoint',
    model: 'claude-test',
    credentialScope: 'anthropic-account'
  };

  it('derives active, closed, and orphaned states from call relationships', () => {
    const programCall = nativeTestItem({
      item_type: 'program',
      native_id: 'prog_1',
      group_id: 'prog_1',
      call_id: 'prog_1',
      raw_payload: { type: 'program', id: 'prog_1' }
    });
    const programOutput = nativeTestItem({
      item_type: 'program_output',
      native_id: 'prog_out_1',
      group_id: 'prog_1',
      call_id: 'prog_1',
      pair_id: 'prog_1',
      raw_payload: { type: 'program_output', id: 'prog_out_1', call_id: 'prog_1' }
    });

    expect(deriveProviderNativeGroups(nativeHistory([programCall]))[0]?.state)
      .toBe('active_waiting_tool');
    expect(deriveProviderNativeGroups(nativeHistory([programOutput]))[0]?.state)
      .toBe('active_waiting_model');
    expect(deriveProviderNativeGroups(nativeHistory([programCall, programOutput]))[0]?.state)
      .toBe('active_waiting_model');

    const closed = nativeHistory([programCall, programOutput]);
    if (typeof closed.input !== 'string') {
      closed.input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'input_text', text: 'later answer' }]
      });
    }
    expect(deriveProviderNativeGroups(closed)[0]?.state).toBe('historical_closed');

    const orphaned = nativeHistory([programCall]);
    if (typeof orphaned.input !== 'string') {
      orphaned.input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'input_text', text: 'unrelated later answer' }]
      });
    }
    expect(deriveProviderNativeGroups(orphaned)[0]?.state).toBe('orphaned');
  });

  it('merges PTC items through native IDs, depends_on, callers, and pair IDs', () => {
    const program = nativeTestItem({
      item_type: 'program',
      native_id: 'prog_linked',
      raw_payload: { type: 'program', id: 'prog_linked', code: 'await lookup()' }
    });
    const programOutput = nativeTestItem({
      item_type: 'program_output',
      native_id: 'prog_output_linked',
      group_id: 'output_group',
      pair_id: 'prog_linked',
      depends_on: ['prog_linked'],
      raw_payload: {
        type: 'program_output',
        id: 'prog_output_linked',
        program_id: 'prog_linked',
        output: 'scheduled lookup'
      }
    });
    const functionCall = nativeTestItem({
      item_type: 'function_call',
      native_id: 'fc_linked',
      group_id: 'prog_linked',
      call_id: 'call_linked',
      pair_id: 'call_linked',
      raw_payload: {
        type: 'function_call',
        id: 'fc_linked',
        call_id: 'call_linked',
        name: 'lookup',
        arguments: '{}',
        caller: { type: 'program', id: 'prog_linked' }
      }
    });

    const groups = deriveProviderNativeGroups(
      nativeHistory([program, programOutput, functionCall])
    );
    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({
      state: 'active_waiting_tool',
      active: true
    });
    expect(groups[0]?.items).toEqual([program, programOutput, functionCall]);
  });

  it('rejects active orphaned groups and removes historical orphaned groups atomically', () => {
    const call = nativeTestItem({
      item_type: 'function_call',
      native_id: 'fc_orphan',
      group_id: 'orphan_group',
      call_id: 'call_orphan',
      pair_id: 'call_orphan',
      raw_payload: {
        type: 'function_call',
        id: 'fc_orphan',
        call_id: 'call_orphan',
        name: 'lookup',
        arguments: '{}'
      }
    });
    const mismatchedOutput = nativeTestItem({
      item_type: 'function_call_output',
      native_id: 'fco_orphan',
      group_id: 'orphan_group',
      call_id: 'different_call',
      pair_id: 'different_call',
      raw_payload: {
        type: 'function_call_output',
        id: 'fco_orphan',
        call_id: 'different_call',
        output: 'mismatched'
      }
    });
    const active = nativeHistory([call, mismatchedOutput]);
    expect(deriveProviderNativeGroups(active)[0]).toMatchObject({
      state: 'orphaned',
      active: true
    });
    expect(
      prepareReasoningStateForTargetResult(
        active,
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAIOrigin,
        { historyPolicy: 'native' }
      )
    ).toEqual({
      ok: false,
      error: 'incompatible_active_orphaned_group: group=orphan_group'
    });

    const historical = nativeHistory([call, mismatchedOutput]);
    if (typeof historical.input !== 'string') {
      historical.input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'input_text', text: 'safe later answer' }]
      });
    }
    const prepared = prepareReasoningStateForTargetResult(
      historical,
      OPENAI_RESPONSES_REASONING_FORMAT,
      openAIOrigin,
      { historyPolicy: 'native' }
    );
    expect(prepared.ok).toBe(true);
    if (prepared.ok) {
      expect(JSON.stringify(prepared.value.input)).not.toContain('fc_orphan');
      expect(JSON.stringify(prepared.value.input)).not.toContain('fco_orphan');
      expect(JSON.stringify(prepared.value.input)).toContain('safe later answer');
    }
  });

  it('rejects a missing native Gemini 3 signature while active and drops it once historical', () => {
    const geminiOrigin: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'gemini-endpoint',
      model: 'gemini-3.5-flash',
      credentialScope: 'gemini-account'
    };
    const missingSignature = nativeTestItem({
      item_type: 'function_call',
      item_origin: 'native',
      native_id: 'gemini_missing_signature',
      source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      source_origin: geminiOrigin,
      provider_schema_version: 'gemini-generate-content-v1beta',
      group_id: 'gemini_call_missing',
      call_id: 'gemini_call_missing',
      pair_id: 'gemini_call_missing',
      capture_state: 'partial',
      raw_payload: {
        functionCall: {
          id: 'gemini_call_missing',
          name: 'lookup',
          args: {}
        }
      }
    });
    const active: StandardRequest = {
      model: 'gemini-3.5-flash',
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'gemini_call_missing',
              name: 'lookup',
              input: {},
              native_item: missingSignature
            }
          ],
          native_items: [missingSignature]
        }
      ]
    };
    expect(
      prepareReasoningStateForTargetResult(
        active,
        GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
        geminiOrigin,
        { historyPolicy: 'native' }
      )
    ).toEqual({
      ok: false,
      error: 'incomplete_active_native_group: group=gemini_call_missing'
    });

    const historical = structuredClone(active);
    if (typeof historical.input !== 'string') {
      historical.input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'input_text', text: 'continued safely' }]
      });
    }
    const prepared = prepareReasoningStateForTargetResult(
      historical,
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      geminiOrigin,
      { historyPolicy: 'native' }
    );
    expect(prepared.ok).toBe(true);
    if (prepared.ok) {
      expect(JSON.stringify(prepared.value.input)).not.toContain('gemini_call_missing');
      expect(JSON.stringify(prepared.value.input)).toContain('continued safely');
    }
  });

  it('separates incomplete capture from provider incomplete status', () => {
    const partial = nativeTestItem({
      capture_state: 'partial',
      readable_text: 'readable reasoning'
    });
    expect(
      decideProviderNativeItemReplay(
        partial,
        'historical_closed',
        undefined,
        anthropicOrigin,
        { historyPolicy: 'plaintext', plaintextReasoningSupported: true }
      ).decision
    ).toBe('emit_plaintext');

    const providerIncomplete = nativeTestItem({ provider_status: 'incomplete' });
    expect(
      decideProviderNativeItemReplay(
        providerIncomplete,
        'historical_closed',
        OPENAI_RESPONSES_REASONING_FORMAT,
        openAIOrigin,
        { historyPolicy: 'native' }
      ).decision
    ).toBe('strip_optional');

    const interruptedProgram = nativeTestItem({
      item_type: 'program',
      raw_payload: { type: 'program', id: 'prog_interrupted' },
      native_id: 'prog_interrupted',
      capture_state: 'interrupted'
    });
    expect(
      decideProviderNativeItemReplay(
        interruptedProgram,
        'active_waiting_tool',
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        anthropicOrigin
      ).decision
    ).toBe('reject');
  });

  it('keeps ordinary tool calls convertible when only their optional native projection is incompatible', () => {
    const request: StandardRequest = {
      model: 'claude-test',
      input: [
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'call_plain',
              name: 'lookup',
              input: { key: 'value' },
              native_item: nativeTestItem({
                item_type: 'function_call',
                native_id: 'fc_plain',
                group_id: 'call_plain',
                call_id: 'call_plain',
                pair_id: 'call_plain',
                raw_payload: {
                  type: 'function_call',
                  id: 'fc_plain',
                  call_id: 'call_plain',
                  name: 'lookup',
                  arguments: '{"key":"value"}'
                }
              })
            }
          ]
        }
      ]
    };
    const prepared = prepareReasoningStateForTargetResult(
      request,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      anthropicOrigin,
      { historyPolicy: 'native' }
    );
    expect(prepared.ok).toBe(true);
    if (!prepared.ok || typeof prepared.value.input === 'string') {
      return;
    }
    expect(prepared.value.input[0]?.content).toEqual([
      { type: 'tool_use', id: 'call_plain', name: 'lookup', input: { key: 'value' } }
    ]);
  });

  it('keeps a message whose replayable state exists only in native_items', () => {
    const nativeOnly = nativeTestItem({ native_id: 'native_only_reasoning' });
    const prepared = prepareReasoningStateForTargetResult(
      {
        model: 'gpt-5.6-sol',
        input: [{
          type: 'message',
          role: 'assistant',
          content: [],
          native_items: [nativeOnly]
        }]
      },
      OPENAI_RESPONSES_REASONING_FORMAT,
      openAIOrigin,
      { historyPolicy: 'native' }
    );
    expect(prepared.ok).toBe(true);
    if (!prepared.ok || typeof prepared.value.input === 'string') {
      return;
    }
    expect(prepared.value.input).toHaveLength(1);
    expect(prepared.value.input[0]).toMatchObject({
      content: [],
      native_items: [{ native_id: 'native_only_reasoning' }]
    });
  });

  it('rejects an incompatible active Program group and drops a closed historical group atomically', () => {
    const activeProgram = nativeTestItem({
      item_type: 'program',
      native_id: 'prog_active',
      group_id: 'prog_active',
      call_id: 'prog_active',
      raw_payload: { type: 'program', id: 'prog_active' }
    });
    const active = prepareReasoningStateForTargetResult(
      nativeHistory([activeProgram]),
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      anthropicOrigin
    );
    expect(active).toMatchObject({ ok: false });
    if (!active.ok) {
      expect(active.error).toContain('incompatible_active_native_group');
    }

    const output = nativeTestItem({
      item_type: 'program_output',
      native_id: 'prog_output',
      group_id: 'prog_active',
      call_id: 'prog_active',
      pair_id: 'prog_active',
      raw_payload: { type: 'program_output', id: 'prog_output', call_id: 'prog_active' }
    });
    const historical = nativeHistory([activeProgram, output]);
    if (typeof historical.input !== 'string') {
      historical.input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'input_text', text: 'later safe answer' }]
      });
    }
    const prepared = prepareReasoningStateForTargetResult(
      historical,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      anthropicOrigin
    );
    expect(prepared.ok).toBe(true);
    if (prepared.ok && typeof prepared.value.input !== 'string') {
      expect(JSON.stringify(prepared.value.input)).not.toContain('prog_active');
      expect(JSON.stringify(prepared.value.input)).toContain('later safe answer');
    }
  });
});

describe('compaction and stateful route continuity', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'openai',
    endpoint: 'responses-endpoint-a',
    model: 'gpt-5.6-sol',
    credentialScope: 'responses-account'
  };
  const otherOrigin: ReasoningStateOrigin = {
    ...origin,
    endpoint: 'responses-endpoint-b'
  };

  it('fails closed when opaque compaction is the only history unless strip was explicit', () => {
    const compact = nativeTestItem({
      item_type: 'compaction',
      native_id: 'cmp_1',
      raw_payload: { type: 'compaction', id: 'cmp_1', encrypted_content: 'opaque-window' },
      compaction_mode: 'server_side'
    });
    const request = nativeHistory([compact]);
    const incompatible = prepareReasoningStateForTargetResult(
      request,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      {
        provider: 'anthropic',
        endpoint: 'anthropic-endpoint',
        model: 'claude-test',
        credentialScope: 'anthropic-account'
      }
    );
    expect(incompatible).toEqual({ ok: false, error: 'incompatible_compacted_history' });

    const explicitStrip = prepareReasoningStateForTargetResult(
      request,
      ANTHROPIC_CLAUDE_REASONING_FORMAT,
      {
        provider: 'anthropic',
        endpoint: 'anthropic-endpoint',
        model: 'claude-test',
        credentialScope: 'anthropic-account'
      },
      { historyPolicy: 'strip', explicitStrip: true }
    );
    expect(explicitStrip.ok).toBe(true);
  });

  it('does not mistake a post-compaction delta for complete pre-compaction history', () => {
    const compact = nativeTestItem({
      item_type: 'compaction',
      native_id: 'cmp_followup_only',
      raw_payload: {
        type: 'compaction',
        id: 'cmp_followup_only',
        encrypted_content: 'only-old-history'
      },
      compaction_mode: 'server_side'
    });
    const request = nativeHistory([compact]);
    if (typeof request.input !== 'string') {
      request.input.push({
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'new delta after compaction' }]
      });
    }

    expect(
      prepareReasoningStateForTargetResult(
        request,
        ANTHROPIC_CLAUDE_REASONING_FORMAT,
        {
          provider: 'anthropic',
          endpoint: 'anthropic-endpoint',
          model: 'claude-test',
          credentialScope: 'anthropic-account'
        }
      )
    ).toEqual({ ok: false, error: 'incompatible_compacted_history' });
  });

  it('uses full pre-compaction history on a protocol switch and rejects partial standalone windows', () => {
    const compact = nativeTestItem({
      item_type: 'compaction',
      native_id: 'cmp_2',
      raw_payload: { type: 'compaction', id: 'cmp_2', encrypted_content: 'opaque-window' },
      compaction_mode: 'server_side'
    });
    const withHistory = nativeHistory([compact]);
    if (typeof withHistory.input !== 'string') {
      withHistory.input.unshift({
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'full original history' }]
      });
    }
    const switched = prepareReasoningStateForTargetResult(
      withHistory,
      undefined,
      { provider: 'openai', endpoint: 'chat-endpoint' },
      { historyPolicy: 'plaintext', plaintextReasoningSupported: true }
    );
    expect(switched.ok).toBe(true);
    if (switched.ok) {
      expect(JSON.stringify(switched.value.input)).toContain('full original history');
      expect(JSON.stringify(switched.value.input)).not.toContain('opaque-window');
    }

    const partialStandalone = nativeHistory([
      nativeTestItem({
        item_type: 'compaction',
        native_id: 'cmp_partial',
        raw_payload: { type: 'compaction', id: 'cmp_partial', encrypted_content: 'partial' },
        capture_state: 'partial',
        compaction_mode: 'standalone'
      })
    ]);
    expect(
      prepareReasoningStateForTargetResult(
        partialStandalone,
        OPENAI_RESPONSES_REASONING_FORMAT,
        origin,
        { historyPolicy: 'native' }
      )
    ).toEqual({ ok: false, error: 'incomplete_standalone_compaction_window' });
  });

  it('keeps stateful IDs only on the same service/account route and falls back to complete manual history', () => {
    clearStatefulNativeRouteCacheForTests();
    attachReasoningStateOrigin(emptyResponse('resp_stateful'), origin, OPENAI_RESPONSES_REASONING_FORMAT);

    const sameRoute = prepareReasoningStateForTargetResult(
      {
        model: origin.model!,
        input: 'new delta',
        openai_responses: { operation: 'create', previous_response_id: 'resp_stateful' }
      },
      OPENAI_RESPONSES_REASONING_FORMAT,
      origin
    );
    expect(sameRoute.ok).toBe(true);
    if (sameRoute.ok) {
      expect(sameRoute.value.openai_responses?.previous_response_id).toBe('resp_stateful');
    }

    const stateOnlySwitch = prepareReasoningStateForTargetResult(
      {
        model: origin.model!,
        input: 'new delta',
        openai_responses: { operation: 'create', previous_response_id: 'resp_stateful' }
      },
      OPENAI_RESPONSES_REASONING_FORMAT,
      otherOrigin
    );
    expect(stateOnlySwitch).toEqual({ ok: false, error: 'incompatible_previous_response_route' });

    const manualSwitch = prepareReasoningStateForTargetResult(
      {
        model: origin.model!,
        input: [
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'old question' }] },
          { type: 'message', role: 'assistant', content: [{ type: 'input_text', text: 'old answer' }] },
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'new question' }] }
        ],
        openai_responses: { operation: 'create', previous_response_id: 'resp_stateful' }
      },
      OPENAI_RESPONSES_REASONING_FORMAT,
      otherOrigin
    );
    expect(manualSwitch.ok).toBe(true);
    if (manualSwitch.ok) {
      expect(manualSwitch.value.openai_responses?.previous_response_id).toBeUndefined();
    }
  });

  it('allows Interactions cross-model continuation only with store=true and an unchanged route', () => {
    clearStatefulNativeRouteCacheForTests();
    const interactionsOrigin: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'interactions-endpoint',
      model: 'gemini-3-pro',
      credentialScope: 'gemini-account'
    };
    attachReasoningStateOrigin(
      emptyResponse('int_stateful'),
      interactionsOrigin,
      GEMINI_INTERACTIONS_REASONING_FORMAT
    );
    const crossModel = prepareReasoningStateForTargetResult(
      {
        model: 'gemini-3-flash',
        input: 'new delta',
        gemini_interactions: { previous_interaction_id: 'int_stateful', store: true }
      },
      GEMINI_INTERACTIONS_REASONING_FORMAT,
      { ...interactionsOrigin, model: 'gemini-3-flash' }
    );
    expect(crossModel.ok).toBe(true);

    const notStored = prepareReasoningStateForTargetResult(
      {
        model: 'gemini-3-flash',
        input: 'new delta',
        gemini_interactions: { previous_interaction_id: 'int_stateful', store: false }
      },
      GEMINI_INTERACTIONS_REASONING_FORMAT,
      { ...interactionsOrigin, model: 'gemini-3-flash' }
    );
    expect(notStored).toEqual({ ok: false, error: 'incompatible_previous_interaction_route' });
  });
});

describe('bounded carrier validation', () => {
  const origin: ReasoningStateOrigin = {
    provider: 'gemini',
    endpoint: 'gemini-endpoint',
    model: 'gemini-3-pro',
    credentialScope: 'gemini-account'
  };

  it('rejects malformed, oversized, deeply nested, and non-signature ID carriers with bounded errors', () => {
    expect(
      validateReasoningTransportCarriers(
        `${REASONING_TRANSPORT_ENVELOPE_V3_PREFIX}not-valid-base64-json`,
        1024 * 1024
      )
    ).toMatchObject({ ok: false, status: 400, code: 'malformed_reasoning_carrier' });

    const large = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'x'.repeat(2_000),
      undefined,
      'signature',
      origin
    );
    expect(validateReasoningTransportCarriers(large, 1024))
      .toMatchObject({ ok: false, status: 413, code: 'carrier_too_large' });

    const oversizedV2 = `ccr-reasoning-transport-v2:${Buffer.from(JSON.stringify({
      format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      data: 'x'.repeat(3_000),
      kind: 'signature',
      origin
    }), 'utf8').toString('base64url')}`;
    expect(validateReasoningTransportCarriers(oversizedV2, 10_000))
      .toMatchObject({ ok: false, status: 413, code: 'carrier_payload_too_large' });

    let deepPayload: Record<string, unknown> = { leaf: true };
    for (let depth = 0; depth < 40; depth += 1) {
      deepPayload = { nested: deepPayload };
    }
    const deep = encodeReasoningTransportEnvelope(
      OPENAI_RESPONSES_REASONING_FORMAT,
      'opaque',
      'deep_item',
      'encrypted',
      {
        provider: 'openai',
        endpoint: 'openai-endpoint',
        model: 'gpt-5.6-sol',
        credentialScope: 'openai-account'
      },
      { nativeItem: nativeTestItem({ native_id: 'deep_item', raw_payload: deepPayload }) }
    );
    expect(validateReasoningTransportCarriers(deep, 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'malformed_reasoning_carrier' });
    expect(validateReasoningTransportCarriers(deepPayload, 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'malformed_reasoning_carrier' });

    const fullNativeSignature = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'signature',
      'call_full',
      'signature',
      origin,
      {
        nativeItem: nativeTestItem({
          item_type: 'function_call',
          native_id: 'call_full',
          source_format: GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
          source_origin: origin,
          raw_payload: {
            thoughtSignature: 'signature',
            functionCall: { id: 'call_full', name: 'lookup', args: {} }
          }
        })
      }
    );
    expect(
      validateReasoningTransportCarriers(
        { tool_calls: [{ id: `call_full__thought__${fullNativeSignature}` }] },
        64 * 1024 * 1024
      )
    ).toMatchObject({ ok: false, status: 400, code: 'invalid_tool_call_id_carrier' });

    expect(
      validateReasoningTransportCarriers(
        {
          tool_calls: [{
            id: `call_large__thought__${REASONING_TRANSPORT_ENVELOPE_V3_PREFIX}${'x'.repeat(65 * 1024)}`
          }]
        },
        64 * 1024 * 1024
      )
    ).toMatchObject({ ok: false, status: 413, code: 'tool_call_id_carrier_too_large' });
  });

  it('rejects conflicting IDs, pair directions, cycles, and more than 4096 items', () => {
    const first = encodedNativeTestCarrier(nativeTestItem({ native_id: 'duplicate' }), 'opaque-a');
    const second = encodedNativeTestCarrier(
      nativeTestItem({ native_id: 'duplicate', raw_payload: { type: 'reasoning', encrypted_content: 'opaque-b' } }),
      'opaque-b'
    );
    expect(validateReasoningTransportCarriers([first, second], 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'duplicate_native_item_id' });

    const projectedItem = nativeTestItem({ native_id: 'same_native_projection' });
    const projectedEncrypted = encodedNativeTestCarrier(projectedItem, 'encrypted-field');
    const projectedFingerprint = encodedNativeTestCarrier(projectedItem, 'fingerprint-field');
    expect(
      validateReasoningTransportCarriers(
        [projectedEncrypted, projectedFingerprint],
        64 * 1024 * 1024
      )
    ).toMatchObject({ ok: true, itemCount: 1 });

    const callA = encodedNativeTestCarrier(nativeTestItem({
      item_type: 'function_call',
      native_id: 'call_a',
      pair_id: 'pair_shared',
      raw_payload: { type: 'function_call', id: 'call_a', call_id: 'pair_shared' }
    }));
    const callB = encodedNativeTestCarrier(nativeTestItem({
      item_type: 'function_call',
      native_id: 'call_b',
      pair_id: 'pair_shared',
      raw_payload: { type: 'function_call', id: 'call_b', call_id: 'pair_shared' }
    }));
    expect(validateReasoningTransportCarriers([callA, callB], 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'conflicting_native_pair' });

    const resultA = encodedNativeTestCarrier(nativeTestItem({
      item_type: 'function_call_output',
      native_id: 'result_a',
      pair_id: 'pair_once_closed',
      raw_payload: { type: 'function_call_output', id: 'result_a', call_id: 'pair_once_closed' }
    }));
    const callOnce = encodedNativeTestCarrier(nativeTestItem({
      item_type: 'function_call',
      native_id: 'call_once',
      pair_id: 'pair_once_closed',
      raw_payload: { type: 'function_call', id: 'call_once', call_id: 'pair_once_closed' }
    }));
    const resultB = encodedNativeTestCarrier(nativeTestItem({
      item_type: 'function_call_output',
      native_id: 'result_b',
      pair_id: 'pair_once_closed',
      raw_payload: { type: 'function_call_output', id: 'result_b', call_id: 'pair_once_closed' }
    }));
    expect(validateReasoningTransportCarriers([callOnce, resultA, resultB], 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'conflicting_native_pair' });

    const cycleA = encodedNativeTestCarrier(nativeTestItem({
      native_id: 'cycle_a',
      depends_on: ['cycle_b']
    }));
    const cycleB = encodedNativeTestCarrier(nativeTestItem({
      native_id: 'cycle_b',
      depends_on: ['cycle_a']
    }));
    expect(validateReasoningTransportCarriers([cycleA, cycleB], 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 400, code: 'cyclic_native_dependency' });

    const uniqueItems = Array.from({ length: 4097 }, (_, index) =>
      encodedNativeTestCarrier(nativeTestItem({
        native_id: `native_item_${index}`,
        position: { turn: index, step: 0, item: 0 }
      }))
    );
    expect(validateReasoningTransportCarriers(uniqueItems, 64 * 1024 * 1024))
      .toMatchObject({ ok: false, status: 413, code: 'too_many_native_items' });
  });

  it('does not emit any passthrough carrier frame when a streamed turn exceeds the limit', async () => {
    const upstream = new Response(
      [
        `event: content_block_delta\ndata: ${JSON.stringify({
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'signature_delta', signature: 'x'.repeat(2_000) }
        })}\n\n`,
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
      ].join(''),
      { headers: { 'content-type': 'text/event-stream' } }
    );
    const stream = createReasoningAwarePassthroughSseStream(
      upstream,
      'anthropic_messages',
      {
        provider: 'anthropic',
        endpoint: 'anthropic-endpoint',
        model: 'claude-test',
        credentialScope: 'anthropic-account'
      },
      undefined,
      1024
    );
    let emitted = 0;
    const consume = async () => {
      for await (const _chunk of stream) {
        emitted += 1;
      }
    };
    await expect(consume()).rejects.toThrow('carrier_too_large');
    expect(emitted).toBe(0);
  });
});

function nativeTestItem(overrides: Partial<ProviderNativeItem> = {}): ProviderNativeItem {
  return {
    type: 'provider_native_item',
    item_type: overrides.item_type || 'reasoning',
    native_id: overrides.native_id || 'native_reasoning',
    raw_payload: overrides.raw_payload || {
      type: 'reasoning',
      id: overrides.native_id || 'native_reasoning',
      status: 'completed',
      summary: [],
      encrypted_content: 'opaque'
    },
    provider_schema_version: overrides.provider_schema_version || OPENAI_RESPONSES_REASONING_FORMAT,
    item_origin: overrides.item_origin || 'native',
    source_format: overrides.source_format || OPENAI_RESPONSES_REASONING_FORMAT,
    source_origin: overrides.source_origin || {
      provider: 'openai',
      endpoint: 'openai-endpoint',
      model: 'gpt-5.6-sol',
      credentialScope: 'openai-account'
    },
    position: overrides.position || { turn: 0, step: 0, item: 0 },
    capture_state: overrides.capture_state || 'complete',
    ...overrides
  };
}

function nativeHistory(items: ProviderNativeItem[]): StandardRequest {
  return {
    model: 'gpt-5.6-sol',
    input: [
      {
        type: 'message',
        role: 'assistant',
        content: items
      }
    ]
  };
}

function emptyResponse(id: string): StandardResponse {
  return {
    id,
    object: 'response',
    status: 'completed',
    model: 'test-model',
    output_text: '',
    output: [],
    usage: {}
  };
}

function encodedNativeTestCarrier(item: ProviderNativeItem, data = 'opaque'): string {
  return encodeReasoningTransportEnvelope(
    item.source_format,
    data,
    item.native_id,
    'encrypted',
    item.source_origin,
    { nativeItem: item }
  );
}
