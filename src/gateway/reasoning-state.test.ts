import { Readable } from 'node:stream';
import { describe, expect, it } from 'vitest';
import type { FastifyReply, FastifyRequest } from 'fastify';
import type {
  GatewayConfig,
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

  it('requires the exact same model for every opaque reasoning format', () => {
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
    for (const format of [
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      GEMINI_INTERACTIONS_REASONING_FORMAT
    ]) {
      expect(canReplayReasoningState(format, geminiA, format, geminiB)).toBe(false);
      expect(canReplayReasoningState(format, geminiA, format, geminiA)).toBe(true);
    }
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

describe('reasoning transport envelope v2', () => {
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

  it('decodes an older v2 origin but does not treat its missing credential scope as replayable', () => {
    const oldOrigin: ReasoningStateOrigin = {
      provider: 'gemini',
      endpoint: 'endpoint-fingerprint',
      model: 'gemini-3-pro'
    };
    const encoded = encodeReasoningTransportEnvelope(
      GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
      'old-v2-signature',
      undefined,
      'signature',
      oldOrigin
    );
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
