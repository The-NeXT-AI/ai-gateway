import { describe, expect, it } from 'vitest';
import type { GatewayConfig } from '../types';
import { buildProviderRequest, parseProviderStreamChunks } from './provider-adapter';

function indexOfSubarray(haystack: Uint8Array, needle: Uint8Array): number {
  for (let i = 0; i <= haystack.length - needle.length; i += 1) {
    let matched = true;
    for (let j = 0; j < needle.length; j += 1) {
      if (haystack[i + j] !== needle[j]) {
        matched = false;
        break;
      }
    }

    if (matched) {
      return i;
    }
  }

  return -1;
}

describe('parseProviderStreamChunks', () => {
  it('preserves multibyte UTF-8 text when SSE frames split mid-character', async () => {
    const encoder = new TextEncoder();
    const ssePayload =
      'data: {"type":"response.output_text.delta","delta":"工具列表"}\n\n' +
      'data: {"type":"response.completed"}\n\n';
    const fullBytes = encoder.encode(ssePayload);
    const splitTarget = encoder.encode('工具');
    const splitStart = indexOfSubarray(fullBytes, splitTarget);

    expect(splitStart).toBeGreaterThanOrEqual(0);

    const chunk1 = fullBytes.slice(0, splitStart + 1);
    const chunk2 = fullBytes.slice(splitStart + 1, splitStart + 4);
    const chunk3 = fullBytes.slice(splitStart + 4);

    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(chunk1);
          controller.enqueue(chunk2);
          controller.enqueue(chunk3);
          controller.close();
        }
      }),
      {
        status: 200,
        headers: {
          'content-type': 'text/event-stream'
        }
      }
    );

    const chunks = [];
    for await (const chunk of parseProviderStreamChunks('openai', response)) {
      chunks.push(chunk);
    }

    const text = chunks
      .filter((chunk) => chunk.type === 'text_delta')
      .map((chunk) => chunk.text)
      .join('');
    const doneChunk = chunks.find((chunk) => chunk.type === 'done');

    expect(text).toBe('工具列表');
    expect(doneChunk).toEqual({
      type: 'done',
      text: '工具列表'
    });
  });

  it('accumulates streamed openai chat tool-call arguments until finish_reason tool_calls', async () => {
    const frames = [
      {
        id: 'chatcmpl_tool_1',
        object: 'chat.completion.chunk',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_weather',
                  type: 'function',
                  function: {
                    name: 'code_tool.workflow',
                    arguments: '{"format":"pl'
                  }
                }
              ]
            }
          }
        ]
      },
      {
        id: 'chatcmpl_tool_1',
        object: 'chat.completion.chunk',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_weather',
                  type: 'function',
                  function: {
                    arguments: 'an","plan":{"steps":[1]}}'
                  }
                }
              ]
            }
          }
        ]
      },
      {
        id: 'chatcmpl_tool_1',
        object: 'chat.completion.chunk',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'tool_calls'
          }
        ]
      }
    ];
    const ssePayload = frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('');

    const response = new Response(
      ssePayload,
      {
        status: 200,
        headers: {
          'content-type': 'text/event-stream'
        }
      }
    );

    const chunks = [];
    for await (const chunk of parseProviderStreamChunks('openai', response)) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual([
      {
        type: 'tool_call',
        toolName: 'code_tool.workflow',
        arguments: {
          format: 'plan',
          plan: {
            steps: [1]
          }
        },
        reason: 'LLM requested function call.'
      }
    ]);
  });
});

describe('buildProviderRequest', () => {
  it('propagates gateway identity headers for internal agent model calls', () => {
    const config = {
      openaiApiKey: 'provider-openai-key',
      defaultOpenAIModel: 'gpt-4o-mini',
      auth: {
        enabled: true,
        mode: 'http_introspection',
        required: true,
        trustedCidrs: [],
        identityHeaders: {
          userId: 'x-auth-user-id',
          tenantId: 'x-auth-tenant-id',
          subject: 'x-auth-sub',
          organizationId: 'x-auth-organization-id',
          plan: 'x-auth-plan',
          apiKeyId: 'x-auth-api-key-id'
        },
        signature: {
          enabled: false,
          header: 'x-auth-signature',
          timestampHeader: 'x-auth-ts',
          secretEnv: 'AUTH_HEADER_SIGNING_SECRET',
          maxSkewSec: 120
        },
        introspection: {
          endpoint: 'http://auth.local/introspect',
          timeoutMs: 3000,
          tokenHeader: 'authorization',
          tokenBearerOnly: true,
          requestTokenField: 'token',
          credentialHeader: 'x-gateway-auth',
          credentialEnv: 'AUTH_INTROSPECTION_SHARED_SECRET',
          responseMap: {
            active: 'active',
            userId: 'userId',
            tenantId: 'tenantId',
            subject: 'sub',
            organizationId: 'organizationId',
            plan: 'plan',
            apiKeyId: 'apiKeyId'
          }
        }
      }
    } as unknown as GatewayConfig;

    const previousSecret = process.env.AUTH_INTROSPECTION_SHARED_SECRET;
    process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'internal-secret';

    try {
      const prepared = buildProviderRequest(
        { provider: 'openai' },
        'system',
        'hello',
        [],
        undefined,
        config,
        'http://127.0.0.1:3000',
        {
          source: 'http_introspection',
          billingSubjectKey: 'tenant-a:user-1',
          userId: 'user-1',
          tenantId: 'tenant-a',
          subject: 'user-1',
          organizationId: 'org-1',
          plan: 'project',
          apiKeyId: 'key-1'
        },
        {
          agentId: '92d91b78-639f-4d44-b87f-b9b0ca6f38f4',
          sessionId: 'session-456',
          runId: 'corr-123',
          stepId: 'event-456',
          workflow: 'event_driven_agent_runtime',
          promptVersion: 'router-v12'
        }
      );

      expect(prepared.ok).toBe(true);
      if (!prepared.ok) {
        return;
      }

      expect(prepared.request.headers).toMatchObject({
        authorization: 'Bearer provider-openai-key',
        'x-gateway-agent-internal': '1',
        'x-gateway-auth': 'internal-secret',
        'x-auth-user-id': 'user-1',
        'x-auth-tenant-id': 'tenant-a',
        'x-auth-sub': 'user-1',
        'x-auth-organization-id': 'org-1',
        'x-auth-plan': 'project',
        'x-auth-api-key-id': 'key-1',
        'x-agent-id': '92d91b78-639f-4d44-b87f-b9b0ca6f38f4',
        'x-agent-session-id': 'session-456',
        'x-agent-run-id': 'corr-123',
        'x-agent-step-id': 'event-456',
        'x-agent-workflow': 'event_driven_agent_runtime',
        'x-agent-prompt-version': 'router-v12',
        'x-target-provider': 'openai'
      });
    } finally {
      if (previousSecret === undefined) {
        delete process.env.AUTH_INTROSPECTION_SHARED_SECRET;
      } else {
        process.env.AUTH_INTROSPECTION_SHARED_SECRET = previousSecret;
      }
    }
  });
});
