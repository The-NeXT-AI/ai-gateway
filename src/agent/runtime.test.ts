import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GatewayConfig } from '../types';
import { createAgentRuntime } from './runtime';
import type { AgentToolProvider } from './tools';
import type { AgentModelClient } from './types';

vi.setConfig({ testTimeout: 20_000 });

// Create a minimal test config
function createTestConfig(): GatewayConfig {
  return {
    host: '127.0.0.1',
    port: 3000,
    openaiApiKey: 'test-openai-key',
    anthropicApiKey: 'test-anthropic-key',
    geminiApiKey: 'test-gemini-key',
    defaultOpenAIModel: 'gpt-4o-mini',
    defaultAnthropicModel: 'claude-3-5-sonnet-20241022',
    defaultGeminiModel: 'gemini-1.5-flash',
    providers: [],
    defaultTargetProviders: ['openai'],
    defaultTargetProvider: 'openai',
    upstreamTimeoutMs: 30000,
    geminiApiVersion: 'v1beta',
    openaiApiKeyHeader: 'authorization',
    anthropicApiKeyHeader: 'x-api-key',
    geminiApiKeyQueryParam: 'key',
    agent: {
      mcpServers: [],
      storage: {
        type: 'memory'
      }
    },
    mcpGateway: {
      enabled: false,
      endpoint: '/mcp',
      websocket: {
        enabled: false,
        endpoint: '/mcp-ws'
      }
    },
    billingQueue: {
      enabled: false
    },
    billingWebhook: {
      enabled: false
    }
  } as unknown as GatewayConfig;
}

describe('EventDrivenAgentRuntime cancellation', () => {
  const toolProvider: AgentToolProvider = {
    listDefinitions: async () => [],
    has: async () => false,
    execute: async () => ({}),
    close: async () => {}
  };

  const testConfig = createTestConfig();
  const runtimes: ReturnType<typeof createAgentRuntime>[] = [];

  afterEach(async () => {
    await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  });

  it('aborts in-flight model decision by correlation id', async () => {
    const modelClient: AgentModelClient = {
      generate: async ({ signal }) =>
        new Promise((resolve, reject) => {
          const timer = setTimeout(() => resolve({ type: 'reply', text: 'late reply' }), 150);
          const onAbort = () => {
            clearTimeout(timer);
            const error = new Error('aborted');
            error.name = 'AbortError';
            reject(error);
          };

          if (signal?.aborted) {
            onAbort();
            return;
          }

          signal?.addEventListener('abort', onAbort, { once: true });
        })
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'cancel-agent' });
    const created = runtime.createSession({
      sessionId: `cancel-session-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    const accepted = runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: 'hello'
      }
    });

    runtime.abortCorrelation(accepted.correlationId);
    await new Promise((resolve) => setTimeout(resolve, 250));

    const events = runtime.listEvents(created.session.sessionId, 20);
    expect(events.some((event) => event.type === 'AGENT_REPLY')).toBe(false);
  });

  it('hydrates agents and sessions from external endpoint when enabled', async () => {
    const originalFetch = global.fetch;
    let capturedInit: RequestInit | undefined;

    global.fetch = (async (_input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      capturedInit = init;
      return new Response(
        JSON.stringify({
          agents: [
            {
              agentId: 'external-agent-1',
              name: 'external-agent',
              systemPrompt: 'external-prompt',
              allowedTools: [],
              createdAt: '2026-01-01T00:00:00.000Z',
              updatedAt: '2026-01-01T00:00:00.000Z'
            }
          ],
          sessions: [
            {
              state: {
                sessionId: 'external-session-1',
                agentId: 'external-agent-1',
                systemPrompt: 'session-prompt',
                allowedTools: [],
                memoryRefs: [],
                messages: [],
                pendingToolCalls: {},
                lastEventOffset: 0,
                updatedAt: '2026-01-01T00:00:00.000Z'
              },
              events: []
            }
          ]
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    }) as typeof fetch;

    const externalConfig = createTestConfig();
    (externalConfig.agent as unknown as Record<string, unknown>).external = {
      enabled: true,
      transport: 'http',
      endpoint: 'https://external.example.com/agent-state',
      timeoutMs: 1000,
      apiKeyHeader: 'x-agent-external-key',
      apiKey: 'external-secret',
      headers: {
        'x-client-id': 'gateway-test'
      }
    };

    const runtime = createAgentRuntime({
      config: externalConfig,
      toolProvider
    });
    runtimes.push(runtime);

    try {
      await runtime.initialize();

      const agents = runtime.listAgents();
      expect(agents).toHaveLength(1);
      expect(agents[0].agentId).toBe('external-agent-1');

      const session = runtime.getSession('external-session-1');
      expect(session).toBeDefined();
      expect(session?.agentId).toBe('external-agent-1');

      const headers = new Headers(capturedInit?.headers);
      expect(capturedInit?.method).toBe('GET');
      expect(headers.get('x-agent-external-key')).toBe('external-secret');
      expect(headers.get('x-client-id')).toBe('gateway-test');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('hydrates agents and sessions from stdio external source when enabled', async () => {
    const externalConfig = createTestConfig();
    (externalConfig.agent as unknown as Record<string, unknown>).external = {
      enabled: true,
      transport: 'stdio',
      command: process.execPath,
      args: [
        '-e',
        'let input="";process.stdin.on("data",c=>input+=c);process.stdin.on("end",()=>process.stdout.write(JSON.stringify({agents:[{agentId:"stdio-agent-1",name:"stdio-agent",systemPrompt:"stdio-prompt",allowedTools:[],createdAt:"2026-01-01T00:00:00.000Z",updatedAt:"2026-01-01T00:00:00.000Z"}],sessions:[{state:{sessionId:"stdio-session-1",agentId:"stdio-agent-1",systemPrompt:"session-prompt",allowedTools:[],memoryRefs:[],messages:[],pendingToolCalls:{},lastEventOffset:0,updatedAt:"2026-01-01T00:00:00.000Z"},events:[]}] })));'
      ],
      timeoutMs: 1000,
      apiKeyHeader: 'x-agent-external-key',
      headers: {}
    };

    const runtime = createAgentRuntime({
      config: externalConfig,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agents = runtime.listAgents();
    expect(agents).toHaveLength(1);
    expect(agents[0].agentId).toBe('stdio-agent-1');
    expect(runtime.getSession('stdio-session-1')?.agentId).toBe('stdio-agent-1');
  });

  it('syncs external sessions through stdio action payloads', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gateway-agent-external-'));
    const outPath = join(dir, 'requests.jsonl');
    const script = [
      'const fs=require("fs");',
      'let input="";',
      'process.stdin.on("data",c=>input+=c);',
      'process.stdin.on("end",()=>{',
      'fs.appendFileSync(process.env.OUT,input.trim()+"\\n");',
      'const request=JSON.parse(input);',
      'if(request.type==="agent_state_request"){',
      'process.stdout.write(JSON.stringify({agents:[{agentId:"stdio-sync-agent",name:"stdio-sync-agent",systemPrompt:"sync",allowedTools:[],createdAt:"2026-01-01T00:00:00.000Z",updatedAt:"2026-01-01T00:00:00.000Z"}],sessions:[]}));',
      '}else{process.stdout.write(JSON.stringify({ok:true}));}',
      '});'
    ].join('');
    const externalConfig = createTestConfig();
    (externalConfig.agent as unknown as Record<string, unknown>).external = {
      enabled: true,
      transport: 'stdio',
      command: process.execPath,
      args: ['-e', script],
      env: {
        OUT: outPath
      },
      timeoutMs: 1000,
      apiKeyHeader: 'x-agent-external-key',
      headers: {}
    };

    try {
      const runtime = createAgentRuntime({
        config: externalConfig,
        toolProvider
      });
      runtimes.push(runtime);
      await runtime.initialize();

      const created = runtime.createSession({
        sessionId: 'stdio-sync-session',
        agentId: 'stdio-sync-agent'
      });
      expect(created.ok).toBe(true);

      await waitFor(() => {
        try {
          const lines = readFileSync(outPath, 'utf8').trim().split('\n').filter(Boolean);
          return lines.some((line) => JSON.parse(line).type === 'agent_session_upsert');
        } catch {
          return false;
        }
      }, 2000);

      const requests = readFileSync(outPath, 'utf8')
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line));
      expect(requests.map((request) => request.type)).toContain('agent_state_request');
      expect(requests.map((request) => request.type)).toContain('agent_session_upsert');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('migrates legacy allowedTools to code_tool.call when code-tool exposure is active', async () => {
    const codeToolProvider: AgentToolProvider = {
      listDefinitions: async () => [
        {
          name: 'code_tool.search',
          description: 'Search internal tools'
        },
        {
          name: 'code_tool.call',
          description: 'Call internal tools'
        },
        {
          name: 'code_tool.runCode',
          description: 'Run orchestration code'
        }
      ],
      has: async (name) => name.startsWith('code_tool.'),
      execute: async () => ({}),
      close: async () => {}
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      toolProvider: codeToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'legacy-agent',
      allowedTools: ['filesystem.read_file', 'filesystem.write_file']
    });
    const created = runtime.createSession({
      sessionId: `legacy-session-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    // Initially, the session should have the legacy tool names
    const initialSession = runtime.getSession(created.session.sessionId);
    expect(initialSession?.allowedTools).toEqual(['filesystem.read_file', 'filesystem.write_file']);

    // Publish a USER_INPUT event to trigger the normalization
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: 'read file'
      }
    });

    // Wait for the event to be processed and session to be updated
    await waitFor(() => {
      const session = runtime.getSession(created.session.sessionId);
      return session?.allowedTools.includes('code_tool.call') ?? false;
    }, 1000);

    // After processing, the session should have the migrated tool name
    const latest = runtime.getSession(created.session.sessionId);
    expect(latest?.allowedTools).toEqual(['code_tool.call']);
  });

  it('clears session allowedTools when available tool list becomes empty', async () => {
    let providerOnline = true;
    const cliToolProvider: AgentToolProvider = {
      listDefinitions: async () =>
        providerOnline
          ? [
              {
                name: 'filesystem',
                description: 'MCP filesystem CLI wrapper'
              }
            ]
          : [],
      has: async (name) => providerOnline && name === 'filesystem',
      execute: async () => ({}),
      close: async () => {}
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      toolProvider: cliToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'clear-tools-agent',
      allowedTools: ['filesystem']
    });
    const created = runtime.createSession({
      sessionId: `clear-tools-session-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    const initialSession = runtime.getSession(created.session.sessionId);
    expect(initialSession?.allowedTools).toEqual(['filesystem']);

    providerOnline = false;
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: 'trigger tool refresh'
      }
    });

    await waitFor(() => {
      const session = runtime.getSession(created.session.sessionId);
      return Array.isArray(session?.allowedTools) && session.allowedTools.length === 0;
    }, 1000);

    const latest = runtime.getSession(created.session.sessionId);
    expect(latest?.allowedTools).toEqual([]);
  });

  it('does not auto-expand explicit empty allowedTools to all available tools', async () => {
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ output_text: 'ok' }), {
        status: 200,
        headers: {
          'content-type': 'application/json'
        }
      });

    try {
      const cliToolProvider: AgentToolProvider = {
        listDefinitions: async () => [
          {
            name: 'filesystem',
            description: 'MCP filesystem CLI wrapper'
          },
          {
            name: 'browser',
            description: 'MCP browser wrapper'
          }
        ],
        has: async () => false,
        execute: async () => ({}),
        close: async () => {}
      };

      const runtime = createAgentRuntime({
        config: testConfig,
        toolProvider: cliToolProvider
      });
      runtimes.push(runtime);
      await runtime.initialize();

      const agent = runtime.createAgent({
        name: 'explicit-empty-tools-agent',
        allowedTools: []
      });
      const created = runtime.createSession({
        sessionId: `explicit-empty-tools-session-${Date.now()}`,
        agentId: agent.agentId
      });
      expect(created.ok).toBe(true);
      if (!created.ok) {
        return;
      }
      expect(created.session.allowedTools).toEqual([]);

      runtime.publishEvent({
        sessionId: created.session.sessionId,
        type: 'USER_INPUT',
        payload: {
          text: 'hello'
        }
      });

      await waitFor(
        () => runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'AGENT_REPLY'),
        1000
      );

      const latest = runtime.getSession(created.session.sessionId);
      expect(latest?.allowedTools).toEqual([]);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('does not rewrite code_tool.workflow arguments from latest resolve metadata', async () => {
    const codeToolProvider: AgentToolProvider = {
      listDefinitions: async () => [
        { name: 'code_tool.resolve', description: 'Resolve a tool bundle' },
        { name: 'code_tool.workflow', description: 'Run a workflow' }
      ],
      has: async (name) => name === 'code_tool.resolve' || name === 'code_tool.workflow',
      execute: async () => ({ ok: true }),
      close: async () => {}
    };

    let generateCount = 0;
    const modelClient: AgentModelClient = {
      generate: async ({ triggerEvent }) => {
        generateCount += 1;
        if (triggerEvent.type === 'USER_INPUT') {
          return {
            type: 'tool_call',
            toolName: 'code_tool.workflow',
            arguments: {},
            reason: 'LLM requested function call.'
          };
        }
        return {
          type: 'reply',
          text: 'done'
        };
      }
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider: codeToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'workflow-rewrite-agent',
      allowedTools: ['code_tool.resolve', 'code_tool.workflow']
    });
    const created = runtime.createSession({
      sessionId: `workflow-rewrite-session-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    const resolveToolCallId = `resolve-${Date.now()}`;
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_CALL_REQUESTED',
      payload: {
        toolCallId: resolveToolCallId,
        toolName: 'code_tool.resolve',
        arguments: { task: 'Search Baidu for Hangzhou weather' }
      }
    });
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: resolveToolCallId,
        toolName: 'code_tool.resolve',
        status: 'ok',
        result: {
          reasoningSummary: 'Resolve only selected the relevant tool bundle.',
          tsDefinitions:
            '/**\n * Open a URL.\n * Exact tool name: "mcp.local-http-mcp.browser.open_url".\n */\ntype OpenUrlArgs = { url: string; };',
          nextAction: {
            toolName: 'code_tool.workflow',
            arguments: {
              format: 'plan',
              plan: {
                steps: [
                  {
                    type: 'invoke',
                    tool: 'mcp.local-http-mcp.browser.open_url',
                    args: { url: 'https://www.baidu.com/s?wd=%E6%9D%AD%E5%B7%9E%E5%A4%A9%E6%B0%94' }
                  }
                ]
              }
            }
          }
        }
      }
    });

    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some(
        (event) =>
          event.type === 'AGENT_REPLY'
          && (event.payload as { text?: string }).text === 'done'
      );
    }, 2500);

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: { text: '使用百度搜索杭州天气怎么样' }
    });
    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some(
        (event) =>
          event.type === 'TOOL_CALL_REQUESTED'
          && (event.payload as { toolName?: string }).toolName === 'code_tool.workflow'
      );
    }, 2500);

    const workflowRequested = runtime
      .listEvents(created.session.sessionId, 30)
      .filter(
        (event) =>
          event.type === 'TOOL_CALL_REQUESTED'
          && (event.payload as { toolName?: string }).toolName === 'code_tool.workflow'
      )
      .pop();

    expect(workflowRequested).toBeDefined();
    const payload = workflowRequested?.payload as {
      arguments: Record<string, unknown>;
      reason?: string;
    };
    expect(payload.arguments).toEqual({});
    expect(payload.reason || '').not.toContain(
      'gateway reused latest code_tool.resolve result.nextAction.arguments'
    );
  }, 20_000);

  it('does not correct code_tool.invoke arguments from latest resolve metadata', async () => {
    const codeToolProvider: AgentToolProvider = {
      listDefinitions: async () => [
        { name: 'code_tool.resolve', description: 'Resolve a tool bundle' },
        { name: 'code_tool.invoke', description: 'Invoke a resolved tool' }
      ],
      has: async (name) => name === 'code_tool.resolve' || name === 'code_tool.invoke',
      execute: async () => ({ ok: true }),
      close: async () => {}
    };

    let generateCount = 0;
    const modelClient: AgentModelClient = {
      generate: async ({ triggerEvent }) => {
        generateCount += 1;
        if (triggerEvent.type === 'USER_INPUT') {
          return {
            type: 'tool_call',
            toolName: 'code_tool.invoke',
            arguments: {
              tool: 'mcp.local_http_mcp.browser.get_page_context',
              args: {
                tabId: 'tab-123'
              }
            },
            reason: 'LLM requested function call.'
          };
        }
        return {
          type: 'reply',
          text: 'done'
        };
      }
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider: codeToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'invoke-no-correction-agent',
      allowedTools: ['code_tool.resolve', 'code_tool.invoke']
    });
    const created = runtime.createSession({
      sessionId: `invoke-no-correction-session-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    const resolveToolCallId = `resolve-no-correction-${Date.now()}`;
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_CALL_REQUESTED',
      payload: {
        toolCallId: resolveToolCallId,
        toolName: 'code_tool.resolve',
        arguments: { task: 'Get 12306 page context' }
      }
    });
    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: resolveToolCallId,
        toolName: 'code_tool.resolve',
        status: 'ok',
        result: {
          reasoningSummary: 'Resolve only selected the relevant tool bundle.',
          tsDefinitions:
            '/**\n * Get page context.\n * Exact tool name: "mcp.local_http_mcp.browser.get_page_context".\n */\ntype GetPageContextArgs = { session: { sessionId: string; tabId: string; }; };',
          nextAction: {
            toolName: 'code_tool.invoke',
            arguments: {
              tool: 'mcp.local_http_mcp.browser.get_page_context',
              args: {
                session: {
                  sessionId: 'session-123',
                  tabId: 'tab-123'
                }
              }
            }
          }
        }
      }
    });

    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some(
        (event) =>
          event.type === 'AGENT_REPLY'
          && (event.payload as { text?: string }).text === 'done'
      );
    }, 2500);

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: { text: '继续获取页面上下文' }
    });
    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some(
        (event) =>
          event.type === 'TOOL_CALL_REQUESTED'
          && (event.payload as { toolName?: string }).toolName === 'code_tool.invoke'
      );
    }, 2500);

    const invokeRequested = runtime
      .listEvents(created.session.sessionId, 30)
      .filter(
        (event) =>
          event.type === 'TOOL_CALL_REQUESTED'
          && (event.payload as { toolName?: string }).toolName === 'code_tool.invoke'
      )
      .pop();

    expect(invokeRequested).toBeDefined();
    const payload = invokeRequested?.payload as {
      arguments: Record<string, unknown>;
      reason?: string;
    };
    expect(payload.arguments).toEqual({
      tool: 'mcp.local_http_mcp.browser.get_page_context',
      args: {
        tabId: 'tab-123'
      }
    });
    expect(payload.reason || '').not.toContain(
      'gateway corrected model-supplied arguments using latest code_tool.resolve result.nextAction.arguments'
    );
  }, 20_000);

  it('retries transient modelClient failures before succeeding', async () => {
    let attempts = 0;
    const retryConfig = createTestConfig();
    retryConfig.agent = {
      ...(retryConfig.agent as GatewayConfig['agent']),
      runtime: {
        llmRetry: {
          maxAttempts: 3,
          baseDelayMs: 1,
          maxDelayMs: 2,
          backoffMultiplier: 2,
          jitterMs: 0
        }
      }
    } as GatewayConfig['agent'];

    const modelClient: AgentModelClient = {
      generate: async () => {
        attempts += 1;
        if (attempts < 3) {
          const error = new Error('rate limit exceeded') as Error & { status?: number };
          error.status = 429;
          throw error;
        }

        return {
          type: 'reply',
          text: 'retried reply'
        };
      }
    };

    const runtime = createAgentRuntime({
      config: retryConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'model-client-retry-agent' });
    const created = runtime.createSession({
      sessionId: `model-client-retry-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: 'hello'
      }
    });

    await waitFor(
      () => runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'AGENT_REPLY'),
      1000
    );

    const replyEvent = runtime
      .listEvents(created.session.sessionId, 20)
      .find((event) => event.type === 'AGENT_REPLY');

    expect(attempts).toBe(3);
    expect((replyEvent?.payload as { text?: string }).text).toBe('retried reply');
  });

  it('recovers the full reply when streaming stalls after partial chunks', async () => {
    const originalFetch = global.fetch;
    const encoder = new TextEncoder();
    let callCount = 0;

    global.fetch = async (_input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      callCount += 1;
      const body =
        typeof init?.body === 'string'
          ? (JSON.parse(init.body) as Record<string, unknown>)
          : {};

      if (body.stream === true) {
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(
                encoder.encode(
                  'data: {"type":"response.output_text.delta","delta":"我"}\n\n'
                )
              );
            }
          }),
          {
            status: 200,
            headers: {
              'content-type': 'text/event-stream'
            }
          }
        );
      }

      return new Response(JSON.stringify({ output_text: '我有哪些工具' }), {
        status: 200,
        headers: {
          'content-type': 'application/json'
        }
      });
    };

    try {
      const timeoutConfig = createTestConfig();
      timeoutConfig.upstreamTimeoutMs = 50;

      const runtime = createAgentRuntime({
        config: timeoutConfig,
        toolProvider
      });
      runtimes.push(runtime);
      await runtime.initialize();

      const agent = runtime.createAgent({ name: 'stream-recovery-agent' });
      const created = runtime.createSession({
        sessionId: `stream-recovery-${Date.now()}`,
        agentId: agent.agentId
      });
      expect(created.ok).toBe(true);
      if (!created.ok) {
        return;
      }

      runtime.publishEvent({
        sessionId: created.session.sessionId,
        type: 'USER_INPUT',
        payload: {
          text: '你有哪些工具'
        }
      });

      await waitFor(
        () =>
          runtime
            .listEvents(created.session.sessionId, 50)
            .some((event) => event.type === 'AGENT_REPLY'),
        2000
      );

      const events = runtime.listEvents(created.session.sessionId, 50);
      const chunkText = events
        .filter((event) => event.type === 'AGENT_REPLY_CHUNK')
        .map((event) => (event.payload as { text?: string }).text || '')
        .join('');
      const replyEvent = events.find((event) => event.type === 'AGENT_REPLY');

      expect(chunkText).toBe('我有哪些工具');
      expect((replyEvent?.payload as { text?: string }).text).toBe(
        '我有哪些工具'
      );
      expect(callCount).toBe(2);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('continues with the recovered tool call when streaming stalls after partial chunks', async () => {
    const originalFetch = global.fetch;
    const encoder = new TextEncoder();
    let streamCallCount = 0;
    let nonStreamCallCount = 0;

    global.fetch = async (_input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      const body =
        typeof init?.body === 'string'
          ? (JSON.parse(init.body) as Record<string, unknown>)
          : {};

      if (body.stream === true) {
        streamCallCount += 1;
        if (streamCallCount === 1) {
          return new Response(
            new ReadableStream({
              start(controller) {
                controller.enqueue(
                  encoder.encode(
                    'data: {"type":"response.output_text.delta","delta":"我"}\n\n'
                  )
                );
              }
            }),
            {
              status: 200,
              headers: {
                'content-type': 'text/event-stream'
              }
            }
          );
        }

        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(
                encoder.encode(
                  'data: {"type":"response.output_text.delta","delta":"工具执行完成"}\n\n'
                )
              );
              controller.enqueue(
                encoder.encode(
                  'data: {"type":"response.output_text.done","text":"工具执行完成"}\n\n'
                )
              );
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
      }

      nonStreamCallCount += 1;
      return new Response(
        JSON.stringify({
          output: [
            {
              type: 'function_call',
              name: 'filesystem.read_file',
              arguments: '{"path":"/tmp/demo.txt"}'
            }
          ]
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json'
          }
        }
      );
    };

    try {
      const timeoutConfig = createTestConfig();
      timeoutConfig.upstreamTimeoutMs = 50;

      const recoveryToolProvider: AgentToolProvider = {
        listDefinitions: async () => [
          {
            name: 'filesystem.read_file',
            description: 'Read a file'
          }
        ],
        has: async (name) => name === 'filesystem.read_file',
        execute: async () => ({
          ok: true,
          content: 'demo'
        }),
        close: async () => {}
      };

      const runtime = createAgentRuntime({
        config: timeoutConfig,
        toolProvider: recoveryToolProvider
      });
      runtimes.push(runtime);
      await runtime.initialize();

      const agent = runtime.createAgent({
        name: 'stream-tool-recovery-agent',
        allowedTools: ['filesystem.read_file']
      });
      const created = runtime.createSession({
        sessionId: `stream-tool-recovery-${Date.now()}`,
        agentId: agent.agentId
      });
      expect(created.ok).toBe(true);
      if (!created.ok) {
        return;
      }

      runtime.publishEvent({
        sessionId: created.session.sessionId,
        type: 'USER_INPUT',
        payload: {
          text: '读取 /tmp/demo.txt'
        }
      });

      await waitFor(
        () =>
          runtime
            .listEvents(created.session.sessionId, 50)
            .some(
              (event) =>
                event.type === 'AGENT_REPLY' &&
                (event.payload as { text?: string }).text === '工具执行完成'
            ),
        3500
      );

      const events = runtime.listEvents(created.session.sessionId, 50);
      const requestedTool = events.find(
        (event) => event.type === 'TOOL_CALL_REQUESTED'
      );
      const toolResult = events.find((event) => event.type === 'TOOL_RESULT');
      const replyEvent = events.find(
        (event) =>
          event.type === 'AGENT_REPLY' &&
          (event.payload as { text?: string }).text === '工具执行完成'
      );
      const chunkText = events
        .filter((event) => event.type === 'AGENT_REPLY_CHUNK')
        .map((event) => (event.payload as { text?: string }).text || '')
        .join('');
      const chunkPayloads = events
        .filter((event) => event.type === 'AGENT_REPLY_CHUNK')
        .map((event) => event.payload as { metadata?: { kind?: string; actionType?: string } });

      expect((requestedTool?.payload as { toolName?: string }).toolName).toBe(
        'filesystem.read_file'
      );
      expect(toolResult).toBeDefined();
      expect(replyEvent).toBeDefined();
      expect(chunkText).toContain('Action: calling filesystem.read_file.');
      expect(chunkPayloads.some((payload) => payload.metadata?.kind === 'committed_step')).toBe(true);
      expect(chunkPayloads.some((payload) => payload.metadata?.actionType === 'tool_call')).toBe(true);
      expect(streamCallCount).toBe(2);
      expect(nonStreamCallCount).toBe(1);
    } finally {
      global.fetch = originalFetch;
    }
  }, 20_000);

  it('caps streaming idle timeout even when upstream timeout is much larger', async () => {
    const runtime = createAgentRuntime({
      config: {
        ...createTestConfig(),
        upstreamTimeoutMs: 30_000
      },
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    expect((runtime as any).resolveStreamChunkIdleTimeoutMs()).toBe(5_000);
  });

  it('applies session model override over agent default model', async () => {
    const runtime = createAgentRuntime({
      config: testConfig,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'model-agent',
      model: 'openai/gpt-4o-mini'
    });

    const created = runtime.createSession({
      sessionId: `model-session-${Date.now()}`,
      agentId: agent.agentId,
      model: 'anthropic/claude-3-5-sonnet-latest'
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    expect(created.session.model).toBe('anthropic/claude-3-5-sonnet-latest');

    const latest = runtime.getSession(created.session.sessionId);
    expect(latest?.model).toBe('anthropic/claude-3-5-sonnet-latest');
  });

  it('publishes ERROR instead of AGENT_REPLY when TOOL_RESULT follow-up generation fails', async () => {
    const modelClient: AgentModelClient = {
      generate: async ({ triggerEvent }) => {
        if (triggerEvent.type === 'TOOL_RESULT') {
          return {
            type: 'error',
            message: 'LLM generation failed after TOOL_RESULT.'
          };
        }

        return undefined;
      }
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'tool-result-error-agent' });
    const created = runtime.createSession({
      sessionId: `tool-result-error-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: 'call-error-1',
        toolName: 'filesystem',
        status: 'ok',
        result: {
          ok: true
        }
      }
    });

    await waitFor(
      () => runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'ERROR'),
      5000
    );
    const events = runtime.listEvents(created.session.sessionId, 20);
    expect(events.some((event) => event.type === 'AGENT_REPLY')).toBe(false);
    const errorEvent = events.find((event) => event.type === 'ERROR');
    expect(errorEvent).toBeDefined();
    expect((errorEvent?.payload as { message?: string }).message).toContain('LLM generation failed');
  });

  it('emits TOOL_RESULT event without waiting for model generation', async () => {
    let resolveGenerate!: (value: { type: 'reply'; text: string }) => void;
    let generateStarted = false;
    const modelClient: AgentModelClient = {
      generate: async () =>
        new Promise((resolve) => {
          generateStarted = true;
          resolveGenerate = resolve;
        })
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'tool-result-fast-event-agent' });
    const created = runtime.createSession({
      sessionId: `tool-result-fast-event-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: 'call-1',
        toolName: 'filesystem',
        status: 'ok',
        result: {
          ok: true
        }
      }
    });

    await waitFor(
      () =>
        runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'TOOL_RESULT'),
      600
    );
    expect(
      runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'AGENT_REPLY')
    ).toBe(false);

    await waitFor(() => generateStarted, 1000);
    resolveGenerate({
      type: 'reply',
      text: 'done'
    });
    await waitFor(
      () =>
        runtime.listEvents(created.session.sessionId, 20).some((event) => event.type === 'AGENT_REPLY'),
      1000
    );
  });

  it('promotes structured tool failures into TOOL_RESULT error status', async () => {
    let generateCalls = 0;
    const modelClient: AgentModelClient = {
      generate: async () => {
        generateCalls += 1;
        if (generateCalls === 1) {
          return {
            type: 'tool_call',
            toolName: 'code_tool.workflow',
            arguments: {
              format: 'code',
              language: 'ts',
              code: 'return 1;'
            },
            reason: 'Run the workflow.'
          };
        }

        return {
          type: 'reply',
          text: 'failure captured'
        };
      }
    };
    const failingToolProvider: AgentToolProvider = {
      listDefinitions: async () => [
        { name: 'code_tool.workflow', description: 'Run a workflow' }
      ],
      has: async (name) => name === 'code_tool.workflow',
      execute: async () => ({
        ok: false,
        error: {
          code: 'WORKFLOW_FAILED',
          message: 'Input validation failed for browser.extract_data'
        },
        correction: {
          toolName: 'mcp.browser.browser.extract_data',
          suggestedArguments: {
            session: {
              sessionId: 'session-1',
              tabId: 'tab-1'
            }
          }
        }
      }),
      close: async () => {}
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider: failingToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'semantic-tool-failure-agent',
      allowedTools: ['code_tool.workflow']
    });
    const created = runtime.createSession({
      sessionId: `semantic-tool-failure-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: 'run workflow'
      }
    });

    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some((event) => event.type === 'TOOL_RESULT');
    }, 2500);

    const events = runtime.listEvents(created.session.sessionId, 30);
    const toolResult = events.find((event) => event.type === 'TOOL_RESULT');
    expect(toolResult).toBeDefined();
    expect((toolResult?.payload as { status?: string }).status).toBe('error');
    expect((toolResult?.payload as { error?: string }).error).toContain('browser.extract_data');

    const session = runtime.getSession(created.session.sessionId);
    expect(session?.guards.doNotForget.some((entry) => entry.includes('browser.extract_data'))).toBe(true);
    expect(
      session?.transcriptWindow.items.some(
        (item) => item.type === 'failure' && item.text.includes('browser.extract_data')
      )
    ).toBe(true);
  });

  it('stores compact tool messages and transcript items for tool results', async () => {
    const modelClient: AgentModelClient = {
      generate: async () =>
        new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              type: 'reply',
              text: 'done'
            });
          }, 400);
        })
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'tool-result-message-agent' });
    const created = runtime.createSession({
      sessionId: `tool-result-message-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: 'call-structured-1',
        toolName: 'filesystem',
        status: 'ok',
        result: {
          ok: true,
          items: [
            {
              label: 'alpha'
            }
          ]
        }
      }
    });

    await waitFor(() => {
      const session = runtime.getSession(created.session.sessionId);
      return Boolean(session?.messages.some((message) => message.role === 'tool'));
    }, 1000);

    const session = runtime.getSession(created.session.sessionId);
    const toolMessage = session?.messages.find((message) => message.role === 'tool');
    expect(toolMessage).toBeDefined();
    expect(toolMessage?.content).toContain('[filesystem ok]');
    expect(
      session?.transcriptWindow.items.some(
        (item) => item.type === 'tool_result' && item.tool === 'filesystem'
      )
    ).toBe(true);
  });

  it('keeps raw tool-result payloads in the transcript window without rewriting them', async () => {
    const modelClient: AgentModelClient = {
      generate: async () => ({
        type: 'reply',
        text: 'done'
      })
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({ name: 'semantic-workflow-result-agent' });
    const created = runtime.createSession({
      sessionId: `semantic-workflow-result-${Date.now()}`,
      agentId: agent.agentId
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'TOOL_RESULT',
      payload: {
        toolCallId: 'call-workflow-1',
        toolName: 'code_tool.workflow',
        status: 'ok',
        result: {
          ok: true,
          stateStatus: 'continuation',
          observation: {
            summary: 'Observed reusable runtime state from mcp.browser.browser.open_url.',
            toolName: 'mcp.browser.browser.open_url',
            availableContextKeys: ['session', 'sessionId', 'tabId', 'url']
          },
          continuation: {
            stateStatus: 'continuation',
            summary: 'Workflow reached a continuation checkpoint.',
            availableContextKeys: ['session', 'sessionId', 'tabId', 'url'],
            reusableContext: {
              session: {
                sessionId: 'session-browser-1',
                tabId: 'tab-browser-1'
              },
              sessionId: 'session-browser-1',
              tabId: 'tab-browser-1',
              url: 'https://www.12306.cn/index/'
            },
            recentSteps: [
              {
                toolName: 'mcp.browser.browser.open_url',
                summary: 'Opened 12306 homepage.'
              }
            ],
            dataSummary: 'Keys: session, url.'
          },
          data: {
            session: {
              sessionId: 'session-browser-1',
              tabId: 'tab-browser-1'
            },
            url: 'https://www.12306.cn/index/',
            hugeRawPayload: 'x'.repeat(4000)
          }
        }
      }
    });

    await waitFor(() => {
      const session = runtime.getSession(created.session.sessionId);
      return Boolean(session?.messages.some((message) => message.role === 'tool'));
    }, 1200);

    const session = runtime.getSession(created.session.sessionId);
    const toolMessage = session?.messages.find((message) => message.role === 'tool');
    const transcriptResult = session?.transcriptWindow.items.find(
      (item) => item.type === 'tool_result' && item.tool === 'code_tool.workflow'
    );

    expect(toolMessage?.content).toContain('[code_tool.workflow ok]');
    expect(transcriptResult).toBeDefined();
    expect(
      transcriptResult && transcriptResult.type === 'tool_result'
        ? (transcriptResult.raw || transcriptResult.output).includes('session-browser-1')
        : false
    ).toBe(true);
    expect(
      transcriptResult && transcriptResult.type === 'tool_result'
        ? (transcriptResult.raw || transcriptResult.output).includes('hugeRawPayload')
        : false
    ).toBe(true);
    expect(
      transcriptResult && transcriptResult.type === 'tool_result'
        ? (transcriptResult.raw || transcriptResult.output).length > 4000
        : false
    ).toBe(true);
  });

  it('blocks a repeated identical tool call after repeated failures add a guard', async () => {
    const repeatedToolProvider: AgentToolProvider = {
      listDefinitions: async () => [
        {
          name: 'filesystem.read_file',
          description: 'Read a file'
        }
      ],
      has: async (name) => name === 'filesystem.read_file',
      execute: async () => {
        throw new Error('permission denied');
      },
      close: async () => {}
    };

    const modelClient: AgentModelClient = {
      generate: async ({ triggerEvent }) => {
        if (triggerEvent.type === 'USER_INPUT') {
          return {
            type: 'tool_call',
            toolName: 'filesystem.read_file',
            arguments: { path: '/tmp/demo.txt' },
            reason: 'Retry the same file read.'
          };
        }
        return {
          type: 'reply',
          text: 'observed failure'
        };
      }
    };

    const runtime = createAgentRuntime({
      config: testConfig,
      modelClient,
      toolProvider: repeatedToolProvider
    });
    runtimes.push(runtime);
    await runtime.initialize();

    const agent = runtime.createAgent({
      name: 'guarded-repeat-agent',
      allowedTools: ['filesystem.read_file']
    });
    const created = runtime.createSession({
      sessionId: `guarded-repeat-${Date.now()}`,
      agentId: agent.agentId,
      prompt: '第一次读取文件'
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      return;
    }

    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 30);
      return events.some((event) => event.type === 'TOOL_RESULT');
    }, 2500);

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: '第二次读取文件'
      }
    });

    await waitFor(() => {
      const session = runtime.getSession(created.session.sessionId);
      return Boolean(session?.guards.doNotRepeat.some((entry) => entry.includes('filesystem.read_file')));
    }, 2500);

    runtime.publishEvent({
      sessionId: created.session.sessionId,
      type: 'USER_INPUT',
      payload: {
        text: '第三次读取文件'
      }
    });

    await waitFor(() => {
      const events = runtime.listEvents(created.session.sessionId, 60);
      return events.some(
        (event) =>
          event.type === 'ERROR'
          && String((event.payload as { message?: string }).message || '').includes('Rejected guarded action')
      );
    }, 2500);

    const session = runtime.getSession(created.session.sessionId);
    expect(session?.guards.doNotRepeat.some((entry) => entry.includes('filesystem.read_file'))).toBe(true);
  }, 20_000);
});

async function waitFor(check: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + Math.max(timeoutMs, 10_000);
  while (Date.now() < deadline) {
    if (check()) {
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 20));
  }

  throw new Error('waitFor timeout');
}
