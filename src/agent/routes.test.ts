import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Fastify from 'fastify';
import type { GatewayConfig } from '../types';
import { createAgentRuntime, registerAgentRoutes } from './index';
import type { AgentToolDefinition, AgentToolProvider } from './index';
import type { FastifyInstance } from 'fastify';
import type { EventDrivenAgentRuntime } from './runtime';

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

function createTestAuthConfig(): GatewayConfig['auth'] {
  return {
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
      tokenHeader: 'x-api-key',
      tokenBearerOnly: false,
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
  } as GatewayConfig['auth'];
}

describe('Agent Routes', () => {
  let fastify: FastifyInstance;
  let runtime: EventDrivenAgentRuntime;
  const testConfig = createTestConfig();
  const previousIntrospectionSecret = process.env.AUTH_INTROSPECTION_SHARED_SECRET;

  beforeEach(async () => {
    fastify = Fastify({ logger: false });
    runtime = createAgentRuntime({
      config: testConfig,
      logger: fastify.log
    });
    await runtime.initialize();
    runtime.createAgent({
      name: '测试基础Agent'
    });
    registerAgentRoutes(fastify, runtime);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    if (previousIntrospectionSecret === undefined) {
      delete process.env.AUTH_INTROSPECTION_SHARED_SECRET;
    } else {
      process.env.AUTH_INTROSPECTION_SHARED_SECRET = previousIntrospectionSecret;
    }
    await fastify.close();
    await runtime.close();
  });

  describe('GET /agent/tools', () => {
    it('应该返回可用工具列表', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/agent/tools'
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('tools');
      expect(Array.isArray(body.tools)).toBe(true);
    });
  });

  describe('POST /agent/agents', () => {
    it('应该成功创建agent', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '测试Agent',
          description: '这是一个测试Agent',
          systemPrompt: '你是一个测试助手',
          tools: ['tool1', 'tool2']
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('agent');
      expect(body.agent.name).toBe('测试Agent');
      expect(body.agent.description).toBe('这是一个测试Agent');
      expect(body.agent.systemPrompt).toBe('你是一个测试助手');
      expect(body.agent.allowedTools).toEqual(['tool1', 'tool2']);
      expect(body.agent).toHaveProperty('agentId');
      expect(body.agent).toHaveProperty('createdAt');
      expect(body.agent).toHaveProperty('updatedAt');
    });

    it('应该支持创建agent时指定model(provider/model)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '带模型Agent',
          model: 'openai/gpt-4o-mini',
          tools: ['tool1']
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.agent.model).toBe('openai/gpt-4o-mini');
    });

    it('应该拒绝无效model格式的agent创建请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '无效模型Agent',
          model: 'gpt-4o-mini'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('provider/model');
    });

    it('应该拒绝没有name的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          description: '没有name的Agent'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('name');
    });

    it('应该拒绝空name的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '   '
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('name');
    });

    it('应该拒绝无效的tools字段', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '测试Agent',
          tools: 'not-an-array'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('tools');
    });

    it('应该支持allowedTools字段(旧字段名)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '测试Agent',
          allowedTools: ['tool1']
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.agent.allowedTools).toEqual(['tool1']);
    });

    it('未显式传tools时应该自动注册MCP工具', async () => {
      const testFastify = Fastify({ logger: false });
      const mockTools: AgentToolDefinition[] = [
        {
          name: 'filesystem.read_file',
          description: 'read'
        },
        {
          name: 'filesystem.write_file',
          description: 'write'
        }
      ];
      const mockToolProvider: AgentToolProvider = {
        listDefinitions: async () => mockTools,
        has: async () => true,
        execute: async () => ({}),
        close: async () => {}
      };

      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log,
        toolProvider: mockToolProvider
      });

      await testRuntime.initialize();
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: '自动工具Agent'
          }
        });

        expect(response.statusCode).toBe(201);
        const body = JSON.parse(response.body);
        expect(body.agent.allowedTools).toEqual(['filesystem.read_file', 'filesystem.write_file']);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('显式传tools时不应自动注入MCP工具', async () => {
      const testFastify = Fastify({ logger: false });
      const mockTools: AgentToolDefinition[] = [
        {
          name: 'filesystem.read_file',
          description: 'read'
        },
        {
          name: 'filesystem.write_file',
          description: 'write'
        }
      ];
      const mockToolProvider: AgentToolProvider = {
        listDefinitions: async () => mockTools,
        has: async () => true,
        execute: async () => ({}),
        close: async () => {}
      };

      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log,
        toolProvider: mockToolProvider
      });

      await testRuntime.initialize();
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: '手动工具Agent',
            tools: ['custom.echo']
          }
        });

        expect(response.statusCode).toBe(201);
        const body = JSON.parse(response.body);
        expect(body.agent.allowedTools).toEqual(['custom.echo']);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('显式传allowedTools时不应自动注入MCP工具', async () => {
      const testFastify = Fastify({ logger: false });
      const mockTools: AgentToolDefinition[] = [
        {
          name: 'filesystem.read_file',
          description: 'read'
        }
      ];
      const mockToolProvider: AgentToolProvider = {
        listDefinitions: async () => mockTools,
        has: async () => true,
        execute: async () => ({}),
        close: async () => {}
      };

      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log,
        toolProvider: mockToolProvider
      });

      await testRuntime.initialize();
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: '手动工具Agent-legacy',
            allowedTools: ['legacy.tool']
          }
        });

        expect(response.statusCode).toBe(201);
        const body = JSON.parse(response.body);
        expect(body.agent.allowedTools).toEqual(['legacy.tool']);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('显式传空tools时不应自动注入MCP工具', async () => {
      const testFastify = Fastify({ logger: false });
      const mockTools: AgentToolDefinition[] = [
        {
          name: 'filesystem.read_file',
          description: 'read'
        }
      ];
      const mockToolProvider: AgentToolProvider = {
        listDefinitions: async () => mockTools,
        has: async () => true,
        execute: async () => ({}),
        close: async () => {}
      };

      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log,
        toolProvider: mockToolProvider
      });

      await testRuntime.initialize();
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: '空工具Agent',
            tools: []
          }
        });

        expect(response.statusCode).toBe(201);
        const body = JSON.parse(response.body);
        expect(body.agent.allowedTools).toEqual([]);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('应该拒绝非对象类型的请求体', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: 'not-an-object',
        headers: {
          'content-type': 'text/plain'
        }
      });

      // Fastify在content-type不匹配时返回415或400
      expect([400, 415]).toContain(response.statusCode);
    });
  });

  describe('GET /agent/agents/:agentId', () => {
    it('应该返回已创建的agent', async () => {
      // 先创建agent
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '测试Agent'
        }
      });
      const createBody = JSON.parse(createResponse.body);
      const agentId = createBody.agent.agentId;

      // 获取agent
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/agents/${agentId}`
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.agent.agentId).toBe(agentId);
      expect(body.agent.name).toBe('测试Agent');
    });

    it('应该为不存在的agent返回404', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/agent/agents/non-existent-id'
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Agent not found');
    });
  });

  describe('POST /agent/sessions', () => {
    it('应该成功创建session(无参数)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('sessionId');
      expect(body).toHaveProperty('agentId');
      expect(body).toHaveProperty('session');
      expect(body).toHaveProperty('events');
      expect(Array.isArray(body.events)).toBe(true);
    });

    it('创建带prompt的session时应透传gateway身份到消息metadata', async () => {
      const testFastify = Fastify({ logger: false });
      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log
      });
      await testRuntime.initialize();
      testRuntime.createAgent({
        name: '测试基础Agent'
      });
      testFastify.addHook('preHandler', async (request) => {
        request.gatewayIdentity = {
          source: 'http_introspection',
          billingSubjectKey: 'tenant-a:user-1',
          userId: 'user-1',
          tenantId: 'tenant-a',
          organizationId: 'org-1',
          apiKeyId: 'key-1'
        };
      });
      registerAgentRoutes(testFastify, testRuntime);

      const response = await testFastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          prompt: 'hello',
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.initialEvent.payload.metadata.gatewayRequestIdentity).toMatchObject({
        source: 'http_introspection',
        billingSubjectKey: 'tenant-a:user-1',
        userId: 'user-1',
        tenantId: 'tenant-a',
        organizationId: 'org-1',
        apiKeyId: 'key-1'
      });

      await testFastify.close();
      await testRuntime.close();
    });

    it('启用auth时应拒绝未携带API key的agent会话请求', async () => {
      const testFastify = Fastify({ logger: false });
      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log
      });
      await testRuntime.initialize();
      testRuntime.createAgent({
        name: '测试基础Agent'
      });
      registerAgentRoutes(testFastify, testRuntime, {
        authConfig: createTestAuthConfig()
      });

      const response = await testFastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          prompt: 'hello',
          stream: false
        }
      });

      expect(response.statusCode).toBe(401);

      await testFastify.close();
      await testRuntime.close();
    });

    it('启用auth时应将introspection身份写入session metadata', async () => {
      process.env.AUTH_INTROSPECTION_SHARED_SECRET = 'introspection-secret';
      const fetchMock = vi.fn(async (_input: Parameters<typeof fetch>[0]) => {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            active: true,
            userId: 'user-1',
            tenantId: 'tenant-a',
            sub: 'user-1',
            organizationId: 'org-1',
            plan: 'project',
            apiKeyId: 'key-1'
          })
        } as Response;
      });
      vi.stubGlobal('fetch', fetchMock);

      const testFastify = Fastify({ logger: false });
      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log
      });
      await testRuntime.initialize();
      testRuntime.createAgent({
        name: '测试基础Agent'
      });
      registerAgentRoutes(testFastify, testRuntime, {
        authConfig: createTestAuthConfig()
      });

      const response = await testFastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        headers: {
          'x-api-key': 'gateway-user-key'
        },
        payload: {
          prompt: 'hello',
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.initialEvent.payload.metadata.gatewayRequestIdentity).toMatchObject({
        source: 'http_introspection',
        billingSubjectKey: 'tenant-a:user-1',
        userId: 'user-1',
        tenantId: 'tenant-a',
        subject: 'user-1',
        organizationId: 'org-1',
        plan: 'project',
        apiKeyId: 'key-1'
      });
      expect(fetchMock).toHaveBeenCalled();
      expect(fetchMock.mock.calls.some((call) => call[0] === 'http://auth.local/introspect')).toBe(true);

      await testFastify.close();
      await testRuntime.close();
    });

    it('应按gateway身份隔离agent访问', async () => {
      const testFastify = Fastify({ logger: false });
      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log
      });
      await testRuntime.initialize();
      testFastify.addHook('preHandler', async (request) => {
        const userId = typeof request.headers['x-test-user'] === 'string' ? request.headers['x-test-user'] : 'user-a';
        request.gatewayIdentity = {
          source: 'trusted_header',
          billingSubjectKey: `tenant-a:${userId}`,
          tenantId: 'tenant-a',
          userId
        };
      });
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const createA = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          headers: {
            'x-test-user': 'user-a'
          },
          payload: {
            name: 'agent-a',
            allowedTools: []
          }
        });
        const createB = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          headers: {
            'x-test-user': 'user-b'
          },
          payload: {
            name: 'agent-b',
            allowedTools: []
          }
        });
        expect(createA.statusCode).toBe(201);
        expect(createB.statusCode).toBe(201);
        const agentA = JSON.parse(createA.body).agent;
        const agentB = JSON.parse(createB.body).agent;

        const listA = await testFastify.inject({
          method: 'GET',
          url: '/agent/agents',
          headers: {
            'x-test-user': 'user-a'
          }
        });
        expect(listA.statusCode).toBe(200);
        expect(JSON.parse(listA.body).agents.map((agent: { agentId: string }) => agent.agentId)).toEqual([
          agentA.agentId
        ]);

        const getAFromB = await testFastify.inject({
          method: 'GET',
          url: `/agent/agents/${agentA.agentId}`,
          headers: {
            'x-test-user': 'user-b'
          }
        });
        expect(getAFromB.statusCode).toBe(404);

        const updateAFromB = await testFastify.inject({
          method: 'PUT',
          url: `/agent/agents/${agentA.agentId}`,
          headers: {
            'x-test-user': 'user-b'
          },
          payload: {
            name: 'stolen-agent'
          }
        });
        expect(updateAFromB.statusCode).toBe(404);

        const getBFromB = await testFastify.inject({
          method: 'GET',
          url: `/agent/agents/${agentB.agentId}`,
          headers: {
            'x-test-user': 'user-b'
          }
        });
        expect(getBFromB.statusCode).toBe(200);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('应按gateway身份隔离session访问和事件写入', async () => {
      const testFastify = Fastify({ logger: false });
      const testRuntime = createAgentRuntime({
        config: testConfig,
        logger: testFastify.log
      });
      await testRuntime.initialize();
      testFastify.addHook('preHandler', async (request) => {
        const userId = typeof request.headers['x-test-user'] === 'string' ? request.headers['x-test-user'] : 'user-a';
        request.gatewayIdentity = {
          source: 'trusted_header',
          billingSubjectKey: `tenant-a:${userId}`,
          tenantId: 'tenant-a',
          userId
        };
      });
      registerAgentRoutes(testFastify, testRuntime);

      try {
        const createAgent = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          headers: {
            'x-test-user': 'user-a'
          },
          payload: {
            name: 'agent-a',
            allowedTools: []
          }
        });
        expect(createAgent.statusCode).toBe(201);
        const agentId = JSON.parse(createAgent.body).agent.agentId;

        const createSession = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          headers: {
            'x-test-user': 'user-a'
          },
          payload: {
            agentId,
            stream: false
          }
        });
        expect(createSession.statusCode).toBe(201);
        const sessionId = JSON.parse(createSession.body).sessionId;

        const listFromB = await testFastify.inject({
          method: 'GET',
          url: '/agent/sessions',
          headers: {
            'x-test-user': 'user-b'
          }
        });
        expect(listFromB.statusCode).toBe(200);
        expect(JSON.parse(listFromB.body).sessions).toEqual([]);

        const getFromB = await testFastify.inject({
          method: 'GET',
          url: `/agent/sessions/${sessionId}`,
          headers: {
            'x-test-user': 'user-b'
          }
        });
        expect(getFromB.statusCode).toBe(404);

        const inputFromB = await testFastify.inject({
          method: 'POST',
          url: `/agent/sessions/${sessionId}/input`,
          headers: {
            'x-test-user': 'user-b'
          },
          payload: {
            text: 'cross tenant input'
          }
        });
        expect(inputFromB.statusCode).toBe(404);

        const getFromA = await testFastify.inject({
          method: 'GET',
          url: `/agent/sessions/${sessionId}`,
          headers: {
            'x-test-user': 'user-a'
          }
        });
        expect(getFromA.statusCode).toBe(200);
      } finally {
        await testFastify.close();
        await testRuntime.close();
      }
    });

    it('应该成功创建带prompt的session', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          prompt: '你好,请介绍一下自己',
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('sessionId');
      expect(body).toHaveProperty('initialEvent');
      expect(body.initialEvent.type).toBe('USER_INPUT');
      expect(body.initialEvent.payload.text).toBe('你好,请介绍一下自己');
    });

    it('应该成功创建指定agentId的session', async () => {
      // 先创建agent
      const createAgentResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '自定义Agent'
        }
      });
      const agentBody = JSON.parse(createAgentResponse.body);
      const agentId = agentBody.agent.agentId;

      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          agentId,
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.agentId).toBe(agentId);
    });

    it('应该成功创建自定义sessionId的session', async () => {
      const customSessionId = `custom-session-${Date.now()}`;
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          sessionId: customSessionId,
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.sessionId).toBe(customSessionId);
    });

    it('应该拒绝重复的sessionId', async () => {
      const customSessionId = 'duplicate-session-id';

      // 第一次创建
      await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          sessionId: customSessionId,
          stream: false
        }
      });

      // 第二次创建应该失败
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          sessionId: customSessionId,
          stream: false
        }
      });

      expect(response.statusCode).toBe(409);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session already exists');
    });

    it('应该拒绝不存在的agentId', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          agentId: 'non-existent-agent-id',
          stream: false
        }
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Agent not found');
    });

    it('应该成功创建带metadata的session', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          metadata: {
            userId: 'user123',
            environment: 'test'
          },
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
    });

    it('应该拒绝无效的metadata类型', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          metadata: 'not-an-object',
          stream: false
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('metadata');
    });

    it('应该成功创建带tools的session', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          tools: ['tool1', 'tool2'],
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.session.allowedTools).toEqual(['tool1', 'tool2']);
    });

    it('应该成功创建带memoryRefs的session', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          memoryRefs: ['memory1', 'memory2'],
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.session.memoryRefs).toEqual(['memory1', 'memory2']);
    });

    it('创建session时应支持model覆盖agent默认model', async () => {
      const modelAgent = runtime.createAgent({
        name: '模型覆盖Agent',
        model: 'openai/gpt-4o-mini',
        allowedTools: []
      });

      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          agentId: modelAgent.agentId,
          model: 'anthropic/claude-3-5-sonnet-latest',
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      expect(body.session.model).toBe('anthropic/claude-3-5-sonnet-latest');
    });

    it('应该拒绝无效model格式的session创建请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          model: 'gpt-4o-mini',
          stream: false
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('provider/model');
    });

    it.skip('应该支持流式响应(stream=true)', async () => {
      // SSE流测试会超时,因为它是长连接
      // 跳过此测试,在实际应用中应该手动测试
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: true
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
    });

    it.skip('应该根据Accept header自动启用流式响应', async () => {
      // SSE流测试会超时,跳过
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: true
        },
        headers: {
          Accept: 'text/event-stream'
        }
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
    });

    it('stream=true在AGENT_REPLY事件后应关闭流', async () => {
      const testFastify = Fastify({ logger: false });
      const createdAt = new Date().toISOString();
      const sessionId = `stream-close-after-reply-${Date.now()}`;
      const session = {
        sessionId,
        agentId: 'agent-stream-close',
        systemPrompt: 'prompt',
        allowedTools: [],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        lastEventOffset: 3,
        updatedAt: createdAt
      };
      const events = [
        {
          id: 'event-1',
          type: 'USER_INPUT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-close',
          payload: { text: 'first' },
          offset: 1
        },
        {
          id: 'event-2',
          type: 'AGENT_REPLY',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-close',
          payload: { text: '任务完成' },
          offset: 2
        },
        {
          id: 'event-3',
          type: 'TOOL_RESULT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-close',
          payload: {
            toolCallId: 'tool-3',
            toolName: 'ignored-tool',
            status: 'ok'
          },
          offset: 3
        }
      ];

      const mockRuntime = {
        createSession: () => ({
          ok: true as const,
          session,
          agent: {
            agentId: session.agentId,
            name: 'stream-close-agent',
            systemPrompt: 'prompt',
            allowedTools: [],
            createdAt,
            updatedAt: createdAt
          },
          createdAt,
          initialEvent: {
            id: 'initial-event-1',
            type: 'USER_INPUT',
            sessionId,
            timestamp: createdAt,
            correlationId: 'corr-close',
            payload: { text: 'first' },
            offset: 1
          }
        }),
        listEventsAfter: (_sessionId: string, afterOffset: number) =>
          events.filter((event) => event.offset > afterOffset),
        subscribeSessionEvents: () => () => {},
        abortCorrelation: () => {}
      } as unknown as EventDrivenAgentRuntime;

      registerAgentRoutes(testFastify, mockRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          payload: {
            stream: true
          }
        });

        expect(response.statusCode).toBe(200);
        expect(response.headers['content-type']).toContain('text/event-stream');
        expect(response.body).toContain('event: event');
        expect(response.body).toContain('"type":"AGENT_REPLY"');
        expect(response.body).toContain('"text":"任务完成"');
        expect(response.body).not.toContain('"offset":3');
      } finally {
        await testFastify.close();
      }
    });

    it('stream=true应推送订阅后新增的event事件', async () => {
      const testFastify = Fastify({ logger: false });
      const createdAt = new Date().toISOString();
      const sessionId = `stream-live-events-${Date.now()}`;
      const session = {
        sessionId,
        agentId: 'agent-stream-live',
        systemPrompt: 'prompt',
        allowedTools: [],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        lastEventOffset: 1,
        updatedAt: createdAt
      };
      const events: Array<Record<string, unknown>> = [
        {
          id: 'event-1',
          type: 'USER_INPUT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-live',
          payload: { text: 'first' },
          offset: 1
        }
      ];
      let subscriber: (() => void) | undefined;

      const mockRuntime = {
        createSession: () => ({
          ok: true as const,
          session,
          agent: {
            agentId: session.agentId,
            name: 'stream-live-agent',
            systemPrompt: 'prompt',
            allowedTools: [],
            createdAt,
            updatedAt: createdAt
          },
          createdAt,
          initialEvent: {
            id: 'event-1',
            type: 'USER_INPUT',
            sessionId,
            timestamp: createdAt,
            correlationId: 'corr-live',
            payload: { text: 'first' },
            offset: 1
          }
        }),
        listEventsAfter: (_sessionId: string, afterOffset: number) =>
          events.filter((event) => Number(event.offset) > afterOffset),
        subscribeSessionEvents: (_sessionId: string, cb: () => void) => {
          subscriber = cb;
          return () => {};
        },
        abortCorrelation: () => {}
      } as unknown as EventDrivenAgentRuntime;

      registerAgentRoutes(testFastify, mockRuntime);

      try {
        const responsePromise = testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          payload: {
            stream: true
          }
        });

        await new Promise((resolve) => setTimeout(resolve, 5));

        events.push({
          id: 'event-2',
          type: 'AGENT_REPLY_CHUNK',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-live',
          payload: { text: '分片', done: false },
          offset: 2
        });
        subscriber?.();

        events.push({
          id: 'event-3',
          type: 'AGENT_REPLY',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-live',
          payload: { text: '完成' },
          offset: 3
        });
        subscriber?.();

        const response = await responsePromise;
        expect(response.statusCode).toBe(200);
        expect(response.headers['content-type']).toContain('text/event-stream');
        expect(response.body).toContain('event: event');
        expect(response.body).toContain('"type":"AGENT_REPLY_CHUNK"');
        expect(response.body).toContain('"text":"分片"');
        expect(response.body).toContain('"type":"AGENT_REPLY"');
        expect(response.body).toContain('"text":"完成"');
      } finally {
        await testFastify.close();
      }
    });

    it('stream=false返回的events应按offset升序', async () => {
      const testFastify = Fastify({ logger: false });
      const createdAt = new Date().toISOString();
      const sessionId = `ordered-session-${Date.now()}`;
      const session = {
        sessionId,
        agentId: 'agent-ordered',
        systemPrompt: 'prompt',
        allowedTools: [],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        lastEventOffset: 3,
        updatedAt: createdAt
      };
      const events = [
        {
          id: 'event-3',
          type: 'AGENT_REPLY',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-1',
          payload: { text: 'third' },
          offset: 3
        },
        {
          id: 'event-1',
          type: 'USER_INPUT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-1',
          payload: { text: 'first' },
          offset: 1
        },
        {
          id: 'event-2',
          type: 'TOOL_RESULT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-1',
          payload: {
            toolCallId: 'tool-1',
            toolName: 'test-tool',
            status: 'ok'
          },
          offset: 2
        }
      ];

      const mockRuntime = {
        createSession: () => ({
          ok: true as const,
          session,
          agent: {
            agentId: session.agentId,
            name: 'ordered-agent',
            systemPrompt: 'prompt',
            allowedTools: [],
            createdAt,
            updatedAt: createdAt
          },
          createdAt,
          initialEvent: undefined
        }),
        getSession: () => session,
        listEventsAfter: () => events
      } as unknown as EventDrivenAgentRuntime;

      registerAgentRoutes(testFastify, mockRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          payload: {
            stream: false
          }
        });

        expect(response.statusCode).toBe(201);
        const body = JSON.parse(response.body);
        expect(body.events.map((event: { offset: number }) => event.offset)).toEqual([1, 2, 3]);
      } finally {
        await testFastify.close();
      }
    });
  });

  describe('GET /agent/sessions/:sessionId', () => {
    it('应该返回已创建的session', async () => {
      // 先创建session
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      const sessionId = createBody.sessionId;

      // 获取session
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}`
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.session.sessionId).toBe(sessionId);
    });

    it('应该为不存在的session返回404', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/agent/sessions/non-existent-session-id'
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session not found');
    });
  });

  describe('POST /agent/sessions/:sessionId/resume', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该成功恢复session', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/resume`,
        payload: {
          stream: false
        }
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('session');
      expect(body).toHaveProperty('events');
    });

    it('应该成功恢复session并发送prompt', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/resume`,
        payload: {
          prompt: '继续对话',
          stream: false
        }
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('acceptedEvent');
      expect(body.acceptedEvent.type).toBe('USER_INPUT');
      expect(body.acceptedEvent.payload.text).toBe('继续对话');
    });

    it('应该支持fromOffset参数', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/resume`,
        payload: {
          fromOffset: 10,
          stream: false
        }
      });

      expect(response.statusCode).toBe(200);
    });

    it('应该正确处理负数的fromOffset(转换为0)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/resume`,
        payload: {
          fromOffset: -1,
          stream: false
        }
      });

      // 负数fromOffset会被转换为0,不应该返回错误
      expect(response.statusCode).toBe(200);
    });

    it('应该为不存在的session返回404', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions/non-existent-session-id/resume'
      });

      expect(response.statusCode).toBe(404);
    });

    it('stream=true恢复时在reply事件后应关闭流', async () => {
      const testFastify = Fastify({ logger: false });
      const createdAt = new Date().toISOString();
      const streamSessionId = `resume-stream-close-${Date.now()}`;
      const streamSession = {
        sessionId: streamSessionId,
        agentId: 'agent-resume-stream-close',
        systemPrompt: 'prompt',
        allowedTools: [],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        lastEventOffset: 3,
        updatedAt: createdAt
      };
      const events = [
        {
          id: 'resume-stream-event-1',
          type: 'USER_INPUT',
          sessionId: streamSessionId,
          timestamp: createdAt,
          correlationId: 'corr-resume-close',
          payload: { text: 'first' },
          offset: 1
        },
        {
          id: 'resume-stream-event-2',
          type: 'AGENT_REPLY',
          sessionId: streamSessionId,
          timestamp: createdAt,
          correlationId: 'corr-resume-close',
          payload: { text: '任务完成' },
          offset: 2
        },
        {
          id: 'resume-stream-event-3',
          type: 'TOOL_RESULT',
          sessionId: streamSessionId,
          timestamp: createdAt,
          correlationId: 'corr-resume-close',
          payload: {
            toolCallId: 'tool-3',
            toolName: 'ignored-tool',
            status: 'ok'
          },
          offset: 3
        }
      ];

      const mockRuntime = {
        getSession: (id: string) => (id === streamSessionId ? streamSession : undefined),
        publishEvent: () => {
          throw new Error('publishEvent should not be called in this test');
        },
        listEventsAfter: (_sessionId: string, afterOffset: number) =>
          events.filter((event) => event.offset > afterOffset),
        subscribeSessionEvents: () => () => {},
        abortCorrelation: () => {}
      } as unknown as EventDrivenAgentRuntime;

      registerAgentRoutes(testFastify, mockRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: `/agent/sessions/${streamSessionId}/resume`,
          payload: {
            stream: true
          }
        });

        expect(response.statusCode).toBe(200);
        expect(response.headers['content-type']).toContain('text/event-stream');
        expect(response.body).toContain('event: event');
        expect(response.body).toContain('"type":"AGENT_REPLY"');
        expect(response.body).toContain('"text":"任务完成"');
        expect(response.body).not.toContain('"offset":3');
      } finally {
        await testFastify.close();
      }
    });

    it('stream=false恢复时返回的events应按offset升序', async () => {
      const testFastify = Fastify({ logger: false });
      const createdAt = new Date().toISOString();
      const sessionId = `resume-ordered-${Date.now()}`;
      const session = {
        sessionId,
        agentId: 'agent-resume',
        systemPrompt: 'prompt',
        allowedTools: [],
        memoryRefs: [],
        messages: [],
        pendingToolCalls: {},
        lastEventOffset: 3,
        updatedAt: createdAt
      };
      const events = [
        {
          id: 'resume-event-2',
          type: 'TOOL_RESULT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-2',
          payload: {
            toolCallId: 'tool-2',
            toolName: 'test-tool',
            status: 'ok'
          },
          offset: 2
        },
        {
          id: 'resume-event-3',
          type: 'AGENT_REPLY',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-2',
          payload: { text: 'third' },
          offset: 3
        },
        {
          id: 'resume-event-1',
          type: 'USER_INPUT',
          sessionId,
          timestamp: createdAt,
          correlationId: 'corr-2',
          payload: { text: 'first' },
          offset: 1
        }
      ];

      const mockRuntime = {
        getSession: (id: string) => (id === sessionId ? session : undefined),
        publishEvent: () => {
          throw new Error('publishEvent should not be called in this test');
        },
        listEventsAfter: () => events
      } as unknown as EventDrivenAgentRuntime;

      registerAgentRoutes(testFastify, mockRuntime);

      try {
        const response = await testFastify.inject({
          method: 'POST',
          url: `/agent/sessions/${sessionId}/resume`,
          payload: {
            stream: false
          }
        });

        expect(response.statusCode).toBe(200);
        const body = JSON.parse(response.body);
        expect(body.events.map((event: { offset: number }) => event.offset)).toEqual([1, 2, 3]);
      } finally {
        await testFastify.close();
      }
    });
  });

  describe('GET /agent/sessions/:sessionId/stream', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it.skip('应该返回SSE流', async () => {
      // SSE流测试会超时,跳过
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/stream`
      });

      expect(response.statusCode).toBe(200);
      expect(response.headers['content-type']).toContain('text/event-stream');
      expect(response.headers['cache-control']).toContain('no-cache');
    });

    it.skip('应该支持fromOffset查询参数', async () => {
      // SSE流测试会超时,跳过
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/stream?fromOffset=5`
      });

      expect(response.statusCode).toBe(200);
    });

    it('应该为不存在的session返回404', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/agent/sessions/non-existent-session-id/stream'
      });

      expect(response.statusCode).toBe(404);
    });
  });

  describe('GET /agent/sessions/:sessionId/events', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          prompt: '测试事件',
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该返回session事件列表', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events`
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('events');
      expect(Array.isArray(body.events)).toBe(true);
    });

    it('应该支持limit查询参数', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events?limit=10`
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body.events.length).toBeLessThanOrEqual(10);
    });

    it('应该支持afterOffset查询参数', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events?afterOffset=0`
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.body);
      expect(body).toHaveProperty('events');
    });
  });

  describe('POST /agent/sessions/:sessionId/input', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该成功发送用户输入', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/input`,
        payload: {
          text: '这是用户输入'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.accepted).toBe(true);
      expect(body).toHaveProperty('event');
      expect(body.event.type).toBe('USER_INPUT');
      expect(body.event.payload.text).toBe('这是用户输入');
    });

    it('应该支持带metadata的用户输入', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/input`,
        payload: {
          text: '带metadata的输入',
          metadata: {
            source: 'test',
            priority: 'high'
          }
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.metadata).toEqual({
        source: 'test',
        priority: 'high'
      });
    });

    it('应该支持correlationId', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/input`,
        payload: {
          text: '带correlationId的输入',
          correlationId: 'correlation-123'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.correlationId).toBe('correlation-123');
    });

    it('session不存在时应返回404且不应隐式创建session', async () => {
      const missingSessionId = `missing-input-session-${Date.now()}`;
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${missingSessionId}/input`,
        payload: {
          text: 'should fail'
        }
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session not found');

      const getResponse = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${missingSessionId}`
      });
      expect(getResponse.statusCode).toBe(404);
    });

    it('应该拒绝没有text字段的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/input`,
        payload: {
          metadata: {}
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('text');
    });

    it('应该拒绝非对象类型的请求体', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/input`,
        payload: 'not-an-object',
        headers: {
          'content-type': 'text/plain'
        }
      });

      expect([400, 415]).toContain(response.statusCode);
    });
  });

  describe('POST /agent/sessions/:sessionId/config', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该成功更新systemPrompt', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          systemPrompt: '新的系统提示词'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.accepted).toBe(true);
      expect(body.event.type).toBe('SESSION_CONFIG_UPDATED');
      expect(body.event.payload.systemPrompt).toBe('新的系统提示词');
    });

    it('应该成功更新session model(provider/model)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          model: 'gemini/gemini-2.5-pro'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.model).toBe('gemini/gemini-2.5-pro');
    });

    it('应该成功更新allowedTools', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          allowedTools: ['tool1', 'tool2', 'tool3']
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.allowedTools).toEqual(['tool1', 'tool2', 'tool3']);
    });

    it('应该成功更新memoryRefs', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          memoryRefs: ['mem1', 'mem2']
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.memoryRefs).toEqual(['mem1', 'mem2']);
    });

    it('应该成功同时更新多个配置', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          systemPrompt: '新提示词',
          allowedTools: ['tool1'],
          memoryRefs: ['mem1']
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.systemPrompt).toBe('新提示词');
      expect(body.event.payload.allowedTools).toEqual(['tool1']);
      expect(body.event.payload.memoryRefs).toEqual(['mem1']);
    });

    it('session不存在时config更新应返回404', async () => {
      const missingSessionId = `missing-config-session-${Date.now()}`;
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${missingSessionId}/config`,
        payload: {
          systemPrompt: 'new prompt'
        }
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session not found');
    });

    it('应该拒绝没有任何配置字段的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {}
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('At least one field is required');
    });

    it('应该拒绝无效model格式的config更新请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: {
          model: 'gemini-2.5-pro'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('provider/model');
    });

    it('应该拒绝非对象类型的请求体', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/config`,
        payload: 'not-an-object',
        headers: {
          'content-type': 'text/plain'
        }
      });

      expect([400, 415]).toContain(response.statusCode);
    });
  });

  describe('POST /agent/sessions/:sessionId/tool-result', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该成功提交工具结果(status=ok)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: {
          toolCallId: 'tool-call-123',
          toolName: 'testTool',
          status: 'ok',
          result: { data: '工具执行成功' }
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.accepted).toBe(true);
      expect(body.event.type).toBe('TOOL_RESULT');
      expect(body.event.payload.toolCallId).toBe('tool-call-123');
      expect(body.event.payload.toolName).toBe('testTool');
      expect(body.event.payload.status).toBe('ok');
    });

    it('应该成功提交工具错误结果(status=error)', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: {
          toolCallId: 'tool-call-456',
          toolName: 'testTool',
          status: 'error',
          error: '工具执行失败'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.status).toBe('error');
      expect(body.event.payload.error).toBe('工具执行失败');
    });

    it('应该根据error字段自动设置status为error', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: {
          toolCallId: 'tool-call-789',
          toolName: 'testTool',
          error: '自动设置error状态'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.payload.status).toBe('error');
    });

    it('session不存在时tool-result应返回404', async () => {
      const missingSessionId = `missing-tool-result-session-${Date.now()}`;
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${missingSessionId}/tool-result`,
        payload: {
          toolCallId: 'tool-call-789',
          toolName: 'testTool',
          status: 'ok',
          result: { ok: true }
        }
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session not found');
    });

    it('应该拒绝缺少toolCallId的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: {
          toolName: 'testTool'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('toolCallId');
    });

    it('应该拒绝缺少toolName的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: {
          toolCallId: 'tool-call-123'
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('toolName');
    });

    it('应该拒绝非对象类型的请求体', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/tool-result`,
        payload: 'not-an-object',
        headers: {
          'content-type': 'text/plain'
        }
      });

      expect([400, 415]).toContain(response.statusCode);
    });
  });

  describe('POST /agent/sessions/:sessionId/events', () => {
    let sessionId: string;

    beforeEach(async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      sessionId = createBody.sessionId;
    });

    it('应该成功发布USER_INPUT事件', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          type: 'USER_INPUT',
          payload: {
            text: '通过事件API发送的输入'
          }
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.accepted).toBe(true);
      expect(body.event.type).toBe('USER_INPUT');
    });

    it('应该成功发布SESSION_CONFIG_UPDATED事件', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          type: 'SESSION_CONFIG_UPDATED',
          payload: {
            systemPrompt: '通过事件更新系统提示词'
          }
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.type).toBe('SESSION_CONFIG_UPDATED');
    });

    it('应该成功发布TOOL_RESULT事件', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          type: 'TOOL_RESULT',
          payload: {
            toolCallId: 'event-tool-call-123',
            toolName: 'eventTool',
            status: 'ok',
            result: { success: true }
          }
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.type).toBe('TOOL_RESULT');
    });

    it('应该支持correlationId和causationId', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          type: 'USER_INPUT',
          payload: {
            text: '带关联ID的事件'
          },
          correlationId: 'correlation-abc',
          causationId: 'causation-xyz'
        }
      });

      expect(response.statusCode).toBe(202);
      const body = JSON.parse(response.body);
      expect(body.event.correlationId).toBe('correlation-abc');
      expect(body.event.causationId).toBe('causation-xyz');
    });

    it('session不存在时events发布应返回404', async () => {
      const missingSessionId = `missing-events-session-${Date.now()}`;
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${missingSessionId}/events`,
        payload: {
          type: 'USER_INPUT',
          payload: {
            text: 'should fail'
          }
        }
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Session not found');
    });

    it('应该拒绝不支持的事件类型', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          type: 'UNSUPPORTED_EVENT_TYPE',
          payload: {}
        }
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.body);
      expect(body.error.message).toContain('Unsupported event type');
    });

    it('应该拒绝没有type字段的请求', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: {
          payload: {}
        }
      });

      expect(response.statusCode).toBe(400);
    });

    it('应该拒绝非对象类型的请求体', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: `/agent/sessions/${sessionId}/events`,
        payload: 'not-an-object',
        headers: {
          'content-type': 'text/plain'
        }
      });

      expect([400, 415]).toContain(response.statusCode);
    });
  });

  describe('边界情况测试', () => {
    it('应该处理空字符串参数', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: '   ',
          description: '   '
        }
      });

      expect(response.statusCode).toBe(400);
    });

    it('应该处理超长字符串数组', async () => {
      const tools = Array(1000).fill('tool');
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          tools,
          stream: false
        }
      });

      expect(response.statusCode).toBe(201);
    });

    it('应该处理tools数组中的重复项', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: 'TestAgent',
          tools: ['tool1', 'tool1', 'tool2', 'tool2']
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      // 应该去重
      expect(body.agent.allowedTools).toEqual(['tool1', 'tool2']);
    });

    it('应该处理tools数组中的空字符串', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/agent/agents',
        payload: {
          name: 'TestAgent',
          tools: ['tool1', '', '  ', 'tool2']
        }
      });

      expect(response.statusCode).toBe(201);
      const body = JSON.parse(response.body);
      // 应该过滤掉空字符串
      expect(body.agent.allowedTools).toEqual(['tool1', 'tool2']);
    });

    it('应该处理非常大的limit参数', async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          prompt: '测试',
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      const sessionId = createBody.sessionId;

      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events?limit=999999`
      });

      expect(response.statusCode).toBe(200);
    });

    it('应该处理无效的limit参数', async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      const sessionId = createBody.sessionId;

      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events?limit=invalid`
      });

      // 应该使用默认值并返回200
      expect(response.statusCode).toBe(200);
    });

    it('应该处理无效的afterOffset参数', async () => {
      const createResponse = await fastify.inject({
        method: 'POST',
        url: '/agent/sessions',
        payload: {
          stream: false
        }
      });
      const createBody = JSON.parse(createResponse.body);
      const sessionId = createBody.sessionId;

      const response = await fastify.inject({
        method: 'GET',
        url: `/agent/sessions/${sessionId}/events?afterOffset=invalid`
      });

      // 应该使用默认值并返回200
      expect(response.statusCode).toBe(200);
    });
  });

  describe('外部数据源模式', () => {
    it('应禁用agent/session写管理接口', async () => {
      const testFastify = Fastify({ logger: false });
      const mockRuntime = {
        listTools: async () => [],
        listAgents: () => [],
        listSessions: () => []
      } as unknown as EventDrivenAgentRuntime;
      registerAgentRoutes(testFastify, mockRuntime, {
        managementEnabled: false
      });
      await testFastify.ready();

      try {
        const createAgentResp = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: 'blocked'
          }
        });
        expect(createAgentResp.statusCode).toBe(405);

        const createSessionResp = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          payload: {}
        });
        expect(createSessionResp.statusCode).toBe(405);

        const inputResp = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions/external-session/input',
          payload: {
            text: 'hello'
          }
        });
        expect(inputResp.statusCode).toBe(405);
      } finally {
        await testFastify.close();
      }
    });

    it('应支持禁用agent管理但保留session写接口', async () => {
      const testFastify = Fastify({ logger: false });
      const mockRuntime = {
        listTools: async () => [],
        listAgents: () => [],
        listSessions: () => [],
        getAgent: () => ({
          agentId: 'agent-external-1',
          name: 'external-agent',
          systemPrompt: 'external',
          allowedTools: [],
          createdAt: '2026-01-01T00:00:00.000Z',
          updatedAt: '2026-01-01T00:00:00.000Z'
        }),
        createSession: () => ({
          ok: true,
          createdAt: '2026-01-01T00:00:00.000Z',
          agent: {
            agentId: 'agent-external-1'
          },
          session: {
            sessionId: 'session-external-1',
            agentId: 'agent-external-1'
          },
          initialEvent: undefined
        }),
        getSession: () => ({
          sessionId: 'session-external-1',
          agentId: 'agent-external-1',
          systemPrompt: 'external-session',
          allowedTools: [],
          memoryRefs: [],
          messages: [],
          pendingToolCalls: {},
          lastEventOffset: 0,
          updatedAt: '2026-01-01T00:00:00.000Z'
        }),
        listEventsAfter: () => []
      } as unknown as EventDrivenAgentRuntime;
      registerAgentRoutes(testFastify, mockRuntime, {
        agentManagementEnabled: false,
        sessionManagementEnabled: true
      });
      await testFastify.ready();

      try {
        const createAgentResp = await testFastify.inject({
          method: 'POST',
          url: '/agent/agents',
          payload: {
            name: 'blocked'
          }
        });
        expect(createAgentResp.statusCode).toBe(405);

        const createSessionResp = await testFastify.inject({
          method: 'POST',
          url: '/agent/sessions',
          payload: {
            agentId: 'agent-external-1',
            prompt: 'hello',
            stream: false
          }
        });
        expect(createSessionResp.statusCode).toBe(201);
        const body = JSON.parse(createSessionResp.body);
        expect(body.sessionId).toBe('session-external-1');
        expect(body.agentId).toBe('agent-external-1');
      } finally {
        await testFastify.close();
      }
    });

    it('应保留只读查询接口', async () => {
      const testFastify = Fastify({ logger: false });
      const mockRuntime = {
        listTools: async () => [],
        listAgents: () => [
          {
            agentId: 'agent-external-1',
            name: 'external-agent',
            systemPrompt: 'external',
            allowedTools: [],
            createdAt: '2026-01-01T00:00:00.000Z',
            updatedAt: '2026-01-01T00:00:00.000Z'
          }
        ],
        listSessions: () => [
          {
            sessionId: 'session-external-1',
            agentId: 'agent-external-1',
            systemPrompt: 'external-session',
            allowedTools: [],
            memoryRefs: [],
            messages: [],
            pendingToolCalls: {},
            lastEventOffset: 0,
            updatedAt: '2026-01-01T00:00:00.000Z'
          }
        ]
      } as unknown as EventDrivenAgentRuntime;
      registerAgentRoutes(testFastify, mockRuntime, {
        managementEnabled: false
      });
      await testFastify.ready();

      try {
        const agentResp = await testFastify.inject({
          method: 'GET',
          url: '/agent/agents'
        });
        expect(agentResp.statusCode).toBe(200);
        const agentBody = JSON.parse(agentResp.body);
        expect(agentBody.agents).toHaveLength(1);

        const sessionResp = await testFastify.inject({
          method: 'GET',
          url: '/agent/sessions'
        });
        expect(sessionResp.statusCode).toBe(200);
        const sessionBody = JSON.parse(sessionResp.body);
        expect(sessionBody.sessions).toHaveLength(1);
      } finally {
        await testFastify.close();
      }
    });
  });
});
