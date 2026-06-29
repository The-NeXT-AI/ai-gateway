import { afterEach, describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from './config';

describe('Gateway config providerPlugins', () => {
  afterEach(() => {
    delete process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE;
    delete process.env.MCP_GATEWAY_WS_ENDPOINT;
    delete process.env.PRECHECK_STORAGE_TYPE;
    delete process.env.GATEWAY_POLICY_ENABLED;
    delete process.env.GATEWAY_POLICY_ALLOW_PROVIDERS;
    delete process.env.GATEWAY_POLICY_DENY_MODELS;
    delete process.env.PROVIDER_HEALTH_CHECK_ENABLED;
    delete process.env.PROVIDER_HEALTH_CHECK_INTERVAL_MS;
    delete process.env.PROVIDER_HEALTH_CHECK_INTERVAL_SECONDS;
    delete process.env.PROVIDER_HEALTH_CHECK_TIMEOUT_MS;
    delete process.env.PROVIDER_HEALTH_CHECK_TIMEOUT_SECONDS;
    delete process.env.PROVIDER_HEALTH_CHECK_INITIAL_DELAY_MS;
    delete process.env.PROVIDER_HEALTH_CHECK_INITIAL_DELAY_SECONDS;
    delete process.env.GATEWAY_METRICS_ENABLED;
    delete process.env.GATEWAY_METRICS_INCLUDE_PROVIDER_HEALTH;
    delete process.env.GATEWAY_CORS_ENABLED;
    delete process.env.GATEWAY_CORS_ORIGIN;
    delete process.env.GATEWAY_CORS_ORIGINS;
    delete process.env.CORS_ORIGIN;
    delete process.env.GATEWAY_CORS_ALLOWED_HEADERS;
    delete process.env.GATEWAY_CORS_ALLOWED_METHODS;
    delete process.env.GATEWAY_CORS_ALLOW_CREDENTIALS;
    delete process.env.GATEWAY_CORS_MAX_AGE_SECONDS;
    delete process.env.GATEWAY_IDEMPOTENCY_ENABLED;
    delete process.env.GATEWAY_IDEMPOTENCY_HEADER;
    delete process.env.GATEWAY_IDEMPOTENCY_TTL_MS;
    delete process.env.GATEWAY_IDEMPOTENCY_TTL_SECONDS;
    delete process.env.GATEWAY_IDEMPOTENCY_MAX_ENTRIES;
    delete process.env.GATEWAY_IDEMPOTENCY_CACHE_ERROR_RESPONSES;
    delete process.env.GATEWAY_UPSTREAM_CONCURRENCY_ENABLED;
    delete process.env.GATEWAY_UPSTREAM_MAX_IN_FLIGHT_PER_PROVIDER;
    delete process.env.GATEWAY_UPSTREAM_CONCURRENCY_QUEUE_TIMEOUT_MS;
    delete process.env.GATEWAY_UPSTREAM_CONCURRENCY_QUEUE_TIMEOUT_SECONDS;
    delete process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_ENABLED;
    delete process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_THRESHOLD;
    delete process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_COOLDOWN_MS;
    delete process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_COOLDOWN_SECONDS;
    delete process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_STATUS_CODES;
    delete process.env.GATEWAY_UPSTREAM_RETRY_ENABLED;
    delete process.env.GATEWAY_UPSTREAM_RETRY_MAX_ATTEMPTS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_BASE_DELAY_MS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_BASE_DELAY_SECONDS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_MAX_DELAY_MS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_MAX_DELAY_SECONDS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_BACKOFF_MULTIPLIER;
    delete process.env.GATEWAY_UPSTREAM_RETRY_JITTER_MS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_JITTER_SECONDS;
    delete process.env.GATEWAY_UPSTREAM_RETRY_STATUS_CODES;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ENABLED;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TURNS;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TOOL_CALLS;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_REQUIRE_CLIENT_DECLARATION;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_UNKNOWN_TOOL_POLICY;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ALLOW_TOOLS;
    delete process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_DENY_TOOLS;
    delete process.env.TEST_PROVIDER_API_KEY;
    delete process.env.AUTH_MODE;
    delete process.env.AUTH_STATIC_API_KEY;
    delete process.env.AUTH_STATIC_API_KEYS;
    delete process.env.AUTH_STATIC_API_KEY_ENV;
    delete process.env.AUTH_STATIC_API_KEYS_ENV;
    delete process.env.AUTH_STATIC_API_KEY_HEADER;
    delete process.env.AUTH_STATIC_API_KEY_BEARER_ONLY;
    delete process.env.TEST_GATEWAY_API_KEYS;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_ENABLED;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_TRANSPORT;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_ENDPOINT;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_URL;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_METHOD;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_TIMEOUT_MS;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_INTERVAL_MS;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_INTERVAL_SECONDS;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY_HEADER;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_API_KEY_ENV;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_COMMAND;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_ARGS;
    delete process.env.GATEWAY_CONFIG_EXTERNAL_STDIO_CWD;
    delete process.env.AGENT_EVENT_WEBHOOK_ENABLED;
    delete process.env.AGENT_EVENT_WEBHOOK_TRANSPORT;
    delete process.env.AGENT_EVENT_WEBHOOK_ENDPOINT;
    delete process.env.AGENT_EVENT_WEBHOOK_URL;
    delete process.env.AGENT_EVENT_WEBHOOK_TIMEOUT_MS;
    delete process.env.AGENT_EVENT_WEBHOOK_MAX_ATTEMPTS;
    delete process.env.AGENT_EVENT_WEBHOOK_BASE_DELAY_MS;
    delete process.env.AGENT_EVENT_WEBHOOK_BASE_DELAY_SECONDS;
    delete process.env.AGENT_EVENT_WEBHOOK_MAX_DELAY_MS;
    delete process.env.AGENT_EVENT_WEBHOOK_MAX_DELAY_SECONDS;
    delete process.env.AGENT_EVENT_WEBHOOK_REQUIRE_ACK;
    delete process.env.AGENT_EVENT_WEBHOOK_WEBSOCKET_REQUIRE_ACK;
    delete process.env.AGENT_EVENT_WEBHOOK_API_KEY_HEADER;
    delete process.env.AGENT_EVENT_WEBHOOK_API_KEY;
    delete process.env.AGENT_EVENT_WEBHOOK_API_KEY_ENV;
    delete process.env.AGENT_EVENT_WEBHOOK_AUTHORIZATION;
    delete process.env.AGENT_EVENT_WEBHOOK_STDIO_COMMAND;
    delete process.env.AGENT_EVENT_WEBHOOK_STDIO_ARGS;
    delete process.env.AGENT_EVENT_WEBHOOK_STDIO_CWD;
    delete process.env.BILLING_WEBHOOK_TRANSPORT;
    delete process.env.BILLING_WEBHOOK_MAX_ATTEMPTS;
    delete process.env.BILLING_WEBHOOK_BASE_DELAY_MS;
    delete process.env.BILLING_WEBHOOK_BASE_DELAY_SECONDS;
    delete process.env.BILLING_WEBHOOK_MAX_DELAY_MS;
    delete process.env.BILLING_WEBHOOK_MAX_DELAY_SECONDS;
    delete process.env.BILLING_WEBHOOK_REQUIRE_ACK;
    delete process.env.BILLING_WEBHOOK_WEBSOCKET_REQUIRE_ACK;
    delete process.env.BILLING_WEBHOOK_STDIO_COMMAND;
    delete process.env.BILLING_WEBHOOK_STDIO_ARGS;
    delete process.env.BILLING_WEBHOOK_STDIO_CWD;
    delete process.env.RAW_TRACE_SYNC_TRANSPORT;
    delete process.env.RAW_TRACE_SYNC_ENDPOINT;
    delete process.env.RAW_TRACE_SYNC_URL;
    delete process.env.RAW_TRACE_SYNC_MAX_ATTEMPTS;
    delete process.env.RAW_TRACE_SYNC_BASE_DELAY_MS;
    delete process.env.RAW_TRACE_SYNC_BASE_DELAY_SECONDS;
    delete process.env.RAW_TRACE_SYNC_MAX_DELAY_MS;
    delete process.env.RAW_TRACE_SYNC_MAX_DELAY_SECONDS;
    delete process.env.RAW_TRACE_SYNC_REQUIRE_ACK;
    delete process.env.RAW_TRACE_SYNC_WEBSOCKET_REQUIRE_ACK;
    delete process.env.RAW_TRACE_SYNC_STDIO_COMMAND;
    delete process.env.RAW_TRACE_SYNC_STDIO_ARGS;
    delete process.env.RAW_TRACE_SYNC_STDIO_CWD;
    delete process.env.AGENT_STORAGE_TYPE;
    delete process.env.AGENT_STORAGE_DIR;
    delete process.env.AGENT_EXTERNAL_TRANSPORT;
    delete process.env.AGENT_EXTERNAL_ENDPOINT;
    delete process.env.AGENT_EXTERNAL_URL;
    delete process.env.AGENT_EXTERNAL_STDIO_COMMAND;
    delete process.env.AGENT_EXTERNAL_STDIO_ARGS;
    delete process.env.AGENT_EXTERNAL_STDIO_CWD;
    delete process.env.PROVIDER_EXTERNAL_TRANSPORT;
    delete process.env.PROVIDER_EXTERNAL_ENDPOINT;
    delete process.env.PROVIDER_EXTERNAL_URL;
    delete process.env.PROVIDER_EXTERNAL_STDIO_COMMAND;
    delete process.env.PROVIDER_EXTERNAL_STDIO_ARGS;
    delete process.env.PROVIDER_EXTERNAL_STDIO_CWD;
  });

  it('resolves provider api keys from apiKeyEnv', () => {
    process.env.TEST_PROVIDER_API_KEY = 'provider-env-secret';

    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-env',
          type: 'openai_responses',
          apiKeyEnv: 'TEST_PROVIDER_API_KEY',
          models: ['gpt-4.1-mini']
        }
      ]
    });

    expect(config.providers[0]).toMatchObject({
      name: 'openai-env',
      apikey: 'provider-env-secret',
      apiKeyEnv: 'TEST_PROVIDER_API_KEY'
    });
  });

  it('parses OpenAI chat tools format compatibility config', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'r9s-openai',
          type: 'openai_chat_completions',
          openaiChatToolsFormat: 'anthropic',
          models: ['glm-5.1']
        }
      ]
    });

    expect(config.providers[0]?.openaiChatToolsFormat).toBe('anthropic');
  });

  it('parses OpenAI chat stream usage compatibility config', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'usage-enabled',
          type: 'openai_chat_completions',
          openaiChatStreamUsage: 'include_usage',
          models: ['glm-5.1']
        },
        {
          name: 'usage-disabled',
          type: 'openai_chat_completions',
          openaiChatStreamUsage: false,
          models: ['legacy-chat']
        }
      ]
    });

    expect(config.providers[0]?.openaiChatStreamUsage).toBe('include_usage');
    expect(config.providers[1]?.openaiChatStreamUsage).toBe('disabled');
  });

  it('parses static API key auth config from file and environment', () => {
    process.env.AUTH_MODE = 'static-api-key';
    process.env.AUTH_STATIC_API_KEYS = 'env-key-1,env-key-2';
    process.env.TEST_GATEWAY_API_KEYS = 'ref-key-1,ref-key-2';

    const config = parseGatewayConfigFromRaw({
      auth: {
        enabled: true,
        staticApiKeys: {
          keys: ['file-key'],
          keyEnv: 'TEST_GATEWAY_API_KEYS',
          keyHeader: 'x-api-key',
          keyBearerOnly: false
        }
      }
    });

    expect(config.auth.mode).toBe('static_api_key');
    expect(config.auth.staticApiKeys).toMatchObject({
      keys: ['env-key-1', 'env-key-2', 'ref-key-1', 'ref-key-2', 'file-key'],
      keyEnv: 'TEST_GATEWAY_API_KEYS',
      keyHeader: 'x-api-key',
      keyBearerOnly: false
    });
  });

  it('parses codexOauth provider plugin config with defaults', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-api-key',
          models: ['glm-5']
        }
      ],
      providerPlugins: [
        {
          key: 'openai-main-codex-oauth',
          providerName: 'openai-main',
          codexOauth: {
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            },
            refreshToken: {
              from: 'request.headers.x-codex-refresh-token'
            }
          }
        }
      ]
    });

    const plugin = config.providerPlugins?.[0];
    expect(plugin?.codexOauth).toBeDefined();
    expect(plugin?.codexOauth?.tokenEndpoint).toBe('https://auth.openai.com/oauth/token');
    expect(plugin?.codexOauth?.clientId).toBe('app_EMoamEEZ73f0CkXaXp7hrann');
    expect(plugin?.codexOauth?.scope).toBe(
      'openid profile email offline_access api.connectors.read api.connectors.invoke'
    );
    expect(plugin?.codexOauth?.authHeader).toBe('authorization');
    expect(plugin?.codexOauth?.authScheme).toBe('Bearer');
    expect(plugin?.codexOauth?.refreshIfMissingAccessToken).toBe(true);
    expect(plugin?.codexOauth?.forceRefresh).toBe(false);
    expect(plugin?.codexOauth?.required).toBe(true);
  });

  it('parses HTTP external gateway config source', () => {
    const config = parseGatewayConfigFromRaw({
      externalConfig: {
        enabled: true,
        transport: 'https',
        endpoint: 'https://config.example.com/gateway',
        method: 'post',
        timeoutMs: 3000,
        intervalSeconds: 30,
        apiKeyHeader: 'X-Config-Key',
        apiKey: 'config-secret',
        headers: {
          'x-source': 'gateway'
        }
      }
    });

    expect(config.configExternal).toEqual({
	      enabled: true,
	      transport: 'http',
	      endpoint: 'https://config.example.com/gateway',
	      command: undefined,
	      args: [],
	      cwd: undefined,
	      env: {},
	      method: 'POST',
      timeoutMs: 3000,
      intervalMs: 30000,
      apiKeyHeader: 'x-config-key',
      apiKey: 'config-secret',
      headers: {
        'x-source': 'gateway'
      }
    });
  });

  it('rejects unsupported external gateway config source transports', () => {
    expect(() =>
      parseGatewayConfigFromRaw({
        configExternal: {
          enabled: true,
          transport: 'mqtt',
          endpoint: 'mqtt://config.example.com'
        }
      })
    ).toThrow('configExternal.transport currently supports only "http", "websocket", "grpc", or "stdio".');
  });

  it('parses websocket, grpc, and stdio external gateway config sources', () => {
    const websocketConfig = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        endpoint: 'wss://config.example.com/gateway',
        timeoutMs: 3000
      }
    });
    expect(websocketConfig.configExternal?.transport).toBe('websocket');

    const grpcConfig = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        endpoint: 'grpc://config.example.com/gateway.config.v1.ConfigService/GetConfig',
        timeoutMs: 3000
      }
    });
    expect(grpcConfig.configExternal?.transport).toBe('grpc');

    const stdioConfig = parseGatewayConfigFromRaw({
      configExternal: {
        enabled: true,
        command: 'node',
        args: ['config-source.js'],
        cwd: '/tmp/config-source',
        env: {
          SOURCE_NAME: 'gateway'
        }
      }
    });
    expect(stdioConfig.configExternal?.transport).toBe('stdio');
    expect(stdioConfig.configExternal?.command).toBe('node');
    expect(stdioConfig.configExternal?.args).toEqual(['config-source.js']);
    expect(stdioConfig.configExternal?.cwd).toBe('/tmp/config-source');
    expect(stdioConfig.configExternal?.env).toEqual({
      SOURCE_NAME: 'gateway'
    });
  });

  it('parses agent event webhook config with environment secret header', () => {
    process.env.AGENT_EVENT_WEBHOOK_API_KEY_HEADER = 'X-Agent-Event-Key';
    process.env.AGENT_EVENT_WEBHOOK_API_KEY = 'agent-event-secret';

    const config = parseGatewayConfigFromRaw({
      agent: {
        eventWebhook: {
          enabled: true,
          endpoint: 'https://agent.example.com/events',
          timeoutMs: 2500,
          headers: {
            'x-source': 'gateway'
          }
        }
      }
    });

    expect(config.agent.eventWebhook).toEqual({
      enabled: true,
      transport: 'http',
      endpoint: 'https://agent.example.com/events',
      command: undefined,
      args: [],
      cwd: undefined,
      env: {},
      timeoutMs: 2500,
      maxAttempts: 3,
      baseDelayMs: 200,
      maxDelayMs: 2000,
      requireAck: false,
      headers: {
        'x-source': 'gateway',
        'x-agent-event-key': 'agent-event-secret'
      }
    });
  });

  it('infers websocket and grpc transport for event sink endpoints', () => {
    const config = parseGatewayConfigFromRaw({
      billingWebhook: {
        enabled: true,
        endpoint: 'wss://billing.example.com/events'
      },
      agent: {
        eventWebhook: {
          enabled: true,
          endpoint: 'grpc://agent.example.com/gateway.events.v1.EventSink/Publish'
        }
      }
    });

    expect(config.billingWebhook.transport).toBe('websocket');
    expect(config.agent.eventWebhook?.transport).toBe('grpc');
  });

  it('infers stdio transport for event sink commands', () => {
    const config = parseGatewayConfigFromRaw({
      billingWebhook: {
        enabled: true,
        command: 'node',
        args: ['sink.js'],
        cwd: '/tmp/billing-sink',
        env: {
          SINK_NAME: 'billing'
        }
      },
      agent: {
        eventWebhook: {
          enabled: true,
          command: 'node',
          args: ['agent-sink.js']
        }
      }
    });

    expect(config.billingWebhook.transport).toBe('stdio');
    expect(config.billingWebhook.command).toBe('node');
    expect(config.billingWebhook.args).toEqual(['sink.js']);
    expect(config.billingWebhook.cwd).toBe('/tmp/billing-sink');
    expect(config.billingWebhook.env).toEqual({
      SINK_NAME: 'billing'
    });
    expect(config.agent.eventWebhook?.transport).toBe('stdio');
    expect(config.agent.eventWebhook?.command).toBe('node');
    expect(config.agent.eventWebhook?.args).toEqual(['agent-sink.js']);
  });

  it('parses external event sink retry settings', () => {
    process.env.BILLING_WEBHOOK_MAX_ATTEMPTS = '5';
    process.env.BILLING_WEBHOOK_BASE_DELAY_MS = '25';
    process.env.BILLING_WEBHOOK_MAX_DELAY_MS = '250';
    process.env.BILLING_WEBHOOK_WEBSOCKET_REQUIRE_ACK = 'true';

    const config = parseGatewayConfigFromRaw({
      billingWebhook: {
        enabled: true,
        endpoint: 'https://billing.example.com/events',
        maxAttempts: 2,
        baseDelayMs: 10,
        maxDelayMs: 100
      },
      rawTrace: {
        sync: {
          enabled: true,
          endpoint: 'https://trace.example.com/manifests',
          maxAttempts: 4,
          baseDelaySeconds: 0.5,
          maxDelaySeconds: 5
        }
      },
      agent: {
        eventWebhook: {
          enabled: true,
          endpoint: 'https://agent.example.com/events',
          requireAck: true,
          maxAttempts: 6,
          baseDelayMs: 75,
          maxDelayMs: 750
        }
      }
    });

    expect(config.billingWebhook).toMatchObject({
      maxAttempts: 5,
      baseDelayMs: 25,
      maxDelayMs: 250,
      requireAck: true
    });
    expect(config.rawTrace.sync).toMatchObject({
      maxAttempts: 4,
      baseDelayMs: 500,
      maxDelayMs: 5000,
      requireAck: false
    });
    expect(config.agent.eventWebhook).toMatchObject({
      maxAttempts: 6,
      baseDelayMs: 75,
      maxDelayMs: 750,
      requireAck: true
    });
  });

  it('parses raw trace sync as an external protocol sink without S3 storage', () => {
    const grpcConfig = parseGatewayConfigFromRaw({
      rawTrace: {
        enabled: true,
        mode: 'body_full',
        sync: {
          enabled: true,
          endpoint: 'grpc://trace.example.com/gateway.rawtrace.v1.TraceSink/Publish',
          headers: {
            'x-trace-source': 'gateway'
          }
        }
      }
    });

    expect(grpcConfig.rawTrace.sync).toMatchObject({
      enabled: true,
      transport: 'grpc',
      endpoint: 'grpc://trace.example.com/gateway.rawtrace.v1.TraceSink/Publish',
      command: undefined,
      args: [],
      cwd: undefined,
      env: {},
      headers: {
        'x-trace-source': 'gateway'
      }
    });
    expect(grpcConfig.rawTrace).not.toHaveProperty('s3');

    const stdioConfig = parseGatewayConfigFromRaw({
      rawTrace: {
        enabled: true,
        sync: {
          enabled: true,
          command: 'node',
          args: ['trace-sink.js'],
          cwd: '/tmp/trace-sink',
          env: {
            TRACE_SINK: 'local'
          }
        }
      }
    });

    expect(stdioConfig.rawTrace.sync.transport).toBe('stdio');
    expect(stdioConfig.rawTrace.sync.command).toBe('node');
    expect(stdioConfig.rawTrace.sync.args).toEqual(['trace-sink.js']);
    expect(stdioConfig.rawTrace.sync.cwd).toBe('/tmp/trace-sink');
    expect(stdioConfig.rawTrace.sync.env).toEqual({
      TRACE_SINK: 'local'
    });
  });

  it('defaults agent storage to memory and only uses filesystem when explicit', () => {
    const defaultConfig = parseGatewayConfigFromRaw({});
    expect(defaultConfig.agent.storage).toEqual({
      type: 'memory'
    });

    const filesystemConfig = parseGatewayConfigFromRaw({
      agent: {
        storage: {
          type: 'filesystem',
          dir: '.agent-data'
        }
      }
    });
    expect(filesystemConfig.agent.storage).toEqual({
      type: 'filesystem',
      dir: '.agent-data'
    });

    expect(() =>
      parseGatewayConfigFromRaw({
        agent: {
          storage: {
            type: 'http',
            endpoint: 'https://agent-store.example.com'
          }
        }
      })
    ).toThrow('agent.storage.type=http has been removed. Use agent.external for external agent/session state.');

    expect(() =>
      parseGatewayConfigFromRaw({
        agent: {
          storage: {
            type: 'durable'
          }
        }
      })
    ).toThrow('agent.storage.type currently supports only "memory" or "filesystem".');
  });

  it('parses provider and agent external sources with protocol transports', () => {
    const config = parseGatewayConfigFromRaw({
      providerExternal: {
        enabled: true,
        endpoint: 'wss://provider-source.example.com/config'
      },
      agent: {
        external: {
          enabled: true,
          endpoint: 'grpc://agent-source.example.com/gateway.agent.v1.AgentStateSource/GetState'
        }
      }
    });

    expect(config.providerExternal).toMatchObject({
      enabled: true,
      transport: 'websocket',
      endpoint: 'wss://provider-source.example.com/config'
    });
    expect(config.agent.external).toMatchObject({
      enabled: true,
      transport: 'grpc',
      endpoint: 'grpc://agent-source.example.com/gateway.agent.v1.AgentStateSource/GetState'
    });

    const stdioConfig = parseGatewayConfigFromRaw({
      providerExternal: {
        enabled: true,
        command: 'node',
        args: ['provider-source.js']
      },
      agent: {
        external: {
          enabled: true,
          command: 'node',
          args: ['agent-source.js'],
          cwd: '/tmp/agent-source',
          env: {
            AGENT_SOURCE: 'local'
          }
        }
      }
    });

    expect(stdioConfig.providerExternal?.transport).toBe('stdio');
    expect(stdioConfig.providerExternal?.command).toBe('node');
    expect(stdioConfig.providerExternal?.args).toEqual(['provider-source.js']);
    expect(stdioConfig.agent.external?.transport).toBe('stdio');
    expect(stdioConfig.agent.external?.command).toBe('node');
    expect(stdioConfig.agent.external?.args).toEqual(['agent-source.js']);
    expect(stdioConfig.agent.external?.cwd).toBe('/tmp/agent-source');
    expect(stdioConfig.agent.external?.env).toEqual({
      AGENT_SOURCE: 'local'
    });
  });

  it('uses CODEX_REFRESH_TOKEN_URL_OVERRIDE as default token endpoint', () => {
    process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE = 'https://custom.example.com/oauth/token';

    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-api-key',
          models: ['glm-5']
        }
      ],
      providerPlugins: [
        {
          key: 'openai-main-codex-oauth',
          providerName: 'openai-main',
          codexOauth: {
            accessToken: {
              from: 'request.headers.x-codex-access-token'
            }
          }
        }
      ]
    });

    expect(config.providerPlugins?.[0]?.codexOauth?.tokenEndpoint).toBe(
      'https://custom.example.com/oauth/token'
    );
  });

  it('requires mcp prefix for mcp websocket endpoint from config', () => {
    expect(() =>
      parseGatewayConfigFromRaw({
        providers: [
          {
            name: 'openai-main',
            type: 'openai_responses',
            apikey: 'provider-api-key',
            models: ['glm-5']
          }
        ],
        mcpGateway: {
          websocket: {
            endpoint: '/ws'
          }
        }
      })
    ).toThrow('mcpGateway.websocket.endpoint must start with "/mcp" for path protection.');
  });

  it('requires mcp prefix for mcp websocket endpoint from env override', () => {
    process.env.MCP_GATEWAY_WS_ENDPOINT = '/ws';

    expect(() =>
      parseGatewayConfigFromRaw({
        providers: [
          {
            name: 'openai-main',
            type: 'openai_responses',
            apikey: 'provider-api-key',
            models: ['glm-5']
          }
        ],
        mcpGateway: {
          websocket: {
            endpoint: '/mcp/ws'
          }
        }
      })
    ).toThrow('mcpGateway.websocket.endpoint must start with "/mcp" for path protection.');
  });

  it('accepts mcp websocket endpoint with /mcp prefix', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-api-key',
          models: ['glm-5']
        }
      ],
      mcpGateway: {
        websocket: {
          endpoint: '/mcp/private/ws'
        }
      }
    });

    expect(config.mcpGateway.websocket.endpoint).toBe('/mcp/private/ws');
  });

  it('parses provider health, health-aware routing, and precheck config', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-api-key',
          models: ['gpt-test'],
          health: {
            status: 'degraded',
            available: true,
            priority: 2,
            latencyMs: 120,
            checkedAt: '2026-06-08T00:00:00.000Z'
          }
        }
      ],
      healthAwareRouting: {
        enabled: true,
        skipUnavailable: true,
        unhealthyStatuses: ['down', 'degraded'],
        preferHealthy: true,
        preferLowerLatency: false
      },
      providerHealthCheck: {
        enabled: true,
        intervalSeconds: 45,
        timeoutMs: 2500,
        initialDelaySeconds: 3
      },
      precheck: {
        enabled: true,
        rateLimit: {
          enabled: true,
          windowSeconds: 30,
          maxRequests: 10,
          rpm: 20,
          rpd: 2000,
          tpm: 10000,
          tpd: 100000,
          ipm: 30,
          subject: 'api-key',
          scope: 'provider',
          limits: [
            {
              name: 'custom-images-hour',
              metric: 'images',
              windowSeconds: 3600,
              max: 300,
              subject: 'tenant',
              scope: 'model'
            }
          ]
        },
        quota: {
          enabled: true,
          windowMs: 60000,
          maxTokens: 1000,
          subject: 'tenant',
          scope: 'model'
        },
        budget: {
          enabled: true,
          windowMs: 60000,
          maxCostUsd: 1.5,
          subject: 'organization',
          scope: 'provider-model'
        },
        estimation: {
          charsPerToken: 3,
          defaultMaxOutputTokens: 256
        },
        storage: {
          type: 'memory'
        }
      }
    });

    expect(config.providers[0]?.health).toEqual({
      status: 'degraded',
      available: true,
      priority: 2,
      latencyMs: 120,
      checkedAt: '2026-06-08T00:00:00.000Z'
    });
    expect(config.healthAwareRouting).toEqual({
      enabled: true,
      skipUnavailable: true,
      unhealthyStatuses: ['down', 'degraded'],
      preferHealthy: true,
      preferLowerLatency: false
    });
    expect(config.providerHealthCheck).toEqual({
      enabled: true,
      intervalMs: 45000,
      timeoutMs: 2500,
      initialDelayMs: 3000
    });
    expect(config.precheck.rateLimit).toMatchObject({
      enabled: true,
      windowMs: 30000,
      maxRequests: 10,
      rpm: 20,
      rpd: 2000,
      tpm: 10000,
      tpd: 100000,
      ipm: 30,
      subject: 'api_key',
      scope: 'provider'
    });
    expect(config.precheck.rateLimit.limits.map((limit) => limit.name)).toEqual([
      'requests',
      'rpm',
      'rpd',
      'tpm',
      'tpd',
      'ipm',
      'custom-images-hour'
    ]);
    expect(config.precheck.rateLimit.limits[3]).toMatchObject({
      name: 'tpm',
      metric: 'tokens',
      windowMs: 60000,
      max: 10000,
      subject: 'api_key',
      scope: 'provider'
    });
    expect(config.precheck.rateLimit.limits[6]).toMatchObject({
      name: 'custom-images-hour',
      metric: 'images',
      windowMs: 3600000,
      max: 300,
      subject: 'tenant',
      scope: 'model'
    });
    expect(config.precheck.quota).toMatchObject({
      enabled: true,
      maxTokens: 1000,
      subject: 'tenant',
      scope: 'model'
    });
    expect(config.precheck.budget).toMatchObject({
      enabled: true,
      maxCostUsd: 1.5,
      subject: 'organization',
      scope: 'provider_model'
    });
    expect(config.precheck.estimation).toEqual({
      charsPerToken: 3,
      defaultMaxOutputTokens: 256
    });
    expect(config.precheck.storage).toMatchObject({
      type: 'memory'
    });
  });

  it('parses provider health scheduler env overrides', () => {
    process.env.PROVIDER_HEALTH_CHECK_ENABLED = 'true';
    process.env.PROVIDER_HEALTH_CHECK_INTERVAL_MS = '30000';
    process.env.PROVIDER_HEALTH_CHECK_TIMEOUT_MS = '1200';
    process.env.PROVIDER_HEALTH_CHECK_INITIAL_DELAY_MS = '500';

    const config = parseGatewayConfigFromRaw({
      providerHealthCheck: {
        enabled: false,
        intervalMs: 60000,
        timeoutMs: 5000,
        initialDelayMs: 0
      }
    });

    expect(config.providerHealthCheck).toEqual({
      enabled: true,
      intervalMs: 30000,
      timeoutMs: 1200,
      initialDelayMs: 500
    });
  });

  it('parses metrics config and env overrides', () => {
    process.env.GATEWAY_METRICS_ENABLED = 'true';
    process.env.GATEWAY_METRICS_INCLUDE_PROVIDER_HEALTH = 'false';

    const config = parseGatewayConfigFromRaw({
      metrics: {
        enabled: false,
        includeProviderHealth: true
      }
    });

    expect(config.metrics).toEqual({
      enabled: true,
      includeProviderHealth: false
    });
  });

  it('parses cors config and env overrides', () => {
    process.env.GATEWAY_CORS_ORIGINS = 'https://env-console.example, https://env-admin.example';
    process.env.GATEWAY_CORS_ALLOWED_HEADERS = 'Content-Type, Authorization, X-Tenant';
    process.env.GATEWAY_CORS_ALLOWED_METHODS = 'GET, POST, OPTIONS';
    process.env.GATEWAY_CORS_ALLOW_CREDENTIALS = 'true';
    process.env.GATEWAY_CORS_MAX_AGE_SECONDS = '600';

    const config = parseGatewayConfigFromRaw({
      cors: {
        enabled: true,
        origins: ['https://file.example'],
        allowedHeaders: ['Content-Type'],
        allowedMethods: ['GET'],
        allowCredentials: false,
        maxAgeSeconds: 60
      }
    });

    expect(config.cors).toEqual({
      enabled: true,
      origins: ['https://env-console.example', 'https://env-admin.example'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Tenant'],
      allowedMethods: ['GET', 'POST', 'OPTIONS'],
      allowCredentials: true,
      maxAgeSeconds: 600
    });
  });

  it('parses idempotency config and env overrides', () => {
    process.env.GATEWAY_IDEMPOTENCY_ENABLED = 'true';
    process.env.GATEWAY_IDEMPOTENCY_HEADER = 'X-Request-Idempotency-Key';
    process.env.GATEWAY_IDEMPOTENCY_TTL_MS = '120000';
    process.env.GATEWAY_IDEMPOTENCY_MAX_ENTRIES = '25';
    process.env.GATEWAY_IDEMPOTENCY_CACHE_ERROR_RESPONSES = 'true';

    const config = parseGatewayConfigFromRaw({
      idempotency: {
        enabled: false,
        headerName: 'idempotency-key',
        ttlMs: 1000,
        maxEntries: 10,
        cacheErrorResponses: false
      }
    });

    expect(config.idempotency).toEqual({
      enabled: true,
      headerName: 'X-Request-Idempotency-Key',
      ttlMs: 120000,
      maxEntries: 25,
      cacheErrorResponses: true
    });
  });

  it('parses upstream concurrency config and env overrides', () => {
    process.env.GATEWAY_UPSTREAM_CONCURRENCY_ENABLED = 'true';
    process.env.GATEWAY_UPSTREAM_MAX_IN_FLIGHT_PER_PROVIDER = '3';
    process.env.GATEWAY_UPSTREAM_CONCURRENCY_QUEUE_TIMEOUT_MS = '250';

    const config = parseGatewayConfigFromRaw({
      upstreamConcurrency: {
        enabled: false,
        maxInFlightPerProvider: 10,
        queueTimeoutMs: 1000
      }
    });

    expect(config.upstreamConcurrency).toEqual({
      enabled: true,
      maxInFlightPerProvider: 3,
      queueTimeoutMs: 250
    });
  });

  it('parses upstream circuit breaker config and env overrides', () => {
    process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_ENABLED = 'true';
    process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = '2';
    process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_COOLDOWN_MS = '5000';
    process.env.GATEWAY_UPSTREAM_CIRCUIT_BREAKER_FAILURE_STATUS_CODES = '500,503,504';

    const config = parseGatewayConfigFromRaw({
      upstreamCircuitBreaker: {
        enabled: false,
        failureThreshold: 5,
        cooldownMs: 30000,
        failureStatusCodes: [429, 500]
      }
    });

    expect(config.upstreamCircuitBreaker).toEqual({
      enabled: true,
      failureThreshold: 2,
      cooldownMs: 5000,
      failureStatusCodes: [500, 503, 504]
    });
  });

  it('parses upstream retry config and env overrides', () => {
    process.env.GATEWAY_UPSTREAM_RETRY_ENABLED = 'true';
    process.env.GATEWAY_UPSTREAM_RETRY_MAX_ATTEMPTS = '4';
    process.env.GATEWAY_UPSTREAM_RETRY_BASE_DELAY_MS = '10';
    process.env.GATEWAY_UPSTREAM_RETRY_MAX_DELAY_MS = '80';
    process.env.GATEWAY_UPSTREAM_RETRY_BACKOFF_MULTIPLIER = '2';
    process.env.GATEWAY_UPSTREAM_RETRY_JITTER_MS = '5';
    process.env.GATEWAY_UPSTREAM_RETRY_STATUS_CODES = '429,500,503';

    const config = parseGatewayConfigFromRaw({
      upstreamRetry: {
        enabled: false,
        maxAttempts: 2,
        baseDelayMs: 150,
        maxDelayMs: 150,
        backoffMultiplier: 1,
        jitterMs: 0,
        retryStatusCodes: []
      }
    });

    expect(config.upstreamRetry).toEqual({
      enabled: true,
      maxAttempts: 4,
      baseDelayMs: 10,
      maxDelayMs: 80,
      backoffMultiplier: 2,
      jitterMs: 5,
      retryStatusCodes: [429, 500, 503]
    });
  });

  it('parses transparent tool execution config and env overrides', () => {
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ENABLED = 'true';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TURNS = '6';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_MAX_TOOL_CALLS = '12';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_REQUIRE_CLIENT_DECLARATION = 'false';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_UNKNOWN_TOOL_POLICY = 'fail';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_ALLOW_TOOLS = 'browser.*,search_web';
    process.env.GATEWAY_TRANSPARENT_TOOL_EXECUTION_DENY_TOOLS = 'danger.*';

    const config = parseGatewayConfigFromRaw({
      transparentToolExecution: {
        enabled: false,
        maxTurns: 2,
        maxToolCalls: 3,
        requireClientDeclaration: true,
        unknownToolPolicy: 'return_to_client',
        allowTools: ['file.*'],
        denyTools: ['shell.*']
      }
    });

    expect(config.transparentToolExecution).toEqual({
      enabled: true,
      maxTurns: 6,
      maxToolCalls: 12,
      requireClientDeclaration: false,
      unknownToolPolicy: 'fail',
      allowTools: ['browser.*', 'search_web'],
      denyTools: ['danger.*']
    });
  });

  it('parses gateway routing policy config and env defaults', () => {
    process.env.GATEWAY_POLICY_ALLOW_PROVIDERS = 'openai,anthropic';
    process.env.GATEWAY_POLICY_DENY_MODELS = 'blocked-from-env';

    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_responses',
          apikey: 'provider-api-key',
          models: ['gpt-test']
        }
      ],
      policy: {
        enabled: true,
        defaults: {
          denyProviders: ['gemini'],
          allowModels: ['gpt-*']
        },
        byTenant: {
          'tenant-a': {
            allowProviderNames: ['openai-main'],
            denyProviderModels: ['openai-main/gpt-test-bad']
          }
        },
        byPlan: {
          free: {
            denyModels: ['gpt-expensive']
          }
        }
      }
    });

    expect(config.policy.enabled).toBe(true);
    expect(config.policy.defaults).toMatchObject({
      allowProviders: ['openai', 'anthropic'],
      denyProviders: ['gemini'],
      allowModels: ['gpt-*'],
      denyModels: ['blocked-from-env']
    });
    expect(config.policy.byTenant['tenant-a']).toMatchObject({
      allowProviderNames: ['openai-main'],
      denyProviderModels: ['openai-main/gpt-test-bad']
    });
    expect(config.policy.byPlan.free).toMatchObject({
      denyModels: ['gpt-expensive']
    });
  });

  it('parses virtual model optimistic stream mode', () => {
    const config = parseGatewayConfigFromRaw({
      providers: [
        {
          name: 'openai-main',
          type: 'openai_chat_completions',
          apikey: 'provider-api-key',
          models: ['glm-5']
        }
      ],
      virtualModelProfiles: [
        {
          id: 'websearch',
          key: 'websearch',
          displayName: 'Web Search',
          match: {
            suffixes: [':websearch']
          },
          tools: [
            {
              name: 'web_search',
              visibility: 'internal'
            }
          ],
          execution: {
            mode: 'tool_loop',
            streamMode: 'optimistic'
          }
        },
        {
          id: 'buffered',
          key: 'buffered',
          displayName: 'Buffered',
          match: {
            suffixes: [':buffered']
          },
          tools: [],
          execution: {
            mode: 'tool_loop'
          }
        }
      ]
    });

    expect(config.virtualModelProfiles?.[0]?.execution.streamMode).toBe('optimistic');
    expect(config.virtualModelProfiles?.[1]?.execution.streamMode).toBe('buffered');
  });
});
