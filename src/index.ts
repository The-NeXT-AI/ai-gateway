import { randomUUID } from 'node:crypto';
import Fastify from 'fastify';
import {
  closeAgentEventPublisher,
  createAgentRuntime,
  createAgentEventBus,
  createMcpAgentToolProvider,
  initializeAgentEventPublisher,
  publishAgentEventToExternalSink,
  registerAgentRoutes
} from './agent';
import { closeBillingPublisher, initializeBillingPublisher } from './billing';
import { config } from './config';
import {
  refreshGatewayConfigFromExternalSource,
  startGatewayExternalConfigPoller,
  type GatewayExternalConfigPoller
} from './external-config';
import { buildCorsResponseHeaders } from './gateway/cors';
import { registerGatewayIdempotencyHooks } from './gateway/idempotency';
import { registerLenientJsonParser } from './gateway/lenient-json-parser';
import { recordGatewayHttpRequest, renderGatewayMetrics } from './gateway/metrics';
import { registerGatewayRoutes } from './gateway/routes';
import { createGatewayRuntime } from './gateway/runtime';
import { registerGatewayResponsesWebSocketRoute } from './gateway/websocket';
import { closeGatewayPrecheckStore } from './gateway/precheck';
import {
  closeProviderHealthScheduler,
  initializeProviderHealthScheduler
} from './gateway/provider-health-scheduler';
import { registerManagerRoutes } from './manager';
import { hydrateProvidersFromExternalSource, isProviderExternalSourceEnabled } from './provider/external';
import { closeCodexOauthStateStore, syncProviderPluginsFromConfig } from './provider/plugins';
import { registerProviderWebhookRoutes } from './provider/webhook';
import {
  createMcpGatewayRuntime,
  registerMcpGatewayRoutes,
  registerMcpGatewayWebSocketRoute
} from './mcp-gateway';
import { closeRawTraceManager, initializeRawTraceManager } from './raw-trace';
import type { GatewayConfig } from './types';

const codexResponsesWebSocketPath = '/v1/responses';
const metricsRequestStarts = new WeakMap<object, bigint>();

const fastify = Fastify({
  logger: true,
  genReqId: () => randomUUID(),
  bodyLimit: config.bodyLimitBytes
});
fastify.addHook('onRequest', async (request, reply) => {
  metricsRequestStarts.set(request, process.hrtime.bigint());
  applyCorsHeaders(reply, request.headers.origin);
  if (request.method.toUpperCase() === 'OPTIONS') {
    return reply.code(204).send();
  }
});
fastify.addHook('onSend', async (request, reply, payload) => {
  applyCorsHeaders(reply, request.headers.origin);
  return payload;
});
fastify.addHook('onResponse', async (request, reply) => {
  if (!config.metrics.enabled) {
    return;
  }

  const startedAt = metricsRequestStarts.get(request);
  metricsRequestStarts.delete(request);
  if (!startedAt) {
    return;
  }

  recordGatewayHttpRequest({
    method: request.method,
    route: resolveMetricsRouteLabel(request),
    statusCode: reply.statusCode,
    durationMs: Number(process.hrtime.bigint() - startedAt) / 1_000_000
  });
});
registerLenientJsonParser(fastify);
registerGatewayIdempotencyHooks(fastify, config);
const agentEventBus = createAgentEventBus({
  queueConfig: config.agent.eventQueue,
  runtimeConfig: config.agent.runtime,
  onSubscriberError: (error, event) => {
    fastify.log.error(
      {
        eventId: event.id,
        type: event.type,
        sessionId: event.sessionId,
        details: error instanceof Error ? error.message : String(error)
      },
      'Agent event subscriber failed.'
    );
  },
  onPublishError: (error, event) => {
    fastify.log.warn(
      {
        eventId: event.id,
        type: event.type,
        sessionId: event.sessionId,
        details: error instanceof Error ? error.message : String(error)
      },
      'Failed to enqueue agent event into runtime queue.'
    );
  },
  onQueueError: (error) => {
    fastify.log.warn(
      {
        details: error instanceof Error ? error.message : String(error)
      },
      'Agent runtime event queue worker error.'
    );
  }
});
const agentToolProvider = createMcpAgentToolProvider({
  servers: config.agent.mcpServers,
  exposureMode: 'passthrough',
  logger: fastify.log
});
const runtime = createGatewayRuntime(config, agentToolProvider);
const agentRuntime = createAgentRuntime({
  eventBus: agentEventBus,
  logger: fastify.log,
  toolProvider: agentToolProvider,
  storage: config.agent.storage,
  config
});
const unsubscribeAgentEventQueueSubscriber = agentEventBus.subscribe((event) => {
  void publishAgentEventToExternalSink(event).catch((error) => {
    fastify.log.warn(
      {
        eventId: event.id,
        type: event.type,
        sessionId: event.sessionId,
        details: error instanceof Error ? error.message : String(error)
      },
      'Failed to deliver agent event to external publisher.'
    );
  });
});
const mcpGatewayRuntime = createMcpGatewayRuntime({
  config: config.mcpGateway,
  servers: config.agent.mcpServers,
  logger: fastify.log
});
let gatewayExternalConfigPoller: GatewayExternalConfigPoller | undefined;

if (
  config.mcpGateway.enabled &&
  config.mcpGateway.websocket.enabled &&
  normalizeRoutePath(config.mcpGateway.websocket.endpoint) === codexResponsesWebSocketPath
) {
  throw new Error(
    `mcpGateway.websocket.endpoint cannot be ${codexResponsesWebSocketPath}; MCP WebSocket and Codex responses WebSocket must use different endpoints.`
  );
}
const agentManagementEnabled = !config.agent.external?.enabled;
const agentSessionManagementEnabled = true;
const providerManagementEnabled = !isProviderExternalSourceEnabled(config);

fastify.addHook('onClose', async () => {
  unsubscribeAgentEventQueueSubscriber();
  await closeAgentEventPublisher();
  await closeBillingPublisher();
  await closeRawTraceManager();
  gatewayExternalConfigPoller?.close();
  gatewayExternalConfigPoller = undefined;
  closeProviderHealthScheduler();
  await closeGatewayPrecheckStore();
  await closeCodexOauthStateStore();
  await mcpGatewayRuntime.close();
  await agentRuntime.close();
});

fastify.get('/health', async () => {
  return {
    runtimeId: process.env.CCR_GATEWAY_RUNTIME_ID,
    status: 'ok',
    timestamp: new Date().toISOString()
  };
});

fastify.get('/metrics', async (_request, reply) => {
  if (!config.metrics.enabled) {
    return reply.code(404).send({
      error: {
        message: 'Metrics endpoint is disabled.'
      }
    });
  }

  return reply
    .header('content-type', 'text/plain; version=0.0.4; charset=utf-8')
    .send(renderGatewayMetrics(config));
});

fastify.get('/', async () => {
  const agentReadEndpoints = [
    'GET /agent/tools',
    'GET /agent/agents',
    'GET /agent/agents/:agentId',
    'GET /agent/sessions',
    'GET /agent/sessions/:sessionId',
    'GET /agent/sessions/:sessionId/stream',
    'GET /agent/sessions/:sessionId/events'
  ];
  const agentManageEndpoints = [
    'POST /agent/agents',
    'PUT /agent/agents/:agentId',
    'DELETE /agent/agents/:agentId'
  ];
  const agentSessionManageEndpoints = [
    'POST /agent/sessions',
    'DELETE /agent/sessions/:sessionId',
    'POST /agent/sessions/:sessionId/resume',
    'POST /agent/sessions/:sessionId/input',
    'POST /agent/sessions/:sessionId/config',
    'POST /agent/sessions/:sessionId/tool-result',
    'POST /agent/sessions/:sessionId/events'
  ];

  return {
    name: 'next-ai-gateway',
    standard_model: 'openai_responses',
    endpoints: [
      'POST /v1/responses',
      'WS /v1/responses',
      'GET /v1/models',
      'GET /v1/models/:model',
      'POST /v1/chat/completions',
      'POST /v1/embeddings',
      'POST /v1/moderations',
      'POST /v1/images/generations',
      config.metrics.enabled ? 'GET /metrics' : undefined,
      'POST /v1/messages',
      'POST /v1beta/models/:model:generateContent',
      config.mcpGateway.enabled ? `POST ${config.mcpGateway.endpoint}` : undefined,
      config.mcpGateway.enabled && config.mcpGateway.websocket.enabled
        ? `WS ${config.mcpGateway.websocket.endpoint}`
        : undefined,
      ...agentReadEndpoints,
      ...(agentManagementEnabled ? agentManageEndpoints : []),
      ...(agentSessionManagementEnabled ? agentSessionManageEndpoints : []),
      ...(agentManagementEnabled
        ? [
            'GET /manager/config',
            'POST /manager/config/validate',
            'PUT /manager/config',
            'GET /manager/providers/health',
            'POST /manager/providers/health/check'
          ]
        : []),
      ...(agentManagementEnabled && providerManagementEnabled ? ['GET /manager/providers'] : [])
    ].filter((item): item is string => Boolean(item))
  };
});

registerGatewayRoutes(fastify, config, runtime);
registerGatewayResponsesWebSocketRoute(fastify, config, runtime);
registerAgentRoutes(fastify, agentRuntime, {
  agentManagementEnabled,
  sessionManagementEnabled: agentSessionManagementEnabled,
  authConfig: config.auth
});
registerMcpGatewayRoutes(fastify, mcpGatewayRuntime);
registerMcpGatewayWebSocketRoute(fastify, mcpGatewayRuntime);
registerProviderWebhookRoutes(fastify, {
  config,
  onConfigReload: async (nextConfig) => {
    await reloadRuntimeFromConfig(nextConfig);
  },
  onAgentRefresh: async (reason) => {
    await agentRuntime.refreshFromStorage(reason);
  }
});
if (agentManagementEnabled) {
  registerManagerRoutes(fastify, {
    config,
    beforeApplyConfig: async (nextConfig) => {
      await hydrateProvidersFromExternalSource(nextConfig, fastify.log);
    },
    onConfigReload: async (nextConfig) => {
      await reloadRuntimeFromConfig(nextConfig);
    }
  });
}

const start = async () => {
  try {
    await refreshGatewayConfigFromExternalSource({
      config,
      logger: fastify.log,
      onConfigReload: applyStaticRuntimeConfig,
      reason: 'external_config_startup'
    });
    await hydrateProvidersFromExternalSource(config, fastify.log);
    applyStaticRuntimeConfig(config);
    await agentRuntime.initialize();

    try {
      await initializeBillingPublisher(config.billingQueue, config.billingWebhook, fastify.log);
    } catch (error) {
      fastify.log.warn(
        {
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to initialize billing publishers. Gateway will continue without billing event delivery.'
      );
    }

    try {
      await initializeRawTraceManager(config.rawTrace, fastify.log);
    } catch (error) {
      fastify.log.warn(
        {
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to initialize raw trace manager. Gateway will continue without raw trace delivery.'
      );
    }

    try {
      await initializeAgentEventPublisher(config.agent.eventQueue, config.agent.eventWebhook, fastify.log);
    } catch (error) {
      fastify.log.warn(
        {
          details: error instanceof Error ? error.message : String(error)
        },
        'Failed to initialize agent event publisher. Gateway will continue without agent event delivery.'
      );
    }

    gatewayExternalConfigPoller = startGatewayExternalConfigPoller({
      config,
      logger: fastify.log,
      onConfigReload: async (nextConfig) => {
        await hydrateProvidersFromExternalSource(nextConfig, fastify.log);
        await reloadRuntimeFromConfig(nextConfig);
      }
    });

    await fastify.listen({ host: config.host, port: config.port });
    fastify.log.info(`Gateway running at http://${config.host}:${config.port}`);
  } catch (error) {
    fastify.log.error(error);
    process.exit(1);
  }
};

start();

async function reloadRuntimeFromConfig(nextConfig: GatewayConfig): Promise<void> {
  applyStaticRuntimeConfig(nextConfig);
  await initializeBillingPublisher(nextConfig.billingQueue, nextConfig.billingWebhook, fastify.log);
  await initializeAgentEventPublisher(nextConfig.agent.eventQueue, nextConfig.agent.eventWebhook, fastify.log);
  await initializeRawTraceManager(nextConfig.rawTrace, fastify.log);
}

function applyStaticRuntimeConfig(nextConfig: GatewayConfig): void {
  syncProviderPluginsFromConfig(runtime.providerPlugins, nextConfig);
  initializeProviderHealthScheduler(nextConfig, fastify.log);
}

function normalizeRoutePath(path: string): string {
  const normalized = path.trim().replace(/\/+$/, '');
  return normalized || '/';
}

function applyCorsHeaders(reply: {
  header: (name: string, value: string) => unknown;
}, requestOrigin: string | string[] | undefined): void {
  for (const [name, value] of Object.entries(buildCorsResponseHeaders(config.cors, requestOrigin))) {
    reply.header(name, value);
  }
}

function resolveMetricsRouteLabel(request: { url: string; routeOptions?: { url?: string } }): string {
  return request.routeOptions?.url || request.url.split('?')[0] || 'unknown';
}
