import type { ServerResponse } from 'node:http';
import type { FastifyInstance, FastifyReply, FastifyRequest, RouteShorthandOptions } from 'fastify';
import { createGatewayAuthPreHandler } from '../gateway/auth';
import type { GatewayAuthConfig, GatewayRequestIdentity } from '../types';
import { isObject, parseProvider } from '../utils';
import { mergeGatewayRequestIdentityMetadata } from './request-identity';
import type { EventDrivenAgentRuntime } from './runtime';
import type { AgentDefinition, AgentEventRecord, AgentEventType, AgentSessionState } from './types';

const EXTERNAL_EVENT_TYPES: AgentEventType[] = [
  'SESSION_CONFIG_UPDATED',
  'USER_INPUT',
  'TOOL_RESULT'
];
const DEFAULT_EVENT_LIMIT = 50;
const MAX_EVENT_LIMIT = 500;
const STREAM_BATCH_SIZE = 200;
const MANAGEMENT_DISABLED_MESSAGE =
  'Agent/session management API is disabled because data source is configured as external.';
const CORS_ALLOWED_HEADERS =
  'Content-Type, Authorization, X-API-Key, X-Codex-Access-Token';
const CORS_ALLOWED_METHODS = 'GET,POST,PUT,PATCH,DELETE,OPTIONS';

interface AgentRouteOptions {
  managementEnabled?: boolean;
  agentManagementEnabled?: boolean;
  sessionManagementEnabled?: boolean;
  authConfig?: GatewayAuthConfig;
}

export function registerAgentRoutes(
  fastify: FastifyInstance,
  runtime: EventDrivenAgentRuntime,
  options: AgentRouteOptions = {}
) {
  const defaultManagementEnabled = options.managementEnabled ?? true;
  const agentManagementEnabled =
    options.agentManagementEnabled ?? defaultManagementEnabled;
  const sessionManagementEnabled =
    options.sessionManagementEnabled ?? agentManagementEnabled;
  const activeResumeStreams = new Set<string>();
  const agentRouteOptions: RouteShorthandOptions = {};

  if (options.authConfig) {
    agentRouteOptions.preHandler = createGatewayAuthPreHandler(options.authConfig);
  }

  fastify.get('/agent/tools', agentRouteOptions, async () => {
    const tools = await runtime.listTools();
    return {
      tools
    };
  });

  fastify.get('/agent/agents', agentRouteOptions, async (request) => {
    const agents = filterAccessibleAgents(request, runtime.listAgents());
    return {
      agents
    };
  });

  fastify.post<{ Body: unknown }>('/agent/agents', agentRouteOptions, async (request, reply) => {
    if (!agentManagementEnabled) {
      return sendManagementDisabled(reply);
    }

    const body = request.body;
    if (!isObject(body)) {
      return sendBadRequest(reply, 'Request body must be a JSON object.');
    }

    const name = readRequiredString(body.name);
    if (!name) {
      return sendBadRequest(reply, 'Request body must include non-empty string field: name');
    }

    const description = readOptionalString(body.description);
    if (description === null) {
      return sendBadRequest(reply, 'Field description must be a string.');
    }
    const systemPrompt = readOptionalString(body.systemPrompt);
    if (systemPrompt === null) {
      return sendBadRequest(reply, 'Field systemPrompt must be a string.');
    }
    const model = readOptionalModelReference(body.model);
    if (model === null) {
      return sendBadRequest(reply, 'Field model must use format: provider/model');
    }

    const toolsValue = body.tools === undefined ? body.allowedTools : body.tools;
    const tools = readStrictStringArray(toolsValue);
    if (tools === null) {
      return sendBadRequest(reply, 'Field tools (or allowedTools) must be an array of strings.');
    }

    const agent =
      tools === undefined
        ? await runtime.createAgentWithAutoTools({
            name,
            description,
            ownerIdentity: request.gatewayIdentity,
            systemPrompt,
            model
          })
        : runtime.createAgent({
            name,
            description,
            ownerIdentity: request.gatewayIdentity,
            systemPrompt,
            model,
            allowedTools: tools
          });

    return reply.code(201).send({
      agent
    });
  });

  fastify.get<{ Params: { agentId: string } }>('/agent/agents/:agentId', agentRouteOptions, async (request, reply) => {
    const agent = getAccessibleAgent(request, runtime, request.params.agentId);
    if (!agent) {
      return reply.code(404).send({
        error: {
          message: `Agent not found: ${request.params.agentId}`
        }
      });
    }

    return {
      agent
    };
  });

  fastify.put<{ Params: { agentId: string }; Body: unknown }>('/agent/agents/:agentId', agentRouteOptions, async (request, reply) => {
    if (!agentManagementEnabled) {
      return sendManagementDisabled(reply);
    }

    const body = request.body;
    if (!isObject(body)) {
      return sendBadRequest(reply, 'Request body must be a JSON object.');
    }

    const name = readOptionalString(body.name);
    if (name === null) {
      return sendBadRequest(reply, 'Field name must be a string.');
    }
    const description = readOptionalString(body.description);
    if (description === null) {
      return sendBadRequest(reply, 'Field description must be a string.');
    }
    const systemPrompt = readOptionalString(body.systemPrompt);
    if (systemPrompt === null) {
      return sendBadRequest(reply, 'Field systemPrompt must be a string.');
    }
    const model = readOptionalModelReference(body.model);
    if (model === null) {
      return sendBadRequest(reply, 'Field model must use format: provider/model');
    }

    const toolsValue = body.tools === undefined ? body.allowedTools : body.tools;
    const tools = readStrictStringArray(toolsValue);
    if (tools === null) {
      return sendBadRequest(reply, 'Field tools (or allowedTools) must be an array of strings.');
    }

    const updateInput: Partial<{
      name: string;
      description?: string;
      systemPrompt?: string;
      model?: string;
      allowedTools: string[];
    }> = {};

    if (name !== undefined) updateInput.name = name;
    if (description !== undefined) updateInput.description = description;
    if (systemPrompt !== undefined) updateInput.systemPrompt = systemPrompt;
    if (model !== undefined) updateInput.model = model;
    if (tools !== undefined) updateInput.allowedTools = tools;

    if (!getAccessibleAgent(request, runtime, request.params.agentId)) {
      return reply.code(404).send({
        error: {
          message: `Agent not found: ${request.params.agentId}`
        }
      });
    }

    const agent = runtime.updateAgent(request.params.agentId, updateInput);
    if (!agent) {
      return reply.code(404).send({
        error: {
          message: `Agent not found: ${request.params.agentId}`
        }
      });
    }

    return {
      agent
    };
  });

  fastify.delete<{ Params: { agentId: string } }>('/agent/agents/:agentId', agentRouteOptions, async (request, reply) => {
    if (!agentManagementEnabled) {
      return sendManagementDisabled(reply);
    }

    if (!getAccessibleAgent(request, runtime, request.params.agentId)) {
      return reply.code(404).send({
        error: {
          message: `Agent not found: ${request.params.agentId}`
        }
      });
    }

    runtime.deleteAgent(request.params.agentId);
    return reply.code(204).send();
  });

  fastify.post<{ Body: unknown }>('/agent/sessions', agentRouteOptions, async (request, reply) => {
    if (!sessionManagementEnabled) {
      return sendManagementDisabled(reply);
    }

    const body = request.body;
    if (body !== undefined && !isObject(body)) {
      return sendBadRequest(reply, 'Request body must be a JSON object.');
    }

    const payload = isObject(body) ? body : {};
    const agentId = readOptionalString(payload.agentId);
    if (agentId === null) {
      return sendBadRequest(reply, 'Field agentId must be a string.');
    }
    const sessionId = readOptionalString(payload.sessionId);
    if (sessionId === null) {
      return sendBadRequest(reply, 'Field sessionId must be a string.');
    }
    const prompt = readOptionalString(payload.prompt);
    if (prompt === null) {
      return sendBadRequest(reply, 'Field prompt must be a string.');
    }
    const systemPrompt = readOptionalString(payload.systemPrompt);
    if (systemPrompt === null) {
      return sendBadRequest(reply, 'Field systemPrompt must be a string.');
    }
    const model = readOptionalModelReference(payload.model);
    if (model === null) {
      return sendBadRequest(reply, 'Field model must use format: provider/model');
    }
    const correlationId = readOptionalString(payload.correlationId);
    if (correlationId === null) {
      return sendBadRequest(reply, 'Field correlationId must be a string.');
    }

    const metadata = payload.metadata;
    if (metadata !== undefined && !isObject(metadata)) {
      return sendBadRequest(reply, 'Field metadata must be a JSON object.');
    }
    const mergedMetadata = mergeGatewayRequestIdentityMetadata(
      isObject(metadata) ? metadata : undefined,
      request.gatewayIdentity
    );

    const stream = readOptionalBoolean(payload.stream);
    if (stream === null) {
      return sendBadRequest(reply, 'Field stream must be a boolean.');
    }

    const toolsValue = payload.tools === undefined ? payload.allowedTools : payload.tools;
    const tools = readStrictStringArray(toolsValue);
    if (tools === null) {
      return sendBadRequest(reply, 'Field tools (or allowedTools) must be an array of strings.');
    }

    const memoryRefs = readStrictStringArray(payload.memoryRefs);
    if (memoryRefs === null) {
      return sendBadRequest(reply, 'Field memoryRefs must be an array of strings.');
    }

    if (agentId && !getAccessibleAgent(request, runtime, agentId)) {
      return reply.code(404).send({
        error: {
          message: `Agent not found: ${agentId}`
        }
      });
    }

    const result = runtime.createSession({
      agentId,
      sessionId,
      prompt,
      metadata: mergedMetadata,
      ownerIdentity: request.gatewayIdentity,
      correlationId,
      systemPrompt,
      model,
      allowedTools: tools,
      memoryRefs
    });

    if (!result.ok) {
      if (result.error === 'AGENT_NOT_FOUND') {
        const message = agentId
          ? `Agent not found: ${agentId}`
          : 'Agent not found. Create an agent first or provide agentId.';
        return reply.code(404).send({
          error: {
            message
          }
        });
      }

      return reply.code(409).send({
        error: {
          message: `Session already exists: ${result.sessionId || sessionId || 'unknown'}`
        }
      });
    }

    const shouldStream = resolveStreamPreference(stream);
    if (shouldStream) {
      const correlationId = result.initialEvent?.correlationId;
      const streamStartedAt = Date.now();
      request.log.info(
        {
          sessionId: result.session.sessionId,
          correlationId,
          mode: 'created'
        },
        'Agent session stream started.'
      );
      startSessionEventStream(request, reply, runtime, result.session.sessionId, 0, {
        mode: 'created',
        createdAt: result.createdAt,
        agentId: result.agent.agentId
      }, () => {
        request.log.info(
          {
            sessionId: result.session.sessionId,
            correlationId,
            mode: 'created',
            durationMs: Date.now() - streamStartedAt
          },
          'Agent session stream closed.'
        );
        if (correlationId) {
          runtime.abortCorrelation(correlationId);
        }
      }, true);
      return reply;
    }

    return reply.code(201).send({
      sessionId: result.session.sessionId,
      agentId: result.session.agentId,
      session: result.session,
      initialEvent: result.initialEvent,
      events: sortEventsByOffset(runtime.listEventsAfter(result.session.sessionId, 0, STREAM_BATCH_SIZE))
    });
  });

  fastify.get('/agent/sessions', agentRouteOptions, async (request) => {
    const sessions = filterAccessibleSessions(request, runtime.listSessions());
    return {
      sessions
    };
  });

  fastify.post<{ Params: { sessionId: string }; Body: unknown }>(
    '/agent/sessions/:sessionId/resume',
    agentRouteOptions,
    async (request, reply) => {
      if (!sessionManagementEnabled) {
        return sendManagementDisabled(reply);
      }

      const sessionId = request.params.sessionId;
      const session = getAccessibleSession(request, runtime, sessionId);
      if (!session) {
        return reply.code(404).send({
          error: {
            message: `Session not found: ${sessionId}`
          }
        });
      }

      const body = request.body;
      if (body !== undefined && !isObject(body)) {
        return sendBadRequest(reply, 'Request body must be a JSON object.');
      }

      const payload = isObject(body) ? body : {};
      const prompt = readOptionalString(payload.prompt);
      if (prompt === null) {
        return sendBadRequest(reply, 'Field prompt must be a string.');
      }
      const correlationId = readOptionalString(payload.correlationId);
      if (correlationId === null) {
        return sendBadRequest(reply, 'Field correlationId must be a string.');
      }
      const metadata = payload.metadata;
      if (metadata !== undefined && !isObject(metadata)) {
        return sendBadRequest(reply, 'Field metadata must be a JSON object.');
      }
      const mergedMetadata = mergeGatewayRequestIdentityMetadata(
        isObject(metadata) ? metadata : undefined,
        request.gatewayIdentity
      );

      const fromOffset = readOptionalOffset(payload.fromOffset);
      if (fromOffset === null) {
        return sendBadRequest(reply, 'Field fromOffset must be a non-negative integer.');
      }

      const stream = readOptionalBoolean(payload.stream);
      if (stream === null) {
        return sendBadRequest(reply, 'Field stream must be a boolean.');
      }

      const normalizedFromOffset = fromOffset ?? 0;
      const shouldStream = resolveStreamPreference(stream);
      if (shouldStream && activeResumeStreams.has(sessionId)) {
        return reply.code(409).send({
          error: {
            message: `Session is already resumed by another request: ${sessionId}`
          }
        });
      }

      if (shouldStream) {
        activeResumeStreams.add(sessionId);
      }

      let acceptedEvent: ReturnType<EventDrivenAgentRuntime['publishEvent']> | undefined;
      try {
        acceptedEvent = prompt
          ? runtime.publishEvent({
              sessionId,
              type: 'USER_INPUT',
              payload: {
                text: prompt,
                metadata: mergedMetadata
              },
              correlationId
            })
          : undefined;
      } catch (error) {
        if (shouldStream) {
          activeResumeStreams.delete(sessionId);
        }
        throw error;
      }

      if (shouldStream) {
        const correlationId = acceptedEvent?.correlationId;
        const streamStartedAt = Date.now();
        request.log.info(
          {
            sessionId,
            correlationId,
            mode: 'resumed',
            fromOffset: normalizedFromOffset
          },
          'Agent session stream started.'
        );
        try {
          startSessionEventStream(
            request,
            reply,
            runtime,
            sessionId,
            normalizedFromOffset,
            {
              mode: 'resumed',
              resumedAt: new Date().toISOString()
            },
            () => {
              request.log.info(
                {
                  sessionId,
                  correlationId,
                  mode: 'resumed',
                  durationMs: Date.now() - streamStartedAt
                },
                'Agent session stream closed.'
              );
              activeResumeStreams.delete(sessionId);
              if (correlationId) {
                runtime.abortCorrelation(correlationId);
              }
            },
            true
          );
        } catch (error) {
          activeResumeStreams.delete(sessionId);
          throw error;
        }
        return reply;
      }

      return {
        session: getAccessibleSession(request, runtime, sessionId),
        acceptedEvent,
        events: sortEventsByOffset(runtime.listEventsAfter(sessionId, normalizedFromOffset, STREAM_BATCH_SIZE))
      };
    }
  );

  fastify.get<{ Params: { sessionId: string }; Querystring: { fromOffset?: string } }>(
    '/agent/sessions/:sessionId/stream',
    agentRouteOptions,
    async (request, reply) => {
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return reply.code(404).send({
          error: {
            message: `Session not found: ${request.params.sessionId}`
          }
        });
      }

      const fromOffset = parseOffset(request.query.fromOffset);
      startSessionEventStream(request, reply, runtime, request.params.sessionId, fromOffset, {
        mode: 'stream'
      });
      return reply;
    }
  );

  fastify.get<{ Params: { sessionId: string } }>('/agent/sessions/:sessionId', agentRouteOptions, async (request, reply) => {
    const session = getAccessibleSession(request, runtime, request.params.sessionId);
    if (!session) {
      return reply.code(404).send({
        error: {
          message: `Session not found: ${request.params.sessionId}`
        }
      });
    }

    return {
      session
    };
  });

  fastify.delete<{ Params: { sessionId: string } }>('/agent/sessions/:sessionId', agentRouteOptions, async (request, reply) => {
    if (!sessionManagementEnabled) {
      return sendManagementDisabled(reply);
    }

    if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
      return reply.code(404).send({
        error: {
          message: `Session not found: ${request.params.sessionId}`
        }
      });
    }

    runtime.deleteSession(request.params.sessionId);
    return reply.code(204).send();
  });

  fastify.get<{ Params: { sessionId: string }; Querystring: { limit?: string; afterOffset?: string } }>(
    '/agent/sessions/:sessionId/events',
    agentRouteOptions,
    async (request, reply) => {
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return sendSessionNotFound(reply, request.params.sessionId);
      }

      const limit = parseLimit(request.query.limit);
      const afterOffset = parseOffset(request.query.afterOffset);
      const hasAfterOffset = request.query.afterOffset !== undefined;
      return {
        events:
          hasAfterOffset
            ? runtime.listEventsAfter(request.params.sessionId, afterOffset, limit)
            : runtime.listEvents(request.params.sessionId, limit)
      };
    }
  );

  fastify.post<{ Params: { sessionId: string }; Body: unknown }>(
    '/agent/sessions/:sessionId/input',
    agentRouteOptions,
    async (request, reply) => {
      if (!sessionManagementEnabled) {
        return sendManagementDisabled(reply);
      }
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return sendSessionNotFound(reply, request.params.sessionId);
      }

      const body = request.body;
      if (!isObject(body) || typeof body.text !== 'string') {
        return sendBadRequest(reply, 'Request body must include string field: text');
      }

      const event = runtime.publishEvent({
        sessionId: request.params.sessionId,
        type: 'USER_INPUT',
        payload: {
          text: body.text,
          metadata: mergeGatewayRequestIdentityMetadata(
            isObject(body.metadata) ? body.metadata : undefined,
            request.gatewayIdentity
          )
        },
        correlationId: typeof body.correlationId === 'string' ? body.correlationId : undefined
      });

      return reply.code(202).send({
        accepted: true,
        event
      });
    }
  );

  fastify.post<{ Params: { sessionId: string }; Body: unknown }>(
    '/agent/sessions/:sessionId/config',
    agentRouteOptions,
    async (request, reply) => {
      if (!sessionManagementEnabled) {
        return sendManagementDisabled(reply);
      }
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return sendSessionNotFound(reply, request.params.sessionId);
      }

      const body = request.body;
      if (!isObject(body)) {
        return sendBadRequest(reply, 'Request body must be a JSON object.');
      }

      const systemPrompt = typeof body.systemPrompt === 'string' ? body.systemPrompt : undefined;
      const model = readOptionalModelReference(body.model);
      if (model === null) {
        return sendBadRequest(reply, 'Field model must use format: provider/model');
      }
      const allowedTools = readStringArray(body.allowedTools);
      const memoryRefs = readStringArray(body.memoryRefs);

      if (systemPrompt === undefined && model === undefined && allowedTools === undefined && memoryRefs === undefined) {
        return sendBadRequest(
          reply,
          'At least one field is required: systemPrompt, model, allowedTools, memoryRefs'
        );
      }

      const event = runtime.publishEvent({
        sessionId: request.params.sessionId,
        type: 'SESSION_CONFIG_UPDATED',
        payload: {
          systemPrompt,
          model,
          allowedTools,
          memoryRefs
        },
        correlationId: typeof body.correlationId === 'string' ? body.correlationId : undefined
      });

      return reply.code(202).send({
        accepted: true,
        event
      });
    }
  );

  fastify.post<{ Params: { sessionId: string }; Body: unknown }>(
    '/agent/sessions/:sessionId/tool-result',
    agentRouteOptions,
    async (request, reply) => {
      if (!sessionManagementEnabled) {
        return sendManagementDisabled(reply);
      }
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return sendSessionNotFound(reply, request.params.sessionId);
      }

      const body = request.body;
      if (!isObject(body)) {
        return sendBadRequest(reply, 'Request body must be a JSON object.');
      }

      if (typeof body.toolCallId !== 'string' || typeof body.toolName !== 'string') {
        return sendBadRequest(reply, 'Request body must include string fields: toolCallId, toolName');
      }

      const status =
        body.status === 'ok' || body.status === 'error'
          ? body.status
          : typeof body.error === 'string'
          ? 'error'
          : 'ok';
      const event = runtime.publishEvent({
        sessionId: request.params.sessionId,
        type: 'TOOL_RESULT',
        payload: {
          toolCallId: body.toolCallId,
          toolName: body.toolName,
          status,
          result: body.result,
          error: typeof body.error === 'string' ? body.error : undefined
        },
        correlationId: typeof body.correlationId === 'string' ? body.correlationId : undefined
      });

      return reply.code(202).send({
        accepted: true,
        event
      });
    }
  );

  fastify.post<{ Params: { sessionId: string }; Body: unknown }>(
    '/agent/sessions/:sessionId/events',
    agentRouteOptions,
    async (request, reply) => {
      if (!sessionManagementEnabled) {
        return sendManagementDisabled(reply);
      }
      if (!getAccessibleSession(request, runtime, request.params.sessionId)) {
        return sendSessionNotFound(reply, request.params.sessionId);
      }

      const body = request.body;
      if (!isObject(body)) {
        return sendBadRequest(reply, 'Request body must be a JSON object.');
      }

      if (typeof body.type !== 'string' || !isExternalEventType(body.type)) {
        return sendBadRequest(
          reply,
          `Unsupported event type. Allowed types: ${EXTERNAL_EVENT_TYPES.join(', ')}`
        );
      }

      const eventPayload =
        body.type === 'USER_INPUT' && isObject(body.payload)
          ? {
              ...body.payload,
              metadata: mergeGatewayRequestIdentityMetadata(
                isObject(body.payload.metadata) ? body.payload.metadata : undefined,
                request.gatewayIdentity
              )
            }
          : body.payload;

      const event = runtime.publishEvent({
        sessionId: request.params.sessionId,
        type: body.type,
        payload: eventPayload,
        correlationId: typeof body.correlationId === 'string' ? body.correlationId : undefined,
        causationId: typeof body.causationId === 'string' ? body.causationId : undefined
      });

      return reply.code(202).send({
        accepted: true,
        event
      });
    }
  );
}

function filterAccessibleAgents(request: FastifyRequest, agents: AgentDefinition[]): AgentDefinition[] {
  return agents.filter((agent) => hasResourceAccess(agent.ownerIdentity, request.gatewayIdentity));
}

function getAccessibleAgent(
  request: FastifyRequest,
  runtime: EventDrivenAgentRuntime,
  agentId: string
): AgentDefinition | undefined {
  const agent = runtime.getAgent(agentId);
  if (!agent || !hasResourceAccess(agent.ownerIdentity, request.gatewayIdentity)) {
    return undefined;
  }

  return agent;
}

function filterAccessibleSessions(request: FastifyRequest, sessions: AgentSessionState[]): AgentSessionState[] {
  return sessions.filter((session) => hasResourceAccess(session.ownerIdentity, request.gatewayIdentity));
}

function getAccessibleSession(
  request: FastifyRequest,
  runtime: EventDrivenAgentRuntime,
  sessionId: string
): AgentSessionState | undefined {
  const session = runtime.getSession(sessionId);
  if (!session || !hasResourceAccess(session.ownerIdentity, request.gatewayIdentity)) {
    return undefined;
  }

  return session;
}

function hasResourceAccess(
  ownerIdentity: GatewayRequestIdentity | undefined,
  requestIdentity: GatewayRequestIdentity | undefined
): boolean {
  if (!ownerIdentity?.billingSubjectKey) {
    return true;
  }

  return ownerIdentity.billingSubjectKey === requestIdentity?.billingSubjectKey;
}

function startSessionEventStream(
  request: FastifyRequest,
  reply: FastifyReply,
  runtime: EventDrivenAgentRuntime,
  sessionId: string,
  fromOffset: number,
  metadata: Record<string, unknown>,
  onClose?: () => void,
  closeAfterReply = false
) {
  reply.hijack();
  const response = reply.raw;
  const rawRequest = request.raw;
  response.statusCode = 200;
  response.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  response.setHeader('Cache-Control', 'no-cache, no-transform');
  response.setHeader('Connection', 'keep-alive');
  response.setHeader('Access-Control-Allow-Origin', resolveCorsOrigin());
  response.setHeader('Access-Control-Allow-Headers', CORS_ALLOWED_HEADERS);
  response.setHeader('Access-Control-Allow-Methods', CORS_ALLOWED_METHODS);
  response.setHeader('Access-Control-Max-Age', '86400');
  response.setHeader('X-Accel-Buffering', 'no');
  response.flushHeaders?.();

  let closed = false;
  let lastOffset = Math.max(0, fromOffset);
  let flushing = false;
  let pendingFlush = false;
  let unsubscribe: (() => void) | undefined;
  const handleRequestAborted = () => {
    close();
  };
  const handleResponseClose = () => {
    close();
  };
  const handleResponseError = () => {
    close();
  };

  const close = () => {
    if (closed) {
      return;
    }

    closed = true;
    clearInterval(heartbeatTimer);
    rawRequest.off('aborted', handleRequestAborted);
    response.off('close', handleResponseClose);
    response.off('error', handleResponseError);
    unsubscribe?.();
    try {
      onClose?.();
    } catch {
      // Ignore close-hook errors so stream cleanup can continue.
    }
    if (!response.writableEnded) {
      response.end();
    }
  };

  const emit = (eventName: string, data: unknown, id?: number | string) => {
    if (closed) {
      return;
    }

    const written = writeSseEvent(response, eventName, data, id);
    if (!written) {
      close();
    }
  };

  const flushEvents = () => {
    if (closed) {
      return;
    }

    if (flushing) {
      pendingFlush = true;
      return;
    }

    flushing = true;
    try {
      while (!closed) {
        const events = sortEventsByOffset(runtime.listEventsAfter(sessionId, lastOffset, STREAM_BATCH_SIZE));
        if (events.length === 0) {
          break;
        }

        for (const event of events) {
          if (event.offset <= lastOffset) {
            continue;
          }
          lastOffset = event.offset;
          emit('event', event, event.offset);
          if (event.type === 'AGENT_REPLY') {
            if (closeAfterReply) {
              close();
              break;
            }
          }
          if (closed) {
            break;
          }
        }

        if (events.length < STREAM_BATCH_SIZE) {
          break;
        }
      }
    } finally {
      flushing = false;
      if (pendingFlush && !closed) {
        pendingFlush = false;
        flushEvents();
      } else {
        pendingFlush = false;
      }
    }
  };

  const heartbeatTimer = setInterval(() => {
    emit('ping', {
      sessionId,
      timestamp: new Date().toISOString(),
      lastOffset
    });
  }, 15_000);
  heartbeatTimer.unref?.();

  emit('session', {
    sessionId,
    fromOffset: lastOffset,
    ...metadata
  });
  flushEvents();

  if (closed) {
    return;
  }

  unsubscribe = runtime.subscribeSessionEvents(sessionId, () => {
    flushEvents();
  });

  rawRequest.once('aborted', handleRequestAborted);
  response.once('close', handleResponseClose);
  response.once('error', handleResponseError);
}

function writeSseEvent(
  response: ServerResponse,
  eventName: string,
  data: unknown,
  id?: number | string
): boolean {
  if (response.writableEnded || response.destroyed) {
    return false;
  }

  try {
    if (id !== undefined) {
      response.write(`id: ${String(id)}\n`);
    }
    response.write(`event: ${eventName}\n`);

    const serialized = JSON.stringify(data ?? null);
    const lines = serialized.split('\n');
    for (const line of lines) {
      response.write(`data: ${line}\n`);
    }
    response.write('\n');
    return true;
  } catch {
    return false;
  }
}

function isExternalEventType(value: string): value is AgentEventType {
  return EXTERNAL_EVENT_TYPES.includes(value as AgentEventType);
}

function sendBadRequest(reply: FastifyReply, message: string) {
  return reply.code(400).send({ error: { message } });
}

function sendManagementDisabled(reply: FastifyReply) {
  return reply.code(405).send({
    error: {
      message: MANAGEMENT_DISABLED_MESSAGE
    }
  });
}

function sendSessionNotFound(reply: FastifyReply, sessionId: string) {
  return reply.code(404).send({
    error: {
      message: `Session not found: ${sessionId}`
    }
  });
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const items = value.filter((item): item is string => typeof item === 'string').map((item) => item.trim());
  const deduped = [...new Set(items.filter(Boolean))];
  return deduped;
}

function readStrictStringArray(value: unknown): string[] | undefined | null {
  if (value === undefined) {
    return undefined;
  }

  if (!Array.isArray(value)) {
    return null;
  }

  if (!value.every((item) => typeof item === 'string')) {
    return null;
  }

  const normalized = value.map((item) => item.trim()).filter(Boolean);
  return [...new Set(normalized)];
}

function readRequiredString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
}

function readOptionalString(value: unknown): string | undefined | null {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'string') {
    return null;
  }

  return readRequiredString(value);
}

function readOptionalBoolean(value: unknown): boolean | undefined | null {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'boolean') {
    return null;
  }

  return value;
}

function readOptionalModelReference(value: unknown): string | undefined | null {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'string') {
    return null;
  }

  const normalized = normalizeProviderModelReference(value);
  return normalized || null;
}

function normalizeProviderModelReference(value: string): string | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }

  const separator = trimmed.indexOf('/');
  if (separator <= 0 || separator === trimmed.length - 1) {
    return undefined;
  }

  const providerToken = trimmed.slice(0, separator).trim();
  const modelToken = trimmed.slice(separator + 1).trim();
  if (!providerToken || !modelToken) {
    return undefined;
  }

  const provider = parseProvider(providerToken);
  const normalizedProvider = provider || providerToken;
  return `${normalizedProvider}/${modelToken}`;
}

function readOptionalOffset(value: unknown): number | undefined | null {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }

  return Math.max(0, Math.floor(value));
}

function parseLimit(value: string | undefined): number {
  if (!value) {
    return DEFAULT_EVENT_LIMIT;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_EVENT_LIMIT;
  }

  return Math.max(1, Math.min(MAX_EVENT_LIMIT, Math.floor(parsed)));
}

function parseOffset(value: string | undefined): number {
  if (!value) {
    return 0;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }

  return Math.max(0, Math.floor(parsed));
}

function resolveStreamPreference(stream: boolean | undefined): boolean {
  if (stream !== undefined) {
    return stream;
  }

  return true;
}

function resolveCorsOrigin(): string {
  const fromGatewayEnv = process.env.GATEWAY_CORS_ORIGIN?.trim();
  if (fromGatewayEnv) {
    return fromGatewayEnv;
  }

  const fromSharedEnv = process.env.CORS_ORIGIN?.trim();
  if (fromSharedEnv) {
    return fromSharedEnv;
  }

  return '*';
}

function sortEventsByOffset<TPayload = unknown>(events: AgentEventRecord<TPayload>[]): AgentEventRecord<TPayload>[] {
  if (events.length < 2) {
    return events;
  }

  return [...events].sort((left, right) => left.offset - right.offset);
}
