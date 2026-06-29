import { randomUUID } from 'node:crypto';
import { ConcurrencyLimiter } from './concurrency-limiter';
import type {
  AgentExternalSourceConfig,
  AgentRetryPolicyConfig,
  AgentStorageConfig,
  GatewayConfig,
  GatewayRequestIdentity,
  Provider,
  ProviderConfig
} from '../types';
import { asString, isObject, parseProvider, trimTrailingSlash } from '../utils';
import { requestExternalJson } from '../external-json-source';
import { readUpstreamPayload } from '../upstream/client';
import { type AgentEventBus, InMemoryAgentEventBus } from './event-bus';
import {
  applySessionMemoryRefsToGuards,
  applyToolFailureToGuards,
  applyUserInputToGuards,
  shouldBlockToolCall
} from './guards';
import { createAgentPersistenceStore, type AgentPersistenceStore } from './persistence';
import { SessionLockAcquireTimeoutError, SessionLockManager } from './session-lock';
import {
  buildProviderRequest,
  buildStreamingProviderRequest,
  callProvider,
  callProviderStreaming,
  parseProviderOutput,
  parseProviderStreamChunks,
  summarizeUpstreamErrorPayload,
  type ProviderCallResult,
  type ProviderRequest
} from './provider-adapter';
import { findLatestGatewayRequestIdentityInMessages } from './request-identity';
import {
  buildModelInputText,
  buildSystemPrompt,
  parseTriggerToolResult,
  type ParsedTriggerToolResult
} from './prompt-builder';
import {
  formatRouteLabel,
  parseProviderModelReference,
  resolveProviderRoutes,
  routeMatchesModelReference,
  type ProviderRoute,
  type ProviderModelReference
} from './provider-router';
import { isRetryableError, isRetryableHttpStatus, runWithRetry } from './retry';
import { InMemoryAgentSessionStore, type PersistedSessionSnapshot } from './store';
import {
  applyAgentReplyToTaskState,
  applyErrorToTaskState,
  applySessionConfigToTaskState,
  applyToolCallToTaskState,
  applyToolResultToTaskState,
  applyUserInputToTaskState,
  createInitialTaskState
} from './task-state';
import {
  recordAssistantReply,
  recordError,
  recordToolCall,
  recordToolResult,
  recordUserInput
} from './transcript-window';
import { createMcpAgentToolProvider, type AgentToolProvider } from './tools';
import type {
  AgentDefinition,
  AgentEvent,
  AgentEventRecord,
  AgentEventType,
  AgentModelClient,
  AgentModelOutput,
  AgentReplyPayload,
  AgentRuntimeLogger,
  AgentSessionState,
  AgentToolDefinition,
  ErrorPayload,
  SessionConfigUpdatedPayload,
  ToolCallRequestedPayload,
  ToolResultPayload,
  UserInputPayload
} from './types';

interface ExternalStateSnapshot {
  agents: unknown[];
  sessions: unknown[];
}

interface CorrelationLoopState {
  turnCount: number;
  lastTriggerEventId: string;
  lastTriggerEventType: AgentEventType;
}

const DEFAULT_SYSTEM_PROMPT =
  'You are an event-driven AI agent. Keep track of session context, use tools when needed, and produce concise replies.';
const ABORTED_CORRELATION_TTL_MS = 5 * 60_000;
const SLOW_STAGE_WARN_MS = 2_000;
const SLOW_TASK_WARN_MS = 10_000;
const DEFAULT_SESSION_LOCK_TIMEOUT_MS = 15_000;
const DEFAULT_STREAM_CHUNK_IDLE_TIMEOUT_MS = 5_000;
const MIN_STREAM_CHUNK_IDLE_TIMEOUT_MS = 1_000;
const DEFAULT_LLM_RETRY_POLICY: AgentRetryPolicyConfig = {
  maxAttempts: 3,
  baseDelayMs: 200,
  maxDelayMs: 2_000,
  backoffMultiplier: 2,
  jitterMs: 100
};
const DEFAULT_TOOL_RETRY_POLICY: AgentRetryPolicyConfig = {
  maxAttempts: 2,
  baseDelayMs: 150,
  maxDelayMs: 1_500,
  backoffMultiplier: 2,
  jitterMs: 50
};

export interface CreateAgentInput {
  name: string;
  description?: string;
  ownerIdentity?: GatewayRequestIdentity;
  systemPrompt?: string;
  model?: string;
  allowedTools?: string[];
}

export interface CreateAgentSessionInput {
  agentId?: string;
  sessionId?: string;
  prompt?: string;
  metadata?: Record<string, unknown>;
  ownerIdentity?: GatewayRequestIdentity;
  correlationId?: string;
  systemPrompt?: string;
  model?: string;
  allowedTools?: string[];
  memoryRefs?: string[];
}

export type CreateAgentSessionResult =
  | {
      ok: true;
      session: AgentSessionState;
      agent: AgentDefinition;
      createdAt: string;
      initialEvent?: AgentEvent;
    }
  | {
      ok: false;
      error: 'AGENT_NOT_FOUND' | 'SESSION_EXISTS';
      sessionId?: string;
    };

export interface PublishAgentEventInput {
  sessionId: string;
  type: AgentEventType;
  payload: unknown;
  correlationId?: string;
  causationId?: string;
}

export interface CreateAgentRuntimeOptions {
  eventBus?: AgentEventBus;
  store?: InMemoryAgentSessionStore;
  toolProvider?: AgentToolProvider;
  modelClient?: AgentModelClient;
  config: GatewayConfig;
  maxTranscriptMessages?: number;
  storage?: AgentStorageConfig;
  persistenceStore?: AgentPersistenceStore;
  logger?: AgentRuntimeLogger;
}

export class EventDrivenAgentRuntime {
  private readonly eventBus: AgentEventBus;
  private readonly store: InMemoryAgentSessionStore;
  private readonly toolProvider: AgentToolProvider;
  private readonly modelClient?: AgentModelClient;
  private readonly config: GatewayConfig;
  private readonly maxTranscriptMessages: number;
  private readonly gatewayBaseUrl: string;
  private readonly logger?: AgentRuntimeLogger;
  private readonly unsubscribe: () => void;
  private readonly storage: AgentStorageConfig;
  private readonly externalSource?: AgentExternalSourceConfig;
  private persistenceStore?: AgentPersistenceStore;
  private readonly agents = new Map<string, AgentDefinition>();
  private readonly sessionLocks = new SessionLockManager();
  private llmConcurrencyLimiter!: ConcurrencyLimiter;
  private readonly inflightAbortControllers = new Map<string, Set<AbortController>>();
  private readonly abortedCorrelations = new Map<string, number>();
  private readonly correlationTaskStartedAt = new Map<string, number>();
  private readonly detachedTasks = new Set<Promise<void>>();
  private readonly llmRetryPolicy: AgentRetryPolicyConfig;
  private readonly toolRetryPolicy: AgentRetryPolicyConfig;
  private readonly sessionLockTimeoutMs: number;
  private readonly correlationLoopStates = new Map<string, CorrelationLoopState>();
  private initializePromise?: Promise<void>;
  private initialized = false;
  private shuttingDown = false;
  private persistenceWriteQueue: Promise<void> = Promise.resolve();

  constructor(options: CreateAgentRuntimeOptions) {
    this.logger = options.logger;
    this.config = options.config;
    this.maxTranscriptMessages = Math.max(4, options.maxTranscriptMessages ?? 24);
    this.gatewayBaseUrl = this.resolveAgentGatewayBaseUrl();
    this.llmConcurrencyLimiter = new ConcurrencyLimiter(10);
    const runtimeConfig = resolveAgentRuntimeConfig(options.config);
    this.sessionLockTimeoutMs = runtimeConfig.sessionLockTimeoutMs;
    this.llmRetryPolicy = runtimeConfig.llmRetry;
    this.toolRetryPolicy = runtimeConfig.toolRetry;
    this.modelClient = options.modelClient;

    this.toolProvider =
      options.toolProvider ||
      createMcpAgentToolProvider({
        servers: [],
        logger: options.logger
      });

    this.store =
      options.store ||
      new InMemoryAgentSessionStore({
        defaultSystemPrompt: DEFAULT_SYSTEM_PROMPT,
        defaultAllowedTools: []
      });

    this.eventBus =
      options.eventBus ||
      new InMemoryAgentEventBus({
        onSubscriberError: (error, event) => {
          this.logger?.error?.(
            {
              eventId: event.id,
              type: event.type,
              sessionId: event.sessionId,
              details: toErrorMessage(error)
            },
            'Agent event subscriber failed.'
          );
        }
      });

    this.storage = options.storage || {
      type: 'memory'
    };
    this.externalSource = options.config.agent?.external?.enabled
      ? options.config.agent.external
      : undefined;
    this.persistenceStore = options.persistenceStore;
    this.unsubscribe = this.eventBus.subscribe((event) => this.handleEvent(event));
  }

  async close(): Promise<void> {
    this.shuttingDown = true;
    if (this.initializePromise) {
      try {
        await this.initializePromise;
      } catch {
        // ignore initialize failures during shutdown
      }
    }

    this.abortInflightOperations();
    await this.waitForDetachedTasks();

    if (!this.externalSource) {
      this.persistAgents();
      this.persistSessions();
    }
    await this.flushPersistenceWrites();
    await this.persistenceStore?.close?.();
    this.unsubscribe();
    await this.eventBus.close();
    this.sessionLocks.clear();
    await this.toolProvider.close();
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (this.initializePromise) {
      return this.initializePromise;
    }

    this.initializePromise = this.initializeInternal();
    try {
      await this.initializePromise;
    } finally {
      this.initializePromise = undefined;
    }
  }

  async refreshFromStorage(reason?: string): Promise<void> {
    if (this.externalSource) {
      await this.hydrateFromExternalSource(this.externalSource);
      this.logger?.info?.(
        {
          mode: 'external',
          reason: normalizeOptionalString(reason)
        },
        'Agent runtime refreshed from external source.'
      );
      return;
    }

    if (!this.persistenceStore) {
      this.persistenceStore = await createAgentPersistenceStore(this.storage, this.logger);
    }

    const [storedAgents, storedSessions] = await Promise.all([
      this.persistenceStore.loadAgents(),
      this.persistenceStore.loadSessions()
    ]);

    const normalizedAgents = new Map<string, AgentDefinition>();
    for (const agent of storedAgents) {
      const normalized = normalizeAgentDefinition(agent);
      if (!normalized) {
        continue;
      }

      normalizedAgents.set(normalized.agentId, normalized);
    }

    this.agents.clear();
    for (const [agentId, agent] of normalizedAgents.entries()) {
      this.agents.set(agentId, agent);
    }
    this.store.importSessions(storedSessions);

    this.logger?.info?.(
      {
        mode: this.storage.type,
        reason: normalizeOptionalString(reason),
        agents: this.agents.size,
        sessions: storedSessions.length
      },
      'Agent runtime refreshed from storage.'
    );
  }

  publishEvent(input: PublishAgentEventInput): AgentEvent {
    const event = createEvent({
      sessionId: input.sessionId,
      type: input.type,
      payload: input.payload,
      correlationId: input.correlationId,
      causationId: input.causationId
    });

    if (!this.shuttingDown) {
      this.eventBus.publish(event);
    }
    return event;
  }

  createAgent(input: CreateAgentInput): AgentDefinition {
    const timestamp = new Date().toISOString();
    const normalizedAllowedTools = dedupeStringArray(input.allowedTools);
    const agent: AgentDefinition = {
      agentId: randomUUID(),
      name: input.name.trim(),
      description: normalizeOptionalString(input.description),
      ownerIdentity: cloneGatewayRequestIdentity(input.ownerIdentity),
      systemPrompt: normalizeOptionalString(input.systemPrompt) || DEFAULT_SYSTEM_PROMPT,
      model: normalizeProviderModelReference(input.model),
      allowedTools: normalizedAllowedTools,
      allowedToolsConfigured: input.allowedTools !== undefined ? true : normalizedAllowedTools.length > 0,
      createdAt: timestamp,
      updatedAt: timestamp
    };
    this.agents.set(agent.agentId, agent);
    this.persistAgents();
    return cloneAgent(agent);
  }

  async createAgentWithAutoTools(input: CreateAgentInput): Promise<AgentDefinition> {
    if (input.allowedTools !== undefined) {
      const explicitTools = dedupeStringArray(input.allowedTools);
      return this.createAgent({
        ...input,
        allowedTools: explicitTools
      });
    }

    try {
      const availableTools = await this.toolProvider.listDefinitions();
      return this.createAgent({
        ...input,
        allowedTools: availableTools.map((tool) => tool.name)
      });
    } catch (error) {
      this.logger?.warn?.(
        {
          details: toErrorMessage(error)
        },
        'Failed to auto-register MCP tools for agent. Creating agent without tools.'
      );
      return this.createAgent(input);
    }
  }

  listAgents(): AgentDefinition[] {
    return [...this.agents.values()].map((agent) => cloneAgent(agent));
  }

  getAgent(agentId: string): AgentDefinition | undefined {
    const agent = this.agents.get(agentId);
    if (!agent) {
      return undefined;
    }

    return cloneAgent(agent);
  }

  updateAgent(agentId: string, input: Partial<Omit<CreateAgentInput, 'name'>> & { name?: string }): AgentDefinition | undefined {
    const agent = this.agents.get(agentId);
    if (!agent) {
      return undefined;
    }

    const timestamp = new Date().toISOString();
    const updatedAgent: AgentDefinition = {
      ...agent,
      name: input.name !== undefined ? input.name.trim() : agent.name,
      description: input.description !== undefined ? normalizeOptionalString(input.description) : agent.description,
      systemPrompt: input.systemPrompt !== undefined
        ? (normalizeOptionalString(input.systemPrompt) || DEFAULT_SYSTEM_PROMPT)
        : agent.systemPrompt,
      model: input.model !== undefined ? normalizeProviderModelReference(input.model) : agent.model,
      allowedTools: input.allowedTools !== undefined ? dedupeStringArray(input.allowedTools) : agent.allowedTools,
      allowedToolsConfigured:
        input.allowedTools !== undefined
          ? true
          : (agent.allowedToolsConfigured ?? agent.allowedTools.length > 0),
      updatedAt: timestamp
    };

    this.agents.set(agentId, updatedAgent);
    this.persistAgents();
    return cloneAgent(updatedAgent);
  }

  deleteAgent(agentId: string): boolean {
    if (!this.agents.has(agentId)) {
      return false;
    }

    this.agents.delete(agentId);
    this.persistAgents();
    return true;
  }

  listSessions(): AgentSessionState[] {
    const snapshots = this.store.exportSessions();
    return snapshots.map(snapshot => snapshot.state);
  }

  createSession(input: CreateAgentSessionInput): CreateAgentSessionResult {
    const requestedAgentId = normalizeOptionalString(input.agentId);
    const agent = requestedAgentId
      ? this.agents.get(requestedAgentId)
      : this.pickFallbackSessionAgent(input.ownerIdentity);
    if (!agent || !canAccessOwnerIdentity(agent.ownerIdentity, input.ownerIdentity)) {
      return {
        ok: false,
        error: 'AGENT_NOT_FOUND'
      };
    }

    const providedSessionId = normalizeOptionalString(input.sessionId);
    const sessionId = providedSessionId || randomUUID();
    if (this.store.hasSession(sessionId)) {
      return {
        ok: false,
        error: 'SESSION_EXISTS',
        sessionId
      };
    }

    const createdAt = new Date().toISOString();
    const customAllowedTools = dedupeStringArray(input.allowedTools);
    const customAllowedToolsProvided = input.allowedTools !== undefined;
    const customMemoryRefs = dedupeStringArray(input.memoryRefs);
    const customModel = normalizeProviderModelReference(input.model);
    const inheritedAllowedToolsConfigured = agent.allowedToolsConfigured ?? agent.allowedTools.length > 0;

    this.store.ensureSession(sessionId);
    this.store.setSessionAgent(sessionId, agent.agentId, createdAt);
    this.store.setSessionOwner(sessionId, input.ownerIdentity, createdAt);
    this.store.updateSessionConfig(
      sessionId,
      {
        systemPrompt: normalizeOptionalString(input.systemPrompt) || agent.systemPrompt,
        model: customModel || agent.model,
        allowedTools: customAllowedToolsProvided ? customAllowedTools : agent.allowedTools,
        allowedToolsConfigured: customAllowedToolsProvided ? true : inheritedAllowedToolsConfigured,
        memoryRefs: customMemoryRefs
      },
      createdAt
    );
    this.persistSessionSnapshot(sessionId);

    let initialEvent: AgentEvent | undefined;
    const prompt = normalizeOptionalString(input.prompt);
    if (prompt) {
      initialEvent = this.publishEvent({
        sessionId,
        type: 'USER_INPUT',
        payload: {
          text: prompt,
          metadata: isObject(input.metadata) ? input.metadata : undefined
        },
        correlationId: normalizeOptionalString(input.correlationId)
      });
    }

    const session = this.store.snapshotSession(sessionId);
    if (!session) {
      return {
        ok: false,
        error: 'SESSION_EXISTS',
        sessionId
      };
    }

    return {
      ok: true,
      session,
      agent: cloneAgent(agent),
      createdAt,
      initialEvent
    };
  }

  getSession(sessionId: string): AgentSessionState | undefined {
    return this.store.snapshotSession(sessionId);
  }

  hasSession(sessionId: string): boolean {
    return this.store.hasSession(sessionId);
  }

  deleteSession(sessionId: string): boolean {
    const deleted = this.store.deleteSession(sessionId);
    if (deleted) {
      this.deletePersistedSessionSnapshot(sessionId);
    }
    return deleted;
  }

  supportsSessionManagement(): boolean {
    return true;
  }

  listEvents(sessionId: string, limit = 50): AgentEventRecord[] {
    return this.store.listEvents(sessionId, limit);
  }

  listEventsAfter(sessionId: string, afterOffset = 0, limit = 500): AgentEventRecord[] {
    return this.store.listEventsAfter(sessionId, afterOffset, limit);
  }

  getLastEventOffset(sessionId: string): number {
    return this.store.getLastEventOffset(sessionId);
  }

  subscribeSessionEvents(sessionId: string, onEvent: (event: AgentEvent) => void): () => void {
    return this.eventBus.subscribe((event) => {
      if (event.sessionId !== sessionId) {
        return;
      }

      onEvent(event);
    });
  }

  abortCorrelation(correlationId: string): void {
    const normalized = normalizeOptionalString(correlationId);
    if (!normalized) {
      return;
    }

    this.cleanupAbortedCorrelations();
    this.abortedCorrelations.set(normalized, Date.now());
    this.finalizeCorrelationTask(normalized, 'aborted');

    const controllers = this.inflightAbortControllers.get(normalized);
    if (!controllers) {
      return;
    }

    for (const controller of controllers) {
      controller.abort();
    }
  }

  async listTools(): Promise<AgentToolDefinition[]> {
    return this.toolProvider.listDefinitions();
  }

  private async initializeInternal() {
    if (this.externalSource) {
      await this.hydrateFromExternalSource(this.externalSource);
      this.initialized = true;
      return;
    }

    if (!this.persistenceStore) {
      this.persistenceStore = await createAgentPersistenceStore(this.storage, this.logger);
    }

    await this.hydrateFromPersistence();
    this.persistAgents();
    this.persistSessions();
    await this.flushPersistenceWrites();
    this.initialized = true;
  }

  private async hydrateFromPersistence() {
    if (!this.persistenceStore) {
      return;
    }

    try {
      const storedAgents = await this.persistenceStore.loadAgents();
      for (const agent of storedAgents) {
        const normalized = normalizeAgentDefinition(agent);
        if (!normalized) {
          continue;
        }

        this.agents.set(normalized.agentId, normalized);
      }
    } catch (error) {
      this.logger?.warn?.(
        {
          details: toErrorMessage(error)
        },
        'Failed to load persisted agents. Continue with in-memory defaults.'
      );
    }

    try {
      const storedSessions = await this.persistenceStore.loadSessions();
      this.store.importSessions(storedSessions);
    } catch (error) {
      this.logger?.warn?.(
        {
          details: toErrorMessage(error)
        },
        'Failed to load persisted sessions. Continue with empty session store.'
      );
    }
  }

  private async hydrateFromExternalSource(source: AgentExternalSourceConfig) {
    try {
      const snapshot = await fetchExternalStateSnapshot(source, this.logger);
      const normalizedSessions = normalizeExternalSessionSnapshots(snapshot.sessions);

      this.agents.clear();
      for (const agent of snapshot.agents) {
        const normalized = normalizeAgentDefinition(agent);
        if (!normalized) {
          continue;
        }

        this.agents.set(normalized.agentId, normalized);
      }
      this.store.importSessions(normalizedSessions);

      this.logger?.info?.(
        {
          endpoint: source.endpoint,
          agents: this.agents.size,
          sessions: normalizedSessions.length
        },
        'Loaded agent/session data from external endpoint.'
      );
    } catch (error) {
      this.logger?.error?.(
        {
          endpoint: source.endpoint,
          details: toErrorMessage(error)
        },
        'Failed to load agent/session data from external endpoint.'
      );
      throw error;
    }
  }

  private persistAgents() {
    if (this.externalSource) {
      return;
    }

    const snapshot = this.listAgents();
    this.queuePersistenceWrite(async () => {
      if (!this.persistenceStore) {
        return;
      }

      await this.persistenceStore.saveAgents(snapshot);
    });
  }

  private persistSessions() {
    if (this.externalSource) {
      return;
    }

    const snapshot = this.store.exportSessions();
    this.queuePersistenceWrite(async () => {
      if (!this.persistenceStore) {
        return;
      }

      await this.persistenceStore.saveSessions(snapshot);
    });
  }

  private persistSessionSnapshot(sessionId: string) {
    if (this.externalSource) {
      const snapshot = this.store.exportSession(sessionId);
      if (!snapshot) {
        return;
      }

      this.queuePersistenceWrite(async () => {
        await upsertExternalSessionSnapshot(
          this.externalSource as AgentExternalSourceConfig,
          snapshot,
          this.logger
        );
      });
      return;
    }

    this.persistSessions();
  }

  private deletePersistedSessionSnapshot(sessionId: string) {
    if (this.externalSource) {
      this.queuePersistenceWrite(async () => {
        await deleteExternalSessionSnapshot(
          this.externalSource as AgentExternalSourceConfig,
          sessionId,
          this.logger
        );
      });
      return;
    }

    this.persistSessions();
  }

  private queuePersistenceWrite(task: () => Promise<void>) {
    this.persistenceWriteQueue = this.persistenceWriteQueue
      .then(async () => {
        await task();
      })
      .catch((error) => {
        this.logger?.warn?.(
          {
            details: toErrorMessage(error)
          },
          'Agent persistence write failed.'
        );
      });
  }

  private async flushPersistenceWrites() {
    try {
      await this.persistenceWriteQueue;
    } catch {
      return;
    }
  }

  private pickFallbackSessionAgent(ownerIdentity?: GatewayRequestIdentity): AgentDefinition | undefined {
    if (this.agents.size === 0) {
      return undefined;
    }

    const sorted = [...this.agents.values()]
      .filter((agent) => canAccessOwnerIdentity(agent.ownerIdentity, ownerIdentity))
      .sort((a, b) => {
        const aTime = a.createdAt || '';
        const bTime = b.createdAt || '';
        if (aTime !== bTime) {
          return aTime < bTime ? -1 : 1;
        }
        return a.agentId < b.agentId ? -1 : 1;
      });

    return sorted[0];
  }

  private async handleEvent(event: AgentEvent): Promise<void> {
    if (this.shuttingDown) {
      return;
    }

    const eventStartedAt = Date.now();
    const accepted = this.store.appendEvent(event);
    if (!accepted) {
      return;
    }

    if (event.type === 'USER_INPUT') {
      this.startCorrelationTask(event);
    }

    if (this.isCorrelationAborted(event.correlationId)) {
      this.persistSessionSnapshot(event.sessionId);
      this.logDuration(
        'event.total',
        eventStartedAt,
        {
          eventId: event.id,
          type: event.type,
          sessionId: event.sessionId,
          correlationId: event.correlationId,
          status: 'aborted'
        },
        SLOW_STAGE_WARN_MS
      );
      return;
    }

    const abortController = this.registerInflightAbortController(event.correlationId);
    try {
      switch (event.type) {
        case 'SESSION_CONFIG_UPDATED':
          await this.processSessionConfigEvent(event);
          break;
        case 'USER_INPUT':
          await this.processUserInputEvent(event, abortController.signal);
          break;
        case 'TOOL_CALL_REQUESTED':
          // Tool execution runs detached to avoid blocking the event bus
          await this.processToolCallRequestedEvent(event, abortController.signal);
          break;
        case 'TOOL_RESULT':
          await this.processToolResultEvent(event, abortController.signal);
          break;
        case 'CONTEXT_REQUESTED':
        case 'CONTEXT_HYDRATED':
          // Context hydration is no longer part of the active runtime model.
          break;
        case 'AGENT_REPLY':
          this.processAgentReplyEvent(event);
          break;
        case 'AGENT_REPLY_CHUNK':
          // Chunks are only logged; session messages are updated on AGENT_REPLY
          break;
        case 'ERROR':
          this.processErrorEvent(event);
          break;
      }
    } catch (error) {
      if (isAbortError(error) || this.isCorrelationAborted(event.correlationId)) {
        return;
      }

      this.logger?.error?.(
        {
          eventId: event.id,
          type: event.type,
          sessionId: event.sessionId,
          details: toErrorMessage(error)
        },
        'Agent event processing failed.'
      );

      if (event.type !== 'ERROR') {
        this.publishInternalEvent(event, 'ERROR', {
          message: `Failed to process event type ${event.type}`,
          details: toErrorMessage(error)
        });
      }
    } finally {
      this.unregisterInflightAbortController(event.correlationId, abortController);
      this.persistSessionSnapshot(event.sessionId);
      this.logDuration(
        'event.total',
        eventStartedAt,
        {
          eventId: event.id,
          type: event.type,
          sessionId: event.sessionId,
          correlationId: event.correlationId
        },
        SLOW_STAGE_WARN_MS
      );
    }
  }

  private async processSessionConfigEvent(event: AgentEvent): Promise<void> {
    const stageStartedAt = Date.now();
    try {
      const payload = parseSessionConfigPayload(event.payload);
      if (!payload) {
        this.publishInternalEvent(event, 'ERROR', {
          message: 'SESSION_CONFIG_UPDATED payload 格式错误。'
        });
        return;
      }

      const listToolsStartedAt = Date.now();
      const availableTools = await this.toolProvider.listDefinitions();
      this.logDuration(
        'session_config.list_tools',
        listToolsStartedAt,
        {
          eventId: event.id,
          sessionId: event.sessionId,
          correlationId: event.correlationId,
          availableTools: availableTools.length
        },
        SLOW_STAGE_WARN_MS
      );

      const availableToolNameSet = new Set(availableTools.map((tool) => tool.name));
      const normalizedAllowedTools = payload.allowedTools?.filter((name) => availableToolNameSet.has(name));
      const unknownTools = payload.allowedTools?.filter((name) => !availableToolNameSet.has(name)) || [];

      this.store.updateSessionConfig(
        event.sessionId,
        {
          systemPrompt: payload.systemPrompt,
          model: payload.model,
          allowedTools: normalizedAllowedTools,
          allowedToolsConfigured: payload.allowedToolsConfigured,
          memoryRefs: payload.memoryRefs
        },
        event.timestamp
      );

      this.mutateSessionState(
        event.sessionId,
        event.timestamp,
        'session configuration updated',
        (session) => ({
          ...session,
          taskState: applySessionConfigToTaskState(session.taskState, payload, event.id),
          guards: applySessionMemoryRefsToGuards(session.guards, payload.memoryRefs || [])
        })
      );

      if (unknownTools.length > 0) {
        this.publishInternalEvent(event, 'ERROR', {
          message: `以下工具未注册，已忽略: ${unknownTools.join(', ')}`
        });
      }
    } finally {
      this.logDuration(
        'session_config.total',
        stageStartedAt,
        {
          eventId: event.id,
          sessionId: event.sessionId,
          correlationId: event.correlationId
        },
        SLOW_STAGE_WARN_MS
      );
    }
  }

  private async processUserInputEvent(event: AgentEvent, signal?: AbortSignal): Promise<void> {
    const stageStartedAt = Date.now();
    let textLength = 0;
    try {
      const payload = parseUserInputPayload(event.payload);
      if (!payload) {
        this.publishInternalEvent(event, 'ERROR', {
          message: 'USER_INPUT payload 必须包含字符串字段 text。'
        });
        return;
      }

      if (signal?.aborted) {
        return;
      }

      textLength = payload.text.length;
      this.store.appendMessage(event.sessionId, {
        role: 'user',
        content: payload.text,
        timestamp: event.timestamp,
        metadata: isObject(payload.metadata) ? payload.metadata : undefined
      });

      this.mutateSessionState(
        event.sessionId,
        event.timestamp,
        'user input received',
        (session) => ({
          ...session,
          taskState: applyUserInputToTaskState(session.taskState, payload, event.id),
          transcriptWindow: recordUserInput(session.transcriptWindow, {
            eventId: event.id,
            timestamp: event.timestamp,
            text: payload.text
          }),
          guards: applyUserInputToGuards(session.guards, payload.text)
        })
      );

      this.runModelResponseDetached(event);
    } finally {
      this.logDuration(
        'user_input.total',
        stageStartedAt,
        {
          eventId: event.id,
          sessionId: event.sessionId,
          correlationId: event.correlationId,
          textLength
        },
        SLOW_STAGE_WARN_MS
      );
    }
  }

  private runModelResponseDetached(triggerEvent: AgentEvent): void {
    if (this.shuttingDown || this.isCorrelationAborted(triggerEvent.correlationId)) {
      return;
    }

    const detachedStartedAt = Date.now();
    const abortController = this.registerInflightAbortController(triggerEvent.correlationId);

    const task = (async () => {
      let releaseLock: (() => void) | undefined;
      let releaseConcurrency: (() => void) | undefined;
      try {
        releaseLock = await this.sessionLocks.acquire(triggerEvent.sessionId, {
          timeoutMs: this.sessionLockTimeoutMs,
          signal: abortController.signal
        });
        releaseConcurrency = await this.llmConcurrencyLimiter.acquire();
        if (abortController.signal.aborted || this.shuttingDown) {
          return;
        }
        await this.runModelResponse(triggerEvent, abortController.signal);
      } catch (error) {
        if (isAbortError(error) || this.isCorrelationAborted(triggerEvent.correlationId)) {
          return;
        }

        if (error instanceof SessionLockAcquireTimeoutError) {
          this.publishInternalEvent(triggerEvent, 'ERROR', {
            message: `Session lock timeout for ${triggerEvent.sessionId}.`,
            details: toErrorMessage(error)
          });
          return;
        }

        this.logger?.error?.(
          {
            eventId: triggerEvent.id,
            type: triggerEvent.type,
            sessionId: triggerEvent.sessionId,
            details: toErrorMessage(error)
          },
          'Agent event processing failed.'
        );

        if (triggerEvent.type !== 'ERROR') {
          this.publishInternalEvent(triggerEvent, 'ERROR', {
            message: `Failed to process event type ${triggerEvent.type}`,
            details: toErrorMessage(error)
          });
        }
      } finally {
        releaseConcurrency?.();
        releaseLock?.();
        this.unregisterInflightAbortController(triggerEvent.correlationId, abortController);
        this.persistSessionSnapshot(triggerEvent.sessionId);
        this.logDuration(
          'model_response.background_total',
          detachedStartedAt,
          {
            eventId: triggerEvent.id,
            triggerEventType: triggerEvent.type,
            sessionId: triggerEvent.sessionId,
            correlationId: triggerEvent.correlationId
          },
          SLOW_STAGE_WARN_MS
        );
      }
    })();

    this.trackDetachedTask(task);
  }

  private async processToolCallRequestedEvent(event: AgentEvent, signal?: AbortSignal): Promise<void> {
    const stageStartedAt = Date.now();

    const payload = parseToolCallRequestedPayload(event.payload);
    if (!payload) {
      this.publishInternalEvent(event, 'ERROR', {
        message: 'TOOL_CALL_REQUESTED payload 格式错误。'
      });
      return;
    }

    const toolName = payload.toolName;
    const toolCallId = payload.toolCallId;

    if (signal?.aborted) {
      return;
    }

    const session = this.store.ensureSession(event.sessionId);
    if (!isToolAllowedInSession(session, payload.toolName)) {
      this.publishInternalEvent(event, 'TOOL_RESULT', {
        toolCallId: payload.toolCallId,
        toolName: payload.toolName,
        status: 'error',
        error: `Tool is not allowed in this session: ${payload.toolName}`
      });
      return;
    }

    this.store.setPendingToolCall(event.sessionId, {
      toolCallId: payload.toolCallId,
      toolName: payload.toolName,
      arguments: payload.arguments,
      status: 'pending',
      requestedAt: event.timestamp
    });

    const hasToolStartedAt = Date.now();
    const hasTool = await this.toolProvider.has(payload.toolName);
    this.logDuration(
      'tool.has',
      hasToolStartedAt,
      {
        eventId: event.id,
        sessionId: event.sessionId,
        correlationId: event.correlationId,
        toolName: payload.toolName
      },
      SLOW_STAGE_WARN_MS
    );

    if (signal?.aborted) {
      return;
    }

    if (!hasTool) {
      this.publishInternalEvent(event, 'TOOL_RESULT', {
        toolCallId: payload.toolCallId,
        toolName: payload.toolName,
        status: 'error',
        error: `Tool is not available: ${payload.toolName}`
      });
      return;
    }

    this.mutateSessionState(
      event.sessionId,
      event.timestamp,
      `tool ${payload.toolName} requested`,
      (session) => ({
        ...session,
        taskState: applyToolCallToTaskState(session.taskState, payload, event.id),
        transcriptWindow: recordToolCall(session.transcriptWindow, {
          eventId: event.id,
          timestamp: event.timestamp,
          payload
        })
      })
    );

    this.logDuration(
      'tool_call_requested.validation',
      stageStartedAt,
      {
        eventId: event.id,
        sessionId: event.sessionId,
        correlationId: event.correlationId,
        toolName,
        toolCallId,
        status: 'detached'
      },
      SLOW_STAGE_WARN_MS
    );

    this.runToolExecutionDetached(event, payload);
  }

  private runToolExecutionDetached(triggerEvent: AgentEvent, payload: ToolCallRequestedPayload): void {
    if (this.shuttingDown || this.isCorrelationAborted(triggerEvent.correlationId)) {
      return;
    }

    const detachedStartedAt = Date.now();
    const abortController = this.registerInflightAbortController(triggerEvent.correlationId);
    let finalStatus: 'ok' | 'error' | 'aborted' = 'aborted';

    const task = (async () => {
      let releaseLock: (() => void) | undefined;
      try {
        releaseLock = await this.sessionLocks.acquire(triggerEvent.sessionId, {
          timeoutMs: this.sessionLockTimeoutMs,
          signal: abortController.signal
        });
        if (abortController.signal.aborted || this.shuttingDown) {
          return;
        }

        const executeStartedAt = Date.now();
        try {
          const sessionSnapshot = this.store.snapshotSession(triggerEvent.sessionId);
          if (!sessionSnapshot) {
            return;
          }
          const result = await runWithRetry({
            stage: 'tool.execute',
            policy: this.toolRetryPolicy,
            signal: abortController.signal,
            logger: this.logger,
            context: {
              eventId: triggerEvent.id,
              sessionId: triggerEvent.sessionId,
              correlationId: triggerEvent.correlationId,
              toolName: payload.toolName,
              toolCallId: payload.toolCallId
            },
            shouldRetry: (error) => isRetryableError(error),
            operation: async () =>
              this.toolProvider.execute(payload.toolName, {
                args: payload.arguments,
                session: sessionSnapshot,
                event: triggerEvent as AgentEvent<ToolCallRequestedPayload>,
                signal: abortController.signal
              })
          });
          this.logDuration(
            'tool.execute',
            executeStartedAt,
            {
              eventId: triggerEvent.id,
              sessionId: triggerEvent.sessionId,
              correlationId: triggerEvent.correlationId,
              toolName: payload.toolName,
              toolCallId: payload.toolCallId,
              status: 'ok'
            },
            SLOW_STAGE_WARN_MS
          );

          if (abortController.signal.aborted || this.shuttingDown) {
            return;
          }

          const normalizedOutcome = normalizeExecutedToolOutcome(result);
          finalStatus = normalizedOutcome.status === 'ok' ? 'ok' : 'error';
          this.publishInternalEvent(triggerEvent, 'TOOL_RESULT', {
            toolCallId: payload.toolCallId,
            toolName: payload.toolName,
            status: normalizedOutcome.status,
            result,
            error: normalizedOutcome.error
          });
        } catch (error) {
          if (isAbortError(error) || abortController.signal.aborted) {
            return;
          }

          this.logDuration(
            'tool.execute',
            executeStartedAt,
            {
              eventId: triggerEvent.id,
              sessionId: triggerEvent.sessionId,
              correlationId: triggerEvent.correlationId,
              toolName: payload.toolName,
              toolCallId: payload.toolCallId,
              status: 'error',
              details: toErrorMessage(error)
            },
            SLOW_STAGE_WARN_MS
          );

          finalStatus = 'error';
          this.publishInternalEvent(triggerEvent, 'TOOL_RESULT', {
            toolCallId: payload.toolCallId,
            toolName: payload.toolName,
            status: 'error',
            error: toErrorMessage(error)
          });
        }
      } catch (error) {
        if (isAbortError(error) || this.isCorrelationAborted(triggerEvent.correlationId)) {
          return;
        }

        if (error instanceof SessionLockAcquireTimeoutError) {
          this.publishInternalEvent(triggerEvent, 'ERROR', {
            message: `Session lock timeout for ${triggerEvent.sessionId}.`,
            details: toErrorMessage(error)
          });
          return;
        }

        this.logger?.error?.(
          {
            eventId: triggerEvent.id,
            type: triggerEvent.type,
            sessionId: triggerEvent.sessionId,
            details: toErrorMessage(error)
          },
          'Agent tool execution failed.'
        );

        if (triggerEvent.type !== 'ERROR') {
          this.publishInternalEvent(triggerEvent, 'ERROR', {
            message: `Failed to execute tool ${payload.toolName}`,
            details: toErrorMessage(error)
          });
        }
      } finally {
        releaseLock?.();
        this.unregisterInflightAbortController(triggerEvent.correlationId, abortController);
        this.persistSessionSnapshot(triggerEvent.sessionId);
        this.logDuration(
          'tool_execution.background_total',
          detachedStartedAt,
          {
            eventId: triggerEvent.id,
            sessionId: triggerEvent.sessionId,
            correlationId: triggerEvent.correlationId,
            toolName: payload.toolName,
            toolCallId: payload.toolCallId,
            status: finalStatus
          },
          SLOW_STAGE_WARN_MS
        );
      }
    })();

    this.trackDetachedTask(task);
  }

  private async processToolResultEvent(event: AgentEvent, signal?: AbortSignal): Promise<void> {
    const stageStartedAt = Date.now();
    let toolName = 'unknown';
    let toolCallId = 'unknown';
    let status: ToolResultPayload['status'] | 'invalid' = 'invalid';
    try {
      const payload = parseToolResultPayload(event.payload);
      if (!payload) {
        this.publishInternalEvent(event, 'ERROR', {
          message: 'TOOL_RESULT payload 格式错误。'
        });
        return;
      }

      const effectivePayload = normalizeSemanticToolResultPayload(payload);
      toolName = effectivePayload.toolName;
      toolCallId = effectivePayload.toolCallId;
      status = effectivePayload.status;
      if (signal?.aborted) {
        return;
      }

      let sessionBeforeUpdate = this.store.snapshotSession(event.sessionId);
      if (!sessionBeforeUpdate) {
        this.store.ensureSession(event.sessionId);
        sessionBeforeUpdate = this.store.snapshotSession(event.sessionId);
      }
      const toolArguments = sessionBeforeUpdate?.pendingToolCalls[payload.toolCallId]?.arguments || {};
      this.store.completePendingToolCall(event.sessionId, effectivePayload, event.timestamp);
      this.store.appendMessage(event.sessionId, {
        role: 'tool',
        toolCallId: effectivePayload.toolCallId,
        toolName: effectivePayload.toolName,
        content: describeToolResultForMessage(effectivePayload),
        timestamp: event.timestamp,
        metadata: {
          status: effectivePayload.status,
          result: effectivePayload.result,
          error: effectivePayload.error
        }
      });

      this.mutateSessionState(
        event.sessionId,
        event.timestamp,
        `tool ${payload.toolName} completed`,
        (session) => {
          const pendingArguments =
            session.pendingToolCalls[effectivePayload.toolCallId]?.arguments || toolArguments;
          const nextGuards =
            effectivePayload.status === 'error'
              ? applyToolFailureToGuards(session.guards, {
                  toolName: effectivePayload.toolName,
                  arguments: pendingArguments,
                  error: effectivePayload.error,
                  transcriptWindow: session.transcriptWindow
                })
              : session.guards;

          return {
            ...session,
            taskState: applyToolResultToTaskState(session.taskState, effectivePayload, event.id),
            transcriptWindow: recordToolResult(session.transcriptWindow, {
              eventId: event.id,
              timestamp: event.timestamp,
              payload: effectivePayload
            }),
            guards: nextGuards
          };
        }
      );

      this.advanceCorrelationLoopState(event);
      this.runModelResponseDetached(event);
    } finally {
      this.logDuration(
        'tool_result.total',
        stageStartedAt,
        {
          eventId: event.id,
          sessionId: event.sessionId,
          correlationId: event.correlationId,
          toolName,
          toolCallId,
          status
        },
        SLOW_STAGE_WARN_MS
      );
    }
  }

  private async processContextRequestedEvent(_event: AgentEvent): Promise<void> {}

  private async processContextHydratedEvent(_event: AgentEvent, _signal?: AbortSignal): Promise<void> {}

  private processAgentReplyEvent(event: AgentEvent): void {
    const payload = parseAgentReplyPayload(event.payload);
    if (!payload) {
      this.publishInternalEvent(event, 'ERROR', {
        message: 'AGENT_REPLY payload 必须包含字符串字段 text。'
      });
      return;
    }

    this.store.appendMessage(event.sessionId, {
      role: 'assistant',
      content: payload.text,
      timestamp: event.timestamp,
      metadata: isObject(payload.metadata) ? payload.metadata : undefined
    });

    this.mutateSessionState(
      event.sessionId,
      event.timestamp,
      'agent reply emitted',
      (session) => ({
        ...session,
        taskState: applyAgentReplyToTaskState(session.taskState, payload, event.id),
        transcriptWindow: recordAssistantReply(session.transcriptWindow, {
          eventId: event.id,
          timestamp: event.timestamp,
          text: payload.text
        })
      })
    );

    this.finalizeCorrelationTask(event.correlationId, 'reply', {
      sessionId: event.sessionId,
      eventId: event.id,
      eventType: event.type
    });
  }

  private processErrorEvent(event: AgentEvent): void {
    const payload = parseErrorPayload(event.payload);
    if (!payload) {
      return;
    }

    this.store.appendMessage(event.sessionId, {
      role: 'system',
      content: `[error] ${payload.message}`,
      timestamp: event.timestamp,
      metadata: isObject(payload.details)
        ? {
            details: payload.details
          }
        : undefined
    });

    this.mutateSessionState(
      event.sessionId,
      event.timestamp,
      'error event recorded',
      (session) => ({
        ...session,
        taskState: applyErrorToTaskState(session.taskState, payload, event.id),
        transcriptWindow: recordError(session.transcriptWindow, {
          eventId: event.id,
          timestamp: event.timestamp,
          message: payload.message
        })
      })
    );

    this.finalizeCorrelationTask(event.correlationId, 'error', {
      sessionId: event.sessionId,
      eventId: event.id,
      eventType: event.type
    });
  }

  private async runModelResponse(triggerEvent: AgentEvent, signal?: AbortSignal): Promise<void> {
    const stageStartedAt = Date.now();
    const turnCount = this.correlationLoopStates.get(triggerEvent.correlationId)?.turnCount;
    let model = 'default';
    let toolCount = 0;
    let outputType: 'reply' | 'tool_call' | 'error' | 'none' = 'none';
    let finalStatus: 'completed' | 'aborted' | 'no_session' | 'no_output' | 'failed' = 'completed';

    try {
      if (signal?.aborted) {
        finalStatus = 'aborted';
        return;
      }

      const eventType = triggerEvent.type;
      if (eventType !== 'USER_INPUT' && eventType !== 'TOOL_RESULT') {
        return;
      }

      let session = this.store.snapshotSession(triggerEvent.sessionId);
      if (!session) {
        finalStatus = 'no_session';
        return;
      }
      model = session.model || 'default';

      const listToolsStartedAt = Date.now();
      const availableTools = await this.toolProvider.listDefinitions();
      this.logDuration(
        'model.list_tools',
        listToolsStartedAt,
        {
          eventId: triggerEvent.id,
          triggerEventType: triggerEvent.type,
          sessionId: triggerEvent.sessionId,
          correlationId: triggerEvent.correlationId,
          turnCount,
          availableTools: availableTools.length
        },
        SLOW_STAGE_WARN_MS
      );
      if (signal?.aborted) {
        finalStatus = 'aborted';
        return;
      }

      const normalizedAllowedTools = normalizeAllowedToolsForAvailableDefinitions(
        session.allowedTools,
        availableTools.map((tool) => tool.name)
      );
      if (normalizedAllowedTools) {
        this.store.updateSessionConfig(
          session.sessionId,
          {
            allowedTools: normalizedAllowedTools
          },
          new Date().toISOString()
        );

        const latestSession = this.store.snapshotSession(session.sessionId);
        if (latestSession) {
          session = latestSession;
        }
      }

      const tools = this.resolveAllowedTools(session, availableTools);
      toolCount = tools.length;
      model = session.model || 'default';

      const modelGenerateStartedAt = Date.now();

      // Migrated from LlmToolDecisionModelClient.generate()
      const triggerToolResult = parseTriggerToolResult(triggerEvent);
      const modelInput = buildModelInputText(
        triggerEvent,
        session,
        this.maxTranscriptMessages,
        triggerToolResult,
        tools
      );
      const systemPrompt = buildSystemPrompt(session, triggerToolResult, tools);
      const requestIdentity = findLatestGatewayRequestIdentityInMessages(
        session.messages
      );
      const traceContext = {
        agentId: session.agentId,
        sessionId: session.sessionId,
        runId: triggerEvent.correlationId,
        stepId: triggerEvent.id,
        workflow: 'event_driven_agent_runtime'
      };
      const attemptErrors: string[] = [];
      let output: AgentModelOutput | undefined;
      let selectedModelRef: ProviderModelReference | undefined;
      let providerRoutes: ProviderRoute[] = [];
      if (this.modelClient) {
        try {
          output = await this.generateModelClientOutputWithRetry(
            {
              triggerEvent,
              session,
              tools,
              signal
            },
            {
              eventId: triggerEvent.id,
              triggerEventType: triggerEvent.type,
              sessionId: triggerEvent.sessionId,
              correlationId: triggerEvent.correlationId,
              model
            }
          );
        } catch (error) {
          if (isAbortError(error) || signal?.aborted) {
            throw createAbortError();
          }

          const details = error instanceof Error ? error.message : String(error);
          attemptErrors.push(`modelClient: ${details}`);
        }
      } else {
        selectedModelRef = parseProviderModelReference(session.model);
        if (session.model && !selectedModelRef) {
          this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
            text: `Invalid session model format "${session.model}". Use provider/model or providerName/model, e.g. openai/gpt-4o-mini or custom-provider/custom-model.`
          });
          return;
        }

        providerRoutes = resolveProviderRoutes(this.config);
        if (selectedModelRef) {
          const modelRef = selectedModelRef;
          providerRoutes = providerRoutes.filter((route) => routeMatchesModelReference(route, modelRef));
          if (providerRoutes.length === 0) {
            this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
              text: `No configured provider route matches session model "${modelRef.raw}".`
            });
            return;
          }
        }

        if (providerRoutes.length === 0) {
          this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
            text: 'No available LLM provider is configured for agent tool-calling.'
          });
          return;
        }

        for (const route of providerRoutes) {
          if (signal?.aborted) {
            throw createAbortError();
          }

          // --- Try streaming path first ---
          const streamingPrepared = buildStreamingProviderRequest(
            route,
            systemPrompt,
            modelInput,
            tools,
            selectedModelRef?.model,
            this.config,
            this.gatewayBaseUrl,
            requestIdentity,
            traceContext
          );

          if (streamingPrepared.ok) {
            try {
              const streamResponse = await this.callProviderStreamingWithRetry(
                streamingPrepared.request,
                signal,
                {
                  eventId: triggerEvent.id,
                  triggerEventType: triggerEvent.type,
                  sessionId: triggerEvent.sessionId,
                  correlationId: triggerEvent.correlationId,
                  route: formatRouteLabel(route)
                }
              );

              if (streamResponse.ok) {
                let accumulatedText = '';
                let streamedPartialText: string | undefined;
                let streamToolCall: { toolName: string; arguments: Record<string, unknown>; reason?: string } | undefined;
                const streamIterator = parseProviderStreamChunks(
                  route.provider,
                  streamResponse
                )[Symbol.asyncIterator]();
                let streamTimedOut = false;

                try {
                  while (true) {
                    const nextChunk = await this.nextStreamChunkWithIdleTimeout(
                      streamIterator,
                      this.resolveStreamChunkIdleTimeoutMs()
                    );
                    if (nextChunk.timedOut) {
                      streamTimedOut = true;
                      break;
                    }

                    const iteration = nextChunk.result;
                    if (!iteration || iteration.done) {
                      break;
                    }

                    const chunk = iteration.value;
                    if (signal?.aborted) {
                      throw createAbortError();
                    }

                    if (chunk.type === 'text_delta') {
                      accumulatedText += chunk.text;
                    } else if (chunk.type === 'tool_call') {
                      streamToolCall = chunk;
                      break;
                    } else if (chunk.type === 'done') {
                      accumulatedText = chunk.text || accumulatedText;
                    }
                  }
                } finally {
                  if (streamTimedOut) {
                    try {
                      streamIterator.return?.(undefined)?.catch?.(() => undefined);
                    } catch {
                      // Ignore generator cleanup failures; timeout handling continues below.
                    }
                    try {
                      streamResponse.body?.cancel?.()?.catch?.(() => undefined);
                    } catch {
                      // Ignore upstream body cancel failures during timeout cleanup.
                    }
                  }
                }

                if (streamTimedOut) {
                  const idleTimeoutMs = this.resolveStreamChunkIdleTimeoutMs();
                  this.logger?.warn?.(
                    {
                      eventId: triggerEvent.id,
                      triggerEventType: triggerEvent.type,
                      sessionId: triggerEvent.sessionId,
                      correlationId: triggerEvent.correlationId,
                      route: formatRouteLabel(route),
                      idleTimeoutMs,
                      partialTextLength: accumulatedText.length
                    },
                    accumulatedText
                      ? 'Streaming provider stalled mid-response. Retrying the same route without streaming.'
                      : 'Streaming provider stalled before producing output. Falling back to non-streaming.'
                  );
                  if (accumulatedText) {
                    streamedPartialText = accumulatedText;
                  }
                }

                if (streamToolCall) {
                  output = {
                    type: 'tool_call',
                    toolName: streamToolCall.toolName,
                    arguments: streamToolCall.arguments,
                    reason: streamToolCall.reason
                  };
                } else if (accumulatedText && !streamTimedOut) {
                  output = {
                    type: 'reply',
                    text: accumulatedText
                  };
                }

                if (output) {
                  break;
                }

                if (streamedPartialText) {
                  const prepared = buildProviderRequest(
                    route,
                    systemPrompt,
                    modelInput,
                    tools,
                    selectedModelRef?.model,
                    this.config,
                    this.gatewayBaseUrl,
                    requestIdentity,
                    traceContext
                  );

                  if (prepared.ok) {
                    try {
                      const recoveryResult = await this.callProviderWithRetry(
                        prepared.request,
                        signal,
                        {
                          eventId: triggerEvent.id,
                          triggerEventType: triggerEvent.type,
                          sessionId: triggerEvent.sessionId,
                          correlationId: triggerEvent.correlationId,
                          route: formatRouteLabel(route),
                          recoveryMode: 'stream_timeout'
                        }
                      );

                      if (recoveryResult.ok) {
                        const recoveredOutput = parseProviderOutput(
                          route.provider,
                          recoveryResult.payload
                        );

                        if (recoveredOutput) {
                          output = recoveredOutput;
                          break;
                        }
                      }
                    } catch (error) {
                      this.logger?.warn?.(
                        {
                          eventId: triggerEvent.id,
                          triggerEventType: triggerEvent.type,
                          sessionId: triggerEvent.sessionId,
                          correlationId: triggerEvent.correlationId,
                          route: formatRouteLabel(route),
                          details: toErrorMessage(error)
                        },
                        'Non-streaming recovery after streaming stall failed. Using partial streamed reply.'
                      );
                    }
                  }

                  output = {
                    type: 'reply',
                    text: streamedPartialText
                  };
                  break;
                }
              }

              if (!streamResponse.ok) {
                const upstreamError = summarizeUpstreamErrorPayload(await readUpstreamPayload(streamResponse));
                attemptErrors.push(
                  upstreamError
                    ? `${formatRouteLabel(route)}: streaming upstream status ${streamResponse.status}; response: ${upstreamError}`
                    : `${formatRouteLabel(route)}: streaming upstream status ${streamResponse.status}`
                );
              }
            } catch (error) {
              if (isAbortError(error) || signal?.aborted) {
                throw createAbortError();
              }
              const details = error instanceof Error ? error.message : String(error);
              attemptErrors.push(`${formatRouteLabel(route)}: streaming call failed: ${details}`);
              this.logger?.warn?.(
                {
                  route: formatRouteLabel(route),
                  details
                },
                'Streaming provider call failed, falling back to non-streaming.'
              );
            }
          }

          if (output) {
            break;
          }

          const prepared = buildProviderRequest(
            route,
            systemPrompt,
            modelInput,
            tools,
            selectedModelRef?.model,
            this.config,
            this.gatewayBaseUrl,
            requestIdentity,
            traceContext
          );
          if (!prepared.ok) {
            attemptErrors.push(`${formatRouteLabel(route)}: ${prepared.error}`);
            continue;
          }

          let callResult: ProviderCallResult;
          try {
            callResult = await this.callProviderWithRetry(prepared.request, signal, {
              eventId: triggerEvent.id,
              triggerEventType: triggerEvent.type,
              sessionId: triggerEvent.sessionId,
              correlationId: triggerEvent.correlationId,
              route: formatRouteLabel(route)
            });
          } catch (error) {
            if (isAbortError(error) || signal?.aborted) {
              throw createAbortError();
            }
            const details = error instanceof Error ? error.message : String(error);
            attemptErrors.push(`${formatRouteLabel(route)}: ${details}`);
            continue;
          }

          if (!callResult.ok) {
            const upstreamError = summarizeUpstreamErrorPayload(callResult.payload);
            attemptErrors.push(
              upstreamError
                ? `${formatRouteLabel(route)}: upstream status ${callResult.status}; response: ${upstreamError}`
                : `${formatRouteLabel(route)}: upstream status ${callResult.status}`
            );
            continue;
          }

          output = parseProviderOutput(route.provider, callResult.payload);
          if (output) {
            break;
          }

          attemptErrors.push(`${formatRouteLabel(route)}: empty model output`);
        }
      }

      outputType = output?.type || 'none';
      this.logDuration(
        'model.generate',
        modelGenerateStartedAt,
        {
          eventId: triggerEvent.id,
          triggerEventType: triggerEvent.type,
          sessionId: triggerEvent.sessionId,
          correlationId: triggerEvent.correlationId,
          model,
          tools: toolCount,
          turnCount,
          outputType
        },
        SLOW_STAGE_WARN_MS
      );

      if (signal?.aborted) {
        finalStatus = 'aborted';
        return;
      }

      if (!output) {
        const detail =
          attemptErrors.length > 0
            ? attemptErrors.slice(0, 3).join(' | ')
            : 'no provider attempt details available';

        this.logger?.warn?.(
          {
            providerAttempts: attemptErrors
          },
          'LLM model output generation failed for agent session.'
        );

        if (triggerToolResult) {
          this.publishInternalEvent(triggerEvent, 'ERROR', {
            message: `LLM generation failed after TOOL_RESULT. ${detail}`,
            details: {
              providerAttempts: attemptErrors,
              toolCallId: triggerToolResult.toolCallId,
              toolName: triggerToolResult.toolName,
              toolStatus: triggerToolResult.status
            }
          });
          finalStatus = 'no_output';
          return;
        }

        this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
          text: `LLM generation failed. ${detail}`
        });
        finalStatus = 'no_output';
        return;
      }

      if (output.type === 'reply') {
        this.publishCommittedReplyChunk(triggerEvent, output.text, output.metadata);
        this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
          text: output.text,
          metadata: output.metadata
        });
        return;
      }

      if (output.type === 'error') {
        this.publishInternalEvent(triggerEvent, 'ERROR', {
          message: output.message,
          details: output.details
        });
        return;
      }

      if (toolCount === 0) {
        this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
          text: '当前没有可用工具。请在无需工具的情况下继续回答。'
        });
        return;
      }

      const decisionHasToolStartedAt = Date.now();
      const hasDecisionTool = await this.toolProvider.has(output.toolName);
      this.logDuration(
        'model.decision_tool_has',
        decisionHasToolStartedAt,
        {
          eventId: triggerEvent.id,
          triggerEventType: triggerEvent.type,
          sessionId: triggerEvent.sessionId,
          correlationId: triggerEvent.correlationId,
          toolName: output.toolName
        },
        SLOW_STAGE_WARN_MS
      );
      if (signal?.aborted) {
        finalStatus = 'aborted';
        return;
      }

      if (!hasDecisionTool) {
        this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
          text: `工具不存在: ${output.toolName}`
        });
        return;
      }

      if (!isToolAllowedInSession(session, output.toolName)) {
        this.publishInternalEvent(triggerEvent, 'AGENT_REPLY', {
          text: `工具未授权: ${output.toolName}`
        });
        return;
      }

      const normalizedToolCall = this.normalizeToolCallRequest(
        session,
        output.toolName,
        isObject(output.arguments) ? output.arguments : {}
      );
      const normalizedArguments = normalizedToolCall.arguments;
      const blockedByGuard = shouldBlockToolCall(session.guards, output.toolName, normalizedArguments);
      if (blockedByGuard) {
        this.publishInternalEvent(triggerEvent, 'ERROR', {
          message: `Rejected guarded action ${output.toolName}: ${blockedByGuard}`
        });
        return;
      }

      this.publishCommittedStepChunk(
        triggerEvent,
        buildToolCallChunkText(output.toolName, output.reason || normalizedToolCall.reasonNote),
        {
          actionType: 'tool_call',
          toolName: output.toolName
        }
      );
      this.publishInternalEvent(triggerEvent, 'TOOL_CALL_REQUESTED', {
        toolCallId: randomUUID(),
        toolName: output.toolName,
        arguments: normalizedArguments,
        reason: normalizedToolCall.reasonNote
          ? output.reason
            ? `${output.reason} (${normalizedToolCall.reasonNote})`
            : normalizedToolCall.reasonNote
          : output.reason
      });
    } catch (error) {
      finalStatus = 'failed';
      throw error;
    } finally {
      this.logDuration(
        'model_response.total',
        stageStartedAt,
        {
          eventId: triggerEvent.id,
          triggerEventType: triggerEvent.type,
          sessionId: triggerEvent.sessionId,
          correlationId: triggerEvent.correlationId,
          model,
          tools: toolCount,
          turnCount,
          outputType,
          status: finalStatus
        },
        SLOW_STAGE_WARN_MS
      );
    }
  }

  private normalizeToolCallRequest(
    _session: AgentSessionState,
    toolName: string,
    rawArguments: Record<string, unknown>
  ): { arguments: Record<string, unknown>; reasonNote?: string } {
    const normalizedArguments = isObject(rawArguments) ? cloneToolArguments(rawArguments) : {};
    return { arguments: normalizedArguments };
  }

  private publishCommittedReplyChunk(
    triggerEvent: AgentEvent,
    text: string,
    metadata?: Record<string, unknown>
  ): void {
    const normalized = normalizeOptionalString(text);
    if (!normalized) {
      return;
    }
    this.publishInternalEvent(triggerEvent, 'AGENT_REPLY_CHUNK', {
      text: normalized,
      done: false,
      metadata: {
        ...metadata,
        kind: 'final_reply',
        committed: true,
        actionType: 'reply'
      }
    });
  }

  private publishCommittedStepChunk(
    triggerEvent: AgentEvent,
    text: string,
    metadata?: Record<string, unknown>
  ): void {
    const normalized = normalizeOptionalString(text);
    if (!normalized) {
      return;
    }
    this.publishInternalEvent(triggerEvent, 'AGENT_REPLY_CHUNK', {
      text: normalized,
      done: false,
      metadata: {
        ...metadata,
        kind: 'committed_step',
        committed: true
      }
    });
  }

  private async callProviderWithRetry(
    request: ProviderRequest,
    signal: AbortSignal | undefined,
    context: Record<string, unknown>
  ): Promise<ProviderCallResult> {
    let lastResult: ProviderCallResult | undefined;
    try {
      return await runWithRetry({
        stage: 'llm.call_provider',
        policy: this.llmRetryPolicy,
        signal,
        logger: this.logger,
        context,
        shouldRetry: (error) => error instanceof RetryableUpstreamStatusError || isRetryableError(error),
        operation: async () => {
          const result = await callProvider(request, this.config.upstreamTimeoutMs, signal, this.logger);
          lastResult = result;
          if (!result.ok && isRetryableHttpStatus(result.status)) {
            throw new RetryableUpstreamStatusError(
              result.status,
              summarizeUpstreamErrorPayload(result.payload)
            );
          }
          return result;
        }
      });
    } catch (error) {
      if (error instanceof RetryableUpstreamStatusError && lastResult) {
        return lastResult;
      }
      throw error;
    }
  }

  private async callProviderStreamingWithRetry(
    request: ProviderRequest,
    signal: AbortSignal | undefined,
    context: Record<string, unknown>
  ): Promise<Response> {
    let lastResponse: Response | undefined;
    try {
      return await runWithRetry({
        stage: 'llm.call_provider_streaming',
        policy: this.llmRetryPolicy,
        signal,
        logger: this.logger,
        context,
        shouldRetry: (error) => error instanceof RetryableUpstreamStatusError || isRetryableError(error),
        operation: async () => {
          const response = await callProviderStreaming(
            request,
            this.config.upstreamTimeoutMs,
            signal,
            this.logger
          );
          lastResponse = response;

          if (!response.ok && isRetryableHttpStatus(response.status)) {
            const payload = await readUpstreamPayload(response.clone());
            throw new RetryableUpstreamStatusError(
              response.status,
              summarizeUpstreamErrorPayload(payload)
            );
          }

          return response;
        }
      });
    } catch (error) {
      if (error instanceof RetryableUpstreamStatusError && lastResponse) {
        return lastResponse;
      }
      throw error;
    }
  }

  private async generateModelClientOutputWithRetry(
    input: Parameters<AgentModelClient['generate']>[0],
    context: Record<string, unknown>
  ): Promise<AgentModelOutput | undefined> {
    const modelClient = this.modelClient;
    if (!modelClient) {
      throw new Error('Agent model client is not configured.');
    }

    return runWithRetry({
      stage: 'llm.model_client_generate',
      policy: this.llmRetryPolicy,
      signal: input.signal,
      logger: this.logger,
      context,
      shouldRetry: (error) => isRetryableError(error),
      operation: async () => modelClient.generate(input)
    });
  }

  private resolveStreamChunkIdleTimeoutMs(): number {
    if (
      Number.isFinite(this.config.upstreamTimeoutMs) &&
      this.config.upstreamTimeoutMs > 0
    ) {
      return Math.min(
        DEFAULT_STREAM_CHUNK_IDLE_TIMEOUT_MS,
        Math.max(
          MIN_STREAM_CHUNK_IDLE_TIMEOUT_MS,
          Math.floor(this.config.upstreamTimeoutMs)
        )
      );
    }

    return DEFAULT_STREAM_CHUNK_IDLE_TIMEOUT_MS;
  }

  private async nextStreamChunkWithIdleTimeout<T>(
    iterator: AsyncIterator<T>,
    idleTimeoutMs: number
  ): Promise<{ timedOut: boolean; result?: IteratorResult<T> }> {
    const nextPromise = iterator.next();
    nextPromise.catch(() => undefined);

    if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs <= 0) {
      return {
        timedOut: false,
        result: await nextPromise
      };
    }

    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        nextPromise.then((result) => ({
          timedOut: false as const,
          result
        })),
        new Promise<{ timedOut: true }>((resolve) => {
          timeoutHandle = setTimeout(() => {
            resolve({ timedOut: true });
          }, idleTimeoutMs);
          timeoutHandle.unref?.();
        })
      ]);
    } finally {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
    }
  }

  private trackDetachedTask(task: Promise<void>): void {
    this.detachedTasks.add(task);
    task
      .catch(() => undefined)
      .finally(() => {
        this.detachedTasks.delete(task);
      });
  }

  private async waitForDetachedTasks(): Promise<void> {
    if (this.detachedTasks.size === 0) {
      return;
    }

    await Promise.allSettled([...this.detachedTasks]);
  }

  private abortInflightOperations(): void {
    for (const controllers of this.inflightAbortControllers.values()) {
      for (const controller of controllers) {
        controller.abort();
      }
    }
  }

  private registerInflightAbortController(correlationId: string): AbortController {
    const controller = new AbortController();
    const existing = this.inflightAbortControllers.get(correlationId);
    if (existing) {
      existing.add(controller);
      return controller;
    }

    this.inflightAbortControllers.set(correlationId, new Set([controller]));
    return controller;
  }

  private unregisterInflightAbortController(correlationId: string, controller: AbortController): void {
    const existing = this.inflightAbortControllers.get(correlationId);
    if (!existing) {
      return;
    }

    existing.delete(controller);
    if (existing.size === 0) {
      this.inflightAbortControllers.delete(correlationId);
    }
  }

  private isCorrelationAborted(correlationId: string): boolean {
    this.cleanupAbortedCorrelations();
    return this.abortedCorrelations.has(correlationId);
  }

  private cleanupAbortedCorrelations(): void {
    if (this.abortedCorrelations.size === 0) {
      return;
    }

    const now = Date.now();
    for (const [correlationId, abortedAt] of this.abortedCorrelations.entries()) {
      if (now - abortedAt > ABORTED_CORRELATION_TTL_MS) {
        this.abortedCorrelations.delete(correlationId);
      }
    }
  }

  private startCorrelationTask(event: AgentEvent): void {
    if (this.correlationTaskStartedAt.has(event.correlationId)) {
      return;
    }

    const startedAt = Date.now();
    this.correlationTaskStartedAt.set(event.correlationId, startedAt);
    this.correlationLoopStates.set(event.correlationId, {
      turnCount: 1,
      lastTriggerEventId: event.id,
      lastTriggerEventType: event.type
    });
    this.logger?.info?.(
      {
        stage: 'task.start',
        startedAt: new Date(startedAt).toISOString(),
        eventId: event.id,
        eventType: event.type,
        sessionId: event.sessionId,
        correlationId: event.correlationId,
        turnCount: 1
      },
      'Agent task started.'
    );
  }

  private advanceCorrelationLoopState(event: AgentEvent): number {
    const current = this.correlationLoopStates.get(event.correlationId);
    const nextTurnCount = current ? current.turnCount + 1 : 1;
    this.correlationLoopStates.set(event.correlationId, {
      turnCount: nextTurnCount,
      lastTriggerEventId: event.id,
      lastTriggerEventType: event.type
    });
    return nextTurnCount;
  }

  private finalizeCorrelationTask(
    correlationId: string,
    outcome: 'reply' | 'error' | 'aborted',
    metadata: {
      sessionId?: string;
      eventId?: string;
      eventType?: AgentEventType;
    } = {}
  ): void {
    const startedAt = this.correlationTaskStartedAt.get(correlationId);
    const loopState = this.correlationLoopStates.get(correlationId);
    this.correlationLoopStates.delete(correlationId);
    if (startedAt === undefined) {
      return;
    }

    this.correlationTaskStartedAt.delete(correlationId);
    this.logDuration(
      'task.total',
      startedAt,
      {
        correlationId,
        outcome,
        sessionId: metadata.sessionId,
        eventId: metadata.eventId,
        eventType: metadata.eventType,
        turnCount: loopState?.turnCount
      },
      SLOW_TASK_WARN_MS
    );
  }

  private logDuration(
    stage: string,
    startedAt: number,
    context: Record<string, unknown>,
    warnThresholdMs: number,
    message = 'Agent timing.'
  ): number {
    const durationMs = Date.now() - startedAt;
    const logContext = {
      stage,
      durationMs,
      ...context
    };

    if (durationMs >= warnThresholdMs) {
      this.logger?.warn?.(logContext, message);
    } else {
      this.logger?.info?.(logContext, message);
    }

    return durationMs;
  }

  private mutateSessionState(
    sessionId: string,
    timestamp: string,
    _reason: string,
    mutator: (session: AgentSessionState) => AgentSessionState
  ): AgentSessionState {
    const current = this.store.snapshotSession(sessionId) || this.store.ensureSession(sessionId);
    const next = mutator(current);
    next.updatedAt = timestamp;

    this.store.replaceSessionState(next);
    return next;
  }

  private resolveAllowedTools(
    session: AgentSessionState,
    availableTools: AgentToolDefinition[]
  ): AgentToolDefinition[] {
    if (!isSessionAllowedToolsConfigured(session)) {
      return availableTools;
    }

    const allowedToolSet = new Set(session.allowedTools);
    const tools: AgentToolDefinition[] = [];
    for (const definition of availableTools) {
      if (allowedToolSet.has(definition.name)) {
        tools.push(definition);
      }
    }

    return tools;
  }

  private resolveAgentGatewayBaseUrl(): string {
    return resolveAgentGatewayBaseUrl(this.config);
  }

  private publishInternalEvent(
    sourceEvent: AgentEvent,
    type: AgentEventType,
    payload: unknown
  ): AgentEvent {
    const event = createEvent({
      sessionId: sourceEvent.sessionId,
      type,
      payload,
      correlationId: sourceEvent.correlationId,
      causationId: sourceEvent.id
    });

    if (!this.shuttingDown) {
      this.eventBus.publish(event);
    }
    return event;
  }
}

export function createAgentRuntime(options: CreateAgentRuntimeOptions): EventDrivenAgentRuntime {
  return new EventDrivenAgentRuntime(options);
}

function createEvent(input: PublishAgentEventInput): AgentEvent {
  return {
    id: randomUUID(),
    type: input.type,
    sessionId: input.sessionId,
    timestamp: new Date().toISOString(),
    correlationId: input.correlationId || randomUUID(),
    causationId: input.causationId,
    payload: input.payload
  };
}

function parseSessionConfigPayload(payload: unknown): SessionConfigUpdatedPayload | undefined {
  if (!isObject(payload)) {
    return undefined;
  }

  const hasAllowedTools =
    Object.prototype.hasOwnProperty.call(payload, 'allowedTools');
  const allowedTools =
    Array.isArray(payload.allowedTools) && payload.allowedTools.every((item) => typeof item === 'string')
      ? payload.allowedTools
      : undefined;
  if (hasAllowedTools && allowedTools === undefined) {
    return undefined;
  }
  const memoryRefs =
    Array.isArray(payload.memoryRefs) && payload.memoryRefs.every((item) => typeof item === 'string')
      ? payload.memoryRefs
      : undefined;
  const modelRaw = typeof payload.model === 'string' ? payload.model : undefined;
  const model = modelRaw === undefined ? undefined : normalizeProviderModelReference(modelRaw);
  if (modelRaw !== undefined && model === undefined) {
    return undefined;
  }

  const systemPrompt = typeof payload.systemPrompt === 'string' ? payload.systemPrompt : undefined;
  if (systemPrompt === undefined && model === undefined && allowedTools === undefined && memoryRefs === undefined) {
    return undefined;
  }

  return {
    systemPrompt,
    model,
    allowedTools,
    allowedToolsConfigured: allowedTools !== undefined ? true : undefined,
    memoryRefs
  };
}

function parseUserInputPayload(payload: unknown): UserInputPayload | undefined {
  if (!isObject(payload) || typeof payload.text !== 'string') {
    return undefined;
  }

  return {
    text: payload.text,
    metadata: isObject(payload.metadata) ? payload.metadata : undefined
  };
}

function parseToolCallRequestedPayload(payload: unknown): ToolCallRequestedPayload | undefined {
  if (
    !isObject(payload) ||
    typeof payload.toolCallId !== 'string' ||
    typeof payload.toolName !== 'string' ||
    !isObject(payload.arguments)
  ) {
    return undefined;
  }

  return {
    toolCallId: payload.toolCallId,
    toolName: payload.toolName,
    arguments: payload.arguments,
    reason: typeof payload.reason === 'string' ? payload.reason : undefined
  };
}

function parseToolResultPayload(payload: unknown): ToolResultPayload | undefined {
  if (
    !isObject(payload) ||
    typeof payload.toolCallId !== 'string' ||
    typeof payload.toolName !== 'string' ||
    (payload.status !== 'ok' && payload.status !== 'error')
  ) {
    return undefined;
  }

  return {
    toolCallId: payload.toolCallId,
    toolName: payload.toolName,
    status: payload.status,
    result: payload.result,
    error: typeof payload.error === 'string' ? payload.error : undefined
  };
}

function normalizeSemanticToolResultPayload(payload: ToolResultPayload): ToolResultPayload {
  const outcome = normalizeExecutedToolOutcome(payload.result, payload.error);
  if (payload.status === 'error') {
    return {
      ...payload,
      error: payload.error || outcome.error
    };
  }

  if (outcome.status === 'error') {
    return {
      ...payload,
      status: 'error',
      error: outcome.error
    };
  }

  return payload;
}

function describeToolResultForMessage(payload: ToolResultPayload): string {
  if (payload.status === 'error') {
    return payload.error
      ? `[${payload.toolName} error] ${payload.error}`
      : `[${payload.toolName} error] Tool failed.`;
  }

  return `[${payload.toolName} ok] ${safeStringify(payload.result)}`;
}

function normalizeExecutedToolOutcome(
  result: unknown,
  fallbackError?: string
): { status: 'ok' | 'error'; error?: string } {
  if (!isObject(result) || result.ok !== false) {
    return fallbackError
      ? {
          status: 'error',
          error: fallbackError
        }
      : {
          status: 'ok'
        };
  }

  return {
    status: 'error',
    error:
      normalizeOptionalString(readNestedToolErrorMessage(result)) ||
      fallbackError ||
      'Tool reported a structured failure.'
  };
}

function readNestedToolErrorMessage(value: Record<string, unknown>): string | undefined {
  if (typeof value.error === 'string') {
    return value.error;
  }
  if (isObject(value.error)) {
    return normalizeOptionalString(value.error.message) || normalizeOptionalString(value.error.code);
  }
  return normalizeOptionalString(value.message);
}

function cloneToolArguments(args: Record<string, unknown>): Record<string, unknown> {
  try {
    return JSON.parse(JSON.stringify(args)) as Record<string, unknown>;
  } catch {
    return { ...args };
  }
}

function parseAgentReplyPayload(payload: unknown): AgentReplyPayload | undefined {
  if (!isObject(payload) || typeof payload.text !== 'string') {
    return undefined;
  }

  return {
    text: payload.text,
    metadata: isObject(payload.metadata) ? payload.metadata : undefined
  };
}

function parseErrorPayload(payload: unknown): ErrorPayload | undefined {
  if (!isObject(payload) || typeof payload.message !== 'string') {
    return undefined;
  }

  return {
    message: payload.message,
    details: payload.details
  };
}

function safeStringify(value: unknown): string {
  if (value === undefined) {
    return 'undefined';
  }

  if (typeof value === 'string') {
    return value;
  }

  try {
    const serialized = JSON.stringify(value);
    if (!serialized) {
      return String(value);
    }

    return serialized.length > 500 ? `${serialized.slice(0, 500)}...` : serialized;
  } catch {
    return String(value);
  }
}

function cloneAgent(agent: AgentDefinition): AgentDefinition {
  return {
    ...agent,
    ownerIdentity: cloneGatewayRequestIdentity(agent.ownerIdentity),
    allowedTools: [...agent.allowedTools]
  };
}

function normalizeAgentDefinition(value: unknown): AgentDefinition | undefined {
  if (!isObject(value)) {
    return undefined;
  }

  const agentId = normalizeOptionalString(value.agentId);
  const name = normalizeOptionalString(value.name);
  if (!agentId || !name) {
    return undefined;
  }

  const createdAt = normalizeOptionalString(value.createdAt) || new Date().toISOString();
  const updatedAt = normalizeOptionalString(value.updatedAt) || createdAt;
  const normalizedAllowedTools = dedupeStringArray(value.allowedTools);
  const allowedToolsConfigured = typeof value.allowedToolsConfigured === 'boolean'
    ? value.allowedToolsConfigured
    : normalizedAllowedTools.length > 0;

  return {
    agentId,
    name,
    description: normalizeOptionalString(value.description),
    ownerIdentity: normalizeGatewayRequestIdentity(value.ownerIdentity),
    systemPrompt: normalizeOptionalString(value.systemPrompt) || DEFAULT_SYSTEM_PROMPT,
    model: normalizeProviderModelReference(value.model),
    allowedTools: normalizedAllowedTools,
    allowedToolsConfigured,
    createdAt,
    updatedAt
  };
}

function canAccessOwnerIdentity(
  ownerIdentity: GatewayRequestIdentity | undefined,
  requestIdentity: GatewayRequestIdentity | undefined
): boolean {
  if (!ownerIdentity?.billingSubjectKey) {
    return true;
  }

  return ownerIdentity.billingSubjectKey === requestIdentity?.billingSubjectKey;
}

function cloneGatewayRequestIdentity(identity: GatewayRequestIdentity | undefined): GatewayRequestIdentity | undefined {
  if (!identity?.billingSubjectKey || !identity.source) {
    return undefined;
  }

  return {
    source: identity.source,
    billingSubjectKey: identity.billingSubjectKey,
    userId: normalizeOptionalString(identity.userId),
    tenantId: normalizeOptionalString(identity.tenantId),
    subject: normalizeOptionalString(identity.subject),
    organizationId: normalizeOptionalString(identity.organizationId),
    plan: normalizeOptionalString(identity.plan),
    apiKeyId: normalizeOptionalString(identity.apiKeyId)
  };
}

function normalizeGatewayRequestIdentity(value: unknown): GatewayRequestIdentity | undefined {
  if (!isObject(value)) {
    return undefined;
  }

  const billingSubjectKey = normalizeOptionalString(value.billingSubjectKey);
  const source = value.source;
  if (
    !billingSubjectKey ||
    (source !== 'trusted_header' && source !== 'http_introspection' && source !== 'static_api_key')
  ) {
    return undefined;
  }

  return cloneGatewayRequestIdentity({
    source,
    billingSubjectKey,
    userId: normalizeOptionalString(value.userId),
    tenantId: normalizeOptionalString(value.tenantId),
    subject: normalizeOptionalString(value.subject),
    organizationId: normalizeOptionalString(value.organizationId),
    plan: normalizeOptionalString(value.plan),
    apiKeyId: normalizeOptionalString(value.apiKeyId)
  });
}

function normalizeProviderModelReference(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

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

function normalizeOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
}

function buildToolCallChunkText(toolName: string, reason?: string): string {
  const normalizedReason = normalizeOptionalString(reason);
  if (!normalizedReason) {
    return `Action: calling ${toolName}.`;
  }
  return `Action: calling ${toolName}. ${truncateChunkDetail(normalizedReason)}`;
}

function truncateChunkDetail(value: string, maxLength = 220): string {
  const normalized = normalizeOptionalString(value) || '';
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength)}...`;
}

function dedupeStringArray(values: unknown): string[] {
  if (!Array.isArray(values)) {
    return [];
  }

  const normalized: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    if (typeof value !== 'string') {
      continue;
    }

    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }

    seen.add(trimmed);
    normalized.push(trimmed);
  }

  return normalized;
}

async function fetchExternalStateSnapshot(
  source: AgentExternalSourceConfig,
  logger?: AgentRuntimeLogger
): Promise<ExternalStateSnapshot> {
  const endpoint = normalizeOptionalString(source.endpoint);
  if (source.transport !== 'stdio' && !endpoint) {
    throw new Error('agent.external.endpoint is required when external mode is enabled.');
  }
  if (source.transport === 'stdio' && !normalizeOptionalString(source.command)) {
    throw new Error('agent.external.command is required when agent.external.transport=stdio.');
  }

  const timeoutMs = Number.isFinite(source.timeoutMs) && source.timeoutMs > 0
    ? Math.floor(source.timeoutMs)
    : 5000;

  try {
    const payload = await requestExternalJson(source, {
      label: 'agent.external',
      httpMethod: 'GET',
      payload: {
        type: 'agent_state_request'
      },
      grpcDefaultPath: '/gateway.agent.v1.AgentStateSource/GetState'
    });

    if (!isObject(payload)) {
      throw new Error('External agent endpoint payload must be a JSON object.');
    }

    const envelope = isObject(payload.data) ? payload.data : payload;
    const agents = Array.isArray(envelope.agents) ? envelope.agents : [];
    const sessions = Array.isArray(envelope.sessions) ? envelope.sessions : [];
    if (!Array.isArray(envelope.agents) && !Array.isArray(envelope.sessions)) {
      throw new Error('External agent endpoint payload must include agents and/or sessions arrays.');
    }

    return {
      agents,
      sessions
    };
  } catch (error) {
    if (isTimeoutError(error)) {
      logger?.warn?.(
        {
          endpoint,
          command: source.command,
          transport: source.transport,
          timeoutMs
        },
        'External agent endpoint request timed out.'
      );
    }

    throw error;
  }
}

async function upsertExternalSessionSnapshot(
  source: AgentExternalSourceConfig,
  snapshot: PersistedSessionSnapshot,
  logger?: AgentRuntimeLogger
): Promise<void> {
  const endpoint = normalizeOptionalString(source.endpoint);
  if (source.transport !== 'stdio' && !endpoint) {
    throw new Error('agent.external.endpoint is required when external mode is enabled.');
  }

  if (source.transport !== 'http') {
    await requestExternalJson(source, {
      label: 'agent.external',
      payload: {
        type: 'agent_session_upsert',
        sessionId: snapshot.state.sessionId,
        session: snapshot
      },
      grpcDefaultPath: '/gateway.agent.v1.AgentStateSource/UpsertSession'
    });
    logger?.info?.(
      {
        sessionId: snapshot.state.sessionId,
        endpoint,
        command: source.command,
        transport: source.transport
      },
      'Synced agent session snapshot to external source.'
    );
    return;
  }

  const httpEndpoint = endpoint as string;
  const response = await fetchExternalSource(
    source,
    `${trimTrailingSlash(httpEndpoint)}/sessions/${encodeURIComponent(snapshot.state.sessionId)}`,
    {
      method: 'PUT',
      body: JSON.stringify(snapshot)
    }
  );

  if (!response.ok) {
    const details = await safeReadExternalResponseText(response);
    throw new Error(
      details
        ? `External session sync failed (${response.status}): ${details}`
        : `External session sync failed (${response.status}).`
    );
  }

  logger?.info?.(
    {
      sessionId: snapshot.state.sessionId,
      endpoint
    },
    'Synced agent session snapshot to external source.'
  );
}

async function deleteExternalSessionSnapshot(
  source: AgentExternalSourceConfig,
  sessionId: string,
  logger?: AgentRuntimeLogger
): Promise<void> {
  const endpoint = normalizeOptionalString(source.endpoint);
  if (source.transport !== 'stdio' && !endpoint) {
    throw new Error('agent.external.endpoint is required when external mode is enabled.');
  }

  if (source.transport !== 'http') {
    await requestExternalJson(source, {
      label: 'agent.external',
      payload: {
        type: 'agent_session_delete',
        sessionId
      },
      grpcDefaultPath: '/gateway.agent.v1.AgentStateSource/DeleteSession'
    });
    logger?.info?.(
      {
        sessionId,
        endpoint,
        command: source.command,
        transport: source.transport
      },
      'Deleted agent session snapshot from external source.'
    );
    return;
  }

  const httpEndpoint = endpoint as string;
  const response = await fetchExternalSource(
    source,
    `${trimTrailingSlash(httpEndpoint)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'DELETE'
    }
  );

  if (!response.ok && response.status !== 404) {
    const details = await safeReadExternalResponseText(response);
    throw new Error(
      details
        ? `External session delete failed (${response.status}): ${details}`
        : `External session delete failed (${response.status}).`
    );
  }

  logger?.info?.(
    {
      sessionId,
      endpoint
    },
    'Deleted agent session snapshot from external source.'
  );
}

async function fetchExternalSource(
  source: AgentExternalSourceConfig,
  url: string,
  init: RequestInit
): Promise<Response> {
  const headers = buildExternalSourceHeaders(source, init.body ? 'application/json' : undefined);
  const timeoutMs = Number.isFinite(source.timeoutMs) && source.timeoutMs > 0
    ? Math.floor(source.timeoutMs)
    : 5000;
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(url, {
      ...init,
      headers,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeout);
  }
}

function buildExternalSourceHeaders(
  source: AgentExternalSourceConfig,
  contentType?: string
): Record<string, string> {
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...source.headers
  };
  if (contentType) {
    headers['content-type'] = contentType;
  }
  if (source.apiKey) {
    headers[source.apiKeyHeader] = source.apiKey;
  }
  return headers;
}

async function safeReadExternalResponseText(response: Response): Promise<string | undefined> {
  try {
    const text = (await response.text()).trim();
    return text ? text.slice(0, 500) : undefined;
  } catch {
    return undefined;
  }
}

function normalizeExternalSessionSnapshots(value: unknown[]): PersistedSessionSnapshot[] {
  const snapshots: PersistedSessionSnapshot[] = [];

  for (const item of value) {
    if (!isObject(item)) {
      continue;
    }

    const stateCandidate = isObject(item.state) ? item.state : item;
    if (!isObject(stateCandidate) || typeof stateCandidate.sessionId !== 'string') {
      continue;
    }

    const eventsCandidate = Array.isArray(item.events) ? item.events : [];
    snapshots.push({
      state: stateCandidate as unknown as AgentSessionState,
      events: eventsCandidate as AgentEventRecord[]
    });
  }

  return snapshots;
}

function normalizeAllowedToolsForAvailableDefinitions(
  currentAllowedTools: string[],
  availableToolNames: string[]
): string[] | undefined {
  if (availableToolNames.length === 0) {
    return currentAllowedTools.length > 0 ? [] : undefined;
  }

  const availableSet = new Set(availableToolNames);

  if (currentAllowedTools.length === 0) {
    return undefined;
  }

  const matchedTools = currentAllowedTools.filter((name) => availableSet.has(name));
  const migratedServerTools: string[] = [];
  const migratedCodeToolMeta: string[] = [];

  for (const toolName of currentAllowedTools) {
    if (availableSet.has(toolName)) {
      continue;
    }

    const separator = toolName.indexOf('.');
    if (separator <= 0) {
      continue;
    }

    const serverName = toolName.slice(0, separator).trim();
    if (!serverName || !availableSet.has(serverName)) {
      continue;
    }

    migratedServerTools.push(serverName);
  }

  if (availableSet.has('code_tool.call')) {
    const hasLegacyUnknownTool = currentAllowedTools.some(
      (name) => !availableSet.has(name) && !name.startsWith('code_tool.')
    );
    if (hasLegacyUnknownTool) {
      migratedCodeToolMeta.push('code_tool.call');
    }
  }

  const normalized = dedupeStringArray([...matchedTools, ...migratedServerTools, ...migratedCodeToolMeta]);
  if (normalized.length === 0 || sameStringArray(currentAllowedTools, normalized)) {
    return undefined;
  }

  return normalized;
}

function sameStringArray(left: string[], right: string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

function isSessionAllowedToolsConfigured(session: AgentSessionState): boolean {
  if (typeof session.allowedToolsConfigured === 'boolean') {
    return session.allowedToolsConfigured;
  }

  return session.allowedTools.length > 0;
}

function isToolAllowedInSession(session: AgentSessionState, toolName: string): boolean {
  if (!isSessionAllowedToolsConfigured(session)) {
    return true;
  }

  return session.allowedTools.includes(toolName);
}

function resolveAgentRuntimeConfig(config: GatewayConfig): {
  sessionLockTimeoutMs: number;
  llmRetry: AgentRetryPolicyConfig;
  toolRetry: AgentRetryPolicyConfig;
} {
  const runtime = isObject(config.agent?.runtime)
    ? (config.agent.runtime as Partial<Record<string, unknown>>)
    : undefined;
  const configuredSessionLockTimeoutMs = runtime?.sessionLockTimeoutMs;

  const sessionLockTimeoutMs =
    typeof configuredSessionLockTimeoutMs === 'number' &&
    Number.isFinite(configuredSessionLockTimeoutMs) &&
    configuredSessionLockTimeoutMs > 0
      ? Math.floor(configuredSessionLockTimeoutMs)
      : DEFAULT_SESSION_LOCK_TIMEOUT_MS;

  return {
    sessionLockTimeoutMs,
    llmRetry: normalizeRetryPolicy(runtime?.llmRetry, DEFAULT_LLM_RETRY_POLICY),
    toolRetry: normalizeRetryPolicy(runtime?.toolRetry, DEFAULT_TOOL_RETRY_POLICY)
  };
}

function normalizeRetryPolicy(
  value: unknown,
  fallback: AgentRetryPolicyConfig
): AgentRetryPolicyConfig {
  const candidate = isObject(value) ? value : undefined;
  const maxAttempts = readPositiveInteger(candidate?.maxAttempts, fallback.maxAttempts);
  const baseDelayMs = readNonNegativeInteger(candidate?.baseDelayMs, fallback.baseDelayMs);
  const maxDelayMs = Math.max(
    baseDelayMs,
    readNonNegativeInteger(candidate?.maxDelayMs, fallback.maxDelayMs)
  );
  const backoffMultiplier = readPositiveInteger(
    candidate?.backoffMultiplier,
    fallback.backoffMultiplier
  );
  const jitterMs = readNonNegativeInteger(candidate?.jitterMs, fallback.jitterMs);

  return {
    maxAttempts,
    baseDelayMs,
    maxDelayMs,
    backoffMultiplier,
    jitterMs
  };
}

function readPositiveInteger(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }

  const normalized = Math.floor(value);
  return normalized > 0 ? normalized : fallback;
}

function readNonNegativeInteger(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }

  const normalized = Math.floor(value);
  return normalized >= 0 ? normalized : fallback;
}

class RetryableUpstreamStatusError extends Error {
  constructor(
    readonly status: number,
    readonly detail?: string
  ) {
    super(
      detail
        ? `Retryable upstream status ${status}: ${detail}`
        : `Retryable upstream status ${status}.`
    );
    this.name = 'RetryableUpstreamStatusError';
  }
}

function isAbortError(error: unknown): boolean {
  if (error instanceof Error) {
    return error.name === 'AbortError';
  }

  return isObject(error) && (error as { name?: unknown }).name === 'AbortError';
}

function isTimeoutError(error: unknown): boolean {
  return error instanceof Error && /timed out|timeout/i.test(error.message);
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resolveAgentGatewayBaseUrl(config: GatewayConfig): string {
  const configuredBaseUrl = process.env.AGENT_GATEWAY_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return trimTrailingSlash(configuredBaseUrl);
  }

  const protocol = process.env.AGENT_GATEWAY_PROTOCOL?.trim() || 'http';
  const host = resolveAgentGatewayHost(config.host);
  const port = Number.isFinite(config.port) && config.port > 0 ? config.port : 3000;
  return trimTrailingSlash(`${protocol}://${host}:${port}`);
}

function resolveAgentGatewayHost(host: string | undefined): string {
  const normalized = host?.trim();
  if (!normalized || normalized === '0.0.0.0' || normalized === '::' || normalized === '[::]') {
    return '127.0.0.1';
  }

  return normalized;
}

function createAbortError(): Error {
  const error = new Error('Agent model generation aborted.');
  error.name = 'AbortError';
  return error;
}
