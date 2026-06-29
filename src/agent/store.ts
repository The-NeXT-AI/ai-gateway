import type {
  AgentEvent,
  AgentEventRecord,
  AgentMessage,
  AgentPendingToolCall,
  AgentSessionState,
  SessionConfigUpdatedPayload,
  ToolResultPayload
} from './types';
import type { GatewayRequestIdentity } from '../types';
import { createInitialGuards, normalizeGuards } from './guards';
import { createInitialTaskState, normalizeTaskState } from './task-state';
import { createTranscriptWindow, normalizeTranscriptWindow } from './transcript-window';

const UNASSIGNED_AGENT_ID = 'unassigned';
const LEGACY_DEFAULT_AGENT_ID = 'default';

interface SessionEnvelope {
  state: AgentSessionState;
  events: AgentEventRecord[];
  processedEventIds: Set<string>;
}

export interface PersistedSessionSnapshot {
  state: AgentSessionState;
  events: AgentEventRecord[];
}

export interface AgentSessionStoreOptions {
  defaultSystemPrompt: string;
  defaultAllowedTools: string[];
  maxMessagesPerSession?: number;
  maxEventsPerSession?: number;
}

export class InMemoryAgentSessionStore {
  private readonly sessions = new Map<string, SessionEnvelope>();
  private readonly maxMessagesPerSession: number;
  private readonly maxEventsPerSession: number;

  constructor(private readonly options: AgentSessionStoreOptions) {
    this.maxMessagesPerSession = options.maxMessagesPerSession ?? 200;
    this.maxEventsPerSession = options.maxEventsPerSession ?? 1000;
  }

  exportSessions(): PersistedSessionSnapshot[] {
    const snapshots: PersistedSessionSnapshot[] = [];
    for (const envelope of this.sessions.values()) {
      snapshots.push({
        state: cloneSessionState(envelope.state),
        events: envelope.events.map((item) => ({ ...item }))
      });
    }
    return snapshots;
  }

  exportSession(sessionId: string): PersistedSessionSnapshot | undefined {
    const envelope = this.sessions.get(sessionId);
    if (!envelope) {
      return undefined;
    }

    return {
      state: cloneSessionState(envelope.state),
      events: envelope.events.map((item) => ({ ...item }))
    };
  }

  importSessions(snapshots: PersistedSessionSnapshot[]): void {
    this.sessions.clear();

    for (const snapshot of snapshots) {
      try {
        if (!snapshot?.state || typeof snapshot.state.sessionId !== 'string') {
          continue;
        }

        const sessionId = snapshot.state.sessionId.trim();
        if (!sessionId) {
          continue;
        }

        const state = cloneSessionState(snapshot.state);
        state.sessionId = sessionId;
        const normalizedAgentId =
          typeof state.agentId === 'string' && state.agentId.trim() ? state.agentId.trim() : UNASSIGNED_AGENT_ID;
        state.agentId =
          normalizedAgentId === LEGACY_DEFAULT_AGENT_ID ? UNASSIGNED_AGENT_ID : normalizedAgentId;
        state.ownerIdentity = normalizeGatewayRequestIdentity(state.ownerIdentity);
        state.systemPrompt =
          typeof state.systemPrompt === 'string' && state.systemPrompt.trim()
            ? state.systemPrompt.trim()
            : this.options.defaultSystemPrompt;
        state.model =
          typeof state.model === 'string' && state.model.trim()
            ? state.model.trim()
            : undefined;
        state.allowedTools = dedupeStringArray(state.allowedTools || []);
        state.allowedToolsConfigured =
          typeof state.allowedToolsConfigured === 'boolean'
            ? state.allowedToolsConfigured
            : state.allowedTools.length > 0;
        state.memoryRefs = dedupeStringArray(state.memoryRefs || []);
        state.taskState = normalizeTaskState((state as Partial<AgentSessionState>).taskState, sessionId);
        state.transcriptWindow = normalizeTranscriptWindow(
          (state as Partial<AgentSessionState>).transcriptWindow,
          state.messages
        );
        state.guards = normalizeGuards((state as Partial<AgentSessionState>).guards, state.memoryRefs);
        state.updatedAt =
          typeof state.updatedAt === 'string' && state.updatedAt ? state.updatedAt : new Date().toISOString();

        const events = Array.isArray(snapshot.events)
          ? snapshot.events
              .filter((item): item is AgentEventRecord => isEventRecord(item) && item.sessionId === sessionId)
              .map((item) => ({ ...item }))
          : [];
        events.sort((a, b) => a.offset - b.offset);

        if (events.length > this.maxEventsPerSession) {
          events.splice(0, events.length - this.maxEventsPerSession);
        }

        if (state.messages.length > this.maxMessagesPerSession) {
          state.messages.splice(0, state.messages.length - this.maxMessagesPerSession);
        }

        const maxOffset = events.length > 0 ? events[events.length - 1].offset : 0;
        const stateOffset = Number.isFinite(state.lastEventOffset) ? Math.max(0, Math.floor(state.lastEventOffset)) : 0;
        state.lastEventOffset = Math.max(stateOffset, maxOffset);

        const processedEventIds = new Set<string>();
        for (const event of events) {
          processedEventIds.add(event.id);
        }

        this.sessions.set(sessionId, {
          state,
          events,
          processedEventIds
        });
      } catch {
        continue;
      }
    }
  }

  hasSession(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }

  deleteSession(sessionId: string): boolean {
    if (!this.sessions.has(sessionId)) {
      return false;
    }

    this.sessions.delete(sessionId);
    return true;
  }

  appendEvent(event: AgentEvent): boolean {
    const envelope = this.ensureEnvelope(event.sessionId);
    if (envelope.processedEventIds.has(event.id)) {
      return false;
    }

    envelope.processedEventIds.add(event.id);
    envelope.state.lastEventOffset += 1;
    envelope.state.updatedAt = event.timestamp;
    envelope.events.push({
      ...event,
      offset: envelope.state.lastEventOffset
    });

    if (envelope.events.length > this.maxEventsPerSession) {
      envelope.events.splice(0, envelope.events.length - this.maxEventsPerSession);
    }

    return true;
  }

  snapshotSession(sessionId: string): AgentSessionState | undefined {
    const envelope = this.sessions.get(sessionId);
    if (!envelope) {
      return undefined;
    }

    return cloneSessionState(envelope.state);
  }

  replaceSessionState(state: AgentSessionState): void {
    const envelope = this.ensureEnvelope(state.sessionId);
    envelope.state = cloneSessionState(state);
  }

  listEvents(sessionId: string, limit = 50): AgentEventRecord[] {
    const envelope = this.sessions.get(sessionId);
    if (!envelope) {
      return [];
    }

    const normalizedLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : 50;
    return envelope.events.slice(-normalizedLimit).map((item) => ({ ...item }));
  }

  getLastEventOffset(sessionId: string): number {
    const envelope = this.sessions.get(sessionId);
    return envelope?.state.lastEventOffset ?? 0;
  }

  listEventsAfter(sessionId: string, afterOffset = 0, limit = 500): AgentEventRecord[] {
    const envelope = this.sessions.get(sessionId);
    if (!envelope) {
      return [];
    }

    const normalizedOffset = Number.isFinite(afterOffset) ? Math.max(0, Math.floor(afterOffset)) : 0;
    const normalizedLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : 500;
    const events = envelope.events;
    const startIdx = findFirstIndexAfterOffset(events, normalizedOffset);
    if (startIdx >= events.length) {
      return [];
    }

    const end = Math.min(startIdx + normalizedLimit, events.length);
    const result: AgentEventRecord[] = [];
    for (let i = startIdx; i < end; i++) {
      result.push({ ...events[i] });
    }
    return result;
  }

  appendMessage(sessionId: string, message: AgentMessage): void {
    const envelope = this.ensureEnvelope(sessionId);
    envelope.state.messages.push({ ...message });

    if (envelope.state.messages.length > this.maxMessagesPerSession) {
      envelope.state.messages.splice(0, envelope.state.messages.length - this.maxMessagesPerSession);
    }

    envelope.state.updatedAt = message.timestamp;
  }

  updateSessionConfig(sessionId: string, payload: SessionConfigUpdatedPayload, timestamp: string): void {
    const envelope = this.ensureEnvelope(sessionId);

    if (typeof payload.systemPrompt === 'string') {
      envelope.state.systemPrompt = payload.systemPrompt.trim();
    }

    if (typeof payload.model === 'string') {
      const normalizedModel = payload.model.trim();
      envelope.state.model = normalizedModel || undefined;
    }

    if (Array.isArray(payload.allowedTools)) {
      envelope.state.allowedTools = dedupeStringArray(payload.allowedTools);
      envelope.state.allowedToolsConfigured =
        typeof payload.allowedToolsConfigured === 'boolean' ? payload.allowedToolsConfigured : true;
    } else if (typeof payload.allowedToolsConfigured === 'boolean') {
      envelope.state.allowedToolsConfigured = payload.allowedToolsConfigured;
    }

    if (Array.isArray(payload.memoryRefs)) {
      envelope.state.memoryRefs = dedupeStringArray(payload.memoryRefs);
    }

    envelope.state.updatedAt = timestamp;
  }

  setSessionAgent(sessionId: string, agentId: string, timestamp: string): void {
    const envelope = this.ensureEnvelope(sessionId);
    envelope.state.agentId = agentId;
    envelope.state.updatedAt = timestamp;
  }

  setSessionOwner(sessionId: string, ownerIdentity: GatewayRequestIdentity | undefined, timestamp: string): void {
    const envelope = this.ensureEnvelope(sessionId);
    envelope.state.ownerIdentity = cloneGatewayRequestIdentity(ownerIdentity);
    envelope.state.updatedAt = timestamp;
  }

  setPendingToolCall(sessionId: string, call: AgentPendingToolCall): void {
    const envelope = this.ensureEnvelope(sessionId);
    envelope.state.pendingToolCalls[call.toolCallId] = {
      ...call,
      arguments: cloneArguments(call.arguments)
    };
    envelope.state.updatedAt = call.requestedAt;
  }

  completePendingToolCall(sessionId: string, payload: ToolResultPayload, completedAt: string): void {
    const envelope = this.ensureEnvelope(sessionId);
    const existing = envelope.state.pendingToolCalls[payload.toolCallId];

    envelope.state.pendingToolCalls[payload.toolCallId] = {
      toolCallId: payload.toolCallId,
      toolName: payload.toolName,
      arguments: existing?.arguments || {},
      status: payload.status,
      requestedAt: existing?.requestedAt || completedAt,
      completedAt,
      result: payload.result,
      error: payload.error
    };
    envelope.state.updatedAt = completedAt;
  }

  ensureSession(sessionId: string): AgentSessionState {
    return this.ensureEnvelope(sessionId).state;
  }

  private ensureEnvelope(sessionId: string): SessionEnvelope {
    const existing = this.sessions.get(sessionId);
    if (existing) {
      return existing;
    }

    const timestamp = new Date().toISOString();
    const memoryRefs: string[] = [];
    const state: AgentSessionState = {
      sessionId,
      agentId: UNASSIGNED_AGENT_ID,
      systemPrompt: this.options.defaultSystemPrompt,
      model: undefined,
      allowedTools: dedupeStringArray(this.options.defaultAllowedTools),
      allowedToolsConfigured: this.options.defaultAllowedTools.length > 0,
      memoryRefs,
      messages: [],
      pendingToolCalls: {},
      taskState: createInitialTaskState(sessionId),
      transcriptWindow: createTranscriptWindow(),
      guards: createInitialGuards(memoryRefs),
      lastEventOffset: 0,
      updatedAt: timestamp
    };

    const envelope: SessionEnvelope = {
      state,
      events: [],
      processedEventIds: new Set<string>()
    };

    this.sessions.set(sessionId, envelope);
    return envelope;
  }
}

function cloneSessionState(state: AgentSessionState): AgentSessionState {
  const pendingToolCalls: Record<string, AgentPendingToolCall> = {};
  const pendingToolCallEntries = state.pendingToolCalls || {};
  for (const [toolCallId, call] of Object.entries(pendingToolCallEntries)) {
    pendingToolCalls[toolCallId] = {
      ...call,
      arguments: cloneArguments(call.arguments)
    };
  }

  return {
    ...state,
    ownerIdentity: cloneGatewayRequestIdentity(state.ownerIdentity),
    allowedTools: [...state.allowedTools],
    memoryRefs: [...state.memoryRefs],
    messages: state.messages.map((message) => ({ ...message })),
    pendingToolCalls,
    taskState: normalizeTaskState((state as Partial<AgentSessionState>).taskState, state.sessionId),
    transcriptWindow: normalizeTranscriptWindow(
      (state as Partial<AgentSessionState>).transcriptWindow,
      state.messages
    ),
    guards: normalizeGuards((state as Partial<AgentSessionState>).guards, state.memoryRefs)
  };
}

function cloneGatewayRequestIdentity(identity: GatewayRequestIdentity | undefined): GatewayRequestIdentity | undefined {
  if (!identity?.billingSubjectKey || !identity.source) {
    return undefined;
  }

  return {
    source: identity.source,
    billingSubjectKey: identity.billingSubjectKey,
    userId: readOptionalString(identity.userId),
    tenantId: readOptionalString(identity.tenantId),
    subject: readOptionalString(identity.subject),
    organizationId: readOptionalString(identity.organizationId),
    plan: readOptionalString(identity.plan),
    apiKeyId: readOptionalString(identity.apiKeyId)
  };
}

function normalizeGatewayRequestIdentity(value: unknown): GatewayRequestIdentity | undefined {
  if (!isObject(value)) {
    return undefined;
  }

  const billingSubjectKey = readOptionalString(value.billingSubjectKey);
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
    userId: readOptionalString(value.userId),
    tenantId: readOptionalString(value.tenantId),
    subject: readOptionalString(value.subject),
    organizationId: readOptionalString(value.organizationId),
    plan: readOptionalString(value.plan),
    apiKeyId: readOptionalString(value.apiKeyId)
  });
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
}

function cloneArguments(args: Record<string, unknown>): Record<string, unknown> {
  try {
    return JSON.parse(JSON.stringify(args)) as Record<string, unknown>;
  } catch {
    return { ...args };
  }
}

function dedupeStringArray(values: string[]): string[] {
  const set = new Set<string>();
  for (const value of values) {
    if (typeof value !== 'string') {
      continue;
    }

    const normalized = value.trim();
    if (!normalized) {
      continue;
    }

    set.add(normalized);
  }

  return [...set];
}

function isEventRecord(value: unknown): value is AgentEventRecord {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const item = value as AgentEventRecord;
  return (
    typeof item.id === 'string' &&
    typeof item.type === 'string' &&
    typeof item.sessionId === 'string' &&
    typeof item.timestamp === 'string' &&
    typeof item.correlationId === 'string' &&
    Number.isFinite(item.offset)
  );
}

function findFirstIndexAfterOffset(events: AgentEventRecord[], afterOffset: number): number {
  if (events.length === 0) {
    return 0;
  }

  let low = 0;
  let high = events.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (events[mid].offset <= afterOffset) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}
