import type { GatewayRequestIdentity } from '../types';

export type AgentEventType =
  | 'SESSION_CONFIG_UPDATED'
  | 'USER_INPUT'
  | 'TOOL_CALL_REQUESTED'
  | 'TOOL_RESULT'
  | 'CONTEXT_REQUESTED'
  | 'CONTEXT_HYDRATED'
  | 'AGENT_REPLY'
  | 'AGENT_REPLY_CHUNK'
  | 'ERROR';

export interface AgentEvent<TPayload = unknown> {
  id: string;
  type: AgentEventType;
  sessionId: string;
  timestamp: string;
  correlationId: string;
  causationId?: string;
  payload: TPayload;
}

export interface SessionConfigUpdatedPayload {
  systemPrompt?: string;
  model?: string;
  allowedTools?: string[];
  allowedToolsConfigured?: boolean;
  memoryRefs?: string[];
}

export interface UserInputPayload {
  text: string;
  metadata?: Record<string, unknown>;
}

export interface ToolCallRequestedPayload {
  toolCallId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  reason?: string;
}

export interface ToolResultPayload {
  toolCallId: string;
  toolName: string;
  status: 'ok' | 'error';
  result?: unknown;
  error?: string;
}

export interface AgentReplyPayload {
  text: string;
  metadata?: Record<string, unknown>;
}

export interface AgentReplyChunkPayload {
  text: string;
  done: boolean;
  metadata?: Record<string, unknown> & {
    kind?: 'final_reply' | 'committed_step';
    committed?: boolean;
    actionType?: 'reply' | 'tool_call';
    toolName?: string;
  };
}

export interface ErrorPayload {
  message: string;
  details?: unknown;
}

export type TaskStatus = 'running' | 'blocked' | 'done';

export interface TaskState {
  id: string;
  goal: string;
  activeStep: string | null;
  constraints: string[];
  done: string[];
  todo: string[];
  status: TaskStatus;
}

export type TranscriptItem =
  | {
      id?: string;
      timestamp?: string;
      type: 'user';
      text: string;
      raw?: string;
    }
  | {
      id?: string;
      timestamp?: string;
      type: 'assistant';
      text: string;
      raw?: string;
    }
  | {
      id?: string;
      timestamp?: string;
      type: 'tool_call';
      tool: string;
      args: string;
      raw?: string;
    }
  | {
      id?: string;
      timestamp?: string;
      type: 'tool_result';
      tool: string;
      output: string;
      raw?: string;
    }
  | {
      id?: string;
      timestamp?: string;
      type: 'failure';
      text: string;
      raw?: string;
    };

export interface TranscriptWindow {
  items: TranscriptItem[];
}

export interface Guards {
  doNotRepeat: string[];
  doNotForget: string[];
  doNotViolate: string[];
}

export type AgentMessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface AgentMessage {
  role: AgentMessageRole;
  content: string;
  timestamp: string;
  toolCallId?: string;
  toolName?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentPendingToolCall {
  toolCallId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'ok' | 'error';
  requestedAt: string;
  completedAt?: string;
  result?: unknown;
  error?: string;
}

export interface AgentSessionState {
  sessionId: string;
  agentId: string;
  ownerIdentity?: GatewayRequestIdentity;
  systemPrompt: string;
  model?: string;
  allowedTools: string[];
  allowedToolsConfigured?: boolean;
  memoryRefs: string[];
  messages: AgentMessage[];
  pendingToolCalls: Record<string, AgentPendingToolCall>;
  taskState: TaskState;
  transcriptWindow: TranscriptWindow;
  guards: Guards;
  lastEventOffset: number;
  updatedAt: string;
}

export interface AgentDefinition {
  agentId: string;
  name: string;
  description?: string;
  ownerIdentity?: GatewayRequestIdentity;
  systemPrompt: string;
  model?: string;
  allowedTools: string[];
  allowedToolsConfigured?: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface AgentEventRecord<TPayload = unknown> extends AgentEvent<TPayload> {
  offset: number;
}

export interface AgentToolDefinition {
  name: string;
  description: string;
  inputSchema?: Record<string, unknown>;
}

export interface AgentToolExecutionInput {
  args: Record<string, unknown>;
  session: AgentSessionState;
  event: AgentEvent<ToolCallRequestedPayload>;
  mcpMeta?: Record<string, unknown>;
  signal?: AbortSignal;
}

export type AgentToolHandler = (input: AgentToolExecutionInput) => Promise<unknown>;

export type AgentModelOutput =
  | { type: 'reply'; text: string; metadata?: Record<string, unknown> }
  | { type: 'tool_call'; toolName: string; arguments: Record<string, unknown>; reason?: string }
  | { type: 'error'; message: string; details?: unknown };

export type AgentModelStreamChunk =
  | { type: 'text_delta'; text: string }
  | { type: 'tool_call'; toolName: string; arguments: Record<string, unknown>; reason?: string }
  | { type: 'done'; text: string };

export interface AgentModelInput {
  triggerEvent: AgentEvent;
  session: AgentSessionState;
  tools: AgentToolDefinition[];
  signal?: AbortSignal;
}

export interface AgentModelClient {
  generate(input: AgentModelInput): Promise<AgentModelOutput | undefined>;
}

export interface AgentRuntimeLogger {
  info?(context: unknown, message?: string): void;
  warn?(context: unknown, message?: string): void;
  error?(context: unknown, message?: string): void;
}
