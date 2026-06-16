import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { WebSocket, type RawData } from 'ws';
import type {
  AgentMcpServerConfig,
  AgentMcpStdioServerConfig,
  AgentMcpWebSocketServerConfig
} from '../types';
import { isObject } from '../utils';
import type {
  AgentRuntimeLogger,
  AgentToolDefinition,
  AgentToolExecutionInput
} from './types';

interface JsonRpcSuccessResponse {
  jsonrpc: '2.0';
  id: string | number;
  result: unknown;
}

interface JsonRpcErrorResponse {
  jsonrpc: '2.0';
  id: string | number | null;
  error: {
    code: number;
    message: string;
    data?: unknown;
  };
}

interface McpIndexedTool {
  serverName: string;
  remoteToolName: string;
  definition: AgentToolDefinition;
}

interface CodeToolCatalogEntry {
  toolName: string;
  alias: string;
  serverName: string;
  serverNamespace: string;
  remoteToolName: string;
  description: string;
  inputSchema?: Record<string, unknown>;
}

interface CodeToolNamespaceBindings {
  mcp: Record<string, Record<string, string>>;
  topLevel: Record<string, Record<string, string>>;
}

interface SandboxToolCallMessage {
  type: 'tool_call';
  id: number;
  tool: string;
  input: unknown;
}

interface SandboxResultMessage {
  type: 'result';
  result: unknown;
}

interface SandboxErrorMessage {
  type: 'error';
  error: string;
}

interface SandboxToolResultMessage {
  type: 'tool_result';
  id: number;
  ok: boolean;
  output?: unknown;
  error?: string;
}

interface McpServerToolCatalog {
  tools: AgentToolDefinition[];
  toolMap: Map<string, AgentToolDefinition>;
}

export type McpToolExposureMode = 'canonical' | 'server-cli' | 'code-tool' | 'passthrough';

export interface AgentToolProvider {
  listDefinitions(): Promise<AgentToolDefinition[]>;
  has(name: string): Promise<boolean>;
  execute(name: string, input: AgentToolExecutionInput): Promise<unknown>;
  close(): Promise<void>;
}

export interface CreateMcpAgentToolProviderOptions {
  servers: AgentMcpServerConfig[];
  logger?: AgentRuntimeLogger;
  exposureMode?: McpToolExposureMode;
}

interface McpServerClient {
  listTools(): Promise<AgentToolDefinition[]>;
  callTool(name: string, args: Record<string, unknown>, meta?: Record<string, unknown>): Promise<unknown>;
  close(): Promise<void>;
}

export class McpAgentToolProvider implements AgentToolProvider {
  private readonly clients = new Map<string, McpServerClient>();
  private readonly toolIndex = new Map<string, McpIndexedTool>();
  private readonly serverCatalogs = new Map<string, McpServerToolCatalog>();
  private readonly codeToolServerNamespaces = new Map<string, string>();
  private readonly exposureMode: McpToolExposureMode;
  private refreshPromise?: Promise<void>;
  private lastRefreshAt = 0;

  constructor(private readonly options: CreateMcpAgentToolProviderOptions) {
    this.exposureMode = options.exposureMode || 'canonical';
    for (const server of options.servers) {
      this.clients.set(server.name, createMcpServerClient(server, options.logger));
    }
  }

  async listDefinitions(): Promise<AgentToolDefinition[]> {
    await this.refreshTools(false);
    return [...this.toolIndex.values()].map((item) => ({ ...item.definition }));
  }

  async has(name: string): Promise<boolean> {
    await this.refreshTools(false);
    return Boolean(this.resolveTool(name));
  }

  async execute(name: string, input: AgentToolExecutionInput): Promise<unknown> {
    if (input.signal?.aborted) {
      throw createAbortError();
    }

    await this.refreshTools(false);
    let indexedTool = this.resolveTool(name);
    if (!indexedTool) {
      await this.refreshTools(true);
      indexedTool = this.resolveTool(name);
    }

    if (!indexedTool) {
      throw new Error(`Tool is not available from MCP servers: ${name}`);
    }

    if (this.exposureMode === 'code-tool') {
      return this.executeCodeTool(name, input);
    }

    const client = this.clients.get(indexedTool.serverName);
    if (!client) {
      throw new Error(`MCP server client not found: ${indexedTool.serverName}`);
    }

    if (this.exposureMode === 'server-cli') {
      return this.executeServerCliTool(indexedTool, client, input);
    }

    return raceWithAbort(
      client.callTool(indexedTool.remoteToolName, input.args, input.mcpMeta),
      input.signal
    );
  }

  async close(): Promise<void> {
    await Promise.allSettled(
      [...this.clients.values()].map((client) => client.close())
    );
  }

  private resolveTool(name: string): McpIndexedTool | undefined {
    return this.toolIndex.get(name);
  }

  private async refreshTools(force: boolean): Promise<void> {
    if (!force && Date.now() - this.lastRefreshAt < 3000) {
      return;
    }

    if (this.refreshPromise) {
      await this.refreshPromise;
      return;
    }

    this.refreshPromise = this.performRefresh();
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = undefined;
    }
  }

  private async performRefresh(): Promise<void> {
    const nextCanonicalTools: McpIndexedTool[] = [];
    const nextCatalogs = new Map<string, McpServerToolCatalog>();
    const passthroughNames = new Set<string>();

    for (const [serverName, client] of this.clients.entries()) {
      let listedTools: AgentToolDefinition[];
      try {
        listedTools = await client.listTools();
      } catch (error) {
        this.options.logger?.warn?.(
          {
            server: serverName,
            details: toErrorMessage(error)
          },
          'Failed to list MCP tools from server. Skipping this server until it becomes reachable.'
        );
        continue;
      }

      const normalizedTools = normalizeMcpTools(listedTools);
      const toolMap = new Map<string, AgentToolDefinition>();
      for (const tool of normalizedTools) {
        toolMap.set(tool.name, tool);
      }
      nextCatalogs.set(serverName, {
        tools: normalizedTools,
        toolMap
      });

      if (this.exposureMode === 'server-cli') {
        nextCanonicalTools.push({
          serverName,
          remoteToolName: serverName,
          definition: buildMcpCliWrapperDefinition(serverName, normalizedTools)
        });
        continue;
      }

      if (this.exposureMode === 'code-tool') {
        continue;
      }

      for (const tool of normalizedTools) {
        if (this.exposureMode === 'passthrough') {
          if (passthroughNames.has(tool.name)) {
            this.options.logger?.warn?.(
              {
                toolName: tool.name,
                server: serverName
              },
              'Skipping duplicate passthrough MCP tool name.'
            );
            continue;
          }

          passthroughNames.add(tool.name);
          nextCanonicalTools.push({
            serverName,
            remoteToolName: tool.name,
            definition: {
              ...tool,
              name: tool.name
            }
          });
          continue;
        }

        const canonicalName = `${serverName}.${tool.name}`;
        nextCanonicalTools.push({
          serverName,
          remoteToolName: tool.name,
          definition: {
            ...tool,
            name: canonicalName
          }
        });
      }
    }

    this.serverCatalogs.clear();
    for (const [serverName, catalog] of nextCatalogs.entries()) {
      this.serverCatalogs.set(serverName, catalog);
    }

    if (this.exposureMode === 'code-tool') {
      this.rebuildCodeToolServerNamespaces([...nextCatalogs.keys()]);
      nextCanonicalTools.push(
        ...buildCodeToolMetaDefinitions().map((definition) => ({
          serverName: '__code_tool__',
          remoteToolName: definition.name,
          definition
        }))
      );
    } else {
      this.codeToolServerNamespaces.clear();
    }

    this.toolIndex.clear();

    for (const tool of nextCanonicalTools) {
      this.toolIndex.set(tool.definition.name, tool);
    }

    this.lastRefreshAt = Date.now();
  }

  private async executeServerCliTool(
    indexedTool: McpIndexedTool,
    client: McpServerClient,
    input: AgentToolExecutionInput
  ): Promise<unknown> {
    const serverName = indexedTool.serverName;
    const invocation = parseMcpCliInvocation(input.args);
    const command = invocation.tokens[0];
    let catalog = this.serverCatalogs.get(serverName);
    let tools = catalog?.tools || [];

    if (!command || isHelpCommand(command)) {
      return buildMcpCliHelpResult(serverName, tools);
    }

    if (command === '--list-tools') {
      if (tools.length === 0) {
        await this.refreshTools(true);
        catalog = this.serverCatalogs.get(serverName);
        tools = catalog?.tools || [];
      }
      if (tools.length === 0) {
        try {
          const liveTools = await raceWithAbort(client.listTools(), input.signal);
          const normalizedLiveTools = normalizeMcpTools(liveTools);
          const toolMap = new Map<string, AgentToolDefinition>();
          for (const tool of normalizedLiveTools) {
            toolMap.set(tool.name, tool);
          }
          this.serverCatalogs.set(serverName, {
            tools: normalizedLiveTools,
            toolMap
          });
          tools = normalizedLiveTools;
        } catch (error) {
          throw new Error(
            `Failed to discover MCP tools from server "${serverName}". ` +
              `Underlying error: ${toErrorMessage(error)}. ` +
              `Ensure the MCP server is reachable, then retry "${serverName} --list-tools".`
          );
        }
      }
      return buildMcpCliListToolsResult(serverName, tools);
    }

    let targetCatalog = catalog;
    let resolvedCommand = command;
    let targetTool = targetCatalog?.toolMap.get(resolvedCommand);
    if (!targetTool) {
      const alias = resolveMcpCliSubcommandAlias(resolvedCommand, targetCatalog?.tools || []);
      if (alias) {
        resolvedCommand = alias;
        targetTool = targetCatalog?.toolMap.get(resolvedCommand);
      }
    }
    if (!targetTool) {
      await this.refreshTools(true);
      targetCatalog = this.serverCatalogs.get(serverName);
      targetTool = targetCatalog?.toolMap.get(resolvedCommand);
      if (!targetTool) {
        const alias = resolveMcpCliSubcommandAlias(resolvedCommand, targetCatalog?.tools || []);
        if (alias) {
          resolvedCommand = alias;
          targetTool = targetCatalog?.toolMap.get(resolvedCommand);
        }
      }
    }

    if (!targetTool) {
      const available = (targetCatalog?.tools || [])
        .map((tool) => tool.name)
        .sort((a, b) => a.localeCompare(b));
      throw new Error(
        [
          `Unknown MCP subcommand: ${serverName} ${command}`,
          `Try "${serverName} --list-tools".`,
          available.length > 0 ? `Available tools: ${available.join(', ')}` : undefined
        ]
          .filter(Boolean)
          .join(' ')
      );
    }

    const subcommandArgsTokens = invocation.tokens.slice(1);
    if (!invocation.directArgs && subcommandArgsTokens.length > 0 && isHelpCommand(subcommandArgsTokens[0])) {
      return buildMcpCliSubcommandHelpResult(serverName, resolvedCommand, targetTool);
    }

    let toolArgs: Record<string, unknown>;
    if (invocation.directArgs) {
      const validationError = validateMcpCliSubcommandArgs(invocation.directArgs, targetTool.inputSchema);
      if (validationError) {
        return buildMcpCliSubcommandHelpResult(serverName, resolvedCommand, targetTool, validationError);
      }
      toolArgs = invocation.directArgs;
    } else {
      const parsedArgs = parseMcpCliSubcommandArgs(subcommandArgsTokens, targetTool.inputSchema);
      if (!parsedArgs.ok) {
        return buildMcpCliSubcommandHelpResult(
          serverName,
          resolvedCommand,
          targetTool,
          parsedArgs.error || 'Failed to parse subcommand arguments.'
        );
      }
      toolArgs = parsedArgs.args;
    }

    try {
      return await raceWithAbort(client.callTool(resolvedCommand, toolArgs), input.signal);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }

      throw buildMcpCliSubcommandExecutionError(serverName, resolvedCommand, targetTool, error);
    }
  }

  private async executeCodeTool(name: string, input: AgentToolExecutionInput): Promise<unknown> {
    if (name === 'code_tool.search') {
      return this.executeCodeToolSearch(input);
    }

    if (name === 'code_tool.call') {
      return this.executeCodeToolCall(input);
    }

    if (name === 'code_tool.runCode') {
      return this.executeCodeToolRunCode(input);
    }

    throw new Error(`Unsupported code tool meta tool: ${name}`);
  }

  private async executeCodeToolSearch(input: AgentToolExecutionInput): Promise<unknown> {
    const query = readNonEmptyString(input.args.query);
    if (!query) {
      throw new Error('code_tool.search requires a non-empty "query" string.');
    }

    const topK = resolveCodeToolTopK(input.args.topK);
    const entries = this.listCodeToolCatalogEntries();
    const ranked = rankCodeToolCatalogEntries(entries, query).slice(0, topK);

    return {
      query,
      totalTools: entries.length,
      selectedCount: ranked.length,
      selectedTools: ranked.map((entry) => ({
        name: entry.toolName,
        alias: entry.alias,
        description: entry.description,
        inputSchema: entry.inputSchema
      })),
      summary:
        ranked.length > 0
          ? `Selected ${ranked.length} tool(s): ${ranked.map((entry) => entry.toolName).join(', ')}`
          : 'No relevant tools matched the query.'
    };
  }

  private async executeCodeToolCall(input: AgentToolExecutionInput): Promise<unknown> {
    const requestedTool = readNonEmptyString(input.args.tool);
    if (!requestedTool) {
      throw new Error('code_tool.call requires a non-empty "tool" string.');
    }

    const entries = this.listCodeToolCatalogEntries();
    const entry = this.resolveCodeToolEntry(requestedTool, entries);
    if (!entry) {
      const preview = entries.slice(0, 20).map((item) => item.toolName);
      throw new Error(
        [
          `Unknown tool for code_tool.call: ${requestedTool}`,
          preview.length > 0 ? `Available tools: ${preview.join(', ')}` : 'No MCP tools are currently available.'
        ].join(' ')
      );
    }

    const nestedArguments = input.args.arguments !== undefined ? input.args.arguments : input.args.params;
    if (nestedArguments !== undefined && !isObject(nestedArguments)) {
      throw new Error('code_tool.call field "arguments" (or "params") must be an object when provided.');
    }

    const callArgs = isObject(nestedArguments) ? { ...nestedArguments } : {};
    return this.invokeCodeToolEntry(entry, callArgs, input.signal, resolveCodeToolTimeoutMs(input.args.timeoutMs));
  }

  private async executeCodeToolRunCode(input: AgentToolExecutionInput): Promise<unknown> {
    const code = readNonEmptyString(input.args.code);
    if (!code) {
      throw new Error('code_tool.runCode requires a non-empty "code" string.');
    }

    const timeoutMs = resolveCodeToolTimeoutMs(input.args.timeoutMs);
    const entries = this.listCodeToolCatalogEntries();
    const namespaceBindings = buildCodeToolNamespaceBindings(entries);

    return this.runCodeInDenoSandbox({
      code,
      timeoutMs,
      namespaceBindings,
      signal: input.signal,
      callTool: async (toolNameOrAlias: string, rawArgs: unknown) => {
        if (typeof toolNameOrAlias !== 'string' || !toolNameOrAlias.trim()) {
          throw new Error('code_tool.runCode callTool requires a non-empty tool name.');
        }

        const entry = this.resolveCodeToolEntry(toolNameOrAlias, entries);
        if (!entry) {
          throw new Error(`code_tool.runCode referenced unknown tool: ${toolNameOrAlias}`);
        }

        const normalizedArgs = rawArgs === undefined ? {} : rawArgs;
        if (!isObject(normalizedArgs)) {
          throw new Error(`code_tool.runCode tool arguments must be an object: ${toolNameOrAlias}`);
        }

        return this.invokeCodeToolEntry(entry, { ...normalizedArgs }, input.signal, timeoutMs);
      }
    });
  }

  private listCodeToolCatalogEntries(): CodeToolCatalogEntry[] {
    const entries: CodeToolCatalogEntry[] = [];

    for (const [serverName, catalog] of this.serverCatalogs.entries()) {
      const serverNamespace =
        this.codeToolServerNamespaces.get(serverName) || sanitizeCodeToolServerNamespace(serverName);
      for (const tool of catalog.tools) {
        const toolName = `mcp.${serverNamespace}.${tool.name}`;
        entries.push({
          toolName,
          alias: toCodeToolAlias(toolName),
          serverName,
          serverNamespace,
          remoteToolName: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema
        });
      }
    }

    entries.sort((left, right) => left.toolName.localeCompare(right.toolName));
    return entries;
  }

  private resolveCodeToolEntry(
    requestedToolOrAlias: string,
    entries: CodeToolCatalogEntry[]
  ): CodeToolCatalogEntry | undefined {
    const normalized = requestedToolOrAlias.trim();
    if (!normalized) {
      return undefined;
    }

    for (const entry of entries) {
      if (entry.toolName === normalized || entry.alias === normalized) {
        return entry;
      }

      const executableMcpPath = buildCodeToolExecutableReference(entry, true);
      if (executableMcpPath === normalized) {
        return entry;
      }

      const executableTopLevelPath = buildCodeToolExecutableReference(entry, false);
      if (executableTopLevelPath === normalized) {
        return entry;
      }

      const namespaceAliasToolName = `mcp.${toCodeToolMethodName(entry.serverNamespace)}.${entry.remoteToolName}`;
      if (namespaceAliasToolName === normalized) {
        return entry;
      }
    }

    for (const entry of entries) {
      if (`${entry.serverName}.${entry.remoteToolName}` === normalized) {
        return entry;
      }
    }

    return undefined;
  }

  private async invokeCodeToolEntry(
    entry: CodeToolCatalogEntry,
    args: Record<string, unknown>,
    signal: AbortSignal | undefined,
    timeoutMs: number
  ): Promise<unknown> {
    const client = this.clients.get(entry.serverName);
    if (!client) {
      throw new Error(`MCP server client not found: ${entry.serverName}`);
    }

    return runWithTimeout(
      raceWithAbort(client.callTool(entry.remoteToolName, args), signal),
      timeoutMs,
      `code_tool.call timed out after ${timeoutMs}ms: ${entry.toolName}`
    );
  }

  private rebuildCodeToolServerNamespaces(serverNames: string[]): void {
    this.codeToolServerNamespaces.clear();

    const used = new Set<string>();
    for (const serverName of serverNames.slice().sort((left, right) => left.localeCompare(right))) {
      const base = sanitizeCodeToolServerNamespace(serverName);
      let candidate = base;
      let index = 2;

      while (used.has(candidate)) {
        candidate = `${base}_${index}`;
        index += 1;
      }

      used.add(candidate);
      this.codeToolServerNamespaces.set(serverName, candidate);
    }
  }

  private async runCodeInDenoSandbox(input: {
    code: string;
    timeoutMs: number;
    namespaceBindings: CodeToolNamespaceBindings;
    signal?: AbortSignal;
    callTool: (toolNameOrAlias: string, rawArgs: unknown) => Promise<unknown>;
  }): Promise<unknown> {
    const tempDir = await mkdtemp(join(tmpdir(), 'gateway-code-tool-deno-sandbox-'));
    const scriptPath = join(tempDir, 'sandbox.ts');
    await writeFile(scriptPath, buildCodeToolDenoRunnerScript(input.code, input.namespaceBindings), 'utf8');

    try {
      return await this.executeCodeToolDenoScript(scriptPath, input);
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  }

  private async executeCodeToolDenoScript(
    scriptPath: string,
    input: {
      timeoutMs: number;
      signal?: AbortSignal;
      callTool: (toolNameOrAlias: string, rawArgs: unknown) => Promise<unknown>;
    }
  ): Promise<unknown> {
    const { timeoutMs, signal } = input;
    const child = spawn(
      'deno',
      [
        'run',
        '--quiet',
        '--no-prompt',
        '--no-remote',
        '--cached-only',
        '--deny-env',
        '--deny-net',
        '--deny-import',
        '--deny-read',
        '--deny-write',
        '--deny-run',
        '--deny-ffi',
        '--deny-sys',
        scriptPath
      ],
      {
        stdio: 'pipe'
      }
    );

    let settled = false;
    let stdoutBuffer = '';
    let stderrBuffer = '';
    let timeoutHandle: NodeJS.Timeout | undefined;
    const inFlightToolCalls = new Map<number, string>();
    const abortError = createAbortError();

    let resolvePromise!: (value: unknown) => void;
    let rejectPromise!: (reason: Error) => void;
    const promise = new Promise<unknown>((resolve, reject) => {
      resolvePromise = resolve;
      rejectPromise = reject;
    });

    const cleanup = (): void => {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      signal?.removeEventListener('abort', onAbort);
      if (!child.stdin.destroyed) {
        child.stdin.end();
      }
      if (!child.killed) {
        child.kill('SIGKILL');
      }
    };

    const settleWithResult = (result: unknown): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolvePromise(result);
    };

    const settleWithError = (error: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      rejectPromise(error);
    };

    const sendToolResult = (message: SandboxToolResultMessage): void => {
      if (child.stdin.destroyed) {
        return;
      }
      child.stdin.write(`${JSON.stringify(message)}\n`);
    };

    const handleToolCall = async (message: SandboxToolCallMessage): Promise<void> => {
      inFlightToolCalls.set(message.id, message.tool);
      try {
        const output = await input.callTool(message.tool, message.input);
        sendToolResult({
          type: 'tool_result',
          id: message.id,
          ok: true,
          output: toJsonSafeValue(output)
        });
      } catch (error) {
        sendToolResult({
          type: 'tool_result',
          id: message.id,
          ok: false,
          error: toErrorMessage(error)
        });
      } finally {
        inFlightToolCalls.delete(message.id);
      }
    };

    const handleStdoutChunk = (chunk: Buffer): void => {
      stdoutBuffer += chunk.toString('utf8');
      while (true) {
        const newlineIndex = stdoutBuffer.indexOf('\n');
        if (newlineIndex < 0) {
          break;
        }

        const line = stdoutBuffer.slice(0, newlineIndex).trim();
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
        if (!line) {
          continue;
        }

        let payload: unknown;
        try {
          payload = JSON.parse(line) as unknown;
        } catch {
          continue;
        }

        if (isSandboxToolCallMessage(payload)) {
          void handleToolCall(payload);
          continue;
        }

        if (isSandboxResultMessage(payload)) {
          settleWithResult(payload.result);
          continue;
        }

        if (isSandboxErrorMessage(payload)) {
          settleWithError(new Error(payload.error));
          continue;
        }
      }
    };

    const onAbort = (): void => {
      settleWithError(abortError);
    };

    if (signal?.aborted) {
      settleWithError(abortError);
      return promise;
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    child.stdout.on('data', (chunk: Buffer | string) => {
      handleStdoutChunk(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
    });

    child.stderr.on('data', (chunk: Buffer | string) => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
      stderrBuffer += text;
      this.options.logger?.warn?.(
        {
          stderr: text.trim().slice(0, 400)
        },
        'code_tool.runCode deno stderr.'
      );
    });

    child.once('error', (error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') {
        settleWithError(new Error('code_tool.runCode requires Deno in PATH, but "deno" was not found.'));
        return;
      }
      settleWithError(new Error(`Failed to start code_tool.runCode deno sandbox: ${toErrorMessage(error)}`));
    });

    child.once('exit', (code, exitSignal) => {
      if (settled) {
        return;
      }

      if (code === 0) {
        settleWithError(new Error('Deno sandbox exited without returning a final result.'));
        return;
      }

      const stderrTail = stderrBuffer
        .trim()
        .split('\n')
        .slice(-20)
        .join('\n');
      settleWithError(
        new Error(
          `Deno sandbox exited unexpectedly (code=${String(code)}, signal=${String(exitSignal)})` +
            (stderrTail ? `: ${stderrTail}` : '')
        )
      );
    });

    timeoutHandle = setTimeout(() => {
      const inFlight = [...inFlightToolCalls.entries()].map(([id, tool]) => ({ id, tool }));
      const diagnostics = {
        timeoutMs,
        inFlight,
        stderrTail: stderrBuffer
          .trim()
          .split('\n')
          .slice(-20)
          .join('\n')
      };
      settleWithError(new Error(`code_tool.runCode timed out after ${timeoutMs}ms. ${JSON.stringify(diagnostics)}`));
    }, timeoutMs);

    return promise;
  }
}

function createMcpServerClient(
  server: AgentMcpServerConfig,
  logger?: AgentRuntimeLogger
): McpServerClient {
  if (server.transport === 'websocket') {
    return new WebSocketMcpServerClient(server, logger);
  }

  return new StdioMcpServerClient(server, logger);
}

class StdioMcpServerClient {
  private child?: ChildProcessWithoutNullStreams;
  private stdoutBuffer = Buffer.alloc(0);
  private requestId = 0;
  private isInitialized = false;
  private connectPromise?: Promise<void>;
  private readonly pendingRequests = new Map<
    string,
    {
      resolve: (value: unknown) => void;
      reject: (reason?: unknown) => void;
      timer: NodeJS.Timeout;
    }
  >();

  constructor(
    private readonly server: AgentMcpStdioServerConfig,
    private readonly logger?: AgentRuntimeLogger
  ) {}

  async listTools(): Promise<AgentToolDefinition[]> {
    await this.ensureConnected();

    const tools: AgentToolDefinition[] = [];
    let cursor: string | undefined;

    while (true) {
      const result = await this.sendRequest('tools/list', cursor ? { cursor } : {});
      if (!isObject(result) || !Array.isArray(result.tools)) {
        throw new Error(`Invalid tools/list response from MCP server ${this.server.name}`);
      }

      for (const rawTool of result.tools) {
        if (!isObject(rawTool) || typeof rawTool.name !== 'string') {
          continue;
        }

        tools.push({
          name: rawTool.name,
          description:
            typeof rawTool.description === 'string'
              ? rawTool.description
              : `Tool from MCP server ${this.server.name}`,
          inputSchema: isObject(rawTool.inputSchema)
            ? (rawTool.inputSchema as Record<string, unknown>)
            : undefined
        });
      }

      cursor = typeof result.nextCursor === 'string' ? result.nextCursor : undefined;
      if (!cursor) {
        break;
      }
    }

    return tools;
  }

  async callTool(name: string, args: Record<string, unknown>, meta?: Record<string, unknown>): Promise<unknown> {
    await this.ensureConnected();
    const params: Record<string, unknown> = {
      name,
      arguments: hydrateMcpCallArgumentsFromMeta(args, meta)
    };
    const result = await this.sendRequest('tools/call', params);

    if (isObject(result) && result.isError === true) {
      throw new Error(extractMcpErrorMessage(result));
    }

    return result;
  }

  async close(): Promise<void> {
    this.teardownProcess(new Error(`MCP server client closed: ${this.server.name}`));
  }

  private async ensureConnected(): Promise<void> {
    if (this.isInitialized && this.child && !this.child.killed) {
      return;
    }

    if (this.connectPromise) {
      await this.connectPromise;
      return;
    }

    this.connectPromise = this.connect();
    try {
      await this.connectPromise;
    } finally {
      this.connectPromise = undefined;
    }
  }

  private async connect(): Promise<void> {
    this.teardownProcess(new Error(`Reset MCP process: ${this.server.name}`), false);

    const env = {
      ...process.env,
      ...this.server.env
    };

    const child = spawn(this.server.command, this.server.args, {
      cwd: this.server.cwd,
      env,
      stdio: 'pipe'
    });
    this.child = child;
    this.isInitialized = false;
    this.stdoutBuffer = Buffer.alloc(0);

    child.stdout.on('data', (chunk: Buffer | string) => {
      this.handleStdoutData(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
    });

    child.stderr.on('data', (chunk: Buffer | string) => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
      if (text.trim()) {
        this.logger?.info?.(
          {
            server: this.server.name,
            stderr: text.trim()
          },
          'MCP stderr.'
        );
      }
    });

    child.on('error', (error) => {
      this.logger?.error?.(
        {
          server: this.server.name,
          details: toErrorMessage(error)
        },
        'MCP process error.'
      );
      this.teardownProcess(error);
    });

    child.on('exit', (code, signal) => {
      this.logger?.warn?.(
        {
          server: this.server.name,
          code,
          signal
        },
        'MCP process exited.'
      );
      this.teardownProcess(new Error(`MCP process exited: code=${code}, signal=${signal}`), false);
    });

    await this.withTimeout(
      (async () => {
        await this.sendRequest('initialize', {
          protocolVersion: this.server.protocolVersion,
          capabilities: {},
          clientInfo: {
            name: 'next-ai-gateway',
            version: '1.0.0'
          }
        });

        await this.sendNotification('notifications/initialized', {});
        this.isInitialized = true;
      })(),
      this.server.startupTimeoutMs,
      `MCP initialize timeout: ${this.server.name}`
    );
  }

  private handleStdoutData(chunk: Buffer): void {
    if (this.server.stdioMessageMode === 'newline-json') {
      this.handleNewlineJsonStdoutData(chunk);
      return;
    }

    this.handleContentLengthStdoutData(chunk);
  }

  private handleContentLengthStdoutData(chunk: Buffer): void {
    this.stdoutBuffer = Buffer.concat([this.stdoutBuffer, chunk]);

    while (true) {
      const headerEnd = this.stdoutBuffer.indexOf('\r\n\r\n');
      if (headerEnd < 0) {
        return;
      }

      const headerText = this.stdoutBuffer.slice(0, headerEnd).toString('utf8');
      const contentLength = readContentLength(headerText);
      if (contentLength === undefined || contentLength < 0) {
        this.logger?.warn?.(
          {
            server: this.server.name,
            headers: headerText
          },
          'Invalid MCP frame header.'
        );
        this.stdoutBuffer = Buffer.alloc(0);
        return;
      }

      const payloadStart = headerEnd + 4;
      const payloadEnd = payloadStart + contentLength;
      if (this.stdoutBuffer.length < payloadEnd) {
        return;
      }

      const body = this.stdoutBuffer.slice(payloadStart, payloadEnd).toString('utf8');
      this.stdoutBuffer = this.stdoutBuffer.slice(payloadEnd);
      this.handleMessage(body);
    }
  }

  private handleNewlineJsonStdoutData(chunk: Buffer): void {
    this.stdoutBuffer = Buffer.concat([this.stdoutBuffer, chunk]);

    while (true) {
      const lineEnd = this.stdoutBuffer.indexOf(0x0a);
      if (lineEnd < 0) {
        return;
      }

      let line = this.stdoutBuffer.slice(0, lineEnd);
      this.stdoutBuffer = this.stdoutBuffer.slice(lineEnd + 1);

      if (line.length > 0 && line[line.length - 1] === 0x0d) {
        line = line.slice(0, line.length - 1);
      }

      const body = line.toString('utf8').trim();
      if (!body) {
        continue;
      }

      this.handleMessage(body);
    }
  }

  private handleMessage(rawPayload: string): void {
    let payload: unknown;
    try {
      payload = JSON.parse(rawPayload);
    } catch (error) {
      this.logger?.warn?.(
        {
          server: this.server.name,
          details: toErrorMessage(error),
          payload: rawPayload
        },
        'Failed to parse MCP payload.'
      );
      return;
    }

    if (!isObject(payload) || payload.jsonrpc !== '2.0' || !('id' in payload)) {
      return;
    }

    const idValue = payload.id;
    const id = typeof idValue === 'number' || typeof idValue === 'string' ? String(idValue) : undefined;
    if (!id) {
      return;
    }

    const pending = this.pendingRequests.get(id);
    if (!pending) {
      return;
    }

    clearTimeout(pending.timer);
    this.pendingRequests.delete(id);

    if ('error' in payload && isObject(payload.error)) {
      const errorPayload = payload as unknown as JsonRpcErrorResponse;
      pending.reject(
        new Error(
          `MCP request failed (${this.server.name}): ${errorPayload.error.code} ${errorPayload.error.message}`
        )
      );
      return;
    }

    if ('result' in payload) {
      const successPayload = payload as unknown as JsonRpcSuccessResponse;
      pending.resolve(successPayload.result);
      return;
    }

    pending.reject(new Error(`MCP response missing result: ${this.server.name}`));
  }

  private async sendRequest(method: string, params: unknown): Promise<unknown> {
    const child = this.child;
    if (!child || !child.stdin.writable) {
      throw new Error(`MCP process is not writable: ${this.server.name}`);
    }

    const id = String(++this.requestId);
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    const responsePromise = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`MCP request timeout (${this.server.name}): ${method}`));
      }, this.server.requestTimeoutMs);

      this.pendingRequests.set(id, {
        resolve,
        reject,
        timer
      });
    });

    this.writeFramedMessage(request);
    return responsePromise;
  }

  private async sendNotification(method: string, params: unknown): Promise<void> {
    this.writeFramedMessage({
      jsonrpc: '2.0',
      method,
      params
    });
  }

  private writeFramedMessage(payload: Record<string, unknown>): void {
    const child = this.child;
    if (!child || !child.stdin.writable) {
      throw new Error(`MCP process is not writable: ${this.server.name}`);
    }

    const body = JSON.stringify(payload);
    if (this.server.stdioMessageMode === 'newline-json') {
      child.stdin.write(`${body}\n`, 'utf8');
      return;
    }

    const frame = `Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n\r\n${body}`;
    child.stdin.write(frame, 'utf8');
  }

  private teardownProcess(error: unknown, killChild = true): void {
    for (const [id, pending] of this.pendingRequests.entries()) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pendingRequests.delete(id);
    }

    if (killChild && this.child && !this.child.killed) {
      this.child.kill('SIGTERM');
    }

    this.child = undefined;
    this.isInitialized = false;
    this.stdoutBuffer = Buffer.alloc(0);
  }

  private async withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(() => reject(new Error(message)), timeoutMs);
        })
      ]);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }
}

class WebSocketMcpServerClient implements McpServerClient {
  private socket?: WebSocket;
  private requestId = 0;
  private isInitialized = false;
  private connectPromise?: Promise<void>;
  private readonly pendingRequests = new Map<
    string,
    {
      resolve: (value: unknown) => void;
      reject: (reason?: unknown) => void;
      timer: NodeJS.Timeout;
    }
  >();

  constructor(
    private readonly server: AgentMcpWebSocketServerConfig,
    private readonly logger?: AgentRuntimeLogger
  ) {}

  async listTools(): Promise<AgentToolDefinition[]> {
    await this.ensureConnected();

    const tools: AgentToolDefinition[] = [];
    let cursor: string | undefined;

    while (true) {
      const result = await this.sendRequest('tools/list', cursor ? { cursor } : {});
      if (!isObject(result) || !Array.isArray(result.tools)) {
        throw new Error(`Invalid tools/list response from MCP server ${this.server.name}`);
      }

      for (const rawTool of result.tools) {
        if (!isObject(rawTool) || typeof rawTool.name !== 'string') {
          continue;
        }

        tools.push({
          name: rawTool.name,
          description:
            typeof rawTool.description === 'string'
              ? rawTool.description
              : `Tool from MCP server ${this.server.name}`,
          inputSchema: isObject(rawTool.inputSchema)
            ? (rawTool.inputSchema as Record<string, unknown>)
            : undefined
        });
      }

      cursor = typeof result.nextCursor === 'string' ? result.nextCursor : undefined;
      if (!cursor) {
        break;
      }
    }

    return tools;
  }

  async callTool(name: string, args: Record<string, unknown>, meta?: Record<string, unknown>): Promise<unknown> {
    await this.ensureConnected();

    const params: Record<string, unknown> = {
      name,
      arguments: args
    };
    if (meta && Object.keys(meta).length > 0) {
      params._meta = meta;
    }
    const result = await this.sendRequest('tools/call', params);

    if (isObject(result) && result.isError === true) {
      throw new Error(extractMcpErrorMessage(result));
    }

    return result;
  }

  async close(): Promise<void> {
    this.teardownConnection(new Error(`MCP websocket client closed: ${this.server.name}`));
  }

  private async ensureConnected(): Promise<void> {
    if (this.isInitialized && this.socket && this.socket.readyState === WebSocket.OPEN) {
      return;
    }

    if (this.connectPromise) {
      await this.connectPromise;
      return;
    }

    this.connectPromise = this.connect();
    try {
      await this.connectPromise;
    } finally {
      this.connectPromise = undefined;
    }
  }

  private async connect(): Promise<void> {
    this.teardownConnection(new Error(`Reset MCP websocket connection: ${this.server.name}`), false);

    const socket = await this.createSocket();
    this.socket = socket;
    this.isInitialized = false;

    socket.on('message', (raw) => {
      this.handleMessage(raw);
    });

    socket.on('error', (error) => {
      this.logger?.warn?.(
        {
          server: this.server.name,
          details: toErrorMessage(error)
        },
        'MCP websocket error.'
      );
      this.teardownConnection(error, false);
    });

    socket.on('close', (code, reason) => {
      this.logger?.warn?.(
        {
          server: this.server.name,
          code,
          reason: rawToText(reason)
        },
        'MCP websocket connection closed.'
      );
      this.teardownConnection(
        new Error(`MCP websocket closed: server=${this.server.name}, code=${code}`),
        false
      );
    });

    await this.withTimeout(
      (async () => {
        await this.sendRequest('initialize', {
          protocolVersion: this.server.protocolVersion,
          capabilities: {},
          clientInfo: {
            name: 'next-ai-gateway',
            version: '1.0.0'
          }
        });

        await this.sendNotification('notifications/initialized', {});
        this.isInitialized = true;
      })(),
      this.server.startupTimeoutMs,
      `MCP websocket initialize timeout: ${this.server.name}`
    );
  }

  private async createSocket(): Promise<WebSocket> {
    const headers = this.resolveHeaders();

    const socket = new WebSocket(this.server.url, {
      headers,
      handshakeTimeout: this.server.startupTimeoutMs
    });

    await this.withTimeout(
      new Promise<void>((resolve, reject) => {
        const onOpen = (): void => {
          cleanup();
          resolve();
        };

        const onError = (error: Error): void => {
          cleanup();
          reject(error);
        };

        const onClose = (code: number): void => {
          cleanup();
          reject(new Error(`MCP websocket closed before open: server=${this.server.name}, code=${code}`));
        };

        const cleanup = (): void => {
          socket.off('open', onOpen);
          socket.off('error', onError);
          socket.off('close', onClose);
        };

        socket.on('open', onOpen);
        socket.on('error', onError);
        socket.on('close', onClose);
      }),
      this.server.startupTimeoutMs,
      `MCP websocket connect timeout: ${this.server.name}`
    );

    return socket;
  }

  private resolveHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      ...this.server.headers
    };

    if (!this.hasAuthHeader(headers)) {
      const apiKey = this.resolveApiKey();
      if (apiKey) {
        headers.authorization = `Bearer ${apiKey}`;
      }
    }

    return headers;
  }

  private resolveApiKey(): string | undefined {
    if (this.server.apiKey) {
      return this.server.apiKey.trim();
    }

    if (this.server.apiKeyEnv) {
      const fromEnv = process.env[this.server.apiKeyEnv];
      return fromEnv?.trim() || undefined;
    }

    return undefined;
  }

  private hasAuthHeader(headers: Record<string, string>): boolean {
    for (const [key, value] of Object.entries(headers)) {
      if (!value.trim()) {
        continue;
      }

      const lower = key.toLowerCase();
      if (lower === 'authorization' || lower === 'x-api-key' || lower === 'x-mcp-key') {
        return true;
      }
    }

    return false;
  }

  private handleMessage(rawPayload: RawData): void {
    let payload: unknown;
    try {
      payload = JSON.parse(rawToText(rawPayload));
    } catch (error) {
      this.logger?.warn?.(
        {
          server: this.server.name,
          details: toErrorMessage(error)
        },
        'Failed to parse MCP websocket payload.'
      );
      return;
    }

    if (!isObject(payload) || payload.jsonrpc !== '2.0' || !('id' in payload)) {
      return;
    }

    const idValue = payload.id;
    const id = typeof idValue === 'number' || typeof idValue === 'string' ? String(idValue) : undefined;
    if (!id) {
      return;
    }

    const pending = this.pendingRequests.get(id);
    if (!pending) {
      return;
    }

    clearTimeout(pending.timer);
    this.pendingRequests.delete(id);

    if ('error' in payload && isObject(payload.error)) {
      const errorPayload = payload as unknown as JsonRpcErrorResponse;
      pending.reject(
        new Error(
          `MCP request failed (${this.server.name}): ${errorPayload.error.code} ${errorPayload.error.message}`
        )
      );
      return;
    }

    if ('result' in payload) {
      const successPayload = payload as unknown as JsonRpcSuccessResponse;
      pending.resolve(successPayload.result);
      return;
    }

    pending.reject(new Error(`MCP response missing result: ${this.server.name}`));
  }

  private async sendRequest(method: string, params: unknown): Promise<unknown> {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error(`MCP websocket is not open: ${this.server.name}`);
    }

    const id = String(++this.requestId);
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    const responsePromise = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`MCP request timeout (${this.server.name}): ${method}`));
      }, this.server.requestTimeoutMs);

      this.pendingRequests.set(id, {
        resolve,
        reject,
        timer
      });
    });

    try {
      this.sendJsonPayload(request, id);
    } catch (error) {
      const pending = this.pendingRequests.get(id);
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingRequests.delete(id);
        pending.reject(error);
      }
    }

    return responsePromise;
  }

  private async sendNotification(method: string, params: unknown): Promise<void> {
    this.sendJsonPayload({
      jsonrpc: '2.0',
      method,
      params
    });
  }

  private sendJsonPayload(payload: Record<string, unknown>, requestId?: string): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error(`MCP websocket is not open: ${this.server.name}`);
    }

    const message = JSON.stringify(payload);
    socket.send(message, (error) => {
      if (!error || !requestId) {
        return;
      }

      const pending = this.pendingRequests.get(requestId);
      if (!pending) {
        return;
      }

      clearTimeout(pending.timer);
      this.pendingRequests.delete(requestId);
      pending.reject(error);
    });
  }

  private teardownConnection(error: unknown, closeSocket = true): void {
    for (const [id, pending] of this.pendingRequests.entries()) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pendingRequests.delete(id);
    }

    const socket = this.socket;
    this.socket = undefined;
    this.isInitialized = false;

    if (!socket) {
      return;
    }

    socket.removeAllListeners('message');
    socket.removeAllListeners('error');
    socket.removeAllListeners('close');

    if (
      closeSocket &&
      (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)
    ) {
      socket.terminate();
    }
  }

  private async withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(() => reject(new Error(message)), timeoutMs);
        })
      ]);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }
}

export function createMcpAgentToolProvider(
  options: CreateMcpAgentToolProviderOptions
): AgentToolProvider {
  return new McpAgentToolProvider(options);
}

const CODE_TOOL_DEFAULT_TIMEOUT_MS = 180_000;
const CODE_TOOL_MIN_TIMEOUT_MS = 120_000;
const CODE_TOOL_MAX_TIMEOUT_MS = 900_000;
const CODE_TOOL_DEFAULT_TOP_K = 5;
const CODE_TOOL_MAX_TOP_K = 20;
const CODE_TOOL_RESERVED_NAMESPACE_NAMES = new Set([
  'callTool',
  'tools',
  'input',
  'console',
  'mcp',
  'Deno',
  'process',
  'global',
  'require',
  'module',
  'exports',
  'Function',
  'eval',
  'constructor',
  '__proto__',
  'prototype'
]);
const CODE_TOOL_RESERVED_METHOD_NAMES = new Set(['constructor', '__proto__', 'prototype']);

function toJsLiteral(value: unknown): string {
  try {
    const encoded = JSON.stringify(value);
    return encoded === undefined ? 'undefined' : encoded;
  } catch {
    return 'undefined';
  }
}

function toJsonSafeValue(value: unknown): unknown {
  try {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) {
      return null;
    }
    return JSON.parse(encoded) as unknown;
  } catch {
    return String(value);
  }
}

function isSandboxToolCallMessage(value: unknown): value is SandboxToolCallMessage {
  return (
    isObject(value) &&
    value.type === 'tool_call' &&
    typeof value.id === 'number' &&
    typeof value.tool === 'string'
  );
}

function isSandboxResultMessage(value: unknown): value is SandboxResultMessage {
  return isObject(value) && value.type === 'result';
}

function isSandboxErrorMessage(value: unknown): value is SandboxErrorMessage {
  return isObject(value) && value.type === 'error' && typeof value.error === 'string';
}

function buildCodeToolNamespaceBindings(entries: CodeToolCatalogEntry[]): CodeToolNamespaceBindings {
  const grouped = new Map<string, CodeToolCatalogEntry[]>();
  for (const entry of entries) {
    const existing = grouped.get(entry.serverNamespace);
    if (existing) {
      existing.push(entry);
    } else {
      grouped.set(entry.serverNamespace, [entry]);
    }
  }

  const result: CodeToolNamespaceBindings = {
    mcp: {},
    topLevel: {}
  };
  const usedNamespaceNames = new Set<string>(CODE_TOOL_RESERVED_NAMESPACE_NAMES);

  for (const [serverNamespace, groupEntries] of [...grouped.entries()].sort((a, b) =>
    a[0].localeCompare(b[0])
  )) {
    const namespaceName = createUniqueCodeToolMemberName(
      toCodeToolMethodName(serverNamespace),
      usedNamespaceNames
    );
    const methodMap: Record<string, string> = {};
    const usedMethodNames = new Set<string>(CODE_TOOL_RESERVED_METHOD_NAMES);
    for (const entry of groupEntries.slice().sort((a, b) => a.toolName.localeCompare(b.toolName))) {
      const methodName = createUniqueCodeToolMemberName(
        toCodeToolMethodName(entry.remoteToolName),
        usedMethodNames
      );
      methodMap[methodName] = entry.toolName;
    }

    result.mcp[namespaceName] = methodMap;
    result.topLevel[namespaceName] = methodMap;
  }

  return result;
}

function buildCodeToolDenoRunnerScript(
  code: string,
  namespaceBindings: CodeToolNamespaceBindings
): string {
  const namespaceBindingsLiteral = toJsLiteral(namespaceBindings);
  return [
    'const __Deno = Deno;',
    `const __namespaceBindings = ${namespaceBindingsLiteral};`,
    'const __encoder = new TextEncoder();',
    'const __decoder = new TextDecoder();',
    'let __seq = 0;',
    'const __pending = new Map();',
    'const __inflightToolCalls = new Set();',
    'let __lastToolOutput;',
    '',
    'function __stringifyError(error) {',
    '  return error instanceof Error ? error.message : String(error);',
    '}',
    '',
    'function __stderrLog(...args) {',
    '  const text = args.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(" ");',
    '  __Deno.stderr.writeSync(__encoder.encode(text + "\\n"));',
    '}',
    '',
    'Object.defineProperty(globalThis, "console", {',
    '  value: Object.freeze({',
    '    log: (...args) => __stderrLog(...args),',
    '    warn: (...args) => __stderrLog(...args),',
    '    error: (...args) => __stderrLog(...args),',
    '  }),',
    '  writable: false,',
    '  configurable: false,',
    '});',
    '',
    'for (const key of ["Deno", "process", "global", "require", "module", "exports", "Function", "eval"]) {',
    '  try {',
    '    Object.defineProperty(globalThis, key, {',
    '      value: undefined,',
    '      writable: false,',
    '      configurable: false,',
    '      enumerable: false,',
    '    });',
    '  } catch {',
    '    // ignore',
    '  }',
    '}',
    '',
    'async function __emit(message) {',
    '  const payload = JSON.stringify(message);',
    '  await __Deno.stdout.write(__encoder.encode(payload + "\\n"));',
    '}',
    '',
    'function __trackToolCall(promise) {',
    '  __inflightToolCalls.add(promise);',
    '  promise',
    '    .then((output) => {',
    '      __lastToolOutput = output;',
    '    })',
    '    .catch(() => {',
    '      // ignore',
    '    })',
    '    .finally(() => {',
    '      __inflightToolCalls.delete(promise);',
    '    });',
    '  return promise;',
    '}',
    '',
    'async function __waitForInflightToolCalls() {',
    '  while (__inflightToolCalls.size > 0) {',
    '    await Promise.allSettled([...__inflightToolCalls]);',
    '  }',
    '}',
    '',
    'async function __callToolRaw(toolName, input) {',
    '  const id = ++__seq;',
    '  const normalizedInput = input === undefined ? {} : input;',
    '  return await new Promise((resolve, reject) => {',
    '    __pending.set(id, { resolve, reject });',
    '    __emit({ type: "tool_call", id, tool: toolName, input: normalizedInput }).catch((error) => {',
    '      __pending.delete(id);',
    '      reject(error);',
    '    });',
    '  });',
    '}',
    '',
    'function callTool(toolName, input) {',
    '  return __trackToolCall(__callToolRaw(toolName, input));',
    '}',
    '',
    'const tools = new Proxy({}, {',
    '  get(_target, property) {',
    '    if (typeof property !== "string") return undefined;',
    '    if (property === "constructor" || property === "__proto__" || property === "prototype") {',
    '      return undefined;',
    '    }',
    '    if (property === "call") {',
    '      return (toolName, input) => callTool(toolName, input);',
    '    }',
    '    return (input) => callTool(property, input);',
    '  },',
    '  set() { return false; },',
    '  defineProperty() { return false; },',
    '  deleteProperty() { return false; },',
    '});',
    '',
    'function __createNamespaceObject(methodMap) {',
    '  const namespaceObject = {};',
    '  for (const [methodName, toolName] of Object.entries(methodMap)) {',
    '    if (methodName === "constructor" || methodName === "__proto__" || methodName === "prototype") {',
    '      continue;',
    '    }',
    '    Object.defineProperty(namespaceObject, methodName, {',
    '      value: (input) => callTool(toolName, input),',
    '      writable: false,',
    '      configurable: false,',
    '      enumerable: true,',
    '    });',
    '  }',
    '  return Object.freeze(namespaceObject);',
    '}',
    '',
    'function __safeDefineGlobal(name, value) {',
    '  if (name === "constructor" || name === "__proto__" || name === "prototype") return;',
    '  if (name in globalThis) return;',
    '  try {',
    '    Object.defineProperty(globalThis, name, {',
    '      value,',
    '      writable: false,',
    '      configurable: false,',
    '      enumerable: false,',
    '    });',
    '  } catch {',
    '    // ignore',
    '  }',
    '}',
    '',
    'const __mcpNamespaceObject = {};',
    'for (const [namespaceName, methodMap] of Object.entries(__namespaceBindings.mcp)) {',
    '  const namespaceObject = __createNamespaceObject(methodMap);',
    '  Object.defineProperty(__mcpNamespaceObject, namespaceName, {',
    '    value: namespaceObject,',
    '    writable: false,',
    '    configurable: false,',
    '    enumerable: true,',
    '  });',
    '  __safeDefineGlobal(namespaceName, namespaceObject);',
    '}',
    'if (Object.keys(__mcpNamespaceObject).length > 0) {',
    '  __safeDefineGlobal("mcp", Object.freeze(__mcpNamespaceObject));',
    '}',
    '',
    'for (const [namespaceName, methodMap] of Object.entries(__namespaceBindings.topLevel)) {',
    '  __safeDefineGlobal(namespaceName, __createNamespaceObject(methodMap));',
    '}',
    '',
    '(async () => {',
    '  try {',
    '    let result = await (async () => {',
    '      const process = undefined;',
    '      const Deno = undefined;',
    '      const global = undefined;',
    '      const require = undefined;',
    '      const module = undefined;',
    '      const exports = undefined;',
    '      const Function = undefined;',
    code,
    '    })();',
    '    await __waitForInflightToolCalls();',
    '    if (result === undefined && __lastToolOutput !== undefined) {',
    '      result = __lastToolOutput;',
    '    }',
    '    await __emit({ type: "result", result });',
    '  } catch (error) {',
    '    await __emit({ type: "error", error: __stringifyError(error) });',
    '  } finally {',
    '    __Deno.exit(0);',
    '  }',
    '})();',
    '',
    'let __buffer = "";',
    'for await (const chunk of __Deno.stdin.readable) {',
    '  __buffer += __decoder.decode(chunk, { stream: true });',
    '  while (true) {',
    '    const newlineIndex = __buffer.indexOf("\\n");',
    '    if (newlineIndex < 0) break;',
    '    const line = __buffer.slice(0, newlineIndex).trim();',
    '    __buffer = __buffer.slice(newlineIndex + 1);',
    '    if (line.length === 0) continue;',
    '    let message;',
    '    try {',
    '      message = JSON.parse(line);',
    '    } catch {',
    '      continue;',
    '    }',
    '    if (message && message.type === "tool_result" && typeof message.id === "number") {',
    '      const pending = __pending.get(message.id);',
    '      if (!pending) continue;',
    '      __pending.delete(message.id);',
    '      if (message.ok) {',
    '        pending.resolve(message.output);',
    '      } else {',
    '        pending.reject(new Error(typeof message.error === "string" ? message.error : "Tool call failed"));',
    '      }',
    '    }',
    '  }',
    '}',
    ''
  ].join('\n');
}

function buildCodeToolMetaDefinitions(): AgentToolDefinition[] {
  return [
    {
      name: 'code_tool.search',
      description:
        '接收自然语言查询，返回最相关的 MCP 内部工具候选，便于后续通过 code_tool.call 调用。',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Natural-language search query for finding relevant internal tools.'
          },
          code: {
            type: 'string',
            description: 'Optional orchestration code snippet used as additional search context.'
          },
          topK: {
            type: 'number',
            description: `Maximum number of tools to return (1-${CODE_TOOL_MAX_TOP_K}).`
          },
          model: {
            type: 'string',
            description: 'Optional model hint for compatibility with code-tool API.'
          },
          maxTurns: {
            type: 'number',
            description: 'Optional search-agent turn limit hint for compatibility with code-tool API.'
          }
        },
        required: ['query'],
        additionalProperties: false
      }
    },
    {
      name: 'code_tool.call',
      description:
        '按名称调用内部 MCP 工具。输入形如 {"tool":"mcp.<server>.<tool>","arguments":{...}}。',
      inputSchema: {
        type: 'object',
        properties: {
          tool: {
            type: 'string',
            description: 'Target tool name or alias, e.g. "mcp.filesystem.read_file".'
          },
          arguments: {
            type: 'object',
            description: 'JSON object arguments passed to the target tool.',
            additionalProperties: true
          },
          timeoutMs: {
            type: 'number',
            description: `Optional timeout override in ms (${CODE_TOOL_MIN_TIMEOUT_MS}-${CODE_TOOL_MAX_TIMEOUT_MS}).`
          }
        },
        required: ['tool'],
        additionalProperties: false
      }
    },
    {
      name: 'code_tool.runCode',
      description:
        '执行编排代码。代码中可通过 callTool/tools/mcp.<server>.<toolMethod> 调用内部 MCP 工具。',
      inputSchema: {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'Async JavaScript/TypeScript-like orchestration code to execute.'
          },
          timeoutMs: {
            type: 'number',
            description: `Optional timeout override in ms (${CODE_TOOL_MIN_TIMEOUT_MS}-${CODE_TOOL_MAX_TIMEOUT_MS}).`
          }
        },
        required: ['code'],
        additionalProperties: false
      }
    }
  ];
}

function rankCodeToolCatalogEntries(
  entries: CodeToolCatalogEntry[],
  query: string
): CodeToolCatalogEntry[] {
  const normalizedQuery = query.trim().toLowerCase();
  const tokens = normalizedQuery
    .split(/[^a-zA-Z0-9_]+/g)
    .map((token) => token.trim())
    .filter(Boolean);

  const scored = entries.map((entry) => {
    let score = 0;
    const toolName = entry.toolName.toLowerCase();
    const alias = entry.alias.toLowerCase();
    const description = entry.description.toLowerCase();
    const serverName = entry.serverName.toLowerCase();
    const remoteToolName = entry.remoteToolName.toLowerCase();

    if (toolName === normalizedQuery || alias === normalizedQuery) {
      score += 160;
    }
    if (toolName.includes(normalizedQuery) || alias.includes(normalizedQuery)) {
      score += 90;
    }
    if (remoteToolName.includes(normalizedQuery)) {
      score += 70;
    }
    if (serverName.includes(normalizedQuery)) {
      score += 40;
    }
    if (description.includes(normalizedQuery)) {
      score += 25;
    }

    for (const token of tokens) {
      if (toolName.includes(token)) {
        score += 20;
      }
      if (alias.includes(token)) {
        score += 16;
      }
      if (remoteToolName.includes(token)) {
        score += 14;
      }
      if (serverName.includes(token)) {
        score += 8;
      }
      if (description.includes(token)) {
        score += 6;
      }
    }

    return {
      entry,
      score
    };
  });

  scored.sort((left, right) => {
    if (left.score !== right.score) {
      return right.score - left.score;
    }

    return left.entry.toolName.localeCompare(right.entry.toolName);
  });

  return scored.filter((item) => item.score > 0).map((item) => item.entry);
}

function resolveCodeToolTopK(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return CODE_TOOL_DEFAULT_TOP_K;
  }

  const normalized = Math.floor(value);
  if (normalized <= 0) {
    return CODE_TOOL_DEFAULT_TOP_K;
  }

  return Math.min(normalized, CODE_TOOL_MAX_TOP_K);
}

function resolveCodeToolTimeoutMs(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return CODE_TOOL_DEFAULT_TIMEOUT_MS;
  }

  const normalized = Math.floor(value);
  if (normalized <= 0) {
    return CODE_TOOL_DEFAULT_TIMEOUT_MS;
  }

  return Math.max(CODE_TOOL_MIN_TIMEOUT_MS, Math.min(normalized, CODE_TOOL_MAX_TIMEOUT_MS));
}

function buildCodeToolMcpProxy(
  entries: CodeToolCatalogEntry[],
  callTool: (toolNameOrAlias: string, rawArgs?: unknown) => Promise<unknown>
): Record<string, Record<string, (rawArgs?: unknown) => Promise<unknown>>> {
  const grouped = new Map<string, CodeToolCatalogEntry[]>();
  for (const entry of entries) {
    const list = grouped.get(entry.serverNamespace);
    if (list) {
      list.push(entry);
    } else {
      grouped.set(entry.serverNamespace, [entry]);
    }
  }

  const root: Record<string, Record<string, (rawArgs?: unknown) => Promise<unknown>>> = {};
  const usedNamespaceNames = new Set<string>();
  for (const [serverNamespace, tools] of grouped.entries()) {
    const namespaceName = createUniqueCodeToolMemberName(
      toCodeToolMethodName(serverNamespace),
      usedNamespaceNames
    );
    const namespaceObject: Record<string, (rawArgs?: unknown) => Promise<unknown>> = {};
    const usedMemberNames = new Set<string>();

    for (const entry of tools.sort((left, right) => left.remoteToolName.localeCompare(right.remoteToolName))) {
      const memberName = createUniqueCodeToolMemberName(
        toCodeToolMethodName(entry.remoteToolName),
        usedMemberNames
      );
      namespaceObject[memberName] = (rawArgs?: unknown) => callTool(entry.toolName, rawArgs);
    }

    root[namespaceName] = namespaceObject;
  }

  return root;
}

function buildCodeToolExecutableReference(
  entry: CodeToolCatalogEntry,
  includeMcpRoot: boolean
): string {
  const parts = [
    ...(includeMcpRoot ? ['mcp'] : []),
    toCodeToolMethodName(entry.serverNamespace),
    toCodeToolMethodName(entry.remoteToolName)
  ];
  return parts.join('.');
}

function createUniqueCodeToolMemberName(base: string, used: Set<string>): string {
  let candidate = base || 'tool';
  let index = 2;
  while (used.has(candidate)) {
    candidate = `${base}_${index}`;
    index += 1;
  }

  used.add(candidate);
  return candidate;
}

function toCodeToolMethodName(remoteToolName: string): string {
  const normalized = remoteToolName.replace(/[^a-zA-Z0-9_]/g, '_').replace(/_+/g, '_');
  if (!normalized) {
    return 'tool';
  }

  return /^\d/.test(normalized) ? `_${normalized}` : normalized;
}

function toCodeToolAlias(toolName: string): string {
  const normalized = toolName.replace(/[^a-zA-Z0-9_]/g, '_').replace(/_+/g, '_');
  if (!normalized) {
    return 'tool';
  }

  return /^\d/.test(normalized) ? `_${normalized}` : normalized;
}

function sanitizeCodeToolServerNamespace(serverName: string): string {
  const normalized = serverName.trim().replace(/[^a-zA-Z0-9_-]/g, '_');
  if (!normalized) {
    return 'default';
  }

  return /^\d/.test(normalized) ? `_${normalized}` : normalized;
}

async function runWithTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => {
          const timeoutError = new Error(message);
          timeoutError.name = 'TimeoutError';
          reject(timeoutError);
        }, timeoutMs);
      })
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

interface ParsedMcpCliInvocation {
  tokens: string[];
  directArgs?: Record<string, unknown>;
}

interface ParsedMcpCliSubcommandArgs {
  ok: boolean;
  args: Record<string, unknown>;
  error?: string;
}

function buildMcpCliWrapperDefinition(
  serverName: string,
  discoveredTools: AgentToolDefinition[]
): AgentToolDefinition {
  const previewNames = discoveredTools.slice(0, 5).map((tool) => tool.name);
  const previewSuffix =
    discoveredTools.length > previewNames.length
      ? ` +${discoveredTools.length - previewNames.length} more`
      : '';
  const preview = previewNames.length > 0 ? `${previewNames.join(', ')}${previewSuffix}` : 'none';
  const filesystemFirstCallRule =
    serverName === 'filesystem'
      ? `First call in a session: run "${serverName} --help" to get usage instructions before other commands. `
      : '';
  const commandDescriptionPrefix =
    serverName === 'filesystem'
      ? 'On first call in a session, use "--help" to get usage instructions. '
      : '';

  return {
    name: serverName,
    description:
      `CLI wrapper for MCP server "${serverName}". ` +
      filesystemFirstCallRule +
      `Use "${serverName} --help" or "${serverName} --list-tools". ` +
      'Use exact subcommand names from --list-tools output (including prefixes such as "fs.read"). ' +
      `Discovered tools: ${preview}.`,
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description:
            commandDescriptionPrefix +
            'CLI command tail without the server name. Examples: "--help", "--list-tools", "<tool> <args>", "fs.read /tmp/a.txt", "fs.write --path /tmp/a.txt --content hello".'
        }
      },
      required: ['command'],
      additionalProperties: true
    }
  };
}

function buildMcpCliHelpResult(
  serverName: string,
  discoveredTools: AgentToolDefinition[]
): Record<string, unknown> {
  const toolNames = discoveredTools.map((tool) => tool.name).sort((a, b) => a.localeCompare(b));
  const firstTool = toolNames[0];

  return {
    server: serverName,
    usage: [`${serverName} --help`, `${serverName} --list-tools`, `${serverName} <tool> [args]`],
    argumentFormats: [
      '<tool> {"key":"value"}',
      '<tool> --key value --flag',
      '<tool> key=value',
      '<tool> <positional...>'
    ],
    examples: [
      `${serverName} --list-tools`,
      firstTool ? `${serverName} ${firstTool}` : undefined,
      firstTool ? `${serverName} ${firstTool} {"path":"/tmp/example.txt"}` : undefined
    ].filter((item): item is string => Boolean(item)),
    subcommandRule: 'Use the exact subcommand name returned by --list-tools.',
    toolCount: toolNames.length,
    toolsPreview: toolNames.slice(0, 20)
  };
}

function buildMcpCliListToolsResult(
  serverName: string,
  discoveredTools: AgentToolDefinition[]
): Record<string, unknown> {
  return {
    server: serverName,
    toolCount: discoveredTools.length,
    tools: discoveredTools
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema
      })),
    usageHint: `Call as "${serverName} <tool> [args]". Use "${serverName} --help" for formats.`
  };
}

function buildMcpCliSubcommandHelpResult(
  serverName: string,
  subcommand: string,
  tool: AgentToolDefinition,
  reason?: string
): Record<string, unknown> {
  const schema = isObject(tool.inputSchema)
    ? (tool.inputSchema as Record<string, unknown>)
    : undefined;
  const properties = isObject(schema?.properties)
    ? (schema.properties as Record<string, unknown>)
    : {};
  const required = Array.isArray(schema?.required)
    ? schema.required.filter((item): item is string => typeof item === 'string')
    : [];

  return {
    server: serverName,
    subcommand,
    mode: 'subcommand-help',
    reason: reason ? `Argument parse failed: ${reason}` : 'Subcommand help requested.',
    usage: [
      `${serverName} ${subcommand} --help`,
      `${serverName} ${subcommand} --key value`,
      `${serverName} ${subcommand} key=value`,
      `${serverName} ${subcommand} {"key":"value"}`
    ],
    requiredArgs: required,
    arguments: Object.entries(properties).map(([name, descriptor]) => {
      const normalized = isObject(descriptor) ? (descriptor as Record<string, unknown>) : {};
      return {
        name,
        required: required.includes(name),
        type: readNonEmptyString(normalized.type) || 'any',
        description: readNonEmptyString(normalized.description)
      };
    }),
    inputSchema: tool.inputSchema,
    tips: [
      'Use --key value, --key=value, or a JSON object.',
      'Do not use "key: value" syntax.'
    ]
  };
}

function buildMcpCliSubcommandExecutionError(
  serverName: string,
  subcommand: string,
  tool: AgentToolDefinition,
  error: unknown
): Error {
  const baseMessage = toErrorMessage(error);
  const usageLines = buildMcpCliSubcommandUsageLines(serverName, subcommand, tool);
  const requiredArgs = readMcpCliRequiredArgs(tool.inputSchema);
  const requiredHint =
    requiredArgs.length > 0 ? `Required args: ${requiredArgs.join(', ')}` : undefined;
  const extraTip = /EISDIR/i.test(baseMessage)
    ? 'Tip: the target path is a directory. Use a file path, or switch to a directory/listing subcommand.'
    : undefined;

  return new Error(
    [baseMessage, 'Usage:', ...usageLines.map((line) => `- ${line}`), requiredHint, extraTip]
      .filter(Boolean)
      .join('\n')
  );
}

function buildMcpCliSubcommandUsageLines(
  serverName: string,
  subcommand: string,
  tool: AgentToolDefinition
): string[] {
  const schema = isObject(tool.inputSchema)
    ? (tool.inputSchema as Record<string, unknown>)
    : undefined;
  const properties = isObject(schema?.properties)
    ? (schema.properties as Record<string, unknown>)
    : {};
  const requiredArgs = readMcpCliRequiredArgs(tool.inputSchema);
  const requiredFlags = requiredArgs.map((name) => {
    const descriptor = isObject(properties[name]) ? (properties[name] as Record<string, unknown>) : undefined;
    return `--${name} <${readMcpCliSchemaType(descriptor)}>`;
  });

  const withRequired =
    requiredFlags.length > 0
      ? `${serverName} ${subcommand} ${requiredFlags.join(' ')}`
      : `${serverName} ${subcommand}`;

  return [
    `${serverName} ${subcommand} --help`,
    withRequired,
    `${serverName} ${subcommand} {"key":"value"}`
  ];
}

function readMcpCliRequiredArgs(inputSchema: unknown): string[] {
  if (!isObject(inputSchema) || !Array.isArray(inputSchema.required)) {
    return [];
  }

  return inputSchema.required.filter((item): item is string => typeof item === 'string');
}

function readMcpCliSchemaType(descriptor: Record<string, unknown> | undefined): string {
  if (!descriptor) {
    return 'value';
  }

  const typeValue = descriptor.type;
  if (typeof typeValue === 'string') {
    return typeValue;
  }

  if (Array.isArray(typeValue)) {
    const first = typeValue.find((item) => typeof item === 'string');
    if (typeof first === 'string') {
      return first;
    }
  }

  return 'value';
}

function normalizeMcpTools(tools: AgentToolDefinition[]): AgentToolDefinition[] {
  return tools.map((tool) => ({
    ...tool,
    inputSchema: isObject(tool.inputSchema)
      ? ({ ...tool.inputSchema } as Record<string, unknown>)
      : undefined
  }));
}

function isHelpCommand(token: string): boolean {
  return token === '--help' || token === '-h' || token === 'help';
}

function resolveMcpCliSubcommandAlias(
  command: string,
  discoveredTools: AgentToolDefinition[]
): string | undefined {
  if (!command || command.includes('.')) {
    return undefined;
  }

  const candidates = discoveredTools
    .map((tool) => tool.name)
    .filter((name) => {
      if (name === command) {
        return true;
      }

      const separator = name.lastIndexOf('.');
      if (separator < 0) {
        return false;
      }

      return name.slice(separator + 1) === command;
    });

  if (candidates.length !== 1) {
    return undefined;
  }

  return candidates[0];
}

function parseMcpCliInvocation(args: Record<string, unknown>): ParsedMcpCliInvocation {
  const fromCommand = args.command;
  if (typeof fromCommand === 'string') {
    return {
      tokens: tokenizeMcpCliCommand(fromCommand)
    };
  }

  const fromCommandList = readStringTokenList(fromCommand);
  if (fromCommandList) {
    return {
      tokens: fromCommandList
    };
  }

  const fromArgv = readStringTokenList(args.argv) || readStringTokenList(args.args);
  if (fromArgv) {
    return {
      tokens: fromArgv
    };
  }

  const tool = readNonEmptyString(args.tool) || readNonEmptyString(args.subcommand);
  if (!tool) {
    return {
      tokens: []
    };
  }

  const nestedArgs =
    isObject(args.arguments) ? (args.arguments as Record<string, unknown>) : undefined;
  if (nestedArgs) {
    return {
      tokens: [tool],
      directArgs: { ...nestedArgs }
    };
  }

  const nestedParams = isObject(args.params) ? (args.params as Record<string, unknown>) : undefined;
  if (nestedParams) {
    return {
      tokens: [tool],
      directArgs: { ...nestedParams }
    };
  }

  const directArgs: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (
      key === 'tool' ||
      key === 'subcommand' ||
      key === 'command' ||
      key === 'argv' ||
      key === 'args' ||
      key === 'arguments' ||
      key === 'params'
    ) {
      continue;
    }

    directArgs[key] = value;
  }

  return {
    tokens: [tool],
    directArgs: Object.keys(directArgs).length > 0 ? directArgs : undefined
  };
}

function parseMcpCliSubcommandArgs(
  tokens: string[],
  inputSchema?: Record<string, unknown>
): ParsedMcpCliSubcommandArgs {
  if (tokens.length === 0) {
    const emptyArgs: Record<string, unknown> = {};
    const validationError = validateMcpCliSubcommandArgs(emptyArgs, inputSchema);
    if (validationError) {
      return {
        ok: false,
        args: emptyArgs,
        error: validationError
      };
    }

    return {
      ok: true,
      args: emptyArgs
    };
  }

  const merged = tokens.join(' ');
  const trimmedMerged = merged.trim();
  if (trimmedMerged.startsWith('{')) {
    if (!trimmedMerged.endsWith('}')) {
      return {
        ok: false,
        args: {},
        error: 'JSON object arguments must end with "}".'
      };
    }

    try {
      const parsed = JSON.parse(trimmedMerged) as unknown;
      if (!isObject(parsed)) {
        return {
          ok: false,
          args: {},
          error: 'JSON arguments must be an object.'
        };
      }

      const validationError = validateMcpCliSubcommandArgs(parsed, inputSchema);
      if (validationError) {
        return {
          ok: false,
          args: parsed,
          error: validationError
        };
      }

      return {
        ok: true,
        args: parsed
      };
    } catch {
      return {
        ok: false,
        args: {},
        error: 'Failed to parse JSON arguments.'
      };
    }
  }

  const parsed: Record<string, unknown> = {};
  const positional: string[] = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];

    if (token.startsWith('--')) {
      const body = token.slice(2);
      if (!body) {
        continue;
      }

      const separator = body.indexOf('=');
      if (separator > 0) {
        const key = body.slice(0, separator).trim();
        if (!key) {
          continue;
        }
        if (isLikelyKeyColonToken(key)) {
          const normalizedKey = key.slice(0, -1);
          return {
            ok: false,
            args: parsed,
            error: `Unsupported key format "${key}". Use --${normalizedKey} value, ${normalizedKey}=value, or JSON object arguments.`
          };
        }

        parsed[key] = parseMcpCliValue(body.slice(separator + 1));
        continue;
      }

      if (isLikelyKeyColonToken(body)) {
        const normalizedKey = body.slice(0, -1);
        return {
          ok: false,
          args: parsed,
          error: `Unsupported key format "${body}". Use --${normalizedKey} value, ${normalizedKey}=value, or JSON object arguments.`
        };
      }

      const nextToken = tokens[index + 1];
      if (nextToken && !nextToken.startsWith('--')) {
        parsed[body] = parseMcpCliValue(nextToken);
        index += 1;
      } else {
        parsed[body] = true;
      }

      continue;
    }

    const equalsIndex = token.indexOf('=');
    if (equalsIndex > 0) {
      const key = token.slice(0, equalsIndex).trim();
      if (key) {
        parsed[key] = parseMcpCliValue(token.slice(equalsIndex + 1));
        continue;
      }
    }

    if (isLikelyKeyColonToken(token)) {
      const normalizedKey = token.slice(0, -1);
      return {
        ok: false,
        args: parsed,
        error: `Unsupported key format "${token}". Use --${normalizedKey} value, ${normalizedKey}=value, or JSON object arguments.`
      };
    }

    positional.push(token);
  }

  if (positional.length === 0) {
    const validationError = validateMcpCliSubcommandArgs(parsed, inputSchema);
    if (validationError) {
      return {
        ok: false,
        args: parsed,
        error: validationError
      };
    }

    return {
      ok: true,
      args: parsed
    };
  }

  const positionalNames = resolvePositionalArgumentNames(inputSchema, new Set(Object.keys(parsed)));
  const extraTokens: string[] = [];

  for (let index = 0; index < positional.length; index += 1) {
    const value = positional[index];
    const key = positionalNames[index];
    if (key) {
      parsed[key] = parseMcpCliValue(value);
      continue;
    }

    extraTokens.push(value);
  }

  if (extraTokens.length > 0) {
    parsed._ = extraTokens.map((item) => parseMcpCliValue(item));
  }

  const validationError = validateMcpCliSubcommandArgs(parsed, inputSchema);
  if (validationError) {
    return {
      ok: false,
      args: parsed,
      error: validationError
    };
  }

  return {
    ok: true,
    args: parsed
  };
}

function resolvePositionalArgumentNames(
  inputSchema: Record<string, unknown> | undefined,
  existingKeys: Set<string>
): string[] {
  if (!isObject(inputSchema)) {
    return [];
  }

  const properties = isObject(inputSchema.properties)
    ? Object.keys(inputSchema.properties as Record<string, unknown>)
    : [];
  const required = Array.isArray(inputSchema.required)
    ? inputSchema.required
        .filter((item): item is string => typeof item === 'string')
        .filter((item) => properties.includes(item))
    : [];

  const ordered = [...required, ...properties];
  const deduped: string[] = [];
  for (const name of ordered) {
    if (!name || existingKeys.has(name) || deduped.includes(name)) {
      continue;
    }

    deduped.push(name);
  }

  return deduped;
}

function parseMcpCliValue(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) {
    return '';
  }

  if (trimmed === 'true') {
    return true;
  }

  if (trimmed === 'false') {
    return false;
  }

  if (trimmed === 'null') {
    return null;
  }

  if (/^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      return numeric;
    }
  }

  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']')) ||
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return trimmed;
    }
  }

  return trimmed;
}

function validateMcpCliSubcommandArgs(
  args: Record<string, unknown>,
  inputSchema?: Record<string, unknown>
): string | undefined {
  if (!isObject(inputSchema)) {
    return undefined;
  }

  const required = Array.isArray(inputSchema.required)
    ? inputSchema.required.filter((item): item is string => typeof item === 'string')
    : [];
  const missingRequired = required.filter((key) => args[key] === undefined);
  if (missingRequired.length > 0) {
    return `Missing required arguments: ${missingRequired.join(', ')}`;
  }

  const properties = isObject(inputSchema.properties)
    ? Object.keys(inputSchema.properties as Record<string, unknown>)
    : [];
  const additionalPropertiesAllowed = inputSchema.additionalProperties !== false;
  if (!additionalPropertiesAllowed && properties.length > 0) {
    const unknown = Object.keys(args).filter((key) => key !== '_' && !properties.includes(key));
    if (unknown.length > 0) {
      return `Unknown arguments: ${unknown.join(', ')}`;
    }
  }

  return undefined;
}

function isLikelyKeyColonToken(token: string): boolean {
  if (!token.endsWith(':')) {
    return false;
  }
  if (!/^[a-zA-Z_][a-zA-Z0-9_.-]*:$/.test(token)) {
    return false;
  }
  if (/^[a-zA-Z]:$/.test(token)) {
    return false;
  }
  return true;
}

function tokenizeMcpCliCommand(command: string): string[] {
  const trimmed = command.trim();
  if (!trimmed) {
    return [];
  }

  const tokens: string[] = [];
  let current = '';
  let quote: '"' | "'" | undefined;
  let escaping = false;

  for (const char of trimmed) {
    if (escaping) {
      current += char;
      escaping = false;
      continue;
    }

    if (char === '\\') {
      escaping = true;
      continue;
    }

    if (quote) {
      if (char === quote) {
        quote = undefined;
      } else {
        current += char;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }

    if (char.trim() === '') {
      if (current) {
        tokens.push(current);
        current = '';
      }
      continue;
    }

    current += char;
  }

  if (escaping) {
    current += '\\';
  }

  if (current) {
    tokens.push(current);
  }

  return tokens;
}

function readStringTokenList(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const tokens = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);

  return tokens.length > 0 ? tokens : undefined;
}

function readNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
}

function rawToText(raw: RawData | Buffer): string {
  if (typeof raw === 'string') {
    return raw;
  }

  if (Buffer.isBuffer(raw)) {
    return raw.toString('utf8');
  }

  if (Array.isArray(raw)) {
    return Buffer.concat(raw).toString('utf8');
  }

  return Buffer.from(raw).toString('utf8');
}

function readContentLength(headers: string): number | undefined {
  for (const line of headers.split('\r\n')) {
    const separator = line.indexOf(':');
    if (separator <= 0) {
      continue;
    }

    const key = line.slice(0, separator).trim().toLowerCase();
    if (key !== 'content-length') {
      continue;
    }

    const value = Number(line.slice(separator + 1).trim());
    if (!Number.isFinite(value)) {
      return undefined;
    }

    return value;
  }

  return undefined;
}

function extractMcpErrorMessage(result: Record<string, unknown>): string {
  if (Array.isArray(result.content)) {
    const textItems = result.content
      .filter((item): item is Record<string, unknown> => isObject(item))
      .filter((item) => item.type === 'text' && typeof item.text === 'string')
      .map((item) => String(item.text).trim())
      .filter(Boolean);

    if (textItems.length > 0) {
      return textItems.join('\n');
    }
  }

  return 'MCP tool call failed.';
}

interface McpCallMediaReference {
  id: string;
  value: string;
}

function hydrateMcpCallArgumentsFromMeta(
  args: Record<string, unknown>,
  meta: Record<string, unknown> | undefined
): Record<string, unknown> {
  const references = readMcpCallMediaReferences(meta?.virtualMultimodalReferences);
  if (references.length === 0) {
    return args;
  }

  return hydrateMcpCallMediaReferences(args, references) as Record<string, unknown>;
}

function readMcpCallMediaReferences(value: unknown): McpCallMediaReference[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const references: McpCallMediaReference[] = [];
  for (const item of value) {
    if (!isObject(item) || typeof item.id !== 'string' || typeof item.value !== 'string') {
      continue;
    }
    references.push({
      id: item.id,
      value: item.value
    });
  }
  return references;
}

function hydrateMcpCallMediaReferences(value: unknown, references: McpCallMediaReference[]): unknown {
  if (typeof value === 'string') {
    return hydrateMcpCallMediaReferenceString(value, references);
  }

  if (Array.isArray(value)) {
    return value.map((item) => hydrateMcpCallMediaReferences(item, references));
  }

  if (!isObject(value)) {
    return value;
  }

  const hydrated: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    hydrated[key] = hydrateMcpCallMediaReferences(child, references);
  }
  return hydrated;
}

function hydrateMcpCallMediaReferenceString(value: string, references: McpCallMediaReference[]): string {
  const trimmed = value.trim();
  for (const reference of references) {
    if (
      trimmed === reference.id ||
      trimmed === `media_ref:${reference.id}` ||
      trimmed === `[media_ref:${reference.id}]`
    ) {
      return reference.value;
    }
  }

  let hydrated = value;
  for (const reference of references) {
    hydrated = hydrated
      .split(`[media_ref:${reference.id}]`)
      .join(reference.value)
      .split(`media_ref:${reference.id}`)
      .join(reference.value);
  }
  return hydrated;
}

function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) {
    return promise;
  }

  if (signal.aborted) {
    return Promise.reject(createAbortError());
  }

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(createAbortError());
    };

    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
}

function createAbortError(): Error {
  const error = new Error('Agent tool execution aborted.');
  error.name = 'AbortError';
  return error;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
