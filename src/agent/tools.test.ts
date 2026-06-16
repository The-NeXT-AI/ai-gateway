import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { createInitialGuards } from './guards';
import { createInitialTaskState } from './task-state';
import { createTranscriptWindow } from './transcript-window';
import { createMcpAgentToolProvider } from './tools';
import type { AgentToolExecutionInput } from './types';

interface FakeCall {
  name: string;
  args: Record<string, unknown>;
}

function isDenoAvailable(): boolean {
  const result = spawnSync('deno', ['--version'], {
    stdio: 'ignore'
  });
  return !result.error && result.status === 0;
}

describe('McpAgentToolProvider exposure modes', () => {
  it('exposes one CLI wrapper tool per server in server-cli mode', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));

    const tools = await provider.listDefinitions();
    expect(tools.map((tool) => tool.name)).toEqual(['fs']);
    expect(tools[0]?.inputSchema).toBeDefined();

    await provider.close();
  });

  it('supports --list-tools without invoking a remote sub-tool', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    await provider.listDefinitions();

    const result = await provider.execute(
      'fs',
      buildExecutionInput({
        command: '--list-tools'
      })
    );

    expect(calls).toHaveLength(0);
    expect(result).toMatchObject({
      server: 'fs',
      toolCount: 2
    });

    const tools = Array.isArray((result as Record<string, unknown>).tools)
      ? ((result as Record<string, unknown>).tools as Array<Record<string, unknown>>)
      : [];
    expect(tools.map((item) => item.name)).toEqual(['read', 'write']);

    await provider.close();
  });

  it('hides wrapper when server is unreachable and restores it after recovery', async () => {
    const calls: FakeCall[] = [];
    let shouldFail = true;
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', {
      listTools: async () => {
        if (shouldFail) {
          throw new Error('temporary list failure');
        }

        return [
          {
            name: 'fs.read',
            description: 'Read file',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string'
                }
              },
              required: ['path']
            }
          }
        ];
      },
      callTool: async (name: string, args: Record<string, unknown>) => {
        calls.push({ name, args });
        return {
          ok: true,
          name,
          args
        };
      },
      close: async () => {}
    });

    const beforeRecovery = await provider.listDefinitions();
    expect(beforeRecovery).toHaveLength(0);

    shouldFail = false;
    (provider as any).lastRefreshAt = 0;
    const afterRecovery = await provider.listDefinitions();
    expect(afterRecovery.map((tool) => tool.name)).toEqual(['filesystem']);

    const result = await provider.execute(
      'filesystem',
      buildExecutionInput({
        command: '--list-tools'
      })
    );

    expect(calls).toHaveLength(0);
    expect((result as Record<string, unknown>).toolCount).toBe(1);

    const listed = Array.isArray((result as Record<string, unknown>).tools)
      ? ((result as Record<string, unknown>).tools as Array<Record<string, unknown>>)
      : [];
    expect(listed.map((item) => item.name)).toEqual(['fs.read']);

    await provider.close();
  });

  it('returns tool unavailable when server is unreachable', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', {
      listTools: async () => {
        throw new Error('connect ECONNREFUSED 127.0.0.1:3102');
      },
      callTool: async (name: string, args: Record<string, unknown>) => {
        calls.push({ name, args });
        return {
          ok: true,
          name,
          args
        };
      },
      close: async () => {}
    });

    const tools = await provider.listDefinitions();
    expect(tools).toHaveLength(0);

    const execute = provider.execute(
      'filesystem',
      buildExecutionInput({
        command: '--list-tools'
      })
    );

    await expect(execute).rejects.toThrowError('Tool is not available from MCP servers: filesystem');
    expect(calls).toHaveLength(0);

    await provider.close();
  });

  it('removes wrapper after refresh fails on a previously reachable server', async () => {
    const calls: FakeCall[] = [];
    let shouldFail = false;
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', {
      listTools: async () => {
        if (shouldFail) {
          throw new Error('list failed');
        }

        return [
          {
            name: 'fs.read',
            description: 'Read file',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string'
                }
              },
              required: ['path']
            }
          }
        ];
      },
      callTool: async (name: string, args: Record<string, unknown>) => {
        calls.push({ name, args });
        return {
          ok: true,
          name,
          args
        };
      },
      close: async () => {}
    });

    const initial = await provider.listDefinitions();
    expect(initial.map((tool) => tool.name)).toEqual(['filesystem']);

    shouldFail = true;
    (provider as any).lastRefreshAt = 0;

    const afterFailure = await provider.listDefinitions();
    expect(afterFailure).toHaveLength(0);

    const execute = provider.execute(
      'filesystem',
      buildExecutionInput({
        command: '--list-tools'
      })
    );

    await expect(execute).rejects.toThrowError('Tool is not available from MCP servers: filesystem');
    expect(calls).toHaveLength(0);

    await provider.close();
  });

  it('includes first-call --help guidance in filesystem wrapper description', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', createPrefixedFakeClient(calls));
    const tools = await provider.listDefinitions();
    const filesystemTool = tools.find((tool) => tool.name === 'filesystem');

    expect(filesystemTool).toBeDefined();
    expect(filesystemTool?.description).toContain('First call in a session: run "filesystem --help"');
    expect(
      (
        filesystemTool?.inputSchema as {
          properties?: {
            command?: {
              description?: string;
            };
          };
        }
      )?.properties?.command?.description
    ).toContain('On first call in a session, use "--help"');

    await provider.close();
  });

  it('routes positional CLI arguments to required schema fields', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    await provider.listDefinitions();

    await provider.execute(
      'fs',
      buildExecutionInput({
        command: 'read /tmp/demo.txt'
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/demo.txt'
        }
      }
    ]);

    await provider.close();
  });

  it('accepts unique suffix alias for dotted MCP subcommand names', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', createPrefixedFakeClient(calls));
    await provider.listDefinitions();

    await provider.execute(
      'filesystem',
      buildExecutionInput({
        command: 'glob --pattern *'
      })
    );

    expect(calls).toEqual([
      {
        name: 'fs.glob',
        args: {
          pattern: '*'
        }
      }
    ]);

    await provider.close();
  });

  it('returns subcommand help when key:value style arguments fail to parse', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', createPrefixedFakeClient(calls));
    await provider.listDefinitions();

    const result = await provider.execute(
      'filesystem',
      buildExecutionInput({
        command: 'glob pattern: **/*'
      })
    );

    expect(calls).toHaveLength(0);
    expect(result).toMatchObject({
      server: 'filesystem',
      subcommand: 'fs.glob',
      mode: 'subcommand-help'
    });
    expect((result as Record<string, unknown>).reason).toContain('Unsupported key format');

    await provider.close();
  });

  it('returns subcommand help when required arguments are missing', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', createPrefixedFakeClient(calls));
    await provider.listDefinitions();

    const result = await provider.execute(
      'filesystem',
      buildExecutionInput({
        command: 'glob'
      })
    );

    expect(calls).toHaveLength(0);
    expect(result).toMatchObject({
      server: 'filesystem',
      subcommand: 'fs.glob',
      mode: 'subcommand-help'
    });
    expect((result as Record<string, unknown>).reason).toContain('Missing required arguments: pattern');

    await provider.close();
  });

  it('includes subcommand usage in errors when remote execution fails', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'server-cli'
    });

    (provider as any).clients.set('filesystem', createFailingPrefixedFakeClient(calls));
    await provider.listDefinitions();

    const execute = provider.execute(
      'filesystem',
      buildExecutionInput({
        command: 'read --path /tmp'
      })
    );

    await expect(execute).rejects.toThrowError(
      'MCP request failed (filesystem): -32603 EISDIR: illegal operation on a directory, read'
    );
    await expect(execute).rejects.toThrowError('Usage:');
    await expect(execute).rejects.toThrowError('- filesystem fs.read --help');
    await expect(execute).rejects.toThrowError('- filesystem fs.read --path <string>');
    await expect(execute).rejects.toThrowError('Required args: path');

    expect(calls).toEqual([
      {
        name: 'fs.read',
        args: {
          path: '/tmp'
        }
      }
    ]);

    await provider.close();
  });

  it('exposes code_tool meta tools in code-tool mode', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'code-tool'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    const tools = await provider.listDefinitions();

    expect(tools.map((tool) => tool.name).sort((a, b) => a.localeCompare(b))).toEqual([
      'code_tool.call',
      'code_tool.runCode',
      'code_tool.search'
    ]);

    await provider.close();
  });

  it('searches internal MCP tools via code_tool.search', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'code-tool'
    });

    (provider as any).clients.set('filesystem', createPrefixedFakeClient(calls));
    await provider.listDefinitions();

    const result = await provider.execute(
      'code_tool.search',
      buildExecutionInput({
        query: 'glob files',
        topK: 3
      })
    );

    expect((result as Record<string, unknown>).selectedCount).toBe(1);
    const selectedTools = Array.isArray((result as Record<string, unknown>).selectedTools)
      ? ((result as Record<string, unknown>).selectedTools as Array<Record<string, unknown>>)
      : [];
    expect(selectedTools[0]?.name).toBe('mcp.filesystem.fs.glob');

    await provider.close();
  });

  it('invokes MCP tools by mcp.<server>.<tool> via code_tool.call', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'code-tool'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    await provider.listDefinitions();

    await provider.execute(
      'code_tool.call',
      buildExecutionInput({
        tool: 'mcp.fs.read',
        arguments: {
          path: '/tmp/demo.txt'
        }
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/demo.txt'
        }
      }
    ]);

    await provider.close();
  });

  it('supports alias calls via code_tool.call', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'code-tool'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    await provider.listDefinitions();

    await provider.execute(
      'code_tool.call',
      buildExecutionInput({
        tool: 'mcp_fs_read',
        arguments: {
          path: '/tmp/alias.txt'
        }
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/alias.txt'
        }
      }
    ]);

    await provider.close();
  });

  it('runs orchestration code via code_tool.runCode', async () => {
    if (!isDenoAvailable()) {
      return;
    }

    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'code-tool'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));
    await provider.listDefinitions();

    const result = await provider.execute(
      'code_tool.runCode',
      buildExecutionInput({
        code: 'await mcp.fs.read({ path: "/tmp/run-code.txt" });'
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/run-code.txt'
        }
      }
    ]);
    expect(result).toMatchObject({
      ok: true,
      name: 'read'
    });

    await provider.close();
  });

  it('keeps canonical tool exposure for canonical mode', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'canonical'
    });

    (provider as any).clients.set('fs', createFakeClient(calls));

    const tools = await provider.listDefinitions();
    expect(tools.map((tool) => tool.name)).toEqual(['fs.read', 'fs.write']);

    await provider.execute(
      'fs.read',
      buildExecutionInput({
        path: '/tmp/demo.txt'
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/demo.txt'
        }
      }
    ]);

    await provider.close();
  });

  it('keeps remote tool names unchanged in passthrough mode', async () => {
    const calls: FakeCall[] = [];
    const provider = createMcpAgentToolProvider({
      servers: [],
      exposureMode: 'passthrough'
    });

    (provider as any).clients.set('remote-toolhub', createFakeClient(calls));

    const tools = await provider.listDefinitions();
    expect(tools.map((tool) => tool.name)).toEqual(['read', 'write']);

    await provider.execute(
      'read',
      buildExecutionInput({
        path: '/tmp/passthrough.txt'
      })
    );

    expect(calls).toEqual([
      {
        name: 'read',
        args: {
          path: '/tmp/passthrough.txt'
        }
      }
    ]);

    await provider.close();
  });
});

function createFakeClient(calls: FakeCall[]) {
  return {
    listTools: async () => [
      {
        name: 'read',
        description: 'Read file',
        inputSchema: {
          type: 'object',
          properties: {
            path: {
              type: 'string'
            }
          },
          required: ['path']
        }
      },
      {
        name: 'write',
        description: 'Write file',
        inputSchema: {
          type: 'object',
          properties: {
            path: {
              type: 'string'
            },
            content: {
              type: 'string'
            }
          },
          required: ['path', 'content']
        }
      }
    ],
    callTool: async (name: string, args: Record<string, unknown>) => {
      calls.push({
        name,
        args
      });
      return {
        ok: true,
        name,
        args
      };
    },
    close: async () => {}
  };
}

function createPrefixedFakeClient(calls: FakeCall[]) {
  return {
    listTools: async () => [
      {
        name: 'fs.glob',
        description: 'Find files',
        inputSchema: {
          type: 'object',
          properties: {
            pattern: {
              type: 'string'
            }
          },
          required: ['pattern']
        }
      }
    ],
    callTool: async (name: string, args: Record<string, unknown>) => {
      calls.push({
        name,
        args
      });
      return {
        ok: true,
        name,
        args
      };
    },
    close: async () => {}
  };
}

function createFailingPrefixedFakeClient(calls: FakeCall[]) {
  return {
    listTools: async () => [
      {
        name: 'fs.read',
        description: 'Read file',
        inputSchema: {
          type: 'object',
          properties: {
            path: {
              type: 'string'
            },
            offset: {
              type: 'number'
            },
            limit: {
              type: 'number'
            }
          },
          required: ['path']
        }
      }
    ],
    callTool: async (name: string, args: Record<string, unknown>) => {
      calls.push({
        name,
        args
      });
      throw new Error('MCP request failed (filesystem): -32603 EISDIR: illegal operation on a directory, read');
    },
    close: async () => {}
  };
}

function buildExecutionInput(args: Record<string, unknown>): AgentToolExecutionInput {
  const now = new Date().toISOString();
  return {
    args,
    session: {
      sessionId: 'session-1',
      agentId: 'agent-1',
      systemPrompt: 'test',
      allowedTools: ['fs'],
      memoryRefs: [],
      messages: [],
      pendingToolCalls: {},
      taskState: createInitialTaskState('session-1'),
      transcriptWindow: createTranscriptWindow(),
      guards: createInitialGuards(),
      lastEventOffset: 0,
      updatedAt: now
    },
    event: {
      id: 'event-1',
      type: 'TOOL_CALL_REQUESTED',
      sessionId: 'session-1',
      timestamp: now,
      correlationId: 'corr-1',
      payload: {
        toolCallId: 'call-1',
        toolName: 'fs',
        arguments: args
      }
    }
  };
}
