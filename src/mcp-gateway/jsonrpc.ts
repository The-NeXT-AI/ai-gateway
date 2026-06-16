import { isObject } from '../utils';
import { McpGatewayError, type McpGatewayPrincipalContext, type McpGatewayRuntime } from './runtime';

interface JsonRpcRequestPayload {
  jsonrpc?: unknown;
  id?: unknown;
  method?: unknown;
  params?: unknown;
}

export interface ParsedJsonRpcRequest {
  id: string | number | null;
  method: string;
  params: unknown;
}

export function parseJsonRpcRequest(
  body: unknown
): { ok: true; request: ParsedJsonRpcRequest } | { ok: false; error: string } {
  if (!isObject(body) || Array.isArray(body)) {
    return {
      ok: false,
      error: 'Request body must be a JSON object.'
    };
  }

  const payload = body as JsonRpcRequestPayload;
  if (payload.jsonrpc !== '2.0') {
    return {
      ok: false,
      error: 'jsonrpc must be "2.0".'
    };
  }

  if (typeof payload.method !== 'string' || !payload.method.trim()) {
    return {
      ok: false,
      error: 'method must be a non-empty string.'
    };
  }

  if (!isValidJsonRpcId(payload.id)) {
    return {
      ok: false,
      error: 'id must be string, number, or null.'
    };
  }

  return {
    ok: true,
    request: {
      id: payload.id ?? null,
      method: payload.method.trim(),
      params: payload.params
    }
  };
}

export async function handleMcpJsonRpcMethod(
  runtime: McpGatewayRuntime,
  context: McpGatewayPrincipalContext,
  method: string,
  params: unknown
): Promise<unknown> {
  if (method === 'initialize') {
    return runtime.buildInitializeResponse();
  }

  if (method === 'ping') {
    return {};
  }

  if (method === 'tools/list') {
    const tools = await runtime.listTools(context);
    return {
      tools: tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema || {
          type: 'object',
          additionalProperties: true
        }
      }))
    };
  }

  if (method === 'tools/call') {
    if (!isObject(params)) {
      throw new McpGatewayError(-32602, 'tools/call params must be an object.', 400);
    }

    const name = typeof params.name === 'string' ? params.name.trim() : '';
    if (!name) {
      throw new McpGatewayError(-32602, 'tools/call params.name must be a non-empty string.', 400);
    }

    const argsValue = params.arguments;
    if (argsValue !== undefined && !isObject(argsValue)) {
      throw new McpGatewayError(-32602, 'tools/call params.arguments must be an object.', 400);
    }

    const meta = isObject(params._meta) ? params._meta : undefined;
    const result = await runtime.callTool(context, name, isObject(argsValue) ? argsValue : {}, meta);
    if (isObject(result)) {
      return result;
    }

    return {
      content: [
        {
          type: 'text',
          text: safeJsonStringify(result)
        }
      ]
    };
  }

  throw new McpGatewayError(-32601, `Method not found: ${method}`, 404);
}

export function buildJsonRpcErrorResponse(
  id: string | number | null,
  code: number,
  message: string,
  data?: unknown
): Record<string, unknown> {
  return {
    jsonrpc: '2.0',
    id,
    error: {
      code,
      message,
      data
    }
  };
}

function isValidJsonRpcId(value: unknown): value is string | number | null | undefined {
  return value === undefined || value === null || typeof value === 'string' || typeof value === 'number';
}

function safeJsonStringify(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    const serialized = JSON.stringify(value);
    return serialized === undefined ? 'null' : serialized;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}
