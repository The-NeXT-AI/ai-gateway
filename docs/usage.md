# Next AI Gateway 使用文档

> 本文档由原 README 迁移而来，保留完整架构、配置、路由和调用示例。项目根目录 README 仅保留快速入口。

基于 TypeScript + Fastify 的 AI 协议网关，支持以下 API 格式：

- OpenAI `POST /v1/chat/completions`
- OpenAI `POST /v1/responses`
- OpenAI `POST /v1/embeddings`
- OpenAI `POST /v1/moderations`
- OpenAI `POST /v1/images/generations`
- Anthropic Claude `POST /v1/messages`
- Google Gemini `POST /v1beta/models/{model}:generateContent`（同时支持 `/v1/models/*`）

## 架构与标准模型

- 项目已按模块拆分为：`gateway`（路由与主流程）、`adapters`（可注册协议适配器）、`upstream`（上游调用）、`config/types/utils`（配置与通用能力）。
- 网关内部标准模型统一使用 **OpenAI Responses 协议**：
  - 入站：各协议请求先转换为 OpenAI Responses Request 结构。
  - 出站：上游响应先转换为 OpenAI Responses Response 结构。
  - 最终：再按调用入口协议格式转换后返回。

## 请求体大小

- 默认请求体上限为 `52428800` 字节（50 MiB），用于支持包含 base64 图片/文件的多模态请求。
- 可通过环境变量 `GATEWAY_BODY_LIMIT_BYTES` 覆盖，也可在 `gateway.config.json` 中设置 `bodyLimitBytes`。

## Adapter / Provider 插件机制

- Source Adapter 不再通过 `sourceFormat` 分支写死，而是通过 `SourceAdapterRegistry` 按 `adapterKey` 查找。
- Target Adapter 通过 `TargetAdapterRegistry` 按 provider 查找。
- Provider Plugin 通过 `ProviderPluginRegistry` 按 provider / providerName 匹配，可定制供应商鉴权、请求参数转换、响应参数转换。
- 内置适配器在 `createGatewayRuntime()` 中默认注册，你可以按需新增或覆盖。
- 内置 adapter 已按协议拆分为独立文件（例如 `chat/completions`、`responses`、`messages`、`generateContent`）。

协议文件示例：

- Source:
  - `src/adapters/builtins/source/openai-chat-completions.ts`
  - `src/adapters/builtins/source/openai-responses.ts`
  - `src/adapters/builtins/source/anthropic-messages.ts`
  - `src/adapters/builtins/source/gemini-generate-content.ts`
  - `src/adapters/builtins/source/gemini-stream-generate-content.ts`
- Target:
  - `src/adapters/builtins/target/openai-responses.ts`
  - `src/adapters/builtins/target/anthropic-messages.ts`
  - `src/adapters/builtins/target/gemini-generate-content.ts`

示例（注册自定义 Source Adapter）：

```ts
import Fastify from 'fastify';
import { config } from './config';
import { registerGatewayRoutes } from './gateway/routes';
import { handleGatewayRequest } from './gateway/handler';
import { createGatewayRuntime } from './gateway/runtime';
import type { SourceAdapter } from './types';

const runtime = createGatewayRuntime(config);

const customSourceAdapter: SourceAdapter = {
  key: 'my_source',
  provider: 'openai',
  toStandardRequest({ body }) { /* ... */ throw new Error('impl'); },
  fromStandardResponse({ response }) { return response; },
  isStreamingRequest() { return false; },
  buildPassthroughRequest() { /* ... */ throw new Error('impl'); }
};

runtime.sourceAdapters.register(customSourceAdapter);

const app = Fastify();
registerGatewayRoutes(app, config, runtime);

app.post('/my/source/endpoint', async (request, reply) => {
  return handleGatewayRequest(
    request,
    reply,
    { adapterKey: 'my_source' },
    config,
    runtime
  );
});
```

示例（注册供应商插件，按 providerName 定制鉴权/请求/响应转换）：

```ts
import { config } from './config';
import { createGatewayRuntime } from './gateway/runtime';
import type { ProviderPlugin } from './types';

const runtime = createGatewayRuntime(config);

const openAIMainPlugin: ProviderPlugin = {
  key: 'openai-main-plugin',
  providerName: 'openai-main',
  authenticate({ upstreamRequest }) {
    return {
      ok: true,
      value: {
        ...upstreamRequest,
        headers: {
          ...upstreamRequest.headers,
          'x-provider-signature': 'signed-value'
        }
      }
    };
  },
  transformRequest({ upstreamRequest }) {
    return {
      ok: true,
      value: {
        ...upstreamRequest,
        body: {
          ...(upstreamRequest.body as Record<string, unknown>),
          customParam: 'from-plugin'
        }
      }
    };
  },
  transformResponse({ upstreamPayload }) {
    const payload = upstreamPayload as Record<string, unknown>;
    return {
      ok: true,
      value: {
        ...payload,
        pluginHandled: true
      }
    };
  }
};

runtime.providerPlugins.register(openAIMainPlugin);
```

执行顺序（同一次 provider 尝试内）：

- `authenticate` -> `transformRequest` -> 上游调用 -> `transformResponse`
- 多个插件按注册顺序依次执行
- `providerName` 优先精确匹配；仅配置 `provider` 时按供应商类型匹配
- 对纯流式透传（event-stream）响应不会强制做 payload 级改写
- 也支持在 `gateway.config.json` 中通过 `providerPlugins` 声明式加载（进程启动自动注册，manager/webhook 热更新后自动重载）
- `providerPlugins[].codexOauth` 可启用 Codex OAuth refresh token 流程（`client_id + grant_type=refresh_token + refresh_token`），并自动写入上游鉴权头

## 能力说明

- 同协议透传：默认将请求透传到对应上游。
- 跨协议转换：通过 `x-target-provider` 指定目标厂商，实现请求/响应格式转换。
- 多供应商管理：支持 `x-target-providers`（逗号分隔）定义目标供应商优先级链路。
- 支持按供应商名称路由：`x-target-provider` / `x-target-providers` 可使用 provider `type` 或 `name`（如 `openai-main`）。
- 支持模型内联路由：`x-target-model` 或请求体 `model` 可写成 `providerName/modelName`（如 `openai-main/GPT-5.3`）。
- 支持 OpenAI 兼容 JSON 端点透传：`/v1/embeddings`、`/v1/moderations`、`/v1/images/generations` 复用命名 provider、fallback、插件、precheck、policy、health 与用量计费。
- 请求失败自动 fallback：按供应商顺序逐个尝试，直到成功或全部失败。
- 主动健康检查：manager API 可触发 provider 探测，也可通过 `providerHealthCheck.enabled=true` 启用定时探测并更新运行态 health，用于 health-aware routing。
- 运行指标导出：可通过 `metrics.enabled=true` 开启 `GET /metrics`，输出 Prometheus 文本格式的请求计数、耗时与 provider health 指标。
- 幂等重试保护：可通过 `idempotency.enabled=true` 启用 `Idempotency-Key` 缓存，避免 `/v1*` JSON POST 的非流式成功响应在客户端重试时重复调用上游。
- 上游并发隔离：可通过 `upstreamConcurrency.enabled=true` 限制每个 provider 的并发上游调用数，队列等待超时后返回 429，避免单个慢上游拖垮网关。
- 上游熔断保护：可通过 `upstreamCircuitBreaker.enabled=true` 在 provider 连续连接失败或返回指定状态码后进入冷却期，冷却期内直接 fallback 或返回 503。
- 上游重试策略：`upstreamRetry` 可配置连接错误重试次数、退避参数和可选 HTTP 状态码重试；默认保留现有连接错误 2 次尝试行为，HTTP 状态码重试默认关闭。
- 普通请求透明工具执行：`transparentToolExecution.enabled=true` 时，非流式标准化请求会自动执行已声明且可由 MCP tool provider 唯一解析的工具调用，并把 tool result 追加回模型继续生成；未知工具默认回传客户端。
- 供应商插件扩展：支持按 `provider` / `providerName` 注入鉴权逻辑、上游请求改写、上游响应改写。
- DeepSeek 思考模式插件：仅在 `providerPlugins[].deepseekThinking.enabled=true` 时启用；启用后会把 Anthropic `output_config.effort` 或 OpenAI `reasoning.effort` / `reasoning_effort` 转成 DeepSeek OpenAI chat 的 `reasoning_effort`，并支持 `thinking.type` 开关。
- 配置管理：支持 JSON 配置文件（`gateway.config.json`）、环境变量、HTTP/WebSocket/gRPC/stdio 外部配置源与 manager/webhook 热更新。
- 客户 Auth 接入：支持 `trusted_header`（身份头透传）与 `http_introspection`（远程校验）两种模式。
- 网关策略治理：支持按全局、用户、租户、组织、subject、plan、API key 维度 allow/deny provider、providerName、model、provider/model。
- 用量计费：按上游响应中的 `usage` 字段计算费用，并通过响应头返回。
- 计费事件投递：支持通过 HTTP、WebSocket、gRPC 或 stdio transport 将计费明细上报到外部服务，外部服务可再写库、入队或转发到消息总线。
- 内置事件驱动 Agent：维护会话提示词/上下文/工具，支持通过事件接入用户输入与工具结果。
- Agent 事件上报：支持通过 HTTP、WebSocket、gRPC 或 stdio transport 将 Agent 事件投递到外部审计、观测或编排服务。
- 内置 MCP Gateway：固定 endpoint 聚合 MCP tools，支持按 key/team/org 做工具可见性控制与调用拦截。
- MCP WebSocket RPC：支持通过 `ws://.../mcp/ws` 远端 JSON-RPC 调用 MCP tools，复用同一套鉴权与策略。

> 边界说明：gateway 进程不直接连接数据库、Redis、BullMQ 或对象存储 SDK。需要持久化、队列、审计、动态配置时，应通过 HTTP/WebSocket/gRPC/stdio 等协议接入外部服务，由外部服务负责存储或分发。
> 外部服务实现可参考 [docs/external-protocols.md](docs/external-protocols.md)。

> 说明：网关支持跨厂商流式转换；同厂商流式请求仍优先透传上游流。

## 运行

```bash
npm install
npm run dev
# 或
npm run build && npm start
```

### Docker Compose（含 ToolHub Deno 沙箱）

`code_tool.runCode` 现由 `toolhub` 实现，`gateway` 不再本地执行 Deno 沙箱代码。运行编排代码时需要 `toolhub` 容器内可执行 `deno`。

```bash
docker compose up --build
```

默认行为：
- 服务名：`gateway`
- 端口映射：`3000:3000`
- 使用 `gateway.config.compose.json` 挂载到容器内 `/app/gateway.config.json`
- `node_modules` 使用独立 volume：`gateway_node_modules`

### 远端工具热插拔（方案4：ToolHub 稳定入口）

为避免在 `gateway` 内频繁增删 MCP 进程，建议采用：
- `gateway`（Edge）只连接一个远端 MCP WS：`remote-toolhub`
- `toolhub` 统一承载真实 MCP tools（stdio / websocket）

本仓库已提供 compose 配置：
- `gateway/gateway.config.compose.json`：Edge 配置，只保留 `remote-toolhub`
- `gateway/toolhub.config.compose.json`：ToolHub 配置（`toolExposureMode: "code-tool"`），暴露 `WS /mcp/ws`，默认包含 `local-http-mcp -> http://127.0.0.1:3789/mcp`
- 父级 `../docker-compose.yml`：包含 `gateway + toolhub` 双服务编排

热插拔方式：
1. 修改 `toolhub.config.compose.json` 里的 `mcpServers`（新增/删除/替换工具）
2. 重启 `toolhub` 服务（无需重启 `gateway`）
3. `gateway` 在下次工具发现时会刷新可用工具列表（默认短周期刷新）

鉴权注意：
- `gateway` 侧使用环境变量 `MCP_REMOTE_KEY`
- `toolhub.config.compose.json` 里 `mcpGateway.principals[].key` 需要与其保持一致

网络注意：
- 在 Docker 内运行时，`127.0.0.1` 指向容器自身；如果 MCP 服务跑在宿主机，需改为 `http://host.docker.internal:3789/mcp`。

## 环境变量

### JSON 配置文件

- 默认会读取项目根目录 `gateway.config.json`（文件存在时生效）
- 可通过 `GATEWAY_CONFIG_PATH` 指定 JSON 配置文件路径
- 示例文件见 `gateway.config.example.json`
- 推荐使用 `Providers` 数组配置供应商（数组顺序即默认 fallback 顺序）
- `Providers` 单项字段：`name`、`type`、`apikey|apiKeyEnv`、`baseurl`、`models`、`extraHeaders`、`extraBody`、`billing`
- `providerPlugins` 用于声明供应商插件（`key`、`provider|providerName`、`codexOauth`、`deepseekThinking`、`auth`、`request`、`response`）
- `type` 同时用于声明 provider 类别和上游协议，支持：
  - OpenAI：`openai`（等价 `openai_responses`）、`openai_responses`、`openai_chat_completions`
  - Anthropic：`anthropic`（等价 `anthropic_messages`）、`anthropic_messages`
  - Gemini：`gemini`（等价 `gemini_generate_content`）、`gemini_generate_content`
- `extraHeaders` / `extraBody` / `billing` 支持按模型配置（`default` + 指定模型名）
- `providerPlugins` 的 `auth/request/response` 支持声明式规则：`headers`、`query`、`bodySet`、`bodyMerge`、`bodyRemove`
- `providerPlugins` 支持值引用：`{"from":"env.XXX"}`、`{"from":"request.headers.x-foo"}`、`{"from":"request.body.user.id"}`、`{"from":"upstreamPayload.data.id"}`、`{"from":"target.providerName"}`
- `providerPlugins.codexOauth` 字段：`accessToken`、`refreshToken`、`tokenEndpoint`、`clientId`、`scope`、`refreshIfMissingAccessToken`、`forceRefresh`、`required`、`timeoutMs`、`authHeader`、`authScheme`
- `providerExternal` 用于通过 HTTP/WebSocket/gRPC/stdio 从外部服务动态加载 provider、providerPlugins 与 virtualModelProfiles。
- `billing` 支持 `cacheReadPerMillionUsd` / `cacheWritePerMillionUsd` 与 `tiers`（阶梯计费）
- `configExternal` 用于从外部服务动态获取完整 gateway 配置（`enabled`、`transport=http|websocket|grpc|stdio`、`endpoint`、`command`、`args`、`cwd`、`env`、`method`、`timeoutMs`、`intervalMs|intervalSeconds`、`apiKeyHeader`、`apiKey|apiKeyEnv`、`headers`）；外部返回体可为完整配置对象、`{"config": {...}}` 或 `{"gatewayConfig": {...}}`。gRPC 使用 JSON unary，默认 path 为 `/gateway.config.v1.ConfigService/GetConfig`。
- `billingWebhook` 用于配置事件上报（`enabled`、`transport=http|websocket|grpc|stdio`、`endpoint`、`command`、`args`、`cwd`、`env`、`timeoutMs`、`maxAttempts`、`baseDelayMs`、`maxDelayMs`、`requireAck`、`headers`）；gRPC 使用 JSON unary，默认 path 为 `/gateway.events.v1.EventSink/Publish`。
- `billingQueue` 为历史兼容字段；gateway 不会创建队列连接，开启后也只记录禁用日志。需要队列时请通过 `billingWebhook` 或外部协议适配服务实现。
- `rawTrace` 用于捕获原始请求/上游链路包；gateway 只写本地 spool bundle，`rawTrace.sync` 通过 HTTP/WebSocket/gRPC/stdio 上报 manifest，支持失败重试，由外部服务负责持久化、索引和归档。
- `auth` 用于配置客户侧鉴权（`enabled`、`mode`、`required`、`trustedCidrs`、`identityHeaders`、`signature`、`introspection`）
- `precheck` 用于请求前治理（`rateLimit`、`quota`、`budget`、`estimation`），`precheck.storage.type` 仅使用进程内 `memory`。多实例全局限流/预算应由外部治理服务提供。
- `providerHealthCheck` 用于配置定时 provider 探测（默认关闭）：`enabled`、`intervalMs|intervalSeconds`、`timeoutMs|timeoutSeconds`、`initialDelayMs|initialDelaySeconds`。
- `metrics` 用于配置 Prometheus 指标导出（默认关闭）：`enabled`、`includeProviderHealth`；开启后可访问 `GET /metrics`。
- `cors` 用于配置跨域响应头（默认开启）：`enabled`、`origins|origin`、`allowedHeaders`、`allowedMethods`、`allowCredentials`、`maxAgeSeconds|maxAge`。
- `idempotency` 用于配置幂等重试缓存（默认关闭）：`enabled`、`headerName`、`ttlMs|ttlSeconds`、`maxEntries`、`cacheErrorResponses`；默认只缓存 `/v1*` JSON POST 的非流式成功响应。
- `upstreamConcurrency` 用于配置 provider 维度上游并发隔离（默认关闭）：`enabled`、`maxInFlightPerProvider`、`queueTimeoutMs|queueTimeoutSeconds`。
- `upstreamCircuitBreaker` 用于配置 provider 维度上游熔断（默认关闭）：`enabled`、`failureThreshold`、`cooldownMs|cooldownSeconds`、`failureStatusCodes`。
- `upstreamRetry` 用于配置上游传输重试：`enabled`、`maxAttempts`、`baseDelayMs|baseDelaySeconds`、`maxDelayMs|maxDelaySeconds`、`backoffMultiplier`、`jitterMs|jitterSeconds`、`retryStatusCodes`。
- `transparentToolExecution` 用于配置普通请求透明工具执行（默认关闭）：`enabled`、`maxTurns`、`maxToolCalls`、`requireClientDeclaration`、`unknownToolPolicy=return_to_client|fail`、`allowTools`、`denyTools`。启用后会绕过同协议非流式 passthrough，以便解析 tool calls；仅执行请求中声明且 MCP 工具源可唯一解析的工具。
- `agent.mcpServers` 用于配置 MCP 工具源：
  - `transport=stdio`：`name`、`command`、`args`、`env`、`cwd`、`protocolVersion`、`startupTimeoutMs`、`requestTimeoutMs`
  - `transport=websocket`：`name`、`url`、`headers`、`apiKey|apiKeyEnv`、`protocolVersion`、`startupTimeoutMs`、`requestTimeoutMs`
- `agent.storage` 默认使用进程内 `memory`；仅保留本地开发用 `filesystem`。通用网关部署中如需外部托管 agent/session，请使用 `agent.external` 让外部服务负责状态。
- `agent.runtime` 用于配置运行时健壮性（`sessionLockTimeoutMs`、`eventWorkerConcurrency`、`llmRetry`、`toolRetry`）
- `agent.external` 可配置通过 HTTP/WebSocket/gRPC/stdio 从外部服务加载 agent/session，并将 session 变更同步回外部服务。
- `agent.eventWebhook` 用于通过 HTTP、WebSocket、gRPC 或 stdio transport 上报 Agent 事件（`enabled`、`transport=http|websocket|grpc|stdio`、`endpoint`、`command`、`args`、`cwd`、`env`、`timeoutMs`、`maxAttempts`、`baseDelayMs`、`maxDelayMs`、`requireAck`、`headers`）
- `agent.eventQueue` 为历史兼容字段；gateway 不会创建队列连接。Agent 事件需要上报时优先使用 `agent.eventWebhook`。
- `mcpGateway` 用于配置 MCP 网关访问控制（`enabled`、`endpoint`、`websocket`、`principals`、`serverExposure`、`guardrails`、`oauth`）
- 配置优先级：请求头 > 环境变量 > JSON 配置文件 > 内置默认值

`billing.tiers` 结构示例：

```json
{
  "billing": {
    "default": {
      "inputPerMillionUsd": 0.15,
      "outputPerMillionUsd": 0.6,
      "cacheReadPerMillionUsd": 0.03,
      "tiers": {
        "input": [{ "upToTokens": 1000000, "perMillionUsd": 0.15 }, { "perMillionUsd": 0.12 }],
        "output": [{ "upToTokens": 1000000, "perMillionUsd": 0.6 }, { "perMillionUsd": 0.5 }],
        "cacheRead": [{ "upToTokens": 2000000, "perMillionUsd": 0.03 }, { "perMillionUsd": 0.02 }]
      }
    }
  }
}
```

### 基础

- `HOST`：监听地址，默认 `0.0.0.0`
- `PORT`：监听端口，默认 `3000`
- `GATEWAY_CONFIG_PATH`：JSON 配置文件路径（可选）
- `CODEX_REFRESH_TOKEN_URL_OVERRIDE`：`providerPlugins.codexOauth.tokenEndpoint` 默认值覆盖（可选；默认 `https://auth.openai.com/oauth/token`）
- `MANAGER_API_KEY`：管理接口密钥（可选；设置后需通过 `x-manager-key` 或 `Authorization: Bearer` 访问管理接口）
- `AGENT_STORAGE_TYPE`：Agent 存储类型，支持 `memory` / `filesystem`，默认 `memory`；`filesystem` 适合本地开发，通用网关部署建议使用 `agent.external` 托管状态
- `AGENT_STORAGE_DIR`：仅 `AGENT_STORAGE_TYPE=filesystem` 时生效的本地存储目录（默认 `<project>/.agent-data`）
- `AGENT_EXTERNAL_ENABLED`：是否启用 agent/session 外部数据源（默认 `false`）
- `AGENT_EXTERNAL_TRANSPORT`：外部数据源协议，支持 `http` / `websocket` / `grpc` / `stdio`；未配置时会根据 endpoint 的 `ws://` / `wss://` / `grpc://` / `grpcs://` 或 command 自动推断
- `AGENT_EXTERNAL_ENDPOINT` / `AGENT_EXTERNAL_URL`：外部接口地址（返回 JSON，包含 `agents` 与/或 `sessions` 数组）
- `AGENT_EXTERNAL_STDIO_COMMAND` / `AGENT_EXTERNAL_STDIO_ARGS` / `AGENT_EXTERNAL_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录
- `AGENT_EXTERNAL_TIMEOUT_MS`：外部接口请求超时毫秒（默认 `5000`）
- `AGENT_EXTERNAL_API_KEY_HEADER`：外部接口 API Key 头名（默认 `x-agent-external-key`）
- `AGENT_EXTERNAL_API_KEY`：外部接口 API Key 值（可选）
- `AGENT_EXTERNAL_API_KEY_ENV`：外部接口 API Key 所在环境变量名（可选）
- `PROVIDER_EXTERNAL_ENABLED`：是否启用外部 provider 配置源（默认 `false`）
- `PROVIDER_EXTERNAL_TRANSPORT`：外部 provider 源协议，支持 `http` / `websocket` / `grpc` / `stdio`
- `PROVIDER_EXTERNAL_ENDPOINT` / `PROVIDER_EXTERNAL_URL`：外部 provider 源地址
- `PROVIDER_EXTERNAL_STDIO_COMMAND` / `PROVIDER_EXTERNAL_STDIO_ARGS` / `PROVIDER_EXTERNAL_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录
- `PROVIDER_EXTERNAL_TIMEOUT_MS` / `PROVIDER_EXTERNAL_API_KEY_HEADER` / `PROVIDER_EXTERNAL_API_KEY` / `PROVIDER_EXTERNAL_API_KEY_ENV`：外部 provider 源超时和鉴权配置
- `MCP_GATEWAY_ENABLED`：是否启用 MCP Gateway（默认：当 `mcpGateway.principals` 非空时自动启用）
- `MCP_GATEWAY_ENDPOINT`：MCP Gateway 固定入口（默认 `/mcp`）
- `MCP_GATEWAY_WS_ENABLED`：是否启用 MCP WebSocket RPC（默认 `false`）
- `MCP_GATEWAY_WS_ENDPOINT`：MCP WebSocket RPC 入口（默认 `/mcp/ws`）
- `MCP_GATEWAY_WS_ALLOW_QUERY_TOKEN`：是否允许 URL query token 鉴权（默认 `true`）
- `MCP_GATEWAY_WS_QUERY_TOKEN_PARAM`：URL query token 参数名（默认 `token`）
- `MCP_GATEWAY_GUARDRAILS_ENABLED`：是否启用 tool call guardrails（默认 `true`）
- `MCP_GATEWAY_MAX_ARGUMENT_BYTES`：tool arguments 最大字节数（默认 `65536`）
- `MCP_GATEWAY_OAUTH_ENABLED`：是否启用 `.well-known` OAuth discovery（默认 `false`）
- `MCP_GATEWAY_OAUTH_RESOURCE`：OAuth protected resource 值（可选）
- `MCP_GATEWAY_OAUTH_ISSUER`：OAuth issuer（可选）
- `MCP_GATEWAY_OAUTH_AUTHORIZATION_ENDPOINT`：OAuth authorization endpoint（可选）
- `MCP_GATEWAY_OAUTH_TOKEN_ENDPOINT`：OAuth token endpoint（可选）
- `MCP_GATEWAY_OAUTH_DEFAULT_PRINCIPAL_KEY`：OAuth 换取 token 后映射的默认 principal key（可选；未配置时取 `mcpGateway.principals[0]`）
- `MCP_GATEWAY_OAUTH_AUTH_CODE_TTL_SEC`：授权码有效期秒数（默认 `180`）
- `MCP_GATEWAY_OAUTH_ACCESS_TOKEN_TTL_SEC`：access token 有效期秒数（默认 `3600`）
- `MCP_GATEWAY_OAUTH_REFRESH_TOKEN_TTL_SEC`：refresh token 有效期秒数（默认 `2592000`）
- `DEFAULT_TARGET_PROVIDER`：默认目标 provider（`openai|anthropic|gemini`）
- `DEFAULT_TARGET_PROVIDERS`：默认目标 provider 列表（逗号分隔，如 `openai,anthropic,gemini`）
- `UPSTREAM_TIMEOUT_MS`：兼容旧配置字段；当前不再对模型上游请求设置网关侧固定超时

### 客户 Auth（Header / Introspection）

- `AUTH_ENABLED`：是否开启客户身份鉴权，默认 `false`
- `AUTH_MODE`：鉴权模式，`trusted_header|http_introspection`（默认 `trusted_header`）
- `AUTH_REQUIRED`：是否强制要求身份头，默认 `true`
- `AUTH_TRUSTED_CIDRS`：允许注入身份头的来源网段（逗号分隔，可选）
- `AUTH_HEADER_USER_ID`：用户 ID 头名，默认 `x-auth-user-id`
- `AUTH_HEADER_TENANT_ID`：租户 ID 头名，默认 `x-auth-tenant-id`
- `AUTH_HEADER_SUBJECT`：主体 ID 头名，默认 `x-auth-sub`
- `AUTH_HEADER_ORGANIZATION_ID`：组织 ID 头名，默认 `x-auth-organization-id`
- `AUTH_HEADER_PLAN`：套餐/版本头名，默认 `x-auth-plan`
- `AUTH_HEADER_API_KEY_ID`：API Key ID 头名，默认 `x-auth-api-key-id`
- `AUTH_SIGNATURE_ENABLED`：是否启用身份头签名校验，默认 `false`
- `AUTH_SIGNATURE_HEADER`：签名头名，默认 `x-auth-signature`
- `AUTH_SIGNATURE_TIMESTAMP_HEADER`：签名时间戳头名，默认 `x-auth-ts`
- `AUTH_SIGNATURE_SECRET_ENV`：签名密钥对应的环境变量名，默认 `AUTH_HEADER_SIGNING_SECRET`
- `AUTH_SIGNATURE_MAX_SKEW_SEC`：签名允许的最大时间偏移秒数，默认 `120`
- `AUTH_HEADER_SIGNING_SECRET`：签名密钥（当 `AUTH_SIGNATURE_ENABLED=true` 时必填）

`http_introspection` 模式（网关主动调用客户 Auth 服务）：

- `AUTH_INTROSPECTION_ENDPOINT`：客户 Auth introspection URL（必填）
- `AUTH_INTROSPECTION_TIMEOUT_MS`：调用超时毫秒（默认 `3000`）
- `AUTH_INTROSPECTION_TOKEN_HEADER`：从入站请求读取 token 的 Header（默认 `authorization`）
- `AUTH_INTROSPECTION_TOKEN_BEARER_ONLY`：是否要求 `Bearer <token>` 格式（默认 `true`）
- `AUTH_INTROSPECTION_REQUEST_TOKEN_FIELD`：请求体 token 字段名（默认 `token`）
- `AUTH_INTROSPECTION_CREDENTIAL_HEADER`：网关到 Auth 服务的共享密钥 Header（默认 `x-gateway-auth`）
- `AUTH_INTROSPECTION_CREDENTIAL_ENV`：共享密钥环境变量名（默认 `AUTH_INTROSPECTION_SHARED_SECRET`）
- `AUTH_INTROSPECTION_SHARED_SECRET`：共享密钥值（可选）
- `AUTH_INTROSPECTION_RESPONSE_ACTIVE_FIELD`：响应中的 token 激活字段路径（默认 `active`）
- `AUTH_INTROSPECTION_RESPONSE_USER_ID_FIELD`：用户 ID 字段路径（默认 `userId`）
- `AUTH_INTROSPECTION_RESPONSE_TENANT_ID_FIELD`：租户 ID 字段路径（默认 `tenantId`）
- `AUTH_INTROSPECTION_RESPONSE_SUBJECT_FIELD`：主体 ID 字段路径（默认 `sub`）
- `AUTH_INTROSPECTION_RESPONSE_ORGANIZATION_ID_FIELD`：组织 ID 字段路径（默认 `organizationId`）
- `AUTH_INTROSPECTION_RESPONSE_PLAN_FIELD`：套餐字段路径（默认 `plan`）
- `AUTH_INTROSPECTION_RESPONSE_API_KEY_ID_FIELD`：API Key ID 字段路径（默认 `apiKeyId`）

签名串规则（HMAC-SHA256）：

```text
<x-auth-ts>\n<HTTP_METHOD>\n<PATH>\n<tenantId>\n<userId>\n<subject>\n<organizationId>\n<plan>
```

其中 `tenantId/userId/subject/organizationId/plan` 来自你配置的 `identityHeaders`；可空字段使用空字符串占位。

> `signature` 仅在 `AUTH_MODE=trusted_header` 时生效。

### OpenAI

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）
- `DEFAULT_OPENAI_MODEL`
- JSON（推荐）：
  `{"name":"openai-main","type":"openai","apikey":"...","baseurl":"...","models":["gpt-4.1-mini"]}`

### Anthropic

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`（默认 `https://api.anthropic.com`）
- `DEFAULT_ANTHROPIC_MODEL`
- JSON（推荐）：
  `{"name":"anthropic-main","type":"anthropic","apikey":"...","baseurl":"...","models":["claude-3-5-sonnet-latest"]}`

### Gemini

- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`（默认 `https://generativelanguage.googleapis.com`）
- `GEMINI_API_VERSION`（默认 `v1beta`）
- `DEFAULT_GEMINI_MODEL`
- JSON（推荐）：
  `{"name":"gemini-main","type":"gemini","apikey":"...","baseurl":"...","models":["gemini-2.0-flash"]}`

### 计费

- `BILLING_ENABLED`：是否开启计费，默认 `true`
- `OPENAI_INPUT_PRICE_PER_1M`：OpenAI 输入 token 单价（USD/百万 token）
- `OPENAI_OUTPUT_PRICE_PER_1M`：OpenAI 输出 token 单价（USD/百万 token）
- `OPENAI_CACHE_READ_PRICE_PER_1M`：OpenAI cache read token 单价（USD/百万 token）
- `OPENAI_CACHE_WRITE_PRICE_PER_1M`：OpenAI cache write token 单价（USD/百万 token）
- `ANTHROPIC_INPUT_PRICE_PER_1M`：Anthropic 输入 token 单价（USD/百万 token）
- `ANTHROPIC_OUTPUT_PRICE_PER_1M`：Anthropic 输出 token 单价（USD/百万 token）
- `ANTHROPIC_CACHE_READ_PRICE_PER_1M`：Anthropic cache read token 单价（USD/百万 token）
- `ANTHROPIC_CACHE_WRITE_PRICE_PER_1M`：Anthropic cache write token 单价（USD/百万 token）
- `GEMINI_INPUT_PRICE_PER_1M`：Gemini 输入 token 单价（USD/百万 token）
- `GEMINI_OUTPUT_PRICE_PER_1M`：Gemini 输出 token 单价（USD/百万 token）
- `GEMINI_CACHE_READ_PRICE_PER_1M`：Gemini cache read token 单价（USD/百万 token）
- `GEMINI_CACHE_WRITE_PRICE_PER_1M`：Gemini cache write token 单价（USD/百万 token）

> 未配置单价时默认按 `0` 计费。
> 阶梯计费通过 JSON `billing.tiers` 配置（见 `gateway.config.example.json`）。

### CORS

- `GATEWAY_CORS_ENABLED`：是否返回 CORS 响应头，默认 `true`
- `GATEWAY_CORS_ORIGINS` / `GATEWAY_CORS_ORIGIN` / `CORS_ORIGIN`：允许的 origin，逗号分隔，默认 `*`
- `GATEWAY_CORS_ALLOWED_HEADERS`：允许的请求头，逗号分隔
- `GATEWAY_CORS_ALLOWED_METHODS`：允许的方法，逗号分隔
- `GATEWAY_CORS_ALLOW_CREDENTIALS`：是否返回 `Access-Control-Allow-Credentials: true`，默认 `false`
- `GATEWAY_CORS_MAX_AGE_SECONDS`：preflight 缓存秒数，默认 `86400`

### 外部动态配置源

- `GATEWAY_CONFIG_EXTERNAL_ENABLED`：是否启用外部配置源，默认 `false`
- `GATEWAY_CONFIG_EXTERNAL_TRANSPORT`：配置源协议，支持 `http` / `websocket` / `grpc` / `stdio`
- `GATEWAY_CONFIG_EXTERNAL_ENDPOINT` / `GATEWAY_CONFIG_EXTERNAL_URL`：外部配置服务地址；未配置 transport 时会根据 `ws://` / `wss://` / `grpc://` / `grpcs://` 自动推断
- `GATEWAY_CONFIG_EXTERNAL_METHOD`：HTTP 请求方法，支持 `GET` / `POST`，默认 `GET`
- `GATEWAY_CONFIG_EXTERNAL_STDIO_COMMAND` / `GATEWAY_CONFIG_EXTERNAL_STDIO_ARGS` / `GATEWAY_CONFIG_EXTERNAL_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录；请求 JSON 写入 stdin，配置 JSON 从 stdout 读取
- `GATEWAY_CONFIG_EXTERNAL_TIMEOUT_MS`：请求超时毫秒，默认 `5000`
- `GATEWAY_CONFIG_EXTERNAL_INTERVAL_MS` / `GATEWAY_CONFIG_EXTERNAL_INTERVAL_SECONDS`：定时刷新间隔；为 `0` 时不轮询
- `GATEWAY_CONFIG_EXTERNAL_API_KEY_HEADER` / `GATEWAY_CONFIG_EXTERNAL_API_KEY` / `GATEWAY_CONFIG_EXTERNAL_API_KEY_ENV`：可选，用于给配置源请求增加共享密钥头

### 计费队列兼容项

- `BILLING_QUEUE_ENABLED` / `BILLING_QUEUE_NAME` / `BILLING_QUEUE_JOB_NAME`：历史兼容项；gateway 不会建立队列或数据库连接。
- `BILLING_QUEUE_REMOVE_ON_COMPLETE` / `BILLING_QUEUE_REMOVE_ON_FAIL`：历史兼容项，不影响外部投递。

### 计费 Webhook

- `BILLING_WEBHOOK_ENABLED`：是否开启 Webhook 投递，默认 `false`
- `BILLING_WEBHOOK_TRANSPORT`：投递协议，支持 `http` / `websocket` / `grpc` / `stdio`；未配置时会根据 endpoint 的 `ws://` / `wss://` / `grpc://` / `grpcs://` 或 command 自动推断
- `BILLING_WEBHOOK_ENDPOINT`：Webhook 地址（例如 `https://billing.example.com/webhooks/usage`）
- `BILLING_WEBHOOK_STDIO_COMMAND` / `BILLING_WEBHOOK_STDIO_ARGS` / `BILLING_WEBHOOK_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录；事件 JSON 会写入 stdin
- `BILLING_WEBHOOK_TIMEOUT_MS`：Webhook 超时毫秒，默认 `5000`
- `BILLING_WEBHOOK_MAX_ATTEMPTS` / `BILLING_WEBHOOK_BASE_DELAY_MS` / `BILLING_WEBHOOK_MAX_DELAY_MS`：事件投递失败时的重试次数与指数退避窗口，默认 `3` / `200` / `2000`
- `BILLING_WEBHOOK_REQUIRE_ACK` / `BILLING_WEBHOOK_WEBSOCKET_REQUIRE_ACK`：WebSocket transport 是否等待接收端 ACK 消息，默认 `false`
- `BILLING_WEBHOOK_API_KEY_HEADER` / `BILLING_WEBHOOK_API_KEY`：可选，用于给 webhook 请求增加共享密钥头。
- `BILLING_WEBHOOK_AUTHORIZATION`：可选，用于给 webhook 请求增加 `Authorization` 头。

### Agent 事件队列兼容项

- `AGENT_EVENT_QUEUE_ENABLED` / `AGENT_EVENT_QUEUE_NAME` / `AGENT_EVENT_QUEUE_JOB_NAME`：历史兼容项；gateway 不会建立队列或数据库连接。
- `AGENT_EVENT_QUEUE_REMOVE_ON_COMPLETE` / `AGENT_EVENT_QUEUE_REMOVE_ON_FAIL`：历史兼容项，不影响外部投递。

### Agent 事件 Webhook

- `AGENT_EVENT_WEBHOOK_ENABLED`：是否开启 Agent 事件 Webhook 投递，默认随 endpoint 是否配置自动开启
- `AGENT_EVENT_WEBHOOK_TRANSPORT`：投递协议，支持 `http` / `websocket` / `grpc` / `stdio`；未配置时会根据 endpoint 的 `ws://` / `wss://` / `grpc://` / `grpcs://` 或 command 自动推断
- `AGENT_EVENT_WEBHOOK_ENDPOINT` / `AGENT_EVENT_WEBHOOK_URL`：Agent 事件接收地址
- `AGENT_EVENT_WEBHOOK_STDIO_COMMAND` / `AGENT_EVENT_WEBHOOK_STDIO_ARGS` / `AGENT_EVENT_WEBHOOK_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录；事件 JSON 会写入 stdin
- `AGENT_EVENT_WEBHOOK_TIMEOUT_MS`：Webhook 超时毫秒，默认 `5000`
- `AGENT_EVENT_WEBHOOK_MAX_ATTEMPTS` / `AGENT_EVENT_WEBHOOK_BASE_DELAY_MS` / `AGENT_EVENT_WEBHOOK_MAX_DELAY_MS`：事件投递失败时的重试次数与指数退避窗口，默认 `3` / `200` / `2000`
- `AGENT_EVENT_WEBHOOK_REQUIRE_ACK` / `AGENT_EVENT_WEBHOOK_WEBSOCKET_REQUIRE_ACK`：WebSocket transport 是否等待接收端 ACK 消息，默认 `false`
- `AGENT_EVENT_WEBHOOK_API_KEY_HEADER` / `AGENT_EVENT_WEBHOOK_API_KEY` / `AGENT_EVENT_WEBHOOK_API_KEY_ENV`：可选，用于给 webhook 请求增加共享密钥头
- `AGENT_EVENT_WEBHOOK_AUTHORIZATION`：可选，用于给 webhook 请求增加 `Authorization` 头

### Raw Trace Sync

- `RAW_TRACE_SYNC_ENABLED`：是否上报 raw trace manifest，默认随 endpoint 或 command 是否配置自动开启
- `RAW_TRACE_SYNC_TRANSPORT`：上报协议，支持 `http` / `websocket` / `grpc` / `stdio`；未配置时会根据 endpoint 的 `ws://` / `wss://` / `grpc://` / `grpcs://` 或 command 自动推断
- `RAW_TRACE_SYNC_ENDPOINT` / `RAW_TRACE_SYNC_URL`：raw trace manifest 接收地址
- `RAW_TRACE_SYNC_STDIO_COMMAND` / `RAW_TRACE_SYNC_STDIO_ARGS` / `RAW_TRACE_SYNC_STDIO_CWD`：stdio transport 的外部命令、逗号分隔参数和工作目录；manifest JSON 会写入 stdin
- `RAW_TRACE_SYNC_TIMEOUT_MS`：上报超时毫秒，默认 `5000`
- `RAW_TRACE_SYNC_MAX_ATTEMPTS` / `RAW_TRACE_SYNC_BASE_DELAY_MS` / `RAW_TRACE_SYNC_MAX_DELAY_MS`：manifest 上报失败时的重试次数与指数退避窗口，默认 `3` / `200` / `2000`
- `RAW_TRACE_SYNC_REQUIRE_ACK` / `RAW_TRACE_SYNC_WEBSOCKET_REQUIRE_ACK`：WebSocket transport 是否等待接收端 ACK 消息，默认 `false`
- `RAW_TRACE_SYNC_API_KEY_HEADER` / `RAW_TRACE_SYNC_API_KEY` / `RAW_TRACE_SYNC_AUTHORIZATION`：可选，用于给上报请求增加共享密钥头或 `Authorization` 头

### Agent Runtime 健壮性

- `AGENT_SESSION_LOCK_TIMEOUT_MS`：session lock 获取超时毫秒，默认 `15000`
- `AGENT_EVENT_WORKER_CONCURRENCY`：Agent 持久化事件队列 worker 并发数，默认 `16`
- `AGENT_LLM_RETRY_MAX_ATTEMPTS`：LLM 调用最大尝试次数，默认 `3`
- `AGENT_LLM_RETRY_BASE_DELAY_MS`：LLM 重试基础退避毫秒，默认 `200`
- `AGENT_LLM_RETRY_MAX_DELAY_MS`：LLM 重试最大退避毫秒，默认 `2000`
- `AGENT_LLM_RETRY_BACKOFF_MULTIPLIER`：LLM 重试指数倍数，默认 `2`
- `AGENT_LLM_RETRY_JITTER_MS`：LLM 重试抖动毫秒上限，默认 `100`
- `AGENT_TOOL_RETRY_MAX_ATTEMPTS`：工具执行最大尝试次数，默认 `2`
- `AGENT_TOOL_RETRY_BASE_DELAY_MS`：工具重试基础退避毫秒，默认 `150`
- `AGENT_TOOL_RETRY_MAX_DELAY_MS`：工具重试最大退避毫秒，默认 `1500`
- `AGENT_TOOL_RETRY_BACKOFF_MULTIPLIER`：工具重试指数倍数，默认 `2`
- `AGENT_TOOL_RETRY_JITTER_MS`：工具重试抖动毫秒上限，默认 `50`

## 路由

- `GET /health`
- `GET /metrics`（需 `metrics.enabled=true`）
- `GET /`
- `GET /v1/models`：模型列表接口，默认返回 OpenAI 格式；携带 `anthropic-version`/`anthropic-beta` 或 `?format=anthropic` 时返回 Anthropic 格式。
- `GET /v1/models/:model`：读取单个 OpenAI 格式模型元数据，支持 `providerName/modelName`。
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/moderations`
- `POST /v1/images/generations`
- `WS /v1/responses`
- `WS /v1/responses` 支持将 `chat/completions`、`messages`、`gemini generateContent/streamGenerateContent` 请求体自动转换为 Codex `response.create`；可通过 `?source_adapter=` 显式指定（如 `openai_chat`、`anthropic_messages`、`gemini_generate`、`gemini_stream`）。
- `POST /v1/messages`
- `POST /v1beta/models/*`
- `POST /v1/models/*`
- `POST /mcp`（可通过 `mcpGateway.endpoint` 修改）
- `WS /mcp/ws`（可通过 `mcpGateway.websocket.endpoint` 修改）
- `GET /.well-known/oauth-protected-resource`（启用 `mcpGateway.oauth.enabled` 时）
- `GET /.well-known/oauth-authorization-server`（启用 `mcpGateway.oauth.enabled` 时）
- `POST /agent/agents`
- `GET /agent/agents/:agentId`
- `GET /agent/tools`
- `POST /agent/sessions`
- `POST /agent/sessions/:sessionId/resume`
- `GET /agent/sessions/:sessionId/stream`
- `GET /agent/sessions/:sessionId`
- `GET /agent/sessions/:sessionId/events`
- `POST /agent/sessions/:sessionId/input`
- `POST /agent/sessions/:sessionId/config`
- `POST /agent/sessions/:sessionId/tool-result`
- `POST /agent/sessions/:sessionId/events`
- `GET /manager/config`
- `POST /manager/config/validate`
- `PUT /manager/config`

> 当 `agent.external.enabled=true` 时，`/agent/*` 的写管理接口（POST/PUT/DELETE）会返回 `405`，且不会注册 `/manager/*` 接口。

## Manager 接口

- `GET /manager/config`：读取当前 `gateway.config.json` 文件内容与生效中的内存配置；默认脱敏 API key、password、token、secret 等敏感字段。
- `POST /manager/config/validate`：解析并校验候选配置，返回生效后的配置与重启/热加载 warning，但不写文件、不热更新；默认脱敏敏感字段。
- `PUT /manager/config`：直接覆盖写入 `gateway.config.json`，并立即热加载到网关进程；若提交体中的敏感字段值为 `[REDACTED]`，会保留当前配置文件里的原 secret。
- 如确需排障查看完整 secret，可在受控环境中追加 `?revealSecrets=true`。
- 鉴权策略：
  - 配置了 `MANAGER_API_KEY`：必须携带 `x-manager-key` 或 `Authorization: Bearer <key>`。
  - 未配置 `MANAGER_API_KEY`：仅允许本机地址访问。

示例：

```bash
curl -s http://127.0.0.1:3000/manager/config \
  -H 'x-manager-key: your-manager-key'

curl -s -X POST http://127.0.0.1:3000/manager/config/validate \
  -H 'Content-Type: application/json' \
  -H 'x-manager-key: your-manager-key' \
  -d @gateway.config.json

curl -s -X PUT http://127.0.0.1:3000/manager/config \
  -H 'Content-Type: application/json' \
  -H 'x-manager-key: your-manager-key' \
  -d @gateway.config.json
```

## MCP Gateway

- 固定入口：默认 `POST /mcp`，所有 MCP `tools/list` 与 `tools/call` 都经网关代理。
- WebSocket RPC 入口：默认 `WS /mcp/ws`，消息体为 JSON-RPC 2.0，method 与 HTTP 入口一致（`initialize`/`tools/list`/`tools/call`）。
- 路由隔离：`mcpGateway.websocket.endpoint` 不能配置为 `/v1/responses`；`WS /v1/responses` 专用于 Codex Responses WebSocket 协议。
- 路径保护：`mcpGateway.websocket.endpoint` 必须以 `/mcp` 开头（如 `/mcp/ws`、`/mcp/private/ws`）。
- key/team/org 权限：`mcpGateway.principals` 里按 API key 定义所属团队与工具可见范围。
- 工具发现过滤：客户端请求 `tools/list` 时，仅返回当前 key 被允许访问的工具。
- 公网暴露控制：`mcpGateway.serverExposure` 支持按 MCP server 设置 `internal|public`；外网请求只能访问 `public` server 的工具。
- Guardrails：可限制工具入参大小、阻断敏感字段（`blockedArgumentKeys`）、审计日志自动脱敏（`redactArgumentKeys`）。
- OAuth 发现端点：可启用 `.well-known/oauth-protected-resource` 与 `.well-known/oauth-authorization-server` 元数据。
- OAuth 登录/换 token：`GET /oauth/authorize`、`POST /oauth/token`（授权码 + refresh token）。
- 鉴权方式：`Authorization: Bearer <key>`、`x-api-key`、`x-mcp-key`；WebSocket 可选支持 `?token=<key>`（由 `mcpGateway.websocket.auth` 控制）。
- 单机/多机 RPC 部署模板：`deploy/mcp-ws/README.md`

示例：列工具

```bash
curl -s http://localhost:3000/mcp \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer mcp-demo-key' \\
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'
```

示例：调用工具

```bash
curl -s http://localhost:3000/mcp \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer mcp-demo-key' \\
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"filesystem.read_file","arguments":{"path":"README.md"}}}'
```

示例：WebSocket RPC（`wscat`）

```bash
npx wscat -c ws://localhost:3000/mcp/ws -H 'Authorization: Bearer mcp-demo-key'
# 连接后发送
{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}
```

## 事件驱动 Agent

- Agent 运行时位于 `src/agent/*`，事件总线默认在内存中，未显式配置 `agent.storage.type=filesystem` 时不会写入本地 Agent 状态文件。
- 每个会话维护：`systemPrompt`、`messages`、`allowedTools`、`pendingToolCalls`、`memoryRefs`。
- 工具来源改为 MCP 协议：网关启动后会连接 `agent.mcpServers` 中的 MCP server，并自动暴露工具到会话。
- 创建 Agent 时如果不传 `tools`/`allowedTools`，会自动把当前 MCP 工具注册到该 Agent 的 `allowedTools`。
- Agent 侧工具由远端 MCP server 透传，推荐通过 `toolhub` 暴露 `code_tool.search`、`code_tool.call`、`code_tool.runCode` 三个元工具。
- `gateway` 不再实现任何本地 code-tool 逻辑，仅负责连接远端 MCP（例如 `remote-toolhub`）。
- `code_tool.search` 在 `toolhub` 内部通过 LLM + tree-sitter 分析实现，不再依赖 `gateway /agent/*` 搜索链路。
- `code_tool.call` 的调用格式：`{"tool":"mcp.<server>.<tool>","arguments":{...}}`。
- `code_tool.runCode` 的 Deno 沙箱执行位于 `toolhub`，运行环境需要 `toolhub` 侧 `deno` 在 PATH 中可用。
- 网关使用 `LlmToolDecisionModelClient`：会将会话消息和工具定义发送给 LLM，由 LLM 决定是否调用工具。
- 网关不再提供本地内置工具注册逻辑，所有工具调用都通过 MCP `tools/list` 与 `tools/call` 完成。

快速示例：

```bash
curl -s http://localhost:3000/agent/tools

curl -s -X POST http://localhost:3000/agent/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"fs-agent","description":"文件助手","systemPrompt":"你是文件助手","tools":["code_tool.call"]}'

curl -N -X POST http://localhost:3000/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"agentId":"<your-agent-id>","sessionId":"demo","prompt":"你好"}'

curl -s -X POST http://localhost:3000/agent/sessions/demo/config \
  -H "Content-Type: application/json" \
  -d '{"systemPrompt":"你是一个事件驱动助手","allowedTools":["code_tool.call"]}'

curl -s -X POST http://localhost:3000/agent/sessions/demo/input \
  -H "Content-Type: application/json" \
  -d '{"text":"请帮我读取 README.md 文件"}'

curl -N http://localhost:3000/agent/sessions/demo/stream?fromOffset=0

curl -N -X POST http://localhost:3000/agent/sessions/demo/resume \
  -H "Content-Type: application/json" \
  -d '{"prompt":"继续","fromOffset":0}'

# 如需非流式 JSON 响应，请显式传 stream=false
curl -s -X POST http://localhost:3000/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"demo-json","stream":false}'

curl -s http://localhost:3000/agent/sessions/demo
curl -s http://localhost:3000/agent/sessions/demo/events?limit=20
```

## 转换规则

- 目标 provider 优先级 = `x-target-providers` 请求头 > `x-target-provider` 请求头 > `x-target-model`/请求体 `model` 中的 `provider/model` 前缀 > 默认 provider 列表（`DEFAULT_TARGET_PROVIDERS` 或 JSON `Providers` 数组顺序）> 默认单 provider（`DEFAULT_TARGET_PROVIDER`）> 当前输入协议所属 provider。
- 当目标 provider 是列表时，网关会按顺序尝试并在失败时自动 fallback 到下一个 provider。
- 流式请求支持跨厂商转换：网关会先将目标厂商响应归一化，再按源协议流式回写。
- 目标模型优先级 = `x-target-model` 请求头 > 请求体 `model` > `DEFAULT_{PROVIDER}_MODEL`。
- `x-target-provider` / `x-target-providers` 支持 provider `type`（`openai|anthropic|gemini`）和 provider `name`（来自 JSON 配置 `Providers[].name`）。
- `x-target-model` 与请求体 `model` 支持 `provider/model` 语法（如 `openai-main/GPT-5.3`），会将请求路由到 `openai-main` 并使用模型 `GPT-5.3`。
- 实际上游协议由目标 provider 的 `type` 决定；例如 `type=openai_chat_completions` 时，`/v1/messages` 入站请求会自动转换为 OpenAI Chat Completions 请求。

### 计费响应头

- `x-gateway-billing-provider`
- `x-gateway-billing-currency`
- `x-gateway-billing-input-tokens`
- `x-gateway-billing-output-tokens`
- `x-gateway-billing-cache-read-tokens`
- `x-gateway-billing-cache-write-tokens`
- `x-gateway-billing-total-tokens`
- `x-gateway-billing-input-cost`
- `x-gateway-billing-output-cost`
- `x-gateway-billing-cache-read-cost`
- `x-gateway-billing-cache-write-cost`
- `x-gateway-billing-total-cost`

> 流式请求与上游错误响应不会附带计费头。

### 路由与 fallback 响应头

- `x-gateway-target-provider`：最终成功处理请求的 provider
- `x-gateway-target-provider-name`：最终成功处理请求的 provider name（若命中命名 provider）
- `x-gateway-fallback-used`：是否发生 fallback（`true`）
- `x-gateway-fallback-count`：成功前累计失败次数

### 计费 Webhook 事件（下游集成）

- 请求成功且计费开启时，网关会异步通过 `billingWebhook` 配置的 HTTP/WebSocket/gRPC/stdio transport 投递计费事件；投递失败不会影响主请求返回。
- 事件结构示例：

```json
{
  "eventId": "c5af7a5c-7d5b-4a2c-8be9-2d3830a58e8b",
  "emittedAt": "2026-03-09T12:34:56.789Z",
  "requestId": "req-123",
  "route": { "method": "POST", "url": "/v1/chat/completions" },
  "source": { "provider": "openai", "adapterKey": "openai_chat" },
  "target": { "provider": "anthropic", "model": "claude-3-5-sonnet-latest", "providerName": "anthropic-main" },
  "fallback": { "used": true, "attempts": 1 },
  "identity": {
    "source": "trusted_header",
    "billingSubjectKey": "tenant-a:user-1",
    "tenantId": "tenant-a",
    "userId": "user-1",
    "subject": "sub-1",
    "organizationId": "org-1",
    "plan": "pro"
  },
  "billing": {
    "provider": "anthropic",
    "currency": "USD",
    "usage": {
      "input_tokens": 100,
      "output_tokens": 200,
      "cache_read_tokens": 0,
      "cache_write_tokens": 0,
      "total_tokens": 300
    },
    "rates": {
      "input_per_million_usd": 3,
      "output_per_million_usd": 15,
      "cache_read_per_million_usd": 0,
      "cache_write_per_million_usd": 0
    },
    "cost": {
      "input": 0.0003,
      "output": 0.003,
      "cache_read": 0,
      "cache_write": 0,
      "total": 0.0033
    }
  }
}
```

`identity.source` 可能为 `trusted_header` 或 `http_introspection`。

- 下游接收端示例：

```ts
import Fastify from 'fastify';

const app = Fastify();

app.post('/webhooks/usage', async (request, reply) => {
  const event = request.body as {
    eventId: string;
    billing: { cost: { total: number } };
  };

  // 在这里写入计费系统、消息总线、审计系统或结算流程。
  console.log('billing event', event.eventId, event.billing.cost.total);
  return reply.code(204).send();
});

await app.listen({ host: '0.0.0.0', port: 4000 });
```

## 调用示例

### 1) OpenAI Chat 格式 -> Anthropic

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-target-provider: anthropic' \
  -H 'x-target-model: claude-sonnet-4-20250514' \
  -d '{
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "temperature": 0.2
  }'
```

### 2) Anthropic 格式 -> Gemini

```bash
curl -X POST http://localhost:3000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-target-provider: gemini' \
  -H 'x-target-model: gemini-2.0-flash' \
  -d '{
    "messages": [{"role": "user", "content": "写一个冒泡排序"}],
    "max_tokens": 300
  }'
```

### 3) Gemini 格式透传（默认同协议）

```bash
curl -X POST 'http://localhost:3000/v1beta/models/gemini-2.0-flash:generateContent' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
  }'
```

### 4) 多供应商 fallback（OpenAI -> Anthropic -> Gemini）

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-target-providers: openai,anthropic,gemini' \
  -d '{
    "messages": [{"role": "user", "content": "写一个 hello world"}]
  }'
```

### 5) 按 provider name + model 指定路由

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai-main/GPT-5.3",
    "messages": [{"role": "user", "content": "给我一个发布计划模板"}]
  }'
```

### 6) Anthropic 入站自动转 OpenAI Chat Completions

前提：目标 provider（如 `openai-chat`）在 `gateway.config.json` 中配置了 `"type": "openai_chat_completions"`。

```bash
curl -X POST http://localhost:3000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-target-provider: openai-chat' \
  -d '{
    "model": "openai-chat/gpt-4.1",
    "messages": [{"role": "user", "content": "写一个快速排序"}],
    "max_tokens": 300
  }'
```
