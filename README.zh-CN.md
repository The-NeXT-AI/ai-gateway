# Next AI Gateway

[English](README.md)

基于 TypeScript + Fastify 的 AI 协议网关，支持 OpenAI、Anthropic、Gemini、MCP Gateway 与事件驱动 Agent。

## 快速开始

```bash
npm install
npm run dev
```

生产运行：

```bash
npm run build
npm start
```

Docker Compose：

```bash
docker compose up --build
```

## 主要能力

- OpenAI `chat/completions`、`responses`、`embeddings`、`moderations`、`images/generations`
- Anthropic `messages`
- Gemini `generateContent` / `streamGenerateContent`
- 跨协议转换、供应商 fallback、provider plugin、健康检查、指标、幂等、并发隔离、熔断与重试
- MCP Gateway、MCP WebSocket RPC、事件驱动 Agent
- HTTP / WebSocket / gRPC / stdio 外部配置源与事件投递

## 文档

- [完整使用文档](docs/usage.md)
- [外部协议集成](docs/external-protocols.md)
- [发布与 CI/CD](docs/publishing.md)
- [MCP WebSocket 部署模板](deploy/mcp-ws/README.md)

## npm 发布

本仓库包含 `.npmignore`、`prepack` 构建、GitHub Actions CI/CD，以及可指定版本发布的 release 命令：

```bash
npm run release -- 1.2.3
```

更多选项见 [发布与 CI/CD](docs/publishing.md)。
