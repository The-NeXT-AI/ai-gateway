# Next AI Gateway

[简体中文](README.zh-CN.md)

A TypeScript + Fastify AI protocol gateway for OpenAI, Anthropic, Gemini, MCP Gateway, and event-driven agent workflows.

## Quick Start

```bash
npm install
npm run dev
```

Production:

```bash
npm run build
npm start
```

Docker Compose:

```bash
docker compose up --build
```

## Features

- OpenAI `chat/completions`, `responses`, `embeddings`, `moderations`, and `images/generations`
- Anthropic `messages`
- Gemini `generateContent` / `streamGenerateContent`
- Cross-protocol conversion, provider fallback, provider plugins, health checks, metrics, idempotency, concurrency isolation, circuit breaking, and retries
- MCP Gateway, MCP WebSocket RPC, and event-driven agents
- HTTP / WebSocket / gRPC / stdio external config sources and event delivery

## Documentation

- [Full usage guide](docs/usage.md)
- [External protocol integration](docs/external-protocols.md)
- [Publishing and CI/CD](docs/publishing.md)
- [MCP WebSocket deployment templates](deploy/mcp-ws/README.md)

## npm Publishing

This repository includes `.npmignore`, `prepack` builds, GitHub Actions CI/CD, and a release command that publishes a specific npm version:

```bash
npm run release -- 1.2.3
```

See [Publishing and CI/CD](docs/publishing.md) for more options.
