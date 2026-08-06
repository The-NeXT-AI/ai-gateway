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
export AUTH_STATIC_API_KEYS='replace-with-gateway-client-key'
export MANAGER_API_KEY='replace-with-manager-admin-key'
export OPENAI_API_KEY='sk-...'
export MCP_REMOTE_KEY='replace-with-strong-mcp-key'
export TOOLHUB_MANAGEMENT_TOKEN='replace-with-toolhub-admin-key'
export MINIMAX_API_KEY='replace-with-minimax-api-key'
export ANTHROPIC_API_KEY='replace-with-tool-search-provider-key'
docker compose up --build
```

## Features

- OpenAI `chat/completions`, `responses`, `embeddings`, `moderations`, `images/generations`, `images/edits`, `videos`, `videos/:id`, and `videos/:id/content`
- xAI `videos/generations` and `videos/:id`, with OpenAI ↔ xAI video request/response conversion
- Authenticated, encrypted, expiring video IDs preserve routing/model/owner context without exposing it; configure a shared `media.videoIdSigningSecret` for multi-instance deployments
- The gateway does not hard-code an OpenAI video model. OpenAI has announced removal of the Videos API, `sora-2`, and `sora-2-pro` on September 24, 2026, with no replacement; pin provider models explicitly and plan migrations accordingly ([official deprecation notice](https://developers.openai.com/api/docs/deprecations#2026-03-24-sora-2-video-generation-models-and-videos-api))
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

## Provider-native reasoning continuity

The gateway carries lossless OpenAI Responses, Anthropic thinking, Gemini Part, and Gemini Interactions state in a bounded carrier-v3 envelope. Native state is replayed only when its protocol, provider service, endpoint, credential scope, model rules, capture completeness, provider status, and dependency group are compatible. Readable reasoning can be projected only to protocols that explicitly support it; otherwise optional reasoning is removed. Active native tool groups fail closed, while closed historical groups are removed atomically so calls and results are never orphaned.

Responses history configuration remains backward compatible: `encrypted` means native history (including encrypted reasoning), `plaintext` sends only readable reasoning, and `strip` removes reasoning. An explicit `strip` may discard compacted context. An `auto` decision that resolves to `strip` is not treated as consent to lose a compaction that is the only representation of older history; that request fails with `incompatible_compacted_history`. Stateful IDs are reused only on the same service, endpoint, and credential route; otherwise the gateway falls back to complete manual history or returns a compatibility error.

Carrier replay is designed for trusted local clients and single-tenant deployments. Carrier origin fields are routing provenance, not multi-tenant authentication, and the carrier intentionally has no HMAC. Public proxy and untrusted-client deployments are outside this trust boundary; a non-loopback listener emits a startup warning. Logs redact raw native payloads, ciphertext, signatures, fingerprints, and carrier strings.

Limits derive from `bodyLimitBytes`: total encoded carriers are capped at `min(16 MiB, bodyLimitBytes / 2)`, each decoded payload at `min(8 MiB, bodyLimitBytes / 4)`, native items at 4096, and JSON nesting at 32. Gemini signature envelopes in `tool_call.id` are signature-only and capped at 64 KiB. Oversize carriers return 413; malformed, duplicate/conflicting, or cyclic state returns 400.

## npm Publishing

This repository includes `.npmignore`, `prepack` builds, GitHub Actions CI/CD, and a release command that publishes a specific npm version:

```bash
npm run release -- 1.2.3
```

See [Publishing and CI/CD](docs/publishing.md) for more options.
