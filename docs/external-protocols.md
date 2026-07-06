# Gateway External Protocol Contracts

Gateway does not connect to databases, Redis, queue backends, or object storage SDKs directly. Persistent state, audit pipelines, queues, and dynamic configuration are owned by external services. Gateway talks to those services through JSON contracts over HTTP, WebSocket, gRPC JSON unary, or stdio.

## Shared Transport Rules

### HTTP

- Source requests use `GET` unless the config field explicitly sets `method: "POST"`.
- Event sink requests use `POST` with a JSON body.
- A `2xx` response is treated as success. Non-`2xx` responses are failures and may be retried by event sinks.
- Configured headers are forwarded. If `apiKey` is present, Gateway sends it in `apiKeyHeader`.

### WebSocket

- Gateway opens a connection, sends one JSON request or event, and then waits for one JSON response for source requests.
- For event sinks, successful socket send is treated as delivery by default. Set `requireAck: true` to wait for one receiver message before treating delivery as successful.
- Gateway closes the socket after the response or successful send.

### gRPC JSON Unary

- Gateway uses HTTP/2 unary calls with `application/grpc+json` framing.
- The endpoint may include the full path, for example `grpc://config.example.com/gateway.config.v1.ConfigService/GetConfig`.
- If the endpoint path is omitted, Gateway uses the default path listed in each contract below.

### stdio

- Gateway starts the configured command for each request or event.
- Gateway writes one JSON document plus a newline to stdin.
- Source commands must write one JSON document to stdout and exit `0`.
- Sink commands signal success by exiting `0`; stderr is included in the failure message when the command exits non-zero.

## Dynamic Gateway Config

Config key: `configExternal`

Request:

```json
{ "type": "gateway_config_request" }
```

Response may be any of:

```json
{ "Providers": [], "billing": { "enabled": true } }
```

```json
{ "config": { "Providers": [] } }
```

```json
{ "gatewayConfig": { "Providers": [] } }
```

Default gRPC path:

```text
/gateway.config.v1.ConfigService/GetConfig
```

## Provider Source

Config key: `providerExternal`

Request:

```json
{ "type": "provider_config_request" }
```

Response:

```json
{
  "providers": [
    {
      "name": "openai-main",
      "type": "openai_responses",
      "models": ["gpt-4.1-mini"],
      "apikey": "secret"
    }
  ],
  "plugins": [],
  "providerPlugins": [],
  "virtualModelProfiles": [],
  "credentialEncryption": {}
}
```

The `providers` field may also be returned as `data.providers`. Optional fields are applied only when present.
`plugins` is the preferred unified plugin field. `providerPlugins` remains supported for backward compatibility.

Default gRPC path:

```text
/gateway.providers.v1.ProviderSource/GetProviders
```

## Agent State Source

Config key: `agent.external`

State load request:

```json
{ "type": "agent_state_request" }
```

State load response:

```json
{
  "agents": [
    {
      "agentId": "agent-1",
      "name": "Support Agent",
      "systemPrompt": "You are helpful.",
      "model": "gpt-4.1-mini",
      "allowedTools": []
    }
  ],
  "sessions": [
    {
      "state": {
        "sessionId": "session-1",
        "agentId": "agent-1",
        "systemPrompt": "You are helpful.",
        "messages": [],
        "pendingToolCalls": {},
        "lastEventOffset": 0,
        "updatedAt": "2026-06-08T00:00:00.000Z"
      },
      "events": []
    }
  ]
}
```

Session upsert for WebSocket, gRPC, and stdio:

```json
{
  "type": "agent_session_upsert",
  "sessionId": "session-1",
  "session": { "state": { "sessionId": "session-1" }, "events": [] }
}
```

Session delete for WebSocket, gRPC, and stdio:

```json
{ "type": "agent_session_delete", "sessionId": "session-1" }
```

For HTTP, session mutation keeps REST-compatible paths:

- `PUT {endpoint}/sessions/{sessionId}` with the persisted session snapshot as JSON.
- `DELETE {endpoint}/sessions/{sessionId}`.

Default gRPC paths:

```text
/gateway.agent.v1.AgentStateSource/GetState
/gateway.agent.v1.AgentStateSource/UpsertSession
/gateway.agent.v1.AgentStateSource/DeleteSession
```

## Event Sinks

Config keys:

- `billingWebhook`
- `agent.eventWebhook`
- `rawTrace.sync`

All event sinks accept the event object as the JSON body or message. Gateway treats delivery as successful when:

- HTTP returns `2xx`.
- WebSocket send succeeds.
- gRPC returns `grpc-status: 0`.
- stdio exits `0`.

Event sinks support retry with exponential backoff:

```json
{
  "maxAttempts": 3,
  "baseDelayMs": 200,
  "maxDelayMs": 2000,
  "requireAck": false
}
```

When `requireAck` is enabled for WebSocket sinks, the receiver must send one message after handling the event. Empty messages, non-JSON messages, and JSON objects without `ok: false` or `success: false` are accepted as ACKs. `{"ok":false,"error":"..."}` or `{"success":false,"message":"..."}` rejects the delivery and can be retried.

Default gRPC path:

```text
/gateway.events.v1.EventSink/Publish
```

Billing events include request identity, route, target provider/model, fallback attempts, outcome, trace metadata, and calculated usage/cost.

Agent events include:

```json
{
  "eventId": "event-1",
  "emittedAt": "2026-06-08T00:00:00.000Z",
  "eventType": "USER_INPUT",
  "sessionId": "session-1",
  "correlationId": "corr-1",
  "causationId": "event-0",
  "eventTimestamp": "2026-06-08T00:00:00.000Z",
  "payload": {}
}
```

Raw trace sync events are manifests. Gateway stores raw trace parts in the local spool bundle and sends metadata only:

```json
{
  "requestId": "request-1",
  "captureMode": "body_full",
  "status": "uploaded",
  "uploadAttempts": 1,
  "uploadedAt": "2026-06-08T00:00:00.000Z",
  "route": { "method": "POST", "url": "/v1/responses" },
  "parts": [
    {
      "partType": "client_request",
      "storageBackend": "local",
      "filePath": "/var/spool/gateway/raw-trace/request-1/client_request.json",
      "contentType": "application/json",
      "sha256": "..."
    }
  ]
}
```

## Security Expectations

- Prefer mTLS or a private network for HTTP/gRPC/WebSocket endpoints.
- Use `apiKeyHeader`, `apiKey`, `apiKeyEnv`, `Authorization`, or transport-level credentials for shared authentication.
- Do not return database credentials to Gateway. External services should perform storage access on Gateway's behalf.
- For durable event handling, make the external sink idempotent by `eventId` or `requestId`.
