# Gateway Plugins

Gateway plugins provide one unified configuration surface for extending the gateway.
They replace the need to think about separate "provider plugin" and "protocol adapter"
configuration formats.

The single top-level field is `plugins`.

```json
{
  "plugins": [
    {
      "key": "openai-main-patch",
      "enabled": true,
      "match": {
        "provider": "openai",
        "providerName": "openai-main"
      },
      "providerHooks": {
        "request": {
          "headers": {
            "x-custom-feature": "enabled"
          },
          "bodySet": {
            "metadata.gateway": "next-ai"
          }
        }
      }
    },
    {
      "key": "acme",
      "enabled": true,
      "modulePath": "./plugins/acme/index.mjs"
    }
  ]
}
```

`providerPlugins` is still supported for backward compatibility. New configurations
should prefer `plugins`.

## Capabilities

A plugin can provide one or more capabilities.

| Capability | Purpose | Use when |
| --- | --- | --- |
| `providerHooks` | Patch an existing provider request or response flow. | The upstream is mostly OpenAI, Anthropic, or Gemini compatible, but needs different headers, auth, query params, or small body/response changes. |
| `targetAdapters` | Define a complete upstream protocol. | The upstream request or response format is not compatible with the built-in protocols. |
| `sourceAdapters` | Define a client-facing request protocol. | Clients send a custom inbound request format to the gateway. |

In other words:

- Hook capability modifies an already-built request or already-read response payload.
- Adapter capability defines the protocol itself.

## Provider Hooks

Provider hooks are inline, declarative rules under `plugins[].providerHooks`.

```json
{
  "plugins": [
    {
      "key": "vendor-header-patch",
      "enabled": true,
      "match": {
        "providerName": "vendor-main"
      },
      "providerHooks": {
        "auth": {
          "headers": {
            "authorization": "Bearer {{ env.VENDOR_TOKEN }}"
          }
        },
        "request": {
          "query": {
            "api-version": "2026-01-01"
          },
          "bodyRemove": ["unsupported_field"],
          "bodySet": {
            "stream_options.include_usage": true
          }
        },
        "response": {
          "bodySet": {
            "usage.total_tokens": "{{ upstreamPayload.token_count }}"
          }
        }
      }
    }
  ]
}
```

`providerHooks` accepts either one object or an array of objects. The plugin-level
`match` applies as the default match for each hook. A hook can override that default
with its own `provider` or `providerName`.

Supported declarative fields:

- `credentialScope`
- `auth.headers`
- `auth.query`
- `auth.bodySet`
- `auth.bodyMerge`
- `auth.bodyRemove`
- `request.headers`
- `request.query`
- `request.bodySet`
- `request.bodyMerge`
- `request.bodyRemove`
- `response.bodySet`
- `response.bodyMerge`
- `response.bodyRemove`
- `codexOauth`
- `deepseekThinking`

Supported value references include:

- `{{ env.NAME }}`
- `{{ request.headers.x-header }}`
- `{{ request.body.user.id }}`
- `{{ upstreamRequest.body.model }}`
- `{{ upstreamPayload.data.id }}`
- `{{ target.provider }}`
- `{{ target.providerName }}`
- `{{ model }}`

`credentialScope` identifies the stable upstream account used for opaque reasoning
state. It may use the same value references as other declarative fields. The resolved
value is hashed before it is added to a reasoning transport carrier; the original
account identifier or credential is never written to the carrier. Codex OAuth derives
this scope automatically from `accountId`, a JWT account claim, or, as a final stable
fallback, its refresh token.

Strict mode can be enabled on a hook:

```json
{
  "providerHooks": {
    "request": {
      "strict": true,
      "headers": {
        "x-user-id": "{{ request.headers.x-auth-user-id }}"
      }
    }
  }
}
```

With `strict: true`, a missing reference fails the provider attempt instead of being
silently skipped.

## Module Plugins

Use `modulePath` when a plugin needs code. Module plugins are local, trusted Node.js
modules loaded by the gateway process.

```json
{
  "plugins": [
    {
      "key": "acme",
      "enabled": true,
      "modulePath": "./plugins/acme/index.mjs"
    }
  ]
}
```

The module must export `createGatewayPlugin()` or a default factory/object.

```js
export function createGatewayPlugin({ config, plugin }) {
  return {
    targetAdapters: [acmeMessagesTargetAdapter],
    sourceAdapters: [],
    providerHooks: []
  };
}
```

The returned object supports:

```ts
interface GatewayPluginModuleResult {
  sourceAdapters?: SourceAdapter[];
  targetAdapters?: TargetAdapter[];
  providerHooks?: ProviderPlugin[];
  providerPlugins?: ProviderPlugin[];
}
```

`providerPlugins` is accepted as an alias for `providerHooks` for code modules.
Authentication hooks that manage rotating credentials should implement
`resolveCredentialScope()` and return a stable, non-sensitive account identifier. If
an authentication hook cannot resolve a scope, opaque reasoning state is not replayed.

## Target Adapters

A target adapter defines an upstream protocol. It converts the gateway's
`StandardRequest` into an upstream request, and converts the upstream response payload
back into `StandardResponse`.

```js
export const acmeMessagesTargetAdapter = {
  key: 'acme_messages',
  provider: 'acme',
  providerTypes: ['acme_messages'],

  buildRequestFromStandard(input) {
    return {
      ok: true,
      value: {
        method: 'POST',
        url: `${input.targetProviderConfig.baseurl}/messages`,
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${input.targetProviderConfig.apikey}`
        },
        body: {
          model: input.standardRequest.model,
          prompt: input.standardRequest.input
        }
      }
    };
  },

  toStandardResponse(payload) {
    return {
      ok: true,
      value: {
        id: payload.id || 'resp_acme',
        object: 'response',
        status: 'completed',
        model: payload.model || 'unknown',
        output_text: payload.text || '',
        output: [
          {
            id: 'msg_acme',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: payload.text || '',
                annotations: []
              }
            ]
          }
        ],
        usage: payload.usage || {}
      }
    };
  }
};
```

Then configure a provider that uses the adapter's provider type:

```json
{
  "providers": [
    {
      "name": "acme-main",
      "type": "acme_messages",
      "baseurl": "https://api.acme.example",
      "apikey": "secret",
      "models": ["acme-large"]
    }
  ],
  "plugins": [
    {
      "key": "acme",
      "enabled": true,
      "modulePath": "./plugins/acme/index.mjs"
    }
  ],
  "defaultTargetProvider": "acme"
}
```

Provider type conventions:

- Built-in provider types remain supported, such as `openai_responses`,
  `anthropic_messages`, and `gemini_generate_content`.
- Custom provider types are accepted as safe lowercase tokens.
- The provider group is inferred from the part before the first underscore. For
  example, `acme_messages` maps to provider `acme`, and `my-provider_messages`
  maps to provider `my-provider`.
- For custom protocols, route by provider name (`x-target-provider: acme-main`) or
  provider group (`x-target-provider: acme`).

## Upstream Request Shape

Adapters return an `UpstreamRequest`.

```ts
interface UpstreamRequest {
  method?: string;
  url: string;
  headers: Record<string, string>;
  body: unknown;
  bodyEncoding?: 'json' | 'text' | 'form' | 'bytes' | 'none';
}
```

Defaults:

- `method` defaults to `POST`.
- `bodyEncoding` defaults to `json`.

`bodyEncoding` controls how the body is sent:

- `json`: `JSON.stringify(body)`
- `text`: send a string body, or JSON stringify non-strings
- `form`: send `URLSearchParams`
- `bytes`: send the body as a Fetch `BodyInit`
- `none`: send no request body

## Source Adapters

Most plugins only need target adapters because the gateway already accepts OpenAI,
Anthropic, and Gemini-style client requests.

Use a source adapter only when clients send a custom inbound protocol.

```ts
interface SourceAdapter {
  key: string;
  provider: Provider;
  toStandardRequest(input: SourceAdapterRequestInput): Result<StandardRequest>;
  fromStandardResponse(input: SourceAdapterResponseInput): unknown;
  isStreamingRequest(input: SourceAdapterRequestInput): boolean;
  buildPassthroughRequest(input: SourceAdapterRequestInput): Result<UpstreamRequest>;
}
```

Registering a source adapter through a module plugin makes it available in the runtime
registry. A route still needs to call the gateway handler with that source adapter key.

## Load And Reload Behavior

Plugin modules are loaded during runtime config application:

- server startup
- manager config reload
- provider webhook config reload
- external config reload

When config reloads, previously loaded module adapters and module provider hooks are
unregistered and the currently configured modules are loaded again.

Inline `providerHooks` are converted to the same runtime provider hook interface as
legacy `providerPlugins`.

## Security Model

`modulePath` loads code into the gateway process. Treat module plugins as trusted code.

Do not load plugin modules from untrusted users or writable shared directories. A module
plugin can execute arbitrary Node.js code with the gateway process permissions.

For untrusted extension points, use an external service over HTTP, WebSocket, gRPC, or
stdio and expose only data/configuration, not executable modules.

## Migration From providerPlugins

Old config:

```json
{
  "providerPlugins": [
    {
      "key": "openai-main-dynamic",
      "providerName": "openai-main",
      "request": {
        "bodySet": {
          "metadata.gateway": "next-ai"
        }
      }
    }
  ]
}
```

New config:

```json
{
  "plugins": [
    {
      "key": "openai-main-dynamic",
      "match": {
        "providerName": "openai-main"
      },
      "providerHooks": {
        "request": {
          "bodySet": {
            "metadata.gateway": "next-ai"
          }
        }
      }
    }
  ]
}
```

The old format remains valid, so migration can happen incrementally.

## Choosing A Capability

Use `providerHooks` when:

- the upstream is already compatible with a built-in target adapter
- only small request or response JSON changes are needed
- only auth headers, query params, or credentials need patching

Use `targetAdapters` when:

- the upstream request format is structurally different
- the response format is structurally different
- streaming events need custom parsing
- usage, tools, or reasoning have provider-specific semantics

Use `sourceAdapters` when:

- clients call the gateway with a new client-facing protocol
- the gateway must return that client-facing response shape
