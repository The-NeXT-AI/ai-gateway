# Obit reasoning controls

For an `openai_chat_completions` provider whose `baseurl` host is
`relay.obitmc.com`, protocol conversion preserves the caller's reasoning controls:

| Incoming control | OpenAI chat request |
| --- | --- |
| Anthropic `thinking: { type: "disabled" }` | `reasoning_effort: "none"` |
| Anthropic `output_config.effort` | The same `reasoning_effort` value |
| Responses `reasoning.effort` | The same `reasoning_effort` value |
| No effort or thinking control | No reasoning override |

The supported effort values are `none`, `minimal`, `low`, `medium`, `high`, and
`xhigh`. An explicit disabled thinking control takes precedence over effort.
An existing chat `reasoning_effort` otherwise takes precedence over translated
controls. The Obit worker remains responsible for mapping effort to each model's
chat template; the gateway does not choose a new default effort.

The provider's existing `openaiChatThinkingOptions: "disabled"` policy still strips
thinking controls. Other providers retain their existing behavior. Anthropic
thinking budgets and display settings are not translated to Obit token budgets or
response filtering.
