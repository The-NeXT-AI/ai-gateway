import type { ProviderPlugin, ProviderPluginRequestInput } from '../types';
import { isPlainRecord } from '../utils';

const builtinOpenCodePluginKey = 'builtin:opencode';

/**
 * OpenCode (opencode-go) compatibility plugin.
 *
 * opencode-go rejects OpenAI chat-completions requests that carry fields it
 * does not understand, returning HTTP 400 invalid_request_error. The
 * ai-gateway OpenAI source adapter injects `reasoning_split: true` on every
 * chat-completions request; opencode-go rejects it. Anthropic-era conversion
 * may also surface `reasoning`, `reasoning_effort`, and `cache_control`.
 *
 * This passthrough transformer (mirror of the deepseek-thinking builtin) strips
 * those fields right before the upstream fetch. Gated on the opencode.ai host
 * so other openai providers are unaffected.
 */
export function createOpenCodeProviderPlugin(): ProviderPlugin {
  return {
    key: builtinOpenCodePluginKey,
    provider: 'openai',
    transformRequest(input) {
      if (!isOpenCodeChatCompletionsRequest(input)) {
        return {
          ok: true,
          value: input.upstreamRequest
        };
      }

      const body = isPlainRecord(input.upstreamRequest.body)
        ? { ...input.upstreamRequest.body }
        : undefined;
      if (!body) {
        return {
          ok: true,
          value: input.upstreamRequest
        };
      }

      stripOpenCodeUnsupportedFields(body);

      return {
        ok: true,
        value: {
          ...input.upstreamRequest,
          body
        }
      };
    }
  };
}

function isOpenCodeChatCompletionsRequest(input: ProviderPluginRequestInput): boolean {
  if (input.targetProvider !== 'openai') {
    return false;
  }

  try {
    const url = new URL(input.upstreamRequest.url);
    if (url.hostname !== 'opencode.ai') {
      return false;
    }
    return url.pathname.endsWith('/chat/completions');
  } catch {
    return input.upstreamRequest.url.includes('opencode.ai');
  }
}

function stripOpenCodeUnsupportedFields(body: Record<string, unknown>): void {
  delete body.reasoning_split;
  delete body.reasoning;
  delete body.reasoning_effort;
  delete body.cache_control;

  const messages = body.messages;
  if (!Array.isArray(messages)) {
    return;
  }

  body.messages = messages.map((message) => {
    if (!isPlainRecord(message)) {
      return message;
    }

    const nextMessage: Record<string, unknown> = { ...message };
    delete nextMessage.cache_control;

    if (Array.isArray(nextMessage.content)) {
      nextMessage.content = nextMessage.content.map((item) => {
        if (!isPlainRecord(item)) {
          return item;
        }
        const nextItem: Record<string, unknown> = { ...item };
        delete nextItem.cache_control;
        return nextItem;
      });
    }

    return nextMessage;
  });
}

