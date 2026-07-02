import type { ProviderPlugin, ProviderPluginRequestInput } from '../types';
import { isPlainRecord } from '../utils';

type DeepSeekEffort = 'high' | 'max';
type DeepSeekThinkingType = 'enabled' | 'disabled';

const builtinDeepSeekThinkingPluginKey = 'builtin:deepseek-thinking';

export function createDeepSeekThinkingProviderPlugin(): ProviderPlugin {
  return {
    key: builtinDeepSeekThinkingPluginKey,
    provider: 'openai',
    transformRequest(input) {
      if (!isOpenAIChatCompletionsRequest(input)) {
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

      const thinkingType = resolveThinkingType(input, body);
      const effort = resolveReasoningEffort(input, body);
      if (!thinkingType && !effort) {
        return {
          ok: true,
          value: input.upstreamRequest
        };
      }

      if (thinkingType === 'disabled') {
        body.thinking = { type: 'disabled' };
        delete body.reasoning_effort;
        stripOutputConfigEffort(body);
        return {
          ok: true,
          value: {
            ...input.upstreamRequest,
            body
          }
        };
      }

      body.thinking = { type: thinkingType || 'enabled' };
      if (effort) {
        body.reasoning_effort = effort;
      }
      stripOutputConfigEffort(body);

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

function isOpenAIChatCompletionsRequest(input: ProviderPluginRequestInput): boolean {
  if (input.targetProvider !== 'openai') {
    return false;
  }

  try {
    const url = new URL(input.upstreamRequest.url);
    return url.pathname.endsWith('/chat/completions');
  } catch {
    return input.upstreamRequest.url.includes('/chat/completions');
  }
}

function resolveThinkingType(
  input: ProviderPluginRequestInput,
  body: Record<string, unknown>
): DeepSeekThinkingType | undefined {
  return (
    readThinkingType(body.thinking) ||
    readThinkingType(input.standardRequest?.thinking) ||
    readThinkingType(readOriginalBody(input).thinking)
  );
}

function resolveReasoningEffort(
  input: ProviderPluginRequestInput,
  body: Record<string, unknown>
): DeepSeekEffort | undefined {
  return (
    normalizeReasoningEffort(body.reasoning_effort) ||
    normalizeReasoningEffort(readNestedValue(body.output_config, 'effort')) ||
    normalizeReasoningEffort(readNestedValue(input.standardRequest?.output_config, 'effort')) ||
    normalizeReasoningEffort(readNestedValue(input.standardRequest?.reasoning, 'effort')) ||
    normalizeReasoningEffort(readOriginalBody(input).reasoning_effort) ||
    normalizeReasoningEffort(readNestedValue(readOriginalBody(input).output_config, 'effort')) ||
    normalizeReasoningEffort(readNestedValue(readOriginalBody(input).reasoning, 'effort'))
  );
}

function readOriginalBody(input: ProviderPluginRequestInput): Record<string, unknown> {
  return isPlainRecord(input.request.body) ? input.request.body : {};
}

function readThinkingType(value: unknown): DeepSeekThinkingType | undefined {
  if (typeof value === 'boolean') {
    return value ? 'enabled' : 'disabled';
  }

  if (typeof value === 'string') {
    return normalizeThinkingType(value);
  }

  if (!isPlainRecord(value)) {
    return undefined;
  }

  return normalizeThinkingType(value.type);
}

function normalizeThinkingType(value: unknown): DeepSeekThinkingType | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (['enabled', 'enable', 'on', 'true'].includes(normalized)) {
    return 'enabled';
  }
  if (['disabled', 'disable', 'off', 'false'].includes(normalized)) {
    return 'disabled';
  }
  return undefined;
}

function normalizeReasoningEffort(value: unknown): DeepSeekEffort | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase().replace(/[-_\s]+/g, '');
  if (normalized === 'max' || normalized === 'xhigh') {
    return 'max';
  }
  if (normalized === 'high' || normalized === 'medium' || normalized === 'low') {
    return 'high';
  }
  return undefined;
}

function stripOutputConfigEffort(body: Record<string, unknown>): void {
  const outputConfig = body.output_config;
  if (!isPlainRecord(outputConfig) || !Object.prototype.hasOwnProperty.call(outputConfig, 'effort')) {
    return;
  }

  const nextOutputConfig = { ...outputConfig };
  delete nextOutputConfig.effort;
  if (Object.keys(nextOutputConfig).length === 0) {
    delete body.output_config;
    return;
  }

  body.output_config = nextOutputConfig;
}

function readNestedValue(value: unknown, key: string): unknown {
  return isPlainRecord(value) ? value[key] : undefined;
}

