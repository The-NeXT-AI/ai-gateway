import type { Provider, ProviderType, StandardRequestInputMessage } from './types';

export function parseProvider(value: string | undefined): Provider | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === 'openai') {
    return 'openai';
  }

  if (normalized === 'anthropic' || normalized === 'claude') {
    return 'anthropic';
  }

  if (normalized === 'gemini' || normalized === 'google') {
    return 'gemini';
  }

  return undefined;
}

export function parseProviderList(value: string | undefined): Provider[] {
  if (!value) {
    return [];
  }

  const providers = value
    .split(',')
    .map((item) => parseProvider(item))
    .filter((item): item is Provider => Boolean(item));

  if (providers.length <= 1) {
    return providers;
  }

  const deduped: Provider[] = [];
  for (const provider of providers) {
    if (!deduped.includes(provider)) {
      deduped.push(provider);
    }
  }

  return deduped;
}

export function providerFromProviderType(type: ProviderType): Provider {
  if (type === 'openai_chat_completions' || type === 'openai_responses') {
    return 'openai';
  }

  if (type === 'anthropic_messages') {
    return 'anthropic';
  }

  return 'gemini';
}

export function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value;
}

export function readHeader(value: string | string[] | undefined): string | undefined {
  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value) && value.length > 0) {
    return value[0];
  }

  return undefined;
}

export function readBearerToken(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  const match = value.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return undefined;
  }

  return match[1].trim();
}

export function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

export function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  return undefined;
}

export function asBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

export function asStop(value: unknown): string | string[] | undefined {
  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value)) {
    const stops = value.filter((item): item is string => typeof item === 'string');
    return stops.length > 0 ? stops : undefined;
  }

  return undefined;
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return isObject(value) && !Array.isArray(value);
}

export function extractTextFromPart(part: unknown): string {
  if (typeof part === 'string') {
    return part.trim();
  }

  if (!isObject(part)) {
    return '';
  }

  if (typeof part.text === 'string') {
    return part.text.trim();
  }

  if (typeof part.input_text === 'string') {
    return part.input_text.trim();
  }

  if (typeof part.output_text === 'string') {
    return part.output_text.trim();
  }

  if (part.type === 'input_text' || part.type === 'output_text') {
    return asString(part.text) || '';
  }

  return '';
}

export function normalizeMessageRole(role: unknown): 'system' | 'user' | 'assistant' {
  const value = String(role || '').toLowerCase();

  if (value === 'assistant' || value === 'model') {
    return 'assistant';
  }

  if (value === 'system' || value === 'developer') {
    return 'system';
  }

  return 'user';
}

export function normalizeConversationRole(role: unknown): 'user' | 'assistant' {
  return normalizeMessageRole(role) === 'assistant' ? 'assistant' : 'user';
}

export function collectStandardInputMessages(input: string | StandardRequestInputMessage[]): StandardRequestInputMessage[] {
  if (typeof input === 'string') {
    const text = input.trim();
    if (!text) {
      return [];
    }

    return [
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text }]
      }
    ];
  }

  return input;
}

export function extractStandardInputText(input: string | StandardRequestInputMessage[]): string {
  return collectStandardInputMessages(input)
    .flatMap((message) => message.content)
    .map((item) => (item.type === 'input_text' ? item.text : ''))
    .filter(Boolean)
    .join('\n')
    .trim();
}
