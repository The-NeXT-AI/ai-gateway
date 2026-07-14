import type { Provider, ProviderType, StandardRequestInputMessage } from './types';

const maxErrorCauseDepth = 8;

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

  return isSafeProviderToken(normalized) ? normalized : undefined;
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

  if (type === 'gemini_generate_content' || type === 'gemini_interactions') {
    return 'gemini';
  }

  const normalized = String(type).trim().toLowerCase();
  if (!normalized) {
    return 'unknown';
  }

  const separatorIndex = findProviderTypeSeparatorIndex(normalized);
  const provider = separatorIndex > 0 ? normalized.slice(0, separatorIndex) : normalized;
  return isSafeProviderToken(provider) ? provider : 'unknown';
}

export function isSafeProviderToken(value: string): boolean {
  return /^[a-z0-9][a-z0-9_.:-]*$/.test(value);
}

function findProviderTypeSeparatorIndex(value: string): number {
  const underscoreIndex = value.indexOf('_');
  if (underscoreIndex > 0) {
    return underscoreIndex;
  }

  const colonIndex = value.indexOf(':');
  if (colonIndex > 0) {
    return colonIndex;
  }

  const dotIndex = value.indexOf('.');
  return dotIndex > 0 ? dotIndex : -1;
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

export function formatErrorWithCause(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }

  const messages: string[] = [];
  const seen = new Set<Error>();
  let current: Error | undefined = error;

  for (let depth = 0; current && depth < maxErrorCauseDepth; depth += 1) {
    if (seen.has(current)) {
      messages.push('[circular cause]');
      return messages.join(' => ');
    }

    seen.add(current);
    messages.push(formatSingleError(current));
    current = current.cause instanceof Error ? current.cause : undefined;
  }

  if (current) {
    messages.push('[cause chain truncated]');
  }

  return messages.join(' => ');
}

function formatSingleError(error: Error): string {
  const message = error.message || error.name || 'Error';
  const code = (error as Error & { code?: unknown }).code;
  return typeof code === 'string' && code ? `${message} (${code})` : message;
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
