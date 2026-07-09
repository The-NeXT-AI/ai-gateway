import type { ProviderConfig, StandardRequest } from '../../../../types';

export interface OpenAIChatProviderThinkingMatchInput {
  providerConfig: ProviderConfig;
  model?: string;
}

export interface OpenAIChatProviderThinkingRewriteInput extends OpenAIChatProviderThinkingMatchInput {
  body: Record<string, unknown>;
  standardRequest?: StandardRequest;
}

export interface OpenAIChatProviderThinkingAdapter {
  key: string;
  matches(input: OpenAIChatProviderThinkingMatchInput): boolean;
  rewriteRequest(input: OpenAIChatProviderThinkingRewriteInput): void;
}
