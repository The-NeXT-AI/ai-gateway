import type { ProviderConfig } from '../../../../types';
import { deepSeekOpenAIChatProviderThinkingAdapter } from './deepseek';
import { minimaxOpenAIChatProviderThinkingAdapter } from './minimax';
import type { OpenAIChatProviderThinkingAdapter } from './types';
import { xiaomiMimoOpenAIChatProviderThinkingAdapter } from './xiaomi-mimo';
import { zhipuOpenAIChatProviderThinkingAdapter } from './zhipu';

const openAIChatProviderThinkingAdapters: OpenAIChatProviderThinkingAdapter[] = [
  minimaxOpenAIChatProviderThinkingAdapter,
  deepSeekOpenAIChatProviderThinkingAdapter,
  xiaomiMimoOpenAIChatProviderThinkingAdapter,
  zhipuOpenAIChatProviderThinkingAdapter
];

export function resolveOpenAIChatProviderThinkingAdapter(
  providerConfig: ProviderConfig,
  model?: string
): OpenAIChatProviderThinkingAdapter | undefined {
  return openAIChatProviderThinkingAdapters.find((adapter) =>
    adapter.matches({
      providerConfig,
      model
    })
  );
}
