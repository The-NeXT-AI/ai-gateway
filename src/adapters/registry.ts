import type { Provider, ProviderConfig, ProviderPlugin, SourceAdapter, TargetAdapter } from '../types';

interface RegisterOptions {
  overwrite?: boolean;
}

export class SourceAdapterRegistry {
  private readonly adapters = new Map<string, SourceAdapter>();

  register(adapter: SourceAdapter, options?: RegisterOptions): string {
    const exists = this.adapters.has(adapter.key);
    if (exists && !options?.overwrite) {
      throw new Error(`Source adapter already registered: ${adapter.key}`);
    }

    this.adapters.set(adapter.key, adapter);
    return adapter.key;
  }

  get(key: string): SourceAdapter | undefined {
    return this.adapters.get(key);
  }

  unregister(key: string): boolean {
    return this.adapters.delete(key);
  }

  list(): SourceAdapter[] {
    return Array.from(this.adapters.values());
  }
}

export class TargetAdapterRegistry {
  private readonly adapters = new Map<string, TargetAdapter>();

  register(adapter: TargetAdapter, options?: RegisterOptions): string[] {
    const keys = resolveTargetAdapterKeys(adapter);
    for (const key of keys) {
      const exists = this.adapters.has(key);
      if (exists && !options?.overwrite) {
        throw new Error(`Target adapter already registered: ${key}`);
      }
    }

    for (const key of keys) {
      this.adapters.set(key, adapter);
    }
    return keys;
  }

  get(provider: Provider, providerConfig?: ProviderConfig): TargetAdapter | undefined {
    const byProvider = this.adapters.get(provider);
    if (byProvider && byProvider.providerFallback !== true) {
      return byProvider;
    }

    if (providerConfig?.type) {
      const byProviderType = this.adapters.get(providerConfig.type);
      if (byProviderType) {
        return byProviderType;
      }
    }

    if (byProvider) {
      return byProvider;
    }

    return undefined;
  }

  getByKey(key: string): TargetAdapter | undefined {
    return this.adapters.get(key);
  }

  unregister(key: string): boolean {
    return this.adapters.delete(key);
  }

  list(): TargetAdapter[] {
    return Array.from(new Set(this.adapters.values()));
  }
}

export class ProviderPluginRegistry {
  private readonly plugins = new Map<string, ProviderPlugin>();

  register(plugin: ProviderPlugin, options?: RegisterOptions): void {
    const exists = this.plugins.has(plugin.key);
    if (exists && !options?.overwrite) {
      throw new Error(`Provider plugin already registered: ${plugin.key}`);
    }

    this.plugins.set(plugin.key, plugin);
  }

  get(key: string): ProviderPlugin | undefined {
    return this.plugins.get(key);
  }

  unregister(key: string): boolean {
    return this.plugins.delete(key);
  }

  clear(): void {
    this.plugins.clear();
  }

  resolve(provider: Provider, providerName?: string): ProviderPlugin[] {
    const normalizedProviderName = normalizeProviderName(providerName);
    const matched: ProviderPlugin[] = [];

    for (const plugin of this.plugins.values()) {
      if (plugin.provider && plugin.provider !== provider) {
        continue;
      }

      if (plugin.providerName) {
        const pluginProviderName = normalizeProviderName(plugin.providerName);
        if (!normalizedProviderName || pluginProviderName !== normalizedProviderName) {
          continue;
        }
      }

      matched.push(plugin);
    }

    return matched;
  }

  list(): ProviderPlugin[] {
    return Array.from(this.plugins.values());
  }
}

function normalizeProviderName(value: string | undefined): string | undefined {
  const normalized = value?.trim().toLowerCase();
  return normalized ? normalized : undefined;
}

export function resolveTargetAdapterKeys(adapter: TargetAdapter): string[] {
  const keys = new Set<string>();
  if (adapter.key) {
    keys.add(adapter.key);
  }
  for (const providerType of adapter.providerTypes || []) {
    keys.add(providerType);
  }
  if (keys.size === 0 || adapter.providerFallback === true) {
    keys.add(adapter.provider);
  }
  return Array.from(keys);
}
