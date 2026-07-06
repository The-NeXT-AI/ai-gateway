import { existsSync, statSync } from 'node:fs';
import { isAbsolute, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { GatewayConfig, GatewayPluginConfig, ProviderPlugin, SourceAdapter, TargetAdapter } from '../types';
import type { GatewayRuntime } from '../gateway/runtime';
import { resolveTargetAdapterKeys } from '../adapters/registry';

export interface GatewayPluginLoaderLogger {
  info?: (...args: unknown[]) => void;
  warn?: (...args: unknown[]) => void;
}

export interface GatewayPluginFactoryInput {
  config: GatewayConfig;
  plugin: GatewayPluginConfig;
}

export interface GatewayPluginModuleResult {
  sourceAdapters?: SourceAdapter[];
  targetAdapters?: TargetAdapter[];
  providerHooks?: ProviderPlugin[];
  providerPlugins?: ProviderPlugin[];
}

type GatewayPluginFactory = (
  input: GatewayPluginFactoryInput
) => GatewayPluginModuleResult | Promise<GatewayPluginModuleResult>;

interface GatewayPluginModuleExports {
  createGatewayPlugin?: GatewayPluginFactory;
  default?: GatewayPluginFactory | GatewayPluginModuleResult;
}

interface RegistrySnapshot<T> {
  key: string;
  hadPrevious: boolean;
  previous?: T;
}

interface RegisteredModulePluginState {
  sourceAdapters: RegistrySnapshot<SourceAdapter>[];
  targetAdapters: RegistrySnapshot<TargetAdapter>[];
  providerPlugins: RegistrySnapshot<ProviderPlugin>[];
}

interface LoadedGatewayPluginModule {
  plugin: GatewayPluginConfig;
  result: GatewayPluginModuleResult;
}

const registeredModulePluginState = new WeakMap<GatewayRuntime, RegisteredModulePluginState>();

export async function syncGatewayPluginModulesFromConfig(
  runtime: GatewayRuntime,
  config: GatewayConfig,
  logger?: GatewayPluginLoaderLogger
): Promise<void> {
  const loadedModules: LoadedGatewayPluginModule[] = [];
  for (const plugin of config.plugins || []) {
    if (!plugin.enabled || !plugin.modulePath) {
      continue;
    }

    loadedModules.push({
      plugin,
      result: await loadGatewayPluginModule(plugin, config)
    });
  }

  unregisterPreviousModulePlugins(runtime);

  const registered: RegisteredModulePluginState = {
    sourceAdapters: [],
    targetAdapters: [],
    providerPlugins: []
  };

  for (const { plugin, result: moduleResult } of loadedModules) {
    for (const sourceAdapter of moduleResult.sourceAdapters || []) {
      registered.sourceAdapters.push(captureSourceAdapterSnapshot(runtime, sourceAdapter));
      runtime.sourceAdapters.register(sourceAdapter, { overwrite: true });
    }
    for (const targetAdapter of moduleResult.targetAdapters || []) {
      for (const key of resolveTargetAdapterKeys(targetAdapter)) {
        registered.targetAdapters.push(captureTargetAdapterSnapshot(runtime, key));
      }
      runtime.targetAdapters.register(targetAdapter, { overwrite: true });
    }
    for (const providerPlugin of [
      ...(moduleResult.providerHooks || []),
      ...(moduleResult.providerPlugins || [])
    ]) {
      registered.providerPlugins.push(captureProviderPluginSnapshot(runtime, providerPlugin));
      runtime.providerPlugins.register(providerPlugin, { overwrite: true });
    }

    logger?.info?.(
      {
        key: plugin.key,
        modulePath: plugin.modulePath,
        sourceAdapters: moduleResult.sourceAdapters?.length || 0,
        targetAdapters: moduleResult.targetAdapters?.length || 0,
        providerHooks: (moduleResult.providerHooks?.length || 0) + (moduleResult.providerPlugins?.length || 0)
      },
      'Loaded gateway plugin module.'
    );
  }

  registeredModulePluginState.set(runtime, registered);
}

function unregisterPreviousModulePlugins(runtime: GatewayRuntime): void {
  const previous = registeredModulePluginState.get(runtime);
  if (!previous) {
    return;
  }

  restoreProviderPluginSnapshots(runtime, previous.providerPlugins);
  restoreTargetAdapterSnapshots(runtime, previous.targetAdapters);
  restoreSourceAdapterSnapshots(runtime, previous.sourceAdapters);

  registeredModulePluginState.delete(runtime);
}

function captureSourceAdapterSnapshot(
  runtime: GatewayRuntime,
  adapter: SourceAdapter
): RegistrySnapshot<SourceAdapter> {
  const previous = runtime.sourceAdapters.get(adapter.key);
  return {
    key: adapter.key,
    hadPrevious: previous !== undefined,
    previous
  };
}

function captureTargetAdapterSnapshot(
  runtime: GatewayRuntime,
  key: string
): RegistrySnapshot<TargetAdapter> {
  const previous = runtime.targetAdapters.getByKey(key);
  return {
    key,
    hadPrevious: previous !== undefined,
    previous
  };
}

function captureProviderPluginSnapshot(
  runtime: GatewayRuntime,
  plugin: ProviderPlugin
): RegistrySnapshot<ProviderPlugin> {
  const previous = runtime.providerPlugins.get(plugin.key);
  return {
    key: plugin.key,
    hadPrevious: previous !== undefined,
    previous
  };
}

function restoreSourceAdapterSnapshots(
  runtime: GatewayRuntime,
  snapshots: RegistrySnapshot<SourceAdapter>[]
): void {
  for (const snapshot of [...snapshots].reverse()) {
    if (snapshot.hadPrevious && snapshot.previous) {
      runtime.sourceAdapters.register(snapshot.previous, { overwrite: true });
    } else {
      runtime.sourceAdapters.unregister(snapshot.key);
    }
  }
}

function restoreTargetAdapterSnapshots(
  runtime: GatewayRuntime,
  snapshots: RegistrySnapshot<TargetAdapter>[]
): void {
  for (const snapshot of [...snapshots].reverse()) {
    if (snapshot.hadPrevious && snapshot.previous) {
      runtime.targetAdapters.register(snapshot.previous, { overwrite: true });
    } else {
      runtime.targetAdapters.unregister(snapshot.key);
    }
  }
}

function restoreProviderPluginSnapshots(
  runtime: GatewayRuntime,
  snapshots: RegistrySnapshot<ProviderPlugin>[]
): void {
  for (const snapshot of [...snapshots].reverse()) {
    if (snapshot.hadPrevious && snapshot.previous) {
      runtime.providerPlugins.register(snapshot.previous, { overwrite: true });
    } else {
      runtime.providerPlugins.unregister(snapshot.key);
    }
  }
}

async function loadGatewayPluginModule(
  plugin: GatewayPluginConfig,
  config: GatewayConfig
): Promise<GatewayPluginModuleResult> {
  const modulePath = resolveGatewayPluginModulePath(plugin.modulePath);
  if (!existsSync(modulePath)) {
    throw new Error(`Gateway plugin "${plugin.key}" modulePath does not exist: ${plugin.modulePath}`);
  }

  const moduleUrl = pathToFileURL(modulePath);
  moduleUrl.searchParams.set('gatewayPlugin', plugin.key);
  moduleUrl.searchParams.set('mtime', String(statSync(modulePath).mtimeMs));

  const exports = (await import(moduleUrl.href)) as GatewayPluginModuleExports;
  const factory = exports.createGatewayPlugin || exports.default;
  if (!factory) {
    throw new Error(
      `Gateway plugin "${plugin.key}" must export createGatewayPlugin() or a default plugin factory/object.`
    );
  }

  const result =
    typeof factory === 'function'
      ? await factory({ config, plugin })
      : factory;
  return normalizeGatewayPluginModuleResult(plugin, result);
}

function resolveGatewayPluginModulePath(modulePath: string | undefined): string {
  const normalized = modulePath?.trim();
  if (!normalized) {
    throw new Error('Gateway plugin modulePath is required.');
  }

  return isAbsolute(normalized) ? normalized : resolve(process.cwd(), normalized);
}

function normalizeGatewayPluginModuleResult(
  plugin: GatewayPluginConfig,
  result: GatewayPluginModuleResult | undefined
): GatewayPluginModuleResult {
  if (!result || typeof result !== 'object') {
    throw new Error(`Gateway plugin "${plugin.key}" factory must return a plugin object.`);
  }

  return {
    sourceAdapters: normalizeArray<SourceAdapter>(result.sourceAdapters),
    targetAdapters: normalizeArray<TargetAdapter>(result.targetAdapters),
    providerHooks: normalizeArray<ProviderPlugin>(result.providerHooks),
    providerPlugins: normalizeArray<ProviderPlugin>(result.providerPlugins)
  };
}

function normalizeArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value.filter(Boolean) as T[]) : [];
}
