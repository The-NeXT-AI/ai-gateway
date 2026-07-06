import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import { createGatewayRuntime } from '../gateway/runtime';
import { syncGatewayPluginModulesFromConfig } from './loader';

describe('syncGatewayPluginModulesFromConfig', () => {
  it('keeps previously loaded module plugins when a new module fails to load', async () => {
    const pluginDir = await mkdtemp(join(tmpdir(), 'gateway-plugin-loader-'));
    const modulePath = join(pluginDir, 'acme.mjs');
    await writePluginModule(modulePath, {
      key: 'acme_messages',
      provider: 'acme',
      providerTypes: ['acme_messages'],
      marker: 'old-module'
    });

    const runtime = createGatewayRuntime();
    const initialConfig = parseGatewayConfigFromRaw({
      plugins: [
        {
          key: 'acme',
          modulePath
        }
      ]
    });

    try {
      await syncGatewayPluginModulesFromConfig(runtime, initialConfig);
      expect(readAdapterMarker(runtime.targetAdapters.getByKey('acme_messages'))).toBe('old-module');

      const badConfig = parseGatewayConfigFromRaw({
        plugins: [
          {
            key: 'missing',
            modulePath: join(pluginDir, 'missing.mjs')
          }
        ]
      });
      await expect(syncGatewayPluginModulesFromConfig(runtime, badConfig)).rejects.toThrow(
        /modulePath does not exist/
      );

      expect(readAdapterMarker(runtime.targetAdapters.getByKey('acme_messages'))).toBe('old-module');
    } finally {
      await rm(pluginDir, { recursive: true, force: true });
    }
  });

  it('restores overwritten target adapters when module plugins are removed', async () => {
    const pluginDir = await mkdtemp(join(tmpdir(), 'gateway-plugin-loader-'));
    const modulePath = join(pluginDir, 'openai-override.mjs');
    await writePluginModule(modulePath, {
      key: 'openai_responses',
      provider: 'openai',
      providerTypes: ['openai_responses'],
      marker: 'module-override'
    });

    const runtime = createGatewayRuntime();
    const builtinAdapter = runtime.targetAdapters.getByKey('openai_responses');
    expect(builtinAdapter).toBeDefined();

    const enabledConfig = parseGatewayConfigFromRaw({
      plugins: [
        {
          key: 'openai-override',
          modulePath
        }
      ]
    });
    const disabledConfig = parseGatewayConfigFromRaw({
      plugins: []
    });

    try {
      await syncGatewayPluginModulesFromConfig(runtime, enabledConfig);
      expect(readAdapterMarker(runtime.targetAdapters.getByKey('openai_responses'))).toBe(
        'module-override'
      );

      await syncGatewayPluginModulesFromConfig(runtime, disabledConfig);
      expect(runtime.targetAdapters.getByKey('openai_responses')).toBe(builtinAdapter);
    } finally {
      await rm(pluginDir, { recursive: true, force: true });
    }
  });
});

async function writePluginModule(
  modulePath: string,
  adapter: {
    key: string;
    provider: string;
    providerTypes: string[];
    marker: string;
  }
): Promise<void> {
  await writeFile(
    modulePath,
    `
export function createGatewayPlugin() {
  return {
    targetAdapters: [
      {
        key: ${JSON.stringify(adapter.key)},
        provider: ${JSON.stringify(adapter.provider)},
        providerTypes: ${JSON.stringify(adapter.providerTypes)},
        buildRequestFromStandard() {
          return {
            ok: true,
            value: {
              url: 'https://plugin.example/messages',
              headers: {},
              body: {
                marker: ${JSON.stringify(adapter.marker)}
              }
            }
          };
        },
        toStandardResponse(payload) {
          return {
            ok: true,
            value: payload
          };
        }
      }
    ]
  };
}
`,
    'utf8'
  );
}

function readAdapterMarker(adapter: unknown): string | undefined {
  if (!adapter || typeof adapter !== 'object' || !('buildRequestFromStandard' in adapter)) {
    return undefined;
  }

  const result = (adapter as {
    buildRequestFromStandard: (input: unknown) => { ok: true; value: { body?: { marker?: string } } };
  }).buildRequestFromStandard({});
  return result.ok ? result.value.body?.marker : undefined;
}
