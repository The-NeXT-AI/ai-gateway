import { createDecipheriv } from 'node:crypto';
import {
  asString,
  findDefaultProviderConfig,
  isMediaOnlyProviderType,
  isObject,
  parseProvider,
  parseProviderList,
  providerFromProviderType
} from '../utils';
import {
  parseGatewayPluginsFromRaw,
  parseProviderPluginsFromRaw,
  parseProvidersFromRaw,
  parseVirtualModelProfilesFromRaw
} from '../config';
import type {
  GatewayConfig,
  GatewayPluginConfig,
  Provider,
  ProviderConfig,
  ProviderExternalSourceConfig,
  ProviderPluginConfig,
  VirtualModelProfileConfig
} from '../types';
import { requestExternalJson } from '../external-json-source';
import { updateDistributedCredentialEncryption } from './plugins';

export interface ProviderExternalLogger {
  info?: (...args: unknown[]) => void;
  warn?: (...args: unknown[]) => void;
}

interface ExternalProviderSnapshot {
  providers: ProviderConfig[];
  plugins?: GatewayPluginConfig[];
  pluginsProvided: boolean;
  providerPlugins?: ProviderPluginConfig[];
  providerPluginsProvided: boolean;
  virtualModelProfiles?: VirtualModelProfileConfig[];
  virtualModelProfilesProvided: boolean;
  credentialEncryption?: ExternalCredentialEncryptionConfig;
  credentialEncryptionProvided: boolean;
}

interface ExternalCredentialEncryptionConfig {
  key?: string;
  keyVersion?: string;
  algorithm?: string;
}

const DEFAULT_PROVIDER_TIMEOUT_MS = 5000;
const OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com';
const GEMINI_DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com';
const EXTERNAL_CREDENTIAL_ENCRYPTED_PREFIX = 'enc:v1:';
const EXTERNAL_CREDENTIAL_ENCRYPTION_ALGORITHM = 'aes-256-gcm';

export function isProviderExternalSourceEnabled(config: GatewayConfig): boolean {
  return Boolean(config.providerExternal?.enabled);
}

export async function hydrateProvidersFromExternalSource(
  config: GatewayConfig,
  logger?: ProviderExternalLogger
): Promise<void> {
  const source = config.providerExternal;
  if (!source?.enabled) {
    return;
  }

  const snapshot = await fetchExternalProviders(source, logger);
  applyProvidersSnapshot(
    config,
    snapshot.providers,
    snapshot.plugins,
    snapshot.pluginsProvided,
    snapshot.providerPlugins,
    snapshot.providerPluginsProvided,
    snapshot.virtualModelProfiles,
    snapshot.virtualModelProfilesProvided
  );
  updateDistributedCredentialEncryption(
    snapshot.credentialEncryptionProvided ? snapshot.credentialEncryption : undefined
  );

  logger?.info?.(
    {
      endpoint: source.endpoint,
      command: source.command,
      transport: source.transport,
      providers: snapshot.providers.length,
      plugins: snapshot.pluginsProvided
        ? snapshot.plugins?.length || 0
        : undefined,
      providerPlugins: snapshot.providerPluginsProvided
        ? snapshot.providerPlugins?.length || 0
        : undefined,
      virtualModelProfiles: snapshot.virtualModelProfilesProvided
        ? snapshot.virtualModelProfiles?.length || 0
        : undefined,
      credentialEncryption: snapshot.credentialEncryptionProvided
    },
    'Loaded provider config from external endpoint.'
  );
}

export function applyProvidersSnapshot(
  config: GatewayConfig,
  providers: ProviderConfig[],
  plugins?: GatewayPluginConfig[],
  pluginsProvided = false,
  providerPlugins?: ProviderPluginConfig[],
  providerPluginsProvided = false,
  virtualModelProfiles?: VirtualModelProfileConfig[],
  virtualModelProfilesProvided = false
): void {
  applyProvidersToGatewayConfig(config, providers);
  if (pluginsProvided) {
    config.plugins = [...(plugins || [])];
  }
  if (providerPluginsProvided) {
    config.providerPlugins = [...(providerPlugins || [])];
  }
  if (virtualModelProfilesProvided) {
    config.virtualModelProfiles = [...(virtualModelProfiles || [])];
  }
}

async function fetchExternalProviders(
  source: ProviderExternalSourceConfig,
  logger?: ProviderExternalLogger
): Promise<ExternalProviderSnapshot> {
  const endpoint = normalizeOptionalString(source.endpoint);
  if (source.transport !== 'stdio' && !endpoint) {
    throw new Error('provider.external.endpoint is required when provider.external.enabled=true.');
  }
  if (source.transport === 'stdio' && !normalizeOptionalString(source.command)) {
    throw new Error('provider.external.command is required when provider.external.transport=stdio.');
  }

  const timeoutMs =
    Number.isFinite(source.timeoutMs) && source.timeoutMs > 0
      ? Math.floor(source.timeoutMs)
      : DEFAULT_PROVIDER_TIMEOUT_MS;
  try {
    const payload = await requestExternalJson(source, {
      label: 'provider.external',
      httpMethod: 'GET',
      payload: {
        type: 'provider_config_request'
      },
      grpcDefaultPath: '/gateway.providers.v1.ProviderSource/GetProviders'
    });

    const snapshotPayload = extractProvidersPayload(payload);
    const providersRaw = snapshotPayload.providers;
    const providers = parseProvidersFromRaw(providersRaw);
    if (providersRaw.length > 0 && providers.length === 0) {
      throw new Error('External provider endpoint payload does not contain valid provider items.');
    }

    let providerPlugins: ProviderPluginConfig[] | undefined;
    let plugins: GatewayPluginConfig[] | undefined;
    if (snapshotPayload.pluginsProvided) {
      if (!Array.isArray(snapshotPayload.plugins)) {
        throw new Error(
          'External provider endpoint payload field "plugins" must be an array when present.'
        );
      }

      plugins = parseGatewayPluginsFromRaw(snapshotPayload.plugins);
      if (snapshotPayload.plugins.length > 0 && plugins.length === 0) {
        throw new Error('External provider endpoint payload does not contain valid plugin items.');
      }
    }

    if (snapshotPayload.providerPluginsProvided) {
      if (!Array.isArray(snapshotPayload.providerPlugins)) {
        throw new Error(
          'External provider endpoint payload field "providerPlugins" must be an array when present.'
        );
      }

      providerPlugins = parseProviderPluginsFromRaw(snapshotPayload.providerPlugins);
      if (snapshotPayload.providerPlugins.length > 0 && providerPlugins.length === 0) {
        throw new Error('External provider endpoint payload does not contain valid provider plugin items.');
      }
    }

    let virtualModelProfiles: VirtualModelProfileConfig[] | undefined;
    if (snapshotPayload.virtualModelProfilesProvided) {
      if (!Array.isArray(snapshotPayload.virtualModelProfiles)) {
        throw new Error(
          'External provider endpoint payload field "virtualModelProfiles" must be an array when present.'
        );
      }

      virtualModelProfiles = parseVirtualModelProfilesFromRaw(snapshotPayload.virtualModelProfiles);
      if (
        snapshotPayload.virtualModelProfiles.length > 0 &&
        virtualModelProfiles.length === 0
      ) {
        throw new Error(
          'External provider endpoint payload does not contain valid virtual model profile items.'
        );
      }
    }

    let credentialEncryption: ExternalCredentialEncryptionConfig | undefined;
    if (snapshotPayload.credentialEncryptionProvided) {
      if (
        snapshotPayload.credentialEncryption !== undefined &&
        snapshotPayload.credentialEncryption !== null &&
        !isObject(snapshotPayload.credentialEncryption)
      ) {
        throw new Error(
          'External provider endpoint payload field "credentialEncryption" must be an object when present.'
        );
      }

      credentialEncryption = normalizeCredentialEncryptionPayload(
        snapshotPayload.credentialEncryption
      );
    }
    const decryptedProviders = decryptProviderCredentials(
      providers,
      credentialEncryption
    );

    return {
      providers: decryptedProviders,
      plugins,
      pluginsProvided: snapshotPayload.pluginsProvided,
      providerPlugins,
      providerPluginsProvided: snapshotPayload.providerPluginsProvided,
      virtualModelProfiles,
      virtualModelProfilesProvided: snapshotPayload.virtualModelProfilesProvided,
      credentialEncryption,
      credentialEncryptionProvided: snapshotPayload.credentialEncryptionProvided
    };
  } catch (error) {
    if (isTimeoutError(error)) {
      logger?.warn?.(
        {
          endpoint,
          command: source.command,
          transport: source.transport,
          timeoutMs
        },
        'External provider endpoint request timed out.'
      );
    }

    throw error;
  }
}

function applyProvidersToGatewayConfig(config: GatewayConfig, providers: ProviderConfig[]): void {
  const previousProviders = [...config.providers];
  const previousDefaultTargets = defaultProviderTypes(previousProviders);
  const previousDefaultTarget = previousDefaultTargets[0];

  config.providers.length = 0;
  for (const provider of providers) {
    config.providers.push(provider);
  }

  const nextDefaultTargets = defaultProviderTypes(providers);
  const envDefaultTargets = parseProviderList(process.env.DEFAULT_TARGET_PROVIDERS);
  const hasExplicitDefaultTargets = envDefaultTargets.length > 0;
  if (!hasExplicitDefaultTargets) {
    if (
      config.defaultTargetProviders.length === 0 ||
      areProviderListsEqual(config.defaultTargetProviders, previousDefaultTargets)
    ) {
      config.defaultTargetProviders = nextDefaultTargets;
    }
  }

  const envDefaultTarget = parseProvider(process.env.DEFAULT_TARGET_PROVIDER);
  if (!envDefaultTarget) {
    if (!config.defaultTargetProvider || config.defaultTargetProvider === previousDefaultTarget) {
      config.defaultTargetProvider = config.defaultTargetProviders[0];
    }
  }

  syncProviderDerivedFields(config, previousProviders, providers);
}

function syncProviderDerivedFields(
  config: GatewayConfig,
  previousProviders: ProviderConfig[],
  nextProviders: ProviderConfig[]
): void {
  const previousOpenAI = findProviderByType(previousProviders, 'openai');
  const previousAnthropic = findProviderByType(previousProviders, 'anthropic');
  const previousGemini = findProviderByType(previousProviders, 'gemini');
  const nextOpenAI = findProviderByType(nextProviders, 'openai');
  const nextAnthropic = findProviderByType(nextProviders, 'anthropic');
  const nextGemini = findProviderByType(nextProviders, 'gemini');

  config.openaiApiKey = deriveProviderCredentialValue(
    process.env.OPENAI_API_KEY,
    config.openaiApiKey,
    previousOpenAI?.apikey,
    nextOpenAI?.apikey
  );
  config.anthropicApiKey = deriveProviderCredentialValue(
    process.env.ANTHROPIC_API_KEY,
    config.anthropicApiKey,
    previousAnthropic?.apikey,
    nextAnthropic?.apikey
  );
  config.geminiApiKey = deriveProviderCredentialValue(
    process.env.GEMINI_API_KEY,
    config.geminiApiKey,
    previousGemini?.apikey,
    nextGemini?.apikey
  );

  config.openaiBaseUrl = deriveProviderBaseUrlValue(
    process.env.OPENAI_BASE_URL,
    config.openaiBaseUrl,
    previousOpenAI?.baseurl,
    nextOpenAI?.baseurl,
    OPENAI_DEFAULT_BASE_URL
  );
  config.anthropicBaseUrl = deriveProviderBaseUrlValue(
    process.env.ANTHROPIC_BASE_URL,
    config.anthropicBaseUrl,
    previousAnthropic?.baseurl,
    nextAnthropic?.baseurl,
    ANTHROPIC_DEFAULT_BASE_URL
  );
  config.geminiBaseUrl = deriveProviderBaseUrlValue(
    process.env.GEMINI_BASE_URL,
    config.geminiBaseUrl,
    previousGemini?.baseurl,
    nextGemini?.baseurl,
    GEMINI_DEFAULT_BASE_URL
  );

  config.defaultOpenAIModel = deriveProviderDefaultModelValue(
    process.env.DEFAULT_OPENAI_MODEL,
    config.defaultOpenAIModel,
    previousOpenAI?.models[0],
    nextOpenAI?.models[0]
  );
  config.defaultAnthropicModel = deriveProviderDefaultModelValue(
    process.env.DEFAULT_ANTHROPIC_MODEL,
    config.defaultAnthropicModel,
    previousAnthropic?.models[0],
    nextAnthropic?.models[0]
  );
  config.defaultGeminiModel = deriveProviderDefaultModelValue(
    process.env.DEFAULT_GEMINI_MODEL,
    config.defaultGeminiModel,
    previousGemini?.models[0],
    nextGemini?.models[0]
  );
}

function deriveProviderCredentialValue(
  envValue: string | undefined,
  current: string | undefined,
  previousDerived: string | undefined,
  nextDerived: string | undefined
): string | undefined {
  if (normalizeOptionalString(envValue)) {
    return current;
  }

  if (!current) {
    return nextDerived;
  }

  if (previousDerived && current === previousDerived) {
    return nextDerived || current;
  }

  return current;
}

function deriveProviderBaseUrlValue(
  envValue: string | undefined,
  current: string,
  previousDerived: string | undefined,
  nextDerived: string | undefined,
  fallbackDefault: string
): string {
  if (normalizeOptionalString(envValue)) {
    return current;
  }

  if (!nextDerived) {
    return current;
  }

  if (previousDerived && current === previousDerived) {
    return nextDerived;
  }

  if (!previousDerived && current === fallbackDefault) {
    return nextDerived;
  }

  return current;
}

function deriveProviderDefaultModelValue(
  envValue: string | undefined,
  current: string | undefined,
  previousDerived: string | undefined,
  nextDerived: string | undefined
): string | undefined {
  if (normalizeOptionalString(envValue)) {
    return current;
  }

  if (!current) {
    return nextDerived;
  }

  if (previousDerived && current === previousDerived) {
    return nextDerived || current;
  }

  return current;
}

function findProviderByType(
  providers: ProviderConfig[],
  provider: Provider
): ProviderConfig | undefined {
  return findDefaultProviderConfig(providers, provider);
}

function defaultProviderTypes(providers: ProviderConfig[]): Provider[] {
  return dedupeProviderTypes(
    providers
      .filter((item) => !isMediaOnlyProviderType(item.type))
      .map((item) => providerFromProviderType(item.type))
  );
}

function dedupeProviderTypes(value: Provider[]): Provider[] {
  const deduped: Provider[] = [];
  for (const provider of value) {
    if (!deduped.includes(provider)) {
      deduped.push(provider);
    }
  }
  return deduped;
}

function areProviderListsEqual(a: Provider[], b: Provider[]): boolean {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

function extractProvidersPayload(payload: unknown): {
  providers: unknown[];
  plugins: unknown;
  pluginsProvided: boolean;
  providerPlugins: unknown;
  providerPluginsProvided: boolean;
  virtualModelProfiles: unknown;
  virtualModelProfilesProvided: boolean;
  credentialEncryption: unknown;
  credentialEncryptionProvided: boolean;
} {
  if (Array.isArray(payload)) {
    return {
      providers: payload,
      plugins: undefined,
      pluginsProvided: false,
      providerPlugins: undefined,
      providerPluginsProvided: false,
      virtualModelProfiles: undefined,
      virtualModelProfilesProvided: false,
      credentialEncryption: undefined,
      credentialEncryptionProvided: false
    };
  }

  if (!isObject(payload)) {
    throw new Error('External provider endpoint payload must be an array or object.');
  }

  const pluginsResolution = resolvePluginsPayload(payload);
  const providerPluginsResolution = resolveProviderPluginsPayload(payload);
  const virtualModelProfilesResolution = resolveVirtualModelProfilesPayload(payload);
  const credentialEncryptionResolution = resolveCredentialEncryptionPayload(payload);
  const payloadData = isObject(payload.data) ? payload.data : undefined;

  if (Array.isArray(payload.providers)) {
    return {
      providers: payload.providers,
      plugins: pluginsResolution.value,
      pluginsProvided: pluginsResolution.provided,
      providerPlugins: providerPluginsResolution.value,
      providerPluginsProvided: providerPluginsResolution.provided,
      virtualModelProfiles: virtualModelProfilesResolution.value,
      virtualModelProfilesProvided: virtualModelProfilesResolution.provided,
      credentialEncryption: credentialEncryptionResolution.value,
      credentialEncryptionProvided: credentialEncryptionResolution.provided
    };
  }

  if (Array.isArray(payload.Providers)) {
    return {
      providers: payload.Providers,
      plugins: pluginsResolution.value,
      pluginsProvided: pluginsResolution.provided,
      providerPlugins: providerPluginsResolution.value,
      providerPluginsProvided: providerPluginsResolution.provided,
      virtualModelProfiles: virtualModelProfilesResolution.value,
      virtualModelProfilesProvided: virtualModelProfilesResolution.provided,
      credentialEncryption: credentialEncryptionResolution.value,
      credentialEncryptionProvided: credentialEncryptionResolution.provided
    };
  }

  if (payloadData) {
    if (Array.isArray(payloadData.providers)) {
      return {
        providers: payloadData.providers,
        plugins: pluginsResolution.value,
        pluginsProvided: pluginsResolution.provided,
        providerPlugins: providerPluginsResolution.value,
        providerPluginsProvided: providerPluginsResolution.provided,
        virtualModelProfiles: virtualModelProfilesResolution.value,
        virtualModelProfilesProvided: virtualModelProfilesResolution.provided,
        credentialEncryption: credentialEncryptionResolution.value,
        credentialEncryptionProvided: credentialEncryptionResolution.provided
      };
    }

    if (Array.isArray(payloadData.Providers)) {
      return {
        providers: payloadData.Providers,
        plugins: pluginsResolution.value,
        pluginsProvided: pluginsResolution.provided,
        providerPlugins: providerPluginsResolution.value,
        providerPluginsProvided: providerPluginsResolution.provided,
        virtualModelProfiles: virtualModelProfilesResolution.value,
        virtualModelProfilesProvided: virtualModelProfilesResolution.provided,
        credentialEncryption: credentialEncryptionResolution.value,
        credentialEncryptionProvided: credentialEncryptionResolution.provided
      };
    }
  }

  throw new Error('External provider endpoint payload must include providers array.');
}

function resolvePluginsPayload(payload: Record<string, unknown>): {
  provided: boolean;
  value: unknown;
} {
  if (Object.prototype.hasOwnProperty.call(payload, 'plugins')) {
    return {
      provided: true,
      value: payload.plugins
    };
  }

  if (Object.prototype.hasOwnProperty.call(payload, 'Plugins')) {
    return {
      provided: true,
      value: payload.Plugins
    };
  }

  if (isObject(payload.data)) {
    if (Object.prototype.hasOwnProperty.call(payload.data, 'plugins')) {
      return {
        provided: true,
        value: payload.data.plugins
      };
    }

    if (Object.prototype.hasOwnProperty.call(payload.data, 'Plugins')) {
      return {
        provided: true,
        value: payload.data.Plugins
      };
    }
  }

  return {
    provided: false,
    value: undefined
  };
}

function resolveVirtualModelProfilesPayload(payload: Record<string, unknown>): {
  provided: boolean;
  value: unknown;
} {
  if (Object.prototype.hasOwnProperty.call(payload, 'virtualModelProfiles')) {
    return {
      provided: true,
      value: payload.virtualModelProfiles
    };
  }

  if (isObject(payload.data) && Object.prototype.hasOwnProperty.call(payload.data, 'virtualModelProfiles')) {
    return {
      provided: true,
      value: (payload.data as Record<string, unknown>).virtualModelProfiles
    };
  }

  return {
    provided: false,
    value: undefined
  };
}

function resolveCredentialEncryptionPayload(payload: Record<string, unknown>): {
  provided: boolean;
  value: unknown;
} {
  if (Object.prototype.hasOwnProperty.call(payload, 'credentialEncryption')) {
    return {
      provided: true,
      value: payload.credentialEncryption
    };
  }

  if (Object.prototype.hasOwnProperty.call(payload, 'CredentialEncryption')) {
    return {
      provided: true,
      value: payload.CredentialEncryption
    };
  }

  if (isObject(payload.data)) {
    if (Object.prototype.hasOwnProperty.call(payload.data, 'credentialEncryption')) {
      return {
        provided: true,
        value: payload.data.credentialEncryption
      };
    }

    if (Object.prototype.hasOwnProperty.call(payload.data, 'CredentialEncryption')) {
      return {
        provided: true,
        value: payload.data.CredentialEncryption
      };
    }
  }

  return {
    provided: false,
    value: undefined
  };
}

function normalizeCredentialEncryptionPayload(value: unknown): ExternalCredentialEncryptionConfig | undefined {
  if (!isObject(value)) {
    return undefined;
  }

  const key = normalizeOptionalString(value.key);
  const keyVersion = normalizeOptionalString(value.keyVersion ?? value.key_version);
  const algorithm = normalizeOptionalString(value.algorithm);

  if (!key && !keyVersion && !algorithm) {
    return undefined;
  }

  return {
    key,
    keyVersion,
    algorithm
  };
}

function decryptProviderCredentials(
  providers: ProviderConfig[],
  credentialEncryption?: ExternalCredentialEncryptionConfig
): ProviderConfig[] {
  return providers.map((provider) => {
    const apikey = decryptExternalCredentialIfNeeded(
      provider.apikey,
      credentialEncryption,
      `providers.${provider.name}.apikey`
    );
    const credentials = provider.credentials?.map((credential) => ({
      ...credential,
      apikey: decryptExternalCredentialIfNeeded(
        credential.apikey,
        credentialEncryption,
        `providers.${provider.name}.credentials.${credential.id}.apikey`
      )
    }));

    return {
      ...provider,
      apikey,
      credentials
    };
  });
}

function decryptExternalCredentialIfNeeded(
  value: string | undefined,
  credentialEncryption: ExternalCredentialEncryptionConfig | undefined,
  fieldName: string
): string | undefined {
  if (!value || !value.startsWith(EXTERNAL_CREDENTIAL_ENCRYPTED_PREFIX)) {
    return value;
  }

  const key = resolveExternalCredentialEncryptionKey(credentialEncryption);
  if (!key) {
    throw new Error(`${fieldName} is encrypted but no credential encryption key is configured.`);
  }

  const encodedEnvelope = value.slice(EXTERNAL_CREDENTIAL_ENCRYPTED_PREFIX.length);
  if (!encodedEnvelope) {
    throw new Error(`${fieldName} encrypted credential payload is empty.`);
  }

  let envelope: {
    iv?: unknown;
    tag?: unknown;
    data?: unknown;
    keyVersion?: unknown;
  };
  try {
    envelope = JSON.parse(Buffer.from(encodedEnvelope, 'base64url').toString('utf8'));
  } catch {
    throw new Error(`${fieldName} encrypted credential payload is malformed.`);
  }

  const iv = normalizeOptionalString(envelope.iv);
  const tag = normalizeOptionalString(envelope.tag);
  const data = normalizeOptionalString(envelope.data);
  if (!iv || !tag || !data) {
    throw new Error(`${fieldName} encrypted credential payload is incomplete.`);
  }

  const keyVersion = normalizeOptionalString(envelope.keyVersion);
  const expectedKeyVersion =
    normalizeOptionalString(credentialEncryption?.keyVersion) ||
    normalizeOptionalString(process.env.GATEWAY_CREDENTIAL_ENCRYPTION_KEY_VERSION);
  if (keyVersion && expectedKeyVersion && keyVersion !== expectedKeyVersion) {
    throw new Error(
      `${fieldName} encrypted credential key version mismatch: expected ${expectedKeyVersion}, got ${keyVersion}.`
    );
  }

  try {
    const decipher = createDecipheriv(
      EXTERNAL_CREDENTIAL_ENCRYPTION_ALGORITHM,
      key,
      Buffer.from(iv, 'base64')
    );
    decipher.setAuthTag(Buffer.from(tag, 'base64'));
    const decrypted = Buffer.concat([
      decipher.update(Buffer.from(data, 'base64')),
      decipher.final()
    ]).toString('utf8');
    const normalized = normalizeOptionalString(decrypted);
    if (!normalized) {
      throw new Error(`${fieldName} decrypted credential is empty.`);
    }
    return normalized;
  } catch (error) {
    if (error instanceof Error && error.message.includes('decrypted credential is empty')) {
      throw error;
    }
    throw new Error(`${fieldName} encrypted credential decryption failed.`);
  }
}

function resolveExternalCredentialEncryptionKey(
  credentialEncryption?: ExternalCredentialEncryptionConfig
): Buffer | undefined {
  const configured =
    normalizeOptionalString(credentialEncryption?.key) ||
    normalizeOptionalString(process.env.GATEWAY_CREDENTIAL_ENCRYPTION_KEY);
  if (!configured) {
    return undefined;
  }

  return (
    tryDecodeExternalCredentialKey(configured, 'base64') ||
    tryDecodeExternalCredentialKey(configured, 'hex') ||
    tryDecodeExternalCredentialUtf8Key(configured)
  );
}

function tryDecodeExternalCredentialKey(value: string, encoding: 'base64' | 'hex'): Buffer | undefined {
  try {
    const decoded = Buffer.from(value, encoding);
    if (decoded.length !== 32) {
      return undefined;
    }
    if (encoding === 'base64' && decoded.toString('base64').replace(/=+$/g, '') !== value.replace(/=+$/g, '')) {
      return undefined;
    }
    if (encoding === 'hex' && decoded.toString('hex') !== value.toLowerCase()) {
      return undefined;
    }
    return decoded;
  } catch {
    return undefined;
  }
}

function tryDecodeExternalCredentialUtf8Key(value: string): Buffer | undefined {
  const key = Buffer.from(value, 'utf8');
  return key.length === 32 ? key : undefined;
}

function resolveProviderPluginsPayload(payload: Record<string, unknown>): {
  provided: boolean;
  value: unknown;
} {
  if (Object.prototype.hasOwnProperty.call(payload, 'providerPlugins')) {
    return {
      provided: true,
      value: payload.providerPlugins
    };
  }

  if (Object.prototype.hasOwnProperty.call(payload, 'ProviderPlugins')) {
    return {
      provided: true,
      value: payload.ProviderPlugins
    };
  }

  if (isObject(payload.data)) {
    if (Object.prototype.hasOwnProperty.call(payload.data, 'providerPlugins')) {
      return {
        provided: true,
        value: payload.data.providerPlugins
      };
    }

    if (Object.prototype.hasOwnProperty.call(payload.data, 'ProviderPlugins')) {
      return {
        provided: true,
        value: payload.data.ProviderPlugins
      };
    }
  }

  return {
    provided: false,
    value: undefined
  };
}

function summarizeErrorPayload(payload: unknown): string | undefined {
  if (payload === undefined || payload === null) {
    return undefined;
  }

  if (typeof payload === 'string') {
    const text = payload.trim();
    return text ? truncateText(text, 300) : undefined;
  }

  if (!isObject(payload)) {
    return undefined;
  }

  const directMessage =
    asString(payload.message) ||
    asString(payload.detail) ||
    asString(payload.error_description) ||
    asString(payload.reason);
  if (directMessage) {
    return truncateText(directMessage, 300);
  }

  if (typeof payload.error === 'string') {
    const text = payload.error.trim();
    return text ? truncateText(text, 300) : undefined;
  }

  if (isObject(payload.error)) {
    const errorMessage =
      asString(payload.error.message) || asString(payload.error.detail) || asString(payload.error.code);
    if (errorMessage) {
      return truncateText(errorMessage, 300);
    }
  }

  if (typeof payload.raw === 'string' && payload.raw.trim()) {
    return truncateText(payload.raw, 300);
  }

  return undefined;
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
}

function isTimeoutError(error: unknown): boolean {
  return error instanceof Error && /timed out|timeout/i.test(error.message);
}
