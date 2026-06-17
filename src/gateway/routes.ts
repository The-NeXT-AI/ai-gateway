import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import type { GatewayConfig, ProviderConfig, VirtualModelProfileConfig } from '../types';
import { providerFromProviderType, readHeader } from '../utils';
import { createGatewayAuthPreHandler } from './auth';
import { handleOpenAIEmbeddingsRequest } from './embeddings';
import { handleOpenAIImageGenerationsRequest, handleOpenAIModerationsRequest } from './openai-json';
import { handleGatewayRequest, parseGeminiTail } from './handler';
import { createGatewayIdempotencyPreHandler } from './idempotency';
import type { GatewayRuntime } from './runtime';

type ModelListFormat = 'openai' | 'anthropic';

interface ModelListQuery {
  format?: string;
  protocol?: string;
}

interface ModelPathParams {
  model: string;
}

interface ModelWildcardParams {
  '*': string;
}

interface GatewayModelListEntry {
  id: string;
  displayName: string;
  ownedBy: string;
  created: number;
  createdAt: string;
}

interface BaseModelListEntry extends GatewayModelListEntry {
  providerName: string;
  modelName: string;
}

const unknownModelCreated = 0;
const unknownModelCreatedAt = '1970-01-01T00:00:00Z';

export function registerGatewayRoutes(
  fastify: FastifyInstance,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  const gatewayAuthPreHandler = createGatewayAuthPreHandler(config.auth);
  const gatewayIdempotencyPreHandler = createGatewayIdempotencyPreHandler(config);
  const gatewayWritePreHandlers = [gatewayAuthPreHandler, gatewayIdempotencyPreHandler];

  fastify.get<{ Querystring: ModelListQuery }>(
    '/v1/models',
    { preHandler: gatewayAuthPreHandler },
    async (request) => {
      const entries = buildGatewayModelListEntries(config);
      const format = resolveModelListFormat(request);

      if (format === 'anthropic') {
        return formatAnthropicModelList(entries);
      }

      return formatOpenAIModelList(entries);
    }
  );

  fastify.get<{ Params: ModelPathParams }>(
    '/v1/models/:model',
    { preHandler: gatewayAuthPreHandler },
    async (request, reply) => {
      return handleGetGatewayModel(request.params.model, reply, config);
    }
  );

  fastify.get<{ Params: ModelWildcardParams }>(
    '/v1/models/*',
    { preHandler: gatewayAuthPreHandler },
    async (request, reply) => {
      return handleGetGatewayModel(request.params['*'], reply, config);
    }
  );

  fastify.post('/v1/chat/completions', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleGatewayRequest(
      request,
      reply,
      {
        adapterKey: 'openai_chat'
      },
      config,
      runtime
    );
  });

  fastify.post('/v1/responses', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleGatewayRequest(
      request,
      reply,
      {
        adapterKey: 'openai_responses'
      },
      config,
      runtime
    );
  });

  fastify.post('/v1/embeddings', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleOpenAIEmbeddingsRequest(request, reply, config, runtime);
  });

  fastify.post('/v1/moderations', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleOpenAIModerationsRequest(request, reply, config, runtime);
  });

  fastify.post('/v1/images/generations', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleOpenAIImageGenerationsRequest(request, reply, config, runtime);
  });

  fastify.post('/v1/messages', { preHandler: gatewayWritePreHandlers }, async (request, reply) => {
    return handleGatewayRequest(
      request,
      reply,
      {
        adapterKey: 'anthropic_messages'
      },
      config,
      runtime
    );
  });

  fastify.post<{ Params: { '*': string } }>(
    '/v1beta/models/*',
    { preHandler: gatewayWritePreHandlers },
    async (request, reply) => {
      return handleGeminiRequest(request, reply, 'v1beta', config, runtime);
    }
  );

  fastify.post<{ Params: { '*': string } }>(
    '/v1/models/*',
    { preHandler: gatewayWritePreHandlers },
    async (request, reply) => {
      return handleGeminiRequest(request, reply, 'v1', config, runtime);
    }
  );
}

function handleGetGatewayModel(rawModelId: string, reply: FastifyReply, config: GatewayConfig) {
  const modelId = decodeModelPathParam(rawModelId);
  const entry = buildGatewayModelListEntries(config).find((item) => item.id === modelId);
  if (!entry) {
    return reply.code(404).send({
      error: {
        message: `Model not found: ${modelId}`,
        type: 'invalid_request_error',
        code: 'model_not_found'
      }
    });
  }

  return formatOpenAIModelEntry(entry);
}

function resolveModelListFormat(
  request: FastifyRequest<{ Querystring: ModelListQuery }>
): ModelListFormat {
  const explicitFormat =
    parseModelListFormat(request.query?.format) ||
    parseModelListFormat(request.query?.protocol) ||
    parseModelListFormat(readHeader(request.headers['x-gateway-model-list-format']));

  if (explicitFormat) {
    return explicitFormat;
  }

  if (
    readHeader(request.headers['anthropic-version']) ||
    readHeader(request.headers['anthropic-beta'])
  ) {
    return 'anthropic';
  }

  return 'openai';
}

function parseModelListFormat(value: string | undefined): ModelListFormat | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === 'anthropic' || normalized === 'claude') {
    return 'anthropic';
  }

  if (normalized === 'openai') {
    return 'openai';
  }

  return undefined;
}

function formatOpenAIModelList(entries: GatewayModelListEntry[]) {
  return {
    object: 'list',
    data: entries.map(formatOpenAIModelEntry)
  };
}

function formatOpenAIModelEntry(entry: GatewayModelListEntry) {
  return {
    id: entry.id,
    object: 'model',
    created: entry.created,
    owned_by: entry.ownedBy
  };
}

function formatAnthropicModelList(entries: GatewayModelListEntry[]) {
  return {
    data: entries.map((entry) => ({
      created_at: entry.createdAt,
      display_name: entry.displayName,
      id: entry.id,
      type: 'model'
    })),
    first_id: entries[0]?.id ?? null,
    has_more: false,
    last_id: entries.at(-1)?.id ?? null
  };
}

function buildGatewayModelListEntries(config: GatewayConfig): GatewayModelListEntry[] {
  const seen = new Set<string>();
  const entries: GatewayModelListEntry[] = [];
  const baseEntries: BaseModelListEntry[] = [];
  const bareModelIds = config.modelList?.bareModelIds === true;

  const pushEntry = (entry: GatewayModelListEntry): void => {
    if (!entry.id || seen.has(entry.id)) {
      return;
    }

    seen.add(entry.id);
    entries.push(entry);
  };

  for (const providerConfig of config.providers) {
    const ownedBy = resolveProviderOwner(providerConfig);
    for (const rawModelName of providerConfig.models) {
      const modelName = rawModelName.trim();
      if (!modelName) {
        continue;
      }

      const id = buildBaseModelListId(providerConfig.name, modelName, bareModelIds);
      const entry: BaseModelListEntry = {
        id,
        displayName: modelName,
        ownedBy,
        created: unknownModelCreated,
        createdAt: unknownModelCreatedAt,
        providerName: providerConfig.name,
        modelName
      };
      baseEntries.push(entry);
      pushEntry(entry);
    }
  }

  for (const entry of materializeVirtualModelListEntries(config, baseEntries)) {
    pushEntry(entry);
  }

  return entries;
}

function materializeVirtualModelListEntries(
  config: GatewayConfig,
  baseEntries: BaseModelListEntry[]
): GatewayModelListEntry[] {
  const entries: GatewayModelListEntry[] = [];
  const configuredProviderNames = new Set(baseEntries.map((entry) => entry.providerName));
  const bareModelIds = config.modelList?.bareModelIds === true;

  for (const profile of config.virtualModelProfiles || []) {
    if (!shouldMaterializeVirtualModel(profile)) {
      continue;
    }

    for (const baseEntry of baseEntries) {
      for (const prefix of profile.match.prefixes) {
        const id = buildPrefixedVirtualModelListId(baseEntry, prefix, bareModelIds);
        entries.push(createVirtualModelListEntry(profile, id, baseEntry.id, baseEntry.ownedBy));
      }

      for (const suffix of profile.match.suffixes) {
        const id = buildSuffixedVirtualModelListId(baseEntry, suffix, bareModelIds);
        entries.push(createVirtualModelListEntry(profile, id, baseEntry.id, baseEntry.ownedBy));
      }
    }

    if (!profile.baseModel?.fixedModel) {
      continue;
    }

    for (const alias of profile.match.exactAliases) {
      const id = resolveExactVirtualModelAlias(alias, profile.baseModel.fixedModel, bareModelIds);
      if (!id) {
        continue;
      }

      const owner = extractProviderName(id) || extractProviderName(profile.baseModel.fixedModel);
      if (!owner || !configuredProviderNames.has(owner)) {
        continue;
      }

      entries.push(createVirtualModelListEntry(profile, id, profile.baseModel.fixedModel, owner));
    }
  }

  return entries;
}

function buildBaseModelListId(providerName: string, modelName: string, bareModelIds: boolean): string {
  return bareModelIds ? modelName : `${providerName}/${modelName}`;
}

function buildPrefixedVirtualModelListId(
  baseEntry: BaseModelListEntry,
  prefix: string,
  bareModelIds: boolean
): string {
  return bareModelIds
    ? `${prefix}${baseEntry.modelName}`
    : `${baseEntry.providerName}/${prefix}${baseEntry.modelName}`;
}

function buildSuffixedVirtualModelListId(
  baseEntry: BaseModelListEntry,
  suffix: string,
  bareModelIds: boolean
): string {
  return bareModelIds
    ? `${baseEntry.modelName}${suffix}`
    : `${baseEntry.providerName}/${baseEntry.modelName}${suffix}`;
}

function shouldMaterializeVirtualModel(profile: VirtualModelProfileConfig): boolean {
  return (
    profile.enabled !== false &&
    profile.materialization.enabled !== false &&
    profile.materialization.includeInGatewayModels !== false
  );
}

function createVirtualModelListEntry(
  profile: VirtualModelProfileConfig,
  id: string,
  baseModelId: string,
  ownedBy: string | undefined
): GatewayModelListEntry {
  return {
    id,
    displayName: renderVirtualModelDisplayName(profile, id, baseModelId),
    ownedBy: ownedBy || extractProviderName(id) || 'gateway',
    created: unknownModelCreated,
    createdAt: unknownModelCreatedAt
  };
}

function renderVirtualModelDisplayName(
  profile: VirtualModelProfileConfig,
  aliasModelId: string,
  baseModelId: string
): string {
  const template = profile.materialization.displayNameTemplate;
  if (template) {
    return template
      .replaceAll('{alias}', aliasModelId)
      .replaceAll('{baseModel}', baseModelId)
      .replaceAll('{profileKey}', profile.key)
      .replaceAll('{profileDisplayName}', profile.displayName);
  }

  return extractModelName(aliasModelId) || aliasModelId;
}

function resolveProviderOwner(providerConfig: ProviderConfig): string {
  return providerConfig.name || providerFromProviderType(providerConfig.type);
}

function resolveExactVirtualModelAlias(
  alias: string,
  fixedModelId: string,
  bareModelIds: boolean
): string | undefined {
  const normalizedAlias = alias.trim();
  if (!normalizedAlias) {
    return undefined;
  }

  if (bareModelIds || normalizedAlias.includes('/')) {
    return normalizedAlias;
  }

  const providerName = extractProviderName(fixedModelId);
  return providerName ? `${providerName}/${normalizedAlias}` : undefined;
}

function extractProviderName(modelId: string | undefined): string | undefined {
  const normalized = modelId?.trim();
  if (!normalized) {
    return undefined;
  }

  const slashIndex = normalized.indexOf('/');
  if (slashIndex <= 0 || slashIndex >= normalized.length - 1) {
    return undefined;
  }

  return normalized.slice(0, slashIndex);
}

function extractModelName(modelId: string | undefined): string | undefined {
  const normalized = modelId?.trim();
  if (!normalized) {
    return undefined;
  }

  const slashIndex = normalized.indexOf('/');
  if (slashIndex <= 0 || slashIndex >= normalized.length - 1) {
    return normalized;
  }

  return normalized.slice(slashIndex + 1);
}

function decodeModelPathParam(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

async function handleGeminiRequest(
  request: FastifyRequest<{ Params: { '*': string } }>,
  reply: FastifyReply,
  apiVersion: string,
  config: GatewayConfig,
  runtime: GatewayRuntime
) {
  const tail = String(request.params['*'] || '');
  const parsed = parseGeminiTail(tail);

  if (!parsed) {
    return reply.code(400).send({
      error: {
        message: 'Invalid Gemini route. Expected /models/{model}:generateContent'
      }
    });
  }

  return handleGatewayRequest(
    request,
    reply,
    {
      adapterKey: parsed.action === 'streamGenerateContent' ? 'gemini_stream' : 'gemini_generate',
      metadata: {
        model: parsed.model,
        action: parsed.action,
        apiVersion
      }
    },
    config,
    runtime
  );
}
