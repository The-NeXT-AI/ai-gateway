import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import type {
  GatewayConfig,
  NativeHistoryPolicy,
  ProviderNativeGroupState,
  ProviderNativeItem,
  ProviderNativeReplayDecision,
  Provider,
  ProviderConfig,
  ProviderPlugin,
  ProviderPluginCredentialScopeInput,
  ReasoningStateOrigin,
  Result,
  StandardRequest,
  StandardRequestInputContent,
  StandardResponse
} from '../types';
import { err, ok } from '../types';
import {
  ANTHROPIC_CLAUDE_REASONING_FORMAT,
  decodeReasoningTransportEnvelope,
  encodeReasoningTransportEnvelope,
  GEMINI_GENERATE_CONTENT_REASONING_FORMAT,
  GEMINI_INTERACTIONS_REASONING_FORMAT,
  isProviderNativePayloadStructurallyValid,
  OPENAI_RESPONSES_REASONING_FORMAT,
  validateReasoningTransportCarrierSseStream
} from '../adapters/builtins/reasoning-envelope';
import { parseSseChunks } from '../sse';
import { readBearerToken, readHeader } from '../utils';

export interface ReasoningStateCredentialContext extends ProviderPluginCredentialScopeInput {
  plugins: ProviderPlugin[];
}

export interface NativeStatePreparationOptions {
  historyPolicy?: NativeHistoryPolicy;
  explicitStrip?: boolean;
  plaintextReasoningSupported?: boolean;
  failover?: boolean;
}

export interface ProviderNativeGroupAnalysis {
  id: string;
  state: ProviderNativeGroupState;
  items: ProviderNativeItem[];
  active: boolean;
}

export interface ProviderNativeItemDecision {
  item: ProviderNativeItem;
  groupState: ProviderNativeGroupState;
  decision: ProviderNativeReplayDecision;
  reason?: string;
}

interface StatefulRouteRecord {
  format: string;
  origin: ReasoningStateOrigin;
  recordedAt: number;
}

const STATEFUL_ROUTE_CACHE_LIMIT = 10_000;
const STATEFUL_ROUTE_CACHE_TTL_MS = 60 * 60 * 1000;
const statefulRouteByResponseId = new Map<string, StatefulRouteRecord>();

export function buildReasoningStateOrigin(
  provider: Provider,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig,
  model: string | undefined,
  credentialContext?: ReasoningStateCredentialContext
): ReasoningStateOrigin {
  const providerFamily = resolveProviderFamily(provider, providerConfig);
  const baseUrl = resolveProviderBaseUrl(providerFamily, providerConfig, config);
  const normalizedEndpoint = normalizeReasoningEndpoint(baseUrl);
  const endpoint = createHash('sha256')
    .update(`${providerFamily}\0${normalizedEndpoint}`, 'utf8')
    .digest('base64url');
  const credentialScope = resolveReasoningCredentialScope(
    providerFamily,
    providerConfig,
    config,
    credentialContext
  );

  return {
    provider: providerFamily,
    endpoint,
    ...(model?.trim() ? { model: model.trim() } : {}),
    ...(credentialScope ? { credentialScope } : {})
  };
}

export function normalizeReasoningEndpoint(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return '';
  }

  try {
    const parsed = new URL(trimmed);
    parsed.protocol = parsed.protocol.toLowerCase();
    parsed.hostname = parsed.hostname.toLowerCase();
    parsed.username = '';
    parsed.password = '';
    parsed.search = '';
    parsed.hash = '';
    if (
      (parsed.protocol === 'https:' && parsed.port === '443') ||
      (parsed.protocol === 'http:' && parsed.port === '80')
    ) {
      parsed.port = '';
    }
    const path = parsed.pathname.replace(/\/+$/, '');
    return `${parsed.protocol}//${parsed.host}${path}`;
  } catch {
    return trimmed.replace(/\/+$/, '').toLowerCase();
  }
}

export function resolveReasoningTargetFormat(
  provider: Provider,
  providerConfig?: ProviderConfig
): string | undefined {
  switch (providerConfig?.type) {
    case 'openai_responses':
      return OPENAI_RESPONSES_REASONING_FORMAT;
    case 'anthropic_messages':
      return ANTHROPIC_CLAUDE_REASONING_FORMAT;
    case 'gemini_generate_content':
      return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
    case 'gemini_interactions':
      return GEMINI_INTERACTIONS_REASONING_FORMAT;
    case 'openai_chat_completions':
      return undefined;
  }

  if (provider === 'openai') {
    return OPENAI_RESPONSES_REASONING_FORMAT;
  }
  if (provider === 'anthropic') {
    return ANTHROPIC_CLAUDE_REASONING_FORMAT;
  }
  if (provider === 'gemini') {
    return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
  }
  return undefined;
}

export function attachReasoningStateOrigin(
  response: StandardResponse,
  origin: ReasoningStateOrigin,
  sourceFormat?: string
): StandardResponse {
  const resolvedSourceFormat = sourceFormat || inferFormatFromOrigin(origin);
  if (resolvedSourceFormat && response.id) {
    rememberStatefulRoute(response.id, resolvedSourceFormat, origin);
  }
  return {
    ...response,
    output: response.output.map((item) => {
      if (item.type === 'provider_native_item') {
        return {
          ...item,
          source_origin: origin,
          ...(resolvedSourceFormat ? { source_format: resolvedSourceFormat } : {})
        };
      }
      if (item.type === 'reasoning') {
        return {
          ...item,
          ...(hasOpaqueReasoningState(item) ? { source_origin: origin } : {}),
          ...(item.native_item
            ? {
                native_item: {
                  ...item.native_item,
                  source_origin: origin,
                  ...(resolvedSourceFormat ? { source_format: resolvedSourceFormat } : {})
                }
              }
            : {})
        };
      }
      if (item.type === 'function_call') {
        return {
          ...item,
          ...(item.thought_signature ? { thought_signature_origin: origin } : {}),
          ...(item.native_item
            ? {
                native_item: {
                  ...item.native_item,
                  source_origin: origin,
                  ...(resolvedSourceFormat ? { source_format: resolvedSourceFormat } : {})
                }
              }
            : {})
        };
      }
      if (item.type === 'message' && item.native_item) {
        return {
          ...item,
          native_item: {
            ...item.native_item,
            source_origin: origin,
            ...(resolvedSourceFormat ? { source_format: resolvedSourceFormat } : {})
          }
        };
      }
      return item;
    })
  };
}

export function prepareReasoningStateForTarget(
  request: StandardRequest,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin,
  options: NativeStatePreparationOptions = {}
): StandardRequest {
  const result = prepareReasoningStateForTargetResult(
    request,
    targetFormat,
    targetOrigin,
    options
  );
  if (!result.ok) {
    return { ...request, native_state_error: result.error };
  }
  return result.value;
}

export function prepareReasoningStateForTargetResult(
  request: StandardRequest,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin,
  options: NativeStatePreparationOptions = {}
): Result<StandardRequest> {
  if (typeof request.input === 'string') {
    return prepareStatefulRequestForTarget(request, targetFormat, targetOrigin, options);
  }

  const statefulResult = prepareStatefulRequestForTarget(request, targetFormat, targetOrigin, options);
  if (!statefulResult.ok) {
    return statefulResult;
  }
  request = statefulResult.value;
  if (typeof request.input === 'string') {
    return ok(request);
  }

  const groups = deriveProviderNativeGroups(request);
  const groupByItem = new Map<ProviderNativeItem, ProviderNativeGroupAnalysis>();
  for (const group of groups) {
    for (const item of group.items) {
      groupByItem.set(item, group);
    }
  }

  const decisions = new Map<ProviderNativeItem, ProviderNativeItemDecision>();
  const droppedGroupIds = new Set<string>();
  for (const group of groups) {
    if (group.state === 'orphaned') {
      if (group.active) {
        return err(`incompatible_active_orphaned_group: group=${group.id}`);
      }
      droppedGroupIds.add(group.id);
    }
    for (const item of group.items) {
      const decision = decideProviderNativeItemReplay(
        item,
        group.state,
        targetFormat,
        targetOrigin,
        options
      );
      decisions.set(item, decision);
      if (decision.decision === 'reject') {
        return err(`${decision.reason || 'incompatible_provider_native_state'}: group=${group.id}`);
      }
      if (decision.decision === 'drop_group') {
        droppedGroupIds.add(group.id);
      }
    }
  }

  const compactionResult = validateCompactionContinuity(
    request,
    groups,
    decisions,
    targetFormat,
    targetOrigin,
    options
  );
  if (!compactionResult.ok) {
    return compactionResult;
  }

  const preparedInput = request.input
    .map((message) => {
      const nativeItems = message.native_items?.filter((item) => {
        const group = groupByItem.get(item);
        if (group && droppedGroupIds.has(group.id)) {
          return false;
        }
        return decisions.get(item)?.decision === 'replay_native';
      });
      const content = message.content.flatMap<StandardRequestInputContent>((item) => {
        const nativeItem = nativeItemFromContent(item);
        if (nativeItem) {
          const group = groupByItem.get(nativeItem);
          if (group && droppedGroupIds.has(group.id)) {
            return [];
          }
          const nativeDecision = decisions.get(nativeItem);
          if (nativeDecision) {
            if (nativeDecision.decision === 'replay_native') {
              return [item];
            }
            if (nativeDecision.decision === 'emit_plaintext') {
              return projectReadableNativeItem(item, nativeItem);
            }
            if (
              nativeDecision.decision === 'strip_optional' &&
              item.type !== 'provider_native_item' &&
              item.type !== 'reasoning'
            ) {
              return [stripOptionalNativeProjection(item, targetFormat, targetOrigin)];
            }
            return [];
          }
        }

        if (item.type === 'provider_native_item') {
          return [];
        }
        if (item.type === 'tool_use') {
          if (!item.thought_signature) {
            return [item];
          }
          if (
            targetFormat &&
            canReplayReasoningState(
              item.thought_signature_format,
              item.thought_signature_origin,
              targetFormat,
              targetOrigin
            )
          ) {
            return [item];
          }
          const {
            thought_signature: _thoughtSignature,
            thought_signature_format: _thoughtSignatureFormat,
            thought_signature_origin: _thoughtSignatureOrigin,
            native_item: _nativeItem,
            ...toolUse
          } = item;
          return [toolUse];
        }

        if (item.type !== 'reasoning' || !hasOpaqueReasoningState(item)) {
          return [item];
        }
        if (
          targetFormat &&
          canReplayReasoningState(item.source_format, item.source_origin, targetFormat, targetOrigin)
        ) {
          return [item];
        }

        const plaintextSupported = options.plaintextReasoningSupported ??
          (targetFormat === OPENAI_RESPONSES_REASONING_FORMAT || !targetFormat);
        if (plaintextSupported && options.historyPolicy !== 'strip') {
          const readableReasoning = stripOpaqueReasoningState(item);
          return readableReasoning ? [readableReasoning] : [];
        }
        return [];
      });
      return {
        ...message,
        content,
        ...(nativeItems && nativeItems.length > 0 ? { native_items: nativeItems } : { native_items: undefined })
      };
    })
    .filter((message) => message.content.length > 0 || (message.native_items?.length || 0) > 0);

  return ok({
    ...request,
    input: preparedInput
  });
}

export function canReplayReasoningState(
  sourceFormat: string | undefined,
  sourceOrigin: ReasoningStateOrigin | undefined,
  targetFormat: string,
  targetOrigin: ReasoningStateOrigin
): boolean {
  if (
    sourceFormat !== targetFormat ||
    !sourceOrigin ||
    sourceOrigin.provider !== targetOrigin.provider ||
    sourceOrigin.endpoint !== targetOrigin.endpoint ||
    !sourceOrigin.credentialScope ||
    !targetOrigin.credentialScope ||
    sourceOrigin.credentialScope !== targetOrigin.credentialScope
  ) {
    return false;
  }

  if (sourceFormat === GEMINI_INTERACTIONS_REASONING_FORMAT) {
    // Interactions officially supports carrying thought steps across Gemini models.
    return true;
  }

  return Boolean(sourceOrigin.model && targetOrigin.model && sourceOrigin.model === targetOrigin.model);
}

export function deriveProviderNativeGroups(request: StandardRequest): ProviderNativeGroupAnalysis[] {
  if (typeof request.input === 'string') {
    return [];
  }

  const occurrences: Array<{ item: ProviderNativeItem; order: number; messageIndex: number }> = [];
  const seen = new Set<string>();
  let order = 0;
  for (let messageIndex = 0; messageIndex < request.input.length; messageIndex += 1) {
    const message = request.input[messageIndex];
    const candidates = [
      ...(message.native_items || []),
      ...message.content.flatMap((content) => {
        if (content.type === 'provider_native_item') {
          return [content];
        }
        const nativeItem = nativeItemFromContent(content);
        return nativeItem ? [nativeItem] : [];
      })
    ];
    for (const item of candidates) {
      const identity = providerNativeIdentity(item);
      if (seen.has(identity)) {
        continue;
      }
      seen.add(identity);
      occurrences.push({ item, order, messageIndex });
      order += 1;
    }
  }

  const parents = occurrences.map((_, index) => index);
  const find = (index: number): number => {
    let root = index;
    while (parents[root] !== root) {
      root = parents[root]!;
    }
    while (parents[index] !== index) {
      const next = parents[index]!;
      parents[index] = root;
      index = next;
    }
    return root;
  };
  const union = (left: number, right: number): void => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) {
      parents[rightRoot] = leftRoot;
    }
  };
  const relationOwners = new Map<string, number>();
  for (let index = 0; index < occurrences.length; index += 1) {
    const item = occurrences[index]!.item;
    for (const relationId of providerNativeRelationIds(item)) {
      const relationKey = `${providerNativeRelationNamespace(item)}\0${relationId}`;
      const owner = relationOwners.get(relationKey);
      if (owner === undefined) {
        relationOwners.set(relationKey, index);
      } else {
        union(owner, index);
      }
    }
  }

  const groupsByRoot = new Map<number, typeof occurrences>();
  for (let index = 0; index < occurrences.length; index += 1) {
    const root = find(index);
    const group = groupsByRoot.get(root) || [];
    group.push(occurrences[index]!);
    groupsByRoot.set(root, group);
  }

  const lastAssistantMessageIndex = request.input.reduce(
    (latest, message, index) => message.role === 'assistant' && message.content.length > 0 ? index : latest,
    -1
  );
  return [...groupsByRoot.values()].map((occurrencesInGroup) => {
    const items = occurrencesInGroup.map(({ item }) => item);
    const latestMessageIndex = Math.max(...occurrencesInGroup.map(({ messageIndex }) => messageIndex));
    const hasLaterAssistant = lastAssistantMessageIndex > latestMessageIndex;
    const { state, active } = deriveProviderNativeGroupState(items, hasLaterAssistant);
    const first = occurrencesInGroup[0]!;
    const id = deriveProviderNativeGroupId(first.item, first.order);
    return { id, state, items, active };
  });
}

export function decideProviderNativeItemReplay(
  item: ProviderNativeItem,
  groupState: ProviderNativeGroupState,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin,
  options: NativeStatePreparationOptions = {}
): ProviderNativeItemDecision {
  const dependent = isDependencyCriticalNativeItem(item);
  const active = groupState === 'active_waiting_tool' || groupState === 'active_waiting_model';

  if (groupState === 'orphaned') {
    return {
      item,
      groupState,
      decision: 'drop_group',
      reason: 'orphaned_native_group'
    };
  }

  if (item.capture_state !== 'complete') {
    const hasReadable = Boolean(item.readable_text?.trim() || item.readable_summary?.trim());
    const plaintextSupported = options.plaintextReasoningSupported ??
      (targetFormat === OPENAI_RESPONSES_REASONING_FORMAT || !targetFormat);
    if (
      hasReadable &&
      plaintextSupported &&
      options.historyPolicy !== 'strip' &&
      isReasoningLikeNativeItem(item) &&
      !dependent
    ) {
      return { item, groupState, decision: 'emit_plaintext', reason: 'untrusted_native_state' };
    }
    if (active && dependent) {
      return { item, groupState, decision: 'reject', reason: 'incomplete_active_native_group' };
    }
    return {
      item,
      groupState,
      decision: dependent ? 'drop_group' : 'strip_optional',
      reason: 'incomplete_native_capture'
    };
  }

  const policyAllowsNative = options.historyPolicy === undefined || options.historyPolicy === 'native';
  if (
    policyAllowsNative &&
    targetFormat &&
    isReplayableNativeItemType(item, targetFormat) &&
    isProviderNativeItemStructurallyComplete(item) &&
    canReplayReasoningState(item.source_format, item.source_origin, targetFormat, targetOrigin)
  ) {
    return { item, groupState, decision: 'replay_native' };
  }

  const hasReadable = Boolean(item.readable_text?.trim() || item.readable_summary?.trim());
  const plaintextSupported = options.plaintextReasoningSupported ??
    (targetFormat === OPENAI_RESPONSES_REASONING_FORMAT || !targetFormat);
  if (
    hasReadable &&
    plaintextSupported &&
    options.historyPolicy !== 'strip' &&
    isReasoningLikeNativeItem(item) &&
    !dependent
  ) {
    return { item, groupState, decision: 'emit_plaintext' };
  }

  if (!dependent && isReasoningLikeNativeItem(item)) {
    return { item, groupState, decision: 'strip_optional' };
  }
  if (active && dependent) {
    return { item, groupState, decision: 'reject', reason: 'incompatible_active_native_group' };
  }
  return {
    item,
    groupState,
    decision: dependent ? 'drop_group' : 'strip_optional'
  };
}

export function clearStatefulNativeRouteCacheForTests(): void {
  statefulRouteByResponseId.clear();
}

function prepareStatefulRequestForTarget(
  request: StandardRequest,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin,
  options: NativeStatePreparationOptions
): Result<StandardRequest> {
  const previousResponseId = request.openai_responses?.previous_response_id;
  const previousInteractionId = request.gemini_interactions?.previous_interaction_id;
  if (!previousResponseId && !previousInteractionId) {
    return ok(request);
  }

  const id = previousResponseId || previousInteractionId!;
  const expectedFormat = previousResponseId
    ? OPENAI_RESPONSES_REASONING_FORMAT
    : GEMINI_INTERACTIONS_REASONING_FORMAT;
  const record = readStatefulRoute(id);
  const interactionsStoreValid = !previousInteractionId || request.gemini_interactions?.store === true;
  const routeValid = Boolean(
    !options.failover &&
    interactionsStoreValid &&
    targetFormat === expectedFormat &&
    record?.format === expectedFormat &&
    sameStatefulRoute(record.origin, targetOrigin)
  );
  if (routeValid) {
    return ok({ ...request, input: keepOnlyStatefulDeltaInput(request.input) });
  }

  if (!hasCompleteManualHistory(request)) {
    return err(previousResponseId
      ? 'incompatible_previous_response_route'
      : 'incompatible_previous_interaction_route');
  }

  if (previousResponseId) {
    const openaiResponses = { ...request.openai_responses };
    delete openaiResponses.previous_response_id;
    return ok({ ...request, openai_responses: openaiResponses });
  }
  const geminiInteractions = { ...request.gemini_interactions };
  delete geminiInteractions.previous_interaction_id;
  return ok({ ...request, gemini_interactions: geminiInteractions });
}

function validateCompactionContinuity(
  request: StandardRequest,
  groups: ProviderNativeGroupAnalysis[],
  decisions: Map<ProviderNativeItem, ProviderNativeItemDecision>,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin,
  options: NativeStatePreparationOptions
): Result<StandardRequest> {
  const compactItems = groups.flatMap(({ items }) => items).filter(isCompactionNativeItem);
  for (const compactItem of compactItems) {
    const decision = decisions.get(compactItem);
    if (compactItem.compaction_mode === 'standalone' && compactItem.capture_state !== 'complete') {
      return err('incomplete_standalone_compaction_window');
    }
    if (decision?.decision === 'replay_native') {
      if (
        compactItem.compaction_mode === 'standalone' &&
        (compactItem.capture_state !== 'complete' ||
          !canReplayReasoningState(compactItem.source_format, compactItem.source_origin, targetFormat!, targetOrigin))
      ) {
        return err('incomplete_standalone_compaction_window');
      }
      continue;
    }
    if (hasHistoryBeforeCompaction(request, compactItem)) {
      continue;
    }
    if (options.explicitStrip === true && options.historyPolicy === 'strip') {
      continue;
    }
    return err('incompatible_compacted_history');
  }
  return ok(request);
}

function projectReadableNativeItem(
  original: StandardRequestInputContent,
  nativeItem: ProviderNativeItem
): StandardRequestInputContent[] {
  if (original.type === 'reasoning') {
    const readable = stripOpaqueReasoningState(original);
    return readable ? [readable] : [];
  }
  const text = nativeItem.readable_text?.trim() || nativeItem.readable_summary?.trim();
  if (!text || !isReasoningLikeNativeItem(nativeItem)) {
    return [];
  }
  return [{
    type: 'reasoning',
    text,
    source_format: nativeItem.source_format
  }];
}

function stripOptionalNativeProjection(
  item: Exclude<StandardRequestInputContent, { type: 'provider_native_item' | 'reasoning' }>,
  targetFormat: string | undefined,
  targetOrigin: ReasoningStateOrigin
): StandardRequestInputContent {
  if (item.type === 'tool_use') {
    const {
      native_item: _nativeItem,
      thought_signature: thoughtSignature,
      thought_signature_format: thoughtSignatureFormat,
      thought_signature_origin: thoughtSignatureOrigin,
      ...toolUse
    } = item;
    if (
      thoughtSignature &&
      targetFormat &&
      canReplayReasoningState(
        thoughtSignatureFormat,
        thoughtSignatureOrigin,
        targetFormat,
        targetOrigin
      )
    ) {
      return {
        ...toolUse,
        thought_signature: thoughtSignature,
        thought_signature_format: thoughtSignatureFormat,
        ...(thoughtSignatureOrigin ? { thought_signature_origin: thoughtSignatureOrigin } : {})
      };
    }
    return toolUse;
  }
  const { native_item: _nativeItem, ...withoutNativeItem } = item;
  return withoutNativeItem;
}

function hasCompleteManualHistory(request: StandardRequest): boolean {
  if (typeof request.input === 'string') {
    return false;
  }
  const allNativeItems = deriveProviderNativeGroups(request).flatMap(({ items }) => items);
  const compactItems = allNativeItems.filter(isCompactionNativeItem);
  if (compactItems.some((item) => !hasHistoryBeforeCompaction(request, item))) {
    return false;
  }
  const nativeItems = allNativeItems.filter((item) => !isCompactionNativeItem(item));
  if (nativeItems.length > 0) {
    return nativeItems.every((item) => item.capture_state === 'complete');
  }
  return request.input.some((message) =>
    message.role === 'assistant' && message.content.some((item) =>
      item.type !== 'provider_native_item' || !isCompactionNativeItem(item)
    )
  );
}

function keepOnlyStatefulDeltaInput(
  input: StandardRequest['input']
): StandardRequest['input'] {
  if (typeof input === 'string') {
    return input;
  }
  let lastAssistantIndex = -1;
  for (let index = 0; index < input.length; index += 1) {
    if (input[index]?.role === 'assistant') {
      lastAssistantIndex = index;
    }
  }
  return lastAssistantIndex >= 0 ? input.slice(lastAssistantIndex + 1) : input;
}

function hasHistoryBeforeCompaction(request: StandardRequest, compactItem: ProviderNativeItem): boolean {
  if (typeof request.input === 'string') {
    return false;
  }
  let compactMessageIndex = -1;
  let compactContentIndex = -1;
  for (let messageIndex = 0; messageIndex < request.input.length; messageIndex += 1) {
    const message = request.input[messageIndex]!;
    const contentIndex = message.content.findIndex((content) => {
      const nativeItem = nativeItemFromContent(content);
      return nativeItem === compactItem ||
        (nativeItem !== undefined && providerNativeIdentity(nativeItem) === providerNativeIdentity(compactItem));
    });
    if (contentIndex >= 0) {
      compactMessageIndex = messageIndex;
      compactContentIndex = contentIndex;
      break;
    }
  }

  const compactPosition = compactItem.position;
  const nativeBeforeCompaction = (content: StandardRequestInputContent): boolean => {
    const nativeItem = nativeItemFromContent(content);
    return Boolean(
      nativeItem &&
      !isCompactionNativeItem(nativeItem) &&
      compareNativePosition(nativeItem.position, compactPosition) < 0
    );
  };
  if (compactMessageIndex < 0) {
    // Without a carrier location, only position-bearing native items can prove
    // that manual history predates the compacted window. Text after a compact
    // item must never be mistaken for the original history.
    return request.input.some((message) => message.content.some(nativeBeforeCompaction));
  }

  for (let messageIndex = 0; messageIndex <= compactMessageIndex; messageIndex += 1) {
    const message = request.input[messageIndex]!;
    const limit = messageIndex === compactMessageIndex
      ? compactContentIndex
      : message.content.length;
    for (let contentIndex = 0; contentIndex < limit; contentIndex += 1) {
      const content = message.content[contentIndex]!;
      if (nativeBeforeCompaction(content)) {
        return true;
      }
      if (!nativeItemFromContent(content) && isMeaningfulManualHistoryContent(content)) {
        return true;
      }
    }
  }
  return false;
}

function compareNativePosition(
  left: ProviderNativeItem['position'],
  right: ProviderNativeItem['position']
): number {
  return left.turn - right.turn || left.step - right.step || left.item - right.item;
}

function nativeItemFromContent(content: StandardRequestInputContent): ProviderNativeItem | undefined {
  if (content.type === 'provider_native_item') {
    return content;
  }
  return 'native_item' in content ? content.native_item : undefined;
}

function isProviderNativeItemStructurallyComplete(item: ProviderNativeItem): boolean {
  if (item.provider_status && item.provider_status !== 'completed') {
    return false;
  }
  if (
    !item.item_type ||
    !item.provider_schema_version ||
    !isPlainRecord(item.raw_payload) ||
    !isProviderNativePayloadStructurallyValid(item)
  ) {
    return false;
  }
  if (isProviderNativeCallItem(item)) {
    return Boolean(item.call_id || readNativeString(item.raw_payload, 'call_id') || readNativeString(item.raw_payload, 'id'));
  }
  if (isProviderNativeResultItem(item)) {
    return Boolean(item.pair_id || item.call_id || readNativeString(item.raw_payload, 'call_id'));
  }
  return true;
}

function isReplayableNativeItemType(item: ProviderNativeItem, targetFormat: string): boolean {
  const itemType = item.item_type.toLowerCase();
  if (targetFormat === OPENAI_RESPONSES_REASONING_FORMAT) {
    return new Set([
      'reasoning', 'message', 'function_call', 'function_call_output', 'program',
      'program_output', 'compaction', 'tool_search_call', 'tool_search_output'
    ]).has(itemType);
  }
  if (targetFormat === ANTHROPIC_CLAUDE_REASONING_FORMAT) {
    return itemType === 'thinking' || itemType === 'redacted_thinking' ||
      itemType === 'tool_use' || itemType === 'tool_result';
  }
  if (targetFormat === GEMINI_GENERATE_CONTENT_REASONING_FORMAT) {
    return new Set(['part', 'thought', 'function_call', 'function_response', 'thought_signature']).has(itemType);
  }
  if (targetFormat === GEMINI_INTERACTIONS_REASONING_FORMAT) {
    return itemType === 'thought' || isGeminiInteractionsSignedBuiltInStep(itemType);
  }
  return false;
}

function isGeminiInteractionsSignedBuiltInStep(itemType: string): boolean {
  return /^(?:code_execution|file_search|google_maps|google_search|retrieval)_(?:call|result)$/.test(itemType);
}

function isReasoningLikeNativeItem(item: ProviderNativeItem): boolean {
  return /reasoning|thinking|thought/.test(item.item_type.toLowerCase());
}

function isCompactionNativeItem(item: ProviderNativeItem): boolean {
  return item.item_type.toLowerCase() === 'compaction';
}

function isDependencyCriticalNativeItem(item: ProviderNativeItem): boolean {
  const type = item.item_type.toLowerCase();
  if (
    isCompactionNativeItem(item) ||
    type === 'program' ||
    type === 'program_output' ||
    Boolean(item.depends_on?.length)
  ) {
    return true;
  }
  if (
    item.source_format === OPENAI_RESPONSES_REASONING_FORMAT &&
    (isPlainRecord(item.raw_payload.caller) ||
      readNativeString(item.raw_payload, 'fingerprint') ||
      readNativeString(item.raw_payload, 'program_id'))
  ) {
    return true;
  }
  if (
    item.source_format === ANTHROPIC_CLAUDE_REASONING_FORMAT &&
    Boolean(item.group_id) &&
    (type === 'thinking' || type === 'redacted_thinking' || type === 'tool_use' || type === 'tool_result')
  ) {
    return true;
  }
  if (
    item.source_format === GEMINI_GENERATE_CONTENT_REASONING_FORMAT &&
    type === 'function_call'
  ) {
    const hasSignature = Boolean(
      readNativeString(item.raw_payload, 'thoughtSignature') ||
      readNativeString(item.raw_payload, 'thought_signature') ||
      (isPlainRecord(item.raw_payload.functionCall) &&
        (readNativeString(item.raw_payload.functionCall, 'thoughtSignature') ||
          readNativeString(item.raw_payload.functionCall, 'thought_signature')))
    );
    const nativeGemini3Call = item.item_origin === 'native' &&
      /^gemini-3(?:[.-]|$)/i.test(item.source_origin.model?.trim() || '');
    return hasSignature || nativeGemini3Call;
  }
  return false;
}

function isProviderNativeCallItem(item: ProviderNativeItem): boolean {
  const type = item.item_type.toLowerCase();
  return type === 'program' || type.endsWith('_call') || type === 'tool_use';
}

function isProviderNativeResultItem(item: ProviderNativeItem): boolean {
  const type = item.item_type.toLowerCase();
  return type.endsWith('_result') || type.endsWith('_output') ||
    type.endsWith('_response') || type === 'tool_result';
}

function deriveProviderNativeGroupState(
  items: ProviderNativeItem[],
  hasLaterAssistant: boolean
): { state: ProviderNativeGroupState; active: boolean } {
  const callCounts = new Map<string, number>();
  const resultCounts = new Map<string, number>();
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index]!;
    if (isProviderNativeCallItem(item)) {
      const key = providerNativePairKey(item, `unpaired-call-${index}`);
      callCounts.set(key, (callCounts.get(key) || 0) + 1);
    }
    if (isProviderNativeResultItem(item)) {
      const key = providerNativePairKey(item, `unpaired-result-${index}`);
      resultCounts.set(key, (resultCounts.get(key) || 0) + 1);
    }
  }

  const keys = new Set([...callCounts.keys(), ...resultCounts.keys()]);
  const hasUnmatchedCall = [...keys].some(
    (key) => (callCounts.get(key) || 0) > (resultCounts.get(key) || 0)
  );
  const hasUnmatchedResult = [...keys].some(
    (key) => (resultCounts.get(key) || 0) > (callCounts.get(key) || 0)
  );
  if (hasUnmatchedCall && hasUnmatchedResult) {
    return { state: 'orphaned', active: !hasLaterAssistant };
  }
  if (hasUnmatchedCall) {
    return hasLaterAssistant
      ? { state: 'orphaned', active: false }
      : { state: 'active_waiting_tool', active: true };
  }
  if (hasUnmatchedResult) {
    return hasLaterAssistant
      ? { state: 'orphaned', active: false }
      : { state: 'active_waiting_model', active: true };
  }
  if (callCounts.size > 0 || resultCounts.size > 0) {
    return hasLaterAssistant
      ? { state: 'historical_closed', active: false }
      : { state: 'active_waiting_model', active: true };
  }
  return { state: 'historical_closed', active: false };
}

function providerNativePairKey(item: ProviderNativeItem, fallback: string): string {
  return item.pair_id || item.call_id ||
    readNativeString(item.raw_payload, 'call_id') ||
    readNativeString(item.raw_payload, 'program_id') ||
    readNativeString(item.raw_payload, 'tool_use_id') ||
    item.native_id || fallback;
}

function providerNativeRelationIds(item: ProviderNativeItem): string[] {
  const caller = isPlainRecord(item.raw_payload.caller) ? item.raw_payload.caller : undefined;
  const functionCall = isPlainRecord(item.raw_payload.functionCall)
    ? item.raw_payload.functionCall
    : undefined;
  const functionResponse = isPlainRecord(item.raw_payload.functionResponse)
    ? item.raw_payload.functionResponse
    : undefined;
  return [...new Set([
    item.group_id,
    item.pair_id,
    item.call_id,
    item.native_id,
    readNativeString(item.raw_payload, 'id'),
    readNativeString(item.raw_payload, 'call_id'),
    readNativeString(item.raw_payload, 'program_id'),
    readNativeString(item.raw_payload, 'tool_use_id'),
    readNativeString(caller, 'id'),
    readNativeString(caller, 'caller_id'),
    readNativeString(functionCall, 'id'),
    readNativeString(functionCall, 'call_id'),
    readNativeString(functionResponse, 'id'),
    readNativeString(functionResponse, 'call_id'),
    ...(item.depends_on || [])
  ].filter((value): value is string => Boolean(value?.trim())))]
    .map((value) => value.trim());
}

function providerNativeRelationNamespace(item: ProviderNativeItem): string {
  const origin = item.source_origin;
  return [
    item.source_format,
    origin.provider,
    origin.endpoint,
    origin.model || '',
    origin.credentialScope || ''
  ].join('\0');
}

function isMeaningfulManualHistoryContent(content: StandardRequestInputContent): boolean {
  if (content.type === 'input_text') {
    return Boolean(content.text.trim());
  }
  if (content.type === 'reasoning') {
    return Boolean(content.text?.trim() || content.summary?.trim());
  }
  return content.type !== 'provider_native_item';
}

function deriveProviderNativeGroupId(item: ProviderNativeItem, order: number): string {
  const caller = isPlainRecord(item.raw_payload.caller) ? item.raw_payload.caller : undefined;
  return readNativeString(caller, 'caller_id') || item.group_id || item.pair_id || item.call_id ||
    readNativeString(item.raw_payload, 'call_id') || item.native_id || `native-${order}`;
}

function providerNativeIdentity(item: ProviderNativeItem): string {
  return [
    item.source_format,
    item.native_id || '',
    item.item_type,
    item.position.turn,
    item.position.step,
    item.position.item
  ].join('\0');
}

function readNativeString(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key];
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function rememberStatefulRoute(id: string, format: string, origin: ReasoningStateOrigin): void {
  pruneStatefulRoutes();
  if (statefulRouteByResponseId.size >= STATEFUL_ROUTE_CACHE_LIMIT) {
    const oldest = statefulRouteByResponseId.keys().next().value as string | undefined;
    if (oldest) {
      statefulRouteByResponseId.delete(oldest);
    }
  }
  statefulRouteByResponseId.set(id, { format, origin: { ...origin }, recordedAt: Date.now() });
}

function readStatefulRoute(id: string): StatefulRouteRecord | undefined {
  pruneStatefulRoutes();
  return statefulRouteByResponseId.get(id);
}

function pruneStatefulRoutes(): void {
  const cutoff = Date.now() - STATEFUL_ROUTE_CACHE_TTL_MS;
  for (const [id, record] of statefulRouteByResponseId) {
    if (record.recordedAt >= cutoff) {
      break;
    }
    statefulRouteByResponseId.delete(id);
  }
}

function sameStatefulRoute(left: ReasoningStateOrigin, right: ReasoningStateOrigin): boolean {
  return left.provider === right.provider && left.endpoint === right.endpoint &&
    Boolean(left.credentialScope) && left.credentialScope === right.credentialScope;
}

function inferFormatFromOrigin(origin: ReasoningStateOrigin): string | undefined {
  if (origin.provider === 'openai') {
    return OPENAI_RESPONSES_REASONING_FORMAT;
  }
  if (origin.provider === 'anthropic') {
    return ANTHROPIC_CLAUDE_REASONING_FORMAT;
  }
  if (origin.provider === 'gemini') {
    return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
  }
  return undefined;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveReasoningCredentialScope(
  providerFamily: string,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig,
  credentialContext: ReasoningStateCredentialContext | undefined
): string | undefined {
  const providerIdentity = (
    providerConfig?.credentialSourceProviderName ||
    providerConfig?.name ||
    providerFamily
  ).trim().toLowerCase();
  const credentialId = providerConfig?.credentialId?.trim() || 'default';
  const pluginResolution = resolveProviderPluginCredentialScope(credentialContext);
  if (pluginResolution.present && !pluginResolution.scope) {
    return undefined;
  }
  const credentialMaterial = pluginResolution.scope
    ? `plugins\0${pluginResolution.scope}`
    : resolveBuiltInCredential(
        providerFamily,
        providerConfig,
        config,
        credentialContext?.request
      );
  if (!credentialMaterial) {
    return undefined;
  }

  return createHash('sha256')
    .update(
      `${providerFamily}\0${providerIdentity}\0${credentialId}\0${credentialMaterial}`,
      'utf8'
    )
    .digest('base64url');
}

function resolveProviderPluginCredentialScope(
  credentialContext: ReasoningStateCredentialContext | undefined
): { present: boolean; scope?: string } {
  if (!credentialContext) {
    return { present: false };
  }

  // Provider plugins execute in registration order, so keep that order in the
  // fingerprint as well. Reordering two auth plugins can change the final credential.
  const credentialPlugins = credentialContext.plugins.filter(
    (plugin) => plugin.authenticate || plugin.resolveCredentialScope
  );
  if (credentialPlugins.length === 0) {
    return { present: false };
  }

  const scopes: string[] = [];
  for (const plugin of credentialPlugins) {
    if (!plugin.resolveCredentialScope) {
      return { present: true };
    }
    try {
      const scope = plugin.resolveCredentialScope(credentialContext)?.trim();
      if (!scope) {
        return { present: true };
      }
      scopes.push(`${plugin.key}\0${scope}`);
    } catch {
      return { present: true };
    }
  }

  return { present: true, scope: scopes.join('\0') };
}

function resolveBuiltInCredential(
  providerFamily: string,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig,
  request: ProviderPluginCredentialScopeInput['request'] | undefined
): string | undefined {
  const configuredProviderKey =
    providerConfig?.apikey?.trim() ||
    (providerConfig?.apiKeyEnv ? process.env[providerConfig.apiKeyEnv]?.trim() : undefined);
  const preferManaged = Boolean(
    config.auth?.enabled &&
      (config.auth.mode === 'http_introspection' || config.auth.mode === 'static_api_key')
  );

  if (providerFamily === 'openai') {
    const managed = configuredProviderKey || config.openaiApiKey?.trim() || process.env.OPENAI_API_KEY?.trim();
    const bearer = readBearerToken(readHeader(request?.headers.authorization));
    const apiKeyHeader =
      readHeader(request?.headers['x-api-key']) || readHeader(request?.headers['api-key']);
    return preferManaged
      ? managed || bearer || apiKeyHeader
      : bearer || apiKeyHeader || managed;
  }

  if (providerFamily === 'anthropic') {
    const managed = configuredProviderKey || config.anthropicApiKey?.trim() || process.env.ANTHROPIC_API_KEY?.trim();
    const apiKeyHeader = readHeader(request?.headers['x-api-key']);
    const bearer = readBearerToken(readHeader(request?.headers.authorization));
    return preferManaged
      ? managed || apiKeyHeader || bearer
      : apiKeyHeader || bearer || managed;
  }

  if (providerFamily === 'gemini') {
    return (
      readRequestQueryValue(request?.query, 'key') ||
      configuredProviderKey ||
      config.geminiApiKey?.trim() ||
      process.env.GEMINI_API_KEY?.trim()
    );
  }

  return configuredProviderKey;
}

function readRequestQueryValue(value: unknown, key: string): string | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }
  const raw = (value as Record<string, unknown>)[key];
  if (typeof raw === 'string') {
    return raw.trim() || undefined;
  }
  if (Array.isArray(raw)) {
    const first = raw.find((item): item is string => typeof item === 'string' && Boolean(item.trim()));
    return first?.trim();
  }
  return undefined;
}

export function wrapPassthroughReasoningPayload(
  payload: unknown,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin,
  options: { compactionMode?: ProviderNativeItem['compaction_mode'] } = {}
): { payload: unknown; changed: boolean } {
  const format = sourceFormatForAdapter(sourceAdapterKey);
  if (!format || (typeof payload !== 'object' && !Array.isArray(payload))) {
    return { payload, changed: false };
  }

  let changed = false;
  let nativeItemIndex = 0;
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (typeof value !== 'object' || value === null) {
      return;
    }

    const record = value as Record<string, unknown>;
    const rawPayload = cloneNativePayload(record);
    const itemType = typeof record.type === 'string' ? record.type : 'unknown';
    const callId = readNativeString(record, 'call_id') || readNativeString(record, 'program_id');
    const nativeItem: Partial<ProviderNativeItem> = {
      item_type: itemType,
      ...(readNativeString(record, 'id') ? { native_id: readNativeString(record, 'id') } : {}),
      raw_payload: rawPayload,
      provider_schema_version: format,
      item_origin: 'native',
      position: { turn: 0, step: 0, item: nativeItemIndex },
      ...(callId ? { group_id: callId, call_id: callId, pair_id: callId } : {}),
      capture_state: 'complete',
      ...(readNativeString(record, 'status') ? { provider_status: readNativeString(record, 'status') } : {})
    };
    nativeItemIndex += 1;
    if (format === OPENAI_RESPONSES_REASONING_FORMAT && record.type === 'reasoning') {
      changed = wrapRecordField(record, 'encrypted_content', format, origin, {
        id: typeof record.id === 'string' ? record.id : undefined,
        kind: 'encrypted',
        nativeItem
      }) || changed;
    } else if (format === OPENAI_RESPONSES_REASONING_FORMAT) {
      if (record.type === 'compaction') {
        nativeItem.compaction_mode = options.compactionMode || 'server_side';
        changed = wrapRecordField(record, 'encrypted_content', format, origin, {
          id: readNativeString(record, 'id'),
          kind: 'encrypted',
          nativeItem
        }) || changed;
      }
      changed = wrapRecordField(record, 'fingerprint', format, origin, {
        id: readNativeString(record, 'id'),
        kind: 'signature',
        nativeItem
      }) || changed;
      if (typeof record.caller === 'string') {
        changed = wrapRecordField(record, 'caller', format, origin, {
          id: readNativeString(record, 'id'),
          kind: 'signature',
          nativeItem
        }) || changed;
      } else if (isPlainRecord(record.caller)) {
        changed = wrapRecordField(record.caller, 'fingerprint', format, origin, {
          id: readNativeString(record, 'id'),
          kind: 'signature',
          nativeItem
        }) || changed;
      }
    } else if (format === ANTHROPIC_CLAUDE_REASONING_FORMAT) {
      if (record.type === 'thinking') {
        changed = wrapRecordField(record, 'signature', format, origin, { kind: 'signature', nativeItem }) || changed;
      } else if (record.type === 'redacted_thinking') {
        changed = wrapRecordField(record, 'data', format, origin, { kind: 'encrypted', nativeItem }) || changed;
      }
    } else if (format === GEMINI_GENERATE_CONTENT_REASONING_FORMAT) {
      changed = wrapRecordField(record, 'thoughtSignature', format, origin, { kind: 'signature', nativeItem }) || changed;
      changed = wrapRecordField(record, 'thought_signature', format, origin, { kind: 'signature', nativeItem }) || changed;
    } else if (format === GEMINI_INTERACTIONS_REASONING_FORMAT) {
      if (record.type === 'thought' || record.type === 'thought_signature' ||
          isGeminiInteractionsSignedBuiltInStep(itemType)) {
        changed = wrapRecordField(record, 'signature', format, origin, { kind: 'signature', nativeItem }) || changed;
      }
    }

    Object.values(record).forEach(visit);
  };

  visit(payload);
  return { payload, changed };
}

export function createReasoningAwarePassthroughSseStream(
  response: Response,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin,
  abortSignal?: AbortSignal,
  bodyLimitBytes = 32 * 1024 * 1024
): Readable {
  return Readable.from(
    validateReasoningTransportCarrierSseStream(
      relayReasoningAwarePassthroughSse(response, sourceAdapterKey, origin, abortSignal),
      bodyLimitBytes
    )
  );
}

async function* relayReasoningAwarePassthroughSse(
  response: Response,
  sourceAdapterKey: string,
  origin: ReasoningStateOrigin,
  abortSignal?: AbortSignal
): AsyncGenerator<string> {
  const anthropicSignatureByIndex = new Map<number, string>();
  const anthropicThinkingByIndex = new Map<number, string>();

  const flushAnthropicSignature = (index: number): string | undefined => {
    const signature = anthropicSignatureByIndex.get(index);
    const thinking = anthropicThinkingByIndex.get(index) || '';
    anthropicThinkingByIndex.delete(index);
    if (!signature) {
      return undefined;
    }
    anthropicSignatureByIndex.delete(index);
    const nativeItem: Partial<ProviderNativeItem> = {
      item_type: 'thinking',
      raw_payload: {
        type: 'thinking',
        thinking,
        signature
      },
      provider_schema_version: ANTHROPIC_CLAUDE_REASONING_FORMAT,
      item_origin: 'native',
      position: { turn: 0, step: 0, item: index },
      capture_state: 'complete',
      ...(thinking ? { readable_text: thinking } : {})
    };
    return encodeSseChunk('content_block_delta', {
      type: 'content_block_delta',
      index,
      delta: {
        type: 'signature_delta',
        signature: encodeReasoningTransportEnvelope(
          ANTHROPIC_CLAUDE_REASONING_FORMAT,
          signature,
          undefined,
          'signature',
          origin,
          { nativeItem }
        )
      }
    });
  };

  for await (const chunk of parseSseChunks(response, abortSignal)) {
    const data = chunk.data.trim();
    if (!data) {
      continue;
    }
    if (data === '[DONE]') {
      for (const index of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
        const frame = flushAnthropicSignature(index);
        if (frame) {
          yield frame;
        }
      }
      yield encodeRawSseChunk(chunk.event, data);
      continue;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data) as unknown;
    } catch {
      yield encodeRawSseChunk(chunk.event, data);
      continue;
    }

    if (sourceAdapterKey === 'anthropic_messages' && isRecord(payload)) {
      const eventType = typeof payload.type === 'string' ? payload.type : chunk.event;
      const index = typeof payload.index === 'number' ? payload.index : undefined;
      const delta = isRecord(payload.delta) ? payload.delta : undefined;
      const contentBlock = isRecord(payload.content_block) ? payload.content_block : undefined;
      if (
        eventType === 'content_block_start' &&
        index !== undefined &&
        contentBlock?.type === 'thinking'
      ) {
        anthropicThinkingByIndex.set(
          index,
          typeof contentBlock.thinking === 'string' ? contentBlock.thinking : ''
        );
      }
      if (
        eventType === 'content_block_delta' &&
        index !== undefined &&
        delta?.type === 'thinking_delta' &&
        typeof delta.thinking === 'string'
      ) {
        anthropicThinkingByIndex.set(
          index,
          `${anthropicThinkingByIndex.get(index) || ''}${delta.thinking}`
        );
      }
      if (
        eventType === 'content_block_delta' &&
        index !== undefined &&
        delta?.type === 'signature_delta' &&
        typeof delta.signature === 'string'
      ) {
        anthropicSignatureByIndex.set(
          index,
          `${anthropicSignatureByIndex.get(index) || ''}${delta.signature}`
        );
        continue;
      }
      if (eventType === 'content_block_stop' && index !== undefined) {
        const frame = flushAnthropicSignature(index);
        if (frame) {
          yield frame;
        }
      } else if (eventType === 'message_stop') {
        for (const pendingIndex of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
          const frame = flushAnthropicSignature(pendingIndex);
          if (frame) {
            yield frame;
          }
        }
        anthropicThinkingByIndex.clear();
      }
    }

    const wrapped = wrapPassthroughReasoningPayload(payload, sourceAdapterKey, origin);
    yield encodeSseChunk(chunk.event, wrapped.payload);
  }

  for (const index of [...anthropicSignatureByIndex.keys()].sort((a, b) => a - b)) {
    const frame = flushAnthropicSignature(index);
    if (frame) {
      yield frame;
    }
  }
  anthropicThinkingByIndex.clear();
}

function resolveProviderFamily(provider: Provider, providerConfig?: ProviderConfig): string {
  switch (providerConfig?.type) {
    case 'openai_responses':
    case 'openai_chat_completions':
      return 'openai';
    case 'anthropic_messages':
      return 'anthropic';
    case 'gemini_generate_content':
    case 'gemini_interactions':
      return 'gemini';
    default:
      return provider.trim().toLowerCase();
  }
}

function resolveProviderBaseUrl(
  providerFamily: string,
  providerConfig: ProviderConfig | undefined,
  config: GatewayConfig
): string {
  if (providerConfig?.baseurl?.trim()) {
    return providerConfig.baseurl.trim();
  }
  if (providerFamily === 'openai') {
    return config.openaiBaseUrl;
  }
  if (providerFamily === 'anthropic') {
    return config.anthropicBaseUrl;
  }
  if (providerFamily === 'gemini') {
    return config.geminiBaseUrl;
  }
  return providerFamily;
}

function hasOpaqueReasoningState(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }> | StandardResponse['output'][number]
): boolean {
  if (item.type !== 'reasoning') {
    return false;
  }
  if (item.encrypted_content) {
    return true;
  }
  return (item.reasoning_details || []).some((detail) => {
    if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) {
      return false;
    }
    const record = detail as Record<string, unknown>;
    return Boolean(
      record.data ||
      record.encrypted_content ||
      record.signature ||
      record.thoughtSignature ||
      record.thought_signature
    );
  });
}

function stripOpaqueReasoningState(
  item: Extract<StandardRequestInputContent, { type: 'reasoning' }>
): Extract<StandardRequestInputContent, { type: 'reasoning' }> | undefined {
  const readableDetails = sanitizeReadableReasoningDetails(item.reasoning_details);
  const {
    encrypted_content: _encryptedContent,
    source_origin: _sourceOrigin,
    ...readableReasoning
  } = item;
  if (
    !readableReasoning.text?.trim() &&
    !readableReasoning.summary?.trim() &&
    readableDetails.length === 0
  ) {
    return undefined;
  }
  return {
    ...readableReasoning,
    ...(readableDetails.length > 0
      ? { reasoning_details: readableDetails }
      : { reasoning_details: undefined })
  };
}

function sanitizeReadableReasoningDetails(value: unknown[] | undefined): unknown[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const readableDetails: unknown[] = [];
  for (const detail of value) {
    if (typeof detail === 'string') {
      if (detail.trim()) {
        readableDetails.push(detail);
      }
      continue;
    }
    if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) {
      continue;
    }
    const record = detail as Record<string, unknown>;
    const {
      data: _data,
      encrypted_content: _encryptedContent,
      signature: _signature,
      thoughtSignature: _thoughtSignature,
      thought_signature: _thoughtSignatureSnake,
      ...readable
    } = record;
    const hasReadableText = ['text', 'summary', 'reasoning', 'thinking'].some(
      (key) => typeof readable[key] === 'string' && (readable[key] as string).trim()
    );
    if (hasReadableText) {
      readableDetails.push(readable);
    }
  }
  return readableDetails;
}

function sourceFormatForAdapter(sourceAdapterKey: string): string | undefined {
  if (sourceAdapterKey === 'openai_responses') {
    return OPENAI_RESPONSES_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'anthropic_messages') {
    return ANTHROPIC_CLAUDE_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'gemini_generate' || sourceAdapterKey === 'gemini_stream') {
    return GEMINI_GENERATE_CONTENT_REASONING_FORMAT;
  }
  if (sourceAdapterKey === 'gemini_interactions') {
    return GEMINI_INTERACTIONS_REASONING_FORMAT;
  }
  return undefined;
}

function wrapRecordField(
  record: Record<string, unknown>,
  field: string,
  format: string,
  origin: ReasoningStateOrigin,
  options: {
    id?: string;
    kind: 'signature' | 'encrypted';
    nativeItem?: Partial<ProviderNativeItem>;
  }
): boolean {
  const value = typeof record[field] === 'string' ? record[field] as string : '';
  if (!value || decodeReasoningTransportEnvelope(value)) {
    return false;
  }
  record[field] = encodeReasoningTransportEnvelope(
    format,
    value,
    options.id,
    options.kind,
    origin,
    options.nativeItem ? { nativeItem: options.nativeItem } : undefined
  );
  return true;
}

function cloneNativePayload(value: Record<string, unknown>): Record<string, unknown> {
  try {
    return structuredClone(value);
  } catch {
    return { ...value };
  }
}

function encodeSseChunk(event: string | undefined, payload: unknown): string {
  return encodeRawSseChunk(event, JSON.stringify(payload));
}

function encodeRawSseChunk(event: string | undefined, data: string): string {
  return `${event ? `event: ${event}\n` : ''}${data
    .split('\n')
    .map((line) => `data: ${line}`)
    .join('\n')}\n\n`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
