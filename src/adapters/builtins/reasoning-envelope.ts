import type {
  ProviderNativeItem,
  ProviderNativeItemPosition,
  ReasoningStateOrigin
} from '../../types';

export const OPENAI_RESPONSES_REASONING_FORMAT = 'openai-responses-v1';
export const ANTHROPIC_CLAUDE_REASONING_FORMAT = 'anthropic-claude-v1';
export const GEMINI_GENERATE_CONTENT_REASONING_FORMAT = 'google-generate-content-v1';
export const GEMINI_INTERACTIONS_REASONING_FORMAT = 'google-interactions-v1';

const OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX = 'ccr-openai-responses-reasoning-v1:';
const REASONING_TRANSPORT_ENVELOPE_PREFIX = 'ccr-reasoning-transport-v1:';
const REASONING_TRANSPORT_ENVELOPE_V2_PREFIX = 'ccr-reasoning-transport-v2:';
export const REASONING_TRANSPORT_ENVELOPE_V3_PREFIX = 'ccr-reasoning-transport-v3:';
export const GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR = '__thought__';

export const PROVIDER_NATIVE_MAX_ITEMS = 4096;
export const PROVIDER_NATIVE_MAX_JSON_DEPTH = 32;
export const GEMINI_SIGNATURE_ID_MAX_BYTES = 64 * 1024;

const MIB = 1024 * 1024;
const DEFAULT_BODY_LIMIT_BYTES = 32 * MIB;

export interface OpenAIResponsesReasoningEnvelope {
  id: string;
  encryptedContent: string;
}

export interface ReasoningTransportEnvelope {
  format: string;
  data: string;
  id?: string;
  kind?: 'signature' | 'encrypted';
  origin?: ReasoningStateOrigin;
  carrierVersion?: 1 | 2 | 3;
  nativeItem?: ProviderNativeItem;
}

export interface ReasoningTransportEncodeOptions {
  nativeItem?: Partial<ProviderNativeItem>;
  providerSchemaVersion?: string;
  position?: ProviderNativeItemPosition;
  groupId?: string;
  callId?: string;
  pairId?: string;
  dependsOn?: string[];
  captureState?: ProviderNativeItem['capture_state'];
  providerStatus?: string;
  providerMode?: string;
  readableText?: string;
  readableSummary?: string;
  compactionMode?: ProviderNativeItem['compaction_mode'];
}

export interface ReasoningCarrierLimits {
  maxEncodedBytes: number;
  maxDecodedPayloadBytes: number;
  maxItems: number;
  maxJsonDepth: number;
  maxGeminiSignatureIdBytes: number;
}

export type ReasoningCarrierValidationResult =
  | { ok: true; itemCount: number; encodedBytes: number }
  | { ok: false; status: 400 | 413; code: string; message: string };

export function reasoningCarrierLimits(bodyLimitBytes = DEFAULT_BODY_LIMIT_BYTES): ReasoningCarrierLimits {
  const normalizedBodyLimit = Number.isFinite(bodyLimitBytes) && bodyLimitBytes > 0
    ? Math.floor(bodyLimitBytes)
    : DEFAULT_BODY_LIMIT_BYTES;
  return {
    maxEncodedBytes: Math.min(16 * MIB, Math.max(1, Math.floor(normalizedBodyLimit / 2))),
    maxDecodedPayloadBytes: Math.min(8 * MIB, Math.max(1, Math.floor(normalizedBodyLimit / 4))),
    maxItems: PROVIDER_NATIVE_MAX_ITEMS,
    maxJsonDepth: PROVIDER_NATIVE_MAX_JSON_DEPTH,
    maxGeminiSignatureIdBytes: GEMINI_SIGNATURE_ID_MAX_BYTES
  };
}

export function normalizeAnthropicThinkingMode(value: unknown): string {
  if (!isRecord(value)) {
    return 'default';
  }
  const type = asTrimmedString(value.type)?.toLowerCase();
  if (type === 'off' || type === 'none') {
    return 'disabled';
  }
  return type || 'default';
}

export function encodeOpenAIResponsesReasoningEnvelope(id: string, encryptedContent: string): string {
  const normalizedId = id.trim();
  if (!normalizedId || !encryptedContent) {
    return encryptedContent;
  }
  const payload = Buffer.from(JSON.stringify({ id: normalizedId, encrypted_content: encryptedContent }), 'utf8')
    .toString('base64url');
  return `${OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX}${payload}`;
}

export function decodeOpenAIResponsesReasoningEnvelope(
  value: string
): OpenAIResponsesReasoningEnvelope | undefined {
  if (!value.startsWith(OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX)) {
    return undefined;
  }
  const decoded = decodeBase64Json(value.slice(OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX.length));
  if (!isRecord(decoded)) {
    return undefined;
  }
  const id = asTrimmedString(decoded.id);
  const encryptedContent = typeof decoded.encrypted_content === 'string' ? decoded.encrypted_content : '';
  return id && encryptedContent ? { id, encryptedContent } : undefined;
}

/** New emissions use v3 whenever an origin is available; v1/v2 remain decode-only. */
export function encodeReasoningTransportEnvelope(
  format: string,
  data: string,
  id?: string,
  kind?: ReasoningTransportEnvelope['kind'],
  origin?: ReasoningStateOrigin,
  options: ReasoningTransportEncodeOptions = {}
): string {
  const normalizedFormat = format.trim();
  const normalizedId = id?.trim();
  if (!normalizedFormat || !data) {
    return data;
  }

  const normalizedOrigin = normalizeReasoningStateOrigin(origin);
  if (!normalizedOrigin) {
    if (normalizedFormat === OPENAI_RESPONSES_REASONING_FORMAT && normalizedId) {
      return encodeOpenAIResponsesReasoningEnvelope(normalizedId, data);
    }
    return encodeLegacyTransportEnvelope(normalizedFormat, data, normalizedId, kind);
  }

  const nativeItem = buildProviderNativeItem(
    normalizedFormat,
    data,
    normalizedId,
    kind,
    normalizedOrigin,
    options
  );
  const payload = Buffer.from(JSON.stringify({
    version: 3,
    format: normalizedFormat,
    data,
    ...(normalizedId ? { id: normalizedId } : {}),
    ...(kind ? { kind } : {}),
    origin: normalizedOrigin,
    native_item: nativeItem
  }), 'utf8').toString('base64url');
  return `${REASONING_TRANSPORT_ENVELOPE_V3_PREFIX}${payload}`;
}

export function decodeReasoningTransportEnvelope(
  value: string,
  limits: ReasoningCarrierLimits = reasoningCarrierLimits()
): ReasoningTransportEnvelope | undefined {
  const v3 = decodeTransportEnvelopeWithPrefix(value, REASONING_TRANSPORT_ENVELOPE_V3_PREFIX, 3, limits);
  if (v3) {
    return v3;
  }
  const v2 = decodeTransportEnvelopeWithPrefix(value, REASONING_TRANSPORT_ENVELOPE_V2_PREFIX, 2, limits);
  if (v2) {
    return v2;
  }
  const openAIEnvelope = decodeOpenAIResponsesReasoningEnvelope(value);
  if (openAIEnvelope) {
    return {
      format: OPENAI_RESPONSES_REASONING_FORMAT,
      id: openAIEnvelope.id,
      data: openAIEnvelope.encryptedContent,
      carrierVersion: 1
    };
  }
  return decodeTransportEnvelopeWithPrefix(value, REASONING_TRANSPORT_ENVELOPE_PREFIX, 1, limits);
}

export function appendGeminiThoughtSignatureToToolCallId(
  toolCallId: string,
  encodedSignature: string
): string {
  const envelope = decodeReasoningTransportEnvelope(encodedSignature);
  if (
    !toolCallId ||
    !envelope?.origin ||
    envelope.format !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT ||
    envelope.kind !== 'signature'
  ) {
    return toolCallId;
  }
  const carrierId = `${toolCallId}${GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR}${encodedSignature}`;
  return Buffer.byteLength(carrierId, 'utf8') <= GEMINI_SIGNATURE_ID_MAX_BYTES ? carrierId : toolCallId;
}

export function decodeGeminiThoughtSignatureToolCallId(value: string):
  | { toolCallId: string; envelope: ReasoningTransportEnvelope; encodedSignature: string }
  | undefined;
export function decodeGeminiThoughtSignatureToolCallId(
  value: string,
  limits: ReasoningCarrierLimits
): { toolCallId: string; envelope: ReasoningTransportEnvelope; encodedSignature: string } | undefined;
export function decodeGeminiThoughtSignatureToolCallId(
  value: string,
  limits: ReasoningCarrierLimits = reasoningCarrierLimits()
): { toolCallId: string; envelope: ReasoningTransportEnvelope; encodedSignature: string } | undefined {
  if (Buffer.byteLength(value, 'utf8') > limits.maxGeminiSignatureIdBytes) {
    return undefined;
  }
  const separatorIndex = value.lastIndexOf(GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR);
  if (separatorIndex <= 0) {
    return undefined;
  }
  const toolCallId = value.slice(0, separatorIndex);
  const encodedSignature = value.slice(separatorIndex + GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR.length);
  const envelope = decodeReasoningTransportEnvelope(encodedSignature, limits);
  if (
    !toolCallId ||
    !envelope?.origin ||
    envelope.format !== GEMINI_GENERATE_CONTENT_REASONING_FORMAT ||
    envelope.kind !== 'signature' ||
    (envelope.nativeItem && !isSignatureOnlyNativeItem(envelope.nativeItem))
  ) {
    return undefined;
  }
  return { toolCallId, envelope, encodedSignature };
}

export function containsReasoningTransportCarrier(value: unknown): boolean {
  if (typeof value === 'string') {
    return isCarrierPrefixed(value) || value.includes(GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR);
  }
  if (Array.isArray(value)) {
    return value.some(containsReasoningTransportCarrier);
  }
  return isRecord(value) && Object.values(value).some(containsReasoningTransportCarrier);
}

/** Performs bounded validation before a request is normalized or routed. */
export function validateReasoningTransportCarriers(
  value: unknown,
  bodyLimitBytes: number
): ReasoningCarrierValidationResult {
  const limits = reasoningCarrierLimits(bodyLimitBytes);
  let encodedBytes = 0;
  let itemCount = 0;
  const nativeIds = new Map<string, string>();
  const pairDirections = new Map<string, { call: boolean; result: boolean }>();
  const dependencies = new Map<string, string[]>();
  let failure: ReasoningCarrierValidationResult | undefined;

  const reject = (status: 400 | 413, code: string, message: string): void => {
    failure ??= { ok: false, status, code, message };
  };

  const visit = (candidate: unknown, parentKey?: string, depth = 1): void => {
    if (failure) {
      return;
    }
    if (depth > limits.maxJsonDepth) {
      reject(400, 'malformed_reasoning_carrier', 'Reasoning carrier JSON nesting exceeds the configured limit.');
      return;
    }
    if (typeof candidate === 'string') {
      const hasGeminiIdCarrier = parentKey === 'id' &&
        candidate.includes(GEMINI_THOUGHT_SIGNATURE_TOOL_CALL_ID_SEPARATOR);
      if (
        hasGeminiIdCarrier &&
        Buffer.byteLength(candidate, 'utf8') > limits.maxGeminiSignatureIdBytes
      ) {
        reject(413, 'tool_call_id_carrier_too_large', 'Gemini signature carrier in tool_call.id exceeds 64 KiB.');
        return;
      }
      const embedded = hasGeminiIdCarrier
        ? decodeGeminiThoughtSignatureToolCallId(candidate, limits)
        : undefined;
      if (hasGeminiIdCarrier && !embedded) {
        reject(400, 'invalid_tool_call_id_carrier', 'tool_call.id may contain only a Gemini signature envelope.');
        return;
      }
      const encoded = embedded?.encodedSignature || (isCarrierPrefixed(candidate) ? candidate : undefined);
      if (!encoded) {
        return;
      }
      const bytes = Buffer.byteLength(encoded, 'utf8');
      encodedBytes += bytes;
      if (encodedBytes > limits.maxEncodedBytes) {
        reject(413, 'carrier_too_large', 'Reasoning carrier total exceeds the configured limit.');
        return;
      }
      const decodedPayloadBytes = carrierDecodedPayloadBytes(encoded);
      if (decodedPayloadBytes === undefined) {
        reject(400, 'malformed_reasoning_carrier', 'Reasoning carrier is malformed.');
        return;
      }
      if (decodedPayloadBytes > limits.maxDecodedPayloadBytes) {
        reject(413, 'carrier_payload_too_large', 'A decoded reasoning carrier payload exceeds the configured limit.');
        return;
      }
      if (parentKey === 'id' && Buffer.byteLength(candidate, 'utf8') > limits.maxGeminiSignatureIdBytes) {
        reject(413, 'tool_call_id_carrier_too_large', 'Gemini signature carrier in tool_call.id exceeds 64 KiB.');
        return;
      }
      const envelope = decodeReasoningTransportEnvelope(encoded, limits);
      if (!envelope) {
        reject(400, 'malformed_reasoning_carrier', 'Reasoning carrier is malformed or exceeds structural limits.');
        return;
      }
      if (parentKey === 'id' && (!envelope.kind || envelope.kind !== 'signature' ||
          (envelope.nativeItem && !isSignatureOnlyNativeItem(envelope.nativeItem)))) {
        reject(400, 'invalid_tool_call_id_carrier', 'tool_call.id may contain only a Gemini signature envelope.');
        return;
      }
      const nativeItem = envelope.nativeItem;
      const nativeId = nativeItem?.native_id;
      if (nativeItem && nativeId) {
        const nativeFingerprint = JSON.stringify(nativeItem);
        const previous = nativeIds.get(nativeId);
        if (previous) {
          if (previous !== nativeFingerprint) {
            reject(400, 'duplicate_native_item_id', `Conflicting duplicate provider native item ID: ${nativeId}`);
          }
          // A provider object can project the same complete carrier into more
          // than one opaque field. It is still one native item, so validate its
          // graph edge only once while continuing to count encoded bytes.
          return;
        }
        nativeIds.set(nativeId, nativeFingerprint);
        dependencies.set(nativeId, nativeItem.depends_on || []);
      }
      itemCount += 1;
      if (itemCount > limits.maxItems) {
        reject(413, 'too_many_native_items', `Reasoning carrier contains more than ${limits.maxItems} native items.`);
        return;
      }
      if (!nativeItem) {
        return;
      }
      if (nativeItem.pair_id) {
        const direction = nativeItem.item_type.includes('result') || nativeItem.item_type.includes('output')
          ? 'result'
          : 'call';
        const pair = pairDirections.get(nativeItem.pair_id) || { call: false, result: false };
        if (pair[direction]) {
          reject(400, 'conflicting_native_pair', `Conflicting provider native pair: ${nativeItem.pair_id}`);
          return;
        }
        pair[direction] = true;
        pairDirections.set(nativeItem.pair_id, pair);
      }
      return;
    }
    if (Array.isArray(candidate)) {
      for (const entry of candidate) {
        visit(entry, parentKey, depth + 1);
      }
      return;
    }
    if (!isRecord(candidate)) {
      return;
    }
    for (const [key, entry] of Object.entries(candidate)) {
      visit(entry, key, depth + 1);
    }
  };

  visit(value);
  if (failure) {
    return failure;
  }
  if (hasDependencyCycle(dependencies)) {
    return { ok: false, status: 400, code: 'cyclic_native_dependency', message: 'Provider native item dependencies contain a cycle.' };
  }
  return { ok: true, itemCount, encodedBytes };
}

/** Validates all carrier-bearing JSON payloads in a complete set of SSE frames. */
export function validateReasoningTransportCarriersInSseFrames(
  frames: readonly string[],
  bodyLimitBytes: number
): ReasoningCarrierValidationResult {
  const payloads: unknown[] = [];
  for (const frame of frames) {
    for (const eventBlock of frame.split(/\r?\n\r?\n/)) {
      const data = eventBlock
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice('data:'.length).trimStart())
        .join('\n')
        .trim();
      if (!data || data === '[DONE]') {
        continue;
      }
      try {
        payloads.push(JSON.parse(data) as unknown);
      } catch {
        // Non-JSON SSE data cannot contain a structured provider-native carrier.
      }
    }
  }
  return validateReasoningTransportCarriers(payloads, bodyLimitBytes);
}

/**
 * Verifies that a v3 native item contains an actual provider object rather than
 * the legacy `{ data }` carrier projection. This is deliberately protocol
 * specific: a complete capture is necessary, but it is not sufficient to
 * authorize native replay.
 */
export function isProviderNativePayloadStructurallyValid(
  item: ProviderNativeItem,
  expectedFormat = item.source_format
): boolean {
  if (item.source_format !== expectedFormat || !isRecord(item.raw_payload)) {
    return false;
  }

  const payload = item.raw_payload;
  const itemType = item.item_type.toLowerCase();
  if (expectedFormat === OPENAI_RESPONSES_REASONING_FORMAT) {
    return asTrimmedString(payload.type)?.toLowerCase() === itemType;
  }

  if (expectedFormat === ANTHROPIC_CLAUDE_REASONING_FORMAT) {
    if (asTrimmedString(payload.type)?.toLowerCase() !== itemType) {
      return false;
    }
    if (itemType === 'thinking') {
      return typeof payload.thinking === 'string' && typeof payload.signature === 'string' &&
        payload.signature.length > 0;
    }
    if (itemType === 'redacted_thinking') {
      return typeof payload.data === 'string' && payload.data.length > 0;
    }
    return itemType === 'tool_use' || itemType === 'tool_result';
  }

  if (expectedFormat === GEMINI_GENERATE_CONTENT_REASONING_FORMAT) {
    const signature = asTrimmedString(payload.thoughtSignature) ||
      asTrimmedString(payload.thought_signature) ||
      (isRecord(payload.functionCall)
        ? asTrimmedString(payload.functionCall.thoughtSignature) ||
          asTrimmedString(payload.functionCall.thought_signature)
        : undefined);
    if (itemType === 'function_call') {
      const requiresSignature = /^gemini-3(?:[.-]|$)/i.test(item.source_origin.model?.trim() || '');
      return isRecord(payload.functionCall) && Boolean(asTrimmedString(payload.functionCall.name)) &&
        (!requiresSignature || Boolean(signature));
    }
    if (itemType === 'function_response') {
      return isRecord(payload.functionResponse);
    }
    if (itemType === 'thought') {
      return payload.thought === true && Boolean(signature);
    }
    if (itemType === 'part') {
      return typeof payload.text === 'string' && Boolean(signature);
    }
    if (itemType === 'thought_signature') {
      return Boolean(signature);
    }
    return false;
  }

  if (expectedFormat === GEMINI_INTERACTIONS_REASONING_FORMAT) {
    if (asTrimmedString(payload.type)?.toLowerCase() !== itemType) {
      return false;
    }
    return (itemType === 'thought' ||
      /^(?:code_execution|file_search|google_maps|google_search|retrieval)_(?:call|result)$/.test(itemType)) &&
      Boolean(asTrimmedString(payload.signature));
  }

  return false;
}

function buildProviderNativeItem(
  format: string,
  data: string,
  id: string | undefined,
  kind: ReasoningTransportEnvelope['kind'],
  origin: ReasoningStateOrigin,
  options: ReasoningTransportEncodeOptions
): ProviderNativeItem {
  const supplied = options.nativeItem;
  return {
    type: 'provider_native_item',
    item_type: supplied?.item_type || (kind === 'signature' ? 'thought_signature' : 'reasoning'),
    ...(supplied?.native_id || id ? { native_id: supplied?.native_id || id } : {}),
    raw_payload: supplied?.raw_payload && isRecord(supplied.raw_payload)
      ? supplied.raw_payload
      : { data },
    provider_schema_version: supplied?.provider_schema_version || options.providerSchemaVersion || format,
    ...(supplied?.item_origin ? { item_origin: supplied.item_origin } : {}),
    source_format: format,
    source_origin: origin,
    position: normalizePosition(supplied?.position || options.position),
    ...(supplied?.group_id || options.groupId ? { group_id: supplied?.group_id || options.groupId } : {}),
    ...(supplied?.call_id || options.callId ? { call_id: supplied?.call_id || options.callId } : {}),
    ...(supplied?.pair_id || options.pairId ? { pair_id: supplied?.pair_id || options.pairId } : {}),
    ...(supplied?.depends_on || options.dependsOn ? { depends_on: supplied?.depends_on || options.dependsOn } : {}),
    capture_state: supplied?.capture_state || options.captureState ||
      (supplied?.raw_payload && isRecord(supplied.raw_payload) ? 'complete' : 'partial'),
    ...(supplied?.provider_status || options.providerStatus
      ? { provider_status: supplied?.provider_status || options.providerStatus }
      : {}),
    ...(supplied?.provider_mode || options.providerMode
      ? { provider_mode: supplied?.provider_mode || options.providerMode }
      : {}),
    ...(supplied?.readable_text || options.readableText
      ? { readable_text: supplied?.readable_text || options.readableText }
      : {}),
    ...(supplied?.readable_summary || options.readableSummary
      ? { readable_summary: supplied?.readable_summary || options.readableSummary }
      : {}),
    ...(supplied?.compaction_mode || options.compactionMode
      ? { compaction_mode: supplied?.compaction_mode || options.compactionMode }
      : {})
  };
}

function decodeTransportEnvelopeWithPrefix(
  value: string,
  prefix: string,
  version: 1 | 2 | 3,
  limits: ReasoningCarrierLimits
): ReasoningTransportEnvelope | undefined {
  if (!value.startsWith(prefix)) {
    return undefined;
  }
  const encoded = value.slice(prefix.length);
  if (!isCanonicalBase64Url(encoded)) {
    return undefined;
  }
  let decodedText: string;
  try {
    const decodedBuffer = Buffer.from(encoded, 'base64url');
    if (decodedBuffer.byteLength > limits.maxDecodedPayloadBytes) {
      return undefined;
    }
    decodedText = decodedBuffer.toString('utf8');
  } catch {
    return undefined;
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(decodedText) as unknown;
  } catch {
    return undefined;
  }
  if (!isRecord(decoded) || jsonDepthExceeds(decoded, limits.maxJsonDepth)) {
    return undefined;
  }
  const format = asTrimmedString(decoded.format);
  const data = typeof decoded.data === 'string' ? decoded.data : '';
  const id = asTrimmedString(decoded.id);
  const kind = decoded.kind === 'signature' || decoded.kind === 'encrypted' ? decoded.kind : undefined;
  const origin = normalizeReasoningStateOrigin(decoded.origin);
  if (!format || !data || (version >= 2 && !origin)) {
    return undefined;
  }
  const nativeItem = version === 3 ? normalizeProviderNativeItem(decoded.native_item, format, origin) : undefined;
  if (version === 3 && (!nativeItem || decoded.version !== 3)) {
    return undefined;
  }
  return {
    format,
    data,
    ...(id ? { id } : {}),
    ...(kind ? { kind } : {}),
    ...(origin ? { origin } : {}),
    carrierVersion: version,
    ...(nativeItem ? { nativeItem } : {})
  };
}

function normalizeProviderNativeItem(
  value: unknown,
  format: string,
  envelopeOrigin: ReasoningStateOrigin | undefined
): ProviderNativeItem | undefined {
  if (!isRecord(value) || value.type !== 'provider_native_item' || !envelopeOrigin) {
    return undefined;
  }
  const itemType = asTrimmedString(value.item_type);
  const rawPayload = isRecord(value.raw_payload) ? value.raw_payload : undefined;
  const schemaVersion = asTrimmedString(value.provider_schema_version);
  const sourceFormat = asTrimmedString(value.source_format);
  const sourceOrigin = normalizeReasoningStateOrigin(value.source_origin);
  const position = normalizePositionFromUnknown(value.position);
  const captureState = value.capture_state;
  if (
    !itemType || !rawPayload || !schemaVersion || sourceFormat !== format || !sourceOrigin || !position ||
    (captureState !== 'complete' && captureState !== 'partial' && captureState !== 'interrupted') ||
    !sameOrigin(sourceOrigin, envelopeOrigin)
  ) {
    return undefined;
  }
  const dependsOn = Array.isArray(value.depends_on)
    ? value.depends_on.map(asTrimmedString).filter((entry): entry is string => Boolean(entry))
    : undefined;
  return {
    type: 'provider_native_item',
    item_type: itemType,
    ...(asTrimmedString(value.native_id) ? { native_id: asTrimmedString(value.native_id) } : {}),
    raw_payload: rawPayload,
    provider_schema_version: schemaVersion,
    ...(value.item_origin === 'native' || value.item_origin === 'converted' || value.item_origin === 'synthetic'
      ? { item_origin: value.item_origin }
      : {}),
    source_format: sourceFormat,
    source_origin: sourceOrigin,
    position,
    ...(asTrimmedString(value.group_id) ? { group_id: asTrimmedString(value.group_id) } : {}),
    ...(asTrimmedString(value.call_id) ? { call_id: asTrimmedString(value.call_id) } : {}),
    ...(asTrimmedString(value.pair_id) ? { pair_id: asTrimmedString(value.pair_id) } : {}),
    ...(dependsOn && dependsOn.length > 0 ? { depends_on: dependsOn } : {}),
    capture_state: captureState,
    ...(asTrimmedString(value.provider_status) ? { provider_status: asTrimmedString(value.provider_status) } : {}),
    ...(asTrimmedString(value.provider_mode) ? { provider_mode: asTrimmedString(value.provider_mode) } : {}),
    ...(typeof value.readable_text === 'string' ? { readable_text: value.readable_text } : {}),
    ...(typeof value.readable_summary === 'string' ? { readable_summary: value.readable_summary } : {}),
    ...(value.compaction_mode === 'standalone' || value.compaction_mode === 'server_side'
      ? { compaction_mode: value.compaction_mode }
      : {})
  };
}

function encodeLegacyTransportEnvelope(
  format: string,
  data: string,
  id: string | undefined,
  kind: ReasoningTransportEnvelope['kind']
): string {
  const payload = Buffer.from(JSON.stringify({
    format,
    data,
    ...(id ? { id } : {}),
    ...(kind ? { kind } : {})
  }), 'utf8').toString('base64url');
  return `${REASONING_TRANSPORT_ENVELOPE_PREFIX}${payload}`;
}

function normalizeReasoningStateOrigin(value: unknown): ReasoningStateOrigin | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const provider = asTrimmedString(value.provider);
  const endpoint = asTrimmedString(value.endpoint);
  const model = asTrimmedString(value.model);
  const credentialScope = asTrimmedString(value.credentialScope);
  if (!provider || !endpoint) {
    return undefined;
  }
  return {
    provider,
    endpoint,
    ...(model ? { model } : {}),
    ...(credentialScope ? { credentialScope } : {})
  };
}

function normalizePosition(value: ProviderNativeItemPosition | undefined): ProviderNativeItemPosition {
  return value && Number.isInteger(value.turn) && Number.isInteger(value.step) && Number.isInteger(value.item)
    ? { turn: value.turn, step: value.step, item: value.item }
    : { turn: 0, step: 0, item: 0 };
}

function normalizePositionFromUnknown(value: unknown): ProviderNativeItemPosition | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const turn = typeof value.turn === 'number' ? value.turn : NaN;
  const step = typeof value.step === 'number' ? value.step : NaN;
  const item = typeof value.item === 'number' ? value.item : NaN;
  return Number.isInteger(turn) && turn >= 0 && Number.isInteger(step) && step >= 0 && Number.isInteger(item) && item >= 0
    ? { turn, step, item }
    : undefined;
}

function isSignatureOnlyNativeItem(item: ProviderNativeItem): boolean {
  if (item.item_type !== 'thought_signature' && item.item_type !== 'signature') {
    return false;
  }
  return Object.keys(item.raw_payload).every((key) => key === 'data' || key === 'signature');
}

function isCarrierPrefixed(value: string): boolean {
  return value.startsWith(REASONING_TRANSPORT_ENVELOPE_V3_PREFIX) ||
    value.startsWith(REASONING_TRANSPORT_ENVELOPE_V2_PREFIX) ||
    value.startsWith(REASONING_TRANSPORT_ENVELOPE_PREFIX) ||
    value.startsWith(OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX);
}

function carrierDecodedPayloadBytes(value: string): number | undefined {
  const prefixes = [
    REASONING_TRANSPORT_ENVELOPE_V3_PREFIX,
    REASONING_TRANSPORT_ENVELOPE_V2_PREFIX,
    REASONING_TRANSPORT_ENVELOPE_PREFIX,
    OPENAI_RESPONSES_REASONING_ENVELOPE_PREFIX
  ];
  const prefix = prefixes.find((candidate) => value.startsWith(candidate));
  if (!prefix) {
    return undefined;
  }
  const encoded = value.slice(prefix.length);
  if (!isCanonicalBase64Url(encoded)) {
    return undefined;
  }
  try {
    return Buffer.from(encoded, 'base64url').byteLength;
  } catch {
    return undefined;
  }
}

function jsonDepthExceeds(value: unknown, maximumDepth: number): boolean {
  const pending: Array<{ value: unknown; depth: number }> = [{ value, depth: 1 }];
  while (pending.length > 0) {
    const current = pending.pop()!;
    if (current.depth > maximumDepth) {
      return true;
    }
    if (Array.isArray(current.value)) {
      for (const entry of current.value) {
        pending.push({ value: entry, depth: current.depth + 1 });
      }
    } else if (isRecord(current.value)) {
      for (const entry of Object.values(current.value)) {
        pending.push({ value: entry, depth: current.depth + 1 });
      }
    }
  }
  return false;
}

function isCanonicalBase64Url(value: string): boolean {
  if (!value || value.length % 4 === 1 || !/^[A-Za-z0-9_-]+$/.test(value)) {
    return false;
  }
  try {
    return Buffer.from(value, 'base64url').toString('base64url') === value;
  } catch {
    return false;
  }
}

function hasDependencyCycle(graph: Map<string, string[]>): boolean {
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (id: string): boolean => {
    if (visiting.has(id)) {
      return true;
    }
    if (visited.has(id)) {
      return false;
    }
    visiting.add(id);
    for (const dependency of graph.get(id) || []) {
      if (graph.has(dependency) && visit(dependency)) {
        return true;
      }
    }
    visiting.delete(id);
    visited.add(id);
    return false;
  };
  return [...graph.keys()].some(visit);
}

function sameOrigin(left: ReasoningStateOrigin, right: ReasoningStateOrigin): boolean {
  return left.provider === right.provider && left.endpoint === right.endpoint &&
    left.model === right.model && left.credentialScope === right.credentialScope;
}

function decodeBase64Json(value: string): unknown {
  if (!isCanonicalBase64Url(value)) {
    return undefined;
  }
  try {
    return JSON.parse(Buffer.from(value, 'base64url').toString('utf8')) as unknown;
  } catch {
    return undefined;
  }
}

function asTrimmedString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
