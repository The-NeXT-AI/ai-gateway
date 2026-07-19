import {
  createCipheriv,
  createDecipheriv,
  createHash,
  createHmac,
  randomBytes,
  timingSafeEqual
} from 'node:crypto';
import type { Provider, ProviderType } from '../types';
import { asNumber, isObject } from '../utils';

export type VideoApiProtocol = 'openai' | 'xai';

export interface GatewayVideoReference {
  version: 2;
  upstreamId: string;
  sourceProtocol: VideoApiProtocol;
  targetProtocol: VideoApiProtocol;
  targetProvider: Provider;
  targetProviderName?: string;
  targetProviderKey?: string;
  targetCredentialId?: string;
  model?: string;
  duration?: number;
  size?: string;
  createdAt: number;
  ownerKey?: string;
}

export interface VideoIdCodecOptions {
  signingSecret?: string;
  ttlMs?: number;
}

export interface VideoCreateMetadata {
  duration?: number;
  size?: string;
}

interface GatewayVideoIdPayload {
  v: 2 | 3;
  u: string;
  s: VideoApiProtocol;
  t: VideoApiProtocol;
  p: Provider;
  n?: string;
  k?: string;
  r?: string;
  m?: string;
  d?: number;
  z?: string;
  c: number;
  e: number;
  o?: string;
}

const gatewayVideoIdPrefix = 'gv3';
const legacyGatewayVideoIdPrefix = 'gv2';
const defaultVideoReferenceTtlMs = 24 * 60 * 60 * 1000;
const maxGatewayVideoIdLength = 8_192;
const fallbackVideoIdSecret = randomBytes(32).toString('base64url');
const gatewayVideoIdEncryptionAlgorithm = 'aes-256-gcm';
const gatewayVideoIdNonceBytes = 12;
const gatewayVideoIdAuthTagBytes = 16;
const gatewayVideoIdAdditionalData = Buffer.from('next-ai-gateway:video-id:gv3', 'utf8');
const maxVideoReferences = 10_000;
const openAIToXAIConvertibleFields = new Set([
  'model',
  'prompt',
  'seconds',
  'size',
  'input_reference'
]);
const xaiToOpenAIConvertibleFields = new Set([
  'model',
  'prompt',
  'duration',
  'aspect_ratio',
  'resolution',
  'image',
  'reference_images'
]);
const videoReferences = new Map<
  string,
  { reference: GatewayVideoReference; expiresAt: number }
>();
const videoBillingClaims = new Map<
  string,
  { state: 'pending' | 'completed'; expiresAt: number }
>();

export function videoProtocolForTarget(
  provider: Provider,
  providerType: ProviderType | undefined,
  fallback: VideoApiProtocol
): VideoApiProtocol {
  if (providerType === 'xai_video_generations' || provider === 'xai') {
    return 'xai';
  }
  if (providerType === 'openai_video_generations' || provider === 'openai') {
    return 'openai';
  }
  return fallback;
}

export function convertVideoCreateBody(
  body: Record<string, unknown>,
  source: VideoApiProtocol,
  target: VideoApiProtocol
): Record<string, unknown> {
  if (source === target) {
    return { ...body };
  }

  const conversionError = validateVideoCreateConversion(body, source, target);
  if (conversionError) {
    throw new Error(conversionError);
  }

  if (source === 'openai') {
    return convertOpenAIVideoCreateBodyToXAI(body);
  }
  return convertXAIVideoCreateBodyToOpenAI(body);
}

export function validateVideoCreateConversion(
  body: Record<string, unknown>,
  source: VideoApiProtocol,
  target: VideoApiProtocol
): string | undefined {
  if (source === target) {
    return undefined;
  }

  if (source === 'openai') {
    const duration = readPositiveNumber(body.seconds);
    if (body.seconds !== undefined && (duration === undefined || ![4, 8, 12].includes(duration))) {
      return 'OpenAI video seconds must be 4, 8, or 12 for cross-provider conversion.';
    }
    const size = readNonEmptyString(body.size);
    if (size && !openAISizeToXAIFormat(size)) {
      return `OpenAI video size "${size}" cannot be represented exactly by xAI. Cross-provider conversion supports 720x1280 and 1280x720.`;
    }
    if (body.input_reference !== undefined) {
      const reference = isObject(body.input_reference) ? body.input_reference : undefined;
      const imageUrl = readNonEmptyString(reference?.image_url);
      const fileId = readNonEmptyString(reference?.file_id);
      if (!reference || Boolean(imageUrl) === Boolean(fileId)) {
        return 'OpenAI input_reference must contain exactly one of image_url or file_id for xAI conversion.';
      }
      if (fileId) {
        return 'OpenAI input_reference.file_id cannot be reused with xAI. Use input_reference.image_url for cross-provider conversion.';
      }
    }
    const unsupportedFields = findUnsupportedVideoFields(
      body,
      openAIToXAIConvertibleFields
    );
    if (unsupportedFields.length > 0) {
      return `OpenAI video fields cannot be converted to xAI without data loss: ${unsupportedFields.join(', ')}.`;
    }
    return undefined;
  }

  const duration = readPositiveNumber(body.duration);
  if (duration === undefined || ![4, 8, 12].includes(duration)) {
    if (body.duration === undefined) {
      return 'xAI duration must be explicit for OpenAI conversion and must be 4, 8, or 12 seconds.';
    }
    return `xAI video duration "${String(body.duration)}" cannot be represented by OpenAI, which supports 4, 8, or 12 seconds.`;
  }

  const aspectRatio = readNonEmptyString(body.aspect_ratio);
  const resolution = readNonEmptyString(body.resolution);
  if (!aspectRatio || !resolution) {
    return 'xAI aspect_ratio and resolution must be explicit for OpenAI conversion; use 16:9 or 9:16 at 720p.';
  }
  if ((aspectRatio || resolution) && !xaiFormatToOpenAISize(aspectRatio, resolution)) {
    return 'xAI video format cannot be represented exactly by OpenAI. Cross-provider conversion supports 16:9 or 9:16 at 720p.';
  }

  const hasReferenceImages = Array.isArray(body.reference_images)
    ? body.reference_images.length > 0
    : body.reference_images !== undefined;
  if (body.image !== undefined || hasReferenceImages) {
    return 'xAI image and reference_images inputs cannot be converted to OpenAI because OpenAI input_reference requires a multipart file upload.';
  }

  const unsupportedFields = findUnsupportedVideoFields(
    body,
    xaiToOpenAIConvertibleFields
  );
  if (unsupportedFields.length > 0) {
    return `xAI video fields cannot be converted to OpenAI without data loss: ${unsupportedFields.join(', ')}.`;
  }

  return undefined;
}

export function readVideoCreateMetadata(
  body: Record<string, unknown>,
  protocol: VideoApiProtocol
): VideoCreateMetadata {
  if (protocol === 'openai') {
    return {
      duration: readPositiveNumber(body.seconds) ?? 4,
      size: readNonEmptyString(body.size) ?? '720x1280'
    };
  }

  return {
    duration: readPositiveNumber(body.duration),
    size: xaiFormatToOpenAISize(
      readNonEmptyString(body.aspect_ratio),
      readNonEmptyString(body.resolution)
    )
  };
}

export function encodeGatewayVideoId(
  reference: GatewayVideoReference,
  options: VideoIdCodecOptions = {}
): string {
  const now = Date.now();
  pruneVideoReferences(now);
  const providerKey = reference.targetProviderName
    ? videoProviderKey(reference.targetProviderName)
    : undefined;
  const payload: GatewayVideoIdPayload = {
    v: 3,
    u: reference.upstreamId,
    s: reference.sourceProtocol,
    t: reference.targetProtocol,
    p: reference.targetProvider,
    n: reference.targetProviderName,
    k: providerKey,
    r: reference.targetCredentialId,
    m: reference.model,
    d: reference.duration,
    z: reference.size,
    c: Math.max(0, Math.floor(reference.createdAt)),
    e: Math.ceil((now + normalizeVideoIdTtlMs(options.ttlMs)) / 1000),
    o: reference.ownerKey
  };
  const nonce = randomBytes(gatewayVideoIdNonceBytes);
  const cipher = createCipheriv(
    gatewayVideoIdEncryptionAlgorithm,
    deriveVideoIdEncryptionKey(options.signingSecret),
    nonce
  );
  cipher.setAAD(gatewayVideoIdAdditionalData);
  const encryptedPayload = Buffer.concat([
    cipher.update(JSON.stringify(payload), 'utf8'),
    cipher.final()
  ]);
  const publicId = [
    gatewayVideoIdPrefix,
    nonce.toString('base64url'),
    encryptedPayload.toString('base64url'),
    cipher.getAuthTag().toString('base64url')
  ].join('.');
  videoReferences.set(publicId, {
    reference: { ...reference, targetProviderKey: providerKey },
    expiresAt: payload.e * 1000
  });
  pruneVideoReferences(now);
  return publicId;
}

export function decodeGatewayVideoId(
  value: string,
  options: VideoIdCodecOptions = {}
): GatewayVideoReference | undefined {
  const now = Date.now();
  if (value.length > maxGatewayVideoIdLength) {
    return undefined;
  }
  try {
    const payload = value.startsWith(`${gatewayVideoIdPrefix}.`)
      ? decryptGatewayVideoIdPayload(value, options.signingSecret)
      : value.startsWith(`${legacyGatewayVideoIdPrefix}.`)
        ? verifyLegacyGatewayVideoIdPayload(value, options.signingSecret)
        : undefined;
    if (!payload || !isValidVideoIdPayload(payload, now)) {
      return undefined;
    }
    const entry = videoReferences.get(value);
    if (entry && entry.expiresAt > now) {
      return entry.reference;
    }
    if (entry) {
      videoReferences.delete(value);
    }
    const reference: GatewayVideoReference = {
      version: 2,
      upstreamId: payload.u,
      sourceProtocol: payload.s,
      targetProtocol: payload.t,
      targetProvider: payload.p,
      targetProviderName: payload.n,
      targetProviderKey: payload.k,
      targetCredentialId: payload.r,
      model: payload.m,
      duration: payload.d,
      size: payload.z,
      createdAt: payload.c,
      ownerKey: payload.o
    };
    videoReferences.set(value, { reference, expiresAt: payload.e * 1000 });
    pruneVideoReferences(now);
    return reference;
  } catch {
    return undefined;
  }
}

export function isGatewayVideoId(value: string): boolean {
  return (
    value.startsWith(`${gatewayVideoIdPrefix}.`) ||
    value.startsWith(`${legacyGatewayVideoIdPrefix}.`) ||
    /^gv1[ox][ox]\./.test(value)
  );
}

export function gatewayVideoIdPrefixForTests(): string {
  return gatewayVideoIdPrefix;
}

export function resetGatewayVideoReferencesForTests(): void {
  videoReferences.clear();
  videoBillingClaims.clear();
}

export function claimVideoBillingEvent(publicId: string, ttlMs?: number): boolean {
  const now = Date.now();
  pruneVideoBillingClaims(now);
  if (videoBillingClaims.has(publicId)) {
    return false;
  }
  videoBillingClaims.set(publicId, {
    state: 'pending',
    expiresAt: now + normalizeVideoIdTtlMs(ttlMs)
  });
  return true;
}

export function completeVideoBillingEvent(publicId: string): void {
  const claim = videoBillingClaims.get(publicId);
  if (claim) {
    claim.state = 'completed';
  }
}

export function releaseVideoBillingEvent(publicId: string): void {
  if (videoBillingClaims.get(publicId)?.state === 'pending') {
    videoBillingClaims.delete(publicId);
  }
}

export function videoProviderKey(providerName: string): string {
  return createHash('sha256').update(providerName.trim().toLowerCase()).digest('base64url').slice(0, 11);
}

export function videoOwnerKey(subject: string | undefined): string | undefined {
  const normalized = subject?.trim();
  return normalized
    ? createHash('sha256').update(normalized).digest('base64url').slice(0, 22)
    : undefined;
}

function convertOpenAIVideoCreateBodyToXAI(
  body: Record<string, unknown>
): Record<string, unknown> {
  const converted = copyVideoFields(body, ['model', 'prompt']);

  converted.duration = readPositiveNumber(body.seconds) ?? 4;

  const size = readNonEmptyString(body.size) ?? '720x1280';
  const format = openAISizeToXAIFormat(size);
  if (format) {
    converted.aspect_ratio = format.aspectRatio;
    converted.resolution = format.resolution;
  }

  const inputReference = isObject(body.input_reference) ? body.input_reference : undefined;
  const imageUrl =
    readNonEmptyString(inputReference?.image_url) || readNonEmptyString(inputReference?.url);
  if (imageUrl) {
    converted.image = { url: imageUrl };
  }

  return converted;
}

function convertXAIVideoCreateBodyToOpenAI(
  body: Record<string, unknown>
): Record<string, unknown> {
  const converted = copyVideoFields(body, ['model', 'prompt']);

  const duration = readPositiveNumber(body.duration);
  if (duration !== undefined) {
    converted.seconds = String(duration);
  }

  const size = xaiFormatToOpenAISize(
    readNonEmptyString(body.aspect_ratio),
    readNonEmptyString(body.resolution)
  );
  if (size) {
    converted.size = size;
  }

  return converted;
}

function findUnsupportedVideoFields(
  body: Record<string, unknown>,
  allowedFields: ReadonlySet<string>
): string[] {
  return Object.keys(body)
    .filter((field) => !allowedFields.has(field))
    .sort();
}

function copyVideoFields(
  body: Record<string, unknown>,
  fields: readonly string[]
): Record<string, unknown> {
  const copied: Record<string, unknown> = {};
  for (const field of fields) {
    if (body[field] !== undefined) {
      copied[field] = body[field];
    }
  }
  return copied;
}

function openAISizeToXAIFormat(
  value: string
): { aspectRatio: string; resolution: string } | undefined {
  switch (value.trim().toLowerCase()) {
    case '720x1280':
      return { aspectRatio: '9:16', resolution: '720p' };
    case '1280x720':
      return { aspectRatio: '16:9', resolution: '720p' };
    default:
      return undefined;
  }
}

function xaiFormatToOpenAISize(
  aspectRatio: string | undefined,
  resolution: string | undefined
): string | undefined {
  if (!aspectRatio && !resolution) {
    return undefined;
  }
  const normalizedAspectRatio = (aspectRatio || '16:9').trim();
  const normalizedResolution = (resolution || '720p').trim().toLowerCase();
  if (normalizedResolution !== '720p') {
    return undefined;
  }
  if (normalizedAspectRatio === '16:9') {
    return '1280x720';
  }
  if (normalizedAspectRatio === '9:16') {
    return '720x1280';
  }
  return undefined;
}

function readPositiveNumber(value: unknown): number | undefined {
  const parsed =
    asNumber(value) ??
    (typeof value === 'string' && value.trim() ? Number(value.trim()) : undefined);
  return parsed !== undefined && Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function readNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function pruneVideoReferences(now: number): void {
  for (const [key, entry] of videoReferences) {
    if (entry.expiresAt <= now) {
      videoReferences.delete(key);
    }
  }
  while (videoReferences.size >= maxVideoReferences) {
    const oldest = videoReferences.keys().next().value as string | undefined;
    if (!oldest) {
      break;
    }
    videoReferences.delete(oldest);
  }
}

function pruneVideoBillingClaims(now: number): void {
  for (const [key, claim] of videoBillingClaims) {
    if (claim.expiresAt <= now) {
      videoBillingClaims.delete(key);
    }
  }
  while (videoBillingClaims.size >= maxVideoReferences) {
    const oldest = videoBillingClaims.keys().next().value as string | undefined;
    if (!oldest) {
      break;
    }
    videoBillingClaims.delete(oldest);
  }
}

function normalizeVideoIdTtlMs(value: number | undefined): number {
  return Number.isFinite(value) && (value || 0) > 0
    ? Math.floor(value as number)
    : defaultVideoReferenceTtlMs;
}

function deriveVideoIdEncryptionKey(signingSecret: string | undefined): Buffer {
  return createHash('sha256')
    .update('next-ai-gateway:video-id-encryption-key:gv3\0', 'utf8')
    .update(signingSecret || fallbackVideoIdSecret, 'utf8')
    .digest();
}

function decryptGatewayVideoIdPayload(
  value: string,
  signingSecret: string | undefined
): Partial<GatewayVideoIdPayload> | undefined {
  const match = value.match(
    /^gv3\.([A-Za-z0-9_-]{16})\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]{22})$/
  );
  if (!match) {
    return undefined;
  }
  const nonce = decodeCanonicalBase64Url(match[1] || '');
  const encryptedPayload = decodeCanonicalBase64Url(match[2] || '');
  const authTag = decodeCanonicalBase64Url(match[3] || '');
  if (
    !nonce ||
    !encryptedPayload ||
    !authTag ||
    nonce.byteLength !== gatewayVideoIdNonceBytes ||
    encryptedPayload.byteLength === 0 ||
    authTag.byteLength !== gatewayVideoIdAuthTagBytes
  ) {
    return undefined;
  }
  const decipher = createDecipheriv(
    gatewayVideoIdEncryptionAlgorithm,
    deriveVideoIdEncryptionKey(signingSecret),
    nonce
  );
  decipher.setAAD(gatewayVideoIdAdditionalData);
  decipher.setAuthTag(authTag);
  const payload = Buffer.concat([
    decipher.update(encryptedPayload),
    decipher.final()
  ]).toString('utf8');
  return JSON.parse(payload) as Partial<GatewayVideoIdPayload>;
}

function decodeCanonicalBase64Url(value: string): Buffer | undefined {
  const decoded = Buffer.from(value, 'base64url');
  return decoded.toString('base64url') === value ? decoded : undefined;
}

function verifyLegacyGatewayVideoIdPayload(
  value: string,
  signingSecret: string | undefined
): Partial<GatewayVideoIdPayload> | undefined {
  const match = value.match(/^gv2\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]{43})$/);
  if (!match) {
    return undefined;
  }
  const unsignedId = `${legacyGatewayVideoIdPrefix}.${match[1]}`;
  if (!verifyVideoIdSignature(unsignedId, match[2] || '', signingSecret)) {
    return undefined;
  }
  return JSON.parse(
    Buffer.from(match[1] || '', 'base64url').toString('utf8')
  ) as Partial<GatewayVideoIdPayload>;
}

function signVideoId(unsignedId: string, signingSecret: string | undefined): string {
  return createHmac('sha256', signingSecret || fallbackVideoIdSecret)
    .update(unsignedId)
    .digest('base64url');
}

function verifyVideoIdSignature(
  unsignedId: string,
  signature: string,
  signingSecret: string | undefined
): boolean {
  const expected = Buffer.from(signVideoId(unsignedId, signingSecret), 'ascii');
  const actual = Buffer.from(signature, 'ascii');
  return expected.length === actual.length && timingSafeEqual(expected, actual);
}

function isValidVideoIdPayload(
  payload: Partial<GatewayVideoIdPayload>,
  now: number
): payload is GatewayVideoIdPayload {
  return (
    (payload.v === 2 || payload.v === 3) &&
    typeof payload.u === 'string' &&
    Boolean(payload.u.trim()) &&
    (payload.s === 'openai' || payload.s === 'xai') &&
    (payload.t === 'openai' || payload.t === 'xai') &&
    typeof payload.p === 'string' &&
    Boolean(payload.p.trim()) &&
    Number.isSafeInteger(payload.c) &&
    Number.isSafeInteger(payload.e) &&
    (payload.e as number) * 1000 > now &&
    (payload.n === undefined || typeof payload.n === 'string') &&
    (payload.k === undefined || typeof payload.k === 'string') &&
    (payload.r === undefined || typeof payload.r === 'string') &&
    (payload.m === undefined || typeof payload.m === 'string') &&
    (payload.d === undefined || (typeof payload.d === 'number' && Number.isFinite(payload.d))) &&
    (payload.z === undefined || typeof payload.z === 'string') &&
    (payload.o === undefined || typeof payload.o === 'string')
  );
}
