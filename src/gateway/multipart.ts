import type { FastifyRequest } from 'fastify';
import { readHeader } from '../utils';

const multipartHeaderSeparator = Buffer.from('\r\n\r\n');
const multipartHeaderSeparatorLf = Buffer.from('\n\n');
const multipartLineBreak = Buffer.from('\r\n');
const multipartLineBreakLf = Buffer.from('\n');
const maxMultipartBoundaryLength = 200;
const maxMultipartHeaderBytes = 16 * 1024;
const maxMultipartMetadataFieldBytes = 256 * 1024;
const metadataFieldNames = new Set(['model', 'prompt', 'seconds', 'size']);
const requestMetadataCache = new WeakMap<FastifyRequest, OpenAIMultipartMetadataResult>();

export interface OpenAIMultipartMetadata {
  fields: Record<string, string>;
  imageCount: number;
  unsupportedFields: string[];
  fileFields: string[];
}

export type OpenAIMultipartMetadataResult =
  | { ok: true; value: OpenAIMultipartMetadata }
  | { ok: false; error: string };

export function readOpenAIMultipartRequestMetadata(
  request: FastifyRequest
): OpenAIMultipartMetadataResult | undefined {
  if (!Buffer.isBuffer(request.body)) {
    return undefined;
  }
  const contentType = readHeader(request.headers['content-type']);
  if (!/^multipart\/form-data(?:\s*;|$)/i.test(contentType?.trim() || '')) {
    return undefined;
  }

  const cached = requestMetadataCache.get(request);
  if (cached) {
    return cached;
  }
  const parsed = parseOpenAIMultipartMetadata(request.body, contentType || '');
  requestMetadataCache.set(request, parsed);
  return parsed;
}

export function parseOpenAIMultipartMetadata(
  body: Buffer,
  contentType: string
): OpenAIMultipartMetadataResult {
  const boundary = readMultipartBoundary(contentType);
  if (!boundary) {
    return { ok: false, error: 'Multipart request is missing a valid boundary.' };
  }

  const boundaryMarker = Buffer.from(`--${boundary}`);
  const boundaryWithCrlf = Buffer.concat([multipartLineBreak, boundaryMarker]);
  const boundaryWithLf = Buffer.concat([multipartLineBreakLf, boundaryMarker]);
  const fields: Record<string, string> = {};
  const unsupportedFields = new Set<string>();
  const fileFields = new Set<string>();
  let imageCount = 0;
  let boundaryIndex = body.indexOf(boundaryMarker);

  if (boundaryIndex < 0) {
    return { ok: false, error: 'Multipart request body does not contain its declared boundary.' };
  }

  while (boundaryIndex >= 0) {
    let cursor = boundaryIndex + boundaryMarker.length;
    if (body[cursor] === 45 && body[cursor + 1] === 45) {
      return {
        ok: true,
        value: {
          fields,
          imageCount,
          unsupportedFields: [...unsupportedFields],
          fileFields: [...fileFields]
        }
      };
    }

    if (body[cursor] === 13 && body[cursor + 1] === 10) {
      cursor += 2;
    } else if (body[cursor] === 10) {
      cursor += 1;
    } else {
      return { ok: false, error: 'Multipart boundary is not followed by a line break.' };
    }

    const headerEndCrlf = body.indexOf(multipartHeaderSeparator, cursor);
    const headerEndLf = body.indexOf(multipartHeaderSeparatorLf, cursor);
    const headerEnd = firstNonNegative(headerEndCrlf, headerEndLf);
    if (headerEnd < 0 || headerEnd - cursor > maxMultipartHeaderBytes) {
      return { ok: false, error: 'Multipart part headers are missing or too large.' };
    }

    const headerSeparatorLength = headerEnd === headerEndCrlf ? multipartHeaderSeparator.length : 2;
    const dataStart = headerEnd + headerSeparatorLength;
    const nextBoundaryCrlf = body.indexOf(boundaryWithCrlf, dataStart);
    const nextBoundaryLf = body.indexOf(boundaryWithLf, dataStart);
    const nextBoundary = firstNonNegative(nextBoundaryCrlf, nextBoundaryLf);
    if (nextBoundary < 0) {
      return { ok: false, error: 'Multipart request body is missing its closing boundary.' };
    }

    const headerText = body.toString('latin1', cursor, headerEnd);
    const dispositionHeaders = readPartHeaders(headerText, 'content-disposition');
    if (dispositionHeaders.length !== 1) {
      return {
        ok: false,
        error: 'Each multipart part must contain exactly one Content-Disposition header.'
      };
    }
    const dispositionResult = readDispositionParameters(dispositionHeaders[0]);
    if (!dispositionResult.ok) {
      return dispositionResult;
    }
    const { fieldName, filename } = dispositionResult.value;
    if (!fieldName) {
      return { ok: false, error: 'Each multipart form-data part must have a name.' };
    }
    const canonicalControlFieldName = fieldName.trim().toLowerCase();
    if (
      metadataFieldNames.has(canonicalControlFieldName) &&
      fieldName !== canonicalControlFieldName
    ) {
      return {
        ok: false,
        error: `Multipart field "${fieldName}" must use the exact lowercase name "${canonicalControlFieldName}".`
      };
    }
    const contentTypeHeaders = readPartHeaders(headerText, 'content-type');
    if (contentTypeHeaders.length > 1) {
      return {
        ok: false,
        error: `Multipart field "${fieldName}" must not repeat the Content-Type header.`
      };
    }
    const partContentType = contentTypeHeaders[0]?.toLowerCase();
    const dataLength = nextBoundary - dataStart;

    if (isImagePart(fieldName, filename, partContentType)) {
      imageCount += 1;
    }

    if (filename !== undefined && metadataFieldNames.has(fieldName)) {
      return {
        ok: false,
        error: `Multipart field "${fieldName}" must be a text field without a filename.`
      };
    }

    if (filename !== undefined) {
      fileFields.add(fieldName);
    } else if (metadataFieldNames.has(fieldName)) {
      if (dataLength > maxMultipartMetadataFieldBytes) {
        return {
          ok: false,
          error: `Multipart field "${fieldName}" is too large.`
        };
      }
      if (Object.prototype.hasOwnProperty.call(fields, fieldName)) {
        return {
          ok: false,
          error: `Multipart field "${fieldName}" must not be repeated.`
        };
      }
      fields[fieldName] = body.toString('utf8', dataStart, nextBoundary);
    } else {
      unsupportedFields.add(fieldName);
    }

    boundaryIndex =
      nextBoundary + (nextBoundary === nextBoundaryCrlf ? multipartLineBreak.length : 1);
  }

  return { ok: false, error: 'Multipart request body is malformed.' };
}

function readMultipartBoundary(contentType: string): string | undefined {
  if (!/^multipart\/form-data(?:\s*;|$)/i.test(contentType.trim())) {
    return undefined;
  }

  const match = contentType.match(/(?:^|;)\s*boundary\s*=\s*(?:"([^"]*)"|([^;]*))/i);
  const boundary = (match?.[1] ?? match?.[2])?.trim();
  if (
    !boundary ||
    boundary.length > maxMultipartBoundaryLength ||
    boundary.includes('\r') ||
    boundary.includes('\n')
  ) {
    return undefined;
  }
  return boundary;
}

function readPartHeaders(headers: string, name: string): string[] {
  const target = name.toLowerCase();
  const values: string[] = [];
  for (const line of headers.split(/\r?\n/)) {
    const separator = line.indexOf(':');
    if (separator <= 0 || line.slice(0, separator).trim().toLowerCase() !== target) {
      continue;
    }
    values.push(line.slice(separator + 1).trim());
  }
  return values;
}

function readDispositionParameters(
  value: string
):
  | { ok: true; value: { fieldName?: string; filename?: string } }
  | { ok: false; error: string } {
  const segments: string[] = [];
  let segmentStart = 0;
  let quoted = false;
  let escaped = false;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && character === '\\') {
      escaped = true;
      continue;
    }
    if (character === '"') {
      quoted = !quoted;
      continue;
    }
    if (!quoted && character === ';') {
      segments.push(value.slice(segmentStart, index).trim());
      segmentStart = index + 1;
    }
  }
  if (quoted) {
    return { ok: false, error: 'Multipart Content-Disposition contains an unterminated quote.' };
  }
  segments.push(value.slice(segmentStart).trim());
  if (segments[0]?.toLowerCase() !== 'form-data') {
    return { ok: false, error: 'Multipart Content-Disposition must use form-data.' };
  }

  const parameters = new Map<string, string>();
  for (const segment of segments.slice(1)) {
    const separator = segment.indexOf('=');
    if (separator <= 0) {
      continue;
    }
    const name = segment.slice(0, separator).trim().toLowerCase();
    if (name !== 'name' && name !== 'filename') {
      continue;
    }
    if (parameters.has(name)) {
      return {
        ok: false,
        error: `Multipart Content-Disposition parameter "${name}" must not be repeated.`
      };
    }
    const raw = segment.slice(separator + 1).trim();
    if (!raw) {
      parameters.set(name, '');
      continue;
    }
    if (raw.startsWith('"')) {
      if (!raw.endsWith('"') || raw.length < 2) {
        return {
          ok: false,
          error: `Multipart Content-Disposition parameter "${name}" is malformed.`
        };
      }
      parameters.set(name, raw.slice(1, -1).replace(/\\(["\\])/g, '$1'));
      continue;
    }
    if (/\s/.test(raw)) {
      return {
        ok: false,
        error: `Multipart Content-Disposition parameter "${name}" is malformed.`
      };
    }
    parameters.set(name, raw);
  }

  return {
    ok: true,
    value: {
      fieldName: parameters.get('name'),
      filename: parameters.get('filename')
    }
  };
}

function isImagePart(
  fieldName: string | undefined,
  filename: string | undefined,
  contentType: string | undefined
): boolean {
  if (contentType?.startsWith('image/')) {
    return true;
  }
  if (!fieldName) {
    return false;
  }
  const isImageField =
    fieldName === 'image' ||
    fieldName.startsWith('image[') ||
    fieldName === 'input_reference';
  return isImageField && (filename !== undefined || contentType === undefined);
}

function firstNonNegative(left: number, right: number): number {
  if (left < 0) {
    return right;
  }
  if (right < 0) {
    return left;
  }
  return Math.min(left, right);
}
