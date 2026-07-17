import { afterEach, describe, expect, it, vi } from 'vitest';
import { createHmac } from 'node:crypto';
import {
  convertVideoCreateBody,
  decodeGatewayVideoId,
  encodeGatewayVideoId,
  resetGatewayVideoReferencesForTests,
  validateVideoCreateConversion,
  videoProviderKey
} from './video-compat';

describe('video API compatibility', () => {
  afterEach(() => {
    resetGatewayVideoReferencesForTests();
    vi.useRealTimers();
  });

  it('converts the common OpenAI and xAI video creation fields in both directions', () => {
    const xai = convertVideoCreateBody(
      {
        model: 'grok-imagine-video',
        prompt: 'A lighthouse in fog',
        seconds: '8',
        size: '1280x720'
      },
      'openai',
      'xai'
    );
    expect(xai).toEqual({
      model: 'grok-imagine-video',
      prompt: 'A lighthouse in fog',
      duration: 8,
      aspect_ratio: '16:9',
      resolution: '720p'
    });

    expect(convertVideoCreateBody(xai, 'xai', 'openai')).toEqual({
      model: 'grok-imagine-video',
      prompt: 'A lighthouse in fog',
      seconds: '8',
      size: '1280x720'
    });
  });

  it('rejects xAI image inputs because OpenAI requires a multipart upload', () => {
    const body = {
      duration: 8,
      aspect_ratio: '16:9',
      resolution: '720p',
      image: { url: 'https://example.test/frame.png' }
    };

    expect(validateVideoCreateConversion(body, 'xai', 'openai')).toContain(
      'requires a multipart file upload'
    );
    expect(() => convertVideoCreateBody(body, 'xai', 'openai')).toThrow(
      'requires a multipart file upload'
    );
  });

  it('rejects OpenAI sizes that xAI cannot represent exactly', () => {
    expect(
      validateVideoCreateConversion({ size: '1024x1792' }, 'openai', 'xai')
    ).toContain('cannot be represented exactly');
    expect(
      validateVideoCreateConversion({ size: '1792x1024' }, 'openai', 'xai')
    ).toContain('cannot be represented exactly');
  });

  it('rejects xAI settings that OpenAI cannot represent', () => {
    expect(
      validateVideoCreateConversion(
        { duration: 6, aspect_ratio: '1:1', resolution: '720p' },
        'xai',
        'openai'
      )
    ).toContain('duration');
    expect(
      validateVideoCreateConversion(
        { aspect_ratio: '16:9', resolution: '720p' },
        'xai',
        'openai'
      )
    ).toContain('must be explicit');
    expect(
      validateVideoCreateConversion({ duration: 8 }, 'xai', 'openai')
    ).toContain('must be explicit');
    expect(
      validateVideoCreateConversion(
        { duration: 8, aspect_ratio: '1:1', resolution: '720p' },
        'xai',
        'openai'
      )
    ).toContain('format');
    expect(
      validateVideoCreateConversion(
        { duration: 8, aspect_ratio: '16:9', resolution: '480p' },
        'xai',
        'openai'
      )
    ).toContain('format');
    expect(
      validateVideoCreateConversion(
        {
          duration: 8,
          aspect_ratio: '16:9',
          resolution: '720p',
          reference_images: [{ url: 'a' }, { url: 'b' }]
        },
        'xai',
        'openai'
      )
    ).toContain('requires a multipart file upload');
  });

  it('materializes OpenAI defaults when converting to xAI', () => {
    expect(convertVideoCreateBody({ prompt: 'A quiet lake' }, 'openai', 'xai')).toEqual({
      prompt: 'A quiet lake',
      duration: 4,
      aspect_ratio: '9:16',
      resolution: '720p'
    });
  });

  it('rejects provider-local OpenAI file ids when converting to xAI', () => {
    const body = { input_reference: { file_id: 'file-provider-local' } };
    expect(validateVideoCreateConversion(body, 'openai', 'xai')).toContain(
      'cannot be reused with xAI'
    );
    expect(() => convertVideoCreateBody(body, 'openai', 'xai')).toThrow(
      'cannot be reused with xAI'
    );
  });

  it('rejects source-only JSON fields during cross-provider conversion', () => {
    const openAIBody = {
      prompt: 'A recurring character',
      seconds: '8',
      size: '1280x720',
      characters: [{ id: 'char_123' }]
    };
    expect(validateVideoCreateConversion(openAIBody, 'openai', 'xai')).toContain(
      'characters'
    );
    expect(() => convertVideoCreateBody(openAIBody, 'openai', 'xai')).toThrow(
      'without data loss'
    );

    const xaiBody = {
      prompt: 'A seeded clip',
      duration: 8,
      aspect_ratio: '16:9',
      resolution: '720p',
      seed: 42
    };
    expect(validateVideoCreateConversion(xaiBody, 'xai', 'openai')).toContain('seed');
    expect(() => convertVideoCreateBody(xaiBody, 'xai', 'openai')).toThrow(
      'without data loss'
    );
  });

  it('keeps cross-protocol routing information in an encrypted route-safe stateless id', () => {
    const publicId = encodeGatewayVideoId({
      version: 2,
      upstreamId: 'video_68d7512d07848190b3e45da0ecbebcde004da08e1e0678d5',
      sourceProtocol: 'xai',
      targetProtocol: 'openai',
      targetProvider: 'openai',
      targetProviderName: 'openai-video',
      targetCredentialId: 'account-a',
      model: 'sora-2',
      duration: 8,
      size: '1280x720',
      createdAt: Math.floor(Date.now() / 1000),
      ownerKey: 'owner-key'
    }, { signingSecret: 'test-video-secret' });
    expect(publicId).toMatch(/^gv3\./);
    const encodedSegments = publicId
      .split('.')
      .slice(1)
      .map((segment) => Buffer.from(segment, 'base64url').toString('utf8'))
      .join('');
    expect(encodedSegments).not.toContain('account-a');
    expect(encodedSegments).not.toContain('owner-key');
    expect(encodedSegments).not.toContain('openai-video');

    resetGatewayVideoReferencesForTests();
    expect(decodeGatewayVideoId(publicId, { signingSecret: 'test-video-secret' })).toMatchObject({
      upstreamId: 'video_68d7512d07848190b3e45da0ecbebcde004da08e1e0678d5',
      sourceProtocol: 'xai',
      targetProtocol: 'openai',
      targetProvider: 'openai',
      targetProviderKey: videoProviderKey('openai-video'),
      targetCredentialId: 'account-a',
      model: 'sora-2',
      ownerKey: 'owner-key'
    });
  });

  it('continues to decode legacy gv2 ids during the migration window', () => {
    const secret = 'legacy-video-secret';
    const payload = {
      v: 2,
      u: 'legacy-upstream-id',
      s: 'openai',
      t: 'xai',
      p: 'xai',
      n: 'legacy-xai-video',
      m: 'grok-imagine-video',
      c: Math.floor(Date.now() / 1000),
      e: Math.ceil((Date.now() + 60_000) / 1000),
      o: 'legacy-owner-key'
    };
    const encodedPayload = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url');
    const unsignedId = `gv2.${encodedPayload}`;
    const signature = createHmac('sha256', secret).update(unsignedId).digest('base64url');

    expect(decodeGatewayVideoId(`${unsignedId}.${signature}`, { signingSecret: secret })).toMatchObject({
      upstreamId: 'legacy-upstream-id',
      sourceProtocol: 'openai',
      targetProtocol: 'xai',
      targetProvider: 'xai',
      targetProviderName: 'legacy-xai-video',
      model: 'grok-imagine-video',
      ownerKey: 'legacy-owner-key'
    });
  });

  it('keeps long upstream ids stateless across process-local cache loss', () => {
    const upstreamId = `video_${'a'.repeat(256)}`;
    const publicId = encodeGatewayVideoId({
      version: 2,
      upstreamId,
      sourceProtocol: 'openai',
      targetProtocol: 'xai',
      targetProvider: 'xai',
      targetProviderName: 'xai-video',
      createdAt: Math.floor(Date.now() / 1000)
    }, { signingSecret: 'test-video-secret' });

    resetGatewayVideoReferencesForTests();
    expect(decodeGatewayVideoId(publicId, { signingSecret: 'test-video-secret' })).toMatchObject({
      upstreamId,
      sourceProtocol: 'openai',
      targetProtocol: 'xai',
      targetProviderKey: videoProviderKey('xai-video')
    });
  });

  it('does not expire a signed video id before its configured TTL', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-17T00:00:00.900Z'));
    const publicId = encodeGatewayVideoId(
      {
        version: 2,
        upstreamId: 'video-near-second-boundary',
        sourceProtocol: 'openai',
        targetProtocol: 'openai',
        targetProvider: 'openai',
        model: 'sora-2',
        createdAt: Math.floor(Date.now() / 1000)
      },
      { signingSecret: 'ttl-secret', ttlMs: 1_000 }
    );

    resetGatewayVideoReferencesForTests();
    vi.advanceTimersByTime(500);
    expect(decodeGatewayVideoId(publicId, { signingSecret: 'ttl-secret' })).toMatchObject({
      upstreamId: 'video-near-second-boundary'
    });
  });

  it('rejects tampered, wrongly keyed, and expired gateway video ids', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-17T00:00:00Z'));
    const publicId = encodeGatewayVideoId(
      {
        version: 2,
        upstreamId: 'video-secure',
        sourceProtocol: 'openai',
        targetProtocol: 'xai',
        targetProvider: 'xai',
        model: 'grok-imagine-video',
        createdAt: Math.floor(Date.now() / 1000)
      },
      { signingSecret: 'correct-secret', ttlMs: 1_000 }
    );

    resetGatewayVideoReferencesForTests();
    expect(decodeGatewayVideoId(`${publicId}x`, { signingSecret: 'correct-secret' })).toBeUndefined();
    expect(decodeGatewayVideoId(publicId, { signingSecret: 'wrong-secret' })).toBeUndefined();
    vi.advanceTimersByTime(1_100);
    expect(decodeGatewayVideoId(publicId, { signingSecret: 'correct-secret' })).toBeUndefined();
  });
});
