import { createServer } from 'node:http';
import { once } from 'node:events';
import Fastify from 'fastify';
import { describe, expect, it, vi } from 'vitest';
import { parseGatewayConfigFromRaw } from '../config';
import { registerGatewayRoutes } from './routes';
import { createGatewayRuntime } from './runtime';

describe('Anthropic to Obit HTTP wire conversion', () => {
  it('sends disabled thinking and explicit effort to the actual upstream transport', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const upstream = createServer(async (request, response) => {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      if (bodies.at(-1)?.stream === true) {
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.write(`data: ${JSON.stringify({
          id: 'test-completion', model: 'unsloth/Qwen3.8-27B-GGUF',
          choices: [{ index: 0, delta: { role: 'assistant', content: 'OK' }, finish_reason: null }]
        })}\n\n`);
        response.write(`data: ${JSON.stringify({
          id: 'test-completion', model: 'unsloth/Qwen3.8-27B-GGUF',
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          usage: { prompt_tokens: 17, completion_tokens: 1, total_tokens: 18 }
        })}\n\n`);
        response.end('data: [DONE]\n\n');
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        id: 'test-completion', object: 'chat.completion', model: 'unsloth/Qwen3.8-27B-GGUF',
        choices: [{ index: 0, message: { role: 'assistant', content: 'OK' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 17, completion_tokens: 1, total_tokens: 18 }
      }));
    });
    upstream.listen(0, '127.0.0.1');
    await once(upstream, 'listening');
    const address = upstream.address();
    if (!address || typeof address === 'string') throw new Error('Missing loopback address');
    const realFetch = globalThis.fetch.bind(globalThis);
    const transport = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      const target = new URL(String(url));
      expect(target.hostname).toBe('relay.obitmc.com');
      expect(target.pathname).toBe('/v1/chat/completions');
      return realFetch(`http://127.0.0.1:${address.port}${target.pathname}`, init);
    });
    const config = parseGatewayConfigFromRaw({
      logging: { enabled: false },
      providers: [{ name: 'obit', type: 'openai_chat_completions', baseurl: 'https://relay.obitmc.com/v1',
        apikey: 'test-only', models: ['unsloth/Qwen3.8-27B-GGUF'] }]
    });
    const app = Fastify({ logger: false });
    registerGatewayRoutes(app, config, createGatewayRuntime(config));
    try {
      for (const controls of [
        { thinking: { type: 'disabled' }, output_config: { effort: 'high' }, stream: false },
        { thinking: { type: 'adaptive', display: 'omitted' }, output_config: { effort: 'low' }, stream: false },
        { stream: false },
        { thinking: { type: 'disabled' }, output_config: { effort: 'high' }, stream: true },
        { thinking: { type: 'adaptive', display: 'omitted' }, output_config: { effort: 'low' }, stream: true }
      ]) {
        const response = await app.inject({ method: 'POST', url: '/v1/messages',
          headers: { 'x-target-provider': 'obit', 'content-type': 'application/json' },
          payload: { model: 'unsloth/Qwen3.8-27B-GGUF', max_tokens: 64,
            messages: [{ role: 'user', content: 'Reply with OK.' }], ...controls }
        });
        expect(response.statusCode, response.body).toBe(200);
        if (controls.stream) {
          expect(response.body).toContain('"text":"OK"');
          expect(response.body).toContain('event: message_stop');
        } else {
          expect(response.json().content).toEqual([{ type: 'text', text: 'OK' }]);
        }
      }
      expect(bodies.map((body) => body.reasoning_effort)).toEqual(['none', 'low', undefined, 'none', 'low']);
      for (const body of bodies) {
        expect(body).not.toHaveProperty('thinking');
        expect(body).not.toHaveProperty('output_config');
      }
    } finally {
      await app.close();
      transport.mockRestore();
      await new Promise<void>((resolve) => upstream.close(() => resolve()));
    }
  });
});
