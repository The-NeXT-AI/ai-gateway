import { describe, expect, it } from 'vitest';
import { parseAnthropicMessagesRequest, parseOpenAIResponsesRequest } from '../source/parsers';
import { openAIResponsesTargetAdapter } from './openai-responses';

// Property test for parallel tool batches that carry images, across both source
// shapes and all three OpenAI target modes. Each case is a random conversation:
// call groups, their outputs in arbitrary order, and plain user notes sprinkled
// anywhere. The invariants below are what every conversion must preserve; a
// failing seed can be replayed with FUZZ_SEED=<seed> FUZZ_ROUNDS=1.

type Event =
  | { kind: 'calls'; ids: string[]; searchIds: string[] }
  | { kind: 'output'; id: string; image: boolean; text?: string }
  | { kind: 'search'; id: string }
  | { kind: 'note'; text: string };

// mulberry32: small, seedable, deterministic.
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateConversation(random: () => number): Event[] {
  const events: Event[] = [];
  let callCount = 0;
  let noteCount = 0;
  const note = (): Event => ({ kind: 'note', text: `note${noteCount++}` });

  const turns = 1 + Math.floor(random() * 3);
  for (let turn = 0; turn < turns; turn += 1) {
    if (random() < 0.3) {
      events.push(note());
    }
    const ids = Array.from({ length: 1 + Math.floor(random() * 3) }, () => `c${callCount++}`);
    // Client-executed tool search calls answer within the same batch.
    const searchIds = random() < 0.3 ? [`s${callCount++}`] : [];
    events.push({ kind: 'calls', ids, searchIds });
    const answers = [...ids.map((id) => ({ id, search: false })), ...searchIds.map((id) => ({ id, search: true }))];
    for (const answer of answers.sort(() => random() - 0.5)) {
      if (answer.search) {
        events.push({ kind: 'search', id: answer.id });
      } else {
        events.push({
          kind: 'output',
          id: answer.id,
          image: random() < 0.6,
          text: random() < 0.5 ? `out-${answer.id}` : undefined
        });
      }
      if (random() < 0.35) {
        events.push(note());
      }
    }
  }
  if (random() < 0.5) {
    events.push(note());
  }
  return events;
}

const imageData = (id: string) => Buffer.from(id).toString('base64');

// Responses source: one input item per call, output and note.
function toResponsesInput(events: Event[]): unknown[] {
  const items: unknown[] = [{ type: 'message', role: 'user', content: 'start' }];
  for (const event of events) {
    if (event.kind === 'calls') {
      for (const id of event.ids) {
        items.push({ type: 'function_call', call_id: id, name: 'shot', arguments: '{}' });
      }
      for (const id of event.searchIds) {
        items.push({ type: 'tool_search_call', execution: 'client', call_id: id, arguments: { query: 'q' } });
      }
    } else if (event.kind === 'search') {
      items.push({ type: 'tool_search_output', execution: 'client', call_id: event.id, tools: [] });
    } else if (event.kind === 'output') {
      const output: unknown[] = [];
      if (event.text) {
        output.push({ type: 'input_text', text: event.text });
      }
      if (event.image) {
        output.push({ type: 'input_image', image_url: `data:image/png;base64,${imageData(event.id)}` });
      }
      items.push({
        type: 'function_call_output',
        call_id: event.id,
        output: output.length > 0 ? output : 'ok'
      });
    } else {
      items.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: event.text }] });
    }
  }
  return items;
}

// Anthropic source: a user turn groups several tool_results and text, and a
// batch may be split across turns. tool_result blocks have to lead their turn,
// so a note closes the turn for further results.
function toAnthropicMessages(events: Event[]): unknown[] {
  const messages: Array<{ role: string; content: unknown[]; closed?: boolean }> = [
    { role: 'user', content: [{ type: 'text', text: 'start' }], closed: true }
  ];
  const userTurn = (forResult: boolean) => {
    const last = messages[messages.length - 1];
    if (last.role === 'user' && !(forResult && last.closed)) {
      return last;
    }
    messages.push({ role: 'user', content: [] });
    return messages[messages.length - 1];
  };

  for (const event of events) {
    if (event.kind === 'search') {
      throw new Error('Anthropic input has no client tool-search item');
    }
    if (event.kind === 'calls') {
      messages.push({
        role: 'assistant',
        content: event.ids.map((id) => ({ type: 'tool_use', id, name: 'shot', input: {} }))
      });
    } else if (event.kind === 'output') {
      const content: unknown[] = [];
      if (event.text) {
        content.push({ type: 'text', text: event.text });
      }
      if (event.image) {
        content.push({
          type: 'image',
          source: { type: 'base64', media_type: 'image/png', data: imageData(event.id) }
        });
      }
      userTurn(true).content.push({
        type: 'tool_result',
        tool_use_id: event.id,
        content: content.length > 0 ? content : 'ok'
      });
    } else {
      const turn = userTurn(false);
      turn.content.push({ type: 'text', text: event.text });
      turn.closed = true;
    }
  }
  return messages.map(({ role, content }) => ({ role, content }));
}

const targetModes: Array<[string, Record<string, unknown>]> = [
  ['native', {}],
  ['text', { openaiResponsesToolOutputFormat: 'text' }],
  ['chat', { type: 'openai_chat_completions' }]
];

// Every string value in the body, one entry per line, so merged text parts
// (joined with "\n") still yield one entry per original note.
function collectTextLines(value: unknown, out: string[] = []): string[] {
  if (typeof value === 'string') {
    out.push(...value.split('\n'));
  } else if (Array.isArray(value)) {
    for (const entry of value) {
      collectTextLines(entry, out);
    }
  } else if (value && typeof value === 'object') {
    for (const entry of Object.values(value)) {
      collectTextLines(entry, out);
    }
  }
  return out;
}

function assertInvariants(
  context: string,
  events: Event[],
  body: Record<string, unknown>,
  mode: string
) {
  const serialized = JSON.stringify(body);

  // 1. Every user note survives exactly once and in its source order.
  const expectedNotes = events.flatMap((event) => (event.kind === 'note' ? [event.text] : []));
  const actualNotes = collectTextLines(body).filter((line) => /^note\d+$/.test(line));
  expect(actualNotes, `${context}: note order`).toEqual(expectedNotes);

  // 2. Every tool image survives exactly once as an image, never as text.
  for (const event of events) {
    if (event.kind !== 'output' || !event.image) {
      continue;
    }
    const occurrences =
      serialized.split(`data:image/png;base64,${imageData(event.id)}"`).length - 1;
    expect(occurrences, `${context}: image of ${event.id}`).toBe(1);
  }

  if (mode === 'chat') {
    const messages = body.messages as Array<Record<string, unknown>>;
    // 3. An assistant's parallel tool_calls are answered by tool messages right
    //    after it, before any user turn opens.
    for (const [index, message] of messages.entries()) {
      const calls = (message.tool_calls as Array<{ id: string }> | undefined) ?? [];
      if (calls.length === 0) {
        continue;
      }
      const answers = messages.slice(index + 1, index + 1 + calls.length);
      expect(answers.map((answer) => answer.role), `${context}: batch contiguity`).toEqual(
        calls.map(() => 'tool')
      );
      expect(new Set(answers.map((answer) => answer.tool_call_id))).toEqual(
        new Set(calls.map((call) => call.id))
      );
    }
    return;
  }

  const items = body.input as Array<Record<string, unknown>>;
  const groups = events.flatMap((event) =>
    event.kind === 'calls' ? [{ ids: event.ids, searchIds: event.searchIds }] : []
  );
  const indexOf = (type: string, id: string) =>
    items.findIndex((item) => item.type === type && item.call_id === id);
  for (const { ids, searchIds } of groups) {
    const callIndexes = [
      ...ids.map((id) => indexOf('function_call', id)),
      ...searchIds.map((id) => indexOf('tool_search_call', id))
    ];
    const outputIndexes = [
      ...ids.map((id) => indexOf('function_call_output', id)),
      ...searchIds.map((id) => indexOf('tool_search_output', id))
    ];
    expect(callIndexes.every((index) => index >= 0), `${context}: every call present`).toBe(true);
    expect(outputIndexes.every((index) => index >= 0), `${context}: every output present`).toBe(
      true
    );
    // 3. No message carrying a tool image opens between a group's calls and
    //    its last output. (User text placed ahead of a message's own outputs is
    //    the long-standing Responses layout and is not asserted on here.)
    const inside = items.slice(Math.min(...callIndexes), Math.max(...outputIndexes) + 1);
    const imageMessagesInside = inside.filter(
      (item) => item.type === 'message' && JSON.stringify(item).includes('input_image')
    );
    expect(imageMessagesInside, `${context}: batch contiguity`).toEqual([]);
  }

  // 4. Native mode keeps every image on its own function_call_output.
  if (mode === 'native') {
    const userMessages = items.filter((item) => item.type === 'message');
    expect(JSON.stringify(userMessages), `${context}: native attribution`).not.toContain(
      'input_image'
    );
  }
}

function convert(standardRequest: unknown, targetProviderConfig: Record<string, unknown>) {
  const built = openAIResponsesTargetAdapter.buildRequestFromStandard({
    request: { headers: {} } as never,
    standardRequest: standardRequest as never,
    config: { openaiApiKey: 'sk-test', openaiBaseUrl: 'https://mock.local/v1' } as never,
    targetProviderConfig: targetProviderConfig as never
  });
  expect(built.ok).toBe(true);
  if (!built.ok) {
    throw new Error(built.error);
  }
  return built.value.body as Record<string, unknown>;
}

describe('parallel tool batch conversion properties', () => {
  const rounds = Number(process.env.FUZZ_ROUNDS ?? 300);
  const firstSeed = Number(process.env.FUZZ_SEED ?? 1);

  it('holds for Responses-shaped input', () => {
    for (let offset = 0; offset < rounds; offset += 1) {
      const seed = firstSeed + offset;
      const events = generateConversation(createRandom(seed));
      const parsed = parseOpenAIResponsesRequest({ model: 'm', input: toResponsesInput(events) });
      expect(parsed.ok, `responses seed ${seed}`).toBe(true);
      if (!parsed.ok) {
        return;
      }
      for (const [mode, targetProviderConfig] of targetModes) {
        const body = convert(parsed.value, targetProviderConfig);
        assertInvariants(`responses seed=${seed} ${mode}`, events, body, mode);
      }
    }
  });

  it('holds for Anthropic-shaped input', () => {
    for (let offset = 0; offset < rounds; offset += 1) {
      const seed = firstSeed + 100000 + offset;
      // Anthropic has no client tool-search item, so the conversation is
      // generated without search answers.
      const events = generateConversation(createRandom(seed)).flatMap((event): Event[] =>
        event.kind === 'search'
          ? []
          : event.kind === 'calls'
            ? [{ ...event, searchIds: [] }]
            : [event]
      );
      const parsed = parseAnthropicMessagesRequest({
        model: 'm',
        max_tokens: 16,
        messages: toAnthropicMessages(events)
      });
      expect(parsed.ok, `anthropic seed ${seed}`).toBe(true);
      if (!parsed.ok) {
        return;
      }
      for (const [mode, targetProviderConfig] of targetModes) {
        const body = convert(parsed.value, targetProviderConfig);
        assertInvariants(`anthropic seed=${seed} ${mode}`, events, body, mode);
      }
    }
  });
});
