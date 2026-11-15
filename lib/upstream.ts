// The upstream wires the starter does not speak: every translation between the two client wires
// (OpenAI Chat Completions, Anthropic Messages) and the three provider wires (openai, anthropic,
// gemini), in both directions, for JSON bodies and for server-sent events.
//
// The translations are written to keep the provider's echo identical to what the client sent. The
// trial's providers answer with an echo of what they parsed (system text, user text, tool names,
// max_tokens, stop sequences, image bytes, JSON mode), and the client compares that echo with its own
// request, so a translation must carry every one of those fields across the wire unchanged:
//
//   - system text: one string, whatever the source wire's shape (a system message, a system field,
//     a systemInstruction); the providers join multi-part system text with "\n", so multi-part
//     sources are joined with "\n" first and then carried as one string.
//   - user text: the text blocks of the last user turn, joined with "\n" on every wire.
//   - images: the same bytes. A data URL becomes inlineData / an image block with the same base64;
//     a plain URL becomes fileData / an image block with the same URL.
//   - tools: the same names, in the same order (functionDeclarations on gemini, input_schema tools
//     on anthropic, function tools on openai).
//   - max_tokens, temperature, top_p, stop sequences: passed through, renamed per wire.
//   - JSON mode: response_format json_object/json_schema becomes responseMimeType application/json
//     on gemini. The anthropic wire cannot carry it at all, so main.ts never routes a JSON-mode
//     OpenAI request to an anthropic-wire provider.
//
// Everything here is a pure function of its arguments: no clock, no randomness, no I/O. The same
// provider answer always translates to the same client bytes, which is what a replayed trial needs.

import { isRecord, type Json } from './providers.ts';
import { sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for translated completion objects, as wire.ts uses for its lists. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url: string = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI message as an Anthropic message (or several: a tool message becomes tool_results). */
function openAIMessageToAnthropic(message: Json, out: Json[]): void {
  const role = message.role;
  if (role === 'system') return; // the system prompt travels in the `system` field, not in messages
  if (role === 'tool') {
    out.push({
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
          content: typeof message.content === 'string' ? message.content : textOf(message.content),
        },
      ],
    });
    return;
  }
  const content = message.content;
  if (role === 'assistant') {
    const blocks: Json[] = [];
    const text = textOf(content);
    if (text !== '') blocks.push({ type: 'text', text });
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: the tool_use carries an empty input.
        }
      }
      blocks.push({
        type: 'tool_use',
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        input,
      });
    }
    if (blocks.length > 0) out.push({ role: 'assistant', content: blocks });
    return;
  }
  if (typeof content === 'string') {
    out.push({ role: 'user', content });
    return;
  }
  if (Array.isArray(content)) {
    const blocks = content.flatMap((part) => {
      if (!isRecord(part)) return [];
      const block = openAIPartToAnthropic(part);
      return block === null ? [] : [block];
    });
    if (blocks.length > 0) out.push({ role: 'user', content: blocks });
  }
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. The system
 * prompt becomes the `system` field, images become image blocks, tool definitions and tool choice
 * are rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass through.
 * JSON mode has no anthropic spelling: callers must not route a JSON-mode request here.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    openAIMessageToAnthropic(raw, messages);
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Anthropic content block as a Gemini content part: text or an inline image. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini content part. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url: string = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** Tool definitions as Gemini functionDeclarations, from either client wire's spelling. */
function toolsToGemini(body: Json, api: 'openai' | 'anthropic'): Json[] {
  const raw = Array.isArray(body.tools) ? body.tools : [];
  const out: Json[] = [];
  for (const tool of raw) {
    if (!isRecord(tool)) continue;
    if (api === 'openai') {
      if (!isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      out.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    } else {
      if (typeof tool.name !== 'string') continue;
      out.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

/**
 * A client request (either wire) as a Gemini generateContent body. The model travels in the URL
 * path, not in this body. Contents keep the turns' order; the system prompt becomes the
 * systemInstruction; tool calls become functionCall parts and tool results functionResponse parts.
 */
export function clientToGemini(body: Json, api: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  let systemText = '';
  if (api === 'anthropic') {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        if (Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block) || block.type !== 'tool_use') continue;
            let args: Json = {};
            if (isRecord(block.input)) args = block.input;
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_result') {
            const content = block.content;
            parts.push({
              functionResponse: {
                name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
                response: { result: typeof content === 'string' ? content : textOf(content) },
              },
            });
            continue;
          }
          const part = anthropicBlockToGemini(block);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  } else {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty args object.
            }
          }
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
                response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const geminiPart = openAIPartToGemini(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = toolsToGemini(body, api);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = api === 'anthropic' ? body.max_tokens : (body.max_tokens ?? body.max_completion_tokens);
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = api === 'anthropic' ? stringList(body.stop_sequences) : stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (api === 'openai' && isRecord(body.response_format)) {
    const format = body.response_format.type;
    if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Responses back to the client
// ---------------------------------------------------------------------------

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** The text of a Gemini candidate's content parts, joined in order. */
function geminiText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini usageMetadata object as OpenAI usage counts. */
function geminiUsage(body: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(meta.promptTokenCount);
  const output = numberOrZero(meta.candidatesTokenCount);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(candidate) },
        finish_reason: geminiStopToOpenAI(candidate.finishReason),
        logprobs: null,
      },
    ],
    usage,
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(candidate) }],
    stop_reason: geminiStopToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  let text = '';
  const calls: Json[] = [];
  const blocks = Array.isArray(body.content) ? body.content : [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  const message: Json = { role: 'assistant', content: calls.length > 0 && text === '' ? null : text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One server-sent-events frame (event name + data) of a response stream, split on blank lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Read a whole response body as text (a stream the caller buffered first). */
export async function readBodyText(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return out;
}

function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/**
 * An Anthropic SSE stream as the OpenAI event sequence: a role chunk, content deltas, tool-call
 * deltas, a finish chunk, a usage chunk, then [DONE]. The provider's message_start carries the
 * input tokens and its message_delta the output tokens; both land in the final usage chunk.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readBodyText(upstream);
  const frames = parseSseFrames(text);
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  const toolIndexes = new Map<number, number>();
  let nextToolIndex = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent(null, openAIChunk(model, {}, stop)),
      sseEvent(null, {
        id: 'chatcmpl-sim',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        yield* start();
        break;
      }
      case 'content_block_start': {
        yield* start();
        const index = numberOrZero(data.index);
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolIndexes.set(index, nextToolIndex);
          nextToolIndex += 1;
          yield sseEvent(
            null,
            openAIChunk(model, {
              tool_calls: [
                {
                  index: toolIndexes.get(index),
                  id: typeof block.id === 'string' ? block.id : `toolu_${index}`,
                  type: 'function',
                  function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
                },
              ],
            }, null),
          );
        }
        break;
      }
      case 'content_block_delta': {
        yield* start();
        const index = numberOrZero(data.index);
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          yield sseEvent(null, openAIChunk(model, { content: delta.text }, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const toolIndex = toolIndexes.get(index) ?? nextToolIndex;
          yield sseEvent(null, openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null));
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') stop = anthropicStopToOpenAI(delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        yield* start();
        yield* finish();
        break;
      }
      default:
        break;
    }
  }
  if (!finished) {
    yield* start();
    yield* finish();
  }
}

/**
 * A Gemini SSE stream as the OpenAI event sequence. Gemini frames carry no event names; each holds
 * one candidate's text, and the last carries the finishReason and the usageMetadata. The stream is
 * buffered first so the usage is known before the first chunk is written.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readBodyText(upstream);
  const frames = parseSseFrames(text);
  let fullText = '';
  let stop = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    fullText += geminiText(candidate);
    if (typeof candidate.finishReason === 'string') stop = geminiStopToOpenAI(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
  for (let at = 0; at < fullText.length; at += 13) {
    yield sseEvent(null, openAIChunk(model, { content: fullText.slice(at, at + 13) }, null));
  }
  yield sseEvent(null, openAIChunk(model, {}, stop));
  yield sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start (with the input tokens, known
 * because the stream is buffered first), one text block's deltas, message_delta with the stop reason
 * and the output tokens, then message_stop.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readBodyText(upstream);
  const frames = parseSseFrames(text);
  let fullText = '';
  let stop = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    fullText += geminiText(candidate);
    if (typeof candidate.finishReason === 'string') stop = geminiStopToAnthropic(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (let at = 0; at < fullText.length; at += 13) {
    yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: fullText.slice(at, at + 13) } });
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

/** OpenAI's finish_reason as Anthropic's stop_reason (the reverse of wire.ts's stopReasonOf). */
export function openAIStopToAnthropic(finish: unknown): string {
  return stopReasonOf(finish);
}
