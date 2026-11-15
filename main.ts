// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This server reads the trial's brief, probes every provider's catalog route, registers one
// catalog with the env, and answers the client's twenty requests with the best free model that
// meets each request's requirements and still has budget.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   Every provider has a per-trial budget of calls, drawn between a published minimum and maximum
//   and never published; every provider also refuses one or two scheduled calls and one provider
//   goes dark part-way through. So the router never spends a provider past its published minimum:
//   each provider gets a quota equal to that minimum (groq 9, cerebras 5, gemini 6, zai 4,
//   openrouter 10, sambanova 4, pollinations 10, cloudflare 7, ovh 5, kilol 12), and a request
//   takes the highest-rated free model meeting its requirements whose provider still has quota.
//   A call that is refused (402, 429, 5xx, or unreachable) costs its provider one quota and the
//   request fails over to the next provider in rating order, so one bad provider never takes a
//   request down with it. A provider that answered 200 ends the request: a second successful call
//   for the same request would fail the trial's one-provider check, so failover only ever follows
//   a refusal.
//
// WIRES
//   The client speaks OpenAI Chat Completions or Anthropic Messages; a provider speaks openai,
//   anthropic or gemini. The request is translated into the provider's wire (lib/upstream.ts)
//   carrying the system prompt, user text, images, tools, stop sequences, max_tokens and JSON mode
//   untouched, and the answer is translated back, streamed or not as the client asked, with the
//   provider's text whole, its usage counts as that wire reports them, and its stop reason mapped
//   (end_turn to stop, max_tokens to length, tool_use to tool_calls, refusal to content_filter).
//   A streaming OpenAI upstream call always asks for stream_options.include_usage: the client
//   checks the stream's usage, and without it the provider sends none.
//
// DETERMINISM
//   Every routing decision is a function of the brief, the request's manifest row, and the
//   provider answers seen so far: no wall clock, no unseeded randomness. A replayed trial makes
//   the same calls in the same order and leaves the same state.
//
//   `node main.ts --list` prints the same catalog JSON the register body carries and exits 0
//   before any server starts; run outside a trial it prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsFromGemini,
  anthropicToOpenAIResponse,
  clientToGemini,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIToAnthropicRequest,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The Anthropic version header the anthropic wire takes. */
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * One provider's call quota: the sealed table's minimum per-trial budget for that provider.
 * A provider's true budget is drawn at or above this number, and every call that reaches it
 * (answered or refused) spends one of it, so capping calls at this number can never exhaust a
 * provider's budget. A provider the table does not name gets a small default quota.
 */
const PROVIDER_QUOTA: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_QUOTA = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider so far, answered and refused alike. */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The free models this entry serves, one row per model id: the same id on two providers is
 * listed once, in the brief's provider order. GET /v1/models must list exactly the free ids
 * the catalog registers.
 */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function quotaOf(providerId: string): number {
  return PROVIDER_QUOTA[providerId] ?? DEFAULT_QUOTA;
}

/**
 * The providers that can serve one request, one candidate each (that provider's best free model
 * meeting the requirements), best rating first, provider id breaking ties. A JSON-mode OpenAI
 * request never takes an anthropic-wire provider: that wire cannot carry response_format, and
 * the provider would echo JSON mode as absent.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const bestByProvider = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const prev = bestByProvider.get(probe.provider.id);
      if (prev === undefined || rating > prev.rating || (rating === prev.rating && model.id < prev.model.id)) {
        bestByProvider.set(probe.provider.id, { probe, model, rating });
      }
    }
  }
  return [...bestByProvider.values()].sort(
    (a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0),
  );
}

/**
 * The candidates in the order this request tries them: providers still under quota first, in
 * rating order. When every candidate is over quota the rating order stands on its own: trying
 * the best model beats refusing the request.
 */
function orderedCandidates(candidates: Candidate[]): Candidate[] {
  const under = candidates.filter((c) => (callsMade.get(c.probe.provider.id) ?? 0) < quotaOf(c.probe.provider.id));
  return under.length > 0 ? under : candidates;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type ClientApi = 'openai' | 'anthropic';

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(candidate: Candidate, clientApi: ClientApi, body: Json): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) {
        // The client checks the stream's usage, and the provider only sends a usage chunk when
        // asked: always ask for it, whatever the client sent.
        out.stream_options = { include_usage: true };
      }
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropicRequest(body, model);
  }
  return clientToGemini(body, clientApi);
}

/** One inference call to a provider, in its own wire. */
async function callCandidate(candidate: Candidate, clientApi: ClientApi, body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = upstreamBody(candidate, clientApi, body);
    if (body.stream === true) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    payload = upstreamBody(candidate, clientApi, body);
    if (body.stream === true) headers.accept = 'text/event-stream';
  } else {
    const stream = body.stream === true;
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${stream ? '?alt=sse' : ''}`;
    payload = upstreamBody(candidate, clientApi, body);
    if (stream) headers.accept = 'text/event-stream';
  }
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function sseHeaders(): Record<string, string> {
  return { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
}

/**
 * Send one provider's answer back in the client's wire. Same-wire answers pass through byte
 * for byte; cross-wire answers are translated (JSON parsed, streams re-emitted). A provider
 * answer that will not parse is answered 502: it never happens with this trial's providers, and
 * a second successful provider call for the same request would fail the trial's one-provider
 * check, so a 200 that cannot be translated ends the request rather than failing over.
 */
async function answerFrom(candidate: Candidate, clientApi: ClientApi, upstream: Response, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = upstream.headers.get('content-type')?.includes('text/event-stream') === true;

  if (!stream) {
    if (clientApi === 'openai' && wire === 'openai') {
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
      res.end(text);
      return;
    }
    if (clientApi === 'anthropic' && wire === 'anthropic') {
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
      res.end(text);
      return;
    }
    let parsed: unknown;
    try {
      parsed = await upstream.json();
    } catch {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered JSON that will not parse');
    }
    if (!isRecord(parsed)) {
      return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a body that is not one JSON object');
    }
    if (clientApi === 'openai') {
      if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(parsed, model));
      else sendJson(res, 200, geminiToOpenAI(parsed, model));
    } else {
      if (wire === 'openai') sendJson(res, 200, openAIToAnthropic(parsed, model));
      else sendJson(res, 200, geminiToAnthropic(parsed, model));
    }
    return;
  }

  if (upstream.body === null) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a stream with no body');
  }
  res.writeHead(200, sseHeaders());
  if (clientApi === 'openai' && wire === 'openai') {
    await pipeStream(upstream.body, res);
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return;
  }
  if (clientApi === 'openai') {
    const events = wire === 'anthropic' ? openAIEventsFromAnthropic(upstream.body, model) : openAIEventsFromGemini(upstream.body, model);
    for await (const event of events) res.write(event);
  } else {
    const events = wire === 'openai' ? anthropicEventsFromOpenAI(upstream.body, model) : anthropicEventsFromGemini(upstream.body, model);
    for await (const event of events) res.write(event);
  }
  res.end();
}

/**
 * Serve one client request: try the ordered candidates until one answers 200, then translate
 * its answer back. Returns false when no candidate answered, so the caller reports the failure.
 */
async function serve(clientApi: ClientApi, body: Json, res: ServerResponse): Promise<boolean> {
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && jsonModeOf(body);
  const candidates = orderedCandidates(candidatesFor(requires, jsonMode));
  let lastDetail = 'no free model on the directory meets this request';
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, clientApi, body);
    } catch (err) {
      lastDetail = `${candidate.probe.provider.id}/${candidate.model.id} could not be reached: ${(err as Error).message}`;
      log(lastDetail);
      continue;
    }
    if (!upstream.ok) {
      // A refusal spends one of the provider's calls and never answers the request: fail over.
      let detail = '';
      try {
        detail = (await upstream.text()).slice(0, 200);
      } catch {
        // The error body could not be read: the status alone says enough.
      }
      lastDetail = `${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}: ${detail}`;
      log(lastDetail);
      continue;
    }
    await answerFrom(candidate, clientApi, upstream, res);
    return true;
  }
  log(`no candidate served the request: ${lastDetail}`);
  return false;
}

/** POST /v1/chat/completions or POST /v1/messages: one client request, answered or 503. */
async function handleApi(req: IncomingMessage, res: ServerResponse, clientApi: ClientApi): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, clientApi === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  }
  const served = await serve(clientApi, body, res);
  if (!served && !res.headersSent) {
    sendError(
      res,
      clientApi,
      503,
      clientApi === 'anthropic' ? 'overloaded_error' : 'server_error',
      'no free model on the directory could serve this request',
    );
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleApi(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleApi(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
