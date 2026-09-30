/**
 * Live tests for @pairsystems/goodmem-ai-sdk, against a running GoodMem server.
 *
 * They skip entirely unless GOODMEM_API_KEY, GOODMEM_BASE_URL and
 * GOODMEM_TEST_EMBEDDER_ID are set -- which is also the check that no
 * credential is baked into the package.
 *
 *   GOODMEM_BASE_URL=http://localhost:8080 GOODMEM_API_KEY=... \
 *   GOODMEM_TEST_EMBEDDER_ID=... [GOODMEM_TEST_RERANKER_ID=...] \
 *   [GOODMEM_TEST_LATENCY_EMBEDDER_ID=...] [GOODMEM_TEARDOWN_REPORT=path.json] \
 *     npm run test:live
 *
 * The middleware tests that expect real memories pass a generous
 * `retrievalTimeoutMs`: the default 5 s is shorter than the hosted embedder
 * behind the test space sometimes takes to embed one query.
 *
 * Every space is created through the package itself (`space: { name,
 * embedderId }`) under a unique run id. One temporary embedder with an
 * unreachable endpoint is created with the SDK to produce a real
 * EMBEDDER_FAILED. Everything is registered as soon as it exists, deleted at
 * the end with the SDK (every deletion attempted, failures reported
 * together), and the final test asserts against a fresh listing that nothing
 * the run created is left. TLS verification is never turned off here.
 */

import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { createServer, type Server, type Socket } from 'node:net';
import { inspect } from 'node:util';
import { after, before, describe, it } from 'node:test';

import { Goodmem } from '@pairsystems/goodmem';
import { generateText, stepCountIs, streamText } from 'ai';

import {
  addMemories,
  filters,
  GoodMemConfigError,
  GoodMemError,
  GoodMemIndexingError,
  GoodMemIngestionError,
  goodmemTools,
  RETRIEVAL_FAILED_CODE,
  retrieveMemories,
  searchMemories,
  withGoodmem,
  type GoodmemConfig,
} from '../src/index';
import { makePdf } from './support/pdf';
import { scriptedModel, streamingModel, systemText, textResult, toolCall, toolResultsIn } from './support/models';

const BASE = process.env.GOODMEM_BASE_URL;
const KEY = process.env.GOODMEM_API_KEY;
const EMBEDDER = process.env.GOODMEM_TEST_EMBEDDER_ID;
const RERANKER = process.env.GOODMEM_TEST_RERANKER_ID;
// The empty-space latency check embeds a query on the server, so its absolute
// time is bounded below by the embedder's; this lets it use a faster one.
const LATENCY_EMBEDDER = process.env.GOODMEM_TEST_LATENCY_EMBEDDER_ID || process.env.GOODMEM_TEST_EMBEDDER_ID;
const TEARDOWN_REPORT = process.env.GOODMEM_TEARDOWN_REPORT;
const skip = !(BASE && KEY && EMBEDDER)
  ? 'GOODMEM_API_KEY, GOODMEM_BASE_URL and GOODMEM_TEST_EMBEDDER_ID are not set'
  : false;

const RUN = `aisdk-live-${Date.now().toString(36)}-${randomBytes(3).toString('hex')}`;
const TOKEN = RUN.slice(-6).toUpperCase();
const TEXT_CANARY = `ORYX-${TOKEN}`;
const PDF_CANARY = `LYNX-${TOKEN}`;
const TOOL_CANARY = `IBEX-${TOKEN}`;
const TEXT = `The live text canary is ${TEXT_CANARY}. O'Brien filed it for the AI SDK suite.`;
const PDF = makePdf(`The live PDF canary is ${PDF_CANARY} for the AI SDK suite.`);
const MISSING_SPACE = '00000000-0000-7000-8000-000000000000';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

class Logs {
  readonly lines: string[] = [];
  warn = (message: string) => {
    this.lines.push(message);
  };
}

describe('live', { skip }, () => {
  // The SDK, used directly only for what the package does not do on purpose:
  // creating the unreachable embedder, reading back server state, and teardown.
  let admin: Goodmem;
  const created = { spaces: new Set<string>(), embedders: new Set<string>() };
  let tornDown = false;

  let serverVersion = '';
  const inventoryBefore = { spaces: 0, embedders: 0 };
  let badEmbedderId = '';
  let mainConfig: GoodmemConfig;
  let mainSpaceId = '';
  let textMemoryId = '';
  let pdfMemoryId = '';
  let weatherMemoryId = '';

  // Generous, because the hosted embedder behind the test space has been seen
  // taking over 30 s to embed a single query. Timeouts themselves are tested
  // against a local server that never answers.
  const config = (overrides: Partial<GoodmemConfig>): GoodmemConfig => ({
    apiKey: KEY!,
    baseUrl: BASE!,
    timeoutMs: 90_000,
    logger: new Logs(),
    ...overrides,
  });

  async function waitCompleted(memoryId: string): Promise<string> {
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      const status = (await admin.memories.get(memoryId)).processingStatus;
      if (status === 'COMPLETED' || status === 'FAILED') return status;
      await sleep(1000);
    }
    throw new Error(`memory ${memoryId} did not finish indexing`);
  }

  async function spacesNamed(prefix: string) {
    const out: Array<{ spaceId: string; name: string; embedderIds: string[] }> = [];
    for await (const space of await admin.spaces.list({ nameFilter: `${prefix}*`, maxResults: 1000 })) {
      out.push({
        spaceId: space.spaceId,
        name: space.name,
        embedderIds: (space.spaceEmbedders ?? []).map((e) => e.embedderId),
      });
    }
    return out;
  }

  async function teardown() {
    tornDown = true;
    const failures: string[] = [];
    // Sweep by run id as well as by registration, so a space created by a
    // test that then failed is still found.
    for (const space of await spacesNamed(RUN)) created.spaces.add(space.spaceId);
    for (const embedder of await admin.embedders.list()) {
      if (embedder.displayName.startsWith(RUN)) created.embedders.add(embedder.embedderId);
    }
    for (const id of created.spaces) {
      try {
        await admin.spaces.delete(id);
      } catch (error) {
        failures.push(`space ${id}: ${(error as Error).message}`);
      }
    }
    for (const id of created.embedders) {
      try {
        await admin.embedders.delete(id);
      } catch (error) {
        failures.push(`embedder ${id}: ${(error as Error).message}`);
      }
    }

    // Verify against a fresh, complete listing -- not the deletes' status.
    const allSpaces: Array<{ spaceId: string; name: string }> = [];
    for await (const space of await admin.spaces.list({ maxResults: 1000 })) {
      allSpaces.push({ spaceId: space.spaceId, name: space.name });
    }
    const allEmbedders = (await admin.embedders.list()).map((e) => ({ embedderId: e.embedderId, displayName: e.displayName }));
    const leftSpaces = allSpaces.filter((s) => s.name.startsWith(RUN) || created.spaces.has(s.spaceId));
    const leftEmbedders = allEmbedders.filter((e) => e.displayName.startsWith(RUN) || created.embedders.has(e.embedderId));
    // The report names only this run's resources: other spaces on a shared
    // server are counted, never listed, so it can be kept as evidence.
    const report = {
      run: RUN,
      serverVersion,
      checkedAt: new Date().toISOString(),
      deleted: { spaces: [...created.spaces], embedders: [...created.embedders] },
      deleteFailures: failures,
      leftFromThisRun: { spaces: leftSpaces, embedders: leftEmbedders },
      freshListing: {
        before: inventoryBefore,
        after: { spaces: allSpaces.length, embedders: allEmbedders.length },
        spacesNamedAisdkLive: allSpaces.filter((s) => s.name.startsWith('aisdk-live-')),
        embeddersNamedAisdkLive: allEmbedders.filter((e) => e.displayName.startsWith('aisdk-live-')),
      },
    };
    console.log(`# teardown verification: ${JSON.stringify(report.leftFromThisRun)} left from ${RUN}; ` +
      `server had ${inventoryBefore.spaces} spaces / ${inventoryBefore.embedders} embedders before the run and ` +
      `${allSpaces.length} / ${allEmbedders.length} after`);
    if (TEARDOWN_REPORT) writeFileSync(TEARDOWN_REPORT, `${JSON.stringify(report, null, 2)}\n`);
    return { failures, leftSpaces, leftEmbedders };
  }

  before(async () => {
    admin = new Goodmem({ baseUrl: BASE!, apiKey: KEY!, timeoutMs: 90_000 });
    serverVersion = (await admin.system.info()).version;
    for await (const _space of await admin.spaces.list({ maxResults: 1000 })) inventoryBefore.spaces += 1;
    inventoryBefore.embedders = (await admin.embedders.list()).length;
    console.log(`# GoodMem ${serverVersion} at ${BASE}; run ${RUN}; ${inventoryBefore.spaces} spaces and ${inventoryBefore.embedders} embedders before`);

    const bad = await admin.embedders.create({
      displayName: `${RUN}-unreachable`,
      providerType: 'OPENAI',
      endpointUrl: 'http://127.0.0.1:9/v1',
      modelIdentifier: `unreachable-${TOKEN.toLowerCase()}`,
      dimensionality: 8,
      distributionType: 'DENSE',
    });
    created.embedders.add(bad.embedderId);
    badEmbedderId = bad.embedderId;

    mainConfig = config({ space: { name: `${RUN}-main`, embedderId: EMBEDDER! } });
    const stored = await addMemories(
      [
        { text: TEXT, metadata: { tenant: 'acme', year: 2026, active: true } },
        { data: PDF, contentType: 'application/pdf', metadata: { tenant: 'acme', kind: 'pdf' } },
        { text: 'An unrelated note about the weather in Amman.', metadata: { tenant: 'globex', year: 2025, active: false } },
      ],
      mainConfig,
      { waitForIndexing: { timeoutMs: 120_000 } }
    );
    for (const m of stored.memories) created.spaces.add(m.spaceId);
    [textMemoryId, pdfMemoryId, weatherMemoryId] = stored.memories.map((m) => m.memoryId);
    mainSpaceId = stored.memories[0].spaceId;
    assert.deepEqual(stored.memories.map((m) => m.processingStatus), ['COMPLETED', 'COMPLETED', 'COMPLETED']);
  });

  after(async () => {
    if (!tornDown && admin) {
      const { failures, leftSpaces, leftEmbedders } = await teardown();
      assert.deepEqual([failures, leftSpaces, leftEmbedders], [[], [], []], 'teardown after an aborted run left resources');
    }
  });

  // ---------------------------------------------------------------- spaces --

  it('creates its space through the package, and an identical second config reuses it', async () => {
    const named = (await spacesNamed(`${RUN}-main`)).filter((s) => s.name === `${RUN}-main`);
    assert.equal(named.length, 1);
    assert.equal(named[0].spaceId, mainSpaceId);
    assert.deepEqual(named[0].embedderIds, [EMBEDDER]);

    const again = config({ space: { name: `${RUN}-main`, embedderId: EMBEDDER! } });
    const out = await searchMemories(TEXT_CANARY, again);
    assert.ok(out.results.every((r) => r.spaceId === mainSpaceId));
    assert.equal((await spacesNamed(`${RUN}-main`)).filter((s) => s.name === `${RUN}-main`).length, 1, 'a duplicate space was created');
  });

  it('refuses a same-named space with another embedder, and the original embedder stays in effect', async () => {
    const conflicting = config({ space: { name: `${RUN}-main`, embedderId: badEmbedderId } });
    await assert.rejects(searchMemories('anything', conflicting), (e: any) => {
      assert.ok(GoodMemConfigError.isInstance(e));
      assert.match(e.message, new RegExp(`indexed by embedder\\(s\\) \\["${EMBEDDER}"\\], not "${badEmbedderId}"`));
      return true;
    });
    const space = await admin.spaces.get(mainSpaceId);
    assert.deepEqual((space.spaceEmbedders ?? []).map((e) => e.embedderId), [EMBEDDER]);
  });

  it('refuses a malformed embedder id with the server\'s field-level message, creating nothing', async () => {
    await assert.rejects(addMemories('x', config({ space: { name: `${RUN}-invalid`, embedderId: 'not-a-uuid' } })), (e: any) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.equal(e.statusCode, 400);
      assert.match(e.message, /Creating space ".*-invalid" failed: GoodMem answered HTTP 400: .*Invalid embedder ID format/);
      return true;
    });
    assert.deepEqual(await spacesNamed(`${RUN}-invalid`), []);
  });

  it('an unknown space id is a clear 404 for searches and a typed ingestion error for writes', async () => {
    const missing = config({ spaceId: MISSING_SPACE });
    await assert.rejects(searchMemories('x', missing), (e: any) => {
      assert.equal(e.statusCode, 404);
      assert.match(e.message, /HTTP 404: Space not found/);
      return true;
    });
    await assert.rejects(addMemories('x', missing), (e: any) => {
      assert.ok(GoodMemIngestionError.isInstance(e));
      assert.equal(e.failedIndex, 0);
      assert.deepEqual(e.createdMemoryIds, []);
      assert.ok(e.statusCode === 404 || e.statusCode === 400, `status ${e.statusCode}`);
      return true;
    });
  });

  // ----------------------------------------------------- tools, round trip --

  it('round-trips add and search through generateText tool calling', async () => {
    const tools = goodmemTools(mainConfig);
    const writer = scriptedModel(
      () => toolCall('addMemory', { text: `The live tool canary is ${TOOL_CANARY}.` }),
      () => textResult('Saved.')
    );
    const wrote = await generateText({ model: writer, tools, prompt: 'Remember the tool canary.', stopWhen: stepCountIs(3) });
    const saved = wrote.steps[0].toolResults[0].output as any;
    assert.equal(saved.spaceId, mainSpaceId);
    assert.equal(saved.processingStatus, 'PENDING');
    assert.equal(await waitCompleted(saved.memoryId), 'COMPLETED');

    const offered = writer.doGenerateCalls[0].tools as any[];
    assert.deepEqual(offered.map((t) => t.name).sort(), ['addMemory', 'searchMemories']);

    const reader = scriptedModel(
      () => toolCall('searchMemories', { query: `tool canary ${TOOL_CANARY}`, topK: 3 }),
      (prompt) => {
        const found = toolResultsIn(prompt)[0].output.value;
        const hit = found.results.find((r: any) => r.text.includes(TOOL_CANARY));
        return textResult(hit ? `Found it: ${hit.text.trim()}` : 'Not found.');
      }
    );
    const read = await generateText({ model: reader, tools, prompt: 'What is the tool canary?', stopWhen: stepCountIs(3) });
    assert.match(read.text, new RegExp(`Found it: .*${TOOL_CANARY}`));
    const output = read.steps[0].toolResults[0].output as any;
    assert.equal(output.partial, false);
    assert.ok(output.results.some((r: any) => r.memoryId === saved.memoryId));
  });

  // -------------------------------------------------------------- content --

  it('decodes text content as text', async () => {
    const { results } = await searchMemories(TEXT_CANARY, mainConfig, { includeContent: true });
    const hit = results.find((r) => r.memoryId === textMemoryId);
    assert.ok(hit, 'the text memory was not found');
    assert.equal(hit.contentEncoding, 'text');
    assert.equal(hit.content, TEXT);
  });

  it('decodes PDF content as base64, byte-identical to what was uploaded', async () => {
    const { results } = await searchMemories(`PDF canary ${PDF_CANARY}`, mainConfig, { includeContent: true });
    const hit = results.find((r) => r.memoryId === pdfMemoryId);
    assert.ok(hit, 'the PDF memory was not found');
    assert.equal(hit.contentType, 'application/pdf');
    assert.equal(hit.contentEncoding, 'base64');
    assert.ok(Buffer.from(hit.content as string, 'base64').equals(PDF));
    assert.match(hit.text, new RegExp(PDF_CANARY), 'the extracted PDF text was not searchable');
    JSON.stringify(results);
  });

  // --------------------------------------------------------------- scores --

  it('hits carry an oriented vector score, the raw score, ids and metadata', async () => {
    const { results, partial } = await searchMemories(`text canary ${TEXT_CANARY}`, mainConfig, { topK: 3 });
    assert.equal(partial, false);
    const hit = results[0];
    assert.equal(hit.memoryId, textMemoryId);
    assert.ok((hit.rawScore as number) < 0, 'GoodMem vector scores are negative');
    assert.equal(hit.score, -(hit.rawScore as number));
    assert.equal(hit.scoreKind, 'vector');
    assert.equal(hit.metadata.tenant, 'acme');
    assert.ok(hit.chunkId && hit.spaceId === mainSpaceId);
  });

  it('reranked results are scored as reranker, unflipped', { skip: RERANKER ? false : 'GOODMEM_TEST_RERANKER_ID is not set' }, async () => {
    const { results, partial } = await searchMemories(TEXT_CANARY, { ...mainConfig, rerankerId: RERANKER });
    assert.equal(partial, false);
    assert.ok(results.length > 0);
    for (const r of results) {
      assert.equal(r.scoreKind, 'reranker');
      assert.equal(r.score, r.rawScore);
    }
    assert.equal(results[0].memoryId, textMemoryId);
  });

  it('a reranker that does not exist degrades the search but keeps its hits (Q4a)', async () => {
    const logs = new Logs();
    const out = await searchMemories(TEXT_CANARY, { ...mainConfig, rerankerId: MISSING_SPACE, logger: logs });
    assert.equal(out.partial, true);
    assert.ok(out.results.length > 0, 'hits were discarded');
    assert.deepEqual(out.statuses.map((s) => s.code).sort(), ['NOT_FOUND', 'RERANKING_FAILED']);
    assert.equal(out.results[0].scoreKind, 'vector', 'vector scores were labelled as reranker scores');
    assert.match(logs.lines[0], /^\[goodmem\] GoodMem reported a problem during retrieval -- NOT_FOUND/);
  });

  // -------------------------------------------------- degraded embedder, Q4b --

  it('a failing embedder is reported through helpers, tools and middleware, never as "no memories"', async () => {
    const logs = new Logs();
    const broken = config({ space: { name: `${RUN}-broken`, embedderId: badEmbedderId }, logger: logs });

    // A write lands, then fails processing: reported, with the write kept.
    await assert.rejects(addMemories('Doomed canary.', broken, { waitForIndexing: { timeoutMs: 60_000 } }), (e: any) => {
      assert.ok(GoodMemIndexingError.isInstance(e));
      assert.equal(e.failedMemoryIds.length, 1);
      assert.match(e.message, /failed processing on the server.*The writes succeeded/);
      return true;
    });
    for (const s of await spacesNamed(`${RUN}-broken`)) created.spaces.add(s.spaceId);

    const helper = await searchMemories('doomed canary', broken);
    assert.equal(helper.partial, true);
    assert.deepEqual(helper.results, []);
    assert.equal(helper.statuses[0].code, 'EMBEDDER_FAILED');
    assert.match(helper.warning ?? '', /EMBEDDER_FAILED/);
    assert.ok(logs.lines.some((l) => /this is not an empty index/.test(l)));

    let toolOutput: any;
    await generateText({
      model: scriptedModel(
        () => toolCall('searchMemories', { query: 'doomed canary' }),
        (prompt) => {
          toolOutput = toolResultsIn(prompt)[0].output.value;
          return textResult('ok');
        }
      ),
      tools: goodmemTools(broken),
      prompt: 'search',
      stopWhen: stepCountIs(3),
    });
    assert.equal(toolOutput.partial, true);
    assert.equal(toolOutput.statuses[0].code, 'EMBEDDER_FAILED');

    const model = scriptedModel(() => textResult('answer'));
    const result = await generateText({ model: withGoodmem(model, broken), prompt: 'What did I save?' });
    assert.match(systemText(model.doGenerateCalls[0].prompt), /retrieval failed for this request \(EMBEDDER_FAILED\)/);
    assert.ok(result.warnings?.some((w: any) => /EMBEDDER_FAILED/.test(w.message)));
    assert.equal((result.providerMetadata?.goodmem as any).partial, true);
  });

  // -------------------------------------------------------- empty, latency --

  it('an empty space answers in under a second with a single request: nothing polls', async () => {
    // Counts requests, and times each one until its body has been read, so
    // the package's own share of the latency can be separated from the
    // server's.
    const stats = { requests: 0, serverMs: 0 };
    const timed: typeof fetch = async (input, init) => {
      stats.requests += 1;
      const t0 = performance.now();
      const response = await fetch(input, init);
      if (!response.body) {
        stats.serverMs += performance.now() - t0;
        return response;
      }
      const body = response.body.pipeThrough(
        new TransformStream({
          flush() {
            stats.serverMs += performance.now() - t0;
          },
        })
      );
      return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
    };
    const empty = config({ space: { name: `${RUN}-empty`, embedderId: LATENCY_EMBEDDER! }, fetch: timed });
    await searchMemories('warm up: resolves and creates the space', empty);
    for (const s of await spacesNamed(`${RUN}-empty`)) created.spaces.add(s.spaceId);

    // Every search embeds its query through the space's embedder on the
    // server; with a hosted embedder that alone varies from a few hundred
    // milliseconds to tens of seconds. So each attempt must make exactly one
    // request and add almost nothing to the server's own time, and at least
    // one of up to five attempts must finish in under a second -- a read path
    // that polled (0.1.x waited 23 s on an empty space) never would.
    const measure = async (label: string, run: (query: string) => Promise<void>) => {
      const attempts: Array<{ totalMs: number; serverMs: number }> = [];
      for (let i = 0; i < 5; i += 1) {
        stats.requests = 0;
        stats.serverMs = 0;
        const started = performance.now();
        await run(`nothing at all ${TOKEN} ${label} ${i}`);
        const totalMs = performance.now() - started;
        assert.equal(stats.requests, 1, `${label} made ${stats.requests} requests for one search`);
        assert.ok(totalMs - stats.serverMs < 150, `${label} added ${(totalMs - stats.serverMs).toFixed(0)} ms to the server's time`);
        attempts.push({ totalMs: Math.round(totalMs), serverMs: Math.round(stats.serverMs) });
        if (totalMs < 1000) break;
      }
      console.log(`# empty-space ${label} (embedder ${LATENCY_EMBEDDER}): ${JSON.stringify(attempts)}`);
      assert.ok(
        attempts.some((a) => a.totalMs < 1000),
        `no ${label} empty search finished in under a second: ${JSON.stringify(attempts)}. The package added ` +
          'under 150 ms to each; the rest is the server embedding the query -- set ' +
          'GOODMEM_TEST_LATENCY_EMBEDDER_ID to a faster embedder to measure the read path alone.'
      );
    };

    await measure('search', async (query) => {
      const out = await searchMemories(query, empty);
      assert.deepEqual(out.results, []);
      assert.equal(out.partial, false);
    });
    const emptyMiddleware = { ...empty, retrievalTimeoutMs: 90_000 };
    await generateText({ model: withGoodmem(scriptedModel(() => textResult('ok')), emptyMiddleware), prompt: 'warm up the space lookup' });
    await measure('middleware', async (query) => {
      const model = scriptedModel(() => textResult('ok'));
      const result = await generateText({ model: withGoodmem(model, emptyMiddleware), prompt: query });
      assert.equal(systemText(model.doGenerateCalls[0].prompt), '', 'something was injected for an empty space');
      assert.equal((result.providerMetadata?.goodmem as any).partial, false);
    });
  });

  // ----------------------------------------------------------- middleware --

  it('the middleware injects memories from a real space and reports them', async () => {
    const model = scriptedModel(() => textResult('answer'));
    const messages = [{ role: 'user' as const, content: `What is the live text canary ${TEXT_CANARY}?` }];
    const before = JSON.stringify(messages);
    const patient = { ...mainConfig, retrievalTimeoutMs: 90_000 };
    const result = await generateText({ model: withGoodmem(model, patient), system: 'Answer briefly.', messages });
    const injected = systemText(model.doGenerateCalls[0].prompt);
    assert.match(injected, /^Answer briefly\.\n\nRelevant memories from GoodMem, most relevant first:\n1\. /);
    assert.ok(injected.includes(TEXT_CANARY), 'the canary memory was not injected');
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.partial, false);
    assert.ok(meta.memoryIds.includes(textMemoryId));
    assert.equal(JSON.stringify(messages), before);

    const streamed = streamText({ model: withGoodmem(streamingModel(), patient), prompt: `canary ${TEXT_CANARY}` });
    assert.equal(await streamed.text, 'ok');
    assert.ok(((await streamed.providerMetadata)?.goodmem as any).memoryIds.includes(textMemoryId));
  });

  it('retrieveMemories formats a context block from a real space, in result order', async () => {
    const out = await retrieveMemories(`live text canary ${TEXT_CANARY}`, mainConfig, { topK: 2 });
    assert.equal(out.results.length, 2);
    assert.equal(
      out.context,
      ['Relevant memories from GoodMem, most relevant first:', ...out.results.map((r, i) => `${i + 1}. ${r.text.trim()}`)].join('\n')
    );
    assert.ok(out.context.includes(TEXT_CANARY));
  });

  // ---------------------------------------------------- filters and scope --

  it('scope and filters apply server-side, and an injection payload matches nothing', async () => {
    const scoped = await searchMemories('canary', { ...mainConfig, scope: { tenant: 'acme' }, topK: 10 });
    assert.ok(scoped.results.length > 0);
    assert.ok(scoped.results.every((r) => r.metadata.tenant === 'acme'));
    assert.ok(!scoped.results.some((r) => r.memoryId === weatherMemoryId));

    const injected = await searchMemories('canary', { ...mainConfig, scope: { tenant: "x' OR '1'='1" }, topK: 10 });
    assert.equal(injected.partial, false);
    assert.deepEqual(injected.results, [], 'filter injection matched rows');

    const recent = await searchMemories('note', { ...mainConfig, filter: filters.compare('year', '>=', 2026), topK: 10 });
    assert.ok(recent.results.length > 0 && recent.results.every((r) => r.metadata.year === 2026));

    const active = await searchMemories('note', { ...mainConfig, filter: filters.equals('active', true), topK: 10 });
    assert.deepEqual([...new Set(active.results.map((r) => r.memoryId))], [textMemoryId]);
    const inactive = await searchMemories('note', { ...mainConfig, filter: filters.not(filters.equals('active', true)), topK: 10 });
    assert.ok(inactive.results.some((r) => r.memoryId === weatherMemoryId));
    assert.ok(!inactive.results.some((r) => r.memoryId === textMemoryId));
  });

  it('scope is written onto stored memories', async () => {
    const scope = { userId: `u-${TOKEN.toLowerCase()}` };
    const { memories } = await addMemories({ text: 'A scoped note.', metadata: { topic: 'scoping' } }, { ...mainConfig, scope });
    const memory = await admin.memories.get(memories[0].memoryId);
    assert.deepEqual(memory.metadata, { topic: 'scoping', userId: scope.userId });
  });

  it("a malformed raw filter surfaces the server's parse error", async () => {
    await assert.rejects(searchMemories('x', { ...mainConfig, filter: "CAST(val('$.tenant') AS TEXT) = ''acme''" }), (e: any) => {
      assert.equal(e.statusCode, 400);
      assert.match(e.message, /HTTP 400: Invalid filter .*Parse error/);
      return true;
    });
  });

  // ------------------------------------------------ dead connection, timeout --

  it('a dead connection: helpers throw, tools report a tool error, the middleware carries on flagged unless told not to', async () => {
    const closed = await new Promise<number>((resolve) => {
      const server = createServer();
      server.listen(0, '127.0.0.1', () => {
        const port = (server.address() as { port: number }).port;
        server.close(() => resolve(port));
      });
    });
    const dead = config({ baseUrl: `http://127.0.0.1:${closed}`, spaceId: mainSpaceId });

    const started = Date.now();
    await assert.rejects(searchMemories('x', dead), (e: any) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.match(e.message, /could not reach the GoodMem server at http:\/\/127\.0\.0\.1:\d+ \(ECONNREFUSED\)/);
      return true;
    });
    assert.ok(Date.now() - started < 5000);

    // Default skipMemoryOnError: the call goes ahead, told memories may be missing, and is flagged.
    const logs = new Logs();
    const model = scriptedModel(() => textResult('answered without memories'));
    const carried = await generateText({ model: withGoodmem(model, { ...dead, logger: logs }), prompt: 'What do you remember about me?' });
    assert.equal(carried.text, 'answered without memories');
    assert.match(systemText(model.doGenerateCalls[0].prompt), /retrieval failed for this request \(RETRIEVAL_FAILED\).*do not assume there are none/);
    const meta = carried.providerMetadata?.goodmem as any;
    assert.equal(meta.partial, true);
    assert.equal(meta.resultCount, 0);
    assert.equal(meta.statuses[0].code, RETRIEVAL_FAILED_CODE);
    assert.match(meta.statuses[0].message, /could not reach the GoodMem server .*ECONNREFUSED/);
    assert.ok(carried.warnings?.some((w: any) => /memory lookup failed/.test(w.message)));
    assert.match(logs.lines[0], /skipMemoryOnError is on/);

    // skipMemoryOnError: false restores failing the call.
    const strict = scriptedModel(() => textResult('never'));
    await assert.rejects(
      generateText({ model: withGoodmem(strict, { ...dead, skipMemoryOnError: false }), prompt: 'hi' }),
      /could not reach the GoodMem server/
    );
    assert.equal(strict.doGenerateCalls.length, 0, 'the model was called with no memories');

    const result = await generateText({
      model: scriptedModel(() => toolCall('searchMemories', { query: 'x' }), () => textResult('sorry')),
      tools: goodmemTools(dead),
      prompt: 'hi',
      stopWhen: stepCountIs(3),
    });
    const toolError = result.steps[0].content.find((p: any) => p.type === 'tool-error') as any;
    assert.match(String(toolError?.error?.message), /could not reach the GoodMem server/);
  });

  it('a server that accepts but never answers: timeoutMs bounds helpers, retrievalTimeoutMs bounds the middleware', async () => {
    const sockets = new Set<Socket>();
    const silent: Server = createServer((socket) => {
      sockets.add(socket);
    });
    await new Promise<void>((resolve) => silent.listen(0, '127.0.0.1', resolve));
    const port = (silent.address() as { port: number }).port;
    try {
      const hung = config({ baseUrl: `http://127.0.0.1:${port}`, spaceId: mainSpaceId, timeoutMs: 500 });
      const started = Date.now();
      await assert.rejects(searchMemories('x', hung), (e: any) => {
        assert.equal(e.timedOut, true);
        assert.match(e.message, /timed out after 500 ms/);
        return true;
      });
      const elapsed = Date.now() - started;
      assert.ok(elapsed >= 450 && elapsed < 3000, `took ${elapsed} ms`);
      // The middleware gives up after retrievalTimeoutMs and carries on, flagged.
      const model = scriptedModel(() => textResult('ok'));
      const t0 = Date.now();
      const carried = await generateText({ model: withGoodmem(model, { ...hung, retrievalTimeoutMs: 700 }), prompt: 'hi' });
      const waited = Date.now() - t0;
      assert.ok(waited >= 650 && waited < 3000, `the middleware waited ${waited} ms`);
      const meta = carried.providerMetadata?.goodmem as any;
      assert.equal(meta.statuses[0].code, RETRIEVAL_FAILED_CODE);
      assert.deepEqual(meta.statuses[0].details, { timedOut: true });
      assert.match(meta.statuses[0].message, /timed out after 700 ms: .*raise retrievalTimeoutMs/);
      assert.equal(model.doGenerateCalls.length, 1);

      const strict = scriptedModel(() => textResult('never'));
      await assert.rejects(
        generateText({ model: withGoodmem(strict, { ...hung, skipMemoryOnError: false, retrievalTimeoutMs: 700 }), prompt: 'hi' }),
        /timed out after 700 ms/
      );
      assert.equal(strict.doGenerateCalls.length, 0);
    } finally {
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => silent.close(() => resolve()));
    }
  });

  // ------------------------------------------- environment, availability --

  it('falls back to GOODMEM_API_KEY and GOODMEM_BASE_URL end to end, and an explicit option wins', async () => {
    // This suite runs with both variables set; none of these configs name a server or key.
    const fromEnv = { spaceId: mainSpaceId, logger: new Logs() };
    const found = await searchMemories(`live text canary ${TEXT_CANARY}`, fromEnv);
    assert.ok(found.results.some((r) => r.memoryId === textMemoryId));

    const model = scriptedModel(() => textResult('ok'));
    await generateText({ model: withGoodmem(model, { ...fromEnv, retrievalTimeoutMs: 90_000 }), prompt: `canary ${TEXT_CANARY}` });
    assert.ok(systemText(model.doGenerateCalls[0].prompt).includes(TEXT_CANARY));

    const tool = goodmemTools(fromEnv).searchMemories as any;
    const output = await tool.execute({ query: TEXT_CANARY }, { toolCallId: 'env', messages: [], context: {} });
    assert.ok(output.results.length > 0);

    const saved = process.env.GOODMEM_BASE_URL;
    process.env.GOODMEM_BASE_URL = 'http://127.0.0.1:9';
    try {
      // Explicit baseUrl wins over a broken environment...
      const explicit = await searchMemories(TEXT_CANARY, { ...fromEnv, baseUrl: BASE! });
      assert.ok(explicit.results.length > 0);
      // ...and the environment really is what an omitted baseUrl uses.
      await assert.rejects(searchMemories(TEXT_CANARY, fromEnv), /could not reach the GoodMem server at http:\/\/127\.0\.0\.1:9 /);
    } finally {
      if (saved === undefined) delete process.env.GOODMEM_BASE_URL;
      else process.env.GOODMEM_BASE_URL = saved;
    }
  });

  it('a rejected API key (HTTP 401) still fails the middleware call with skipMemoryOnError on', async () => {
    const badKey = config({ apiKey: 'gm_invalid_live_test_key', spaceId: mainSpaceId });
    await assert.rejects(searchMemories('x', badKey), (e: any) => {
      assert.equal(e.statusCode, 401);
      assert.equal(e.isRetryable, false);
      return true;
    });
    const logs = new Logs();
    const model = scriptedModel(() => textResult('never'));
    await assert.rejects(generateText({ model: withGoodmem(model, { ...badKey, logger: logs }), prompt: 'hi' }), (e: any) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.equal(e.statusCode, 401);
      assert.match(e.message, /GoodMem answered HTTP 401/);
      return true;
    });
    assert.equal(model.doGenerateCalls.length, 0, 'the model was called although the key is wrong');
    assert.deepEqual(logs.lines, []);
  });

  it('a lookup slower than retrievalTimeoutMs against the real server carries on, flagged', async () => {
    const logs = new Logs();
    const model = scriptedModel(() => textResult('ok'));
    const started = Date.now();
    // A space id, so the time limit falls on the retrieval request itself.
    const result = await generateText({
      model: withGoodmem(model, config({ spaceId: mainSpaceId, logger: logs, retrievalTimeoutMs: 1 } as GoodmemConfig)),
      prompt: `live text canary ${TEXT_CANARY}`,
    });
    assert.ok(Date.now() - started < 3000);
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.partial, true);
    assert.equal(meta.statuses[0].code, RETRIEVAL_FAILED_CODE);
    assert.deepEqual(meta.statuses[0].details, { timedOut: true });
    assert.equal(model.doGenerateCalls.length, 1);
    assert.ok(!systemText(model.doGenerateCalls[0].prompt).includes(TEXT_CANARY));
    assert.match(logs.lines[0], /Searching GoodMem timed out after 1 ms: .*raise retrievalTimeoutMs/);
  });

  // --------------------------------------------------------------- secrets --

  it('no credential appears in results, tools or errors', async () => {
    const out = await searchMemories(TEXT_CANARY, mainConfig);
    let error: unknown;
    try {
      await searchMemories('x', config({ spaceId: MISSING_SPACE }));
    } catch (e) {
      error = e;
    }
    const dumps = [JSON.stringify(out), inspect(goodmemTools(mainConfig), { depth: 8 }), inspect(error, { depth: 8, showHidden: true })];
    for (const dump of dumps) assert.ok(!dump.includes(KEY!), 'the API key leaked');
  });

  // -------------------------------------------------------------- teardown --

  it('teardown deletes everything the run created, verified by a fresh listing', async () => {
    const { failures, leftSpaces, leftEmbedders } = await teardown();
    assert.deepEqual(failures, []);
    assert.deepEqual(leftSpaces, [], 'spaces survived teardown');
    assert.deepEqual(leftEmbedders, [], 'the temporary embedder survived teardown');
    assert.ok(created.spaces.size >= 3, `expected the run's spaces to be registered, got ${created.spaces.size}`);
  });
});
