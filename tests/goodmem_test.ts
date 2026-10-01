/**
 * Offline tests for @pairsystems/goodmem-vercel-ai-sdk.
 *
 * These drive the real @pairsystems/goodmem SDK over a fake `fetch` that
 * replays bytes captured from a live GoodMem server (see
 * tests/fixtures/manifest.json for the version), and drive the tools and the
 * middleware through the real `ai` package -- `generateText` and `streamText`
 * with the AI SDK's own mock language model -- rather than calling our
 * functions in isolation. No network, no environment variables.
 */

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { inspect } from 'node:util';
import { runInNewContext } from 'node:vm';
import { describe, it } from 'node:test';

import { generateText, stepCountIs, streamText, tool } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { z } from 'zod';

import {
  addMemories,
  FACT_EXTRACTION_INSTRUCTIONS,
  filters,
  GoodMemConfigError,
  GoodMemError,
  GoodMemFilterError,
  GoodMemIndexingError,
  GoodMemIngestionError,
  goodmemTools,
  MALFORMED_STREAM_CODE,
  RETRIEVAL_FAILED_CODE,
  retrieveMemories,
  searchMemories,
  UNKNOWN_CODE,
  withGoodmem,
  type GoodmemConfig,
} from '../src/index';
import { MAX_SPACES_SCANNED } from '../src/config';
import { environmentDefaults } from '../src/env';
import { DEFAULT_RETRIEVAL_TIMEOUT_MS } from '../src/middleware';
import { decodeContent, orientScore } from '../src/results';
import { FakeGoodmem, fixture, FIXTURES, hang, jsonResponse, manifest, replay } from './support/fake-goodmem';
import { scriptedModel, streamingModel, systemText, textResult, toolCall, toolResultsIn } from './support/models';

const ROOT = join(__dirname, '..');
const BASE = 'https://goodmem.test';
const KEY = 'gm_offline_test_key';

// Ids come from the captured fixtures, so a re-capture needs no edits here.
const OK_EVENTS = fixture('retrieve_ok.ndjson').toString('utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const memoryIdOf = (needle: string): string =>
  OK_EVENTS.find((e) => e.retrievedItem?.chunk.chunk.chunkText.includes(needle)).retrievedItem.chunk.chunk.memoryId;
const CANARY_MEMORY = memoryIdOf('ORYX-4471');
const WEATHER_MEMORY = memoryIdOf('Amman');
const SPACE: string = OK_EVENTS.find((e) => e.memoryDefinition).memoryDefinition.spaceId;

// ---------------------------------------------------------------- helpers --

class Logs {
  readonly lines: string[] = [];
  warn = (message: string) => {
    this.lines.push(message);
  };
}

function setup(overrides: Partial<GoodmemConfig> = {}) {
  const fake = new FakeGoodmem();
  const logs = new Logs();
  const config: GoodmemConfig = {
    apiKey: KEY,
    baseUrl: BASE,
    spaceId: SPACE,
    fetch: fake.fetch,
    logger: logs,
    ...overrides,
  };
  return { fake, logs, config };
}

function lines(name: string): string[] {
  return fixture(name).toString('utf8').split('\n').filter(Boolean);
}

function ndjson(body: string[] | string, status = 200): Response {
  const text = Array.isArray(body) ? `${body.join('\n')}\n` : body;
  return new Response(text, { status, headers: { 'content-type': 'application/x-ndjson; charset=utf-8' } });
}

async function rejectsWith<T extends Error>(promise: Promise<unknown>, check: (e: T) => void) {
  let caught: unknown;
  try {
    await promise;
  } catch (error) {
    caught = error;
  }
  assert.ok(caught, 'expected a rejection');
  check(caught as T);
}

/** Run `fn` with environment variables set (a string) or removed (undefined), then restore them. */
async function withEnv<T>(vars: Record<string, string | undefined>, fn: () => T | Promise<T>): Promise<T> {
  const saved = Object.fromEntries(Object.keys(vars).map((k) => [k, process.env[k]]));
  const apply = (values: Record<string, string | undefined>) => {
    for (const [k, v] of Object.entries(values)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  };
  apply(vars);
  try {
    return await fn();
  } finally {
    apply(saved);
  }
}

// --------------------------------------------------------------- fixtures --

describe('fixtures', () => {
  it('are bytes captured from a live server, with their recorded status', () => {
    assert.match(manifest.serverVersion, /^server-v\d+\.\d+\.\d+$/);
    const events = lines('retrieve_ok.ndjson').map((l) => JSON.parse(l));
    assert.ok(events.some((e) => e.resultSetBoundary));
    assert.ok(events.some((e) => e.retrievedItem));
    assert.match(fixture('retrieve_ok.ndjson').toString(), /[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-/);
    assert.equal(manifest.files['space_create.json'].status, 201);
    assert.equal(manifest.files['error_retrieve_missing_space.json'].status, 404);
  });

  it('carry no credential', () => {
    for (const name of readdirSync(FIXTURES)) {
      assert.doesNotMatch(fixture(name).toString('latin1'), /(^|[^A-Za-z0-9_])gm_[A-Za-z0-9]{20,}/, name);
    }
  });
});

// ---------------------------------------------------------- configuration --

describe('configuration', () => {
  it('requires apiKey, baseUrl and exactly one space option, naming what is missing', async () => {
    const fake = new FakeGoodmem();
    const base = { apiKey: KEY, baseUrl: BASE, spaceId: SPACE, fetch: fake.fetch };
    await withEnv({ GOODMEM_API_KEY: undefined, GOODMEM_BASE_URL: undefined }, () => {
      assert.throws(
        () => goodmemTools({ ...base, apiKey: undefined }),
        (e: any) =>
          GoodMemConfigError.isInstance(e) &&
          /^goodmemTools: apiKey is required: pass apiKey, or set the GOODMEM_API_KEY environment variable\. Get an instance URL and API key from GoodMem Cloud at https:\/\/cloud\.goodmem\.ai\/login .*or self-host: https:\/\/docs\.goodmem\.ai\.$/.test(e.message)
      );
      assert.throws(
        () => goodmemTools({ ...base, baseUrl: undefined }),
        /baseUrl is required: pass baseUrl, or set the GOODMEM_BASE_URL environment variable, e\.g\. https:\/\/gm-<name>-<id>\.app\.goodmem\.ai or http:\/\/localhost:8080\. There is no default server\. Get an instance URL and API key from GoodMem Cloud at https:\/\/cloud\.goodmem\.ai\/login/
      );
    });
    assert.throws(() => goodmemTools({ ...base, apiKey: '' }), /apiKey is empty: .*or omit apiKey to use GOODMEM_API_KEY/);
    assert.throws(() => goodmemTools({ ...base, baseUrl: ' ' }), /baseUrl is empty: .*or omit baseUrl to use GOODMEM_BASE_URL/);
    assert.throws(() => goodmemTools({ ...base, baseUrl: 'localhost:8080' }), /http:\/\/ or https:\/\//);
    assert.throws(() => goodmemTools({ apiKey: KEY, baseUrl: BASE } as GoodmemConfig), /a space is required.*never chooses a space/);
    assert.throws(() => goodmemTools({ ...base, spaceIds: [SPACE] }), /only one of spaceId, spaceIds and space/);
    assert.throws(() => goodmemTools({ ...base, spaceId: undefined, spaceIds: [] }), /non-empty array/);
    assert.throws(
      () => goodmemTools({ ...base, spaceId: undefined, space: { name: 'x' } as any }),
      /space must be \{ name, embedderId \}.*never picked for you/
    );
  });

  it('rejects unknown options instead of ignoring a typo', () => {
    const { config } = setup();
    assert.throws(
      () => goodmemTools({ ...config, topk: 3 } as any),
      (e: any) => /unknown option\(s\) "topk"/.test(e.message) && /Valid options: .*topK/.test(e.message)
    );
  });

  it('bounds every request: timeoutMs must be a positive number and cannot be turned off', () => {
    const { config } = setup();
    for (const timeoutMs of [0, -1, Number.POSITIVE_INFINITY, Number.NaN]) {
      assert.throws(() => goodmemTools({ ...config, timeoutMs }), /timeoutMs must be a positive number.*no way to turn the timeout off/);
    }
  });

  it('refuses minScore without a reranker, explaining why', () => {
    const { config } = setup();
    assert.throws(() => goodmemTools({ ...config, minScore: 0.5 }), /minScore applies only to reranker scores.*not on a 0-1 scale/);
  });

  it('validates scope field names and values up front', () => {
    const { config } = setup();
    assert.throws(() => goodmemTools({ ...config, scope: { 'user-id': 'u1' } }), /scope is not usable: .*hyphenated name/);
    assert.throws(() => goodmemTools({ ...config, scope: { userId: { nested: true } as any } }), /Unsupported filter value type/);
    assert.throws(() => goodmemTools({ ...config, topK: 0 }), /topK must be an integer from 1 to 100/);
  });

  it('withGoodmem refuses a model id string and an unknown position', () => {
    const { config } = setup();
    assert.throws(() => withGoodmem('openai/gpt-4o' as any, config), /language model instance/);
    assert.throws(() => withGoodmem(streamingModel(), { ...config, position: 'middle' as any }), /position must be 'system' or 'user'/);
  });
});

// ---------------------------------------------------- environment fallback --

describe('environment fallback (GOODMEM_API_KEY, GOODMEM_BASE_URL)', () => {
  const ENV_KEY = 'gm_environment_test_key';
  const ENV_URL = 'https://env.goodmem.test';

  it('uses both variables when the options are omitted', async () => {
    const fake = new FakeGoodmem().on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await withEnv({ GOODMEM_API_KEY: ENV_KEY, GOODMEM_BASE_URL: ENV_URL }, () =>
      searchMemories('canary', { spaceId: SPACE, fetch: fake.fetch })
    );
    assert.equal(fake.requests[0].url.origin, ENV_URL);
    assert.equal(fake.requests[0].headers.get('x-api-key'), ENV_KEY);
  });

  it('an explicit option always wins over the environment', async () => {
    const fake = new FakeGoodmem().on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await withEnv({ GOODMEM_API_KEY: ENV_KEY, GOODMEM_BASE_URL: ENV_URL }, () =>
      searchMemories('canary', { apiKey: KEY, baseUrl: BASE, spaceId: SPACE, fetch: fake.fetch })
    );
    assert.equal(fake.requests[0].url.origin, BASE);
    assert.equal(fake.requests[0].headers.get('x-api-key'), KEY);
  });

  it('each option falls back on its own, and an explicit empty value never does', async () => {
    const fake = new FakeGoodmem().on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await withEnv({ GOODMEM_API_KEY: ENV_KEY, GOODMEM_BASE_URL: ENV_URL }, async () => {
      await searchMemories('canary', { apiKey: KEY, spaceId: SPACE, fetch: fake.fetch });
      assert.throws(() => goodmemTools({ apiKey: '', spaceId: SPACE, fetch: fake.fetch }), /apiKey is empty/);
    });
    assert.equal(fake.requests[0].url.origin, ENV_URL);
    assert.equal(fake.requests[0].headers.get('x-api-key'), KEY);
  });

  it('neither option nor variable: the error names the option, the variable and where to get one', async () => {
    await withEnv({ GOODMEM_API_KEY: undefined, GOODMEM_BASE_URL: undefined }, async () => {
      await assert.rejects(searchMemories('q', { spaceId: SPACE }), (e: any) => {
        assert.ok(GoodMemConfigError.isInstance(e));
        assert.match(e.message, /apiKey is required: pass apiKey, or set the GOODMEM_API_KEY environment variable/);
        assert.match(e.message, /https:\/\/cloud\.goodmem\.ai\/login/);
        return true;
      });
      assert.throws(
        () => withGoodmem(streamingModel(), { apiKey: KEY, spaceId: SPACE }),
        /withGoodmem: baseUrl is required: pass baseUrl, or set the GOODMEM_BASE_URL environment variable.*https:\/\/cloud\.goodmem\.ai\/login/
      );
    });
  });

  it('a malformed GOODMEM_BASE_URL is reported as coming from the environment', async () => {
    await withEnv({ GOODMEM_API_KEY: ENV_KEY, GOODMEM_BASE_URL: 'not a url' }, () => {
      assert.throws(() => goodmemTools({ spaceId: SPACE }), /baseUrl "not a url" \(from GOODMEM_BASE_URL\) is not a valid URL/);
    });
  });

  it('a key from the environment never appears in tools, wrapped models, results or errors', async () => {
    const fake = new FakeGoodmem().on('POST', '/v1/memories:retrieve', 'error_retrieve_missing_space.json');
    await withEnv({ GOODMEM_API_KEY: ENV_KEY, GOODMEM_BASE_URL: ENV_URL }, async () => {
      const config = { spaceId: SPACE, fetch: fake.fetch };
      let error: unknown;
      try {
        await searchMemories('x', config);
      } catch (e) {
        error = e;
      }
      const dumps = [
        JSON.stringify(goodmemTools(config)),
        inspect(goodmemTools(config), { depth: 10, showHidden: true }),
        inspect(withGoodmem(streamingModel(), config), { depth: 10, showHidden: true }),
        inspect(error, { depth: 10, showHidden: true }),
        JSON.stringify(error),
      ];
      for (const dump of dumps) assert.ok(!dump.includes(ENV_KEY), 'the environment key leaked');
    });
  });

  it('runtimes without a Node environment, or that refuse access to it, get no fallback', () => {
    // The function's own compiled source, run where `process` is missing,
    // refuses access, or exists. Results are compared as JSON: they come from
    // another realm.
    const run = (sandbox: object) => JSON.parse(JSON.stringify(runInNewContext(`(${environmentDefaults.toString()})()`, sandbox)));
    assert.deepEqual(run({}), {}, 'no process global');
    const locked = { process: { get env() { throw new Error('Requires env access'); } } };
    assert.deepEqual(run(locked), {}, 'env access refused');
    assert.deepEqual(run({ process: {} }), {}, 'process without env');
    const node = { process: { env: { GOODMEM_API_KEY: ' k ', GOODMEM_BASE_URL: '', OTHER: 'x' } } };
    assert.deepEqual(run(node), { apiKey: 'k' });
  });

  it('only src/env.ts reads the environment, and only these two variables', () => {
    for (const file of readdirSync(join(ROOT, 'src'))) {
      const code = readFileSync(join(ROOT, 'src', file), 'utf8');
      if (file === 'env.ts') continue;
      assert.doesNotMatch(code, /\bprocess\s*(\.|\?\.|\[)|import\.meta\.env|Deno\.env|Bun\.env/, file);
    }
    const reads = readFileSync(join(ROOT, 'src', 'env.ts'), 'utf8')
      .split('\n')
      .filter((l) => !/^\s*(\*|\/\/|\/\*)/.test(l))
      .join('\n')
      .match(/\bprocess\b[^\s;,)]*/g);
    assert.deepEqual([...new Set(reads)].sort(), [
      'process',
      'process.env.GOODMEM_API_KEY',
      'process.env.GOODMEM_BASE_URL',
    ]);
  });
});

// ----------------------------------- middleware when GoodMem is unavailable --

describe('middleware when GoodMem is unavailable (skipMemoryOnError, retrievalTimeoutMs)', () => {
  const unreachable = (async () => {
    throw new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }) });
  }) as typeof fetch;

  function assertContinued(model: ReturnType<typeof scriptedModel>, result: any, logs: Logs, message: RegExp) {
    assert.equal(model.doGenerateCalls.length, 1, 'the model was not called');
    assert.match(
      systemText(model.doGenerateCalls[0].prompt),
      /GoodMem memory retrieval failed for this request \(RETRIEVAL_FAILED\)\. Relevant memories may exist but could not be retrieved; do not assume there are none\./
    );
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.partial, true);
    assert.equal(meta.resultCount, 0);
    assert.deepEqual(meta.memoryIds, []);
    assert.equal(meta.statuses.length, 1);
    assert.equal(meta.statuses[0].code, RETRIEVAL_FAILED_CODE);
    assert.match(meta.statuses[0].message, message);
    assert.ok(result.warnings?.some((w: any) => w.type === 'other' && /memory lookup failed, so this call went ahead without memories/.test(w.message)));
    assert.equal(logs.lines.length, 1);
    assert.match(logs.lines[0], /^\[goodmem\] memory lookup failed; calling the model without memories because skipMemoryOnError is on -- /);
    return meta;
  }

  it('an unreachable server: the call goes ahead, told memories may be missing, and is flagged', async () => {
    const { logs, config } = setup({ fetch: unreachable });
    const model = scriptedModel(() => textResult('answered anyway'));
    const result = await generateText({ model: withGoodmem(model, config), prompt: 'What is my name?' });
    assert.equal(result.text, 'answered anyway');
    assertContinued(model, result, logs, /could not reach the GoodMem server at https:\/\/goodmem\.test \(ECONNREFUSED\)/);
  });

  for (const status of [503, 500, 429, 408]) {
    it(`HTTP ${status} is an availability failure: the call goes ahead, flagged with the status`, async () => {
      const { fake, logs, config } = setup();
      fake.on('POST', '/v1/memories:retrieve', () => jsonResponse({ error: `server says ${status}` }, status));
      const model = scriptedModel(() => textResult('ok'));
      const result = await generateText({ model: withGoodmem(model, config), prompt: 'hi' });
      const meta = assertContinued(model, result, logs, new RegExp(`HTTP ${status}: server says ${status}`));
      assert.deepEqual(meta.statuses[0].details, { statusCode: status });
    });
  }

  it('a broken response (a 200 with no events) is an availability failure too', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', () => ndjson(''));
    const model = scriptedModel(() => textResult('ok'));
    const result = await generateText({ model: withGoodmem(model, config), prompt: 'hi' });
    assertContinued(model, result, logs, /empty stream/);
  });

  it('a lookup slower than retrievalTimeoutMs is cut off and the call goes ahead, flagged as a timeout', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', hang);
    const model = scriptedModel(() => textResult('ok'));
    const started = Date.now();
    const result = await generateText({ model: withGoodmem(model, { ...config, retrievalTimeoutMs: 100 }), prompt: 'hi' });
    const elapsed = Date.now() - started;
    assert.ok(elapsed >= 90 && elapsed < 1500, `took ${elapsed} ms`);
    const meta = assertContinued(model, result, logs, /timed out after 100 ms: .*raise retrievalTimeoutMs\./);
    assert.deepEqual(meta.statuses[0].details, { timedOut: true });
  });

  it('retrievalTimeoutMs bounds the whole lookup, including finding a named space', async () => {
    // timeoutMs still bounds the space listing itself, which carries on in the
    // background (and is cached for the next call) after the call went ahead.
    const { fake, logs, config } = setup({ spaceId: undefined, space: { name: 'slow-lookup', embedderId: 'e-1' }, timeoutMs: 300 });
    fake.on('GET', '/v1/spaces', hang);
    const model = scriptedModel(() => textResult('ok'));
    const started = Date.now();
    const result = await generateText({ model: withGoodmem(model, { ...config, retrievalTimeoutMs: 100 }), prompt: 'hi' });
    assert.ok(Date.now() - started < 1500);
    assertContinued(model, result, logs, /Looking up the configured space timed out after 100 ms.*raise retrievalTimeoutMs/);
  });

  it('retrievalTimeoutMs is the middleware\'s alone: helpers keep timeoutMs', async () => {
    const { fake, config } = setup({ timeoutMs: 400 });
    fake.on('POST', '/v1/memories:retrieve', hang);
    const started = Date.now();
    await assert.rejects(searchMemories('q', config), /timed out after 400 ms: .*raise timeoutMs\./);
    assert.ok(Date.now() - started >= 380);
    assert.throws(() => goodmemTools({ ...config, retrievalTimeoutMs: 100 } as any), /unknown option\(s\) "retrievalTimeoutMs"/);
    assert.equal(DEFAULT_RETRIEVAL_TIMEOUT_MS, 5000);
  });

  it('the streaming path goes ahead and is flagged the same way', async () => {
    const { config } = setup({ fetch: unreachable });
    const result = streamText({ model: withGoodmem(streamingModel(), config), prompt: 'hi' });
    assert.equal(await result.text, 'ok');
    assert.ok((await result.warnings)?.some((w: any) => /memory lookup failed/.test(w.message)));
    const meta = (await result.providerMetadata)?.goodmem as any;
    assert.equal(meta.partial, true);
    assert.equal(meta.statuses[0].code, RETRIEVAL_FAILED_CODE);
  });

  for (const [status, body] of [
    [400, { error: 'Invalid filter for space x: Parse error' }],
    [401, { error: 'Invalid API key' }],
    [403, { error: 'Permission denied' }],
  ] as const) {
    it(`HTTP ${status} still throws with skipMemoryOnError on: it will not fix itself`, async () => {
      const { fake, logs, config } = setup();
      fake.on('POST', '/v1/memories:retrieve', () => jsonResponse(body, status));
      const model = scriptedModel(() => textResult('never'));
      await rejectsWith<GoodMemError>(generateText({ model: withGoodmem(model, config), prompt: 'hi' }), (e) => {
        assert.ok(GoodMemError.isInstance(e));
        assert.equal(e.statusCode, status);
        assert.equal(e.isRetryable, false);
        assert.match(e.message, new RegExp(`HTTP ${status}: ${body.error}`));
      });
      assert.equal(model.doGenerateCalls.length, 0);
      assert.deepEqual(logs.lines, []);
    });
  }

  it('HTTP 404 (a missing space) still throws with skipMemoryOnError on', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'error_retrieve_missing_space.json');
    const model = scriptedModel(() => textResult('never'));
    await assert.rejects(generateText({ model: withGoodmem(model, config), prompt: 'hi' }), /HTTP 404: Space not found/);
    assert.equal(model.doGenerateCalls.length, 0);
  });

  it('a configuration error found during the lookup still throws', async () => {
    const page = JSON.parse(fixture('spaces_page2.json').toString());
    const { fake, config } = setup({ spaceId: undefined, space: { name: page.spaces[0].name, embedderId: 'another-embedder' } });
    fake.on('GET', '/v1/spaces', () => jsonResponse(page));
    const model = scriptedModel(() => textResult('never'));
    await rejectsWith(generateText({ model: withGoodmem(model, config), prompt: 'hi' }), (e) => {
      assert.ok(GoodMemConfigError.isInstance(e));
    });
    assert.equal(model.doGenerateCalls.length, 0);
  });

  it('skipMemoryOnError: false throws on a timeout, naming retrievalTimeoutMs', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', hang);
    const model = scriptedModel(() => textResult('never'));
    await rejectsWith<GoodMemError>(
      generateText({ model: withGoodmem(model, { ...config, skipMemoryOnError: false, retrievalTimeoutMs: 80 }), prompt: 'hi' }),
      (e) => {
        assert.equal(e.timedOut, true);
        assert.equal(e.isRetryable, true);
        assert.match(e.message, /timed out after 80 ms: .*raise retrievalTimeoutMs/);
      }
    );
    assert.equal(model.doGenerateCalls.length, 0);
  });

  it('validates skipMemoryOnError and retrievalTimeoutMs', () => {
    const { config } = setup();
    for (const retrievalTimeoutMs of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.throws(() => withGoodmem(streamingModel(), { ...config, retrievalTimeoutMs }), /retrievalTimeoutMs must be a positive number/);
    }
    assert.throws(() => withGoodmem(streamingModel(), { ...config, skipMemoryOnError: 'yes' as any }), /skipMemoryOnError must be true or false/);
  });
});

// ----------------------------------------------------- automatic saving --

describe('automatic saving (addMemory, extractionModel, extractFacts)', () => {
  const EXTRACTION_USAGE = {
    inputTokens: { total: 42, noCache: 42, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 9, text: 9, reasoning: 0 },
  };
  const CREATED = () => JSON.parse(fixture('memory_create.json').toString()).memoryId as string;

  /**
   * One mock model serving both kinds of call: structured-output calls (the
   * extraction, which asks for JSON) and ordinary calls (the chat).
   */
  function memoryModel(options: {
    facts?: unknown;
    extraction?: (call: any) => Promise<unknown>;
    answer?: (prompt: any[]) => unknown;
  } = {}) {
    const extraction: any[] = [];
    const main: any[] = [];
    const extract = async (call: any) => {
      extraction.push(call);
      if (options.extraction) return options.extraction(call);
      return {
        content: [{ type: 'text', text: JSON.stringify({ facts: options.facts ?? [] }) }],
        finishReason: { unified: 'stop', raw: undefined },
        usage: EXTRACTION_USAGE,
        warnings: [],
      };
    };
    const model = new MockLanguageModelV3({
      doGenerate: (async (call: any) => {
        if (call.responseFormat?.type === 'json') return extract(call);
        main.push(call);
        return options.answer ? options.answer(call.prompt) : textResult('ok');
      }) as any,
      doStream: (async (call: any) => {
        main.push(call);
        return streamingModel().doStream(call);
      }) as any,
    });
    return { model, extraction, main };
  }

  function saving(overrides: Partial<GoodmemConfig> = {}) {
    const env = setup({ scope: { userId: 'u-7' }, ...overrides });
    env.fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson').on('POST', '/v1/memories', 'memory_create.json');
    return env;
  }

  it('is off by default: no extraction call and nothing stored', async () => {
    const { fake, config } = saving();
    const { model, extraction } = memoryModel({ facts: ['User is vegetarian.'] });
    const result = await generateText({ model: withGoodmem(model, config), prompt: 'I am vegetarian.' });
    assert.equal(extraction.length, 0);
    assert.equal(fake.calls('POST', '/v1/memories').length, 0);
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.saved, undefined);
    assert.equal(meta.saveError, undefined);
  });

  it("'always': one extraction call per turn, facts stored with the scope, the outcome reported with usage", async () => {
    const { fake, logs, config } = saving();
    const { model, extraction, main } = memoryModel({ facts: ['User is vegetarian.', 'User lives in Amman.'] });
    const result = await generateText({
      model: withGoodmem(model, { ...config, addMemory: 'always' }),
      prompt: "I'm vegetarian and I live in Amman. Any dinner ideas?",
    });
    assert.equal(result.text, 'ok');
    assert.equal(main.length, 1);
    assert.equal(extraction.length, 1);
    const writes = fake.calls('POST', '/v1/memories').map((r) => r.body);
    assert.deepEqual(writes, [
      { spaceId: SPACE, originalContent: 'User is vegetarian.', contentType: 'text/plain', metadata: { userId: 'u-7' } },
      { spaceId: SPACE, originalContent: 'User lives in Amman.', contentType: 'text/plain', metadata: { userId: 'u-7' } },
    ]);
    const meta = result.providerMetadata?.goodmem as any;
    assert.deepEqual(meta.saved, {
      facts: ['User is vegetarian.', 'User lives in Amman.'],
      memoryIds: [CREATED(), CREATED()],
      duplicates: [],
      usage: { inputTokens: 42, outputTokens: 9, totalTokens: 51 },
    });
    assert.equal(meta.partial, false, 'the retrieval outcome is still reported');
    assert.deepEqual(logs.lines, []);
  });

  // OpenAI and Azure refuse response_format {type:'json_object'} unless the
  // messages contain the word "json": HTTP 400 "'messages' must contain the
  // word 'json' in some form, to use 'response_format' of type
  // 'json_object'." OpenAI-compatible providers send exactly that format for
  // a structured-output call and drop the schema, so the instructions alone
  // must name JSON and spell out the shape. Found against OpenRouter.
  it('the extraction instructions name JSON and spell out the exact shape', () => {
    assert.match(FACT_EXTRACTION_INSTRUCTIONS, /json/i);
    assert.ok(FACT_EXTRACTION_INSTRUCTIONS.includes('{"facts": ['), 'the output shape is not stated');
    assert.ok(FACT_EXTRACTION_INSTRUCTIONS.includes('{"facts": []}'), 'the empty answer is not stated');
    assert.match(FACT_EXTRACTION_INSTRUCTIONS, /no other keys/);
    assert.match(FACT_EXTRACTION_INSTRUCTIONS, /third person/);
    assert.match(FACT_EXTRACTION_INSTRUCTIONS, /questions/);
  });

  it('through an OpenAI-compatible provider: the request asks for json_object, its messages say json, and the answer round-trips', async () => {
    const { fake, logs, config } = saving();
    const sent: any[] = [];
    // Behaves like OpenAI/OpenRouter: rejects json_object unless a message mentions json.
    const provider = createOpenAICompatible({
      name: 'fake-openai',
      baseURL: 'https://llm.test/v1',
      apiKey: 'test-llm-key',
      fetch: (async (_url: any, init: any) => {
        const body = JSON.parse(String(init.body));
        sent.push(body);
        const mentionsJson = body.messages.some((m: any) => /json/i.test(typeof m.content === 'string' ? m.content : JSON.stringify(m.content)));
        if (body.response_format?.type === 'json_object' && !mentionsJson) {
          return jsonResponse(
            { error: { message: "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", type: 'invalid_request_error' } },
            400
          );
        }
        return jsonResponse({
          id: 'chatcmpl-test',
          object: 'chat.completion',
          created: 1790000000,
          model: body.model,
          choices: [{ index: 0, message: { role: 'assistant', content: '{"facts": ["User is vegetarian.", "User lives in Amman."]}' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 120, completion_tokens: 18, total_tokens: 138 },
        });
      }) as typeof fetch,
    });
    const { model } = memoryModel();
    const result = await generateText({
      model: withGoodmem(model, { ...config, addMemory: 'always', extractionModel: provider.chatModel('openai/gpt-4o-mini') }),
      prompt: "I'm vegetarian and I live in Amman.",
    });
    assert.equal(sent.length, 1);
    const request = sent[0];
    assert.deepEqual(request.response_format, { type: 'json_object' });
    assert.deepEqual(request.messages, [
      { role: 'system', content: FACT_EXTRACTION_INSTRUCTIONS },
      { role: 'user', content: "I'm vegetarian and I live in Amman." },
    ]);
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.saveError, undefined, meta.saveError?.message);
    assert.deepEqual(meta.saved.facts, ['User is vegetarian.', 'User lives in Amman.']);
    assert.deepEqual(meta.saved.usage, { inputTokens: 120, outputTokens: 18, totalTokens: 138 });
    assert.equal(fake.calls('POST', '/v1/memories').length, 2);
    assert.deepEqual(logs.lines, []);
  });

  it('extracts from the latest user message only: no injected memories, no assistant reply, no history', async () => {
    const { config } = saving();
    const { model, extraction } = memoryModel({ facts: [] });
    await generateText({
      model: withGoodmem(model, { ...config, addMemory: 'always' }),
      system: 'Be brief.',
      messages: [
        { role: 'user', content: 'My cat is called Pixel.' },
        { role: 'assistant', content: 'The user owns a yacht and speaks nine languages.' },
        { role: 'user', content: 'I just moved to Irbid.' },
      ],
    });
    const prompt = extraction[0].prompt as any[];
    assert.equal(prompt.length, 2);
    assert.deepEqual(prompt[0], { role: 'system', content: FACT_EXTRACTION_INSTRUCTIONS });
    assert.match(prompt[0].content, /json/i, 'the extraction prompt must mention JSON for json_object providers');
    assert.equal(prompt[1].role, 'user');
    assert.deepEqual(prompt[1].content.map((p: any) => p.text), ['I just moved to Irbid.']);
    const all = JSON.stringify(prompt);
    assert.doesNotMatch(all, /Relevant memories|ORYX-4471|yacht|Pixel|Be brief/);
    assert.equal(extraction[0].responseFormat.type, 'json');
    assert.deepEqual(Object.keys(extraction[0].responseFormat.schema.properties), ['facts']);
  });

  it('a tool loop extracts once, on the step that answers the user', async () => {
    const { config } = saving();
    let step = 0;
    const { model, extraction, main } = memoryModel({
      facts: ['User prefers aisle seats.'],
      answer: () => (++step === 1 ? toolCall('lookUpFlights', {}) : textResult('Booked an aisle seat.')),
    });
    const result = await generateText({
      model: withGoodmem(model, { ...config, addMemory: 'always' }),
      tools: { lookUpFlights: tool({ inputSchema: z.object({}), execute: async () => 'two flights' }) },
      prompt: 'I always sit in the aisle. Find me a flight to Cairo.',
      stopWhen: stepCountIs(3),
    });
    assert.equal(main.length, 2, 'expected two model steps');
    assert.equal(extraction.length, 1, 'the tool-result step extracted again');
    assert.deepEqual((result.steps[0].providerMetadata?.goodmem as any).saved.facts, ['User prefers aisle seats.']);
    assert.equal((result.steps[1].providerMetadata?.goodmem as any).saved, undefined);
  });

  it("skips a fact a memory retrieved for this turn already states, and duplicates within the turn", async () => {
    const { fake, config } = saving();
    const { model } = memoryModel({
      facts: ["the fixture canary is ORYX-4471.  O'Brien filed it", 'User is vegetarian.', 'user is vegetarian'],
    });
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'canary?' });
    const meta = result.providerMetadata?.goodmem as any;
    assert.deepEqual(meta.saved.facts, ['User is vegetarian.']);
    assert.deepEqual(meta.saved.duplicates, ["the fixture canary is ORYX-4471.  O'Brien filed it"]);
    assert.equal(fake.calls('POST', '/v1/memories').length, 1);
  });

  it('a question or small talk stores nothing', async () => {
    const { fake, config } = saving();
    const { model, extraction } = memoryModel({ facts: [] });
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'What time is it?' });
    assert.equal(extraction.length, 1);
    assert.equal(fake.calls('POST', '/v1/memories').length, 0);
    assert.deepEqual((result.providerMetadata?.goodmem as any).saved, {
      facts: [],
      memoryIds: [],
      duplicates: [],
      usage: { inputTokens: 42, outputTokens: 9, totalTokens: 51 },
    });
  });

  it('a failed extraction is flagged -- saveError, a warning, a log line -- and the call succeeds', async () => {
    const { fake, logs, config } = saving();
    const { model } = memoryModel({ extraction: async () => { throw new Error('extraction model overloaded'); } });
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am a nurse.' });
    assert.equal(result.text, 'ok');
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.saveError.stage, 'extraction');
    assert.match(meta.saveError.message, /^Extracting facts to remember failed: .*extraction model overloaded/);
    assert.ok(result.warnings?.some((w: any) => /automatic saving failed -- Extracting facts/.test(w.message)));
    assert.match(logs.lines[0], /^\[goodmem\] automatic saving failed at the extraction stage -- /);
    assert.equal(fake.calls('POST', '/v1/memories').length, 0);
  });

  it('extraction output that is not the expected JSON is a flagged extraction failure', async () => {
    const { config } = saving();
    const { model } = memoryModel({
      extraction: async () => ({ content: [{ type: 'text', text: 'Sure! The user is a nurse.' }], finishReason: { unified: 'stop', raw: undefined }, usage: EXTRACTION_USAGE, warnings: [] }),
    });
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am a nurse.' });
    assert.equal((result.providerMetadata?.goodmem as any).saveError.stage, 'extraction');
  });

  it('a failed save is flagged with the facts and the memories already stored, and the call succeeds', async () => {
    const { fake, logs, config } = setup({ scope: { userId: 'u-7' } });
    let writes = 0;
    fake
      .on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson')
      .on('POST', '/v1/memories', () => (++writes === 1 ? replay('memory_create.json') : jsonResponse({ error: 'overloaded' }, 503)));
    const { model } = memoryModel({ facts: ['User is a nurse.', 'User works nights.'] });
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am a night nurse.' });
    assert.equal(result.text, 'ok');
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.saveError.stage, 'save');
    assert.match(meta.saveError.message, /Storing memory 2 of 2 failed: GoodMem answered HTTP 503: overloaded/);
    assert.deepEqual(meta.saveError.facts, ['User is a nurse.', 'User works nights.']);
    assert.deepEqual(meta.saveError.memoryIds, [CREATED()]);
    assert.ok(result.warnings?.some((w: any) => /automatic saving failed/.test(w.message)));
    assert.match(logs.lines[0], /at the save stage/);
  });

  it('streaming: the outcome rides in the finish part, success or failure', async () => {
    const ok = saving();
    const streamed = memoryModel({ facts: ['User is left-handed.'] });
    const result = streamText({ model: withGoodmem(streamed.model, { ...ok.config, addMemory: 'always' }), prompt: "I'm left-handed." });
    assert.equal(await result.text, 'ok');
    assert.deepEqual(((await result.providerMetadata)?.goodmem as any).saved.facts, ['User is left-handed.']);
    assert.equal(streamed.extraction.length, 1);

    const failing = setup();
    failing.fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson').on('POST', '/v1/memories', () => jsonResponse({ error: 'down' }, 503));
    const broken = streamText({
      model: withGoodmem(memoryModel({ facts: ['User is left-handed.'] }).model, { ...failing.config, addMemory: 'always' }),
      prompt: "I'm left-handed.",
    });
    assert.equal(await broken.text, 'ok');
    assert.equal(((await broken.providerMetadata)?.goodmem as any).saveError.stage, 'save');
    assert.match(failing.logs.lines[0], /automatic saving failed at the save stage/);
  });

  it('extractionModel does the extraction when given; the wrapped model only chats', async () => {
    const { config } = saving();
    const chat = memoryModel({ facts: ['should not be asked'] });
    const cheap = memoryModel({ facts: ['User is a pilot.'] });
    const result = await generateText({
      model: withGoodmem(chat.model, { ...config, addMemory: 'always', extractionModel: cheap.model }),
      prompt: 'I fly planes for a living.',
    });
    assert.equal(chat.extraction.length, 0);
    assert.equal(chat.main.length, 1);
    assert.equal(cheap.extraction.length, 1);
    assert.deepEqual((result.providerMetadata?.goodmem as any).saved.facts, ['User is a pilot.']);
  });

  it('extractFacts replaces model extraction; its output is trimmed, de-duplicated and capped at 10', async () => {
    const { fake, config } = saving();
    const { model, extraction } = memoryModel();
    const seen: Array<{ text: string; signal: AbortSignal }> = [];
    const extractFacts = async (input: { text: string; signal: AbortSignal }) => {
      seen.push(input);
      return ['  User likes tea.  ', 'User likes tea', '', ...Array.from({ length: 12 }, (_, i) => `User fact ${i}.`)];
    };
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always', extractFacts }), prompt: 'I like tea.' });
    assert.equal(extraction.length, 0);
    assert.equal(seen[0].text, 'I like tea.');
    assert.ok(seen[0].signal instanceof AbortSignal);
    const saved = (result.providerMetadata?.goodmem as any).saved;
    assert.equal(saved.facts.length, 10);
    assert.equal(saved.facts[0], 'User likes tea.');
    assert.equal(saved.usage, undefined);
    assert.equal(fake.calls('POST', '/v1/memories').length, 10);
    assert.throws(
      () => withGoodmem(model, { ...config, addMemory: 'always', extractFacts, extractionModel: model }),
      /pass extractionModel or extractFacts, not both/
    );
  });

  it('the save is bounded by timeoutMs and reported as timed out', async () => {
    const { fake, logs, config } = saving({ timeoutMs: 150 });
    const { model } = memoryModel({
      extraction: (call: any) =>
        new Promise((_resolve, reject) => call.abortSignal?.addEventListener('abort', () => reject(call.abortSignal.reason))),
    });
    const started = Date.now();
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am a runner.' });
    assert.ok(Date.now() - started < 2000);
    const meta = result.providerMetadata?.goodmem as any;
    assert.match(meta.saveError.message, /Automatic saving timed out after 150 ms \(timeoutMs\)/);
    assert.match(logs.lines[0], /timed out/);
    assert.equal(fake.calls('POST', '/v1/memories').length, 0);
  });

  it("the caller's abort stops the save; nothing is stored", async () => {
    const { fake, config } = saving();
    let extractionSignal: AbortSignal | undefined;
    const { model } = memoryModel({
      extraction: (call: any) => {
        extractionSignal = call.abortSignal;
        return new Promise((_resolve, reject) => call.abortSignal?.addEventListener('abort', () => reject(call.abortSignal.reason)));
      },
    });
    const controller = new AbortController();
    setTimeout(() => controller.abort(new Error('user left')), 50);
    await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am a runner.', abortSignal: controller.signal }).catch(
      () => undefined
    );
    assert.equal(extractionSignal?.aborted, true, 'the extraction call was not aborted');
    assert.equal(fake.calls('POST', '/v1/memories').length, 0);
  });

  it('a failed model call cancels the save', async () => {
    const { fake, logs, config } = saving();
    const { model, extraction } = memoryModel({
      facts: ['User is a runner.'],
      answer: () => {
        throw new Error('provider down');
      },
    });
    await assert.rejects(generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I run.', maxRetries: 0 }), /provider down/);
    await new Promise((r) => setTimeout(r, 100));
    assert.equal(extraction.length, 1, 'the extraction was expected to have run alongside the call');
    assert.equal(fake.calls('POST', '/v1/memories').length, 0, 'facts were saved for a failed call');
    assert.deepEqual(logs.lines, []);
  });

  it('extraction runs alongside the model call, not after it', async () => {
    const { config } = saving();
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const { model } = memoryModel({
      extraction: async () => {
        await delay(250);
        return { content: [{ type: 'text', text: '{"facts":["User is tall."]}' }], finishReason: { unified: 'stop', raw: undefined }, usage: EXTRACTION_USAGE, warnings: [] };
      },
      answer: () => delay(250).then(() => textResult('ok')),
    });
    const started = Date.now();
    const result = await generateText({ model: withGoodmem(model, { ...config, addMemory: 'always' }), prompt: 'I am tall.' });
    const elapsed = Date.now() - started;
    assert.deepEqual((result.providerMetadata?.goodmem as any).saved.facts, ['User is tall.']);
    assert.ok(elapsed < 450, `extraction and the model call took ${elapsed} ms together: they did not overlap`);
  });

  it('a stream that fails part-way stores nothing', async () => {
    const { fake, config } = saving();
    const { model: extractor } = memoryModel({ facts: ['User is a chef.'] });
    const failingStream = new MockLanguageModelV3({
      doStream: (async () => ({
        stream: new ReadableStream({
          start(controller) {
            controller.enqueue({ type: 'stream-start', warnings: [] });
            controller.enqueue({ type: 'text-start', id: 't' });
            controller.enqueue({ type: 'text-delta', id: 't', delta: 'par' });
            controller.error(new Error('connection reset'));
          },
        }),
      })) as any,
    });
    const result = streamText({
      model: withGoodmem(failingStream, { ...config, addMemory: 'always', extractionModel: extractor }),
      prompt: 'I am a chef.',
      onError: () => {},
    });
    await Promise.resolve(result.consumeStream()).catch(() => undefined);
    await new Promise((r) => setTimeout(r, 100));
    assert.equal(fake.calls('POST', '/v1/memories').length, 0, 'facts were stored for a failed stream');
  });

  it('validates addMemory, extractionModel and extractFacts', () => {
    const { config } = saving();
    const model = streamingModel();
    assert.throws(() => withGoodmem(model, { ...config, addMemory: 'sometimes' as any }), /addMemory must be 'never' or 'always'/);
    assert.throws(() => withGoodmem(model, { ...config, extractionModel: 42 as any }), /extractionModel must be a language model/);
    assert.throws(() => withGoodmem(model, { ...config, extractFacts: 'yes' as any }), /extractFacts must be a function/);
  });
});

// ------------------------------------------------------------ isRetryable --

describe('errors say whether a retry can help (isRetryable)', () => {
  const cases: Array<[string, () => Response | Promise<Response>, boolean]> = [
    ['HTTP 503', () => jsonResponse({ error: 'overloaded' }, 503), true],
    ['HTTP 429', () => jsonResponse({ error: 'slow down' }, 429), true],
    ['HTTP 401', () => jsonResponse({ error: 'Invalid API key' }, 401), false],
    ['HTTP 404', () => replay('error_retrieve_missing_space.json'), false],
    ['an empty 200 stream', () => ndjson(''), true],
  ];
  for (const [label, respond, retryable] of cases) {
    it(`${label}: isRetryable is ${retryable}`, async () => {
      const { fake, config } = setup();
      fake.on('POST', '/v1/memories:retrieve', respond);
      await rejectsWith<GoodMemError>(searchMemories('q', config), (e) => assert.equal(e.isRetryable, retryable));
    });
  }

  it('an unreachable server and a timeout are retryable; a configuration error is not', async () => {
    const down = setup({ fetch: (async () => { throw new TypeError('fetch failed'); }) as typeof fetch });
    await rejectsWith<GoodMemError>(searchMemories('q', down.config), (e) => assert.equal(e.isRetryable, true));
    const slow = setup({ timeoutMs: 50 });
    slow.fake.on('POST', '/v1/memories:retrieve', hang);
    await rejectsWith<GoodMemError>(searchMemories('q', slow.config), (e) => {
      assert.equal(e.timedOut, true);
      assert.equal(e.isRetryable, true);
    });
    assert.throws(() => goodmemTools({ ...down.config, topK: 0 }), (e: any) => GoodMemConfigError.isInstance(e) && e.isRetryable === false);
  });
});

// -------------------------------------------------- retrieval status contract

describe('retrieval status contract', () => {
  it('a clean stream is not partial and logs nothing', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const out = await searchMemories('canary', config);
    assert.equal(out.partial, false);
    assert.deepEqual(out.statuses, []);
    assert.equal(out.warning, undefined);
    assert.equal(out.results.length, 2);
    assert.deepEqual(logs.lines, []);
  });

  it('Q1: FEATURE_DISABLED is noise by code alone', async () => {
    const { fake, config } = setup({ rerankerId: '019e6da0-8a5a-72b0-8656-04dddfb25762' });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_reranked.ndjson');
    assert.match(fixture('retrieve_reranked.ndjson').toString(), /FEATURE_DISABLED/);
    const out = await searchMemories('canary', config);
    assert.equal(out.partial, false);
    assert.deepEqual(out.statuses, []);
  });

  it('Q4a: a degraded retrieval still returns its hits, flagged, with a WARNING log', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_rerank_failed.ndjson');
    const out = await searchMemories('canary', config);
    assert.equal(out.results.length, 2, 'hits were discarded');
    assert.equal(out.partial, true);
    assert.deepEqual(out.statuses.map((s) => s.code), ['NOT_FOUND', 'RERANKING_FAILED']);
    assert.match(out.warning ?? '', /RERANKING_FAILED/);
    assert.equal(logs.lines.length, 1);
    assert.match(logs.lines[0], /^\[goodmem\] .*returning the 2 result\(s\) that did arrive/);
  });

  it('Q4b: a degraded retrieval with no hits is flagged, not an empty success, and does not throw', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_embedder_failed.ndjson');
    const out = await searchMemories('anything', config);
    assert.deepEqual(out.results, []);
    assert.equal(out.partial, true);
    assert.equal(out.statuses[0].code, 'EMBEDDER_FAILED');
    assert.match(out.warning ?? '', /EMBEDDER_FAILED: Embedding failed/);
    assert.match(logs.lines[0], /this is not an empty index: the retrieval failed/);
  });

  it('Q3: a status code this build does not know becomes UNKNOWN, partial, results kept', async () => {
    const { fake, config } = setup();
    const stream = lines('retrieve_ok.ndjson');
    stream.splice(1, 0, '{"status":{"code":"SOME_FUTURE_CODE","message":"a status from a newer server"}}');
    fake.on('POST', '/v1/memories:retrieve', () => ndjson(stream));
    const out = await searchMemories('canary', config);
    assert.equal(out.partial, true);
    assert.equal(out.statuses[0].code, UNKNOWN_CODE);
    assert.equal(out.statuses[0].message, 'a status from a newer server');
    assert.equal(out.results.length, 2);
  });

  it('a stream that breaks off keeps what arrived and says so', async () => {
    const { fake, config } = setup();
    const whole = fixture('retrieve_ok.ndjson');
    fake.on('POST', '/v1/memories:retrieve', () => replay('retrieve_ok.ndjson', whole.subarray(0, Math.floor(whole.length * 0.6))));
    const out = await searchMemories('canary', config);
    assert.equal(out.partial, true);
    assert.ok(out.statuses.some((s) => s.code === MALFORMED_STREAM_CODE));
    assert.ok(out.results.length >= 1, 'the chunk that did arrive was dropped');
  });

  it('a request that yields no event at all throws rather than reporting "nothing found"', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', () => ndjson(''));
    await rejectsWith(searchMemories('canary', config), (e) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.match(e.message, /Searching GoodMem failed: .*empty stream/);
    });
  });

  it('an empty space is a clean, empty result', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_empty.ndjson');
    const out = await retrieveMemories('anything', config);
    assert.deepEqual(out.results, []);
    assert.equal(out.partial, false);
    assert.equal(out.context, '');
    assert.deepEqual(logs.lines, []);
  });
});

// -------------------------------- degraded status through every entry point --

describe('degraded retrieval surfaces through tools, middleware and helpers', () => {
  it('tool: the model receives partial, statuses and a warning', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_embedder_failed.ndjson');
    let seen: any;
    const model = scriptedModel(
      () => toolCall('searchMemories', { query: 'what do you know?' }),
      (prompt) => {
        seen = toolResultsIn(prompt)[0].output.value;
        return textResult('done');
      }
    );
    const result = await generateText({ model, tools: goodmemTools(config), prompt: 'hi', stopWhen: stepCountIs(3) });
    assert.equal(result.text, 'done');
    assert.equal(seen.partial, true);
    assert.equal(seen.totalResults, 0);
    assert.equal(seen.statuses[0].code, 'EMBEDDER_FAILED');
    assert.match(seen.warning, /EMBEDDER_FAILED/);
  });

  it('middleware: the injected text says retrieval failed, and the call carries a warning and partial metadata', async () => {
    const { fake, logs, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_embedder_failed.ndjson');
    const model = scriptedModel(() => textResult('answer'));
    const result = await generateText({ model: withGoodmem(model, config), prompt: 'What is my name?' });
    const injected = systemText(model.doGenerateCalls[0].prompt);
    assert.match(injected, /GoodMem memory retrieval failed for this request \(EMBEDDER_FAILED\).*do not assume there are none/);
    assert.ok(result.warnings?.some((w: any) => w.type === 'other' && /EMBEDDER_FAILED/.test(w.message)));
    const meta = result.providerMetadata?.goodmem as any;
    assert.equal(meta.partial, true);
    assert.equal(meta.resultCount, 0);
    assert.equal(meta.statuses[0].code, 'EMBEDDER_FAILED');
    assert.equal(logs.lines.length, 1);
  });

  it('middleware (streaming): warning and partial metadata reach streamText', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_rerank_failed.ndjson');
    const result = streamText({ model: withGoodmem(streamingModel(), config), prompt: 'canary?' });
    assert.equal(await result.text, 'ok');
    assert.ok((await result.warnings)?.some((w: any) => /RERANKING_FAILED/.test(w.message)));
    assert.equal(((await result.providerMetadata)?.goodmem as any).partial, true);
  });

  it('retrieveMemories: the context says results may be missing', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_rerank_failed.ndjson');
    const out = await retrieveMemories('canary', config);
    assert.match(out.context, /^Relevant memories from GoodMem, most relevant first:\n1\. The fixture canary is ORYX-4471/);
    assert.match(out.context, /Note: GoodMem reported a problem during this retrieval \(NOT_FOUND, RERANKING_FAILED\)/);
    assert.equal(out.partial, true);
  });
});

// ---------------------------------------------------------------- failures --

describe('failures throw, carrying the server\'s explanation', () => {
  it('an HTTP error keeps its status and body', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'error_retrieve_missing_space.json');
    await rejectsWith<GoodMemError>(searchMemories('canary', config), (e) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.equal(e.statusCode, 404);
      assert.equal(e.body, fixture('error_retrieve_missing_space.json').toString());
      assert.match(e.message, /^Searching GoodMem failed: GoodMem answered HTTP 404: Space not found: 0{8}-/);
      assert.equal(e.timedOut, false);
    });
  });

  it('an unreachable server says which URL and why', async () => {
    const { config } = setup({
      fetch: (async () => {
        throw new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }) });
      }) as typeof fetch,
    });
    await rejectsWith<GoodMemError>(searchMemories('canary', config), (e) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.match(e.message, /could not reach the GoodMem server at https:\/\/goodmem\.test \(ECONNREFUSED\)\. Check baseUrl/);
    });
  });

  it('tool: a failed request becomes an AI SDK tool error with the server message', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'error_retrieve_missing_space.json');
    let errorPart: any;
    const model = scriptedModel(
      () => toolCall('searchMemories', { query: 'x' }),
      (prompt) => {
        errorPart = toolResultsIn(prompt)[0];
        return textResult('sorry');
      }
    );
    const result = await generateText({ model, tools: goodmemTools(config), prompt: 'hi', stopWhen: stepCountIs(3) });
    const toolError = result.steps[0].content.find((p: any) => p.type === 'tool-error') as any;
    assert.ok(toolError, 'no tool-error part');
    assert.ok(GoodMemError.isInstance(toolError.error));
    assert.match(String(toolError.error.message), /HTTP 404: Space not found/);
    assert.match(JSON.stringify(errorPart.output), /HTTP 404: Space not found/);
  });

  it('middleware with skipMemoryOnError: false: an unreachable server fails the call; the model is never called', async () => {
    const { config } = setup({ fetch: (async () => { throw new TypeError('fetch failed'); }) as typeof fetch });
    const model = scriptedModel(() => textResult('should not run'));
    await rejectsWith(generateText({ model: withGoodmem(model, { ...config, skipMemoryOnError: false }), prompt: 'hello' }), (e) => {
      assert.ok(GoodMemError.isInstance(e), `got ${String(e)}`);
      assert.match(e.message, /could not reach the GoodMem server/);
    });
    assert.equal(model.doGenerateCalls.length, 0);
  });
});

// ---------------------------------------------------------------- timeouts --

describe('timeouts and cancellation', () => {
  it('a request that never answers is cut off at timeoutMs', async () => {
    const { fake, config } = setup({ timeoutMs: 150 });
    fake.on('POST', '/v1/memories:retrieve', hang);
    const started = Date.now();
    await rejectsWith<GoodMemError>(searchMemories('canary', config), (e) => {
      assert.ok(GoodMemError.isInstance(e));
      assert.equal(e.timedOut, true);
      assert.match(e.message, /timed out after 150 ms: the GoodMem server at https:\/\/goodmem\.test did not answer in time/);
    });
    const elapsed = Date.now() - started;
    assert.ok(elapsed >= 140 && elapsed < 2000, `took ${elapsed} ms`);
  });

  it('a stream that stalls mid-way is cut off too, keeping what arrived', async () => {
    const { fake, config } = setup({ timeoutMs: 150 });
    const head = `${lines('retrieve_ok.ndjson').slice(0, 3).join('\n')}\n`;
    fake.on('POST', '/v1/memories:retrieve', (request) => {
      const body = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(head));
          request.signal?.addEventListener('abort', () => controller.error(request.signal?.reason));
        },
      });
      return new Response(body, { status: 200, headers: { 'content-type': 'application/x-ndjson' } });
    });
    const started = Date.now();
    const out = await searchMemories('canary', config);
    assert.ok(Date.now() - started < 2000);
    assert.equal(out.partial, true);
    const broken = out.statuses.find((s) => s.code === MALFORMED_STREAM_CODE);
    assert.match(broken?.message ?? '', /request timed out/);
    assert.equal(out.results.length, 1);
  });

  it('every request the package makes carries an abort signal', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'bounded', embedderId: 'e-1' } });
    fake
      .on('GET', '/v1/spaces', () => jsonResponse({ spaces: [] }))
      .on('POST', '/v1/spaces', 'space_create.json')
      .on('POST', '/v1/memories', 'memory_create.json')
      .on('GET', /^\/v1\/memories\/[^/]+$/, 'memory_get_completed.json')
      .on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await addMemories('remember this', config, { waitForIndexing: { pollIntervalMs: 5 } });
    await searchMemories('canary', config);
    assert.ok(fake.requests.length >= 5);
    for (const request of fake.requests) {
      assert.ok(request.signal instanceof AbortSignal, `${request.method} ${request.url.pathname} had no signal`);
    }
  });

  it('a caller abort cancels the request and is rethrown as the abort', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', hang);
    const controller = new AbortController();
    const reason = new Error('user navigated away');
    setTimeout(() => controller.abort(reason), 30);
    await rejectsWith(searchMemories('canary', config, { signal: controller.signal }), (e) => assert.equal(e, reason));
  });

  it('middleware forwards the call\'s abortSignal to GoodMem', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', hang);
    const controller = new AbortController();
    setTimeout(() => controller.abort(new Error('stop')), 30);
    const model = scriptedModel(() => textResult('never'));
    await rejectsWith(generateText({ model: withGoodmem(model, config), prompt: 'hi', abortSignal: controller.signal }), (e) =>
      assert.match(String(e.message ?? e), /stop|abort/i)
    );
    assert.equal(model.doGenerateCalls.length, 0);
  });
});

// ------------------------------------------------------------------ scores --

describe('scores', () => {
  it('vector scores are flipped to higher-is-better, raw kept, order preserved', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const { results } = await searchMemories('canary', config);
    for (const r of results) {
      assert.equal(r.scoreKind, 'vector');
      assert.ok((r.rawScore as number) < 0, 'GoodMem vector scores are negative');
      assert.equal(r.score, -(r.rawScore as number));
    }
    assert.match(results[0].text, /ORYX-4471/, 'server order was not kept');
    assert.ok((results[0].score as number) > (results[1].score as number));
  });

  it('reranker scores come from the rerank stage and are never flipped', async () => {
    const { fake, config } = setup({ rerankerId: '019e6da0-8a5a-72b0-8656-04dddfb25762' });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_reranked.ndjson');
    const { results } = await searchMemories('canary', config);
    assert.equal(results[0].scoreKind, 'reranker');
    assert.equal(results[0].score, results[0].rawScore);
    assert.ok((results[0].score as number) > 0.5);
    const body = fake.calls('POST', '/v1/memories:retrieve')[0].body;
    assert.deepEqual(body.postProcessor, {
      name: 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory',
      config: { reranker_id: '019e6da0-8a5a-72b0-8656-04dddfb25762' },
    });
  });

  it('a reranker that failed does not mislabel the vector scores that came back', async () => {
    const { fake, config } = setup({ rerankerId: '00000000-0000-7000-8000-000000000000' });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_rerank_failed.ndjson');
    const { results, partial } = await searchMemories('canary', config);
    assert.equal(partial, true);
    assert.equal(results[0].scoreKind, 'vector');
    assert.equal(results[0].score, -(results[0].rawScore as number));
  });

  it('minScore filters reranker scores only, and warns with the observed range when it removes everything', async () => {
    const { fake, logs, config } = setup({ rerankerId: 'r-1', minScore: 0.95 });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_reranked.ndjson');
    const out = await searchMemories('canary', config);
    assert.deepEqual(out.results, []);
    assert.match(logs.lines[0], /minScore=0\.95 removed all 2 reranked result\(s\); observed scores ranged 0\.17\d+\.\.0\.81\d+.*not 0-1/);

    const vector = setup({ rerankerId: 'r-1', minScore: 0.95 });
    vector.fake.on('POST', '/v1/memories:retrieve', 'retrieve_rerank_failed.ndjson');
    assert.equal((await searchMemories('canary', vector.config)).results.length, 2, 'a threshold was applied to vector scores');
  });

  it('sends no reranker or threshold unless configured', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('canary', config);
    const body = fake.calls('POST', '/v1/memories:retrieve')[0].body;
    assert.equal(body.postProcessor, undefined);
    assert.doesNotMatch(JSON.stringify(body), /threshold/i);
  });

  it('orientScore flips only vector scores', () => {
    assert.equal(orientScore(-0.51, 'vector'), 0.51);
    assert.equal(orientScore(-0.14, 'reranker'), -0.14);
    assert.equal(orientScore(null, 'vector'), null);
  });
});

// ----------------------------------------------------------- stream joining --

describe('stream joining', () => {
  it('joins chunks to memories by UUID, whatever the order and memoryIndex', async () => {
    const { fake, config } = setup();
    const events = lines('retrieve_ok.ndjson').map((l) => JSON.parse(l));
    const definitions = events.filter((e) => e.memoryDefinition);
    const items = events.filter((e) => e.retrievedItem);
    for (const item of items) item.retrievedItem.chunk.memoryIndex = 7; // positional index made useless
    const reordered = [events[0], ...items, ...definitions.reverse(), events[events.length - 1]].map((e) => JSON.stringify(e));
    fake.on('POST', '/v1/memories:retrieve', () => ndjson(reordered));
    const { results } = await searchMemories('canary', config);
    const canary = results.find((r) => r.text.includes('ORYX-4471'));
    const weather = results.find((r) => r.text.includes('Amman'));
    assert.equal(canary?.metadata.tenant, 'acme');
    assert.equal(weather?.metadata.tenant, 'globex');
    assert.equal(canary?.spaceId, SPACE);
  });

  it('de-duplicates by chunk id, never by memory id', async () => {
    const { fake, config } = setup();
    const events = lines('retrieve_ok.ndjson');
    const item = JSON.parse(events[2]);
    const sibling = JSON.parse(events[2]);
    sibling.retrievedItem.chunk.chunk.chunkId = '01a0f254-c917-77fd-ba8c-3472cb87ffff';
    sibling.retrievedItem.chunk.chunk.chunkText = 'A second chunk of the same memory.';
    const stream = [...events.slice(0, 3), JSON.stringify(item), JSON.stringify(sibling), ...events.slice(3)];
    fake.on('POST', '/v1/memories:retrieve', () => ndjson(stream));
    const { results } = await searchMemories('canary', config);
    assert.equal(results.filter((r) => r.chunkId === item.retrievedItem.chunk.chunk.chunkId).length, 1, 'duplicate chunk kept');
    assert.ok(results.some((r) => r.text === 'A second chunk of the same memory.'), 'a distinct chunk of the same memory was dropped');
  });
});

// --------------------------------------------------------- content decoding --

describe('content decoding', () => {
  it('includeContent returns text as text and a PDF as base64, byte-identical', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_with_content.ndjson');
    const { results } = await searchMemories('canary', config, { includeContent: true, topK: 10 });
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.fetchMemoryContent, true);
    const text = results.find((r) => r.contentType === 'text/plain' && r.text.includes('ORYX'));
    assert.equal(text?.contentEncoding, 'text');
    assert.equal(text?.content, "The fixture canary is ORYX-4471. O'Brien filed it.");
    const pdf = results.find((r) => r.contentType === 'application/pdf');
    assert.equal(pdf?.contentEncoding, 'base64');
    assert.ok(Buffer.from(pdf?.content as string, 'base64').equals(fixture('canary.pdf')));
    assert.match(pdf?.text ?? '', /LYNX-2208/);
    JSON.stringify(results);
  });

  it('does not ask for content unless requested', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const { results } = await searchMemories('canary', config);
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.fetchMemoryContent, undefined);
    assert.equal(results[0].content, undefined);
  });

  it('decodes by content type and charset, and keeps undecodable text as base64', () => {
    assert.deepEqual(decodeContent(new TextEncoder().encode('héllo'), 'text/plain'), { content: 'héllo', encoding: 'text' });
    assert.deepEqual(decodeContent(Uint8Array.from([0x68, 0xe9]), 'text/plain; charset=iso-8859-1'), { content: 'hé', encoding: 'text' });
    assert.equal(decodeContent(Uint8Array.from([0xff, 0xfe, 0xfd]), 'text/plain').encoding, 'base64');
    const binary = Uint8Array.from([0x25, 0x50, 0x44, 0x46, 0x00, 0xff]);
    const out = decodeContent(binary, 'application/pdf');
    assert.equal(out.encoding, 'base64');
    assert.deepEqual(Uint8Array.from(Buffer.from(out.content, 'base64')), binary);
    assert.equal(decodeContent(new TextEncoder().encode('{"a":1}'), 'application/json').encoding, 'text');
  });
});

// ------------------------------------------------------------------- tools --

describe('tools', () => {
  it('the model is offered exactly two tools, with minimal inputs (as sent to the provider)', async () => {
    const { config } = setup();
    const model = scriptedModel(() => textResult('hi'));
    await generateText({ model, tools: goodmemTools(config), prompt: 'hello' });
    const offered = (model.doGenerateCalls[0].tools ?? []) as any[];
    assert.deepEqual(offered.map((t) => t.name).sort(), ['addMemory', 'searchMemories']);
    const search = offered.find((t) => t.name === 'searchMemories');
    const add = offered.find((t) => t.name === 'addMemory');
    assert.deepEqual(Object.keys(search.inputSchema.properties).sort(), ['query', 'topK']);
    assert.deepEqual(search.inputSchema.required, ['query']);
    assert.equal(search.inputSchema.properties.topK.maximum, 50);
    assert.deepEqual(Object.keys(add.inputSchema.properties), ['text']);
    for (const tool of offered) {
      assert.doesNotMatch(JSON.stringify(tool.inputSchema), /space|path|file|filter|delete/i);
      assert.ok(tool.description.length > 40);
    }
  });

  it('round-trips through generateText: the model calls searchMemories and receives real results', async () => {
    const { fake, config } = setup({ scope: { tenant: 'acme' } });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const model = scriptedModel(
      () => toolCall('searchMemories', { query: 'fixture canary', topK: 3 }),
      (prompt) => {
        const found = toolResultsIn(prompt)[0].output.value;
        return textResult(`The canary is in: ${found.results[0].text.trim()}`);
      }
    );
    const result = await generateText({ model, tools: goodmemTools(config), prompt: 'What is the canary?', stopWhen: stepCountIs(3) });
    assert.match(result.text, /ORYX-4471/);
    const request = fake.calls('POST', '/v1/memories:retrieve')[0].body;
    assert.equal(request.message, 'fixture canary');
    assert.equal(request.requestedSize, 3);
    assert.deepEqual(request.spaceKeys, [{ spaceId: SPACE, filter: "CAST(val('$.tenant') AS TEXT) = 'acme'" }]);
    const output = result.steps[0].toolResults[0].output as any;
    assert.equal(output.results[0].memoryId, CANARY_MEMORY);
    assert.equal(output.results[0].metadata.tenant, 'acme');
    assert.equal(output.results[0].content, undefined);
  });

  it('addMemory stores text in the configured space with the scope, and does not wait', async () => {
    const { fake, config } = setup({ scope: { userId: 'u-42' } });
    fake.on('POST', '/v1/memories', 'memory_create.json');
    const model = scriptedModel(
      () => toolCall('addMemory', { text: 'The user prefers window seats.' }),
      () => textResult('saved')
    );
    const result = await generateText({ model, tools: goodmemTools(config), prompt: 'remember', stopWhen: stepCountIs(3) });
    const create = fake.calls('POST', '/v1/memories');
    assert.equal(create.length, 1);
    assert.deepEqual(create[0].body, {
      spaceId: SPACE,
      originalContent: 'The user prefers window seats.',
      contentType: 'text/plain',
      metadata: { userId: 'u-42' },
    });
    assert.equal(fake.requests.length, 1, 'the tool polled after writing');
    const output = result.steps[0].toolResults[0].output as any;
    assert.deepEqual(Object.keys(output).sort(), ['memoryId', 'processingStatus', 'spaceId']);
    assert.equal(output.processingStatus, 'PENDING');
  });

  it('an out-of-range topK from the model is rejected before any request', async () => {
    const { fake, config } = setup();
    const model = scriptedModel(() => toolCall('searchMemories', { query: 'x', topK: 500 }), () => textResult('ok'));
    const result = await generateText({ model, tools: goodmemTools(config), prompt: 'hi', stopWhen: stepCountIs(3) });
    assert.equal(fake.requests.length, 0);
    assert.ok(result.steps[0].content.some((p: any) => p.type === 'tool-error'));
  });
});

// -------------------------------------------------------------- middleware --

describe('middleware', () => {
  it('appends memories to the leading system message, or adds one', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const model = scriptedModel(() => textResult('ok'));
    const wrapped = withGoodmem(model, config);
    await generateText({ model: wrapped, system: 'You are terse.', prompt: 'What is the canary?' });
    const first = model.doGenerateCalls[0].prompt as any[];
    assert.equal(first.filter((m) => m.role === 'system').length, 1);
    assert.match(first[0].content, /^You are terse\.\n\nRelevant memories from GoodMem, most relevant first:\n1\. The fixture canary is ORYX-4471/);

    await generateText({ model: wrapped, prompt: 'What is the canary?' });
    const second = model.doGenerateCalls[1].prompt as any[];
    assert.equal(second[0].role, 'system');
    assert.match(second[0].content, /^Relevant memories from GoodMem/);
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.message, 'What is the canary?');
  });

  it("position 'user' prepends memories to the latest user message", async () => {
    const { fake, config } = setup({});
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const model = scriptedModel(() => textResult('ok'));
    await generateText({ model: withGoodmem(model, { ...config, position: 'user' }), prompt: 'canary?' });
    const prompt = model.doGenerateCalls[0].prompt as any[];
    assert.equal(prompt.some((m) => m.role === 'system'), false);
    const user = prompt[prompt.length - 1];
    assert.match(user.content[0].text, /^Relevant memories from GoodMem/);
    assert.equal(user.content[1].text, 'canary?');
  });

  it('a template controls the injected text; returning an empty string injects nothing', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const model = scriptedModel(() => textResult('ok'));
    const template = (r: any) => `<memories>${r.results.map((x: any) => x.memoryId).join(',')}</memories>`;
    await generateText({ model: withGoodmem(model, { ...config, template }), prompt: 'canary?' });
    assert.equal(systemText(model.doGenerateCalls[0].prompt), `<memories>${CANARY_MEMORY},${WEATHER_MEMORY}</memories>`);
    const silent = scriptedModel(() => textResult('ok'));
    const result = await generateText({ model: withGoodmem(silent, { ...config, template: () => '' }), prompt: 'canary?' });
    assert.equal(systemText(silent.doGenerateCalls[0].prompt), '');
    assert.equal((result.providerMetadata?.goodmem as any).resultCount, 2, 'the retrieval outcome was not reported');
  });

  it("does not modify the caller's messages or put memories into the conversation", async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const messages = [
      { role: 'user' as const, content: 'What is the canary?' },
      { role: 'assistant' as const, content: 'Let me check.' },
      { role: 'user' as const, content: 'Go on.' },
    ];
    const before = JSON.stringify(messages);
    const model = scriptedModel(() => textResult('It is ORYX-4471.'));
    const result = await generateText({ model: withGoodmem(model, config), system: 'Be brief.', messages });
    assert.equal(JSON.stringify(messages), before);
    assert.doesNotMatch(JSON.stringify(result.response.messages), /Relevant memories/);
  });

  it('a call with no user text is passed through without a retrieval', async () => {
    const { fake, config } = setup();
    const model = scriptedModel(() => textResult('ok'));
    const result = await generateText({ model: withGoodmem(model, config), system: 'Only a system prompt.', messages: [{ role: 'assistant', content: 'hello' }] });
    assert.equal(fake.requests.length, 0);
    assert.equal(result.providerMetadata?.goodmem, undefined);
  });

  it('searches the latest user message and reports the memories used', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const model = scriptedModel(() => textResult('ok'));
    const result = await generateText({
      model: withGoodmem(model, config),
      messages: [
        { role: 'user', content: 'first question' },
        { role: 'assistant', content: 'first answer' },
        { role: 'user', content: 'second question' },
      ],
    });
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.message, 'second question');
    assert.deepEqual(result.providerMetadata?.goodmem, {
      partial: false,
      resultCount: 2,
      memoryIds: [CANARY_MEMORY, WEATHER_MEMORY],
      statuses: [],
    });
    assert.equal(result.warnings?.length ?? 0, 0);
  });
});

// ------------------------------------------------------------------ writes --

describe('writes', () => {
  it('never sends publicRead (the server rejects it with HTTP 400)', async () => {
    assert.match(fixture('error_space_publicread_400.json').toString(), /Unrecognized field \\"publicRead\\"/);
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'no-public-read', embedderId: 'e-1' } });
    fake
      .on('GET', '/v1/spaces', () => jsonResponse({ spaces: [] }))
      .on('POST', '/v1/spaces', 'space_create.json')
      .on('POST', '/v1/memories', 'memory_create.json');
    await addMemories({ text: 'x', metadata: { a: 1 } }, config);
    for (const request of fake.requests) {
      assert.doesNotMatch(JSON.stringify(request.body ?? {}), /publicRead|public_read/i);
    }
    assert.deepEqual(Object.keys(fake.calls('POST', '/v1/spaces')[0].body).sort(), ['defaultChunkingConfig', 'name', 'spaceEmbedders']);
    for (const file of readdirSync(join(ROOT, 'src'))) {
      const code = readFileSync(join(ROOT, 'src', file), 'utf8')
        .split('\n')
        .filter((l) => !/^\s*(\*|\/\/)/.test(l))
        .join('\n');
      assert.doesNotMatch(code, /publicRead|public_read/i, file);
    }
  });

  it('uploads a PDF as bytes with its content type, as multipart through the SDK', async () => {
    const { fake, config } = setup({ scope: { tenant: 'acme' } });
    fake.on('POST', '/v1/memories', 'memory_create.json');
    const pdf = fixture('canary.pdf');
    const out = await addMemories({ data: pdf, contentType: 'application/pdf', filename: 'canary.pdf', metadata: { kind: 'pdf' } }, config);
    const form = fake.calls('POST', '/v1/memories')[0].body as FormData;
    assert.ok(form instanceof FormData, 'not a multipart upload');
    const part = form.get('request') as string | Blob;
    const request = JSON.parse(typeof part === 'string' ? part : await part.text());
    assert.deepEqual(request, { spaceId: SPACE, contentType: 'application/pdf', metadata: { kind: 'pdf', tenant: 'acme' } });
    const file = form.get('file') as File;
    assert.equal(file.name, 'canary.pdf');
    assert.ok(Buffer.from(await file.arrayBuffer()).equals(pdf));
    assert.equal(out.memories[0].memoryId, JSON.parse(fixture('memory_create.json').toString()).memoryId);
  });

  it('validates every input before writing anything', async () => {
    const { fake, config } = setup({ scope: { userId: 'u-1' } });
    fake.on('POST', '/v1/memories', 'memory_create.json');
    await assert.rejects(addMemories(['fine', { data: new Uint8Array(), contentType: 'application/pdf' }], config), /input 1: data must be a non-empty/);
    await assert.rejects(addMemories(['fine', { text: 'x', metadata: { userId: 'someone-else' } }], config), /scope requires "u-1".*cannot be overridden/);
    await assert.rejects(addMemories([{ data: new Uint8Array([1]) } as any], config), /contentType is required with data/);
    await assert.rejects(addMemories([], config), /at least one memory/);
    assert.equal(fake.requests.length, 0);
  });

  it('a failure part-way reports the memories already stored and where to resume', async () => {
    const { fake, config } = setup();
    let n = 0;
    fake.on('POST', '/v1/memories', () => (++n === 1 ? replay('memory_create.json') : jsonResponse({ error: 'Space not found: x', status: 404 }, 404)));
    const created = JSON.parse(fixture('memory_create.json').toString()).memoryId;
    await rejectsWith<GoodMemIngestionError>(addMemories(['one', 'two', 'three'], config), (e) => {
      assert.ok(GoodMemIngestionError.isInstance(e));
      assert.deepEqual(e.createdMemoryIds, [created]);
      assert.equal(e.failedIndex, 1);
      assert.equal(e.statusCode, 404);
      assert.match(e.message, /^Storing memory 2 of 3 failed: GoodMem answered HTTP 404: Space not found: x \(1 earlier memory was stored: .*retry from input 1/);
    });
    assert.equal(n, 2, 'inputs after the failure were attempted');
  });

  it('does not wait for indexing unless asked', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories', 'memory_create.json');
    const out = await addMemories('x', config);
    assert.equal(fake.requests.length, 1);
    assert.equal(out.memories[0].processingStatus, 'PENDING');
  });

  it('waitForIndexing polls only the memories it created until they complete', async () => {
    const { fake, config } = setup();
    let polls = 0;
    fake
      .on('POST', '/v1/memories', 'memory_create.json')
      .on('GET', /^\/v1\/memories\/[^/]+$/, () => (++polls < 3 ? replay('memory_get_pending.json') : replay('memory_get_completed.json')));
    const out = await addMemories('x', config, { waitForIndexing: { pollIntervalMs: 5 } });
    assert.equal(out.memories[0].processingStatus, 'COMPLETED');
    const id = out.memories[0].memoryId;
    assert.ok(fake.calls('GET', /^\/v1\/memories\//).every((r) => r.url.pathname === `/v1/memories/${id}`));
    assert.equal(polls, 3);
    assert.equal(fake.calls('POST', '/v1/memories:retrieve').length, 0, 'waited by searching');
  });

  it('a memory that fails processing is reported, and the write is said to have succeeded', async () => {
    const { fake, config } = setup();
    const failed = JSON.parse(fixture('memory_get_completed.json').toString());
    failed.processingStatus = 'FAILED';
    failed.processingHistory = { attempts: [{ statusMessage: 'Embedding failed: Failed to create inference client' }] };
    fake.on('POST', '/v1/memories', 'memory_create.json').on('GET', /^\/v1\/memories\/[^/]+$/, () => jsonResponse(failed));
    await rejectsWith<GoodMemIndexingError>(addMemories('x', config, { waitForIndexing: true }), (e) => {
      assert.ok(GoodMemIndexingError.isInstance(e));
      assert.deepEqual(e.failedMemoryIds, [failed.memoryId]);
      assert.deepEqual(e.pendingMemoryIds, []);
      assert.match(e.message, /1 memory was stored, but 1 memory failed processing on the server .*Embedding failed.*The writes succeeded; do not store them again/);
    });
  });

  it('a wait that runs out is bounded and reported as pending, not failed', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories', 'memory_create.json').on('GET', /^\/v1\/memories\/[^/]+$/, 'memory_get_pending.json');
    const started = Date.now();
    await rejectsWith<GoodMemIndexingError>(addMemories('x', config, { waitForIndexing: { timeoutMs: 60, pollIntervalMs: 10 } }), (e) => {
      assert.equal(e.timedOut, true);
      assert.equal(e.pendingMemoryIds.length, 1);
      assert.match(e.message, /had not finished indexing after 60 ms/);
    });
    assert.ok(Date.now() - started < 2000);
  });
});

// ---------------------------------------------- named spaces and pagination --

describe('named space: create, reuse and pagination', () => {
  const page1 = JSON.parse(fixture('spaces_page1.json').toString());
  const page2 = JSON.parse(fixture('spaces_page2.json').toString());
  const onSecondPage = page2.spaces[0];
  const embedderOf = (space: any) => space.spaceEmbedders[0].embedderId;

  function listing(fake: FakeGoodmem) {
    fake.on('GET', '/v1/spaces', (request) => (request.url.searchParams.get('nextToken') ? replay('spaces_page2.json') : replay('spaces_page1.json')));
  }

  it('follows the listing to a match on the second page and reuses it', async () => {
    assert.ok(page1.nextToken, 'fixture page 1 must have a next page');
    const { fake, config } = setup({ spaceId: undefined, space: { name: onSecondPage.name, embedderId: embedderOf(onSecondPage) } });
    listing(fake);
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('canary', config);
    const lists = fake.calls('GET', '/v1/spaces');
    assert.equal(lists.length, 2);
    assert.equal(lists[1].url.searchParams.get('nextToken'), page1.nextToken);
    assert.equal(lists[0].url.searchParams.get('nameFilter'), onSecondPage.name);
    assert.equal(fake.calls('POST', '/v1/spaces').length, 0, 'created a duplicate');
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.spaceKeys[0].spaceId, onSecondPage.spaceId);
  });

  it('creates the space with the requested embedder when none exists, then reuses it', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'brand-new', embedderId: '019cfd94-2844-7117-85ca-1b9919758a26' } });
    fake
      .on('GET', '/v1/spaces', () => jsonResponse({ spaces: [] }))
      .on('POST', '/v1/spaces', 'space_create.json')
      .on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('one', config);
    await searchMemories('two', config);
    const creates = fake.calls('POST', '/v1/spaces');
    assert.equal(creates.length, 1, 'an identical second use created another space');
    assert.equal(creates[0].body.name, 'brand-new');
    assert.deepEqual(creates[0].body.spaceEmbedders, [{ embedderId: '019cfd94-2844-7117-85ca-1b9919758a26', defaultRetrievalWeight: 1 }]);
    const created = JSON.parse(fixture('space_create.json').toString()).spaceId;
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[1].body.spaceKeys[0].spaceId, created);
  });

  it('refuses to reuse a same-named space indexed by another embedder, naming both', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: onSecondPage.name, embedderId: 'another-embedder' } });
    listing(fake);
    await rejectsWith<GoodMemConfigError>(searchMemories('canary', config), (e) => {
      assert.ok(GoodMemConfigError.isInstance(e));
      assert.match(e.message, new RegExp(`indexed by embedder\\(s\\) \\["${embedderOf(onSecondPage)}"\\], not "another-embedder".*cannot be\\s+changed after creation`));
    });
    assert.equal(fake.calls('POST', '/v1/spaces').length, 0);
  });

  it('refuses to guess between two spaces with the same name', async () => {
    const twin = { ...onSecondPage, spaceId: '01a0f255-0000-7000-8000-000000000001' };
    const { fake, config } = setup({ spaceId: undefined, space: { name: onSecondPage.name, embedderId: embedderOf(onSecondPage) } });
    fake.on('GET', '/v1/spaces', () => jsonResponse({ spaces: [onSecondPage, twin] }));
    await assert.rejects(searchMemories('canary', config), /2 spaces visible to this API key are named .*refusing to guess/);
  });

  it('a server that repeats a page token fails loudly instead of looping', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'loop', embedderId: 'e' } });
    fake.on('GET', '/v1/spaces', () => jsonResponse({ spaces: [], nextToken: 'same-token' }));
    await assert.rejects(searchMemories('canary', config), /Looking up space "loop" failed: .*Pagination loop detected/);
  });

  it('a runaway listing is bounded and refused, never silently truncated', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'needle', embedderId: 'e' } });
    let page = 0;
    fake.on('GET', '/v1/spaces', () => {
      page += 1;
      const spaces = Array.from({ length: 1000 }, (_, i) => ({ spaceId: `s-${page}-${i}`, name: `other-${page}-${i}`, spaceEmbedders: [] }));
      return jsonResponse({ spaces, nextToken: `t-${page}` });
    });
    await assert.rejects(searchMemories('canary', config), new RegExp(`paged through ${MAX_SPACES_SCANNED} spaces without reaching the end.*Pass spaceId`));
    assert.equal(fake.calls('POST', '/v1/spaces').length, 0);
  });

  it('a space created concurrently by someone else (409) is found and reused', async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: onSecondPage.name, embedderId: embedderOf(onSecondPage) } });
    let lists = 0;
    fake
      .on('GET', '/v1/spaces', () => (++lists === 1 ? jsonResponse({ spaces: [] }) : jsonResponse({ spaces: [onSecondPage] })))
      .on('POST', '/v1/spaces', 'error_space_create_409.json')
      .on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('canary', config);
    assert.equal(fake.calls('POST', '/v1/memories:retrieve')[0].body.spaceKeys[0].spaceId, onSecondPage.spaceId);
  });

  it("a rejected create carries the server's field-level message", async () => {
    const { fake, config } = setup({ spaceId: undefined, space: { name: 'bad-embedder', embedderId: 'not-a-uuid' } });
    fake.on('GET', '/v1/spaces', () => jsonResponse({ spaces: [] })).on('POST', '/v1/spaces', 'error_space_create_400.json');
    await rejectsWith<GoodMemError>(addMemories('x', config), (e) => {
      assert.equal(e.statusCode, 400);
      assert.match(e.message, /Creating space "bad-embedder" failed: GoodMem answered HTTP 400: spaceEmbedders\[0\]\.embedderId: Invalid embedder ID format/);
    });
  });
});

// ----------------------------------------------------------------- filters --

describe('filters', () => {
  it('escapes an apostrophe with a backslash, not by doubling, and doubles backslashes', () => {
    assert.equal(filters.equals('n', "o'brien"), "CAST(val('$.n') AS TEXT) = 'o\\'brien'");
    assert.equal(filters.equals('p', 'a\\b'), "CAST(val('$.p') AS TEXT) = 'a\\\\b'");
  });

  it('refuses control characters', () => {
    assert.throws(() => filters.equals('f', 'a\nb'), GoodMemFilterError);
    assert.throws(() => filters.escapeLiteral('tab\there'), /control characters/);
  });

  it('casts booleans as BOOLEAN and numbers as NUMERIC, never as TEXT', () => {
    assert.equal(filters.equals('a', true), "CAST(val('$.a') AS BOOLEAN) = true");
    assert.equal(filters.equals('year', 2026), "CAST(val('$.year') AS NUMERIC) = 2026");
    assert.throws(() => filters.equals('n', Number.NaN), /finite/);
  });

  it('refuses unsafe and hyphenated field names, allows nested ones', () => {
    assert.throws(() => filters.equals("a' OR '1", 'x'), GoodMemFilterError);
    assert.throws(() => filters.equals('user-id', 'x'), /hyphenated name is accepted by the server but never matches/);
    assert.equal(filters.equals('profile.team', 'x'), "CAST(val('$.profile.team') AS TEXT) = 'x'");
  });

  it('builds comparisons, sets, negation and combinations', () => {
    assert.equal(filters.compare('year', '>=', 2000), "CAST(val('$.year') AS NUMERIC) >= 2000");
    assert.equal(filters.oneOf('tag', ['a', 'b']), "CAST(val('$.tag') AS TEXT) IN ('a', 'b')");
    assert.throws(() => filters.oneOf('tag', ['a', 1]), /same type/);
    assert.equal(filters.not(filters.equals('a', true)), "NOT (CAST(val('$.a') AS BOOLEAN) = true)");
    assert.equal(filters.allOf('x', '', undefined, 'y'), '(x) AND (y)');
    assert.equal(filters.anyOf('x'), 'x');
    assert.equal(filters.fromMapping({ b: 1, a: 'z' }), "(CAST(val('$.a') AS TEXT) = 'z') AND (CAST(val('$.b') AS NUMERIC) = 1)");
  });

  it('scope and filter reach every space key, and an injection payload stays a literal', async () => {
    const { fake, config } = setup({
      spaceId: undefined,
      spaceIds: [SPACE, 'second-space'],
      scope: { tenant: "x' OR '1'='1" },
      filter: filters.compare('year', '>=', 2026),
    });
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('canary', config);
    const expected = "(CAST(val('$.tenant') AS TEXT) = 'x\\' OR \\'1\\'=\\'1') AND (CAST(val('$.year') AS NUMERIC) >= 2026)";
    assert.deepEqual(fake.calls('POST', '/v1/memories:retrieve')[0].body.spaceKeys, [
      { spaceId: SPACE, filter: expected },
      { spaceId: 'second-space', filter: expected },
    ]);
  });
});

// ------------------------------------------------------ secrets, isolation --

describe('secrets and isolation', () => {
  it('the API key never appears in tools, wrapped models, results or errors', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'error_retrieve_missing_space.json');
    const tools = goodmemTools(config);
    const wrapped = withGoodmem(streamingModel(), config);
    let error: unknown;
    try {
      await searchMemories('x', config);
    } catch (e) {
      error = e;
    }
    const dumps = [
      JSON.stringify(tools),
      inspect(tools, { depth: 10, showHidden: true }),
      inspect(wrapped, { depth: 10, showHidden: true }),
      inspect(error, { depth: 10, showHidden: true }),
      JSON.stringify(error),
    ];
    for (const dump of dumps) assert.doesNotMatch(dump, new RegExp(KEY));
  });

  it('the key is sent only in the x-api-key header', async () => {
    const { fake, config } = setup();
    fake.on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    await searchMemories('canary', config);
    const [request] = fake.requests;
    assert.equal(request.headers.get('x-api-key'), KEY);
    assert.doesNotMatch(request.url.href + JSON.stringify(request.body), new RegExp(KEY));
  });

  it('two configurations keep their own client, spaces and scope', async () => {
    const a = setup({ spaceId: 'space-a', scope: { tenant: 'a' } });
    const b = setup({ spaceId: 'space-b', scope: { tenant: 'b' } });
    a.fake.on('POST', '/v1/memories', 'memory_create.json').on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    b.fake.on('POST', '/v1/memories', 'memory_create.json').on('POST', '/v1/memories:retrieve', 'retrieve_ok.ndjson');
    const toolsA = goodmemTools(a.config);
    const toolsB = goodmemTools(b.config);
    const call = (t: any, input: any) => t.execute(input, { toolCallId: 'c', messages: [], context: {} });
    await call(toolsB.addMemory, { text: 'for b' });
    await call(toolsA.addMemory, { text: 'for a' });
    await call(toolsA.searchMemories, { query: 'q' });
    await call(toolsB.searchMemories, { query: 'q' });
    assert.equal(a.fake.calls('POST', '/v1/memories')[0].body.spaceId, 'space-a');
    assert.equal(b.fake.calls('POST', '/v1/memories')[0].body.spaceId, 'space-b');
    assert.deepEqual(a.fake.calls('POST', '/v1/memories')[0].body.metadata, { tenant: 'a' });
    assert.match(b.fake.calls('POST', '/v1/memories:retrieve')[0].body.spaceKeys[0].filter, /'b'$/);
    assert.equal(a.fake.requests.length, 2);
    assert.equal(b.fake.requests.length, 2);
  });
});

// ------------------------------------------------------------------ README --

describe('README', () => {
  it('every TypeScript example compiles against the package', () => {
    const run = spawnSync(process.execPath, [join(ROOT, 'scripts', 'check-readme.mjs'), '--against', 'src', '--compile-only'], {
      cwd: ROOT,
      encoding: 'utf8',
    });
    assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  });
});
