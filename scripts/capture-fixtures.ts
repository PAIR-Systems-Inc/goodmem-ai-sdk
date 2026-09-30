/**
 * Captures the offline suite's fixtures from a live GoodMem server.
 *
 * The offline tests feed these bytes to the real @pairsystems/goodmem SDK
 * through a fake `fetch`, so every shape they assert on is one the server
 * actually produced. Setup and teardown go through the SDK; the responses
 * that become fixtures are read with plain `fetch` so the bytes are kept
 * exactly as they came off the wire.
 *
 *   GOODMEM_BASE_URL=http://localhost:8080 GOODMEM_API_KEY=... \
 *   GOODMEM_TEST_EMBEDDER_ID=... [GOODMEM_TEST_RERANKER_ID=...] \
 *     npm run capture-fixtures
 *
 * Every resource it creates is deleted before it exits, and the manifest it
 * writes records the server version and each response's HTTP status.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { Goodmem } from '@pairsystems/goodmem';

import { makePdf } from '../tests/support/pdf';

const BASE = process.env.GOODMEM_BASE_URL;
const KEY = process.env.GOODMEM_API_KEY;
const EMBEDDER = process.env.GOODMEM_TEST_EMBEDDER_ID;
const RERANKER = process.env.GOODMEM_TEST_RERANKER_ID;
if (!BASE || !KEY || !EMBEDDER) {
  console.error('Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_TEST_EMBEDDER_ID.');
  process.exit(2);
}

const OUT = join(__dirname, '..', 'tests', 'fixtures');
mkdirSync(OUT, { recursive: true });
const client = new Goodmem({ baseUrl: BASE as string, apiKey: KEY, timeoutMs: 30_000 });
const run = `fx${Date.now().toString(36)}`;
const manifest: { capturedAt: string; serverVersion?: string; files: Record<string, unknown> } = {
  capturedAt: new Date().toISOString(),
  files: {},
};
const spaces: string[] = [];
const embedders: string[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function capture(name: string, method: string, path: string, body?: unknown, accept = 'application/json') {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      'x-api-key': KEY as string,
      accept,
      ...(body === undefined ? {} : { 'content-type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const bytes = Buffer.from(await response.arrayBuffer());
  writeFileSync(join(OUT, name), bytes);
  manifest.files[name] = {
    method,
    path: path.replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, '{id}'),
    status: response.status,
    contentType: response.headers.get('content-type'),
  };
  return { status: response.status, text: bytes.toString('utf8') };
}

async function waitDone(memoryId: string) {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const memory = await client.memories.get(memoryId);
    if (memory.processingStatus === 'COMPLETED' || memory.processingStatus === 'FAILED') {
      return memory.processingStatus;
    }
    await sleep(1000);
  }
  throw new Error(`memory ${memoryId} did not finish indexing within 60s`);
}

const retrieveBody = (spaceIds: string[], extra: Record<string, unknown> = {}) => ({
  message: 'Which canary does the fixture carry?',
  spaceKeys: spaceIds.map((spaceId) => ({ spaceId })),
  requestedSize: 5,
  fetchMemory: true,
  ...extra,
});

async function main(): Promise<void> {
try {
  const info = await client.system.info();
  manifest.serverVersion = info.version;

  // --- a healthy space with two text memories -------------------------------
  const created = await capture('space_create.json', 'POST', '/v1/spaces', {
    name: `aisdk-${run}`,
    spaceEmbedders: [{ embedderId: EMBEDDER, defaultRetrievalWeight: 1.0 }],
    defaultChunkingConfig: { recursive: { chunkSize: 512, chunkOverlap: 64 } },
  });
  const space = JSON.parse(created.text);
  spaces.push(space.spaceId);

  const mem1 = await capture('memory_create.json', 'POST', '/v1/memories', {
    spaceId: space.spaceId,
    contentType: 'text/plain',
    originalContent: "The fixture canary is ORYX-4471. O'Brien filed it.",
    metadata: { tenant: 'acme', year: 2026, active: true },
  });
  const first = JSON.parse(mem1.text);
  const second = await client.memories.create({
    spaceId: space.spaceId,
    originalContent: 'An unrelated note about the weather in Amman.',
    metadata: { tenant: 'globex', year: 2025, active: false },
  });
  await capture('memory_get_pending.json', 'GET', `/v1/memories/${first.memoryId}`);
  await waitDone(first.memoryId);
  await waitDone(second.memoryId);
  await capture('memory_get_completed.json', 'GET', `/v1/memories/${first.memoryId}`);

  await capture('retrieve_ok.ndjson', 'POST', '/v1/memories:retrieve', retrieveBody([space.spaceId]), 'application/x-ndjson');
  await capture(
    'retrieve_rerank_failed.ndjson',
    'POST',
    '/v1/memories:retrieve',
    retrieveBody([space.spaceId], {
      postProcessor: {
        name: 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory',
        config: { reranker_id: '00000000-0000-7000-8000-000000000000' },
      },
    }),
    'application/x-ndjson'
  );
  if (RERANKER) {
    await capture(
      'retrieve_reranked.ndjson',
      'POST',
      '/v1/memories:retrieve',
      retrieveBody([space.spaceId], {
        postProcessor: {
          name: 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory',
          config: { reranker_id: RERANKER },
        },
      }),
      'application/x-ndjson'
    );
  }

  // --- original content: text and a tiny PDF --------------------------------
  const pdf = makePdf('The PDF fixture canary is LYNX-2208.');
  const pdfMemory = await client.memories.createFromBytes({
    spaceId: space.spaceId,
    bytes: pdf,
    contentType: 'application/pdf',
    filename: 'canary.pdf',
    metadata: { tenant: 'acme', kind: 'pdf' },
  });
  await waitDone(pdfMemory.memoryId);
  await capture(
    'retrieve_with_content.ndjson',
    'POST',
    '/v1/memories:retrieve',
    retrieveBody([space.spaceId], { fetchMemoryContent: true, requestedSize: 10 }),
    'application/x-ndjson'
  );
  writeFileSync(join(OUT, 'canary.pdf'), pdf);

  // --- an empty space, and pagination over two spaces -----------------------
  const empty = await client.spaces.create({
    name: `aisdk-${run}-empty`,
    spaceEmbedders: [{ embedderId: EMBEDDER as string, defaultRetrievalWeight: 1.0 }],
  });
  spaces.push(empty.spaceId);
  await capture('retrieve_empty.ndjson', 'POST', '/v1/memories:retrieve', retrieveBody([empty.spaceId]), 'application/x-ndjson');

  const page1 = await capture('spaces_page1.json', 'GET', `/v1/spaces?nameFilter=aisdk-${run}*&maxResults=1`);
  const token = JSON.parse(page1.text).nextToken;
  if (!token) throw new Error('expected a second page of spaces');
  await capture('spaces_page2.json', 'GET', `/v1/spaces?nameFilter=aisdk-${run}*&maxResults=1&nextToken=${encodeURIComponent(token)}`);

  // --- a space whose embedder cannot be reached -----------------------------
  const bad = await client.embedders.create({
    displayName: `aisdk-${run}-unreachable`,
    providerType: 'OPENAI',
    endpointUrl: 'http://127.0.0.1:9/v1',
    modelIdentifier: `unreachable-${run}`,
    dimensionality: 8,
    distributionType: 'DENSE',
  });
  embedders.push(bad.embedderId);
  const broken = await client.spaces.create({
    name: `aisdk-${run}-broken`,
    spaceEmbedders: [{ embedderId: bad.embedderId, defaultRetrievalWeight: 1.0 }],
  });
  spaces.push(broken.spaceId);
  await capture('retrieve_embedder_failed.ndjson', 'POST', '/v1/memories:retrieve', retrieveBody([broken.spaceId]), 'application/x-ndjson');

  // --- errors the server explains in its body -------------------------------
  await capture('error_space_create_400.json', 'POST', '/v1/spaces', {
    name: `aisdk-${run}-invalid`,
    spaceEmbedders: [{ embedderId: 'not-a-uuid' }],
    defaultChunkingConfig: { recursive: { chunkSize: 512, chunkOverlap: 64 } },
  });
  await capture('error_space_create_409.json', 'POST', '/v1/spaces', {
    name: `aisdk-${run}`,
    spaceEmbedders: [{ embedderId: EMBEDDER }],
    defaultChunkingConfig: { recursive: { chunkSize: 512, chunkOverlap: 64 } },
  });
  await capture(
    'error_retrieve_missing_space.json',
    'POST',
    '/v1/memories:retrieve',
    retrieveBody(['00000000-0000-7000-8000-000000000000']),
    'application/x-ndjson'
  );
  await capture('error_space_publicread_400.json', 'PUT', `/v1/spaces/${space.spaceId}`, { publicRead: true });
} finally {
  const failures = [];
  for (const id of spaces.reverse()) {
    try {
      await client.spaces.delete(id);
    } catch (error) {
      failures.push(`space ${id}: ${(error as Error).message}`);
    }
  }
  for (const id of embedders) {
    try {
      await client.embedders.delete(id);
    } catch (error) {
      failures.push(`embedder ${id}: ${(error as Error).message}`);
    }
  }
  writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  if (failures.length) {
    console.error(`Teardown failed:\n${failures.join('\n')}`);
    process.exitCode = 1;
  }
}
console.log(JSON.stringify(manifest, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
