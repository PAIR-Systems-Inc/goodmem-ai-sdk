# @pairsystems/goodmem-ai-sdk

[GoodMem](https://docs.goodmem.ai) memory for the [Vercel AI SDK](https://ai-sdk.dev):
tools a model can call, a middleware that gives every call relevant memories,
and explicit helpers for your own code. GoodMem chunks, embeds and searches
server-side; this package talks to it only through the official
`@pairsystems/goodmem` SDK.

**Version 0.1.0.** Verified against GoodMem server **v1.0.323**, AI SDK **6**
and **7**, and Node **20** and later. See [Compatibility](#compatibility).

## Install

```bash
npm install @pairsystems/goodmem-ai-sdk ai zod
```

With another package manager:

```bash
pnpm add @pairsystems/goodmem-ai-sdk ai zod
yarn add @pairsystems/goodmem-ai-sdk ai zod
bun add @pairsystems/goodmem-ai-sdk ai zod
```

`ai` and `zod` are peer dependencies, so the package uses your copies. All
four package managers install it without peer warnings, and it also runs on
the Bun runtime.

## Get an instance URL and API key

You need a GoodMem server URL and an API key for it.

- **GoodMem Cloud.** Sign up or sign in at
  [cloud.goodmem.ai/login](https://cloud.goodmem.ai/login) with Google or
  GitHub. The free 14-day trial needs no credit card, and a trial instance is
  provisioned automatically. Instance URLs look like
  `https://gm-<name>-<id>.app.goodmem.ai`; take the URL and an API key from
  the GoodMem Cloud app. To get a first space, open Quick Start in the
  instance's console (`https://gm-<name>-<id>.app.goodmem.ai/console/quick-start`)
  and paste an API key from a model provider -- OpenAI, OpenRouter, Voyage,
  Cohere, Jina, Gemini or DashScope. One click creates an embedder and a first
  space (plus an LLM and a reranker where that provider offers them); use that
  space's id as `spaceId`.
- **Self-hosted.** Install a server with
  `curl -s https://get.goodmem.ai | bash` (see
  [docs.goodmem.ai](https://docs.goodmem.ai)), then use its URL and an API key
  it issued. The self-hosted server this package is tested against serves
  `http://localhost:8080`.

Then either put them in the environment:

```bash
export GOODMEM_BASE_URL=https://gm-<name>-<id>.app.goodmem.ai
export GOODMEM_API_KEY=<your-api-key>
```

or pass them as `baseUrl` and `apiKey`.

## Configure

Every entry point takes the same configuration object. With the two variables
above set, a space is all it needs:

```ts
import type { GoodmemConfig } from '@pairsystems/goodmem-ai-sdk';

const goodmem: GoodmemConfig = { spaceId: '<space-uuid>' };

// The same, with the server and key passed explicitly:
const explicit: GoodmemConfig = {
  apiKey: '<your-api-key>',
  baseUrl: 'https://gm-<name>-<id>.app.goodmem.ai',
  spaceId: '<space-uuid>',
};
```

| Option | Default | Meaning |
| --- | --- | --- |
| `apiKey` | `GOODMEM_API_KEY` | Your GoodMem API key. |
| `baseUrl` | `GOODMEM_BASE_URL` | Your GoodMem server. There is no default server. |
| `spaceId` / `spaceIds` / `space` | one required | Where memories are read and written; see [Spaces and scoping](#spaces-and-scoping). The model never chooses a space. |
| `topK` | `5` | Results per search (1-100). |
| `timeoutMs` | `30000` | Upper bound on every request, including reading a result stream to its end. It cannot be turned off. |
| `scope` | none | Metadata written onto every memory and required on every search, e.g. `{ userId: 'u-123' }`. |
| `filter` | none | An extra metadata filter for every search, built with `filters`. |
| `rerankerId` | none | A GoodMem reranker to apply to results. |
| `minScore` | none | Drop results scoring below this. Requires `rerankerId`; see [Scores](#scores). |
| `fetch` | global `fetch` | A `fetch` for the GoodMem SDK to use (a proxy, a custom CA, tests). |
| `logger` | `console` | Where warnings about degraded retrievals are written. |

`apiKey` falls back to `GOODMEM_API_KEY`, and `baseUrl` to `GOODMEM_BASE_URL`,
each only when that option is omitted. An explicit option always wins, an
explicit empty string is an error rather than a fallback, and no other
environment variable is read. Where there is no Node-style environment (a
browser, some edge runtimes), pass both options.

Configuration mistakes throw `GoodMemConfigError` straight away, naming the
option and how to fix it. A missing key, for example, says to pass `apiKey`
or set `GOODMEM_API_KEY`, and where to get one. An unknown option (a typo such
as "topk" for `topK`) is an error rather than silently ignored.

## Tools

The examples from here on assume `GOODMEM_API_KEY` and `GOODMEM_BASE_URL` are
set; otherwise add `apiKey` and `baseUrl` to each configuration.

```ts
import { openai } from '@ai-sdk/openai';
import { generateText, stepCountIs } from 'ai';
import { goodmemTools } from '@pairsystems/goodmem-ai-sdk';

const { text } = await generateText({
  model: openai('gpt-4o'),
  tools: goodmemTools({ spaceId: '<space-uuid>' }),
  stopWhen: stepCountIs(5),
  prompt: 'Remember that I prefer window seats. What do you know about how I like to travel?',
});
```

`goodmemTools(config)` returns two tools. These are the names and inputs the
model sees:

| Tool | Model input | Returns to the model |
| --- | --- | --- |
| `searchMemories` | `query`, optional `topK` (1-50) | `results` (text, ids, `score`, `rawScore`, `scoreKind`, metadata), `totalResults`, `partial`, `statuses`, and `warning` when `partial` |
| `addMemory` | `text` | `memoryId`, `spaceId`, `processingStatus` |

There is no delete, update or space-management tool, and no tool takes a file
path. Spaces, scope, filters, reranking and timeouts come from your
configuration, never from the model. For read-only memory, pass only the
search tool:

```ts
import { goodmemTools } from '@pairsystems/goodmem-ai-sdk';

const { searchMemories } = goodmemTools({ spaceId: '<space-uuid>' });
const tools = { searchMemories };
```

A memory saved with `addMemory` becomes searchable once GoodMem has indexed it,
usually within a few seconds; the tool does not wait. When a request fails the
tool throws `GoodMemError`, which the AI SDK reports to the model as a tool
error carrying the server's message.

## Middleware

`withGoodmem(model, config)` wraps a model with `wrapLanguageModel`. On every
call it searches GoodMem for the latest user message and adds the results to
the prompt:

```ts
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { withGoodmem } from '@pairsystems/goodmem-ai-sdk';

const model = withGoodmem(openai('gpt-4o'), {
  spaceId: '<space-uuid>',
  scope: { userId: 'u-123' },
});

const { text, providerMetadata } = await generateText({
  model,
  prompt: 'Where do I like to sit on flights?',
});

console.log(providerMetadata?.goodmem);
// { partial: false, resultCount: 3, memoryIds: [...], statuses: [] }
```

By default the memories are appended to the leading system message (one is
added if there is none), under the heading "Relevant memories from GoodMem,
most relevant first". The middleware takes the shared configuration plus:

| Option | Default | Meaning |
| --- | --- | --- |
| `position` | `'system'` | `'user'` prepends the memories to the latest user message instead. |
| `template` | the result's `context` | `(retrieved) => string` builds the injected text; return `''` to inject nothing. |
| `skipMemoryOnError` | `true` | When GoodMem is unavailable, call the model anyway (flagged) instead of failing the call. See below. |
| `retrievalTimeoutMs` | `5000` | Upper bound on each call's memory lookup, space lookup included. Only the middleware uses it; everything else keeps `timeoutMs`. |
| `addMemory` | `'never'` | `'always'` saves the durable facts the user states. See [Automatic saving](#automatic-saving). |
| `extractionModel` | the wrapped model | The model that extracts those facts. |
| `extractFacts` | none | Your own extraction function, instead of a model. |

```ts
import { openai } from '@ai-sdk/openai';
import { withGoodmem, type RetrieveMemoriesResult } from '@pairsystems/goodmem-ai-sdk';

const model = withGoodmem(openai('gpt-4o'), {
  spaceId: '<space-uuid>',
  position: 'user',
  template: (retrieved: RetrieveMemoriesResult) =>
    retrieved.results.length
      ? `<memories>\n${retrieved.results.map((r) => `- ${r.text.trim()}`).join('\n')}\n</memories>`
      : retrieved.context, // '' when nothing was found; the failure note when the lookup failed
});
```

Only the request sent to the model is changed. Your messages are never
modified, so injected memories are not saved into your chat history. A call
whose latest user message has no text (or that has no user message at all) is
passed through without a search. The search runs once per model call, so in a
multi-step tool loop it runs again, with the same user message, for each step.

The outcome of each call's search is on the result as
`providerMetadata.goodmem`: `partial`, `resultCount`, `memoryIds` and
`statuses`, plus `saved` or `saveError` when [automatic saving](#automatic-saving)
ran. It works the same way with `streamText`.

### When GoodMem is unavailable

A memory service being down should not take the chat down with it. With
`skipMemoryOnError` on (the default), a lookup that fails because GoodMem is
unavailable -- the server cannot be reached, the lookup takes longer than
`retrievalTimeoutMs`, the server answers HTTP 5xx, 429 or 408, or the response
is broken -- does not fail the call. Instead:

- the model is called, with a note that memory retrieval failed and that
  relevant memories may exist, so it never tells the user they have none;
- the call's `warnings` include a GoodMem warning, and the logger gets a
  warning line;
- `providerMetadata.goodmem` is `{ partial: true, resultCount: 0, memoryIds: [], statuses }`,
  with one status whose `code` is `RETRIEVAL_FAILED` (exported as
  `RETRIEVAL_FAILED_CODE`) and whose `message` says what failed.

Failures that will not fix themselves still throw, whatever the setting:
configuration errors, HTTP 400, 401, 403 and 404 (a rejected key, a missing
space, an invalid filter), and your own abort signal. Hiding those would mean
memory silently never works. Set `skipMemoryOnError: false` to fail the call on
every failure instead.

The server embeds each query with the space's embedder before it can search,
so `retrievalTimeoutMs` covers that too. A fast hosted embedder answers well
within the 5 s default (Voyage took about 0.3 s against the test server), but a
slow one may not: Qwen3 through OpenRouter took anywhere from 1 s to over 30 s
on the same server. With a slow embedder, raise `retrievalTimeoutMs`, or expect
some calls to go ahead without memories, flagged as above.

### Automatic saving

Off by default. With `addMemory: 'always'` the middleware also saves what the
user tells it about themselves:

```ts
import { openai } from '@ai-sdk/openai';
import { withGoodmem } from '@pairsystems/goodmem-ai-sdk';

const model = withGoodmem(openai('gpt-4o'), {
  spaceId: '<space-uuid>',
  scope: { userId: 'u-123' },
  addMemory: 'always',
  extractionModel: openai('gpt-4o-mini'), // optional: a cheaper model for extraction
});
```

On each user turn it makes one extra model call. That call gets
`FACT_EXTRACTION_INSTRUCTIONS` and the latest user message, and returns the
durable facts the user stated about themselves, their preferences or their
world, as short third-person sentences ("User is vegetarian."). Each fact is
stored with the configuration's `scope`, so the next turn finds it. Questions
and small talk yield nothing. The extraction uses `extractionModel`, or the
wrapped model itself when that is omitted, called directly rather than through
the middleware.

What is never saved: the model's replies, its reasoning, streamed output,
earlier turns, and the memories injected into the prompt. A fact that matches
a memory retrieved for the same turn, after normalising case, spacing and final
punctuation, is skipped.

The extraction runs while the model answers. The facts are written to GoodMem
only once the model call has succeeded, so a failed or aborted turn stores
nothing. In a multi-step tool loop, it runs once per user turn, not once per
step. The save is bounded by `timeoutMs` and follows your abort signal.

The outcome is on `providerMetadata.goodmem`:

- `saved: { facts, memoryIds, duplicates, usage }`, where `usage` is the
  extraction call's token count, so the extra cost is visible;
- or `saveError: { stage, message, facts?, memoryIds? }`. A failed extraction or
  save never fails the call. It is reported there, in the logger, and, for
  `generateText`, in the call's `warnings`. With `streamText` the outcome is
  only known at the end, so it comes in the finish part's metadata and the log.

`extractFacts: ({ text, signal }) => Promise<string[]>` replaces the model call
with your own extraction. It cannot be combined with `extractionModel`.

This differs from memory services that extract facts on their own servers with
their own model. Here the extraction is an ordinary AI SDK call on the model
you choose. It shows up in your usage and your telemetry like any other call,
and its instructions are exported and can be replaced.

## Helpers

For your own code, three functions take the same configuration:

```ts
import { addMemories, retrieveMemories, searchMemories, type GoodmemConfig } from '@pairsystems/goodmem-ai-sdk';
import { readFile } from 'node:fs/promises';

const goodmem: GoodmemConfig = { spaceId: '<space-uuid>' };

// Store text, and a PDF as bytes, then wait until both are searchable.
await addMemories(
  [
    'The quarterly review moved to Thursday.',
    { data: await readFile('report.pdf'), contentType: 'application/pdf', filename: 'report.pdf' },
  ],
  goodmem,
  { waitForIndexing: true },
);

// Structured results: ids, oriented scores, metadata, status.
const { results, partial, statuses } = await searchMemories('When is the review?', goodmem);

// The same, plus a context block ready for a system prompt.
const { context } = await retrieveMemories('When is the review?', goodmem, { topK: 3 });
```

- `searchMemories(query, config, options?)` returns `{ results, partial, statuses, warning?, resultSetId }`.
- `retrieveMemories(query, config, options?)` returns the same plus `context`.
  Both accept `topK`, `signal`, and `includeContent: true`, which adds each
  memory's original `content` to its result: text as text, anything else (a
  PDF, say) base64, with `contentEncoding` saying which.
- `addMemories(input, config, options?)` stores a string, `{ text }`, or
  `{ data, contentType }` bytes, or an array of them, and returns their
  `memoryId`s. `data` is bytes you pass in; nothing in this package reads a
  file path. Every input is validated before anything is written.

Nothing waits or polls unless you ask. `waitForIndexing: true` (or
`{ timeoutMs, pollIntervalMs }`, default 60 s) makes `addMemories` poll only the
memories it just created until they finish indexing. If one fails processing
or the wait runs out, it throws `GoodMemIndexingError`, which lists
`failedMemoryIds` and `pendingMemoryIds` and says the writes themselves
succeeded. Searches never poll or wait: each makes one request and returns
what the server sends, so an empty space costs a single round trip.

## Spaces and scoping

Choose exactly one of:

- `spaceId: '<uuid>'` -- one space for reads and writes.
- `spaceIds: ['<uuid>', ...]` -- search several spaces; writes go to the first.
- `space: { name, embedderId }` -- use the space with exactly this name, or
  create it with this embedder if there is none. Every page of the listing is
  read, so a match on a later page is found, and a listing that never ends is
  refused rather than cut short. An existing space is reused only if it uses
  `embedderId` (a space's embedder cannot be changed after creation), and two
  visible spaces with the name are refused as ambiguous. The embedder is never
  chosen for you. The lookup runs the first time each configuration object is
  used and is then remembered for that object.

For per-user or per-session memory, either give each user their own space
(GoodMem enforces access per space) or share a space and set `scope`:

```ts
import { goodmemTools } from '@pairsystems/goodmem-ai-sdk';

function toolsFor(userId: string) {
  return goodmemTools({
    space: { name: 'assistant-memory', embedderId: '<embedder-uuid>' },
    scope: { userId },
  });
}
```

`scope` is written into the metadata of every memory stored through that
configuration and applied as an equality filter on every search. It is a
filter within a space, not an access control: use separate spaces when users
must not be able to reach each other's memories. A per-memory `metadata` value
that contradicts the scope is refused rather than overwritten.

## When retrieval goes wrong

`partial` means exactly one thing: **the server reported a real problem during
this retrieval**, whether or not results came back. The rules follow the
GoodMem retrieval status contract:

| Situation | What you get |
| --- | --- |
| `FEATURE_DISABLED`, `LLM_CAPABILITY_INFERRED` | Ignored: they report an optional feature you did not configure. |
| A problem, and results arrived | The results, `partial: true`, the server's `statuses`, a `warning`, and a warning logged. |
| A problem, and no results | Empty results, `partial: true`, `statuses`, `warning`, a warning logged -- never a silent empty success. |
| A status code this version does not know | Reported as `UNKNOWN`, `partial: true`; results kept. |
| The result stream broke off part-way | What arrived, plus a `MALFORMED_STREAM` status. |
| No usable response: unreachable server, timeout, HTTP error | Helpers throw `GoodMemError`; tools throw it, and the AI SDK hands the model a tool error. The middleware: see below. |

The middleware follows the same rules. A degraded search still injects what
arrived, the injected text says memories may be missing (or that retrieval
failed, when nothing came back), the warning is added to the call's
`warnings`, and `providerMetadata.goodmem.partial` is `true`. When the lookup
gets no usable response at all:

| Failure | Middleware, `skipMemoryOnError: true` (default) | `skipMemoryOnError: false` |
| --- | --- | --- |
| Unreachable server, lookup over `retrievalTimeoutMs`, HTTP 5xx / 429 / 408, broken response | The call goes ahead with the "retrieval failed" note, a warning, a logged warning, and a `RETRIEVAL_FAILED` status in `providerMetadata.goodmem` | Throws `GoodMemError` |
| HTTP 400 / 401 / 403 / 404, a configuration error | Throws | Throws |
| Your own abort signal | Rethrows the abort | Rethrows the abort |

Either way the model is never quietly called as if there were no memories.

## Errors

| Error | Raised when | Carries |
| --- | --- | --- |
| `GoodMemError` | A request failed. | `statusCode` and `body` (the server's response, verbatim) when the server answered; `timedOut` when the time limit was reached; `isRetryable`, true for availability failures (unreachable, timed out, HTTP 5xx / 429 / 408, broken response) that may go away on their own. The message quotes the server's explanation. |
| `GoodMemConfigError` | The configuration or an argument cannot be used. | The option at fault and what to pass instead. |
| `GoodMemIngestionError` | An `addMemories` write failed part-way. | `createdMemoryIds` (already stored) and `failedIndex` (where to resume). |
| `GoodMemIndexingError` | A `waitForIndexing` wait did not end in success. | `memoryIds`, `failedMemoryIds`, `pendingMemoryIds`. |
| `GoodMemFilterError` | A filter value or field cannot be expressed safely. | |

All of them extend `GoodMemError`. Each class has a static `isInstance()`,
like the AI SDK's own errors, which also works when two copies of the package
are installed. A caller-requested abort is rethrown as the abort, unchanged.

## Scores

Every result carries three score fields. GoodMem produces two kinds of score,
and they are not comparable:

- `scoreKind: 'vector'` -- the raw score is a negated inner product: negative,
  and more negative is a closer match. `score` is its negation, so higher is
  better; `rawScore` keeps the server's value.
- `scoreKind: 'reranker'` -- set when the reranking stage produced the results.
  Already higher-is-better, on a scale that depends on the reranker (on the
  same five documents and GoodMem server, Voyage `rerank-2.5` scored 0.27 to
  0.93 and Jina `jina-reranker-v3` -0.14 to 0.43). `score` equals `rawScore`.

The kind is read from the result stream, so if a configured reranker fails,
the vector scores that come back are still labelled `vector`. Neither kind is
a 0-1 scale, so there is no default threshold: `minScore` applies only to
reranker scores and requires `rerankerId`, and if it removes every result a
warning names the scores that were observed.

## Metadata filters

GoodMem filters are expressions evaluated server-side, not SQL. Build them
with `filters`, so values are escaped the way the server accepts:

```ts
import { filters, searchMemories } from '@pairsystems/goodmem-ai-sdk';

const recentFromAcme = filters.allOf(
  filters.equals('tenant', "O'Brien & Co"),
  filters.compare('year', '>=', 2026),
  filters.not(filters.equals('archived', true)),
);

await searchMemories('pricing decisions', {
  spaceId: '<space-uuid>',
  filter: recentFromAcme,
});
```

Available: `equals`, `notEquals`, `compare`, `oneOf`, `allOf`, `anyOf`, `not`,
`fromMapping` and `escapeLiteral`. The rules they enforce were checked against
a live server:

- `'` is escaped as `\'` and `\` as `\\`; SQL-style `''` doubling is rejected by the server with HTTP 400.
- Control characters, including newlines, are refused.
- Each value is cast to the type GoodMem stored: `TEXT`, `NUMERIC` or `BOOLEAN`.
  A boolean compared as text is accepted by the server and matches nothing,
  so booleans are never stringified.
- Field names are letters, digits and underscores, with dots for nested
  fields. A hyphenated name such as `user-id` is accepted by the server and
  matches nothing, so it is refused here.

## Compatibility

| | Supported | Verified |
| --- | --- | --- |
| `ai` | `^6.0.0 \|\| ^7.0.0` | 6.0.0, 6.0.297, 7.0.0 and 7.0.123 |
| `zod` | `^3.25.76 \|\| ^4.1.8` | 3.25.76, 4.1.8 and 4.6.5 |
| Node | `>=20` | 20 and 24 locally; 20 and 22 in CI |
| GoodMem server | | v1.0.323 (live suite) |
| `@pairsystems/goodmem` | `^0.1.9` | 0.1.9 |

Each verified combination ran the type check (package and tests), the whole
offline suite -- which drives the tools and middleware through that `ai`
version's own `generateText`, `streamText` and mock model -- the README check
against the built types, and a clean install loaded with both `require` and
`import`. The floors are tested, not assumed: `ai` 6.0.0 with `zod` 3.25.76,
and `ai` 7.0.0.

`ai` 7 itself requires Node 22 and ships only ES modules. This package ships
both CommonJS and ES module builds; `require()` of it with `ai` 7 relies on
Node's support for requiring ES modules (Node 20.19+ or 22.12+). With `ai` 6,
both builds work on any Node 20.

## Security

- The API key, whether passed as `apiKey` or read from `GOODMEM_API_KEY`, is
  sent only as the `x-api-key` header. It is not stored on anything this
  package returns: tools, wrapped models, results and errors can be logged or
  serialised without leaking it.
- The only environment variables read are `GOODMEM_API_KEY` and
  `GOODMEM_BASE_URL`, each only when its option is omitted, so an explicit
  configuration is never redirected by the environment.
- TLS certificate verification is never disabled by this package. For a
  private CA, pass a `fetch` configured with it.
- No file-system path is read, so no model output can make the package open a
  file on your host.
- Metadata filters are built with escaping verified against the server; a
  value such as `x' OR '1'='1` stays a literal and matches only itself.

## Development

These are the commands CI runs:

```bash
npm ci
npx tsc --noEmit        # types
npm test                # offline suite: no network; sets and clears the GOODMEM_ variables it tests
npm run build           # CommonJS + ES module builds and types in lib/
npm run check:readme    # README examples type-check; named identifiers exist
npm run test:live       # live suite; skips without credentials
```

CI also runs the gates in `.github/workflows/ci.yml`: no credential-shaped
string in any tracked file, no TLS-verification bypass anywhere, no direct
HTTP calls in `src/`, environment reads only in `src/env.ts` and only of
`GOODMEM_API_KEY` and `GOODMEM_BASE_URL`, and the built package installed into
a clean directory and loaded with both `require` and `import`.

The live suite needs a running GoodMem server:

```bash
GOODMEM_BASE_URL=http://localhost:8080 GOODMEM_API_KEY=... \
GOODMEM_TEST_EMBEDDER_ID=<embedder-uuid> npm run test:live
```

It creates its spaces through the package under a unique run id (and one
temporary embedder with an unreachable endpoint, to produce a real
`EMBEDDER_FAILED`), deletes all of them afterwards, and then asserts against a
fresh listing that none is left. Set `GOODMEM_TEST_RERANKER_ID` to include the
reranker checks. The empty-space check requires a search to finish in under a
second with one request; since the server embeds every query, a slow hosted
embedder can dominate that time, and `GOODMEM_TEST_LATENCY_EMBEDDER_ID` points
that one check at a faster embedder. `GOODMEM_TEARDOWN_REPORT=<file>` writes
the post-teardown server listing to a file.

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/goodmem_test.ts` | 125 | Nothing. The real GoodMem SDK over a fake `fetch` replaying responses captured from server v1.0.323, and the real `ai` package driving the tools and middleware with its mock language model. |
| `tests/goodmem_live_test.ts` | 25 | `GOODMEM_API_KEY`, `GOODMEM_BASE_URL`, `GOODMEM_TEST_EMBEDDER_ID`; skips without them. |

## License

Apache-2.0.
