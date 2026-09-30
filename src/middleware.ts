/**
 * Automatic memory: a language-model middleware that retrieves relevant
 * memories for every call and adds them to the prompt.
 *
 * Only the outgoing request is augmented. The caller's messages are never
 * modified, so the injected memories are not saved into chat history as if
 * the user or the model had said them.
 */

import { wrapLanguageModel, type LanguageModelMiddleware } from 'ai';

import { connect, MAX_TIMEOUT_MS, type GoodmemConfig } from './config.js';
import { GoodMemConfigError, GoodMemError } from './errors.js';
import { RETRIEVAL_FAILED_CODE } from './results.js';
import { formatContext, retrieve, type RetrieveMemoriesResult } from './retrieval.js';

type TransformOptions = Parameters<NonNullable<LanguageModelMiddleware['transformParams']>>[0];
type CallParams = TransformOptions['params'];
type Prompt = CallParams['prompt'];
type Message = Prompt[number];
type GenerateResult = Awaited<ReturnType<NonNullable<LanguageModelMiddleware['wrapGenerate']>>>;
type StreamResult = Awaited<ReturnType<NonNullable<LanguageModelMiddleware['wrapStream']>>>;
type StreamPart = StreamResult['stream'] extends ReadableStream<infer P> ? P : never;
type Warning = NonNullable<GenerateResult['warnings']>[number];

/** Default for `retrievalTimeoutMs`: how long a call waits for memories. */
export const DEFAULT_RETRIEVAL_TIMEOUT_MS = 5_000;

/** The model `withGoodmem` wraps: any model `wrapLanguageModel` accepts. */
export type GoodmemWrappableModel = Parameters<typeof wrapLanguageModel>[0]['model'];

/** Where retrieved memories are placed in the prompt. */
export type GoodmemInjectionPosition = 'system' | 'user';

/** Configuration for `withGoodmem`. */
export interface GoodmemMiddlewareConfig extends GoodmemConfig {
  /**
   * Where the memories go. `system` (default) appends them to the leading
   * system message, or adds one; `user` prepends them to the latest user
   * message.
   */
  position?: GoodmemInjectionPosition;
  /**
   * Builds the text that is injected. Receives the retrieval result; return
   * '' to inject nothing. Defaults to the result's `context`, which says so
   * when a retrieval was degraded.
   */
  template?: (retrieved: RetrieveMemoriesResult) => string;
  /**
   * When the memory lookup fails because GoodMem is unavailable -- the server
   * cannot be reached, the lookup exceeds `retrievalTimeoutMs`, the server
   * answers HTTP 5xx, 429 or 408, or the response is broken -- call the model
   * anyway, telling it that memories may be missing, and flag the call
   * (`warnings`, `providerMetadata.goodmem`, a logged warning). Defaults to
   * true. Failures that will not fix themselves still throw: a configuration
   * error, HTTP 400/401/403/404 (a rejected key, a missing space, an invalid
   * filter), and the caller's own abort. Set false to throw on every failure.
   */
  skipMemoryOnError?: boolean;
  /**
   * Upper bound, in milliseconds, on the memory lookup for one call, space
   * lookup included. Defaults to 5000. Only the middleware's lookup uses it;
   * every other request keeps `timeoutMs`. A lookup that runs out is an
   * availability failure, handled by `skipMemoryOnError`. The server embeds
   * each query with the space's embedder, so a slow hosted embedder can take
   * longer than this on its own.
   */
  retrievalTimeoutMs?: number;
}

function lastUserText(prompt: Prompt): string {
  for (let i = prompt.length - 1; i >= 0; i -= 1) {
    const message = prompt[i];
    if (message.role !== 'user') continue;
    return (message.content as Array<{ type: string; text?: string }>)
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => part.text as string)
      .join('\n')
      .trim();
  }
  return '';
}

function inject(prompt: Prompt, text: string, position: GoodmemInjectionPosition): Prompt {
  const next = [...prompt] as Message[];
  if (position === 'user') {
    for (let i = next.length - 1; i >= 0; i -= 1) {
      const message = next[i];
      if (message.role !== 'user') continue;
      next[i] = { ...message, content: [{ type: 'text', text }, ...message.content] } as Message;
      return next as Prompt;
    }
    return prompt;
  }
  let lastSystem = -1;
  while (next[lastSystem + 1]?.role === 'system') lastSystem += 1;
  if (lastSystem >= 0) {
    const message = next[lastSystem] as Extract<Message, { role: 'system' }>;
    next[lastSystem] = { ...message, content: `${message.content}\n\n${text}` } as Message;
  } else {
    next.unshift({ role: 'system', content: text } as Message);
  }
  return next as Prompt;
}

function metadataFor(retrieved: RetrieveMemoriesResult) {
  return {
    partial: retrieved.partial,
    resultCount: retrieved.results.length,
    memoryIds: [...new Set(retrieved.results.map((r) => r.memoryId))],
    statuses: retrieved.statuses.map((s) => ({
      code: s.code,
      message: s.message,
      ...(s.details ? { details: JSON.parse(JSON.stringify(s.details)) } : {}),
    })),
  };
}

function warningsFor(retrieved: RetrieveMemoriesResult): Warning[] {
  return retrieved.partial && retrieved.warning
    ? [{ type: 'other', message: `[goodmem] ${retrieved.warning}` } as Warning]
    : [];
}

/**
 * The retrieval result reported when a lookup failed outright and the call
 * went ahead without memories: no results, partial, and one status saying why.
 */
function lookupFailed(error: GoodMemError): RetrieveMemoriesResult {
  const details: Record<string, unknown> = {};
  if (error.statusCode !== undefined) details.statusCode = error.statusCode;
  if (error.timedOut) details.timedOut = true;
  const statuses = [
    { code: RETRIEVAL_FAILED_CODE, message: error.message, ...(Object.keys(details).length ? { details } : {}) },
  ];
  const result = {
    results: [],
    statuses,
    partial: true,
    warning: `GoodMem memory lookup failed, so this call went ahead without memories -- ${error.message}`,
    resultSetId: '',
  };
  return { ...result, context: formatContext(result) };
}

/**
 * Wrap a language model so every call is given relevant memories.
 *
 * For each call, the text of the latest user message is searched in the
 * configured space(s) and the results are added to the prompt. A call with
 * no user text is passed through untouched.
 *
 * Retrieval follows the GoodMem status contract, and a failed lookup is
 * never presented to the model as "no memories":
 * - a degraded retrieval (the server reported a problem) still injects what
 *   arrived, says in the injected text that memories may be missing, adds a
 *   warning to the call's `warnings`, and logs it;
 * - a lookup that fails because GoodMem is unavailable (unreachable, slower
 *   than `retrievalTimeoutMs`, HTTP 5xx/429/408, a broken response) is
 *   handled the same way with no results and a `RETRIEVAL_FAILED` status, and
 *   the call goes ahead -- unless `skipMemoryOnError` is false, in which case
 *   it throws `GoodMemError`;
 * - a failure that will not fix itself (a configuration error, HTTP
 *   400/401/403/404) and the caller's own abort always throw.
 *
 * The outcome is reported on each result as `providerMetadata.goodmem`:
 * `{ partial, resultCount, memoryIds, statuses }`.
 */
export function withGoodmem(
  model: GoodmemWrappableModel,
  config: GoodmemMiddlewareConfig
): ReturnType<typeof wrapLanguageModel> {
  const conn = connect(config, 'withGoodmem', ['position', 'template', 'skipMemoryOnError', 'retrievalTimeoutMs']);
  if (!model || typeof model !== 'object' || typeof (model as { doGenerate?: unknown }).doGenerate !== 'function') {
    throw new GoodMemConfigError(
      'withGoodmem: pass a language model instance, e.g. openai("gpt-4o"), not a model id string.'
    );
  }
  const position = config.position ?? 'system';
  if (position !== 'system' && position !== 'user') {
    throw new GoodMemConfigError(`withGoodmem: position must be 'system' or 'user' (got ${JSON.stringify(config.position)}).`);
  }
  const template = config.template;
  if (template !== undefined && typeof template !== 'function') {
    throw new GoodMemConfigError('withGoodmem: template must be a function returning the text to inject.');
  }
  const skipMemoryOnError = config.skipMemoryOnError ?? true;
  if (typeof skipMemoryOnError !== 'boolean') {
    throw new GoodMemConfigError(`withGoodmem: skipMemoryOnError must be true or false (got ${JSON.stringify(config.skipMemoryOnError)}).`);
  }
  const retrievalTimeoutMs = config.retrievalTimeoutMs ?? DEFAULT_RETRIEVAL_TIMEOUT_MS;
  if (
    typeof retrievalTimeoutMs !== 'number' ||
    !Number.isFinite(retrievalTimeoutMs) ||
    retrievalTimeoutMs <= 0 ||
    retrievalTimeoutMs > MAX_TIMEOUT_MS
  ) {
    throw new GoodMemConfigError(
      `withGoodmem: retrievalTimeoutMs must be a positive number of milliseconds (got ${String(config.retrievalTimeoutMs)}).`
    );
  }

  /** The lookup for one call; availability failures become a flagged, empty result when allowed. */
  const lookUp = async (query: string, signal: AbortSignal | undefined): Promise<RetrieveMemoriesResult> => {
    try {
      return await retrieve(conn, query, { signal, timeoutMs: retrievalTimeoutMs, timeoutOption: 'retrievalTimeoutMs' });
    } catch (error) {
      const skippable =
        skipMemoryOnError && !signal?.aborted && GoodMemError.isInstance(error) && error.isRetryable;
      if (!skippable) throw error;
      const failed = lookupFailed(error as GoodMemError);
      conn.logger.warn(
        `[goodmem] memory lookup failed; calling the model without memories because skipMemoryOnError is on -- ${(error as Error).message}`
      );
      return failed;
    }
  };

  const outcomes = new WeakMap<object, RetrieveMemoriesResult>();

  const middleware: LanguageModelMiddleware = {
    specificationVersion: 'v3',

    transformParams: async ({ params }) => {
      const query = lastUserText(params.prompt);
      if (!query) return params;
      const retrieved = await lookUp(query, params.abortSignal);
      const text = template ? template(retrieved) : retrieved.context;
      if (typeof text !== 'string') {
        throw new GoodMemConfigError('withGoodmem: template must return a string.');
      }
      const next = { ...params, prompt: text ? inject(params.prompt, text, position) : params.prompt };
      outcomes.set(next, retrieved);
      return next;
    },

    wrapGenerate: async ({ doGenerate, params }) => {
      const result = await doGenerate();
      const retrieved = outcomes.get(params);
      if (!retrieved) return result;
      return {
        ...result,
        warnings: [...(result.warnings ?? []), ...warningsFor(retrieved)],
        providerMetadata: { ...(result.providerMetadata ?? {}), goodmem: metadataFor(retrieved) },
      } as GenerateResult;
    },

    wrapStream: async ({ doStream, params }) => {
      const result = await doStream();
      const retrieved = outcomes.get(params);
      if (!retrieved) return result;
      const warnings = warningsFor(retrieved);
      const goodmem = metadataFor(retrieved);
      const annotate = new TransformStream<StreamPart, StreamPart>({
        transform(part, controller) {
          if (part.type === 'stream-start' && warnings.length) {
            controller.enqueue({ ...part, warnings: [...part.warnings, ...warnings] } as StreamPart);
          } else if (part.type === 'finish') {
            controller.enqueue({
              ...part,
              providerMetadata: { ...(part.providerMetadata ?? {}), goodmem },
            } as StreamPart);
          } else {
            controller.enqueue(part);
          }
        },
      });
      return { ...result, stream: result.stream.pipeThrough(annotate) } as StreamResult;
    },
  };

  return wrapLanguageModel({ model, middleware });
}
