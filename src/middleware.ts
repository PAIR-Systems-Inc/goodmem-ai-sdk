/**
 * Automatic memory: a language-model middleware that retrieves relevant
 * memories for every call and adds them to the prompt, and -- when asked --
 * saves the durable facts the user states.
 *
 * Only the outgoing request is augmented. The caller's messages are never
 * modified, so the injected memories are not saved into chat history as if
 * the user or the model had said them.
 */

import { wrapLanguageModel, type LanguageModelMiddleware } from 'ai';

import {
  extractWithModel,
  runExtraction,
  storeFacts,
  type GoodmemExtractionModel,
  type GoodmemFactExtractor,
  type TurnSave,
} from './autosave.js';
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
  /**
   * Save what the user says. `'never'` (default) saves nothing. `'always'`
   * extracts the durable facts the user stated in the latest user message --
   * about themselves, their preferences or their world, as short third-person
   * sentences -- and stores them with the configured `scope`. It costs one
   * extra model call per user turn (see `extractionModel`). Only the user's
   * own message is used, never the model's reply. The outcome is reported as
   * `providerMetadata.goodmem.saved`, or `saveError`; a failed save never
   * fails the call.
   */
  addMemory?: 'never' | 'always';
  /**
   * The model that extracts facts when `addMemory` is `'always'`. Defaults to
   * the model passed to `withGoodmem` (called directly, without this
   * middleware). A smaller, cheaper model is usually enough.
   */
  extractionModel?: GoodmemExtractionModel;
  /**
   * Replaces model-based extraction entirely: receives the latest user
   * message's text and returns the facts to store. Cannot be combined with
   * `extractionModel`.
   */
  extractFacts?: GoodmemFactExtractor;
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

function metadataFor(retrieved: RetrieveMemoriesResult, save?: TurnSave) {
  return JSON.parse(
    JSON.stringify({
      partial: retrieved.partial,
      resultCount: retrieved.results.length,
      memoryIds: [...new Set(retrieved.results.map((r) => r.memoryId))],
      statuses: retrieved.statuses.map((s) => ({
        code: s.code,
        message: s.message,
        ...(s.details ? { details: s.details } : {}),
      })),
      ...(save ?? {}),
    })
  );
}

function warningsFor(retrieved: RetrieveMemoriesResult, save?: TurnSave): Warning[] {
  const warnings: Warning[] = [];
  if (retrieved.partial && retrieved.warning) {
    warnings.push({ type: 'other', message: `[goodmem] ${retrieved.warning}` } as Warning);
  }
  if (save && 'saveError' in save) {
    warnings.push({ type: 'other', message: `[goodmem] automatic saving failed -- ${save.saveError.message}` } as Warning);
  }
  return warnings;
}

/**
 * A save started for one turn: the extraction runs alongside the lookup and
 * the model call; storing waits for `commit`, which is only called once the
 * model call has succeeded.
 */
interface PendingSave {
  /** Store the extracted facts (once); resolves with the outcome, never rejects. */
  commit(retrieved: RetrieveMemoriesResult['results']): Promise<TurnSave>;
  /** Stop it: the call failed or was aborted, so nothing is stored or reported. */
  cancel(reason: unknown): void;
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
 * With `addMemory: 'always'`, the durable facts the user stated in the latest
 * user message are extracted with `extractionModel` (or the wrapped model,
 * called directly) and stored, once per user turn, alongside the model call.
 *
 * The outcome is reported on each result as `providerMetadata.goodmem`:
 * `{ partial, resultCount, memoryIds, statuses }`, plus `saved` or
 * `saveError` when automatic saving ran.
 */
export function withGoodmem(
  model: GoodmemWrappableModel,
  config: GoodmemMiddlewareConfig
): ReturnType<typeof wrapLanguageModel> {
  const conn = connect(config, 'withGoodmem', [
    'position',
    'template',
    'skipMemoryOnError',
    'retrievalTimeoutMs',
    'addMemory',
    'extractionModel',
    'extractFacts',
  ]);
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

  const addMemory = config.addMemory ?? 'never';
  if (addMemory !== 'never' && addMemory !== 'always') {
    throw new GoodMemConfigError(`withGoodmem: addMemory must be 'never' or 'always' (got ${JSON.stringify(config.addMemory)}).`);
  }
  const { extractionModel, extractFacts } = config;
  if (extractFacts !== undefined && typeof extractFacts !== 'function') {
    throw new GoodMemConfigError('withGoodmem: extractFacts must be a function returning the facts to store.');
  }
  if (extractFacts !== undefined && extractionModel !== undefined) {
    throw new GoodMemConfigError('withGoodmem: pass extractionModel or extractFacts, not both; extractFacts replaces model-based extraction.');
  }
  if (
    extractionModel !== undefined &&
    !(typeof extractionModel === 'string' && extractionModel.trim()) &&
    !(extractionModel && typeof extractionModel === 'object' && typeof (extractionModel as { doGenerate?: unknown }).doGenerate === 'function')
  ) {
    throw new GoodMemConfigError('withGoodmem: extractionModel must be a language model, e.g. openai("gpt-4o-mini").');
  }
  // The extraction call goes to the model itself, never through this
  // middleware: no memories are injected into it and it cannot recurse.
  const extract = extractFacts
    ? async (text: string, signal: AbortSignal) => ({ facts: await extractFacts({ text, signal }) })
    : (text: string, signal: AbortSignal) =>
        extractWithModel(extractionModel ?? (model as GoodmemExtractionModel), text, signal);

  /** Start saving one turn's facts, bounded by timeoutMs and by the caller's abort. */
  const startSave = (text: string, callerSignal: AbortSignal | undefined): PendingSave => {
    const controller = new AbortController();
    const timer = setTimeout(
      () =>
        controller.abort(
          new GoodMemError(`Automatic saving timed out after ${conn.timeoutMs} ms (timeoutMs).`, { timedOut: true, isRetryable: true })
        ),
      conn.timeoutMs
    );
    const onAbort = () => cancel(callerSignal?.reason);
    const cleanUp = () => {
      clearTimeout(timer);
      callerSignal?.removeEventListener('abort', onAbort);
    };
    function cancel(reason: unknown) {
      cleanUp();
      controller.abort(reason);
    }
    if (callerSignal?.aborted) cancel(callerSignal.reason);
    else callerSignal?.addEventListener('abort', onAbort, { once: true });

    const extraction = runExtraction(text, extract, controller.signal);
    let committed: Promise<TurnSave> | undefined;
    return {
      cancel,
      commit(retrieved) {
        committed ??= extraction
          .then((extracted) => storeFacts(conn, extracted, retrieved, controller.signal))
          .then((result) => {
            cleanUp();
            if ('saveError' in result) {
              conn.logger.warn(`[goodmem] automatic saving failed at the ${result.saveError.stage} stage -- ${result.saveError.message}`);
            }
            return result;
          });
        return committed;
      },
    };
  };

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

  const outcomes = new WeakMap<object, { retrieved: RetrieveMemoriesResult; save?: PendingSave }>();

  const middleware: LanguageModelMiddleware = {
    specificationVersion: 'v3',

    transformParams: async ({ params }) => {
      const query = lastUserText(params.prompt);
      if (!query) return params;
      // Save once per turn: only on the step that answers a user message, not
      // on the later steps of a tool loop. Extraction starts now and runs
      // alongside the lookup and the model call; nothing is stored unless the
      // model call succeeds.
      const firstStep = params.prompt[params.prompt.length - 1]?.role === 'user';
      const save = addMemory === 'always' && firstStep ? startSave(query, params.abortSignal) : undefined;
      let retrieved: RetrieveMemoriesResult;
      try {
        retrieved = await lookUp(query, params.abortSignal);
      } catch (error) {
        save?.cancel(error);
        throw error;
      }
      const text = template ? template(retrieved) : retrieved.context;
      if (typeof text !== 'string') {
        save?.cancel(new Error('template returned a non-string'));
        throw new GoodMemConfigError('withGoodmem: template must return a string.');
      }
      const next = { ...params, prompt: text ? inject(params.prompt, text, position) : params.prompt };
      outcomes.set(next, { retrieved, save });
      return next;
    },

    wrapGenerate: async ({ doGenerate, params }) => {
      const entry = outcomes.get(params);
      let result: GenerateResult;
      try {
        result = await doGenerate();
      } catch (error) {
        entry?.save?.cancel(error);
        throw error;
      }
      if (!entry) return result;
      const save = entry.save ? await entry.save.commit(entry.retrieved.results) : undefined;
      return {
        ...result,
        warnings: [...(result.warnings ?? []), ...warningsFor(entry.retrieved, save)],
        providerMetadata: { ...(result.providerMetadata ?? {}), goodmem: metadataFor(entry.retrieved, save) },
      } as GenerateResult;
    },

    wrapStream: async ({ doStream, params }) => {
      const entry = outcomes.get(params);
      let result: StreamResult;
      try {
        result = await doStream();
      } catch (error) {
        entry?.save?.cancel(error);
        throw error;
      }
      if (!entry) return result;
      // The save's outcome is only known later, so a failed save is reported
      // in the finish part's metadata and the log, not in the warnings that
      // open the stream. A stream that fails, ends without a finish part or
      // is cancelled stores nothing.
      const warnings = warningsFor(entry.retrieved);
      const source = result.stream.getReader();
      let finished = false;
      const annotated = new ReadableStream<StreamPart>({
        async pull(controller) {
          let chunk: ReadableStreamReadResult<StreamPart>;
          try {
            chunk = await source.read();
          } catch (error) {
            entry.save?.cancel(error);
            controller.error(error);
            return;
          }
          if (chunk.done) {
            if (!finished) entry.save?.cancel(new Error('the stream ended without a finish part'));
            controller.close();
            return;
          }
          const part = chunk.value;
          if (part.type === 'stream-start' && warnings.length) {
            controller.enqueue({ ...part, warnings: [...part.warnings, ...warnings] } as StreamPart);
          } else if (part.type === 'finish') {
            finished = true;
            const save = entry.save ? await entry.save.commit(entry.retrieved.results) : undefined;
            controller.enqueue({
              ...part,
              providerMetadata: { ...(part.providerMetadata ?? {}), goodmem: metadataFor(entry.retrieved, save) },
            } as StreamPart);
          } else {
            controller.enqueue(part);
          }
        },
        cancel(reason) {
          entry.save?.cancel(reason);
          return source.cancel(reason);
        },
      });
      return { ...result, stream: annotated } as StreamResult;
    },
  };

  return wrapLanguageModel({ model, middleware });
}
