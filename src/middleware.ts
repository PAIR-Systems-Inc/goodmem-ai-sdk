/**
 * Automatic memory: a language-model middleware that retrieves relevant
 * memories for every call and adds them to the prompt.
 *
 * Only the outgoing request is augmented. The caller's messages are never
 * modified, so the injected memories are not saved into chat history as if
 * the user or the model had said them.
 */

import { wrapLanguageModel, type LanguageModelMiddleware } from 'ai';

import { connect, type GoodmemConfig } from './config.js';
import { GoodMemConfigError } from './errors.js';
import { retrieve, type RetrieveMemoriesResult } from './retrieval.js';

type TransformOptions = Parameters<NonNullable<LanguageModelMiddleware['transformParams']>>[0];
type CallParams = TransformOptions['params'];
type Prompt = CallParams['prompt'];
type Message = Prompt[number];
type GenerateResult = Awaited<ReturnType<NonNullable<LanguageModelMiddleware['wrapGenerate']>>>;
type StreamResult = Awaited<ReturnType<NonNullable<LanguageModelMiddleware['wrapStream']>>>;
type StreamPart = StreamResult['stream'] extends ReadableStream<infer P> ? P : never;
type Warning = NonNullable<GenerateResult['warnings']>[number];

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
 * Wrap a language model so every call is given relevant memories.
 *
 * For each call, the text of the latest user message is searched in the
 * configured space(s) and the results are added to the prompt. A call with
 * no user text is passed through untouched.
 *
 * Retrieval follows the GoodMem status contract:
 * - a degraded retrieval (the server reported a problem) still injects what
 *   arrived, says in the injected text that memories may be missing, adds a
 *   warning to the call's `warnings`, and logs it -- it is never presented as
 *   "no memories";
 * - an unreachable server, a timeout or a rejected request throws
 *   `GoodMemError`, failing the call rather than silently injecting nothing.
 *
 * The outcome is reported on each result as `providerMetadata.goodmem`:
 * `{ partial, resultCount, memoryIds, statuses }`.
 */
export function withGoodmem(
  model: GoodmemWrappableModel,
  config: GoodmemMiddlewareConfig
): ReturnType<typeof wrapLanguageModel> {
  const conn = connect(config, 'withGoodmem', ['position', 'template']);
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

  const outcomes = new WeakMap<object, RetrieveMemoriesResult>();

  const middleware: LanguageModelMiddleware = {
    specificationVersion: 'v3',

    transformParams: async ({ params }) => {
      const query = lastUserText(params.prompt);
      if (!query) return params;
      const retrieved = await retrieve(conn, query, { signal: params.abortSignal });
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
