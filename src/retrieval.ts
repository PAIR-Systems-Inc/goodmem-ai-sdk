/**
 * Search: the one retrieval path behind the tool, the middleware and the
 * developer helpers.
 */

import { connect, MAX_TOP_K, type Connection, type GoodmemConfig } from './config.js';
import { GoodMemConfigError, wrapError } from './errors.js';
import { foldRetrieval, type RetrievalStatus, type SearchMemoriesResult } from './results.js';

const CHAT_POST_PROCESSOR = 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory';

/** Options for a single search. */
export interface SearchOptions {
  /** How many results to return. Defaults to the configuration's `topK`. */
  topK?: number;
  /**
   * Also return each memory's original content (`content` and
   * `contentEncoding` on every result): text as text, anything else base64.
   */
  includeContent?: boolean;
  /** Cancels the request. */
  signal?: AbortSignal;
}

/** A search result together with a context block ready to put in a prompt. */
export interface RetrieveMemoriesResult extends SearchMemoriesResult {
  /**
   * The results formatted for a prompt, most relevant first. Empty when
   * nothing was found and nothing went wrong; when the retrieval was
   * degraded it says so, so a model is never told there are no memories
   * when the lookup failed.
   */
  context: string;
}

function codes(statuses: RetrievalStatus[]): string {
  return [...new Set(statuses.map((s) => s.code))].join(', ');
}

/** Format a search result as a context block for a prompt. */
export function formatContext(result: SearchMemoriesResult): string {
  if (result.results.length === 0) {
    return result.partial
      ? `GoodMem memory retrieval failed for this request (${codes(result.statuses)}). ` +
          'Relevant memories may exist but could not be retrieved; do not assume there are none.'
      : '';
  }
  const lines = ['Relevant memories from GoodMem, most relevant first:'];
  result.results.forEach((r, i) => lines.push(`${i + 1}. ${r.text.trim()}`));
  if (result.partial) {
    lines.push(
      `Note: GoodMem reported a problem during this retrieval (${codes(result.statuses)}); ` +
        'some relevant memories may be missing.'
    );
  }
  return lines.join('\n');
}

/** Run one retrieval over a connection. Shared by every entry point. */
export async function search(
  conn: Connection,
  query: string,
  options: SearchOptions = {}
): Promise<SearchMemoriesResult> {
  if (typeof query !== 'string' || query.trim().length === 0) {
    throw new GoodMemConfigError('A search needs a non-empty query string.');
  }
  const topK = options.topK ?? conn.topK;
  if (!Number.isInteger(topK) || topK < 1 || topK > MAX_TOP_K) {
    throw new GoodMemConfigError(`topK must be an integer from 1 to ${MAX_TOP_K} (got ${String(options.topK)}).`);
  }
  const context = conn.errorContext(options.signal);
  const spaceIds = await conn.spaceIds();
  const request = {
    message: query,
    spaceKeys: spaceIds.map((spaceId) => (conn.readFilter ? { spaceId, filter: conn.readFilter } : { spaceId })),
    requestedSize: topK,
    fetchMemory: true,
    ...(options.includeContent ? { fetchMemoryContent: true } : {}),
    ...(conn.rerankerId
      ? { postProcessor: { name: CHAT_POST_PROCESSOR, config: { reranker_id: conn.rerankerId } } }
      : {}),
  };

  let result: SearchMemoriesResult;
  try {
    result = await foldRetrieval(
      conn.client.memories.retrieve(request, options.signal ? { signal: options.signal } : undefined)
    );
  } catch (error) {
    throw wrapError(error, 'Searching GoodMem', context);
  }

  if (conn.minScore !== undefined) {
    const threshold = conn.minScore;
    const reranked = result.results.filter((r) => r.scoreKind === 'reranker');
    const kept = result.results.filter(
      (r) => r.scoreKind !== 'reranker' || (r.score !== null && r.score >= threshold)
    );
    if (reranked.length > 0 && kept.length === 0) {
      const scores = reranked.map((r) => r.score).filter((s): s is number => s !== null);
      conn.logger.warn(
        `[goodmem] minScore=${threshold} removed all ${reranked.length} reranked result(s); observed scores ` +
          `ranged ${Math.min(...scores).toFixed(4)}..${Math.max(...scores).toFixed(4)}. Reranker scales are ` +
          'provider-dependent, not 0-1: calibrate minScore for this reranker.'
      );
    }
    result.results = kept;
  }

  if (result.partial) {
    conn.logger.warn(
      `[goodmem] ${result.warning}` +
        (result.results.length === 0
          ? ' -- no results came back, and this is not an empty index: the retrieval failed.'
          : ` -- returning the ${result.results.length} result(s) that did arrive.`)
    );
  }
  return result;
}

/** Search and format the results as a context block. */
export async function retrieve(
  conn: Connection,
  query: string,
  options: SearchOptions = {}
): Promise<RetrieveMemoriesResult> {
  const result = await search(conn, query, options);
  return { ...result, context: formatContext(result) };
}

/**
 * Search GoodMem and return structured results.
 *
 * Each result carries its chunk and memory ids, the memory's metadata, a
 * score oriented so higher is better, the raw server score and its kind.
 * `partial` is true when the server reported a problem, whether or not
 * results came back; a request that fails outright throws `GoodMemError`.
 */
export async function searchMemories(
  query: string,
  config: GoodmemConfig,
  options: SearchOptions = {}
): Promise<SearchMemoriesResult> {
  return search(connect(config, 'searchMemories'), query, options);
}

/**
 * Search GoodMem and return the results together with a formatted context
 * block, ready to place in a system prompt.
 */
export async function retrieveMemories(
  query: string,
  config: GoodmemConfig,
  options: SearchOptions = {}
): Promise<RetrieveMemoriesResult> {
  return retrieve(connect(config, 'retrieveMemories'), query, options);
}
