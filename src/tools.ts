/**
 * Model-facing tools.
 *
 * The model sees two tools and chooses only what to search for and what to
 * remember. Spaces, filters, scope, reranking and timeouts are fixed by the
 * developer's configuration; there is no delete, update or space-management
 * tool, and no tool takes a file path.
 */

import { tool, type Tool } from 'ai';
import { z } from 'zod';

import { connect, type GoodmemConfig } from './config.js';
import { add } from './memories.js';
import type { MemoryResult, RetrievalStatus } from './results.js';
import { search } from './retrieval.js';

/** Most results the model may ask for in one search. */
export const MAX_TOOL_TOP_K = 50;

/** What the model passes to `searchMemories`. */
export interface SearchMemoriesToolInput {
  query: string;
  topK?: number;
}

/** What `searchMemories` returns to the model. */
export interface SearchMemoriesToolOutput {
  results: Array<Omit<MemoryResult, 'content' | 'contentEncoding'>>;
  totalResults: number;
  /** True when GoodMem reported a problem; results may be missing. */
  partial: boolean;
  statuses: RetrievalStatus[];
  warning?: string;
}

/** What the model passes to `addMemory`. */
export interface AddMemoryToolInput {
  text: string;
}

/** What `addMemory` returns to the model. */
export interface AddMemoryToolOutput {
  memoryId: string;
  spaceId: string;
  processingStatus: string;
}

/**
 * The tools returned by `goodmemTools`. A type alias rather than an
 * interface, so it is assignable to the AI SDK's `ToolSet`.
 */
export type GoodmemTools = {
  /** Search long-term memory. */
  searchMemories: Tool<SearchMemoriesToolInput, SearchMemoriesToolOutput>;
  /** Store a piece of text in long-term memory. */
  addMemory: Tool<AddMemoryToolInput, AddMemoryToolOutput>;
};

/**
 * GoodMem tools for `generateText`, `streamText` and agents.
 *
 * Pass both, or only `searchMemories` for read-only memory:
 *
 * ```ts
 * const { searchMemories } = goodmemTools(config);
 * ```
 *
 * A request that fails throws `GoodMemError`, which the AI SDK reports to the
 * model as a tool error. A degraded search is not an error: it returns the
 * results that arrived with `partial: true` and the server's statuses.
 */
export function goodmemTools(config: GoodmemConfig): GoodmemTools {
  const conn = connect(config, 'goodmemTools');

  const searchMemories = tool({
    description:
      'Search long-term memory for information relevant to the current request: facts, preferences, ' +
      'decisions, documents or context saved in earlier conversations. Returns matching memory text, ' +
      'most relevant first, with a relevance score (higher is better). If "partial" is true, memory ' +
      'retrieval ran into a problem and relevant memories may be missing.',
    inputSchema: z.object({
      query: z.string().min(1).describe('What to look for, in natural language.'),
      topK: z
        .number()
        .int()
        .min(1)
        .max(MAX_TOOL_TOP_K)
        .optional()
        .describe(`How many memories to return (default ${conn.topK}).`),
    }),
    execute: async ({ query, topK }, { abortSignal }): Promise<SearchMemoriesToolOutput> => {
      const result = await search(conn, query, { topK: topK ?? conn.topK, signal: abortSignal });
      return {
        results: result.results.map(({ content: _content, contentEncoding: _encoding, ...rest }) => rest),
        totalResults: result.results.length,
        partial: result.partial,
        statuses: result.statuses,
        ...(result.warning ? { warning: result.warning } : {}),
      };
    },
  });

  const addMemory = tool({
    description:
      'Save information to long-term memory so it can be recalled in later conversations. Use it for ' +
      'durable facts, preferences or decisions worth remembering, written as a self-contained statement. ' +
      'A saved memory becomes searchable a few seconds after it is stored.',
    inputSchema: z.object({
      text: z.string().min(1).describe('The information to remember, as a self-contained statement.'),
    }),
    execute: async ({ text }, { abortSignal }): Promise<AddMemoryToolOutput> => {
      const { memories } = await add(conn, [text], { signal: abortSignal });
      const [memory] = memories;
      return {
        memoryId: memory.memoryId,
        spaceId: memory.spaceId,
        processingStatus: memory.processingStatus,
      };
    },
  });

  return { searchMemories, addMemory };
}
