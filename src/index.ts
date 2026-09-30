/**
 * GoodMem memory for the Vercel AI SDK.
 *
 * - `goodmemTools(config)` -- `searchMemories` and `addMemory` tools for a model.
 * - `withGoodmem(model, config)` -- a middleware that adds relevant memories
 *   to every call.
 * - `retrieveMemories`, `searchMemories`, `addMemories` -- explicit helpers.
 * - `filters` -- builds metadata filter expressions with correct escaping.
 *
 * Everything talks to GoodMem through the official `@pairsystems/goodmem` SDK.
 */

export { goodmemTools } from './tools.js';
export type {
  AddMemoryToolInput,
  AddMemoryToolOutput,
  GoodmemTools,
  SearchMemoriesToolInput,
  SearchMemoriesToolOutput,
} from './tools.js';

export { withGoodmem } from './middleware.js';
export type {
  GoodmemInjectionPosition,
  GoodmemMiddlewareConfig,
  GoodmemWrappableModel,
} from './middleware.js';

export { retrieveMemories, searchMemories } from './retrieval.js';
export type { RetrieveMemoriesResult, SearchOptions } from './retrieval.js';

export { addMemories } from './memories.js';
export type { AddedMemory, AddMemoriesOptions, AddMemoriesResult, MemoryInput } from './memories.js';

export * as filters from './filters.js';
export { GoodMemFilterError } from './filters.js';
export type { FilterValue } from './filters.js';

export {
  GoodMemConfigError,
  GoodMemError,
  GoodMemIndexingError,
  GoodMemIngestionError,
} from './errors.js';
export type { GoodMemErrorOptions } from './errors.js';

export { MALFORMED_STREAM_CODE, RETRIEVAL_FAILED_CODE, UNKNOWN_CODE } from './results.js';
export type { MemoryResult, RetrievalStatus, ScoreKind, SearchMemoriesResult } from './results.js';

export type { GoodmemConfig, GoodmemLogger, GoodmemNamedSpace } from './config.js';
