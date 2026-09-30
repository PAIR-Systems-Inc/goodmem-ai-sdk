/**
 * Shared handling of a GoodMem retrieval stream.
 *
 * Every retrieval in this package -- the model-facing tool, the middleware
 * and the developer helpers -- is folded through this module, so they cannot
 * drift apart in how they classify a status, join a chunk to its memory,
 * orient a score or decode content.
 */

/**
 * Status codes that report an optional feature the caller never configured.
 * The server files both under "Informational status messages (non-error)":
 * nothing the caller asked for is missing, so they are noise unconditionally
 * and their `details` are never inspected to decide (contract rule Q1).
 */
export const INFORMATIONAL_CODES: ReadonlySet<string> = new Set([
  'LLM_CAPABILITY_INFERRED',
  'FEATURE_DISABLED',
]);

/** Surfaced in place of a status code this build does not recognise. */
export const UNKNOWN_CODE = 'UNKNOWN';

/** Reported when the stream broke off or carried an undecodable line. */
export const MALFORMED_STREAM_CODE = 'MALFORMED_STREAM';

/**
 * Reported by the middleware when a lookup failed outright for an
 * availability reason and the call went ahead without memories
 * (`skipMemoryOnError`).
 */
export const RETRIEVAL_FAILED_CODE = 'RETRIEVAL_FAILED';

/** Codes this build knows about. Anything else becomes `UNKNOWN`. */
export const KNOWN_CODES: ReadonlySet<string> = new Set([
  'GOODMEM_STATUS_CODE_UNSPECIFIED',
  'INVALID_ARGUMENT',
  'NOT_FOUND',
  'PERMISSION_DENIED',
  'FAILED_PRECONDITION',
  'EMBEDDER_FAILED',
  'EMBEDDER_UNAVAILABLE',
  'EMBEDDER_TIMEOUT',
  'VECTOR_SEARCH_FAILED',
  'VECTOR_SEARCH_PARTIAL',
  'VECTOR_SEARCH_TIMEOUT',
  'SPACE_INACCESSIBLE',
  'SPACE_NOT_FOUND',
  'SPACE_NO_EMBEDDERS',
  'CHUNK_NOT_FOUND',
  'MEMORY_LOAD_FAILED',
  'MEMORY_CONTENT_UNAVAILABLE',
  'RERANKING_FAILED',
  'SUMMARIZATION_FAILED',
  'SUMMARIZATION_TIMEOUT',
  'RATE_LIMITED',
  'RESOURCE_EXHAUSTED',
  'CONFIGURATION_ERROR',
  'LLM_CAPABILITY_INFERRED',
  'FEATURE_DISABLED',
]);

/** One problem the server reported during a retrieval. */
export interface RetrievalStatus {
  /** The server's code, or `UNKNOWN` when this build does not recognise it. */
  code: string;
  /** The server's own human-readable message. */
  message: string;
  /** Any structured detail the server attached. */
  details?: Record<string, unknown>;
}

/** How a score was produced. The two kinds are not on a common scale. */
export type ScoreKind = 'vector' | 'reranker';

/** One retrieved chunk, joined to the memory it came from. */
export interface MemoryResult {
  /** The chunk's text. */
  text: string;
  /** Unique id of the chunk. */
  chunkId: string;
  /** The memory the chunk belongs to. */
  memoryId: string;
  /** The space the memory is stored in. */
  spaceId: string;
  /** Relevance oriented so that higher is better. */
  score: number | null;
  /** The score exactly as the server sent it. */
  rawScore: number | null;
  /** `vector` (negated distance) or `reranker` (provider-dependent scale). */
  scoreKind: ScoreKind;
  /** MIME type of the memory's original content. */
  contentType: string;
  /** The memory's metadata. */
  metadata: Record<string, unknown>;
  /**
   * The memory's original content, present when requested with
   * `includeContent`: text as text, anything else base64.
   */
  content?: string;
  /** How `content` is encoded. */
  contentEncoding?: 'text' | 'base64';
}

/** Everything one retrieval produced. */
export interface SearchMemoriesResult {
  /** Matching chunks, most relevant first. */
  results: MemoryResult[];
  /** Problems the server reported. Empty when `partial` is false. */
  statuses: RetrievalStatus[];
  /**
   * True when the server reported a real problem during this retrieval.
   * Independent of whether results came back: a degraded retrieval returns
   * whatever arrived, flagged.
   */
  partial: boolean;
  /** A one-line summary of the problem, present when `partial` is true. */
  warning?: string;
  /** The server's id for the result set. */
  resultSetId: string;
}

/**
 * Classify one status event.
 *
 * Implements Q1 and Q3 of the retrieval status contract: the two
 * informational codes are noise unconditionally, and a code this build does
 * not recognise is surfaced as `UNKNOWN` rather than dropped or thrown on.
 * The SDK maps a code it does not know to `null`; that is `UNKNOWN` too.
 */
export function classifyStatus(
  rawCode: string | null | undefined,
  message: string
): { status: RetrievalStatus; informational: boolean } {
  if (!rawCode || !KNOWN_CODES.has(rawCode)) {
    return {
      status: {
        code: UNKNOWN_CODE,
        message,
        ...(rawCode ? { details: { serverCode: rawCode } } : {}),
      },
      informational: false,
    };
  }
  return {
    status: { code: rawCode, message },
    informational: INFORMATIONAL_CODES.has(rawCode),
  };
}

/**
 * Return a score oriented so that a higher number is a better match.
 *
 * GoodMem vector scores are negated inner products -- `-0.51` is a closer
 * match than `-0.29` -- while reranker scores are already higher-is-better on
 * a provider-dependent scale. Negating a reranker score would invert the
 * ranking, so only vector scores are flipped.
 */
export function orientScore(raw: number | null | undefined, kind: ScoreKind): number | null {
  if (raw === null || raw === undefined || !Number.isFinite(raw)) return null;
  return kind === 'reranker' ? raw : -raw;
}

/** A one-line summary of why a retrieval was degraded. */
export function warningText(statuses: RetrievalStatus[]): string {
  if (statuses.length === 0) return '';
  const parts = statuses.map((s) => (s.message ? `${s.code}: ${s.message}` : s.code));
  return `GoodMem reported a problem during retrieval -- ${parts.join('; ')}`;
}

/** Decode memory content by its content type: text as text, else base64. */
export function decodeContent(
  raw: Uint8Array,
  contentType: string
): { content: string; encoding: 'text' | 'base64' } {
  const [primaryPart, ...params] = (contentType || '').split(';');
  const primary = primaryPart.trim().toLowerCase();
  let charset = 'utf-8';
  for (const part of params) {
    const [key, value] = part.split('=');
    if (key?.trim().toLowerCase() === 'charset' && value?.trim()) {
      charset = value.trim().replace(/^"|"$/g, '');
    }
  }
  const textual =
    primary.startsWith('text/') ||
    primary.endsWith('+json') ||
    primary.endsWith('+xml') ||
    ['application/json', 'application/xml', 'application/javascript', 'application/x-ndjson'].includes(primary);
  const base64 = Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength).toString('base64');
  if (!textual) return { content: base64, encoding: 'base64' };
  try {
    return { content: new TextDecoder(charset, { fatal: true }).decode(raw), encoding: 'text' };
  } catch {
    // An undecodable body or an unknown charset: keep the exact bytes.
    return { content: base64, encoding: 'base64' };
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function toNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

interface PendingHit {
  resultSetId: string;
  chunkId: string;
  text: string;
  memoryId: string;
  rawScore: number | null;
}

/**
 * Fold a GoodMem retrieval stream into a result.
 *
 * - Chunks are joined to their memory by memory UUID, not by the positional
 *   `memoryIndex` the stream also carries, and independently of arrival
 *   order, so a reordered stream cannot attach a chunk to the wrong memory.
 * - Chunks are de-duplicated by chunk id, never by memory id, which would
 *   collapse distinct chunks of one document.
 * - Only the final stage is returned: when a reranking stage produced
 *   results, those are kept and scored as `reranker`; otherwise the retrieval
 *   stage's results are scored as `vector`. The kind comes from the stream's
 *   own stage name, so a reranker that failed does not mislabel vector scores.
 * - A stream that breaks off after some events is reported as
 *   `MALFORMED_STREAM` with whatever arrived kept. A request that never
 *   yields a single event throws: that is a failed request, not a search
 *   that found nothing.
 */
export async function foldRetrieval(events: AsyncIterable<unknown>): Promise<SearchMemoriesResult> {
  const statuses: RetrievalStatus[] = [];
  const memories = new Map<string, Record<string, unknown>>();
  const stages = new Map<string, string>();
  const pending: PendingHit[] = [];
  let resultSetId = '';
  let received = 0;

  try {
    for await (const raw of events) {
      received += 1;
      const event = asRecord(raw);
      if (event.resultSetBoundary) {
        const boundary = asRecord(event.resultSetBoundary);
        const id = String(boundary.resultSetId ?? '');
        if (id && boundary.kind === 'BEGIN') stages.set(id, String(boundary.stageName ?? ''));
        if (id && !resultSetId) resultSetId = id;
        continue;
      }
      if (event.status) {
        const status = asRecord(event.status);
        const classified = classifyStatus(
          typeof status.code === 'string' ? status.code : null,
          String(status.message ?? '')
        );
        const details = asRecord(status.details);
        if (Object.keys(details).length) {
          classified.status.details = { ...(classified.status.details ?? {}), ...details };
        }
        if (!classified.informational) statuses.push(classified.status);
        continue;
      }
      if (event.memoryDefinition) {
        const memory = asRecord(event.memoryDefinition);
        const id = String(memory.memoryId ?? '');
        if (id) memories.set(id, memory);
        continue;
      }
      const reference = asRecord(asRecord(event.retrievedItem).chunk);
      const chunk = asRecord(reference.chunk);
      const chunkId = String(chunk.chunkId ?? '');
      if (!chunkId) continue;
      pending.push({
        resultSetId: String(reference.resultSetId ?? ''),
        chunkId,
        text: String(chunk.chunkText ?? ''),
        memoryId: String(chunk.memoryId ?? ''),
        rawScore: toNumber(reference.relevanceScore),
      });
    }
  } catch (error) {
    if (received === 0) throw error;
    const cause = (error as { cause?: unknown })?.cause;
    const reason =
      (error instanceof Error ? error.message : String(error)) +
      (cause instanceof Error && cause.message ? ` (${cause.message})` : '');
    statuses.push({
      code: MALFORMED_STREAM_CODE,
      message: `The retrieval stream broke off after ${received} event(s); results may be incomplete: ${reason}`,
    });
  }

  const isRerank = (hit: PendingHit) => /rerank/i.test(stages.get(hit.resultSetId) ?? '');
  const reranked = pending.filter(isRerank);
  const final = reranked.length > 0 ? reranked : pending;
  const kind: ScoreKind = reranked.length > 0 ? 'reranker' : 'vector';

  const seen = new Set<string>();
  const results: MemoryResult[] = [];
  for (const hit of final) {
    if (seen.has(hit.chunkId)) continue;
    seen.add(hit.chunkId);
    const memory = memories.get(hit.memoryId) ?? {};
    const contentType = String(memory.contentType ?? '');
    const result: MemoryResult = {
      text: hit.text,
      chunkId: hit.chunkId,
      memoryId: hit.memoryId,
      spaceId: String(memory.spaceId ?? ''),
      score: orientScore(hit.rawScore, kind),
      rawScore: hit.rawScore,
      scoreKind: kind,
      contentType,
      metadata: { ...asRecord(memory.metadata) },
    };
    if (typeof memory.originalContent === 'string') {
      const decoded = decodeContent(Buffer.from(memory.originalContent, 'base64'), contentType);
      result.content = decoded.content;
      result.contentEncoding = decoded.encoding;
    }
    results.push(result);
  }

  const partial = statuses.length > 0;
  return {
    results,
    statuses,
    partial,
    ...(partial ? { warning: warningText(statuses) } : {}),
    resultSetId: final[0]?.resultSetId || resultSetId,
  };
}
