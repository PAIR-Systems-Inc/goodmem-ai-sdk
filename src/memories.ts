/**
 * Writes: storing text and binary documents as memories.
 *
 * Content is passed as text or bytes. There is deliberately no file-path
 * input anywhere in this package, so no path chosen by a model -- or by
 * anyone else -- can make it read a file from the host.
 */

import { connect, type Connection, type GoodmemConfig } from './config.js';
import {
  GoodMemConfigError,
  GoodMemError,
  GoodMemIndexingError,
  GoodMemIngestionError,
  wrapError,
} from './errors.js';

/** A memory to store: plain text, or a document as bytes with its MIME type. */
export type MemoryInput =
  | string
  | {
      /** The text to store. */
      text: string;
      /** MIME type of the text. Defaults to `text/plain`. */
      contentType?: string;
      /** Metadata to attach; merged with the configured `scope`. */
      metadata?: Record<string, unknown>;
    }
  | {
      /** The document's bytes, e.g. a PDF. */
      data: Uint8Array | ArrayBuffer;
      /** The document's MIME type, e.g. `application/pdf`. Required. */
      contentType: string;
      /** A file name to record with the upload. */
      filename?: string;
      /** Metadata to attach; merged with the configured `scope`. */
      metadata?: Record<string, unknown>;
    };

/** Options for `addMemories`. */
export interface AddMemoriesOptions {
  /**
   * Wait until every stored memory has finished indexing (and so is
   * searchable) before returning. Off by default: nothing in this package
   * waits or polls unless asked. The wait polls only the memories this call
   * created, and is bounded by `timeoutMs` (default 60000).
   */
  waitForIndexing?: boolean | { timeoutMs?: number; pollIntervalMs?: number };
  /** Cancels the requests. */
  signal?: AbortSignal;
}

/** One stored memory. */
export interface AddedMemory {
  memoryId: string;
  spaceId: string;
  contentType: string;
  /** `PENDING` right after the write; `COMPLETED` after a successful wait. */
  processingStatus: string;
}

/** The memories one `addMemories` call stored, in input order. */
export interface AddMemoriesResult {
  memories: AddedMemory[];
}

type Normalized =
  | { kind: 'text'; text: string; contentType: string; metadata?: Record<string, unknown> }
  | { kind: 'bytes'; bytes: Uint8Array; contentType: string; filename?: string; metadata?: Record<string, unknown> };

const DEFAULT_WAIT_MS = 60_000;
const DEFAULT_POLL_MS = 1_000;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalize(input: MemoryInput, index: number, scope: Readonly<Record<string, unknown>>): Normalized {
  const where = `addMemories: input ${index}`;
  const item = typeof input === 'string' ? { text: input } : input;
  if (!isPlainObject(item)) {
    throw new GoodMemConfigError(`${where} must be a string, { text } or { data, contentType }.`);
  }
  if (item.metadata !== undefined && !isPlainObject(item.metadata)) {
    throw new GoodMemConfigError(`${where}: metadata must be a plain object.`);
  }
  const own = (item.metadata ?? {}) as Record<string, unknown>;
  for (const [key, value] of Object.entries(scope)) {
    if (key in own && own[key] !== value) {
      throw new GoodMemConfigError(
        `${where} sets metadata.${key}=${JSON.stringify(own[key])}, but the configured scope requires ` +
          `${JSON.stringify(value)}. Scope keys cannot be overridden per memory.`
      );
    }
  }
  const merged = { ...own, ...scope };
  const metadata = Object.keys(merged).length ? merged : undefined;

  if ('data' in item && item.data !== undefined) {
    const data = item.data as unknown;
    const bytes =
      data instanceof Uint8Array ? data : data instanceof ArrayBuffer ? new Uint8Array(data) : undefined;
    if (!bytes || bytes.byteLength === 0) {
      throw new GoodMemConfigError(`${where}: data must be a non-empty Uint8Array, Buffer or ArrayBuffer.`);
    }
    if (typeof item.contentType !== 'string' || !item.contentType.trim()) {
      throw new GoodMemConfigError(`${where}: contentType is required with data, e.g. 'application/pdf'.`);
    }
    if (item.filename !== undefined && typeof item.filename !== 'string') {
      throw new GoodMemConfigError(`${where}: filename must be a string.`);
    }
    return { kind: 'bytes', bytes, contentType: item.contentType.trim(), filename: item.filename, metadata };
  }
  const text = (item as { text?: unknown }).text;
  if (typeof text !== 'string' || text.trim().length === 0) {
    throw new GoodMemConfigError(`${where} needs non-empty text, or data with a contentType.`);
  }
  const contentType = (item as { contentType?: unknown }).contentType ?? 'text/plain';
  if (typeof contentType !== 'string' || !contentType.trim()) {
    throw new GoodMemConfigError(`${where}: contentType must be a non-empty string.`);
  }
  return { kind: 'text', text, contentType: contentType.trim(), metadata };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

const plural = (n: number, word: string) => `${n} ${n === 1 ? word : word.replace(/y$/, 'ie') + 's'}`;
const were = (n: number) => (n === 1 ? 'was' : 'were');

/** Store memories over a connection. Shared by `addMemories` and the tool. */
export async function add(
  conn: Connection,
  inputs: MemoryInput[],
  options: AddMemoriesOptions = {}
): Promise<AddMemoriesResult> {
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new GoodMemConfigError('addMemories needs at least one memory to store.');
  }
  // Every input is checked before anything is written.
  const items = inputs.map((input, i) => normalize(input, i, conn.scope));
  const wait = options.waitForIndexing;
  const waitMs = typeof wait === 'object' ? wait.timeoutMs ?? DEFAULT_WAIT_MS : DEFAULT_WAIT_MS;
  const pollMs = typeof wait === 'object' ? wait.pollIntervalMs ?? DEFAULT_POLL_MS : DEFAULT_POLL_MS;
  for (const [name, value] of [['timeoutMs', waitMs], ['pollIntervalMs', pollMs]] as const) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new GoodMemConfigError(`waitForIndexing.${name} must be a positive number of milliseconds.`);
    }
  }

  const { signal } = options;
  const requestOptions = signal ? { signal } : undefined;
  const context = conn.errorContext(signal);
  const [spaceId] = await conn.spaceIds();
  const stored: AddedMemory[] = [];

  for (const [index, item] of items.entries()) {
    try {
      const memory =
        item.kind === 'text'
          ? await conn.client.memories.create(
              {
                spaceId,
                originalContent: item.text,
                contentType: item.contentType,
                ...(item.metadata ? { metadata: item.metadata } : {}),
              },
              requestOptions
            )
          : await conn.client.memories.createFromBytes(
              {
                spaceId,
                bytes: item.bytes,
                contentType: item.contentType,
                ...(item.filename ? { filename: item.filename } : {}),
                ...(item.metadata ? { metadata: item.metadata } : {}),
              },
              requestOptions
            );
      stored.push({
        memoryId: String(memory.memoryId),
        spaceId: String(memory.spaceId ?? spaceId),
        contentType: String(memory.contentType ?? item.contentType),
        processingStatus: String(memory.processingStatus ?? 'PENDING'),
      });
    } catch (error) {
      const wrapped = wrapError(error, `Storing memory ${index + 1} of ${items.length}`, context);
      if (!GoodMemError.isInstance(wrapped) && stored.length === 0) throw wrapped;
      const created = stored.map((m) => m.memoryId);
      const base = GoodMemError.isInstance(wrapped)
        ? wrapped
        : new GoodMemError(`Storing memory ${index + 1} of ${items.length} was aborted`, { cause: wrapped });
      throw new GoodMemIngestionError(
        base.message +
          (created.length
            ? ` (${plural(created.length, 'earlier memory')} ${were(created.length)} stored: ${created.join(', ')}; ` +
              `retry from input ${index} to avoid duplicates)`
            : ''),
        {
          createdMemoryIds: created,
          failedIndex: index,
          statusCode: base.statusCode,
          body: base.body,
          timedOut: base.timedOut,
          cause: base,
        }
      );
    }
  }

  if (wait) await waitForIndexing(conn, stored, waitMs, pollMs, signal);
  return { memories: stored };
}

async function waitForIndexing(
  conn: Connection,
  stored: AddedMemory[],
  waitMs: number,
  pollMs: number,
  signal?: AbortSignal
): Promise<void> {
  const requestOptions = signal ? { signal } : undefined;
  const deadline = Date.now() + waitMs;
  const pending = new Map(stored.map((m) => [m.memoryId, m]));
  const failed: string[] = [];
  const reasons: string[] = [];
  const memoryIds = stored.map((m) => m.memoryId);

  try {
    while (pending.size > 0) {
      for (const [id, entry] of [...pending]) {
        const memory = await conn.client.memories.get(id, undefined, requestOptions);
        entry.processingStatus = String(memory.processingStatus ?? entry.processingStatus);
        if (entry.processingStatus === 'COMPLETED') pending.delete(id);
        if (entry.processingStatus === 'FAILED') {
          pending.delete(id);
          failed.push(id);
          const history = await conn.client.memories.get(id, { includeProcessingHistory: true }, requestOptions);
          const attempts = history.processingHistory?.attempts ?? [];
          const reason = attempts.length ? attempts[attempts.length - 1]?.statusMessage : undefined;
          if (reason) reasons.push(`${id}: ${reason}`);
        }
      }
      const left = deadline - Date.now();
      if (pending.size === 0 || left <= 0) break;
      await sleep(Math.min(pollMs, left), signal);
    }
  } catch (error) {
    if (signal?.aborted) throw signal.reason ?? error;
    const wrapped = wrapError(error, 'Checking indexing status', conn.errorContext(signal)) as Error;
    throw new GoodMemIndexingError(
      `${plural(stored.length, 'memory')} ${were(stored.length)} stored, but checking whether they finished indexing failed: ` +
        `${wrapped.message}. The writes succeeded; do not store them again.`,
      { memoryIds, failedMemoryIds: failed, pendingMemoryIds: [...pending.keys()], cause: wrapped }
    );
  }

  if (failed.length || pending.size) {
    const parts: string[] = [];
    if (failed.length) {
      parts.push(
        `${plural(failed.length, 'memory')} failed processing on the server (${failed.join(', ')})` +
          (reasons.length ? `: ${reasons.join('; ')}` : '')
      );
    }
    if (pending.size) {
      parts.push(`${plural(pending.size, 'memory')} had not finished indexing after ${waitMs} ms (${[...pending.keys()].join(', ')})`);
    }
    throw new GoodMemIndexingError(
      `${plural(stored.length, 'memory')} ${were(stored.length)} stored, but ${parts.join(', and ')}. ` +
        'The writes succeeded; do not store them again.',
      {
        memoryIds,
        failedMemoryIds: failed,
        pendingMemoryIds: [...pending.keys()],
        timedOut: pending.size > 0,
      }
    );
  }
}

/**
 * Store one or more memories: plain text, or documents as bytes (a PDF, for
 * example) with their MIME type.
 *
 * Every memory is written to the configuration's write space (the first of
 * `spaceIds`, or the named `space`) with the configured `scope` merged into
 * its metadata. Inputs are validated before anything is written; if a write
 * fails part-way, `GoodMemIngestionError` lists the memories already stored.
 */
export async function addMemories(
  input: MemoryInput | MemoryInput[],
  config: GoodmemConfig,
  options: AddMemoriesOptions = {}
): Promise<AddMemoriesResult> {
  const conn = connect(config, 'addMemories');
  return add(conn, Array.isArray(input) ? input : [input], options);
}
