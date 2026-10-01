/**
 * Typed errors for GoodMem operations.
 *
 * Every failure that reaches a caller is one of these, carrying the server's
 * own explanation: the HTTP status and response body of a rejected request
 * are kept verbatim rather than reduced to "400 Bad Request". Like the AI
 * SDK's own errors, each class has a static `isInstance()` that works across
 * duplicate copies of this package, where `instanceof` does not.
 */

const MARKER = Symbol.for('pairsystems.goodmem-vercel-ai-sdk.error');

/** Options shared by every GoodMem error. */
export interface GoodMemErrorOptions {
  /** HTTP status the GoodMem server answered with, when it answered. */
  statusCode?: number;
  /** The server's response body, exactly as sent. */
  body?: string;
  /** True when the request was cut off by `timeoutMs`. */
  timedOut?: boolean;
  /**
   * True when the failure is about availability -- the server could not be
   * reached, was too slow, was overloaded (HTTP 5xx, 429, 408) or sent a broken
   * response -- so the same request may succeed later.
   */
  isRetryable?: boolean;
  /** The underlying error. */
  cause?: unknown;
}

/** Raised when a GoodMem operation fails. */
export class GoodMemError extends Error {
  /** HTTP status the GoodMem server answered with, when it answered. */
  readonly statusCode?: number;
  /** The server's response body, exactly as sent. */
  readonly body?: string;
  /** True when the request was cut off by `timeoutMs`. */
  readonly timedOut: boolean;
  /**
   * True for availability failures that may go away on their own: the server
   * could not be reached, was too slow, answered HTTP 5xx, 429 or 408, or sent
   * a broken response. False for failures that will repeat until something is
   * changed: a rejected key, a missing space, an invalid filter, a bad
   * configuration.
   */
  readonly isRetryable: boolean;

  constructor(message: string, options: GoodMemErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    Object.defineProperty(this, MARKER, { value: true });
    this.name = 'GoodMemError';
    this.statusCode = options.statusCode;
    this.body = options.body;
    this.timedOut = options.timedOut ?? false;
    this.isRetryable = options.isRetryable ?? false;
  }

  /** True for any error raised by this package, including subclasses. */
  static isInstance(error: unknown): error is GoodMemError {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[MARKER] === true;
  }
}

/** Raised when the configuration or an argument cannot be used as given. */
export class GoodMemConfigError extends GoodMemError {
  constructor(message: string) {
    super(message);
    this.name = 'GoodMemConfigError';
  }

  static isInstance(error: unknown): error is GoodMemConfigError {
    return GoodMemError.isInstance(error) && error.name === 'GoodMemConfigError';
  }
}

/**
 * Raised when a batch of writes fails part-way. The memories created before
 * the failure are listed, so a retry can resume at `failedIndex` instead of
 * writing duplicates.
 */
export class GoodMemIngestionError extends GoodMemError {
  /** Memories that were created before the failure, in input order. */
  readonly createdMemoryIds: string[];
  /** Index of the input that failed; inputs after it were not attempted. */
  readonly failedIndex: number;

  constructor(
    message: string,
    options: GoodMemErrorOptions & { createdMemoryIds: string[]; failedIndex: number }
  ) {
    super(message, options);
    this.name = 'GoodMemIngestionError';
    this.createdMemoryIds = options.createdMemoryIds;
    this.failedIndex = options.failedIndex;
  }

  static isInstance(error: unknown): error is GoodMemIngestionError {
    return GoodMemError.isInstance(error) && error.name === 'GoodMemIngestionError';
  }
}

/**
 * Raised by `addMemories(..., { waitForIndexing })` when a memory was stored
 * but did not become searchable: its processing failed, or the wait ran out.
 * The writes themselves succeeded; nothing needs to be uploaded again.
 */
export class GoodMemIndexingError extends GoodMemError {
  /** Every memory the call created. */
  readonly memoryIds: string[];
  /** Memories whose processing ended in FAILED. */
  readonly failedMemoryIds: string[];
  /** Memories still pending when the wait ran out. */
  readonly pendingMemoryIds: string[];

  constructor(
    message: string,
    options: GoodMemErrorOptions & {
      memoryIds: string[];
      failedMemoryIds: string[];
      pendingMemoryIds: string[];
    }
  ) {
    super(message, options);
    this.name = 'GoodMemIndexingError';
    this.memoryIds = options.memoryIds;
    this.failedMemoryIds = options.failedMemoryIds;
    this.pendingMemoryIds = options.pendingMemoryIds;
  }

  static isInstance(error: unknown): error is GoodMemIndexingError {
    return GoodMemError.isInstance(error) && error.name === 'GoodMemIndexingError';
  }
}

/** Context used to phrase an SDK failure for the caller. */
export interface ErrorContext {
  baseUrl: string;
  timeoutMs: number;
  /** The option that set `timeoutMs`, named in timeout messages. */
  timeoutOption?: string;
  signal?: AbortSignal;
}

const TIMEOUT_MESSAGE = 'request timed out';
const MAX_DETAIL = 1000;

function causes(error: unknown): unknown[] {
  const chain: unknown[] = [];
  let current: unknown = error;
  while (current && chain.length < 6) {
    chain.push(current);
    current = (current as { cause?: unknown }).cause;
  }
  return chain;
}

function isTimeout(error: unknown): boolean {
  return causes(error).some(
    (e) => e instanceof Error && e.message === TIMEOUT_MESSAGE
  );
}

/** The server's own explanation, pulled out of a JSON error body. */
export function serverMessage(body: string | undefined): string {
  if (!body) return '';
  try {
    const parsed = JSON.parse(body) as Record<string, unknown>;
    if (typeof parsed.error === 'string' && parsed.error) return parsed.error;
    if (Array.isArray(parsed.errors)) {
      const parts = parsed.errors
        .map((e) => {
          const item = (e ?? {}) as Record<string, unknown>;
          const field = typeof item.field === 'string' ? `${item.field}: ` : '';
          return typeof item.message === 'string' ? `${field}${item.message}` : '';
        })
        .filter(Boolean);
      if (parts.length) return parts.join('; ');
    }
    if (typeof parsed.message === 'string' && parsed.message) return parsed.message;
  } catch {
    // not JSON: fall through to the raw text
  }
  const text = body.trim();
  return text.length > MAX_DETAIL ? `${text.slice(0, MAX_DETAIL)}...` : text;
}

function lowLevelReason(error: unknown): string {
  for (const e of causes(error).slice(1)) {
    const code = (e as { code?: unknown }).code;
    if (typeof code === 'string' && code) return code;
  }
  for (const e of causes(error).slice(1)) {
    if (e instanceof Error && e.message) return e.message;
  }
  return '';
}

/**
 * Turn anything the SDK threw into a `GoodMemError` that says what was being
 * done, what the server said, and -- for timeouts and unreachable servers --
 * what to check. An abort requested by the caller is rethrown unchanged, so
 * the AI SDK and other callers still recognise it as an abort.
 */
export function wrapError(error: unknown, what: string, context: ErrorContext): unknown {
  if (GoodMemError.isInstance(error)) return error;
  if (context.signal?.aborted) return context.signal.reason ?? error;

  const e = error as { statusCode?: unknown; body?: unknown; message?: unknown; name?: unknown };
  if (typeof e?.statusCode === 'number') {
    const body = typeof e.body === 'string' ? e.body : undefined;
    const detail = serverMessage(body);
    return new GoodMemError(
      `${what} failed: GoodMem answered HTTP ${e.statusCode}${detail ? `: ${detail}` : ''}`,
      { statusCode: e.statusCode, body, cause: error, isRetryable: retryableStatus(e.statusCode) }
    );
  }
  if (isTimeout(error)) {
    return new GoodMemError(
      `${what} timed out after ${context.timeoutMs} ms: the GoodMem server at ${context.baseUrl} ` +
        `did not answer in time. Check that the server is healthy, or raise ${context.timeoutOption ?? 'timeoutMs'}.`,
      { timedOut: true, cause: error, isRetryable: true }
    );
  }
  const message = typeof e?.message === 'string' ? e.message : String(error);
  if (e?.name === 'NetworkError') {
    const reason = lowLevelReason(error);
    return new GoodMemError(
      `${what} failed: could not reach the GoodMem server at ${context.baseUrl}` +
        `${reason ? ` (${reason})` : ''}. Check baseUrl and that the server is running.`,
      { cause: error, isRetryable: true }
    );
  }
  // A 2xx response whose body was empty or not valid NDJSON/JSON: a broken
  // response, which a retry may not repeat.
  const brokenResponse = e?.name === 'ParseError';
  return new GoodMemError(`${what} failed: ${message}`, { cause: error, isRetryable: brokenResponse });
}

/** HTTP statuses that report availability rather than a mistake in the request. */
function retryableStatus(status: number): boolean {
  return status >= 500 || status === 429 || status === 408;
}
