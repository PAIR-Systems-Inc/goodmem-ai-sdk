/**
 * Configuration shared by every entry point, and the connection built from it.
 *
 * There is no default server. `apiKey` and `baseUrl` fall back to the
 * GOODMEM_API_KEY and GOODMEM_BASE_URL environment variables only when the
 * option is omitted (see env.ts, the one place the environment is read); an
 * explicit option always wins, so a configuration that names its server and
 * key can never be redirected by the environment.
 */

import { Goodmem } from '@pairsystems/goodmem';

import { API_KEY_ENV, BASE_URL_ENV, environmentDefaults, GET_STARTED } from './env.js';
import { GoodMemConfigError, GoodMemError, wrapError, type ErrorContext } from './errors.js';
import { allOf, fromMapping, type FilterValue } from './filters.js';

/** A space found by name, or created with this embedder if none exists. */
export interface GoodmemNamedSpace {
  /** Exact space name. */
  name: string;
  /**
   * The embedder the space must use. An existing space with this name is
   * reused only if it is indexed by this embedder; a space's embedder cannot
   * be changed after creation.
   */
  embedderId: string;
}

/** Where warnings about degraded retrievals are written. */
export interface GoodmemLogger {
  warn(message: string): void;
}

/** Configuration for every GoodMem entry point in this package. */
export interface GoodmemConfig {
  /**
   * The GoodMem API key. When omitted, the `GOODMEM_API_KEY` environment
   * variable is used; an explicit value always wins.
   */
  apiKey?: string;
  /**
   * The GoodMem server URL, e.g. `https://gm-<name>-<id>.app.goodmem.ai` or
   * `http://localhost:8080`. When omitted, the `GOODMEM_BASE_URL` environment
   * variable is used; an explicit value always wins. There is no default server.
   */
  baseUrl?: string;
  /** The space to read from and write to. Use one of `spaceId`, `spaceIds` or `space`. */
  spaceId?: string;
  /** Spaces to read from; writes go to the first. */
  spaceIds?: string[];
  /** A space found by exact name, or created with `embedderId` if missing. */
  space?: GoodmemNamedSpace;
  /** How many results a search returns by default. Defaults to 5; at most 100. */
  topK?: number;
  /**
   * Upper bound, in milliseconds, on every request this package makes,
   * including reading a retrieval stream to its end. Defaults to 30000.
   */
  timeoutMs?: number;
  /**
   * A metadata filter every retrieval must satisfy, applied server-side.
   * Build it with `filters` so values are escaped correctly.
   */
  filter?: string;
  /**
   * Metadata that scopes this configuration to a user, session or tenant:
   * written onto every memory it stores, and required (as equality filters)
   * on every retrieval it runs. This is a filter within the configured
   * spaces, not an access control; use separate spaces for hard isolation.
   */
  scope?: Record<string, FilterValue>;
  /** A reranker to apply to retrieval. */
  rerankerId?: string;
  /**
   * Drop results whose reranker score is below this value. Requires
   * `rerankerId`: reranker scales are provider-dependent and vector scores
   * are not on a 0-1 scale, so there is no default threshold.
   */
  minScore?: number;
  /**
   * A `fetch` implementation for the GoodMem SDK to use -- for a proxy, a
   * custom CA, or tests. TLS verification is never disabled by this package.
   */
  fetch?: typeof fetch;
  /** Where warnings are written. Defaults to `console`. */
  logger?: GoodmemLogger;
}

export const DEFAULT_TOP_K = 5;
export const MAX_TOP_K = 100;
export const DEFAULT_TIMEOUT_MS = 30_000;
export const MAX_TIMEOUT_MS = 2_147_483_647;
/** Most spaces a lookup by name will page through before giving up. */
export const MAX_SPACES_SCANNED = 10_000;

const CONFIG_KEYS = [
  'apiKey',
  'baseUrl',
  'spaceId',
  'spaceIds',
  'space',
  'topK',
  'timeoutMs',
  'filter',
  'scope',
  'rerankerId',
  'minScore',
  'fetch',
  'logger',
] as const;

/** A validated configuration bound to its own SDK client. */
export interface Connection {
  readonly client: Goodmem;
  readonly baseUrl: string;
  readonly timeoutMs: number;
  readonly topK: number;
  /** The scope and filter combined, or '' when neither is set. */
  readonly readFilter: string;
  readonly scope: Readonly<Record<string, FilterValue>>;
  readonly rerankerId?: string;
  readonly minScore?: number;
  readonly logger: GoodmemLogger;
  /** The spaces to search; the first is where writes go. */
  spaceIds(): Promise<string[]>;
  errorContext(signal?: AbortSignal): ErrorContext;
}

function fail(entry: string, message: string): never {
  throw new GoodMemConfigError(`${entry}: ${message}`);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

/** Cached lookups of named spaces, one per configuration object. */
const namedSpaces = new WeakMap<object, { key: string; promise: Promise<string> }>();

/**
 * Validate a configuration and build a connection for it.
 *
 * Every problem is reported up front, naming the option and what to pass
 * instead, rather than surfacing later as a confusing server error.
 */
export function connect(
  config: GoodmemConfig,
  entry: string,
  extraKeys: readonly string[] = []
): Connection {
  if (!config || typeof config !== 'object') {
    fail(entry, 'a configuration object is required, e.g. { spaceId }, with apiKey and baseUrl passed or set in the environment.');
  }
  const allowed = new Set<string>([...CONFIG_KEYS, ...extraKeys]);
  const unknown = Object.keys(config).filter((k) => !allowed.has(k));
  if (unknown.length) {
    fail(
      entry,
      `unknown option(s) ${unknown.map((k) => JSON.stringify(k)).join(', ')}. ` +
        `Valid options: ${[...allowed].join(', ')}.`
    );
  }

  // An omitted (undefined or null) option falls back to the environment; an
  // explicit value, even an unusable one, never does.
  const env = environmentDefaults();
  const omitted = (value: unknown) => value === undefined || value === null;
  const apiKey = omitted(config.apiKey) ? env.apiKey : config.apiKey;
  const baseUrl = omitted(config.baseUrl) ? env.baseUrl : config.baseUrl;
  const baseUrlSource = omitted(config.baseUrl) ? ` (from ${BASE_URL_ENV})` : '';

  if (apiKey === undefined) {
    fail(entry, `apiKey is required: pass apiKey, or set the ${API_KEY_ENV} environment variable. ${GET_STARTED}.`);
  }
  if (!nonEmptyString(apiKey)) {
    fail(entry, `apiKey is empty: pass the key your GoodMem server issued, or omit apiKey to use ${API_KEY_ENV}.`);
  }
  if (baseUrl === undefined) {
    fail(
      entry,
      `baseUrl is required: pass baseUrl, or set the ${BASE_URL_ENV} environment variable, e.g. ` +
        `https://gm-<name>-<id>.app.goodmem.ai or http://localhost:8080. There is no default server. ${GET_STARTED}.`
    );
  }
  if (!nonEmptyString(baseUrl)) {
    fail(entry, `baseUrl is empty: pass your GoodMem server's URL, or omit baseUrl to use ${BASE_URL_ENV}.`);
  }
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    fail(entry, `baseUrl ${JSON.stringify(baseUrl)}${baseUrlSource} is not a valid URL.`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    fail(entry, `baseUrl${baseUrlSource} must be an http:// or https:// URL, not ${JSON.stringify(baseUrl)}.`);
  }

  const given = (['spaceId', 'spaceIds', 'space'] as const).filter((k) => config[k] !== undefined);
  if (given.length !== 1) {
    fail(
      entry,
      given.length === 0
        ? 'a space is required: pass spaceId, spaceIds, or space: { name, embedderId }. The model never chooses a space.'
        : `pass only one of spaceId, spaceIds and space (got ${given.join(' and ')}).`
    );
  }
  let staticSpaceIds: string[] | undefined;
  if (config.spaceId !== undefined) {
    if (!nonEmptyString(config.spaceId)) fail(entry, 'spaceId must be a non-empty string.');
    staticSpaceIds = [config.spaceId];
  } else if (config.spaceIds !== undefined) {
    if (!Array.isArray(config.spaceIds) || config.spaceIds.length === 0) {
      fail(entry, 'spaceIds must be a non-empty array of space ids.');
    }
    if (!config.spaceIds.every(nonEmptyString)) fail(entry, 'every entry of spaceIds must be a non-empty string.');
    staticSpaceIds = [...new Set(config.spaceIds)];
  } else {
    const space = config.space as GoodmemNamedSpace;
    if (!space || typeof space !== 'object' || !nonEmptyString(space.name) || !nonEmptyString(space.embedderId)) {
      fail(entry, 'space must be { name, embedderId } with both set; the embedder is never picked for you.');
    }
    const extra = Object.keys(space).filter((k) => k !== 'name' && k !== 'embedderId');
    if (extra.length) fail(entry, `space accepts only name and embedderId (got ${extra.join(', ')}).`);
  }

  const topK = config.topK ?? DEFAULT_TOP_K;
  if (!Number.isInteger(topK) || topK < 1 || topK > MAX_TOP_K) {
    fail(entry, `topK must be an integer from 1 to ${MAX_TOP_K} (got ${String(config.topK)}).`);
  }
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (typeof timeoutMs !== 'number' || !Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > MAX_TIMEOUT_MS) {
    fail(
      entry,
      `timeoutMs must be a positive number of milliseconds (got ${String(config.timeoutMs)}). ` +
        'Every request is bounded; there is no way to turn the timeout off.'
    );
  }
  if (config.filter !== undefined && typeof config.filter !== 'string') {
    fail(entry, 'filter must be a string; build it with the filters helpers.');
  }
  if (config.scope !== undefined && (typeof config.scope !== 'object' || config.scope === null || Array.isArray(config.scope))) {
    fail(entry, 'scope must be an object of metadata values, e.g. { userId: "u-123" }.');
  }
  const scope = { ...(config.scope ?? {}) };
  let readFilter: string;
  try {
    readFilter = allOf(fromMapping(scope), config.filter?.trim());
  } catch (error) {
    fail(entry, `scope is not usable: ${(error as Error).message}`);
  }
  if (config.rerankerId !== undefined && !nonEmptyString(config.rerankerId)) {
    fail(entry, 'rerankerId must be a non-empty string.');
  }
  if (config.minScore !== undefined) {
    if (typeof config.minScore !== 'number' || !Number.isFinite(config.minScore)) {
      fail(entry, 'minScore must be a finite number.');
    }
    if (!config.rerankerId) {
      fail(
        entry,
        'minScore applies only to reranker scores, whose scale depends on the reranker; set rerankerId as ' +
          'well. Vector scores are not on a 0-1 scale, so a threshold on them is not meaningful.'
      );
    }
  }
  if (config.fetch !== undefined && typeof config.fetch !== 'function') {
    fail(entry, 'fetch must be a fetch-compatible function.');
  }
  if (config.logger !== undefined && typeof config.logger?.warn !== 'function') {
    fail(entry, 'logger must have a warn(message) method.');
  }

  const client = new Goodmem({
    baseUrl,
    apiKey,
    timeoutMs,
    ...(config.fetch ? { fetch: config.fetch } : {}),
  });
  const errorContext = (signal?: AbortSignal): ErrorContext => ({ baseUrl, timeoutMs, signal });

  let spaceIds: () => Promise<string[]>;
  if (staticSpaceIds) {
    const ids = staticSpaceIds;
    spaceIds = async () => [...ids];
  } else {
    const { name, embedderId } = config.space as GoodmemNamedSpace;
    const key = [apiKey, baseUrl, name, embedderId].join('\u0000');
    spaceIds = async () => {
      let cached = namedSpaces.get(config);
      if (!cached || cached.key !== key) {
        const promise = resolveNamedSpace(client, name, embedderId, errorContext());
        cached = { key, promise };
        namedSpaces.set(config, cached);
        promise.catch(() => {
          if (namedSpaces.get(config)?.promise === promise) namedSpaces.delete(config);
        });
      }
      return [await cached.promise];
    };
  }

  return {
    client,
    baseUrl,
    timeoutMs,
    topK,
    readFilter,
    scope,
    rerankerId: config.rerankerId,
    minScore: config.minScore,
    logger: config.logger ?? console,
    spaceIds,
    errorContext,
  };
}

function embedderIdsOf(space: { spaceEmbedders?: Array<{ embedderId?: string }> }): string[] {
  return (space.spaceEmbedders ?? []).map((e) => String(e?.embedderId ?? '')).filter(Boolean);
}

async function findByName(
  client: Goodmem,
  name: string,
  context: ErrorContext
): Promise<Array<{ spaceId: string; embedderIds: string[] }>> {
  // The server's nameFilter is a glob; an exact name without glob characters
  // narrows the listing to candidates. Names are compared exactly either way.
  const glob = /[*?[\]\\]/.test(name);
  const matches: Array<{ spaceId: string; embedderIds: string[] }> = [];
  let scanned = 0;
  try {
    const pages = await client.spaces.list({ maxResults: 1000, ...(glob ? {} : { nameFilter: name }) });
    for await (const space of pages) {
      scanned += 1;
      if (space.name === name) matches.push({ spaceId: String(space.spaceId), embedderIds: embedderIdsOf(space) });
      if (scanned >= MAX_SPACES_SCANNED) {
        throw new GoodMemError(
          `Looking up space ${JSON.stringify(name)} paged through ${MAX_SPACES_SCANNED} spaces without ` +
            'reaching the end of the listing; refusing to guess from a partial listing. Pass spaceId instead.'
        );
      }
    }
  } catch (error) {
    throw wrapError(error, `Looking up space ${JSON.stringify(name)}`, context);
  }
  return matches;
}

function pick(
  matches: Array<{ spaceId: string; embedderIds: string[] }>,
  name: string,
  embedderId: string
): string | undefined {
  if (matches.length > 1) {
    throw new GoodMemError(
      `${matches.length} spaces visible to this API key are named ${JSON.stringify(name)} ` +
        `(${matches.map((m) => m.spaceId).join(', ')}); refusing to guess which one was meant. Pass spaceId instead.`
    );
  }
  if (matches.length === 0) return undefined;
  const [match] = matches;
  if (!match.embedderIds.includes(embedderId)) {
    throw new GoodMemConfigError(
      `Space ${JSON.stringify(name)} (${match.spaceId}) already exists and is indexed by embedder(s) ` +
        `${JSON.stringify(match.embedderIds)}, not ${JSON.stringify(embedderId)}. A space's embedder cannot be ` +
        'changed after creation: use that embedder, choose another space name, or pass spaceId.'
    );
  }
  return match.spaceId;
}

/**
 * Find a space by exact name, or create it.
 *
 * The listing is followed to its end (the SDK detects a repeated page token),
 * so a match on a later page is not missed, and it is bounded so a runaway
 * listing fails loudly instead of being silently truncated. Reuse requires the
 * requested embedder; two spaces with the name are refused as ambiguous.
 */
export async function resolveNamedSpace(
  client: Goodmem,
  name: string,
  embedderId: string,
  context: ErrorContext
): Promise<string> {
  const existing = pick(await findByName(client, name, context), name, embedderId);
  if (existing) return existing;
  try {
    const created = await client.spaces.create({
      name,
      spaceEmbedders: [{ embedderId, defaultRetrievalWeight: 1.0 }],
    });
    return String(created.spaceId);
  } catch (error) {
    if ((error as { statusCode?: unknown })?.statusCode === 409) {
      // Created by someone else between the lookup and the create.
      const raced = pick(await findByName(client, name, context), name, embedderId);
      if (raced) return raced;
    }
    throw wrapError(error, `Creating space ${JSON.stringify(name)}`, context);
  }
}
