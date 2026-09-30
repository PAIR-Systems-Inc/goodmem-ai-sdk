/**
 * The only place this package reads the environment.
 *
 * `apiKey` and `baseUrl` fall back to `GOODMEM_API_KEY` and
 * `GOODMEM_BASE_URL`, each only when the caller omitted that option: an
 * explicit value always wins, and no other variable is read. Runtimes without
 * a Node-style environment (browsers, some edge runtimes) or that refuse
 * access to it (Deno without --allow-env) simply get no fallback.
 */

/** Environment variable holding the default API key. */
export const API_KEY_ENV = 'GOODMEM_API_KEY';
/** Environment variable holding the default server URL. */
export const BASE_URL_ENV = 'GOODMEM_BASE_URL';

/** Where to get a server and a key, for error messages. */
export const GET_STARTED =
  'Get an instance URL and API key from GoodMem Cloud at https://cloud.goodmem.ai/login ' +
  '(free 14-day trial), or self-host: https://docs.goodmem.ai';

/**
 * The two fallbacks, read afresh on each call. Self-contained on purpose (no
 * helpers), so the tests can run this exact function where `process` is
 * missing or refuses access.
 */
export function environmentDefaults(): { apiKey?: string; baseUrl?: string } {
  if (typeof process === 'undefined') return {};
  let apiKey: unknown;
  let baseUrl: unknown;
  try {
    apiKey = process.env.GOODMEM_API_KEY;
    baseUrl = process.env.GOODMEM_BASE_URL;
  } catch {
    return {};
  }
  const found: { apiKey?: string; baseUrl?: string } = {};
  if (typeof apiKey === 'string' && apiKey.trim()) found.apiKey = apiKey.trim();
  if (typeof baseUrl === 'string' && baseUrl.trim()) found.baseUrl = baseUrl.trim();
  return found;
}
