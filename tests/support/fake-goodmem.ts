/**
 * A fake GoodMem server for the offline suite.
 *
 * It is handed to the package as its `fetch`, so every request still goes
 * through the real @pairsystems/goodmem SDK -- URL building, headers, JSON
 * and NDJSON parsing, error mapping, timeouts -- and only the network is
 * replaced. Responses are the bytes captured from a live server
 * (`scripts/capture-fixtures.ts`), served with the status and content type
 * the server used.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const FIXTURES = join(__dirname, '..', 'fixtures');

interface ManifestEntry {
  method: string;
  path: string;
  status: number;
  contentType: string | null;
}

export const manifest: { serverVersion: string; capturedAt: string; files: Record<string, ManifestEntry> } =
  JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));

export function fixture(name: string): Buffer {
  return readFileSync(join(FIXTURES, name));
}

/** A captured fixture as the server sent it: same bytes, status and content type. */
export function replay(name: string, bytes: Uint8Array = fixture(name)): Response {
  const entry = manifest.files[name];
  if (!entry) throw new Error(`fixture ${name} is not in the manifest`);
  return new Response(Uint8Array.from(bytes), {
    status: entry.status,
    headers: { 'content-type': entry.contentType ?? 'application/octet-stream' },
  });
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

export interface CapturedRequest {
  method: string;
  url: URL;
  headers: Headers;
  body: any;
  rawBody: unknown;
  signal?: AbortSignal | null;
}

type Handler = (request: CapturedRequest) => Response | Promise<Response>;

export class FakeGoodmem {
  readonly requests: CapturedRequest[] = [];
  private readonly routes: Array<{ method: string; path: string | RegExp; handler: Handler }> = [];

  on(method: string, path: string | RegExp, handler: Handler | string): this {
    const h: Handler = typeof handler === 'string' ? () => replay(handler) : handler;
    this.routes.push({ method, path, handler: h });
    return this;
  }

  /** Requests for one method and path. */
  calls(method: string, path: string | RegExp): CapturedRequest[] {
    return this.requests.filter(
      (r) => r.method === method && (typeof path === 'string' ? r.url.pathname === path : path.test(r.url.pathname))
    );
  }

  readonly fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url);
    const method = (init?.method ?? 'GET').toUpperCase();
    let body: any;
    if (typeof init?.body === 'string') {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    } else if (init?.body instanceof FormData) {
      body = init.body;
    }
    const request: CapturedRequest = {
      method,
      url,
      headers: new Headers(init?.headers),
      body,
      rawBody: init?.body,
      signal: init?.signal,
    };
    this.requests.push(request);
    const route = this.routes.find(
      (r) => r.method === method && (typeof r.path === 'string' ? r.path === url.pathname : r.path.test(url.pathname))
    );
    if (!route) {
      return jsonResponse({ error: `no fake route for ${method} ${url.pathname}` }, 599);
    }
    return route.handler(request);
  }) as typeof fetch;
}

/** A response whose body never arrives, released only when the request is aborted. */
export function hang(request: CapturedRequest): Promise<Response> {
  return new Promise((_resolve, reject) => {
    const signal = request.signal;
    if (!signal) return; // never settles: the test fails by timing out, which is the point
    if (signal.aborted) return reject(signal.reason);
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
}
