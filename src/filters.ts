/**
 * Builds GoodMem metadata filter expressions safely.
 *
 * GoodMem filters are expressions evaluated server-side, not SQL.
 * Interpolating a value into one is both a filter-injection hole and a
 * correctness bug: an ordinary apostrophe produces a malformed expression.
 *
 * The escaping and casting rules below were verified live against GoodMem
 * (v1.0.320, and re-checked on v1.0.323 for this package):
 *
 * - a literal is single-quoted; `'` escapes as `\'` and a backslash as `\\`.
 *   SQL-style `''` doubling and double-quoted strings are rejected with
 *   HTTP 400.
 * - a raw newline inside a literal is rejected, so control characters are
 *   refused here rather than sent.
 * - `val()` yields JSON, so a comparison must cast to the stored type:
 *   TEXT for strings, NUMERIC for numbers, BOOLEAN for booleans. The cast has
 *   to match: comparing a boolean as TEXT is accepted with HTTP 200 and
 *   matches nothing.
 * - a field name containing a hyphen (`$.user-id`) is accepted with HTTP 200
 *   and matches nothing, so field names are limited to letters, digits and
 *   underscores, with dots for nested fields.
 */

import { GoodMemError } from './errors.js';

const SAFE_FIELD = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/;
// eslint-disable-next-line no-control-regex
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

/** A value a filter can compare against. */
export type FilterValue = string | number | boolean;

/** Thrown when a filter cannot be expressed safely. */
export class GoodMemFilterError extends GoodMemError {
  constructor(message: string) {
    super(message);
    this.name = 'GoodMemFilterError';
  }

  static isInstance(error: unknown): error is GoodMemFilterError {
    return GoodMemError.isInstance(error) && error.name === 'GoodMemFilterError';
  }
}

/** Quote a string as a GoodMem filter literal. */
export function escapeLiteral(value: string): string {
  if (typeof value !== 'string') {
    throw new GoodMemFilterError(`escapeLiteral() takes a string, not ${typeof value}.`);
  }
  if (CONTROL_CHARS.test(value)) {
    throw new GoodMemFilterError(
      'Filter values cannot contain control characters (including newlines ' +
        'and tabs); the server rejects them inside a literal.'
    );
  }
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

function checkField(field: string): string {
  if (typeof field !== 'string' || !SAFE_FIELD.test(field)) {
    throw new GoodMemFilterError(
      `Unsupported metadata field name ${JSON.stringify(field)}. Field names may ` +
        'contain letters, digits and underscores, must start with a letter or ' +
        'underscore, and may use dots for nested fields (a hyphenated name ' +
        'is accepted by the server but never matches).'
    );
  }
  return field;
}

function accessor(field: string, cast: string): string {
  return `CAST(val('$.${checkField(field)}') AS ${cast})`;
}

function render(value: unknown): [string, string] {
  // Booleans first: stringifying one produces a filter the server accepts
  // and that matches nothing.
  if (typeof value === 'boolean') return ['BOOLEAN', value ? 'true' : 'false'];
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new GoodMemFilterError('Filter numbers must be finite.');
    }
    return ['NUMERIC', String(value)];
  }
  if (typeof value === 'string') return ['TEXT', escapeLiteral(value)];
  throw new GoodMemFilterError(
    `Unsupported filter value type ${value === null ? 'null' : typeof value}; use a string, number or boolean.`
  );
}

/** Matches memories whose metadata `field` equals `value`. */
export function equals(field: string, value: FilterValue): string {
  const [cast, literal] = render(value);
  return `${accessor(field, cast)} = ${literal}`;
}

/** Matches memories whose metadata `field` differs from `value`. */
export function notEquals(field: string, value: FilterValue): string {
  const [cast, literal] = render(value);
  return `${accessor(field, cast)} != ${literal}`;
}

/** An ordering comparison against a numeric metadata field. */
export function compare(field: string, operator: '>' | '>=' | '<' | '<=', value: number): string {
  if (!['>', '>=', '<', '<='].includes(operator)) {
    throw new GoodMemFilterError(`Unsupported comparison operator ${JSON.stringify(operator)}.`);
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new GoodMemFilterError('Ordering comparisons apply to finite numbers only.');
  }
  return `${accessor(field, 'NUMERIC')} ${operator} ${value}`;
}

/** Matches memories whose metadata `field` is one of `values`. */
export function oneOf(field: string, values: readonly FilterValue[]): string {
  if (!Array.isArray(values) || values.length === 0) {
    throw new GoodMemFilterError('oneOf() needs at least one value.');
  }
  const casts = new Set<string>();
  const rendered = values.map((v) => {
    const [cast, literal] = render(v);
    casts.add(cast);
    return literal;
  });
  if (casts.size > 1) {
    throw new GoodMemFilterError('oneOf() values must all be of the same type.');
  }
  return `${accessor(field, [...casts][0])} IN (${rendered.join(', ')})`;
}

/** Combine filter expressions with `AND`. Empty parts are skipped. */
export function allOf(...expressions: Array<string | null | undefined>): string {
  const parts = expressions.filter((e): e is string => Boolean(e));
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return parts.map((p) => `(${p})`).join(' AND ');
}

/** Combine filter expressions with `OR`. Empty parts are skipped. */
export function anyOf(...expressions: Array<string | null | undefined>): string {
  const parts = expressions.filter((e): e is string => Boolean(e));
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return parts.map((p) => `(${p})`).join(' OR ');
}

/** Negate a filter expression. */
export function not(expression: string): string {
  if (typeof expression !== 'string' || !expression) {
    throw new GoodMemFilterError('not() needs a non-empty expression.');
  }
  return `NOT (${expression})`;
}

/** An `AND` of equality filters built from a plain object, keys sorted. */
export function fromMapping(metadata: Record<string, FilterValue> | undefined): string {
  if (!metadata) return '';
  const keys = Object.keys(metadata).sort();
  if (keys.length === 0) return '';
  return allOf(...keys.map((k) => equals(k, metadata[k])));
}
