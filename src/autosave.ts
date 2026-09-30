/**
 * Automatic saving for the middleware: after a user turn, extract the durable
 * facts the user stated and store them in GoodMem.
 *
 * Extraction runs on the developer's own model, as an ordinary AI SDK
 * structured-output call, so its cost shows up in the developer's usage and
 * its behaviour can be inspected or replaced (`extractFacts`). Only the
 * latest user message is given to it: never the model's replies, its
 * reasoning or streamed deltas, and never the memories injected for the turn.
 */

import { generateText, Output } from 'ai';
import { z } from 'zod';

import type { Connection } from './config.js';
import { GoodMemError, GoodMemIngestionError, wrapError } from './errors.js';
import { add } from './memories.js';
import type { MemoryResult } from './results.js';

/** A model `generateText` accepts, used to extract facts. */
export type GoodmemExtractionModel = Parameters<typeof generateText>[0]['model'];

/** What a custom fact extractor receives. */
export interface GoodmemFactExtractionInput {
  /** The text of the latest user message. */
  text: string;
  /** Aborted when the call is aborted or the save runs out of time. */
  signal: AbortSignal;
}

/** Replaces the built-in, model-based extraction. Return the facts to store. */
export type GoodmemFactExtractor = (input: GoodmemFactExtractionInput) => Promise<string[]>;

/** Most facts stored from one turn. */
export const MAX_FACTS_PER_TURN = 10;
/** Longest fact stored, in characters. */
export const MAX_FACT_LENGTH = 500;

/**
 * The instructions given to the extraction model.
 *
 * They name JSON and spell out the exact output shape on purpose. For a
 * structured-output call, OpenAI-compatible providers send
 * `response_format: { type: 'json_object' }` and drop the schema, so the model
 * sees only these instructions; and OpenAI (and Azure, and OpenRouter in
 * front of them) reject `json_object` with HTTP 400 unless the messages
 * contain the word "json".
 */
export const FACT_EXTRACTION_INSTRUCTIONS = [
  'You pick out facts worth remembering from one message a user wrote to an assistant.',
  'Respond with only a JSON object of the form {"facts": ["...", "..."]}: a single key "facts" whose value is an array of strings,',
  'with no other keys and no text before or after the JSON.',
  'Put in the array only durable facts the user stated about themselves, their preferences, plans, relationships or their world,',
  'that would still be useful in a later conversation.',
  'Write each fact as a short standalone sentence in the third person, starting with "User",',
  'for example "User is vegetarian." or "User\'s daughter is called Lina."',
  'Do not include questions, requests, instructions to the assistant, greetings or small talk,',
  'and do not guess or add anything the user did not say.',
  `Return at most ${MAX_FACTS_PER_TURN} facts.`,
  'If there is nothing durable to remember, respond with {"facts": []}.',
].join(' ');

const FactsSchema = z.object({
  facts: z
    .array(z.string().max(MAX_FACT_LENGTH))
    .max(MAX_FACTS_PER_TURN)
    .describe('Durable facts the user stated, one short third-person sentence each; empty when there are none.'),
});

/** Token usage of the extraction call, as the AI SDK reports it. */
export interface GoodmemExtractionUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

/** What automatic saving did for one turn, reported on `providerMetadata.goodmem.saved`. */
export interface GoodmemSaveOutcome {
  /** The facts that were stored, in order. */
  facts: string[];
  /** The ids of the stored memories, in the same order. */
  memoryIds: string[];
  /** Extracted facts not stored because a memory retrieved for this turn already says the same. */
  duplicates: string[];
  /** Token usage of the extraction call, when a model did the extraction. */
  usage?: GoodmemExtractionUsage;
}

/** Why automatic saving failed for one turn, reported on `providerMetadata.goodmem.saveError`. */
export interface GoodmemSaveError {
  /** `extraction` when no facts could be extracted, `save` when storing them failed. */
  stage: 'extraction' | 'save';
  message: string;
  /** Facts extracted before a save failure. */
  facts?: string[];
  /** Memories stored before a save failure; they are not stored again. */
  memoryIds?: string[];
  usage?: GoodmemExtractionUsage;
}

export type TurnSave = { saved: GoodmemSaveOutcome } | { saveError: GoodmemSaveError };

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').replace(/[\s.!?]+$/, '').trim();
}

function usageOf(usage: unknown): GoodmemExtractionUsage | undefined {
  if (!usage || typeof usage !== 'object') return undefined;
  const u = usage as Record<string, unknown>;
  const out: GoodmemExtractionUsage = {};
  for (const key of ['inputTokens', 'outputTokens', 'totalTokens'] as const) {
    if (typeof u[key] === 'number') out[key] = u[key] as number;
  }
  return Object.keys(out).length ? out : undefined;
}

/** Extract facts with a model through the AI SDK's structured output. */
export async function extractWithModel(
  model: GoodmemExtractionModel,
  text: string,
  signal: AbortSignal
): Promise<{ facts: string[]; usage?: GoodmemExtractionUsage }> {
  const result = await generateText({
    model,
    system: FACT_EXTRACTION_INSTRUCTIONS,
    prompt: text,
    output: Output.object({
      schema: FactsSchema,
      name: 'memories',
      description: 'Facts about the user worth remembering.',
    }),
    abortSignal: signal,
    maxRetries: 1,
  });
  return { facts: result.output.facts, usage: usageOf(result.usage) };
}

/** Clean, cap and de-duplicate extracted facts, against each other and against this turn's memories. */
export function selectFacts(extracted: unknown, retrieved: MemoryResult[]): { facts: string[]; duplicates: string[] } {
  const known = new Set(retrieved.map((r) => normalize(r.text)));
  const seen = new Set<string>();
  const facts: string[] = [];
  const duplicates: string[] = [];
  for (const raw of Array.isArray(extracted) ? extracted : []) {
    if (typeof raw !== 'string') continue;
    const fact = raw.trim().slice(0, MAX_FACT_LENGTH);
    const key = normalize(fact);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    if (known.has(key)) duplicates.push(fact);
    else facts.push(fact);
    if (facts.length >= MAX_FACTS_PER_TURN) break;
  }
  return { facts, duplicates };
}

/** The result of the extraction phase: facts to store, or why there are none. */
export type Extraction =
  | { ok: true; facts: string[]; usage?: GoodmemExtractionUsage }
  | { ok: false; error: GoodmemSaveError };

/**
 * Phase one, started as soon as the turn begins so it runs alongside the
 * model call: extract the facts. Never rejects.
 */
export async function runExtraction(
  text: string,
  extract: (text: string, signal: AbortSignal) => Promise<{ facts: string[]; usage?: GoodmemExtractionUsage }>,
  signal: AbortSignal
): Promise<Extraction> {
  try {
    const extracted = await extract(text, signal);
    if (!extracted || !Array.isArray(extracted.facts)) {
      throw new Error('the extractor did not return a list of facts');
    }
    return { ok: true, facts: extracted.facts, usage: extracted.usage };
  } catch (error) {
    const deadline = signal.aborted && GoodMemError.isInstance(signal.reason) ? signal.reason : undefined;
    const reason = deadline
      ? deadline.message
      : signal.aborted
        ? 'the call was aborted'
        : error instanceof Error
          ? error.message
          : String(error);
    return { ok: false, error: { stage: 'extraction', message: `Extracting facts to remember failed: ${reason}` } };
  }
}

/**
 * Phase two, run only once the model call has succeeded, so a failed or
 * aborted turn stores nothing: de-duplicate against this turn's memories and
 * store what is left. Never rejects: every failure comes back as a
 * `saveError`, so a failed save cannot fail the chat.
 */
export async function storeFacts(
  conn: Connection,
  extraction: Extraction,
  retrieved: MemoryResult[],
  signal: AbortSignal
): Promise<TurnSave> {
  if (!extraction.ok) return { saveError: extraction.error };
  const { facts, duplicates } = selectFacts(extraction.facts, retrieved);
  const usage = extraction.usage;
  if (facts.length === 0) {
    return { saved: { facts: [], memoryIds: [], duplicates, ...(usage ? { usage } : {}) } };
  }
  try {
    const { memories } = await add(conn, facts, { signal });
    return { saved: { facts, memoryIds: memories.map((m) => m.memoryId), duplicates, ...(usage ? { usage } : {}) } };
  } catch (error) {
    const wrapped = GoodMemError.isInstance(error)
      ? error
      : GoodMemError.isInstance(signal.reason)
        ? signal.reason
        : (wrapError(error, 'Saving facts', conn.errorContext()) as Error);
    return {
      saveError: {
        stage: 'save',
        message: wrapped.message,
        facts,
        memoryIds: GoodMemIngestionError.isInstance(error) ? error.createdMemoryIds : [],
        ...(usage ? { usage } : {}),
      },
    };
  }
}
