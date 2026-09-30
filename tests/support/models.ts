/**
 * Scripted language models for driving the tools and the middleware through
 * the real `ai` package (`generateText`, `streamText`), using the AI SDK's
 * own `MockLanguageModelV3`, which both AI SDK 6 and 7 export.
 */

import { MockLanguageModelV3 } from 'ai/test';

export const USAGE = {
  inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 5, text: 5, reasoning: 0 },
};

export function textResult(text: string) {
  return { content: [{ type: 'text', text }], finishReason: { unified: 'stop', raw: undefined }, usage: USAGE, warnings: [] };
}

export function toolCall(toolName: string, input: unknown, toolCallId = 'call-1') {
  return {
    content: [{ type: 'tool-call', toolCallId, toolName, input: JSON.stringify(input) }],
    finishReason: { unified: 'tool-calls', raw: undefined },
    usage: USAGE,
    warnings: [],
  };
}

/** A mock model answering each step from a script; each step sees the prompt. */
export function scriptedModel(...steps: Array<(prompt: any[]) => unknown>) {
  let step = 0;
  return new MockLanguageModelV3({
    doGenerate: (async (options: any) => {
      const next = steps[Math.min(step, steps.length - 1)];
      step += 1;
      return next(options.prompt);
    }) as any,
  });
}

/** A mock model that streams "ok". */
export function streamingModel() {
  return new MockLanguageModelV3({
    doStream: (async () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: 'stream-start', warnings: [] });
          controller.enqueue({ type: 'text-start', id: 't' });
          controller.enqueue({ type: 'text-delta', id: 't', delta: 'ok' });
          controller.enqueue({ type: 'text-end', id: 't' });
          controller.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: USAGE });
          controller.close();
        },
      }),
    })) as any,
  });
}

/** Tool results the AI SDK passed back to the model, from the prompt it saw. */
export function toolResultsIn(prompt: any[]): any[] {
  return prompt
    .filter((m) => m.role === 'tool')
    .flatMap((m) => m.content)
    .filter((p: any) => p.type === 'tool-result');
}

/** All system text in a prompt the model received. */
export function systemText(prompt: any[]): string {
  return prompt.filter((m) => m.role === 'system').map((m) => m.content).join('\n');
}
