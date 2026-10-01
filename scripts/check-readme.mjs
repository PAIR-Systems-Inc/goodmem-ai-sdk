#!/usr/bin/env node
/**
 * Mechanical README check.
 *
 * 1. Every ```ts block in README.md is type-checked, as a consumer would
 *    write it, against the package -- `--against lib` uses the built types
 *    (lib/index.d.mts), `--against src` the sources.
 * 2. Every identifier in an inline code span (`goodmemTools`, `topK`,
 *    `providerMetadata.goodmem`, ...) must exist in the built types, in the
 *    installed `ai` or `@pairsystems/goodmem` types (server status codes
 *    come from the latter), or on a short, printed allowlist of names that
 *    are not code (environment variables). Skipped with `--compile-only`.
 * 3. The model-facing tool names must be documented.
 *
 * Exits non-zero on any failure, printing what failed.
 */

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const against = args.includes('--against') ? args[args.indexOf('--against') + 1] : 'lib';
const compileOnly = args.includes('--compile-only');
if (!['lib', 'src'].includes(against)) {
  console.error('usage: check-readme.mjs --against lib|src [--compile-only]');
  process.exit(2);
}

const target = against === 'lib' ? join(ROOT, 'lib', 'index.d.mts') : join(ROOT, 'src', 'index.ts');
if (!existsSync(target)) {
  console.error(`${target} does not exist${against === 'lib' ? ' -- run npm run build first' : ''}.`);
  process.exit(2);
}

const readme = readFileSync(join(ROOT, 'README.md'), 'utf8');
const failures = [];

// ---- 1. compile every TypeScript block ------------------------------------
const blocks = [...readme.matchAll(/^```(ts|typescript)\n([\s\S]*?)^```$/gm)].map((m) => m[2]);
const work = join(ROOT, '.readme-check');
rmSync(work, { recursive: true, force: true });
mkdirSync(work, { recursive: true });
try {
  blocks.forEach((code, i) => writeFileSync(join(work, `example-${i + 1}.ts`), `export {};\n${code}`));
  // Provider packages are not dependencies of this repo; the examples only
  // need a function that returns a model.
  writeFileSync(
    join(work, 'providers.d.ts'),
    "declare module '@ai-sdk/openai' {\n" +
      "  export function openai(modelId: string): import('@ai-sdk/provider').LanguageModelV3;\n" +
      '}\n'
  );
  writeFileSync(
    join(work, 'tsconfig.json'),
    JSON.stringify(
      {
        compilerOptions: {
          module: 'esnext',
          moduleResolution: 'bundler',
          target: 'es2022',
          lib: ['es2022', 'DOM'],
          strict: true,
          noEmit: true,
          skipLibCheck: true,
          types: ['node'],
          typeRoots: [join(ROOT, 'node_modules', '@types')],
          paths: { '@pairsystems/goodmem-vercel-ai-sdk': [target] },
        },
        include: ['*.ts'],
      },
      null,
      2
    )
  );
  const tsc = spawnSync(process.execPath, [join(ROOT, 'node_modules', 'typescript', 'bin', 'tsc'), '-p', work], {
    encoding: 'utf8',
  });
  if (tsc.status !== 0) {
    failures.push(`README examples do not type-check against ${against}:\n${tsc.stdout}${tsc.stderr}`);
  } else {
    console.log(`ok - ${blocks.length} TypeScript example(s) type-check against ${against}`);
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}

// ---- 2. inline identifiers exist ------------------------------------------
if (!compileOnly) {
  const ours = readFileSync(join(ROOT, 'lib', 'index.d.ts'), 'utf8');
  const aiTypes = readFileSync(join(ROOT, 'node_modules', 'ai', 'dist', 'index.d.ts'), 'utf8');
  const sdkTypes = readFileSync(join(ROOT, 'node_modules', '@pairsystems', 'goodmem', 'dist', 'index.d.ts'), 'utf8');
  const ALLOW = new Set([
    // environment variables read by the live test suite, not by the package
    'GOODMEM_API_KEY', 'GOODMEM_BASE_URL', 'GOODMEM_TEST_EMBEDDER_ID', 'GOODMEM_TEST_RERANKER_ID',
    'GOODMEM_TEST_LATENCY_EMBEDDER_ID', 'GOODMEM_TEARDOWN_REPORT',
  ]);
  const prose = readme.replace(/^```[\s\S]*?^```$/gm, '');
  const spans = [...prose.matchAll(/`([^`\n]+)`/g)].map((m) => m[1].trim());
  const identifier = /^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*(\(\))?$/;
  const found = { ours: new Set(), ai: new Set(), sdk: new Set(), allowed: new Set(), missing: new Set() };
  const has = (text, word) => new RegExp(`(^|[^\\w$])${word.replace(/\$/g, '\\$')}([^\\w$]|$)`).test(text);
  for (const span of spans) {
    if (!identifier.test(span)) continue;
    for (const word of span.replace(/\(\)$/, '').split('.')) {
      if (has(ours, word)) found.ours.add(word);
      else if (has(aiTypes, word)) found.ai.add(word);
      else if (has(sdkTypes, word)) found.sdk.add(word);
      else if (ALLOW.has(word)) found.allowed.add(word);
      else found.missing.add(word);
    }
  }
  console.log(`ok - ${found.ours.size} inline identifier(s) found in lib/index.d.ts: ${[...found.ours].sort().join(', ')}`);
  console.log(`ok - ${found.ai.size} found in the ai package types: ${[...found.ai].sort().join(', ')}`);
  console.log(`ok - ${found.sdk.size} found in the @pairsystems/goodmem types: ${[...found.sdk].sort().join(', ')}`);
  console.log(`ok - ${found.allowed.size} allowlisted (not code): ${[...found.allowed].sort().join(', ')}`);
  if (found.missing.size) {
    failures.push(`README names identifier(s) that exist nowhere in the built types: ${[...found.missing].sort().join(', ')}`);
  }

  // ---- 3. model-facing names ----------------------------------------------
  for (const name of ['searchMemories', 'addMemory']) {
    if (!readme.includes(`\`${name}\``)) failures.push(`README does not document the model-facing tool name ${name}`);
  }
  if (!failures.length) console.log('ok - model-facing tool names searchMemories and addMemory are documented');
}

if (failures.length) {
  console.error(`\n${failures.join('\n\n')}`);
  process.exit(1);
}
