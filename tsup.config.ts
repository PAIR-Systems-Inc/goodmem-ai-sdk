import { defineConfig } from 'tsup';

// One bundled entry per format. The reference plugin compiled every source
// file separately (`bundle: false`), which left its ESM build importing the
// CommonJS copies of its own modules; bundling keeps each format
// self-contained. `ai`, `zod` and `@pairsystems/goodmem` stay external.
export default defineConfig({
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  shims: false,
  outDir: 'lib',
  entry: ['src/index.ts'],
  bundle: true,
  treeshake: false,
  target: 'es2022',
});
