## 1. Checkpoint types + codecs

- [x] 1.1 Define exported TypeScript interfaces for **`TensorManifestEntry`** and **`PackedCheckpointSidecar`** matching `docs/project.md` manifest fields (`offset`, `shape`, `dtype: "f32"`) plus **`config`** / **`tensors`** / optional **`metadata`**.
- [x] 1.2 Implement **`packPackedCheckpoint`** (record of named `Float32Array` payloads + shapes) producing `{ bin: Uint8Array; sidecar: PackedCheckpointSidecar }` with deterministic lexical tensor ordering.
- [x] 1.3 Implement **`unpackPackedCheckpoint`** validating manifest vs binary blob and yielding `Record<string, Float32Array>` (copied views).
- [x] 1.4 Implement **`writePackedCheckpointFiles`** / **`readPackedCheckpointFiles`** atop `node:fs/promises` with recursive `mkdir` for parent dirs.

## 2. JSON + glob loaders

- [x] 1.5 Implement **`readJsonFile`** returning `unknown` and **`validatePackedCheckpointSidecar`** narrowing to `PackedCheckpointSidecar`.
- [x] 1.6 Implement **`discoverTrainingFiles`** wrapping the `glob` package (multi-pattern support, sorted deduped paths resolved via `cwd`).

## 3. Tests + verification

- [x] 1.7 Vitest specs for codecs (round-trip), validation errors (length mismatch / dtype / gaps), filesystem IO with temp dirs, and glob loader behavior.
- [x] 1.8 `pnpm exec nx run @llm-transformer-diy/io:build` and `:test`; fix regressions until green.
- [x] 1.9 Archive this folder to `openspec/changes/archive/2026-05-10-io-checkpoint-bootstrap/` and commit conventionally scoped to `@llm-transformer-diy/io` + archived OpenSpec tree.
