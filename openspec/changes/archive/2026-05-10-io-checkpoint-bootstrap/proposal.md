## Why

Inference and training need a deterministic on-disk checkpoint format (`model.bin` + `model.json` style sidecar per `docs/project.md`). `packages/io` should own serialization of packed `f32` blobs, JSON manifests, config loading helpers, and training text discovery via `glob`, without coupling to WebGPU tensors so the layer stays usable even when GPUs are unavailable (CI, pure file tests).

## What Changes

- Implement **`checkpoint-format.ts`**: types for manifest entries (`offset`, `shape`, `dtype: "f32"`), lexical-order packing of contiguous `Float32Array` blobs, validators, and filesystem helpers (`writePackedCheckpointFiles`, `readPackedCheckpointFiles`).
- Implement **`config-loader.ts`**: `readJsonFile` / typed parse hooks plus runtime validation turning unknown JSON into a validated checkpoint sidecar.
- Implement **`glob-loader.ts`**: async glob expansion for corpus patterns (`llm train --data` style) with sorted, deduped paths.
- Add **`glob`** runtime dependency under `packages/io` (explicitly allowed in `docs/project.md`).
- Replace the stub **`packages/io`** surface with Vitest-backed specs using small temp dirs; keep **`IO_LAYER_MARKER`** exported for `@llm-transformer-diy/training` stub layering.

## Capabilities

### New Capabilities

- `io-checkpoint-bootstrap`: binary + JSON checkpoint read/write aligned with Component 9 in `docs/project.md`; JSON schema shape (`config`, `tensors`, optional `metadata`); minimal glob-based file discovery helpers.

### Modified Capabilities

- _(none in other packages — training continues to consume only `IO_LAYER_MARKER` until checkpoints are wired in.)_

## Impact

- **Primary**: `packages/io` (+ new OpenSpec artifact tree under this change folder).
- **Dependencies**: Adds `glob` (utility allowlist entry). No `@llm-transformer-diy/tensor` edge — blobs are plain `Float32Array`/`Buffer`/`Uint8Array`; upper layers attach GPU uploads later.
