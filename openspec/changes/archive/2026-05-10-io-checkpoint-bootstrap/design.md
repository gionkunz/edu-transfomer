## Approach

Checkpoint sidecars mirror `docs/project.md`: top-level **`config`** (opaque JSON subtree), **`tensors`** map (manifest), optional **`metadata`**. Packed `.bin` is raw byte concatenation with **lexicographically sorted tensor names** so serialization is deterministic. Each manifest entry carries `dtype: "f32"` and `shape` validated as positive finite integers matching `Float32Array.length`.

Readers verify **coverage**: after sorting names, sequential `(offset → offset + nbytes)` intervals must lie within the binary slice, remain non-overlapping, and collectively span bytes `[0, bin.byteLength)` with no gaps (matches the canonical writer).

## Out of Scope

GPU buffer upload, tokenizer JSON, tokenizer merge rules persistence, gzip/sharding, versioning beyond optional `metadata` keys, protobuf, and automatic migration from hypothetical older formats.

## Testing

Vitest suites under `packages/io/src/**/*.spec.ts` cover pack/unpack symmetry, corrupted manifest rejection, filesystem round-trip in `os.tmpdir()`, and glob discovery over nested temp dirs.
