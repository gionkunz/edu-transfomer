## Requirements

### REQ-CHK-001 — Deterministic packing

Packed checkpoints MUST serialize tensors in lexical name order without padding or alignment filler between tensors.

### REQ-CHK-002 — Manifest integrity

Readers MUST validate that every declared tensor slice fits entirely inside the binary payload, manifests use `dtype === "f32"`, offsets are multiples of four, contiguous segments cover `[0, byteLength)`, and summed element counts equal each `Float32Array` length requirement.

### REQ-CHK-003 — Neutral config payload

Checkpoint sidecars MUST keep `config` as parsed JSON compatible with downstream `packages/model`-defined structures in later changes; loaders MUST NOT silently coerce unknown keys beyond validation of the envelope fields.

### REQ-GLOB-001 — Pattern expansion

Glob helpers MUST dedupe overlapping matches from multiple patterns, return sorted deterministic paths suitable for deterministic dataset iteration, and default to sensible Node-friendly option flags (follow symlinks off unless callers override later).
