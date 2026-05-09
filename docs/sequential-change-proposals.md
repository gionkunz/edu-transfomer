# Sequential change proposals (suggested order)

This document suggests how to slice the full implementation into **ordered OpenSpec changes**. It is guidance only—actual proposal names, scopes, and task lists should follow `.cursor/skills/openspec-propose` and stay consistent with `docs/project.md` (layering, dependency policy, GPU-first stance).

Canonical architecture order—each layer depends only on layers below it:

`gpu` → `tensor` → `autograd` → `model` → `training` / `inference` → `apps/cli`

Cross-cutting packages (`tokenizer`, `io`, `utils`) should depend **downward** only; wire them in when their consumers exist or when they have no upward dependency.

---

## Principles for sequencing

1. **Bottom-up delivery**: Land device/buffer/pipeline plumbing before tensor ops; land tensor ops before autograd; land autograd before the transformer stack.
2. **Kernel batches**: The GPU layer is large. Prefer **several smaller changes** (e.g. device + buffer pool + registry → core matmul/elementwise → attention-related kernels → optimizer/clip) rather than one mega-change, so each merge stays reviewable and testable.
3. **Prove the stack vertically early**: After a minimal matmul + a few elementwise ops exist on GPU tensors, add **one** end-to-end path (e.g. tiny `Linear` or two-op graph + backward smoke test) before filling the entire kernel table.
4. **Tokenizer and I/O early where cheap**: `tokenizer` (char-level first) and `io` (checkpoint format readers/writers) can progress in parallel with lower layers **as long as** they do not pull forbidden dependencies or shortcut compute.
5. **CLI last for “product”**: Reserve wolf-tui dashboards and full subcommands until `training` and `inference` expose stable programmatic APIs; keep CLI thin.

---

## Already completed anchor (example)

| Change (archived) | Role |
|-------------------|------|
| Monorepo skeleton / project foundation | pnpm + Nx, package placeholders, TS baseline, test/lint gates—**no** WGSL, ops, or model semantics. |

Treat anything listed in `docs/project.md` as the north star for what each later change must eventually satisfy.

---

## Suggested sequence (proposals)

The numbers are **execution order**, not package names. Several steps can be parallelized among different contributors once interfaces are clear (e.g. tokenizer vs GPU), but **merge order** should still respect downward dependencies.

### 1. GPU bootstrap — device, queues, buffers, kernel registry

- **Owns**: `packages/gpu` — adapter/device init, queue, `GPUBuffer` helpers, optional buffer pool, compute pipeline + bind group helpers, WGSL loading/compilation, **kernel registry** with deterministic naming.
- **Does not own yet**: full kernel inventory; tensor shape algebra.
- **Exit criteria**: one **trivial** compute shader (e.g. fill buffer or add two buffers) runs in CI-friendly conditions Node WebGPU allows; “no GPU” fails loudly per project policy.

### 2. Core WGSL kernels — matmul, elementwise, transpose

- **Owns**: `matmul`, elementwise (`add`, `mul`, `scale`, `gelu` as needed), `transpose` — enough to build dense layers and simple graphs.
- **Notes**: Start naive matmul if needed for velocity; follow with a **separate** optimization change (tiled/shared memory) to keep diffs readable.
- **Exit criteria**: microbenchmark-style tests or numerical checks vs CPU reference for small shapes.

### 3. Tensor layer — types, storage, dispatch

- **Owns**: `packages/tensor` — `Tensor` type, strides/shape, dtype, **GPU-resident** data, sync upload/download where required, routing from op API to registry kernels.
- **Does not own**: autograd; high-level model modules.
- **Exit criteria**: construct tensors, run matmul + a few elementwise ops, read back results for tests.

### 4. Reduction / softmax / normalization kernels (forward)

- **Owns**: stable softmax (two-pass), `reduce_max` / `reduce_sum`, `rmsnorm` forward, optional `causal_mask` as separate small change if it keeps reviews focused.
- **Depends on**: tensor dispatch patterns from step 3.

### 5. Autograd — graph, topo backward, parameters

- **Owns**: `packages/autograd` — `Variable`/operation nodes, topological backward, `Parameter` and registry (`zeroGrad`, etc.), ops matching **implemented** forward kernels first (`matmul`, `add`, `mul`, `gelu`, `softmax`, `reshape`/`transpose` views).
- **Strategy**: add backward coverage **pairwise** with forward kernels to avoid a giant “all gradients at once” change.

### 6. Embedding + cross-entropy (forward + backward)

- **Owns**: `embedding` gather/scatter grad, fused or staged `cross_entropy` with logits grads, tying into softmax from step 4.
- **Rationale**: unlocks a **real loss** for later training/inference without yet assembling a full GPT.

### 7. Optimizer and training utilities on GPU

- **Owns**: `adamw` fused step (or minimal CPU fallback only if explicitly allowed—prefer GPU per `docs/project.md`), `clip_norm`, and any remaining reducer kernels training needs.
- **Depends on**: autograd leaves with `.grad` populated; tensor API stable enough for in-place or fused updates.

### 8. Tokenizer package — char-level then BPE

- **Owns**: `packages/tokenizer` — interface from `docs/project.md`, char tokenizer first for pipeline testing, then BPE training/save/load.
- **Parallelism**: can start after step 1–2 if API is clear; must not introduce forbidden tokenizer dependencies.

### 9. I/O — checkpoint format + config loaders

- **Owns**: `packages/io` — binary tensor checkpoint + JSON sidecar, config loading, glob discovery as specified.
- **Pairs with**: step 7 for optimizer state if checkpoints include it.

### 10. Model — GPT blocks

- **Owns**: `packages/model` — `ModelConfig`, embeddings, **causal** multi-head attention, FFN, RMSNorm, residual stream, `TransformerBlock`, full GPT forward. Order inside this step: submodules with unit tests → composed block → full model.
- **Depends on**: autograd ops for every primitive used; masking kernel if not already landed.

### 11. Training loop

- **Owns**: `packages/training` — dataset batching, LR schedule, gradient clipping wiring, `Trainer`, checkpoint save/resume, logging hooks.
- **Exit criteria**: train a tiny model on toy data; loss decreases; checkpoint round-trip.

### 12. Inference — KV cache and sampling

- **Owns**: `packages/inference` — generation loop, greedy/temperature/top-k, KV cache per layer, reuse paths in model forward for cached decode.

### 13. CLI — wolf-tui and commands

- **Owns**: `apps/cli` — `tokenize`, `train`, `generate`, `inspect` per `docs/project.md`, wiring to packages only (no new ML logic in CLI).
- **Depends on**: stable CLIs from `training` + `inference` + `io` + `tokenizer`.

### 14. Hardening and performance passes

- **Examples**: tiled matmul, fused kernels where profiling shows bottlenecks, memory pool tuning, larger sequence-length testing, documentation nits **only if** tracked as their own small changes.

---

## Optional splits (when a step feels too large)

- **GPU**: “Device/registry” vs “numeric kernels” vs “optimizer/kernels for training only.”
- **Autograd**: “Core engine + primitive ops” vs “higher-level fused op nodes.”
- **Model**: “Single-head attention + tiny FFN smoke block” vs “full multi-layer GPT + config matrix.”
- **Training**: “Single-process toy trainer” vs “resume + full scheduler + production defaults.”

---

## How to use this with OpenSpec

1. For each row above, open **`openspec-propose`** with a concrete capability name and a tight **What changes / Explicitly out of scope** section (mirroring the archived foundation proposal).
2. After merge, use **`openspec-apply-change`** task lists; archive when stable per **`openspec-archive-change`**.
3. Keep **`docs/project.md`** authoritative for file layout and dependency allowlists; adjust this sequencing doc if the canonical architecture shifts.
