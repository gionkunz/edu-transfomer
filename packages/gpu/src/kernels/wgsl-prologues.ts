/**
 * 16-bit kernel WGSL uses one storage layout, prepended from disk before the per-kernel
 * `*.frag.wgsl` body by {@link ./load-core-wgsl.ts}:
 *
 * - **bf16** — `shared-bf16.wgsl` (`array<u32>`, two packed bf16 scalars per `u32` word).
 *   Race-free writes rely on "one thread per pair" dispatch — see the doc comment in
 *   `shared-bf16.wgsl` and `softmax.bf16.frag.wgsl` for the disjoint-word invariant.
 *
 * `fp16` storage was removed: bf16 matches fp16's memory and bandwidth wins while keeping
 * f32's exponent range (no loss scaling needed during training). See
 * `docs/project.md` → Numerical Stability for the rationale.
 */

export {};
