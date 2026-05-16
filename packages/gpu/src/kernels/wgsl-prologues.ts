/**
 * 16-bit kernel WGSL uses three storage layouts, each prepended from disk before the per-kernel
 * `*.frag.wgsl` body by {@link ./load-core-wgsl.ts}:
 *
 * - **bf16** — `shared-bf16.wgsl` (`array<u32>`, one bf16 in low 16 bits per index — avoids RMW races)
 * - **fp16 packed** — `shared-fp16-packed.wgsl` (`unpack2x16float` / `pack2x16float`, one f16 per u32)
 * - **fp16 native** — `shared-fp16-native.wgsl` (`enable f16`, `array<f16>`), compiled only when
 *   `GPUDevice.features` includes `shader-f16`.
 */

export {};
