// bf16 packed storage: two bf16 scalars per `u32` word.
//   word bits  0..15  = element 2k     (low half)
//   word bits 16..31  = element 2k + 1 (high half)
// Buffer byte length is `ceil(N/2) * 4` (matches `bf16ElementsByteLength`).
//
// Concurrency invariants enforced by the dispatch layer / shaders:
//   * Ordinary kernels dispatch one thread per **pair** (each thread owns one whole word)
//     and use the `*_pair_*` helpers — no in-word RMW between threads, so no race.
//   * Loose reads (`load_bf16` by element index) are race-free with concurrent writers
//     **inside the same dispatch** only when the source buffer is in a `read` binding;
//     in `read_write` accumulator buffers, callers MUST go through the atomic
//     `atomicCompareExchangeWeak`-RMW helpers defined alongside the scatter kernel.

// --- pack / unpack a bf16 element <-> 16-bit RNE bit pattern ---

fn bf16_to_f32(bits16: u32) -> f32 {
  return bitcast<f32>((bits16 & 0xffffu) << 16u);
}

fn f32_to_bf16(v: f32) -> u32 {
  let bits = bitcast<u32>(v);
  // RNE: bias 0x8000 on the dropped mantissa half.
  return ((bits + 0x8000u) >> 16u) & 0xffffu;
}

// --- per-element load (returns f32). Safe for read-only / non-racy bindings. ---

fn load_bf16(p: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
  let w = (*p)[idx >> 1u];
  let half = (w >> ((idx & 1u) * 16u)) & 0xffffu;
  return bf16_to_f32(half);
}

fn load_bf16_rw(p: ptr<storage, array<u32>, read_write>, idx: u32) -> f32 {
  let w = (*p)[idx >> 1u];
  let half = (w >> ((idx & 1u) * 16u)) & 0xffffu;
  return bf16_to_f32(half);
}

// --- pair-aware load / store: one whole `u32` word per thread (race-free). ---

fn load_bf16_pair(p: ptr<storage, array<u32>, read>, word_idx: u32) -> vec2<f32> {
  let w = (*p)[word_idx];
  return vec2<f32>(bf16_to_f32(w & 0xffffu), bf16_to_f32((w >> 16u) & 0xffffu));
}

fn load_bf16_pair_rw(p: ptr<storage, array<u32>, read_write>, word_idx: u32) -> vec2<f32> {
  let w = (*p)[word_idx];
  return vec2<f32>(bf16_to_f32(w & 0xffffu), bf16_to_f32((w >> 16u) & 0xffffu));
}

fn store_bf16_pair(p: ptr<storage, array<u32>, read_write>, word_idx: u32, v: vec2<f32>) {
  (*p)[word_idx] = f32_to_bf16(v.x) | (f32_to_bf16(v.y) << 16u);
}

// --- single-element store (in-place RMW on the containing u32) ---
//
// RACE-UNSAFE between concurrent threads targeting the same u32 word. Callers MUST
// dispatch such that the issuing thread is the sole writer of every u32 word it
// touches across the lifetime of its store sequence. The canonical pattern is the
// "pair-of-rows" dispatch used by `softmax.*.frag.wgsl` / `rmsnorm.*.frag.wgsl`:
// thread `k` owns the contiguous element range `[2k·cols, (2k+2)·cols)`, which is
// always 2·cols elements long, starts on an even index, and ends on an even index,
// so the u32 words covered are disjoint between threads.
fn store_bf16(p: ptr<storage, array<u32>, read_write>, idx: u32, v: f32) {
  let word_idx = idx >> 1u;
  let half = idx & 1u;
  let new_bits = f32_to_bf16(v);
  let old_w = (*p)[word_idx];
  let preserve = select(old_w & 0xffff0000u, old_w & 0x0000ffffu, half == 1u);
  let injected = select(new_bits, new_bits << 16u, half == 1u);
  (*p)[word_idx] = preserve | injected;
}
