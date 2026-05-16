// Shared bf16 storage helpers: one logical bf16 per `array<u32>` entry (low 16 bits, high bits 0).
// Adjacent threads must not share a u32 word — packed 2×bf16 per u32 caused non-atomic RMW races.
fn load_bf16(p: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
  let h = (*p)[idx] & 0xffffu;
  return bitcast<f32>(h << 16u);
}

fn load_bf16_rw(p: ptr<storage, array<u32>, read_write>, idx: u32) -> f32 {
  let h = (*p)[idx] & 0xffffu;
  return bitcast<f32>(h << 16u);
}

fn store_bf16(p: ptr<storage, array<u32>, read_write>, idx: u32, v: f32) {
  let bits = bitcast<u32>(v);
  let h = (bits + 0x8000u) >> 16u;
  (*p)[idx] = h & 0xffffu;
}
