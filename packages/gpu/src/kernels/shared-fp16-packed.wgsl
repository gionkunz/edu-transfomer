// fp16: one logical IEEE binary16 per `array<u32>` (low lane only). Avoids RMW races from 2×f16 per u32.
fn load_fp16p(p: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
  return unpack2x16float((*p)[idx]).x;
}

fn load_fp16p_rw(p: ptr<storage, array<u32>, read_write>, idx: u32) -> f32 {
  return unpack2x16float((*p)[idx]).x;
}

fn store_fp16p(p: ptr<storage, array<u32>, read_write>, idx: u32, v: f32) {
  (*p)[idx] = pack2x16float(vec2<f32>(v, 0.0));
}
