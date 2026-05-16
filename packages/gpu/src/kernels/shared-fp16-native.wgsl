enable f16;

fn load_f16n(p: ptr<storage, array<f16>, read>, idx: u32) -> f32 {
  return f32((*p)[idx]);
}

fn load_f16n_rw(p: ptr<storage, array<f16>, read_write>, idx: u32) -> f32 {
  return f32((*p)[idx]);
}

fn store_f16n(p: ptr<storage, array<f16>, read_write>, idx: u32, v: f32) {
  (*p)[idx] = f16(v);
}
