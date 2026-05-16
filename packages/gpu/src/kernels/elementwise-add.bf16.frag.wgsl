struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

// One thread per packed pair (one `u32` of `out`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  if (i0 >= params.element_count) {
    return;
  }
  let av = load_bf16_pair(&a, word_idx);
  let bv = load_bf16_pair(&b, word_idx);
  store_bf16_pair(&out, word_idx, av + bv);
}
