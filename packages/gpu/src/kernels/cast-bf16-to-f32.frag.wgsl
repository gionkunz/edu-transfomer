struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// One thread per packed bf16 pair (one `u32` of `inp`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  let n = params.element_count;
  if (i0 >= n) {
    return;
  }
  let pair = load_bf16_pair(&inp, word_idx);
  out[i0] = pair.x;
  if (i0 + 1u < n) {
    out[i0 + 1u] = pair.y;
  }
}
