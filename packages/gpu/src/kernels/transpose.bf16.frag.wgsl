struct TransposeDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: TransposeDims;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// One thread per destination pair (one `u32` of `dst`). Source reads are scattered
// element-wise, but reads from `read`-bound storage are race-free with concurrent
// writers in a different binding, so element-wise `load_bf16` is safe here.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let out_i0 = word_idx << 1u;
  let m = dims.rows;
  let n = dims.cols;
  let total = m * n;
  if (out_i0 >= total) {
    return;
  }
  // dst is row-major `[cols, rows]`; dst[out_i] = src[(out_i % rows) * cols + out_i / rows].
  let lo = load_bf16(&src, (out_i0 % m) * n + (out_i0 / m));
  var hi: f32 = 0.0;
  let out_i1 = out_i0 + 1u;
  if (out_i1 < total) {
    hi = load_bf16(&src, (out_i1 % m) * n + (out_i1 / m));
  }
  store_bf16_pair(&dst, word_idx, vec2<f32>(lo, hi));
}
