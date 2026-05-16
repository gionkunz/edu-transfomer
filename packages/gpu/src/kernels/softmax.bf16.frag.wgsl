struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn softmax_row(base: u32, cols: u32) {
  var m: f32 = load_bf16(&input, base);
  for (var j = 1u; j < cols; j++) {
    m = max(m, load_bf16(&input, base + j));
  }

  var s: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let e = exp(load_bf16(&input, base + j) - m);
    store_bf16(&out, base + j, e);
    s = s + e;
  }

  let inv = 1.0 / s;
  for (var j = 0u; j < cols; j++) {
    let ev = load_bf16_rw(&out, base + j);
    store_bf16(&out, base + j, ev * inv);
  }
}

// Pair-of-rows dispatch: thread `k` owns rows `2k` and `2k+1` so its stores cover the
// contiguous element range `[2k·cols, (2k+2)·cols)`. With the leading index even and
// the span size `2·cols` also even, the touched u32 words are disjoint across threads
// regardless of `cols`'s parity (see `store_bf16` invariant in `shared-bf16.wgsl`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  let i0 = pair_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;
  softmax_row(i0 * cols, cols);
  let i1 = i0 + 1u;
  if (i1 < rows) {
    softmax_row(i1 * cols, cols);
  }
}
