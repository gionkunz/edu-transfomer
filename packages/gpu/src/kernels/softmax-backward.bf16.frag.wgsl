struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> grad_out: array<u32>;
@group(0) @binding(2) var<storage, read> y: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<u32>;

fn softmax_backward_row(base: u32, cols: u32) {
  var dot: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let idx = base + j;
    dot = dot + load_bf16(&y, idx) * load_bf16(&grad_out, idx);
  }
  for (var j = 0u; j < cols; j++) {
    let idx = base + j;
    let gv = load_bf16(&y, idx) * (load_bf16(&grad_out, idx) - dot);
    store_bf16(&grad_in, idx, gv);
  }
}

// Pair-of-rows dispatch (see `softmax.bf16.frag.wgsl`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  let i0 = pair_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;
  softmax_backward_row(i0 * cols, cols);
  let i1 = i0 + 1u;
  if (i1 < rows) {
    softmax_backward_row(i1 * cols, cols);
  }
}
