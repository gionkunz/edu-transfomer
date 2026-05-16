struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> gamma: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

fn rmsnorm_row(base: u32, cols: u32, epsilon: f32) {
  var sumSq: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let v = load_bf16(&input, base + j);
    sumSq = sumSq + v * v;
  }
  let inv = 1.0 / sqrt(sumSq / f32(cols) + epsilon);
  for (var j = 0u; j < cols; j++) {
    let x = load_bf16(&input, base + j);
    let g = load_bf16(&gamma, j);
    store_bf16(&out, base + j, x * inv * g);
  }
}

// Pair-of-rows dispatch — see `softmax.bf16.frag.wgsl` for the disjoint-word invariant.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  let i0 = pair_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;
  rmsnorm_row(i0 * cols, cols, dims.epsilon);
  let i1 = i0 + 1u;
  if (i1 < rows) {
    rmsnorm_row(i1 * cols, cols, dims.epsilon);
  }
}
