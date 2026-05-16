struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> gamma: array<u32>;
@group(0) @binding(3) var<storage, read> grad_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<u32>;

fn rmsnorm_grad_input_row(base: u32, cols: u32, epsilon: f32) {
  var sum_sq: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let v = load_bf16(&input, base + j);
    sum_sq = sum_sq + v * v;
  }
  let inv = 1.0 / sqrt(sum_sq / f32(cols) + epsilon);

  var sum_term: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let xv = load_bf16(&input, base + j);
    sum_term = sum_term + xv * load_bf16(&gamma, j) * load_bf16(&grad_out, base + j);
  }
  let coeff = inv * inv * inv / f32(cols);

  for (var j = 0u; j < cols; j++) {
    let xv = load_bf16(&input, base + j);
    let gv = inv * load_bf16(&gamma, j) * load_bf16(&grad_out, base + j) - xv * coeff * sum_term;
    store_bf16(&grad_input, base + j, gv);
  }
}

// Pair-of-rows dispatch — see `softmax.bf16.frag.wgsl`.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pair_idx = gid.x;
  let i0 = pair_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;
  rmsnorm_grad_input_row(i0 * cols, cols, dims.epsilon);
  let i1 = i0 + 1u;
  if (i1 < rows) {
    rmsnorm_grad_input_row(i1 * cols, cols, dims.epsilon);
  }
}
