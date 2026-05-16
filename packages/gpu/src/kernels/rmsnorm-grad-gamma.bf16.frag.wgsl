struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> grad_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<u32>;

fn grad_gamma_col(j: u32, cols: u32, rows: u32, epsilon: f32) -> f32 {
  var acc: f32 = 0.0;
  for (var i = 0u; i < rows; i++) {
    let base = i * cols;
    var sum_sq: f32 = 0.0;
    for (var k = 0u; k < cols; k++) {
      let v = load_bf16(&input, base + k);
      sum_sq = sum_sq + v * v;
    }
    let inv = 1.0 / sqrt(sum_sq / f32(cols) + epsilon);
    let xv = load_bf16(&input, base + j);
    let go = load_bf16(&grad_out, base + j);
    acc = acc + go * xv * inv;
  }
  return acc;
}

// One thread per packed output pair: each thread emits `grad_gamma[2k]` / `[2k+1]`
// as a single `u32` write. (Going through `store_bf16_pair` keeps the output writes
// race-free without relying on the single-element `store_bf16` invariant.)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let j0 = word_idx << 1u;
  let cols = dims.cols;
  if (j0 >= cols) {
    return;
  }
  let rows = dims.rows;
  let eps = dims.epsilon;
  let lo = grad_gamma_col(j0, cols, rows, eps);
  var hi: f32 = 0.0;
  let j1 = j0 + 1u;
  if (j1 < cols) {
    hi = grad_gamma_col(j1, cols, rows, eps);
  }
  store_bf16_pair(&grad_gamma, word_idx, vec2<f32>(lo, hi));
}
