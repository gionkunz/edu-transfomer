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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= dims.cols) {
    return;
  }
  let cols = dims.cols;
  let rows = dims.rows;
  var acc: f32 = 0.0;

  for (var i = 0u; i < rows; i++) {
    let base = i * cols;
    var sum_sq: f32 = 0.0;
    for (var k = 0u; k < cols; k++) {
      let v = load_bf16(&input, base + k);
      sum_sq += v * v;
    }
    let mean_sq = sum_sq / f32(cols);
    let inv = 1.0 / sqrt(mean_sq + dims.epsilon);
    let xv = load_bf16(&input, base + j);
    let go = load_bf16(&grad_out, base + j);
    acc += go * xv * inv;
  }

  store_bf16(&grad_gamma, j, acc);
}
