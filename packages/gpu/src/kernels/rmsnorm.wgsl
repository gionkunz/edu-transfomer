// Row-wise RMSNorm of row-major f32 matrix [rows, cols] with learned scale
// gamma of length cols. For each row i:
//   meanSq = sum_j(input[i,j]^2) / cols
//   inv    = 1 / sqrt(meanSq + epsilon)
//   out[i,j] = input[i,j] * inv * gamma[j]
// Matches docs/project.md RMSNorm definition: y = x / sqrt(mean(x^2) + eps) * gamma.
struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = i * cols;

  var sumSq: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let v = input[base + j];
    sumSq += v * v;
  }
  let meanSq = sumSq / f32(cols);
  let inv = 1.0 / sqrt(meanSq + dims.epsilon);

  for (var j = 0u; j < cols; j++) {
    out[base + j] = input[base + j] * inv * gamma[j];
  }
}
