// ∂L/∂x for row-wise RMSNorm (matches forward in rmsnorm.wgsl).
// grad_input[row,j] = inv*gamma[j]*grad_out[row,j]
//                   - x[row,j] * inv^3 / cols * sum_k(x[row,k]*gamma[k]*grad_out[row,k])
struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> grad_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = row * cols;

  var sum_sq: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let v = input[base + j];
    sum_sq += v * v;
  }
  let mean_sq = sum_sq / f32(cols);
  let inv = 1.0 / sqrt(mean_sq + dims.epsilon);

  var sum_term: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let xv = input[base + j];
    sum_term += xv * gamma[j] * grad_out[base + j];
  }
  let coeff = inv * inv * inv / f32(cols);

  for (var j = 0u; j < cols; j++) {
    let xv = input[base + j];
    grad_input[base + j] = inv * gamma[j] * grad_out[base + j] - xv * coeff * sum_term;
  }
}
