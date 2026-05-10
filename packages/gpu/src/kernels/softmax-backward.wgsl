// Jacobian–vector product for row-wise softmax w.r.t. logits.
// Given forward output probabilities `y` (row-wise softmax of logits) and
// upstream gradient `grad_out` w.r.t. those probabilities, per row i:
//   dot = Σ_j y[i,j] * grad_out[i,j]
//   grad_in[i,j] = y[i,j] * (grad_out[i,j] - dot)
//
// `grad_in` is the gradient w.r.t. the pre-softmax logits. Layout matches
// softmax.wgsl: row-major [rows, cols].
struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = i * cols;

  var dot: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let idx = base + j;
    dot += y[idx] * grad_out[idx];
  }

  for (var j = 0u; j < cols; j++) {
    let idx = base + j;
    grad_in[idx] = y[idx] * (grad_out[idx] - dot);
  }
}
