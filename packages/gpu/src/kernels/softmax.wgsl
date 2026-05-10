// Numerically stable row-wise softmax of row-major f32 matrix [rows, cols].
// Each thread owns one row and runs the textbook two-pass algorithm:
//   1. m = max(row)
//   2. for each j: out[j] = exp(row[j] - m); accumulate s += out[j]
//   3. for each j: out[j] /= s
// Subtracting the per-row maximum before exp() prevents overflow on rows with
// large-magnitude values (e.g. attention scores at 1000+). See
// docs/project.md "Numerical Stability" — Softmax overflow row.
struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = i * cols;

  var m: f32 = input[base];
  for (var j = 1u; j < cols; j++) {
    m = max(m, input[base + j]);
  }

  var s: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let e = exp(input[base + j] - m);
    out[base + j] = e;
    s += e;
  }

  for (var j = 0u; j < cols; j++) {
    out[base + j] = out[base + j] / s;
  }
}
