// Row-wise sum of row-major f32 matrix [rows, cols] -> f32 vector [rows].
// One thread per output row; sequential accumulation over cols. Naive on
// purpose — workgroup-shared parallel reduction is a separate perf change.
struct ReduceDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: ReduceDims;
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
  var acc: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    acc += input[base + j];
  }
  out[i] = acc;
}
