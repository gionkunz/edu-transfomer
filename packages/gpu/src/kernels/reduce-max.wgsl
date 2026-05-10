// Row-wise max of row-major f32 matrix [rows, cols] -> f32 vector [rows].
// Accumulator is initialized from input[base] (first element of the row) since
// the dispatch helper rejects cols <= 0 — Tint rejects bitcast<f32>(0xff800000u)
// because -inf is not representable as a literal, so first-element init is the
// pragmatic equivalent of "start at the smallest possible row value".
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
  var acc: f32 = input[base];
  for (var j = 1u; j < cols; j++) {
    acc = max(acc, input[base + j]);
  }
  out[i] = acc;
}
