// Stable per-row softmax cross-entropy from logits alone:
// loss[i] = log(sum_j exp(logits[i,j] - m_i)) + m_i - logits[i, target_i]
// where m_i = max_j logits[i,j].
struct Dims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> losses: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let rows = dims.rows;
  let cols = dims.cols;
  if (row >= rows) {
    return;
  }
  let base = row * cols;
  let t = targets[row];

  var m = logits[base];
  for (var j = 1u; j < cols; j++) {
    m = max(m, logits[base + j]);
  }

  var sumexp: f32 = 0.0;
  for (var k = 0u; k < cols; k++) {
    sumexp += exp(logits[base + k] - m);
  }

  let logz = log(sumexp);
  losses[row] = logz + m - logits[base + t];
}
