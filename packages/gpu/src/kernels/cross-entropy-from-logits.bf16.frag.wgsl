struct Dims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> logits: array<u32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> losses: array<u32>;

fn ce_row(row: u32, cols: u32) -> f32 {
  let base = row * cols;
  let t = targets[row];
  var m = load_bf16(&logits, base);
  for (var j = 1u; j < cols; j++) {
    m = max(m, load_bf16(&logits, base + j));
  }
  var sumexp: f32 = 0.0;
  for (var k = 0u; k < cols; k++) {
    sumexp = sumexp + exp(load_bf16(&logits, base + k) - m);
  }
  return log(sumexp) + m - load_bf16(&logits, base + t);
}

// One thread per packed output pair (one `u32` of `losses`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;
  let lo = ce_row(i0, cols);
  var hi: f32 = 0.0;
  let i1 = i0 + 1u;
  if (i1 < rows) {
    hi = ce_row(i1, cols);
  }
  store_bf16_pair(&losses, word_idx, vec2<f32>(lo, hi));
}
