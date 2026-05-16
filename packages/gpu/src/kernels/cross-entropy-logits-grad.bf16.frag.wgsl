struct Dims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> probs: array<u32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read> upstream: array<u32>;
@group(0) @binding(4) var<storage, read_write> grad_logits: array<u32>;

fn ce_grad_at(ix: u32, cols: u32) -> f32 {
  let row = ix / cols;
  let col = ix % cols;
  let tgt = targets[row];
  let u = load_bf16(&upstream, row);
  let p = load_bf16(&probs, ix);
  var g = u * p;
  if (col == tgt) {
    g = g - u;
  }
  return g;
}

// One thread per packed output pair (one `u32` of `grad_logits`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let ix0 = word_idx << 1u;
  let cols = dims.cols;
  let total = dims.rows * cols;
  if (ix0 >= total) {
    return;
  }
  let lo = ce_grad_at(ix0, cols);
  var hi: f32 = 0.0;
  let ix1 = ix0 + 1u;
  if (ix1 < total) {
    hi = ce_grad_at(ix1, cols);
  }
  store_bf16_pair(&grad_logits, word_idx, vec2<f32>(lo, hi));
}
