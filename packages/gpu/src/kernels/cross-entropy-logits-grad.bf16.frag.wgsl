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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let ix = gid.x;
  let rows = dims.rows;
  let cols = dims.cols;
  let total = rows * cols;
  if (ix >= total) {
    return;
  }
  let row = ix / cols;
  let col = ix % cols;
  let tgt = targets[row];
  let u = load_bf16(&upstream, row);
  let p = load_bf16(&probs, ix);
  var g = u * p;
  if (col == tgt) {
    g = g - u;
  }
  store_bf16(&grad_logits, ix, g);
}
