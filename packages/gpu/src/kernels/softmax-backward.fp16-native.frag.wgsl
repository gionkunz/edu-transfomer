enable f16;

struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> grad_out: array<f16>;
@group(0) @binding(2) var<storage, read> y: array<f16>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<f16>;

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
    dot += load_f16n(&y, idx) * load_f16n(&grad_out, idx);
  }

  for (var j = 0u; j < cols; j++) {
    let idx = base + j;
    let gv = load_f16n(&y, idx) * (load_f16n(&grad_out, idx) - dot);
    store_f16n(&grad_in, idx, gv);
  }
}
