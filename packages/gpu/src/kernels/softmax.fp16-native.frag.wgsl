struct SoftmaxDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: SoftmaxDims;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = i * cols;

  var m: f32 = load_f16n(&input, base);
  for (var j = 1u; j < cols; j++) {
    m = max(m, load_f16n(&input, base + j));
  }

  var s: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let e = exp(load_f16n(&input, base + j) - m);
    store_f16n(&out, base + j, e);
    s += e;
  }

  for (var j = 0u; j < cols; j++) {
    let ev = load_f16n_rw(&out, base + j);
    store_f16n(&out, base + j, ev / s);
  }
}
