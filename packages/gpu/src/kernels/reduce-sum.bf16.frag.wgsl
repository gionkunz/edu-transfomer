struct ReduceDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: ReduceDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

// One thread per packed output pair (one `u32` of `out`); accumulators stay in f32.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  let rows = dims.rows;
  if (i0 >= rows) {
    return;
  }
  let cols = dims.cols;

  var acc0: f32 = 0.0;
  let base0 = i0 * cols;
  for (var j = 0u; j < cols; j++) {
    acc0 = acc0 + load_bf16(&input, base0 + j);
  }

  var acc1: f32 = 0.0;
  let i1 = i0 + 1u;
  if (i1 < rows) {
    let base1 = i1 * cols;
    for (var j = 0u; j < cols; j++) {
      acc1 = acc1 + load_bf16(&input, base1 + j);
    }
  }
  store_bf16_pair(&out, word_idx, vec2<f32>(acc0, acc1));
}
