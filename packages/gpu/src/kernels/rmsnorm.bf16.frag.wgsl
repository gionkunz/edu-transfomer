struct RmsNormDims {
  rows: u32,
  cols: u32,
  epsilon: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: RmsNormDims;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> gamma: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= dims.rows) {
    return;
  }
  let cols = dims.cols;
  let base = i * cols;

  var sumSq: f32 = 0.0;
  for (var j = 0u; j < cols; j++) {
    let v = load_bf16(&input, base + j);
    sumSq += v * v;
  }
  let meanSq = sumSq / f32(cols);
  let inv = 1.0 / sqrt(meanSq + dims.epsilon);

  for (var j = 0u; j < cols; j++) {
    let x = load_bf16(&input, base + j);
    let g = load_bf16(&gamma, j);
    store_bf16(&out, base + j, x * inv * g);
  }
}
