struct MatMulDims {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: MatMulDims;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;
  let total = M * N;
  if (idx >= total) {
    return;
  }
  let row = idx / N;
  let col = idx % N;
  var sum = 0.0;
  for (var k = 0u; k < K; k++) {
    let a = load_f16n(&A, row * K + k);
    let b = load_f16n(&B, k * N + col);
    sum += a * b;
  }
  store_f16n(&C, idx, sum);
}
