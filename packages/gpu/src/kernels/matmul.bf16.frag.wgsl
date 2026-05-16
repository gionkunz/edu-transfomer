struct MatMulDims {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: MatMulDims;
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read> B: array<u32>;
@group(0) @binding(3) var<storage, read_write> C: array<u32>;

// One thread per packed output pair (one `u32` of `C`). Each thread accumulates
// in f32 and rounds to bf16 once at the end (preserves the wider-accumulator
// promise from `docs/project.md`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let out_i0 = word_idx << 1u;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;
  let total = M * N;
  if (out_i0 >= total) {
    return;
  }
  let row0 = out_i0 / N;
  let col0 = out_i0 % N;

  var sum0: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k++) {
    sum0 = sum0 + load_bf16(&A, row0 * K + k) * load_bf16(&B, k * N + col0);
  }

  var sum1: f32 = 0.0;
  let out_i1 = out_i0 + 1u;
  if (out_i1 < total) {
    let row1 = out_i1 / N;
    let col1 = out_i1 % N;
    for (var k: u32 = 0u; k < K; k++) {
      sum1 = sum1 + load_bf16(&A, row1 * K + k) * load_bf16(&B, k * N + col1);
    }
  }
  store_bf16_pair(&C, word_idx, vec2<f32>(sum0, sum1));
}
