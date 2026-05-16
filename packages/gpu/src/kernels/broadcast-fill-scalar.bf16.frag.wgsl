struct Buf {
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> buf: Buf;
@group(0) @binding(1) var<storage, read> scalar: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

// One thread per packed pair (one `u32` of `out`).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  let n = buf.n;
  if (i0 >= n) {
    return;
  }
  let v = load_bf16(&scalar, 0u);
  let hi = select(0.0, v, i0 + 1u < n);
  store_bf16_pair(&out, word_idx, vec2<f32>(v, hi));
}
