struct Buf {
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> buf: Buf;
@group(0) @binding(1) var<storage, read> scalar: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = buf.n;
  if (i >= n) {
    return;
  }
  let v = load_f16n(&scalar, 0u);
  store_f16n(&out, i, v);
}
