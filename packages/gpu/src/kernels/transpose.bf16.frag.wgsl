struct TransposeDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: TransposeDims;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let m = dims.rows;
  let n = dims.cols;
  let total = m * n;
  if (idx >= total) {
    return;
  }
  let i = idx / n;
  let j = idx % n;
  let out_idx = j * m + i;
  let v = load_bf16(&src, idx);
  store_bf16(&dst, out_idx, v);
}
