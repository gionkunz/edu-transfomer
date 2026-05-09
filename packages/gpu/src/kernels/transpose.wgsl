// Row-major f32 matrix src with shape rows×cols; dst is cols×rows (transpose).
struct TransposeDims {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: TransposeDims;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

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
  dst[out_idx] = src[idx];
}
