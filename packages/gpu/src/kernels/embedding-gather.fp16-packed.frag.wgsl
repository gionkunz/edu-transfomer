struct GatherDims {
  num_tokens: u32,
  embed_dim: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: GatherDims;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  let ed = dims.embed_dim;
  let total = dims.num_tokens * ed;
  if (tid >= total) {
    return;
  }
  let token = tid / ed;
  let d = tid % ed;
  let vi = indices[token];
  let widx = vi * ed + d;
  let v = load_fp16p(&weights, widx);
  store_fp16p(&out, tid, v);
}
