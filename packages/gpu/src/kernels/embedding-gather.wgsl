// Gather token embeddings: weights [vocab_size, embed_dim] row-major flat,
// indices [num_tokens] u32 → out [num_tokens, embed_dim] row-major flat.
struct GatherDims {
  num_tokens: u32,
  embed_dim: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: GatherDims;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

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
  out[tid] = weights[widx];
}
