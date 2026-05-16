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

// One thread per packed output pair (one `u32` of `out`). Each thread emits one
// row tile of 2 consecutive embedding dimensions — both sides of the pair come
// from the same token row, so `vi` is computed once per word.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let out_i0 = word_idx << 1u;
  let ed = dims.embed_dim;
  let total = dims.num_tokens * ed;
  if (out_i0 >= total) {
    return;
  }
  let token0 = out_i0 / ed;
  let d0 = out_i0 % ed;
  let vi0 = indices[token0];
  let lo = load_bf16(&weights, vi0 * ed + d0);

  var hi: f32 = 0.0;
  let out_i1 = out_i0 + 1u;
  if (out_i1 < total) {
    let token1 = out_i1 / ed;
    let d1 = out_i1 % ed;
    let vi1 = indices[token1];
    hi = load_bf16(&weights, vi1 * ed + d1);
  }
  store_bf16_pair(&out, word_idx, vec2<f32>(lo, hi));
}
