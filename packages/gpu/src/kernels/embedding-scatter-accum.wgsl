// Atomic float accumulate into grad_weights (dense [vocab, embed_dim] backing),
// keyed by duplicated indices across tokens — matches embedding gather backward.

fn atomic_add_f32(addr: ptr<storage, atomic<i32>, read_write>, v: f32) {
  var old_bits = atomicLoad(addr);
  loop {
    let old_f = bitcast<f32>(old_bits);
    let new_f = old_f + v;
    let new_bits = bitcast<i32>(new_f);
    let r = atomicCompareExchangeWeak(addr, old_bits, new_bits);
    if (r.exchanged) {
      break;
    }
    old_bits = r.old_value;
  }
}

struct ScatterDims {
  num_tokens: u32,
  embed_dim: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: ScatterDims;
@group(0) @binding(1) var<storage, read> grad_tokens: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_weights_atomic: array<atomic<i32>>;

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
  atomic_add_f32(&grad_weights_atomic[widx], grad_tokens[tid]);
}
