// Atomic bf16 (low 16 bits per u32) float accumulate into grad_weights[vocab × embed].
fn atomic_add_bf16_slot(addr: ptr<storage, atomic<u32>, read_write>, delta: f32) {
  var old_u = atomicLoad(addr);
  loop {
    let old_f = bitcast<f32>((old_u & 0xffffu) << 16u);
    let new_f = old_f + delta;
    let bits = bitcast<u32>(new_f);
    let new_h = ((bits + 0x8000u) >> 16u) & 0xffffu;
    let r = atomicCompareExchangeWeak(addr, old_u, new_h);
    if (r.exchanged) {
      break;
    }
    old_u = r.old_value;
  }
}

struct ScatterDims {
  num_tokens: u32,
  embed_dim: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> dims: ScatterDims;
@group(0) @binding(1) var<storage, read> grad_tokens: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_weights_atomic: array<atomic<u32>>;

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
  let g = load_bf16(&grad_tokens, tid);
  atomic_add_bf16_slot(&grad_weights_atomic[widx], g);
}
