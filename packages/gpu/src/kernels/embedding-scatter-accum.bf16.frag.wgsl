// Atomic bf16 (packed pair, 2 elements per u32) accumulate into grad_weights[vocab × embed].
// `half` selects which 16-bit lane within the word to RMW: 0 = low (bits  0..15),
// 1 = high (bits 16..31). The other lane is preserved across the CAS loop.
fn atomic_add_bf16_packed(addr: ptr<storage, atomic<u32>, read_write>, half: u32, delta: f32) {
  var old_u = atomicLoad(addr);
  loop {
    let lo_old = old_u & 0xffffu;
    let hi_old = (old_u >> 16u) & 0xffffu;
    var lo_new = lo_old;
    var hi_new = hi_old;
    if (half == 0u) {
      let old_f = bitcast<f32>(lo_old << 16u);
      let new_bits = bitcast<u32>(old_f + delta);
      lo_new = ((new_bits + 0x8000u) >> 16u) & 0xffffu;
    } else {
      let old_f = bitcast<f32>(hi_old << 16u);
      let new_bits = bitcast<u32>(old_f + delta);
      hi_new = ((new_bits + 0x8000u) >> 16u) & 0xffffu;
    }
    let new_u = lo_new | (hi_new << 16u);
    let r = atomicCompareExchangeWeak(addr, old_u, new_u);
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
  atomic_add_bf16_packed(&grad_weights_atomic[widx >> 1u], widx & 1u, g);
}
