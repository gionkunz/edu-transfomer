const GELU_TANH_COEF: f32 = 0.7978845608028654;
const GELU_CUBE_COEF: f32 = 0.044715;

fn gelu_f32(x: f32) -> f32 {
  let t = tanh(GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x));
  return 0.5 * x * (1.0 + t);
}

struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  if (i0 >= params.element_count) {
    return;
  }
  let x = load_bf16_pair(&inp, word_idx);
  store_bf16_pair(&out, word_idx, vec2<f32>(gelu_f32(x.x), gelu_f32(x.y)));
}
