const GELU_TANH_COEF: f32 = 0.7978845608028654;
const GELU_CUBE_COEF: f32 = 0.044715;
const GELU_LINEAR_THRESHOLD: f32 = 8.0;

fn gelu_grad_f32(x: f32) -> f32 {
  if (x > GELU_LINEAR_THRESHOLD) {
    return 1.0;
  }
  if (x < -GELU_LINEAR_THRESHOLD) {
    return 0.0;
  }
  let u = GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x);
  let t = tanh(u);
  let dtanh = 1.0 - t * t;
  let dudx = GELU_TANH_COEF * (1.0 + 3.0 * GELU_CUBE_COEF * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * dtanh * dudx;
}

struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<u32>;
@group(0) @binding(2) var<storage, read> inp: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  if (i0 >= params.element_count) {
    return;
  }
  let x = load_bf16_pair(&inp, word_idx);
  let g = load_bf16_pair(&grad_out, word_idx);
  let out = vec2<f32>(g.x * gelu_grad_f32(x.x), g.y * gelu_grad_f32(x.y));
  store_bf16_pair(&grad_in, word_idx, out);
}
