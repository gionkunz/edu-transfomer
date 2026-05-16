struct ScaleParams {
  element_count: u32,
  scale: f32,
  _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> scale_params: ScaleParams;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let word_idx = gid.x;
  let i0 = word_idx << 1u;
  if (i0 >= scale_params.element_count) {
    return;
  }
  let pair = load_bf16_pair(&a, word_idx);
  store_bf16_pair(&out, word_idx, pair * scale_params.scale);
}
