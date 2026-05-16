struct ScaleParams {
  element_count: u32,
  scale: f32,
  _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> scale_params: ScaleParams;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= scale_params.element_count) {
    return;
  }
  let av = load_f16n(&a, i);
  store_f16n(&out, i, av * scale_params.scale);
}
