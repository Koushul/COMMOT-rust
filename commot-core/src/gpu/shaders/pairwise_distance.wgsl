struct Params {
    n: u32,
    dim: u32,
    output_squared: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> spatial: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_mat: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let n = params.n;
    let dim = params.dim;
    if (i >= n || j >= n) {
        return;
    }
    var s = 0.0;
    let base_i = i * dim;
    let base_j = j * dim;
    for (var k = 0u; k < dim; k++) {
        let a = spatial[base_i + k];
        let b = spatial[base_j + k];
        let d = a - b;
        s += d * d;
    }
    let idx = i * n + j;
    if (params.output_squared == 1u) {
        out_mat[idx] = s;
    } else {
        out_mat[idx] = sqrt(s);
    }
}
