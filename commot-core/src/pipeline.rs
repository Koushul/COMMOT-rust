use crate::cot::{build_m_max_sp_from_spatial, cot_combine_sparse, cot_combine_sparse_from_m_max};
use crate::sparse::CooMatrix;
use std::collections::HashMap;

pub fn euclidean_distance_matrix(spatial: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..dim {
                let a = spatial[i * dim + k];
                let b = spatial[j * dim + k];
                s += (a - b).powi(2);
            }
            m[i * n + j] = s.sqrt();
        }
    }
    m
}

pub fn square_distance_matrix(m: &[f64], _n: usize) -> Vec<f64> {
    m.iter().map(|x| x * x).collect()
}

pub fn run_cot_combine(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    rho: f64,
    nitermax: usize,
    weights: [f64; 4],
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    cot_combine_sparse(
        s, d, a, m, cutoff, n_pos, n_pos, ns_s, ns_d, eps_p, None, None, rho, weights, nitermax,
        1e-8,
    )
}

pub fn run_cot_combine_streaming(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    spatial: &[f64],
    cutoff: &[f64],
    n_pos: usize,
    spatial_dim: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    rho: f64,
    nitermax: usize,
    weights: [f64; 4],
    cost_squared: bool,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let m_max_sp = build_m_max_sp_from_spatial(spatial, n_pos, spatial_dim, cutoff, cost_squared);
    cot_combine_sparse_from_m_max(
        s, d, a, cutoff, n_pos, n_pos, ns_s, ns_d, eps_p, None, None, rho, weights, nitermax, 1e-8,
        &m_max_sp,
    )
}
