use std::collections::HashMap;

use rayon::prelude::*;

use crate::sparse::{coo_expand_to_csr, coo_submatrix_pull, csr_get_block, CooMatrix, CsrMatrix};
use crate::unot::unot_sinkhorn_l1_sparse;

fn flatten_f_order(s: &[f64], n_pos: usize, n_spec: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_pos * n_spec];
    for j in 0..n_spec {
        for i in 0..n_pos {
            out[j * n_pos + i] = s[i * n_spec + j];
        }
    }
    out
}

fn nz_indices(v: &[f64]) -> Vec<usize> {
    v.iter()
        .enumerate()
        .filter(|(_, x)| **x > 0.0)
        .map(|(i, _)| i)
        .collect()
}

pub fn build_m_max_sp(m: &[f64], n_pos_s: usize, n_pos_d: usize, max_cutoff: f64) -> CooMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n_pos_s {
        for j in 0..n_pos_d {
            let v = m[i * n_pos_d + j];
            if v <= max_cutoff {
                rows.push(i);
                cols.push(j);
                data.push(v);
            }
        }
    }
    CooMatrix::new(rows, cols, data, n_pos_s, n_pos_d)
}

const SPATIAL_DIST_TILE: usize = 512;

pub fn build_m_max_sp_from_spatial(
    spatial: &[f64],
    n_pos: usize,
    dim: usize,
    cutoff: &[f64],
    cost_squared: bool,
) -> CooMatrix {
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let tile = SPATIAL_DIST_TILE;
    for i0 in (0..n_pos).step_by(tile) {
        let i1 = (i0 + tile).min(n_pos);
        for j0 in (0..n_pos).step_by(tile) {
            let j1 = (j0 + tile).min(n_pos);
            for i in i0..i1 {
                for j in j0..j1 {
                    let mut acc = 0.0_f64;
                    for k in 0..dim {
                        let a = spatial[i * dim + k];
                        let b = spatial[j * dim + k];
                        let d = a - b;
                        acc += d * d;
                    }
                    let cost = if cost_squared {
                        let e = acc.sqrt();
                        e * e
                    } else {
                        acc.sqrt()
                    };
                    if cost <= max_cutoff {
                        rows.push(i);
                        cols.push(j);
                        data.push(cost);
                    }
                }
            }
        }
    }
    CooMatrix::new(rows, cols, data, n_pos, n_pos)
}

fn max_mmax_data_le_cutoff(m_max: &CooMatrix, cutoff_ij: f64) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    for k in 0..m_max.nnz() {
        let d = m_max.data[k];
        if d <= cutoff_ij {
            mx = mx.max(d);
        }
    }
    mx
}

fn cost_scale_for_pair(m_max: &CooMatrix, cutoff_ij: f64, a_ij: f64) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    for k in 0..m_max.nnz() {
        if m_max.data[k] <= cutoff_ij {
            let c = m_max.data[k] * a_ij;
            mx = mx.max(c);
        }
    }
    mx
}

fn precompute_max_d_uniform_cutoff(m_max: &CooMatrix, cutoff: &[f64]) -> Option<f64> {
    let c0 = *cutoff.first()?;
    if !cutoff.iter().all(|&c| c == c0) {
        return None;
    }
    Some(max_mmax_data_le_cutoff(m_max, c0))
}

#[inline]
fn cost_scale_with_precomputed_max_d(
    m_max: &CooMatrix,
    cut: f64,
    a_ij: f64,
    uniform_max_d: Option<f64>,
) -> f64 {
    if let Some(mx_d) = uniform_max_d {
        if a_ij > 0.0 {
            return mx_d * a_ij;
        }
        if a_ij == 0.0 {
            return 0.0;
        }
    }
    cost_scale_for_pair(m_max, cut, a_ij)
}

fn csr_to_coo_scaled(csr: &CsrMatrix, scale: f64) -> CooMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..csr.nrows {
        for k in csr.indptr[i]..csr.indptr[i + 1] {
            rows.push(i);
            cols.push(csr.indices[k]);
            data.push(csr.data[k] * scale);
        }
    }
    CooMatrix::new(rows, cols, data, csr.nrows, csr.ncols)
}

fn unot_to_csr(
    a: &[f64],
    b: &[f64],
    c: &CooMatrix,
    eps_p: f64,
    rho: f64,
    eps_mu: f64,
    eps_nu: f64,
    nitermax: usize,
    stopthr: f64,
) -> Result<CsrMatrix, &'static str> {
    if (eps_p - eps_mu).abs() > 1e-8 || (eps_p - eps_nu).abs() > 1e-8 {
        return Err("momentum solver (eps_mu/eps_nu != eps_p) not implemented in Rust port");
    }
    let nza = nz_indices(a);
    let nzb = nz_indices(b);
    if nza.is_empty() || nzb.is_empty() {
        return Ok(CsrMatrix {
            indptr: vec![0; a.len() + 1],
            indices: vec![],
            data: vec![],
            nrows: a.len(),
            ncols: b.len(),
        });
    }
    let a_s: Vec<f64> = nza.iter().map(|&i| a[i]).collect();
    let b_s: Vec<f64> = nzb.iter().map(|&i| b[i]).collect();
    let c_nz = coo_submatrix_pull(c, &nza, &nzb);
    let p = unot_sinkhorn_l1_sparse(&a_s, &b_s, &c_nz, eps_p, rho, nitermax, stopthr, false);
    Ok(coo_expand_to_csr(&p, a.len(), b.len(), &nza, &nzb))
}

pub fn is_inf_entry(x: f64) -> bool {
    x.is_infinite()
}

fn cot_sparse_mmax(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: f64,
    eps_nu: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    m_max_sp: &CooMatrix,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let uniform_max_d = precompute_max_d_uniform_cutoff(m_max_sp, cutoff);
    let sum_s: f64 = s.iter().sum();
    let sum_d: f64 = d.iter().sum();
    let max_amount = sum_s.max(sum_d);
    let s: Vec<f64> = s.iter().map(|x| x / max_amount).collect();
    let d: Vec<f64> = d.iter().map(|x| x / max_amount).collect();
    let a_flat = flatten_f_order(&s, n_pos_s, ns_s);
    let b_flat = flatten_f_order(&d, n_pos_d, ns_d);
    let mut c_rows = Vec::new();
    let mut c_cols = Vec::new();
    let mut c_data = Vec::new();
    let mut cost_scales = Vec::new();
    for i in 0..ns_s {
        for j in 0..ns_d {
            if is_inf_entry(a[i * ns_d + j]) {
                continue;
            }
            let a_ij = a[i * ns_d + j];
            let cut = cutoff[i * ns_d + j];
            let tmp_nz_s: Vec<usize> = (0..n_pos_s).filter(|&r| s[r * ns_s + i] > 0.0).collect();
            let tmp_nz_d: Vec<usize> = (0..n_pos_d).filter(|&c| d[c * ns_d + j] > 0.0).collect();
            if tmp_nz_s.is_empty() || tmp_nz_d.is_empty() {
                continue;
            }
            cost_scales.push(cost_scale_with_precomputed_max_d(
                m_max_sp,
                cut,
                a_ij,
                uniform_max_d,
            ));
            let sub = coo_submatrix_pull(&m_max_sp, &tmp_nz_s, &tmp_nz_d);
            for k in 0..sub.nnz() {
                if sub.data[k] <= cut {
                    let r = tmp_nz_s[sub.rows[k]];
                    let c = tmp_nz_d[sub.cols[k]];
                    c_rows.push(r + i * n_pos_s);
                    c_cols.push(c + j * n_pos_d);
                    c_data.push(sub.data[k] * a_ij);
                }
            }
        }
    }
    if cost_scales.is_empty() {
        let mut empty = HashMap::new();
        for i in 0..ns_s {
            for j in 0..ns_d {
                if !is_inf_entry(a[i * ns_d + j]) {
                    empty.insert((i, j), CooMatrix::empty(n_pos_s, n_pos_d));
                }
            }
        }
        return Ok(empty);
    }
    let cost_scale = cost_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for v in &mut c_data {
        *v /= cost_scale;
    }
    let len_a = a_flat.len();
    let len_b = b_flat.len();
    let c_full = CooMatrix::new(c_rows, c_cols, c_data, len_a, len_b);
    let p_csr = unot_to_csr(
        &a_flat, &b_flat, &c_full, eps_p, rho, eps_mu, eps_nu, nitermax, stopthr,
    )?;
    let mut out = HashMap::new();
    for i in 0..ns_s {
        for j in 0..ns_d {
            if is_inf_entry(a[i * ns_d + j]) {
                continue;
            }
            let block = csr_get_block(
                &p_csr,
                i * n_pos_s,
                (i + 1) * n_pos_s,
                j * n_pos_d,
                (j + 1) * n_pos_d,
            );
            let scaled: Vec<f64> = block.data.iter().map(|x| x * max_amount).collect();
            out.insert(
                (i, j),
                CooMatrix::new(block.rows, block.cols, scaled, n_pos_s, n_pos_d),
            );
        }
    }
    Ok(out)
}

pub fn cot_sparse(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let eps_mu = eps_mu.unwrap_or(eps_p);
    let eps_nu = eps_nu.unwrap_or(eps_p);
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_max_sp = build_m_max_sp(m, n_pos_s, n_pos_d, max_cutoff);
    cot_sparse_mmax(
        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu, eps_nu, rho, nitermax,
        stopthr, &m_max_sp,
    )
}

fn cot_row_mmax(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: f64,
    eps_nu: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    m_max_sp: &CooMatrix,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let uniform_max_d = precompute_max_d_uniform_cutoff(m_max_sp, cutoff);
    let mut p_expand: HashMap<(usize, usize), CooMatrix> = HashMap::new();
    for i in 0..ns_s {
        let d_ind: Vec<usize> = (0..ns_d)
            .filter(|&j| !is_inf_entry(a[i * ns_d + j]))
            .collect();
        let a_col: Vec<f64> = (0..n_pos_s).map(|r| s[r * ns_s + i]).collect();
        let b_flat: Vec<f64> = {
            let mut v = Vec::with_capacity(n_pos_d * d_ind.len());
            for &dj in &d_ind {
                for r in 0..n_pos_d {
                    v.push(d[r * ns_d + dj]);
                }
            }
            v
        };
        let nz_a = nz_indices(&a_col);
        let nz_b = nz_indices(&b_flat);
        if nz_a.is_empty() || nz_b.is_empty() {
            for &dj in &d_ind {
                p_expand.insert((i, dj), CooMatrix::empty(n_pos_s, n_pos_d));
            }
            continue;
        }
        let max_amount: f64 = nz_a
            .iter()
            .map(|&k| a_col[k])
            .sum::<f64>()
            .max(nz_b.iter().map(|&k| b_flat[k]).sum());
        let a_n: Vec<f64> = a_col.iter().map(|x| x / max_amount).collect();
        let b_n: Vec<f64> = b_flat.iter().map(|x| x / max_amount).collect();
        let mut c_rows = Vec::new();
        let mut c_cols = Vec::new();
        let mut c_data = Vec::new();
        let mut cost_scales = Vec::new();
        for (ji, &dj) in d_ind.iter().enumerate() {
            let tmp_nz_s: Vec<usize> = (0..n_pos_s).filter(|&r| s[r * ns_s + i] > 0.0).collect();
            let tmp_nz_d: Vec<usize> = (0..n_pos_d).filter(|&c| d[c * ns_d + dj] > 0.0).collect();
            let cut = cutoff[i * ns_d + dj];
            let a_ij = a[i * ns_d + dj];
            cost_scales.push(cost_scale_with_precomputed_max_d(
                m_max_sp,
                cut,
                a_ij,
                uniform_max_d,
            ));
            let sub = coo_submatrix_pull(&m_max_sp, &tmp_nz_s, &tmp_nz_d);
            for k in 0..sub.nnz() {
                if sub.data[k] <= cut {
                    let r = tmp_nz_s[sub.rows[k]];
                    let c = tmp_nz_d[sub.cols[k]];
                    c_rows.push(r);
                    c_cols.push(c + ji * n_pos_d);
                    c_data.push(sub.data[k] * a_ij);
                }
            }
        }
        let cost_scale = cost_scales
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for v in &mut c_data {
            *v /= cost_scale;
        }
        let len_b = n_pos_d * d_ind.len();
        let c_full = CooMatrix::new(c_rows, c_cols, c_data, n_pos_s, len_b);
        let p_csr = unot_to_csr(
            &a_n, &b_n, &c_full, eps_p, rho, eps_mu, eps_nu, nitermax, stopthr,
        )?;
        for (ji, &dj) in d_ind.iter().enumerate() {
            let block = csr_get_block(&p_csr, 0, n_pos_s, ji * n_pos_d, (ji + 1) * n_pos_d);
            let scaled: Vec<f64> = block.data.iter().map(|x| x * max_amount).collect();
            p_expand.insert(
                (i, dj),
                CooMatrix::new(block.rows, block.cols, scaled, n_pos_s, n_pos_d),
            );
        }
    }
    Ok(p_expand)
}

pub fn cot_row_sparse(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let eps_mu = eps_mu.unwrap_or(eps_p);
    let eps_nu = eps_nu.unwrap_or(eps_p);
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_max_sp = build_m_max_sp(m, n_pos_s, n_pos_d, max_cutoff);
    cot_row_mmax(
        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu, eps_nu, rho, nitermax,
        stopthr, &m_max_sp,
    )
}

fn cot_col_mmax(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: f64,
    eps_nu: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    m_max_sp: &CooMatrix,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let uniform_max_d = precompute_max_d_uniform_cutoff(m_max_sp, cutoff);
    let mut p_expand: HashMap<(usize, usize), CooMatrix> = HashMap::new();
    for j in 0..ns_d {
        let s_ind: Vec<usize> = (0..ns_s)
            .filter(|&i| !is_inf_entry(a[i * ns_d + j]))
            .collect();
        let b_col: Vec<f64> = (0..n_pos_d).map(|r| d[r * ns_d + j]).collect();
        let a_flat: Vec<f64> = {
            let mut v = Vec::with_capacity(n_pos_s * s_ind.len());
            for &si in &s_ind {
                for r in 0..n_pos_s {
                    v.push(s[r * ns_s + si]);
                }
            }
            v
        };
        let nz_a = nz_indices(&a_flat);
        let nz_b = nz_indices(&b_col);
        if nz_a.is_empty() || nz_b.is_empty() {
            for &si in &s_ind {
                p_expand.insert((si, j), CooMatrix::empty(n_pos_s, n_pos_d));
            }
            continue;
        }
        let max_amount: f64 = nz_a
            .iter()
            .map(|&k| a_flat[k])
            .sum::<f64>()
            .max(nz_b.iter().map(|&k| b_col[k]).sum());
        let a_n: Vec<f64> = a_flat.iter().map(|x| x / max_amount).collect();
        let b_n: Vec<f64> = b_col.iter().map(|x| x / max_amount).collect();
        let mut c_rows = Vec::new();
        let mut c_cols = Vec::new();
        let mut c_data = Vec::new();
        let mut cost_scales = Vec::new();
        for (ii, &si) in s_ind.iter().enumerate() {
            let tmp_nz_s: Vec<usize> = (0..n_pos_s).filter(|&r| s[r * ns_s + si] > 0.0).collect();
            let tmp_nz_d: Vec<usize> = (0..n_pos_d).filter(|&c| d[c * ns_d + j] > 0.0).collect();
            let cut = cutoff[si * ns_d + j];
            let a_ij = a[si * ns_d + j];
            cost_scales.push(cost_scale_with_precomputed_max_d(
                m_max_sp,
                cut,
                a_ij,
                uniform_max_d,
            ));
            let sub = coo_submatrix_pull(&m_max_sp, &tmp_nz_s, &tmp_nz_d);
            for k in 0..sub.nnz() {
                if sub.data[k] <= cut {
                    let r = tmp_nz_s[sub.rows[k]];
                    let c = tmp_nz_d[sub.cols[k]];
                    c_rows.push(r + ii * n_pos_s);
                    c_cols.push(c);
                    c_data.push(sub.data[k] * a_ij);
                }
            }
        }
        let cost_scale = cost_scales
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for v in &mut c_data {
            *v /= cost_scale;
        }
        let len_a = n_pos_s * s_ind.len();
        let c_full = CooMatrix::new(c_rows, c_cols, c_data, len_a, n_pos_d);
        let p_csr = unot_to_csr(
            &a_n, &b_n, &c_full, eps_p, rho, eps_mu, eps_nu, nitermax, stopthr,
        )?;
        for (ii, &si) in s_ind.iter().enumerate() {
            let block = csr_get_block(&p_csr, ii * n_pos_s, (ii + 1) * n_pos_s, 0, n_pos_d);
            let scaled: Vec<f64> = block.data.iter().map(|x| x * max_amount).collect();
            p_expand.insert(
                (si, j),
                CooMatrix::new(block.rows, block.cols, scaled, n_pos_s, n_pos_d),
            );
        }
    }
    Ok(p_expand)
}

pub fn cot_col_sparse(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let eps_mu = eps_mu.unwrap_or(eps_p);
    let eps_nu = eps_nu.unwrap_or(eps_p);
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_max_sp = build_m_max_sp(m, n_pos_s, n_pos_d, max_cutoff);
    cot_col_mmax(
        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu, eps_nu, rho, nitermax,
        stopthr, &m_max_sp,
    )
}

fn cot_blk_mmax(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: f64,
    eps_nu: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    m_max_sp: &CooMatrix,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let uniform_max_d = precompute_max_d_uniform_cutoff(m_max_sp, cutoff);
    let mut p_expand: HashMap<(usize, usize), CooMatrix> = HashMap::new();
    for i in 0..ns_s {
        for j in 0..ns_d {
            if is_inf_entry(a[i * ns_d + j]) {
                continue;
            }
            let a_col: Vec<f64> = (0..n_pos_s).map(|r| s[r * ns_s + i]).collect();
            let b_col: Vec<f64> = (0..n_pos_d).map(|r| d[r * ns_d + j]).collect();
            let nz_a = nz_indices(&a_col);
            let nz_b = nz_indices(&b_col);
            if nz_a.is_empty() || nz_b.is_empty() {
                p_expand.insert((i, j), CooMatrix::empty(n_pos_s, n_pos_d));
                continue;
            }
            let max_amount: f64 = nz_a
                .iter()
                .map(|&k| a_col[k])
                .sum::<f64>()
                .max(nz_b.iter().map(|&k| b_col[k]).sum());
            let a_n: Vec<f64> = a_col.iter().map(|x| x / max_amount).collect();
            let b_n: Vec<f64> = b_col.iter().map(|x| x / max_amount).collect();
            let tmp_nz_s: Vec<usize> = (0..n_pos_s).filter(|&r| s[r * ns_s + i] > 0.0).collect();
            let tmp_nz_d: Vec<usize> = (0..n_pos_d).filter(|&c| d[c * ns_d + j] > 0.0).collect();
            let sub = coo_submatrix_pull(&m_max_sp, &tmp_nz_s, &tmp_nz_d);
            let cut = cutoff[i * ns_d + j];
            let a_ij = a[i * ns_d + j];
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut data = Vec::new();
            for k in 0..sub.nnz() {
                if sub.data[k] <= cut {
                    rows.push(tmp_nz_s[sub.rows[k]]);
                    cols.push(tmp_nz_d[sub.cols[k]]);
                    data.push(sub.data[k] * a_ij);
                }
            }
            let cost_scale = cost_scale_with_precomputed_max_d(m_max_sp, cut, a_ij, uniform_max_d);
            for v in &mut data {
                *v /= cost_scale;
            }
            let c_full = CooMatrix::new(rows, cols, data, n_pos_s, n_pos_d);
            let p_csr = unot_to_csr(
                &a_n, &b_n, &c_full, eps_p, rho, eps_mu, eps_nu, nitermax, stopthr,
            )?;
            let coo = csr_to_coo_scaled(&p_csr, max_amount);
            p_expand.insert((i, j), coo);
        }
    }
    Ok(p_expand)
}

pub fn cot_blk_sparse(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let eps_mu = eps_mu.unwrap_or(eps_p);
    let eps_nu = eps_nu.unwrap_or(eps_p);
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_max_sp = build_m_max_sp(m, n_pos_s, n_pos_d, max_cutoff);
    cot_blk_mmax(
        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu, eps_nu, rho, nitermax,
        stopthr, &m_max_sp,
    )
}

pub fn cot_combine_sparse_from_m_max(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    weights: [f64; 4],
    nitermax: usize,
    stopthr: f64,
    m_max_sp: &CooMatrix,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let eps_mu_r = eps_mu.unwrap_or(eps_p);
    let eps_nu_r = eps_nu.unwrap_or(eps_p);
    let ((p_cot, p_row), (p_col, p_blk)) = rayon::join(
        || {
            rayon::join(
                || {
                    cot_sparse_mmax(
                        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu_r, eps_nu_r,
                        rho, nitermax, stopthr, m_max_sp,
                    )
                },
                || {
                    cot_row_mmax(
                        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu_r, eps_nu_r,
                        rho, nitermax, stopthr, m_max_sp,
                    )
                },
            )
        },
        || {
            rayon::join(
                || {
                    cot_col_mmax(
                        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu_r, eps_nu_r,
                        rho, nitermax, stopthr, m_max_sp,
                    )
                },
                || {
                    cot_blk_mmax(
                        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu_r, eps_nu_r,
                        rho, nitermax, stopthr, m_max_sp,
                    )
                },
            )
        },
    );
    let p_cot = p_cot?;
    let p_row = p_row?;
    let p_col = p_col?;
    let p_blk = p_blk?;
    let pairs: Vec<(usize, usize)> = (0..ns_s)
        .flat_map(|i| (0..ns_d).map(move |j| (i, j)))
        .filter(|&(i, j)| !is_inf_entry(a[i * ns_d + j]))
        .collect();
    let w = weights;
    let entries: Vec<((usize, usize), CooMatrix)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let c0 = p_cot.get(&(i, j)).unwrap();
            let c1 = p_row.get(&(i, j)).unwrap();
            let c2 = p_col.get(&(i, j)).unwrap();
            let c3 = p_blk.get(&(i, j)).unwrap();
            let merged = merge_weighted_coo(&[c0, c1, c2, c3], &w, n_pos_s, n_pos_d);
            ((i, j), merged)
        })
        .collect();
    Ok(entries.into_iter().collect())
}

pub fn cot_combine_sparse(
    s: &[f64],
    d: &[f64],
    a: &[f64],
    m: &[f64],
    cutoff: &[f64],
    n_pos_s: usize,
    n_pos_d: usize,
    ns_s: usize,
    ns_d: usize,
    eps_p: f64,
    eps_mu: Option<f64>,
    eps_nu: Option<f64>,
    rho: f64,
    weights: [f64; 4],
    nitermax: usize,
    stopthr: f64,
) -> Result<HashMap<(usize, usize), CooMatrix>, &'static str> {
    let max_cutoff = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m_max_sp = build_m_max_sp(m, n_pos_s, n_pos_d, max_cutoff);
    cot_combine_sparse_from_m_max(
        s, d, a, cutoff, n_pos_s, n_pos_d, ns_s, ns_d, eps_p, eps_mu, eps_nu, rho, weights,
        nitermax, stopthr, &m_max_sp,
    )
}

fn merge_weighted_coo(parts: &[&CooMatrix], w: &[f64; 4], nrows: usize, ncols: usize) -> CooMatrix {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for (pi, p) in parts.iter().enumerate() {
        let wi = w[pi];
        for k in 0..p.nnz() {
            let key = (p.rows[k], p.cols[k]);
            *acc.entry(key).or_insert(0.0) += wi * p.data[k];
        }
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for ((r, c), v) in acc {
        rows.push(r);
        cols.push(c);
        data.push(v);
    }
    CooMatrix::new(rows, cols, data, nrows, ncols)
}
