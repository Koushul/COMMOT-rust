use crate::sparse::{
    coo_col_sums_scipy_style, coo_row_sums_scipy_style, coo_to_csr, csr_col_sums, csr_row_sums,
    CooMatrix, CsrMatrix,
};

fn coo_coords_unique(c: &CooMatrix) -> bool {
    if c.nnz() <= 1 {
        return true;
    }
    let mut pairs: Vec<(usize, usize)> = (0..c.nnz()).map(|k| (c.rows[k], c.cols[k])).collect();
    pairs.sort_unstable();
    !pairs.windows(2).any(|w| w[0] == w[1])
}

fn csr_row_indices_for_nnz(csr: &CsrMatrix) -> Vec<usize> {
    let mut v = Vec::with_capacity(csr.data.len());
    for i in 0..csr.nrows {
        for _ in csr.indptr[i]..csr.indptr[i + 1] {
            v.push(i);
        }
    }
    v
}

fn unot_sinkhorn_l1_sparse_coo(
    a: &[f64],
    b: &[f64],
    c: &CooMatrix,
    eps: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    verbose: bool,
) -> CooMatrix {
    let tmp_rows = c.rows.clone();
    let tmp_cols = c.cols.clone();
    let mut tmp_data = vec![0.0; c.nnz()];
    let mut f = vec![0.0; a.len()];
    let mut g = vec![0.0; b.len()];
    let mut niter = 0usize;
    let mut err = 100.0_f64;
    let mut fprev = vec![0.0; a.len()];
    let mut gprev = vec![0.0; b.len()];
    while niter <= nitermax && err > stopthr {
        if niter % 10 == 0 {
            fprev.copy_from_slice(&f);
            gprev.copy_from_slice(&g);
        }
        for k in 0..c.nnz() {
            tmp_data[k] = ((-c.data[k] + f[c.rows[k]] + g[c.cols[k]]) / eps).exp();
        }
        let tmp_coo = CooMatrix::new(
            tmp_rows.clone(),
            tmp_cols.clone(),
            tmp_data.clone(),
            c.nrows,
            c.ncols,
        );
        let row_sum = coo_row_sums_scipy_style(&tmp_coo);
        for i in 0..a.len() {
            let denom = row_sum[i] + ((-rho + f[i]) / eps).exp();
            f[i] = eps * a[i].ln() - eps * denom.ln() + f[i];
        }
        for k in 0..c.nnz() {
            tmp_data[k] = ((-c.data[k] + f[c.rows[k]] + g[c.cols[k]]) / eps).exp();
        }
        let tmp_coo2 = CooMatrix::new(
            tmp_rows.clone(),
            tmp_cols.clone(),
            tmp_data.clone(),
            c.nrows,
            c.ncols,
        );
        let col_sum = coo_col_sums_scipy_style(&tmp_coo2);
        for j in 0..b.len() {
            let denom = col_sum[j] + ((-rho + g[j]) / eps).exp();
            g[j] = eps * b[j].ln() - eps * denom.ln() + g[j];
        }
        if niter % 10 == 0 {
            let mut max_diff_f = 0.0_f64;
            let mut max_f = 0.0_f64;
            let mut max_fprev = 0.0_f64;
            for i in 0..a.len() {
                max_diff_f = max_diff_f.max((f[i] - fprev[i]).abs());
                max_f = max_f.max(f[i].abs());
                max_fprev = max_fprev.max(fprev[i].abs());
            }
            let den_f = max_f.max(max_fprev).max(1.0);
            let err_f = max_diff_f / den_f;
            let mut max_diff_g = 0.0_f64;
            let mut max_g = 0.0_f64;
            let mut max_gprev = 0.0_f64;
            for j in 0..b.len() {
                max_diff_g = max_diff_g.max((g[j] - gprev[j]).abs());
                max_g = max_g.max(g[j].abs());
                max_gprev = max_gprev.max(gprev[j].abs());
            }
            let den_g = max_g.max(max_gprev).max(1.0);
            let err_g = max_diff_g / den_g;
            err = 0.5 * (err_f + err_g);
        }
        niter += 1;
    }
    if verbose {
        eprintln!("Number of iterations in unot: {}", niter);
    }
    for k in 0..c.nnz() {
        tmp_data[k] = ((-c.data[k] + f[c.rows[k]] + g[c.cols[k]]) / eps).exp();
    }
    CooMatrix::new(tmp_rows, tmp_cols, tmp_data, c.nrows, c.ncols)
}

fn unot_sinkhorn_l1_sparse_csr(
    a: &[f64],
    b: &[f64],
    c: &CooMatrix,
    eps: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    verbose: bool,
) -> CooMatrix {
    let mut base = coo_to_csr(c);
    let nnz = base.data.len();
    let row_of = csr_row_indices_for_nnz(&base);
    let c_cost = std::mem::take(&mut base.data);
    let col_idx = base.indices.clone();
    let mut tmp_csr = CsrMatrix {
        indptr: base.indptr.clone(),
        indices: col_idx.clone(),
        data: vec![0.0; nnz],
        nrows: base.nrows,
        ncols: base.ncols,
    };
    let mut f = vec![0.0; a.len()];
    let mut g = vec![0.0; b.len()];
    let mut niter = 0usize;
    let mut err = 100.0_f64;
    let mut fprev = vec![0.0; a.len()];
    let mut gprev = vec![0.0; b.len()];
    while niter <= nitermax && err > stopthr {
        if niter % 10 == 0 {
            fprev.copy_from_slice(&f);
            gprev.copy_from_slice(&g);
        }
        for k in 0..nnz {
            let i = row_of[k];
            let j = col_idx[k];
            tmp_csr.data[k] = ((-c_cost[k] + f[i] + g[j]) / eps).exp();
        }
        let row_sum = csr_row_sums(&tmp_csr);
        for i in 0..a.len() {
            let denom = row_sum[i] + ((-rho + f[i]) / eps).exp();
            f[i] = eps * a[i].ln() - eps * denom.ln() + f[i];
        }
        for k in 0..nnz {
            let i = row_of[k];
            let j = col_idx[k];
            tmp_csr.data[k] = ((-c_cost[k] + f[i] + g[j]) / eps).exp();
        }
        let col_sum = csr_col_sums(&tmp_csr);
        for j in 0..b.len() {
            let denom = col_sum[j] + ((-rho + g[j]) / eps).exp();
            g[j] = eps * b[j].ln() - eps * denom.ln() + g[j];
        }
        if niter % 10 == 0 {
            let mut max_diff_f = 0.0_f64;
            let mut max_f = 0.0_f64;
            let mut max_fprev = 0.0_f64;
            for i in 0..a.len() {
                max_diff_f = max_diff_f.max((f[i] - fprev[i]).abs());
                max_f = max_f.max(f[i].abs());
                max_fprev = max_fprev.max(fprev[i].abs());
            }
            let den_f = max_f.max(max_fprev).max(1.0);
            let err_f = max_diff_f / den_f;
            let mut max_diff_g = 0.0_f64;
            let mut max_g = 0.0_f64;
            let mut max_gprev = 0.0_f64;
            for j in 0..b.len() {
                max_diff_g = max_diff_g.max((g[j] - gprev[j]).abs());
                max_g = max_g.max(g[j].abs());
                max_gprev = max_gprev.max(gprev[j].abs());
            }
            let den_g = max_g.max(max_gprev).max(1.0);
            let err_g = max_diff_g / den_g;
            err = 0.5 * (err_f + err_g);
        }
        niter += 1;
    }
    if verbose {
        eprintln!("Number of iterations in unot: {}", niter);
    }
    let mut tmp_data = vec![0.0; c.nnz()];
    for k in 0..c.nnz() {
        tmp_data[k] = ((-c.data[k] + f[c.rows[k]] + g[c.cols[k]]) / eps).exp();
    }
    CooMatrix::new(c.rows.clone(), c.cols.clone(), tmp_data, c.nrows, c.ncols)
}

pub fn unot_sinkhorn_l1_sparse(
    a: &[f64],
    b: &[f64],
    c: &CooMatrix,
    eps: f64,
    rho: f64,
    nitermax: usize,
    stopthr: f64,
    verbose: bool,
) -> CooMatrix {
    if coo_coords_unique(c) {
        unot_sinkhorn_l1_sparse_csr(a, b, c, eps, rho, nitermax, stopthr, verbose)
    } else {
        unot_sinkhorn_l1_sparse_coo(a, b, c, eps, rho, nitermax, stopthr, verbose)
    }
}
