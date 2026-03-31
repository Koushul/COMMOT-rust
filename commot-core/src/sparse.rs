#[derive(Clone, Debug)]
pub struct CooMatrix {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CooMatrix {
    pub fn new(
        rows: Vec<usize>,
        cols: Vec<usize>,
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        assert_eq!(rows.len(), cols.len());
        assert_eq!(rows.len(), data.len());
        Self {
            rows,
            cols,
            data,
            nrows,
            ncols,
        }
    }

    pub fn empty(nrows: usize, ncols: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            data: Vec::new(),
            nrows,
            ncols,
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

pub fn coo_submatrix_pull(m: &CooMatrix, row_pick: &[usize], col_pick: &[usize]) -> CooMatrix {
    let lr = row_pick.len();
    let lc = col_pick.len();
    let mut gr = vec![usize::MAX; m.nrows];
    let mut gc = vec![usize::MAX; m.ncols];
    for (ar, &r) in row_pick.iter().enumerate() {
        gr[r] = ar;
    }
    for (ac, &c) in col_pick.iter().enumerate() {
        gc[c] = ac;
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for k in 0..m.nnz() {
        let r = m.rows[k];
        let c = m.cols[k];
        if gr[r] != usize::MAX && gc[c] != usize::MAX {
            rows.push(gr[r]);
            cols.push(gc[c]);
            data.push(m.data[k]);
        }
    }
    CooMatrix::new(rows, cols, data, lr, lc)
}

#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

pub fn coo_to_csr(coo: &CooMatrix) -> CsrMatrix {
    let mut triples: Vec<(usize, usize, f64)> = (0..coo.nnz())
        .map(|k| (coo.rows[k], coo.cols[k], coo.data[k]))
        .collect();
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut merged: Vec<(usize, usize, f64)> = Vec::new();
    for (r, c, v) in triples {
        if let Some(last) = merged.last_mut() {
            if last.0 == r && last.1 == c {
                last.2 += v;
                continue;
            }
        }
        merged.push((r, c, v));
    }
    let mut indptr = vec![0usize; coo.nrows + 1];
    let mut indices = Vec::with_capacity(merged.len());
    let mut data = Vec::with_capacity(merged.len());
    for (r, c, v) in merged {
        indices.push(c);
        data.push(v);
        indptr[r + 1] += 1;
    }
    for i in 0..coo.nrows {
        indptr[i + 1] += indptr[i];
    }
    CsrMatrix {
        indptr,
        indices,
        data,
        nrows: coo.nrows,
        ncols: coo.ncols,
    }
}

pub fn csr_row_sums(csr: &CsrMatrix) -> Vec<f64> {
    let mut out = vec![0.0; csr.nrows];
    for i in 0..csr.nrows {
        let mut s = 0.0;
        for k in csr.indptr[i]..csr.indptr[i + 1] {
            s += csr.data[k];
        }
        out[i] = s;
    }
    out
}

pub fn csr_col_sums(csr: &CsrMatrix) -> Vec<f64> {
    let mut out = vec![0.0; csr.ncols];
    for i in 0..csr.nrows {
        for k in csr.indptr[i]..csr.indptr[i + 1] {
            let c = csr.indices[k];
            out[c] += csr.data[k];
        }
    }
    out
}

pub fn coo_row_sums_scipy_style(coo: &CooMatrix) -> Vec<f64> {
    let csr = coo_to_csr(coo);
    csr_row_sums(&csr)
}

pub fn coo_col_sums_scipy_style(coo: &CooMatrix) -> Vec<f64> {
    let csr = coo_to_csr(coo);
    csr_col_sums(&csr)
}

pub fn csr_to_dense(csr: &CsrMatrix) -> Vec<f64> {
    let mut d = vec![0.0; csr.nrows * csr.ncols];
    for i in 0..csr.nrows {
        for k in csr.indptr[i]..csr.indptr[i + 1] {
            let j = csr.indices[k];
            d[i * csr.ncols + j] += csr.data[k];
        }
    }
    d
}

pub fn coo_expand_to_csr(
    coo: &CooMatrix,
    nrow_full: usize,
    ncol_full: usize,
    row_map: &[usize],
    col_map: &[usize],
) -> CsrMatrix {
    let mut triples: Vec<(usize, usize, f64)> = Vec::with_capacity(coo.nnz());
    for k in 0..coo.nnz() {
        triples.push((row_map[coo.rows[k]], col_map[coo.cols[k]], coo.data[k]));
    }
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut merged: Vec<(usize, usize, f64)> = Vec::new();
    for (r, c, v) in triples {
        if let Some(last) = merged.last_mut() {
            if last.0 == r && last.1 == c {
                last.2 += v;
                continue;
            }
        }
        merged.push((r, c, v));
    }
    let mut indptr = vec![0usize; nrow_full + 1];
    let mut indices = Vec::with_capacity(merged.len());
    let mut data = Vec::with_capacity(merged.len());
    for (r, c, v) in merged {
        indices.push(c);
        data.push(v);
        indptr[r + 1] += 1;
    }
    for i in 0..nrow_full {
        indptr[i + 1] += indptr[i];
    }
    CsrMatrix {
        indptr,
        indices,
        data,
        nrows: nrow_full,
        ncols: ncol_full,
    }
}

pub fn csr_get_block(
    csr: &CsrMatrix,
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
) -> CooMatrix {
    let br = row1 - row0;
    let bc = col1 - col0;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in row0..row1 {
        for k in csr.indptr[i]..csr.indptr[i + 1] {
            let j = csr.indices[k];
            if j >= col0 && j < col1 {
                rows.push(i - row0);
                cols.push(j - col0);
                data.push(csr.data[k]);
            }
        }
    }
    CooMatrix::new(rows, cols, data, br, bc)
}
