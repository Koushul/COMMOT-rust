use commot_core::{coo_submatrix_pull, CooMatrix};

#[test]
fn coo_submatrix_pull_matches_python_reference() {
    let rows = vec![0usize, 0, 1, 2];
    let cols = vec![1usize, 2, 0, 1];
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let m = CooMatrix::new(rows, cols, data, 3, 3);
    let row_pick = [0usize, 2];
    let col_pick = [1usize, 2];
    let sub = coo_submatrix_pull(&m, &row_pick, &col_pick);
    assert_eq!(sub.nrows, 2);
    assert_eq!(sub.ncols, 2);
    let mut triples: Vec<(usize, usize, f64)> = (0..sub.nnz())
        .map(|k| (sub.rows[k], sub.cols[k], sub.data[k]))
        .collect();
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(triples, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 4.0)]);
}
