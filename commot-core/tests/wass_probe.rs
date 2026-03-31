use commot_core::sparse::CooMatrix;
use commot_core::unot_sinkhorn_l1_sparse;
use ndarray::array;
use wass::unbalanced_sinkhorn_log_with_convergence;

#[test]
fn wass_kl_unbalanced_differs_from_commot_l1_sparse() {
    let a = vec![0.3_f64, 0.7];
    let b = vec![0.4_f64, 0.6];
    let c = CooMatrix::new(
        vec![0, 0, 1, 1],
        vec![0, 1, 0, 1],
        vec![0.1, 0.5, 0.5, 0.1],
        2,
        2,
    );
    let eps = 0.1_f64;
    let rho = 10.0_f64;
    let p_comm = unot_sinkhorn_l1_sparse(&a, &b, &c, eps, rho, 5000, 1e-10, false);
    let cost = array![[0.1_f32, 0.5_f32], [0.5_f32, 0.1_f32]];
    let af = array![0.3_f32, 0.7_f32];
    let bf = array![0.4_f32, 0.6_f32];
    let wass_res = unbalanced_sinkhorn_log_with_convergence(
        &af, &bf, &cost, eps as f32, rho as f32, 5000, 1e-6_f32,
    );
    assert!(wass_res.is_ok());
    let (_plan, _obj, _it) = wass_res.unwrap();
    let mut max_diff = 0.0_f64;
    for k in 0..p_comm.nnz() {
        let r = p_comm.rows[k];
        let col = p_comm.cols[k];
        let wc = _plan[[r, col]] as f64;
        max_diff = max_diff.max((p_comm.data[k] - wc).abs());
    }
    assert!(
        max_diff > 1e-4,
        "wass unexpectedly matched COMMOT L1 sparse (max diff {max_diff})"
    );
}
