use std::collections::HashMap;

use commot_core::{
    build_m_max_sp, build_m_max_sp_from_spatial, euclidean_distance_matrix, run_cot_combine,
    run_cot_combine_streaming, square_distance_matrix, CooMatrix,
};

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn u01(state: &mut u64) -> f64 {
    (lcg(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn synth(
    mut state: u64,
    n: usize,
    dim: usize,
    ns_s: usize,
    ns_d: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut spatial = vec![0.0_f64; n * dim];
    for x in spatial.iter_mut() {
        *x = u01(&mut state) * 12.0;
    }
    let mut s = vec![0.0_f64; n * ns_s];
    for x in s.iter_mut() {
        *x = u01(&mut state) * 4.0 + 0.05;
    }
    let mut d = vec![0.0_f64; n * ns_d];
    for x in d.iter_mut() {
        *x = u01(&mut state) * 4.0 + 0.05;
    }
    let mut a = vec![f64::INFINITY; ns_s * ns_d];
    a[0] = 1.0;
    if ns_s > 1 && ns_d > 1 {
        a[ns_s * ns_d - 1] = 1.0;
    }
    (s, d, a, spatial)
}

fn coo_to_dense(coo: &CooMatrix, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for k in 0..coo.nnz() {
        out[coo.rows[k] * n + coo.cols[k]] += coo.data[k];
    }
    out
}

fn sorted_triplets(coo: &CooMatrix) -> Vec<(usize, usize, f64)> {
    let mut v: Vec<_> = (0..coo.nnz())
        .map(|k| (coo.rows[k], coo.cols[k], coo.data[k]))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));
    v
}

fn assert_m_max_equivalent(spatial: &[f64], dist: &[f64], n: usize, dim: usize, cutoff: &[f64]) {
    let max_c = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let d = build_m_max_sp(dist, n, n, max_c);
    let s = build_m_max_sp_from_spatial(spatial, n, dim, cutoff, false);
    assert_eq!(sorted_triplets(&d), sorted_triplets(&s));
}

fn assert_m_max_equivalent_squared(
    spatial: &[f64],
    sq: &[f64],
    n: usize,
    dim: usize,
    cutoff: &[f64],
) {
    let max_c = cutoff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let d = build_m_max_sp(sq, n, n, max_c);
    let s = build_m_max_sp_from_spatial(spatial, n, dim, cutoff, true);
    assert_eq!(sorted_triplets(&d), sorted_triplets(&s));
}

fn assert_network_close(
    dense: &HashMap<(usize, usize), CooMatrix>,
    stream: &HashMap<(usize, usize), CooMatrix>,
    n: usize,
) {
    assert_eq!(dense.len(), stream.len());
    let tol = 1e-11_f64;
    for (k, a) in dense {
        let b = stream.get(k).expect("missing pair");
        let da = coo_to_dense(a, n);
        let db = coo_to_dense(b, n);
        let mx = da
            .iter()
            .zip(db.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(mx < tol, "pair {k:?} max_abs {mx} (tol {tol})");
    }
}

#[test]
fn m_max_sp_streaming_matches_dense_euclidean() {
    for n in [17, 48, 111] {
        let (_, _, _, spatial) = synth(991 * n as u64, n, 2, 2, 2);
        let dist = euclidean_distance_matrix(&spatial, n, 2);
        let cutoff = vec![2.8_f64; 4];
        assert_m_max_equivalent(&spatial, &dist, n, 2, &cutoff);
    }
}

#[test]
fn m_max_sp_streaming_matches_dense_squared() {
    for n in [19, 55] {
        let (_s, _d, _a, spatial) = synth(77 * n as u64, n, 3, 2, 2);
        let dist = euclidean_distance_matrix(&spatial, n, 3);
        let sq = square_distance_matrix(&dist, n);
        let dis_thr = 3.1_f64;
        let thr = dis_thr * dis_thr;
        let cutoff = vec![thr; 4];
        assert_m_max_equivalent_squared(&spatial, &sq, n, 3, &cutoff);
    }
}

#[test]
fn cot_combine_streaming_matches_dense_euclidean() {
    let ns_s = 2usize;
    let ns_d = 2usize;
    let dim = 2usize;
    let weights = [0.25, 0.25, 0.25, 0.25];
    for n in [24, 61] {
        for seed in [1u64, 42, 99] {
            let (s, d, a, spatial) = synth(seed.wrapping_mul(n as u64), n, dim, ns_s, ns_d);
            let dist = euclidean_distance_matrix(&spatial, n, dim);
            let dis_thr = 2.9_f64;
            let cutoff: Vec<f64> = (0..ns_s * ns_d).map(|_| dis_thr).collect();
            let dense = run_cot_combine(
                &s, &d, &a, &dist, &cutoff, n, ns_s, ns_d, 0.05, 0.1, 200, weights,
            )
            .unwrap();
            let stream = run_cot_combine_streaming(
                &s, &d, &a, &spatial, &cutoff, n, dim, ns_s, ns_d, 0.05, 0.1, 200, weights, false,
            )
            .unwrap();
            assert_network_close(&dense, &stream, n);
        }
    }
}

#[test]
fn cot_combine_streaming_matches_dense_squared_cost() {
    let ns_s = 2usize;
    let ns_d = 2usize;
    let dim = 2usize;
    let weights = [0.25, 0.25, 0.25, 0.25];
    for n in [28, 58] {
        let (s, d, a, spatial) = synth(500 + n as u64, n, dim, ns_s, ns_d);
        let dist = euclidean_distance_matrix(&spatial, n, dim);
        let dist_sq = square_distance_matrix(&dist, n);
        let dis_thr = 3.2_f64;
        let thr = dis_thr * dis_thr;
        let cutoff: Vec<f64> = (0..ns_s * ns_d).map(|_| thr).collect();
        let dense = run_cot_combine(
            &s, &d, &a, &dist_sq, &cutoff, n, ns_s, ns_d, 0.05, 0.1, 200, weights,
        )
        .unwrap();
        let stream = run_cot_combine_streaming(
            &s, &d, &a, &spatial, &cutoff, n, dim, ns_s, ns_d, 0.05, 0.1, 200, weights, true,
        )
        .unwrap();
        assert_network_close(&dense, &stream, n);
    }
}
