use std::fs::File;
use std::path::{Path, PathBuf};

use commot_core::cot_combine_sparse;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::Deserialize;

#[derive(Deserialize)]
struct Meta {
    n_pos: usize,
    ns_s: usize,
    ns_d: usize,
    cot_eps_p: f64,
    cot_rho: f64,
    cot_nitermax: usize,
    cot_weights: [f64; 4],
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/parity")
}

fn load_2d(path: impl AsRef<Path>) -> Array2<f64> {
    read_npy(path.as_ref()).unwrap_or_else(|_| panic!("read_npy {}", path.as_ref().display()))
}

fn coo_to_dense(coo: &commot_core::CooMatrix, n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for k in 0..coo.nnz() {
        d[coo.rows[k] * n + coo.cols[k]] += coo.data[k];
    }
    d
}

#[test]
fn cot_combine_matches_python_golden() {
    let dir = fixture_dir();
    let meta: Meta = serde_json::from_reader(File::open(dir.join("meta.json")).unwrap()).unwrap();
    let s = load_2d(dir.join("S.npy")).into_raw_vec_and_offset().0;
    let d = load_2d(dir.join("D.npy")).into_raw_vec_and_offset().0;
    let a = load_2d(dir.join("A.npy")).into_raw_vec_and_offset().0;
    let m = load_2d(dir.join("M.npy")).into_raw_vec_and_offset().0;
    let cutoff = load_2d(dir.join("cutoff.npy")).into_raw_vec_and_offset().0;
    let n = meta.n_pos;
    let out = cot_combine_sparse(
        &s,
        &d,
        &a,
        &m,
        &cutoff,
        n,
        n,
        meta.ns_s,
        meta.ns_d,
        meta.cot_eps_p,
        None,
        None,
        meta.cot_rho,
        meta.cot_weights,
        meta.cot_nitermax,
        1e-8,
    )
    .expect("cot_combine");
    for i in 0..meta.ns_s {
        for j in 0..meta.ns_d {
            if a[i * meta.ns_d + j].is_infinite() {
                continue;
            }
            let golden = load_2d(dir.join(format!("P_{i}_{j}.npy")));
            let coo = out.get(&(i, j)).expect("pair");
            let got = coo_to_dense(coo, n);
            let exp: Vec<f64> = golden.into_raw_vec_and_offset().0;
            let l1: f64 = got.iter().zip(exp.iter()).map(|(g, e)| (g - e).abs()).sum();
            assert!(l1 < 1e-5, "pair ({i},{j}) L1 diff {l1} exceeds 1e-5");
        }
    }
}
