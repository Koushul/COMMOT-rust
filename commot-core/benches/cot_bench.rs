use std::fs::File;
use std::path::PathBuf;

use commot_core::cot_combine_sparse;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
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

fn bench_cot_combine(c: &mut Criterion) {
    let dir = fixture_dir();
    let meta: Meta = serde_json::from_reader(File::open(dir.join("meta.json")).unwrap()).unwrap();
    let s: ndarray::Array2<f64> = read_npy(dir.join("S.npy")).unwrap();
    let d: ndarray::Array2<f64> = read_npy(dir.join("D.npy")).unwrap();
    let a: ndarray::Array2<f64> = read_npy(dir.join("A.npy")).unwrap();
    let m: ndarray::Array2<f64> = read_npy(dir.join("M.npy")).unwrap();
    let cutoff: ndarray::Array2<f64> = read_npy(dir.join("cutoff.npy")).unwrap();
    let (s, d, a, m, cutoff) = (
        s.into_raw_vec_and_offset().0,
        d.into_raw_vec_and_offset().0,
        a.into_raw_vec_and_offset().0,
        m.into_raw_vec_and_offset().0,
        cutoff.into_raw_vec_and_offset().0,
    );
    let n = meta.n_pos;
    c.bench_function("cot_combine_sparse_toy", |b| {
        b.iter(|| {
            black_box(
                cot_combine_sparse(
                    black_box(&s),
                    black_box(&d),
                    black_box(&a),
                    black_box(&m),
                    black_box(&cutoff),
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
                .unwrap(),
            )
        });
    });
}

criterion_group!(benches, bench_cot_combine);
criterion_main!(benches);
