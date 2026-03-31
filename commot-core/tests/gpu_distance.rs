use commot_core::gpu::spatial_pairwise_costs_f64;
use commot_core::pipeline::{euclidean_distance_matrix, square_distance_matrix};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn gpu_pairwise_euclidean_near_cpu() {
    let n = 64;
    let dim = 3;
    let spatial: Vec<f64> = (0..n * dim).map(|i| (i as f64) * 0.07 - 2.0).collect();
    let cpu = euclidean_distance_matrix(&spatial, n, dim);
    let gpu = match spatial_pairwise_costs_f64(&spatial, n, dim, false) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GPU distance test skipped: {e}");
            return;
        }
    };
    assert_eq!(gpu.len(), n * n);
    let d = max_abs_diff(&cpu, &gpu);
    assert!(
        d < 5e-4,
        "GPU vs CPU euclidean max abs diff {d} (expected < 5e-4 from f32 intermediates)"
    );
}

#[test]
fn gpu_squared_euclidean_matches_fused_reference() {
    let n = 48;
    let dim = 2;
    let spatial: Vec<f64> = (0..n * dim).map(|i| (i as f64).sin() * 0.5).collect();
    let cpu_euc = euclidean_distance_matrix(&spatial, n, dim);
    let cpu_sq = square_distance_matrix(&cpu_euc, n);
    let gpu_sq = match spatial_pairwise_costs_f64(&spatial, n, dim, true) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GPU distance test skipped: {e}");
            return;
        }
    };
    let d = max_abs_diff(&cpu_sq, &gpu_sq);
    assert!(
        d < 1e-3,
        "GPU squared vs CPU sqrt-then-square max abs diff {d}"
    );
}
