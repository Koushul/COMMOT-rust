use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use commot_core::{write_commot_obsp, IoError, Sidecar};

#[cfg(feature = "pprof")]
use pprof::ProfilerGuard;

#[derive(Parser, Debug)]
#[command(name = "commot-cli")]
struct Args {
    #[arg(long)]
    input_h5ad: PathBuf,
    #[arg(long)]
    output_h5ad: PathBuf,
    #[arg(long)]
    sidecar_json: PathBuf,
    #[arg(long, default_value = "testdb")]
    database_name: String,
    #[arg(long, default_value_t = 1.0)]
    dis_thr: f64,
    #[arg(long, default_value_t = false)]
    euc_square: bool,
    #[arg(long, default_value_t = 0.1)]
    cot_eps_p: f64,
    #[arg(long, default_value_t = 10.0)]
    cot_rho: f64,
    #[arg(long, default_value_t = 10000usize)]
    cot_nitermax: usize,
    #[arg(long, default_value = "0.25,0.25,0.25,0.25")]
    cot_weights: String,
    #[arg(long, default_value_t = false)]
    pathway_sum: bool,
    #[cfg(feature = "gpu")]
    #[arg(
        long,
        default_value_t = false,
        help = "Compute pairwise spatial distances on the GPU when building without obsp/spatial_distance (requires --features gpu)"
    )]
    gpu_distance: bool,
    #[cfg(feature = "pprof")]
    #[arg(
        long,
        value_name = "SVG_PATH",
        help = "Write CPU flamegraph (SVG) after the run; requires building with --features pprof"
    )]
    pprof_flamegraph: Option<PathBuf>,
}

fn parse_weights(s: &str) -> Result<[f64; 4]> {
    let p: Vec<f64> = s
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .context("cot_weights must be four comma-separated floats")?;
    if p.len() != 4 {
        anyhow::bail!("cot_weights must have exactly 4 values");
    }
    Ok([p[0], p[1], p[2], p[3]])
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(feature = "pprof")]
    let pprof_guard = match &args.pprof_flamegraph {
        Some(_) => Some(ProfilerGuard::new(100).map_err(|e| anyhow::anyhow!("pprof: {}", e))?),
        None => None,
    };
    let w = parse_weights(&args.cot_weights)?;
    let raw = fs::read_to_string(&args.sidecar_json).context("read sidecar")?;
    let side: Sidecar = serde_json::from_str(&raw).context("parse sidecar JSON")?;
    let run = write_commot_obsp(
        &args.input_h5ad,
        &args.output_h5ad,
        &args.database_name,
        args.dis_thr,
        args.euc_square,
        args.cot_eps_p,
        args.cot_rho,
        args.cot_nitermax,
        w,
        &side,
        args.pathway_sum,
        #[cfg(feature = "gpu")]
        args.gpu_distance,
        #[cfg(not(feature = "gpu"))]
        false,
    )
    .map_err(|e: IoError| anyhow::anyhow!("{}", e));
    #[cfg(feature = "pprof")]
    if let (Some(guard), Some(path)) = (pprof_guard, args.pprof_flamegraph.as_ref()) {
        if let Ok(report) = guard.report().build() {
            let mut f =
                fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
            report.flamegraph(&mut f).context("write flamegraph")?;
        }
    }
    run?;
    Ok(())
}
