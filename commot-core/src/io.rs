use std::collections::HashMap;
use std::path::Path;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::File;
use hdf5_metno::Group;
use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::cot::is_inf_entry;
use crate::pipeline::{
    euclidean_distance_matrix, run_cot_combine, run_cot_combine_streaming, square_distance_matrix,
};
use crate::sparse::CooMatrix;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("HDF5: {0}")]
    Hdf5(#[from] hdf5_metno::Error),
    #[error("missing {0}")]
    Missing(&'static str),
    #[error("unsupported AnnData X layout (expected dense /X dataset)")]
    UnsupportedX,
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

pub struct AnnDataRead {
    pub x: Vec<f64>,
    pub n_obs: usize,
    pub n_var: usize,
    pub var_names: Vec<String>,
    pub spatial: Vec<f64>,
    pub spatial_dim: usize,
    pub spatial_distance: Option<Vec<f64>>,
}

fn read_var_index(f: &File) -> Result<Vec<String>, IoError> {
    let var = f.group("var").map_err(|_| IoError::Missing("var"))?;
    let ds = var
        .dataset("_index")
        .or_else(|_| var.dataset("index"))
        .map_err(|_| IoError::Missing("var/_index"))?;
    let arr = ds.read_1d::<VarLenUnicode>()?;
    Ok(arr.iter().map(|s| s.to_string()).collect())
}

fn read_csr_x_to_dense(xg: &Group, n_var: usize) -> Result<(usize, Vec<f64>), IoError> {
    let indptr: Vec<i32> = xg
        .dataset("indptr")
        .map_err(|_| IoError::Missing("X/indptr"))?
        .read_raw()?;
    let indices: Vec<i32> = xg
        .dataset("indices")
        .map_err(|_| IoError::Missing("X/indices"))?
        .read_raw()?;
    let data: Vec<f64> = xg
        .dataset("data")
        .map_err(|_| IoError::Missing("X/data"))?
        .read_raw()?;
    let n_csr_rows = indptr.len().saturating_sub(1);
    let mut max_ix: usize = 0;
    for &ix in &indices {
        if ix >= 0 {
            max_ix = max_ix.max(ix as usize);
        }
    }

    if let Ok(attr) = xg.attr("shape") {
        let sh: Array1<u64> = attr.read_1d().map_err(|_| IoError::UnsupportedX)?;
        if sh.len() != 2 {
            return Err(IoError::UnsupportedX);
        }
        let n_obs = sh[0] as usize;
        let n_var_sh = sh[1] as usize;
        if n_var_sh != n_var {
            return Err(IoError::UnsupportedX);
        }
        let mut x = vec![0.0_f64; n_obs * n_var];
        if n_csr_rows == n_obs && max_ix < n_var {
            for i in 0..n_obs {
                let a = indptr[i] as usize;
                let b = indptr[i + 1] as usize;
                for k in a..b {
                    let j = indices[k] as usize;
                    if j >= n_var {
                        return Err(IoError::UnsupportedX);
                    }
                    x[i * n_var + j] = data[k];
                }
            }
            return Ok((n_obs, x));
        }
        if n_csr_rows == n_var && max_ix < n_obs {
            for g in 0..n_var {
                let a = indptr[g] as usize;
                let b = indptr[g + 1] as usize;
                for k in a..b {
                    let obs = indices[k] as usize;
                    if obs >= n_obs {
                        return Err(IoError::UnsupportedX);
                    }
                    x[obs * n_var + g] = data[k];
                }
            }
            return Ok((n_obs, x));
        }
        return Err(IoError::UnsupportedX);
    }

    let n_obs = n_csr_rows;
    let mut x = vec![0.0_f64; n_obs * n_var];
    for i in 0..n_obs {
        let a = indptr[i] as usize;
        let b = indptr[i + 1] as usize;
        for k in a..b {
            let j = indices[k] as usize;
            if j >= n_var {
                return Err(IoError::UnsupportedX);
            }
            x[i * n_var + j] = data[k];
        }
    }
    Ok((n_obs, x))
}

pub fn read_anndata_h5ad(path: &Path) -> Result<AnnDataRead, IoError> {
    let f = File::open(path)?;
    let var_names = read_var_index(&f)?;
    let n_var = var_names.len();
    let (n_obs, x) = if let Ok(x_ds) = f.dataset("X") {
        let x_arr: Array2<f64> = x_ds.read_2d()?;
        let nrows = x_arr.nrows();
        if x_arr.ncols() != n_var {
            return Err(IoError::UnsupportedX);
        }
        (nrows, x_arr.into_raw_vec_and_offset().0)
    } else {
        let xg = f.group("X").map_err(|_| IoError::Missing("X"))?;
        read_csr_x_to_dense(&xg, n_var)?
    };
    let obsm = f.group("obsm").map_err(|_| IoError::Missing("obsm"))?;
    let sp = obsm
        .dataset("spatial")
        .map_err(|_| IoError::Missing("obsm/spatial"))?;
    let sp_arr: Array2<f64> = sp.read_2d()?;
    let spatial_dim = sp_arr.ncols();
    if sp_arr.nrows() != n_obs {
        return Err(IoError::UnsupportedX);
    }
    if x.len() != n_obs * n_var {
        return Err(IoError::UnsupportedX);
    }
    let spatial = sp_arr.into_raw_vec_and_offset().0;
    let spatial_distance = if let Ok(obsp) = f.group("obsp") {
        if let Ok(ds) = obsp.dataset("spatial_distance") {
            let d: Array2<f64> = ds.read_2d()?;
            if d.nrows() == n_obs && d.ncols() == n_obs {
                Some(d.into_raw_vec_and_offset().0)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    Ok(AnnDataRead {
        x,
        n_obs,
        n_var,
        var_names,
        spatial,
        spatial_dim,
        spatial_distance,
    })
}

#[derive(serde::Deserialize)]
pub struct Sidecar {
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub pairs: Vec<PairEntry>,
}

#[derive(serde::Deserialize)]
pub struct PairEntry {
    pub ligand: String,
    pub receptor: String,
    #[serde(default)]
    pub pathway: String,
}

pub fn build_s_d_a_from_sidecar(
    adata: &AnnDataRead,
    side: &Sidecar,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize)> {
    let n = adata.n_obs;
    let ng = adata.n_var;
    let ns_s = side.ligands.len();
    let ns_d = side.receptors.len();
    let gene_idx: HashMap<&str, usize> = adata
        .var_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut a = vec![f64::INFINITY; ns_s * ns_d];
    for p in &side.pairs {
        let li = side.ligands.iter().position(|x| x == &p.ligand)?;
        let rj = side.receptors.iter().position(|x| x == &p.receptor)?;
        gene_idx.get(p.ligand.as_str())?;
        gene_idx.get(p.receptor.as_str())?;
        a[li * ns_d + rj] = 1.0;
    }
    let mut s = vec![0.0; n * ns_s];
    let mut d = vec![0.0; n * ns_d];
    for ci in 0..n {
        for j in 0..ns_s {
            let g = *gene_idx.get(side.ligands[j].as_str())?;
            s[ci * ns_s + j] = adata.x[ci * ng + g];
        }
        for j in 0..ns_d {
            let g = *gene_idx.get(side.receptors[j].as_str())?;
            d[ci * ns_d + j] = adata.x[ci * ng + g];
        }
    }
    Some((s, d, a, ns_s, ns_d))
}

use crate::sparse::coo_to_csr;

pub fn coo_to_csr_anndata(coo: &CooMatrix, n: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let csr = coo_to_csr(coo);
    debug_assert_eq!(csr.nrows, n);
    let indptr: Vec<i64> = csr.indptr.iter().map(|&x| x as i64).collect();
    let indices: Vec<i64> = csr.indices.iter().map(|&x| x as i64).collect();
    (indptr, indices, csr.data)
}

fn write_csr_group(
    obsp: &hdf5_metno::Group,
    name: &str,
    coo: &CooMatrix,
    n: usize,
) -> Result<(), IoError> {
    if obsp.link_exists(name) {
        obsp.unlink(name)?;
    }
    let g = obsp.create_group(name)?;
    let (indptr, indices, data) = coo_to_csr_anndata(coo, n);
    let shape = ndarray::array![n as u64, n as u64];
    g.new_attr_builder().with_data(&shape).create("shape")?;
    g.new_attr_builder()
        .with_data("csr_matrix")
        .create("encoding-type")?;
    g.new_dataset_builder()
        .with_data(&indptr[..])
        .create("indptr")?;
    g.new_dataset_builder()
        .with_data(&indices[..])
        .create("indices")?;
    g.new_dataset_builder()
        .with_data(&data[..])
        .create("data")?;
    Ok(())
}

/// Upper bound on allocating a dense `n×n` `f64` distance matrix in RAM. Above this,
/// build pairwise costs via tiled streaming (same numerics, less peak memory).
fn max_dense_dist_matrix_bytes() -> usize {
    std::env::var("COMMOT_MAX_DENSE_DIST_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024)
}

fn should_stream_spatial_distance(adata: &AnnDataRead) -> bool {
    if adata.spatial_distance.is_some() {
        return false;
    }
    if std::env::var("COMMOT_ALWAYS_STREAM_DIST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }
    let n = adata.n_obs;
    let Some(n2) = n.checked_mul(n) else {
        return true;
    };
    let Some(bytes) = n2.checked_mul(std::mem::size_of::<f64>()) else {
        return true;
    };
    bytes > max_dense_dist_matrix_bytes()
}

fn spatial_euclidean_from_coords_cpu(spatial: &[f64], n: usize, dim: usize) -> Vec<f64> {
    euclidean_distance_matrix(spatial, n, dim)
}

#[cfg(feature = "gpu")]
fn spatial_euclidean_from_coords_maybe_gpu(
    spatial: &[f64],
    n: usize,
    dim: usize,
    prefer_gpu: bool,
) -> Vec<f64> {
    if prefer_gpu {
        if let Ok(m) = crate::gpu::spatial_pairwise_costs_f64(spatial, n, dim, false) {
            return m;
        }
    }
    spatial_euclidean_from_coords_cpu(spatial, n, dim)
}

#[cfg(not(feature = "gpu"))]
fn spatial_euclidean_from_coords_maybe_gpu(
    spatial: &[f64],
    n: usize,
    dim: usize,
    _prefer_gpu: bool,
) -> Vec<f64> {
    spatial_euclidean_from_coords_cpu(spatial, n, dim)
}

pub fn write_commot_obsp(
    input_path: &Path,
    output_path: &Path,
    database_name: &str,
    dis_thr: f64,
    cost_euc_square: bool,
    cot_eps_p: f64,
    cot_rho: f64,
    cot_nitermax: usize,
    cot_weights: [f64; 4],
    side: &Sidecar,
    _pathway_sum: bool,
    prefer_gpu_distance: bool,
) -> Result<(), IoError> {
    let adata = read_anndata_h5ad(input_path)?;
    let (s, d, a, ns_s, ns_d) =
        build_s_d_a_from_sidecar(&adata, side).ok_or(IoError::Missing("sidecar/genes"))?;
    let n = adata.n_obs;
    let thr = if cost_euc_square {
        dis_thr * dis_thr
    } else {
        dis_thr
    };
    let cutoff: Vec<f64> = (0..ns_s * ns_d).map(|_| thr).collect();
    let use_streaming = should_stream_spatial_distance(&adata);
    let network = if use_streaming {
        run_cot_combine_streaming(
            &s,
            &d,
            &a,
            &adata.spatial,
            &cutoff,
            n,
            adata.spatial_dim,
            ns_s,
            ns_d,
            cot_eps_p,
            cot_rho,
            cot_nitermax,
            cot_weights,
            cost_euc_square,
        )
    } else {
        let mut dist = if let Some(sd) = &adata.spatial_distance {
            sd.clone()
        } else {
            spatial_euclidean_from_coords_maybe_gpu(
                &adata.spatial,
                n,
                adata.spatial_dim,
                prefer_gpu_distance,
            )
        };
        if cost_euc_square {
            dist = square_distance_matrix(&dist, n);
        }
        run_cot_combine(
            &s,
            &d,
            &a,
            &dist,
            &cutoff,
            n,
            ns_s,
            ns_d,
            cot_eps_p,
            cot_rho,
            cot_nitermax,
            cot_weights,
        )
    }
    .map_err(|_| IoError::UnsupportedX)?;
    std::fs::copy(input_path, output_path)?;
    let f = File::open_rw(output_path)?;
    let obsp = match f.group("obsp") {
        Ok(g) => g,
        Err(_) => f.create_group("obsp")?,
    };
    for i in 0..ns_s {
        for j in 0..ns_d {
            if is_inf_entry(a[i * ns_d + j]) {
                continue;
            }
            let coo = network.get(&(i, j)).unwrap();
            let key = format!(
                "commot-{}-{}-{}",
                database_name, side.ligands[i], side.receptors[j]
            );
            write_csr_group(&obsp, &key, coo, n)?;
        }
    }
    Ok(())
}
