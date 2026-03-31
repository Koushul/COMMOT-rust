pub mod cot;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
pub mod pipeline;
pub mod sparse;
pub mod unot;

pub use cot::{
    build_m_max_sp, build_m_max_sp_from_spatial, cot_blk_sparse, cot_col_sparse,
    cot_combine_sparse, cot_combine_sparse_from_m_max, cot_row_sparse, cot_sparse, is_inf_entry,
};
pub use io::{
    build_s_d_a_from_sidecar, read_anndata_h5ad, write_commot_obsp, AnnDataRead, IoError, Sidecar,
};
pub use pipeline::{
    euclidean_distance_matrix, run_cot_combine, run_cot_combine_streaming, square_distance_matrix,
};
pub use sparse::{coo_submatrix_pull, CooMatrix, CsrMatrix};
pub use unot::unot_sinkhorn_l1_sparse;

#[cfg(feature = "gpu")]
pub use gpu::{spatial_pairwise_costs_f64, GpuDistanceError};
