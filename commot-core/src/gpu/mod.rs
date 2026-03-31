//! Optional GPU helpers (feature `gpu`). See module-level notes on when this pays off.
//!
//! ## Where GPU parallelism helps in COMMOT
//!
//! 1. **Pairwise spatial cost matrix** ([`spatial_pairwise_costs_f64`]) — \(O(n^2 \cdot d)\) dense
//!    work with a regular memory pattern. This dominates preprocessing when `obsp/spatial_distance`
//!    is absent. A 2D grid compute shader maps naturally to \((i,j)\) cell pairs.
//!
//! 2. **Per-iteration Sinkhorn on fixed CSR pattern** — each iteration is \(O(\mathrm{nnz})\)
//!    element-wise `exp` plus row/column reductions. GPUs can help for very large `nnz`, but
//!    host–device transfer and small sparse batches often favor CPU; not implemented here.
//!
//! 3. **Many independent OT subproblems** (`cot_*` over LR pairs) — already parallelized on CPU
//!    with Rayon; batching many small Sinkhorns on GPU would need a different algorithmic layout
//!    (padding / bucketing) to avoid warp divergence.
//!
//! 4. **`build_m_max_sp`** — dense threshold filter over positions; could use GPU compare+compact,
//!    but output is COO and CPU path is often acceptable relative to OT.

mod distance;

pub use distance::{spatial_pairwise_costs_f64, GpuDistanceError};
