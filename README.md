# COMMOT
Screening cell-cell communication in spatial transcriptomics via collective optimal transport 

[![PyPI](https://img.shields.io/pypi/v/commot?logo=PyPI)](https://pypi.org/project/commot)
![pytest](https://github.com/zcang/COMMOT/actions/workflows/python-package.yml/badge.svg)
[![Read the Docs](https://readthedocs.org/projects/commot/badge/?version=latest)](https://commot.readthedocs.io/en/latest/)


## Installation
Install from PyPI by `pip install commot` or install from source by cd to this directory and `pip install .`

[Optional] Install [tradeSeq](https://github.com/statOmics/tradeSeq) in R to analyze the CCC differentially expressed genes. \
Currently, tradeSeq version 1.0.1 with R version 3.6.3 has been tested to work. \
In order for the R-python interface to work properly, rpy2==3.4.2 and anndata2ri==1.0.6 should be installed.
To use this feature, install from PyPI by `pip install commot[tradeSeq]` or from source by `pip install .[tradeSeq]`.

## Usage
**Basic usage**

_Import packages_
```
import commot as ct
import scanpy as sc
import pandas as pd
import numpy as np
```
_Load a spatial dataset_ \
(e.g., a Visium dataset)
```
adata = sc.datasets.visium_sge(sample_id='V1_Mouse_Brain_Sagittal_Posterior')
adata.var_names_make_unique()
```
_Basic processing_
```
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
```
_Specify ligand-receptor pairs_
```
LR=np.array([['Tgfb1', 'Tgfbr1_Tgfbr2', 'Tgfb_pathway'],['Tgfb2', 'Tgfbr1_Tgfbr2', 'Tgfb_pathway'],['Tgfb3', 'Tgfbr1_Tgfbr2', 'Tgfb_pathway']],dtype=str)
df_ligrec = pd.DataFrame(data=LR)
```
(or use pairs from a ligand-receptor database `df_ligrec=ct.pp.ligand_receptor_database(database='CellChat', species='mouse')`.)

_Construct CCC networks_ \
Use collective optimal transport to construct CCC networks for the ligand-receptor pairs with a spatial distance constraint of 200 (coupling between cells with distance greater than 200 is prohibited). For example, the spot-by-spot matrix for the pair Tgfb1 (ligand) and Tgfbr1_Tgfbr2 (receptor)is stored in `adata.obsp['commot-user_database-Tgfb1-Tgfbr1_Tgfbr2']`. The total sent or received signal for each pair is stored in `adata.obsm['commot-user_database-sum-sender']` and `adata.obsm['commot-user_database-sum-receiver']`.
```
ct.tl.spatial_communication(adata,
    database_name='user_database', df_ligrec=df_ligrec, dis_thr=200, heteromeric=True)
```
## Rust port (experimental)

The workspace adds a fast path for the collective OT core and a small CLI:

- `commot-core` — sparse `cot_combine_sparse`, L1-penalty Sinkhorn matching Python `commot._optimal_transport._unot` / `_cot` (not the KL `wass` formulation; see `commot-core/tests/wass_probe.rs`). Sinkhorn uses a CSR fast path when COO coordinates are unique; the four COT variants run in parallel via Rayon.
- `commot-cli` — reads dense `AnnData` `.h5ad` (`X`, `obsm/spatial`), merges LR tables from a JSON sidecar, writes `obsp/commot-{db}-{lig}-{rec}` CSR groups.

**Large-\(n\) streaming:** If `obsp/spatial_distance` is absent, distances default to a **dense** `n×n` matrix when it fits under **`COMMOT_MAX_DENSE_DIST_BYTES`** (default **512 MiB**, i.e. about \(n \lesssim 8000\) for `f64`). Above that budget, pairwise costs are built in **tiles** (streaming: same cutoffs and numerics as dense, no full distance buffer). Set **`COMMOT_ALWAYS_STREAM_DIST=1`** to force the tiled path. If a precomputed `obsp/spatial_distance` dataset exists, behavior is unchanged (dense path from file). For smaller \(n\), the CLI still optionally uses `--gpu-distance` when built with the `gpu` feature.

**GPU (optional):** When the dense distance path is chosen, building the \(n \times n\) Euclidean matrix from `obsm/spatial` is \(O(n^2 \cdot d)\) and may suit a compute shader. Enable `commot-core`’s `gpu` feature (wgpu): `cargo test -p commot-core --features gpu` runs parity checks vs the CPU reference. CLI: `cargo build -p commot-cli --release --features gpu` and `--gpu-distance`. Collective OT (Sinkhorn / `cot_*`) stays on the CPU; see `commot-core/src/gpu/mod.rs` for what is and is not a good GPU fit.

**HDF5:** `hdf5-metno` links the system HDF5 library. On macOS with Homebrew:

```bash
export HDF5_DIR="$(brew --prefix hdf5)"
cargo build --workspace
```

**Fixtures and parity:** Regenerate NumPy goldens after changing the Python OT code:

```bash
uv run --with numpy --with scipy --with pandas python scripts/export_parity_fixtures.py
cargo test -p commot-core --test parity
cargo test -p commot-core --test streaming_parity
```

Dense vs tiled distance + COT is checked in `streaming_parity`; Python vs Rust on real h5ad can use `scripts/benchmark_kidney_h5ad.py` or `benchmark_slideseq_scaling.py` (on macOS the script uses `/usr/bin/time -l` so Rust RSS is captured correctly). For very large \(n\), Python may run out of memory building a dense distance matrix; use subsampling in those scripts or compare Rust streaming output to a sparse/downsized Python baseline.

**Scaling snapshot (example runs):** Slide-seq brain (~27k cells max in file): Rust wall time ~3.9× faster than Python at \(10^4\) cells with `dis_thr=4000` (12 LR pairs); parity at \(n=100\) matched exactly (\(L_1=0\)). VisiumHD lymph (\(\sim\)89k cells): same parity at \(n=100\); at \(n=20{,}000\) Python completed COT in \(\sim\)218 s vs Rust \(\sim\)30 s (\(\sim\)7×) on one workstation. Full \(\sim\)89k-cell runs can still be killed by the OS if the *sparse* OT intermediates (many pairs within `dis_thr`) exceed RAM—that is separate from the dense distance matrix.

**Mock h5ad + sidecar for the CLI:**

```bash
uv run --with anndata --with pandas --with numpy python scripts/prepare_commot_inputs.py
cargo build -p commot-cli --release
./target/release/commot-cli --input-h5ad tests/fixtures/mock_adata.h5ad \
  --output-h5ad /tmp/out.h5ad --sidecar-json tests/fixtures/mock_sidecar.json --database-name testdb
```

**Benchmarks:** `cargo bench -p commot-core --bench cot_bench` · scaling Python vs Rust by cell count: real data `... benchmark_scale_cells.py --h5ad /path/to/file.h5ad --species human`, or synthetic AnnData (no file) `... benchmark_scale_cells.py --synthetic --max-cells 15000 --verify-first-n` (log-spaced grid; JSON includes `python_over_rust_cot_ratio`; use `--help`).

**CPU flamegraph (optional):** `cargo build -p commot-cli --release --features pprof`, then add `--pprof-flamegraph /tmp/commot.svg` to the CLI invocation ([tikv/pprof-rs](https://github.com/tikv/pprof-rs)).

**Python vs Rust timing / matrix check (uses h5py for Rust output):**

```bash
uv run --with numpy --with scipy --with pandas --with anndata --with h5py python scripts/compare_rust_python_timing.py
```

## Documentation

See detailed documentation and examples at [https://commot.readthedocs.io/en/latest/index.html](https://commot.readthedocs.io/en/latest/index.html).

## References

Cang, Z., Zhao, Y., Almet, A.A. et al. Screening cell–cell communication in spatial transcriptomics via collective optimal transport. *Nat Methods* 20, 218–228 (2023). https://doi.org/10.1038/s41592-022-01728-4
