# COMMOT Implementation Reference

> **Source:** Cang, Z. et al. "Screening cell–cell communication in spatial transcriptomics via collective optimal transport." *Nature Methods* 20, 218–228 (2023). [doi:10.1038/s41592-022-01728-4](https://doi.org/10.1038/s41592-022-01728-4)

---

## 1. Problem Statement

Given a spatial transcriptomics dataset, infer **cell–cell communication (CCC)** by simultaneously considering:

1. **Multiple ligand–receptor (LR) pairs** that compete for binding
2. **Spatial distance constraints** (signaling only occurs within limited ranges)
3. **Mass conservation** — total transported signal cannot exceed available ligand or receptor amounts

The output is a 4D tensor **P\*** ∈ ℝ^(n_l × n_r × n_s × n_s) where P\*_{i,j,k,l} scores signaling strength from sender cell *k* to receiver cell *l* through ligand *i* and receptor *j*.

---

## 2. Inputs

| Symbol | Description |
|--------|-------------|
| n_s | Number of spatial locations (cells or spots) |
| n_l | Number of ligand species |
| n_r | Number of receptor species |
| **X^L** ∈ ℝ^(n_l × n_s) | Expression matrix for ligands; X^L_{i,k} = expression of ligand *i* at spot *k* |
| **X^R** ∈ ℝ^(n_r × n_s) | Expression matrix for receptors; X^R_{j,l} = expression of receptor *j* at spot *l* |
| **I** ⊂ {1..n_l} × {1..n_r} | Index set of LR pairs that can bind (from database, e.g. CellChatDB) |
| **D** ∈ ℝ^(n_s × n_s) | Euclidean distance matrix between spots |
| T_{(i,j)} | Spatial signaling range for LR pair (i,j) — distances beyond this are set to ∞ |
| φ(·) | Distance scaling function (square or exponential) |

### Heteromeric complexes

When a receptor (or ligand) is a multi-subunit complex, use the **minimum** expression across subunits to represent the effective amount:

```
X^R_{j,l} = min(subunit_1_expr, subunit_2_expr, ...)
```

### Cost matrix construction

For each LR pair (i,j), construct a species-specific cost matrix:

```
C_{(i,j)}[k,l] = φ(D[k,l])   if D[k,l] ≤ T_{(i,j)}
             = ∞             otherwise
```

Common choices for φ: `φ(d) = d²` or `φ(d) = exp(d)`.

---

## 3. Core Optimization: Collective Optimal Transport

### 3.1 Original formulation (Eq. 1)

```
min_{P ∈ Γ}  Σ_{(i,j) ∈ I}  ⟨P_{i,j,·,·}, C_{(i,j)}⟩_F  +  Σ_i F(μ_i)  +  Σ_j F(ν_j)
```

where the feasible set Γ enforces:

```
Γ = { P ∈ ℝ^(n_l × n_r × n_s × n_s)_+  :
        P_{i,j,·,·} = 0           for (i,j) ∉ I,           // non-binding pairs have zero transport
        Σ_{j,l} P_{i,j,k,l} ≤ X^L_{i,k},                   // ligand capacity constraint
        Σ_{i,k} P_{i,j,k,l} ≤ X^R_{j,l}  }                 // receptor capacity constraint
```

The untransported mass (slack):

```
μ_i(k) = X^L_{i,k} − Σ_{j,l} P_{i,j,k,l}
ν_j(l) = X^R_{j,l} − Σ_{i,k} P_{i,j,k,l}
```

F(·) penalizes untransported mass. The key insight: inequality constraints on marginals (≤ rather than =) with a penalty on slack allows the method to avoid normalizing distributions to probabilities, preserving comparability across species.

### 3.2 Reshaped formulation with entropy regularization (Eq. 2)

Reshape the 4D tensor P into a 2D matrix P̂ where:

```
P̂[(i-1)*n_s + k, (j-1)*n_s + l] = P_{i,j,k,l}
```

This gives P̂ ∈ ℝ^(m × n) where m = n_l * n_s, n = n_r * n_s.

Similarly reshape the cost matrix Ĉ, setting Ĉ entries to ∞ for non-binding LR pairs.

Construct marginal vectors:

```
a[(i-1)*n_s + k] = X^L_{i,k}       (length m = n_l * n_s)
b[(j-1)*n_s + l] = X^R_{j,l}       (length n = n_r * n_s)
```

The optimization becomes:

```
min_{P̂, μ̂, ν̂ ≥ 0}  ⟨P̂, Ĉ⟩_F  +  ε_p H(P̂)  +  ε_μ H(μ̂)  +  ε_ν H(ν̂)  +  ρ(‖μ̂‖_1 + ‖ν̂‖_1)

s.t.  P̂ 1_n = a − μ̂
      P̂^T 1_m = b − ν̂
```

where H(x) = Σ_i x_i (ln(x_i) − 1) is the entropy regularization.

**Parameters:**
- ε_p, ε_μ, ε_ν: entropy regularization coefficients (set equal: ε = ε_p = ε_μ = ε_ν in the paper's implementation)
- ρ: penalty weight on untransported mass (controls the trade-off between transporting more mass vs. cost)

---

## 4. Solver Algorithm: Stabilized Sinkhorn Iteration (Eq. 3)

When ε = ε_p = ε_μ = ε_ν, the problem is solved via **stabilized log-domain Sinkhorn iterations**:

```
Initialize f^(0) and g^(0) arbitrarily (e.g., zeros)

For l = 0, 1, 2, ... until convergence:

    f^(l+1) ← ε * log(a) + f^(l) − ε * log(
        exp(f^(l)/ε) ⊙ exp(−C/ε) @ exp(g^(l)/ε)  +  exp((f^(l) − ρ)/ε)
    )

    g^(l+1) ← ε * log(b) + g^(l) − ε * log(
        exp(g^(l)/ε) ⊙ exp(−C^T/ε) @ exp(f^(l+1)/ε)  +  exp((g^(l) − ρ)/ε)
    )
```

where:
- `⊙` is element-wise (Hadamard) product
- `@` is matrix-vector multiplication (summing over the appropriate axis)
- All operations are element-wise unless otherwise noted
- The `+ exp((f−ρ)/ε)` term handles the slack / untransported mass

**Final solution:**

```
P̂* = exp((f ⊕ g − C) / ε)
```

where `f ⊕ g` is the outer sum: `(f ⊕ g)[k,l] = f[k] + g[l]`.

### Implementation notes

1. **Log-domain stabilization:** The algorithm works in log-space (with f, g as log-domain dual variables) to avoid numerical overflow/underflow from exponentiating large cost values.

2. **Sparsity from ∞ costs:** Entries where C = ∞ result in exp(−∞/ε) = 0, so the matrix-vector products are effectively sparse. Only entries where D[k,l] ≤ T_{(i,j)} contribute. This is critical for performance.

3. **Convergence:** Check marginal constraint violation or change in f, g between iterations.

4. **Memory:** Only store finite values of C and non-zero values of P̂. Both scale linearly with the number of spatial locations (due to the spatial range constraint making the transport plan sparse).

5. **Time complexity:** Scales linearly with the number of non-zero elements in the CCC matrix (confirmed in Supplementary Fig. 35).

---

## 5. Practical Sparse Implementation Strategy

The full Ĉ matrix is (n_l * n_s) × (n_r * n_s) which can be enormous. In practice:

- For each LR pair (i,j) ∈ I, the cost sub-block C_{(i,j)} is n_s × n_s but sparse (only entries where D[k,l] ≤ T_{(i,j)} are finite).
- For (i,j) ∉ I, the entire sub-block is ∞ (zero contribution).
- The Sinkhorn matrix-vector products `exp(−C/ε) @ v` decompose into per-LR-pair sparse operations:

```
For each LR pair (i,j) ∈ I:
    K_{(i,j)}[k,l] = exp(−C_{(i,j)}[k,l] / ε)   // sparse kernel, only for D[k,l] ≤ T
    
    // The row-sum for the f-update at index (i, k):
    result[(i-1)*n_s + k] = Σ_{ (j): (i,j)∈I }  Σ_{ l: D[k,l] ≤ T }  K_{(i,j)}[k,l] * exp(g[(j-1)*n_s + l] / ε)
```

This avoids ever materializing the full (n_l * n_s) × (n_r * n_s) matrix.

---

## 6. Downstream Analyses

### 6.1 Spatial Signaling Direction (Vector Field)

Given a CCC matrix **S** ∈ ℝ^(n_s × n_s) (for a specific LR pair or pathway, obtained by summing relevant P slices), compute vector fields:

**Sending direction** (direction to which spot *i* sends signal):

```
V^s_i = (Σ_j S_{i,j}) * N( Σ_{j ∈ N^s_i} S_{i,j} * (x_j − x_i) )
```

**Receiving direction** (direction from which spot *i* receives signal):

```
V^r_i = (Σ_j S_{j,i}) * N( Σ_{j ∈ N^r_i} S_{j,i} * (x_i − x_j) )
```

where:
- N(x) = x / ‖x‖ (unit vector normalization)
- N^s_i = index set of top-k spots with largest values in row i of S
- N^r_i = index set of top-k spots with largest values in column i of S
- x_i = spatial coordinates of spot i

### 6.2 Cluster-Level CCC

Aggregate spot-level CCC matrix S to cluster-level S^cl:

```
S^cl_{i,j} = Σ_{(k,l) ∈ I^cl_{i,j}} S_{k,l} / |I^cl_{i,j}|
```

where I^cl_{i,j} = {(k,l) : L_k = i, L_l = j} and L_k is the cluster label of spot k.

**Significance testing:** Permute cluster labels n times, compute percentile of original S^cl in the permuted distribution → p-value.

### 6.3 Downstream Gene Analysis

1. **Received signal per spot:** r_i = Σ_j S_{j,i}

2. **Differential expression w.r.t. CCC:** Use tradeSeq-style analysis with received signal r as the cofactor (analogous to pseudotime DE testing).

3. **Random forest prioritization:** Train a random forest where:
   - Output: potential target gene expression
   - Input features: received signal r + top intracellularly correlated genes
   - Feature importance (Gini importance) of r quantifies unique CCC impact on the target gene

---

## 7. Evaluation Metrics

### 7.1 Signaling direction robustness (cosine distance)

```
d_cos(V_full, V_sub) = Σ_i ‖V_full(i)‖ * [1 − V_full(i)·V_sub(i) / (‖V_full(i)‖ * ‖V_sub(i)‖)] / Σ_i ‖V_full(i)‖
```

Weighted cosine distance — spots with stronger signals contribute more.

### 7.2 Cluster-level CCC robustness (Jaccard distance)

```
d_Jaccard(S^cl_1, S^cl_2) = 1 − |S̄^cl_1 ∩ S̄^cl_2| / |S̄^cl_1 ∪ S̄^cl_2|
```

where S̄^cl are binarized edge sets (edges with p < 0.05 kept).

### 7.3 Correlation with known targets (Spearman's ρ)

```
ρ = cov(R(X^LR), R(X^tgt)) / (σ_{R(X^LR)} * σ_{R(X^tgt)})
```

where X^LR_i = average received signal in cluster i, X^tgt_i = activity of known target genes (% DE genes) in cluster i.

Reported median Spearman correlations on three real datasets: **0.237, 0.180, 0.230**.

---

## 8. Validation Strategy (for testing your implementation)

### 8.1 PDE-based synthetic data

The paper validates against a PDE model of ligand diffusion + binding:

```
∂[L_i]/∂t = D∇²[L_i] − a_i[L_i][R] + b_i[L_i R] − c_i[L_i]
∂[L_i R]/∂t = a_i[L_i][R] − b_i[L_i R]
∂[R]/∂t = Σ_i (−a_i[L_i][R] + b_i[L_i R])
```

where D = diffusion coefficient, a_i = binding rate, b_i = dissociation rate, c_i = degradation rate.

**Test cases:** 10 cases of increasing complexity (1–10 LR pairs, varying binding patterns). Compare COMMOT output to PDE ground truth via Spearman correlation and RMSE.

### 8.2 Robustness via subsampling

Subsample cells at various percentages (50%–95%) and compare:
- Signaling direction via cosine distance
- Cluster-level CCC via Jaccard distance
- DE gene overlap via Jaccard index

### 8.3 Real data benchmarks used

| Dataset | Technology | Genes | Cells/Spots | Resolution |
|---------|-----------|-------|-------------|------------|
| Mouse hypothalamic preoptic | MERFISH | 161 | 73,655 | Single-cell |
| Mouse placenta | STARmap | 903 | 7,203 | Single-cell |
| Mouse somatosensory cortex | seqFISH+ | 10,000 | 523 | Single-cell |
| Mouse hippocampus | Slide-seqV2 | 23,264 | 53,173 | Near-single-cell |
| Human breast cancer | Visium | 36,601 | 3,798 | Multi-cell |
| Mouse brain (sagittal) | Visium | 32,285 | 3,355 | Multi-cell |
| Drosophila embryo | In silico (SpaOTsc) | — | — | Single-cell |
| Human epidermis | In silico (SpaOTsc) | — | — | Single-cell |

---

## 9. Key Hyperparameters

| Parameter | Role | Notes |
|-----------|------|-------|
| ε (epsilon) | Entropy regularization | Smooths the solution; smaller ε → sharper but harder to converge |
| ρ (rho) | Penalty on untransported mass | Larger ρ → more mass transported; smaller ρ → more selective |
| T_{(i,j)} | Spatial signaling range | Uniform large value recommended for screening; refine per-pair later |
| φ | Distance scaling | Square or exponential |
| k (top-k) | For vector field computation | Number of top signal-sending/receiving neighbors |

---

## 10. Ligand–Receptor Database

The paper uses **CellChatDB** (secreted signaling category):
- 1,735 secreted LR pairs in Fantom5
- 72% of ligands (372/516) and 60% of receptors (309/512) bind multiple species
- This multi-species binding is exactly why collective OT is needed over pairwise OT

Available at: http://www.cellchat.org/cellchatdb/

---

## 11. Architecture Summary for Rust Implementation

```
Core modules:
├── cost_matrix
│   ├── euclidean_distance(coords) → D
│   ├── threshold_cost(D, T, φ) → sparse C_{(i,j)}
│   └── build_sparse_kernel(C, ε) → sparse K_{(i,j)}
│
├── collective_ot
│   ├── reshape_marginals(X_L, X_R) → (a, b)
│   ├── sinkhorn_iteration(a, b, sparse_kernels, ε, ρ, max_iter, tol) → (f, g)
│   └── reconstruct_transport(f, g, sparse_costs, ε) → sparse P*
│
├── aggregation
│   ├── sum_lr_pair(P*, i, j) → S_{(i,j)} ∈ ℝ^(n_s × n_s)
│   ├── sum_pathway(P*, pairs) → S_pathway
│   └── received_signal(S) → r ∈ ℝ^n_s
│
├── downstream
│   ├── signaling_direction(S, coords, k) → (V_send, V_recv)
│   ├── cluster_ccc(S, labels, n_permutations) → (S_cl, p_values)
│   └── de_gene_analysis(r, expression_matrix) → DE results
│
└── io
    ├── read_anndata / h5ad
    ├── read_cellchatdb
    └── write_results
```

### Key data structures

```rust
// Sparse cost/kernel for one LR pair between spots
struct SparseLRKernel {
    pair: (usize, usize),          // (ligand_idx, receptor_idx)
    row_indices: Vec<usize>,       // sender spot indices
    col_indices: Vec<usize>,       // receiver spot indices  
    values: Vec<f64>,              // cost or kernel values
    // CSR or COO format
}

// The dual variables
struct SinkhornState {
    f: Vec<f64>,  // length n_l * n_s
    g: Vec<f64>,  // length n_r * n_s
}

// Sparse transport plan (output)
struct TransportPlan {
    // Per LR pair, sparse n_s × n_s matrix
    plans: HashMap<(usize, usize), SparseMatrix>,
}
```

### Performance-critical operations

1. **Sparse matrix-vector multiply** in each Sinkhorn iteration (the inner loop)
2. **Log-sum-exp** with numerical stability (for the log-domain updates)
3. **Distance computation** — build spatial neighbor graph once, reuse for all LR pairs

---

## 12. Comparison: COMMOT vs. Simpler Approaches

| Method | Handles competition? | Spatial constraint? | Normalization-free? |
|--------|---------------------|--------------------|--------------------|
| **COMMOT (Collective OT)** | ✅ Yes | ✅ Yes (hard cutoff) | ✅ Yes |
| Pairwise OT | ❌ No | ✅ Yes | ✅ Yes |
| Unbalanced OT | ❌ No | Soft (KL) | Partially (can exceed supply) |
| Partial OT | ❌ No | ✅ Yes | Requires total mass param |
| CellChat | ❌ No | ❌ No (non-spatial) | N/A |
| Giotto | ❌ No | KNN graph | N/A |
| CellPhoneDB v3 | ❌ No | Cluster proximity | N/A |

COMMOT's key advantage: coupling between one species pair affects all other couplings (competition), which cannot be realized in standard OT applied pair-by-pair.

---

## 13. Known Limitations

1. Operates on **mRNA expression** as a proxy for protein abundance — cannot capture post-translational modifications (phosphorylation, glycosylation, cleavage, dimerization)
2. Spatial distance constraint T is hard to accurately estimate per LR pair; a uniform large value is recommended for initial screening
3. The method emphasizes **local short-range interactions** even when T is increased (shown in Supplementary Fig. 36)
4. False positives are inherent — results should be experimentally validated

---

## 14. Reference Implementation

- **Python package:** https://github.com/zcang/COMMOT
- **Reproduction code:** https://doi.org/10.5281/zenodo.7272562
- Dependencies: POT (Python Optimal Transport), scikit-learn, tradeSeq (R)
