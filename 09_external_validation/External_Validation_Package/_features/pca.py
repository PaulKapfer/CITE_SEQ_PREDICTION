"""
Feature 2 – HVG-based PCA projection (50 components).

Projects new cells into the reference PCA space using stored gene-scaling
parameters and PCA loadings.

Reference files used (all in data/HVG_PCA/):
  hvg_genes.csv     – HVG gene names
  gene_scaling.csv  – per-gene mean and std for centering/scaling
  pca_loadings.csv  – genes × PC loading matrix (index = gene, columns = PC1..PC50)

Result stored in adata.obsm["X_External_Validation_Package_PCA"]  shape (n_cells, 50).
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .._data import data_path


def compute_pca(adata) -> None:
    """Project cells into reference PCA space. Adds adata.obsm['X_External_Validation_Package_PCA']."""

    print("[pca] Loading HVG-PCA reference artefacts...")
    hvg_df      = pd.read_csv(data_path("HVG_PCA", "hvg_genes.csv"))
    scaling_df  = pd.read_csv(data_path("HVG_PCA", "gene_scaling.csv"), index_col=0)
    loadings_df = pd.read_csv(data_path("HVG_PCA", "pca_loadings.csv"), index_col=0)

    hvg_genes = hvg_df["gene"].tolist()
    n_pcs     = loadings_df.shape[1]
    print(f"[pca] {len(hvg_genes)} HVGs  |  {n_pcs} PCs")

    # ── Prepare expression matrix ─────────────────────────────────────────────
    if "counts" in adata.layers:
        import anndata as ad
        import scanpy as sc
        from scipy.sparse import csr_matrix
        X_c = adata.layers["counts"].astype(np.float32)
        if not issparse(X_c):
            X_c = csr_matrix(X_c)
        tmp = ad.AnnData(X=X_c, obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        X_log = tmp.X
        del tmp
    else:
        X_log = adata.X

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    n_cells     = adata.n_obs

    # Build HVG sub-matrix; fill genes absent in this dataset with 0.
    # Gather all present genes in one index operation instead of looping column-by-column.
    X_hvg = np.zeros((n_cells, len(hvg_genes)), dtype=np.float32)
    hvg_ref_idx, dat_idx = zip(
        *[(j, gene_to_idx[g]) for j, g in enumerate(hvg_genes) if g in gene_to_idx]
    ) if any(g in gene_to_idx for g in hvg_genes) else ([], [])

    if dat_idx:
        X_sub = X_log[:, list(dat_idx)]
        if issparse(X_sub):
            X_sub = np.asarray(X_sub.todense())
        X_hvg[:, list(hvg_ref_idx)] = X_sub.astype(np.float32)

    # ── Center and scale ──────────────────────────────────────────────────────
    gene_mean = scaling_df.loc[hvg_genes, "mean"].values.astype(np.float32)
    gene_std  = scaling_df.loc[hvg_genes, "std"].values.astype(np.float32)
    gene_std  = np.where(gene_std > 0, gene_std, 1.0)
    X_scaled  = (X_hvg - gene_mean) / gene_std

    # ── Apply loadings  (n_cells × n_hvg) @ (n_hvg × n_pcs) → (n_cells × n_pcs) ──
    L         = loadings_df.loc[hvg_genes].values.astype(np.float32)
    pc_coords = (X_scaled @ L).astype(np.float32)

    adata.obsm["X_External_Validation_Package_PCA"] = pc_coords
    print(f"[pca] Stored adata.obsm['X_External_Validation_Package_PCA']: {pc_coords.shape}")
