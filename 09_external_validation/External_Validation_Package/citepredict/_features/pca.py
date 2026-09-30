"""HVG-PCA projection (50 PCs) into the reference PCA space.

Reference files (data/HVG_PCA/): hvg_genes.csv, gene_scaling.csv (per-gene mean/std
used to centre and scale), pca_loadings.csv (genes x PC1..PC50).
Genes absent from the query dataset enter as 0 expression.
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .._data import data_path
from .._keys import OBSM_PCA


def compute_pca(adata, X_log) -> None:
    hvg_genes = pd.read_csv(data_path("HVG_PCA", "hvg_genes.csv"))["gene"].tolist()
    scaling   = pd.read_csv(data_path("HVG_PCA", "gene_scaling.csv"), index_col=0)
    loadings  = pd.read_csv(data_path("HVG_PCA", "pca_loadings.csv"), index_col=0)

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    present = [(j, gene_to_idx[g]) for j, g in enumerate(hvg_genes) if g in gene_to_idx]
    print(f"[pca] {len(present)} / {len(hvg_genes)} HVGs found in dataset  |  {loadings.shape[1]} PCs")

    X_hvg = np.zeros((adata.n_obs, len(hvg_genes)), dtype=np.float32)
    if present:
        ref_idx, dat_idx = map(list, zip(*present))
        sub = X_log[:, dat_idx]
        X_hvg[:, ref_idx] = sub.toarray() if issparse(sub) else np.asarray(sub)

    mean = scaling.loc[hvg_genes, "mean"].to_numpy(np.float32)
    std  = scaling.loc[hvg_genes, "std"].to_numpy(np.float32)
    std  = np.where(std > 0, std, 1.0)
    L    = loadings.loc[hvg_genes].to_numpy(np.float32)

    adata.obsm[OBSM_PCA] = (((X_hvg - mean) / std) @ L).astype(np.float32)
    print(f"[pca] obsm['{OBSM_PCA}'] {adata.obsm[OBSM_PCA].shape}")
