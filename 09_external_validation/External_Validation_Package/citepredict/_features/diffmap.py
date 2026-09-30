"""
Feature 3 – RPG diffusion-map coordinates (25 DC dimensions).

Maps new cells into the reference RPG co-expression diffusion manifold using
Nyström landmark-cell interpolation.

Algorithm
---------
1. Load the reference RPG gene list and pairwise-pair names.
2. Compute per-cell pairwise-product co-expression for the same RPG pairs
   (upper triangle incl. diagonal, identical to reference_preparation.py).
3. For each new cell, compute adaptive Gaussian kernel weights against all
   landmark cells:
       k(new, lm_j) = exp( -||x_new - x_lm_j||² / (σ_global × σ_lm_j) )
4. Interpolate DC coordinates as a weighted average over landmarks.

Reference files used (data/RPG_diffmap/):
  rpg_gene_list.csv, rpg_pair_names.csv,
  landmark_rpg_vectors.csv, landmark_dc_coords.csv,
  landmark_local_sigma.csv, landmark_sigma_global.txt

Genes absent from the query dataset enter as 0 expression.
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .._data import data_path
from .._keys import OBSM_DIFFMAP

_CHUNK = 2000   # cells processed per iteration (memory / speed trade-off)


def compute_diffmap(adata, X_log) -> None:

    print("[diffmap] Loading RPG diffusion-map reference artefacts...")

    rpg_genes   = pd.read_csv(data_path("RPG_diffmap", "rpg_gene_list.csv"))["gene"].tolist()
    pair_names  = pd.read_csv(data_path("RPG_diffmap", "rpg_pair_names.csv"))["pair"].tolist()
    lm_rpg_df   = pd.read_csv(data_path("RPG_diffmap", "landmark_rpg_vectors.csv"), index_col=0)
    lm_dc_df    = pd.read_csv(data_path("RPG_diffmap", "landmark_dc_coords.csv"),   index_col=0)
    lm_sigma_df = pd.read_csv(data_path("RPG_diffmap", "landmark_local_sigma.csv"))

    with open(data_path("RPG_diffmap", "landmark_sigma_global.txt")) as fh:
        sigma_global = float(fh.read().strip())

    lm_rpg   = lm_rpg_df.values.astype(np.float64)   # (n_lm, n_pairs)
    lm_dc    = lm_dc_df.values.astype(np.float64)     # (n_lm, n_dc)
    lm_sigma = lm_sigma_df["local_sigma"].values.astype(np.float64)  # (n_lm,)
    n_lm, n_dc = lm_dc.shape

    print(f"[diffmap] Landmarks: {n_lm}  |  DCs: {n_dc}  |  "
          f"RPG genes: {len(rpg_genes)}  |  σ_global: {sigma_global:.4f}")

    # ── 1. Extract RPG expression aligned to reference gene order ─────────────
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    n_rpg_ref   = len(rpg_genes)
    n_cells     = adata.n_obs

    rpg_expr_all = np.zeros((n_cells, n_rpg_ref), dtype=np.float32)
    rpg_ref_idx, dat_idx = zip(
        *[(j, gene_to_idx[g]) for j, g in enumerate(rpg_genes) if g in gene_to_idx]
    ) if any(g in gene_to_idx for g in rpg_genes) else ([], [])

    if dat_idx:
        X_sub = X_log[:, list(dat_idx)]
        if issparse(X_sub):
            X_sub = np.asarray(X_sub.todense())
        rpg_expr_all[:, list(rpg_ref_idx)] = X_sub.astype(np.float32)

    print(f"[diffmap] RPG genes found in dataset: {len(dat_idx)} / {n_rpg_ref}")

    # ── 2. Pairwise co-expression products (upper triangle incl. diagonal) ────
    ti, tj  = np.triu_indices(n_rpg_ref)
    n_pairs = len(pair_names)
    print(f"[diffmap] Computing {n_pairs:,} pairwise products for {n_cells:,} cells...")

    co_expr = (rpg_expr_all[:, ti] * rpg_expr_all[:, tj]).astype(np.float32)
    del rpg_expr_all

    # ── 3. Nyström interpolation ──────────────────────────────────────────────
    print(f"[diffmap] Nyström interpolation against {n_lm} landmarks...")

    # Pre-compute squared norms of landmark vectors
    lm_sq = np.sum(lm_rpg ** 2, axis=1)                     # (n_lm,)
    denom = (sigma_global * lm_sigma) + 1e-30                # (n_lm,)

    dc_coords = np.empty((n_cells, n_dc), dtype=np.float32)

    for start in range(0, n_cells, _CHUNK):
        end   = min(start + _CHUNK, n_cells)
        chunk = co_expr[start:end].astype(np.float64)        # (c, n_pairs)

        # Squared Euclidean distances: ||a-b||² = ||a||² + ||b||² - 2 a·b
        a_sq = np.sum(chunk ** 2, axis=1, keepdims=True)     # (c, 1)
        ab   = chunk @ lm_rpg.T                              # (c, n_lm)
        d2   = np.maximum(a_sq + lm_sq - 2.0 * ab, 0.0)    # (c, n_lm)

        # Adaptive Gaussian kernel weights
        K     = np.exp(-d2 / denom)                          # (c, n_lm)
        K_sum = K.sum(axis=1, keepdims=True) + 1e-30

        dc_coords[start:end] = (K @ lm_dc / K_sum).astype(np.float32)

    adata.obsm[OBSM_DIFFMAP] = dc_coords
    print(f"[diffmap] obsm['{OBSM_DIFFMAP}'] {dc_coords.shape}")
