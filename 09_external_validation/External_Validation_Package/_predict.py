"""
Shared prediction utilities used by annotate_unknown (and the feature builders).

Handles:
- Feature matrix construction (must exactly match the 718-dim training layout)
- Isoform resolution (keep the isoform with the higher calibrated Pearson r)
- Linear calibration (slope/intercept, clipped at 0)
- Model and RNA-FM cache loading
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from pathlib import Path

from ._data import data_path, load_model, load_mapping, load_rnafm, \
                  load_rnafm_any, load_calibration, load_protein_quality

# Proteins that have -1 / -2 isoform variants in the model
ISOFORM_BASES = {
    "CD3", "CD4", "CD38", "CD56", "CD11b",
    "CD26", "CD133", "CD138", "CD275", "CD44", "CD45",
}

N_DC = 25
N_PC = 50


def build_protein_id_map(mapping_df: pd.DataFrame, rnafm_cache: dict) -> dict[str, float]:
    """
    Build the protein → integer id map that exactly mirrors training.

    Training assigned ids by sorting all usable (protein, gene) pairs
    alphabetically by protein name and enumerating from 0.
    'Usable' = protein has a matched gene with an RNA-FM vector in rnafm_cache.

    Known proteins receive their real id during prediction; unseen proteins
    receive NaN so the model falls back to the learned default branch.
    """
    usable_sorted = sorted(
        [(row["ADT_feature"], row["RNA_gene"])
         for _, row in mapping_df.iterrows()
         if row["RNA_gene"] in rnafm_cache],
        key=lambda x: x[0],
    )
    return {feat: float(i) for i, (feat, _) in enumerate(usable_sorted)}


def _get_log_expr(adata):
    """
    Return log-normalised expression matrix aligned to adata.var_names.
    Priority: layers['log1p_norm'] → normalize layers['counts'] → adata.X.
    """
    if "log1p_norm" in adata.layers:
        return adata.layers["log1p_norm"]
    if "counts" in adata.layers:
        import scanpy as sc
        import anndata as ad
        from scipy.sparse import csr_matrix
        X_c = adata.layers["counts"].astype(np.float32)
        if not issparse(X_c):
            X_c = csr_matrix(X_c)
        tmp = ad.AnnData(X=X_c, obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        return tmp.X
    return adata.X


def build_rnafm_cache(mapping_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load RNA-FM vectors for all proteins in the mapping. Returns {gene: (640,)}."""
    cache = {}
    for gene in mapping_df["RNA_gene"].unique():
        try:
            cache[gene] = load_rnafm(gene)  # (640,) float32
        except FileNotFoundError:
            pass  # skip genes without RNA-FM features
    return cache


def build_rnafm_cache_unknown(mapping_df: pd.DataFrame,
                              quality_df: pd.DataFrame,
                              requested: list[str] | str) -> dict[str, np.ndarray]:
    """
    Load RNA-FM vectors from Known/ or Unknown/ for all second-dataset proteins.

    All proteins are treated as unknown (protein_id = NaN); embeddings are loaded
    from whichever subdirectory contains the file for each gene.

    Parameters
    ----------
    mapping_df : full ADT→gene mapping for the second dataset
    quality_df : unused; kept for signature compatibility
    requested  : list of ADT_feature names to include, or "all"
    """
    cache = {}
    for _, row in mapping_df.iterrows():
        adt, gene = row["ADT_feature"], row["RNA_gene"]
        if requested != "all" and adt not in requested:
            continue
        try:
            cache[gene] = load_rnafm_any(gene)
        except FileNotFoundError:
            pass                               # no RNA-FM features available — skip
    return cache


def get_unknown_protein_pairs(mapping_df: pd.DataFrame,
                               quality_df: pd.DataFrame,
                               rnafm_cache_unknown: dict) -> list[tuple[str, str]]:
    """Return (ADT_feature, gene) pairs for all second-dataset proteins with RNA-FM features.

    quality_df is unused; kept for signature compatibility.
    """
    pairs = []
    for _, row in mapping_df.iterrows():
        adt, gene = row["ADT_feature"], row["RNA_gene"]
        if gene in rnafm_cache_unknown:
            pairs.append((adt, gene))
    return pairs


def resolve_isoforms(proteins: list[str], quality_df: pd.DataFrame) -> dict[str, str]:
    """
    For each isoform base (e.g. 'CD3'), keep the isoform (-1 or -2) with higher
    calibrated r.  Returns a dict: protein_name_in_model → output_name (no suffix).

    All non-isoform proteins map to themselves.
    """
    q = quality_df.set_index("protein")["r_cal"]
    result = {}

    seen_bases: dict[str, tuple[str, float]] = {}  # base → (best_prot, best_r)
    isoform_prots = set()

    for p in proteins:
        # Check if this is an isoform variant: ends with -1 or -2 and base is in ISOFORM_BASES
        if len(p) > 2 and p[-2] == "-" and p[-1] in ("1", "2"):
            base = p[:-2]
            if base in ISOFORM_BASES:
                isoform_prots.add(p)
                r = float(q.get(p, -np.inf))
                if base not in seen_bases or r > seen_bases[base][1]:
                    seen_bases[base] = (p, r)

    # Build mapping: keep only winner isoforms (renamed), pass through the rest
    for p in proteins:
        if p in isoform_prots:
            base = p[:-2]
            winner, _ = seen_bases[base]
            if p == winner:
                result[p] = base   # rename e.g. "CD3-1" → "CD3"
            # losers are excluded (not added to result)
        else:
            result[p] = p          # unchanged

    return result


def build_feature_matrix(adata, gene: str, rnafm_cache: dict,
                         X_log=None, protein_id_val: float = np.nan) -> np.ndarray:
    """
    Build the (n_cells, 718) XGBoost feature matrix for a single protein/gene.

    Feature layout (must exactly match training):
      [0]       protein_id   – integer id for known proteins; NaN for unseen
      [1]       RNA_expr     – log-normalised expression of matched gene
      [2]       endocytosis  – endocytosis RRS (custom UCell-inspired score)
      [3..27]   DC1..DC25    – RPG diffusion-map coordinates
      [28..77]  PC1..PC50    – HVG-PCA coordinates
      [78..717] RNAFM_0..639 – RNA-FM embedding (640 dims)

    Parameters
    ----------
    adata          : AnnData with RRS_Endocytosis, obsm PCA/diffmap
    gene           : RNA gene name for this protein
    rnafm_cache    : {gene: (640,) array}
    X_log          : optional pre-computed log-normalised expression matrix
    protein_id_val : real integer id for known proteins (from build_protein_id_map);
                     np.nan for unseen proteins so the model applies only effects
                     not tied to a specific antibody
    """
    n = adata.n_obs

    # 0. protein_id (scalar broadcast to all cells)
    pid_col = np.full((n, 1), protein_id_val, dtype=np.float32)

    # 1. RNA expression of matched gene
    if X_log is None:
        X_log = _get_log_expr(adata)
    gene_names  = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    if gene in gene_to_idx:
        col = X_log[:, gene_to_idx[gene]]
        if issparse(col):
            col = np.asarray(col.todense()).flatten()
        rna_expr = col.astype(np.float32).reshape(-1, 1)
    else:
        rna_expr = np.zeros((n, 1), dtype=np.float32)

    # 2. Endocytosis score
    endocyt = adata.obs["RRS_Endocytosis"].values.astype(np.float32).reshape(-1, 1)

    # 3. Diffusion map coords (DC1..DC25)
    dc_coords = adata.obsm["X_External_Validation_Package_diffmap"].astype(np.float32)  # (n, 25)

    # 4. PCA coords (PC1..PC50)
    pc_coords = adata.obsm["X_External_Validation_Package_PCA"].astype(np.float32)      # (n, 50)

    # 5. RNA-FM features (640 dims, same for all cells of this protein)
    rnafm_vec = rnafm_cache[gene].reshape(1, -1).astype(np.float32)  # (1, 640)
    rnafm_mat = np.tile(rnafm_vec, (n, 1))                           # (n, 640)

    return np.concatenate([pid_col, rna_expr, endocyt, dc_coords, pc_coords, rnafm_mat], axis=1)


def calibrate(y_raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply linear calibration and clip at 0."""
    return np.maximum(slope * y_raw + intercept, 0.0).astype(np.float32)


def get_usable_proteins(rnafm_cache: dict) -> list[tuple[str, str]]:
    """
    Return (protein, gene) pairs that are usable: gene has RNA-FM features.
    """
    mapping = load_mapping()
    pairs = []
    for _, row in mapping.iterrows():
        p, g = row["ADT_feature"], row["RNA_gene"]
        if g in rnafm_cache:
            pairs.append((p, g))
    return pairs
