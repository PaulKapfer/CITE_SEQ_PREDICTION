"""
Feature 4 – Endocytosis pathway relative-rank score (RRS).

Computes a per-cell endocytosis RRS for the GO-BP gene-set union, exactly as in
the reference-preparation pipeline.

IMPORTANT: this is a custom, UCell-INSPIRED reimplementation — it does NOT use
the UCell / pyUCell package (none is imported; scoring is done here with
scipy.stats.rankdata). Only the Mann–Whitney-U rank-sum formula and the default
rank cap (max_rank=1500) follow UCell.

The gene set is bundled as JSON (the same GOBP_* pathway terms as the reference).

Result stored in adata.obs["RRS_Endocytosis"]  (float, range [0, 1]).
"""
import math
import numpy as np
from scipy.sparse import issparse
from scipy.stats import rankdata
from joblib import Parallel, delayed

N_JOBS       = 4
UCELL_MAX_RANK = 1500   # rank cap per cell — matches UCell's default (this is a reimplementation)

ENDOCYTOSIS_PATHWAYS = [
    "GOBP_ENDOCYTOSIS",
    "GOBP_CLATHRIN_DEPENDENT_ENDOCYTOSIS",
    "GOBP_CLATHRIN_COAT_ASSEMBLY",
    "GOBP_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_RECEPTOR_INTERNALIZATION",
    "GOBP_MEMBRANE_INVAGINATION",
    "GOBP_PINOCYTOSIS",
    "GOBP_PHAGOCYTOSIS",
    "GOBP_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_CLATHRIN_DEPENDENT_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_NEGATIVE_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_NEGATIVE_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_ENDOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_RECEPTOR_INTERNALIZATION",
    "GOBP_POSITIVE_REGULATION_OF_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_POSITIVE_REGULATION_OF_PHAGOCYTOSIS",
    "GOBP_G_PROTEIN_COUPLED_RECEPTOR_INTERNALIZATION",
    "GOBP_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_EARLY_ENDOSOME_TO_LATE_ENDOSOME_TRANSPORT",
    "GOBP_ENDOCYTIC_RECYCLING",
    "GOBP_REGULATION_OF_ENDOCYTIC_RECYCLING",
    "GOBP_ENDOSOMAL_TRANSPORT",
    "GOBP_ENDOSOMAL_VESICLE_FUSION",
    "GOBP_ENDOSOME_ORGANIZATION",
    "GOBP_ENDOSOME_TO_LYSOSOME_TRANSPORT",
    "GOBP_ENDOSOME_TO_LYSOSOME_TRANSPORT_VIA_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_LATE_ENDOSOME_TO_LYSOSOME_TRANSPORT",
    "GOBP_LATE_ENDOSOME_TO_VACUOLE_TRANSPORT",
    "GOBP_LATE_ENDOSOME_TO_VACUOLE_TRANSPORT_VIA_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_MULTIVESICULAR_BODY_ORGANIZATION",
    "GOBP_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_UBIQUITIN_DEPENDENT_PROTEIN_CATABOLIC_PROCESS_VIA_THE_MULTIVESICULAR_BODY_SORTING_PATHWAY",
    "GOBP_PROTEIN_LOCALIZATION_TO_ENDOSOME",
    "GOBP_PRESYNAPTIC_ENDOCYTOSIS",
]



def _fetch_endocytosis_gene_indices(gene_names: list) -> np.ndarray:
    """Load bundled endocytosis gene sets and return indices into gene_names."""
    import json
    from pathlib import Path
    bundle = Path(__file__).parent.parent / "data" / "endocytosis_gene_sets.json"
    print(f"[endocytosis] Loading bundled gene sets from {bundle.name}...")
    with open(bundle) as f:
        gene_sets = json.load(f)   # {GOBP_name: [gene, ...]}

    endocytosis_genes: set = set()
    for genes in gene_sets.values():
        endocytosis_genes.update(genes)

    print(f"[endocytosis] {len(gene_sets)} pathways  |  union: {len(endocytosis_genes)} genes")
    idx = np.array([i for i, g in enumerate(gene_names) if g in endocytosis_genes])
    print(f"[endocytosis] {len(idx)} pathway genes found in dataset")
    return idx


def _ucell_chunk(cell_indices: np.ndarray, X, gene_set_idx: np.ndarray,
                 max_rank: int) -> np.ndarray:
    """Custom UCell-inspired RRS for a chunk of cells (NOT the UCell package).

    Reimplements UCell's rank-sum score with scipy.stats.rankdata: rank genes
    per cell (cap at max_rank), then score = 1 − normalised Mann–Whitney U of the
    gene set's ranks. One joblib task.
    """
    if issparse(X):
        chunk = X[cell_indices].toarray().astype(np.float32)
    else:
        chunk = np.asarray(X[cell_indices], dtype=np.float32)
    ranks   = np.vstack([rankdata(-row, method="average") for row in chunk]).astype(np.float32)
    np.clip(ranks, None, max_rank, out=ranks)
    n_set   = len(gene_set_idx)
    u_stat  = ranks[:, gene_set_idx].sum(axis=1) - n_set * (n_set + 1) / 2.0
    u_norm  = u_stat / (float(n_set) * max_rank)
    return (1.0 - u_norm).astype(np.float32)


def compute_endocytosis(adata) -> None:
    """Compute the endocytosis RRS (custom UCell-inspired, not the UCell package).
    Adds adata.obs['RRS_Endocytosis']."""

    gene_names = list(adata.var_names)
    valid_idx  = _fetch_endocytosis_gene_indices(gene_names)

    n_cells = adata.n_obs
    if len(valid_idx) == 0:
        print("[endocytosis] WARNING: No pathway genes found — setting scores to 0.")
        adata.obs["RRS_Endocytosis"] = 0.0
        return

    # Reference was built on log-normalized counts (normalize_total 1e4 → log1p).
    # The RRS is rank-based so log is rank-equivalent to raw counts, but using the
    # same normalization as reference ensures consistent behavior.
    # Prefer layers['log1p_norm'] if pre-computed; else compute from layers['counts'].
    if "log1p_norm" in adata.layers:
        X_use = adata.layers["log1p_norm"]
        print("[endocytosis] Using layers['log1p_norm'].")
    elif "counts" in adata.layers:
        import anndata as ad
        import scanpy as sc
        from scipy.sparse import csr_matrix, issparse as _issparse
        X_c = adata.layers["counts"].astype(np.float32)
        if not _issparse(X_c):
            X_c = csr_matrix(X_c)
        tmp = ad.AnnData(X=X_c, obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        X_use = tmp.X
        del tmp
        print("[endocytosis] Using layers['counts'] → normalize_total(1e4) → log1p.")
    else:
        X_use = adata.X
        print("[endocytosis] WARNING: no counts/log1p_norm layer — using adata.X as-is.")

    n_chunks    = max(N_JOBS * 4, 1)
    chunk_size  = math.ceil(n_cells / n_chunks)
    cell_chunks = [np.arange(i, min(i + chunk_size, n_cells))
                   for i in range(0, n_cells, chunk_size)]

    print(f"[endocytosis] RRS (UCell-inspired) scoring: {n_cells:,} cells  |  "
          f"{len(cell_chunks)} chunks  |  {N_JOBS} workers  |  max_rank={UCELL_MAX_RANK}")

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_ucell_chunk)(chunk, X_use, valid_idx, UCELL_MAX_RANK)
        for chunk in cell_chunks
    )
    scores = np.concatenate(results)
    print(f"[endocytosis] Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    adata.obs["RRS_Endocytosis"] = scores
