"""
Feature 4 – Endocytosis pathway score (UCell).

Computes per-cell UCell scores for the endocytosis GO-BP gene set union,
exactly as performed in the reference preparation pipeline.

The gene set is bundled (data/endocytosis_gene_sets.json): the 40 GOBP_* terms
below from Enrichr GO_Biological_Process_2023, union of 740 genes, identical to
what reference_preparation.py fetched. UCell matches the R UCell default
(max_rank=1500).

Result stored in adata.obs["RRS_Endocytosis"]  (float, range [0, 1]).
"""
import json
import math

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import rankdata

from .._data import data_path
from .._keys import OBS_ENDOCYTOSIS

N_JOBS       = 4
UCELL_MAX_RANK = 1500

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
    with open(data_path("endocytosis_gene_sets.json")) as f:
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


def compute_endocytosis(adata, X_log) -> None:
    valid_idx = _fetch_endocytosis_gene_indices(list(adata.var_names))

    n_cells = adata.n_obs
    if len(valid_idx) == 0:
        print("[endocytosis] WARNING: No pathway genes found — setting scores to 0.")
        adata.obs[OBS_ENDOCYTOSIS] = 0.0
        return

    n_chunks    = max(N_JOBS * 4, 1)
    chunk_size  = math.ceil(n_cells / n_chunks)
    cell_chunks = [np.arange(i, min(i + chunk_size, n_cells))
                   for i in range(0, n_cells, chunk_size)]

    print(f"[endocytosis] UCell scoring: {n_cells:,} cells  |  "
          f"{len(cell_chunks)} chunks  |  {N_JOBS} workers  |  max_rank={UCELL_MAX_RANK}")

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_ucell_chunk)(chunk, X_log, valid_idx, UCELL_MAX_RANK)
        for chunk in cell_chunks
    )
    scores = np.concatenate(results)
    print(f"[endocytosis] Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    adata.obs[OBS_ENDOCYTOSIS] = scores
