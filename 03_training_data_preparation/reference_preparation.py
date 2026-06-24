#!/usr/bin/env python3
"""
Reference Preparation Pipeline
================================
Processes annotated Hao CITE-seq data (adata_annotated.h5ad) and produces
a flat reference table ready for model training plus all artefacts needed
for out-of-sample projection of new cells.

Modules
-------
1. RPG pairwise co-expression → diffusion map (up to 25 components)
   + Nyström landmark selection (25000 cells via MiniBatchKMeans)
2. HVG selection (5000 genes, Seurat-style) + PCA (50 components)
   with gene-scaling params and per-gene loadings saved
3. Relative Rank Scoring (RRS) for the Endocytosis GO pathway (chunked).
   NOTE: this is a custom, UCell-INSPIRED reimplementation of the Mann–Whitney-U
   rank score — it does NOT use the UCell / pyUCell package (none is imported).
   Only the scoring formula and the default rank cap (1500) follow UCell.

Final output table columns
--------------------------
  barcode, celltype
  DC1 .. DC25          (RPG-co-expression diffusion map positions)
  PC1 .. PC50          (PCA on HVG expression)
  RRS_Endocytosis      (pathway module score, range [0,1])
  ADT_{protein} ...    (CLR-normalised protein levels)
  (HVG gene loadings are saved separately in HVG_PCA/pca_loadings.csv)

Subdirectory artefacts for out-of-sample mapping
-------------------------------------------------
  RPG_diffmap/
    rpg_gene_list.csv          — RPG genes used (in order)
    rpg_pair_names.csv         — column names for pairwise products
    diffmap_eigenvalues.csv    — eigenvalues DC1..DC10
    landmark_barcodes.csv      — 25000 landmark barcodes + celltype
    landmark_rpg_vectors.csv   — RPG pairwise product vectors (landmarks × pairs)
    landmark_dc_coords.csv     — DC positions for landmarks (landmarks × DC)
    landmark_local_sigma.csv   — per-landmark local sigma from diffusion map
    landmark_sigma_global.txt  — median sigma (scalar) for Nyström kernel
  HVG_PCA/
    hvg_genes.csv              — 5000 HVG names + dispersion stats
    gene_scaling.csv           — mean + std per HVG (center/scale new data)
    pca_loadings.csv           — gene × PC loading matrix
    pca_variance.csv           — explained variance per PC

Dependencies
------------
  pip install anndata numpy pandas scipy pynndescent scikit-learn gseapy
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata
from pynndescent import NNDescent
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
INPUT_H5AD   = Path(r"C:\R\CITE-SEQ\Annotation\Output\Hao\4.Annotation+validation\adata_annotated.h5ad")
RPG_FILE     = Path(r"C:\R\RPGFingerprints\inst\extdata\rpg_lists\human\human_cytosolic.csv")
OUT_DIR      = Path(r"C:\R\CITE-SEQ\Reference_Preparation\Output")
RPG_DIR      = OUT_DIR / "RPG_diffmap"
HVG_DIR      = OUT_DIR / "HVG_PCA"

MAPPING_CSV  = Path(r"C:\R\CITE-SEQ\RNA_data\adt_rna_mapping.csv")

CELLTYPE_COL = "celltype_final"
N_DC         = 25       # diffusion components to retain (max 25)
N_LANDMARKS  = 25000     # Nyström landmark cells
N_HVG        = 5000     # highly variable genes
N_PCS        = 50       # PCA components
N_BINS           = 20      # bins for Seurat-style HVG selection
N_JOBS           = 8       # parallel workers for RRS (UCell-inspired) scoring
UCELL_MAX_RANK   = 1500    # rank cap per cell (matches UCell's default; see note in section 7)

for d in [OUT_DIR, RPG_DIR, HVG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────
def compute_diffusion_map(feature_matrix: np.ndarray, n_components: int = 10):
    """
    Adaptive diffusion map identical to fingerprint_clustering.R / citediffmap_pipeline.py.

    Algorithm:
      1. Adaptive k = clamp(round(sqrt(N)), 20, 50)
      2. Approximate kNN via pynndescent (Euclidean)
      3. Local sigma_i = distance to k-th nearest neighbour
      4. Sparse adaptive Gaussian kernel: K[i,j] = exp(-d² / (σᵢ·σⱼ))
      5. Symmetrise K, then L = D^(-½) K D^(-½)
      6. Sparse eigsh, drop trivial first component
      7. Scale DC_i = λᵢ · vᵢ

    Returns
    -------
    dc_coords   : (n_cells, n_components)
    eigenvalues : (n_components,) non-trivial, descending
    local_sigma : (n_cells,) per-cell sigma from kNN
    """
    n = feature_matrix.shape[0]
    k = int(np.clip(round(np.sqrt(n)), 20, 50))
    print(f"  Adaptive k = {k}  (N = {n:,})")

    print("  Building approximate kNN graph (pynndescent) ...")
    index = NNDescent(feature_matrix, n_neighbors=k + 1, metric="euclidean",
                      random_state=42, verbose=False)
    knn_indices, knn_distances = index.neighbor_graph
    local_sigma = knn_distances[:, k].astype(np.float64)

    print("  Building sparse adaptive Gaussian affinity matrix ...")
    d_knn   = knn_distances[:, 1:].astype(np.float64)
    idx_knn = knn_indices[:, 1:]
    sigma_i = local_sigma[:, np.newaxis]
    sigma_j = local_sigma[idx_knn]
    sp      = sigma_i * sigma_j
    sp      = np.where(sp > 0, sp, np.inf)
    kvals   = np.exp(-d_knn ** 2 / sp).astype(np.float32)

    row_idx  = np.repeat(np.arange(n), k)
    col_idx  = idx_knn.ravel()
    all_rows = np.concatenate([row_idx, np.arange(n)])
    all_cols = np.concatenate([col_idx, np.arange(n)])
    all_vals = np.concatenate([kvals.ravel(), np.ones(n, dtype=np.float32)])

    K = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n, n), dtype=np.float32)
    K = (K + K.T).multiply(0.5)

    print("  Normalising ...")
    row_sums   = np.array(K.sum(axis=1)).flatten()
    row_sums   = np.where(row_sums > 0, row_sums, 1.0)
    D_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
    L          = D_inv_sqrt @ K @ D_inv_sqrt

    print(f"  Eigendecomposition ({n_components + 1} eigenvectors) ...")
    n_eigs = min(n_components + 1, n - 2)
    eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which="LM")

    order        = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues  = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    dc_coords = (eigenvectors * eigenvalues[np.newaxis, :]).astype(np.float32)
    print(f"  Non-trivial eigenvalues (first 5): {np.round(eigenvalues[:5], 5)}")
    return dc_coords, eigenvalues, local_sigma


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 65)
print("1. Loading annotated h5ad ...")
adata = ad.read_h5ad(str(INPUT_H5AD))
print(f"  Cells  : {adata.n_obs:,}")
print(f"  Genes  : {adata.n_vars:,}")
print(f"  Obs    : {list(adata.obs.columns[:10])} ...")

barcodes  = adata.obs_names.values.astype(str)
celltypes = adata.obs[CELLTYPE_COL].astype(str).values
donors    = adata.obs["donor"].astype(str).values
n_cells   = len(barcodes)
gene_names = list(adata.var_names.astype(str))
n_genes    = len(gene_names)

# Compute log-normalized expression (normalize_total → log1p, target_sum=1e4).
# This is used for RPG co-expression diffmap, endocytosis RRS scoring, HVG
# selection, PCA, and RNA gene extraction
if "counts" in adata.layers:
    X_counts = adata.layers["counts"]
    if not issparse(X_counts):
        X_counts = csr_matrix(X_counts)
    row_sums = np.array(X_counts.sum(axis=1)).flatten()
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X_rna = X_counts.multiply((1e4 / row_sums).reshape(-1, 1)).tocsr()
    X_rna.data = np.log1p(X_rna.data)
    print(f"  Log-normalized from layers['counts'] (normalize_total 1e4 → log1p)")
else:
    X_rna = adata.X
    if not issparse(X_rna):
        X_rna = csr_matrix(X_rna)
    print("  WARNING: layers['counts'] not found — using adata.X as-is.")

print(f"  X shape: {X_rna.shape}  nnz={X_rna.nnz:,}")


# ── 2. RPG CO-EXPRESSION ──────────────────────────────────────────────────────
print("\n2. Computing RPG pairwise co-expression ...")

rpg_list = (pd.read_csv(RPG_FILE, header=None, names=["gene"])["gene"]
              .astype(str).tolist())
print(f"  {len(rpg_list)} genes in RPG list")

gene_to_idx = {g: i for i, g in enumerate(gene_names)}
rpg_found   = [g for g in rpg_list if g in gene_to_idx]
rpg_idx     = [gene_to_idx[g] for g in rpg_found]
n_rpg       = len(rpg_found)
print(f"  {n_rpg} / {len(rpg_list)} RPG genes found in dataset")

if n_rpg < 2:
    sys.exit("ERROR: Fewer than 2 RPG genes found — cannot compute pairwise co-expression.")

print("  Extracting RPG expression sub-matrix ...")
rpg_expr = np.asarray(X_rna[:, rpg_idx].todense(), dtype=np.float32)
print(f"  RPG expression: {rpg_expr.shape}")

ti, tj  = np.triu_indices(n_rpg)   # upper triangle, includes diagonal
n_pairs = len(ti)
mem_gb  = n_cells * n_pairs * 4 / 1e9
print(f"  {n_rpg} RPG genes → {n_pairs:,} pairwise products  (~{mem_gb:.2f} GB)")

co_expr = (rpg_expr[:, ti] * rpg_expr[:, tj]).astype(np.float32)
del rpg_expr
print(f"  Co-expression matrix: {co_expr.shape}")

pair_names = [f"{rpg_found[ti[p]]}_{rpg_found[tj[p]]}" for p in range(n_pairs)]


# ── 3. DIFFUSION MAP ──────────────────────────────────────────────────────────
print(f"\n3. Running diffusion map ({N_DC} components) ...")
dc_coords, eigenvalues, local_sigma = compute_diffusion_map(co_expr, n_components=N_DC)
dc_cols = [f"DC{i + 1}" for i in range(N_DC)]
print(f"  DC coordinates: {dc_coords.shape}")

pd.DataFrame({"gene": rpg_found}).to_csv(RPG_DIR / "rpg_gene_list.csv", index=False)
pd.DataFrame({"pair": pair_names}).to_csv(RPG_DIR / "rpg_pair_names.csv", index=False)
pd.DataFrame({
    "component": dc_cols,
    "eigenvalue": eigenvalues,
}).to_csv(RPG_DIR / "diffmap_eigenvalues.csv", index=False)


# ── 4. LANDMARK SELECTION (Nyström out-of-sample projection) ──────────────────
print(f"\n4. Selecting {N_LANDMARKS} Nyström landmark cells ...")
print("  Running MiniBatchKMeans on DC coordinates (low-dim proxy for co-expression) ...")

actual_landmarks = min(N_LANDMARKS, n_cells)
kmeans = MiniBatchKMeans(
    n_clusters   = actual_landmarks,
    random_state = 42,
    batch_size   = min(50_000, n_cells),   # larger batches → faster convergence
    n_init       = 3,
    verbose      = 0,
)
# Cluster in DC space (N_DC dims) rather than co_expr space (n_pairs dims).
# DC coords encode the same co-expression structure at ~200× lower dimensionality,
# giving equivalent manifold coverage at a fraction of the compute cost.
kmeans.fit(dc_coords)

# For each cluster, pick the actual cell closest to the centroid (in DC space)
print("  Finding closest actual cells to centroids ...")
labels    = kmeans.labels_
centroids = kmeans.cluster_centers_

landmark_indices = []
for cluster_id in range(actual_landmarks):
    mask = np.where(labels == cluster_id)[0]
    if len(mask) == 0:
        continue
    dists = np.sum((dc_coords[mask] - centroids[cluster_id]) ** 2, axis=1)
    landmark_indices.append(mask[np.argmin(dists)])

landmark_indices  = np.array(landmark_indices)
n_lm              = len(landmark_indices)
print(f"  Selected {n_lm} landmark cells")

lm_barcodes  = barcodes[landmark_indices]
lm_celltypes = celltypes[landmark_indices]
lm_dc        = dc_coords[landmark_indices]
lm_rpg       = co_expr[landmark_indices]
lm_sigma     = local_sigma[landmark_indices]
sigma_global = float(np.median(local_sigma))

# Save landmark artefacts
pd.DataFrame({
    "barcode": lm_barcodes,
    "celltype": lm_celltypes,
}).to_csv(RPG_DIR / "landmark_barcodes.csv", index=False)

pd.DataFrame(lm_rpg, index=lm_barcodes,
             columns=pair_names).to_csv(RPG_DIR / "landmark_rpg_vectors.csv")

pd.DataFrame(lm_dc, index=lm_barcodes,
             columns=dc_cols).to_csv(RPG_DIR / "landmark_dc_coords.csv")

pd.DataFrame({
    "barcode": lm_barcodes,
    "local_sigma": lm_sigma,
}).to_csv(RPG_DIR / "landmark_local_sigma.csv", index=False)

with open(RPG_DIR / "landmark_sigma_global.txt", "w") as fh:
    fh.write(f"{sigma_global}\n")

print(f"  Global sigma (median of all cells): {sigma_global:.6f}")
print(f"  Landmark artefacts saved to {RPG_DIR}")

del co_expr


# ── 5. HVG SELECTION (Seurat-style binned dispersion) ─────────────────────────
print(f"\n5. Selecting top {N_HVG} HVGs ...")

# Gene-wise statistics from sparse matrix (cells × genes)
mean_g = np.asarray(X_rna.mean(axis=0)).flatten()           # (n_genes,)

X_sq       = X_rna.copy()
X_sq.data **= 2
mean_sq_g  = np.asarray(X_sq.mean(axis=0)).flatten()
del X_sq

var_g = mean_sq_g - mean_g ** 2
cv2_g = np.where(mean_g > 0, var_g / mean_g, 0.0)

expressed = mean_g > 0
print(f"  Expressed genes (mean > 0): {expressed.sum():,}")

log_mean = np.where(expressed, np.log1p(mean_g), 0.0)
log_cv2  = np.where(expressed, np.log1p(cv2_g),  0.0)

bin_edges = np.percentile(log_mean[expressed], np.linspace(0, 100, N_BINS + 1))
bin_edges[-1] += 1e-6
bin_ids = np.clip(np.digitize(log_mean, bin_edges) - 1, 0, N_BINS - 1)

std_disp = np.zeros(n_genes)
for b in range(N_BINS):
    mask = (bin_ids == b) & expressed
    if mask.sum() < 2:
        continue
    vals   = log_cv2[mask]
    mu, sd = vals.mean(), vals.std()
    if sd > 0:
        std_disp[mask] = (vals - mu) / sd

disp_expressed = np.where(expressed, std_disp, -np.inf)
hvg_idx        = np.sort(np.argsort(disp_expressed)[::-1][:N_HVG])
hvg_names      = np.array(gene_names)[hvg_idx]
print(f"  HVG dispersion range: [{std_disp[hvg_idx].min():.2f}, {std_disp[hvg_idx].max():.2f}]")

pd.DataFrame({
    "gene":           hvg_names,
    "mean":           mean_g[hvg_idx],
    "variance":       var_g[hvg_idx],
    "std_dispersion": std_disp[hvg_idx],
}).to_csv(HVG_DIR / "hvg_genes.csv", index=False)


# ── 6. CENTER, SCALE, PCA ─────────────────────────────────────────────────────
print(f"\n6. PCA on {N_HVG} HVGs ({N_PCS} components) ...")

X_hvg = np.asarray(X_rna[:, hvg_idx].todense(), dtype=np.float32)
print(f"  HVG sub-matrix: {X_hvg.shape}  ({X_hvg.nbytes / 1e9:.2f} GB)")

gene_mean_hvg = X_hvg.mean(axis=0)
gene_std_hvg  = X_hvg.std(axis=0)
gene_std_hvg  = np.where(gene_std_hvg > 0, gene_std_hvg, 1.0)

X_scaled = (X_hvg - gene_mean_hvg) / gene_std_hvg
del X_hvg

pca       = PCA(n_components=N_PCS, random_state=42, svd_solver="randomized")
pc_coords = pca.fit_transform(X_scaled).astype(np.float32)
del X_scaled
print(f"  PC coordinates: {pc_coords.shape}")

explained      = pca.explained_variance_
explained_frac = pca.explained_variance_ratio_
cumulative     = np.cumsum(explained_frac)
print(f"  Variance explained PC1–PC{N_PCS}: {cumulative[-1] * 100:.1f}%")

# Gene scaling params — required to center/scale new data before projection
pd.DataFrame({
    "gene": hvg_names,
    "mean": gene_mean_hvg,
    "std":  gene_std_hvg,
}).to_csv(HVG_DIR / "gene_scaling.csv", index=False)

# Per-gene loadings (components_.T gives genes × PCs)
pd.DataFrame(
    pca.components_.T,
    index=hvg_names,
    columns=[f"PC{i + 1}" for i in range(N_PCS)],
).to_csv(HVG_DIR / "pca_loadings.csv")

pd.DataFrame({
    "PC":              [f"PC{i + 1}" for i in range(N_PCS)],
    "explained_var":   explained,
    "explained_frac":  explained_frac,
    "cumulative_frac": cumulative,
}).to_csv(HVG_DIR / "pca_variance.csv", index=False)


# ── 7. ENDOCYTOSIS PATHWAY RRS (custom UCell-inspired score; NOT the UCell pkg) ─
print("\n7. Computing endocytosis RRS scores (custom UCell-inspired) ...")

# Hardcoded Endocytosis MSigDB C5 GO:BP pathway names
# (mirrors pathway_module_scores.R → Endocytosis = c(...))
ENDOCYTOSIS_PATHWAYS = [
    # Clathrin-mediated / core endocytosis
    "GOBP_ENDOCYTOSIS",
    "GOBP_CLATHRIN_DEPENDENT_ENDOCYTOSIS",
    "GOBP_CLATHRIN_COAT_ASSEMBLY",
    "GOBP_RECEPTOR_MEDIATED_ENDOCYTOSIS",
    "GOBP_RECEPTOR_INTERNALIZATION",
    "GOBP_MEMBRANE_INVAGINATION",
    "GOBP_PINOCYTOSIS",
    "GOBP_PHAGOCYTOSIS",
    # Regulation of endocytosis
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
    # Receptor-specific internalization
    "GOBP_G_PROTEIN_COUPLED_RECEPTOR_INTERNALIZATION",
    "GOBP_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_RECEPTOR_RECYCLING",
    "GOBP_REGULATION_OF_EARLY_ENDOSOME_TO_LATE_ENDOSOME_TRANSPORT",
    # Endosomal trafficking
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

# Fetch gene members from MSigDB C5 GO:BP via gseapy (Enrichr).
# Enrichr term names look like "Endocytosis (GO:0006897)"; MSigDB uses GOBP_ENDOCYTOSIS.
# We normalise both sides: strip GOBP_ prefix / GO-ID suffix, collapse non-alpha to spaces.
import re
import gseapy as gp

def _norm_gobp(name: str) -> str:
    """GOBP_CLATHRIN_DEPENDENT_ENDOCYTOSIS → 'clathrin dependent endocytosis'"""
    return name.replace("GOBP_", "").replace("_", " ").lower()

def _norm_enrichr(name: str) -> str:
    """'Clathrin-dependent endocytosis (GO:0072583)' → 'clathrin dependent endocytosis'"""
    name = re.sub(r"\s*\(GO:\d+\)", "", name)   # strip GO ID
    name = re.sub(r"[-/]", " ", name)            # hyphens/slashes → space
    return name.lower().strip()

print("  Fetching GO Biological Process gene sets (gseapy / Enrichr) ...")
go_sets = gp.get_library(name="GO_Biological_Process_2023", organism="Human")

# Build normalised lookup: normalised_term_name → gene list
norm_to_genes: dict = {_norm_enrichr(k): v for k, v in go_sets.items()}

endocytosis_genes: set = set()
found_terms, missing_terms = [], []
for pw in ENDOCYTOSIS_PATHWAYS:
    key = _norm_gobp(pw)
    if key in norm_to_genes:
        endocytosis_genes.update(norm_to_genes[key])
        found_terms.append(pw)
    else:
        missing_terms.append(pw)

print(f"  {len(found_terms)}/{len(ENDOCYTOSIS_PATHWAYS)} pathway terms resolved")
if missing_terms:
    print(f"  Not resolved (skipped): {missing_terms}")
print(f"  Union gene set: {len(endocytosis_genes)} unique genes")

# Intersect with dataset genes
valid_pathway_idx = np.array(
    [i for i, g in enumerate(gene_names) if g in endocytosis_genes]
)
n_pathway = len(valid_pathway_idx)
print(f"  {n_pathway} pathway genes found in dataset ({n_genes} total genes)")

# Custom, UCell-INSPIRED relative rank score (RRS) — NOT the UCell/pyUCell
# package (no UCell library is imported; this is a from-scratch NumPy version
# that reproduces UCell's formula). Per cell: a Mann-Whitney U statistic for the
# gene set, with per-gene ranks capped at UCELL_MAX_RANK (UCell's default of 1500).
# Score = 1 - U_norm, where U_norm = (sum_of_capped_set_ranks - n_set*(n_set+1)/2)
#                                     / (n_set * UCELL_MAX_RANK)
# Higher score → gene set more highly expressed in that cell.
# Column kept as "RRS_Endocytosis" for compatibility with downstream scripts.

def _ucell_chunk(cell_indices: np.ndarray,
                 X_csr,
                 gene_set_idx: np.ndarray,
                 max_rank: int) -> np.ndarray:
    """Custom UCell-inspired RRS for a subset of cells (one joblib task).

    NOT the UCell package — a direct NumPy/SciPy reimplementation of UCell's
    rank-sum score: rank genes per cell (cap at max_rank), then 1 − normalised
    Mann–Whitney U of the gene set's ranks.
    """
    X_chunk = X_csr[cell_indices].toarray().astype(np.float32)
    # Rank each cell descending (rank 1 = highest expression, ties = average)
    ranks = np.vstack(
        [rankdata(-row, method="average") for row in X_chunk]
    ).astype(np.float32)
    np.clip(ranks, None, max_rank, out=ranks)          # cap at max_rank
    n_set     = len(gene_set_idx)
    set_ranks = ranks[:, gene_set_idx]                 # (chunk, n_set)
    u_stat    = set_ranks.sum(axis=1) - n_set * (n_set + 1) / 2.0
    u_norm    = u_stat / (float(n_set) * max_rank)
    return (1.0 - u_norm).astype(np.float32)

if n_pathway == 0:
    print("  WARNING: No pathway genes found — RRS scores set to 0.")
    rrs_scores = np.zeros(n_cells, dtype=np.float32)
else:
    # Split cells into balanced chunks; more chunks than workers for load-balancing
    n_chunks   = N_JOBS * 4
    chunk_size = math.ceil(n_cells / n_chunks)
    cell_chunks = [
        np.arange(i, min(i + chunk_size, n_cells))
        for i in range(0, n_cells, chunk_size)
    ]
    print(f"  RRS (UCell-inspired) scoring: {n_cells:,} cells  |  "
          f"{len(cell_chunks)} chunks of ~{chunk_size:,}  |  "
          f"{N_JOBS} workers  |  max_rank={UCELL_MAX_RANK}")

    chunk_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_ucell_chunk)(chunk, X_rna, valid_pathway_idx, UCELL_MAX_RANK)
        for chunk in cell_chunks
    )
    rrs_scores = np.concatenate(chunk_results)

    print(f"  RRS score range : [{rrs_scores.min():.4f}, {rrs_scores.max():.4f}]")
    print(f"  RRS score mean  : {rrs_scores.mean():.4f}")


# ── 8. EXTRACT ADT-MATCHED RNA EXPRESSION ─────────────────────────────────────
print("\n8. Extracting RNA expression for ADT-matched genes ...")
mapping_df   = pd.read_csv(MAPPING_CSV)
adt_rna_genes = mapping_df["RNA_gene"].dropna().unique().tolist()
print(f"  {len(adt_rna_genes)} unique RNA genes in mapping file")

rna_cols: dict = {}
missing_rna = []
for gene in adt_rna_genes:
    if gene in gene_to_idx:
        idx = gene_to_idx[gene]
        rna_cols[f"RNA_{gene}"] = np.asarray(X_rna[:, idx].todense(), dtype=np.float32).flatten()
    else:
        missing_rna.append(gene)

print(f"  {len(rna_cols)} genes extracted  |  {len(missing_rna)} not found in dataset"
      + (f": {missing_rna}" if missing_rna else ""))


# ── 9. EXTRACT ADT DATA (CLR-normalised) ──────────────────────────────────────
print("\n9. Extracting ADT protein levels ...")
adt_obsm = adata.obsm.get("ADT", None)
adt_cols = {}

if adt_obsm is None:
    print("  WARNING: No 'ADT' key in adata.obsm — ADT columns will be absent.")
else:
    if isinstance(adt_obsm, pd.DataFrame):
        adt_matrix   = adt_obsm.values.astype(np.float32)
        adt_proteins = list(adt_obsm.columns)
    else:
        adt_matrix   = np.asarray(adt_obsm, dtype=np.float32)
        adt_proteins = [f"prot{i}" for i in range(adt_matrix.shape[1])]
    print(f"  ADT shape: {adt_matrix.shape}  ({len(adt_proteins)} proteins)")
    print(f"  ADT raw count range: [{adt_matrix.min():.1f}, {adt_matrix.max():.1f}]")

    # Per-cell CLR normalization (Seurat-compatible: margin=2, i.e. per cell across proteins)
    # CLR(x_ij) = log1p( x_ij / exp(mean_j(log1p(x_ij))) )
    log1p_adt   = np.log1p(adt_matrix)                               # (n_cells, n_proteins)
    geo_mean    = np.exp(log1p_adt.mean(axis=1, keepdims=True))      # per-cell geometric mean
    adt_matrix  = np.log1p(adt_matrix / geo_mean).astype(np.float32) # CLR-normalised
    print(f"  ADT CLR range: [{adt_matrix.min():.3f}, {adt_matrix.max():.3f}]")

    adt_cols = {f"ADT_{p}": adt_matrix[:, i] for i, p in enumerate(adt_proteins)}


# ── 10. ASSEMBLE FINAL REFERENCE TABLE ────────────────────────────────────────
print("\n10. Assembling final reference table ...")

out_dict: dict = {
    "barcode":  barcodes,
    "celltype": celltypes,
    "donor":    donors,
}

# Diffusion map positions
for i, col in enumerate(dc_cols):
    out_dict[col] = dc_coords[:, i]

# PCA dimensions
for i in range(N_PCS):
    out_dict[f"PC{i + 1}"] = pc_coords[:, i]

# Pathway score
out_dict["RRS_Endocytosis"] = rrs_scores

# ADT protein levels
out_dict.update(adt_cols)

# RNA expression for ADT-matched genes only
out_dict.update(rna_cols)

df_out = pd.DataFrame(out_dict)
print(f"  Output: {df_out.shape[0]:,} cells × {df_out.shape[1]} columns")


# ── 11. SAVE ──────────────────────────────────────────────────────────────────
print("\n11. Saving ...")

parquet_path = OUT_DIR / "reference_data.parquet"
csv_path     = OUT_DIR / "reference_data.csv"

df_out.to_parquet(str(parquet_path), index=False)
df_out.to_csv(str(csv_path), index=False)

print(f"  → {parquet_path}")
print(f"  → {csv_path}")

print("\n── Column summary ──────────────────────────────────────────")
print(f"  Metadata       :  3  (barcode, celltype, donor)")
print(f"  DC positions   : {N_DC:2d}  (DC1–DC{N_DC})")
print(f"  PCA dimensions : {N_PCS:2d}  (PC1–PC{N_PCS})")
print(f"  Pathway score  :  1  (RRS_Endocytosis)")
print(f"  ADT proteins   : {len(adt_cols):2d}")
print(f"  RNA (ADT-matched): {len(rna_cols):3d}  (RNA_*)")
print(f"  (HVG loadings saved separately → HVG_PCA/pca_loadings.csv)")

print("\n── Subdirectory artefacts ──────────────────────────────────")
print(f"  {RPG_DIR}/")
print(f"    rpg_gene_list.csv, rpg_pair_names.csv, diffmap_eigenvalues.csv")
print(f"    landmark_barcodes.csv, landmark_rpg_vectors.csv")
print(f"    landmark_dc_coords.csv, landmark_local_sigma.csv, landmark_sigma_global.txt")
print(f"  {HVG_DIR}/")
print(f"    hvg_genes.csv, gene_scaling.csv, pca_loadings.csv, pca_variance.csv")

print("\nDone.")
