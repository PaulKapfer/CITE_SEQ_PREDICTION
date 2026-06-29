"""
External validation — inference pipeline on the Kotliarov et al. 2020 CITE-seq dataset.
=======================================================================================
Orchestrates the full external-validation inference run:
  1. Load the annotated Kotliarov AnnData object.
  2. Compute the three training-matched feature blocks (HVG-PCA, RPG diffusion-map,
     endocytosis RRS — custom UCell-inspired, not the UCell package) in-place
     via External_Validation_Package.prepare_anndata.
  3. Predict every protein with protein_id = NaN (zero-shot) via annotate_unknown.
  4. Save the annotated h5ad and export flat files (CSV / MatrixMarket) for R.
  5. Call assemble_seurat.R to build a Seurat object from the exported files.

The Known/Unknown split shown in the figures is decided later at plotting time
(evaluation_panel.R), based on whether the matched transcript was seen in training.

Run with:
    conda run -n bioinfo_env python run_External_Validation_Package_kotliarov.py

Outputs
-------
  output/singlecellobjects/adata_External_Validation_Package.h5ad   – annotated AnnData
  output/singlecellobjects/r_export/                                – CSVs + MTX for R import
  output/singlecellobjects/seurat_External_Validation_Package.rds   – Seurat object (via R)
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import subprocess
import anndata as ad

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_H5AD  = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Annotation\Kotliarov\Output\4.1.Annotation+validation\adata_annotated.h5ad"
OUT_DIR     = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Second_Dataset\Output2\singlecellobjects"
EXPORT_DIR  = os.path.join(OUT_DIR, "r_export")
OUT_H5AD    = os.path.join(OUT_DIR, "adata_External_Validation_Package.h5ad")
OUT_RDS     = os.path.join(OUT_DIR, "seurat_External_Validation_Package.rds")
R_SCRIPT    = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Second_Dataset\Scripts\assemble_seurat.R"

os.makedirs(EXPORT_DIR, exist_ok=True)

# ── 1. Load adata ─────────────────────────────────────────────────────────────
print("Loading adata...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"  {adata.n_obs:,} cells  ×  {adata.n_vars:,} genes")

# ── 1a. Print ADT proteins ─────────────────────────────────────────────────────
print(f"  obsm keys: {list(adata.obsm.keys())}")
if "protein_counts" in adata.obsm and isinstance(adata.obsm["protein_counts"], pd.DataFrame):
    adt_proteins = list(adata.obsm["protein_counts"].columns)
    print(f"\n  ADT proteins ({len(adt_proteins)} total):")
    for p in adt_proteins:
        print(f"    {p}")
else:
    print("  No ADT DataFrame found in obsm['protein_counts']")

# ── 2. Run External_Validation_Package pipeline ──────────────────────────────────────────────
sys.path.insert(0, r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Second_Dataset\package")
import External_Validation_Package

# Step 1: Compute all required input features in-place:
#   - HVG-PCA (50 PCs) for global transcriptional state
#   - RPG diffusion-map (25 DCs) for cell topology / trajectory
#   - endocytosis RRS (custom UCell-inspired score, not the UCell package) for
#     non-specific antibody uptake
print("\n=== prepare_anndata ===")
External_Validation_Package.prepare_anndata(adata)

# Step 2: Predict protein abundance for all proteins with protein_id = NaN.
# All Kotliarov proteins are treated as unknown regardless of overlap with the
# training panel, because antibody clones differ between datasets and applying
# training-derived per-antibody biases to different clones would be invalid.
print("\n=== annotate_unknown ===")
External_Validation_Package.annotate_unknown(adata)

# ── 3. Save h5ad ──────────────────────────────────────────────────────────────
print(f"\nSaving h5ad -> {OUT_H5AD}")
adata.write_h5ad(OUT_H5AD)
print("  Done.")

# ── 4. Export for R ───────────────────────────────────────────────────────────
# Export all matrices and metadata to flat files (CSV / MatrixMarket) so the
# downstream R evaluation script can assemble a Seurat object without depending on Python.
print(f"\nExporting for R -> {EXPORT_DIR}")

# Remove stale CSV files from previous runs to prevent dimension mismatches
_stale = [
    "obsm_ADT_measured.csv", "obsm_ADT_pred_known.csv",
    "obsm_ADT_pred_unknown.csv", "obsm_ADT_pred_known_from_unknown.csv",
    "obsm_umap.csv", "obsm_pca_harmony.csv",
    "obsm_External_Validation_Package_pca.csv",
    "obsm_External_Validation_Package_diffmap.csv",
]
for _f in _stale:
    _fp = os.path.join(EXPORT_DIR, _f)
    if os.path.exists(_fp):
        os.remove(_fp)
        print(f"  Removed stale: {_f}")

# 4a. Cell barcodes and gene names
pd.Series(adata.obs_names, name="barcode").to_csv(
    os.path.join(EXPORT_DIR, "barcodes.csv"), index=False)
adata.var.to_csv(os.path.join(EXPORT_DIR, "var.csv"))

# 4b. Cell metadata (obs)
adata.obs.to_csv(os.path.join(EXPORT_DIR, "obs.csv"))

# 4c. RNA counts matrix (sparse, genes × cells) as Matrix Market
print("  Writing counts matrix (.mtx)...")
counts = adata.layers["counts"]
if not sp.issparse(counts):
    counts = sp.csc_matrix(counts)
else:
    counts = counts.T.tocsc()   # genes × cells
scipy.io.mmwrite(os.path.join(EXPORT_DIR, "counts.mtx"), counts)

# 4d. Log-normalised matrix
print("  Writing log1p_norm matrix (.mtx)...")
lognorm = adata.layers["log1p_norm"]
if not sp.issparse(lognorm):
    lognorm = sp.csc_matrix(lognorm)
else:
    lognorm = lognorm.T.tocsc()
scipy.io.mmwrite(os.path.join(EXPORT_DIR, "lognorm.mtx"), lognorm)

# 4e. obsm layers as CSV (cells × features)
def save_obsm(key, filename):
    """Export one adata.obsm matrix to CSV (cells × features); skip if the key is absent."""
    val = adata.obsm.get(key)
    if val is None:
        print(f"  obsm['{key}'] not found — skipping")
        return
    if isinstance(val, pd.DataFrame):
        df = val
    else:
        df = pd.DataFrame(val, index=adata.obs_names)
    df.to_csv(os.path.join(EXPORT_DIR, filename))
    print(f"  obsm['{key}'] -> {filename}  {df.shape}")

save_obsm("ADT",                          "obsm_ADT_measured.csv")   # measured ADT (ground truth)
save_obsm("ADT_pred_unknown",             "obsm_ADT_pred_unknown.csv")  # NaN-id predictions (all proteins)
save_obsm("X_umap_corrected",             "obsm_umap.csv")
save_obsm("X_pca_harmony",                "obsm_pca_harmony.csv")
save_obsm("X_External_Validation_Package_PCA",           "obsm_External_Validation_Package_pca.csv")
save_obsm("X_External_Validation_Package_diffmap",       "obsm_External_Validation_Package_diffmap.csv")

print("  Export complete.")

# ── 5. Build Seurat RDS via R ─────────────────────────────────────────────────
print(f"\nCalling R to assemble Seurat object -> {OUT_RDS}")
result = subprocess.run(
    ["Rscript", R_SCRIPT, EXPORT_DIR, OUT_RDS],
    capture_output=False
)
if result.returncode != 0:
    print("  WARNING: R script returned non-zero exit code. Check output above.")
else:
    print("  RDS saved.")

print("\nAll done.")
