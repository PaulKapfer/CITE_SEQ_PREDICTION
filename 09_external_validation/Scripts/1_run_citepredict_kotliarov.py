"""
Step 1/3 - citepredict on the Kotliarov dataset.

  1. Rename the measured ADT columns to the reference naming used by citepredict
     (measured column -> Kotliarov antibody sheet row -> clone -> Hao antibody with
     that clone, else the Kotliarov-only row of the training mapping). Columns
     without a match keep their original name and are neither predicted nor
     evaluated: isotype controls, AnnexinV and TCRgd (no target gene), and
     CD206 (see EXCLUDED_MEASURED).
  2. CLR-normalise the measured ADT counts per cell, exactly as
     reference_preparation.py did for the training targets.
  3. citepredict: prepare_anndata -> annotate_known (Source=Hao, training
     protein_id) -> annotate_unknown (Source=Kotliarov, protein_id = NaN).
  4. Export CSV/MTX files and build the Seurat object (2_assemble_seurat.R).

Run in the env where citepredict is installed (pip install -e <package dir>):
    conda run -n annotation python 1_run_citepredict_kotliarov.py

Outputs (OUT_DIR)
  adata_citepredict.h5ad, adt_rename_table.csv, r_export/, seurat_citepredict.rds
"""

import os
import re
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import citepredict
from citepredict._data import data_path

# ── Paths / settings ──────────────────────────────────────────────────────────
INPUT_H5AD = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Annotation\Kotliarov\Output\4.Annotation+validation\adata_annotated.h5ad"
OUT_DIR    = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Second_Dataset\Output2\singlecellobjects"
EXPORT_DIR = os.path.join(OUT_DIR, "r_export")
OUT_H5AD   = os.path.join(OUT_DIR, "adata_citepredict.h5ad")
OUT_RDS    = os.path.join(OUT_DIR, "seurat_citepredict.rds")
R_SCRIPT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_assemble_seurat.R")

ADT_OBSM = "protein_counts"        # raw measured ADT counts (cells x antibodies)
MEASURED_CLR_OBSM = "ADT_measured_CLR"
PREDICT_ONLY_MEASURED = True       # False: predict every antibody in the training mapping

# CD206 is measured in Kotliarov (CD206_PROT) but missing from the Kotliarov antibody
# sheet, so its clone is unknown: it cannot be assigned to Hao's CD206 (clone 15-2)
# nor established as a distinct antibody. It is excluded from prediction and
# evaluation (stated in the Methods).
EXCLUDED_MEASURED = {"CD206_PROT"}

os.makedirs(EXPORT_DIR, exist_ok=True)


# ── ADT renaming (outside the package) ────────────────────────────────────────
def _name_key(name: str) -> str:
    """'CD197_PROT' / 'CD197 (CCR7)' / 'CX3CR1/GPR13/CCRL1' -> 'CD197' / 'CD197' / 'CX3CR1'."""
    s = re.sub(r"_PROT$", "", name).strip()
    s = re.split(r"\s*\(|/", s)[0]
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _clone_key(clone) -> str:
    """Same normalisation as combine_hao_kotliarov_MODEL-TRAINING.py (case, Excel notation, aliases)."""
    s = str(clone).strip().lower()
    s = re.sub(r"\s*\([^)]*\)", "", s)
    m = re.fullmatch(r"(\d+)\.0*e\+?0*(\d+)", s)
    if m:
        s = f"{m.group(1)}e{m.group(2)}"
    return re.sub(r"\s+", " ", s).strip()


def build_rename_table(measured_cols) -> pd.DataFrame:
    info = data_path("antibody_info")
    kot  = pd.read_csv(info / "kotliarov_adt_mapping.csv", dtype=str)
    hao  = pd.read_csv(info / "hao_adt_mapping.csv", dtype=str)
    comb = pd.read_csv(info / "combined_adt_mapping_MODEL-TRAINING.csv", dtype=str)

    kot_by_key   = {_name_key(a): row for a, row in zip(kot["ADT_feature"], kot.itertuples())}
    hao_by_clone = dict(zip(hao["Clone"].map(_clone_key), hao["ADT_feature"]))
    source       = dict(zip(comb["ADT_feature"], comb["Source"]))

    rows = []
    for col in measured_cols:
        k = None if col in EXCLUDED_MEASURED else kot_by_key.get(_name_key(col))
        ref = None
        if k is not None:
            ref = hao_by_clone.get(_clone_key(k.Clone))
            if ref is None and source.get(k.ADT_feature) == "Kotliarov":
                ref = k.ADT_feature
        rows.append({"measured": col,
                     "kotliarov_antibody": None if k is None else k.ADT_feature,
                     "clone": None if k is None else k.Clone,
                     "reference_name": ref,
                     "Source": source.get(ref)})
    table = pd.DataFrame(rows)
    dup = table["reference_name"].dropna()
    if dup.duplicated().any():
        raise ValueError(f"Several measured columns map to one antibody: {sorted(dup[dup.duplicated()])}")
    return table


def clr(counts: np.ndarray) -> np.ndarray:
    """Per-cell CLR as in reference_preparation.py: log1p(x / exp(mean_j log1p(x_j)))."""
    geo_mean = np.exp(np.log1p(counts).mean(axis=1, keepdims=True))
    return np.log1p(counts / geo_mean).astype(np.float32)


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading adata...")
adata = ad.read_h5ad(INPUT_H5AD)
print(f"  {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")

# ── 2. Measured ADT: rename to reference naming + CLR ─────────────────────────
raw_adt = adata.obsm[ADT_OBSM]
rename = build_rename_table(raw_adt.columns)
rename.to_csv(os.path.join(OUT_DIR, "adt_rename_table.csv"), index=False)
print(f"\n  Measured antibodies: {len(rename)}  |  "
      f"known (Source=Hao): {(rename.Source == 'Hao').sum()}  |  "
      f"unknown (Source=Kotliarov): {(rename.Source == 'Kotliarov').sum()}  |  "
      f"unmatched: {rename.reference_name.isna().sum()} {rename.loc[rename.reference_name.isna(), 'measured'].tolist()}")

new_names = [r if isinstance(r, str) else m for m, r in zip(rename["measured"], rename["reference_name"])]
adata.obsm[MEASURED_CLR_OBSM] = pd.DataFrame(clr(raw_adt.to_numpy(np.float64)),
                                             index=adata.obs_names, columns=new_names)

# ── 3. citepredict ────────────────────────────────────────────────────────────
known_names   = rename.loc[rename.Source == "Hao", "reference_name"].tolist()
unknown_names = rename.loc[rename.Source == "Kotliarov", "reference_name"].tolist()

print("\n=== prepare_anndata ===")
citepredict.prepare_anndata(adata, counts_layer="counts")
print("\n=== annotate_known ===")
citepredict.annotate_known(adata, proteins=known_names if PREDICT_ONLY_MEASURED else None)
print("\n=== annotate_unknown ===")
citepredict.annotate_unknown(adata, proteins=unknown_names if PREDICT_ONLY_MEASURED else None)

print(f"\nSaving h5ad -> {OUT_H5AD}")
adata.write_h5ad(OUT_H5AD)

# ── 4. Export for R ───────────────────────────────────────────────────────────
print(f"\nExporting for R -> {EXPORT_DIR}")
for f in os.listdir(EXPORT_DIR):              # drop files from earlier runs / older layouts
    if f.startswith("obsm_"):
        os.remove(os.path.join(EXPORT_DIR, f))

pd.Series(adata.obs_names, name="barcode").to_csv(os.path.join(EXPORT_DIR, "barcodes.csv"), index=False)
adata.var.to_csv(os.path.join(EXPORT_DIR, "var.csv"))
adata.obs.to_csv(os.path.join(EXPORT_DIR, "obs.csv"))


def write_mtx(mat, filename):
    mat = sp.csc_matrix(mat.T) if not sp.issparse(mat) else mat.T.tocsc()   # genes x cells
    scipy.io.mmwrite(os.path.join(EXPORT_DIR, filename), mat)


print("  Writing counts / lognorm matrices (.mtx)...")
write_mtx(adata.layers["counts"], "counts.mtx")
write_mtx(adata.layers["log1p_norm"], "lognorm.mtx")


def save_obsm(key, filename):
    val = adata.obsm.get(key)
    if val is None:
        print(f"  obsm['{key}'] not found - skipping")
        return
    df = val if isinstance(val, pd.DataFrame) else pd.DataFrame(val, index=adata.obs_names)
    df.to_csv(os.path.join(EXPORT_DIR, filename))
    print(f"  obsm['{key}'] -> {filename}  {df.shape}")


save_obsm(MEASURED_CLR_OBSM,       "obsm_ADT_measured.csv")
save_obsm("ADT_pred_known",        "obsm_ADT_pred_known.csv")
save_obsm("ADT_pred_unknown",      "obsm_ADT_pred_unknown.csv")
save_obsm("X_umap_corrected",      "obsm_umap.csv")
save_obsm("X_pca_harmony",         "obsm_pca_harmony.csv")
save_obsm("X_citepredict_PCA",     "obsm_citepredict_pca.csv")
save_obsm("X_citepredict_diffmap", "obsm_citepredict_diffmap.csv")

# ── 5. Seurat object via R ────────────────────────────────────────────────────
print(f"\nCalling R to assemble Seurat object -> {OUT_RDS}")
result = subprocess.run(["Rscript", R_SCRIPT, EXPORT_DIR, OUT_RDS])
print("  RDS saved." if result.returncode == 0 else "  WARNING: R script returned non-zero exit code.")
print("\nAll done.")
