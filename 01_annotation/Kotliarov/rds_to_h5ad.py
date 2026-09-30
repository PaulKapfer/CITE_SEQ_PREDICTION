"""
Convert Kotliarov neg_control_object.rds → AnnData (.h5ad)

Exports Seurat object components via Rscript, then assembles in Python:
  - RNA raw counts         → layers['counts']
  - RNA log-normalised     → adata.X  (computed in Python if not in Seurat)
  - Cell metadata          → obs
  - ADT raw counts         → obsm['ADT']  (DataFrame, proteins as columns)

Output
------
  C:/R/CITE-SEQ/Annotation/Output/Kotliarov/neg_control.h5ad
"""

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_RDS  = Path(r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Annotation\Kotliarov\Input\H1_day0_demultilexed_singlets.RDS")
OUTPUT_DIR = Path(r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Annotation\Kotliarov\Output\0.Raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_H5AD   = OUTPUT_DIR / "H1_day0.h5ad"

# ── Step 1: Export Seurat components via R ─────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    r_script = textwrap.dedent(f"""
        suppressPackageStartupMessages(library(Seurat))
        suppressPackageStartupMessages(library(Matrix))

        cat("Loading RDS ...\\n")
        obj <- readRDS("{INPUT_RDS.as_posix()}")
        cat(sprintf("  Class   : %s\\n", paste(class(obj), collapse=", ")))

        is_v2 <- inherits(obj, "seurat")   # lowercase = Seurat v2
        is_v3 <- inherits(obj, "Seurat")   # uppercase = Seurat v3+

        if (!is_v2 && !is_v3) {{
            stop("Unsupported class: ", paste(class(obj), collapse=", "))
        }}

        cat(sprintf("  Seurat version: %s\\n", if (is_v2) "v2 (seurat)" else "v3+ (Seurat)"))

        # ── RNA counts ──────────────────────────────────────────────────────────
        if (is_v2) {{
            # Seurat v2: raw counts in @raw.data (genes x cells sparse matrix)
            counts <- obj@raw.data
            # Subset to cells that passed QC (stored in @cell.names)
            cells <- obj@cell.names
            counts <- counts[, cells, drop = FALSE]
            cat(sprintf("  Cells   : %d\\n", length(cells)))
        }} else {{
            # Seurat v3/v4/v5
            rna_assay <- if ("RNA" %in% Assays(obj)) "RNA" else DefaultAssay(obj)
            cat(sprintf("  Cells   : %d\\n  RNA assay: %s\\n", ncol(obj), rna_assay))
            counts <- tryCatch(
                GetAssayData(obj, assay = rna_assay, layer = "counts"),
                error = function(e)
                GetAssayData(obj, assay = rna_assay, slot  = "counts")
            )
        }}
        cat(sprintf("  RNA counts: %d genes x %d cells\\n", nrow(counts), ncol(counts)))

        writeMM(counts, "{(tmp / 'counts.mtx').as_posix()}")
        writeLines(colnames(counts), "{(tmp / 'barcodes.txt').as_posix()}")
        writeLines(rownames(counts), "{(tmp / 'genes.txt').as_posix()}")

        # ── Cell metadata ───────────────────────────────────────────────────────
        meta <- obj@meta.data
        write.csv(meta, "{(tmp / 'metadata.csv').as_posix()}", row.names = TRUE)
        cat(sprintf("  Metadata: %d columns  (%s)\\n", ncol(meta),
                    paste(head(colnames(meta), 8), collapse=", ")))

        # ── ADT counts ──────────────────────────────────────────────────────────
        if (is_v2) {{
            # Seurat v2: other assays stored in obj@assay as a named list
            adt_candidates <- c("CITE", "ADT", "PROT", "Protein", "antibody", "AB")
            adt_key <- adt_candidates[adt_candidates %in% names(obj@assay)][1]
            if (!is.na(adt_key)) {{
                cat(sprintf("  ADT assay (v2): %s\\n", adt_key))
                adt <- obj@assay[[adt_key]]@raw.data
                adt_cells <- intersect(colnames(adt), colnames(counts))
                adt <- adt[, adt_cells, drop = FALSE]
            }} else {{
                adt_key <- NA
            }}
        }} else {{
            adt_candidates <- c("ADT", "PROT", "Protein", "antibody", "AB", "CITE")
            adt_key <- intersect(adt_candidates, Assays(obj))[1]
            if (!is.na(adt_key)) {{
                adt <- tryCatch(
                    GetAssayData(obj, assay = adt_key, layer = "counts"),
                    error = function(e)
                    GetAssayData(obj, assay = adt_key, slot  = "counts")
                )
            }}
        }}

        if (!is.na(adt_key)) {{
            cat(sprintf("  ADT counts: %d proteins x %d cells\\n", nrow(adt), ncol(adt)))
            writeMM(adt, "{(tmp / 'adt_counts.mtx').as_posix()}")
            writeLines(rownames(adt), "{(tmp / 'adt_proteins.txt').as_posix()}")
        }} else {{
            cat("  No ADT assay found\\n")
        }}

        cat("R export complete.\\n")
    """)

    r_script_path = tmp / "export.R"
    r_script_path.write_text(r_script, encoding="utf-8")

    print("=" * 60)
    print("Step 1: Exporting Seurat object via R")
    print("=" * 60)
    result = subprocess.run(["Rscript", str(r_script_path)], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("R export failed — check output above.")

    # ── Step 2: Assemble AnnData ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Assembling AnnData")
    print("=" * 60)

    # RNA counts (MTX is genes × cells → transpose to cells × genes)
    counts_mtx = scipy.io.mmread(tmp / "counts.mtx").T.tocsr().astype(np.float32)
    barcodes   = (tmp / "barcodes.txt").read_text().strip().splitlines()
    genes      = (tmp / "genes.txt").read_text().strip().splitlines()
    print(f"  RNA counts : {counts_mtx.shape[0]:,} cells × {counts_mtx.shape[1]:,} genes")

    # Cell metadata
    meta_df = pd.read_csv(tmp / "metadata.csv", index_col=0)
    meta_df.index = meta_df.index.astype(str)
    # Align to barcode order from the counts matrix
    meta_df = meta_df.reindex(barcodes)
    print(f"  Metadata   : {meta_df.shape[1]} columns")

    # Assemble AnnData
    adata = ad.AnnData(
        X   = counts_mtx,
        obs = meta_df,
        var = pd.DataFrame(index=genes),
    )
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.layers["counts"] = counts_mtx.copy()

    # Log-normalise → store in X and as layer
    print("  Log-normalising RNA (normalize_total 1e4 → log1p) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()

    # Restore raw counts to X is intentional for downstream tools; keep both layers
    # (adata.X = log-normalised, layers['counts'] = raw)

    # ADT
    adt_mtx_path = tmp / "adt_counts.mtx"
    adt_prot_path = tmp / "adt_proteins.txt"
    if adt_mtx_path.exists():
        adt_mtx      = scipy.io.mmread(adt_mtx_path).T.toarray().astype(np.float32)
        adt_proteins = adt_prot_path.read_text().strip().splitlines()
        # Align to RNA barcodes (ADT cells should match)
        adt_df = pd.DataFrame(adt_mtx, index=barcodes, columns=adt_proteins)
        adata.obsm["ADT"] = adt_df
        print(f"  ADT        : {adt_df.shape[1]} proteins")
    else:
        print("  ADT        : not found")

    print(f"\n  Final AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    if "ADT" in adata.obsm:
        print(f"  obsm['ADT'] : {adata.obsm['ADT'].shape}")
    print(f"  layers      : {list(adata.layers.keys())}")

# ── Step 3: Save ───────────────────────────────────────────────────────────────
print(f"\nSaving → {OUT_H5AD}")
adata.write_h5ad(OUT_H5AD)
print("Done.")
