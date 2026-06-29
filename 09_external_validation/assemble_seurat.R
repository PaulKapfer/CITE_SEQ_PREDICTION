#!/usr/bin/env Rscript
# assemble_seurat.R  <export_dir>  <out_rds>
#
# Assembles a Seurat object from the CSV/MTX files written by run_External_Validation_Package.py
# and saves it as an RDS.
#
# Slots populated
# ---------------
#   RNA assay   : counts layer   (raw counts)
#                 data  layer    (log1p-normalised)
#   ADT assay        : measured ADT          (obsm_ADT_measured.csv)
#   ADT_pred_unknown : predicted ADT (all proteins, protein_id = NaN)
#                      (obsm_ADT_pred_unknown.csv)
#   reductions  : pca (X_pca_harmony), umap (X_umap_corrected),
#                 evpca (X_External_Validation_Package_PCA), diffmap (X_External_Validation_Package_diffmap)
#   meta.data   : obs.csv
#
# Any optional matrix CSV that is absent (e.g. ADT_pred_known) is skipped
# automatically — the current pipeline exports only measured + ADT_pred_unknown.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

# ── Args ──────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: assemble_seurat.R <export_dir> <out_rds>")
export_dir <- args[1]
out_rds    <- args[2]
cat(sprintf("Export dir : %s\n", export_dir))
cat(sprintf("Output RDS : %s\n", out_rds))

# ── Helpers ───────────────────────────────────────────────────────────────────
read_obsm <- function(filename) {
  fp <- file.path(export_dir, filename)
  if (!file.exists(fp)) { message("  not found: ", filename); return(NULL) }
  df <- read.csv(fp, row.names = 1, check.names = FALSE)
  if (nrow(df) != length(barcodes)) {
    message("  SKIPPED stale file (", nrow(df), " rows, expected ", length(barcodes), "): ", filename)
    return(NULL)
  }
  cat(sprintf("  %-40s %d x %d\n", filename, nrow(df), ncol(df)))
  df
}

# ── 1. Barcodes and gene metadata ─────────────────────────────────────────────
cat("\n[1] Loading barcodes and gene metadata\n")
barcodes <- read.csv(file.path(export_dir, "barcodes.csv"), stringsAsFactors = FALSE)[[1]]
var_df   <- read.csv(file.path(export_dir, "var.csv"),      row.names = 1, check.names = FALSE)
cat(sprintf("  %d cells  x  %d genes\n", length(barcodes), nrow(var_df)))

# ── 2. RNA count matrix (genes x cells) ───────────────────────────────────────
cat("\n[2] Loading count matrix\n")
counts_mat <- readMM(file.path(export_dir, "counts.mtx"))
rownames(counts_mat) <- rownames(var_df)
colnames(counts_mat) <- barcodes
counts_mat <- as(counts_mat, "CsparseMatrix")
cat(sprintf("  counts: %d x %d\n", nrow(counts_mat), ncol(counts_mat)))

# ── 3. Log-normalised matrix ───────────────────────────────────────────────────
cat("\n[3] Loading log1p-normalised matrix\n")
lognorm_mat <- readMM(file.path(export_dir, "lognorm.mtx"))
rownames(lognorm_mat) <- rownames(var_df)
colnames(lognorm_mat) <- barcodes
lognorm_mat <- as(lognorm_mat, "CsparseMatrix")

# ── 4. Cell metadata ───────────────────────────────────────────────────────────
cat("\n[4] Loading cell metadata (obs)\n")
obs_df <- read.csv(file.path(export_dir, "obs.csv"), row.names = 1, check.names = FALSE)
obs_df <- obs_df[barcodes, , drop = FALSE]   # ensure same order as barcodes
cat(sprintf("  %d cells  x  %d metadata columns\n", nrow(obs_df), ncol(obs_df)))

# ── 5. Create Seurat object (RNA assay) ───────────────────────────────────────
cat("\n[5] Creating Seurat object\n")
seu <- CreateSeuratObject(
  counts    = counts_mat,
  meta.data = obs_df,
  assay     = "RNA"
)
# Store log-normalised values (Seurat v5 uses SetAssayData with layer=)
seu <- SetAssayData(seu, assay = "RNA", layer = "data", new.data = lognorm_mat)
cat(sprintf("  Seurat: %d cells  x  %d genes\n", ncol(seu), nrow(seu)))

# ── 6. ADT assays ─────────────────────────────────────────────────────────────
# Note: Seurat silently converts underscores to dashes in feature names.
# Assay names with underscores (ADT_pred_known, ADT_pred_unknown) are fine;
# only the protein row-names inside each assay are affected.
cat("\n[6] Loading ADT matrices\n")
# Add one ADT matrix (CSV) as a Seurat assay (proteins x cells); skip if missing.
add_adt_assay <- function(csv_file, assay_name) {
  df <- read_obsm(csv_file)
  if (is.null(df)) return(invisible(NULL))
  mat <- t(as.matrix(df))   # proteins x cells
  colnames(mat) <- barcodes
  seu[[assay_name]] <<- CreateAssayObject(counts = mat)
  seu <<- SetAssayData(seu, assay = assay_name, layer = "data", new.data = mat)
}
add_adt_assay("obsm_ADT_measured.csv",                 "ADT")
add_adt_assay("obsm_ADT_pred_known.csv",               "ADT_pred_known")
add_adt_assay("obsm_ADT_pred_unknown.csv",             "ADT_pred_unknown")
add_adt_assay("obsm_ADT_pred_known_from_unknown.csv",  "ADT_pred_known_from_unknown")

# ── 7. Dimensional reductions ─────────────────────────────────────────────────
cat("\n[7] Loading dimensional reductions\n")
# Add one embedding (CSV) as a Seurat dimensional reduction; skip if missing.
add_reduction <- function(csv_file, key, reduction_name) {
  df <- read_obsm(csv_file)
  if (is.null(df)) return(invisible(NULL))
  emb        <- as.matrix(df)
  rownames(emb) <- barcodes
  colnames(emb) <- paste0(key, seq_len(ncol(emb)))
  seu[[reduction_name]] <<- CreateDimReducObject(
    embeddings = emb,
    key        = key,
    assay      = "RNA"
  )
}
add_reduction("obsm_pca_harmony.csv",         "PC_",      "pca")
add_reduction("obsm_umap.csv",                "UMAP_",    "umap")
add_reduction("obsm_External_Validation_Package_pca.csv",    "SCTPC_",   "sctpca")
add_reduction("obsm_External_Validation_Package_diffmap.csv","SCTDM_",   "diffmap")

# ── 8. Save ───────────────────────────────────────────────────────────────────
cat(sprintf("\n[8] Saving RDS -> %s\n", out_rds))
dir.create(dirname(out_rds), showWarnings = FALSE, recursive = TRUE)
saveRDS(seu, file = out_rds)
cat("Done.\n")
