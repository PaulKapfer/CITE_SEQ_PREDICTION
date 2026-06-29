# ============================================================================
# Step 0: Export the Kotliarov Seurat v2 .RDS to plain interchange files
# ============================================================================
# Reads the legacy Seurat v2 object and writes, per barcode:
#   - RNA  raw counts   (MatrixMarket, genes  x cells)
#   - ADT  raw counts   (MatrixMarket, proteins x cells)
#   - cell metadata     (CSV, indexed by barcode)
#   - feature / barcode name lists
# These are then loaded in pure Python by 1.QC+doublet_detection.py (no rpy2),
# which avoids the embedded-R instability when mixed with scanpy/scrublet.
# Run once:  Rscript "0.export_rds_to_mtx.R"

suppressMessages(library(Seurat))
suppressMessages(library(Matrix))

INFILE   <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Input/H1_day0_demultilexed_singlets.RDS"
OUTDIR   <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Input/exported"
ADT_ASSAY <- "CITE"

dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

cat("Reading RDS:", INFILE, "\n")
obj <- readRDS(INFILE)

# Real (singlet) cells live in meta.data; raw.data still holds all droplets.
cells <- rownames(obj@meta.data)
cat("Cells:", length(cells), "\n")

# --- RNA raw counts (genes x cells) ---
rna <- obj@raw.data[, cells, drop = FALSE]
cat("RNA:", nrow(rna), "genes x", ncol(rna), "cells\n")
writeMM(rna, file.path(OUTDIR, "rna_counts.mtx"))
writeLines(rownames(rna), file.path(OUTDIR, "rna_genes.tsv"))

# --- ADT / CITE raw counts (proteins x cells) ---
adt <- obj@assay[[ADT_ASSAY]]@raw.data[, cells, drop = FALSE]
cat("ADT:", nrow(adt), "proteins x", ncol(adt), "cells\n")
writeMM(adt, file.path(OUTDIR, "adt_counts.mtx"))
writeLines(rownames(adt), file.path(OUTDIR, "adt_features.tsv"))

# --- Shared barcodes (column order of both matrices) ---
writeLines(cells, file.path(OUTDIR, "barcodes.tsv"))

# --- Cell metadata (row order = barcodes) ---
write.csv(obj@meta.data, file.path(OUTDIR, "metadata.csv"), row.names = TRUE)

cat("Export complete ->", OUTDIR, "\n")
