# =============================================================================
# multiMiR CACHE GENERATION
# =============================================================================
# Pre-fetches experimentally validated miRNA→target interactions (multiMiR;
# Ru et al. 2014) for the target gene list and caches them to an .rds file.
# rna_features_human.R reads this cache to compute the per-gene miRNA binding
# site count and density (sites / 3'UTR length) without re-querying the database.
#
# Run this BEFORE rna_features_human.R, then point its MULTIMIR_CACHE at the output.
#
# INPUT  : adt_rna_mapping_known.csv  (columns: Ensembl_ID, RNA_gene)
# OUTPUT : multimir_cache.rds         (the multiMiR @data table)
# =============================================================================

library(multiMiR)
library(dplyr)

# ── Paths ─────────────────────────────────────────────────────────────────────
TARGET_CSV   <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual/Input/adt_rna_mapping_known.csv"
OUTPUT_CACHE <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual/Input/multimir_cache.rds"

# ── 1. Load and validate the target gene list ─────────────────────────────────
if (!file.exists(TARGET_CSV))
  stop("Target gene CSV not found: ", TARGET_CSV)

message("--- Loading target genes ---")
targets <- read.csv(TARGET_CSV, stringsAsFactors = FALSE)
required_cols <- c("Ensembl_ID", "RNA_gene")
missing <- setdiff(required_cols, colnames(targets))
if (length(missing) > 0L)
  stop("Missing columns in target CSV: ", paste(missing, collapse = ", "))

# Unique, non-empty HGNC symbols only
genes <- unique(targets$RNA_gene)
genes <- genes[!is.na(genes) & genes != ""]
message("  ", length(genes), " unique HGNC symbols loaded")

# ── 2. Query multiMiR for experimentally validated miRNA→target interactions ──
message("--- Querying multiMiR (validated targets) ---")
mir_data <- get_multimir(
  org     = "hsa",
  target  = genes,
  table   = "validated",   # validated (experimentally confirmed) interactions only
  summary = TRUE
)

# ── 3. Save the interaction table to the cache file ───────────────────────────
message("--- Saving results to: ", OUTPUT_CACHE, " ---")
saveRDS(mir_data@data, file = OUTPUT_CACHE)
message("✓ Cache generated. You can now set MULTIMIR_CACHE in the main script.")
