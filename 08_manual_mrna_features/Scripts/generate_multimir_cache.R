library(multiMiR)
library(dplyr)

# Paths
TARGET_CSV   <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Matching_ADT_Transcript/Output2/combined_adt_mapping_RNA_FM.csv"
OUTPUT_CACHE <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual/Input/multimir_cache.rds"

if (!file.exists(TARGET_CSV))
  stop("Target gene CSV not found: ", TARGET_CSV)

message("--- Loading target genes ---")
targets <- read.csv(TARGET_CSV, stringsAsFactors = FALSE)
required_cols <- c("Ensembl_ID", "RNA_gene")
missing <- setdiff(required_cols, colnames(targets))
if (length(missing) > 0L)
  stop("Missing columns in target CSV: ", paste(missing, collapse = ", "))

genes <- unique(targets$RNA_gene)
genes <- genes[!is.na(genes) & genes != ""]
message("  ", length(genes), " unique HGNC symbols loaded")

message("--- Querying multiMiR (validated targets) ---")
mir_data <- get_multimir(
  org     = "hsa",
  target  = genes,
  table   = "validated",
  summary = TRUE
)

message("--- Saving results to: ", OUTPUT_CACHE, " ---")
saveRDS(mir_data@data, file = OUTPUT_CACHE)
message("✓ Cache generated. You can now set MULTIMIR_CACHE in the main script.")
