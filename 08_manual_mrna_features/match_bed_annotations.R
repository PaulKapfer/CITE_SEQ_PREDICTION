# =============================================================================
# BED ANNOTATION MATCHING — IRES / nTIS / uORF
# =============================================================================
# Reads per-chromosome BED files for three annotation types, maps RefSeq
# transcript IDs to HGNC gene symbols via Ensembl BioMart, and produces one
# output CSV per annotation type listing which target genes carry each feature.
#
# INPUT FILES
# -----------
#   Target gene list (Ensembl ID + gene symbol):
#     C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/
#       Generate_RNA_features_manual/Input/adt_rna_mapping_known.csv
#     Required columns: Ensembl_ID, RNA_gene
#
#   Per-chromosome BED files (space-separated, UCSC track format):
#     .../Input/IRES_Literature/IRES_literature_chr*.bed
#     .../Input/nTIS/           nTIS_chr*.bed               (assumed naming)
#     .../Input/uORF/           uORF_chr*.bed               (assumed naming)
#
#   BED name field format (column 4):
#     <type>_<RefSeqID>_<start>_<end>
#     e.g.  IRES_literature_NM_000254.2_154_426
#           nTIS_NM_001007022.3_3201_3290
#           uORF_NM_001007553.3_1_450
#
# OUTPUT FILES  (written to .../Input/)
# -------------
#   ires_gene_hits.csv   — one row per unique gene with ≥1 IRES site
#   ntis_gene_hits.csv   — one row per unique gene with ≥1 nTIS site
#   uorf_gene_hits.csv   — one row per unique gene with ≥1 uORF site
#   bed_annotation_summary.csv — one row per target gene, TRUE/FALSE per type
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual"
INPUT_DIR   <- file.path(BASE_DIR, "Input")
OUTPUT_DIR  <- INPUT_DIR   # outputs land in the same Input folder

TARGET_CSV  <- file.path(INPUT_DIR, "adt_rna_mapping_known.csv")

# BED subdirectory names and their file prefix patterns
# Each list entry: name used in output | subdirectory | regex to match bed files
BED_SOURCES <- list(
  IRES = list(
    subdir  = "IRES_Literature",
    pattern = "IRES_literature_chr.*\\.bed$"
  ),
  nTIS = list(
    subdir  = "nTIS",
    pattern = ".*chr.*\\.bed$"
  ),
  uORF = list(
    subdir  = "uORF",
    pattern = ".*chr.*\\.bed$"
  )
)

# BioMart mirror preference
BIOMART_MIRRORS <- c("asia", "www", "useast", "uswest")

# Retry settings
BIOMART_RETRIES <- 3L   # attempts per batch before giving up
BIOMART_WAIT    <- 15L  # seconds to wait between retries


# =============================================================================
# PACKAGE LOADING
# =============================================================================

for (pkg in c("dplyr", "stringr", "biomaRt")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Package '", pkg, "' is required but not installed.")
  library(pkg, character.only = TRUE)
}


# =============================================================================
# HELPER: read all BED files for one annotation type
# =============================================================================

# Read and row-bind all BED files for one annotation type into a data frame.
read_bed_files <- function(subdir, pattern, label) {
  dir_path <- file.path(INPUT_DIR, subdir)
  if (!dir.exists(dir_path))
    stop("BED directory not found: ", dir_path)

  files <- list.files(dir_path, pattern = pattern, full.names = TRUE,
                      ignore.case = TRUE)
  if (length(files) == 0L)
    stop("No BED files matched pattern '", pattern, "' in: ", dir_path)

  message("  Reading ", length(files), " BED file(s) for [", label, "]...")

  rows <- lapply(files, function(f) {
    # BED files begin with 'track' and '#' comment lines — skip them
    raw <- readLines(f, warn = FALSE)
    data_lines <- raw[!grepl("^\\s*(track|#|browser)", raw, ignore.case = TRUE) &
                        nchar(trimws(raw)) > 0L]
    if (length(data_lines) == 0L) return(NULL)

    # Parse space- or tab-delimited fields
    # Expected columns: chrom chromStart chromEnd name score strand
    parsed <- do.call(rbind, lapply(data_lines, function(l) {
      fields <- strsplit(trimws(l), "\\s+")[[1L]]
      if (length(fields) < 4L) return(NULL)
      data.frame(
        chrom       = fields[1L],
        chromStart  = as.integer(fields[2L]),
        chromEnd    = as.integer(fields[3L]),
        name        = fields[4L],
        score       = if (length(fields) >= 5L) fields[5L] else NA_character_,
        strand      = if (length(fields) >= 6L) fields[6L] else NA_character_,
        stringsAsFactors = FALSE
      )
    }))
    parsed
  })

  bed <- do.call(rbind, Filter(Negate(is.null), rows))
  message("    → ", nrow(bed), " sites loaded")
  bed
}


# =============================================================================
# HELPER: extract RefSeq ID from BED name field
# =============================================================================
# Name format examples:
#   IRES_literature_NM_000254.2_154_426
#   nTIS_NM_001007022.3_3201_3290
#   uORF_NM_001007553.3_1_450
#
# Strategy: RefSeq accessions follow the pattern NM_\d+\.\d+ (or NR_, XM_, etc.)
# We extract the first occurrence of that pattern from the name string.

# Extract the RefSeq accession (NM_/NR_/XM_/XR_…) from each BED 'name' field.
extract_refseq <- function(name_vec) {
  # Capture NM_/NR_/XM_/XR_ followed by digits, dot, digits
  m <- regmatches(name_vec,
                  regexpr("[NX][MR]_[0-9]+\\.[0-9]+", name_vec, perl = TRUE))
  # regmatches returns character(0) for no-match elements — replace with NA
  ifelse(lengths(regmatches(name_vec,
                             gregexpr("[NX][MR]_[0-9]+\\.[0-9]+",
                                      name_vec, perl = TRUE))) == 0L,
         NA_character_, m)
}


# =============================================================================
# HELPER: map RefSeq → HGNC symbol via BioMart
# =============================================================================

# Connect to Ensembl BioMart: try each mirror, then a host fallback, with retries.
connect_ensembl <- function() {
  # Try useEnsembl() across all mirrors first
  for (m in BIOMART_MIRRORS) {
    for (attempt in seq_len(BIOMART_RETRIES)) {
      mart <- tryCatch(
        useEnsembl("genes", "hsapiens_gene_ensembl", mirror = m),
        error = function(e) NULL
      )
      if (!is.null(mart)) {
        message("  ✓ BioMart connected via useEnsembl (mirror: ", m, ")")
        return(mart)
      }
      if (attempt < BIOMART_RETRIES) {
        message("  ⚠ useEnsembl mirror '", m, "' attempt ", attempt, " failed — retrying in ",
                BIOMART_WAIT, "s...")
        Sys.sleep(BIOMART_WAIT)
      }
    }
  }

  # Fall back to useMart() with the public host
  message("  useEnsembl failed on all mirrors — trying useMart() fallback...")
  for (attempt in seq_len(BIOMART_RETRIES)) {
    mart <- tryCatch(
      useMart("ensembl", dataset = "hsapiens_gene_ensembl",
              host = "https://ensembl.org"),
      error = function(e) NULL
    )
    if (!is.null(mart)) {
      message("  ✓ BioMart connected via useMart() fallback")
      return(mart)
    }
    if (attempt < BIOMART_RETRIES) {
      message("  ⚠ useMart attempt ", attempt, " failed — retrying in ", BIOMART_WAIT, "s...")
      Sys.sleep(BIOMART_WAIT)
    }
  }

  stop("Could not connect to Ensembl BioMart via any mirror or fallback. ",
       "The server may be temporarily down — try again in a few minutes.")
}

# Map RefSeq mRNA IDs to HGNC symbols via BioMart (batched, bisecting on failure).
refseq_to_hgnc <- function(refseq_ids, mart) {
  ids <- unique(refseq_ids[!is.na(refseq_ids)])
  if (length(ids) == 0L) return(data.frame(refseq_mrna = character(),
                                            hgnc_symbol = character()))

  message("  Mapping ", length(ids), " unique RefSeq IDs to HGNC symbols...")

  # BioMart does not accept version suffixes (.1, .2) in refseq_mrna filter
  ids_noversion <- sub("\\.[0-9]+$", "", ids)

  # Batch query (groups of 200 to avoid server timeouts)
  batch_size <- 200L
  batches    <- split(ids_noversion, ceiling(seq_along(ids_noversion) / batch_size))

  # Recursive helper: query a sub-batch, bisecting on failure to isolate bad IDs
  query_with_bisect <- function(ids, depth = 0L) {
    if (length(ids) == 0L) return(NULL)

    for (attempt in seq_len(BIOMART_RETRIES)) {
      res <- tryCatch(
        getBM(
          attributes = c("refseq_mrna", "hgnc_symbol"),
          filters    = "refseq_mrna",
          values     = ids,
          mart       = mart
        ),
        error = function(e) {
          message(strrep("  ", depth + 1L),
                  "⚠ BioMart query (n=", length(ids), ") attempt ", attempt,
                  " failed: ", conditionMessage(e))
          NULL
        }
      )
      if (!is.null(res)) return(res)
      if (attempt < BIOMART_RETRIES) {
        message(strrep("  ", depth + 1L),
                "  Retrying in ", BIOMART_WAIT, "s...")
        Sys.sleep(BIOMART_WAIT)
      }
    }

    # All retries exhausted — bisect if more than one ID remains
    if (length(ids) == 1L) {
      message(strrep("  ", depth + 1L),
              "  ✗ Skipping unresolvable ID: ", ids)
      return(NULL)
    }

    mid  <- length(ids) %/% 2L
    message(strrep("  ", depth + 1L),
            "  Bisecting into halves of ", mid, " and ", length(ids) - mid, "...")
    rbind(
      query_with_bisect(ids[seq_len(mid)],          depth + 1L),
      query_with_bisect(ids[seq(mid + 1L, length(ids))], depth + 1L)
    )
  }

  results <- lapply(seq_along(batches), function(i) {
    message("  Querying batch ", i, "/", length(batches),
            " (n=", length(batches[[i]]), ")...")
    query_with_bisect(batches[[i]])
  })

  mapping <- do.call(rbind, Filter(Negate(is.null), results))
  mapping <- mapping[mapping$hgnc_symbol != "", ]
  message("    → ", nrow(mapping), " RefSeq→HGNC mappings retrieved")
  mapping
}


# =============================================================================
# MAIN
# =============================================================================

# 1. Load target genes
if (!file.exists(TARGET_CSV))
  stop("Target gene CSV not found: ", TARGET_CSV)

targets <- read.csv(TARGET_CSV, stringsAsFactors = FALSE)
required_cols <- c("Ensembl_ID", "RNA_gene")
missing <- setdiff(required_cols, colnames(targets))
if (length(missing) > 0L)
  stop("Missing columns in target CSV: ", paste(missing, collapse = ", "))

targets <- targets %>%
  filter(!is.na(RNA_gene) & RNA_gene != "") %>%
  dplyr::select(Ensembl_ID, RNA_gene) %>%
  distinct()

message("Target genes loaded: ", nrow(targets))

# 2. Connect to BioMart once for all annotation types
mart <- connect_ensembl()

# 3. Process each annotation type
all_hits <- list()

for (label in names(BED_SOURCES)) {
  src <- BED_SOURCES[[label]]
  message("\n=== Processing: ", label, " ===")

  # Read BED
  bed <- tryCatch(
    read_bed_files(src$subdir, src$pattern, label),
    error = function(e) { message("  ✗ ", e$message); NULL }
  )
  if (is.null(bed) || nrow(bed) == 0L) {
    message("  Skipping ", label, " — no data.")
    next
  }

  # Extract RefSeq IDs
  bed$refseq_id <- extract_refseq(bed$name)
  n_parsed <- sum(!is.na(bed$refseq_id))
  message("  RefSeq IDs extracted: ", n_parsed, " / ", nrow(bed), " sites")

  # Map RefSeq → HGNC
  mapping <- refseq_to_hgnc(bed$refseq_id, mart)

  # Strip version from bed refseq_id for join
  bed$refseq_noversion <- sub("\\.[0-9]+$", "", bed$refseq_id)

  # Attach HGNC to BED rows
  bed_mapped <- bed %>%
    left_join(mapping, by = c("refseq_noversion" = "refseq_mrna")) %>%
    filter(!is.na(hgnc_symbol) & hgnc_symbol != "")

  message("  Sites with HGNC symbol: ", nrow(bed_mapped))

  # Match against target genes
  hits <- bed_mapped %>%
    filter(hgnc_symbol %in% targets$RNA_gene) %>%
    group_by(hgnc_symbol) %>%
    summarise(
      n_sites      = n(),
      chromosomes  = paste(sort(unique(chrom)), collapse = ";"),
      strands      = paste(sort(unique(strand[!is.na(strand)])), collapse = ";"),
      refseq_ids   = paste(sort(unique(refseq_id[!is.na(refseq_id)])), collapse = ";"),
      .groups = "drop"
    ) %>%
    rename(RNA_gene = hgnc_symbol)

  message("  Target genes with ≥1 site: ", nrow(hits), " / ", nrow(targets))

  # Save per-type output
  out_file <- file.path(OUTPUT_DIR,
                         paste0(tolower(label), "_gene_hits.csv"))
  write.csv(hits, out_file, row.names = FALSE)
  message("  Saved: ", out_file)

  all_hits[[label]] <- hits
}

# 4. Build summary table: one row per target gene, one column per annotation type
message("\n=== Building summary table ===")

summary_df <- targets %>% dplyr::select(RNA_gene, Ensembl_ID)

for (label in names(BED_SOURCES)) {
  col_name <- paste0("has_", label)
  if (!is.null(all_hits[[label]])) {
    hit_genes <- all_hits[[label]]$RNA_gene
    n_col     <- paste0("n_sites_", label)
    summary_df[[col_name]] <- summary_df$RNA_gene %in% hit_genes
    summary_df[[n_col]]    <- ifelse(
      summary_df$RNA_gene %in% all_hits[[label]]$RNA_gene,
      all_hits[[label]]$n_sites[match(summary_df$RNA_gene, all_hits[[label]]$RNA_gene)],
      0L
    )
  } else {
    summary_df[[col_name]] <- NA
    summary_df[[paste0("n_sites_", label)]] <- NA_integer_
  }
}

summary_out <- file.path(OUTPUT_DIR, "bed_annotation_summary.csv")
write.csv(summary_df, summary_out, row.names = FALSE)
message("Summary saved: ", summary_out)

# 5. Print overview
message("\n=== Summary ===")
for (label in names(BED_SOURCES)) {
  col <- paste0("has_", label)
  if (col %in% colnames(summary_df)) {
    n_pos <- sum(summary_df[[col]] == TRUE, na.rm = TRUE)
    message(sprintf("  %-6s : %d / %d target genes have ≥1 site",
                    label, n_pos, nrow(targets)))
  }
}
message("\nDone. Output files written to: ", OUTPUT_DIR)
