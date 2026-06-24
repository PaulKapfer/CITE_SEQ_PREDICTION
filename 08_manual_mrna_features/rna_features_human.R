# =============================================================================
# RNA FEATURE EXTRACTION — HOMO SAPIENS
# =============================================================================
# Calculates mRNA-level features for a gene list (default: CollecTRI TFs).
# Restricted to RNA/post-transcriptional features only (no DNA/promoter features).
#
# INPUT FILES (all under <INPUT_DIR>, see CONFIGURATION; all optional —
# any missing file makes only its dependent feature(s) NA)
# --------------------
#   TIS_efficiency_Noderer2014.txt  – empirical TIS efficiency lookup table
#       Source: Noderer et al. (2014) Mol Syst Biol 10:748, Suppl. Table S2
#   uorf_gene_hits.csv              – genes carrying a validated uORF
#       Produced by match_bed_annotations.R from uORFdb BED tracks
#       (uORFdb: Manske et al. 2023, Nucleic Acids Res)
#   ires_gene_hits.csv              – genes with a literature-curated IRES
#       Produced by match_bed_annotations.R from Human IRES Atlas BED tracks
#       (Human IRES Atlas: Yang et al. 2021)
#   ntis_gene_hits.csv              – genes with documented noncanonical (non-AUG) TIS
#       Produced by match_bed_annotations.R (same Human IRES Atlas resource)
#   multimir_cache.rds              – validated miRNA→target table
#       Produced by generate_multimir_cache.R (run it first)
#
# OUTPUT FILES
# ------------
#   C:/R/Proteoscores/Input/mRNA_features/rna_features.csv   (main results)
#   C:/R/Proteoscores/Input/mRNA_features/mRNA_sequences_cache.rds  (sequence cache)
#
# FEATURES CALCULATED
# -------------------
#   5'UTR : length, GC, purine content, TIS efficiency (Noderer 2014),
#           TOP motif (terminal only), uORFs (ATG + near-cognate, uORFdb validation),
#           G-quadruplexes (pqsfinder), m6A density (DRACH),
#           MFE / MFE-normalized (optional, requires ViennaRNA),
#           IRES validation (human), noncanonical TIS validation (human)
#   CDS   : length, GC, purine content, CAI (Sharp & Li 1987, KAZUSA weights),
#           ENC (Wright 1990, native R implementation), GC3, m6A density (DRACH)
#   3'UTR : length, GC, purine content, ARE classification (Class I/II/III,
#           Barreau 2005 / Chen & Shyu 1995), PAS type + distance, m6A density
#   mRNA  : whole-transcript m6A count + density
#
# AUTHOR  : Paul [consolidated & extended from three prior scripts]
# UPDATED : 2026-04
# =============================================================================


# =============================================================================
# PACKAGE LOADING
# =============================================================================

required_packages <- c(
  "biomaRt",    # Ensembl sequence retrieval
  "Biostrings", # DNA/RNA sequence manipulation
  "dplyr",      # Data wrangling
  "stringr",    # String operations
  "parallel",   # Parallel processing
  "pbapply",    # Progress bars
  "decoupleR",  # CollecTRI TF list
  "pqsfinder"   # G-quadruplex detection (Hon et al. 2017 Bioinformatics)
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message("Warning: Package '", pkg, "' is not installed.")
  }
}


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR   <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual"
INPUT_DIR  <- file.path(BASE_DIR, "Input")
OUTPUT_DIR <- file.path(BASE_DIR, "Output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET_CSV        <- file.path(INPUT_DIR, "adt_rna_mapping_known.csv")
OUTPUT_FILE_MAIN  <- file.path(OUTPUT_DIR, "rna_features.csv")
OUTPUT_FILE_CACHE <- file.path(OUTPUT_DIR, "mRNA_sequences_cache.rds")

# Performance
N_CORES            <- parallel::detectCores() - 1
BIOMART_BATCH_SIZE <- 50

# Features
ENABLE_RNAFOLD <- TRUE  # requires ViennaRNA (RNAfold) in PATH

# Test mode
TEST_MODE    <- FALSE
TEST_N_GENES <- 4

# Optional external input paths (set to NULL to skip)
NODERER_TIS_TABLE <- file.path(INPUT_DIR, "TIS_efficiency_Noderer2014.txt")
UORFDB_FILE       <- file.path(INPUT_DIR, "uorf_gene_hits.csv")
IRES_FILE         <- file.path(INPUT_DIR, "ires_gene_hits.csv")
NONCANON_TIS_FILE <- file.path(INPUT_DIR, "ntis_gene_hits.csv")
MULTIMIR_CACHE    <- file.path(INPUT_DIR, "multimir_cache.rds")


# =============================================================================
# SECTION 1: GENE LIST RETRIEVAL
# =============================================================================

# Load the unique, non-empty HGNC gene symbols from the target CSV.
load_target_genes <- function() {
  message("--- Loading target genes from: ", TARGET_CSV, " ---")
  if (!file.exists(TARGET_CSV))
    stop("Target gene CSV not found: ", TARGET_CSV)
  df <- read.csv(TARGET_CSV, stringsAsFactors = FALSE)
  missing <- setdiff(c("Ensembl_ID", "RNA_gene"), colnames(df))
  if (length(missing) > 0L)
    stop("Missing columns in target CSV: ", paste(missing, collapse = ", "))
  genes <- unique(df$RNA_gene[!is.na(df$RNA_gene) & df$RNA_gene != ""])
  message("  ✓ ", length(genes), " unique HGNC symbols loaded")
  genes
}


# =============================================================================
# SECTION 2: SEQUENCE RETRIEVAL (Ensembl BioMart with batch + retry)
# =============================================================================

# Fetch 5'UTR/CDS/3'UTR for each gene's canonical transcript from Ensembl BioMart
# (cached; mirror failover; batched queries that bisect to isolate failing IDs).
get_mrna_sequences <- function(gene_symbols, use_cache = TRUE) {

  if (use_cache && file.exists(OUTPUT_FILE_CACHE)) {
    cached <- tryCatch(readRDS(OUTPUT_FILE_CACHE), error = function(e) NULL)
    if (!is.null(cached) && is.data.frame(cached) && nrow(cached) > 0) {
      message("  ✓ Loaded ", nrow(cached), " genes from cache")
      return(cached)
    }
    message("  Cache invalid — re-downloading.")
    unlink(OUTPUT_FILE_CACHE)
  }

  message("--- Retrieving sequences from Ensembl ---")

  # Connect with mirror failover
  mirrors <- c("asia", "www", "useast", "uswest")
  ensembl <- NULL
  for (m in mirrors) {
    ensembl <- tryCatch(
      useEnsembl("genes", "hsapiens_gene_ensembl", mirror = m),
      error = function(e) NULL
    )
    if (!is.null(ensembl)) { message("  ✓ Connected: ", m); break }
  }
  if (is.null(ensembl)) stop("Could not connect to any Ensembl mirror.")

  # Recursive helper: query ids, bisecting on failure to isolate bad IDs
  query_with_bisect <- function(ids, filter, attrs, max_retry = 3, depth = 0L) {
    if (length(ids) == 0L) return(NULL)
    for (attempt in seq_len(max_retry)) {
      res <- tryCatch(
        getBM(attrs, filter, ids, ensembl),
        error = function(e) { Sys.sleep(attempt * 3); NULL }
      )
      if (!is.null(res)) return(res)
    }
    if (length(ids) == 1L) {
      message(strrep("  ", depth + 1L), "✗ Skipping unresolvable ID: ", ids)
      return(NULL)
    }
    mid <- length(ids) %/% 2L
    message(strrep("  ", depth + 1L),
            "Bisecting failed batch (n=", length(ids), ") into ", mid,
            " + ", length(ids) - mid, "...")
    rbind(
      query_with_bisect(ids[seq_len(mid)],               filter, attrs, max_retry, depth + 1L),
      query_with_bisect(ids[seq(mid + 1L, length(ids))], filter, attrs, max_retry, depth + 1L)
    )
  }

  # Helper: batched query with retry + bisection on failure
  safe_batch_query <- function(ids, filter, attrs, batch = BIOMART_BATCH_SIZE, max_retry = 3) {
    batches <- split(ids, ceiling(seq_along(ids) / batch))
    results <- lapply(seq_along(batches), function(bi) {
      message("  Querying batch ", bi, "/", length(batches), " (n=", length(batches[[bi]]), ")...")
      query_with_bisect(batches[[bi]], filter, attrs, max_retry)
    })
    do.call(rbind, Filter(Negate(is.null), results))
  }

  # Query 1–5: annotation pages (split to avoid multi-page errors)
  message("  Fetching transcript metadata...")
  basic    <- safe_batch_query(gene_symbols, "hgnc_symbol",
                               c("hgnc_symbol", "ensembl_transcript_id"))
  t_ids    <- unique(basic$ensembl_transcript_id)

  utr_pos  <- safe_batch_query(t_ids, "ensembl_transcript_id",
                               c("ensembl_transcript_id", "5_utr_start", "5_utr_end"))
  appris   <- safe_batch_query(t_ids, "ensembl_transcript_id",
                               c("ensembl_transcript_id", "transcript_appris"))
  mane     <- safe_batch_query(t_ids, "ensembl_transcript_id",
                               c("ensembl_transcript_id", "transcript_mane_select"))
  tsl      <- safe_batch_query(t_ids, "ensembl_transcript_id",
                               c("ensembl_transcript_id", "transcript_tsl"))

  transcripts <- basic %>%
    left_join(utr_pos, by = "ensembl_transcript_id") %>%
    left_join(appris,  by = "ensembl_transcript_id") %>%
    left_join(mane,    by = "ensembl_transcript_id") %>%
    left_join(tsl,     by = "ensembl_transcript_id")

  # Canonical transcript selection: MANE > APPRIS principal > TSL
  canonical <- transcripts %>%
    filter(!is.na(`5_utr_start`)) %>%
    group_by(hgnc_symbol) %>%
    arrange(
      desc(!is.na(transcript_mane_select) & transcript_mane_select != ""),
      desc(transcript_appris == "principal1"),
      desc(grepl("principal", transcript_appris)),
      transcript_tsl
    ) %>%
    slice(1) %>% ungroup()

  message("  ✓ ", nrow(canonical), " canonical transcripts selected")
  can_ids <- canonical$ensembl_transcript_id

  # Retrieve sequences
  message("  Fetching 5'UTR sequences...")
  utr5 <- safe_batch_query(can_ids, "ensembl_transcript_id",
                            c("hgnc_symbol", "ensembl_transcript_id", "5utr"))

  message("  Fetching 3'UTR sequences...")
  utr3 <- safe_batch_query(can_ids, "ensembl_transcript_id",
                            c("hgnc_symbol", "ensembl_transcript_id", "3utr"))

  message("  Fetching CDS sequences (smaller batches)...")
  cds_batch <- max(10L, BIOMART_BATCH_SIZE %/% 5L)
  cds  <- safe_batch_query(can_ids, "ensembl_transcript_id",
                            c("hgnc_symbol", "ensembl_transcript_id", "coding"),
                            batch = cds_batch, max_retry = 5L)

  # Merge
  result <- canonical %>%
    select(hgnc_symbol, ensembl_transcript_id, transcript_appris, transcript_mane_select) %>%
    left_join(utr5, by = c("hgnc_symbol", "ensembl_transcript_id")) %>%
    left_join(utr3, by = c("hgnc_symbol", "ensembl_transcript_id")) %>%
    left_join(cds,  by = c("hgnc_symbol", "ensembl_transcript_id")) %>%
    filter(!is.na(`5utr`) | !is.na(`3utr`) | !is.na(coding))

  saveRDS(result, OUTPUT_FILE_CACHE)
  message("  ✓ Cached ", nrow(result), " sequences to: ", OUTPUT_FILE_CACHE)
  return(result)
}


# =============================================================================
# SECTION 3: OPTIONAL RESOURCE LOADERS (loaded once, used by all genes)
# =============================================================================

TIS_EFFICIENCY_TABLE <- NULL
UORFDB_DATA          <- NULL
IRES_GENES           <- NULL
NONCANONICAL_TIS_GENES <- NULL

# Load the Noderer 2014 TIS-efficiency lookup table (11-nt context -> score).
load_tis_table <- function() {
  # Noderer WL et al. (2014) Mol Syst Biol 10:748
  # Empirical TIS efficiency scores for all 65,536 possible 11-nt AUG contexts
  # from FACS-seq in mouse pre-B lymphocytes (Kozak context conserved across vertebrates)
  if (!file.exists(NODERER_TIS_TABLE)) {
    message("  ⚠ TIS table not found: ", NODERER_TIS_TABLE); return(NULL)
  }
  tis <- read.delim(NODERER_TIS_TABLE, skip = 1, header = TRUE, stringsAsFactors = FALSE)
  setNames(tis$efficiency, tis$sequence)
}

# Load the validated-uORF reference table (uORFdb), or NULL if absent.
load_uorfdb <- function() {
  if (!file.exists(UORFDB_FILE)) {
    message("  ⚠ uORFdb not found: ", UORFDB_FILE); return(NULL)
  }
  db <- read.csv(UORFDB_FILE, header = TRUE, stringsAsFactors = FALSE, comment.char = "#")
  message("  ✓ Loaded uORFdb: ", nrow(db), " validated uORFs")
  db
}

# Load the list of genes carrying a literature-curated IRES, or NULL if absent.
load_ires_genes <- function() {
  if (!file.exists(IRES_FILE)) {
    message("  ⚠ IRES gene list not found: ", IRES_FILE); return(NULL)
  }
  genes <- read.csv(IRES_FILE, header = FALSE, stringsAsFactors = FALSE)[, 1]
  message("  ✓ Loaded ", length(genes), " IRES-validated genes")
  genes
}

# Load the list of genes with documented non-AUG translation initiation, or NULL.
load_noncanon_tis_genes <- function() {
  if (!file.exists(NONCANON_TIS_FILE)) {
    message("  ⚠ Noncanonical TIS list not found: ", NONCANON_TIS_FILE); return(NULL)
  }
  genes <- read.csv(NONCANON_TIS_FILE, header = FALSE, stringsAsFactors = FALSE)[, 1]
  message("  ✓ Loaded ", length(genes), " noncanonical TIS genes")
  genes
}


# =============================================================================
# SECTION 4: FEATURE FUNCTIONS — 5'UTR
# =============================================================================

# --- Basic 5'UTR properties ---
calculate_5utr_basic <- function(utr5) {
  if (is.na(utr5) || utr5 == "") return(list(
    utr5_length = NA, utr5_gc_content = NA, utr5_purine_content = NA))
  len <- nchar(utr5)
  list(
    utr5_length        = len,
    utr5_gc_content    = round(str_count(utr5, "[GCgc]") / len, 4),
    utr5_purine_content = round(str_count(utr5, "[AGag]") / len, 4)
  )
}

# --- TIS efficiency (Noderer 2014) ---
# Reference: Noderer WL et al. (2014) Mol Syst Biol 10:748
# "Quantitative analysis of mammalian translation initiation sites by FACS-seq"
# Empirical efficiency score (20–150) for the 11-nt window [-6..ATG..+2]
get_tis_efficiency <- function(utr5, cds) {
  if (is.null(TIS_EFFICIENCY_TABLE) || is.na(utr5) || nchar(utr5) < 6) return(NA)
  if (is.na(cds) || nchar(cds) < 5) return(NA)
  utr5 <- toupper(utr5); cds <- toupper(cds)
  upstream <- substr(utr5, nchar(utr5) - 5L, nchar(utr5))
  start    <- substr(cds, 1L, 3L)
  down     <- substr(cds, 4L, 5L)
  tis_seq  <- gsub("T", "U", paste0(upstream, start, down))
  if (substr(tis_seq, 7L, 9L) != "AUG") return(NA)
  eff <- TIS_EFFICIENCY_TABLE[[tis_seq]]
  if (is.null(eff)) return(NA)
  as.numeric(eff)
}

# --- TOP motif detection (terminal only) ---
# Reference: Meyuhas O & Kahan T (2015) Biochim Biophys Acta 1849:801–811
# "An invariable C residue at the cap site, followed by an uninterrupted stretch
#  of 4 to 15 pyrimidines; ... a CG-rich region immediately downstream"
# IMPLEMENTATION NOTE: search is anchored to the 5' terminus (^) so that only
# true terminal TOP motifs are detected — internal pyrimidine runs are excluded.
has_TOP_candidate <- function(utr5) {
  if (is.na(utr5) || utr5 == "") return(NA)
  utr5 <- toupper(gsub("U", "T", utr5))

  # Criterion 1: C at +1 followed by 4–15 consecutive pyrimidines — anchored at 5' end
  pyr_match <- regexpr("^C[CT]{4,15}", utr5)
  if (pyr_match == -1L) return(FALSE)
  pyr_end <- attr(pyr_match, "match.length")  # length of terminal pyrimidine tract

  # Criterion 2: CG-rich region immediately downstream of the tract
  ds_start <- pyr_end + 1L
  if (ds_start > nchar(utr5)) return(NA)  # no downstream sequence
  ds_end    <- min(ds_start + 19L, nchar(utr5))
  ds_region <- substr(utr5, ds_start, ds_end)
  if (nchar(ds_region) < 10L) return(NA)  # insufficient downstream sequence

  cg_ratio <- nchar(gsub("[AT]", "", ds_region)) / nchar(ds_region)
  return(cg_ratio > 0.60)
}

# --- G-quadruplex detection ---
# Reference: Hon J et al. (2017) Bioinformatics 33:3373–3379 (pqsfinder)
# Score threshold ≥25: Puig Lombardi & Londoño-Vallejo (2020) NAR 48:1–15
detect_g_quadruplex <- function(utr5, min_score = 25L) {
  empty <- list(utr5_g4_max_score = NA, utr5_g4_count = NA,
                utr5_g4_mean_score = NA, utr5_g4_max_tetrads = NA)
  if (is.na(utr5) || utr5 == "") return(empty)
  dna <- Biostrings::DNAString(toupper(gsub("U", "T", utr5)))
  res <- tryCatch({
    invisible(capture.output(
      pqs <- pqsfinder::pqsfinder(dna, strand = "+", min_score = min_score,
                                  max_defects = 3L),
      type = "message"
    ))
    pqs
  }, error = function(e) NULL)
  if (is.null(res) || length(res) == 0L)
    return(list(utr5_g4_max_score = 0L, utr5_g4_count = 0L,
                utr5_g4_mean_score = 0, utr5_g4_max_tetrads = 0L))
  pmd <- elementMetadata(res)
  list(
    utr5_g4_max_score   = max(score(res)),
    utr5_g4_count       = length(res),
    utr5_g4_mean_score  = mean(score(res)),
    utr5_g4_max_tetrads = max(pmd$nt)
  )
}

# --- uORF analysis ---
# Reference: Ingolia NT et al. (2011) Science 333:1693–1697 (ribosome profiling)
#            Johnstone TG et al. (2016) Genome Res 26:976–987 (near-cognate uORFs)
#            Wethmar K et al. (2014) Nucleic Acids Res 42:D92–D97 (uORFdb)
analyze_uorfs <- function(utr5, cds = NULL) {
  empty <- list(uORF_ATG_count = NA, uORF_nonATG_count = NA,
                uORF_longest_length = NA, uORF_distance_to_CDS = NA,
                uORF_overlaps_CDS = NA, has_validated_uORF = NA)
  if (is.na(utr5) || utr5 == "") return(empty)

  utr5_u  <- toupper(utr5)
  utr5_len <- nchar(utr5_u)
  stop_codons <- c("TAA", "TAG", "TGA")

  # Validated uORF check via uORFdb
  has_validated <- FALSE
  if (!is.null(UORFDB_DATA)) {
    sp_uorfs <- UORFDB_DATA[UORFDB_DATA$Taxon == "Homo sapiens", ]
    search_seq <- paste0(utr5_u, if (!is.na(cds) && cds != "") toupper(cds) else "")
    for (i in seq_len(nrow(sp_uorfs))) {
      uorf_seq <- toupper(sp_uorfs$uORFnucleotideSeq[i])
      if (is.na(uorf_seq) || uorf_seq == "") next
      positions <- gregexpr(uorf_seq, search_seq, fixed = TRUE)[[1L]]
      if (positions[1L] != -1L && any(positions <= utr5_len)) {
        has_validated <- TRUE; break
      }
    }
  }

  # De-novo uORF detection
  find_uorfs <- function(start_codon) {
    positions <- gregexpr(start_codon, utr5_u)[[1L]]
    if (positions[1L] == -1L) return(list())
    lapply(positions, function(pos) {
      remaining <- substr(utr5_u, pos, utr5_len)
      if (nchar(remaining) < 6L)   # need at least start codon + one codon
        return(list(start = pos, stop = utr5_len, len_codons = 1L, overlaps = TRUE))
      found_stop <- FALSE; stop_offset <- NA
      for (j in seq(4L, nchar(remaining) - 2L, by = 3L)) {
        if (nchar(remaining) < j + 2L) break
        if (substr(remaining, j, j + 2L) %in% stop_codons) {
          stop_offset <- j; found_stop <- TRUE; break
        }
      }
      if (found_stop) {
        stop_abs <- pos + stop_offset + 2L
        list(start = pos, stop = stop_abs,
             len_codons = (stop_abs - pos + 1L) / 3L, overlaps = FALSE)
      } else {
        list(start = pos, stop = utr5_len,
             len_codons = (utr5_len - pos + 1L) / 3L, overlaps = TRUE)
      }
    })
  }

  atg_uorfs     <- find_uorfs("ATG")
  nonatg_uorfs  <- unlist(lapply(c("CTG","GTG","ACG","TTG"), find_uorfs), recursive = FALSE)
  all_uorfs     <- c(atg_uorfs, nonatg_uorfs)

  if (length(all_uorfs) == 0L)
    return(list(uORF_ATG_count = 0L, uORF_nonATG_count = 0L,
                uORF_longest_length = NA, uORF_distance_to_CDS = NA,
                uORF_overlaps_CDS = FALSE, has_validated_uORF = has_validated))

  overlaps <- any(sapply(all_uorfs, `[[`, "overlaps"))
  list(
    uORF_ATG_count        = length(atg_uorfs),
    uORF_nonATG_count     = length(nonatg_uorfs),
    uORF_longest_length   = round(max(sapply(all_uorfs, `[[`, "len_codons")), 1L),
    uORF_distance_to_CDS  = if (overlaps) 0L else
                              utr5_len - max(sapply(all_uorfs, `[[`, "stop")),
    uORF_overlaps_CDS     = overlaps,
    has_validated_uORF    = has_validated
  )
}

# --- RNA secondary structure MFE (5'UTR and 3'UTR) ---
# Reference: Lorenz R et al. (2011) Algorithms Mol Biol 6:26 (ViennaRNA / RNAfold)
# MFE is normalized by sequence length to allow cross-gene comparison
calculate_mfe <- function(seq, prefix) {
  empty <- setNames(list(NA_real_, NA_real_),
                    c(paste0(prefix, "_mfe"), paste0(prefix, "_mfe_normalized")))
  if (!ENABLE_RNAFOLD) return(empty)
  if (is.na(seq) || seq == "") {
    message("  [MFE/", prefix, "] skipped: sequence is NA/empty")
    return(empty)
  }
  if (nchar(seq) < 10L) {
    message("  [MFE/", prefix, "] skipped: sequence too short (", nchar(seq), " nt)")
    return(empty)
  }

  # RNAfold requires only ACGU/ACGT — strip anything else
  seq_clean <- gsub("[^ACGTUacgtu]", "N", seq)

  tmp <- tempfile(fileext = ".fa"); on.exit(unlink(tmp), add = TRUE)
  writeLines(c(">seq", seq_clean), tmp)
  out <- tryCatch(
    system2("RNAfold", c("--noPS", tmp), stdout = TRUE, stderr = FALSE),
    error = function(e) { message("  [MFE/", prefix, "] RNAfold error: ", e$message); character(0) }
  )
  if (length(out) < 3L) {
    message("  [MFE/", prefix, "] unexpected RNAfold output (", length(out), " lines): ",
            paste(out, collapse = " | "))
    return(empty)
  }
  # Structure + MFE is always the last output line; earlier lines are the (possibly
  # wrapped) sequence echo and header — do not assume a fixed line index.
  struct_line <- out[length(out)]
  m <- regmatches(struct_line, regexpr("\\(\\s*[-+]?[0-9]+\\.[0-9]+\\s*\\)\\s*$", struct_line, perl = TRUE))
  if (length(m) == 0L) {
    message("  [MFE/", prefix, "] could not parse MFE from: ", struct_line)
    return(empty)
  }
  mfe <- as.numeric(gsub("[() \r\n]", "", m[1L]))
  setNames(list(round(mfe, 2L), round(mfe / nchar(seq), 4L)),
           c(paste0(prefix, "_mfe"), paste0(prefix, "_mfe_normalized")))
}

# --- IRES & noncanonical TIS (experimental) ---
check_ires  <- function(gene) {
  if (is.null(IRES_GENES)          || is.na(gene)) return(NA)
  gene %in% IRES_GENES
}
# TRUE/FALSE (NA if no list) — is this gene in the noncanonical-TIS set?
check_noncanon_tis <- function(gene) {
  if (is.null(NONCANONICAL_TIS_GENES) || is.na(gene)) return(NA)
  gene %in% NONCANONICAL_TIS_GENES
}


# =============================================================================
# SECTION 5: FEATURE FUNCTIONS — CDS
# =============================================================================

# --- CAI: Codon Adaptation Index ---
# Reference: Sharp PM & Li W-H (1987) Nucleic Acids Res 15:1281–1295
# Weights from KAZUSA Codon Usage Database, Homo sapiens (9606)
# Dataset: 93,487 CDS entries; Nakamura Y et al. (2000) Nucleic Acids Res 28:292
# w(codon) = freq(codon) / max(freq of synonymous codons)
HUMAN_CODON_WEIGHTS <- list(
  "TTT"=0.867,"TTC"=1.000,
  "TTA"=0.194,"TTG"=0.326,"CTT"=0.333,"CTC"=0.495,"CTA"=0.182,"CTG"=1.000,
  "ATT"=0.769,"ATC"=1.000,"ATA"=0.361,
  "ATG"=1.000,
  "GTT"=0.391,"GTC"=0.516,"GTA"=0.253,"GTG"=1.000,
  "TCT"=0.779,"TCC"=0.908,"TCA"=0.626,"TCG"=0.226,"AGT"=0.621,"AGC"=1.000,
  "CCT"=0.884,"CCC"=1.000,"CCA"=0.854,"CCG"=0.348,
  "ACT"=0.693,"ACC"=1.000,"ACA"=0.799,"ACG"=0.323,
  "GCT"=0.664,"GCC"=1.000,"GCA"=0.570,"GCG"=0.267,
  "TAT"=0.797,"TAC"=1.000,
  "CAT"=0.722,"CAC"=1.000,
  "CAA"=0.360,"CAG"=1.000,
  "AAT"=0.890,"AAC"=1.000,
  "AAA"=0.765,"AAG"=1.000,
  "GAT"=0.869,"GAC"=1.000,
  "GAA"=0.732,"GAG"=1.000,
  "TGT"=0.841,"TGC"=1.000,
  "TGG"=1.000,
  "CGT"=0.369,"CGC"=0.852,"CGA"=0.508,"CGG"=0.934,"AGA"=1.000,"AGG"=0.984,
  "GGT"=0.486,"GGC"=1.000,"GGA"=0.743,"GGG"=0.743,
  "TAA"=NA,"TAG"=NA,"TGA"=NA
)

# Codon Adaptation Index: geometric mean of per-codon weights over the CDS.
calculate_cai <- function(cds) {
  if (is.na(cds) || nchar(cds) < 9L) return(NA)
  tryCatch({
    cds <- toupper(gsub("U","T", cds))
    trim <- nchar(cds) - (nchar(cds) %% 3L)
    if (trim < 9L) return(NA)
    cds <- substr(cds, 1L, trim)
    codons <- substring(cds, seq(1L, trim - 2L, 3L), seq(3L, trim, 3L))
    w <- sapply(codons, function(c) {
      v <- HUMAN_CODON_WEIGHTS[[c]]; if (is.null(v)) NA else v
    })
    w <- w[!is.na(w)]
    if (length(w) < 3L) return(NA)
    round(exp(mean(log(w))), 4L)
  }, error = function(e) NA)
}

# --- ENC: Effective Number of Codons ---
# Reference: Wright F (1990) Gene 87:23–29
# "The 'effective number of codons' used in a gene"
# Range: 20 (maximum bias, one codon per AA) to 61 (no bias, all codons equal)
# Minimum 100 codons (300 nt) required for reliable estimation (Wright 1990).
#
# IMPLEMENTATION: Wright's formula computed in native R.
# The coRdon package wraps this calculation but returns class-specific objects
# that routinely produce NA when called on a single sequence via DNAStringSet.
# The native implementation below faithfully reproduces Wright's F_hat estimator:
#   F_k = (sum(n_ij^2 / n_i) - 1) / (n_i - 1) per amino-acid family k
#   ENC  = 2 + 9/F_2 + 1/F_3 + 5/F_4 + 3/F_6   (approximate degeneracy counts)
calculate_enc <- function(cds) {
  if (is.na(cds) || cds == "") return(NA)
  cds <- toupper(gsub("U", "T", cds))
  trim <- nchar(cds) - (nchar(cds) %% 3L)
  if (trim < 300L) return(NA)  # < 100 codons: high variance (Wright 1990)
  cds <- substr(cds, 1L, trim)
  n   <- trim / 3L
  codons <- substring(cds, seq(1L, trim - 2L, 3L), seq(3L, trim, 3L))

  # Codon → amino-acid degeneracy family mapping (DNA, standard genetic code)
  aa_families <- list(
    # 2-fold
    Phe=c("TTT","TTC"), Tyr=c("TAT","TAC"), Cys=c("TGT","TGC"),
    His=c("CAT","CAC"), Gln=c("CAA","CAG"), Asn=c("AAT","AAC"),
    Lys=c("AAA","AAG"), Asp=c("GAT","GAC"), Glu=c("GAA","GAG"),
    # 3-fold (Ile)
    Ile=c("ATT","ATC","ATA"),
    # 4-fold
    Val=c("GTT","GTC","GTA","GTG"),  Pro=c("CCT","CCC","CCA","CCG"),
    Thr=c("ACT","ACC","ACA","ACG"),  Ala=c("GCT","GCC","GCA","GCG"),
    Gly=c("GGT","GGC","GGA","GGG"),
    # 6-fold
    Leu=c("TTA","TTG","CTT","CTC","CTA","CTG"),
    Arg=c("CGT","CGC","CGA","CGG","AGA","AGG"),
    Ser=c("TCT","TCC","TCA","TCG","AGT","AGC"),
    # 1-fold (Met, Trp) — excluded from ENC (no synonymous variation)
    # Stop codons — excluded
    Stop=c("TAA","TAG","TGA")
  )

  # Degeneracy class membership
  deg2 <- c("Phe","Tyr","Cys","His","Gln","Asn","Lys","Asp","Glu")
  deg3 <- c("Ile")
  deg4 <- c("Val","Pro","Thr","Ala","Gly")
  deg6 <- c("Leu","Arg","Ser")

  # Wright's F_hat per amino-acid family:
  #   F = (sum_j(n_j^2 / n_total) - 1) / (n_total - 1)
  # where n_j = count of codon j, n_total = sum(n_j) for that family.
  # Returns NA if n_total < 2 (cannot compute F for unobserved families).
  compute_F <- function(aa_name) {
    codons_aa <- aa_families[[aa_name]]
    counts    <- table(factor(codons[codons %in% codons_aa], levels = codons_aa))
    n_total   <- sum(counts)
    if (n_total < 2L) return(NA_real_)
    sum((counts / n_total)^2)
    # Equivalent to Wright's chi_k: sum(p_ij^2) where p_ij = n_ij/n_i
    # F_hat = (n * sum_p^2 - 1) / (n - 1)
    chi <- sum((as.numeric(counts))^2) / n_total
    (chi - 1) / (n_total - 1)
  }

  # Mean F per degeneracy class (NA-robust)
  mean_F <- function(aa_names) {
    fs <- sapply(aa_names, compute_F)
    fs <- fs[!is.na(fs)]
    if (length(fs) == 0L) return(NA_real_)
    mean(fs)
  }

  F2 <- mean_F(deg2)
  F3 <- mean_F(deg3)
  F4 <- mean_F(deg4)
  F6 <- mean_F(deg6)

  # ENC = 2 + S2/F2 + S3/F3 + S4/F4 + S6/F6
  # where S_k = number of amino-acid families in degeneracy class k
  # 2-fold: 9 AAs; 3-fold: 1 AA (Ile); 4-fold: 5 AAs; 6-fold: 3 AAs
  terms <- c(
    if (!is.na(F2) && F2 > 0) 9  / F2 else NA,
    if (!is.na(F3) && F3 > 0) 1  / F3 else NA,
    if (!is.na(F4) && F4 > 0) 5  / F4 else NA,
    if (!is.na(F6) && F6 > 0) 3  / F6 else NA
  )

  if (all(is.na(terms))) return(NA_real_)
  enc <- 2 + sum(terms, na.rm = TRUE)
  round(min(max(enc, 20), 61), 2L)  # clamp to theoretical range
}

# --- GC3: GC content at third codon position ---
# Correlates with tRNA availability without requiring species-specific tables.
# Reference: Tatarinova T et al. (2010) BMC Genomics 11:395 — GC3 as a gene
# expression correlate; Sharp PM & Li W-H (1987) Nucleic Acids Res 15:1281
calculate_gc3 <- function(cds) {
  if (is.na(cds) || nchar(cds) < 3L) return(NA_real_)
  cds <- toupper(gsub("U","T", cds))
  n_codons <- floor(nchar(cds) / 3L)
  if (n_codons == 0L) return(NA_real_)
  pos3 <- seq(3L, n_codons * 3L, by = 3L)
  b3   <- substring(cds, pos3, pos3)
  round(sum(b3 %in% c("G","C")) / n_codons, 4L)
}


# =============================================================================
# SECTION 6: FEATURE FUNCTIONS — 3'UTR
# =============================================================================

# --- Basic 3'UTR properties ---
calculate_3utr_basic <- function(utr3) {
  if (is.na(utr3) || utr3 == "") return(list(
    utr3_length = NA, utr3_gc_content = NA, utr3_purine_content = NA))
  len <- nchar(utr3)
  list(
    utr3_length         = len,
    utr3_gc_content     = round(str_count(utr3, "[GCgc]") / len, 4),
    utr3_purine_content = round(str_count(utr3, "[AGag]") / len, 4)
  )
}

# --- ARE motifs (AU-rich elements) ---
# References:
#   Chen CY & Shyu AB (1995) Trends Biochem Sci 20:465–470 — original ARE classification
#   Barreau C et al. (2005) Nucleic Acids Res 33:7138–7150 — revised criteria
#
# Class I : ≥2 AUUUA pentamers dispersed in U-rich context (≥40% U in 50-nt window)
# Class II : ≥2 overlapping UUAUUUA(U/A)(U/A) nonamers
# Class III: U-rich region (≥60% U in ≥50 nt), no AUUUA
count_are_motifs <- function(utr3) {
  empty <- list(are_class1=NA, are_class2=NA, are_class3=NA,
                are_nonamers=NA, are_pentamers=NA)
  if (is.na(utr3) || utr3 == "") return(empty)

  rna <- gsub("T","U", toupper(utr3))
  len <- nchar(rna)
  covered <- rep(FALSE, len)

  # Nonamers
  nm_pos <- gregexpr("UUAUUUA[UA][UA]", rna)[[1L]]
  n_nom  <- if (nm_pos[1L] != -1L) length(nm_pos) else 0L

  # Class II: ≥2 overlapping nonamers
  class2 <- 0L
  if (n_nom >= 2L) {
    for (i in seq_len(n_nom - 1L)) {
      if (nm_pos[i + 1L] < nm_pos[i] + 9L) {
        class2 <- class2 + 1L
        r <- nm_pos[i]:(min(nm_pos[i + 1L] + 8L, len))
        covered[r] <- TRUE
      }
    }
  }

  # Pentamers
  pent_pos <- gregexpr("AUUUA", rna)[[1L]]
  n_pent   <- if (pent_pos[1L] != -1L) length(pent_pos) else 0L
  uncov_pent <- if (pent_pos[1L] != -1L)
    Filter(function(p) !any(covered[p:min(p+4L, len)]), pent_pos) else c()

  # Class I: ≥2 uncovered pentamers in U-rich context
  class1 <- 0L
  pent_in_urich <- 0L
  for (p in uncov_pent) {
    ws <- max(1L, p - 22L); we <- min(len, p + 27L)
    win <- substr(rna, ws, we)
    if (str_count(win, "U") / nchar(win) >= 0.40) {
      pent_in_urich <- pent_in_urich + 1L
      covered[p:min(p+4L, len)] <- TRUE
    }
  }
  if (pent_in_urich >= 2L) class1 <- 1L

  # Class III: U-rich (≥60%) ≥50 nt window, no AUUUA
  class3 <- 0L
  if (len >= 50L) {
    for (i in seq_len(len - 49L)) {
      if (any(covered[i:(i + 49L)])) next
      win <- substr(rna, i, i + 49L)
      if (str_count(win, "U") / 50L >= 0.60 && !grepl("AUUUA", win)) {
        class3 <- class3 + 1L
        covered[i:(i + 49L)] <- TRUE
      }
    }
  }

  list(are_class1 = class1, are_class2 = class2, are_class3 = class3,
       are_nonamers = n_nom, are_pentamers = n_pent)
}

# --- PAS: polyadenylation signal ---
# Motif hierarchy from Tian B & Manley JL (2017) Nat Rev Mol Cell Biol 18:18–30
# AATAAA is the canonical PAS (~70% of human genes); 10 alternative hexamers account
# for the majority of remaining genes.
detect_pas <- function(utr3) {
  motifs <- c("AATAAA","ATTAAA","AGTAAA","TATAAA","CATAAA","GATAAA",
              "AATATA","AATACA","AATGAA","ACTAAA","AATAGA")
  if (is.na(utr3) || nchar(utr3) < 50L)
    return(list(PAS_type = "None", PAS_distance = NA))
  tail <- substr(utr3, max(1L, nchar(utr3) - 100L), nchar(utr3))
  for (m in motifs) {
    p <- regexpr(m, tail)
    if (p > 0L)
      return(list(PAS_type = m,
                  PAS_distance = nchar(tail) - (p + nchar(m) - 1L)))
  }
  list(PAS_type = "None", PAS_distance = NA)
}


# =============================================================================
# SECTION 7: FEATURE FUNCTIONS — SHARED (m6A)
# =============================================================================

# --- m6A DRACH motif density ---
# Reference: Dominissini D et al. (2012) Nature 485:201–206 (m6A transcriptome)
#            Linder B et al. (2015) Nat Methods 12:767–772 (DRACH consensus)
# DRACH = [AGU][AG]AC[ACU]
# Density reported per 100 nt; minimum 50 nt for meaningful density.
count_m6a <- function(seq) {
  if (is.na(seq) || seq == "")
    return(list(m6a_count = NA, m6a_density = NA))
  rna <- gsub("T","U", toupper(seq))
  cnt <- str_count(rna, "[AGU][AG]AC[ACU]")
  list(
    m6a_count   = cnt,
    m6a_density = if (nchar(rna) >= 50L) round(cnt / nchar(rna) * 100, 4L) else NA
  )
}


# =============================================================================
# SECTION 8: miRNA density (multiMiR or fallback)
# =============================================================================

# Per-gene miRNA target count and density (sites / 3'UTR length) from the cache.
get_mirna_features <- function(gene, utr3, mir_cache) {
  utr3_len <- if (!is.na(utr3) && utr3 != "") nchar(utr3) else NA_integer_
  if (!is.null(mir_cache) && "target_symbol" %in% colnames(mir_cache)) {
    hits <- mir_cache[mir_cache$target_symbol == gene, ]
    cnt  <- nrow(hits)
  } else {
    cnt <- str_count(if (!is.na(utr3)) utr3 else "", "[AGCT]{7}")
  }
  list(
    miRNA_count   = cnt,
    miRNA_density = if (!is.na(utr3_len) && utr3_len > 0L) cnt / utr3_len else NA
  )
}


# =============================================================================
# SECTION 9: MAIN PROCESSING
# =============================================================================

# Driver: load resources, fetch sequences, compute all features per gene (parallel),
# merge with any prior output, and write rna_features.csv.
extract_rna_features <- function(gene_list, use_cache = TRUE) {

  # --- RNAfold smoke test (runs first so you see immediately if it works) ---
  message("\n=== RNAfold smoke test ===")
  rnafold_ok <- FALSE
  if (ENABLE_RNAFOLD) {
    ver_ok <- tryCatch(
      system2("RNAfold", "--version", stdout = TRUE, stderr = FALSE),
      error = function(e) character(0)
    )
    if (length(ver_ok) > 0L) {
      message("  RNAfold found: ", ver_ok[1L])
      test_seq <- "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
      tmp <- tempfile(fileext = ".fa"); on.exit(unlink(tmp), add = TRUE)
      writeLines(c(">test", test_seq), tmp)
      test_out <- tryCatch(
        system2("RNAfold", c("--noPS", tmp), stdout = TRUE, stderr = FALSE),
        error = function(e) character(0)
      )
      if (length(test_out) >= 3L) {
        message("  ✓ RNAfold output: ", test_out[3L])
        rnafold_ok <- TRUE
      } else {
        message("  ✗ RNAfold ran but produced unexpected output — MFE will be NA")
      }
    } else {
      message("  ✗ RNAfold not found in PATH — MFE features will be NA")
      message("    Install ViennaRNA and ensure 'RNAfold' is on your PATH")
    }
  } else {
    message("  ENABLE_RNAFOLD = FALSE — skipping")
  }
  message("")

  message("--- Loading optional resources ---")
  TIS_EFFICIENCY_TABLE   <<- load_tis_table()
  UORFDB_DATA            <<- load_uorfdb()
  IRES_GENES             <<- load_ires_genes()
  NONCANONICAL_TIS_GENES <<- load_noncanon_tis_genes()
  mir_cache <- if (file.exists(MULTIMIR_CACHE)) readRDS(MULTIMIR_CACHE) else NULL

  # Load prior results and skip already-processed genes
  prior <- NULL
  if (file.exists(OUTPUT_FILE_MAIN)) {
    prior <- tryCatch(read.csv(OUTPUT_FILE_MAIN, stringsAsFactors = FALSE), error = function(e) NULL)
    if (!is.null(prior) && nrow(prior) > 0L) {
      already_done <- intersect(gene_list, prior$Gene)
      if (length(already_done) > 0L) {
        message("  Skipping ", length(already_done), " already-processed gene(s); ",
                length(gene_list) - length(already_done), " remaining")
        gene_list <- setdiff(gene_list, already_done)
      }
    }
  }
  if (length(gene_list) == 0L) {
    message("  All genes already processed — nothing to do.")
    return(invisible(prior))
  }

  data <- get_mrna_sequences(gene_list, use_cache)
  n    <- nrow(data)
  message("\n--- Calculating RNA features for ", n, " genes ---")

  # Compute the full feature row for one gene (5'UTR + CDS + 3'UTR + whole-mRNA).
  process_gene <- function(i) {
    tryCatch({
      gene <- data$hgnc_symbol[i]
      u5   <- data$`5utr`[i]
      u3   <- data$`3utr`[i]
      cds  <- data$coding[i]

      # Defensive: ensure length-1 character or NA
      clean <- function(x) {
        v <- x[[1L]]
        if (length(v) != 1L || is.null(v)) NA_character_ else as.character(v)
      }
      u5  <- clean(data[i, "5utr",  drop = FALSE])
      u3  <- clean(data[i, "3utr",  drop = FALSE])
      cds <- clean(data[i, "coding", drop = FALSE])

      # 5'UTR
      b5   <- calculate_5utr_basic(u5)
      tis  <- get_tis_efficiency(u5, cds)
      top  <- has_TOP_candidate(u5)
      uorf <- analyze_uorfs(u5, cds)
      g4   <- detect_g_quadruplex(u5)
      mfe5 <- calculate_mfe(u5, "utr5")
      m5   <- count_m6a(u5)
      ires <- check_ires(gene)
      nct  <- check_noncanon_tis(gene)

      # CDS
      cai  <- calculate_cai(cds)
      enc  <- calculate_enc(cds)
      gc3  <- calculate_gc3(cds)
      cds_len <- if (!is.na(cds)) nchar(cds) else NA
      cds_gc  <- if (!is.na(cds) && nchar(cds) > 0) round(str_count(cds,"[GCgc]")/nchar(cds),4) else NA
      cds_pur <- if (!is.na(cds) && nchar(cds) > 0) round(str_count(cds,"[AGag]")/nchar(cds),4) else NA
      mc   <- count_m6a(cds)

      # 3'UTR
      b3   <- calculate_3utr_basic(u3)
      are  <- count_are_motifs(u3)
      pas  <- detect_pas(u3)
      mfe3 <- calculate_mfe(u3, "utr3")
      m3   <- count_m6a(u3)
      mir  <- get_mirna_features(gene, u3, mir_cache)

      # Whole mRNA m6A
      full <- paste0(if (is.na(u5)) "" else u5,
                     if (is.na(cds)) "" else cds,
                     if (is.na(u3)) "" else u3)
      mf   <- count_m6a(if (nchar(full) > 0) full else NA)

      data.frame(
        # Metadata
        Gene              = gene,
        transcript_id     = data$ensembl_transcript_id[i],
        transcript_source = ifelse(
          !is.na(data$transcript_mane_select[i]) & data$transcript_mane_select[i] != "",
          "MANE_Select", paste0("APPRIS_", data$transcript_appris[i])),

        # 5'UTR
        utr5_length              = b5$utr5_length,
        utr5_gc_content          = b5$utr5_gc_content,
        utr5_purine_content      = b5$utr5_purine_content,
        utr5_mfe                 = mfe5$utr5_mfe,
        utr5_mfe_normalized      = mfe5$utr5_mfe_normalized,
        tis_efficiency           = tis,
        has_TOP_candidate_motif  = top,
        uORF_ATG_count           = uorf$uORF_ATG_count,
        uORF_nonATG_count        = uorf$uORF_nonATG_count,
        uORF_longest_length      = uorf$uORF_longest_length,
        uORF_distance_to_CDS     = uorf$uORF_distance_to_CDS,
        uORF_overlaps_CDS        = uorf$uORF_overlaps_CDS,
        has_validated_uORF       = uorf$has_validated_uORF,
        utr5_g4_max_score        = g4$utr5_g4_max_score,
        utr5_g4_count            = g4$utr5_g4_count,
        utr5_g4_mean_score       = g4$utr5_g4_mean_score,
        utr5_g4_max_tetrads      = g4$utr5_g4_max_tetrads,
        utr5_m6a_count           = m5$m6a_count,
        utr5_m6a_density         = m5$m6a_density,
        IRES_validated           = ires,
        noncanonical_TIS_validated = nct,

        # CDS
        cds_length               = cds_len,
        cds_gc_content           = cds_gc,
        cds_purine_content       = cds_pur,
        cai                      = cai,
        enc                      = enc,
        gc3                      = gc3,
        cds_m6a_count            = mc$m6a_count,
        cds_m6a_density          = mc$m6a_density,

        # 3'UTR
        utr3_length              = b3$utr3_length,
        utr3_gc_content          = b3$utr3_gc_content,
        utr3_purine_content      = b3$utr3_purine_content,
        are_class1_count         = are$are_class1,
        are_class2_count         = are$are_class2,
        are_class3_count         = are$are_class3,
        are_nonamer_count        = are$are_nonamers,
        are_pentamer_count       = are$are_pentamers,
        PAS_type                 = pas$PAS_type,
        PAS_distance             = as.integer(pas$PAS_distance),
        utr3_mfe                 = mfe3$utr3_mfe,
        utr3_mfe_normalized      = mfe3$utr3_mfe_normalized,
        miRNA_count              = mir$miRNA_count,
        miRNA_density            = mir$miRNA_density,
        utr3_m6a_count           = m3$m6a_count,
        utr3_m6a_density         = m3$m6a_density,

        # Whole mRNA
        mrna_m6a_count           = mf$m6a_count,
        mrna_m6a_density         = mf$m6a_density,

        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      message("  ✗ Failed: ", data$hgnc_symbol[i], " — ", e$message)
      NULL
    })
  }

  # Parallel or sequential — skip cluster overhead for small jobs
  use_parallel <- N_CORES > 1L && n > 4L
  if (use_parallel) {
    cl <- parallel::makeCluster(N_CORES)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, {
      library(Biostrings); library(pqsfinder); library(stringr); library(dplyr)
    })
    parallel::clusterExport(cl,
      c("data","mir_cache","rnafold_ok","ENABLE_RNAFOLD",
        "TIS_EFFICIENCY_TABLE","UORFDB_DATA","IRES_GENES","NONCANONICAL_TIS_GENES",
        ls(envir = globalenv())),
      envir = environment()
    )
    results <- pbapply::pblapply(seq_len(n), process_gene, cl = cl)
  } else {
    results <- pbapply::pblapply(seq_len(n), process_gene)
  }

  new_results <- do.call(rbind, Filter(Negate(is.null), results))

  # Merge with any existing output, replacing rows for re-processed genes
  if (file.exists(OUTPUT_FILE_MAIN) && !is.null(prior)) {
    combined <- rbind(prior[!prior$Gene %in% new_results$Gene, ], new_results)
    combined <- combined[order(combined$Gene), ]
  } else {
    combined <- new_results
  }

  write.csv(combined, OUTPUT_FILE_MAIN, row.names = FALSE)
  message("\n✓ Results saved: ", OUTPUT_FILE_MAIN)
  message("  Genes in file: ", nrow(combined),
          "  (", nrow(new_results), " newly processed)")
  return(invisible(combined))
}


# =============================================================================
# ENTRY POINT
# =============================================================================

# Entry point: resolve the gene list (or TEST subset) and run extract_rna_features.
run <- function(genes = NULL) {
  if (is.null(genes)) {
    genes <- load_target_genes()
    if (TEST_MODE) {
      message("TEST MODE: processing first ", TEST_N_GENES, " genes")
      genes <- head(genes, TEST_N_GENES)
    }
  } else {
    message("Running for ", length(genes), " specified gene(s): ",
            paste(genes, collapse = ", "))
  }
  extract_rna_features(genes, use_cache = TRUE)
}

if (!interactive()) {
  run()
} else {
  message("Script loaded. Run: run()")
}
