# =============================================================================
# EXTERNAL VALIDATION — PREDICTION EVALUATION (Kotliarov et al. 2020)
# =============================================================================
# NOTE: This script produces statistical evaluations not used in the final manuscript; Used for exploratory purposes.
# Check "\10_plotting\second_dataset\evaluation_panel.R" for stats used.

# Evaluates the external-validation predictions stored in the Seurat object
# built by assemble_seurat.R. For each protein it fits a per-protein linear
# calibration (measured ~ predicted), then reports per-protein and per-cell
# Pearson r / R² / RMSE (raw and calibrated), plus per-cell-type means, and
# writes CSVs + diagnostic plots.
#
# In the current pipeline every protein is predicted with protein_id = NaN
# (annotate_unknown), so only the Unknown/ results are produced; the Known/ and
# Known_from_Unknown/ branches are retained but become no-ops when those assays
# are absent. The figure (evaluation_panel.R) reads Unknown/per_protein_*.csv
# and assigns the Known/Unknown display label by training status.
# =============================================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(dplyr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
RDS_PATH    <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Second_Dataset/Output2/singlecellobjects/seurat_External_Validation_Package.rds"
MAPPING_CSV <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Second_Dataset/package/External_Validation_Package/data/adt_rna_mapping_second_dataset.csv"
OUT_DIR     <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Second_Dataset/Output2/Evaluation"

FILTER_THRESHOLD <- 0.5   # cells below this CLR-normalised ADT value are treated as dropout and excluded from evaluation
CELLTYPE_COL     <- NULL  # set to e.g. "celltype" to override auto-detection

for (d in file.path(OUT_DIR, c("Known", "Unknown", "Known_from_Unknown", "Summary")))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading RDS...\n")
seu <- readRDS(RDS_PATH)
cat(sprintf("  %d cells\n", ncol(seu)))
cat(sprintf("  Assays: %s\n", paste(Assays(seu), collapse = ", ")))

mapping <- read.csv(MAPPING_CSV, stringsAsFactors = FALSE)
mapping$Known_protein_id <- trimws(mapping$Known_protein_id)
mapping$protein          <- trimws(mapping$protein)
mapping$RNA_gene         <- trimws(mapping$RNA_gene)

# ── Extract matrices (cells x proteins) ───────────────────────────────────────
get_mat <- function(assay_name) {
  if (!assay_name %in% Assays(seu)) return(NULL)
  as.matrix(t(GetAssayData(seu, assay = assay_name, layer = "counts")))
}
adt_mat              <- get_mat("ADT")
pred_known           <- get_mat("ADT_pred_known")
pred_unknown         <- get_mat("ADT_pred_unknown")
pred_known_from_unk  <- get_mat("ADT_pred_known_from_unknown")

# Seurat v5 silently converts underscores to dashes in all feature names.
# Normalise all three matrices and the mapping table to dashes so names match.
fix_names <- function(x) gsub("_", "-", x)

if (!is.null(adt_mat))      colnames(adt_mat)      <- fix_names(colnames(adt_mat))
if (!is.null(pred_known))          colnames(pred_known)          <- fix_names(colnames(pred_known))
if (!is.null(pred_unknown))        colnames(pred_unknown)        <- fix_names(colnames(pred_unknown))
if (!is.null(pred_known_from_unk)) colnames(pred_known_from_unk) <- fix_names(colnames(pred_known_from_unk))

mapping$protein          <- fix_names(mapping$protein)
mapping$Known_protein_id <- fix_names(mapping$Known_protein_id)

# Strip _PROT / -PROT suffixes Seurat sometimes adds to measured ADT names
if (!is.null(adt_mat))
  colnames(adt_mat) <- sub("[-]PROT$", "", colnames(adt_mat))

cat(sprintf("  ADT proteins              : %d\n", ncol(adt_mat)))
cat(sprintf("  ADT_pred_known            : %d\n", if (!is.null(pred_known))          ncol(pred_known)          else 0))
cat(sprintf("  ADT_pred_unknown          : %d\n", if (!is.null(pred_unknown))        ncol(pred_unknown)        else 0))
cat(sprintf("  ADT_pred_known_from_unk   : %d\n", if (!is.null(pred_known_from_unk)) ncol(pred_known_from_unk) else 0))

# ── Detect cell-type column ───────────────────────────────────────────────────
meta <- seu@meta.data
if (!is.null(CELLTYPE_COL) && CELLTYPE_COL %in% colnames(meta)) {
  celltype_vec <- as.character(meta[[CELLTYPE_COL]])
} else {
  candidates <- c("celltype_final")
  hit <- candidates[candidates %in% colnames(meta)]
  if (length(hit) > 0) {
    CELLTYPE_COL <- hit[1]
    celltype_vec <- as.character(meta[[CELLTYPE_COL]])
    cat(sprintf("  Cell-type column   : %s (%d types)\n",
                CELLTYPE_COL, length(unique(celltype_vec))))
  } else {
    cat("  WARNING: no cell-type column found — per-celltype stats skipped.\n")
    celltype_vec <- NULL
  }
}

# ── Build protein pair tables ─────────────────────────────────────────────────
# Known: ADT[protein] <-> ADT_pred_known[isoform-resolved Known_protein_id]
# Isoform resolution in annotate_known renames e.g. "CD3-2" -> "CD3"
# Strip the trailing -1 / -2 isoform suffix to match the output names written by annotate_known
resolve_isoform <- function(x) sub("-[12]$", "", x)

known_map <- mapping[mapping$Known_protein_id != "" &
                     !is.na(mapping$Known_protein_id), ]
known_map$pred_col <- resolve_isoform(known_map$Known_protein_id)

# All proteins are predicted via annotate_unknown (protein_id = NaN).
# Evaluate all proteins with a valid gene from ADT_pred_unknown regardless of
# whether they were seen during training; Known/Unknown split is display-only.
unknown_map <- mapping[mapping$RNA_gene != "" &
                       !is.na(mapping$RNA_gene) &
                       mapping$RNA_gene != "NA", ]

# Filter to pairs that actually exist in both matrices
filter_pairs <- function(map_df, adt, pred, adt_col, pred_col) {
  if (is.null(pred)) return(data.frame())
  map_df[map_df[[adt_col]]  %in% colnames(adt) &
         map_df[[pred_col]] %in% colnames(pred), , drop = FALSE]
}
known_pairs        <- filter_pairs(known_map,   adt_mat, pred_known,         "protein", "pred_col")
unknown_pairs      <- filter_pairs(unknown_map, adt_mat, pred_unknown,        "protein", "protein")
known_from_unk_pairs <- filter_pairs(known_map, adt_mat, pred_known_from_unk, "protein", "pred_col")

cat(sprintf("  Matched known             : %d proteins\n", nrow(known_pairs)))
cat(sprintf("  Matched unknown           : %d proteins\n", nrow(unknown_pairs)))
cat(sprintf("  Matched known_from_unk    : %d proteins\n", nrow(known_from_unk_pairs)))

# ── Core evaluation function ──────────────────────────────────────────────────
# For each protein pair:
#   1. Fit a per-protein linear model (measured ~ predicted) to calibrate scale and offset.
#   2. Compute per-protein Pearson r, R², and RMSE before and after calibration.
#   3. Compute per-cell Pearson r across all proteins (protein panel correlation per cell).
#   4. Aggregate per-cell metrics by cell type (mean r per cell type).
# Note: Pearson r is invariant to linear (affine) transformation; calibration affects RMSE only.
evaluate_pairs <- function(pairs, adt_col, pred_col,
                           adt, pred, celltypes, threshold = NULL, label = "") {
  if (nrow(pairs) == 0) return(NULL)
  tag <- if (is.null(threshold)) "unfiltered" else sprintf("CLR_gt_%.1f", threshold)

  n_cells <- nrow(adt)
  prots   <- pairs[[adt_col]]
  preds   <- pairs[[pred_col]]

  m_mat <- adt[,  prots, drop = FALSE]  # cells x proteins (measured)
  p_mat <- pred[, preds, drop = FALSE]  # cells x proteins (predicted)
  colnames(p_mat) <- prots              # align column names to the measured ADT names

  # ── Per-protein calibration (lm: measured ~ predicted) ────────────────────
  pp <- lapply(seq_along(prots), function(i) {
    m <- m_mat[, i]; p <- p_mat[, i]
    # If a dropout threshold is set, exclude low-signal cells from calibration fitting
    keep <- if (!is.null(threshold)) !is.na(m) & m > threshold else !is.na(m) & !is.na(p)
    n <- sum(keep)
    if (n < 10) return(NULL)  # skip proteins with too few measurable cells
    mk <- m[keep]; pk <- p[keep]
    r_raw  <- tryCatch(cor(pk, mk), error = function(e) NA_real_)
    r2_raw <- if (!is.na(r_raw)) r_raw^2 else NA_real_
    rmse_raw <- sqrt(mean((pk - mk)^2))
    if (is.na(r_raw) || sd(pk, na.rm = TRUE) == 0) {
      # Skip calibration if predictions have zero variance (model predicted a constant)
      slope <- NA_real_; intercept <- NA_real_; rmse_cal <- NA_real_
    } else {
      fit <- lm(mk ~ pk)
      slope     <- coef(fit)[["pk"]]
      intercept <- coef(fit)[["(Intercept)"]]
      pk_cal    <- slope * pk + intercept
      rmse_cal  <- sqrt(mean((pk_cal - mk)^2))
    }
    # Note: Pearson r is invariant to linear transformation so r_cal == r_raw.
    # RMSE is the correct metric to show the effect of calibration.
    r_cal  <- if (!is.na(slope)) sign(slope) * r_raw else r_raw
    r2_cal <- if (!is.na(r_cal)) r_cal^2 else NA_real_
    data.frame(protein = prots[i], pred_name = preds[i],
               r_raw = r_raw, r2_raw = r2_raw,
               r_cal = r_cal, r2_cal = r2_cal,
               slope = slope, intercept = intercept,
               rmse_raw = rmse_raw, rmse_cal = rmse_cal,
               n_cells = n, stringsAsFactors = FALSE)
  })
  pp <- do.call(rbind, Filter(Negate(is.null), pp))
  if (is.null(pp) || nrow(pp) == 0) return(NULL)

  # ── Calibrated prediction matrix ──────────────────────────────────────────
  # Apply the per-protein slope and intercept to rescale raw predictions to the measured ADT axis
  p_cal <- p_mat[, pp$protein, drop = FALSE]
  for (i in seq_len(nrow(pp))) {
    if (!is.na(pp$slope[i]))
      p_cal[, i] <- pp$slope[i] * p_cal[, i] + pp$intercept[i]
  }

  # ── Per-cell r and R² ─────────────────────────────────────────────────────
  # For each cell, correlate its predicted protein vector against the measured protein vector
  per_cell <- function(pred_m, name_suffix) {
    sapply(seq_len(n_cells), function(i) {
      m_row <- m_mat[i, pp$protein, drop = TRUE]
      p_row <- pred_m[i, pp$protein, drop = TRUE]
      keep  <- if (!is.null(threshold)) !is.na(m_row) & m_row > threshold
               else !is.na(m_row) & !is.na(p_row)
      if (sum(keep) < 3) return(c(r = NA_real_, r2 = NA_real_))
      r <- tryCatch(cor(p_row[keep], m_row[keep]), error = function(e) NA_real_)
      c(r = r, r2 = if (!is.na(r)) r^2 else NA_real_)
    })
  }
  raw_stats <- per_cell(p_mat)
  cal_stats <- per_cell(p_cal)

  cell_df <- data.frame(
    barcode    = rownames(adt),
    r_raw      = raw_stats["r", ],
    r2_raw     = raw_stats["r2", ],
    r_cal      = cal_stats["r", ],
    r2_cal     = cal_stats["r2", ],
    stringsAsFactors = FALSE
  )
  if (!is.null(celltypes)) cell_df$celltype <- celltypes

  # ── Per-celltype r and R² ─────────────────────────────────────────────────
  ct_df <- NULL
  if (!is.null(celltypes)) {
    ct_df <- cell_df %>%
      group_by(celltype) %>%
      summarise(
        n_cells                       = n(),
        mean_r_per_cell_raw           = mean(r_raw,  na.rm = TRUE),
        mean_r2_per_cell_raw          = mean(r2_raw, na.rm = TRUE),
        mean_r_per_cell_cal           = mean(r_cal,  na.rm = TRUE),
        mean_r2_per_cell_cal          = mean(r2_cal, na.rm = TRUE),
        .groups = "drop"
      ) %>% as.data.frame()
  }

  # Align raw prediction matrix columns to matched proteins only
  p_raw_out <- p_mat[, pp$protein, drop = FALSE]

  list(per_protein = pp, per_cell = cell_df, per_celltype = ct_df,
       pred_raw = p_raw_out, pred_cal = p_cal,
       tag = tag, label = label, n_prots = nrow(pp))
}

# ── Save + plot function ───────────────────────────────────────────────────────
save_results <- function(res, sub_dir) {
  if (is.null(res)) return(invisible(NULL))
  tag <- res$tag  # "unfiltered" or "CLR_gt_0.5" — used as a suffix in all output filenames

  # CSVs
  write.csv(res$per_protein,
            file.path(sub_dir, sprintf("per_protein_%s.csv",  tag)), row.names = FALSE)
  write.csv(res$per_cell,
            file.path(sub_dir, sprintf("per_cell_%s.csv",     tag)), row.names = FALSE)
  if (!is.null(res$per_celltype))
    write.csv(res$per_celltype,
              file.path(sub_dir, sprintf("per_celltype_%s.csv", tag)), row.names = FALSE)

  # Raw prediction matrix (cells x proteins)
  pred_raw_out <- as.data.frame(res$pred_raw)
  pred_raw_out <- cbind(barcode = rownames(res$pred_raw), pred_raw_out)
  write.csv(pred_raw_out,
            file.path(sub_dir, sprintf("predictions_raw_%s.csv", tag)),
            row.names = FALSE)

  # Calibrated prediction matrix (cells x proteins)
  pred_cal_out <- as.data.frame(res$pred_cal)
  pred_cal_out <- cbind(barcode = rownames(res$pred_cal), pred_cal_out)
  write.csv(pred_cal_out,
            file.path(sub_dir, sprintf("predictions_calibrated_%s.csv", tag)),
            row.names = FALSE)

  pp <- res$per_protein

  # Plot 1: horizontal bar chart of per-protein Pearson r, raw vs calibrated side by side
  pp_long <- rbind(
    data.frame(protein = pp$protein, r = pp$r_raw, type = "raw"),
    data.frame(protein = pp$protein, r = pp$r_cal, type = "calibrated")
  )
  pp_long$protein <- factor(pp_long$protein,
                             levels = pp$protein[order(pp$r_cal, na.last = TRUE)])
  p1_sub <- sprintf("raw: mean=%.3f  |  calibrated: mean=%.3f",
                    mean(pp$r_raw, na.rm = TRUE), mean(pp$r_cal, na.rm = TRUE))
  p1 <- ggplot(pp_long, aes(x = protein, y = r, fill = type)) +
    geom_col(position = "dodge") +
    scale_fill_manual(values = c(raw = "grey60", calibrated = "steelblue")) +
    coord_flip() +
    labs(title    = sprintf("Per-protein Pearson r — %s | %s", res$label, tag),
         subtitle = p1_sub, x = NULL, y = "Pearson r", fill = NULL) +
    theme_bw(base_size = 9) +
    theme(legend.position = "bottom")
  ggsave(file.path(sub_dir, sprintf("plot_per_protein_%s.png", tag)),
         p1, width = 6, height = max(3, nrow(pp) * 0.28))

  # Plot 2: density plot of per-cell Pearson r distribution across all cells
  cell_long <- rbind(
    data.frame(r = res$per_cell$r_raw, type = "raw"),
    data.frame(r = res$per_cell$r_cal, type = "calibrated")
  )
  p2_sub <- sprintf("raw: mean=%.3f  |  calibrated: mean=%.3f",
                    mean(res$per_cell$r_raw, na.rm = TRUE),
                    mean(res$per_cell$r_cal, na.rm = TRUE))
  p2 <- ggplot(cell_long, aes(x = r, fill = type, colour = type)) +
    geom_density(alpha = 0.35, na.rm = TRUE) +
    scale_fill_manual(values = c(raw = "grey60", calibrated = "steelblue")) +
    scale_colour_manual(values = c(raw = "grey40", calibrated = "steelblue4")) +
    labs(title    = sprintf("Per-cell Pearson r — %s | %s", res$label, tag),
         subtitle = p2_sub, x = "Pearson r", y = "Density", fill = NULL, colour = NULL) +
    theme_bw(base_size = 10)
  ggsave(file.path(sub_dir, sprintf("plot_per_cell_%s.png", tag)),
         p2, width = 7, height = 4)

  # Plot 3: per-cell-type mean Pearson r bar chart (only if cell-type labels are available)
  if (!is.null(res$per_celltype) && nrow(res$per_celltype) > 0) {
    ct <- res$per_celltype
    ct$celltype <- factor(ct$celltype,
                          levels = ct$celltype[order(ct$mean_r_per_cell_cal, na.last = TRUE)])
    ct_long <- rbind(
      data.frame(celltype = ct$celltype, r = ct$mean_r_per_cell_raw, type = "raw"),
      data.frame(celltype = ct$celltype, r = ct$mean_r_per_cell_cal, type = "calibrated")
    )
    p3_sub <- sprintf("raw: mean=%.3f  |  calibrated: mean=%.3f",
                      mean(ct$mean_r_per_cell_raw, na.rm = TRUE),
                      mean(ct$mean_r_per_cell_cal, na.rm = TRUE))
    p3 <- ggplot(ct_long, aes(x = celltype, y = r, fill = type)) +
      geom_col(position = "dodge") +
      scale_fill_manual(values = c(raw = "grey60", calibrated = "steelblue")) +
      coord_flip() +
      labs(title    = sprintf("Per-celltype mean Pearson r — %s | %s", res$label, tag),
           subtitle = p3_sub, x = NULL, y = "Mean Pearson r", fill = NULL) +
      theme_bw(base_size = 9) +
      theme(legend.position = "bottom")
    ggsave(file.path(sub_dir, sprintf("plot_per_celltype_%s.png", tag)),
           p3, width = 7, height = max(3, nrow(ct) * 0.3))
  }

  invisible(res)
}

# ── Run evaluations ───────────────────────────────────────────────────────────
# Up to three protein categories may be evaluated (a category is skipped if its
# prediction assay is absent). In the current NaN-only pipeline only "Unknown" runs:
#   Unknown            – every protein, predicted with protein_id = NaN (the active branch).
#   Known / Known_from_Unknown – legacy branches, evaluated only if those assays exist.
# Each category is evaluated both unfiltered (all cells) and filtered (dropout cells removed).
cat("\n========== KNOWN PROTEINS ==========\n")
res_known_unf <- evaluate_pairs(known_pairs, "protein", "pred_col",
                                adt_mat, pred_known, celltype_vec,
                                threshold = NULL,             label = "Known")
res_known_fil <- evaluate_pairs(known_pairs, "protein", "pred_col",
                                adt_mat, pred_known, celltype_vec,
                                threshold = FILTER_THRESHOLD, label = "Known")

if (!is.null(res_known_unf))
  cat(sprintf("  Unfiltered: %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_known_unf$n_prots,
              mean(res_known_unf$per_cell$r_cal, na.rm = TRUE)))
if (!is.null(res_known_fil))
  cat(sprintf("  Filtered  : %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_known_fil$n_prots,
              mean(res_known_fil$per_cell$r_cal, na.rm = TRUE)))

save_results(res_known_unf, file.path(OUT_DIR, "Known"))
save_results(res_known_fil, file.path(OUT_DIR, "Known"))

cat("\n========== KNOWN-FROM-UNKNOWN (HELD-OUT FOLD MODELS) ==========\n")
res_kfu_unf <- evaluate_pairs(known_from_unk_pairs, "protein", "pred_col",
                              adt_mat, pred_known_from_unk, celltype_vec,
                              threshold = NULL,             label = "Known_from_Unknown")
res_kfu_fil <- evaluate_pairs(known_from_unk_pairs, "protein", "pred_col",
                              adt_mat, pred_known_from_unk, celltype_vec,
                              threshold = FILTER_THRESHOLD, label = "Known_from_Unknown")

if (!is.null(res_kfu_unf))
  cat(sprintf("  Unfiltered: %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_kfu_unf$n_prots,
              mean(res_kfu_unf$per_cell$r_cal, na.rm = TRUE)))
if (!is.null(res_kfu_fil))
  cat(sprintf("  Filtered  : %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_kfu_fil$n_prots,
              mean(res_kfu_fil$per_cell$r_cal, na.rm = TRUE)))

save_results(res_kfu_unf, file.path(OUT_DIR, "Known_from_Unknown"))
save_results(res_kfu_fil, file.path(OUT_DIR, "Known_from_Unknown"))

cat("\n========== UNKNOWN PROTEINS ==========\n")
res_unk_unf <- evaluate_pairs(unknown_pairs, "protein", "protein",
                              adt_mat, pred_unknown, celltype_vec,
                              threshold = NULL,             label = "Unknown")
res_unk_fil <- evaluate_pairs(unknown_pairs, "protein", "protein",
                              adt_mat, pred_unknown, celltype_vec,
                              threshold = FILTER_THRESHOLD, label = "Unknown")

if (!is.null(res_unk_unf))
  cat(sprintf("  Unfiltered: %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_unk_unf$n_prots,
              mean(res_unk_unf$per_cell$r_cal, na.rm = TRUE)))
if (!is.null(res_unk_fil))
  cat(sprintf("  Filtered  : %d proteins  |  per-cell mean r_cal=%.3f\n",
              res_unk_fil$n_prots,
              mean(res_unk_fil$per_cell$r_cal, na.rm = TRUE)))

save_results(res_unk_unf, file.path(OUT_DIR, "Unknown"))
save_results(res_unk_fil, file.path(OUT_DIR, "Unknown"))

# ── Summary table ─────────────────────────────────────────────────────────────
summarise_res <- function(res, label, filter) {
  if (is.null(res)) return(NULL)
  pp <- res$per_protein; cc <- res$per_cell
  data.frame(
    label              = label,
    filter             = filter,
    n_proteins         = nrow(pp),
    mean_r_raw_protein = round(mean(pp$r_raw,  na.rm = TRUE), 3),
    mean_r_cal_protein = round(mean(pp$r_cal,  na.rm = TRUE), 3),
    mean_r2_raw_protein= round(mean(pp$r2_raw, na.rm = TRUE), 3),
    mean_r2_cal_protein= round(mean(pp$r2_cal, na.rm = TRUE), 3),
    mean_r_raw_cell    = round(mean(cc$r_raw,  na.rm = TRUE), 3),
    mean_r_cal_cell    = round(mean(cc$r_cal,  na.rm = TRUE), 3),
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, list(
  summarise_res(res_known_unf, "Known",              "unfiltered"),
  summarise_res(res_known_fil, "Known",              sprintf("measured>%.1f", FILTER_THRESHOLD)),
  summarise_res(res_kfu_unf,   "Known_from_Unknown", "unfiltered"),
  summarise_res(res_kfu_fil,   "Known_from_Unknown", sprintf("measured>%.1f", FILTER_THRESHOLD)),
  summarise_res(res_unk_unf,   "Unknown",            "unfiltered"),
  summarise_res(res_unk_fil,   "Unknown",            sprintf("measured>%.1f", FILTER_THRESHOLD))
))

cat("\n========== SUMMARY ==========\n")
print(summary_df, row.names = FALSE)
write.csv(summary_df, file.path(OUT_DIR, "Summary", "summary.csv"), row.names = FALSE)

# Summary plot: grouped bar chart of mean calibrated r by label x filter
if (!is.null(summary_df) && nrow(summary_df) > 0) {
  sm <- summary_df
  sm$condition <- paste(sm$label, sm$filter, sep = "\n")
  sm_long <- rbind(
    data.frame(condition = sm$condition, r = sm$mean_r_raw_protein, level = "per-protein raw"),
    data.frame(condition = sm$condition, r = sm$mean_r_cal_protein, level = "per-protein cal"),
    data.frame(condition = sm$condition, r = sm$mean_r_raw_cell,    level = "per-cell raw"),
    data.frame(condition = sm$condition, r = sm$mean_r_cal_cell,    level = "per-cell cal")
  )
  ps <- ggplot(sm_long, aes(x = condition, y = r, fill = level)) +
    geom_col(position = "dodge") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Summary: mean Pearson r by condition",
         x = NULL, y = "Mean Pearson r", fill = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom")
  ggsave(file.path(OUT_DIR, "Summary", "plot_summary.png"), ps, width = 8, height = 5)
}

cat(sprintf("\nAll outputs saved to: %s\n", OUT_DIR))
