library(dplyr)
library(ggplot2)
library(ggrepel)
library(cowplot)

col_known   <- "#440154"
col_unknown <- "#D55E00"

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir      <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred"
export_dir    <- file.path(base_dir, "Second_Dataset/Output2/singlecellobjects/r_export")
eval_dir      <- file.path(base_dir, "Second_Dataset/Output2/Evaluation")
mapping_csv   <- file.path(base_dir, "Second_Dataset/package/citepredict/data/antibody_info/combined_adt_mapping_MODEL-TRAINING.csv")
out_dir       <- file.path(base_dir, "Plotting/Output/Second_dataset")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ── Classify proteins by training status ──────────────────────────────────────
# Source = Hao       → antibody measured in training (Known; predicted with its protein_id)
# Source = Kotliarov → antibody absent from training (Unknown; protein_id = NaN)
mapping <- read.csv(mapping_csv, stringsAsFactors = FALSE)

protein_class <- mapping |>
  mutate(protein = ADT_feature,
         training_status = ifelse(Source == "Hao", "Known", "Unknown")) |>
  select(protein, RNA_gene, training_status)

# ── Load prediction and measurement matrices ───────────────────────────────────
# Known and unknown antibodies are predicted separately (obsm ADT_pred_known /
# ADT_pred_unknown); measured ADT is CLR and already renamed to the reference names
# by 1_run_citepredict_kotliarov.py, so columns match by name.
read_obsm <- function(f) read.csv(file.path(export_dir, f), row.names = 1, check.names = FALSE)
pred_known   <- read_obsm("obsm_ADT_pred_known.csv")
pred_unknown <- read_obsm("obsm_ADT_pred_unknown.csv")
pred_mat <- cbind(pred_known, pred_unknown[rownames(pred_known), , drop = FALSE])
meas_mat <- read_obsm("obsm_ADT_measured.csv")

common_cells <- intersect(rownames(pred_mat), rownames(meas_mat))
common_prots <- intersect(colnames(pred_mat), colnames(meas_mat))
pred_mat     <- as.matrix(pred_mat[common_cells, common_prots, drop = FALSE])
meas_mat     <- as.matrix(meas_mat[common_cells, common_prots, drop = FALSE])

# Per-protein OLS calibration across all cells
slope_vec     <- setNames(rep(1, length(common_prots)), common_prots)
intercept_vec <- setNames(rep(0, length(common_prots)), common_prots)
for (p in common_prots) {
  fit              <- lm(meas_mat[, p] ~ pred_mat[, p])
  slope_vec[p]     <- coef(fit)[2]
  intercept_vec[p] <- coef(fit)[1]
}
pred_cal_mat <- sweep(sweep(pred_mat, 2, slope_vec, "*"), 2, intercept_vec, "+")
pred_cal_mat[pred_cal_mat < 0] <- 0

# Protein subsets (intersect with what's in both matrices)
known_prots   <- intersect(protein_class$protein[protein_class$training_status == "Known"],   common_prots)
unknown_prots <- intersect(protein_class$protein[protein_class$training_status == "Unknown"], common_prots)
n_known       <- length(known_prots)
n_unknown     <- length(unknown_prots)

message(sprintf("Known proteins matched in matrices: %d | Unknown: %d | Total in matrices: %d",
                n_known, n_unknown, length(common_prots)))
if (n_known == 0L || n_unknown == 0L) {
  message("  protein_class names (first 10): ",
          paste(head(protein_class$protein, 10), collapse = ", "))
  message("  pred_mat colnames (first 10): ",
          paste(head(colnames(pred_mat), 10), collapse = ", "))
  stop("Protein name mismatch between mapping CSV and exported matrices. Check names above.")
}

grp_k <- paste0("Known\n(n=", n_known, ")")
grp_u <- paste0("Unknown\n(n=", n_unknown, ")")

# Per-cell r restricted to each protein subset
per_cell_r <- function(pred, meas) {
  vapply(seq_len(nrow(pred)),
         function(i) {
           v1 <- as.numeric(pred[i, ])
           v2 <- as.numeric(meas[i, ])
           ok <- is.finite(v1) & is.finite(v2)
           if (sum(ok) < 2L) return(NA_real_)
           cor(v1[ok], v2[ok])
         },
         numeric(1))
}

cell_df <- bind_rows(
  data.frame(
    r_raw = per_cell_r(pred_mat[,     known_prots, drop = FALSE],
                       meas_mat[,     known_prots, drop = FALSE]),
    r_cal = per_cell_r(pred_cal_mat[, known_prots, drop = FALSE],
                       meas_mat[,     known_prots, drop = FALSE]),
    group = grp_k
  ),
  data.frame(
    r_raw = per_cell_r(pred_mat[,     unknown_prots, drop = FALSE],
                       meas_mat[,     unknown_prots, drop = FALSE]),
    r_cal = per_cell_r(pred_cal_mat[, unknown_prots, drop = FALSE],
                       meas_mat[,     unknown_prots, drop = FALSE]),
    group = grp_u
  )
)

col_fill <- c(col_known, col_unknown)
names(col_fill) <- c(grp_k, grp_u)

# ── Load per-protein evaluation results ───────────────────────────────────────
# 3_Evaluate_predictions_kotliarev.r writes Known and Unknown to separate folders.
# Its protein names passed through Seurat ("_" -> "-"), so join on dashed names.
prot_df <- bind_rows(
  read.csv(file.path(eval_dir, "Known",   "per_protein_unfiltered.csv"), check.names = FALSE),
  read.csv(file.path(eval_dir, "Unknown", "per_protein_unfiltered.csv"), check.names = FALSE)
) |>
  left_join(protein_class |> transmute(protein = gsub("_", "-", protein), training_status),
            by = "protein") |>
  mutate(group = case_when(
    training_status == "Known"   ~ "Known",
    training_status == "Unknown" ~ "Unknown",
    TRUE                         ~ NA_character_
  ))

col_map <- c("Known" = col_known, "Unknown" = col_unknown)

# ── Theme ─────────────────────────────────────────────────────────────────────
base_theme <- function() {
  theme_minimal(base_size = 9) +
    theme(
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size = 9),
      panel.grid.minor = element_blank(),
      axis.line        = element_line(colour = "black", linewidth = 0.5),
      plot.margin      = margin(t = 16, r = 5, b = 5, l = 5)
    )
}

# ── A: per-cell r_cal violin — calibrated ─────────────────────────────────────
p_A <- ggplot(cell_df, aes(x = group, y = r_cal, fill = group)) +
  geom_violin(colour = "black", alpha = 0.6, linewidth = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, colour = "black",
               linewidth = 0.4, outlier.shape = NA) +
  scale_fill_manual(values = col_fill) +
  labs(x = "Protein Status", y = "Pearson r per cell (calibrated)") +
  base_theme() +
  theme(legend.position = "none")

# ── B: per-protein r_cal vs rmse_cal scatter — calibrated ─────────────────────
rmse_cal_range   <- diff(range(prot_df$rmse_cal, na.rm = TRUE))
top5_cal_r       <- prot_df |> slice_max(r_cal,    n = 5) |> pull(protein)
bottom5_cal_r    <- prot_df |> slice_min(r_cal,    n = 5) |> pull(protein)
top5_cal_rmse    <- prot_df |> slice_max(rmse_cal, n = 5) |> pull(protein)
bottom3_cal_rmse <- prot_df |> slice_min(rmse_cal, n = 3) |> pull(protein)
labeled_cal      <- unique(c(top5_cal_r, bottom5_cal_r, top5_cal_rmse, bottom3_cal_rmse))

prot_cal <- prot_df |>
  mutate(
    label   = ifelse(protein %in% labeled_cal, protein, NA_character_),
    nudge_x = ifelse(protein %in% top5_cal_r,      0.05, 0) +
              ifelse(protein %in% bottom5_cal_r,   -0.05, 0),
    nudge_y = ifelse(protein %in% top5_cal_rmse,    rmse_cal_range * 0.05, 0) +
              ifelse(protein %in% bottom3_cal_rmse, -rmse_cal_range * 0.05, 0)
  )

p_B <- ggplot(prot_cal, aes(x = r_cal, y = rmse_cal, colour = group, label = label)) +
  geom_point(alpha = 0.7, size = 1.8) +
  scale_colour_manual(values = col_map, na.translate = FALSE) +
  geom_label_repel(
    na.rm              = TRUE,
    size               = 2.5,
    max.overlaps       = Inf,
    segment.size       = 0.3,
    segment.colour     = "grey50",
    box.padding        = 0.4,
    label.padding      = 0.15,
    nudge_x            = prot_cal$nudge_x,
    nudge_y            = prot_cal$nudge_y,
    min.segment.length = 0,
    force              = 3,
    force_pull         = 0.5,
    max.iter           = 20000,
    show.legend        = FALSE
  ) +
  scale_x_continuous(name = "Pearson r (calibrated)") +
  scale_y_continuous(name = "RMSE (calibrated)") +
  base_theme() +
  theme(legend.title = element_blank(), legend.position = "right") +
  guides(colour = guide_legend(override.aes = list(shape = 16, size = 3, alpha = 1)))

# ── C: per-cell r_raw violin — uncalibrated ───────────────────────────────────
p_C <- ggplot(cell_df, aes(x = group, y = r_raw, fill = group)) +
  geom_violin(colour = "black", alpha = 0.6, linewidth = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, colour = "black",
               linewidth = 0.4, outlier.shape = NA) +
  scale_fill_manual(values = col_fill) +
  labs(x = "Protein Status", y = "Pearson r per cell (uncalibrated)") +
  base_theme() +
  theme(legend.position = "none")

# ── D: per-protein r_raw vs rmse_raw scatter — uncalibrated ───────────────────
rmse_raw_range   <- diff(range(prot_df$rmse_raw, na.rm = TRUE))
top5_raw_r       <- prot_df |> slice_max(r_raw,    n = 5) |> pull(protein)
bottom5_raw_r    <- prot_df |> slice_min(r_raw,    n = 5) |> pull(protein)
top5_raw_rmse    <- prot_df |> slice_max(rmse_raw, n = 5) |> pull(protein)
bottom3_raw_rmse <- prot_df |> slice_min(rmse_raw, n = 3) |> pull(protein)
labeled_raw      <- unique(c(top5_raw_r, bottom5_raw_r, top5_raw_rmse, bottom3_raw_rmse))

prot_raw <- prot_df |>
  mutate(
    label   = ifelse(protein %in% labeled_raw, protein, NA_character_),
    nudge_x = ifelse(protein %in% top5_raw_r,      0.05, 0) +
              ifelse(protein %in% bottom5_raw_r,   -0.05, 0),
    nudge_y = ifelse(protein %in% top5_raw_rmse,    rmse_raw_range * 0.05, 0) +
              ifelse(protein %in% bottom3_raw_rmse, -rmse_raw_range * 0.05, 0)
  )

p_D <- ggplot(prot_raw, aes(x = r_raw, y = rmse_raw, colour = group, label = label)) +
  geom_point(alpha = 0.7, size = 1.8) +
  scale_colour_manual(values = col_map, na.translate = FALSE) +
  geom_label_repel(
    na.rm              = TRUE,
    size               = 2.5,
    max.overlaps       = Inf,
    segment.size       = 0.3,
    segment.colour     = "grey50",
    box.padding        = 0.4,
    label.padding      = 0.15,
    nudge_x            = prot_raw$nudge_x,
    nudge_y            = prot_raw$nudge_y,
    min.segment.length = 0,
    force              = 3,
    force_pull         = 0.5,
    max.iter           = 20000,
    show.legend        = FALSE
  ) +
  scale_x_continuous(name = "Pearson r (uncalibrated)") +
  scale_y_continuous(name = "RMSE (uncalibrated)") +
  base_theme() +
  theme(legend.title = element_blank(), legend.position = "right") +
  guides(colour = guide_legend(override.aes = list(shape = 16, size = 3, alpha = 1)))

# ── Assemble 2×2 panel ────────────────────────────────────────────────────────
combined <- plot_grid(
  p_A, p_B, p_C, p_D,
  nrow           = 2,
  rel_widths     = c(1, 1.5),
  labels         = c("A", "B", "C", "D"),
  label_size     = 18L,
  label_fontface = "bold",
  label_x        = 0,
  label_y        = 1.00,
  hjust          = 0,
  vjust          = 1
)

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(
  filename = file.path(out_dir, "evaluation_panel.pdf"),
  plot     = combined,
  width    = 7.5,
  height   = 5.9,
  device   = cairo_pdf
)
ggsave(
  filename = file.path(out_dir, "evaluation_panel.png"),
  plot     = combined,
  width    = 7.5,
  height   = 5.9,
  dpi      = 300
)
ggsave(
  filename = file.path(out_dir, "evaluation_panel.tif"),
  plot     = combined,
  width    = 7.5,
  height   = 5.9,
  dpi      = 600,
  device   = function(filename, width, height, ...) tiff(filename, width = width, height = height, units = "in", res = 600, compression = "lzw")
)

message("Saved to: ", out_dir)

# ── Console summaries ─────────────────────────────────────────────────────────
summarise_group <- function(df, grp, label) {
  sub <- df[!is.na(df$training_status) & df$training_status == grp, ]
  if (nrow(sub) == 0) return(invisible(NULL))
  message(sprintf("\n── %s %s (n=%d) ──────────────────", grp, label, nrow(sub)))
  for (metric in c("r_cal", "r_raw")) {
    vals  <- sub[[metric]]
    tag   <- if (metric == "r_cal") "calibrated  " else "uncalibrated"
    med   <- round(median(vals, na.rm = TRUE), 3)
    best  <- sub$protein[which.max(vals)]
    worst <- sub$protein[which.min(vals)]
    message(sprintf("  r (%s):  median=%.3f  |  best: %s (%.3f)  |  worst: %s (%.3f)",
                    tag, med,
                    best,  round(max(vals, na.rm = TRUE), 3),
                    worst, round(min(vals, na.rm = TRUE), 3)))
  }
}

message("\n===== Per-protein summary =====")
for (grp in c("Known", "Unknown")) summarise_group(prot_df, grp, "proteins")

message("\n===== Per-cell summary =====")
for (grp_label in c(grp_k, grp_u)) {
  sub   <- cell_df[cell_df$group == grp_label, ]
  label <- sub("\\n.*", "", grp_label)   # strip the "\n(n=... proteins)" part
  status <- if (grepl("Known", label)) "Known" else "Unknown"
  if (nrow(sub) == 0) next
  message(sprintf("\n── %s proteins (n=%d cells) ──────────────────", status, nrow(sub)))
  for (metric in c("r_cal", "r_raw")) {
    vals <- sub[[metric]]
    tag  <- if (metric == "r_cal") "calibrated  " else "uncalibrated"
    message(sprintf("  r (%s):  median=%.3f  |  range: [%.3f, %.3f]",
                    tag,
                    round(median(vals, na.rm = TRUE), 3),
                    round(min(vals,    na.rm = TRUE), 3),
                    round(max(vals,    na.rm = TRUE), 3)))
  }
}
