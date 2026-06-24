# =============================================================================
# ABLATION — PER-PROTEIN r HEATMAP (Figure 2)
# =============================================================================
# Heatmap of calibrated per-protein Pearson r (in-fold known) across the six
# ablation conditions (rows = conditions, columns = proteins). Proteins are
# isoform-deduplicated (keep the variant with the highest max r) and restricted
# to those reaching r_cal > 0.85 in at least one condition. Outputs PDF + PNG.
# Reads per-protein in-fold-known stats from the ablation study output (07).
# =============================================================================

library(dplyr)
library(tidyr)
library(ComplexHeatmap)
library(circlize)
library(grid)

# ── User parameter ────────────────────────────────────────────────────────────
R_THRESHOLD <- 0.85  # keep proteins with r_cal > this in at least one condition

# ── Paths ─────────────────────────────────────────────────────────────────────
abl_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output"
out_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Ablation_study/Per_protein_heatmap"

conditions <- c("baseline", "add_dc", "add_endocytosis", "add_pca", "add_rnafm", "all_features")

condition_labels <- c(
  baseline        = "Baseline",
  add_dc          = "Baseline + RPG-DCs",
  add_endocytosis = "Baseline + Endocytosis",
  add_pca         = "Baseline + HVG-PCA",
  add_rnafm       = "Baseline + RNA-FM",
  all_features    = "Full Model"
)

# ── Load known per-protein r_cal ──────────────────────────────────────────────
load_condition <- function(cond, file) {
  read.csv(file.path(abl_dir, cond, "statistics", file)) |>
    mutate(condition = condition_labels[[cond]])
}

known_data <- bind_rows(lapply(conditions, load_condition, file = "infold_known_per_protein.csv"))

# ── Wide matrix: rows = proteins, columns = conditions ────────────────────────
wide <- known_data |>
  select(protein, condition, r_cal) |>
  mutate(protein = as.character(protein)) |>
  pivot_wider(names_from = condition, values_from = r_cal)

mat           <- as.matrix(wide[, -1])
rownames(mat) <- wide$protein

# ── Deduplicate: strip numeric suffix, keep row with highest max r ────────────
base_names    <- sub("-\\d+$", "", rownames(mat))
max_r_per_row <- apply(mat, 1, max, na.rm = TRUE)
keep_idx      <- tapply(seq_len(nrow(mat)), base_names,
                        function(idx) idx[which.max(max_r_per_row[idx])])
mat           <- mat[unlist(keep_idx), , drop = FALSE]

# ── Filter: keep proteins with r_cal > threshold in ≥1 condition ─────────────
keep <- apply(mat, 1, function(x) any(x > R_THRESHOLD, na.rm = TRUE))
mat  <- mat[keep, , drop = FALSE]

message("Known proteins passing r > ", R_THRESHOLD, ": ", nrow(mat))

# ── Transpose: rows = conditions, columns = proteins ─────────────────────────
mat <- t(mat)

# ── Color scale ───────────────────────────────────────────────────────────────
col_fun <- colorRamp2(c(0, 0.5, 1), c("#2166AC", "#F7F7F7", "#B2182B"))

# ── Heatmap ───────────────────────────────────────────────────────────────────
ht <- Heatmap(
  mat,
  name                 = "Pearson r",
  col                  = col_fun,
  cluster_rows         = TRUE,
  cluster_columns      = TRUE,
  row_dend_side        = "right",
  row_names_side       = "left",
  show_row_names       = TRUE,
  show_column_names    = TRUE,
  row_names_gp         = gpar(fontsize = 8),
  column_names_gp      = gpar(fontsize = 7),
  column_names_rot     = 45,
  border               = "black",
  heatmap_legend_param = list(
    title            = "Pearson r",
    title_gp         = gpar(fontsize = 9, fontface = "bold"),
    labels_gp        = gpar(fontsize = 8),
    legend_direction = "vertical"
  )
)

# ── Dynamic dimensions ────────────────────────────────────────────────────────
n_proteins <- ncol(mat)
plot_h     <- 4
plot_w     <- 9

# ── Save ──────────────────────────────────────────────────────────────────────
pdf(file.path(out_dir, "ablation_heatmap_known_horizontal.pdf"),
    width = plot_w, height = plot_h)
draw(ht, padding = unit(c(4, 4, 4, 4), "mm"))
dev.off()

png(file.path(out_dir, "ablation_heatmap_known_horizontal.png"),
    width = plot_w, height = plot_h, units = "in", res = 300)
draw(ht, padding = unit(c(4, 4, 4, 4), "mm"))
dev.off()

message("Saved to: ", out_dir)
