# =============================================================================
# FULL-MODEL PERFORMANCE PANEL (Figure 1)
# =============================================================================
# Assembles the main full-model performance figure from the cross-validation
# outputs of 05_model_training. Four sub-panels:
#   A – per-cell Pearson r (calibrated), Known vs Unknown proteins (violin/box)
#   B – per-protein r vs RMSE scatter for known proteins (top/bottom labelled)
#   C – per-cell Pearson r (uncalibrated), Known vs Unknown proteins
#   D – per-protein r by manually-annotated protein class (UniProt-derived)
# Outputs the combined multi-panel figure as PDF + PNG.
# =============================================================================

library(nanoparquet)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(cowplot)
library(forcats)

col_known   <- "#440154"
col_unknown <- "#D55E00"

# ── Paths ─────────────────────────────────────────────────────────────────────
pred_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Final_Run/Output/Final_Run_stratified_20.05.26/predictions"
stat_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Final_Run/Output/Final_Run_stratified_20.05.26/statistics"
out_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Full_model/Panel1"

# ── Load data ─────────────────────────────────────────────────────────────────
known        <- read_parquet(file.path(pred_dir, "infold_known_proteins_averaged.parquet"))
known_cell   <- read.csv(file.path(stat_dir, "infold_known_per_cell.csv"))
unknown_cell <- read.csv(file.path(stat_dir, "oof_unknown_per_cell.csv"))

ref_data     <- read_parquet("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Final_Run/Input/reference_data.parquet")
ann          <- read.csv("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/4.Annotation+validation/cell_type_assignments.csv")
barcode_ct   <- setNames(ann$celltype_final, ann$X)

# Attach each cell's final cell-type label (barcode lookup) to a per-cell stats df.
add_celltype <- function(df) {
  barcodes <- ref_data$barcode[df$cell_id + 1L]
  df$celltype_final <- barcode_ct[barcodes]
  df$celltype_final[df$celltype_final == "DC"] <- "Dendritic cells"
  df
}

known_cell   <- add_celltype(known_cell)
unknown_cell <- add_celltype(unknown_cell)

# ── Shared theme ──────────────────────────────────────────────────────────────
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

# ── A: per-cell r_cal — Known vs Unknown proteins (calibrated) ────────────────
per_cell_cal <- bind_rows(
  known_cell   |> select(r_cal) |> mutate(group = "Known\n (n=205)"),
  unknown_cell |> select(r_cal) |> mutate(group = "Unknown\n (n=205)")
)

p_A <- ggplot(per_cell_cal, aes(x = group, y = r_cal, fill = group)) +
  geom_violin(colour = "black", alpha = 0.6, linewidth = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, colour = "black",
               linewidth = 0.4, outlier.shape = NA) +
  scale_fill_manual(values = c("Known\n (n=205)" = col_known, "Unknown\n (n=205)" = col_unknown)) +
  labs(x = "Protein Status", y = "Pearson r per cell (calibrated)") +
  base_theme() +
  theme(legend.position = "none")

# ── B: per-protein r vs rmse_cal scatter — known proteins (calibrated) ────────
metrics_known <- known |>
  group_by(protein) |>
  summarise(
    r        = cor(y_true, y_pred_cal, use = "complete.obs"),
    rmse_cal = sqrt(mean((y_true - y_pred_cal)^2, na.rm = TRUE)),
    .groups  = "drop"
  )

top5_known_r      <- metrics_known |> slice_max(r,        n = 5) |> pull(protein)
bottom5_known_r   <- metrics_known |> slice_min(r,        n = 5) |> pull(protein)
top5_known_rmse   <- metrics_known |> slice_max(rmse_cal, n = 5) |> pull(protein)
bottom5_known_rmse <- metrics_known |> slice_min(rmse_cal, n = 5) |> pull(protein)
labeled_known <- unique(c(top5_known_r, bottom5_known_r, top5_known_rmse, bottom5_known_rmse))
metrics_known$label <- ifelse(
  metrics_known$protein %in% labeled_known,
  as.character(metrics_known$protein), NA_character_
)
rmse_cal_range <- diff(range(metrics_known$rmse_cal, na.rm = TRUE))
metrics_known$nudge_x <-
  ifelse(metrics_known$protein %in% top5_known_r,     0.05, 0) +
  ifelse(metrics_known$protein %in% bottom5_known_r, -0.05, 0)
metrics_known$nudge_y <-
  ifelse(metrics_known$protein %in% top5_known_rmse,    rmse_cal_range * 0.05, 0) +
  ifelse(metrics_known$protein %in% bottom5_known_rmse, -rmse_cal_range * 0.05, 0)

p_B <- ggplot(metrics_known, aes(x = r, y = rmse_cal, label = label)) +
  geom_point(colour = col_known, alpha = 0.7, size = 1.8) +
  geom_label_repel(
    na.rm              = TRUE,
    size               = 2.5,
    max.overlaps       = Inf,
    segment.size       = 0.3,
    segment.colour     = "grey50",
    box.padding        = 0.3,
    label.padding      = 0.15,
    nudge_x            = metrics_known$nudge_x,
    nudge_y            = metrics_known$nudge_y,
    min.segment.length = 0
  ) +
  scale_x_continuous(name = "Pearson r") +
  scale_y_continuous(name = "RMSE (calibrated)") +
  base_theme()

# ── C: per-cell r_raw — Known vs Unknown proteins (uncalibrated) ──────────────
per_cell_raw <- bind_rows(
  known_cell   |> select(r_raw) |> mutate(group = "Known\n (n=205)"),
  unknown_cell |> select(r_raw) |> mutate(group = "Unknown\n (n=205)")
)

p_C <- ggplot(per_cell_raw, aes(x = group, y = r_raw, fill = group)) +
  geom_violin(colour = "black", alpha = 0.6, linewidth = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, colour = "black",
               linewidth = 0.4, outlier.shape = NA) +
  scale_fill_manual(values = c("Known\n (n=205)" = col_known, "Unknown\n (n=205)" = col_unknown)) +
  labs(x = "Protein Status", y = "Pearson r per cell (uncalibrated)") +
  base_theme() +
  theme(legend.position = "none")

# ── D: per-protein r_cal by protein class ────────────────────────────────────
prot_ann <- read.csv("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Scripts/Ablation/protein_list_annotated.csv")

prot_class <- metrics_known |>
  left_join(prot_ann |> select(protein, Protein_Class_TopLevel), by = "protein") |>
  filter(!is.na(Protein_Class_TopLevel)) |>
  group_by(Protein_Class_TopLevel) |>
  mutate(
    n     = n(),
    label = paste0(Protein_Class_TopLevel, "\n(n=", n, ")")
  ) |>
  ungroup() |>
  mutate(label = fct_reorder(label, r, median, .desc = TRUE))

p_D <- ggplot(prot_class, aes(x = label, y = r, fill = label)) +
  geom_boxplot(colour = "black", linewidth = 0.4, outlier.shape = NA, alpha = 0.6) +
  scale_fill_viridis_d(option = "turbo", guide = "none") +
  labs(x = "Protein Class", y = "Pearson r per protein") +
  base_theme() +
  theme(
    axis.text.x  = element_text(angle = 50, hjust = 1, size = 7),
    legend.position = "none"
  )

# ── Assemble panel: ABC in row 1, D full-width in row 2 ──────────────────────
row1 <- plot_grid(
  p_C, p_A, p_B,
  nrow           = 1,
  labels         = c("A", "B", "C"),
  label_size     = 18L,
  label_fontface = "bold",
  label_x        = 0,
  label_y        = 1.00,
  hjust          = 0,
  vjust          = 1
)

row2 <- plot_grid(
  p_D,
  nrow           = 1,
  labels         = "D",
  label_size     = 18L,
  label_fontface = "bold",
  label_x        = 0,
  label_y        = 1.00,
  hjust          = 0,
  vjust          = 1
)

combined <- plot_grid(
  row1, row2,
  ncol        = 1,
  rel_heights = c(2.8, 3.9)
)

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(
  filename = file.path(out_dir, "panel1.pdf"),
  plot     = combined,
  width    = 9,
  height   = 7.1,
  device   = cairo_pdf
)
ggsave(
  filename = file.path(out_dir, "panel1.png"),
  plot     = combined,
  width    = 9,
  height   = 7.1,
  dpi      = 300
)

message("Saved to: ", out_dir)
message("Median per-protein Pearson r (Panel C): ", round(median(metrics_known$r, na.rm = TRUE), 3))
