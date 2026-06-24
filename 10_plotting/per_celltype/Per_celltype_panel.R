# =============================================================================
# CELL-TYPE-RESTRICTED PERFORMANCE PANEL (Figure 6)
# =============================================================================
# Summarises the cell-type-restricted models (06) for the three major cell
# types (CD4 T cells, CD8 T cells, Monocytes). One row per cell type, each with:
#   left  – per-cell Pearson r (calibrated), Known vs Unknown proteins
#   right – per-protein r vs RMSE scatter (known proteins, top/bottom labelled)
# Per-cell stats are restricted to cells of the matching annotated cell type.
# Outputs the combined 3×2 figure as PDF + PNG.
# =============================================================================

library(nanoparquet)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(cowplot)

col_known   <- "#440154"
col_unknown <- "#D55E00"

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Per-Celltype-Test/Output"
ref_path <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Final_Run/Input/reference_data.parquet"
ann_path <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/4.Annotation+validation/cell_type_assignments.csv"
out_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Per_celltype"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Annotation lookup ─────────────────────────────────────────────────────────
ref_data   <- read_parquet(ref_path)
ann        <- read.csv(ann_path)
barcode_ct <- setNames(ann$celltype_final, ann$X)

# ── Cell types and their directory names ──────────────────────────────────────
celltypes <- list(
  list(dir = "CD4_T_cells", label = "CD4 T cells"),
  list(dir = "CD8_T_cells", label = "CD8 T cells"),
  list(dir = "Monocytes",   label = "Monocytes")
)

# ── Shared theme ──────────────────────────────────────────────────────────────
base_theme <- function() {
  theme_minimal(base_size = 9) +
    theme(
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size = 9),
      panel.grid.minor = element_blank(),
      axis.line        = element_line(colour = "black", linewidth = 0.5),
      plot.margin      = margin(t = 5, r = 5, b = 5, l = 5)
    )
}

# ── Per-cell-type plot pair ───────────────────────────────────────────────────
make_row_plots <- function(ct_dir, ct_label) {
  pred_dir <- file.path(base_dir, ct_dir, "predictions")
  stat_dir <- file.path(base_dir, ct_dir, "statistics")

  known        <- read_parquet(file.path(pred_dir, "infold_known_proteins_averaged.parquet"))
  known_cell   <- read.csv(file.path(stat_dir, "infold_known_per_cell.csv"))
  unknown_cell <- read.csv(file.path(stat_dir, "oof_unknown_per_cell.csv"))

  # Filter per-cell stats to cells matching this cell type
  filter_to_ct <- function(df) {
    barcodes <- ref_data$barcode[df$cell_id + 1L]
    ct       <- barcode_ct[barcodes]
    df[!is.na(ct) & ct == ct_label, ]
  }

  known_cell   <- filter_to_ct(known_cell)
  unknown_cell <- filter_to_ct(unknown_cell)

  known_lbl   <- "Known\n(n=205)"
  unknown_lbl <- "Unknown\n(n=205)"

  # ── Left: per-cell r_cal violin+boxplot, Known vs Unknown ──────────────────
  per_cell <- bind_rows(
    known_cell   |> select(r_cal) |> mutate(group = known_lbl),
    unknown_cell |> select(r_cal) |> mutate(group = unknown_lbl)
  )

  p_left <- ggplot(per_cell, aes(x = group, y = r_cal, fill = group)) +
    geom_violin(colour = "black", alpha = 0.6, linewidth = 0.4, trim = FALSE) +
    geom_boxplot(width = 0.15, colour = "black",
                 linewidth = 0.4, outlier.shape = NA) +
    scale_fill_manual(values = setNames(
      c(col_known, col_unknown), c(known_lbl, unknown_lbl)
    )) +
    labs(x = "Protein Status", y = "Pearson r per cell (calibrated)",
         title = ct_label) +
    base_theme() +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 10))

  # ── Right: per-protein r vs rmse_cal scatter, known proteins ───────────────
  metrics <- known |>
    group_by(protein) |>
    summarise(
      r        = cor(y_true, y_pred_cal, use = "complete.obs"),
      rmse_cal = sqrt(mean((y_true - y_pred_cal)^2, na.rm = TRUE)),
      .groups  = "drop"
    )

  top5_r      <- metrics |> slice_max(r,        n = 5) |> pull(protein)
  bottom5_r   <- metrics |> slice_min(r,        n = 5) |> pull(protein)
  top5_rmse   <- metrics |> slice_max(rmse_cal, n = 5) |> pull(protein)
  bottom5_rmse <- metrics |> slice_min(rmse_cal, n = 5) |> pull(protein)
  labeled     <- unique(c(top5_r, bottom5_r, top5_rmse, bottom5_rmse))
  metrics$label <- ifelse(metrics$protein %in% labeled,
                          as.character(metrics$protein), NA_character_)

  rmse_range       <- diff(range(metrics$rmse_cal, na.rm = TRUE))
  metrics$nudge_x  <- ifelse(metrics$protein %in% top5_r,      0.05, 0) +
                      ifelse(metrics$protein %in% bottom5_r,  -0.05, 0)
  metrics$nudge_y  <- ifelse(metrics$protein %in% top5_rmse,    rmse_range * 0.05, 0) +
                      ifelse(metrics$protein %in% bottom5_rmse, -rmse_range * 0.05, 0)

  p_right <- ggplot(metrics, aes(x = r, y = rmse_cal, label = label)) +
    geom_point(colour = col_known, alpha = 0.7, size = 1.8) +
    geom_label_repel(
      na.rm              = TRUE,
      size               = 2.5,
      max.overlaps       = Inf,
      segment.size       = 0.3,
      segment.colour     = "grey50",
      box.padding        = 0.3,
      label.padding      = 0.15,
      nudge_x            = metrics$nudge_x,
      nudge_y            = metrics$nudge_y,
      min.segment.length = 0
    ) +
    scale_x_continuous(name = "Pearson r") +
    scale_y_continuous(name = "RMSE (calibrated)") +
    base_theme()

  # ── Console summary ──────────────────────────────────────────────────────────
  med_known   <- median(known_cell$r_cal,   na.rm = TRUE)
  med_unknown <- median(unknown_cell$r_cal, na.rm = TRUE)

  med_r_prot  <- median(metrics$r, na.rm = TRUE)

  best_prot   <- metrics$protein[which.max(metrics$r)]
  worst_prot  <- metrics$protein[which.min(metrics$r)]
  r_max       <- max(metrics$r, na.rm = TRUE)
  r_min       <- min(metrics$r, na.rm = TRUE)

  message(
    ct_label, ":\n",
    "  Per-cell r_cal median  — Known: ", round(med_known, 3),
    "  |  Unknown: ", round(med_unknown, 3), "\n",
    "  Per-protein Pearson r  — Median: ", round(med_r_prot, 3),
    "  |  Range: ", best_prot, " (", round(r_max, 3), ")",
    "  to  ", worst_prot, " (", round(r_min, 3), ")"
  )

  list(left = p_left, right = p_right)
}

# ── Build all rows ────────────────────────────────────────────────────────────
plots <- lapply(celltypes, function(ct) make_row_plots(ct$dir, ct$label))

plot_list <- unlist(lapply(plots, function(p) list(p$left, p$right)), recursive = FALSE)

combined <- plot_grid(
  plotlist      = plot_list,
  ncol           = 2,
  labels         = c("A", "B", "C", "D", "E", "F"),
  label_size     = 18L,
  label_fontface = "bold",
  label_x        = 0,
  label_y        = 1.00,
  hjust          = 0,
  vjust          = 1
)

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(
  filename = file.path(out_dir, "per_celltype_panel.pdf"),
  plot     = combined,
  width    = 9,
  height   = 10,
  device   = cairo_pdf
)
ggsave(
  filename = file.path(out_dir, "per_celltype_panel.png"),
  plot     = combined,
  width    = 9,
  height   = 10,
  dpi      = 300
)

message("Saved to: ", out_dir)
