# =============================================================================
# ABLATION — PER-FEATURE TOP-PROTEIN LOLLIPOPS
# =============================================================================
# For each single-feature ablation condition, shows the proteins that gain the
# most calibrated-r over baseline. Builds a 4-panel figure (RNA-FM, RPG-DCs,
# Endocytosis, HVG-PCA), each a lollipop of the top-20 Δr proteins (restricted
# to proteins reaching r_cal > 0.8 in at least one condition). Outputs PDF + PNG.
# Reads per-protein in-fold-known stats from the ablation study output (07).
# =============================================================================

library(dplyr)
library(ggplot2)
library(cowplot)
library(RColorBrewer)

# ── Paths ─────────────────────────────────────────────────────────────────────
abl_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output"
out_dir <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Ablation_study/Per_feature"

conditions <- c("baseline", "add_dc", "add_endocytosis", "add_pca", "add_rnafm", "all_features")

condition_labels <- c(
  baseline        = "Baseline",
  add_dc          = "Baseline + RPG-DCs",
  add_endocytosis = "Baseline + Endocytosis",
  add_pca         = "Baseline + HVG-PCA",
  add_rnafm       = "Baseline + RNA-FM",
  all_features    = "Full Model"
)

dark2_colors     <- brewer.pal(8, "Dark2")
condition_colors <- setNames(dark2_colors[seq_along(condition_labels)], condition_labels)

R_THRESHOLD <- 0.8
TOP_N       <- 20

# ── Load known per-protein r_cal for all conditions ───────────────────────────
load_condition <- function(cond, file) {
  read.csv(file.path(abl_dir, cond, "statistics", file)) |>
    mutate(condition = condition_labels[[cond]])
}

known_data <- bind_rows(lapply(conditions, load_condition, file = "infold_known_per_protein.csv"))

# ── Filter: proteins with r_cal > threshold in ≥1 condition ──────────────────
keep <- known_data |>
  group_by(protein) |>
  summarise(passes = any(r_cal > R_THRESHOLD, na.rm = TRUE), .groups = "drop") |>
  filter(passes) |>
  pull(protein)

known_data <- known_data |> filter(protein %in% keep)

# ── Compute delta vs baseline ─────────────────────────────────────────────────
baseline_r <- known_data |>
  filter(condition == "Baseline") |>
  select(protein, r_baseline = r_cal)

known_delta <- known_data |>
  filter(condition != "Baseline") |>
  inner_join(baseline_r, by = "protein") |>
  mutate(delta = r_cal - r_baseline)

# ── Shared theme ──────────────────────────────────────────────────────────────
base_theme <- function() {
  theme_minimal(base_size = 9) +
    theme(
      axis.title         = element_text(size = 10),
      axis.text          = element_text(size = 9),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.line          = element_line(colour = "black", linewidth = 0.5),
      plot.title         = element_text(size = 10, face = "bold", hjust = 0.5),
      plot.margin        = margin(t = 8, r = 10, b = 5, l = 5)
    )
}

# ── Lollipop factory ──────────────────────────────────────────────────────────
make_lollipop <- function(cond_label, title, top_n = TOP_N) {
  plot_df <- known_delta |>
    filter(condition == cond_label) |>
    slice_max(delta, n = top_n) |>
    mutate(protein = factor(protein, levels = protein[order(delta)]))

  col <- condition_colors[[cond_label]]

  ggplot(plot_df, aes(x = delta, y = protein)) +
    geom_segment(aes(x = 0, xend = delta, y = protein, yend = protein),
                 colour = col, linewidth = 0.6) +
    geom_point(colour = col, size = 2.5) +
    scale_x_continuous(
      name   = expression(Delta ~ "Pearson r  (vs. Baseline)"),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(title = title, y = NULL) +
    base_theme()
}

# ── Build 4 panels (A=RNA-FM, B=RPG-DCs, C=Endocytosis, D=HVG-PCA) ──────────
p_A <- make_lollipop("Baseline + RNA-FM",       "RNA-FM")
p_B <- make_lollipop("Baseline + RPG-DCs",      "RPG-DCs")
p_C <- make_lollipop("Baseline + Endocytosis",  "Endocytosis")
p_D <- make_lollipop("Baseline + HVG-PCA",      "HVG-PCA")

# ── Assemble: rows 1+2 (2×2 lollipops, −10% height), row 3 (E, full width) ───
row1 <- plot_grid(
  p_A, p_B,
  nrow = 1, labels = c("A", "B"),
  label_size = 18L, label_fontface = "bold",
  label_x = 0, label_y = 1.00, hjust = 0, vjust = 1
)
row2 <- plot_grid(
  p_C, p_D,
  nrow = 1, labels = c("C", "D"),
  label_size = 18L, label_fontface = "bold",
  label_x = 0, label_y = 1.00, hjust = 0, vjust = 1
)
# Rows 1 and 2 at 4.5 in each (5 in × 0.9)
combined <- plot_grid(
  row1, row2,
  ncol        = 1,
  rel_heights = c(4.5, 4.5)
)

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(file.path(out_dir, "ablation_per_feature.pdf"), combined,
       width = 10, height = 9, device = cairo_pdf, limitsize = FALSE)
ggsave(file.path(out_dir, "ablation_per_feature.png"), combined,
       width = 10, height = 9, dpi = 300, limitsize = FALSE)

message("Saved to: ", out_dir)
