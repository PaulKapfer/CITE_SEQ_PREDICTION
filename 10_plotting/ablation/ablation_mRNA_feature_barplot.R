# =============================================================================
# ABLATION mRNA-FEATURE ASSOCIATION — BARPLOT
# =============================================================================
# Plots the significant mRNA-feature associations from ablation_mrna_feature_stats.R.
# Restricts to the HVG-PCA accuracy-gain dimension (delta_pca) — the only
# dimension with significant findings — so a single, correctly-labelled effect
# size (Spearman ρ) is shown per feature. Significance stars: * <0.05, ** <0.01,
# *** <0.001 (BH-FDR adjusted). Outputs PDF + PNG.
# =============================================================================

library(dplyr)
library(ggplot2)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Expects the results CSV produced by 07_ablation_study/ablation_mrna_feature_stats.R
results_file <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output/mrna_feature_stats/mrna_feature_association_results.csv"
out_dir      <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Ablation_study/mRNA_feature_barplot"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Feature name translations ─────────────────────────────────────────────────
feature_labels <- c(
  utr5_length                = "5'UTR length",
  utr5_gc_content            = "5'UTR GC content",
  utr5_purine_content        = "5'UTR purine content",
  utr5_mfe                   = "5'UTR min. free energy",
  utr5_mfe_normalized        = "5'UTR MFE (normalized)",
  tis_efficiency             = "TIS efficiency",
  has_TOP_candidate_motif    = "TOP motif",
  uORF_ATG_count             = "uORF count (ATG)",
  uORF_nonATG_count          = "uORF count (non-ATG)",
  uORF_longest_length        = "uORF longest length",
  uORF_distance_to_CDS       = "uORF distance to CDS",
  uORF_overlaps_CDS          = "uORF overlaps CDS",
  has_validated_uORF         = "validated uORF",
  utr5_g4_max_score          = "5'UTR G4 max score",
  utr5_g4_count              = "5'UTR G4 count",
  utr5_g4_mean_score         = "5'UTR G4 mean score",
  utr5_g4_max_tetrads        = "5'UTR G4 max tetrads",
  utr5_m6a_count             = "5'UTR m6A count",
  utr5_m6a_density           = "5'UTR m6A density",
  IRES_validated             = "validated IRES",
  noncanonical_TIS_validated = "validated non-canonical TIS",
  cds_length                 = "CDS length",
  cds_gc_content             = "CDS GC content",
  cds_purine_content         = "CDS purine content",
  cai                        = "codon adaptation index",
  enc                        = "effective number of codons",
  gc3                        = "GC3 content",
  cds_m6a_count              = "CDS m6A count",
  cds_m6a_density            = "CDS m6A density",
  utr3_length                = "3'UTR length",
  utr3_gc_content            = "3'UTR GC content",
  utr3_purine_content        = "3'UTR purine content",
  are_class1_count           = "3'UTR ARE class I count",
  are_class2_count           = "3'UTR ARE class II count",
  are_class3_count           = "3'UTR ARE class III count",
  are_nonamer_count          = "3'UTR ARE nonamer count",
  are_pentamer_count         = "3'UTR ARE pentamer count",
  PAS_distance               = "PAS distance",
  utr3_mfe                   = "3'UTR minimum free energy",
  utr3_mfe_normalized        = "3'UTR MFE (normalized)",
  miRNA_count                = "3'UTR miRNA target count",
  miRNA_density              = "3'UTR miRNA target density",
  utr3_m6a_count             = "3'UTR m6A-motif count",
  utr3_m6a_density           = "3'UTR m6A density",
  mrna_m6a_count             = "mRNA m6A-motif count",
  mrna_m6a_density           = "mRNA m6A density"
)

# ── Load results and filter to HVG-PCA accuracy gain (delta_pca) ─────────────
# Only HVG-PCA yielded significant mRNA feature associations; plotting a single
# accuracy dimension avoids mixing incommensurable effect sizes on one y-axis.
results <- read.csv(results_file)

sig <- results |>
  filter(significant, dimension == "delta_pca") |>
  mutate(
    sig_label    = case_when(
      p_adj < 0.001 ~ "***",
      p_adj < 0.01  ~ "**",
      TRUE          ~ "*"
    ),
    feature_pretty = feature_labels[feature]
  )

feat_order <- sig |>
  arrange(desc(estimate)) |>
  distinct(feature_pretty) |>
  pull(feature_pretty)

sig <- sig |>
  mutate(
    feature_pretty = factor(feature_pretty, levels = feat_order),
    bar_top        = ifelse(estimate >= 0, estimate + 0.015, estimate - 0.015),
    star_vjust     = ifelse(estimate >= 0, 0, 1)
  )

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(sig, aes(x = feature_pretty, y = estimate)) +
  geom_col(fill = "#89C4E1", colour = "grey30", linewidth = 0.3, width = 0.7) +
  geom_text(aes(y = bar_top, label = sig_label, vjust = star_vjust), size = 3) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.3) +
  labs(x = "mRNA Feature", y = "Correlation with Δr HVG-PCA (Spearman ρ)") +
  theme_minimal(base_size = 9) +
  theme(
    axis.text.x        = element_text(angle = 45, hjust = 1, size = 7),
    axis.text.y        = element_text(size = 9),
    axis.title         = element_text(size = 10),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.line          = element_line(colour = "black", linewidth = 0.4),
    plot.margin        = margin(t = 15, r = 10, b = 5, l = 5)
  )

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(file.path(out_dir, "mRNA_feature_barplot.pdf"), p,
       width = 5, height = 4, device = cairo_pdf)
ggsave(file.path(out_dir, "mRNA_feature_barplot.png"), p,
       width = 5, height = 4, dpi = 300)

message("Saved to: ", out_dir)
