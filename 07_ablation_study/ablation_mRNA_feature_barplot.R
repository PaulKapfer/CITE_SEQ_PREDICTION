library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)

# ── Paths ─────────────────────────────────────────────────────────────────────
abl_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output"
ann_file <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Scripts/Ablation/protein_list_annotated.csv"
rna_file <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual/Output/rna_features.csv"
out_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Output/Ablation_study/mRNA_feature_barplot"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Dimensions & features ─────────────────────────────────────────────────────
dimensions <- c("baseline_r", "delta_rnafm", "delta_dc", "delta_endocytosis", "delta_pca")
dim_labels <- c(
  baseline_r        = "Baseline r",
  delta_rnafm       = "Δr RNA-FM",
  delta_dc          = "Δr RPG-DC",
  delta_endocytosis = "Δr Endocytosis",
  delta_pca         = "Δr HVG-PCA"
)

rna_features <- c(
  "utr5_length", "utr5_gc_content", "utr5_purine_content",
  "utr5_mfe", "utr5_mfe_normalized", "tis_efficiency",
  "has_TOP_candidate_motif", "uORF_ATG_count", "uORF_nonATG_count", 
  "uORF_longest_length", "uORF_distance_to_CDS", "uORF_overlaps_CDS",
  "has_validated_uORF", "utr5_g4_max_score", "utr5_g4_count",
  "utr5_g4_mean_score", "utr5_g4_max_tetrads", "utr5_m6a_count",
  "utr5_m6a_density", "IRES_validated", "noncanonical_TIS_validated",
  "cds_length", "cds_gc_content", "cds_purine_content",
  "cai", "enc", "gc3", "cds_m6a_count", "cds_m6a_density",
  "utr3_length", "utr3_gc_content", "utr3_purine_content",
  "are_class1_count", "are_class2_count", "are_class3_count",
  "are_nonamer_count", "are_pentamer_count", "PAS_distance",
  "utr3_mfe", "utr3_mfe_normalized", "miRNA_count", "miRNA_density",
  "utr3_m6a_count", "utr3_m6a_density", "mrna_m6a_count", "mrna_m6a_density"
)

# ── Build analysis data frame ─────────────────────────────────────────────────
conds <- c(baseline = "Baseline", add_dc = "RPG-DC",
           add_endocytosis = "Endocytosis", add_pca = "HVG-PCA", add_rnafm = "RNA-FM")

all_data <- bind_rows(lapply(names(conds), function(cond) {
  read.csv(file.path(abl_dir, cond, "statistics", "infold_known_per_protein.csv")) |>
    select(protein, r_cal) |>
    mutate(label = conds[[cond]])
}))

wide <- all_data |>
  pivot_wider(names_from = label, values_from = r_cal) |>
  mutate(
    delta_rnafm       = `RNA-FM`    - Baseline,
    delta_dc          = `RPG-DC`    - Baseline,
    delta_endocytosis = Endocytosis - Baseline,
    delta_pca         = `HVG-PCA`   - Baseline
  ) |>
  rename(baseline_r = Baseline) |>
  select(protein, all_of(dimensions))

ann <- read.csv(ann_file) |>
  select(protein, transcript = Alternative.Name.Gene.Symbol)
rna <- read.csv(rna_file)  |> select(Gene, all_of(rna_features))
df  <- wide |>
  inner_join(ann, by = "protein") |>
  left_join(rna, by = c("transcript" = "Gene"))

# ── Statistical tests ─────────────────────────────────────────────────────────
test_association <- function(x, y) {
  keep <- !is.na(x) & !is.na(y)
  x <- x[keep]; y <- y[keep]; n <- length(x)
  if (n < 5) return(tibble(statistic=NA_real_, estimate=NA_real_,
                            p_value=NA_real_, test=NA_character_, n=n))
  if (is.logical(y)) {
    grp <- factor(y)
    if (length(levels(grp)) < 2)
      return(tibble(statistic=NA_real_, estimate=NA_real_,
                    p_value=NA_real_, test="wilcoxon", n=n))
    wt   <- wilcox.test(x ~ grp, exact = FALSE)
    n1   <- sum(grp == levels(grp)[1]); n2 <- sum(grp == levels(grp)[2])
    r_rb <- 1 - (2 * wt$statistic) / (n1 * n2)
    tibble(statistic=wt$statistic, estimate=as.numeric(r_rb),
           p_value=wt$p.value, test="wilcoxon", n=n)
  } else {
    ct <- cor.test(x, y, method = "spearman", exact = FALSE)
    tibble(statistic=ct$statistic, estimate=ct$estimate,
           p_value=ct$p.value, test="spearman", n=n)
  }
}

results <- expand_grid(dimension = dimensions, feature = rna_features) |>
  mutate(res = map2(dimension, feature,
                    ~ test_association(df[[.x]], df[[.y]]))) |>
  unnest(res) |>
  group_by(dimension) |>
  mutate(p_adj = p.adjust(p_value, method = "BH")) |>
  ungroup() |>
  mutate(
    significant = !is.na(p_adj) & p_adj < 0.05,
    dim_label   = factor(dim_labels[dimension], levels = dim_labels)
  )

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

# ── Prepare significant results for plotting ──────────────────────────────────
sig <- results |>
  filter(significant) |>
  mutate(
    sig_label    = case_when(
      p_adj < 0.001 ~ "***",
      p_adj < 0.01  ~ "**",
      TRUE          ~ "*"
    ),
    feature_pretty = feature_labels[feature]
  )

# Order features by effect size (descending)
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

# ── Save statistical results ──────────────────────────────────────────────────
results_out <- results |>
  mutate(
    dimension_label = dim_labels[dimension],
    feature_label   = feature_labels[feature]
  ) |>
  select(dimension, dimension_label, feature, feature_label,
         test, n, statistic, estimate, p_value, p_adj, significant) |>
  arrange(dimension, p_adj)

write.csv(results_out,
          file.path(out_dir, "mRNA_feature_statistics.csv"),
          row.names = FALSE)

# ── Save ──────────────────────────────────────────────────────────────────────
ggsave(file.path(out_dir, "mRNA_feature_barplot.pdf"), p,
       width = 5, height = 4, device = cairo_pdf)
ggsave(file.path(out_dir, "mRNA_feature_barplot.png"), p,
       width = 5, height = 4, dpi = 300)
ggsave(file.path(out_dir, "mRNA_feature_barplot.tif"), p,
       width = 5, height = 4, dpi = 600,
       device = function(filename, width, height, ...) tiff(filename, width = width, height = height, units = "in", res = 600, compression = "lzw"))

message("Saved to: ", out_dir)
