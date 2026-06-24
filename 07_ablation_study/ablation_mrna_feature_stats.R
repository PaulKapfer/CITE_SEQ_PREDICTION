# =============================================================================
# ABLATION mRNA-FEATURE ASSOCIATION — STATISTICS
# =============================================================================
# Tests whether manually-engineered mRNA features (08_manual_mrna_features)
# relate to per-protein prediction accuracy in the ablation study (07).
#
# For each protein it builds five "accuracy dimensions":
#   baseline_r        – calibrated Pearson r of the baseline condition
#   delta_{rnafm,dc,endocytosis,pca} – r gain of each single-feature condition
#                                      over baseline (condition r_cal − baseline r_cal)
# Each dimension is tested against 46 mRNA features:
#   continuous/count → Spearman (ρ); binary → Wilcoxon (rank-biserial r_rb).
# p-values are BH-FDR corrected WITHIN each dimension; FDR < 0.05 = significant.
#
# INPUT : per-protein in-fold-known stats from each ablation condition (07),
#         protein→transcript annotation, and the mRNA feature table (08).
# OUTPUT: mrna_feature_association_results.csv  (consumed by
#         10_plotting/ablation/ablation_mRNA_feature_barplot.R)
# =============================================================================

library(dplyr)
library(tidyr)
library(purrr)

# ── Paths ─────────────────────────────────────────────────────────────────────
abl_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output"
ann_file <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Plotting/Scripts/Ablation/protein_list_annotated.csv"
rna_file <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Generate_RNA_features_manual/Output/rna_features.csv"
out_dir  <- "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output/mrna_feature_stats"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Accuracy dimensions and mRNA features ────────────────────────────────────
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

# ── Build per-protein accuracy data frame ─────────────────────────────────────
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

ann <- read.csv(ann_file) |> select(protein, transcript)
rna <- read.csv(rna_file)  |> select(Gene, all_of(rna_features))
df  <- wide |>
  inner_join(ann, by = "protein") |>
  left_join(rna, by = c("transcript" = "Gene"))

# ── Statistical tests ─────────────────────────────────────────────────────────
# Continuous/count features: Spearman rank correlation (ρ as effect size).
# Binary (logical) features: two-sided Wilcoxon rank-sum test (rank-biserial r_rb as effect size).
# Feature–dimension pairs with n < 5 paired non-missing observations are excluded.
# BH FDR correction is applied within each accuracy dimension separately.
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

# ── Save results ──────────────────────────────────────────────────────────────
out_file <- file.path(out_dir, "mrna_feature_association_results.csv")
write.csv(results, out_file, row.names = FALSE)
message("Results saved to: ", out_file)
message("Significant associations (FDR < 0.05): ", sum(results$significant, na.rm = TRUE))
