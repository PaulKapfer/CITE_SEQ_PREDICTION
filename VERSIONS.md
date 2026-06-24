# Dependencies & Versions

Software dependencies of the analysis code, the exact versions used, and the
scripts that import them. Intended for readers who want to recreate the workflow.

All versions below were used with:
- **Python**: **3.10.19**
- **R**: **4.3.3**

---

## Python packages

| Package | Version | Used in (directory / script) |
|---|---|---|
| numpy | 2.2.6 | 01, 02, 03, 04, 05, 06, 07, 09 (all analysis scripts + package) |
| pandas | 2.3.3 | 01, 02, 03, 04, 05, 06, 07, 09 |
| scipy | 1.15.2 | 03, 04, 05, 06, 07, 09 |
| scikit-learn | 1.7.2 | 03, 04, 05, 06, 07 |
| xgboost | 3.2.0 | 04, 05, 06, 07, 09 |
| anndata | 0.11.4 | 02, 03, 09 |
| scanpy | 1.11.5 | 02, 09 |
| matplotlib | 3.10.8 | 02, 05, 06, 07 |
| seaborn | 0.13.2 | 02 (`4.Annotation+validation.py`) |
| joblib | 1.5.3 | 03, 09 |
| pynndescent | 0.5.13 | 03 (`reference_preparation.py`) |
| gseapy | 1.1.13 | 03 (`reference_preparation.py`) |
| celltypist | 1.7.1 | 02 (`4.Annotation+validation.py`) |
| optuna | 4.8.0 | 04 (`hyperparameter_tuning_xgb_model3_round2.py`) |
| optuna-integration | 4.8.0 | 04 (`hyperparameter_tuning_xgb_model3_round2.py`) |
| torch (PyTorch) | 2.11.0 | 01 (`rnafm_features_final_run.py`) |
| rna-fm | 0.2.2 | 01 (`rnafm_features_final_run.py`) |
| requests | 2.32.5 | 01 (`rnafm_features_final_run.py`) |
| setuptools | 80.9.0 | 09 (`External_Validation_Package/setup.py`, build only) |

Standard-library modules used (no separate install): `os, sys, time, queue,
logging, threading, math, json, re, pickle, gc, shutil, argparse, warnings,
pathlib, collections, subprocess`.

---

## R packages

| Package | Version | Used in (directory / script) |
|---|---|---|
| dplyr | 1.1.4 | 07, 08, 09, 10 (most R scripts) |
| ggplot2 | 4.0.1 | 09 (`Evaluate_predictions_kotliarev.r`), 10 (all panels) |
| tidyr | 1.3.2 | 07 (`ablation_mrna_feature_stats.R`), 10 (ablation heatmap) |
| purrr | 1.0.4 | 07 (`ablation_mrna_feature_stats.R`) |
| forcats | 1.0.1 | 10 (`full_model/Full_model_panel1.R`) |
| stringr | 1.6.0 | 08 (`match_bed_annotations.R`, `rna_features_human.R`) |
| ggrepel | 0.9.6 | 10 (full_model, per_celltype, second_dataset panels) |
| cowplot | 1.2.0 | 10 (full_model, per_celltype, second_dataset, ablation_per_feature) |
| RColorBrewer | 1.1.3 | 10 (`ablation/ablation_per_feature.R`) |
| ComplexHeatmap | 2.26.0 | 10 (`ablation/ablation_heatmap_known_horizontal.R`) |
| circlize | 0.4.17 | 10 (`ablation/ablation_heatmap_known_horizontal.R`) |
| nanoparquet | 0.5.0 | 10 (full_model, per_celltype panels) |
| Seurat | 5.4.0 | 09 (`assemble_seurat.R`, `Evaluate_predictions_kotliarev.r`) |
| Matrix | 1.6.5 | 09 (`assemble_seurat.R`) |
| biomaRt | 2.58.2 | 08 (`match_bed_annotations.R`, `rna_features_human.R`) |
| Biostrings | 2.70.3 | 08 (`rna_features_human.R`) |
| pqsfinder | 2.18.0 | 08 (`rna_features_human.R`) |
| multiMiR | 1.24.0 | 08 (`generate_multimir_cache.R`) |
| decoupleR | 2.16.0 | 08 (`rna_features_human.R`, for the CollecTRI gene list) |
| pbapply | 1.7.4 | 08 (`rna_features_human.R`) |
| parallel | 4.3.3 | 08 (`rna_features_human.R`) — base R |
| grid | 4.3.3 | 10 (`ablation/ablation_heatmap_known_horizontal.R`) — base R |

---

## External tools & data resources

Invoked as binaries or accessed as web services / reference data (needed to
reproduce the workflow):

| Resource | Version | Used in |
|---|---|---|
| ViennaRNA / `RNAfold` (binary, via `system2`) | 2.7.2 | 08 (`rna_features_human.R`) |
| Ensembl REST API | web service | 01 |
| Ensembl BioMart | web service | 08 |
| Enrichr (via `gseapy`) / MSigDB C5 GO:BP gene sets | web service | 03 |
| KAZUSA codon-usage table; Noderer 2014 TIS table | reference data | 08 |
| uORFdb; Human IRES Atlas | reference data | 08 |
| CellTypist model `Immune_All_Low.pkl` | reference model | 02 |
| RNA-FM pretrained weights (`rna_fm_t12`) | reference model | 01 |
| CITE-seq datasets (Hao et al. 2024; Kotliarov et al. 2020) | input data | 02–09 |
