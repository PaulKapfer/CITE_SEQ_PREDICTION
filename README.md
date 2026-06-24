# Single-Cell Surface Protein Abundance, Encoded in the Transcriptome
## Code Repository

Analysis code for the publication
*"Single-Cell Surface Protein Abundance, Encoded in the Transcriptome: Mechanistic Origins of mRNA-Protein Discordance"*.

These scripts are copied directly from the production analysis tree and lightly documented (added headers, per-block comments). No algorithmic decisions were changed; every step is consistent with the methods description. The Code produces statistical evaluations in intermediate steps for exploratory purposes which are not used in the final publication. Statistical tests and results used for publication are mentioned in the methods description.

**Python 3.10.19 · R 4.3.3**

---

## Directory layout

```
Files2/
├── 01_rna_fm_embeddings/            RNA-FM transcript embeddings (Python)
├── 02_annotation/                   CellTypist annotation + validation (Python)
├── 03_training_data_preparation/    Prepare the data table used for XGBoost training (Python)
├── 04_hyperparameter_tuning/        Optuna XGBoost search (Python)
├── 05_model_training/               Full-model training & cross-validation (Within-dataset; Python)
├── 06_celltype_restricted_training/ Per-cell-type re-run of the full model (Python)
├── 07_ablation_study/               Additive feature ablation + mRNA-feature stats (Python + R)
├── 08_manual_mrna_features/         Manually-engineered mRNA features (R)
├── 09_external_validation/          Inference on the Kotliarov dataset (Python + R)
│   └── External_Validation_Package/ Installable inference package
│       └── _features/
└── 10_plotting/                     Figure scripts (R)
    ├── full_model/
    ├── ablation/
    ├── second_dataset/
    └── per_celltype/
```

---

## Scripts, in execution order

The pipeline has two independent inputs that feed model training:
**(A)** per-gene RNA-FM embeddings (step 01) and **(B)** the per-cell reference
table (steps 02–03: annotation → reference preparation). Steps 04–07 consume
both. Steps 08–10 are downstream analysis/validation/plotting.

### 01 · RNA-FM transcript embeddings
`01_rna_fm_embeddings/rnafm_features_final_run.py` (Python)
Maps each target gene to its MANE Select (canonical fallback) transcript via the
Ensembl REST API, fetches the cDNA, converts T→U, and runs RNA-FM (`rna_fm_t12`,
layer 12, 640-dim; positional table extended to 16,000 nt by periodic tiling).
Mean-pools over nucleotide positions (excluding `<cls>`/`<eos>`) → one `.npy`
embedding per gene. **Output:** `rnafm_features/<gene>.npy` (+ `metadata.csv`).

### 02 · Cell-type annotation & validation
`02_annotation/4.Annotation+validation.py` (Python)
De-novo annotation with CellTypist (`Immune_All_Low`, majority voting,
confidence 0.5), validated against canonical markers (DE + cross-tabulation vs
Leiden res-0.8 clusters, ≥70 % purity). Merges fine subtypes into broad classes
and drops ambiguous populations (HSC/MPP, megakaryocytes/platelets, ILC3).
**Output:** `adata_annotated.h5ad`, `cell_type_assignments.csv`.

### 03 · Training-data preparation
`03_training_data_preparation/reference_preparation.py` (Python)
Turns the annotated Hao et al. 2024 CITE-seq dataset into the per-cell reference
table and saves all artefacts needed to project new cells. Computes: log-norm
(target 1e4 → log1p); RPG pairwise co-expression → adaptive-kernel diffusion map
(25 DCs) with 25,000 Nyström landmarks; HVG-PCA (5,000 HVGs → 50 PCs, scalers +
loadings saved); endocytosis RRS — a custom UCell-inspired rank score, **not** the
UCell/pyUCell package (40 GO:BP gene sets, rank cap 1,500);
CLR-normalised ADT. **Input:** `adata_annotated.h5ad` (from step 02).
**Output:** `reference_data.parquet` + `RPG_diffmap/`, `HVG_PCA/` artefacts.

### 04 · Hyperparameter optimisation
`04_hyperparameter_tuning/hyperparameter_tuning_xgb_model3_round2.py` (Python)
Optuna TPE search (50 trials, MedianPruner). Fixed 80/20 donor `GroupShuffleSplit`
(test held out); 3-fold donor `GroupKFold` within the train pool; ≤20,000
cells/protein. Objective = mean MSE over all validation proteins. **Output:**
`best_hyperparameters.json` (the values hard-coded into steps 05–07).

### 05 · Full-model training & cross-validation
`05_model_training/GENESPLIT_no_celltype_proteinid_stratified.py` (Python)
Trains the XGBoost model on the 718-dim feature vector
(`protein_id` + `RNA_expr` + endocytosis + 25 DC + 50 PC + 640 RNA-FM).
Combined cell × protein CV: cells split 80/20 by donor; proteins split into 5
**gene-grouped, RNA-FM-stratified** folds (proteins sharing a transcript stay in
one fold; folds balanced by K-means on RNA-FM). Per fold: in-fold known proteins
(real `protein_id`) and out-of-fold unknown proteins (`protein_id = NaN`, zero-shot).
Post-hoc per-protein OLS calibration. Trains a final all-data model.
**Input:** `reference_data.parquet`, `rnafm_features/`, `adt_rna_mapping.csv`.
**Output:** predictions, per-protein/per-cell statistics, calibration params, models.

### 06 · Cell-type-restricted training
`06_celltype_restricted_training/PER_CELLTYPE_GENESPLIT.py` (Python)
Re-runs the step-05 pipeline independently within Monocytes, CD4 T, and CD8 T
cells (toggle per type), to test whether accuracy reflects cross-cell-type
variation. Identical CV design (gene-grouped, RNA-FM-stratified protein folds).

### 07 · Feature ablation study
`07_ablation_study/GENESPLIT_ablation_study.py` (Python)
Additive ablation over six conditions (baseline → +HVG-PCA → +RPG-DCs →
+Endocytosis → +RNA-FM → all features), reusing one saved cell/protein split for
all conditions with deterministic per-fold seeds; ≤25,000 cells/protein.
`07_ablation_study/ablation_mrna_feature_stats.R` (R)
Associates the manually-engineered mRNA features (step 08) with per-protein
accuracy gains across five "accuracy dimensions" (baseline r + four Δr's).
Spearman (continuous) / Wilcoxon + rank-biserial (binary); BH-FDR within each
dimension. **Output:** `mrna_feature_association_results.csv` (used by step 10).

### 08 · Manually-engineered mRNA features
`08_manual_mrna_features/` (R) — run in this order:
1. `generate_multimir_cache.R` — caches validated miRNA→target interactions (multiMiR).
2. `match_bed_annotations.R` — maps IRES / nTIS / uORF BED tracks (RefSeq) to HGNC
   symbols via BioMart → `ires_gene_hits.csv`, `ntis_gene_hits.csv`, `uorf_gene_hits.csv`.
3. `rna_features_human.R` — main extractor: per-transcript 5′UTR / CDS / 3′UTR
   features (TIS efficiency, TOP tract, G-quadruplexes, uORFs, MFE via RNAfold,
   CAI, ENC, GC3, AREs, PAS, miRNA density, m⁶A-DRACH density). Isoform priority:
   MANE Select > APPRIS Principal1 > TSL. **Output:** `rna_features.csv`.

### 09 · External validation (Kotliarov et al. 2020)
`09_external_validation/` (Python + R) — run in this order:
1. `run_External_Validation_Package_kotliarov.py` — orchestrator: loads the
   annotated Kotliarov AnnData, computes the three training-matched feature
   blocks (`prepare_anndata`), predicts **every protein with `protein_id = NaN`**
   (`annotate_unknown`; raw predictions clipped at 0), saves the h5ad, and exports
   flat files for R.
2. `assemble_seurat.R` — assembles a Seurat object from the exported CSV/MTX files
   (called automatically by the orchestrator).
3. `Evaluate_predictions_kotliarev.r` — per-protein OLS calibration + per-protein /
   per-cell / per-cell-type Pearson r, R², RMSE → CSVs consumed by step 10.

`External_Validation_Package/` — installable inference package (`pip install -e .`):

| Module | Role |
|---|---|
| `prepare.py` | `prepare_anndata`: HVG-PCA projection, RPG diffusion Nyström interpolation, endocytosis RRS (custom UCell-inspired, not the UCell package) |
| `annotate_unknown.py` | Predict all proteins with `protein_id = NaN` (clipped at 0) |
| `_predict.py` | Feature-matrix construction, isoform resolution, calibration helpers |
| `_data.py` | Loaders for bundled model / mapping / RNA-FM / quality tables |
| `_features/` | `pca.py`, `diffmap.py`, `endocytosis.py` (one feature block each) |
| `__init__.py`, `setup.py` | Package API and install metadata |

> The package's `data/` directory (model weights, reference artefacts, RNA-FM
> embeddings) is not bundled in this repository due to size.

### 10 · Plotting (R)

| File | Figure | Reads from |
|---|---|---|
| `full_model/Full_model_panel1.R` | Fig. 1 | 05 predictions/statistics |
| `ablation/ablation_heatmap_known_horizontal.R` | Fig. 2 | 07 per-protein stats |
| `ablation/ablation_per_feature.R` | Fig. 3 | 07 per-protein stats |
| `ablation/ablation_mRNA_feature_barplot.R` | Fig. 4 | 07 `mrna_feature_association_results.csv` |
| `second_dataset/evaluation_panel.R` | Fig. 5 | 09 exported predictions + eval CSVs |
| `per_celltype/Per_celltype_panel.R` | Supplemental Fig. 1 | 06 predictions/statistics |

The Known/Unknown labels in Figures 4–5 are display-only: a protein is "Known" if
its matched transcript was used during training, "Unknown" otherwise. All transfer-dataset
proteins are predicted identically with `protein_id = NaN`.

---

