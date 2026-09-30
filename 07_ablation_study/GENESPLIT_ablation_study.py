#!/usr/bin/env python3
"""
GENESPLIT_ablation_study.py
============================
Additive feature study for the GENESPLIT protein prediction model.

protein_id and RNA_expr are the minimal baseline — always present in every
condition. Additional feature classes are introduced one at a time, then
removed again, so the contribution of each class is measured against the
same two-feature baseline.

Conditions
----------
  baseline        – protein_id + RNA_expr  (minimal model)
  add_pca         – protein_id + RNA_expr + PCA dims
  add_dc          – protein_id + RNA_expr + DC dims
  add_endocytosis – protein_id + RNA_expr + endocytosis / phagocytosis score
  add_rnafm       – protein_id + RNA_expr + RNA-FM embeddings
  add_rnafm_dc    – protein_id + RNA_expr + RNA-FM + DC dims
                    (interaction between structural embeddings and cell topology)

Two prediction sets are collected per fold (mirroring the reference script):
  1. OOF unknown proteins  — te_pairs × cell_test_global (protein_id = NaN)
  2. In-fold known proteins — tr_pairs × cell_test_global (protein_id = real id)
     Each known protein appears in 4 of 5 folds; saved both per-fold and
     fold-averaged.

Notes
-----
- Shared protein fold splits and cell split are written to shared_splits/ once
  and reused across all conditions for a fair comparison.
- Per-fold RNG seeds are deterministic (RANDOM_STATE + fold_i * 1000) so cell
  subsampling is identical across conditions for the same fold, even on resume.
- CELLS_PER_PROTEIN = 25 000 (reduced from full-run default for speed).
- Each condition has its own checkpoint; individual conditions can be re-started
  without repeating completed folds or touching other conditions.
- A summary CSV and two bar-charts (unknown / known proteins) are written to the
  root output directory once all conditions finish.

Output layout
-------------
  <ABLATION_BASE>/
    ablation_summary.csv
    ablation_summary_unknown.png
    ablation_summary_known.png
    shared_splits/
      fold_protein_splits.csv
      cell_split.npz
    <condition>/
      execution.log
      cv_checkpoint.pkl
      predictions/
        oof_unknown_proteins.parquet
        infold_known_proteins_perfold.parquet
        infold_known_proteins_averaged.parquet
      statistics/
        oof_unknown_per_protein.csv
        oof_unknown_per_cell.csv
        infold_known_per_protein.csv
        infold_known_per_cell.csv
      calibration/
        lm_params_per_protein_per_fold.csv
      models/
        model_fold_{0..4}.json
        fold_protein_splits.csv
"""

import os, sys, logging, pickle, gc, shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, spearmanr as _spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
REF_PARQUET   = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Reference_Preparation/Output/reference_data.parquet"
MAPPING_CSV   = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Matching_ADT_Transcript/Output2/combined_adt_mapping_MODEL-TRAINING.csv"
RNAFM_DIR     = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Create_RNA_FM_features/Output/rnafm_features"
ABLATION_BASE = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Ablation_study/Output"

best_params = {
    "max_depth":         8,
    "learning_rate":     0.06161472706880244,
    "subsample":         0.7123043816958816,
    "colsample_bytree":  0.6108327809165853,
    "min_child_weight":  256,
    "reg_alpha":         0.040377333089617114,
    "reg_lambda":        0.00014728652782921222,
    "gamma":             0.2203501322530219,
}

N_SPLITS                     = 5
CELLS_PER_PROTEIN            = 25_000
TUNING_MAX_CELLS_PER_PROTEIN = 20_000
RNAFM_TOP                    = 640
N_PCS                        = 50
DROPOUT_THR                  = 0.5
MIN_CELLS_PER_CELLTYPE       = 10
RANDOM_STATE                 = 42
N_JOBS                       = 12

# Additive ablation design: protein_id and RNA_expr form the mandatory baseline.
# Each subsequent condition adds exactly one additional feature class on top of the baseline,
# allowing the marginal contribution of each feature class to be measured independently.
# The final condition ("all_features") combines all classes and matches the full production model.
# Optional feature-class keys: "pca" (PCA coords), "dc" (diffusion-map coords),
#                               "phagocytosis" (endocytosis UCell score), "rnafm" (RNA-FM embeddings)
CONDITIONS = [
    ("baseline",        frozenset()),                          # protein_id + RNA_expr only
    ("add_pca",         frozenset({"pca"})),                   # + PCA dims
    ("add_dc",          frozenset({"dc"})),                    # + DC dims
    ("add_endocytosis", frozenset({"phagocytosis"})),          # + endocytosis score
    ("add_rnafm",       frozenset({"rnafm"})),                 # + RNA-FM embeddings
    ("all_features",    frozenset({"pca", "dc", "phagocytosis", "rnafm"})),  # full model
]

# ── SHARED OUTPUT / ROOT LOGGER ────────────────────────────────────────────────
os.makedirs(ABLATION_BASE, exist_ok=True)
SHARED_DIR = os.path.join(ABLATION_BASE, "shared_splits")
os.makedirs(SHARED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("GENESPLIT ADDITIVE FEATURE STUDY — loading data")
log.info("=" * 70)
df      = pd.read_parquet(REF_PARQUET)
mapping = pd.read_csv(MAPPING_CSV)
log.info(f"  Reference data: {len(df):,} cells × {len(df.columns)} columns")

log.info("  Loading RNA-FM features...")
rnafm_raw = {fn[:-4]: np.load(os.path.join(RNAFM_DIR, fn)).astype(np.float32)
             for fn in os.listdir(RNAFM_DIR) if fn.endswith(".npy")}
all_vecs    = np.stack(list(rnafm_raw.values()))
top_dims    = np.argsort(all_vecs.var(axis=0))[::-1][:RNAFM_TOP]
rnafm_cache = {g: v[top_dims] for g, v in rnafm_raw.items()}
del rnafm_raw, all_vecs; gc.collect()

protein_to_gene = dict(zip(mapping["ADT_feature"], mapping["RNA_gene"]))
# A missing RNA_<gene> column is not a reason to drop the antibody: the gene was
# simply never detected in the scRNA-seq matrix, so its log-normalised expression
# is 0 in every cell (_build_X substitutes a zero column). The ADT measurement is
# still real and the protein remains predictable from RNA-FM and cell state.
usable = [
    (p, g) for p, g in protein_to_gene.items()
    if f"ADT_{p}" in df.columns and g in rnafm_cache
]
_no_rna = sorted({g for _, g in usable if f"RNA_{g}" not in df.columns})
log.info(f"  Usable proteins: {len(usable)}")
if _no_rna:
    log.info(f"  Genes undetected in scRNA-seq (RNA_expr = 0): "
             f"{len(_no_rna)}  {_no_rna}")

# Stable protein_id mapping (alphabetical → reproducible integer ids)
usable_sorted  = sorted(usable, key=lambda x: x[0])
protein_id_map = {feat: float(i) for i, (feat, _) in enumerate(usable_sorted)}

dc_cols = sorted([c for c in df.columns if c.startswith("DC") and c[2:].isdigit()],
                 key=lambda x: int(x[2:]))
pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1) if f"PC{i}" in df.columns]
log.info(f"  DC dims: {len(dc_cols)}  |  PCA dims: {len(pc_cols)}")

usable_array = np.array(usable)


# ── 2. SHARED SPLITS ──────────────────────────────────────────────────────────
CELL_SPLIT_FILE = os.path.join(SHARED_DIR, "cell_split.npz")
FOLD_SPLIT_CSV  = os.path.join(SHARED_DIR, "fold_protein_splits.csv")

# Shared splits ensure all conditions are evaluated on exactly the same cells and protein folds.
# Without this, random variation across conditions would confound the feature contribution signal.
log.info("Computing / loading shared train/test splits...")

# Cell split — fixed 20% test hold-out by donor
if os.path.exists(CELL_SPLIT_FILE):
    _s = np.load(CELL_SPLIT_FILE)
    cell_train_global = _s["cell_train_global"]
    cell_test_global  = _s["cell_test_global"]
    log.info(f"  Cell split loaded — train: {len(cell_train_global):,} / "
             f"test: {len(cell_test_global):,}")
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    cell_train_global, cell_test_global = next(
        gss.split(np.arange(len(df)), groups=df["donor"].values))
    np.savez(CELL_SPLIT_FILE,
             cell_train_global=cell_train_global,
             cell_test_global=cell_test_global)
    log.info(f"  Cell split saved — train: {len(cell_train_global):,} cells / "
             f"{df['donor'].iloc[cell_train_global].nunique()} donors  |  "
             f"test: {len(cell_test_global):,} cells / "
             f"{df['donor'].iloc[cell_test_global].nunique()} donors")

# Protein fold splits — gene-grouped, RNA-FM-stratified (identical design to full model).
# All proteins mapping to the same transcript land in the same fold, so the model never
# sees a gene's RNA-FM embedding during training when that gene appears as a test protein.
gene_to_prot_idx = defaultdict(list)
for _i, (_, _gene) in enumerate(usable):
    gene_to_prot_idx[_gene].append(_i)

unique_genes   = sorted(gene_to_prot_idx.keys())
n_unique_genes = len(unique_genes)
n_strata       = min(8, max(2, n_unique_genes // 3))
gene_vecs      = np.array([rnafm_cache[g] for g in unique_genes], dtype=np.float32)
strata_labels  = KMeans(n_clusters=n_strata, random_state=RANDOM_STATE, n_init=10).fit_predict(gene_vecs)

prot_rng     = np.random.default_rng(RANDOM_STATE + 99)
gene_to_fold = {}
for _s in range(n_strata):
    _s_genes = [g for g, lbl in zip(unique_genes, strata_labels) if lbl == _s]
    _s_arr   = np.array(_s_genes)
    prot_rng.shuffle(_s_arr)
    for _j, _g in enumerate(_s_arr):
        gene_to_fold[_g] = int(_j % N_SPLITS)

protein_fold = np.array([gene_to_fold[gene] for _, gene in usable])
folds = [
    (np.where(protein_fold != fi)[0], np.where(protein_fold == fi)[0])
    for fi in range(N_SPLITS)
]

log.info(f"  Gene-grouped split: {n_unique_genes} unique transcripts, {n_strata} RNA-FM strata")
for fi in range(N_SPLITS):
    log.info(f"  Fold {fi}: {(protein_fold != fi).sum()} train / "
             f"{(protein_fold == fi).sum()} test proteins")

if not os.path.exists(FOLD_SPLIT_CSV):
    split_log = [
        {"fold": fi,
         "n_train_proteins": len(tr), "n_test_proteins": len(te),
         "train_proteins": "|".join(usable_array[tr, 0]),
         "test_proteins":  "|".join(usable_array[te, 0])}
        for fi, (tr, te) in enumerate(folds)
    ]
    pd.DataFrame(split_log).to_csv(FOLD_SPLIT_CSV, index=False)
    log.info(f"  Fold splits saved: {FOLD_SPLIT_CSV}")
else:
    log.info(f"  Fold splits loaded (split recomputed deterministically): {FOLD_SPLIT_CSV}")


# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def _get_feature_names(include):
    """Return ordered feature name list.
    protein_id and RNA_expr are always first; optional classes follow."""
    names = ["protein_id", "RNA_expr"]
    if "phagocytosis" in include:
        names.append("endocytosis")
    if "dc" in include:
        names.extend(dc_cols)
    if "pca" in include:
        names.extend(pc_cols)
    if "rnafm" in include:
        names.extend([f"RNAFM_{i}" for i in range(RNAFM_TOP)])
    return names


def _build_X(sub, gene, protein_id_val, include):
    """Build feature matrix for a single protein.
    protein_id and RNA_expr are always included.
    `include` is a frozenset of optional feature-class keys to add."""
    n = len(sub)
    rna_col = f"RNA_{gene}"
    parts = [
        np.full((n, 1), protein_id_val, dtype=np.float32),
        (sub[rna_col].values.reshape(-1, 1).astype(np.float32)
         if rna_col in sub.columns
         else np.zeros((n, 1), dtype=np.float32)),
    ]
    if "phagocytosis" in include:
        parts.append(sub["RRS_Endocytosis"].values.reshape(-1, 1).astype(np.float32))
    if "dc" in include:
        parts.append(sub[dc_cols].values.astype(np.float32))
    if "pca" in include:
        parts.append(sub[pc_cols].values.astype(np.float32))
    if "rnafm" in include:
        parts.append(np.tile(rnafm_cache[gene], (n, 1)))
    return np.concatenate(parts, axis=1)


def _stratified_sample(rng, valid_idx, ct_labels, n_target, min_per_ct):
    unique_cts, ct_counts = np.unique(ct_labels, return_counts=True)
    # Guarantee a minimum floor per cell type to preserve rare populations
    floor_alloc = np.minimum(ct_counts, min_per_ct)
    remaining   = max(n_target - floor_alloc.sum(), 0)
    prop_alloc  = np.zeros(len(unique_cts), dtype=np.int64)
    if remaining > 0:
        # Distribute the remaining budget proportionally to cell-type size
        w = ct_counts.astype(float) / ct_counts.sum()
        prop_alloc = np.floor(w * remaining).astype(np.int64)
    allocated = np.minimum(floor_alloc + prop_alloc, ct_counts)
    shortfall = n_target - allocated.sum()
    if shortfall > 0:
        # Fill any rounding shortfall by adding cells to types with most remaining headroom
        headroom = ct_counts - allocated
        for idx in np.argsort(headroom)[::-1]:
            add = min(int(shortfall), int(headroom[idx]))
            allocated[idx] += add; shortfall -= add
            if shortfall <= 0: break
    sampled = []
    for ct, n_alloc in zip(unique_cts, allocated):
        if n_alloc <= 0: continue
        ct_pos = np.where(ct_labels == ct)[0]
        n_draw = min(int(n_alloc), len(ct_pos))
        if n_draw > 0:
            sampled.append(valid_idx[rng.choice(ct_pos, n_draw, replace=False)])
    return np.concatenate(sampled) if sampled else valid_idx[:0]


def _group_metrics(y_true, y_pred):
    # Returns (R², Pearson r, n); NaN for groups too small to compute reliable metrics
    n = len(y_true)
    if n < 3: return float("nan"), float("nan"), n
    try:
        r2 = float(r2_score(y_true, y_pred))
        r, _ = pearsonr(y_true.astype(float), y_pred.astype(float))
        return r2, float(r), n
    except:
        return float("nan"), float("nan"), n


def _calibrate(y_true, y_pred_raw):
    """Per-protein OLS calibration. Returns (y_pred_cal, metrics_dict)."""
    # Skip calibration if predictions are constant (slope undefined) or too few cells
    if len(y_pred_raw) < 2 or np.ptp(y_pred_raw) == 0:
        return y_pred_raw.copy(), {
            "r2_raw": float("nan"), "r2_cal": float("nan"),
            "r_raw":  float("nan"), "r_cal":  float("nan"),
            "rmse_raw": float("nan"), "rmse_cal": float("nan"),
            "lm_slope": 1.0, "lm_intercept": 0.0,
            "n_cells_total":   len(y_true),
            "n_cells_nonzero": int((y_true >= DROPOUT_THR).sum()),
        }
    # Fit y_true ~ y_pred_raw; slope and intercept correct per-protein scale and offset
    res   = linregress(y_pred_raw.astype(float), y_true.astype(float))
    # Clip at 0: ADT abundance cannot be negative
    y_cal = np.clip((res.slope * y_pred_raw + res.intercept).astype(np.float32), 0, None)
    n_nz  = int((y_true >= DROPOUT_THR).sum())
    r2_raw, r_raw, _ = _group_metrics(y_true, y_pred_raw)
    r2_cal, r_cal, _ = _group_metrics(y_true, y_cal)
    return y_cal, {
        "r2_raw": r2_raw, "r2_cal": r2_cal,
        "r_raw":  float(r_raw), "r_cal": float(r_cal),
        "rmse_raw": float(np.sqrt(np.mean((y_true - y_pred_raw) ** 2))),
        "rmse_cal": float(np.sqrt(np.mean((y_true - y_cal) ** 2))),
        "lm_slope":      float(res.slope),
        "lm_intercept":  float(res.intercept),
        "n_cells_total":   int(len(y_true)),
        "n_cells_nonzero": n_nz,
    }


def _predict_proteins(model, pairs, protein_type, include, fold_i):
    """Predict all proteins in `pairs` on cell_test_global.

    protein_type: "unknown" → protein_id = NaN (OOF — transferable signal only)
                  "known"   → protein_id = real id (in-fold — antibody tag used)
    """
    pred_rows = []
    cal_rows  = []
    for feat, gene in pairs:
        pid_val = np.nan if protein_type == "unknown" else protein_id_map[feat]
        adt_col = f"ADT_{feat}"
        valid   = np.intersect1d(np.where(~df[adt_col].isna())[0], cell_test_global)
        if len(valid) == 0:
            continue
        sub    = df.iloc[valid]
        y_true = sub[adt_col].values.astype(np.float32)
        y_raw  = model.predict(
            xgb.DMatrix(_build_X(sub, gene, pid_val, include)),
            iteration_range=(0, model.best_iteration + 1))
        y_cal, cal_m = _calibrate(y_true, y_raw)
        cal_m.update({"protein": feat, "transcript": gene,
                      "fold": fold_i, "protein_type": protein_type})
        cal_rows.append(cal_m)
        pred_rows.append(pd.DataFrame({
            "y_true":       y_true,
            "y_pred_raw":   y_raw,
            "y_pred_cal":   y_cal,
            "protein":      feat,
            "transcript":   gene,
            "celltype":     sub["celltype"].values,
            "donor":        sub["donor"].values,
            "cell_id":      sub.index.values,
            "fold":         fold_i,
            "protein_type": protein_type,
        }))
    return pred_rows, cal_rows


def _protein_stats(data):
    # Compute per-protein summary: Pearson r, R², RMSE for both raw and calibrated predictions
    rows = []
    for prot, g in data.groupby("protein", observed=True):
        if len(g) < 3: continue
        y_t = g["y_true"].values.astype(float)
        row = {"protein": prot, "transcript": g["transcript"].iloc[0], "n_cells": len(g),
               "mean_true": float(np.mean(y_t)), "std_true": float(np.std(y_t, ddof=1))}
        for pt, pc in [("raw", "y_pred_raw"), ("cal", "y_pred_cal")]:
            y_p = g[pc].values.astype(float)
            r2, r, _ = _group_metrics(y_t, y_p)
            row[f"r_{pt}"]    = r
            row[f"r2_{pt}"]   = r2
            row[f"rmse_{pt}"] = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_stats(data):
    # Compute per-cell summary: Pearson r, R², Spearman r across all proteins in a cell
    # Requires ≥5 proteins per cell to produce a reliable correlation estimate
    rows = []
    for cid, g in data.groupby("cell_id", observed=True):
        if len(g) < 5: continue
        y_t = g["y_true"].values.astype(float)
        row = {"cell_id": cid, "celltype": g["celltype"].iloc[0],
               "donor": g["donor"].iloc[0], "n_proteins": len(g)}
        for pt, pc in [("raw", "y_pred_raw"), ("cal", "y_pred_cal")]:
            y_p = g[pc].values.astype(float)
            r2, r, _ = _group_metrics(y_t, y_p)
            sp_r, _  = _spearmanr(y_t, y_p)
            row[f"r_{pt}"]          = r
            row[f"r2_{pt}"]         = r2
            row[f"spearman_r_{pt}"] = float(sp_r)
        rows.append(row)
    return pd.DataFrame(rows)


def _avg_cells_per_protein(pairs, cell_subset=None):
    counts = []
    for feat, _gene in pairs:
        adt_col   = f"ADT_{feat}"
        valid_all = np.where(~df[adt_col].isna())[0]
        valid     = (np.intersect1d(valid_all, cell_subset)
                     if cell_subset is not None else valid_all)
        if len(valid) > 0:
            counts.append(len(valid))
    return int(np.median(counts)) if counts else TUNING_MAX_CELLS_PER_PROTEIN


# ── 4. ADTDataIter ─────────────────────────────────────────────────────────────
class ADTDataIter(xgb.core.DataIter):
    """Streams one protein at a time with a per-fold deterministic RNG so that
    cell subsampling is identical across conditions for the same fold."""

    def __init__(self, protein_pairs, cell_subset, include, rng):
        self.pairs       = protein_pairs
        self.cell_subset = cell_subset
        self.include     = include
        self.rng         = rng
        self._it         = 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        if self._it == len(self.pairs):
            return 0
        feat, gene = self.pairs[self._it]
        pid_val    = protein_id_map[feat]   # always real id during training
        adt_col    = f"ADT_{feat}"
        # Only use cells that have a non-NaN ADT measurement for this protein
        valid_all  = np.where(~df[adt_col].isna())[0]
        valid = (np.intersect1d(valid_all, self.cell_subset)
                 if self.cell_subset is not None else valid_all)
        if len(valid) > 0:
            # If a per-protein cell cap is set, subsample while preserving cell-type proportions
            if CELLS_PER_PROTEIN > 0 and len(valid) > CELLS_PER_PROTEIN:
                valid = _stratified_sample(
                    self.rng, valid,
                    df["celltype"].iloc[valid].values,
                    CELLS_PER_PROTEIN, MIN_CELLS_PER_CELLTYPE)
            sub = df.iloc[valid]
            input_data(data=_build_X(sub, gene, pid_val, self.include),
                       label=sub[adt_col].values.astype(np.float32))
        else:
            n_feat = len(_get_feature_names(self.include))
            input_data(data=np.zeros((0, n_feat), dtype=np.float32),
                       label=np.zeros(0, dtype=np.float32))
        self._it += 1
        return 1

    def reset(self):
        self._it = 0


# ── 5. MAIN LOOP ───────────────────────────────────────────────────────────────
xgb_params_base = {
    "objective":    "reg:squarederror",
    "tree_method":  "hist",
    "random_state": RANDOM_STATE,
    "n_jobs":       N_JOBS,
}
xgb_params_base.update(best_params)

for cond_name, include in CONDITIONS:

    # ── directories ─────────────────────────────────────────────────────────
    cond_dir  = os.path.join(ABLATION_BASE, cond_name)
    pred_dir  = os.path.join(cond_dir, "predictions")
    stat_dir  = os.path.join(cond_dir, "statistics")
    cal_dir   = os.path.join(cond_dir, "calibration")
    model_dir = os.path.join(cond_dir, "models")
    for _d in [cond_dir, pred_dir, stat_dir, cal_dir, model_dir]:
        os.makedirs(_d, exist_ok=True)

    shutil.copy2(FOLD_SPLIT_CSV, os.path.join(model_dir, "fold_protein_splits.csv"))

    # Per-condition file handler
    _fh = logging.FileHandler(os.path.join(cond_dir, "execution.log"), mode="w")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    log.addHandler(_fh)

    feat_names = _get_feature_names(include)
    log.info("=" * 70)
    log.info(f"CONDITION: {cond_name}")
    log.info(f"  Added feature classes  : {include or 'none (baseline)'}")
    log.info(f"  Active features        : {len(feat_names)}  "
             f"(protein_id=yes, RNA_expr=yes, "
             f"endocytosis={'yes' if 'phagocytosis' in include else 'no'}, "
             f"DC={len(dc_cols) if 'dc' in include else 0}, "
             f"PCA={len(pc_cols) if 'pca' in include else 0}, "
             f"RNAFM={RNAFM_TOP if 'rnafm' in include else 0})")
    log.info("=" * 70)

    # ── checkpoint ──────────────────────────────────────────────────────────
    ckpt_file            = os.path.join(cond_dir, "cv_checkpoint.pkl")
    all_cv_preds_unknown = []
    all_cv_preds_known   = []
    all_cal_metrics      = []
    best_iters           = []
    start_fold           = 0

    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)
        start_fold           = ckpt["completed_folds"]
        all_cv_preds_unknown = ckpt["all_cv_preds_unknown"]
        all_cv_preds_known   = ckpt["all_cv_preds_known"]
        all_cal_metrics      = ckpt["all_cal_metrics"]
        best_iters           = ckpt["best_iters"]
        log.info(f"  Resuming from fold {start_fold + 1}/{N_SPLITS}")

    xgb_params = xgb_params_base.copy()

    # ── CV folds ────────────────────────────────────────────────────────────
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if fold_i < start_fold:
            continue

        tr_pairs = usable_array[train_idx]
        te_pairs = usable_array[test_idx]

        # Deterministic per-fold RNG seeds ensure that cell subsampling within a fold
        # is identical across all conditions, so performance differences reflect features only.
        iter_rng = np.random.default_rng(RANDOM_STATE + fold_i * 1000)
        val_rng  = np.random.default_rng(RANDOM_STATE + fold_i * 1000 + 999)

        effective_cells = _avg_cells_per_protein(tr_pairs, cell_train_global)
        if CELLS_PER_PROTEIN > 0:
            effective_cells = min(effective_cells, CELLS_PER_PROTEIN)
        scaled_mcw = max(
            1, round(best_params["min_child_weight"] * effective_cells / TUNING_MAX_CELLS_PER_PROTEIN))
        xgb_params["min_child_weight"] = scaled_mcw

        log.info(f"  Fold {fold_i + 1}/{N_SPLITS} — "
                 f"{len(tr_pairs)} train proteins / {len(te_pairs)} test proteins  |  "
                 f"min_child_weight: {best_params['min_child_weight']} "
                 f"-> {scaled_mcw} (eff. {effective_cells:,} cells)")

        # Training matrix
        it_train = ADTDataIter(tr_pairs, cell_train_global, include, iter_rng)
        dtrain   = xgb.QuantileDMatrix(it_train)

        # Validation matrix: 100 cells per training protein (early stopping)
        X_val_list, y_val_list = [], []
        for feat, gene in tr_pairs:
            adt_col = f"ADT_{feat}"
            valid   = np.intersect1d(np.where(~df[adt_col].isna())[0], cell_train_global)
            if len(valid) == 0:
                continue
            v_idx = val_rng.choice(valid, min(100, len(valid)), replace=False)
            sub   = df.iloc[v_idx]
            X_val_list.append(_build_X(sub, gene, protein_id_map[feat], include))
            y_val_list.append(sub[adt_col].values.astype(np.float32))
        dval = xgb.DMatrix(np.concatenate(X_val_list), label=np.concatenate(y_val_list))

        model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=5000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        log.info(f"    Best iteration: {model.best_iteration}")
        best_iters.append(model.best_iteration)
        model.save_model(os.path.join(model_dir, f"model_fold_{fold_i}.json"))

        # OOF unknown proteins: held out from training; protein_id = NaN.
        # Tests whether the feature class helps generalise to proteins never seen during training.
        oof_rows, oof_cal = _predict_proteins(model, te_pairs, "unknown", include, fold_i)
        all_cv_preds_unknown.extend(oof_rows)
        all_cal_metrics.extend(oof_cal)

        # In-fold known proteins: were in training; protein_id = real id.
        # Tests performance when antibody-specific biases are available.
        known_rows, known_cal = _predict_proteins(model, tr_pairs, "known", include, fold_i)
        all_cv_preds_known.extend(known_rows)
        all_cal_metrics.extend(known_cal)

        log.info(f"    OOF rows: {sum(len(r) for r in oof_rows):,}  |  "
                 f"Known rows: {sum(len(r) for r in known_rows):,}")

        with open(ckpt_file, "wb") as f:
            pickle.dump({
                "completed_folds":       fold_i + 1,
                "all_cv_preds_unknown":  all_cv_preds_unknown,
                "all_cv_preds_known":    all_cv_preds_known,
                "all_cal_metrics":       all_cal_metrics,
                "best_iters":            best_iters,
            }, f)

        del it_train, dtrain, dval, model; gc.collect()

    # ── Save predictions ─────────────────────────────────────────────────────
    log.info(f"  Saving predictions for {cond_name}...")

    cv_oof = pd.concat(all_cv_preds_unknown, ignore_index=True)
    for col in ["protein", "transcript", "celltype", "donor"]:
        cv_oof[col] = cv_oof[col].astype("category")
    cv_oof.to_parquet(os.path.join(pred_dir, "oof_unknown_proteins.parquet"), index=False)
    log.info(f"  OOF unknown — {len(cv_oof):,} rows  |  "
             f"{cv_oof['protein'].nunique()} proteins  |  "
             f"{cv_oof['cell_id'].nunique():,} cells")

    cv_known_perfold = pd.concat(all_cv_preds_known, ignore_index=True)
    for col in ["protein", "transcript", "celltype", "donor"]:
        cv_known_perfold[col] = cv_known_perfold[col].astype("category")
    cv_known_perfold.to_parquet(
        os.path.join(pred_dir, "infold_known_proteins_perfold.parquet"), index=False)

    cv_known_avg = (cv_known_perfold
                    .groupby(["cell_id", "protein", "celltype", "transcript", "donor"],
                              observed=True)
                    [["y_true", "y_pred_raw", "y_pred_cal"]].mean()
                    .reset_index())
    cv_known_avg.to_parquet(
        os.path.join(pred_dir, "infold_known_proteins_averaged.parquet"), index=False)
    log.info(f"  In-fold known — {len(cv_known_perfold):,} per-fold rows  →  "
             f"{len(cv_known_avg):,} averaged  |  "
             f"{cv_known_perfold['protein'].nunique()} proteins")

    pd.DataFrame(all_cal_metrics).to_csv(
        os.path.join(cal_dir, "lm_params_per_protein_per_fold.csv"), index=False)

    # ── Statistics ───────────────────────────────────────────────────────────
    log.info(f"  Computing statistics for {cond_name}...")

    ps_oof = _protein_stats(cv_oof)
    ps_oof.to_csv(os.path.join(stat_dir, "oof_unknown_per_protein.csv"), index=False)
    cs_oof = _cell_stats(cv_oof)
    cs_oof.to_csv(os.path.join(stat_dir, "oof_unknown_per_cell.csv"), index=False)
    log.info(f"  OOF per-cell  — n={len(cs_oof):,}  |  "
             f"median r_raw={cs_oof['r_raw'].median():.4f}  "
             f"median r_cal={cs_oof['r_cal'].median():.4f}  "
             f"median spearman_cal={cs_oof['spearman_r_cal'].median():.4f}")

    ps_known = _protein_stats(cv_known_avg)
    ps_known.to_csv(os.path.join(stat_dir, "infold_known_per_protein.csv"), index=False)
    cs_known = _cell_stats(cv_known_avg)
    cs_known.to_csv(os.path.join(stat_dir, "infold_known_per_cell.csv"), index=False)
    log.info(f"  Known per-cell — n={len(cs_known):,}  |  "
             f"median r_raw={cs_known['r_raw'].median():.4f}  "
             f"median r_cal={cs_known['r_cal'].median():.4f}  "
             f"median spearman_cal={cs_known['spearman_r_cal'].median():.4f}")

    del cv_oof, cv_known_perfold, cv_known_avg; gc.collect()

    log.removeHandler(_fh); _fh.close()


# ── 6. SUMMARY ACROSS CONDITIONS ──────────────────────────────────────────────
# Aggregate per-cell Pearson r, Spearman r, and R² across all conditions.
# The summary CSV and bar-charts show how much each feature class improves
# OOF unknown protein prediction relative to the protein_id + RNA_expr baseline.
log.info("=" * 70)
log.info("Generating summary...")

summary_rows = []
for cond_name, _ in CONDITIONS:
    for stat_file, ptype in [
        ("oof_unknown_per_cell.csv",  "unknown"),
        ("infold_known_per_cell.csv", "known"),
    ]:
        path = os.path.join(ABLATION_BASE, cond_name, "statistics", stat_file)
        if not os.path.exists(path):
            log.warning(f"  Missing: {path}")
            continue
        cs = pd.read_csv(path)
        summary_rows.append({
            "condition":             cond_name,
            "protein_type":          ptype,
            "n_cells":               len(cs),
            "median_r_raw":          round(float(cs["r_raw"].median()), 4),
            "median_r_cal":          round(float(cs["r_cal"].median()), 4),
            "mean_r_cal":            round(float(cs["r_cal"].mean()), 4),
            "median_r2_cal":         round(float(cs["r2_cal"].median()), 4),
            "median_spearman_r_cal": round(float(cs["spearman_r_cal"].median()), 4),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(ABLATION_BASE, "ablation_summary.csv"), index=False)
log.info(f"\n{summary_df.to_string(index=False)}")


def _summary_barplot(df_sub, ptype_label, out_path):
    conditions = df_sub["condition"].tolist()
    colors     = ["#4C9BE8" if c == "baseline" else "#E8834C" for c in conditions]
    metrics    = [
        ("median_r_raw",          "Median Pearson r (raw)"),
        ("median_r_cal",          "Median Pearson r (calibrated)"),
        ("median_spearman_r_cal", "Median Spearman r (calibrated)"),
    ]
    baseline_r = (float(df_sub.loc[df_sub["condition"] == "baseline", "median_r_cal"].iloc[0])
                  if "baseline" in conditions else None)
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5 * len(metrics), max(4, len(conditions) * 0.6 + 1)))
    for ax, (col, label) in zip(axes, metrics):
        vals = df_sub[col].tolist()
        bars = ax.barh(conditions, vals, color=colors, edgecolor="white", height=0.6)
        if baseline_r is not None and col == "median_r_cal":
            ax.axvline(baseline_r, color="navy", linestyle="--", linewidth=1.2,
                       label=f"Baseline={baseline_r:.4f}")
            ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.invert_yaxis()
    fig.suptitle(f"GENESPLIT Additive Feature Study — {ptype_label} (per-cell metrics)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if not summary_df.empty:
    for ptype, label, fname in [
        ("unknown", "OOF Unknown Proteins",   "ablation_summary_unknown.png"),
        ("known",   "In-Fold Known Proteins", "ablation_summary_known.png"),
    ]:
        sub = summary_df[summary_df["protein_type"] == ptype].copy()
        if not sub.empty:
            _summary_barplot(sub, label, os.path.join(ABLATION_BASE, fname))
            log.info(f"  Summary plot saved: {fname}")

log.info("=" * 70)
log.info("ADDITIVE FEATURE STUDY COMPLETE")
log.info(f"Output: {ABLATION_BASE}")
log.info("=" * 70)
