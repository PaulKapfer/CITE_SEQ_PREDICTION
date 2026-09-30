#!/usr/bin/env python3
"""
PER_CELLTYPE_GENESPLIT.py
=========================
Runs the GENESPLIT_no_celltype_proteinid pipeline independently within each of
three major cell types (Monocytes, CD4 T cells, CD8 T cells).

Purpose
-------
Tests whether the model relies on cell-type identity to predict protein
abundances. By training and evaluating exclusively within one cell type at a
time, we isolate how well the model generalises using only intra-type
transcriptional variation.

Cell-type membership is read from cell_type_assignments.csv
(barcode -> celltype_final) and matched to the reference parquet by its
'barcode' column.

Cross-validation design (identical to the full-data run)
---------------------------------------------------------
  - Cells: fixed 20% test hold-out by donor (GroupShuffleSplit).
  - Proteins: 5-fold KFold.
  - Every (test_cell, test_protein) pair is predicted by a model that saw
    neither that cell nor that protein during training.
  - Unknown proteins (OOF): protein_id = NaN.
  - Known proteins (in-fold): protein_id = real integer.

Output layout
-------------
BASE_OUT_DIR/
  Monocytes/
    execution.log
    predictions/  statistics/  calibration/  models/  plots/
  CD4_T_cells/
    ...
  CD8_T_cells/
    ...
"""

import os, sys, logging, pickle, gc
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
REF_PARQUET  = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Reference_Preparation/Output/reference_data.parquet"
MAPPING_CSV  = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Matching_ADT_Transcript/Output2/combined_adt_mapping_MODEL-TRAINING.csv"
RNAFM_DIR    = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Create_RNA_FM_features/Output/rnafm_features"
CELLTYPE_CSV = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/Output/4.Annotation+validation/cell_type_assignments.csv"
BASE_OUT_DIR = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Per-Celltype-Test/Output"

# ── CELL-TYPE TOGGLES ─────────────────────────────────────────────────────────
RUN_MONOCYTES  = True   # toggle Monocytes run on/off
RUN_CD4_T      = True   # toggle CD4 T cells run on/off
RUN_CD8_T      = True   # toggle CD8 T cells run on/off

CELL_TYPES_TO_RUN = (
    (["Monocytes"]    if RUN_MONOCYTES else []) +
    (["CD4 T cells"]  if RUN_CD4_T     else []) +
    (["CD8 T cells"]  if RUN_CD8_T     else [])
)

TRAIN_FINAL_MODEL = True   # set False to skip steps 7-8 and save time

best_params = {
    "max_depth": 8,
    "learning_rate": 0.06161472706880244,
    "subsample": 0.7123043816958816,
    "colsample_bytree": 0.6108327809165853,
    "min_child_weight": 256,
    "reg_alpha": 0.040377333089617114,
    "reg_lambda": 0.00014728652782921222,
    "gamma": 0.2203501322530219,
}

N_SPLITS                     = 5
CELLS_PER_PROTEIN            = 50_000
TUNING_MAX_CELLS_PER_PROTEIN = 20_000
RNAFM_TOP                    = 640
N_PCS                        = 50
DROPOUT_THR                  = 0.5
RANDOM_STATE                 = 42
N_JOBS                       = 15

os.makedirs(BASE_OUT_DIR, exist_ok=True)

# ── HELPERS (stateless — no df reference) ─────────────────────────────────────
rng_sub = np.random.default_rng(RANDOM_STATE)


def _group_metrics(y_true, y_pred):
    n = len(y_true)
    if n < 3:
        return float("nan"), float("nan"), n
    try:
        r2 = float(r2_score(y_true, y_pred))
        r, _ = pearsonr(y_true.astype(float), y_pred.astype(float))
        return r2, float(r), n
    except:
        return float("nan"), float("nan"), n


def calibrate_predictions(y_true, y_pred_raw):
    if len(y_pred_raw) < 2 or np.ptp(y_pred_raw) == 0:
        return y_pred_raw, {
            "r2_raw": float("nan"), "r2_cal": float("nan"),
            "r_raw":  float("nan"), "r_cal":  float("nan"),
            "rmse_raw": float("nan"), "rmse_cal": float("nan"),
            "lm_slope": 1.0, "lm_intercept": 0.0,
            "n_cells_total":   len(y_true),
            "n_cells_nonzero": int((y_true >= DROPOUT_THR).sum()),
        }
    res        = linregress(y_pred_raw.astype(float), y_true.astype(float))
    y_pred_cal = np.clip((res.slope * y_pred_raw + res.intercept).astype(np.float32), 0, None)
    n_nonzero  = int((y_true >= DROPOUT_THR).sum())
    if len(y_true) >= 3:
        r2_raw   = float(r2_score(y_true, y_pred_raw))
        r2_cal   = float(r2_score(y_true, y_pred_cal))
        rmse_raw = float(np.sqrt(mean_squared_error(y_true, y_pred_raw)))
        rmse_cal = float(np.sqrt(mean_squared_error(y_true, y_pred_cal)))
        r_raw, _ = pearsonr(y_true.astype(float), y_pred_raw.astype(float))
        r_cal, _ = pearsonr(y_true.astype(float), y_pred_cal.astype(float))
    else:
        r2_raw = r2_cal = rmse_raw = rmse_cal = r_raw = r_cal = float("nan")
    return y_pred_cal, {
        "r2_raw": r2_raw,   "r2_cal": r2_cal,
        "r_raw":  float(r_raw), "r_cal": float(r_cal),
        "rmse_raw": rmse_raw, "rmse_cal": rmse_cal,
        "lm_slope":      float(res.slope),
        "lm_intercept":  float(res.intercept),
        "n_cells_total":   int(len(y_true)),
        "n_cells_nonzero": n_nonzero,
    }


def _scatter_plot(y_true, y_pred, title, out_path, max_pts=100_000):
    if len(y_true) > max_pts:
        idx = np.random.choice(len(y_true), max_pts, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
    r2, r, n = _group_metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=1, alpha=0.2, rasterized=True, color="steelblue")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True ADT (log-normalised)"); ax.set_ylabel("Predicted ADT")
    ax.set_title(f"{title}\nR²={r2:.4f}  r={r:.4f}  n={n:,}")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _cell_boxplot(cell_df, metric_col, title, out_path):
    if cell_df.empty:
        return
    ct_medians = cell_df.groupby("celltype", observed=True)[metric_col].median().sort_values()
    data   = [cell_df[cell_df["celltype"] == ct][metric_col].dropna().values for ct in ct_medians.index]
    labels = [l for l, d in zip(ct_medians.index, data) if len(d) > 0]
    data   = [d for d in data if len(d) > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    if data:
        bp = ax.boxplot(data, tick_labels=labels, vert=False, patch_artist=True)
        for p in bp["boxes"]: p.set_facecolor("lightblue"); p.set_alpha(0.7)
        for m in bp["medians"]: m.set_color("red"); m.set_linewidth(2)
    overall_med = cell_df[metric_col].median()
    overall_mn  = cell_df[metric_col].mean()
    ax.set_title(f"{title}\nMedian={overall_med:.4f}  Mean={overall_mn:.4f}  n={len(cell_df):,} cells")
    ax.set_xlabel(metric_col); ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ── LOAD SHARED DATA (ONCE) ───────────────────────────────────────────────────
print("Loading shared data (parquet, mapping, RNA-FM)...")

df_full  = pd.read_parquet(REF_PARQUET)
mapping  = pd.read_csv(MAPPING_CSV)
ct_asgn  = pd.read_csv(CELLTYPE_CSV, index_col=0)   # index = barcode

rnafm_raw = {fn[:-4]: np.load(os.path.join(RNAFM_DIR, fn)).astype(np.float32)
             for fn in os.listdir(RNAFM_DIR) if fn.endswith(".npy")}
all_vecs    = np.stack(list(rnafm_raw.values()))
top_dims    = np.argsort(all_vecs.var(axis=0))[::-1][:RNAFM_TOP]
rnafm_cache = {g: v[top_dims] for g, v in rnafm_raw.items()}
del rnafm_raw, all_vecs

# Protein-gene pairs and feature layout derived from the full parquet once
protein_to_gene = dict(zip(mapping["ADT_feature"], mapping["RNA_gene"]))
# A missing RNA_<gene> column is not a reason to drop the antibody: the gene was
# simply never detected in the scRNA-seq matrix, so its log-normalised expression
# is 0 in every cell (_build_X substitutes a zero column). The ADT measurement is
# still real and the protein remains predictable from RNA-FM and cell state.
usable_full = [
    (p, g) for p, g in protein_to_gene.items()
    if f"ADT_{p}" in df_full.columns and g in rnafm_cache
]
_no_rna = sorted({g for _, g in usable_full if f"RNA_{g}" not in df_full.columns})
usable_sorted  = sorted(usable_full, key=lambda x: x[0])
protein_id_map = {feat: float(i) for i, (feat, _) in enumerate(usable_sorted)}

dc_cols = sorted([c for c in df_full.columns if c.startswith("DC") and c[2:].isdigit()],
                 key=lambda x: int(x[2:]))
pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1) if f"PC{i}" in df_full.columns]

feature_names = (["protein_id", "RNA_expr", "endocytosis"]
                 + dc_cols + pc_cols + [f"RNAFM_{i}" for i in range(RNAFM_TOP)])

print(f"  Usable proteins: {len(usable_full)}")
if _no_rna:
    print(f"  Genes undetected in scRNA-seq (RNA_expr = 0): {len(_no_rna)}  {_no_rna}")
print(f"  Full parquet: {len(df_full):,} cells")
print(f"  Cell type assignments loaded: {len(ct_asgn):,} barcodes")
print()


# ── PER-CELL-TYPE LOOP ────────────────────────────────────────────────────────
for CELLTYPE_NAME in CELL_TYPES_TO_RUN:

    ct_slug = CELLTYPE_NAME.replace(" ", "_").replace("/", "_")
    OUT_DIR = os.path.join(BASE_OUT_DIR, ct_slug)

    PRED_DIR  = os.path.join(OUT_DIR, "predictions")
    STAT_DIR  = os.path.join(OUT_DIR, "statistics")
    CAL_DIR   = os.path.join(OUT_DIR, "calibration")
    MODEL_DIR = os.path.join(OUT_DIR, "models")
    PLOT_DIR  = os.path.join(OUT_DIR, "plots")
    for _d in [OUT_DIR, PRED_DIR, STAT_DIR, CAL_DIR, MODEL_DIR, PLOT_DIR,
               os.path.join(PLOT_DIR, "filtered"),
               os.path.join(PLOT_DIR, "unfiltered"),
               os.path.join(PLOT_DIR, "known_vs_unknown"),
               os.path.join(PLOT_DIR, "fold_consistency")]:
        os.makedirs(_d, exist_ok=True)

    # Per-celltype log file — reset root logger handlers each iteration
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        h.close(); root_logger.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(OUT_DIR, "execution.log"), mode="w"),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info(f"CELL TYPE: {CELLTYPE_NAME}")
    log.info(f"Output:    {OUT_DIR}")
    log.info("=" * 60)

    # ── FILTER CELLS FOR THIS CELL TYPE ───────────────────────────────────────
    log.info(f"1. Filtering reference data to {CELLTYPE_NAME}...")
    ct_barcodes = set(ct_asgn[ct_asgn["celltype_final"] == CELLTYPE_NAME].index.astype(str))

    # df is module-level; ADTDataIter and _build_X reference it by name
    df = df_full[df_full["barcode"].isin(ct_barcodes)].copy().reset_index(drop=True)

    if len(df) == 0:
        log.error(f"  No cells found for {CELLTYPE_NAME} — skipping.")
        continue

    # Restrict usable proteins to those that have at least one non-NaN ADT value
    # in this cell-type subset (same proteins as full run in practice, but safeguards
    # against proteins assayed only on cell types not included here)
    usable = [
        (p, g) for p, g in usable_full
        if df[f"ADT_{p}"].notna().any()
    ]
    usable_array = np.array(usable)

    log.info(f"  Cells after filter: {len(df):,}  |  donors: {df['donor'].nunique()}")
    log.info(f"  Usable proteins with non-NaN ADT in subset: {len(usable)}")

    # ── FEATURE BUILDER ───────────────────────────────────────────────────────
    def _build_X(sub, gene, protein_id_val=np.nan):
        n = len(sub)
        rna_col = f"RNA_{gene}"
        rna_expr = (sub[rna_col].values.reshape(-1, 1).astype(np.float32)
                    if rna_col in sub.columns
                    else np.zeros((n, 1), dtype=np.float32))
        return np.concatenate([
            np.full((n, 1), protein_id_val, dtype=np.float32),
            rna_expr,
            sub["RRS_Endocytosis"].values.reshape(-1, 1).astype(np.float32),
            sub[dc_cols].values.astype(np.float32),
            sub[pc_cols].values.astype(np.float32),
            np.tile(rnafm_cache[gene], (n, 1)),
        ], axis=1)

    # ── CELL SPLIT ────────────────────────────────────────────────────────────
    log.info("2. Splitting cells: fixed 20% test hold-out by donor...")
    all_cell_idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    cell_train_global, cell_test_global = next(
        gss.split(all_cell_idx, groups=df["donor"].values))
    log.info(f"  Train: {len(cell_train_global):,} cells / "
             f"{df['donor'].iloc[cell_train_global].nunique()} donors  |  "
             f"Test: {len(cell_test_global):,} cells / "
             f"{df['donor'].iloc[cell_test_global].nunique()} donors")

    # ── DATA ITERATOR ─────────────────────────────────────────────────────────
    class ADTDataIter(xgb.core.DataIter):
        def __init__(self, protein_pairs, cell_subset=None):
            self.pairs = protein_pairs; self.cell_subset = cell_subset; self._it = 0
            super().__init__(cache_prefix=None)

        def next(self, input_data):
            if self._it == len(self.pairs):
                return 0
            feat, gene = self.pairs[self._it]
            pid_val    = protein_id_map[feat]
            adt_col    = f"ADT_{feat}"
            valid_all  = np.where(~df[adt_col].isna())[0]
            valid      = (np.intersect1d(valid_all, self.cell_subset)
                          if self.cell_subset is not None else valid_all)
            if len(valid) > 0:
                sub = df.iloc[valid]
                input_data(data=_build_X(sub, gene, pid_val),
                           label=sub[adt_col].values.astype(np.float32))
            else:
                input_data(data=np.zeros((0, len(feature_names)), dtype=np.float32),
                           label=np.zeros(0, dtype=np.float32))
            self._it += 1
            return 1

        def reset(self):
            self._it = 0

    def _cells_per_protein(pairs, cell_subset=None):
        counts = []
        for feat, gene in pairs:
            adt_col   = f"ADT_{feat}"
            valid_all = np.where(~df[adt_col].isna())[0]
            valid     = (np.intersect1d(valid_all, cell_subset)
                         if cell_subset is not None else valid_all)
            if len(valid) > 0:
                counts.append(len(valid))
        return counts if counts else [TUNING_MAX_CELLS_PER_PROTEIN]

    def _avg_cells_per_protein(pairs, cell_subset=None):
        return int(np.median(_cells_per_protein(pairs, cell_subset)))

    _preview_counts = _cells_per_protein(usable, cell_train_global)
    _preview_cells  = int(np.median(_preview_counts))
    _preview_mcw    = max(1, round(best_params["min_child_weight"]
                                   * _preview_cells / TUNING_MAX_CELLS_PER_PROTEIN))
    log.info(f"  Cells/protein in training — min: {min(_preview_counts):,}  "
             f"median: {_preview_cells:,}  max: {max(_preview_counts):,}  "
             f"(across {len(_preview_counts)} proteins)")
    log.info(f"  min_child_weight preview: {best_params['min_child_weight']} "
             f"-> {_preview_mcw} (scaled @ {_preview_cells:,} cells/protein, "
             f"{_preview_cells / TUNING_MAX_CELLS_PER_PROTEIN:.2f}x)")

    # ── CROSS-VALIDATION ──────────────────────────────────────────────────────
    log.info(f"3. Starting {N_SPLITS}-fold CV "
             f"(gene-grouped, RNA-FM-stratified protein split)...")

    # Group proteins by transcript so proteins sharing a gene always land in the
    # same fold — prevents RNA-FM embedding leakage between train and test sets.
    _gene_to_prot_idx = defaultdict(list)
    for _i, (_, _gene) in enumerate(usable):
        _gene_to_prot_idx[_gene].append(_i)
    _unique_genes  = sorted(_gene_to_prot_idx.keys())
    _n_genes       = len(_unique_genes)

    # K-means on RNA-FM gene embeddings stratifies the fold assignment so that
    # each fold's test set spans diverse transcript space rather than being
    # concentrated in one region of sequence/structure space.
    _n_strata      = min(8, max(2, _n_genes // 3))
    _gene_vecs     = np.array([rnafm_cache[g] for g in _unique_genes], dtype=np.float32)
    _kmeans        = KMeans(n_clusters=_n_strata, random_state=RANDOM_STATE, n_init=10)
    _strata_labels = _kmeans.fit_predict(_gene_vecs)
    _gene_stratum  = dict(zip(_unique_genes, _strata_labels))

    # Round-robin assignment of genes to folds within each stratum.
    _prot_rng     = np.random.default_rng(RANDOM_STATE + 99)
    _gene_to_fold = {}
    for _s in range(_n_strata):
        _s_genes = [g for g, lbl in zip(_unique_genes, _strata_labels) if lbl == _s]
        _s_arr   = np.array(_s_genes)
        _prot_rng.shuffle(_s_arr)
        for _j, _g in enumerate(_s_arr):
            _gene_to_fold[_g] = int(_j % N_SPLITS)

    _protein_fold = np.array([_gene_to_fold[gene] for _, gene in usable])
    folds = [
        (np.where(_protein_fold != fi)[0], np.where(_protein_fold == fi)[0])
        for fi in range(N_SPLITS)
    ]

    log.info(f"  Unique transcripts: {_n_genes}  |  RNA-FM strata: {_n_strata}")
    for _fi in range(N_SPLITS):
        log.info(f"  Fold {_fi}: {(~(_protein_fold == _fi)).sum()} train / "
                 f"{(_protein_fold == _fi).sum()} test proteins")

    split_log = [{"fold": fi,
                  "n_train_proteins": len(tr), "n_test_proteins": len(te),
                  "train_proteins": "|".join(usable_array[tr, 0]),
                  "test_proteins":  "|".join(usable_array[te, 0])}
                 for fi, (tr, te) in enumerate(folds)]
    pd.DataFrame(split_log).to_csv(
        os.path.join(MODEL_DIR, "fold_protein_splits.csv"), index=False)

    pd.DataFrame({
        "protein": [p for p, _ in usable],
        "gene":    [g for _, g in usable],
        "fold":    _protein_fold.tolist(),
        "stratum": [int(_gene_stratum[g]) for _, g in usable],
    }).to_csv(os.path.join(MODEL_DIR, "protein_fold_assignment.csv"), index=False)

    CHECKPOINT_FILE = os.path.join(OUT_DIR, "cv_checkpoint.pkl")
    all_cv_preds, all_cv_preds_known, all_calibration_metrics, best_iters = [], [], [], []
    start_fold = 0

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        start_fold              = ckpt["completed_folds"]
        all_cv_preds            = ckpt["all_cv_preds"]
        all_cv_preds_known      = ckpt.get("all_cv_preds_known", [])
        best_iters              = ckpt["best_iters"]
        all_calibration_metrics = ckpt.get("all_calibration_metrics", [])
        log.info(f"  Resuming from fold {start_fold + 1}")

    xgb_params = {"objective": "reg:squarederror", "tree_method": "hist",
                  "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    xgb_params.update(best_params)

    for fold_i, (train_pair_idx, test_pair_idx) in enumerate(folds):
        if fold_i < start_fold:
            continue
        tr_pairs = usable_array[train_pair_idx]
        te_pairs = usable_array[test_pair_idx]
        log.info(f"  Fold {fold_i+1}/{N_SPLITS} — "
                 f"{len(tr_pairs)} train proteins / {len(te_pairs)} test proteins  |  "
                 f"train cells: {len(cell_train_global):,}  test cells: {len(cell_test_global):,}")

        it_train = ADTDataIter(tr_pairs, cell_subset=cell_train_global)
        dtrain   = xgb.QuantileDMatrix(it_train)

        X_val_list, y_val_list = [], []
        for feat, gene in tr_pairs:
            adt_col = f"ADT_{feat}"
            valid   = np.intersect1d(np.where(~df[adt_col].isna())[0], cell_train_global)
            if len(valid) == 0:
                continue
            v_idx = rng_sub.choice(valid, min(100, len(valid)), replace=False)
            sub   = df.iloc[v_idx]
            X_val_list.append(_build_X(sub, gene, protein_id_map[feat]))
            y_val_list.append(sub[adt_col].values.astype(np.float32))
        dval = xgb.DMatrix(np.concatenate(X_val_list), label=np.concatenate(y_val_list))

        current_params    = xgb_params.copy()
        effective_cells   = _avg_cells_per_protein(tr_pairs, cell_train_global)
        current_params["min_child_weight"] = max(
            1, round(best_params["min_child_weight"]
                     * effective_cells / TUNING_MAX_CELLS_PER_PROTEIN))
        log.info(f"    min_child_weight: {best_params['min_child_weight']} "
                 f"-> {current_params['min_child_weight']} "
                 f"(scaled @ {effective_cells:,} cells/protein, "
                 f"{effective_cells / TUNING_MAX_CELLS_PER_PROTEIN:.2f}x)")

        model = xgb.train(current_params, dtrain, num_boost_round=5000,
                          evals=[(dtrain, "train"), (dval, "val")],
                          early_stopping_rounds=100, verbose_eval=False)
        log.info(f"    Best iteration: {model.best_iteration}")
        best_iters.append(model.best_iteration)
        model.save_model(os.path.join(MODEL_DIR, f"model_fold_{fold_i}.json"))

        def _predict_proteins(pairs, label):
            rows = []
            for feat, gene in pairs:
                pid_val = np.nan if label == "unknown" else protein_id_map[feat]
                adt_col = f"ADT_{feat}"
                valid   = np.intersect1d(np.where(~df[adt_col].isna())[0], cell_test_global)
                if len(valid) == 0:
                    continue
                sub    = df.iloc[valid]
                y_true = sub[adt_col].values.astype(np.float32)
                y_raw  = model.predict(
                    xgb.DMatrix(_build_X(sub, gene, pid_val)),
                    iteration_range=(0, model.best_iteration + 1))
                y_cal, cal_m = calibrate_predictions(y_true, y_raw)
                if label == "unknown":
                    cal_m["protein"]    = feat
                    cal_m["transcript"] = gene
                    cal_m["fold"]       = fold_i
                    all_calibration_metrics.append(cal_m)
                rows.append(pd.DataFrame({
                    "y_true":       y_true,
                    "y_pred_raw":   y_raw,
                    "y_pred_cal":   y_cal,
                    "protein":      feat,
                    "transcript":   gene,
                    "celltype":     sub["celltype"].values,
                    "donor":        sub["donor"].values,
                    "cell_id":      sub.index.values,
                    "fold":         fold_i,
                    "protein_type": label,
                }))
            return rows

        all_cv_preds.extend(_predict_proteins(te_pairs, "unknown"))
        all_cv_preds_known.extend(_predict_proteins(tr_pairs, "known"))

        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump({"completed_folds":      fold_i + 1,
                         "all_cv_preds":          all_cv_preds,
                         "all_cv_preds_known":    all_cv_preds_known,
                         "best_iters":            best_iters,
                         "all_calibration_metrics": all_calibration_metrics}, f)
        del it_train, dtrain, dval, model
        gc.collect()

    # ── SAVE PREDICTIONS ──────────────────────────────────────────────────────
    log.info("4. Saving predictions...")

    cv_res = pd.concat(all_cv_preds, ignore_index=True)
    for col in ["protein", "transcript", "celltype", "donor"]:
        cv_res[col] = cv_res[col].astype("category")
    cv_res.to_parquet(os.path.join(PRED_DIR, "oof_unknown_proteins.parquet"), index=False)
    log.info(f"  OOF unknown: {len(cv_res):,} rows  |  "
             f"{cv_res['protein'].nunique()} proteins  |  "
             f"{cv_res['cell_id'].nunique():,} cells")

    cv_known_perfold = pd.concat(all_cv_preds_known, ignore_index=True)
    for col in ["protein", "transcript", "celltype", "donor"]:
        cv_known_perfold[col] = cv_known_perfold[col].astype("category")
    cv_known_perfold.to_parquet(
        os.path.join(PRED_DIR, "infold_known_proteins_perfold.parquet"), index=False)

    cv_known = (cv_known_perfold
                .groupby(["cell_id", "protein", "celltype", "transcript", "donor"], observed=True)
                [["y_true", "y_pred_raw", "y_pred_cal"]].mean().reset_index())
    cv_known.to_parquet(
        os.path.join(PRED_DIR, "infold_known_proteins_averaged.parquet"), index=False)
    log.info(f"  In-fold known: {len(cv_known_perfold):,} per-fold rows  →  "
             f"{len(cv_known):,} fold-averaged  |  {cv_known['protein'].nunique()} proteins")

    pd.DataFrame(all_calibration_metrics).to_csv(
        os.path.join(CAL_DIR, "oof_lm_params_per_protein_per_fold.csv"), index=False)

    # ── STATISTICS ────────────────────────────────────────────────────────────
    log.info("5. Computing statistics...")

    def _protein_stats(data):
        rows = []
        for prot, g in data.groupby("protein", observed=True):
            if len(g) < 3:
                continue
            y_t = g["y_true"].values.astype(float)
            row = {"protein": prot, "transcript": g["transcript"].iloc[0],
                   "n_cells": len(g),
                   "mean_true": float(np.mean(y_t)),
                   "std_true":  float(np.std(y_t, ddof=1))}
            for pt, pc in [("raw", "y_pred_raw"), ("cal", "y_pred_cal")]:
                y_p = g[pc].values.astype(float)
                r2, r, _ = _group_metrics(y_t, y_p)
                row[f"r_{pt}"]         = r
                row[f"r2_{pt}"]        = r2
                row[f"rmse_{pt}"]      = float(np.sqrt(np.mean((y_t - y_p)**2)))
                row[f"mean_pred_{pt}"] = float(np.mean(y_p))
                row[f"std_pred_{pt}"]  = float(np.std(y_p, ddof=1))
            rows.append(row)
        return pd.DataFrame(rows)

    def _cell_stats(data):
        rows = []
        for cid, g in data.groupby("cell_id", observed=True):
            if len(g) < 5:
                continue
            y_t = g["y_true"].values.astype(float)
            row = {"cell_id": cid, "celltype": g["celltype"].iloc[0],
                   "donor": g["donor"].iloc[0], "n_proteins": len(g)}
            for pt, pc in [("raw", "y_pred_raw"), ("cal", "y_pred_cal")]:
                y_p = g[pc].values.astype(float)
                r2, r, _ = _group_metrics(y_t, y_p)
                sp_r, _  = _spearmanr(y_t, y_p)
                row[f"r_{pt}"]         = r
                row[f"r2_{pt}"]        = r2
                row[f"spearman_r_{pt}"] = float(sp_r)
            rows.append(row)
        return pd.DataFrame(rows)

    _protein_stats(cv_res).to_csv(
        os.path.join(STAT_DIR, "oof_unknown_per_protein.csv"), index=False)
    cell_stats_oof = _cell_stats(cv_res)
    cell_stats_oof.to_csv(os.path.join(STAT_DIR, "oof_unknown_per_cell.csv"), index=False)
    log.info(f"  OOF unknown per-cell: {len(cell_stats_oof):,} cells  |  "
             f"median r_cal={cell_stats_oof['r_cal'].median():.4f}")

    cell_fold_rows = []
    for (cid, fold_i), g in cv_res.groupby(["cell_id", "fold"], observed=True):
        if len(g) < 3:
            continue
        y_t = g["y_true"].values.astype(float)
        row = {"cell_id": cid, "fold": fold_i, "celltype": g["celltype"].iloc[0],
               "donor": g["donor"].iloc[0], "n_proteins": len(g)}
        for pt, pc in [("raw", "y_pred_raw"), ("cal", "y_pred_cal")]:
            y_p = g[pc].values.astype(float)
            r2, r, _ = _group_metrics(y_t, y_p)
            sp_r, _  = _spearmanr(y_t, y_p)
            row[f"r_{pt}"] = r; row[f"r2_{pt}"] = r2; row[f"spearman_r_{pt}"] = float(sp_r)
        cell_fold_rows.append(row)
    cell_fold_df = pd.DataFrame(cell_fold_rows)
    cell_fold_df.to_csv(
        os.path.join(STAT_DIR, "oof_unknown_per_cell_per_fold.csv"), index=False)

    _protein_stats(cv_known).to_csv(
        os.path.join(STAT_DIR, "infold_known_per_protein.csv"), index=False)
    cell_stats_known = _cell_stats(cv_known)
    cell_stats_known.to_csv(os.path.join(STAT_DIR, "infold_known_per_cell.csv"), index=False)
    log.info(f"  In-fold known per-cell: {len(cell_stats_known):,} cells  |  "
             f"median r_cal={cell_stats_known['r_cal'].median():.4f}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    log.info("6. Generating plots...")

    def _make_plots(data, tag):
        tag_dir = os.path.join(PLOT_DIR, tag)
        for p_type in ["raw", "cal"]:
            p_col  = f"y_pred_{p_type}"
            label  = "Raw (no calibration)" if p_type == "raw" else "Calibrated (post-hoc LM)"
            _scatter_plot(data["y_true"].values, data[p_col].values,
                          f"OOF Unknown Proteins — {label} [{tag}] ({CELLTYPE_NAME})",
                          os.path.join(tag_dir, f"oof_unknown_scatter_{p_type}.png"))
            cell_m = []
            for cid, g in data.groupby("cell_id", observed=True):
                if len(g) < 5:
                    continue
                r2, r, _ = _group_metrics(g["y_true"], g[p_col])
                cell_m.append({"cell_id": cid, "celltype": g["celltype"].iloc[0],
                                "r2": r2, "r": r})
            cdf = pd.DataFrame(cell_m)
            if not cdf.empty:
                _cell_boxplot(cdf, "r",
                              f"Pearson r per Cell (unknown proteins) — {label} [{tag}] ({CELLTYPE_NAME})",
                              os.path.join(tag_dir, f"oof_unknown_percell_pearsonr_{p_type}.png"))
                _cell_boxplot(cdf, "r2",
                              f"R² per Cell (unknown proteins) — {label} [{tag}] ({CELLTYPE_NAME})",
                              os.path.join(tag_dir, f"oof_unknown_percell_r2_{p_type}.png"))

    _make_plots(cv_res[cv_res["y_true"] >= DROPOUT_THR], "filtered")
    _make_plots(cv_res, "unfiltered")

    kvu_dir = os.path.join(PLOT_DIR, "known_vs_unknown")
    if not cell_stats_oof.empty and not cell_stats_known.empty:
        for metric, label in [
            ("r_cal",          "Pearson r (Calibrated)"),
            ("r_raw",          "Pearson r (Raw)"),
            ("spearman_r_cal", "Spearman r (Calibrated)"),
        ]:
            d_known   = cell_stats_known[metric].dropna().values
            d_unknown = cell_stats_oof[metric].dropna().values
            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot([d_known, d_unknown],
                            tick_labels=["Known proteins\n(in-fold, novel cells)",
                                         "Unknown proteins\n(OOF, novel cells)"],
                            patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], ["#4C9BE8", "#E8834C"]):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            for m in bp["medians"]: m.set_color("black"); m.set_linewidth(2)
            ax.set_title(
                f"GENESPLIT ({CELLTYPE_NAME}) — {label} per Cell\n"
                f"Known: median={np.nanmedian(d_known):.4f} (n={len(d_known):,})  |  "
                f"Unknown: median={np.nanmedian(d_unknown):.4f} (n={len(d_unknown):,})"
            )
            ax.set_ylabel(metric); ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            fig.savefig(os.path.join(kvu_dir, f"percell_{metric}_comparison.png"), dpi=150)
            plt.close(fig)

        cv_known_filt = cv_known[cv_known["y_true"] >= DROPOUT_THR]
        for p_type in ["raw", "cal"]:
            label = "Raw" if p_type == "raw" else "Calibrated"
            _scatter_plot(cv_known_filt["y_true"].values,
                          cv_known_filt[f"y_pred_{p_type}"].values,
                          f"In-Fold Known Proteins — {label} [filtered] ({CELLTYPE_NAME})",
                          os.path.join(kvu_dir, f"overall_scatter_known_{p_type}.png"))

    fc_dir = os.path.join(PLOT_DIR, "fold_consistency")
    if not cell_fold_df.empty:
        for metric, label in [
            ("r_cal",          "Pearson r (Calibrated)"),
            ("r_raw",          "Pearson r (Raw)"),
            ("spearman_r_cal", "Spearman r (Calibrated)"),
        ]:
            folds_present = sorted(cell_fold_df["fold"].unique())
            fold_data     = [cell_fold_df.loc[cell_fold_df["fold"] == f, metric].dropna().values
                             for f in folds_present]
            fold_data     = [d for d in fold_data if len(d) > 0]
            fold_labels   = [f"Fold {f}\n({len(usable_array[folds[f][1]])} proteins)"
                             for f in folds_present
                             if len(cell_fold_df.loc[cell_fold_df["fold"] == f, metric].dropna()) > 0]
            fig, ax = plt.subplots(figsize=(9, 5))
            bp = ax.boxplot(fold_data, tick_labels=fold_labels, patch_artist=True)
            for p in bp["boxes"]: p.set_facecolor("lightblue"); p.set_alpha(0.7)
            for m in bp["medians"]: m.set_color("red"); m.set_linewidth(2)
            fold_medians = [np.nanmedian(d) for d in fold_data]
            overall_med  = np.nanmedian(cell_fold_df[metric].values)
            ax.set_title(
                f"OOF Unknown Proteins — {label} per Cell by Fold ({CELLTYPE_NAME})\n"
                f"Overall median={overall_med:.4f}  "
                f"fold range=[{min(fold_medians):.4f}, {max(fold_medians):.4f}]"
            )
            ax.set_xlabel("Fold — each fold tests a different ~20% protein subset")
            ax.set_ylabel(metric); ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            fig.savefig(os.path.join(fc_dir, f"oof_unknown_percell_{metric}_by_fold.png"), dpi=150)
            plt.close(fig)

    # ── FINAL MODEL ───────────────────────────────────────────────────────────
    if TRAIN_FINAL_MODEL:
        FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.json")
        if os.path.exists(FINAL_MODEL_PATH):
            log.info("7. Loading final model from checkpoint...")
            final_model = xgb.Booster()
            final_model.load_model(FINAL_MODEL_PATH)
        else:
            log.info("7. Training final model on all proteins × all cells (this cell type)...")
            avg_best = int(np.mean(best_iters)) if best_iters else 1000
            log.info(f"  Rounds: {avg_best}")
            it_final    = ADTDataIter(usable, cell_subset=None)
            dfinal      = xgb.QuantileDMatrix(it_final)
            final_params = xgb_params.copy()
            effective_cells = _avg_cells_per_protein(usable, cell_subset=None)
            final_params["min_child_weight"] = max(
                1, round(best_params["min_child_weight"]
                         * effective_cells / TUNING_MAX_CELLS_PER_PROTEIN))
            log.info(f"  min_child_weight: {best_params['min_child_weight']} "
                     f"-> {final_params['min_child_weight']} "
                     f"(scaled @ {effective_cells:,} cells/protein)")
            final_model = xgb.train(final_params, dfinal,
                                    num_boost_round=avg_best, verbose_eval=100)
            final_model.save_model(FINAL_MODEL_PATH)
            del it_final, dfinal; gc.collect()

        log.info("8. Computing final model per-protein calibration parameters...")
        final_cal_rows = []
        for feat, gene in usable:
            adt_col = f"ADT_{feat}"
            valid   = np.where(~df[adt_col].isna())[0]
            if len(valid) == 0:
                continue
            sub        = df.iloc[valid]
            y_true     = sub[adt_col].values.astype(np.float32)
            y_pred_raw = final_model.predict(
                xgb.DMatrix(_build_X(sub, gene, protein_id_map[feat])))
            y_pred_cal, cal_m = calibrate_predictions(y_true, y_pred_raw)
            cal_m["protein"]    = feat
            cal_m["transcript"] = gene
            final_cal_rows.append(cal_m)
        final_cal_df = pd.DataFrame(final_cal_rows)
        final_cal_df.to_csv(
            os.path.join(CAL_DIR, "final_model_lm_params.csv"), index=False)
        log.info(f"  Final model LM params: {len(final_cal_df)} proteins  |  "
                 f"median r_raw={final_cal_df['r_raw'].median():.4f}  "
                 f"median r_cal={final_cal_df['r_cal'].median():.4f}")

    log.info("=" * 60)
    log.info(f"COMPLETE: {CELLTYPE_NAME}")
    log.info(f"Output:   {OUT_DIR}")
    log.info("=" * 60)

    # Release cell-type-specific memory before next iteration
    del df, cv_res, cv_known_perfold, cv_known, cell_stats_oof, cell_stats_known
    del all_cv_preds, all_cv_preds_known, all_calibration_metrics, best_iters
    gc.collect()

print("\nAll cell types complete.")
