#!/usr/bin/env python3
"""
Hyperparameter Tuning — XGBoost Protein Abundance Prediction (GENESPLIT_no_celltype)
======================================================================================
Model: RNA_expr + Endocytosis + All DCs + PC1-50 + RNA-FM (top-640 dims)
       NO celltype, NO protein_id, NO transcript_id features.

Design
------
* Data          : reference_data.parquet
* Cell split    : donor-based GroupShuffleSplit — fixed 80% train / 20% test
                  The 20% test set is held out and NEVER used during tuning.
* Tuning split  : donor-based GroupKFold (N_FOLDS) within the 80% train cells.
                  Each fold's validation set is a held-out donor group from
                  the train pool — uncoupled from the 20% global test set.
* Protein stack : all usable proteins stacked per fold
                  (train cells capped at MAX_CELLS_PER_PROTEIN via stratified
                  celltype sampling, matching GENESPLIT_no_celltype.py)
* Optimizer     : Optuna TPE sampler + MedianPruner
                  Seeded with current best params from GENESPLIT_no_celltype.py
                  as trial 0 for warm-starting the search.
* Pruning       : XGBoostPruningCallback on fold-0 validation curve;
                  if not pruned → remaining folds evaluated with early stopping
* Objective     : mean MSE across all N_FOLDS folds
* Metrics saved : R² and Pearson r (dropout-filtered, ≥ DROPOUT_THR)
                  per fold and averaged for every completed trial

Outputs (OUT_DIR)
-----------------
  hyperparameter_trials.csv     — per-trial R², r, RMSE per fold + averages
  best_hyperparameters.json     — best params from Optuna study
  best_params_fold_metrics.csv  — final N_FOLDS evaluation with best params
  summary.json                  — overall summary
  hyperparameter_tuning.log

Dependencies
------------
  pip install xgboost optuna optuna-integration pandas numpy scipy scikit-learn
"""

import os
import sys
import json
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import xgboost as xgb
import optuna

try:
    from optuna.integration import XGBoostPruningCallback
except ImportError:
    from optuna_integration.xgboost import XGBoostPruningCallback

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
REF_PARQUET = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Hyperparameter_tuning\Input\reference_data.parquet"
MAPPING_CSV = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Hyperparameter_tuning\Input\adt_rna_mapping.csv"
RNAFM_DIR   = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Hyperparameter_tuning\Input\rnafm_features"
OUT_DIR     = r"C:\Users\Paul\Desktop\Publications\CITE-SEQ_pred\Hyperparameter_tuning\Output"

N_FOLDS               = 3        # donor-based folds within the 80% train pool
N_TRIALS              = 50       # Optuna trials
RNAFM_TOP             = 640      # top RNA-FM dimensions (matches GENESPLIT_no_celltype)
N_PCS                 = 50       # PC1..PC50
DROPOUT_THR           = 0.5      # CLR abundance below this = dropout (excluded from metrics)
MAX_CELLS_PER_PROTEIN = 20_000   # cap cells per protein in train stacking (tuning speed)
MIN_CELLS_PER_CELLTYPE = 10      # minimum cells per celltype for stratified sampling
RANDOM_STATE          = 42
N_JOBS                = 8        # XGBoost parallelism

# Starting point for the Optuna sweep — These starting params are extracted from an exploratory sweep of a similar model that did not make it into the final project. These starting params should not be relied on for model training.
SEED_PARAMS = {
    "max_depth":        6,
    "learning_rate":    0.08810003129071789,
    "subsample":        0.5998368910791798,
    "colsample_bytree": 0.538522716492931,
    "min_child_weight": 60,
    "gamma":            0.23225206359998862,
    "reg_alpha":        0.10907475835157694,
    "reg_lambda":       0.0007122305833333872,
}

# ── SETUP ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "hyperparameter_tuning.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)
rng_sub = np.random.default_rng(RANDOM_STATE + 1)   # for per-protein stratified subsampling


# ── HELPERS ────────────────────────────────────────────────────────────────────
def _stratified_sample(valid_idx, ct_labels, n_target, min_per_ct):
    """Stratified sample of valid_idx by ct_labels, matching GENESPLIT_no_celltype."""
    unique_cts, ct_counts = np.unique(ct_labels, return_counts=True)
    floor_alloc = np.minimum(ct_counts, min_per_ct)
    remaining   = max(n_target - floor_alloc.sum(), 0)
    prop_alloc  = np.zeros(len(unique_cts), dtype=np.int64)
    if remaining > 0:
        w = ct_counts.astype(float) / ct_counts.sum()
        prop_alloc = np.floor(w * remaining).astype(np.int64)
    allocated = np.minimum(floor_alloc + prop_alloc, ct_counts)
    shortfall = n_target - allocated.sum()
    if shortfall > 0:
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
            sampled.append(valid_idx[rng_sub.choice(ct_pos, n_draw, replace=False)])
    return np.concatenate(sampled) if sampled else valid_idx[:0]


# ── 1. LOAD REFERENCE DATA ─────────────────────────────────────────────────────
log.info("=" * 65)
log.info("1. Loading reference data ...")
df = pd.read_parquet(REF_PARQUET)
log.info(f"   Shape : {df.shape[0]:,} cells × {df.shape[1]} columns")
log.info(f"   Donors: {df['donor'].nunique()} unique  | "
         f"Celltypes: {df['celltype'].nunique()} unique")

dc_cols = sorted(
    [c for c in df.columns if c.startswith("DC") and c[2:].isdigit()],
    key=lambda x: int(x[2:]),
)
pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1) if f"PC{i}" in df.columns]

log.info(f"   DC columns : {len(dc_cols)}  ({dc_cols[0]}–{dc_cols[-1]})")
log.info(f"   PC columns : {len(pc_cols)}  ({pc_cols[0]}–{pc_cols[-1]})")


# ── 2. LOAD RNA-FM EMBEDDINGS ─────────────────────────────────────────────────
log.info("\n2. Loading RNA-FM embeddings ...")
rnafm_raw: dict = {}
for fname in os.listdir(RNAFM_DIR):
    if fname.endswith(".npy"):
        gene = fname[:-4]
        rnafm_raw[gene] = np.load(os.path.join(RNAFM_DIR, fname)).astype(np.float32)

if not rnafm_raw:
    log.error("   No RNA-FM embeddings found — aborting.")
    sys.exit(1)

raw_dim = next(iter(rnafm_raw.values())).shape[0]
log.info(f"   {len(rnafm_raw)} embeddings loaded  (dim = {raw_dim})")

all_vecs = np.stack(list(rnafm_raw.values()))
top_dims = np.argsort(all_vecs.var(axis=0))[::-1][:RNAFM_TOP]
rnafm_cache: dict = {g: v[top_dims] for g, v in rnafm_raw.items()}
del rnafm_raw, all_vecs
log.info(f"   Reduced to top-{RNAFM_TOP} most variable dimensions")


# ── 3. IDENTIFY USABLE PROTEINS ────────────────────────────────────────────────
log.info("\n3. Identifying usable proteins ...")
mapping         = pd.read_csv(MAPPING_CSV)
protein_to_gene = dict(zip(mapping["ADT_feature"], mapping["RNA_gene"]))

usable = []
for feat, gene in protein_to_gene.items():
    if f"ADT_{feat}" not in df.columns:
        continue
    if f"RNA_{gene}" not in df.columns:
        continue
    if gene not in rnafm_cache:
        continue
    usable.append((feat, gene))
log.info(f"   {len(usable)} proteins with ADT + RNA + RNA-FM")
if not usable:
    log.error("   No usable proteins — check mapping CSV and column names.")
    sys.exit(1)


# ── 4. PREPARE CELL-LEVEL ARRAYS ──────────────────────────────────────────────
log.info("\n4. Preparing cell-level arrays ...")

n_dc   = len(dc_cols)
n_pc   = len(pc_cols)
n_feat = 1 + 1 + n_dc + n_pc + RNAFM_TOP
log.info(f"   Feature vector size: {n_feat}")
log.info(f"     RNA_expr(1) + endocytosis(1) + DC({n_dc}) + PC({n_pc}) + RNA-FM({RNAFM_TOP})")

dc_mat       = df[dc_cols].values.astype(np.float32)
pc_mat       = df[pc_cols].values.astype(np.float32)
endo_col_arr = df["RRS_Endocytosis"].values.astype(np.float32)
donors_arr   = df["donor"].values
celltype_arr = df["celltype"].values


def _build_X_idx(valid_idx, gene):
    """Build feature matrix for the given cell indices and gene."""
    n = len(valid_idx)
    return np.concatenate([
        df[f"RNA_{gene}"].values[valid_idx].reshape(-1, 1).astype(np.float32),
        endo_col_arr[valid_idx].reshape(-1, 1),
        dc_mat[valid_idx],
        pc_mat[valid_idx],
        np.tile(rnafm_cache[gene], (n, 1)),
    ], axis=1, dtype=np.float32)


# ── 5. DONOR-BASED 80/20 CELL SPLIT ──────────────────────────────────────────
log.info("\n5. Splitting cells: fixed 80% train / 20% test by donor ...")
all_cell_idx = np.arange(len(df))
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
cell_train_global, cell_test_global = next(
    gss.split(all_cell_idx, groups=donors_arr)
)
log.info(f"   Train: {len(cell_train_global):,} cells / "
         f"{len(np.unique(donors_arr[cell_train_global]))} donors  |  "
         f"Test (held out): {len(cell_test_global):,} cells / "
         f"{len(np.unique(donors_arr[cell_test_global]))} donors")


# ── 6. DONOR-BASED GROUPKFOLD WITHIN THE 80% TRAIN POOL ──────────────────────
log.info(f"\n6. Creating {N_FOLDS}-fold donor-based GroupKFold splits "
         f"within the 80% train pool ...")

donors_train = donors_arr[cell_train_global]
gkf   = GroupKFold(n_splits=N_FOLDS)
# Indices here are relative to cell_train_global
folds_rel = list(gkf.split(cell_train_global, groups=donors_train))

for i, (tr_rel, te_rel) in enumerate(folds_rel):
    n_tr_donors = len(np.unique(donors_train[tr_rel]))
    n_te_donors = len(np.unique(donors_train[te_rel]))
    log.info(f"   Fold {i}: train={len(tr_rel):,} rows / {n_tr_donors} donors  |  "
             f"val={len(te_rel):,} rows / {n_te_donors} donors")


# ── 7. BUILD PER-FOLD STACKED FEATURE MATRICES ────────────────────────────────
log.info(f"\n7. Building per-fold stacked matrices "
         f"(train capped at {MAX_CELLS_PER_PROTEIN:,} cells/protein) ...")

fold_arrays = []
for fold_i, (tr_rel, te_rel) in enumerate(folds_rel):
    # Absolute cell indices for this fold's train and validation sets
    fold_train_cells = cell_train_global[tr_rel]
    fold_val_cells   = cell_train_global[te_rel]

    X_tr_parts, y_tr_parts = [], []
    X_te_parts, y_te_parts = [], []

    for feat, gene in usable:
        adt_col = f"ADT_{feat}"
        y_prot  = df[adt_col].values.astype(np.float32)
        valid_mask = ~np.isnan(y_prot)

        # Train block — stratified subsample within fold train cells
        valid_tr = np.intersect1d(np.where(valid_mask)[0], fold_train_cells)
        if len(valid_tr) < 10:
            continue
        if len(valid_tr) > MAX_CELLS_PER_PROTEIN:
            valid_tr = _stratified_sample(
                valid_tr, celltype_arr[valid_tr],
                MAX_CELLS_PER_PROTEIN, MIN_CELLS_PER_CELLTYPE,
            )
        X_tr_parts.append(_build_X_idx(valid_tr, gene))
        y_tr_parts.append(y_prot[valid_tr])

        # Validation block — all fold val cells, no cap
        valid_te = np.intersect1d(np.where(valid_mask)[0], fold_val_cells)
        if len(valid_te) < 3:
            continue
        X_te_parts.append(_build_X_idx(valid_te, gene))
        y_te_parts.append(y_prot[valid_te])

    X_tr = np.concatenate(X_tr_parts, axis=0)
    y_tr = np.concatenate(y_tr_parts, axis=0).astype(np.float32)
    X_te = np.concatenate(X_te_parts, axis=0)
    y_te = np.concatenate(y_te_parts, axis=0).astype(np.float32)

    fold_arrays.append({
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "keep": y_te >= DROPOUT_THR,
    })
    log.info(f"   Fold {fold_i}: X_tr={X_tr.shape}  X_te={X_te.shape}"
             f"  (dropout-pass: {(y_te >= DROPOUT_THR).sum():,}/{len(y_te):,})")
    del X_tr_parts, y_tr_parts, X_te_parts, y_te_parts


# ── 8. OPTUNA HYPERPARAMETER SEARCH ───────────────────────────────────────────
log.info(f"\n8. Starting Optuna search ({N_TRIALS} trials) ...")


def objective(trial):
    """
    Train on N_FOLDS donor-based folds within the 80% train pool.
    Prune early using fold-0 validation curve; remaining folds use early stopping.
    Objective = mean MSE across all N_FOLDS folds.
    R² / r / RMSE stored as user attrs (dropout-filtered ≥ DROPOUT_THR).
    """
    params = {
        "verbosity":        0,
        "objective":        "reg:squarederror",
        "tree_method":      "hist",
        "n_jobs":           N_JOBS,
        "random_state":     RANDOM_STATE,
        "n_estimators":     1000,
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 300),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 100.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 10.0),
    }

    fold_mses = []
    models    = []

    for fold_i, fa in enumerate(fold_arrays):
        fit_kwargs = dict(
            eval_set=[(fa["X_te"], fa["y_te"])],
            verbose=False,
        )

        if fold_i == 0:
            # Fold 0: pruning callback — may raise TrialPruned
            pruning_cb = XGBoostPruningCallback(trial, "validation_0-rmse")
            model = xgb.XGBRegressor(**params, callbacks=[pruning_cb])
            model.fit(fa["X_tr"], fa["y_tr"], **fit_kwargs)
        else:
            # Remaining folds: early stopping only
            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(fa["X_tr"], fa["y_tr"], **fit_kwargs)

        y_pred = model.predict(fa["X_te"]).astype(np.float32)
        fold_mses.append(float(mean_squared_error(fa["y_te"], y_pred)))
        models.append((model, y_pred))

    # Per-fold dropout-filtered metrics
    for fold_i, (fa, (_, y_pred)) in enumerate(zip(fold_arrays, models)):
        keep = fa["keep"]
        if keep.sum() >= 3:
            r2   = float(r2_score(fa["y_te"][keep], y_pred[keep]))
            rmse = float(np.sqrt(mean_squared_error(fa["y_te"][keep], y_pred[keep])))
            r, _ = pearsonr(fa["y_te"][keep].astype(float), y_pred[keep].astype(float))
            r    = float(r)
        else:
            r2, rmse, r = float("nan"), float("nan"), float("nan")

        trial.set_user_attr(f"fold{fold_i}_r2",   r2)
        trial.set_user_attr(f"fold{fold_i}_r",     r)
        trial.set_user_attr(f"fold{fold_i}_rmse",  rmse)

    avg_r2   = float(np.nanmean([trial.user_attrs[f"fold{i}_r2"]   for i in range(N_FOLDS)]))
    avg_r    = float(np.nanmean([trial.user_attrs[f"fold{i}_r"]    for i in range(N_FOLDS)]))
    avg_rmse = float(np.nanmean([trial.user_attrs[f"fold{i}_rmse"] for i in range(N_FOLDS)]))
    trial.set_user_attr("avg_r2",   avg_r2)
    trial.set_user_attr("avg_r",    avg_r)
    trial.set_user_attr("avg_rmse", avg_rmse)

    return float(np.mean(fold_mses))   # minimise mean MSE


def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log a one-line summary after every trial."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        is_best    = trial.number == study.best_trial.number
        tag        = " *** BEST ***" if is_best else ""
        log.info(
            f"  Trial {trial.number:>3}  "
            f"avg_r2={trial.user_attrs.get('avg_r2', float('nan')):+.4f}  "
            f"avg_r={trial.user_attrs.get('avg_r', float('nan')):.4f}  "
            f"avg_rmse={trial.user_attrs.get('avg_rmse', float('nan')):.4f}  "
            f"| depth={trial.params.get('max_depth')}  "
            f"lr={trial.params.get('learning_rate'):.4f}  "
            f"sub={trial.params.get('subsample'):.2f}  "
            f"col={trial.params.get('colsample_bytree'):.2f}  "
            f"mcw={trial.params.get('min_child_weight')}  "
            f"gamma={trial.params.get('gamma'):.3f}  "
            f"alpha={trial.params.get('reg_alpha'):.4f}  "
            f"lambda={trial.params.get('reg_lambda'):.4f}"
            f"  [done={n_complete} pruned={n_pruned}]{tag}"
        )
    elif trial.state == optuna.trial.TrialState.PRUNED:
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        log.info(f"  Trial {trial.number:>3}  PRUNED  [pruned={n_pruned}]")


study = optuna.create_study(
    direction = "minimize",
    sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner    = optuna.pruners.MedianPruner(n_warmup_steps=20),
)
# Warm-start: enqueue current best params as trial 0
study.enqueue_trial(SEED_PARAMS)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
               callbacks=[_trial_callback])


# ── 9. COLLECT TRIAL RESULTS ──────────────────────────────────────────────────
log.info("\n9. Collecting trial results ...")

trial_rows = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    row = {
        "trial_number": t.number,
        "mean_mse":     t.value,
        "avg_r2":       t.user_attrs.get("avg_r2",   float("nan")),
        "avg_r":        t.user_attrs.get("avg_r",    float("nan")),
        "avg_rmse":     t.user_attrs.get("avg_rmse", float("nan")),
    }
    for fold_i in range(N_FOLDS):
        row[f"fold{fold_i}_r2"]   = t.user_attrs.get(f"fold{fold_i}_r2",   float("nan"))
        row[f"fold{fold_i}_r"]    = t.user_attrs.get(f"fold{fold_i}_r",    float("nan"))
        row[f"fold{fold_i}_rmse"] = t.user_attrs.get(f"fold{fold_i}_rmse", float("nan"))
    row.update(t.params)
    trial_rows.append(row)

trials_df  = pd.DataFrame(trial_rows)
n_complete = len(trials_df)
n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
log.info(f"   Completed : {n_complete}  |  Pruned : {n_pruned}  |  Total : {N_TRIALS}")

if n_complete > 0:
    best_row = trials_df.loc[trials_df["mean_mse"].idxmin()]
    log.info(f"   Best trial  #{int(best_row['trial_number'])}"
             f"  avg_r2={best_row['avg_r2']:.4f}"
             f"  avg_r={best_row['avg_r']:.4f}"
             f"  avg_rmse={best_row['avg_rmse']:.4f}")
    log.info(f"   Best params : {study.best_params}")


# ── 10. FINAL N_FOLDS EVALUATION WITH BEST PARAMS ────────────────────────────
log.info(f"\n10. Final {N_FOLDS}-fold evaluation with best hyperparameters ...")

best_params_final = dict(
    verbosity             = 0,
    objective             = "reg:squarederror",
    tree_method           = "hist",
    n_jobs                = N_JOBS,
    random_state          = RANDOM_STATE,
    n_estimators          = 1000,
    early_stopping_rounds = 50,
    **study.best_params,
)

final_fold_rows = []
for fold_i, fa in enumerate(fold_arrays):
    model = xgb.XGBRegressor(**best_params_final)
    model.fit(fa["X_tr"], fa["y_tr"],
              eval_set=[(fa["X_te"], fa["y_te"])],
              verbose=False)
    y_pred = model.predict(fa["X_te"]).astype(np.float32)
    keep   = fa["keep"]

    if keep.sum() >= 3:
        r2   = float(r2_score(fa["y_te"][keep], y_pred[keep]))
        rmse = float(np.sqrt(mean_squared_error(fa["y_te"][keep], y_pred[keep])))
        r, _ = pearsonr(fa["y_te"][keep].astype(float), y_pred[keep].astype(float))
        r    = float(r)
    else:
        r2, rmse, r = float("nan"), float("nan"), float("nan")

    final_fold_rows.append({"fold": fold_i, "r2": r2, "r": r, "rmse": rmse})
    log.info(f"   Fold {fold_i}: R²={r2:.4f}  r={r:.4f}  RMSE={rmse:.4f}"
             f"  (best_iter={model.best_iteration})")

final_df = pd.DataFrame(final_fold_rows)
log.info(f"\n   Average across folds (best params):")
log.info(f"     R²   = {final_df['r2'].mean():.4f} ± {final_df['r2'].std():.4f}")
log.info(f"     r    = {final_df['r'].mean():.4f} ± {final_df['r'].std():.4f}")
log.info(f"     RMSE = {final_df['rmse'].mean():.4f} ± {final_df['rmse'].std():.4f}")


# ── 11. SAVE ──────────────────────────────────────────────────────────────────
log.info("\n11. Saving outputs ...")

trials_csv = os.path.join(OUT_DIR, "hyperparameter_trials.csv")
trials_df.to_csv(trials_csv, index=False)
log.info(f"   → {trials_csv}  ({len(trials_df)} rows)")

fold_csv = os.path.join(OUT_DIR, "best_params_fold_metrics.csv")
final_df.to_csv(fold_csv, index=False)
log.info(f"   → {fold_csv}")

params_json = os.path.join(OUT_DIR, "best_hyperparameters.json")
with open(params_json, "w") as fh:
    json.dump(study.best_params, fh, indent=2)
log.info(f"   → {params_json}")

summary = {
    "n_cells_total":          len(df),
    "n_cells_train":          len(cell_train_global),
    "n_cells_test_held_out":  len(cell_test_global),
    "max_cells_per_protein":  MAX_CELLS_PER_PROTEIN,
    "n_proteins":             len(usable),
    "n_folds":                N_FOLDS,
    "n_trials_requested":     N_TRIALS,
    "n_trials_completed":     n_complete,
    "n_trials_pruned":        n_pruned,
    "n_dc_features":          n_dc,
    "n_pc_features":          n_pc,
    "n_rnafm_features":       RNAFM_TOP,
    "total_features":         n_feat,
    "seed_params":            SEED_PARAMS,
    "best_params":            study.best_params,
    "final_avg_r2":           float(final_df["r2"].mean()),
    "final_std_r2":           float(final_df["r2"].std()),
    "final_avg_r":            float(final_df["r"].mean()),
    "final_std_r":            float(final_df["r"].std()),
    "final_avg_rmse":         float(final_df["rmse"].mean()),
    "final_std_rmse":         float(final_df["rmse"].std()),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
log.info(f"   → summary.json")

log.info("\nDone.")
