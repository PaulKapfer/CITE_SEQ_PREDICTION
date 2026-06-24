"""
External_Validation_Package – annotate_unknown

The single prediction step of the external-validation pipeline: every protein
in the external dataset is predicted with protein_id = NaN, so the model relies
only on transferable (protein-agnostic) signal — appropriate because antibody
clones differ between datasets. RNA-FM embeddings are read from the Unknown/
directory.

Predictions are raw XGBoost outputs clipped at 0 (no per-protein calibration is
applied here; calibration, if any, happens downstream at evaluation/plotting).

Adds to adata.obsm:
  ADT_pred_unknown – (n_cells × n_proteins) DataFrame, raw predictions ≥ 0

Usage
-----
    External_Validation_Package.annotate_unknown(adata)
"""
import numpy as np
import pandas as pd


def annotate_unknown(adata) -> None:
    """
    Predict protein abundances for unknown proteins and annotate adata.obsm.

    Parameters
    ----------
    adata : AnnData prepared with prepare_anndata()
    """
    import xgboost as xgb

    from ._data import load_model, load_protein_quality, load_mapping_second_dataset
    from ._predict import (
        _get_log_expr, build_rnafm_cache_unknown, build_feature_matrix,
        get_unknown_protein_pairs,
    )

    _check_prepared(adata)

    # ── Load reference data ───────────────────────────────────────────────────
    print("[annotate_unknown] Loading model and reference tables...")
    model   = load_model()
    mapping = load_mapping_second_dataset()
    qual_df = load_protein_quality()

    # Build RNA-FM cache from Unknown/ directory, excluding known proteins
    rnafm_cache = build_rnafm_cache_unknown(mapping, qual_df, requested="all")
    unknown_pairs = get_unknown_protein_pairs(mapping, qual_df, rnafm_cache)
    print(f"[annotate_unknown] Unknown proteins with RNA-FM features: {len(unknown_pairs)}")

    if not unknown_pairs:
        print("[annotate_unknown] No unknown proteins found — skipping.")
        adata.obsm["ADT_pred_unknown"] = pd.DataFrame(index=adata.obs_names)
        return

    # ── Pre-compute log expression once ──────────────────────────────────────
    print("[annotate_unknown] Preparing expression matrix...")
    X_log   = _get_log_expr(adata)
    n_cells = adata.n_obs

    # ── Predict all unknown proteins ─────────────────────────────────────────
    pred_cols = {}
    print(f"[annotate_unknown] Predicting {len(unknown_pairs)} proteins "
          f"for {n_cells:,} cells...")
    print(f"  {'Protein':<30} {'Gene':<15}  Note")
    print("  " + "-" * 60)

    for prot, gene in unknown_pairs:
        # protein_id = NaN → model uses only transferable effects
        X     = build_feature_matrix(adata, gene, rnafm_cache, X_log=X_log,
                                      protein_id_val=np.nan)
        y_raw = model.predict(xgb.DMatrix(X)).astype(np.float32)
        # Clip at 0; no linear calibration available for unknown proteins
        pred_cols[prot] = np.maximum(y_raw, 0.0)
        print(f"  {prot:<30} {gene:<15}  (no calibration)")

    print()

    # ── Annotate adata.obsm ───────────────────────────────────────────────────
    protein_names = list(pred_cols.keys())
    pred_mat = np.column_stack([pred_cols[p] for p in protein_names])
    adata.obsm["ADT_pred_unknown"] = pd.DataFrame(
        pred_mat, index=adata.obs_names, columns=protein_names)
    print(f"[annotate_unknown] Added obsm['ADT_pred_unknown'] "
          f"({n_cells} × {len(protein_names)})")
    print("[annotate_unknown] Done.")


def _check_prepared(adata):
    """Raise if prepare_anndata() hasn't run (required PCA/diffmap/endocytosis features missing)."""
    missing = []
    if "RRS_Endocytosis"        not in adata.obs.columns: missing.append("obs['RRS_Endocytosis']")
    if "X_External_Validation_Package_PCA"     not in adata.obsm:        missing.append("obsm['X_External_Validation_Package_PCA']")
    if "X_External_Validation_Package_diffmap" not in adata.obsm:        missing.append("obsm['X_External_Validation_Package_diffmap']")
    if missing:
        raise ValueError(
            f"adata is missing features from prepare_anndata(): {missing}. "
            "Run External_Validation_Package.prepare_anndata(adata) first."
        )
