"""Feature matrix construction and prediction shared by annotate_known / annotate_unknown.

Column layout must match _build_X in the training script exactly:
  [0]       protein_id   training id for known antibodies, NaN for unknown
  [1]       RNA_expr     log-normalised expression of the matched gene (0 if absent)
  [2]       endocytosis  UCell endocytosis score
  [3:28]    DC1..DC25    RPG diffusion-map coordinates
  [28:78]   PC1..PC50    HVG-PCA coordinates
  [78:718]  RNAFM        640-dim RNA-FM embedding of the matched gene (training order)
Predictions are the model's raw output; no calibration is applied.
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ._data import N_FEATURES, load_mapping, load_model, load_protein_ids, load_rnafm
from ._expr import log_normalized
from ._keys import OBS_ENDOCYTOSIS, OBSM_DIFFMAP, OBSM_PCA, UNS


def check_prepared(adata) -> None:
    missing = [k for k, ok in [(f"obs['{OBS_ENDOCYTOSIS}']", OBS_ENDOCYTOSIS in adata.obs),
                               (f"obsm['{OBSM_PCA}']",       OBSM_PCA in adata.obsm),
                               (f"obsm['{OBSM_DIFFMAP}']",   OBSM_DIFFMAP in adata.obsm)] if not ok]
    if missing:
        raise ValueError(f"adata lacks {missing}; run citepredict.prepare_anndata(adata) first.")


def select_antibodies(source: str, proteins) -> pd.DataFrame:
    """Rows of the training mapping with the given Source, optionally restricted to `proteins`."""
    mapping = load_mapping()
    rows = mapping[mapping["Source"] == source]
    if proteins is not None:
        proteins = list(proteins)
        unknown_names = [p for p in proteins if p not in set(rows["ADT_feature"])]
        if unknown_names:
            other = mapping.set_index("ADT_feature")["Source"]
            hints = {p: (f"Source={other[p]}" if p in other.index else "not in mapping")
                     for p in unknown_names}
            raise ValueError(f"Not Source={source} antibodies: {hints}")
        rows = rows.set_index("ADT_feature").loc[proteins].reset_index()
    return rows.reset_index(drop=True)


def predict(adata, rows: pd.DataFrame, known: bool, counts_layer) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every antibody in `rows`. Returns (predictions cells x antibodies, annotation)."""
    check_prepared(adata)
    ids = load_protein_ids() if known else {}
    if known:
        no_id = sorted(set(rows["ADT_feature"]) - set(ids))
        if no_id:
            raise RuntimeError(f"Source=Hao antibodies without a training protein_id: {no_id}")

    model = load_model()
    X_log = log_normalized(adata, counts_layer)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    X = np.empty((adata.n_obs, N_FEATURES), dtype=np.float32)
    X[:, 2]     = adata.obs[OBS_ENDOCYTOSIS].to_numpy(np.float32)
    X[:, 3:28]  = np.asarray(adata.obsm[OBSM_DIFFMAP], dtype=np.float32)
    X[:, 28:78] = np.asarray(adata.obsm[OBSM_PCA], dtype=np.float32)

    import xgboost as xgb
    preds, annot = {}, []
    for adt, gene in zip(rows["ADT_feature"], rows["RNA_gene"]):
        pid = ids[adt] if known else np.nan
        X[:, 0] = pid
        if gene in gene_to_idx:
            col = X_log[:, gene_to_idx[gene]]
            X[:, 1] = col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()
        else:
            X[:, 1] = 0.0
        X[:, 78:] = load_rnafm(gene)
        preds[adt] = model.predict(xgb.DMatrix(X)).astype(np.float32)
        annot.append({"ADT_feature": adt, "RNA_gene": gene, "protein_id": pid,
                      "gene_in_dataset": gene in gene_to_idx})

    pred_df  = pd.DataFrame(preds, index=adata.obs_names)
    annot_df = pd.DataFrame(annot)
    n_absent = int((~annot_df["gene_in_dataset"]).sum()) if len(annot_df) else 0
    print(f"[citepredict] {len(pred_df.columns)} antibodies x {adata.n_obs:,} cells "
          f"({'protein_id from training' if known else 'protein_id = NaN'}); "
          f"{n_absent} target gene(s) absent from dataset -> RNA_expr = 0")
    return pred_df, annot_df


def store(adata, obsm_key: str, uns_key: str, pred_df: pd.DataFrame, annot_df: pd.DataFrame) -> None:
    adata.obsm[obsm_key] = pred_df
    adata.uns.setdefault(UNS, {})[uns_key] = annot_df
    print(f"[citepredict] obsm['{obsm_key}'] {pred_df.shape}  |  uns['{UNS}']['{uns_key}']")
