"""Access to the bundled reference data (data/)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

N_FEATURES = 718   # protein_id, RNA_expr, endocytosis, DC1-25, PC1-50, RNAFM 640
N_RNAFM    = 640


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def load_mapping() -> pd.DataFrame:
    """Antibody -> RNA gene table the model was trained from (one row per clone).

    Source == "Hao"       : antibody measured in the training data (known)
    Source == "Kotliarov" : antibody never seen in training (unknown)
    """
    return pd.read_csv(data_path("antibody_info", "combined_adt_mapping_MODEL-TRAINING.csv"),
                       dtype=str)


def load_protein_ids() -> dict:
    """ADT_feature -> protein_id exactly as assigned during training (0..218)."""
    df = pd.read_csv(data_path("antibody_info", "protein_id_map.csv"))
    return dict(zip(df["ADT_feature"], df["protein_id"].astype(float)))


def load_rnafm(gene: str) -> np.ndarray:
    """640-dim RNA-FM embedding of `gene`, permuted into the order the model was trained on.

    Training reordered every embedding by descending across-gene variance
    (rnafm_dim_order.npy), so raw embeddings must be permuted identically.
    """
    p = data_path("rnafm_features", f"{gene}.npy")
    if not p.exists():
        raise FileNotFoundError(f"No RNA-FM embedding for gene '{gene}' ({p})")
    return np.load(p).astype(np.float32)[_rnafm_order()]


_ORDER = None


def _rnafm_order() -> np.ndarray:
    global _ORDER
    if _ORDER is None:
        _ORDER = np.load(data_path("rnafm_dim_order.npy"))
    return _ORDER


def load_model():
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(data_path("models", "final_model.json")))
    if model.num_features() != N_FEATURES:
        raise RuntimeError(f"Bundled model expects {model.num_features()} features, "
                           f"feature construction produces {N_FEATURES}")
    return model
