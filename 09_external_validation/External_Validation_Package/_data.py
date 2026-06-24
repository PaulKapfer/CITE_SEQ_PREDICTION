"""Central data access for bundled reference files."""
from pathlib import Path
import numpy as np
import pandas as pd

# Bundled data directory sits two levels up from this file (package/data/)
DATA_DIR = Path(__file__).parent / "data"


def data_path(*parts: str) -> Path:
    """Return path to a bundled data file."""
    return DATA_DIR.joinpath(*parts)


def load_calibration() -> pd.DataFrame:
    """Load final model linear calibration parameters (slope, intercept per protein)."""
    return pd.read_csv(data_path("calibration", "final_model_lm_params.csv"))


def load_protein_quality() -> pd.DataFrame:
    """Load per-protein quality metrics (r, r2, rmse, std) from in-fold known proteins."""
    return pd.read_csv(data_path("protein_quality", "infold_known_per_protein.csv"))


def load_mapping() -> pd.DataFrame:
    """Load ADT protein → RNA gene mapping table (columns: ADT_feature, RNA_gene)."""
    return pd.read_csv(data_path("adt_rna_mapping.csv"))


def load_mapping_second_dataset() -> pd.DataFrame:
    """Load ADT → RNA gene mapping for the second dataset (known + unknown proteins).

    Columns: protein, Known_protein_id, Ensembl_ID, RNA_gene.
    Returned with ADT_feature and RNA_gene columns normalised for compatibility
    with build_rnafm_cache_unknown / get_unknown_protein_pairs.
    """
    df = pd.read_csv(data_path("adt_rna_mapping_second_dataset.csv"), dtype=str).fillna("")
    # Normalise to the column names expected by _predict utilities
    df = df.rename(columns={"protein": "ADT_feature"})
    # Drop rows with no gene symbol
    df = df[df["RNA_gene"].str.strip() != ""].reset_index(drop=True)
    return df


def load_rnafm(gene: str) -> np.ndarray:
    """Load 640-dim RNA-FM embedding from Known/. Returns (640,) float32."""
    p = data_path("rnafm_features", "Known", f"{gene}.npy")
    if not p.exists():
        raise FileNotFoundError(f"RNA-FM features not found for gene '{gene}': {p}")
    return np.load(str(p)).astype(np.float32)


def load_rnafm_unknown(gene: str) -> np.ndarray:
    """Load 640-dim RNA-FM embedding from Unknown/. Returns (640,) float32."""
    p = data_path("rnafm_features", "Unknown", f"{gene}.npy")
    if not p.exists():
        raise FileNotFoundError(f"RNA-FM features not found for unknown gene '{gene}': {p}")
    return np.load(str(p)).astype(np.float32)


def load_rnafm_any(gene: str) -> np.ndarray:
    """Load 640-dim RNA-FM embedding from Known/ or Unknown/ (whichever exists).

    Searches Known/ first, then Unknown/. Raises FileNotFoundError if absent from both.
    """
    for subdir in ("Known", "Unknown"):
        p = data_path("rnafm_features", subdir, f"{gene}.npy")
        if p.exists():
            return np.load(str(p)).astype(np.float32)
    raise FileNotFoundError(
        f"RNA-FM features not found for gene '{gene}' in Known/ or Unknown/"
    )


def load_model():
    """Load the final XGBoost model."""
    import xgboost as xgb
    m = xgb.Booster()
    m.load_model(str(data_path("models", "final_model.json")))
    return m


def load_fold_model(fold: int):
    """Load a cross-validation fold model (0–4)."""
    import xgboost as xgb
    m = xgb.Booster()
    m.load_model(str(data_path("models", f"model_fold_{fold}.json")))
    return m


def load_fold_splits() -> "pd.DataFrame":
    """Load CV fold protein splits. Columns: fold, train_proteins, test_proteins (pipe-separated)."""
    import pandas as pd
    return pd.read_csv(data_path("models", "fold_protein_splits.csv"))
