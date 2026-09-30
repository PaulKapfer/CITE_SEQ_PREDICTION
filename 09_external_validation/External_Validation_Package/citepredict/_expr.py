"""Expression normalisation shared by every feature."""
import warnings

import numpy as np
from scipy.sparse import csr_matrix


def log_normalized(adata, counts_layer: str | None = "counts") -> csr_matrix:
    """Raw counts -> per-cell scaling to 1e4 -> log1p, as in reference_preparation.py.

    The reference derived PCA, diffusion map, endocytosis score and RNA_expr all from
    this transform of raw counts, so it is always recomputed from counts rather than
    taken from a pre-normalised layer whose recipe is unknown.

    counts_layer : layer holding raw UMI counts; None means adata.X holds them.
    """
    if counts_layer is None:
        X = adata.X
    elif counts_layer in adata.layers:
        X = adata.layers[counts_layer]
    else:
        raise ValueError(
            f"adata.layers has no '{counts_layer}' layer. citepredict needs raw counts: "
            f"pass counts_layer=<layer name>, or counts_layer=None if adata.X holds them."
        )

    X = csr_matrix(X, dtype=np.float32, copy=True)
    sample = X.data[:100_000]
    if sample.size and not np.allclose(sample, np.round(sample)):
        warnings.warn("The counts matrix contains non-integer values; citepredict expects raw "
                      "counts and will normalise them itself.", stacklevel=2)

    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    X = X.multiply((1e4 / row_sums).astype(np.float32).reshape(-1, 1)).tocsr()
    X.data = np.log1p(X.data)
    return X
