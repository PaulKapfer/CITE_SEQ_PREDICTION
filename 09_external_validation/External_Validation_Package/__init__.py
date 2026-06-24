"""
External_Validation_Package
===========================
Minimal inference package used to apply the trained CITE-seq prediction model
to an independent (external) CITE-seq dataset.

Public API
----------
  prepare_anndata(adata)    – compute the three in-place feature blocks
                              (HVG-PCA, RPG diffusion-map, endocytosis RRS —
                              a custom UCell-inspired score, not the UCell package)
  annotate_unknown(adata)   – predict every protein with protein_id = NaN
                              (zero-shot transfer; raw predictions clipped at 0)

The Known/Unknown distinction used in the figures is applied downstream at
plotting time (by whether the matched transcript was seen during training);
all proteins here are predicted identically with protein_id = NaN.
"""
__version__ = "0.1.0"

from .prepare          import prepare_anndata
from .annotate_unknown import annotate_unknown

__all__ = [
    "prepare_anndata",
    "annotate_unknown",
]
