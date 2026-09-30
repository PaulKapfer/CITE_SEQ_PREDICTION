"""citepredict: predict surface-protein (ADT) abundance from scRNA-seq.

    import citepredict
    citepredict.prepare_anndata(adata)     # cell-state features
    citepredict.annotate_known(adata)      # Source=Hao antibodies   -> obsm['ADT_pred_known']
    citepredict.annotate_unknown(adata)    # Source=Kotliarov        -> obsm['ADT_pred_unknown']

ADT names must already follow the reference naming in
data/antibody_info/combined_adt_mapping_MODEL-TRAINING.csv (rename them beforehand).
"""
__version__ = "0.2.0"

from .annotate_known import annotate_known
from .annotate_unknown import annotate_unknown
from .prepare import prepare_anndata

__all__ = ["prepare_anndata", "annotate_known", "annotate_unknown"]
