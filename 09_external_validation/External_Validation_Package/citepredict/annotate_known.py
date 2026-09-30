"""Predict antibodies the model was trained on (mapping Source == "Hao").

The same antibody clones were measured in training, so each prediction is made
with that antibody's training protein_id, letting the model apply what it learned
about the antibody itself.
"""
from ._keys import OBSM_KNOWN
from ._predict import predict, select_antibodies, store


def annotate_known(adata, proteins=None, counts_layer: str | None = "counts") -> None:
    """Add obsm['ADT_pred_known'] (cells x antibodies) and uns['citepredict']['known'].

    adata        : AnnData processed by citepredict.prepare_anndata
    proteins     : ADT_feature names (reference naming) to predict; default all 219
                   Source=Hao antibodies
    counts_layer : layer with raw counts, used for the target gene's expression;
                   None if adata.X holds raw counts
    """
    rows = select_antibodies("Hao", proteins)
    pred_df, annot_df = predict(adata, rows, known=True, counts_layer=counts_layer)
    store(adata, OBSM_KNOWN, "known", pred_df, annot_df)
