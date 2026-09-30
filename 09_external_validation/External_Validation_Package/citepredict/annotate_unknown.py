"""Predict antibodies absent from training (mapping Source == "Kotliarov").

protein_id is NaN, so the model can only use effects that transfer across
antibodies: target-gene expression, its RNA-FM embedding and cell state.
"""
from ._keys import OBSM_UNKNOWN
from ._predict import predict, select_antibodies, store


def annotate_unknown(adata, proteins=None, counts_layer: str | None = "counts") -> None:
    """Add obsm['ADT_pred_unknown'] (cells x antibodies) and uns['citepredict']['unknown'].

    adata        : AnnData processed by citepredict.prepare_anndata
    proteins     : ADT_feature names to predict; default all Source=Kotliarov antibodies
    counts_layer : layer with raw counts, used for the target gene's expression;
                   None if adata.X holds raw counts
    """
    rows = select_antibodies("Kotliarov", proteins)
    pred_df, annot_df = predict(adata, rows, known=False, counts_layer=counts_layer)
    store(adata, OBSM_UNKNOWN, "unknown", pred_df, annot_df)
