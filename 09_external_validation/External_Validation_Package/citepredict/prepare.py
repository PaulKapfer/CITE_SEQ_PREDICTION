"""Compute the cell-state features the model needs, projected into the reference space.

  obsm['X_citepredict_PCA']      50 HVG-PCA coordinates
  obsm['X_citepredict_diffmap']  25 RPG diffusion-map coordinates (Nystroem)
  obs['RRS_Endocytosis']         UCell endocytosis score

CLI:  citepredict-prepare input.h5ad [-o output.h5ad] [--counts-layer counts]
"""
import argparse

from ._expr import log_normalized


def prepare_anndata(adata, counts_layer: str | None = "counts") -> None:
    """Add the three feature sets to `adata` in place.

    counts_layer : layer with raw UMI counts; None if adata.X holds them.
    """
    from ._features.diffmap import compute_diffmap
    from ._features.endocytosis import compute_endocytosis
    from ._features.pca import compute_pca

    print(f"[citepredict] prepare_anndata: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    X_log = log_normalized(adata, counts_layer)
    compute_pca(adata, X_log)
    compute_diffmap(adata, X_log)
    compute_endocytosis(adata, X_log)


def main():
    parser = argparse.ArgumentParser(description="citepredict: compute prediction features.")
    parser.add_argument("input", help="input .h5ad")
    parser.add_argument("--output", "-o", help="output .h5ad (default: overwrite input)")
    parser.add_argument("--counts-layer", default="counts",
                        help="layer with raw counts; 'X' to use adata.X (default: counts)")
    args = parser.parse_args()

    import anndata as ad
    adata = ad.read_h5ad(args.input)
    prepare_anndata(adata, None if args.counts_layer == "X" else args.counts_layer)
    adata.write_h5ad(args.output or args.input)


if __name__ == "__main__":
    main()
