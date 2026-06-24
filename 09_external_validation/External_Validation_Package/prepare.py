"""
Command 1: External_Validation_Package_prepare_anndata

Sequentially computes three feature sets and stores them in the AnnData object:
  1. HVG-PCA (50 PCs)      → adata.obsm["X_External_Validation_Package_PCA"]
  2. RPG diffmap (25)      → adata.obsm["X_External_Validation_Package_diffmap"]
  3. Endocytosis RRS       → adata.obs["RRS_Endocytosis"]
     (custom UCell-INSPIRED score — not the UCell/pyUCell package)

Usage
-----
Python API:
    import External_Validation_Package
    External_Validation_Package.prepare_anndata(adata)

CLI:
    External_Validation_Package_prepare_anndata input.h5ad --output output.h5ad
"""
import argparse


def prepare_anndata(adata) -> None:
    """
    Compute all feature sets in-place on *adata*.

    After this call the AnnData contains:
      adata.obs["RRS_Endocytosis"]           – endocytosis RRS (custom UCell-inspired)
      adata.obsm["X_External_Validation_Package_PCA"]       – (n_cells, 50) PC coordinates
      adata.obsm["X_External_Validation_Package_diffmap"]   – (n_cells, 25) DC coordinates
    """
    from ._features.pca         import compute_pca
    from ._features.diffmap     import compute_diffmap
    from ._features.endocytosis import compute_endocytosis

    print("=" * 60)
    print("External_Validation_Package: prepare_anndata")
    print(f"  Input: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print("=" * 60)

    print("\n[1/3] HVG-PCA projection (50 PCs)")
    compute_pca(adata)

    print("\n[2/3] RPG diffusion-map projection (25 DCs)")
    compute_diffmap(adata)

    print("\n[3/3] Endocytosis RRS (custom UCell-inspired) score")
    compute_endocytosis(adata)

    print("\n" + "=" * 60)
    print("prepare_anndata complete.")
    print("  adata.obs columns added : RRS_Endocytosis")
    print("  adata.obsm keys added   : X_External_Validation_Package_PCA, X_External_Validation_Package_diffmap")
    print("=" * 60)


def main():
    """CLI entry point: read an .h5ad, run prepare_anndata() in place, write it back."""
    parser = argparse.ArgumentParser(
        description="External_Validation_Package – compute prediction features for an AnnData object."
    )
    parser.add_argument("input",  help="Path to input .h5ad file")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to save annotated .h5ad (default: overwrite input)")
    args = parser.parse_args()

    import anndata as ad
    print(f"Reading: {args.input}")
    adata = ad.read_h5ad(args.input)

    prepare_anndata(adata)

    out = args.output if args.output else args.input
    print(f"\nSaving to: {out}")
    adata.write_h5ad(out)
    print("Done.")


if __name__ == "__main__":
    main()
