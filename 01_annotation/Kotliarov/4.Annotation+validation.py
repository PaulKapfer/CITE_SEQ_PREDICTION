"""
Script 4: Annotation and Validation
Single-cell RNA-seq data preprocessing workflow

Phase 8: Automated Annotation with CellTypist
- Run CellTypist with pre-trained model
- Store predictions and confidence scores

Phase 9: Marker Gene Validation
- Differential expression analysis
- Known marker visualization

Phase 10: Annotation Validation
- Cross-tabulation and confusion matrix
- Cluster purity analysis

Phase 11: Manual Curation
- Framework for manual refinement

NOTE: This script uses 'celltype_final' for all analysis and visualization,
      which is initialized from CellTypist predictions but can be manually refined.
      The original CellTypist predictions are preserved in 'celltypist_predicted_labels'.
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Prevent GDI object limits
import gc

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/3.batch+clustering/adata_batch_clustered.h5ad"
OUTPUT_DIR = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/4.Annotation+validation")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CellTypist parameters
CELLTYPIST_MODEL = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Input/Immune_All_Low.pkl"
MAJORITY_VOTING = True
CONFIDENCE_THRESHOLD = 0.5

# Clustering resolution to use for validation
CLUSTER_KEY = 'leiden_res0.8'

# Known PBMC markers - matched to CellTypist cell types
KNOWN_MARKERS = {
    # T cells
    'CD4_T': ['CD4', 'IL7R', 'CCR7', 'S100A4', 'LTB', 'SELL', 'LEF1', 'TCF7'],  # CD4 T cells
    'CD8_T': ['CD8A', 'CD8B', 'GZMK', 'GZMA', 'PRF1', 'IFNG'],  # CD8 memory T cells
    
    # NK cells
    'NK': ['NKG7', 'GNLY', 'NCAM1', 'KLRD1', 'FGFBP2', 'FCGR3A'],  # Natural Killer cells
    
    # B cells and Plasma cells
    'B': ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'IGHA1', 'IGHG1', 'MZB1', 'SDC1', 'JCHAIN'],  # B cells

    # Plasma cells / Plasmablasts (Speziell für Cluster 6 & COVID PBMCs)
    'Plasma': ['MZB1', 'JCHAIN', 'PRDM1', 'SDC1', 'CD38', 'IGHA1', 'IGHG1', 'XBP1', 'TXNDC5'],

    # Monocytes
    'Monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1', 'FCGR3A', 'MS4A7', 'CDKN1C'],  # Monocytes
    
    # Other cell types
    'Platelet': ['PPBP', 'PF4', 'GNG11', 'TUBB1']  # Platelets
}

# Differential expression parameters
DE_METHOD = 't-test'  # or 't-test'
N_TOP_GENES = 50

# ============================================================================
# PHASE 0: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

adata = sc.read_h5ad(INPUT_FILE)

print(f"\nLoaded data shape: {adata.shape}")
print(f"Cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")
print(f"Samples: {adata.obs['sample'].nunique()}")
print(f"Using cluster column: {CLUSTER_KEY}")

# Remove pre-existing Seurat cell type columns to avoid downstream confusion
for col in ['celltype.l1', 'celltype.l2', 'celltype.l3']:
    if col in adata.obs.columns:
        adata.obs.drop(columns=[col], inplace=True)
        print(f"Removed pre-existing column: '{col}'")

# Check if we need to use raw counts for CellTypist
if 'counts' not in adata.layers:
    print("\nWARNING: 'counts' layer not found. CellTypist needs raw counts.")
    print("Attempting to use X as counts...")

# ============================================================================
# PHASE 8: AUTOMATED ANNOTATION WITH CELLTYPIST
# ============================================================================

print("\n" + "="*80)
print("PHASE 8: AUTOMATED ANNOTATION WITH CELLTYPIST")
print("="*80)

try:
    import celltypist
    from celltypist import models

    print(f"\nLoading CellTypist model: {CELLTYPIST_MODEL}")

    # Load the model
    model = models.Model.load(CELLTYPIST_MODEL)

    print(f"Model loaded successfully")
    print(f"Model cell types: {len(model.cell_types)} types")

    # Get model genes to subset data BEFORE copying (memory optimization)
    model_genes = model.features
    print(f"Model uses {len(model_genes)} genes")

    # Find overlap with dataset
    genes_to_use = [g for g in model_genes if g in adata.var_names]
    print(f"Found {len(genes_to_use)} model genes in dataset ({len(genes_to_use)/len(model_genes)*100:.1f}%)")

    # Prepare data for CellTypist
    # CellTypist needs log-normalized data
    print("\nPreparing data for CellTypist...")
    print(f"Original data: {adata.shape[0]} cells × {adata.shape[1]} genes = {adata.shape[0] * adata.shape[1] / 1e9:.2f}B elements")

    # CRITICAL: Subset to model genes FIRST, then copy (memory optimization)
    # This reduces memory footprint dramatically (27,846 genes → ~2,985 genes)
    adata_subset = adata[:, genes_to_use].copy()
    print(f"Subset data: {adata_subset.shape[0]} cells × {adata_subset.shape[1]} genes = {adata_subset.shape[0] * adata_subset.shape[1] / 1e9:.2f}B elements")

    # If we have raw counts in layers, use them
    if 'counts' in adata_subset.layers:
        print("Using counts from 'counts' layer")
        adata_subset.X = adata_subset.layers['counts'].copy()
        # Normalize and log-transform
        sc.pp.normalize_total(adata_subset, target_sum=1e4)
        sc.pp.log1p(adata_subset)
    else:
        print("Using current X (assuming already log-normalized)")

    # Ensure float32 dtype (biological data doesn't need float64 precision)
    if adata_subset.X.dtype != np.float32:
        print(f"Converting from {adata_subset.X.dtype} to float32 (memory optimization)")
        adata_subset.X = adata_subset.X.astype(np.float32)

    expected_memory_gb = (adata_subset.shape[0] * adata_subset.shape[1] * 4) / (1024**3)
    print(f"Expected memory for CellTypist scaling: ~{expected_memory_gb:.1f} GiB (float32)")

    # Run CellTypist
    print(f"\nRunning CellTypist annotation...")
    print(f"Majority voting: {MAJORITY_VOTING}")

    predictions = celltypist.annotate(
        adata_subset,
        model=model,
        majority_voting=MAJORITY_VOTING,
    )

    # Clean up immediately to free memory
    del adata_subset
    gc.collect()
    print("Cleaned up temporary data")

    # Extract predictions
    print("\nCellTypist annotation complete")

    # Debug: Check what columns are available
    print(f"\nAvailable columns in predictions.predicted_labels:")
    print(predictions.predicted_labels.columns.tolist())
    print(f"\nFirst few rows of predictions.predicted_labels:")
    print(predictions.predicted_labels.head())

    # Store results in original adata
    adata.obs['celltypist_cell_type'] = predictions.predicted_labels.predicted_labels.values

    if MAJORITY_VOTING:
        adata.obs['celltypist_cell_type_majority'] = predictions.predicted_labels.majority_voting.values
        # Use majority voting as the main prediction
        adata.obs['celltypist_predicted_labels'] = adata.obs['celltypist_cell_type_majority']
    else:
        adata.obs['celltypist_predicted_labels'] = adata.obs['celltypist_cell_type']

    # Check for majorTypes column
    if 'majorTypes' in predictions.predicted_labels.columns:
        print("\nFound 'majorTypes' column")
        adata.obs['celltypist_major_types'] = predictions.predicted_labels.majorTypes.values
        print(f"\nMajor cell type lineages:")
        print(adata.obs['celltypist_major_types'].value_counts())

    # Store confidence scores if available
    # Note: When majority_voting=True, CellTypist doesn't provide confidence scores
    # We'll use a different approach to assess prediction quality
    if 'conf_score' in predictions.predicted_labels.columns:
        adata.obs['celltypist_conf_score'] = predictions.predicted_labels.conf_score.values
        print("\nUsing CellTypist confidence scores")
    elif 'confidence_score' in predictions.predicted_labels.columns:
        adata.obs['celltypist_conf_score'] = predictions.predicted_labels.confidence_score.values
        print("\nUsing CellTypist confidence scores")
    else:
        print("\nNote: Confidence scores not available with majority voting.")
        print("Computing prediction consistency as a proxy for confidence...")

        # Calculate a pseudo-confidence based on majority voting consistency
        # For each cell, count how many neighbors agree with the prediction
        # This requires the probability matrix which might be in predictions
        if hasattr(predictions, 'probability_matrix'):
            # Use the max probability as confidence
            adata.obs['celltypist_conf_score'] = predictions.probability_matrix.max(axis=1)
            print("Using max probability from probability matrix as confidence")
        else:
            # Fallback: set all to 1.0 (assume high confidence)
            print("Setting all confidence scores to 1.0 (majority voting assumes high confidence)")
            adata.obs['celltypist_conf_score'] = 1.0

    # Flag low-confidence cells
    adata.obs['low_confidence'] = adata.obs['celltypist_conf_score'] < CONFIDENCE_THRESHOLD

    n_low_conf = adata.obs['low_confidence'].sum()
    print(f"\nLow confidence cells (score < {CONFIDENCE_THRESHOLD}): {n_low_conf} ({n_low_conf/adata.n_obs*100:.1f}%)")

    # Print cell type distribution
    print("\nCell type distribution:")
    print(adata.obs['celltypist_predicted_labels'].value_counts())

    # Check if 'majority_voting' column exists (contains major cell types)
    if 'majority_voting' in predictions.predicted_labels.columns:
        print("\nMajor cell types detected:")
        print(predictions.predicted_labels['majority_voting'].value_counts())

    # Initialize final cell type column with CellTypist predictions
    # Convert to string to allow manual modifications later
    adata.obs['celltype_final'] = adata.obs['celltypist_predicted_labels'].astype(str)

    # -------------------------------------------------------------------------
    # Optional type merging — populate after inspecting predictions
    # -------------------------------------------------------------------------
    # type_mapping = {
    #     'Example fine type': 'Merged coarse type',
    # }
    # adata.obs['celltype_final'] = adata.obs['celltype_final'].replace(type_mapping)
    print("\n[Custom] No type merging applied — edit type_mapping to customise.")

    # Convert to categorical for efficiency
    adata.obs['celltype_final'] = adata.obs['celltype_final'].astype('category')

    CELLTYPIST_SUCCESS = True

except ImportError:
    print("\nERROR: celltypist not installed")
    print("To install: pip install celltypist")
    CELLTYPIST_SUCCESS = False
    adata.obs['celltypist_predicted_labels'] = 'Unknown'
    adata.obs['celltype_final'] = 'Unknown'
    adata.obs['celltypist_conf_score'] = 0.0
    adata.obs['low_confidence'] = True

except Exception as e:
    print(f"\nERROR running CellTypist: {e}")
    print("Continuing without CellTypist annotations...")
    CELLTYPIST_SUCCESS = False
    adata.obs['celltypist_predicted_labels'] = 'Unknown'
    adata.obs['celltype_final'] = 'Unknown'
    adata.obs['celltypist_conf_score'] = 0.0
    adata.obs['low_confidence'] = True

# Visualize CellTypist predictions
if CELLTYPIST_SUCCESS:
    print("\nVisualizing CellTypist predictions...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('CellTypist Annotations', fontsize=16, y=0.995)

    # UMAP colored by predicted cell type
    sc.pl.umap(adata, color='celltype_final', ax=axes[0, 0],
               show=False, title='CellType Annotations')

    # UMAP colored by confidence score
    sc.pl.umap(adata, color='celltypist_conf_score', ax=axes[0, 1],
               show=False, title='Confidence Score', cmap='RdYlGn')

    # UMAP colored by low confidence flag
    sc.pl.umap(adata, color='low_confidence', ax=axes[1, 0],
               show=False, title='Low Confidence Cells')

    # UMAP colored by clusters
    sc.pl.umap(adata, color=CLUSTER_KEY, ax=axes[1, 1],
               show=False, title=f'Leiden Clusters ({CLUSTER_KEY})', legend_loc='on data')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_celltypist_predictions.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"  Saved: {FIGURES_DIR / '01_celltypist_predictions.png'}")

# ============================================================================
# PHASE 9: MARKER GENE VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 9: MARKER GENE VALIDATION")
print("="*80)

# ----------------------------------------------------------------------------
# 9.1 Differential expression analysis
# ----------------------------------------------------------------------------

if CELLTYPIST_SUCCESS:
    print("\n[Step 1/3] Running differential expression analysis...")

    # Set normalized data for differential expression
    # Note: Wilcoxon test requires log-normalized data, not raw counts
    if 'counts' in adata.layers:
        print("Preparing log-normalized data for differential expression...")

        # MEMORY OPTIMIZATION: Create minimal AnnData instead of full copy
        # Only include what's needed: counts matrix + var/obs
        # This avoids copying layers, obsm, varm, uns (saves ~40-60 GB)

        print(f"  Creating minimal AnnData with counts (avoiding full copy)...")
        # Convert counts to float32 BEFORE normalization (saves 50% memory)
        counts_matrix = adata.layers['counts'].astype(np.float32)

        # Create new minimal AnnData object (not a copy of everything!)
        adata_raw = ad.AnnData(
            X=counts_matrix,
            obs=adata.obs.copy(),
            var=adata.var.copy()
        )

        print(f"  Matrix shape: {adata_raw.shape}, dtype: {adata_raw.X.dtype}")
        print(f"  Normalizing and log-transforming...")

        # Normalize and log-transform (already float32, so stays float32)
        sc.pp.normalize_total(adata_raw, target_sum=1e4)
        sc.pp.log1p(adata_raw)

        print(f"  Setting as .raw attribute...")
        adata.raw = adata_raw

        # Clean up immediately
        del adata_raw, counts_matrix
        gc.collect()

        print("  Using log-normalized counts (float32) for differential expression")
    else:
        adata.raw = adata
        print("Using current X for differential expression")

    print(f"Method: {DE_METHOD}")
    print(f"Grouping by: celltype_final")

    # Run differential expression
    sc.tl.rank_genes_groups(
        adata,
        groupby='celltype_final',
        method=DE_METHOD,
        use_raw=True,
        n_genes=N_TOP_GENES
    )

    print("Differential expression complete")

    # Get top marker genes per cell type
    print("\nTop 10 marker genes per cell type:")

    result = adata.uns['rank_genes_groups']
    cell_types = result['names'].dtype.names

    top_markers = {}
    for cell_type in cell_types:
        genes = result['names'][cell_type][:10]
        scores = result['scores'][cell_type][:10]
        print(f"\n{cell_type}:")
        for gene, score in zip(genes, scores):
            print(f"  {gene}: {score:.2f}")
        top_markers[cell_type] = list(genes)

    # Save top markers to file
    markers_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in top_markers.items()]))
    markers_file = OUTPUT_DIR / "top_markers_per_celltype.csv"
    markers_df.to_csv(markers_file, index=False)
    print(f"\nSaved top markers: {markers_file}")

    # Plot ranking
    print("\nPlotting marker gene rankings...")
    sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, show=False)
    plt.savefig(FIGURES_DIR / '02_marker_genes_ranking.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"  Saved: {FIGURES_DIR / '02_marker_genes_ranking.png'}")

# ----------------------------------------------------------------------------
# 9.2 Check known markers
# ----------------------------------------------------------------------------

print("\n[Step 2/3] Checking known PBMC markers...")

# Find which known markers are present in the dataset
available_markers = {}
missing_markers = {}

for cell_type, markers in KNOWN_MARKERS.items():
    available = [m for m in markers if m in adata.var_names]
    missing = [m for m in markers if m not in adata.var_names]

    if available:
        available_markers[cell_type] = available
    if missing:
        missing_markers[cell_type] = missing

print("\nAvailable known markers:")
for cell_type, markers in available_markers.items():
    print(f"  {cell_type}: {', '.join(markers)}")

if missing_markers:
    print("\nMissing markers (not in dataset):")
    for cell_type, markers in missing_markers.items():
        print(f"  {cell_type}: {', '.join(markers)}")

# Create flat list of all available markers
all_available_markers = []
for markers in available_markers.values():
    all_available_markers.extend(markers)
all_available_markers = list(set(all_available_markers))

print(f"\nTotal available markers: {len(all_available_markers)}")

# ----------------------------------------------------------------------------
# 9.3 Visualize markers
# ----------------------------------------------------------------------------

print("\n[Step 3/3] Visualizing marker genes...")

if CELLTYPIST_SUCCESS and len(all_available_markers) > 0:
    # Create a subdirectory for per-celltype plots
    celltype_dir = FIGURES_DIR / "per_celltype"
    celltype_dir.mkdir(exist_ok=True)

    # Dot plots: one per cell type with all its markers
    print("\nCreating dot plots per cell type...")

    for cell_type, markers in available_markers.items():
        if len(markers) > 0:
            # Clean cell type name for filename
            clean_name = cell_type.replace(' ', '_').replace('/', '_')

            try:
                sc.pl.dotplot(
                    adata,
                    markers,
                    groupby='celltype_final',
                    dendrogram=True,
                    show=False,
                    figsize=(max(8, len(markers) * 0.5), 8)
                )
                plt.savefig(celltype_dir / f'03_dotplot_{clean_name}.png', dpi=300, bbox_inches='tight')
                plt.close('all')
                gc.collect()
                print(f"  Saved: {celltype_dir / f'03_dotplot_{clean_name}.png'}")
            except Exception as e:
                print(f"  Warning: Could not create dotplot for {cell_type}: {e}")
                plt.close('all')
                gc.collect()

    # UMAP plots: one combined figure per cell type with all its markers + celltype annotation
    print("\nCreating UMAP plots per cell type...")

    for cell_type, markers in available_markers.items():
        if len(markers) > 0:
            clean_name = cell_type.replace(' ', '_').replace('/', '_')
            n_markers = len(markers)

            # Add one more subplot for the celltype predictions
            n_total_plots = n_markers + 1

            # Determine grid size
            n_cols = min(4, n_total_plots)
            n_rows = int(np.ceil(n_total_plots / n_cols))

            try:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
                fig.suptitle(f'{cell_type} - Marker Expression', fontsize=16, y=0.995)

                # Flatten axes array for easier indexing
                if n_total_plots == 1:
                    axes = np.array([axes])
                elif n_rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Plot marker genes
                for idx, marker in enumerate(markers):
                    sc.pl.umap(adata, color=marker, ax=axes[idx], show=False,
                              title=marker, cmap='Reds', frameon=False)

                # Plot celltype predictions as the last subplot
                # Use 'on data' legend to prevent UMAP compression (legend overlays instead of taking subplot space)
                sc.pl.umap(adata, color='celltype_final', ax=axes[n_markers],
                          show=False, title='Cell Type Annotations', frameon=False, legend_loc='on data',
                          legend_fontsize=6)

                # Hide unused subplots
                for idx in range(n_total_plots, len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                plt.savefig(celltype_dir / f'05_umap_{clean_name}.png', dpi=300, bbox_inches='tight')
                plt.close('all')
                gc.collect()
                print(f"  Saved: {celltype_dir / f'05_umap_{clean_name}.png'}")
            except Exception as e:
                print(f"  Warning: Could not create UMAP for {cell_type}: {e}")
                plt.close('all')
                gc.collect()

# ============================================================================
# PHASE 10: ANNOTATION VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 10: ANNOTATION VALIDATION")
print("="*80)

if CELLTYPIST_SUCCESS:
    # ----------------------------------------------------------------------------
    # 10.1 Cross-tabulation
    # ----------------------------------------------------------------------------

    print("\n[Step 1/3] Creating cross-tabulation...")

    # Create cross-tabulation
    crosstab = pd.crosstab(
        adata.obs[CLUSTER_KEY],
        adata.obs['celltype_final'],
        margins=True
    )

    print("\nCross-tabulation (Leiden clusters vs Cell Type labels):")
    print(crosstab)

    # Save cross-tabulation
    crosstab_file = OUTPUT_DIR / "cluster_celltype_crosstab.csv"
    crosstab.to_csv(crosstab_file)
    print(f"\nSaved cross-tabulation: {crosstab_file}")

    # Visualize as heatmap
    print("\nCreating confusion matrix heatmap...")

    # Normalize by cluster (rows)
    crosstab_norm = pd.crosstab(
        adata.obs[CLUSTER_KEY],
        adata.obs['celltype_final'],
        normalize='index'
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(crosstab_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Cluster vs Cell Type (normalized by cluster)')
    ax.set_xlabel('CellTypist Predicted Cell Type')
    ax.set_ylabel(f'Leiden Cluster ({CLUSTER_KEY})')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"  Saved: {FIGURES_DIR / '06_confusion_matrix.png'}")

    # ----------------------------------------------------------------------------
    # 10.2 Calculate cluster purity
    # ----------------------------------------------------------------------------

    print("\n[Step 2/3] Calculating cluster purity...")

    cluster_purity = {}
    for cluster in adata.obs[CLUSTER_KEY].unique():
        cluster_cells = adata.obs[adata.obs[CLUSTER_KEY] == cluster]
        most_common_type = cluster_cells['celltype_final'].value_counts().iloc[0]
        total_cells = len(cluster_cells)
        purity = most_common_type / total_cells

        cluster_purity[cluster] = {
            'dominant_type': cluster_cells['celltype_final'].value_counts().index[0],
            'purity': purity,
            'n_cells': total_cells
        }

    purity_df = pd.DataFrame(cluster_purity).T
    purity_df = purity_df.sort_values('purity', ascending=False)

    print("\nCluster purity scores:")
    print(purity_df)

    # Save purity scores
    purity_file = OUTPUT_DIR / "cluster_purity.csv"
    purity_df.to_csv(purity_file)
    print(f"\nSaved cluster purity: {purity_file}")

    # Calculate overall statistics
    mean_purity = purity_df['purity'].mean()
    print(f"\nMean cluster purity: {mean_purity:.2%}")

    if mean_purity >= 0.7:
        print("✓ Good agreement (≥70%)")
    elif mean_purity >= 0.5:
        print("⚠ Needs attention (50-70%)")
    else:
        print("✗ Major issues (<50%)")

    # ----------------------------------------------------------------------------
    # 10.3 Identify problems
    # ----------------------------------------------------------------------------

    print("\n[Step 3/3] Identifying potential problems...")

    # Mixed clusters (purity < 0.7)
    mixed_clusters = purity_df[purity_df['purity'] < 0.7]
    print(f"\nMixed clusters (purity < 70%): {len(mixed_clusters)}")
    if len(mixed_clusters) > 0:
        print(mixed_clusters)

    # Low confidence cells per cluster
    low_conf_per_cluster = adata.obs.groupby(CLUSTER_KEY)['low_confidence'].sum()
    print(f"\nLow confidence cells per cluster:")
    print(low_conf_per_cluster)

    # Small clusters (potential outliers)
    small_clusters = purity_df[purity_df['n_cells'] < 50]
    print(f"\nSmall clusters (< 50 cells): {len(small_clusters)}")
    if len(small_clusters) > 0:
        print(small_clusters)

    # Visualize validation results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Annotation Validation', fontsize=16, y=0.995)

    # Cluster purity
    purity_df['purity'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].axhline(y=0.7, color='green', linestyle='--', label='Good (70%)')
    axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', label='Acceptable (50%)')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Purity')
    axes[0, 0].set_title('Cluster Purity')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Low confidence cells
    low_conf_per_cluster.plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Number of low confidence cells')
    axes[0, 1].set_title('Low Confidence Cells per Cluster')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Cluster sizes
    purity_df['n_cells'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].axhline(y=50, color='red', linestyle='--', label='Min size (50)')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of cells')
    axes[1, 0].set_title('Cluster Sizes')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Confidence score distribution
    adata.obs['celltypist_conf_score'].hist(bins=50, ax=axes[1, 1], color='purple', alpha=0.7)
    axes[1, 1].axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--',
                       label=f'Threshold ({CONFIDENCE_THRESHOLD})')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Number of cells')
    axes[1, 1].set_title('Confidence Score Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_validation_summary.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"  Saved: {FIGURES_DIR / '07_validation_summary.png'}")

# ============================================================================
# PHASE 11: MANUAL CURATION FRAMEWORK
# ============================================================================

print("\n" + "="*80)
print("PHASE 11: MANUAL CURATION FRAMEWORK")
print("="*80)

if CELLTYPIST_SUCCESS:
    # Note: celltype_final was already created in Phase 8
    # Here we just flag cells that need manual review

    # Flag cells that need manual review
    adata.obs['needs_manual_review'] = False

    # Flag low confidence cells
    adata.obs.loc[adata.obs['low_confidence'], 'needs_manual_review'] = True

    # Flag cells in mixed clusters
    mixed_cluster_ids = mixed_clusters.index if len(mixed_clusters) > 0 else []
    adata.obs.loc[adata.obs[CLUSTER_KEY].isin(mixed_cluster_ids), 'needs_manual_review'] = True

    # Flag cells in small clusters
    small_cluster_ids = small_clusters.index if len(small_clusters) > 0 else []
    adata.obs.loc[adata.obs[CLUSTER_KEY].isin(small_cluster_ids), 'needs_manual_review'] = True

    n_review = adata.obs['needs_manual_review'].sum()
    print(f"\nCells flagged for manual review: {n_review} ({n_review/adata.n_obs*100:.1f}%)")

    # Create detailed manual review report
    print("\n" + "-"*80)
    print("GENERATING DETAILED MANUAL REVIEW REPORT")
    print("-"*80)

    # Prepare detailed statistics per cell type
    review_report = []

    for cell_type in adata.obs['celltype_final'].unique():
        cell_type_mask = adata.obs['celltype_final'] == cell_type
        cell_type_data = adata.obs[cell_type_mask]

        n_total = cell_type_mask.sum()
        n_low_conf = (cell_type_data['low_confidence']).sum()
        n_review = (cell_type_data['needs_manual_review']).sum()

        # Get cluster distribution for this cell type
        cluster_dist = cell_type_data[CLUSTER_KEY].value_counts()
        n_clusters = len(cluster_dist)
        dominant_cluster = cluster_dist.index[0] if len(cluster_dist) > 0 else 'N/A'
        dominant_cluster_pct = (cluster_dist.iloc[0] / n_total * 100) if len(cluster_dist) > 0 else 0

        # Check if this cell type is spread across mixed clusters
        mixed_clusters_for_type = []
        for cluster_id in cluster_dist.index:
            if cluster_id in mixed_cluster_ids:
                mixed_clusters_for_type.append(cluster_id)

        # Determine review reasons
        review_reasons = []
        if n_low_conf > 0:
            review_reasons.append(f"Low confidence ({n_low_conf} cells, {n_low_conf/n_total*100:.1f}%)")
        if len(mixed_clusters_for_type) > 0:
            n_in_mixed = cell_type_data[CLUSTER_KEY].isin(mixed_clusters_for_type).sum()
            review_reasons.append(f"In mixed clusters ({n_in_mixed} cells in clusters {', '.join(map(str, mixed_clusters_for_type))})")
        if n_total < 50:
            review_reasons.append(f"Small population ({n_total} cells)")
        if n_clusters > 5:
            review_reasons.append(f"Highly fragmented across {n_clusters} clusters")

        review_report.append({
            'Cell_Type': cell_type,
            'Total_Cells': n_total,
            'Cells_Needing_Review': n_review,
            'Percent_Needing_Review': f"{n_review/n_total*100:.1f}%" if n_total > 0 else "0%",
            'Low_Confidence_Cells': n_low_conf,
            'N_Clusters_Spanned': n_clusters,
            'Dominant_Cluster': dominant_cluster,
            'Dominant_Cluster_Percent': f"{dominant_cluster_pct:.1f}%",
            'Mixed_Clusters': ', '.join(map(str, mixed_clusters_for_type)) if mixed_clusters_for_type else 'None',
            'Review_Reasons': '; '.join(review_reasons) if review_reasons else 'None'
        })

    review_df = pd.DataFrame(review_report)
    review_df = review_df.sort_values('Cells_Needing_Review', ascending=False)

    # Save to CSV
    review_file = OUTPUT_DIR / "manual_review_report.csv"
    review_df.to_csv(review_file, index=False)
    print(f"\nSaved detailed review report: {review_file}")

    # Create text report
    text_report = []
    text_report.append("=" * 80)
    text_report.append("MANUAL REVIEW REPORT")
    text_report.append("=" * 80)
    text_report.append("")
    text_report.append(f"Total cells flagged for review: {n_review} ({n_review/adata.n_obs*100:.1f}%)")
    text_report.append("")
    text_report.append("SUMMARY BY CELL TYPE:")
    text_report.append("-" * 80)
    text_report.append("")

    for _, row in review_df.iterrows():
        text_report.append(f"Cell Type: {row['Cell_Type']}")
        text_report.append(f"  Total cells: {row['Total_Cells']}")
        text_report.append(f"  Cells needing review: {row['Cells_Needing_Review']} ({row['Percent_Needing_Review']})")
        text_report.append(f"  Low confidence cells: {row['Low_Confidence_Cells']}")
        text_report.append(f"  Spread across {row['N_Clusters_Spanned']} clusters")
        text_report.append(f"  Dominant cluster: {row['Dominant_Cluster']} ({row['Dominant_Cluster_Percent']} of cells)")
        if row['Mixed_Clusters'] != 'None':
            text_report.append(f"  Found in mixed clusters: {row['Mixed_Clusters']}")
        if row['Review_Reasons'] != 'None':
            text_report.append(f"  ⚠️  REASONS FOR REVIEW: {row['Review_Reasons']}")
        text_report.append("")

    # Add cluster-level analysis
    text_report.append("=" * 80)
    text_report.append("CLUSTER-LEVEL ANALYSIS:")
    text_report.append("-" * 80)
    text_report.append("")

    for cluster in adata.obs[CLUSTER_KEY].unique():
        cluster_data = adata.obs[adata.obs[CLUSTER_KEY] == cluster]
        n_cluster_cells = len(cluster_data)

        # Cell type composition
        celltype_counts = cluster_data['celltype_final'].value_counts()
        dominant_type = celltype_counts.index[0]
        dominant_pct = celltype_counts.iloc[0] / n_cluster_cells * 100

        # Check if mixed
        is_mixed = cluster in mixed_cluster_ids
        is_small = n_cluster_cells < 50

        purity = purity_df.loc[cluster, 'purity'] if cluster in purity_df.index else 0

        text_report.append(f"Cluster {cluster}:")
        text_report.append(f"  Size: {n_cluster_cells} cells")
        text_report.append(f"  Purity: {purity:.1%}")
        text_report.append(f"  Dominant cell type: {dominant_type} ({dominant_pct:.1f}%)")

        if is_mixed:
            text_report.append(f"  ⚠️  MIXED CLUSTER (purity < 70%)")
            text_report.append(f"  Cell type composition:")
            for ct, count in celltype_counts.items():
                text_report.append(f"    - {ct}: {count} cells ({count/n_cluster_cells*100:.1f}%)")

        if is_small:
            text_report.append(f"  ⚠️  SMALL CLUSTER (< 50 cells) - possible rare population or artifact")

        text_report.append("")

    # Add recommendations
    text_report.append("=" * 80)
    text_report.append("RECOMMENDATIONS:")
    text_report.append("-" * 80)
    text_report.append("")
    text_report.append("1. PRIORITY ACTIONS:")

    high_priority_types = review_df[review_df['Cells_Needing_Review'].astype(int) > 100]
    if len(high_priority_types) > 0:
        text_report.append("   Cell types with >100 cells needing review:")
        for _, row in high_priority_types.iterrows():
            text_report.append(f"   - {row['Cell_Type']}: Review {row['Review_Reasons']}")

    if len(mixed_clusters) > 0:
        text_report.append(f"\n   Mixed clusters to investigate: {', '.join(map(str, mixed_cluster_ids[:5]))}")
        text_report.append("   → Check marker expression and consider increasing clustering resolution")

    if len(small_clusters) > 0:
        text_report.append(f"\n   Small clusters to validate: {', '.join(map(str, small_cluster_ids[:5]))}")
        text_report.append("   → Verify if these are real rare populations or technical artifacts")

    text_report.append("")
    text_report.append("2. VALIDATION WORKFLOW:")
    text_report.append("   a) For each flagged cell type, check marker expression plots in:")
    text_report.append(f"      {FIGURES_DIR / 'per_celltype'}")
    text_report.append("   b) Compare UMAP spatial distribution with marker expression")
    text_report.append("   c) For mixed clusters, check if cell types should be merged or clusters split")
    text_report.append("   d) Update 'celltype_final' column in AnnData object:")
    text_report.append("      adata.obs.loc[<condition>, 'celltype_final'] = '<new_cell_type>'")
    text_report.append("")

    # Save text report
    text_report_file = OUTPUT_DIR / "manual_review_report.txt"
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_report))

    print(f"Saved detailed text report: {text_report_file}")

    # Print summary to console
    print("\n" + "-"*80)
    print("MANUAL CURATION GUIDE")
    print("-"*80)
    print("\nCells flagged for review include:")
    print(f"  - Low confidence cells: {adata.obs['low_confidence'].sum()}")
    print(f"  - Cells in mixed clusters: {adata.obs[CLUSTER_KEY].isin(mixed_cluster_ids).sum()}")
    print(f"  - Cells in small clusters: {adata.obs[CLUSTER_KEY].isin(small_cluster_ids).sum()}")
    print(f"\nDetailed reports saved to:")
    print(f"  - CSV: {review_file}")
    print(f"  - Text: {text_report_file}")
    print(f"\nTop 5 cell types needing most review:")
    print(review_df[['Cell_Type', 'Total_Cells', 'Cells_Needing_Review', 'Review_Reasons']].head())

else:
    # celltype_final was already set to 'Unknown' in the exception handler
    adata.obs['needs_manual_review'] = True

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save processed data
output_file = OUTPUT_DIR / "adata_annotated.h5ad"
adata.write(output_file)
print(f"\nSaved annotated AnnData object: {output_file}")

# Save cell type assignments
if CELLTYPIST_SUCCESS:
    cell_type_df = adata.obs[[
        CLUSTER_KEY,
        'celltypist_predicted_labels',
        'celltypist_conf_score',
        'low_confidence',
        'needs_manual_review',
        'celltype_final',
        'sample'
    ]].copy()
    cell_type_file = OUTPUT_DIR / "cell_type_assignments.csv"
    cell_type_df.to_csv(cell_type_file)
    print(f"Saved cell type assignments: {cell_type_file}")

# Save summary statistics
if CELLTYPIST_SUCCESS:
    summary = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_cell_types': adata.obs['celltype_final'].nunique(),
        'mean_confidence_score': adata.obs['celltypist_conf_score'].mean(),
        'low_confidence_cells': adata.obs['low_confidence'].sum(),
        'low_confidence_pct': f"{adata.obs['low_confidence'].sum()/adata.n_obs*100:.1f}%",
        'mean_cluster_purity': f"{mean_purity:.2%}",
        'cells_needing_review': n_review,
        'cells_needing_review_pct': f"{n_review/adata.n_obs*100:.1f}%"
    }

    summary_df = pd.DataFrame([summary])
    summary_file = OUTPUT_DIR / "annotation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved annotation summary: {summary_file}")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nFinal dataset:")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")

if CELLTYPIST_SUCCESS:
    print(f"  Cell types identified: {adata.obs['celltype_final'].nunique()}")
    print(f"  Mean cluster purity: {mean_purity:.2%}")
    print(f"  Cells needing manual review: {n_review} ({n_review/adata.n_obs*100:.1f}%)")
    print(f"\nCell type distribution:")
    print(adata.obs['celltype_final'].value_counts())

print(f"\nOutput saved to: {OUTPUT_DIR}")
