"""
Script 3: Batch Correction and Clustering
Single-cell RNA-seq data preprocessing workflow

Phase 6: Batch Effect Correction
- Check for batch effects (UMAP before correction)
- Apply Harmony batch correction
- Validate correction (UMAP after correction)

Phase 7: Clustering & Visualization
- Compute neighbor graph
- UMAP visualization
- Leiden clustering with multiple resolutions
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GDI object limits
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc
from sklearn.metrics import silhouette_score
from scipy import stats

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/2.HVG+PCA/adata_hvg_pca.h5ad")
OUTPUT_DIR = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Kotliarov/Output/3.batch+clustering")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Batch correction parameters
USE_HARMONY = True  # Use Harmony for batch correction
BATCH_KEY = 'sample'  # Column to correct for

# Neighbor graph parameters
N_NEIGHBORS = 15  # Number of neighbors (15-30)
N_PCS = 40  # Number of PCs to use (from Phase 5)

# UMAP parameters
MIN_DIST = 0.3  # Minimum distance (0.3-0.5)
N_COMPONENTS = 2  # Number of UMAP components

# Clustering parameters
LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.8, 1.0]  # Test different resolutions

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
print(f"PCs available: {adata.obsm['X_pca'].shape[1]}")

# ============================================================================
# PHASE 6: BATCH EFFECT CORRECTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: BATCH EFFECT CORRECTION")
print("="*80)

# ----------------------------------------------------------------------------
# 6.1 Check for batch effects BEFORE correction
# ----------------------------------------------------------------------------

print("\n[Step 1/3] Checking for batch effects (BEFORE correction)...")

# Compute neighbors on uncorrected PCA
print(f"Computing neighbor graph (n_neighbors={N_NEIGHBORS}, n_pcs={N_PCS})...")
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, use_rep='X_pca')

# Compute UMAP before correction
print("Computing UMAP before correction...")
sc.tl.umap(adata, min_dist=MIN_DIST)

# Store uncorrected UMAP
adata.obsm['X_umap_uncorrected'] = adata.obsm['X_umap'].copy()

print("UMAP before correction complete")

# Visualize batch effects
print("\nVisualizing batch effects BEFORE correction...")

# Count number of samples for conditional legend handling
n_samples = adata.obs['sample'].nunique()
print(f"  Number of samples: {n_samples}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Batch Effects - BEFORE Correction', fontsize=16, y=1.00)

# UMAP colored by sample (conditional legend for many samples)
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False, title='UMAP - Sample (before)')
else:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False,
               title=f'UMAP - Sample (before, n={n_samples})', legend_loc=None)

# UMAP colored by n_genes
sc.pl.umap(adata, color='n_genes', ax=axes[0, 1], show=False, title='Colored by n_genes')

# UMAP colored by n_counts
sc.pl.umap(adata, color='n_counts', ax=axes[0, 2], show=False, title='Colored by n_counts')

# UMAP colored by pct_counts_mt
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 0], show=False, title='Colored by % Mitochondrial')

# PCA colored by sample (conditional legend)
if n_samples <= 30:
    sc.pl.pca(adata, color='sample', ax=axes[1, 1], show=False, title='PCA - Sample (before)')
else:
    sc.pl.pca(adata, color='sample', ax=axes[1, 1], show=False,
              title=f'PCA - Sample (before, n={n_samples})', legend_loc=None)

# PCA colored by n_genes
sc.pl.pca(adata, color='n_genes', ax=axes[1, 2], show=False, title='PCA colored by n_genes')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_before_correction.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '01_before_correction.png'}")

# ============================================================================
# COMPUTE BATCH CORRECTION METRICS - BEFORE
# ============================================================================

print("\n" + "="*80)
print("BATCH CORRECTION METRICS - BEFORE")
print("="*80)

print("\n[Metrics] Computing batch correction metrics BEFORE Harmony...")

# Silhouette Score per Batch (BEFORE)
print("  Computing Silhouette score (BEFORE)...")
# Use PCA embeddings (subsample if needed for performance)
if adata.n_obs > 100000:
    # Subsample to 100k cells for faster computation
    print(f"    Subsampling to 100k cells (from {adata.n_obs}) for performance...")
    subsample_idx = np.random.choice(adata.n_obs, 100000, replace=False)
    X_pca_subset = adata.obsm['X_pca'][:, :N_PCS][subsample_idx]
    batch_subset = adata.obs['sample'].iloc[subsample_idx]
    silhouette_before = silhouette_score(X_pca_subset, batch_subset)
else:
    silhouette_before = silhouette_score(adata.obsm['X_pca'][:, :N_PCS], adata.obs['sample'])

print(f"  Silhouette score (BEFORE): {silhouette_before:.4f}")
print("  (Higher score = more batch separation, worse batch mixing)")

# PCA Variance Explained by Batch (BEFORE)
print("\n  Computing variance explained by batch (BEFORE)...")
# Perform ANOVA on each PC to see how much batch explains
batch_variance_before = []
for pc in range(min(10, N_PCS)):  # Test first 10 PCs
    pc_values = adata.obsm['X_pca'][:, pc]
    groups = [pc_values[adata.obs['sample'] == s] for s in adata.obs['sample'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    # Calculate eta-squared (effect size)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(pc_values))**2 for g in groups)
    ss_total = np.sum((pc_values - np.mean(pc_values))**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    batch_variance_before.append(eta_squared)

mean_batch_var_before = np.mean(batch_variance_before)
print(f"  Mean variance explained by batch in top 10 PCs (BEFORE): {mean_batch_var_before:.4f}")
print("  (Higher = more batch effect)")

# ----------------------------------------------------------------------------
# 6.2 Apply Harmony batch correction
# ----------------------------------------------------------------------------

print("\n[Step 2/3] Applying Harmony batch correction...")

if USE_HARMONY:
    try:
        import harmonypy as hm

        print(f"Correcting for batch key: '{BATCH_KEY}'")
        print(f"Using {N_PCS} PCs for correction")

        # Prepare input: Harmony expects (PCs x Cells)
        # Scanpy stores (Cells x PCs), so we must transpose
        pca_input = adata.obsm['X_pca'][:, :N_PCS].T
        print(f"Harmony input shape: {pca_input.shape}")

        # Run Harmony
        ho = hm.run_harmony(
            pca_input,
            adata.obs,
            BATCH_KEY,
            max_iter_harmony=10
        )

        # Process output: Check shape and transpose if necessary
        harmony_out = ho.Z_corr
        print(f"Harmony raw output shape: {harmony_out.shape}")
        
        # Ensure it matches standard numpy array format
        if not isinstance(harmony_out, np.ndarray):
            harmony_out = np.array(harmony_out)
            
        # Scanpy expects (Cells x PCs), so shape[0] must be n_obs
        if harmony_out.shape[0] != adata.n_obs:
            print(f"Transposing output from {harmony_out.shape} to {(harmony_out.shape[1], harmony_out.shape[0])}")
            harmony_out = harmony_out.T
            
        if harmony_out.shape[0] != adata.n_obs:
             print(f"ERROR: Even after transpose, shape {harmony_out.shape} does not match cells {adata.n_obs}")
        
        # Store Harmony-corrected embeddings
        adata.obsm['X_pca_harmony'] = harmony_out

        print("Harmony correction complete")

    except Exception as e:
        print(f"WARNING: Harmony correction failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to uncorrected PCA.")
        USE_HARMONY = False
        adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'][:, :N_PCS].copy()
else:
    print("Batch correction disabled")
    adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'][:, :N_PCS].copy()

# ----------------------------------------------------------------------------
# 6.3 Validate batch correction
# ----------------------------------------------------------------------------

print("\n[Step 3/3] Validating batch correction (AFTER correction)...")

# Compute neighbors on corrected embeddings
print(f"Computing neighbor graph on corrected data...")
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, use_rep='X_pca_harmony')

# Compute UMAP after correction
print("Computing UMAP after correction...")
sc.tl.umap(adata, min_dist=MIN_DIST)

print("UMAP after correction complete")

# Visualize after correction
print("\nVisualizing AFTER batch correction...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Batch Effects - AFTER Harmony Correction', fontsize=16, y=1.00)

# UMAP colored by sample (conditional legend)
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False, title='UMAP - Sample (after)')
else:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False,
               title=f'UMAP - Sample (after, n={n_samples})', legend_loc=None)

# UMAP colored by n_genes
sc.pl.umap(adata, color='n_genes', ax=axes[0, 1], show=False, title='Colored by n_genes')

# UMAP colored by n_counts
sc.pl.umap(adata, color='n_counts', ax=axes[0, 2], show=False, title='Colored by n_counts')

# UMAP colored by pct_counts_mt
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 0], show=False, title='Colored by % Mitochondrial')

# PCA colored by sample (on corrected data, conditional legend)
# Note: For PCA visualization, we need to temporarily use the corrected embedding
adata_temp = adata.copy()
adata_temp.obsm['X_pca'] = adata.obsm['X_pca_harmony']
if n_samples <= 30:
    sc.pl.pca(adata_temp, color='sample', ax=axes[1, 1], show=False, title='Harmony PCA - Sample (after)')
else:
    sc.pl.pca(adata_temp, color='sample', ax=axes[1, 1], show=False,
              title=f'Harmony PCA - Sample (after, n={n_samples})', legend_loc=None)
sc.pl.pca(adata_temp, color='n_genes', ax=axes[1, 2], show=False, title='Harmony PCA colored by n_genes')
del adata_temp

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_after_correction.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '02_after_correction.png'}")

# Save the corrected UMAP for later use
adata.obsm['X_umap_corrected'] = adata.obsm['X_umap'].copy()

# ============================================================================
# COMPUTE BATCH CORRECTION METRICS - AFTER
# ============================================================================

print("\n" + "="*80)
print("BATCH CORRECTION METRICS - AFTER")
print("="*80)

print("\n[Metrics] Computing batch correction metrics AFTER Harmony...")

# Silhouette Score (AFTER)
print("  Computing Silhouette score (AFTER)...")
if adata.n_obs > 100000:
    print(f"    Subsampling to 100k cells (from {adata.n_obs}) for performance...")
    subsample_idx = np.random.choice(adata.n_obs, 100000, replace=False)
    X_harmony_subset = adata.obsm['X_pca_harmony'][subsample_idx]
    batch_subset = adata.obs['sample'].iloc[subsample_idx]
    silhouette_after = silhouette_score(X_harmony_subset, batch_subset)
else:
    silhouette_after = silhouette_score(adata.obsm['X_pca_harmony'], adata.obs['sample'])

print(f"  Silhouette score (AFTER): {silhouette_after:.4f}")
print(f"  Change: {silhouette_after - silhouette_before:+.4f}")
print("  (Lower AFTER = better batch mixing)")

# PCA Variance Explained by Batch (AFTER)
print("\n  Computing variance explained by batch (AFTER)...")
batch_variance_after = []
for pc in range(min(10, adata.obsm['X_pca_harmony'].shape[1])):
    pc_values = adata.obsm['X_pca_harmony'][:, pc]
    groups = [pc_values[adata.obs['sample'] == s] for s in adata.obs['sample'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(pc_values))**2 for g in groups)
    ss_total = np.sum((pc_values - np.mean(pc_values))**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    batch_variance_after.append(eta_squared)

mean_batch_var_after = np.mean(batch_variance_after)
print(f"  Mean variance explained by batch in top 10 PCs (AFTER): {mean_batch_var_after:.4f}")
print(f"  Change: {mean_batch_var_after - mean_batch_var_before:+.4f}")
print("  (Lower AFTER = better batch correction)")

# Print summary
print("\n" + "-"*80)
print("BATCH CORRECTION METRICS SUMMARY")
print("-"*80)
print(f"Silhouette Score:")
print(f"  BEFORE: {silhouette_before:.4f}")
print(f"  AFTER:  {silhouette_after:.4f}")
print(f"  Change: {silhouette_after - silhouette_before:+.4f} {'✓ Improved' if silhouette_after < silhouette_before else '✗ Worse'}")
print(f"\nVariance Explained by Batch (mean of top 10 PCs):")
print(f"  BEFORE: {mean_batch_var_before:.4f}")
print(f"  AFTER:  {mean_batch_var_after:.4f}")
print(f"  Change: {mean_batch_var_after - mean_batch_var_before:+.4f} {'✓ Improved' if mean_batch_var_after < mean_batch_var_before else '✗ Worse'}")

# ----------------------------------------------------------------------------
# Visualize batch variance comparison
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("PLOTTING BATCH METRICS COMPARISON")
print("-"*80)

print("\nPlotting batch variance comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of variance explained per PC
pcs = range(1, len(batch_variance_before) + 1)
width = 0.35
x = np.arange(len(pcs))

axes[0].bar(x - width/2, batch_variance_before, width, label='Before Harmony', alpha=0.8)
axes[0].bar(x + width/2, batch_variance_after, width, label='After Harmony', alpha=0.8)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained by Batch (η²)')
axes[0].set_title('Batch Effect per PC (Before vs After)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(pcs)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Summary metrics
metrics_data = {
    'Silhouette Score': [silhouette_before, silhouette_after],
    'Mean Batch Variance': [mean_batch_var_before, mean_batch_var_after]
}

x_pos = np.arange(len(metrics_data))
width = 0.35

for i, (metric_name, values) in enumerate(metrics_data.items()):
    axes[1].bar(i - width/2, values[0], width, label='Before' if i == 0 else '',
                color='C0', alpha=0.8)
    axes[1].bar(i + width/2, values[1], width, label='After' if i == 0 else '',
                color='C1', alpha=0.8)

axes[1].set_ylabel('Score')
axes[1].set_title('Batch Correction Metrics Summary')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(metrics_data.keys(), rotation=15, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02b_batch_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '02b_batch_metrics_comparison.png'}")

# Compare before and after side by side
print("\nCreating before/after comparison (using pre-computed UMAPs)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle('Batch Correction Comparison', fontsize=16, y=0.995)

# Before correction - colored by sample (conditional legend)
adata.obsm['X_umap'] = adata.obsm['X_umap_uncorrected']
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False, title='BEFORE - Sample')
else:
    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False,
               title=f'BEFORE - Sample (n={n_samples})', legend_loc=None)

# After correction - colored by sample (use pre-computed UMAP)
adata.obsm['X_umap'] = adata.obsm['X_umap_corrected']
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', ax=axes[0, 1], show=False, title='AFTER - Sample')
else:
    sc.pl.umap(adata, color='sample', ax=axes[0, 1], show=False,
               title=f'AFTER - Sample (n={n_samples})', legend_loc=None)

# Before correction - colored by n_genes
adata.obsm['X_umap'] = adata.obsm['X_umap_uncorrected']
sc.pl.umap(adata, color='n_genes', ax=axes[1, 0], show=False, title='BEFORE - Colored by n_genes')

# After correction - colored by n_genes (use pre-computed UMAP)
adata.obsm['X_umap'] = adata.obsm['X_umap_corrected']
sc.pl.umap(adata, color='n_genes', ax=axes[1, 1], show=False, title='AFTER - Colored by n_genes')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_comparison_before_after.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '03_comparison_before_after.png'}")

# Ensure corrected UMAP is active for downstream clustering
adata.obsm['X_umap'] = adata.obsm['X_umap_corrected']

# ============================================================================
# PHASE 7: CLUSTERING & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 7: CLUSTERING & VISUALIZATION")
print("="*80)

# Note: Neighbors and UMAP already computed on corrected data (no need to recompute)
print("\nUsing pre-computed neighbor graph and UMAP for clustering...")

# ----------------------------------------------------------------------------
# 7.1 Leiden clustering with multiple resolutions
# ----------------------------------------------------------------------------

print("\n[Step 1/1] Running Leiden clustering with multiple resolutions...")

for res in LEIDEN_RESOLUTIONS:
    print(f"\nClustering with resolution {res}...")

    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')

    n_clusters = adata.obs[f'leiden_res{res}'].nunique()
    print(f"  Resolution {res}: {n_clusters} clusters")

print("\nClustering complete")

# Print cluster statistics for each resolution
print("\n" + "-"*80)
print("CLUSTER STATISTICS")
print("-"*80)

for res in LEIDEN_RESOLUTIONS:
    cluster_col = f'leiden_res{res}'
    n_clusters = adata.obs[cluster_col].nunique()
    cluster_sizes = adata.obs[cluster_col].value_counts().sort_index()

    print(f"\nResolution {res} ({n_clusters} clusters):")
    print(f"  Mean cluster size: {cluster_sizes.mean():.0f}")
    print(f"  Median cluster size: {cluster_sizes.median():.0f}")
    print(f"  Min cluster size: {cluster_sizes.min()}")
    print(f"  Max cluster size: {cluster_sizes.max()}")

# ----------------------------------------------------------------------------
# 7.2 Visualize clustering results
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("VISUALIZING CLUSTERING RESULTS")
print("-"*80)

# Create UMAP plots for all resolutions
print("\nCreating UMAP plots for all resolutions...")

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('Leiden Clustering - Multiple Resolutions', fontsize=16, y=0.995)

for idx, res in enumerate(LEIDEN_RESOLUTIONS):
    ax = axes[idx // 2, idx % 2]
    cluster_col = f'leiden_res{res}'
    n_clusters = adata.obs[cluster_col].nunique()

    sc.pl.umap(
        adata,
        color=cluster_col,
        ax=ax,
        show=False,
        title=f'Resolution {res} ({n_clusters} clusters)',
        legend_loc='on data',
        legend_fontsize=8
    )

plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_leiden_resolutions.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '04_leiden_resolutions.png'}")

# Visualize sample distribution across clusters for optimal resolution
optimal_res = 0.5
optimal_cluster_col = f'leiden_res{optimal_res}'

print(f"\nVisualizing sample distribution for resolution {optimal_res}...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(f'Leiden Clustering Analysis (Resolution {optimal_res})', fontsize=16, y=0.995)

# UMAP colored by clusters
sc.pl.umap(adata, color=optimal_cluster_col, ax=axes[0, 0], show=False,
           title=f'Clusters (Resolution {optimal_res})', legend_loc='on data')

# UMAP colored by sample (conditional legend)
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', ax=axes[0, 1], show=False, title='Sample')
else:
    sc.pl.umap(adata, color='sample', ax=axes[0, 1], show=False,
               title=f'Sample (n={n_samples})', legend_loc=None)

# Cluster sizes
cluster_sizes = adata.obs[optimal_cluster_col].value_counts().sort_index()
axes[1, 0].bar(range(len(cluster_sizes)), cluster_sizes.values)
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of cells')
axes[1, 0].set_title('Cluster sizes')
axes[1, 0].set_xticks(range(len(cluster_sizes)))
axes[1, 0].set_xticklabels(cluster_sizes.index)

# Sample composition per cluster
sample_cluster = pd.crosstab(adata.obs[optimal_cluster_col], adata.obs['sample'], normalize='index')
sample_cluster.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Proportion')
axes[1, 1].set_title('Sample composition per cluster')
axes[1, 1].legend(title='Sample', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / f'05_clustering_analysis_res{optimal_res}.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / f'05_clustering_analysis_res{optimal_res}.png'}")

# Create individual high-quality UMAP plots
print("\nCreating individual UMAP plots...")

# UMAP by cluster
sc.pl.umap(adata, color=optimal_cluster_col, legend_loc='on data',
           title=f'Leiden Clustering (Resolution {optimal_res})', show=False)
plt.savefig(FIGURES_DIR / '06_umap_clusters.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

# UMAP by sample (conditional legend)
if n_samples <= 30:
    sc.pl.umap(adata, color='sample', title='UMAP colored by Sample', show=False)
else:
    sc.pl.umap(adata, color='sample', title=f'UMAP colored by Sample (n={n_samples}, no legend)',
               show=False, legend_loc=None)
plt.savefig(FIGURES_DIR / '07_umap_sample.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

# UMAP by QC metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='n_genes', ax=axes[0], show=False, title='n_genes')
sc.pl.umap(adata, color='n_counts', ax=axes[1], show=False, title='n_counts')
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[2], show=False, title='% Mitochondrial')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '08_umap_qc_metrics.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '06_umap_clusters.png'}")
print(f"  Saved: {FIGURES_DIR / '07_umap_sample.png'}")
print(f"  Saved: {FIGURES_DIR / '08_umap_qc_metrics.png'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save processed data
output_file = OUTPUT_DIR / "adata_batch_clustered.h5ad"
adata.write(output_file)
print(f"\nSaved processed AnnData object: {output_file}")

# Save cluster assignments for each resolution
for res in LEIDEN_RESOLUTIONS:
    cluster_col = f'leiden_res{res}'
    cluster_df = adata.obs[[cluster_col, 'sample']].copy()
    cluster_file = OUTPUT_DIR / f"leiden_clusters_res{res}.csv"
    cluster_df.to_csv(cluster_file)
    print(f"Saved cluster assignments (res {res}): {cluster_file}")

# Save batch correction metrics
batch_metrics = pd.DataFrame({
    'metric': ['silhouette_score', 'mean_batch_variance_top10pcs'],
    'before_correction': [silhouette_before, mean_batch_var_before],
    'after_correction': [silhouette_after, mean_batch_var_after],
    'change': [silhouette_after - silhouette_before,
               mean_batch_var_after - mean_batch_var_before],
    'interpretation': [
        'Lower is better (less batch separation)',
        'Lower is better (batch explains less variance)'
    ]
})

batch_metrics_file = OUTPUT_DIR / "batch_correction_metrics.csv"
batch_metrics.to_csv(batch_metrics_file, index=False)
print(f"Saved batch correction metrics: {batch_metrics_file}")

# Save clustering statistics
clustering_stats = []
for res in LEIDEN_RESOLUTIONS:
    cluster_col = f'leiden_res{res}'
    n_clusters = adata.obs[cluster_col].nunique()
    cluster_sizes = adata.obs[cluster_col].value_counts()

    clustering_stats.append({
        'resolution': res,
        'n_clusters': n_clusters,
        'mean_cluster_size': cluster_sizes.mean(),
        'median_cluster_size': cluster_sizes.median(),
        'min_cluster_size': cluster_sizes.min(),
        'max_cluster_size': cluster_sizes.max()
    })

clustering_stats_df = pd.DataFrame(clustering_stats)
clustering_stats_file = OUTPUT_DIR / "clustering_statistics.csv"
clustering_stats_df.to_csv(clustering_stats_file, index=False)
print(f"Saved clustering statistics: {clustering_stats_file}")

# Save processing summary
summary = {
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'n_samples': adata.obs['sample'].nunique(),
    'batch_correction_method': 'Harmony' if USE_HARMONY else 'None',
    'batch_key': BATCH_KEY,
    'n_neighbors': N_NEIGHBORS,
    'n_pcs_used': N_PCS,
    'silhouette_before': f"{silhouette_before:.4f}",
    'silhouette_after': f"{silhouette_after:.4f}",
    'silhouette_improvement': f"{silhouette_before - silhouette_after:.4f}",
    'batch_variance_before': f"{mean_batch_var_before:.4f}",
    'batch_variance_after': f"{mean_batch_var_after:.4f}",
    'batch_variance_reduction': f"{mean_batch_var_before - mean_batch_var_after:.4f}",
    'umap_min_dist': MIN_DIST,
    'leiden_resolutions_tested': ', '.join(map(str, LEIDEN_RESOLUTIONS)),
    'recommended_resolution': optimal_res
}

summary_df = pd.DataFrame([summary])
summary_file = OUTPUT_DIR / "processing_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Saved processing summary: {summary_file}")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nFinal dataset:")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Samples: {adata.obs['sample'].nunique()}")
print(f"  Batch correction: {'Harmony' if USE_HARMONY else 'None'}")
print(f"  Clustering resolutions tested: {LEIDEN_RESOLUTIONS}")
print(f"\nRecommended resolution for downstream analysis: {optimal_res}")
print(f"Number of clusters at resolution {optimal_res}: {adata.obs[optimal_cluster_col].nunique()}")
print(f"\nOutput saved to: {OUTPUT_DIR}")
