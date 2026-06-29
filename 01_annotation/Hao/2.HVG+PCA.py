"""
Script 2: Highly Variable Genes Selection and PCA
Single-cell RNA-seq data preprocessing workflow

Phase 4: Normalization & Feature Selection
- Normalize counts (target_sum=1e4)
- Log-transform
- Select Highly Variable Genes (HVGs)

Phase 5: Dimensionality Reduction
- Scale data
- PCA
- Determine optimal number of PCs
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

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/1.QC+doublet_detection/adata_qc_filtered.h5ad"
OUTPUT_DIR = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/2.HVG+PCA")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Normalization parameters
TARGET_SUM = 1e4  # 10,000

# Feature selection parameters
N_TOP_GENES = 2000  # Number of highly variable genes to select

# PCA parameters
N_PCS = 50  # Number of principal components to compute

# Scaling parameters (optional)
REGRESS_OUT = False  # Set to True to regress out confounders
REGRESS_VARS = ['pct_counts_mt', 'n_counts']  # Variables to regress out

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

# Store raw counts in a layer before normalization
adata.layers['counts'] = adata.X.copy()

# ============================================================================
# PHASE 4: NORMALIZATION & FEATURE SELECTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: NORMALIZATION & FEATURE SELECTION")
print("="*80)

# ----------------------------------------------------------------------------
# 4.1 Normalize counts
# ----------------------------------------------------------------------------

print("\n[Step 1/3] Normalizing counts...")
print(f"Target sum: {TARGET_SUM}")

sc.pp.normalize_total(adata, target_sum=TARGET_SUM)

print("Normalization complete")

# ----------------------------------------------------------------------------
# 4.2 Log-transform
# ----------------------------------------------------------------------------

print("\n[Step 2/3] Log-transforming...")

sc.pp.log1p(adata)

print("Log-transformation complete")

# ----------------------------------------------------------------------------
# 4.3 Select Highly Variable Genes
# ----------------------------------------------------------------------------

print("\n[Step 3/3] Selecting highly variable genes...")
print(f"Number of top genes: {N_TOP_GENES}")

# Calculate highly variable genes on raw counts (seurat_v3 requires raw counts)
# Note: batch_key removed due to numerical instability with many samples
# Batch effects will be handled during integration if needed
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_TOP_GENES,
    flavor='seurat_v3',
    layer='counts'  # Use raw counts from the saved layer
)

n_hvg = adata.var['highly_variable'].sum()
print(f"\nHighly variable genes selected: {n_hvg}")
print(f"Non-variable genes: {adata.n_vars - n_hvg}")

# Plot highly variable genes
print("\nPlotting highly variable genes...")

# Increase figure width to prevent label overlap
fig = plt.figure(figsize=(14, 5))  # Wider figure
sc.pl.highly_variable_genes(adata, show=False)
plt.tight_layout(pad=2.0)  # Add padding between subplots
plt.savefig(FIGURES_DIR / '01_highly_variable_genes.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '01_highly_variable_genes.png'}")

# Show top variable genes
print("\nTop 20 highly variable genes:")
top_hvgs = adata.var[adata.var['highly_variable']].sort_values('highly_variable_rank').head(20)

# Select available columns for display
available_cols = ['highly_variable_rank']
for col in ['means', 'variances', 'variances_norm', 'dispersions_norm']:
    if col in top_hvgs.columns:
        available_cols.append(col)

print(top_hvgs[available_cols].to_string())

# ============================================================================
# PHASE 5: DIMENSIONALITY REDUCTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: DIMENSIONALITY REDUCTION")
print("="*80)

# ----------------------------------------------------------------------------
# 5.1 Scale data
# ----------------------------------------------------------------------------

print("\n[Step 1/3] Scaling data (HVGs only to save memory)...")

# Store normalized/log-transformed data before scaling
adata.layers['log1p_norm'] = adata.X.copy()

# Create a copy with only HVGs for scaling (to avoid densifying entire matrix)
print(f"Subsetting to {n_hvg} highly variable genes for scaling...")
adata_hvg = adata[:, adata.var['highly_variable']].copy()

print(f"Memory-efficient subset created: {adata_hvg.shape[0]} cells × {adata_hvg.shape[1]} genes")

if REGRESS_OUT:
    print(f"Regressing out: {', '.join(REGRESS_VARS)}")
    sc.pp.regress_out(adata_hvg, REGRESS_VARS)
    print("Regression complete")

# Scale to zero mean and unit variance (only HVGs - much more memory efficient)
print("Scaling HVGs to zero mean and unit variance...")
sc.pp.scale(adata_hvg, max_value=10)

print("Scaling complete")

# ----------------------------------------------------------------------------
# 5.2 PCA
# ----------------------------------------------------------------------------

print("\n[Step 2/3] Running PCA on scaled HVGs...")
print(f"Number of components: {N_PCS}")
print(f"Using {n_hvg} highly variable genes")

# Run PCA on the scaled HVG subset
sc.tl.pca(adata_hvg, n_comps=N_PCS)

print("PCA complete")

# Copy PCA results back to main object
print("Copying PCA results to main object...")
adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
adata.varm['PCs'] = np.zeros((adata.n_vars, N_PCS))
adata.varm['PCs'][adata.var['highly_variable'], :] = adata_hvg.varm['PCs']
adata.uns['pca'] = adata_hvg.uns['pca']

print("PCA results copied successfully")

# Clean up the HVG subset to free memory
del adata_hvg
gc.collect()
print("HVG subset cleaned up to free memory")

# Print variance explained
variance_ratio = adata.uns['pca']['variance_ratio']
print(f"\nVariance explained by first 10 PCs:")
for i in range(min(10, len(variance_ratio))):
    print(f"  PC{i+1}: {variance_ratio[i]*100:.2f}%")

cumsum_variance = np.cumsum(variance_ratio)
print(f"\nCumulative variance explained:")
print(f"  First 10 PCs: {cumsum_variance[9]*100:.2f}%")
print(f"  First 20 PCs: {cumsum_variance[19]*100:.2f}%")
print(f"  First 30 PCs: {cumsum_variance[29]*100:.2f}%")
print(f"  First 40 PCs: {cumsum_variance[39]*100:.2f}%")
print(f"  First 50 PCs: {cumsum_variance[49]*100:.2f}%")

# ----------------------------------------------------------------------------
# 5.3 Determine optimal number of PCs
# ----------------------------------------------------------------------------

print("\n[Step 3/3] Determining optimal number of PCs...")

# Create elbow plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Variance ratio per PC
axes[0].plot(range(1, len(variance_ratio) + 1), variance_ratio, 'o-')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Ratio')
axes[0].set_title('Variance Explained per PC')
axes[0].axvline(x=30, color='red', linestyle='--', linewidth=1, label='PC 30')
axes[0].axvline(x=40, color='orange', linestyle='--', linewidth=1, label='PC 40')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cumulative variance
axes[1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'o-')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Cumulative Variance Ratio')
axes[1].set_title('Cumulative Variance Explained')
axes[1].axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='80%')
axes[1].axhline(y=0.9, color='blue', linestyle='--', linewidth=1, label='90%')
axes[1].axvline(x=30, color='red', linestyle='--', linewidth=1, label='PC 30')
axes[1].axvline(x=40, color='orange', linestyle='--', linewidth=1, label='PC 40')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Elbow plot (log scale)
axes[2].plot(range(1, len(variance_ratio) + 1), variance_ratio, 'o-')
axes[2].set_xlabel('Principal Component')
axes[2].set_ylabel('Variance Ratio (log scale)')
axes[2].set_title('Elbow Plot (log scale)')
axes[2].set_yscale('log')
axes[2].axvline(x=30, color='red', linestyle='--', linewidth=1, label='PC 30')
axes[2].axvline(x=40, color='orange', linestyle='--', linewidth=1, label='PC 40')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_pca_elbow_plot.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '02_pca_elbow_plot.png'}")

# Find elbow point using simple method (where variance drops below threshold)
threshold_variance = 0.01  # 1% variance
elbow_point = np.where(variance_ratio < threshold_variance)[0]
if len(elbow_point) > 0:
    elbow_point = elbow_point[0] + 1
    print(f"\nSuggested elbow point (variance < {threshold_variance*100}%): PC {elbow_point}")
else:
    print(f"\nNo clear elbow point found (all PCs explain >{threshold_variance*100}% variance)")

# Visualize top PCs
print("\nVisualizing top principal components...")

# PCA variance plot (scanpy built-in)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50, show=False)
plt.savefig(FIGURES_DIR / '03_pca_variance_ratio.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '03_pca_variance_ratio.png'}")

# Plot PCA loadings for top PCs
print("\nPlotting PCA loadings...")

sc.pl.pca_loadings(adata, components='1,2,3,4', show=False)
plt.savefig(FIGURES_DIR / '04_pca_loadings.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '04_pca_loadings.png'}")

# Scatter plot of first 2 PCs colored by QC metrics
# Note: With 282 samples, coloring by sample is not informative
print("\nPlotting PCA scatter (colored by QC metrics)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Flatten axes for easier indexing
axes = axes.flatten()

sc.pl.pca(adata, color='n_genes', ax=axes[0], show=False, title='PCA colored by n_genes', legend_loc='right margin')
sc.pl.pca(adata, color='n_counts', ax=axes[1], show=False, title='PCA colored by n_counts', legend_loc='right margin')
sc.pl.pca(adata, color='pct_counts_mt', ax=axes[2], show=False, title='PCA colored by % mito', legend_loc='right margin')

# Check if metadata columns exist for additional coloring
if 'pct_counts_ribo' in adata.obs.columns:
    sc.pl.pca(adata, color='pct_counts_ribo', ax=axes[3], show=False, title='PCA colored by % ribo', legend_loc='right margin')
else:
    # If no ribo data, just show a simple PCA without coloring
    sc.pl.pca(adata, ax=axes[3], show=False, title='PCA (uncolored)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_pca_scatter.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '05_pca_scatter.png'}")

# Scatter plot colored by sample (useful for datasets with fewer samples)
n_samples = adata.obs['sample'].nunique()
print(f"\nPlotting PCA scatter colored by sample ({n_samples} samples)...")

if n_samples <= 30:
    # Reasonable number of samples - create plot with legend
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.pca(adata, color='sample', ax=ax, show=False, title=f'PCA colored by sample (n={n_samples})')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05b_pca_scatter_by_sample.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()
    print(f"  Saved: {FIGURES_DIR / '05b_pca_scatter_by_sample.png'}")
else:
    # Too many samples - create plot without legend
    print(f"  Note: {n_samples} samples detected - plot will be created without legend")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.pca(adata, color='sample', ax=ax, show=False, title=f'PCA colored by sample (n={n_samples})',
              legend_loc=None, palette='tab20')  # No legend for many samples
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05b_pca_scatter_by_sample.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()
    print(f"  Saved: {FIGURES_DIR / '05b_pca_scatter_by_sample.png'} (no legend - too many samples)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save processed data
output_file = OUTPUT_DIR / "adata_hvg_pca.h5ad"
adata.write(output_file)
print(f"\nSaved processed AnnData object: {output_file}")

# Save highly variable genes list
hvg_df = adata.var[adata.var['highly_variable']].sort_values('highly_variable_rank')
hvg_file = OUTPUT_DIR / "highly_variable_genes.csv"
hvg_df.to_csv(hvg_file)
print(f"Saved highly variable genes list: {hvg_file}")

# Save PCA variance statistics
pca_stats = pd.DataFrame({
    'PC': range(1, len(variance_ratio) + 1),
    'variance_ratio': variance_ratio,
    'cumulative_variance': cumsum_variance
})
pca_stats_file = OUTPUT_DIR / "pca_variance_stats.csv"
pca_stats.to_csv(pca_stats_file, index=False)
print(f"Saved PCA variance statistics: {pca_stats_file}")

# Save summary
summary = {
    'n_cells': adata.n_obs,
    'n_genes_total': adata.n_vars,
    'n_highly_variable_genes': n_hvg,
    'n_pcs_computed': N_PCS,
    'variance_explained_pc30': f"{cumsum_variance[29]*100:.2f}%",
    'variance_explained_pc40': f"{cumsum_variance[39]*100:.2f}%",
    'variance_explained_pc50': f"{cumsum_variance[49]*100:.2f}%",
    'regressed_out': REGRESS_OUT,
    'regress_vars': ', '.join(REGRESS_VARS) if REGRESS_OUT else 'None'
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
print(f"  Total genes: {adata.n_vars}")
print(f"  Highly variable genes: {n_hvg}")
print(f"  Principal components: {N_PCS}")
print(f"  Samples: {adata.obs['sample'].nunique()}")
print(f"\nRecommended number of PCs for downstream analysis: 30-40")
print(f"\nOutput saved to: {OUTPUT_DIR}")
