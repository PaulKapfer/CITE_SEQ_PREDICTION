"""
Script 1: Quality Control and Doublet Detection
Single-cell RNA-seq data preprocessing workflow

Phase 2: Quality Control
- Calculate QC metrics (n_genes, n_counts, % mito, % ribo, % hemoglobin)
- Filter cells and genes based on quality thresholds
- Filter samples based on minimum cell count

Phase 3: Doublet Detection & Removal
- Run Scrublet per sample
- Mark and remove doublets
- Re-check sample sizes
"""

import scanpy as sc
import numpy as np
import pandas as pd
import scrublet as scr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GDI object limits
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csc_matrix
import anndata as ad
import h5py
import gc

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Hao 2021 CITE-seq PBMC reference, exported as an .h5seurat file.
# Contains an SCT (RNA) assay and an ADT (protein/CITE-seq) assay.
INPUT_FILE = "C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/Input/multi.h5seurat"

OUTPUT_DIR = Path("C:/Users/Paul/Desktop/Publications/CITE-SEQ_pred/Annotation/Hao/1.QC+doublet_detection")
FIGURES_DIR = OUTPUT_DIR / "figures" / "qc"

# Assay names inside the .h5seurat file
RNA_ASSAY = "SCT"   # RNA counts used for the annotation pipeline
ADT_ASSAY = "ADT"   # CITE-seq protein, carried along for downstream prediction

# Metadata column to treat as the biological "sample" (used for per-sample
# doublet detection and, later, batch correction). orig.ident = donor x timepoint.
SAMPLE_KEY = "orig.ident"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# QC thresholds
MIN_GENES = 200
MAX_GENES = 6000
MAX_PCT_MITO = 10  # percentage
MIN_CELLS_PER_GENE = 3
MIN_CELLS_PER_SAMPLE = 100

# Doublet detection parameters
EXPECTED_DOUBLET_RATE = 0.06  # 6% for 10X data

# ============================================================================
# PHASE 1: LOAD DATA
# ============================================================================

def _decode(arr):
    """Decode an array of (possibly bytes) strings to python str."""
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in arr])


def _read_sparse_counts(assay_grp):
    """Read a Seurat 'counts' slot (CSC, stored as genes x cells) and return a
    CSR matrix of shape (cells, genes) in float32."""
    cg = assay_grp['counts']
    dims = cg.attrs['dims']  # [n_genes, n_cells]
    n_genes, n_cells = int(dims[0]), int(dims[1])
    mat = csc_matrix(
        (cg['data'][:].astype(np.float32), cg['indices'][:], cg['indptr'][:]),
        shape=(n_genes, n_cells)
    )
    return mat.T.tocsr()  # -> (cells, genes)


def load_and_merge_data():
    """Load the Hao .h5seurat: RNA (SCT) counts as X, cell metadata into obs,
    and the ADT (CITE-seq protein) assay into obsm so it is carried through the
    whole pipeline."""

    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"\nReading h5seurat: {INPUT_FILE}")
    print("This may take a few minutes for large datasets...")

    f = h5py.File(INPUT_FILE, "r")

    # ----------------------------------------------------------------------------
    # RNA assay (SCT) -> AnnData.X (raw-style counts for the QC pipeline)
    # ----------------------------------------------------------------------------
    print(f"\nLoading RNA assay '{RNA_ASSAY}' counts...")
    rna_grp = f['assays'][RNA_ASSAY]
    X = _read_sparse_counts(rna_grp)
    genes = _decode(rna_grp['features'][:])
    cells = _decode(f['cell.names'][:])

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = cells
    adata.var_names_make_unique()
    print(f"  RNA matrix: {adata.n_obs} cells x {adata.n_vars} genes")

    # ----------------------------------------------------------------------------
    # Cell metadata -> obs
    # ----------------------------------------------------------------------------
    print("\nLoading cell metadata...")
    md = f['meta.data']
    meta_cols = [c.decode() if isinstance(c, bytes) else c
                 for c in md.attrs['colnames']]
    obs = {}
    for col in meta_cols:
        if col not in md:
            continue
        vals = md[col][:]
        if vals.dtype == object:
            obs[col] = _decode(vals)
        else:
            obs[col] = vals
    obs_df = pd.DataFrame(obs, index=adata.obs_names)
    adata.obs = obs_df
    print(f"  Loaded {len(obs_df.columns)} metadata columns: {list(obs_df.columns)}")

    # ----------------------------------------------------------------------------
    # ADT assay (CITE-seq protein) -> obsm (raw counts + CLR-normalized)
    # Stored as DataFrames so protein names are preserved and the matrices are
    # automatically subset alongside cells during QC / doublet filtering.
    # ----------------------------------------------------------------------------
    if ADT_ASSAY in f['assays']:
        print(f"\nLoading ADT assay '{ADT_ASSAY}' (CITE-seq protein)...")
        adt_grp = f['assays'][ADT_ASSAY]
        prot_names = _decode(adt_grp['features'][:])

        # Raw protein counts (sparse, genes x cells) -> dense cells x proteins
        adt_counts = _read_sparse_counts(adt_grp).toarray()
        adata.obsm['protein_counts'] = pd.DataFrame(
            adt_counts, index=adata.obs_names, columns=prot_names)

        # CLR-normalized protein matrix ('data' slot, stored cells x proteins)
        if 'data' in adt_grp and isinstance(adt_grp['data'], h5py.Dataset):
            adt_data = adt_grp['data'][:].astype(np.float32)
            if adt_data.shape[1] != len(prot_names) and adt_data.shape[0] == len(prot_names):
                adt_data = adt_data.T
            adata.obsm['protein_clr'] = pd.DataFrame(
                adt_data, index=adata.obs_names, columns=prot_names)

        print(f"  ADT matrix: {adata.n_obs} cells x {len(prot_names)} proteins "
              f"(stored in obsm['protein_counts']"
              f"{', protein_clr' if 'protein_clr' in adata.obsm else ''})")
    else:
        print(f"\nWARNING: ADT assay '{ADT_ASSAY}' not found in file.")

    f.close()

    # ----------------------------------------------------------------------------
    # Define the 'sample' column used downstream
    # ----------------------------------------------------------------------------
    if SAMPLE_KEY in adata.obs.columns:
        adata.obs['sample'] = adata.obs[SAMPLE_KEY].astype(str)
        print(f"\nUsing '{SAMPLE_KEY}' as 'sample' column for downstream analysis")
    else:
        adata.obs['sample'] = 'Hao_PBMC'
        print(f"\nWARNING: '{SAMPLE_KEY}' not found, using single sample 'Hao_PBMC'")

    print("\nSample distribution:")
    sample_counts = adata.obs['sample'].value_counts().sort_index()
    if len(sample_counts) > 20:
        print(sample_counts.head(20))
        print(f"... and {len(sample_counts) - 20} more samples")
    else:
        print(sample_counts)

    print("\n" + "-"*80)
    print(f"Total cells: {adata.n_obs}")
    print(f"Total genes: {adata.n_vars}")
    print(f"Total samples: {adata.obs['sample'].nunique()}")

    return adata

# Load data using the fast loading function
adata = load_and_merge_data()

# ============================================================================
# PHASE 2: QUALITY CONTROL
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: QUALITY CONTROL")
print("="*80)

# ----------------------------------------------------------------------------
# 2.1 Calculate QC metrics
# ----------------------------------------------------------------------------

print("\n[Step 1/4] Calculating QC metrics...")

# Identify mitochondrial genes (MT-)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
# Identify ribosomal genes (RPS, RPL)
adata.var['ribo'] = adata.var_names.str.match('^RP[SL]')
# Identify hemoglobin genes (HB)
adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')

# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'ribo', 'hb'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Add n_genes and n_counts to obs (if not already there)
adata.obs['n_genes'] = adata.obs['n_genes_by_counts']
adata.obs['n_counts'] = adata.obs['total_counts']

print(f"\nQC metrics calculated:")
print(f"  - n_genes per cell: mean={adata.obs['n_genes'].mean():.0f}")
print(f"  - n_counts per cell: mean={adata.obs['n_counts'].mean():.0f}")
print(f"  - % mitochondrial: mean={adata.obs['pct_counts_mt'].mean():.2f}%")
print(f"  - % ribosomal: mean={adata.obs['pct_counts_ribo'].mean():.2f}%")
print(f"  - % hemoglobin: mean={adata.obs['pct_counts_hb'].mean():.2f}%")

# ----------------------------------------------------------------------------
# 2.2 Plot QC metrics before filtering
# ----------------------------------------------------------------------------

print("\n[Step 2/4] Plotting QC metrics before filtering...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('QC Metrics - Before Filtering', fontsize=16, y=1.02)

# n_genes violin plot
sc.pl.violin(adata, 'n_genes', ax=axes[0, 0], show=False)
axes[0, 0].axhline(y=MIN_GENES, color='red', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=MAX_GENES, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title(f'n_genes (filter: {MIN_GENES}-{MAX_GENES})')

# n_counts violin plot
sc.pl.violin(adata, 'n_counts', ax=axes[0, 1], show=False)
axes[0, 1].set_title('n_counts')

# pct_counts_mt violin plot
sc.pl.violin(adata, 'pct_counts_mt', ax=axes[0, 2], show=False)
axes[0, 2].axhline(y=MAX_PCT_MITO, color='red', linestyle='--', linewidth=1)
axes[0, 2].set_title(f'% Mitochondrial (filter: <{MAX_PCT_MITO}%)')

# Scatter: n_counts vs n_genes
axes[1, 0].scatter(adata.obs['n_counts'], adata.obs['n_genes'], alpha=0.3, s=1)
axes[1, 0].set_xlabel('n_counts')
axes[1, 0].set_ylabel('n_genes')
axes[1, 0].axhline(y=MIN_GENES, color='red', linestyle='--', linewidth=1)
axes[1, 0].axhline(y=MAX_GENES, color='red', linestyle='--', linewidth=1)

# Scatter: n_counts vs pct_counts_mt
axes[1, 1].scatter(adata.obs['n_counts'], adata.obs['pct_counts_mt'], alpha=0.3, s=1)
axes[1, 1].set_xlabel('n_counts')
axes[1, 1].set_ylabel('% Mitochondrial')
axes[1, 1].axhline(y=MAX_PCT_MITO, color='red', linestyle='--', linewidth=1)

# pct_counts_ribo and pct_counts_hb
axes[1, 2].scatter(adata.obs['pct_counts_ribo'], adata.obs['pct_counts_hb'], alpha=0.3, s=1)
axes[1, 2].set_xlabel('% Ribosomal')
axes[1, 2].set_ylabel('% Hemoglobin')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_qc_metrics_before_filtering.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"  Saved: {FIGURES_DIR / '01_qc_metrics_before_filtering.png'}")

# Plot per-sample QC metrics if multiple samples
if adata.obs['sample'].nunique() > 1:
    print("\n  Plotting per-sample QC metrics...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Per-Sample QC Metrics - Before Filtering', fontsize=16, y=1.02)

    # n_genes per sample
    sc.pl.violin(adata, 'n_genes', groupby='sample', ax=axes[0, 0], show=False, rotation=45)
    axes[0, 0].axhline(y=MIN_GENES, color='red', linestyle='--', linewidth=1)
    axes[0, 0].axhline(y=MAX_GENES, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title('n_genes per sample')

    # n_counts per sample
    sc.pl.violin(adata, 'n_counts', groupby='sample', ax=axes[0, 1], show=False, rotation=45)
    axes[0, 1].set_title('n_counts per sample')

    # pct_counts_mt per sample
    sc.pl.violin(adata, 'pct_counts_mt', groupby='sample', ax=axes[1, 0], show=False, rotation=45)
    axes[1, 0].axhline(y=MAX_PCT_MITO, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_title('% Mitochondrial per sample')

    # Cell count per sample
    sample_counts = adata.obs['sample'].value_counts().sort_index()
    axes[1, 1].bar(range(len(sample_counts)), sample_counts.values)
    axes[1, 1].set_xticks(range(len(sample_counts)))
    axes[1, 1].set_xticklabels(sample_counts.index, rotation=45, ha='right')
    axes[1, 1].axhline(y=MIN_CELLS_PER_SAMPLE, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_ylabel('Cell count')
    axes[1, 1].set_title('Cell count per sample')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01b_per_sample_qc_before_filtering.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"  Saved: {FIGURES_DIR / '01b_per_sample_qc_before_filtering.png'}")

# ----------------------------------------------------------------------------
# 2.3 Filter cells
# ----------------------------------------------------------------------------

print("\n[Step 3/4] Filtering cells...")
print(f"\nBefore filtering: {adata.n_obs} cells")

# Apply filters
sc.pp.filter_cells(adata, min_genes=MIN_GENES)
print(f"After min_genes filter ({MIN_GENES}): {adata.n_obs} cells")

adata = adata[adata.obs['n_genes'] < MAX_GENES, :].copy()
print(f"After max_genes filter ({MAX_GENES}): {adata.n_obs} cells")

adata = adata[adata.obs['pct_counts_mt'] < MAX_PCT_MITO, :].copy()
print(f"After max_pct_mito filter ({MAX_PCT_MITO}%): {adata.n_obs} cells")

# ----------------------------------------------------------------------------
# 2.4 Filter genes
# ----------------------------------------------------------------------------

print("\n[Step 4/4] Filtering genes...")
print(f"\nBefore filtering: {adata.n_vars} genes")

sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
print(f"After min_cells filter ({MIN_CELLS_PER_GENE}): {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# 2.5 Check sample sizes (if multiple samples exist)
# ----------------------------------------------------------------------------

if 'sample' in adata.obs.columns:
    print("\nFiltering samples by minimum cell count...")
    sample_counts = adata.obs['sample'].value_counts()
    print(f"\nSample cell counts before filtering:")
    print(sample_counts)

    valid_samples = sample_counts[sample_counts >= MIN_CELLS_PER_SAMPLE].index
    adata = adata[adata.obs['sample'].isin(valid_samples), :].copy()

    print(f"\nSamples after filtering (min {MIN_CELLS_PER_SAMPLE} cells): {len(valid_samples)}")
    print(f"Total cells after sample filtering: {adata.n_obs}")
else:
    print("\nNo 'sample' column found - treating as single sample")
    if adata.n_obs < MIN_CELLS_PER_SAMPLE:
        print(f"WARNING: Only {adata.n_obs} cells remaining, less than minimum {MIN_CELLS_PER_SAMPLE}")

# ============================================================================
# PHASE 3: DOUBLET DETECTION & REMOVAL
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: DOUBLET DETECTION & REMOVAL")
print("="*80)

# Determine samples
if 'sample' in adata.obs.columns:
    samples = adata.obs['sample'].unique()
    print(f"\nDetected {len(samples)} samples")
else:
    print("\nNo sample information found - creating single sample")
    adata.obs['sample'] = 'sample_1'
    samples = ['sample_1']

# Initialize doublet columns
adata.obs['doublet_score'] = 0.0
adata.obs['doublet_class'] = 'singlet'

# Run Scrublet per sample
print("\nRunning Scrublet per sample...")

doublet_stats = []

for sample in samples:
    print(f"\n--- Processing sample: {sample} ---")

    # Get cells for this sample
    sample_mask = adata.obs['sample'] == sample
    sample_cells = adata[sample_mask, :].copy()

    print(f"Cells in sample: {sample_cells.n_obs}")

    if sample_cells.n_obs < MIN_CELLS_PER_SAMPLE:
        print(f"WARNING: Sample {sample} has only {sample_cells.n_obs} cells, skipping doublet detection")
        continue

    # Run Scrublet
    scrub = scr.Scrublet(
        sample_cells.X,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE
    )

    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=30
    )

    # Handle case where automatic threshold detection failed
    if predicted_doublets is None:
        print(f"WARNING: Automatic threshold detection failed for {sample}")
        print("Manually calling doublets with threshold=0.25")
        predicted_doublets = scrub.call_doublets(threshold=0.25)

    # Store results
    adata.obs.loc[sample_mask, 'doublet_score'] = doublet_scores
    adata.obs.loc[sample_mask, 'doublet_class'] = ['doublet' if d else 'singlet' for d in predicted_doublets]

    n_doublets = predicted_doublets.sum()
    doublet_rate = n_doublets / len(predicted_doublets) * 100

    print(f"Detected doublets: {n_doublets} ({doublet_rate:.2f}%)")

    doublet_stats.append({
        'sample': sample,
        'n_cells': sample_cells.n_obs,
        'n_doublets': n_doublets,
        'doublet_rate': doublet_rate
    })

    # Plot doublet score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    scrub.plot_histogram()
    plt.title(f'Doublet Score Distribution - {sample}')
    plt.savefig(FIGURES_DIR / f'02_doublet_histogram_{sample}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close specific figure
    plt.close('all')  # Close all figures to be safe
    del fig, ax  # Delete figure objects
    gc.collect()  # Force garbage collection to free memory/GDI objects

# Print and save doublet statistics
print("\n" + "-"*80)
print("DOUBLET DETECTION SUMMARY")
print("-"*80)
doublet_df = pd.DataFrame(doublet_stats)
print(doublet_df.to_string(index=False))

# Save doublet detection summary
doublet_summary_file = OUTPUT_DIR / "doublet_detection_summary.csv"
doublet_df.to_csv(doublet_summary_file, index=False)
print(f"\nSaved doublet detection summary: {doublet_summary_file}")

# ----------------------------------------------------------------------------
# Remove doublets
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("REMOVING DOUBLETS")
print("-"*80)

print(f"\nBefore doublet removal: {adata.n_obs} cells")
n_doublets_total = (adata.obs['doublet_class'] == 'doublet').sum()
print(f"Total doublets to remove: {n_doublets_total}")

adata = adata[adata.obs['doublet_class'] == 'singlet', :].copy()
print(f"After doublet removal: {adata.n_obs} cells")

# ----------------------------------------------------------------------------
# Re-check sample sizes
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("RE-CHECKING SAMPLE SIZES")
print("-"*80)

sample_counts_final = adata.obs['sample'].value_counts()
print("\nFinal sample cell counts:")
print(sample_counts_final)

valid_samples_final = sample_counts_final[sample_counts_final >= MIN_CELLS_PER_SAMPLE].index
if len(valid_samples_final) < len(sample_counts_final):
    print(f"\nWARNING: Removing {len(sample_counts_final) - len(valid_samples_final)} samples with <{MIN_CELLS_PER_SAMPLE} cells")
    adata = adata[adata.obs['sample'].isin(valid_samples_final), :].copy()
    print(f"Final cell count: {adata.n_obs}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save filtered data
output_file = OUTPUT_DIR / "adata_qc_filtered.h5ad"
adata.write(output_file)
print(f"\nSaved filtered AnnData object: {output_file}")

# Save QC summary statistics
qc_summary = {
    'n_cells_final': adata.n_obs,
    'n_genes_final': adata.n_vars,
    'n_samples': len(adata.obs['sample'].unique()),
    'doublets_removed': n_doublets_total,
    'mean_genes_per_cell': adata.obs['n_genes'].mean(),
    'mean_counts_per_cell': adata.obs['n_counts'].mean(),
    'mean_pct_mito': adata.obs['pct_counts_mt'].mean(),
}

qc_summary_df = pd.DataFrame([qc_summary])
qc_summary_file = OUTPUT_DIR / "qc_summary.csv"
qc_summary_df.to_csv(qc_summary_file, index=False)
print(f"Saved QC summary: {qc_summary_file}")

# Save per-sample summary statistics
per_sample_summary = adata.obs.groupby('sample').agg({
    'n_genes': ['mean', 'median'],
    'n_counts': ['mean', 'median'],
    'pct_counts_mt': ['mean', 'median']
}).round(2)

per_sample_summary.columns = ['_'.join(col) for col in per_sample_summary.columns]
per_sample_summary['n_cells'] = adata.obs['sample'].value_counts()
per_sample_summary = per_sample_summary.reset_index()

per_sample_file = OUTPUT_DIR / "per_sample_summary.csv"
per_sample_summary.to_csv(per_sample_file, index=False)
print(f"Saved per-sample summary: {per_sample_file}")

# Plot final QC metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('QC Metrics - After Filtering & Doublet Removal', fontsize=16, y=1.02)

sc.pl.violin(adata, 'n_genes', ax=axes[0, 0], show=False)
axes[0, 0].set_title('n_genes')

sc.pl.violin(adata, 'n_counts', ax=axes[0, 1], show=False)
axes[0, 1].set_title('n_counts')

sc.pl.violin(adata, 'pct_counts_mt', ax=axes[1, 0], show=False)
axes[1, 0].set_title('% Mitochondrial')

axes[1, 1].scatter(adata.obs['n_counts'], adata.obs['n_genes'], alpha=0.3, s=1)
axes[1, 1].set_xlabel('n_counts')
axes[1, 1].set_ylabel('n_genes')
axes[1, 1].set_title('n_counts vs n_genes')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_qc_metrics_after_filtering.png', dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"Saved final QC plots: {FIGURES_DIR / '03_qc_metrics_after_filtering.png'}")

# Plot per-sample QC metrics after filtering if multiple samples
if adata.obs['sample'].nunique() > 1:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Per-Sample QC Metrics - After Filtering & Doublet Removal', fontsize=16, y=1.02)

    # n_genes per sample
    sc.pl.violin(adata, 'n_genes', groupby='sample', ax=axes[0, 0], show=False, rotation=45)
    axes[0, 0].set_title('n_genes per sample')

    # n_counts per sample
    sc.pl.violin(adata, 'n_counts', groupby='sample', ax=axes[0, 1], show=False, rotation=45)
    axes[0, 1].set_title('n_counts per sample')

    # pct_counts_mt per sample
    sc.pl.violin(adata, 'pct_counts_mt', groupby='sample', ax=axes[1, 0], show=False, rotation=45)
    axes[1, 0].set_title('% Mitochondrial per sample')

    # Final cell count per sample
    sample_counts_final_plot = adata.obs['sample'].value_counts().sort_index()
    axes[1, 1].bar(range(len(sample_counts_final_plot)), sample_counts_final_plot.values)
    axes[1, 1].set_xticks(range(len(sample_counts_final_plot)))
    axes[1, 1].set_xticklabels(sample_counts_final_plot.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Cell count')
    axes[1, 1].set_title('Final cell count per sample')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03b_per_sample_qc_after_filtering.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    gc.collect()

    print(f"Saved per-sample final QC plots: {FIGURES_DIR / '03b_per_sample_qc_after_filtering.png'}")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nFinal dataset:")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Samples: {len(adata.obs['sample'].unique())}")
print(f"\nOutput saved to: {OUTPUT_DIR}")
