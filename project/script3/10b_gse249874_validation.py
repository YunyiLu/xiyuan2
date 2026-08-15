"""
Step 10b: GSE249874 Independent scRNA-seq Validation
  18 samples: GC(3HP-/3HP+) + Gastritis(3HP-/3HP+) + IM(3HP-/3HP+)

  Validates 92 candidate genes in independent single-cell data:
  1. Stage-specific expression (Gastritis vs IM vs GC)
  2. HP+ vs HP- effect on candidate genes
  3. Cell-type specificity of candidate genes
  4. Cross-validate with our GSE134520 findings

Input:
  - GSE249874 raw count matrix (mtx format, 1.4GB)
  - 92 candidate genes

Output:
  - results/gse249874_stage_expression.csv
  - results/gse249874_hp_effect.csv
  - results/gse249874_validation_summary.csv
  - figures/gse249874_validation.png
"""
import sys, os, warnings, gzip
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DATA_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset"
RES_DIR = f"{BASE}/results"
FIG_DIR = f"{BASE}/figures"

# Load candidates
candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()
print(f"Loaded {len(CANDIDATES)} candidate genes")

# ============================================================
# Step 1: Load GSE249874 data
# ============================================================
print("\n" + "="*70)
print("STEP 1: Load GSE249874 scRNA-seq data")
print("="*70)

# Read features (genes)
print("  Reading features...")
features = pd.read_csv(f"{DATA_DIR}/GSE249874_raw_feature_features.tsv.gz",
                       sep='\t', header=None, names=['ensembl', 'symbol', 'type'],
                       compression='gzip')
print(f"  Features: {len(features)}")

# Check which candidates are in this dataset
candidates_in_data = [g for g in CANDIDATES if g in features['symbol'].values]
print(f"  Candidates found in features: {len(candidates_in_data)}/{len(CANDIDATES)}")

# Read barcodes
print("  Reading barcodes...")
barcodes = pd.read_csv(f"{DATA_DIR}/GSE249874_raw_feature_barcodes.tsv.gz",
                       sep='\t', header=None, names=['barcode'],
                       compression='gzip')
print(f"  Barcodes (cells): {len(barcodes)}")

# Read sample metadata
samples_meta = pd.read_csv(f"{DATA_DIR}/GSE249874_sample_names_and_accessions.txt.gz",
                           sep='\t', compression='gzip')
print(f"  Samples: {len(samples_meta)}")

# Parse sample metadata
sample_info = []
for _, row in samples_meta.iterrows():
    title = row['*title']
    lib = row['*library name']

    if 'adenocarcinoma' in title.lower() or 'GC' in title:
        stage = 'GC'
    elif 'intestinal metaplasia' in title.lower() or 'IM' in title:
        stage = 'IM'
    elif 'gatritis' in title.lower() or 'gastritis' in title.lower():
        stage = 'Gastritis'
    else:
        stage = 'Unknown'

    hp = 'HP+' if 'pylori-positive' in title.lower() else 'HP-'

    sample_info.append({
        'sample': lib,
        'accession': row['GEO accession'],
        'stage': stage,
        'hp_status': hp,
        'title': title
    })

sample_df = pd.DataFrame(sample_info)
print(f"\n  Sample distribution:")
print(sample_df.groupby(['stage', 'hp_status']).size())

# ============================================================
# Step 2: Load expression matrix (sparse, large)
# ============================================================
print("\n" + "="*70)
print("STEP 2: Load expression matrix")
print("="*70)

# For efficiency, only extract candidate gene rows from the sparse matrix
# First, find indices of candidate genes
gene_indices = {}
for gene in candidates_in_data:
    idx = features[features['symbol'] == gene].index
    if len(idx) > 0:
        gene_indices[gene] = idx[0]

print(f"  Gene indices mapped: {len(gene_indices)}")
print(f"  Loading sparse matrix (this will take a while for 1.4GB)...")

# This is a raw unfiltered matrix (122M barcodes from 18 10x samples)
# 122M × 36601 is too large for scipy.mmread (~7GB RAM for COO arrays)
# Strategy: stream the MTX file, only keep entries for 92 candidate gene rows
# For cell filtering, use barcode suffix (sample number) to assign cells to samples
# Then filter by per-cell candidate total (cells expressing any candidates are likely real)

print(f"  Raw barcodes: {len(barcodes)} (most are empty droplets)")
print("  Using streaming approach to extract only 92 gene entries...")

import scipy.sparse as sp
from collections import defaultdict

try:
    target_rows = set(gene_indices.values())
    row_to_gene = {idx: gene for gene, idx in gene_indices.items()}

    # Stream MTX: only keep entries where row is a candidate gene
    # This reads 324M lines but only stores ~500K-1M entries
    cell_data = defaultdict(dict)  # cell_col -> {gene: count}
    cell_any_count = defaultdict(float)  # total counts in candidates for filtering

    entries_processed = 0
    entries_kept = 0

    print("  Streaming 324M entries (extracting 92 gene rows only)...")
    with gzip.open(f"{DATA_DIR}/GSE249874_raw_feature_matrix.mtx.gz", 'rt') as f:
        # Skip comments
        for line in f:
            if line.startswith('%'):
                continue
            # This is the dimension line
            n_rows, n_cols, n_entries = map(int, line.strip().split())
            print(f"  Dimensions: {n_rows} × {n_cols}, {n_entries} nonzeros")
            break

        for line in f:
            entries_processed += 1
            if entries_processed % 50000000 == 0:
                print(f"    {entries_processed/1e6:.0f}M / {n_entries/1e6:.0f}M entries, kept {entries_kept}")

            parts = line.split()
            row = int(parts[0]) - 1  # 0-indexed
            if row not in target_rows:
                continue

            col = int(parts[1]) - 1
            val = float(parts[2]) if len(parts) > 2 else 1.0

            gene = row_to_gene[row]
            cell_data[col][gene] = val
            cell_any_count[col] += val
            entries_kept += 1

    print(f"  Entries processed: {entries_processed}")
    print(f"  Entries for candidates: {entries_kept}")
    print(f"  Cells with any candidate expression: {len(cell_data)}")

    # Filter: cells with >=5 total candidate counts (real cells expressing our genes)
    min_candidate_counts = 3
    filtered_cells = {c: data for c, data in cell_data.items()
                      if cell_any_count[c] >= min_candidate_counts}
    print(f"  Cells with >={min_candidate_counts} candidate counts: {len(filtered_cells)}")

    del cell_data, cell_any_count

    # Build expression dataframe
    rows_list = []
    for cell_idx, gene_counts in filtered_cells.items():
        row_data = {'cell_idx': cell_idx}
        row_data.update(gene_counts)
        rows_list.append(row_data)

    expr_df = pd.DataFrame(rows_list).fillna(0)
    expr_df = expr_df.set_index('cell_idx')

    # Add barcode
    barcode_array = barcodes['barcode'].values
    expr_df['barcode'] = [barcode_array[i] for i in expr_df.index]

    # Add missing candidate columns
    for gene in candidates_in_data:
        if gene not in expr_df.columns:
            expr_df[gene] = 0

    print(f"  Final expression matrix: {expr_df.shape}")
    del filtered_cells

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Step 3: Assign cells to samples
# ============================================================
print("\n" + "="*70)
print("STEP 3: Assign cells to samples and basic QC")
print("="*70)

# Assign cells to samples based on barcode suffix or position
# GSE249874: barcodes are SEQUENCE-N where N=sample number (1-18)
# Each sample has 6,794,880 barcodes (122,307,840 / 18)
print(f"  Sample barcodes (first 5): {expr_df['barcode'].head().tolist()}")
print(f"  Barcode format example: {expr_df['barcode'].iloc[0]}")

# Extract sample number from barcode suffix
def get_sample_from_barcode(bc):
    parts = bc.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"sample{int(parts[1])}"
    return "unknown"

expr_df['sample'] = expr_df['barcode'].apply(get_sample_from_barcode)

# Check distribution
print(f"\n  Sample distribution in filtered cells:")
print(expr_df['sample'].value_counts().sort_index())

# Merge with sample metadata
expr_df = expr_df.merge(sample_df[['sample', 'stage', 'hp_status']], on='sample', how='left')
print(f"\n  Cells per stage:")
print(expr_df['stage'].value_counts())
print(f"\n  Cells per HP status:")
print(expr_df['hp_status'].value_counts())

# Basic QC: total counts per cell
total_counts = expr_df[candidates_in_data].sum(axis=1)
expr_df['total_candidate_counts'] = total_counts

# Filter cells with at least some counts
min_counts = 1
valid_cells = expr_df[expr_df['total_candidate_counts'] >= min_counts]
print(f"\n  Cells passing QC (total_candidate_counts >= {min_counts}): {len(valid_cells)}/{len(expr_df)}")

# ============================================================
# Step 4: Stage-specific expression analysis
# ============================================================
print("\n" + "="*70)
print("STEP 4: Stage-specific expression of 92 candidates")
print("="*70)

stage_results = []
for gene in candidates_in_data:
    gene_data = valid_cells[[gene, 'stage', 'hp_status']].dropna(subset=['stage'])

    if len(gene_data) < 50:
        continue

    stages = ['Gastritis', 'IM', 'GC']
    stage_means = {}
    for s in stages:
        vals = gene_data[gene_data['stage'] == s][gene].values
        stage_means[s] = np.mean(vals) if len(vals) > 0 else 0

    # Kruskal-Wallis across stages
    groups = [gene_data[gene_data['stage'] == s][gene].values for s in stages]
    groups = [g for g in groups if len(g) > 10]

    if len(groups) >= 2:
        try:
            h_stat, kw_p = kruskal(*groups)
        except:
            h_stat, kw_p = 0, 1
    else:
        h_stat, kw_p = 0, 1

    # IM vs Gastritis (key comparison)
    im_vals = gene_data[gene_data['stage'] == 'IM'][gene].values
    gs_vals = gene_data[gene_data['stage'] == 'Gastritis'][gene].values

    if len(im_vals) > 10 and len(gs_vals) > 10:
        _, im_gs_p = mannwhitneyu(im_vals, gs_vals, alternative='two-sided')
        im_gs_fc = np.log2((np.mean(im_vals) + 0.01) / (np.mean(gs_vals) + 0.01))
    else:
        im_gs_p, im_gs_fc = 1, 0

    # HP effect within IM
    im_hp_pos = gene_data[(gene_data['stage'] == 'IM') & (gene_data['hp_status'] == 'HP+')][gene].values
    im_hp_neg = gene_data[(gene_data['stage'] == 'IM') & (gene_data['hp_status'] == 'HP-')][gene].values

    if len(im_hp_pos) > 5 and len(im_hp_neg) > 5:
        _, hp_p = mannwhitneyu(im_hp_pos, im_hp_neg, alternative='two-sided')
        hp_fc = np.log2((np.mean(im_hp_pos) + 0.01) / (np.mean(im_hp_neg) + 0.01))
    else:
        hp_p, hp_fc = 1, 0

    stage_results.append({
        'gene': gene,
        'mean_gastritis': stage_means.get('Gastritis', 0),
        'mean_IM': stage_means.get('IM', 0),
        'mean_GC': stage_means.get('GC', 0),
        'kruskal_p': kw_p,
        'IM_vs_Gastritis_logFC': im_gs_fc,
        'IM_vs_Gastritis_p': im_gs_p,
        'HP_effect_in_IM_logFC': hp_fc,
        'HP_effect_in_IM_p': hp_p,
        'cascade_direction': 'up' if stage_means.get('GC', 0) > stage_means.get('Gastritis', 0) else 'down'
    })

stage_df = pd.DataFrame(stage_results)
if len(stage_df) > 0:
    _, stage_df['kruskal_fdr'], _, _ = multipletests(stage_df['kruskal_p'], method='fdr_bh')
    _, stage_df['IM_vs_Gastritis_fdr'], _, _ = multipletests(stage_df['IM_vs_Gastritis_p'], method='fdr_bh')

    stage_df = stage_df.sort_values('kruskal_fdr')
    stage_df.to_csv(f"{RES_DIR}/gse249874_stage_expression.csv", index=False)

    print(f"  Genes analyzed: {len(stage_df)}")
    print(f"  Significant stage effect (FDR<0.05): {(stage_df['kruskal_fdr'] < 0.05).sum()}")
    print(f"  IM > Gastritis (FDR<0.05): {((stage_df['IM_vs_Gastritis_fdr'] < 0.05) & (stage_df['IM_vs_Gastritis_logFC'] > 0)).sum()}")

    print("\n  Top genes upregulated in IM vs Gastritis:")
    up_im = stage_df[(stage_df['IM_vs_Gastritis_logFC'] > 0) & (stage_df['IM_vs_Gastritis_fdr'] < 0.05)].head(10)
    for _, row in up_im.iterrows():
        print(f"    {row['gene']}: logFC={row['IM_vs_Gastritis_logFC']:.2f}, FDR={row['IM_vs_Gastritis_fdr']:.4f}")

# ============================================================
# Step 5: HP effect analysis
# ============================================================
print("\n" + "="*70)
print("STEP 5: H. pylori effect on candidate genes within IM")
print("="*70)

hp_results = stage_df[stage_df['HP_effect_in_IM_p'] < 0.05].sort_values('HP_effect_in_IM_p')
if len(hp_results) > 0:
    hp_results.to_csv(f"{RES_DIR}/gse249874_hp_effect.csv", index=False)
    print(f"  Genes with HP effect in IM (p<0.05): {len(hp_results)}")
    for _, row in hp_results.head(10).iterrows():
        direction = "↑HP+" if row['HP_effect_in_IM_logFC'] > 0 else "↓HP+"
        print(f"    {row['gene']}: {direction} logFC={row['HP_effect_in_IM_logFC']:.2f}, p={row['HP_effect_in_IM_p']:.4f}")
else:
    print("  No significant HP effects found")

# ============================================================
# Step 6: Cross-validate with our findings
# ============================================================
print("\n" + "="*70)
print("STEP 6: Cross-validation with GSE134520 findings")
print("="*70)

# Our temporal ordering from mechanism analysis
temporal = pd.read_csv(f"{RES_DIR}/mechanism_temporal_ordering.csv")

# Merge
validation = stage_df.merge(temporal[['gene', 'onset_stage', 'temporal_class']],
                            on='gene', how='left')
validation = validation.merge(candidates_df[['gene', 'TransformationScore']],
                             on='gene', how='left')

# Check consistency: genes with "early" onset should be up in IM vs Gastritis
early_genes = validation[validation['onset_stage'] == 'LGIN']
if len(early_genes) > 0:
    consistent = (early_genes['IM_vs_Gastritis_logFC'] > 0).sum()
    total = len(early_genes)
    print(f"  Early-onset genes (LGIN) validated in GSE249874:")
    print(f"    Consistent (↑ in IM vs Gastritis): {consistent}/{total} ({100*consistent/total:.0f}%)")

# Summary
validation_summary = validation[['gene', 'mean_gastritis', 'mean_IM', 'mean_GC',
                                  'IM_vs_Gastritis_logFC', 'IM_vs_Gastritis_fdr',
                                  'cascade_direction', 'onset_stage', 'TransformationScore']].copy()
validation_summary = validation_summary.sort_values('TransformationScore', ascending=False)
validation_summary.to_csv(f"{RES_DIR}/gse249874_validation_summary.csv", index=False)

print(f"\n  Validation summary saved: {len(validation_summary)} genes")

# ============================================================
# Step 7: Visualization
# ============================================================
print("\n" + "="*70)
print("STEP 7: Visualization")
print("="*70)

if len(stage_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # A: Volcano (IM vs Gastritis)
    ax = axes[0, 0]
    sig = stage_df['IM_vs_Gastritis_fdr'] < 0.05
    ax.scatter(stage_df[~sig]['IM_vs_Gastritis_logFC'],
               -np.log10(stage_df[~sig]['IM_vs_Gastritis_fdr'] + 1e-300),
               c='gray', alpha=0.5, s=30)
    ax.scatter(stage_df[sig]['IM_vs_Gastritis_logFC'],
               -np.log10(stage_df[sig]['IM_vs_Gastritis_fdr'] + 1e-300),
               c='red', alpha=0.7, s=50)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
    ax.set_xlabel('log2FC (IM / Gastritis)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('GSE249874: IM vs Gastritis\n(92 candidate genes)')

    # Label top genes
    top = stage_df[sig].nlargest(5, 'IM_vs_Gastritis_logFC')
    for _, row in top.iterrows():
        ax.annotate(row['gene'], (row['IM_vs_Gastritis_logFC'],
                    -np.log10(row['IM_vs_Gastritis_fdr'] + 1e-300)),
                   fontsize=8)

    # B: Heatmap top 15 genes across stages
    ax = axes[0, 1]
    top15 = stage_df.nlargest(15, 'IM_vs_Gastritis_logFC')
    heatmap_data = top15[['gene', 'mean_gastritis', 'mean_IM', 'mean_GC']].set_index('gene')
    # Normalize per gene
    heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1) + 0.01, axis=0)
    sns.heatmap(heatmap_norm, ax=ax, cmap='YlOrRd', xticklabels=['Gastritis', 'IM', 'GC'])
    ax.set_title('Top 15 IM-upregulated genes\n(normalized expression)')

    # C: TransformationScore vs validation effect
    ax = axes[1, 0]
    merged = stage_df.merge(candidates_df[['gene', 'TransformationScore']], on='gene', how='left')
    ax.scatter(merged['TransformationScore'], merged['IM_vs_Gastritis_logFC'],
               alpha=0.6, s=40)
    ax.set_xlabel('TransformationScore (our ranking)')
    ax.set_ylabel('log2FC IM/Gastritis (GSE249874)')
    ax.set_title('Our Score vs Independent Validation')
    # Add correlation
    from scipy.stats import spearmanr
    valid = merged.dropna(subset=['TransformationScore', 'IM_vs_Gastritis_logFC'])
    if len(valid) > 10:
        r, p = spearmanr(valid['TransformationScore'], valid['IM_vs_Gastritis_logFC'])
        ax.text(0.05, 0.95, f'Spearman r={r:.3f}\np={p:.2e}', transform=ax.transAxes, va='top')

    # D: HP effect
    ax = axes[1, 1]
    hp_sig = stage_df['HP_effect_in_IM_p'] < 0.05
    ax.scatter(stage_df[~hp_sig]['HP_effect_in_IM_logFC'],
               -np.log10(stage_df[~hp_sig]['HP_effect_in_IM_p'] + 1e-300),
               c='gray', alpha=0.5, s=30)
    ax.scatter(stage_df[hp_sig]['HP_effect_in_IM_logFC'],
               -np.log10(stage_df[hp_sig]['HP_effect_in_IM_p'] + 1e-300),
               c='orange', alpha=0.7, s=50)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
    ax.set_xlabel('log2FC (HP+ / HP-) within IM')
    ax.set_ylabel('-log10(p)')
    ax.set_title('H. pylori effect on candidates\n(within IM samples)')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/gse249874_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gse249874_validation.png")

print("\nDone!")
