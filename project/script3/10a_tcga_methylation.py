"""
Step 10a: TCGA-STAD Methylation-Expression Integration
  - Extract 92 candidate gene promoter CpGs from TCGA 450K methylation
  - Match methylation with expression for same tumor samples
  - Compute per-gene methylation-expression correlation
  - Classify genes: hypomethylated-upregulated vs hypermethylated-silenced

Input:
  - TCGA-STAD.methylation450.tsv.gz (486K probes × 397 samples)
  - TCGA-STAD.HiSeqV2.gz (expression, log2 RSEM)
  - 92 candidate genes (unified_discovery_ranked.csv)

Output:
  - results/tcga_methylation_92gene_probes.csv
  - results/tcga_methylation_expression_corr.csv
  - results/tcga_methylation_tumor_vs_normal.csv
  - figures/methylation_expression_scatter.png
"""
import sys, os, warnings, gzip
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
TCGA_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"
RES_DIR = f"{BASE}/results"
FIG_DIR = f"{BASE}/figures"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load 92 candidate genes
candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()
print(f"Loaded {len(CANDIDATES)} candidate genes")

# ============================================================
# Step 1: Build 450K probe-to-gene mapping
# ============================================================
print("\n" + "="*70)
print("STEP 1: Build probe-to-gene mapping for 92 candidates")
print("="*70)

# Use Illumina 450K manifest to map CpG probes to genes
# The manifest maps probes to UCSC_RefGene_Name (gene symbol)
# We'll download it or build from the methylation file itself

# Strategy: scan TCGA methylation file for probes, use external annotation
# Since we don't have the full manifest locally, we'll use a known mapping approach

# First, try to get probe-gene mapping from available sources
MANIFEST_URL = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"

# Alternative: use a pre-built mapping from BioConductor IlluminaHumanMethylation450kanno
# For efficiency, we'll build a targeted probe list using gene coordinates

# Known promoter CpG probes for key genes (curated from literature + UCSC)
# We'll use an approach that doesn't require the full manifest:
# Read the TCGA file header, then grep for probes by downloading probe annotation

manifest_path = f"{BASE}/data/methylation/HM450_manifest_genes.csv"
os.makedirs(f"{BASE}/data/methylation", exist_ok=True)

if not os.path.exists(manifest_path):
    print("  Downloading 450K probe-gene annotation...")
    # Use GEO platform annotation GPL13534
    # Alternative: extract from existing annotation files
    try:
        # Try to build from online source (compact version)
        import urllib.request
        url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
        print(f"  Downloading GPL13534 manifest...")
        urllib.request.urlretrieve(url, f"{BASE}/data/methylation/GPL13534_manifest.csv.gz")
        print("  Download complete, parsing...")

        # Parse manifest - only keep probe ID and gene columns
        probe_genes = []
        with gzip.open(f"{BASE}/data/methylation/GPL13534_manifest.csv.gz", 'rt', errors='replace') as f:
            header_found = False
            col_idx = {}
            for line in f:
                if line.startswith('IlmnID'):
                    header_found = True
                    cols = line.strip().split(',')
                    for i, c in enumerate(cols):
                        col_idx[c] = i
                    continue
                if not header_found:
                    continue
                parts = line.strip().split(',')
                if len(parts) < max(col_idx.values()) + 1:
                    continue
                probe_id = parts[col_idx.get('IlmnID', 0)]
                gene_name = parts[col_idx.get('UCSC_RefGene_Name', 0)]
                gene_group = parts[col_idx.get('UCSC_RefGene_Group', 0)]
                chrom = parts[col_idx.get('CHR', 0)]
                pos = parts[col_idx.get('MAPINFO', 0)]

                if not gene_name:
                    continue
                # A probe can map to multiple genes (semicolon-separated)
                genes = gene_name.split(';')
                groups = gene_group.split(';') if gene_group else [''] * len(genes)

                for g, grp in zip(genes, groups):
                    g = g.strip()
                    if g in CANDIDATES:
                        probe_genes.append({
                            'probe': probe_id,
                            'gene': g,
                            'region': grp.strip(),
                            'chr': chrom,
                            'pos': pos
                        })

        df_manifest = pd.DataFrame(probe_genes).drop_duplicates()
        df_manifest.to_csv(manifest_path, index=False)
        print(f"  Saved {len(df_manifest)} probe-gene mappings for {df_manifest['gene'].nunique()} candidates")

    except Exception as e:
        print(f"  ERROR downloading manifest: {e}")
        print("  Falling back to name-based search in TCGA file...")
        df_manifest = None
else:
    df_manifest = pd.read_csv(manifest_path)
    print(f"  Loaded cached manifest: {len(df_manifest)} probes for {df_manifest['gene'].nunique()} genes")


# ============================================================
# Step 2: Extract 92-gene probes from TCGA methylation
# ============================================================
print("\n" + "="*70)
print("STEP 2: Extract methylation beta values for candidate gene probes")
print("="*70)

if df_manifest is not None and len(df_manifest) > 0:
    target_probes = set(df_manifest['probe'].unique())
    print(f"  Target probes to extract: {len(target_probes)}")

    # Read TCGA methylation file - only extract target probes
    print("  Reading TCGA methylation file (this may take a few minutes)...")

    # Read header first
    with gzip.open(f"{TCGA_DIR}/TCGA-STAD.methylation450.tsv.gz", 'rt') as f:
        header = f.readline().strip().split('\t')

    sample_ids = header[1:]
    print(f"  Total samples: {len(sample_ids)}")

    # Classify samples: tumor (01A/01B) vs normal (11A/11B)
    tumor_samples = [s for s in sample_ids if len(s.split('-')) >= 4 and s.split('-')[3][:2] == '01']
    normal_samples = [s for s in sample_ids if len(s.split('-')) >= 4 and s.split('-')[3][:2] == '11']
    print(f"  Tumor samples: {len(tumor_samples)}, Normal samples: {len(normal_samples)}")

    # Extract target probes
    probe_data = {}
    lines_read = 0
    n_values_expected = len(sample_ids)
    with gzip.open(f"{TCGA_DIR}/TCGA-STAD.methylation450.tsv.gz", 'rt') as f:
        f.readline()  # skip header
        for line in f:
            lines_read += 1
            if lines_read % 100000 == 0:
                print(f"    Processed {lines_read}/486K probes, found {len(probe_data)} targets...")

            parts = line.strip().split('\t')
            probe_id = parts[0]
            if probe_id in target_probes:
                values = []
                for v in parts[1:]:
                    try:
                        values.append(float(v))
                    except:
                        values.append(np.nan)
                # Pad or truncate to match expected length
                if len(values) < n_values_expected:
                    values.extend([np.nan] * (n_values_expected - len(values)))
                elif len(values) > n_values_expected:
                    values = values[:n_values_expected]
                probe_data[probe_id] = values

    print(f"  Extracted {len(probe_data)}/{len(target_probes)} target probes")

    # Build methylation matrix
    methyl_df = pd.DataFrame(probe_data, index=sample_ids).T
    methyl_df.index.name = 'probe'

    # Save raw probe data
    methyl_df.to_csv(f"{RES_DIR}/tcga_methylation_92gene_probes.csv")
    print(f"  Saved: tcga_methylation_92gene_probes.csv ({methyl_df.shape})")

else:
    print("  ERROR: No manifest available, cannot extract probes")
    sys.exit(1)


# ============================================================
# Step 3: Compute per-gene promoter methylation
# ============================================================
print("\n" + "="*70)
print("STEP 3: Aggregate per-gene promoter methylation")
print("="*70)

# Focus on promoter-associated probes (TSS1500, TSS200, 1stExon, 5'UTR)
promoter_regions = ['TSS1500', 'TSS200', '1stExon', "5'UTR"]
promoter_probes = df_manifest[df_manifest['region'].isin(promoter_regions)]
print(f"  Promoter-associated probes: {len(promoter_probes)} for {promoter_probes['gene'].nunique()} genes")

# Aggregate: mean beta per gene per sample (across all promoter probes)
gene_methyl = {}
for gene in CANDIDATES:
    gene_probes = promoter_probes[promoter_probes['gene'] == gene]['probe'].tolist()
    gene_probes_avail = [p for p in gene_probes if p in methyl_df.index]
    if len(gene_probes_avail) > 0:
        gene_methyl[gene] = methyl_df.loc[gene_probes_avail].mean(axis=0)

gene_methyl_df = pd.DataFrame(gene_methyl).T
gene_methyl_df.index.name = 'gene'
print(f"  Genes with promoter methylation data: {len(gene_methyl_df)}/{len(CANDIDATES)}")


# ============================================================
# Step 4: Methylation-Expression Correlation (tumor samples)
# ============================================================
print("\n" + "="*70)
print("STEP 4: Methylation-Expression Correlation")
print("="*70)

# Load TCGA expression
print("  Loading TCGA expression...")
expr_df = pd.read_csv(f"{TCGA_DIR}/TCGA-STAD.HiSeqV2.gz", sep='\t', index_col=0, compression='gzip')
print(f"  Expression matrix: {expr_df.shape}")

# Match samples between methylation and expression
# TCGA barcode: methyl uses TCGA-XX-XXXX-01A, expr uses TCGA-XX-XXXX-01
# Truncate methylation barcodes to match expression
methyl_sample_map = {}
for s in tumor_samples:
    short = '-'.join(s.split('-')[:4])[:15]  # TCGA-XX-XXXX-01
    methyl_sample_map[s] = short

expr_samples = set(expr_df.columns)
matched_pairs = []
for methyl_s, expr_s in methyl_sample_map.items():
    if expr_s in expr_samples:
        matched_pairs.append((methyl_s, expr_s))

print(f"  Matched tumor samples (methyl ∩ expression): {len(matched_pairs)}")

# Compute per-gene correlation
corr_results = []
for gene in gene_methyl_df.index:
    if gene not in expr_df.index:
        continue

    methyl_vals = []
    expr_vals = []
    for methyl_s, expr_s in matched_pairs:
        m = gene_methyl_df.loc[gene, methyl_s]
        e = expr_df.loc[gene, expr_s]
        if not np.isnan(m) and not np.isnan(e):
            methyl_vals.append(m)
            expr_vals.append(e)

    if len(methyl_vals) >= 20:
        rho, pval = spearmanr(methyl_vals, expr_vals)
        r_pearson, p_pearson = pearsonr(methyl_vals, expr_vals)

        corr_results.append({
            'gene': gene,
            'n_samples': len(methyl_vals),
            'mean_beta': np.mean(methyl_vals),
            'spearman_rho': rho,
            'spearman_p': pval,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'mean_expression': np.mean(expr_vals),
            'n_promoter_probes': len(promoter_probes[promoter_probes['gene'] == gene])
        })

corr_df = pd.DataFrame(corr_results)

if len(corr_df) > 0:
    # FDR correction
    _, corr_df['spearman_fdr'], _, _ = multipletests(corr_df['spearman_p'], method='fdr_bh')

    # Classify relationship
    corr_df['methylation_class'] = 'neutral'
    corr_df.loc[(corr_df['spearman_rho'] < -0.2) & (corr_df['spearman_fdr'] < 0.05), 'methylation_class'] = 'hypermethylated_silenced'
    corr_df.loc[(corr_df['spearman_rho'] > 0.2) & (corr_df['spearman_fdr'] < 0.05), 'methylation_class'] = 'positive_correlation'
    corr_df.loc[(corr_df['mean_beta'] < 0.3) & (corr_df['mean_expression'] > 8), 'methylation_class'] = 'hypomethylated_active'

    corr_df = corr_df.sort_values('spearman_rho')
    corr_df.to_csv(f"{RES_DIR}/tcga_methylation_expression_corr.csv", index=False)
    print(f"\n  Results: {len(corr_df)} genes analyzed")
    print(f"  Significant negative correlation (FDR<0.05, rho<-0.2): "
          f"{(corr_df['methylation_class'] == 'hypermethylated_silenced').sum()}")
    print(f"  Significant positive correlation: "
          f"{(corr_df['methylation_class'] == 'positive_correlation').sum()}")
    print(f"  Hypomethylated + highly expressed: "
          f"{(corr_df['methylation_class'] == 'hypomethylated_active').sum()}")

    # Top results
    print("\n  Top genes with strong negative methyl-expression correlation:")
    neg = corr_df[corr_df['spearman_rho'] < -0.2].head(10)
    for _, row in neg.iterrows():
        print(f"    {row['gene']}: rho={row['spearman_rho']:.3f}, beta={row['mean_beta']:.3f}, "
              f"FDR={row['spearman_fdr']:.4f}")


# ============================================================
# Step 5: Tumor vs Normal comparison (limited by n=2 normals)
# ============================================================
print("\n" + "="*70)
print("STEP 5: Tumor vs Normal methylation comparison")
print("="*70)

if len(normal_samples) >= 2:
    tn_results = []
    for gene in gene_methyl_df.index:
        tumor_vals = gene_methyl_df.loc[gene, tumor_samples].dropna().values
        normal_vals = gene_methyl_df.loc[gene, normal_samples].dropna().values

        if len(tumor_vals) >= 10 and len(normal_vals) >= 2:
            delta_beta = np.mean(tumor_vals) - np.mean(normal_vals)
            tn_results.append({
                'gene': gene,
                'mean_beta_tumor': np.mean(tumor_vals),
                'mean_beta_normal': np.mean(normal_vals),
                'delta_beta': delta_beta,
                'direction': 'hypermethylated' if delta_beta > 0.1 else ('hypomethylated' if delta_beta < -0.1 else 'unchanged'),
                'n_tumor': len(tumor_vals),
                'n_normal': len(normal_vals)
            })

    tn_df = pd.DataFrame(tn_results).sort_values('delta_beta')
    tn_df.to_csv(f"{RES_DIR}/tcga_methylation_tumor_vs_normal.csv", index=False)
    print(f"  Saved tumor vs normal comparison ({len(tn_df)} genes)")
    print(f"  NOTE: Only {len(normal_samples)} normal samples - results are indicative only")

    print("\n  Top hypomethylated in tumor (potentially activated):")
    hypo = tn_df[tn_df['delta_beta'] < -0.05].head(5)
    for _, row in hypo.iterrows():
        print(f"    {row['gene']}: delta_beta={row['delta_beta']:.3f} (T:{row['mean_beta_tumor']:.3f} vs N:{row['mean_beta_normal']:.3f})")

    print("\n  Top hypermethylated in tumor (potentially silenced):")
    hyper = tn_df[tn_df['delta_beta'] > 0.05].head(5)
    for _, row in hyper.iterrows():
        print(f"    {row['gene']}: delta_beta={row['delta_beta']:.3f} (T:{row['mean_beta_tumor']:.3f} vs N:{row['mean_beta_normal']:.3f})")
else:
    print("  Insufficient normal samples for comparison")


# ============================================================
# Step 6: Visualization
# ============================================================
print("\n" + "="*70)
print("STEP 6: Visualization")
print("="*70)

if len(corr_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # A: Volcano plot (rho vs -log10 p)
    ax = axes[0, 0]
    colors = ['red' if c == 'hypermethylated_silenced' else
              'blue' if c == 'positive_correlation' else
              'green' if c == 'hypomethylated_active' else 'gray'
              for c in corr_df['methylation_class']]
    ax.scatter(corr_df['spearman_rho'], -np.log10(corr_df['spearman_fdr'] + 1e-300),
               c=colors, alpha=0.7, s=50)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spearman rho (methylation vs expression)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('Methylation-Expression Correlation\n(TCGA-STAD, 92 candidate genes)')
    # Label top genes
    for _, row in corr_df.head(5).iterrows():
        ax.annotate(row['gene'], (row['spearman_rho'], -np.log10(row['spearman_fdr'] + 1e-300)),
                   fontsize=8, ha='right')
    for _, row in corr_df.tail(3).iterrows():
        ax.annotate(row['gene'], (row['spearman_rho'], -np.log10(row['spearman_fdr'] + 1e-300)),
                   fontsize=8, ha='left')

    # B: Distribution of rho values
    ax = axes[0, 1]
    ax.hist(corr_df['spearman_rho'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Spearman rho')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Methylation-Expression\nCorrelations (92 genes)')
    ax.text(0.05, 0.95, f"Mean rho = {corr_df['spearman_rho'].mean():.3f}\n"
            f"Negative (rho<-0.2): {(corr_df['spearman_rho']<-0.2).sum()}\n"
            f"Positive (rho>0.2): {(corr_df['spearman_rho']>0.2).sum()}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # C: Mean beta distribution
    ax = axes[1, 0]
    ax.scatter(corr_df['mean_beta'], corr_df['mean_expression'],
               c=corr_df['spearman_rho'], cmap='RdBu_r', vmin=-0.5, vmax=0.5,
               s=60, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Mean Promoter Beta (methylation)')
    ax.set_ylabel('Mean Expression (log2)')
    ax.set_title('Methylation Level vs Expression Level')
    plt.colorbar(ax.collections[0], ax=ax, label='Spearman rho')

    # D: Classification pie
    ax = axes[1, 1]
    class_counts = corr_df['methylation_class'].value_counts()
    colors_pie = {'hypermethylated_silenced': '#d62728', 'positive_correlation': '#2ca02c',
                  'hypomethylated_active': '#1f77b4', 'neutral': '#7f7f7f'}
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.0f%%',
           colors=[colors_pie.get(c, 'gray') for c in class_counts.index])
    ax.set_title('Methylation-Expression Classification\n(92 candidate genes)')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/methylation_expression_tcga.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/methylation_expression_tcga.png")


# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Candidate genes analyzed: {len(CANDIDATES)}")
print(f"  Genes with promoter probes: {len(gene_methyl_df)}")
if len(corr_df) > 0:
    print(f"  Genes with methyl-expression correlation: {len(corr_df)}")
    print(f"  Strong negative correlation (classic silencing): "
          f"{(corr_df['spearman_rho'] < -0.3).sum()}")
    print(f"  Key findings for mechanism:")
    # Check specific genes of interest
    for gene in ['OLFM4', 'CCL3', 'CLDN7', 'REG4', 'FABP1', 'GKN1', 'GKN2', 'TFF1']:
        if gene in corr_df['gene'].values:
            row = corr_df[corr_df['gene'] == gene].iloc[0]
            print(f"    {gene}: rho={row['spearman_rho']:.3f}, beta={row['mean_beta']:.3f}, "
                  f"class={row['methylation_class']}")

print("\nDone!")
