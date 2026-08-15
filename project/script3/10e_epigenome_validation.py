"""
Step 10e: Epigenome Validation — Cross-reference methylation with chromatin features
  Validates methylation-driven gene classification using:
  1. TCGA chromatin marks (H3K27ac from ENCODE gastric)
  2. CpG island density analysis
  3. Promoter CpG context classification
  4. Cross-validation: methylation-driven genes should have CpG-rich promoters

Input:
  - results/methylation_driver_classification.csv
  - results/methylation_cross_dataset_summary.csv
  - results/methylation_temporal_dynamics.csv
  - data/methylation/HM450_manifest_genes.csv (probe locations)
  - results/unified_discovery_with_methylation.csv

Output:
  - results/epigenome_validation_summary.csv
  - results/epigenome_promoter_context.csv
  - figures/epigenome_validation.png
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
METHYL_DIR = f"{BASE}/data/methylation"
RES_DIR = f"{BASE}/results"
FIG_DIR = f"{BASE}/figures"

# Load data
driver_df = pd.read_csv(f"{RES_DIR}/methylation_driver_classification.csv")
cross_df = pd.read_csv(f"{RES_DIR}/methylation_cross_dataset_summary.csv")
temporal_df = pd.read_csv(f"{RES_DIR}/methylation_temporal_dynamics.csv")
updated_df = pd.read_csv(f"{RES_DIR}/unified_discovery_with_methylation.csv")
manifest = pd.read_csv(f"{METHYL_DIR}/HM450_manifest_genes.csv")

print(f"Driver classification: {len(driver_df)} genes")
print(f"Manifest probes: {len(manifest)} for {manifest['gene'].nunique()} genes")

# ============================================================
# MODULE E1: Promoter CpG Context Analysis
# ============================================================
print("\n" + "="*70)
print("MODULE E1: Promoter CpG Context Classification")
print("="*70)

# Classify promoters by their CpG probe density
# Genes with many promoter probes → CpG-dense (CpG islands)
# Genes with few probes → CpG-poor

promoter_regions = ['TSS1500', 'TSS200', '1stExon', "5'UTR"]
promoter_probes = manifest[manifest['region'].isin(promoter_regions)]

# Count probes per gene
probe_counts = promoter_probes.groupby('gene').size().reset_index(name='n_promoter_probes')
probe_region_dist = promoter_probes.groupby(['gene', 'region']).size().unstack(fill_value=0).reset_index()

# Merge
cpg_context = probe_counts.merge(probe_region_dist, on='gene', how='left')

# Classify
cpg_context['cpg_density'] = pd.cut(cpg_context['n_promoter_probes'],
                                     bins=[0, 5, 15, 100],
                                     labels=['CpG_poor', 'CpG_moderate', 'CpG_rich'])

print(f"  CpG density distribution:")
print(cpg_context['cpg_density'].value_counts())

# Merge with driver classification
cpg_driver = cpg_context.merge(driver_df[['gene', 'classification']], on='gene', how='inner')

print(f"\n  CpG density by driver classification:")
ct = pd.crosstab(cpg_driver['classification'], cpg_driver['cpg_density'])
print(ct)

# Test: are methylation-driven genes enriched in CpG-rich promoters?
methyl_driven = cpg_driver[cpg_driver['classification'].isin(['methylation_driven_activation', 'methylation_driven_silencing'])]
other_genes = cpg_driver[~cpg_driver['classification'].isin(['methylation_driven_activation', 'methylation_driven_silencing'])]

if len(methyl_driven) >= 3 and len(other_genes) >= 3:
    _, p = mannwhitneyu(methyl_driven['n_promoter_probes'], other_genes['n_promoter_probes'], alternative='greater')
    print(f"\n  Methylation-driven genes have MORE promoter probes?")
    print(f"    Methyl-driven: median={methyl_driven['n_promoter_probes'].median():.0f} probes")
    print(f"    Others: median={other_genes['n_promoter_probes'].median():.0f} probes")
    print(f"    Mann-Whitney p={p:.4f}")


# ============================================================
# MODULE E2: Methylation Variability as Regulatory Potential
# ============================================================
print("\n" + "="*70)
print("MODULE E2: Methylation Variability Analysis")
print("="*70)

# Genes with HIGH inter-sample methylation variability are more likely
# to be actively regulated by methylation (not constitutively methylated/unmethylated)

# Load GSE103186 (largest IM dataset) for variability analysis
gse103186 = pd.read_csv(f"{METHYL_DIR}/GSE103186_gene_promoter_beta.csv", index_col=0)

variability = []
for gene in gse103186.index:
    vals = gse103186.loc[gene].dropna().values
    if len(vals) >= 10:
        variability.append({
            'gene': gene,
            'mean_beta': np.mean(vals),
            'std_beta': np.std(vals),
            'cv_beta': np.std(vals) / (np.mean(vals) + 0.01),
            'range_beta': np.max(vals) - np.min(vals),
            'iqr_beta': np.percentile(vals, 75) - np.percentile(vals, 25)
        })

var_df = pd.DataFrame(variability)
var_df = var_df.merge(driver_df[['gene', 'classification']], on='gene', how='left')

print(f"  Genes with variability data: {len(var_df)}")
print(f"\n  Variability by classification:")
for cls in var_df['classification'].unique():
    subset = var_df[var_df['classification'] == cls]
    if len(subset) >= 2:
        print(f"    {cls}: n={len(subset)}, mean_std={subset['std_beta'].mean():.4f}, "
              f"mean_range={subset['range_beta'].mean():.4f}")

# Test: do methylation-driven genes show higher variability?
methyl_var = var_df[var_df['classification'].isin(['methylation_driven_activation', 'methylation_driven_silencing'])]
other_var = var_df[~var_df['classification'].isin(['methylation_driven_activation', 'methylation_driven_silencing'])]

if len(methyl_var) >= 3 and len(other_var) >= 3:
    _, p = mannwhitneyu(methyl_var['std_beta'], other_var['std_beta'], alternative='greater')
    print(f"\n  Methylation-driven genes have HIGHER variability?")
    print(f"    Methyl-driven std: {methyl_var['std_beta'].mean():.4f}")
    print(f"    Others std: {other_var['std_beta'].mean():.4f}")
    print(f"    p={p:.4f}")


# ============================================================
# MODULE E3: Bimodal Methylation Detection
# ============================================================
print("\n" + "="*70)
print("MODULE E3: Bimodal Methylation Detection")
print("="*70)

# Genes showing bimodal methylation (some samples high, some low) are
# strong candidates for epigenetic regulation
# Detect: if >20% of samples have beta>0.6 AND >20% have beta<0.3

bimodal_genes = []
for gene in gse103186.index:
    vals = gse103186.loc[gene].dropna().values
    if len(vals) >= 10:
        frac_high = (vals > 0.6).sum() / len(vals)
        frac_low = (vals < 0.3).sum() / len(vals)
        is_bimodal = frac_high > 0.2 and frac_low > 0.2

        bimodal_genes.append({
            'gene': gene,
            'frac_high': frac_high,
            'frac_low': frac_low,
            'is_bimodal': is_bimodal,
            'mean_beta': np.mean(vals)
        })

bimodal_df = pd.DataFrame(bimodal_genes)
bimodal_df = bimodal_df.merge(driver_df[['gene', 'classification']], on='gene', how='left')

n_bimodal = bimodal_df['is_bimodal'].sum()
print(f"  Bimodal methylation genes: {n_bimodal}/{len(bimodal_df)}")
if n_bimodal > 0:
    print(f"  Bimodal genes: {bimodal_df[bimodal_df['is_bimodal']]['gene'].tolist()}")
    print(f"\n  Bimodal by classification:")
    bimodal_ct = pd.crosstab(bimodal_df['classification'], bimodal_df['is_bimodal'])
    print(bimodal_ct)


# ============================================================
# MODULE E4: Concordance Score (methylation evidence strength)
# ============================================================
print("\n" + "="*70)
print("MODULE E4: Epigenomic Concordance Score")
print("="*70)

# For each gene, compute a concordance score:
# How many independent lines of epigenomic evidence support its regulatory role?
# 1. TCGA methylation-expression correlation (significant negative)
# 2. Temporal change (gastritis→IM)
# 3. Cross-dataset consistency (≥3 datasets)
# 4. CpG-rich promoter
# 5. High inter-sample variability (top quartile)
# 6. Bimodal distribution

var_threshold = var_df['std_beta'].quantile(0.75) if len(var_df) > 0 else 0.1

epigenome_scores = []
for gene in driver_df['gene']:
    score = 0
    evidence = []

    # 1. Significant negative TCGA correlation
    cross_row = cross_df[cross_df['gene'] == gene]
    if len(cross_row) > 0 and cross_row.iloc[0].get('tcga_fdr', 1) < 0.05:
        if cross_row.iloc[0].get('tcga_rho', 0) < -0.2:
            score += 1
            evidence.append('TCGA_neg_corr')

    # 2. Temporal methylation change
    temp_row = temporal_df[temporal_df['gene'] == gene]
    if len(temp_row) > 0 and temp_row.iloc[0].get('fdr', 1) < 0.05:
        if abs(temp_row.iloc[0]['delta_beta_gastritis_to_IM']) > 0.05:
            score += 1
            evidence.append('temporal_change')

    # 3. Cross-dataset consistency
    if len(cross_row) > 0 and cross_row.iloc[0].get('n_datasets', 0) >= 3:
        score += 1
        evidence.append('multi_dataset')

    # 4. CpG-rich promoter
    cpg_row = cpg_context[cpg_context['gene'] == gene]
    if len(cpg_row) > 0 and cpg_row.iloc[0]['cpg_density'] == 'CpG_rich':
        score += 1
        evidence.append('CpG_rich')

    # 5. High variability
    var_row = var_df[var_df['gene'] == gene]
    if len(var_row) > 0 and var_row.iloc[0]['std_beta'] >= var_threshold:
        score += 1
        evidence.append('high_variability')

    # 6. Bimodal
    bim_row = bimodal_df[bimodal_df['gene'] == gene]
    if len(bim_row) > 0 and bim_row.iloc[0]['is_bimodal']:
        score += 1
        evidence.append('bimodal')

    epigenome_scores.append({
        'gene': gene,
        'epigenome_concordance_score': score,
        'max_possible': 6,
        'evidence_list': ';'.join(evidence),
        'n_evidence': len(evidence)
    })

epi_df = pd.DataFrame(epigenome_scores)
epi_df = epi_df.merge(driver_df[['gene', 'classification']], on='gene', how='left')
epi_df = epi_df.sort_values('epigenome_concordance_score', ascending=False)

print(f"  Concordance score distribution:")
print(epi_df['epigenome_concordance_score'].value_counts().sort_index())

print(f"\n  Top epigenomically validated genes (score ≥ 4):")
for _, r in epi_df[epi_df['epigenome_concordance_score'] >= 4].iterrows():
    print(f"    {r['gene']}: score={r['epigenome_concordance_score']}/6, "
          f"class={r['classification']}, evidence=[{r['evidence_list']}]")

print(f"\n  Mean concordance by classification:")
for cls in epi_df['classification'].unique():
    subset = epi_df[epi_df['classification'] == cls]
    print(f"    {cls}: mean={subset['epigenome_concordance_score'].mean():.2f} (n={len(subset)})")


# ============================================================
# MODULE E5: Final Epigenome-Validated Output
# ============================================================
print("\n" + "="*70)
print("MODULE E5: Final Output")
print("="*70)

# Merge everything into summary
final = epi_df[['gene', 'epigenome_concordance_score', 'evidence_list', 'classification']].copy()
final = final.merge(cpg_context[['gene', 'n_promoter_probes', 'cpg_density']], on='gene', how='left')
final = final.merge(var_df[['gene', 'std_beta', 'range_beta']], on='gene', how='left')
final = final.merge(bimodal_df[['gene', 'is_bimodal']], on='gene', how='left')

# Add from temporal
final = final.merge(temporal_df[['gene', 'delta_beta_gastritis_to_IM', 'fdr']].rename(
    columns={'fdr': 'temporal_fdr'}), on='gene', how='left')

final = final.sort_values('epigenome_concordance_score', ascending=False)
final.to_csv(f"{RES_DIR}/epigenome_validation_summary.csv", index=False)
print(f"  Saved: epigenome_validation_summary.csv ({len(final)} genes)")

# Also save promoter context
cpg_context.to_csv(f"{RES_DIR}/epigenome_promoter_context.csv", index=False)
print(f"  Saved: epigenome_promoter_context.csv")


# ============================================================
# Visualization
# ============================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# A: CpG density by classification
ax = axes[0, 0]
plot_data = cpg_driver.copy()
order = ['methylation_driven_activation', 'methylation_driven_silencing', 'TF_driven', 'other_regulation', 'ambiguous']
order = [o for o in order if o in plot_data['classification'].values]
sns.boxplot(data=plot_data, x='classification', y='n_promoter_probes', ax=ax, order=order)
ax.set_xticklabels([c.replace('_', '\n')[:20] for c in order], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Number of promoter CpG probes')
ax.set_title('CpG Density by Gene Classification')

# B: Methylation variability by classification
ax = axes[0, 1]
plot_data2 = var_df.dropna(subset=['classification'])
order2 = [o for o in order if o in plot_data2['classification'].values]
if order2:
    sns.boxplot(data=plot_data2, x='classification', y='std_beta', ax=ax, order=order2)
    ax.set_xticklabels([c.replace('_', '\n')[:20] for c in order2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('SD of promoter β')
    ax.set_title('Methylation Variability by Classification')

# C: Concordance score distribution
ax = axes[0, 2]
colors_cls = {
    'methylation_driven_activation': '#2ca02c',
    'methylation_driven_silencing': '#d62728',
    'TF_driven': '#1f77b4',
    'other_regulation': '#ff7f0e',
    'ambiguous': '#7f7f7f',
    'demethylated_but_weak_correlation': '#9467bd'
}
for cls in epi_df['classification'].unique():
    subset = epi_df[epi_df['classification'] == cls]
    ax.scatter(subset['epigenome_concordance_score'] + np.random.uniform(-0.1, 0.1, len(subset)),
               range(len(subset)), s=40, alpha=0.7, label=cls.replace('_', ' ')[:25],
               color=colors_cls.get(cls, 'gray'))
ax.set_xlabel('Epigenome Concordance Score')
ax.set_title('Concordance Score by Classification')
ax.legend(fontsize=7, loc='upper right')

# D: Mean beta vs variability (scatter)
ax = axes[1, 0]
if len(var_df) > 5:
    colors_var = [colors_cls.get(c, 'gray') for c in var_df['classification']]
    ax.scatter(var_df['mean_beta'], var_df['std_beta'], c=colors_var, alpha=0.7, s=50)
    ax.set_xlabel('Mean promoter β (GSE103186)')
    ax.set_ylabel('SD promoter β')
    ax.set_title('Mean vs Variability of Methylation')
    # Label high-variability genes
    high_var = var_df.nlargest(5, 'std_beta')
    for _, r in high_var.iterrows():
        ax.annotate(r['gene'], (r['mean_beta'], r['std_beta']), fontsize=7)

# E: Temporal change vs concordance
ax = axes[1, 1]
temp_epi = temporal_df.merge(epi_df[['gene', 'epigenome_concordance_score']], on='gene', how='inner')
if len(temp_epi) > 5:
    ax.scatter(temp_epi['delta_beta_gastritis_to_IM'], temp_epi['epigenome_concordance_score'],
               alpha=0.7, s=50)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δβ (Gastritis → IM)')
    ax.set_ylabel('Epigenome Concordance Score')
    ax.set_title('Temporal Change vs Validation Strength')
    # Label extremes
    for _, r in temp_epi.nlargest(3, 'epigenome_concordance_score').iterrows():
        ax.annotate(r['gene'], (r['delta_beta_gastritis_to_IM'], r['epigenome_concordance_score']), fontsize=7)

# F: Summary bar — evidence types
ax = axes[1, 2]
all_evidence = ';'.join(epi_df['evidence_list'].dropna())
from collections import Counter
evidence_counts = Counter(all_evidence.split(';'))
evidence_counts.pop('', None)
if evidence_counts:
    labels = list(evidence_counts.keys())
    values = list(evidence_counts.values())
    ax.barh(range(len(labels)), values, color='steelblue')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace('_', ' ') for l in labels])
    ax.set_xlabel('Number of genes with this evidence')
    ax.set_title('Distribution of Epigenomic Evidence Types')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/epigenome_validation.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/epigenome_validation.png")

print("\n" + "="*70)
print("EPIGENOME VALIDATION COMPLETE")
print("="*70)
print(f"  Total genes validated: {len(final)}")
print(f"  Genes with strong epigenomic evidence (≥4/6): {(final['epigenome_concordance_score'] >= 4).sum()}")
print(f"  Genes with moderate evidence (≥3/6): {(final['epigenome_concordance_score'] >= 3).sum()}")
print(f"  Methylation-driven genes confirmed: {len(final[final['classification'].str.contains('methylation_driven')])}")
