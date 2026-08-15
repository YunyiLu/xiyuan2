"""
Step 10f: Final Multi-Omics Synthesis
  Combines ALL evidence layers into final ranked gene list:
  - Original TransformationScore (scRNA + spatial + bulk + network)
  - Methylation integration (4 datasets)
  - Epigenome validation
  - GSE249874 independent scRNA validation (if available)

  Produces definitive output for the project.

Output:
  - results/final_92gene_multiomics_summary.csv
  - results/final_top20_biomarker_candidates.csv
  - figures/final_multiomics_summary.png
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"
FIG_DIR = f"{BASE}/figures"

# Load all evidence
updated_df = pd.read_csv(f"{RES_DIR}/unified_discovery_with_methylation.csv")
driver_df = pd.read_csv(f"{RES_DIR}/methylation_driver_classification.csv")
cross_df = pd.read_csv(f"{RES_DIR}/methylation_cross_dataset_summary.csv")
temporal_df = pd.read_csv(f"{RES_DIR}/methylation_temporal_dynamics.csv")
epi_df = pd.read_csv(f"{RES_DIR}/epigenome_validation_summary.csv")
temporal_expr = pd.read_csv(f"{RES_DIR}/mechanism_temporal_ordering.csv")
tf_pairs = pd.read_csv(f"{RES_DIR}/mechanism_tf_target_pairs.csv")

# Check if GSE249874 validation is ready
gse249874_path = f"{RES_DIR}/gse249874_validation_summary.csv"
has_249874 = os.path.exists(gse249874_path)
if has_249874:
    gse249874_df = pd.read_csv(gse249874_path)
    print(f"GSE249874 validation loaded: {len(gse249874_df)} genes")
else:
    print("GSE249874 validation not yet available (running in background)")

print(f"Loaded {len(updated_df)} genes with all evidence layers")

# ============================================================
# Build Final Evidence Matrix
# ============================================================
print("\n" + "="*70)
print("BUILDING FINAL MULTI-OMICS EVIDENCE MATRIX")
print("="*70)

final = updated_df[['gene', 'TransformationScore', 'methylation_score',
                     'transformation_score_v2', 'methylation_class', 'rank_v2']].copy()

# Add epigenome concordance
epi_merge = epi_df[['gene', 'epigenome_concordance_score', 'evidence_list']].copy()
final = final.merge(epi_merge, on='gene', how='left')

# Add temporal expression info
temporal_merge = temporal_expr[['gene', 'onset_stage', 'temporal_class']].copy()
final = final.merge(temporal_merge, on='gene', how='left')

# Add driver classification details
driver_merge = driver_df[['gene', 'classification', 'methylation_change',
                           'has_neg_methyl_expr_corr', 'is_tf_target', 'delta_beta']].copy()
final = final.merge(driver_merge, on='gene', how='left')

# Add methylation state
cross_merge = cross_df[['gene', 'mean_beta_all', 'methylation_state', 'n_datasets',
                         'tcga_rho', 'tcga_fdr']].copy()
final = final.merge(cross_merge, on='gene', how='left')

# Add temporal methylation change
if len(temporal_df) > 0:
    temp_merge = temporal_df[['gene', 'delta_beta_gastritis_to_IM', 'fdr']].rename(
        columns={'fdr': 'temporal_methyl_fdr'}).copy()
    final = final.merge(temp_merge, on='gene', how='left')

# Add GSE249874 validation if available
if has_249874:
    val_merge = gse249874_df[['gene', 'IM_vs_Gastritis_logFC', 'IM_vs_Gastritis_fdr']].copy()
    val_merge.columns = ['gene', 'gse249874_logFC', 'gse249874_fdr']
    final = final.merge(val_merge, on='gene', how='left')

# Count TF regulators per gene
tf_count = tf_pairs.groupby('target_gene').size().reset_index(name='n_tf_regulators')
tf_count.columns = ['gene', 'n_tf_regulators']
final = final.merge(tf_count, on='gene', how='left')
final['n_tf_regulators'] = final['n_tf_regulators'].fillna(0).astype(int)

# ============================================================
# Compute Final Composite Score
# ============================================================
print("\n" + "="*70)
print("COMPUTING FINAL COMPOSITE SCORE")
print("="*70)

# Final score weights:
# 0.40 × TransformationScore (original multi-omics discovery)
# 0.20 × methylation_score (methylation evidence)
# 0.15 × epigenome_concordance (normalized to 0-1)
# 0.10 × temporal_consistency (early onset = higher score)
# 0.10 × independent_validation (GSE249874 if available)
# 0.05 × regulatory_complexity (TF connections)

# Normalize components to 0-1
max_ts = final['TransformationScore'].max()
final['norm_transformation'] = final['TransformationScore'] / max_ts if max_ts > 0 else 0

max_epi = final['epigenome_concordance_score'].max()
final['norm_epigenome'] = final['epigenome_concordance_score'] / max_epi if max_epi > 0 else 0

# Temporal: early onset = higher
onset_scores = {'LGIN': 1.0, 'HGIN': 0.7, 'EGC': 0.4, 'none': 0}
final['norm_temporal'] = final['onset_stage'].map(onset_scores).fillna(0)

# Regulatory complexity
max_tf = final['n_tf_regulators'].max()
final['norm_regulatory'] = final['n_tf_regulators'] / max_tf if max_tf > 0 else 0

# Validation (if available)
if has_249874 and 'gse249874_logFC' in final.columns:
    # Positive logFC in IM vs Gastritis = validated
    max_fc = final['gse249874_logFC'].abs().max()
    final['norm_validation'] = final['gse249874_logFC'].clip(0, None) / max_fc if max_fc > 0 else 0
    final['norm_validation'] = final['norm_validation'].fillna(0)
    w_validation = 0.10
    w_transformation = 0.35
else:
    final['norm_validation'] = 0
    w_validation = 0
    w_transformation = 0.40

# Compute final score
final['FinalScore'] = (
    w_transformation * final['norm_transformation'] +
    0.20 * final['methylation_score'] +
    0.15 * final['norm_epigenome'] +
    0.10 * final['norm_temporal'] +
    w_validation * final['norm_validation'] +
    0.05 * final['norm_regulatory']
)

# Rank
final = final.sort_values('FinalScore', ascending=False)
final['FinalRank'] = range(1, len(final) + 1)

# Save
final.to_csv(f"{RES_DIR}/final_92gene_multiomics_summary.csv", index=False)
print(f"  Saved: final_92gene_multiomics_summary.csv")

# Top 20
top20 = final.head(20).copy()
top20.to_csv(f"{RES_DIR}/final_top20_biomarker_candidates.csv", index=False)
print(f"  Saved: final_top20_biomarker_candidates.csv")

print(f"\n  FINAL TOP 20 BIOMARKER CANDIDATES:")
print(f"  {'Rank':<5}{'Gene':<10}{'Score':<8}{'Class':<30}{'Onset':<8}{'Methyl':<8}{'Epi':<5}")
print(f"  {'-'*70}")
for _, r in top20.iterrows():
    print(f"  {r['FinalRank']:<5}{r['gene']:<10}{r['FinalScore']:.4f}  "
          f"{str(r.get('classification','')):<28}{str(r.get('onset_stage','')):<8}"
          f"{r['methylation_score']:.3f}   {r.get('epigenome_concordance_score', 0):.0f}")


# ============================================================
# Key Biological Insights
# ============================================================
print("\n" + "="*70)
print("KEY BIOLOGICAL INSIGHTS")
print("="*70)

# 1. Methylation-driven vs TF-driven in top 20
top20_classes = top20['classification'].value_counts()
print(f"\n  Top 20 composition:")
for cls, n in top20_classes.items():
    print(f"    {cls}: {n}")

# 2. Temporal ordering in top 20
top20_onset = top20['onset_stage'].value_counts()
print(f"\n  Onset stage in top 20:")
for stage, n in top20_onset.items():
    print(f"    {stage}: {n}")

# 3. Multi-omics evidence layers
print(f"\n  Evidence coverage in top 20:")
print(f"    Mean TransformationScore: {top20['TransformationScore'].mean():.4f}")
print(f"    Mean methylation_score: {top20['methylation_score'].mean():.3f}")
print(f"    Mean epigenome concordance: {top20['epigenome_concordance_score'].mean():.1f}/6")
print(f"    Genes with TCGA neg corr: {top20['has_neg_methyl_expr_corr'].sum()}")
print(f"    Genes with temporal methyl change: {(top20['delta_beta_gastritis_to_IM'].abs() > 0.05).sum() if 'delta_beta_gastritis_to_IM' in top20.columns else 'N/A'}")


# ============================================================
# Visualization
# ============================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# A: Final score bar plot (top 20)
ax = axes[0, 0]
colors_bar = []
for cls in top20['classification']:
    if 'methylation_driven_activation' in str(cls):
        colors_bar.append('#2ca02c')
    elif 'methylation_driven_silencing' in str(cls):
        colors_bar.append('#d62728')
    elif 'TF_driven' in str(cls):
        colors_bar.append('#1f77b4')
    elif 'other_regulation' in str(cls):
        colors_bar.append('#ff7f0e')
    else:
        colors_bar.append('#7f7f7f')

ax.barh(range(20), top20['FinalScore'].values, color=colors_bar)
ax.set_yticks(range(20))
ax.set_yticklabels(top20['gene'].values, fontsize=8)
ax.set_xlabel('Final Composite Score')
ax.set_title('Top 20 Biomarker Candidates')
ax.invert_yaxis()

# B: Score decomposition (stacked bar)
ax = axes[0, 1]
components = ['norm_transformation', 'methylation_score', 'norm_epigenome', 'norm_temporal']
weights = [w_transformation, 0.20, 0.15, 0.10]
labels = ['Multi-omics', 'Methylation', 'Epigenome', 'Temporal']
bottom = np.zeros(20)
for comp, w, label in zip(components, weights, labels):
    vals = top20[comp].values * w
    ax.barh(range(20), vals, left=bottom, label=label)
    bottom += vals
ax.set_yticks(range(20))
ax.set_yticklabels(top20['gene'].values, fontsize=8)
ax.set_xlabel('Score Contribution')
ax.set_title('Score Decomposition (Top 20)')
ax.legend(fontsize=8)
ax.invert_yaxis()

# C: Heatmap of evidence layers (top 20)
ax = axes[0, 2]
hm_data = top20[['gene', 'norm_transformation', 'methylation_score',
                  'norm_epigenome', 'norm_temporal', 'norm_regulatory']].copy()
hm_data = hm_data.set_index('gene')
hm_data.columns = ['Discovery', 'Methylation', 'Epigenome', 'Temporal', 'TF Network']
sns.heatmap(hm_data, ax=ax, cmap='YlOrRd', vmin=0, vmax=1, annot=True, fmt='.2f',
            xticklabels=True, yticklabels=True, cbar_kws={'shrink': 0.8})
ax.set_title('Evidence Layers (Top 20)')

# D: Onset stage distribution
ax = axes[1, 0]
onset_counts = final['onset_stage'].value_counts()
ax.pie(onset_counts.values, labels=onset_counts.index, autopct='%1.0f%%',
       colors=['#2ca02c', '#ff7f0e', '#d62728', '#7f7f7f'])
ax.set_title('Expression Onset Stage\n(92 candidates)')

# E: Classification sunburst (simplified as nested pie)
ax = axes[1, 1]
class_counts = final['classification'].value_counts()
colors_pie = [{'methylation_driven_activation': '#2ca02c',
               'methylation_driven_silencing': '#d62728',
               'TF_driven': '#1f77b4',
               'other_regulation': '#ff7f0e',
               'demethylated_but_weak_correlation': '#9467bd',
               'ambiguous': '#7f7f7f'}.get(c, 'gray') for c in class_counts.index]
ax.pie(class_counts.values, labels=[c.replace('_', '\n')[:25] for c in class_counts.index],
       autopct='%1.0f%%', colors=colors_pie, textprops={'fontsize': 7})
ax.set_title('Regulatory Mechanism Classification')

# F: Score correlation matrix
ax = axes[1, 2]
corr_data = final[['TransformationScore', 'methylation_score',
                    'epigenome_concordance_score', 'norm_temporal', 'FinalScore']].copy()
corr_data.columns = ['Discovery', 'Methylation', 'Epigenome', 'Temporal', 'Final']
corr_matrix = corr_data.corr(method='spearman')
sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            vmin=-1, vmax=1)
ax.set_title('Score Layer Correlations\n(Spearman)')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/final_multiomics_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/final_multiomics_summary.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"  Total candidate genes: 92")
print(f"  Data sources integrated: 8+ (scRNA, spatial, 4 bulk, 4 methylation, epigenome)")
print(f"  Top 5 final candidates: {', '.join(top20.head(5)['gene'].tolist())}")
