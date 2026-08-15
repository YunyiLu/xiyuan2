"""
Step 10d: Methylation-Expression Integration Pipeline
  Integrates all methylation data with transcriptomic data for 92 candidates.

Modules:
  M1: Cross-dataset methylation consistency (TCGA + GSE103186 + GSE220511 + GSE178925)
  M2: Methylation temporal dynamics (IM stages vs expression onset)
  M3: Methylation-driven vs TF-driven gene classification
  M4: TF binding site methylation analysis
  M5: Updated TransformationScore with methylation layer

Input:
  - results/tcga_methylation_expression_corr.csv (TCGA analysis)
  - data/methylation/GSE103186_gene_promoter_beta.csv (191 IM samples)
  - data/methylation/GSE220511_gene_promoter_beta.csv (26 IM crypt samples)
  - data/methylation/GSE178925_gene_promoter_beta.csv (24 gastritis samples)
  - results/mechanism_temporal_ordering.csv (expression timing)
  - results/mechanism_tf_target_pairs.csv (TF regulation)
  - results/unified_discovery_ranked.csv (92 genes)

Output:
  - results/methylation_cross_dataset_summary.csv
  - results/methylation_temporal_dynamics.csv
  - results/methylation_driver_classification.csv
  - results/methylation_tf_binding_analysis.csv
  - results/unified_discovery_with_methylation.csv (TransformationScore v2)
  - figures/methylation_integration.png
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
METHYL_DIR = f"{BASE}/data/methylation"
RES_DIR = f"{BASE}/results"
FIG_DIR = f"{BASE}/figures"

# Load core data
candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()
print(f"Loaded {len(CANDIDATES)} candidate genes")

tcga_corr = pd.read_csv(f"{RES_DIR}/tcga_methylation_expression_corr.csv")
temporal = pd.read_csv(f"{RES_DIR}/mechanism_temporal_ordering.csv")
tf_pairs = pd.read_csv(f"{RES_DIR}/mechanism_tf_target_pairs.csv")

# Load methylation datasets
datasets = {}
for name in ['GSE103186', 'GSE220511', 'GSE178925']:
    path = f"{METHYL_DIR}/{name}_gene_promoter_beta.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        datasets[name] = df
        print(f"  {name}: {df.shape[0]} genes × {df.shape[1]} samples")

# ============================================================
# MODULE M1: Cross-Dataset Methylation Consistency
# ============================================================
print("\n" + "="*70)
print("MODULE M1: Cross-Dataset Methylation Consistency")
print("="*70)

# For each gene, compute mean promoter beta across datasets
cross_dataset = []
for gene in CANDIDATES:
    row = {'gene': gene}

    # TCGA (tumor)
    tcga_row = tcga_corr[tcga_corr['gene'] == gene]
    if len(tcga_row) > 0:
        row['tcga_mean_beta'] = tcga_row.iloc[0]['mean_beta']
        row['tcga_rho'] = tcga_row.iloc[0]['spearman_rho']
        row['tcga_fdr'] = tcga_row.iloc[0]['spearman_fdr']
        row['tcga_class'] = tcga_row.iloc[0]['methylation_class']
    else:
        row['tcga_mean_beta'] = np.nan
        row['tcga_rho'] = np.nan
        row['tcga_fdr'] = np.nan
        row['tcga_class'] = 'no_data'

    # GSE103186 (IM tissue, 191 samples)
    if 'GSE103186' in datasets and gene in datasets['GSE103186'].index:
        vals = datasets['GSE103186'].loc[gene].dropna().values
        row['gse103186_mean_beta'] = np.mean(vals)
        row['gse103186_std_beta'] = np.std(vals)
        row['gse103186_n'] = len(vals)
    else:
        row['gse103186_mean_beta'] = np.nan
        row['gse103186_std_beta'] = np.nan
        row['gse103186_n'] = 0

    # GSE220511 (IM crypts, 26 samples)
    if 'GSE220511' in datasets and gene in datasets['GSE220511'].index:
        vals = datasets['GSE220511'].loc[gene].dropna().values
        row['gse220511_mean_beta'] = np.mean(vals)
        row['gse220511_n'] = len(vals)
    else:
        row['gse220511_mean_beta'] = np.nan
        row['gse220511_n'] = 0

    # GSE178925 (gastritis, 24 samples — this is BEFORE IM)
    if 'GSE178925' in datasets and gene in datasets['GSE178925'].index:
        vals = datasets['GSE178925'].loc[gene].dropna().values
        row['gse178925_mean_beta'] = np.mean(vals)
        row['gse178925_n'] = len(vals)
    else:
        row['gse178925_mean_beta'] = np.nan
        row['gse178925_n'] = 0

    # Cross-dataset consensus
    betas = [row.get(f'{k}_mean_beta') for k in ['tcga', 'gse103186', 'gse220511', 'gse178925']]
    valid_betas = [b for b in betas if b is not None and not np.isnan(b)]
    if len(valid_betas) >= 2:
        row['mean_beta_all'] = np.mean(valid_betas)
        row['n_datasets'] = len(valid_betas)
        # Classify overall methylation state
        if row['mean_beta_all'] < 0.2:
            row['methylation_state'] = 'hypomethylated'
        elif row['mean_beta_all'] > 0.6:
            row['methylation_state'] = 'hypermethylated'
        else:
            row['methylation_state'] = 'intermediate'
    else:
        row['mean_beta_all'] = np.nan
        row['n_datasets'] = len(valid_betas)
        row['methylation_state'] = 'insufficient_data'

    cross_dataset.append(row)

cross_df = pd.DataFrame(cross_dataset)
cross_df.to_csv(f"{RES_DIR}/methylation_cross_dataset_summary.csv", index=False)

print(f"\n  Methylation state distribution:")
print(cross_df['methylation_state'].value_counts())
print(f"\n  Key genes methylation:")
for gene in ['OLFM4', 'CCL3', 'CLDN7', 'REG4', 'GKN1', 'TFF1', 'FABP1']:
    r = cross_df[cross_df['gene'] == gene]
    if len(r) > 0:
        r = r.iloc[0]
        print(f"    {gene}: mean_beta={r['mean_beta_all']:.3f}, state={r['methylation_state']}, "
              f"tcga_rho={r['tcga_rho']:.3f}" if not np.isnan(r.get('tcga_rho', np.nan)) else f"    {gene}: no TCGA data")


# ============================================================
# MODULE M2: Methylation Temporal Dynamics
# ============================================================
print("\n" + "="*70)
print("MODULE M2: Methylation Temporal Dynamics")
print("="*70)

# Compare methylation between gastritis (GSE178925) and IM (GSE103186/GSE220511)
# This tells us: did methylation CHANGE during gastritis→IM transition?

temporal_methyl = []
if 'GSE178925' in datasets and ('GSE103186' in datasets or 'GSE220511' in datasets):
    gastritis_data = datasets['GSE178925']
    im_data = datasets.get('GSE103186', datasets.get('GSE220511'))

    common_genes = list(set(gastritis_data.index) & set(im_data.index))
    print(f"  Genes in both gastritis and IM datasets: {len(common_genes)}")

    for gene in common_genes:
        gastritis_vals = gastritis_data.loc[gene].dropna().values
        im_vals = im_data.loc[gene].dropna().values

        if len(gastritis_vals) >= 3 and len(im_vals) >= 3:
            delta_beta = np.mean(im_vals) - np.mean(gastritis_vals)
            _, p_val = mannwhitneyu(im_vals, gastritis_vals, alternative='two-sided')

            # Get expression temporal info
            temp_row = temporal[temporal['gene'] == gene]
            onset = temp_row.iloc[0]['onset_stage'] if len(temp_row) > 0 else 'unknown'

            temporal_methyl.append({
                'gene': gene,
                'mean_beta_gastritis': np.mean(gastritis_vals),
                'mean_beta_IM': np.mean(im_vals),
                'delta_beta_gastritis_to_IM': delta_beta,
                'p_value': p_val,
                'methylation_direction': 'demethylated' if delta_beta < -0.05 else ('hypermethylated' if delta_beta > 0.05 else 'stable'),
                'expression_onset': onset
            })

    temporal_methyl_df = pd.DataFrame(temporal_methyl)
    if len(temporal_methyl_df) > 0:
        _, temporal_methyl_df['fdr'], _, _ = multipletests(temporal_methyl_df['p_value'], method='fdr_bh')
        temporal_methyl_df = temporal_methyl_df.sort_values('delta_beta_gastritis_to_IM')
        temporal_methyl_df.to_csv(f"{RES_DIR}/methylation_temporal_dynamics.csv", index=False)

        print(f"  Demethylated in IM (delta < -0.05): {(temporal_methyl_df['delta_beta_gastritis_to_IM'] < -0.05).sum()}")
        print(f"  Hypermethylated in IM (delta > 0.05): {(temporal_methyl_df['delta_beta_gastritis_to_IM'] > 0.05).sum()}")
        print(f"  Stable (|delta| < 0.05): {(abs(temporal_methyl_df['delta_beta_gastritis_to_IM']) <= 0.05).sum()}")

        print("\n  Top demethylated (activated) genes:")
        for _, r in temporal_methyl_df.head(5).iterrows():
            print(f"    {r['gene']}: Δβ={r['delta_beta_gastritis_to_IM']:.3f}, "
                  f"expr_onset={r['expression_onset']}, FDR={r['fdr']:.4f}")

        print("\n  Top hypermethylated (silenced) genes:")
        for _, r in temporal_methyl_df.tail(5).iterrows():
            print(f"    {r['gene']}: Δβ={r['delta_beta_gastritis_to_IM']:.3f}, "
                  f"expr_onset={r['expression_onset']}, FDR={r['fdr']:.4f}")
else:
    print("  Insufficient datasets for temporal comparison")
    temporal_methyl_df = pd.DataFrame()


# ============================================================
# MODULE M3: Methylation-Driven vs TF-Driven Classification
# ============================================================
print("\n" + "="*70)
print("MODULE M3: Driver Classification")
print("="*70)

# Classify each gene:
# 1. "Methylation-driven activation": demethylated in IM + expression up + negative methyl-expr corr
# 2. "Methylation-driven silencing": hypermethylated + expression down + negative methyl-expr corr
# 3. "TF-driven (methylation-independent)": no methylation change but expression changes (driven by TF)
# 4. "Ambiguous": mixed signals

driver_class = []
for gene in CANDIDATES:
    row = {'gene': gene}

    # Get methylation-expression correlation from TCGA
    tcga_row = tcga_corr[tcga_corr['gene'] == gene]
    has_neg_corr = len(tcga_row) > 0 and tcga_row.iloc[0]['spearman_rho'] < -0.2 and tcga_row.iloc[0]['spearman_fdr'] < 0.05

    # Get temporal methylation change
    if len(temporal_methyl_df) > 0:
        temp_row = temporal_methyl_df[temporal_methyl_df['gene'] == gene]
        if len(temp_row) > 0:
            delta = temp_row.iloc[0]['delta_beta_gastritis_to_IM']
            demethylated = delta < -0.05
            hypermethylated = delta > 0.05
            stable_methyl = abs(delta) <= 0.05
        else:
            demethylated = hypermethylated = False
            stable_methyl = True
            delta = 0
    else:
        demethylated = hypermethylated = False
        stable_methyl = True
        delta = 0

    # Get expression direction (from cascade)
    temp_expr = temporal[temporal['gene'] == gene]
    expr_up = len(temp_expr) > 0 and temp_expr.iloc[0].get('onset_stage', 'none') != 'none'

    # Is this gene a known TF target?
    is_tf_target = gene in tf_pairs['target_gene'].values

    # Classification logic
    if demethylated and expr_up and has_neg_corr:
        classification = 'methylation_driven_activation'
    elif hypermethylated and not expr_up:
        classification = 'methylation_driven_silencing'
    elif stable_methyl and expr_up and is_tf_target:
        classification = 'TF_driven'
    elif stable_methyl and expr_up and not is_tf_target:
        classification = 'other_regulation'
    elif demethylated and expr_up and not has_neg_corr:
        classification = 'demethylated_but_weak_correlation'
    else:
        classification = 'ambiguous'

    row['classification'] = classification
    row['methylation_change'] = 'demethylated' if demethylated else ('hypermethylated' if hypermethylated else 'stable')
    row['expression_up_in_cascade'] = expr_up
    row['has_neg_methyl_expr_corr'] = has_neg_corr
    row['is_tf_target'] = is_tf_target
    row['delta_beta'] = delta

    driver_class.append(row)

driver_df = pd.DataFrame(driver_class)
driver_df.to_csv(f"{RES_DIR}/methylation_driver_classification.csv", index=False)

print(f"  Classification distribution:")
print(driver_df['classification'].value_counts())
print(f"\n  Key gene classifications:")
for gene in ['OLFM4', 'CCL3', 'CLDN7', 'REG4', 'GKN1', 'TFF1', 'FABP1', 'ATF3', 'FOS']:
    r = driver_df[driver_df['gene'] == gene]
    if len(r) > 0:
        r = r.iloc[0]
        print(f"    {gene}: {r['classification']} (Δβ={r['delta_beta']:.3f}, neg_corr={r['has_neg_methyl_expr_corr']}, TF_target={r['is_tf_target']})")


# ============================================================
# MODULE M4: TF Binding Site Methylation
# ============================================================
print("\n" + "="*70)
print("MODULE M4: TF Binding Site Methylation Analysis")
print("="*70)

# Key question: Is methylation at NF-κB/CDX2 target gene promoters different?
# Compare: NF-κB targets vs non-targets methylation levels

nfkb_targets = tf_pairs[tf_pairs['TF'].isin(['RELA', 'NFKB1'])]['target_gene'].unique()
cdx2_targets = tf_pairs[tf_pairs['TF'] == 'CDX2']['target_gene'].unique() if 'CDX2' in tf_pairs['TF'].values else []

print(f"  NF-κB targets in our 92 genes: {len([g for g in nfkb_targets if g in CANDIDATES])}")
print(f"  CDX2 targets in our 92 genes: {len([g for g in cdx2_targets if g in CANDIDATES])}")

# For each TF, compare methylation of its targets vs non-targets
tf_methyl_results = []

for tf_name, targets in [('NFKB', nfkb_targets), ('CDX2', cdx2_targets)]:
    targets_in_candidates = [g for g in targets if g in CANDIDATES]
    non_targets = [g for g in CANDIDATES if g not in targets]

    if len(targets_in_candidates) < 3:
        continue

    # Mean promoter beta for targets vs non-targets (from GSE103186 IM data)
    if 'GSE103186' in datasets:
        im_data = datasets['GSE103186']
        target_betas = []
        nontarget_betas = []

        for gene in targets_in_candidates:
            if gene in im_data.index:
                target_betas.append(im_data.loc[gene].mean())
        for gene in non_targets:
            if gene in im_data.index:
                nontarget_betas.append(im_data.loc[gene].mean())

        if target_betas and nontarget_betas:
            _, p = mannwhitneyu(target_betas, nontarget_betas, alternative='two-sided')
            tf_methyl_results.append({
                'tf': tf_name,
                'n_targets': len(target_betas),
                'n_nontargets': len(nontarget_betas),
                'mean_beta_targets': np.mean(target_betas),
                'mean_beta_nontargets': np.mean(nontarget_betas),
                'delta': np.mean(target_betas) - np.mean(nontarget_betas),
                'p_value': p,
                'interpretation': 'targets_less_methylated' if np.mean(target_betas) < np.mean(nontarget_betas) else 'targets_more_methylated'
            })

if tf_methyl_results:
    tf_methyl_df = pd.DataFrame(tf_methyl_results)
    tf_methyl_df.to_csv(f"{RES_DIR}/methylation_tf_binding_analysis.csv", index=False)
    print("\n  TF target methylation comparison (in IM tissue):")
    for _, r in tf_methyl_df.iterrows():
        print(f"    {r['tf']}: targets β={r['mean_beta_targets']:.3f} vs non-targets β={r['mean_beta_nontargets']:.3f}, "
              f"Δ={r['delta']:.3f}, p={r['p_value']:.4f} → {r['interpretation']}")


# ============================================================
# MODULE M5: Updated TransformationScore with Methylation
# ============================================================
print("\n" + "="*70)
print("MODULE M5: TransformationScore v2 (with methylation)")
print("="*70)

# Add methylation evidence to scoring
# methylation_score = consistency across datasets + strength of methyl-expr correlation + temporal change

def compute_methylation_score(gene):
    score = 0

    # 1. Methyl-expression correlation strength (from TCGA, max 0.4)
    tcga_row = tcga_corr[tcga_corr['gene'] == gene]
    if len(tcga_row) > 0:
        rho = abs(tcga_row.iloc[0]['spearman_rho'])
        fdr = tcga_row.iloc[0]['spearman_fdr']
        if fdr < 0.05:
            score += min(rho * 0.5, 0.4)  # Scale: rho=0.8 → 0.4

    # 2. Cross-dataset consistency (max 0.3)
    cross_row = cross_df[cross_df['gene'] == gene]
    if len(cross_row) > 0:
        n_ds = cross_row.iloc[0]['n_datasets']
        score += n_ds * 0.075  # 4 datasets = 0.3

    # 3. Temporal methylation change matches expression (max 0.3)
    if len(temporal_methyl_df) > 0:
        temp_row = temporal_methyl_df[temporal_methyl_df['gene'] == gene]
        if len(temp_row) > 0:
            delta = temp_row.iloc[0]['delta_beta_gastritis_to_IM']
            fdr = temp_row.iloc[0]['fdr']
            # Demethylation of an upregulated gene = concordant evidence
            expr_row = temporal[temporal['gene'] == gene]
            expr_up = len(expr_row) > 0 and expr_row.iloc[0].get('onset_stage', 'none') != 'none'

            if expr_up and delta < -0.05 and fdr < 0.1:
                score += 0.3  # Strong concordant: demethylated + upregulated
            elif expr_up and delta < 0:
                score += 0.15  # Weak concordant
            elif not expr_up and delta > 0.05 and fdr < 0.1:
                score += 0.2  # Silenced concordant

    return min(score, 1.0)

# Compute methylation score for all genes
methyl_scores = {gene: compute_methylation_score(gene) for gene in CANDIDATES}

# Updated TransformationScore v2
# Original: 0.30×scRNA + 0.30×spatial + 0.25×bulk + 0.15×network
# New: 0.25×scRNA + 0.25×spatial + 0.20×bulk + 0.15×methylation + 0.10×network + 0.05×bonus

updated_df = candidates_df.copy()
updated_df['methylation_score'] = updated_df['gene'].map(methyl_scores)

# Recompute with new weights
if 'TransformationScore' in updated_df.columns:
    updated_df['transformation_score_v2'] = (
        updated_df['TransformationScore'] * 0.85 +
        updated_df['methylation_score'] * 0.15
    )
else:
    updated_df['transformation_score_v2'] = updated_df['methylation_score']

# Add driver classification
driver_map = dict(zip(driver_df['gene'], driver_df['classification']))
updated_df['methylation_class'] = updated_df['gene'].map(driver_map)

# Re-rank
updated_df = updated_df.sort_values('transformation_score_v2', ascending=False)
updated_df['rank_v2'] = range(1, len(updated_df) + 1)

# Save
updated_df.to_csv(f"{RES_DIR}/unified_discovery_with_methylation.csv", index=False)
print(f"  TransformationScore v2 computed for {len(updated_df)} genes")
print(f"\n  Top 15 genes (v2 ranking):")
for _, r in updated_df.head(15).iterrows():
    orig_rank = r.get('rank', '?')
    print(f"    Rank {r['rank_v2']:2d} (was {orig_rank}): {r['gene']:8s} "
          f"score_v2={r['transformation_score_v2']:.4f} "
          f"methyl={r['methylation_score']:.3f} "
          f"class={r.get('methylation_class', '?')}")


# ============================================================
# Visualization
# ============================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# A: Cross-dataset beta comparison
ax = axes[0, 0]
plot_data = cross_df.dropna(subset=['tcga_mean_beta', 'gse103186_mean_beta'])
if len(plot_data) > 5:
    ax.scatter(plot_data['tcga_mean_beta'], plot_data['gse103186_mean_beta'], alpha=0.7, s=40)
    r, p = pearsonr(plot_data['tcga_mean_beta'], plot_data['gse103186_mean_beta'])
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax.set_xlabel('TCGA-STAD (tumor) mean β')
    ax.set_ylabel('GSE103186 (IM) mean β')
    ax.set_title(f'Cross-dataset consistency\n(r={r:.3f}, p={p:.2e})')

# B: Temporal methylation dynamics
ax = axes[0, 1]
if len(temporal_methyl_df) > 0:
    colors = ['green' if d < -0.05 else 'red' if d > 0.05 else 'gray'
              for d in temporal_methyl_df['delta_beta_gastritis_to_IM']]
    ax.scatter(temporal_methyl_df['mean_beta_gastritis'],
               temporal_methyl_df['mean_beta_IM'],
               c=colors, alpha=0.7, s=40)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Mean β in Gastritis')
    ax.set_ylabel('Mean β in IM')
    ax.set_title('Methylation: Gastritis → IM transition')
    # Label top changed genes
    for _, r in temporal_methyl_df.head(3).iterrows():
        ax.annotate(r['gene'], (r['mean_beta_gastritis'], r['mean_beta_IM']), fontsize=7)
    for _, r in temporal_methyl_df.tail(3).iterrows():
        ax.annotate(r['gene'], (r['mean_beta_gastritis'], r['mean_beta_IM']), fontsize=7)

# C: Driver classification pie
ax = axes[0, 2]
class_counts = driver_df['classification'].value_counts()
colors_map = {
    'methylation_driven_activation': '#2ca02c',
    'methylation_driven_silencing': '#d62728',
    'TF_driven': '#1f77b4',
    'other_regulation': '#ff7f0e',
    'demethylated_but_weak_correlation': '#9467bd',
    'ambiguous': '#7f7f7f'
}
ax.pie(class_counts.values, labels=[c.replace('_', '\n') for c in class_counts.index],
       autopct='%1.0f%%', colors=[colors_map.get(c, 'gray') for c in class_counts.index],
       textprops={'fontsize': 7})
ax.set_title('Gene Regulation Classification\n(92 candidates)')

# D: Methylation score distribution
ax = axes[1, 0]
scores = [methyl_scores[g] for g in CANDIDATES]
ax.hist(scores, bins=15, color='steelblue', edgecolor='white')
ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'mean={np.mean(scores):.3f}')
ax.set_xlabel('Methylation Evidence Score')
ax.set_ylabel('Count')
ax.set_title('Distribution of Methylation Scores')
ax.legend()

# E: Score v1 vs v2 comparison
ax = axes[1, 1]
if 'TransformationScore' in updated_df.columns:
    ax.scatter(updated_df['TransformationScore'], updated_df['transformation_score_v2'],
               alpha=0.6, s=40)
    ax.plot([0, 0.5], [0, 0.5], 'r--', alpha=0.5)
    ax.set_xlabel('TransformationScore v1')
    ax.set_ylabel('TransformationScore v2 (+ methylation)')
    ax.set_title('Score Update: v1 vs v2')
    # Label top movers (compute original rank from v1 score)
    updated_df['rank_v1'] = updated_df['TransformationScore'].rank(ascending=False)
    updated_df['rank_change'] = updated_df['rank_v1'] - updated_df['rank_v2']
    top_movers = updated_df.nlargest(3, 'rank_change')
    for _, r in top_movers.iterrows():
        ax.annotate(r['gene'], (r['TransformationScore'], r['transformation_score_v2']), fontsize=8)

# F: Methylation vs expression correlation heatmap (top genes)
ax = axes[1, 2]
top_genes = updated_df.head(20)['gene'].tolist()
heatmap_data = []
for gene in top_genes:
    row_data = {'gene': gene}
    tcga_r = tcga_corr[tcga_corr['gene'] == gene]
    row_data['TCGA rho'] = tcga_r.iloc[0]['spearman_rho'] if len(tcga_r) > 0 else 0
    row_data['Methylation score'] = methyl_scores.get(gene, 0)
    if len(temporal_methyl_df) > 0:
        t = temporal_methyl_df[temporal_methyl_df['gene'] == gene]
        row_data['Δβ (G→IM)'] = t.iloc[0]['delta_beta_gastritis_to_IM'] if len(t) > 0 else 0
    else:
        row_data['Δβ (G→IM)'] = 0
    heatmap_data.append(row_data)

hm_df = pd.DataFrame(heatmap_data).set_index('gene')
sns.heatmap(hm_df, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            xticklabels=True, yticklabels=True, cbar_kws={'shrink': 0.8})
ax.set_title('Top 20 genes: methylation features')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/methylation_integration.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/methylation_integration.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("INTEGRATION SUMMARY")
print("="*70)

print(f"\n  Datasets integrated: TCGA-STAD + GSE103186 + GSE220511 + GSE178925")
print(f"  Genes with methylation data: {cross_df['n_datasets'].ge(1).sum()}/92")
print(f"  Genes with 3+ dataset methylation: {cross_df['n_datasets'].ge(3).sum()}")

if len(temporal_methyl_df) > 0:
    print(f"\n  Temporal methylation changes (Gastritis → IM):")
    print(f"    Demethylated (Δβ < -0.05): {(temporal_methyl_df['delta_beta_gastritis_to_IM'] < -0.05).sum()}")
    print(f"    Hypermethylated (Δβ > 0.05): {(temporal_methyl_df['delta_beta_gastritis_to_IM'] > 0.05).sum()}")
    print(f"    Stable: {(abs(temporal_methyl_df['delta_beta_gastritis_to_IM']) <= 0.05).sum()}")

print(f"\n  Driver classification:")
for cls, count in driver_df['classification'].value_counts().items():
    print(f"    {cls}: {count}")

print(f"\n  TransformationScore v2 top 10:")
for _, r in updated_df.head(10).iterrows():
    print(f"    {r['gene']:8s}: v2={r['transformation_score_v2']:.4f}, methyl_score={r['methylation_score']:.3f}")

print("\nDone! All methylation integration complete.")
