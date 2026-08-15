"""
Phase 25h: PMC Stage Validation
Use PMC_2 and PMC_P gene signatures (from Gao et al. 2025 / OMIX010346)
to annotate IM metacells and check correspondence with fate clusters.

Key question: Do fate clusters correspond to the PMC progression continuum?
  - IM -> PMC_2 -> PMC_P -> EGC
  - If high-EGC-fate cluster = high PMC_P score, this independently validates
    the fate clustering using the original paper's staging system.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25h: PMC Stage Validation of Fate Clusters")
print("  Gao et al. 2025 defined: N < IM < PMC_2 < PMC_P < Tumor")
print("  Question: Do our fate clusters map onto this continuum?")
print("=" * 70)

# Load data
print("\n[1/5] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
val_df = pd.read_csv(RESULTS / "rl_value_function.csv")

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values

X = adata_mc.X
if sparse.issparse(X):
    X = X.toarray()
gene_names = list(adata_mc.var_names)
n_cells = len(adata_mc)
im_idx = np.where(stages == 'IM')[0]
print(f"  {n_cells} metacells, {len(im_idx)} IM cells")

# ===================================================================
# [2/5] Compute PMC scores for all cells
# ===================================================================
print("\n[2/5] Computing PMC signatures ...")

# From Gao et al. 2025 (OMIX010346 original paper)
PMC_SIGNATURES = {
    'PMC_2': ['NAMPT', 'ALDH1A1', 'CD44', 'SOX9', 'OLFM4'],
    'PMC_P': ['AREG', 'NAMPT', 'PHLDA1', 'ITGA2', 'MYC'],
    'EGC_like': ['REG4', 'CEACAM6', 'MUC13', 'CLDN3', 'EPCAM',
                 'KRT20', 'ERBB2', 'MET', 'VEGFA'],
    'Differentiation': ['GKN1', 'PGC', 'TFF1', 'MUC5AC', 'GIF'],
    'Stemness': ['LGR5', 'OLFM4', 'SOX9', 'ASCL2', 'CD44'],
}


def score_signature(genes):
    """Mean z-scored expression of gene set."""
    valid = [g for g in genes if g in gene_names]
    if not valid:
        return np.zeros(n_cells)
    idx = [gene_names.index(g) for g in valid]
    vals = X[:, idx].mean(axis=1)
    std = vals.std()
    if std > 1e-10:
        return (vals - vals.mean()) / std
    return np.zeros(n_cells)


scores = {}
for name, genes in PMC_SIGNATURES.items():
    scores[name] = score_signature(genes)
    found = [g for g in genes if g in gene_names]
    print(f"  {name}: {len(found)}/{len(genes)} genes found")

# ===================================================================
# [3/5] Reconstruct fate clusters (k=4 from 24c)
# ===================================================================
print("\n[3/5] Reconstructing IM fate clusters ...")

# Forward propagation k=20 steps
im_start = np.zeros((len(im_idx), T_cr.shape[0]))
for i, idx in enumerate(im_idx):
    im_start[i, idx] = 1.0

prop = im_start.copy()
for _ in range(20):
    prop = prop @ T_cr

# Fate fractions
egc_mask = (stages == 'EGC') | (stages == 'EGC_multi_region')
gc_mask = stages == 'GC'
im_mask_all = stages == 'IM'
nag_mask = stages == 'NAG'

im_fate_egc = (prop * egc_mask).sum(axis=1)
im_fate_gc = (prop * gc_mask).sum(axis=1)
im_fate_im = (prop * im_mask_all).sum(axis=1)
im_fate_nag = (prop * nag_mask).sum(axis=1)

fate_matrix = np.column_stack([im_fate_egc, im_fate_gc, im_fate_im, im_fate_nag])

# KMeans k=4 (same as 24c)
km = KMeans(n_clusters=4, n_init=10, random_state=42)
im_fate_labels = km.fit_predict(fate_matrix)

print(f"  Fate clusters: {np.bincount(im_fate_labels)}")
for c in range(4):
    mask = im_fate_labels == c
    print(f"    Cluster {c}: n={mask.sum()}, "
          f"P(EGC)={im_fate_egc[mask].mean():.3f}, "
          f"P(GC)={im_fate_gc[mask].mean():.3f}")

# ===================================================================
# [4/5] Cross-tabulate PMC scores vs fate clusters
# ===================================================================
print("\n[4/5] PMC scores vs Fate clusters ...")

# Build IM-level dataframe
im_df = pd.DataFrame({
    'fate_cluster': im_fate_labels,
    'P_EGC': im_fate_egc,
    'P_GC': im_fate_gc,
    'pseudotime': pt[im_idx],
    'V_value': V[im_idx],
})
for name, score in scores.items():
    im_df[name] = score[im_idx]

# Sort clusters by P(EGC) for interpretability
cluster_egc_order = im_df.groupby('fate_cluster')['P_EGC'].mean().sort_values()
cluster_rank = {c: r for r, c in enumerate(cluster_egc_order.index)}
im_df['fate_rank'] = im_df['fate_cluster'].map(cluster_rank)

# Correlation: PMC scores vs P(EGC)
print("\n  Spearman correlations with P(EGC):")
print(f"  {'Score':>15} {'rho':>7} {'p-value':>10}")
print(f"  {'-'*35}")
corr_results = []
for name in PMC_SIGNATURES.keys():
    rho, p = spearmanr(im_df['P_EGC'], im_df[name])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {name:>15} {rho:>+7.3f} {p:>10.2e} {sig}")
    corr_results.append({'score': name, 'rho_vs_P_EGC': rho, 'p_value': p})

# Kruskal-Wallis: PMC scores across fate clusters
print("\n  Kruskal-Wallis: PMC scores across 4 fate clusters:")
print(f"  {'Score':>15} {'H-stat':>8} {'p-value':>10} "
      f"{'Cluster0':>9} {'Cluster1':>9} {'Cluster2':>9} {'Cluster3':>9}")
print(f"  {'-'*75}")

kw_results = []
for name in PMC_SIGNATURES.keys():
    groups = [im_df[im_df['fate_cluster'] == c][name].values for c in range(4)]
    h, p = kruskal(*groups)
    means = [g.mean() for g in groups]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {name:>15} {h:>8.2f} {p:>10.2e} "
          f"{means[0]:>+9.3f} {means[1]:>+9.3f} "
          f"{means[2]:>+9.3f} {means[3]:>+9.3f} {sig}")
    kw_results.append({
        'score': name, 'H_stat': h, 'p_value': p,
        **{f'mean_cluster_{c}': means[c] for c in range(4)}
    })

# Key test: highest-EGC-fate cluster vs lowest
high_egc_cluster = cluster_egc_order.index[-1]
low_egc_cluster = cluster_egc_order.index[0]
print(f"\n  Mann-Whitney: Highest-EGC cluster ({high_egc_cluster}) vs "
      f"Lowest-EGC cluster ({low_egc_cluster}):")

mw_results = []
for name in PMC_SIGNATURES.keys():
    high = im_df[im_df['fate_cluster'] == high_egc_cluster][name].values
    low = im_df[im_df['fate_cluster'] == low_egc_cluster][name].values
    u, p = mannwhitneyu(high, low, alternative='two-sided')
    diff = high.mean() - low.mean()
    direction = "HIGH>LOW" if diff > 0 else "LOW>HIGH"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {name:>15}: diff={diff:>+.3f} ({direction}) p={p:.4f} {sig}")
    mw_results.append({
        'score': name, 'mean_high_EGC': high.mean(),
        'mean_low_EGC': low.mean(), 'diff': diff, 'p_value': p
    })

# ===================================================================
# [5/5] Visualization
# ===================================================================
print("\n[5/5] Visualization ...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Panel A: PMC_P score vs P(EGC)
ax = axes[0, 0]
sc_plot = ax.scatter(im_df['P_EGC'], im_df['PMC_P'], c=im_df['fate_cluster'],
                     cmap='Set1', s=15, alpha=0.6)
rho_pp, _ = spearmanr(im_df['P_EGC'], im_df['PMC_P'])
ax.set_xlabel('P(EGC) from fate propagation')
ax.set_ylabel('PMC_P score')
ax.set_title(f'PMC_P vs EGC fate (rho={rho_pp:.3f})')

# Panel B: PMC_2 score vs P(EGC)
ax = axes[0, 1]
ax.scatter(im_df['P_EGC'], im_df['PMC_2'], c=im_df['fate_cluster'],
           cmap='Set1', s=15, alpha=0.6)
rho_p2, _ = spearmanr(im_df['P_EGC'], im_df['PMC_2'])
ax.set_xlabel('P(EGC) from fate propagation')
ax.set_ylabel('PMC_2 score')
ax.set_title(f'PMC_2 vs EGC fate (rho={rho_p2:.3f})')

# Panel C: PMC_P by fate cluster (boxplot)
ax = axes[0, 2]
cluster_order = list(cluster_egc_order.index)
data_box = [im_df[im_df['fate_cluster'] == c]['PMC_P'].values
            for c in cluster_order]
bp = ax.boxplot(data_box, labels=[f'C{c}\n(P_EGC={cluster_egc_order[c]:.2f})'
                                   for c in cluster_order], patch_artist=True)
colors_box = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('PMC_P score')
ax.set_title('PMC_P across fate clusters\n(ordered by P(EGC))')

# Panel D: Differentiation score vs P(EGC)
ax = axes[1, 0]
ax.scatter(im_df['P_EGC'], im_df['Differentiation'], c=im_df['fate_cluster'],
           cmap='Set1', s=15, alpha=0.6)
rho_diff, _ = spearmanr(im_df['P_EGC'], im_df['Differentiation'])
ax.set_xlabel('P(EGC)')
ax.set_ylabel('Differentiation score (GKN1/PGC/TFF1)')
ax.set_title(f'Differentiation vs EGC fate (rho={rho_diff:.3f})')

# Panel E: PMC progression model
ax = axes[1, 1]
# Show the continuum: pseudotime vs PMC_P, colored by P(EGC)
sc_pt = ax.scatter(im_df['pseudotime'], im_df['PMC_P'],
                   c=im_df['P_EGC'], cmap='RdYlBu_r', s=15, alpha=0.6)
plt.colorbar(sc_pt, ax=ax, label='P(EGC)')
ax.set_xlabel('Pseudotime')
ax.set_ylabel('PMC_P score')
ax.set_title('PMC_P along pseudotime\n(color=EGC fate)')

# Panel F: Summary bar chart of correlations
ax = axes[1, 2]
names = [r['score'] for r in corr_results]
rhos = [r['rho_vs_P_EGC'] for r in corr_results]
colors_bar = ['green' if r > 0 else 'red' for r in rhos]
ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Spearman rho with P(EGC)')
ax.set_title('PMC signatures vs EGC fate')
ax.axvline(0, color='black', lw=0.5)

plt.suptitle("Phase 25h: PMC Stage Validation of Fate Clusters\n"
             "Gao 2025: N < IM < PMC_2 < PMC_P < Tumor", fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES / "pmc_stage_validation.png", dpi=150, bbox_inches='tight')
plt.close()

# Save results
pd.DataFrame(corr_results).to_csv(
    RESULTS / "pmc_fate_correlations.csv", index=False)
pd.DataFrame(kw_results).to_csv(
    RESULTS / "pmc_fate_kruskal.csv", index=False)
pd.DataFrame(mw_results).to_csv(
    RESULTS / "pmc_fate_mannwhitney.csv", index=False)
im_df.to_csv(RESULTS / "im_pmc_fate_scores.csv", index=False)

print(f"\n  Saved: pmc_stage_validation.png")
print(f"  Saved: pmc_fate_correlations.csv")
print(f"  Saved: im_pmc_fate_scores.csv")

# Summary
print("\n" + "=" * 70)
print("Phase 25h COMPLETE")
print("=" * 70)
print(f"\n  KEY FINDINGS:")
rho_pmc_p = [r for r in corr_results if r['score'] == 'PMC_P'][0]
rho_pmc_2 = [r for r in corr_results if r['score'] == 'PMC_2'][0]
rho_diff = [r for r in corr_results if r['score'] == 'Differentiation'][0]

print(f"    PMC_P vs P(EGC): rho={rho_pmc_p['rho_vs_P_EGC']:+.3f} "
      f"(p={rho_pmc_p['p_value']:.2e})")
print(f"    PMC_2 vs P(EGC): rho={rho_pmc_2['rho_vs_P_EGC']:+.3f} "
      f"(p={rho_pmc_2['p_value']:.2e})")
print(f"    Differentiation vs P(EGC): rho={rho_diff['rho_vs_P_EGC']:+.3f} "
      f"(p={rho_diff['p_value']:.2e})")

if rho_pmc_p['rho_vs_P_EGC'] > 0.1 and rho_pmc_p['p_value'] < 0.05:
    print(f"\n  CONCLUSION: VALIDATED")
    print(f"  Fate clusters correspond to PMC progression continuum.")
    print(f"  High-EGC-fate IM cells = high PMC_P (Gao's pre-tumor stage).")
elif rho_diff['rho_vs_P_EGC'] < -0.1 and rho_diff['p_value'] < 0.05:
    print(f"\n  CONCLUSION: PARTIALLY VALIDATED")
    print(f"  Differentiation loss tracks with EGC fate (expected direction).")
else:
    print(f"\n  CONCLUSION: NO CLEAR CORRESPONDENCE")
    print(f"  Fate clusters may capture biology not in PMC staging.")
print("=" * 70)
