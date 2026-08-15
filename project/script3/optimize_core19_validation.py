"""
Optimization 3+4: Core transformation 19-gene score + GSE78523 validation
Input: evidence_ranked_genes.csv, GSE78523 expression/metadata
Output: results/core19_gse78523_validation.csv, figures/core19_validation.png
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
GSE78523_EXPR = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_expression.csv"
GSE78523_META = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_metadata.csv"


def main():
    print("=" * 60)
    print("Core 19-gene TransformationScore + GSE78523 Validation")
    print("=" * 60)

    # Load top 19 genes by TransformationScore (regardless of marker_class)
    evidence = pd.read_csv(f"{BASE}/results/evidence_ranked_genes.csv")
    core = evidence.head(19).copy()
    print(f"\nTop 19 TransformationScore genes: {len(core)}")
    print(f"  {core['gene'].tolist()}")

    # Load GSE78523
    print("\n[1] Loading GSE78523...")
    expr = pd.read_csv(GSE78523_EXPR, index_col=0)
    meta = pd.read_csv(GSE78523_META)
    print(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} samples")
    print(f"  Groups: {meta['group'].value_counts().to_dict()}")

    # Match samples
    common_samples = [s for s in meta['sample_id'] if s in expr.columns]
    meta = meta[meta['sample_id'].isin(common_samples)].copy()
    print(f"  Matched samples: {len(meta)}")

    # Check gene availability
    available = [g for g in core['gene'] if g in expr.index]
    missing = [g for g in core['gene'] if g not in expr.index]
    print(f"\n[2] Gene availability: {len(available)}/{len(core)} in GSE78523")
    if missing:
        print(f"  Missing: {missing}")

    if len(available) < 3:
        print("  ERROR: Too few genes available")
        return

    # Compute multi-gene score
    print(f"\n[3] Computing 19-gene TransformationScore...")
    X = expr.loc[available, meta['sample_id']].values.T  # samples x genes
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # Weighted by TransformationScore
    weights = np.array([core.set_index('gene').loc[g, 'TransformationScore'] for g in available])
    weights = weights / weights.sum()
    score = X_z @ weights
    meta['transformation_score'] = score

    # [4] Progressor vs Non-progressor comparison
    print(f"\n[4] Progressor vs Non-progressor analysis...")
    progressors = meta[meta['group'].str.contains('progressor')]
    controls = meta[meta['group'].isin(['IIM_ctrl', 'CIM_ctrl', 'Healthy'])]

    print(f"  Progressors: {len(progressors)}")
    print(f"  Controls (IIM_ctrl + CIM_ctrl + Healthy): {len(controls)}")

    if len(progressors) >= 5 and len(controls) >= 5:
        stat, p = mannwhitneyu(progressors['transformation_score'],
                               controls['transformation_score'], alternative='greater')
        d = (progressors['transformation_score'].mean() - controls['transformation_score'].mean()) / \
            np.sqrt((progressors['transformation_score'].var() + controls['transformation_score'].var()) / 2 + 1e-10)
        print(f"  Mann-Whitney U (progressor > control): p={p:.4e}")
        print(f"  Cohen's d: {d:.3f}")
        print(f"  Progressor mean: {progressors['transformation_score'].mean():.3f}")
        print(f"  Control mean: {controls['transformation_score'].mean():.3f}")

    # IIM vs CIM subtype analysis
    print(f"\n[5] IM subtype analysis...")
    iim_prog = meta[meta['group'] == 'IIM_GC_progressor']
    iim_ctrl = meta[meta['group'] == 'IIM_ctrl']
    cim_prog = meta[meta['group'] == 'CIM_GC_progressor']
    cim_ctrl = meta[meta['group'] == 'CIM_ctrl']

    for name, prog, ctrl in [('IIM', iim_prog, iim_ctrl), ('CIM', cim_prog, cim_ctrl)]:
        if len(prog) >= 3 and len(ctrl) >= 3:
            _, p_sub = mannwhitneyu(prog['transformation_score'],
                                    ctrl['transformation_score'], alternative='greater')
            d_sub = (prog['transformation_score'].mean() - ctrl['transformation_score'].mean()) / \
                    np.sqrt((prog['transformation_score'].var() + ctrl['transformation_score'].var()) / 2 + 1e-10)
            print(f"  {name}: progressor vs ctrl p={p_sub:.4f}, d={d_sub:.3f}")

    # [6] Single gene analysis
    print(f"\n[6] Single gene progressor vs control...")
    gene_results = []
    for gene in available:
        g_prog = expr.loc[gene, progressors['sample_id']].values.astype(float)
        g_ctrl = expr.loc[gene, controls['sample_id']].values.astype(float)
        _, gp = mannwhitneyu(g_prog, g_ctrl, alternative='two-sided')
        gd = (g_prog.mean() - g_ctrl.mean()) / (np.sqrt((g_prog.var() + g_ctrl.var()) / 2) + 1e-10)
        gene_results.append({'gene': gene, 'p_value': gp, 'cohens_d': gd,
                             'prog_mean': g_prog.mean(), 'ctrl_mean': g_ctrl.mean()})
    gene_df = pd.DataFrame(gene_results).sort_values('p_value')
    n_sig = (gene_df['p_value'] < 0.05).sum()
    print(f"  Significant (p<0.05): {n_sig}/{len(gene_df)}")
    print(gene_df.head(10).to_string(index=False))

    # [7] Visualization
    print(f"\n[7] Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot by group
    groups_order = ['Healthy', 'IIM_ctrl', 'CIM_ctrl', 'IIM_GC_progressor', 'CIM_GC_progressor']
    groups_present = [g for g in groups_order if g in meta['group'].values]
    data_box = [meta[meta['group'] == g]['transformation_score'].values for g in groups_present]
    bp = axes[0].boxplot(data_box, labels=[g.replace('_', '\n') for g in groups_present], patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#c0392b']
    for patch, color in zip(bp['boxes'], colors[:len(groups_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel('Transformation Score')
    axes[0].set_title(f'19-gene Score by Group\n(Prog vs Ctrl: p={p:.2e}, d={d:.2f})')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Volcano-like: effect size vs -log10(p)
    axes[1].scatter(gene_df['cohens_d'], -np.log10(gene_df['p_value']),
                    c=['red' if p < 0.05 else 'gray' for p in gene_df['p_value']], alpha=0.7)
    for _, row in gene_df[gene_df['p_value'] < 0.05].iterrows():
        axes[1].annotate(row['gene'], (row['cohens_d'], -np.log10(row['p_value'])),
                         fontsize=7, ha='center')
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Cohen's d (progressor - control)")
    axes[1].set_ylabel('-log10(p)')
    axes[1].set_title(f'Single Gene Validation ({n_sig}/{len(gene_df)} sig)')

    plt.tight_layout()
    plt.savefig(f"{BASE}/figures/core19_validation.png", dpi=150)
    plt.close()
    print(f"  Saved: figures/core19_validation.png")

    # [8] Save results
    results = {
        'n_genes_available': len(available),
        'n_genes_total': len(core),
        'progressor_vs_ctrl_p': p,
        'cohens_d': d,
        'n_progressors': len(progressors),
        'n_controls': len(controls),
        'n_single_gene_sig': n_sig,
    }
    pd.DataFrame([results]).to_csv(f"{BASE}/results/core19_gse78523_validation.csv", index=False)
    gene_df.to_csv(f"{BASE}/results/core19_single_gene_gse78523.csv", index=False)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"  19-gene score: Progressor > Control, p={p:.4e}, d={d:.3f}")
    print(f"  Single genes significant: {n_sig}/{len(gene_df)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
