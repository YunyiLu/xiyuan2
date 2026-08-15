"""
Step 15: Corrected GSE78523 Audit
Address critical reviewer concerns:
1. Remove 15 healthy controls from comparison — true IM-internal analysis only
2. Three-tier analysis: All IM (14v16), IIM (6v7), CIM (8v9)
3. LOOCV AUC with proper CI
4. BH-FDR correction on all p-values
5. Test OLFM4/GKN1 ratio and other gain/loss combinations
6. Separate "IM detection" analysis (IM vs Healthy) from "progression prediction"

Output:
  results/corrected_gse78523_im_only.csv         — single gene results (14v16)
  results/corrected_gse78523_subtype.csv         — IIM and CIM subgroup results
  results/corrected_gse78523_loocv.csv           — LOOCV AUC for key models
  results/corrected_gse78523_ratio.csv           — Ratio biomarker performance
  results/corrected_gse78523_im_detection.csv    — IM vs Healthy (separate question)
  figures/corrected_gse78523_audit.png
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
GSE78523_EXPR = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_expression.csv"
GSE78523_META = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_metadata.csv"

GAIN_GENES = ['OLFM4', 'REG4', 'ITLN1', 'PRAP1', 'ANPEP', 'FABP1', 'CLDN4', 'CPS1', 'MUC13']
LOSS_GENES = ['GKN1', 'GKN2', 'PGC', 'GIF', 'SST', 'TFF2', 'TCN1']
ALL_CANDIDATES = GAIN_GENES + LOSS_GENES


def load_data():
    expr = pd.read_csv(GSE78523_EXPR, index_col=0)
    meta = pd.read_csv(GSE78523_META)
    common = [s for s in meta['sample_id'] if s in expr.columns]
    meta = meta[meta['sample_id'].isin(common)].copy()
    return expr, meta


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)


def loocv_auc(X, y):
    """Leave-one-out cross-validated AUC with logistic regression."""
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        clf.fit(X_train_s, y_train)
        y_pred[test_idx] = clf.predict_proba(X_test_s)[:, 1]
    try:
        auc = roc_auc_score(y, y_pred)
    except ValueError:
        auc = np.nan
    return auc, y_pred


def bootstrap_auc_ci(y_true, y_scores, n_boot=2000, alpha=0.05):
    """Bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(42)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    aucs = np.array(aucs)
    ci_low = np.percentile(aucs, 100 * alpha / 2)
    ci_high = np.percentile(aucs, 100 * (1 - alpha / 2))
    return ci_low, ci_high


def single_gene_analysis(expr, meta, prog_ids, ctrl_ids, genes, label=""):
    """Mann-Whitney U test for each gene, with BH-FDR correction."""
    results = []
    for gene in genes:
        if gene not in expr.index:
            continue
        g_prog = expr.loc[gene, prog_ids].values.astype(float)
        g_ctrl = expr.loc[gene, ctrl_ids].values.astype(float)
        _, p = mannwhitneyu(g_prog, g_ctrl, alternative='two-sided')
        d = compute_cohens_d(g_prog, g_ctrl)
        log2fc = np.log2((np.mean(g_prog) + 0.01) / (np.mean(g_ctrl) + 0.01))
        results.append({
            'gene': gene,
            'prog_mean': np.mean(g_prog),
            'ctrl_mean': np.mean(g_ctrl),
            'log2FC': log2fc,
            'cohens_d': d,
            'p_raw': p,
            'n_prog': len(g_prog),
            'n_ctrl': len(g_ctrl),
            'analysis': label,
        })
    df = pd.DataFrame(results)
    if len(df) > 0:
        _, fdr, _, _ = multipletests(df['p_raw'], method='fdr_bh')
        df['p_fdr'] = fdr
        df = df.sort_values('p_raw')
    return df


def model_loocv_analysis(expr, meta, prog_ids, ctrl_ids):
    """LOOCV AUC for various gene combinations."""
    all_ids = list(prog_ids) + list(ctrl_ids)
    y = np.array([1]*len(prog_ids) + [0]*len(ctrl_ids))

    models = {
        'M1_OLFM4': ['OLFM4'],
        'M2_REG4': ['REG4'],
        'M3_OLFM4_REG4': ['OLFM4', 'REG4'],
        'M4_OLFM4_REG4_ITLN1': ['OLFM4', 'REG4', 'ITLN1'],
        'M5_4gene': ['OLFM4', 'REG4', 'ITLN1', 'PRAP1'],
        'M6_GKN1_loss': ['GKN1'],
        'M7_PGC_loss': ['PGC'],
        'M8_OLFM4_GKN1': ['OLFM4', 'GKN1'],
        'M9_gain3_loss2': ['OLFM4', 'REG4', 'ITLN1', 'GKN1', 'PGC'],
    }

    results = []
    for model_name, genes in models.items():
        available = [g for g in genes if g in expr.index]
        if len(available) == 0:
            continue
        X = expr.loc[available, all_ids].values.T
        auc, y_pred = loocv_auc(X, y)
        ci_low, ci_high = bootstrap_auc_ci(y, y_pred)

        fpr, tpr, thresholds = roc_curve(y, y_pred)
        sens_at_spec90 = tpr[np.searchsorted(1-fpr, 0.9, side='right')-1] if len(fpr) > 1 else 0
        spec_at_sens90 = 1 - fpr[np.searchsorted(tpr, 0.9)-1] if any(tpr >= 0.9) else 0

        results.append({
            'model': model_name,
            'genes': ';'.join(available),
            'n_genes': len(available),
            'AUC': round(auc, 4),
            'CI_low': round(ci_low, 4),
            'CI_high': round(ci_high, 4),
            'sens_at_spec90': round(sens_at_spec90, 4),
            'spec_at_sens90': round(spec_at_sens90, 4),
            'n_prog': len(prog_ids),
            'n_ctrl': len(ctrl_ids),
        })

    return pd.DataFrame(results)


def ratio_analysis(expr, meta, prog_ids, ctrl_ids):
    """Test ratio biomarkers: gain/loss combinations."""
    all_ids = list(prog_ids) + list(ctrl_ids)
    y = np.array([1]*len(prog_ids) + [0]*len(ctrl_ids))

    ratios = {
        'OLFM4/GKN1': ('OLFM4', 'GKN1'),
        'OLFM4/PGC': ('OLFM4', 'PGC'),
        'REG4/GKN1': ('REG4', 'GKN1'),
        'REG4/PGC': ('REG4', 'PGC'),
        'OLFM4/GKN2': ('OLFM4', 'GKN2'),
        'ITLN1/GKN1': ('ITLN1', 'GKN1'),
        '(OLFM4+REG4)/(GKN1+PGC)': None,
    }

    results = []
    for name, pair in ratios.items():
        if pair is not None:
            num_gene, den_gene = pair
            if num_gene not in expr.index or den_gene not in expr.index:
                continue
            num_vals = expr.loc[num_gene, all_ids].values.astype(float)
            den_vals = expr.loc[den_gene, all_ids].values.astype(float)
            ratio_vals = np.log2((num_vals + 0.1) / (den_vals + 0.1))
        else:
            # Composite ratio
            genes_needed = ['OLFM4', 'REG4', 'GKN1', 'PGC']
            if not all(g in expr.index for g in genes_needed):
                continue
            num_vals = (expr.loc['OLFM4', all_ids].values.astype(float) +
                       expr.loc['REG4', all_ids].values.astype(float))
            den_vals = (expr.loc['GKN1', all_ids].values.astype(float) +
                       expr.loc['PGC', all_ids].values.astype(float))
            ratio_vals = np.log2((num_vals + 0.1) / (den_vals + 0.1))

        # Single-feature LOOCV
        X = ratio_vals.reshape(-1, 1)
        auc, y_pred = loocv_auc(X, y)
        ci_low, ci_high = bootstrap_auc_ci(y, y_pred)

        # Direct Mann-Whitney on ratio
        r_prog = ratio_vals[:len(prog_ids)]
        r_ctrl = ratio_vals[len(prog_ids):]
        _, p = mannwhitneyu(r_prog, r_ctrl, alternative='two-sided')
        d = compute_cohens_d(r_prog, r_ctrl)

        results.append({
            'ratio': name,
            'AUC_loocv': round(auc, 4),
            'CI_low': round(ci_low, 4),
            'CI_high': round(ci_high, 4),
            'cohens_d': round(d, 3),
            'p_mannwhitney': p,
            'prog_mean': round(np.mean(r_prog), 3),
            'ctrl_mean': round(np.mean(r_ctrl), 3),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("Step 15: CORRECTED GSE78523 Audit")
    print("  Addressing reviewer critique: exclude healthy controls")
    print("=" * 70)

    expr, meta = load_data()
    print(f"\n  Total samples: {len(meta)}")
    print(f"  Groups: {meta['group'].value_counts().to_dict()}")

    # Define correct groups
    im_prog = meta[meta['group'].str.contains('progressor')]['sample_id'].tolist()
    im_ctrl_only = meta[meta['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()
    healthy = meta[meta['group'] == 'Healthy']['sample_id'].tolist()

    iim_prog = meta[meta['group'] == 'IIM_GC_progressor']['sample_id'].tolist()
    iim_ctrl = meta[meta['group'] == 'IIM_ctrl']['sample_id'].tolist()
    cim_prog = meta[meta['group'] == 'CIM_GC_progressor']['sample_id'].tolist()
    cim_ctrl = meta[meta['group'] == 'CIM_ctrl']['sample_id'].tolist()

    print(f"\n  === CORRECTED GROUPING (Healthy EXCLUDED) ===")
    print(f"  IM Progressors: {len(im_prog)} (6 IIM + 8 CIM)")
    print(f"  IM Non-progressors: {len(im_ctrl_only)} (7 IIM + 9 CIM)")
    print(f"  Healthy (excluded from main analysis): {len(healthy)}")

    # =========================================================
    # ANALYSIS 1: Single gene — IM internal only (14 vs 16)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 1: Single Gene (14 progressor vs 16 non-progressor)")
    print(f"{'='*70}")

    df_im = single_gene_analysis(expr, meta, im_prog, im_ctrl_only, ALL_CANDIDATES, "IM_only_14v16")
    n_sig = (df_im['p_fdr'] < 0.05).sum()
    n_nom = (df_im['p_raw'] < 0.05).sum()
    print(f"\n  Nominally significant (p<0.05): {n_nom}/{len(df_im)}")
    print(f"  FDR significant (q<0.05): {n_sig}/{len(df_im)}")
    print(f"\n  {'Gene':<8} {'d':>7} {'log2FC':>7} {'p_raw':>10} {'p_fdr':>10}")
    print(f"  {'-'*50}")
    for _, r in df_im.iterrows():
        sig = '**' if r['p_fdr'] < 0.05 else '*' if r['p_raw'] < 0.05 else ''
        print(f"  {r['gene']:<8} {r['cohens_d']:>7.3f} {r['log2FC']:>7.3f} {r['p_raw']:>10.4e} {r['p_fdr']:>10.4e} {sig}")

    df_im.to_csv(f"{BASE}/results/corrected_gse78523_im_only.csv", index=False, encoding='utf-8-sig')

    # =========================================================
    # ANALYSIS 2: Subtype-specific (IIM 6v7, CIM 8v9)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 2: Subtype-specific")
    print(f"{'='*70}")

    df_iim = single_gene_analysis(expr, meta, iim_prog, iim_ctrl, ALL_CANDIDATES, "IIM_6v7")
    df_cim = single_gene_analysis(expr, meta, cim_prog, cim_ctrl, ALL_CANDIDATES, "CIM_8v9")

    print(f"\n  IIM (6 prog vs 7 ctrl):")
    print(f"  {'Gene':<8} {'d':>7} {'p_raw':>10}")
    for _, r in df_iim.head(8).iterrows():
        sig = '*' if r['p_raw'] < 0.05 else ''
        print(f"  {r['gene']:<8} {r['cohens_d']:>7.3f} {r['p_raw']:>10.4e} {sig}")

    print(f"\n  CIM (8 prog vs 9 ctrl):")
    print(f"  {'Gene':<8} {'d':>7} {'p_raw':>10}")
    for _, r in df_cim.head(8).iterrows():
        sig = '*' if r['p_raw'] < 0.05 else ''
        print(f"  {r['gene']:<8} {r['cohens_d']:>7.3f} {r['p_raw']:>10.4e} {sig}")

    df_subtype = pd.concat([df_iim, df_cim], ignore_index=True)
    df_subtype.to_csv(f"{BASE}/results/corrected_gse78523_subtype.csv", index=False, encoding='utf-8-sig')

    # =========================================================
    # ANALYSIS 3: LOOCV AUC — IM internal only (14 vs 16)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 3: LOOCV AUC (14 prog vs 16 ctrl, NO healthy)")
    print(f"{'='*70}")

    df_loocv = model_loocv_analysis(expr, meta, im_prog, im_ctrl_only)
    print(f"\n  {'Model':<25} {'AUC':>6} {'95%CI':>15} {'n':>5}")
    print(f"  {'-'*55}")
    for _, r in df_loocv.sort_values('AUC', ascending=False).iterrows():
        print(f"  {r['model']:<25} {r['AUC']:>6.3f} [{r['CI_low']:.3f}-{r['CI_high']:.3f}]  {r['n_prog']}v{r['n_ctrl']}")

    df_loocv.to_csv(f"{BASE}/results/corrected_gse78523_loocv.csv", index=False, encoding='utf-8-sig')

    # =========================================================
    # ANALYSIS 4: Ratio biomarkers (14 vs 16)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 4: Ratio Biomarkers (14 prog vs 16 ctrl)")
    print(f"{'='*70}")

    df_ratio = ratio_analysis(expr, meta, im_prog, im_ctrl_only)
    print(f"\n  {'Ratio':<25} {'AUC':>6} {'95%CI':>15} {'d':>7} {'p':>10}")
    print(f"  {'-'*70}")
    for _, r in df_ratio.sort_values('AUC_loocv', ascending=False).iterrows():
        sig = '*' if r['p_mannwhitney'] < 0.05 else ''
        print(f"  {r['ratio']:<25} {r['AUC_loocv']:>6.3f} [{r['CI_low']:.3f}-{r['CI_high']:.3f}] {r['cohens_d']:>7.3f} {r['p_mannwhitney']:>10.4e} {sig}")

    df_ratio.to_csv(f"{BASE}/results/corrected_gse78523_ratio.csv", index=False, encoding='utf-8-sig')

    # =========================================================
    # ANALYSIS 5: IM detection (separate question: all IM vs Healthy)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 5: IM Detection (30 IM vs 15 Healthy) — SEPARATE QUESTION")
    print(f"{'='*70}")

    all_im = im_prog + im_ctrl_only
    df_detect = single_gene_analysis(expr, meta, all_im, healthy, ALL_CANDIDATES, "IM_detection_30v15")
    n_sig_det = (df_detect['p_fdr'] < 0.05).sum()
    print(f"\n  This answers: 'Can we detect IM presence?' NOT 'Who will progress?'")
    print(f"  FDR significant: {n_sig_det}/{len(df_detect)}")
    print(f"\n  {'Gene':<8} {'d':>7} {'p_fdr':>10} (IM vs Healthy)")
    for _, r in df_detect.head(10).iterrows():
        direction = 'UP' if r['cohens_d'] > 0 else 'DOWN'
        print(f"  {r['gene']:<8} {r['cohens_d']:>7.3f} {r['p_fdr']:>10.4e} {direction} in IM")

    df_detect.to_csv(f"{BASE}/results/corrected_gse78523_im_detection.csv", index=False, encoding='utf-8-sig')

    # =========================================================
    # ANALYSIS 6: Compare OLD (14v31) vs CORRECTED (14v16)
    # =========================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 6: Impact of Correction (Old vs New)")
    print(f"{'='*70}")

    # Old analysis: 14 vs 31 (including healthy)
    old_ctrl = im_ctrl_only + healthy
    df_old_loocv = model_loocv_analysis(expr, meta, im_prog, old_ctrl)

    print(f"\n  {'Model':<25} {'OLD(14v31)':>10} {'NEW(14v16)':>10} {'delta':>8}")
    print(f"  {'-'*58}")
    for _, r_new in df_loocv.iterrows():
        r_old = df_old_loocv[df_old_loocv['model'] == r_new['model']]
        if len(r_old) > 0:
            old_auc = r_old.iloc[0]['AUC']
            delta = r_new['AUC'] - old_auc
            print(f"  {r_new['model']:<25} {old_auc:>10.3f} {r_new['AUC']:>10.3f} {delta:>+8.3f}")

    # =========================================================
    # Visualization
    # =========================================================
    print(f"\n[Generating figures...]")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Corrected single gene volcano
    ax = axes[0, 0]
    ax.scatter(df_im['cohens_d'], -np.log10(df_im['p_raw']),
               c=['red' if p < 0.05 else 'gray' for p in df_im['p_fdr']], alpha=0.7, s=50)
    for _, r in df_im[df_im['p_raw'] < 0.1].iterrows():
        ax.annotate(r['gene'], (r['cohens_d'], -np.log10(r['p_raw'])),
                    fontsize=7, ha='center', va='bottom')
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.set_xlabel("Cohen's d (prog - ctrl)")
    ax.set_ylabel('-log10(p)')
    ax.set_title('A. Single Gene (14 prog vs 16 ctrl)\nRed = FDR < 0.05')
    ax.legend(fontsize=8)

    # Panel B: LOOCV AUC comparison
    ax = axes[0, 1]
    models_sorted = df_loocv.sort_values('AUC', ascending=True)
    y_pos = range(len(models_sorted))
    ax.barh(y_pos, models_sorted['AUC'], color='steelblue', alpha=0.7, height=0.6)
    ax.errorbar(models_sorted['AUC'], y_pos,
                xerr=[models_sorted['AUC']-models_sorted['CI_low'],
                      models_sorted['CI_high']-models_sorted['AUC']],
                fmt='none', color='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted['model'], fontsize=8)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LOOCV AUC')
    ax.set_title('B. Model Comparison (14v16, no healthy)')
    ax.set_xlim(0.3, 1.0)

    # Panel C: Ratio biomarkers
    ax = axes[0, 2]
    df_ratio_sorted = df_ratio.sort_values('AUC_loocv', ascending=True)
    y_pos = range(len(df_ratio_sorted))
    colors = ['green' if p < 0.05 else 'gray' for p in df_ratio_sorted['p_mannwhitney']]
    ax.barh(y_pos, df_ratio_sorted['AUC_loocv'], color=colors, alpha=0.7, height=0.6)
    ax.errorbar(df_ratio_sorted['AUC_loocv'], y_pos,
                xerr=[df_ratio_sorted['AUC_loocv']-df_ratio_sorted['CI_low'],
                      df_ratio_sorted['CI_high']-df_ratio_sorted['AUC_loocv']],
                fmt='none', color='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_ratio_sorted['ratio'], fontsize=8)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LOOCV AUC')
    ax.set_title('C. Ratio Biomarkers (14v16)\nGreen = p<0.05')
    ax.set_xlim(0.3, 1.0)

    # Panel D: IIM subtype
    ax = axes[1, 0]
    df_iim_top = df_iim.head(10)
    colors = ['coral' if p < 0.05 else 'lightgray' for p in df_iim_top['p_raw']]
    ax.barh(range(len(df_iim_top)), df_iim_top['cohens_d'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(df_iim_top)))
    ax.set_yticklabels(df_iim_top['gene'], fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title('D. IIM subtype (6 prog vs 7 ctrl)')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    # Panel E: CIM subtype
    ax = axes[1, 1]
    df_cim_top = df_cim.head(10)
    colors = ['coral' if p < 0.05 else 'lightgray' for p in df_cim_top['p_raw']]
    ax.barh(range(len(df_cim_top)), df_cim_top['cohens_d'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(df_cim_top)))
    ax.set_yticklabels(df_cim_top['gene'], fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title('E. CIM subtype (8 prog vs 9 ctrl)')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    # Panel F: Old vs New AUC comparison
    ax = axes[1, 2]
    merged = df_loocv.merge(df_old_loocv[['model', 'AUC']], on='model', suffixes=('_new', '_old'))
    ax.scatter(merged['AUC_old'], merged['AUC_new'], s=80, c='navy', alpha=0.7)
    for _, r in merged.iterrows():
        ax.annotate(r['model'].replace('M', '').split('_')[0], (r['AUC_old'], r['AUC_new']),
                    fontsize=7, ha='left')
    ax.plot([0.4, 1], [0.4, 1], 'r--', alpha=0.5)
    ax.set_xlabel('Old AUC (14 vs 31, with healthy)')
    ax.set_ylabel('New AUC (14 vs 16, IM only)')
    ax.set_title('F. Correction Impact\nBelow line = inflated by healthy')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)

    plt.suptitle('CORRECTED GSE78523 Audit: IM-Internal Analysis Only\n'
                 '(14 progressors vs 16 non-progressors, healthy controls excluded)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/corrected_gse78523_audit.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/corrected_gse78523_audit.png")

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*70}")
    print("SUMMARY: CORRECTED RESULTS")
    print(f"{'='*70}")

    best_single = df_im.iloc[0]
    best_model = df_loocv.sort_values('AUC', ascending=False).iloc[0]
    best_ratio = df_ratio.sort_values('AUC_loocv', ascending=False).iloc[0]

    print(f"\n  Best single gene: {best_single['gene']} (d={best_single['cohens_d']:.3f}, p_fdr={best_single['p_fdr']:.4e})")
    print(f"  Best model AUC:   {best_model['model']} = {best_model['AUC']:.3f} [{best_model['CI_low']:.3f}-{best_model['CI_high']:.3f}]")
    print(f"  Best ratio AUC:   {best_ratio['ratio']} = {best_ratio['AUC_loocv']:.3f} [{best_ratio['CI_low']:.3f}-{best_ratio['CI_high']:.3f}]")
    print(f"\n  Key comparison:")
    olfm4_row = df_loocv[df_loocv['model'] == 'M1_OLFM4']
    if len(olfm4_row) > 0:
        print(f"    OLFM4 alone:        AUC = {olfm4_row.iloc[0]['AUC']:.3f}")
    ratio_olfm4_gkn1 = df_ratio[df_ratio['ratio'] == 'OLFM4/GKN1']
    if len(ratio_olfm4_gkn1) > 0:
        print(f"    OLFM4/GKN1 ratio:   AUC = {ratio_olfm4_gkn1.iloc[0]['AUC_loocv']:.3f}")
    m8 = df_loocv[df_loocv['model'] == 'M8_OLFM4_GKN1']
    if len(m8) > 0:
        print(f"    OLFM4+GKN1 (2-var): AUC = {m8.iloc[0]['AUC']:.3f}")

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
