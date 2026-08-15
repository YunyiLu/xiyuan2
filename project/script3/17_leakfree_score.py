"""
Step 17: Leakage-Free TransformationScore + True Independent Validation

Problem: Original TransformationScore uses GSE78523 in two places:
  1. bulk_progression component (0.30 weight within 0.25 total = 7.5% of final score)
  2. IM subtype penalty (CIM_only genes penalized based on GSE78523 subtype effects)

Solution: Recompute TransformationScore WITHOUT any GSE78523 information,
           then validate on GSE78523 (14v16) as a truly independent hold-out.

Output:
  results/leakfree_transformation_score.csv
  results/leakfree_validation_gse78523.csv
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
GSE78523_EXPR = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_expression.csv"
GSE78523_META = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk/GSE78523/GSE78523_metadata.csv"


def normalize_scores(d):
    vals = np.array(list(d.values()))
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-10:
        return {k: 0.5 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)


def loocv_auc(X, y):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        clf.fit(X_train_s, y[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return auc, y_pred


def bootstrap_auc_ci(y_true, y_scores, n_boot=2000):
    rng = np.random.default_rng(42)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def main():
    print("=" * 70)
    print("Step 17: Leakage-Free TransformationScore")
    print("  GSE78523 COMPLETELY removed from scoring")
    print("=" * 70)

    # Load the original evidence table
    evidence = pd.read_csv(f"{BASE}/results/evidence_ranked_genes.csv")
    candidates = evidence['gene'].tolist()
    print(f"\n  Original candidates: {len(candidates)}")

    # Recompute bulk_progression WITHOUT GSE78523
    # Original: 0.40*jt_eff + 0.30*prog_eff + 0.15*kw427_eff + 0.15*kw662_eff
    # New: renormalize without prog_eff: 0.55*jt_eff + 0.225*kw427_eff + 0.225*kw662_eff
    # Actually, simplest approach: just zero out the GSE78523 component and renormalize

    print("\n[1] Recomputing bulk_progression without GSE78523...")

    # We need to re-derive from the stored columns
    # gse55696_jt_z and gse55696_p are available
    # For GSE60427/60662 we don't have stored effects — but we can approximate:
    # Original bulk_progression = 0.40*|jt_z if p<0.1| + 0.30*|78523_eff if p<0.1| + 0.15*427 + 0.15*662
    # Since we don't have 427/662 stored separately, we'll compute:
    # new_bulk = (original_bulk - 0.30*|gse78523_effect if p<0.1|) / 0.70
    # This recovers the non-78523 portion

    new_scores = []
    for _, row in evidence.iterrows():
        gene = row['gene']
        orig_bulk = row['bulk_progression']
        gse78523_eff = row['gse78523_effect']
        gse78523_p = row['gse78523_p']

        # Remove GSE78523 contribution
        prog_contrib = abs(gse78523_eff) * 0.30 if gse78523_p < 0.1 else 0
        remaining = orig_bulk - prog_contrib
        # Renormalize to [0, max] range (will normalize later)
        new_bulk = max(0, remaining / 0.70) if remaining > 0 else 0

        # Recompute TransformationScore
        # Original formula: 0.30*scRNA + 0.30*spatial + 0.25*bulk + 0.15*network
        # Then * direction_penalty, then * IM subtype penalty
        scRNA = row['scRNA_risk']
        spatial = row['spatial_gradient']
        network = row['network_score']
        direction = row['direction_penalty']

        new_scores.append({
            'gene': gene,
            'scRNA_risk': scRNA,
            'spatial_gradient': spatial,
            'bulk_progression_noleak': new_bulk,
            'network_score': network,
            'direction_penalty': direction,
            'original_TransformationScore': row['TransformationScore'],
        })

    df = pd.DataFrame(new_scores)

    # Normalize each component
    for col in ['scRNA_risk', 'spatial_gradient', 'bulk_progression_noleak', 'network_score']:
        vals = df[col].values
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-10:
            df[f'{col}_norm'] = (vals - mn) / (mx - mn)
        else:
            df[f'{col}_norm'] = 0.5

    # Compute new TransformationScore (NO IM subtype penalty from GSE78523)
    df['TransformationScore_leakfree'] = (
        0.30 * df['scRNA_risk_norm'] +
        0.30 * df['spatial_gradient_norm'] +
        0.25 * df['bulk_progression_noleak_norm'] +
        0.15 * df['network_score_norm']
    ) * df['direction_penalty']

    df = df.sort_values('TransformationScore_leakfree', ascending=False).reset_index(drop=True)

    print(f"\n  Top 15 (leakage-free):")
    print(f"  {'Rank':<5} {'Gene':<10} {'NewScore':>9} {'OldScore':>9} {'Change':>8}")
    print(f"  {'-'*45}")
    for i, row in df.head(15).iterrows():
        delta = row['TransformationScore_leakfree'] - row['original_TransformationScore']
        print(f"  {i+1:<5} {row['gene']:<10} {row['TransformationScore_leakfree']:>9.4f} {row['original_TransformationScore']:>9.4f} {delta:>+8.4f}")

    df.to_csv(f"{BASE}/results/leakfree_transformation_score.csv", index=False, encoding='utf-8-sig')
    print(f"\n  Saved: results/leakfree_transformation_score.csv")

    # =========================================================
    # Validate on GSE78523 (truly independent)
    # =========================================================
    print(f"\n{'='*70}")
    print("[2] True Independent Validation on GSE78523 (14v16)")
    print(f"{'='*70}")

    expr = pd.read_csv(GSE78523_EXPR, index_col=0)
    meta = pd.read_csv(GSE78523_META)
    common = [s for s in meta['sample_id'] if s in expr.columns]
    meta = meta[meta['sample_id'].isin(common)]

    im_prog = meta[meta['group'].str.contains('progressor')]['sample_id'].tolist()
    im_ctrl = meta[meta['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()
    all_ids = im_prog + im_ctrl
    y = np.array([1]*len(im_prog) + [0]*len(im_ctrl))

    # Pre-specified models based on leakage-free top genes
    top5_genes = df.head(5)['gene'].tolist()
    top10_genes = df.head(10)['gene'].tolist()
    top3_genes = df.head(3)['gene'].tolist()

    models = {
        'OLFM4_alone': ['OLFM4'],
        'leakfree_top3': top3_genes,
        'leakfree_top5': top5_genes,
        'leakfree_top10': top10_genes,
        'OLFM4+ITLN1': ['OLFM4', 'ITLN1'],
        'gain_core3': ['OLFM4', 'REG4', 'ITLN1'],
    }

    print(f"\n  Leakage-free top 5: {top5_genes}")
    print(f"  Leakage-free top 3: {top3_genes}")

    results = []
    for model_name, genes in models.items():
        avail = [g for g in genes if g in expr.index]
        if not avail:
            continue
        X = expr.loc[avail, all_ids].values.T
        auc, y_pred = loocv_auc(X, y)
        ci_low, ci_high = bootstrap_auc_ci(y, y_pred)
        results.append({
            'model': model_name,
            'genes': ';'.join(avail),
            'n_genes': len(avail),
            'AUC': round(auc, 4),
            'CI_low': round(ci_low, 4),
            'CI_high': round(ci_high, 4),
        })

    df_val = pd.DataFrame(results).sort_values('AUC', ascending=False)

    print(f"\n  {'Model':<20} {'Genes':<35} {'AUC':>6} {'95%CI':>15}")
    print(f"  {'-'*80}")
    for _, r in df_val.iterrows():
        genes_short = r['genes'][:32] + '...' if len(r['genes']) > 32 else r['genes']
        print(f"  {r['model']:<20} {genes_short:<35} {r['AUC']:>6.3f} [{r['CI_low']:.3f}-{r['CI_high']:.3f}]")

    df_val.to_csv(f"{BASE}/results/leakfree_validation_gse78523.csv", index=False, encoding='utf-8-sig')

    # Single gene validation with leakfree top 15
    print(f"\n  Single gene validation (leakfree top 15):")
    gene_results = []
    for gene in df.head(15)['gene']:
        if gene not in expr.index:
            continue
        g_prog = expr.loc[gene, im_prog].values.astype(float)
        g_ctrl = expr.loc[gene, im_ctrl].values.astype(float)
        _, p = mannwhitneyu(g_prog, g_ctrl, alternative='two-sided')
        d = compute_cohens_d(g_prog, g_ctrl)
        gene_results.append({'gene': gene, 'cohens_d': d, 'p_raw': p})

    df_genes = pd.DataFrame(gene_results).sort_values('p_raw')
    _, fdr, _, _ = multipletests(df_genes['p_raw'], method='fdr_bh')
    df_genes['p_fdr'] = fdr

    print(f"  {'Gene':<10} {'d':>7} {'p_raw':>10} {'p_fdr':>10}")
    for _, r in df_genes.iterrows():
        sig = '*' if r['p_raw'] < 0.05 else ''
        print(f"  {r['gene']:<10} {r['cohens_d']:>7.3f} {r['p_raw']:>10.4e} {r['p_fdr']:>10.4e} {sig}")

    print(f"\n{'='*70}")
    print("Step 17 COMPLETE — Truly independent validation")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
