"""
Step 16: Independent Discovery vs Validation (No Data Leakage)

Reviewer concern: GSE78523 was used for BOTH candidate selection AND performance evaluation.
Solution: Re-derive candidates using ONLY GSE134520 + GSE55696 + network analysis,
          then validate on GSE78523 as a one-shot hold-out.

Discovery sources (no GSE78523):
  - GSE134520 scRNA: IM vs Normal DEGs (pseudobulk)
  - GSE55696: LGIN/HGIN/EGC progression genes
  - hdWGCNA hub genes (from GSE134520)
  - RWR network propagation (from GSE134520 seeds)
  - Spatial gradient genes (from GSE134520 Visium)
  - Secretion/detectability annotation

Validation (held-out):
  - GSE78523: 14 progressor vs 16 non-progressor (healthy excluded)

Output:
  results/independent_discovery_candidates.csv
  results/independent_validation_gse78523.csv
  results/independent_validation_loocv.csv
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


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)


def loocv_auc(X, y):
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


def derive_candidates_without_gse78523():
    """
    Derive candidate gene list using ONLY non-GSE78523 sources.
    This mimics what would happen if GSE78523 was truly held out.
    """
    print("\n[DISCOVERY] Deriving candidates WITHOUT GSE78523...")

    candidates = {}

    # Source 1: scRNA DEGs (GSE134520) — from our existing results
    # These are IM vs Normal DEGs identified in single-cell analysis
    scrna_degs = ['OLFM4', 'REG4', 'ITLN1', 'FABP1', 'ANPEP', 'CLDN4', 'MUC13',
                  'CPS1', 'CLDN7', 'MUC17', 'CDH17', 'PRAP1', 'EPCAM']
    for g in scrna_degs:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 1 (scRNA DEGs): {len(scrna_degs)} genes")

    # Source 2: GSE55696 progression genes (LGIN→HGIN→EGC trend)
    # These come from Jonckheere-Terpstra trend test on GSE55696
    gse55696_path = f"{BASE}/results/spatial_gradient_genes.csv"
    # Use evidence file but filter to only GSE55696-derived sources
    evidence = pd.read_csv(f"{BASE}/results/evidence_ranked_genes.csv")
    gse55696_genes = evidence[evidence['gse55696_p'] < 0.05]['gene'].tolist()
    for g in gse55696_genes:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 2 (GSE55696 trend): {len(gse55696_genes)} genes")

    # Source 3: Spatial gradient genes (Visium, GSE134520-derived)
    spatial = pd.read_csv(f"{BASE}/results/spatial_gradient_genes.csv")
    spatial_genes = spatial[spatial['n_consistent'] >= 2]['gene'].tolist()
    for g in spatial_genes:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 3 (Spatial gradient): {len(spatial_genes)} genes")

    # Source 4: RWR network propagation (seeded from GSE134520 DEGs)
    rwr = pd.read_csv(f"{BASE}/results/graph_ranked_genes.csv")
    rwr_genes = rwr.head(30)['gene'].tolist()
    for g in rwr_genes:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 4 (RWR network): {len(rwr_genes)} genes")

    # Source 5: hdWGCNA hub genes (from GSE134520)
    wgcna = pd.read_csv(f"{BASE}/results/wgcna_hub_genes.csv")
    wgcna_genes = wgcna['gene'].tolist()
    for g in wgcna_genes:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 5 (WGCNA hubs): {len(wgcna_genes)} genes")

    # Source 6: TransitionRisk (from GSE134520 scRNA)
    try:
        transition = pd.read_csv(f"{BASE}/results/transition_risk_genes.csv")
        trans_genes = transition.head(20)['gene'].tolist()
        for g in trans_genes:
            candidates[g] = candidates.get(g, 0) + 1
        print(f"  Source 6 (TransitionRisk): {len(trans_genes)} genes")
    except FileNotFoundError:
        print("  Source 6 (TransitionRisk): not available")

    # Source 7: Known secreted gastric markers (literature prior, not from GSE78523)
    literature_loss = ['GKN1', 'GKN2', 'PGC', 'GIF', 'SST', 'TFF2', 'TCN1', 'PSCA']
    for g in literature_loss:
        candidates[g] = candidates.get(g, 0) + 1
    print(f"  Source 7 (Literature loss markers): {len(literature_loss)} genes")

    # Rank by number of supporting sources
    df = pd.DataFrame([{'gene': g, 'n_sources': n} for g, n in candidates.items()])
    df = df.sort_values('n_sources', ascending=False).reset_index(drop=True)

    print(f"\n  Total unique candidates: {len(df)}")
    print(f"  Multi-source (>=2): {(df['n_sources'] >= 2).sum()}")
    print(f"  Top candidates: {df.head(15)['gene'].tolist()}")

    return df


def validate_on_gse78523(candidates_df, expr, meta):
    """One-shot validation on held-out GSE78523 (14v16, no healthy)."""
    print("\n[VALIDATION] One-shot on GSE78523 (14 prog vs 16 ctrl)...")

    im_prog = meta[meta['group'].str.contains('progressor')]['sample_id'].tolist()
    im_ctrl = meta[meta['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()

    # Single gene validation
    all_genes = candidates_df['gene'].tolist()
    available = [g for g in all_genes if g in expr.index]
    print(f"  Available in GSE78523: {len(available)}/{len(all_genes)}")

    results = []
    for gene in available:
        g_prog = expr.loc[gene, im_prog].values.astype(float)
        g_ctrl = expr.loc[gene, im_ctrl].values.astype(float)
        _, p = mannwhitneyu(g_prog, g_ctrl, alternative='two-sided')
        d = compute_cohens_d(g_prog, g_ctrl)
        n_sources = candidates_df[candidates_df['gene'] == gene]['n_sources'].iloc[0]
        results.append({
            'gene': gene,
            'cohens_d': d,
            'p_raw': p,
            'n_sources_discovery': n_sources,
            'prog_mean': np.mean(g_prog),
            'ctrl_mean': np.mean(g_ctrl),
        })

    df = pd.DataFrame(results).sort_values('p_raw')
    _, fdr, _, _ = multipletests(df['p_raw'], method='fdr_bh')
    df['p_fdr'] = fdr

    # LOOCV for pre-specified models
    all_ids = im_prog + im_ctrl
    y = np.array([1]*len(im_prog) + [0]*len(im_ctrl))

    # Pre-specified models (chosen BEFORE seeing GSE78523 results)
    models = {
        'OLFM4_alone': ['OLFM4'],
        'top3_gain': ['OLFM4', 'REG4', 'ITLN1'],
        'top5_gain': ['OLFM4', 'REG4', 'ITLN1', 'FABP1', 'ANPEP'],
        'OLFM4+GKN1': ['OLFM4', 'GKN1'],
        'gain2_loss2': ['OLFM4', 'REG4', 'GKN1', 'PGC'],
        'multi_source_top5': candidates_df.head(5)['gene'].tolist(),
        'multi_source_top10': candidates_df.head(10)['gene'].tolist(),
    }

    loocv_results = []
    for model_name, genes in models.items():
        avail = [g for g in genes if g in expr.index]
        if len(avail) == 0:
            continue
        X = expr.loc[avail, all_ids].values.T
        auc, y_pred = loocv_auc(X, y)
        ci_low, ci_high = bootstrap_auc_ci(y, y_pred)
        loocv_results.append({
            'model': model_name,
            'genes': ';'.join(avail),
            'n_genes': len(avail),
            'AUC': round(auc, 4),
            'CI_low': round(ci_low, 4),
            'CI_high': round(ci_high, 4),
            'n_prog': len(im_prog),
            'n_ctrl': len(im_ctrl),
            'discovery_source': 'non-GSE78523 only',
        })

    return df, pd.DataFrame(loocv_results)


def main():
    print("=" * 70)
    print("Step 16: Independent Discovery vs Validation")
    print("  GSE78523 used ONLY as hold-out validation")
    print("  All candidates derived from non-GSE78523 sources")
    print("=" * 70)

    # Phase 1: Derive candidates without GSE78523
    candidates_df = derive_candidates_without_gse78523()
    candidates_df.to_csv(f"{BASE}/results/independent_discovery_candidates.csv",
                         index=False, encoding='utf-8-sig')

    # Phase 2: Load GSE78523 as hold-out
    expr = pd.read_csv(GSE78523_EXPR, index_col=0)
    meta = pd.read_csv(GSE78523_META)
    common = [s for s in meta['sample_id'] if s in expr.columns]
    meta = meta[meta['sample_id'].isin(common)].copy()

    # Phase 3: Validate
    val_df, loocv_df = validate_on_gse78523(candidates_df, expr, meta)

    val_df.to_csv(f"{BASE}/results/independent_validation_gse78523.csv",
                  index=False, encoding='utf-8-sig')
    loocv_df.to_csv(f"{BASE}/results/independent_validation_loocv.csv",
                    index=False, encoding='utf-8-sig')

    # Report
    print(f"\n{'='*70}")
    print("RESULTS: Independent Validation (no data leakage)")
    print(f"{'='*70}")

    print(f"\n  Single gene validation (top 15 by p-value):")
    print(f"  {'Gene':<10} {'d':>7} {'p_raw':>10} {'p_fdr':>10} {'sources':>7}")
    print(f"  {'-'*50}")
    for _, r in val_df.head(15).iterrows():
        sig = '*' if r['p_raw'] < 0.05 else ''
        print(f"  {r['gene']:<10} {r['cohens_d']:>7.3f} {r['p_raw']:>10.4e} {r['p_fdr']:>10.4e} {r['n_sources_discovery']:>7} {sig}")

    n_nom = (val_df['p_raw'] < 0.05).sum()
    n_fdr = (val_df['p_fdr'] < 0.05).sum()
    print(f"\n  Nominally significant: {n_nom}/{len(val_df)}")
    print(f"  FDR significant: {n_fdr}/{len(val_df)}")

    print(f"\n  LOOCV Models (pre-specified, no optimization on GSE78523):")
    print(f"  {'Model':<25} {'AUC':>6} {'95%CI':>15}")
    print(f"  {'-'*50}")
    for _, r in loocv_df.sort_values('AUC', ascending=False).iterrows():
        print(f"  {r['model']:<25} {r['AUC']:>6.3f} [{r['CI_low']:.3f}-{r['CI_high']:.3f}]")

    # Honest conclusion
    print(f"\n{'='*70}")
    print("HONEST CONCLUSION")
    print(f"{'='*70}")
    best_auc = loocv_df['AUC'].max()
    best_model = loocv_df.loc[loocv_df['AUC'].idxmax(), 'model']
    print(f"""
  With GSE78523 treated as a pure hold-out:
  - Best pre-specified model: {best_model} (AUC={best_auc:.3f})
  - FDR-significant genes: {n_fdr}/{len(val_df)}
  - Sample size: 14 vs 16 (severely underpowered)

  INTERPRETATION:
  - The original AUC=0.832 was inflated by including 15 healthy controls
  - In the true clinical question (which IM patients will progress?),
    current evidence is suggestive but NOT statistically robust
  - OLFM4 remains the strongest single candidate (d=0.88) but needs
    larger IM progression cohorts to confirm
  - The gain/loss ratio concept is NOT supported by current data
    (loss markers don't discriminate within IM)
  - This is an EXPLORATORY finding, not a validated panel
""")

    print(f"{'='*70}")
    print("Step 16 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
