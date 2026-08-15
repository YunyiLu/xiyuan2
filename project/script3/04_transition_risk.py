"""
Step 4: TransitionRisk score + HP discovery + trajectory support.
Input: script3/data/adata_integrated.h5ad (with Step 2 scores)
Output: adata.obs['transition_risk'], script3/results/transition_risk_genes.csv,
        script3/results/candidate_pool_E.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"

SCORE_DIMS = ['EGC_like_score', 'PMC_P_score', 'stemness_score',
              'cnv_score', 'dpt_pseudotime', 'incomplete_IM_score']


def compute_dpt(adata):
    """Diffusion pseudotime with NAG as root."""
    print("  Computing DPT (root=NAG)...")
    sc.tl.diffmap(adata)
    nag_mask = adata.obs['stage'] == 'NAG'
    if nag_mask.sum() > 0:
        nag_idx = np.where(nag_mask)[0]
        root = nag_idx[adata.obsm['X_diffmap'][nag_idx, 0].argmin()]
        adata.uns['iroot'] = root
        sc.tl.dpt(adata)
        print(f"  DPT range: [{adata.obs['dpt_pseudotime'].min():.3f}, "
              f"{adata.obs['dpt_pseudotime'].max():.3f}]")
    else:
        print("  WARNING: No NAG cells, using scVI dim 0 as pseudotime proxy")
        adata.obs['dpt_pseudotime'] = adata.obsm['X_scVI'][:, 0]
        dpt = adata.obs['dpt_pseudotime']
        adata.obs['dpt_pseudotime'] = (dpt - dpt.min()) / (dpt.max() - dpt.min())


def compute_transition_risk(adata, weights):
    """Compute TransitionRisk as weighted sum of 6 dimensions."""
    available = [d for d in SCORE_DIMS if d in adata.obs.columns]
    missing = [d for d in SCORE_DIMS if d not in adata.obs.columns]
    if missing:
        print(f"  WARNING: Missing dimensions: {missing}")

    # Standardize available scores (z-score within epithelial cells)
    epi_mask = adata.obs['is_epithelial']
    scores_matrix = np.zeros((epi_mask.sum(), len(available)))
    for i, dim in enumerate(available):
        vals = adata.obs.loc[epi_mask, dim].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0)
        scores_matrix[:, i] = vals

    scaler = StandardScaler()
    scores_z = scaler.fit_transform(scores_matrix)

    # Weighted sum
    w = np.array([weights.get(d, 0.0) for d in available])
    w = w / w.sum()
    risk = scores_z @ w

    # Normalize to [0, 1]
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)

    adata.obs['transition_risk'] = np.nan
    adata.obs.loc[epi_mask, 'transition_risk'] = risk
    return scores_z, available


def sensitivity_analysis(adata, scores_z, available):
    """Weight sensitivity: equal, expert, PCA-driven, leave-one-out."""
    epi_mask = adata.obs['is_epithelial']
    stage_map = {'NAG': 0, 'CAG': 1, 'IM': 2, 'EGC': 3, 'GC': 3, 'EGC_multi_region': 3}
    stage_num = adata.obs.loc[epi_mask, 'stage'].map(stage_map).values.astype(float)
    valid = ~np.isnan(stage_num)

    results = {}

    # Scheme A: Equal weights
    w_equal = np.ones(len(available)) / len(available)
    risk_A = scores_z @ w_equal
    results['equal'] = {'weights': dict(zip(available, w_equal)), 'risk': risk_A}

    # Scheme B: Expert weights (literature-based priors)
    expert_w = {'EGC_like_score': 0.25, 'PMC_P_score': 0.20, 'stemness_score': 0.15,
                'cnv_score': 0.10, 'dpt_pseudotime': 0.15, 'incomplete_IM_score': 0.15}
    w_expert = np.array([expert_w.get(d, 1/len(available)) for d in available])
    w_expert = w_expert / w_expert.sum()
    risk_B = scores_z @ w_expert
    results['expert'] = {'weights': dict(zip(available, w_expert)), 'risk': risk_B}

    # Scheme C: PCA weights (data-driven, stage as response)
    from sklearn.cross_decomposition import PLSRegression
    if valid.sum() > 50:
        pls = PLSRegression(n_components=1)
        pls.fit(scores_z[valid], stage_num[valid])
        w_pls = np.abs(pls.coef_.flatten())
        w_pls = w_pls / w_pls.sum()
        risk_C = scores_z @ w_pls
        results['PLS'] = {'weights': dict(zip(available, w_pls)), 'risk': risk_C}

    # Scheme D: Leave-one-component-out
    loo_corrs = {}
    for i, dim in enumerate(available):
        mask_dims = [j for j in range(len(available)) if j != i]
        w_loo = np.ones(len(mask_dims)) / len(mask_dims)
        risk_loo = scores_z[:, mask_dims] @ w_loo
        if valid.sum() > 10:
            r, _ = spearmanr(risk_loo[valid], stage_num[valid])
        else:
            r = 0
        loo_corrs[dim] = r
    results['leave_one_out'] = loo_corrs

    # Scheme E: HP sensitivity (recompute using only HP+ samples)
    hp_plus_mask = adata.obs.loc[epi_mask, 'hp_status'] == 'HP+'
    if hp_plus_mask.sum() > 50:
        risk_E = scores_z[hp_plus_mask.values] @ w_equal
        results['HP_only'] = {'risk': risk_E, 'mask': hp_plus_mask.values}
        r_hp, _ = spearmanr(risk_E[~np.isnan(stage_num[hp_plus_mask.values])],
                            stage_num[hp_plus_mask.values][~np.isnan(stage_num[hp_plus_mask.values])])
        print(f"    HP+ only: Spearman r={r_hp:.3f}")

    # Report
    print("\n  Sensitivity analysis:")
    for scheme in ['equal', 'expert', 'PLS']:
        if scheme in results:
            r, _ = spearmanr(results[scheme]['risk'][valid], stage_num[valid])
            print(f"    {scheme}: Spearman r={r:.3f}")
    print(f"    Leave-one-out correlations: {loo_corrs}")

    # Top30 stability: gene-risk correlation under each scheme, count overlap
    X_epi = adata[epi_mask].X
    if hasattr(X_epi, 'toarray'):
        X_epi = X_epi.toarray()
    scheme_top30 = {}
    for scheme_name in ['equal', 'expert', 'PLS']:
        if scheme_name not in results or 'risk' not in results[scheme_name]:
            continue
        r_vals = results[scheme_name]['risk']
        v = ~np.isnan(r_vals)
        gene_corrs = [spearmanr(X_epi[v, i], r_vals[v])[0] for i in range(X_epi.shape[1])]
        top30_idx = np.argsort(gene_corrs)[-30:]
        scheme_top30[scheme_name] = set(adata.var_names[top30_idx])

    if len(scheme_top30) >= 2:
        common = set.intersection(*scheme_top30.values())
        print(f"  Top30 stability: {len(common)}/30 genes stable across {len(scheme_top30)} schemes")
    return results


def hp_analysis(adata):
    """HP+/HP- differential analysis within IM stage → candidate pool E."""
    print("\n  HP infection analysis (GSE249874 IM samples)...")
    epi_mask = adata.obs['is_epithelial']
    im_mask = adata.obs['stage'] == 'IM'
    combined = epi_mask & im_mask

    hp_plus = combined & (adata.obs['hp_status'] == 'HP+')
    hp_minus = combined & (adata.obs['hp_status'] == 'HP-')

    n_plus = hp_plus.sum()
    n_minus = hp_minus.sum()
    print(f"  IM HP+ cells: {n_plus}, IM HP- cells: {n_minus}")

    if n_plus < 50 or n_minus < 50:
        print("  Too few cells for HP analysis, skipping")
        return pd.DataFrame()

    # Pseudobulk DE: aggregate by sample, then compare
    samples_plus = adata.obs.loc[hp_plus, 'sample_id'].unique()
    samples_minus = adata.obs.loc[hp_minus, 'sample_id'].unique()
    print(f"  HP+ samples: {len(samples_plus)}, HP- samples: {len(samples_minus)}")

    if len(samples_plus) < 2 or len(samples_minus) < 2:
        print("  Too few samples for pseudobulk DE, skipping")
        return pd.DataFrame()

    # Pseudobulk per sample
    def get_pseudobulk(sample_ids):
        pb = []
        for sid in sample_ids:
            mask = (adata.obs['sample_id'] == sid) & epi_mask
            if mask.sum() < 10:
                continue
            X = adata[mask].X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            pb.append(X.mean(axis=0))
        return np.array(pb) if pb else None

    pb_plus = get_pseudobulk(samples_plus)
    pb_minus = get_pseudobulk(samples_minus)

    if pb_plus is None or pb_minus is None or len(pb_plus) < 2 or len(pb_minus) < 2:
        print("  Insufficient pseudobulk data")
        return pd.DataFrame()

    # Wilcoxon per gene (sample-level)
    de_results = []
    for i in range(adata.n_vars):
        stat, pval = mannwhitneyu(pb_plus[:, i], pb_minus[:, i], alternative='two-sided')
        log2fc = np.log2((pb_plus[:, i].mean() + 1e-6) / (pb_minus[:, i].mean() + 1e-6))
        de_results.append({'gene': adata.var_names[i], 'log2fc': log2fc, 'pval': pval})

    de_df = pd.DataFrame(de_results)
    from statsmodels.stats.multitest import multipletests
    de_df['padj'] = multipletests(de_df['pval'], method='fdr_bh')[1]

    # HP-specific genes: padj < 0.1 (exploratory, low power with n=3 per group)
    hp_sig = de_df[de_df['padj'] < 0.1].copy()
    print(f"  HP-differential genes (padj<0.1): {len(hp_sig)}")

    # Intersect with TransitionRisk-correlated genes
    if 'transition_risk' in adata.obs.columns:
        epi_data = adata[epi_mask]
        risk = epi_data.obs['transition_risk'].values
        valid_risk = ~np.isnan(risk)
        if valid_risk.sum() > 50:
            X_epi = epi_data.X
            if hasattr(X_epi, 'toarray'):
                X_epi = X_epi.toarray()
            risk_corr = []
            for i in range(X_epi.shape[1]):
                r, p = spearmanr(X_epi[valid_risk, i], risk[valid_risk])
                risk_corr.append(r)
            de_df['risk_corr'] = risk_corr

            # Pool E: HP-differential AND TransitionRisk-correlated
            pool_E = de_df[(de_df['padj'] < 0.1) & (de_df['risk_corr'].abs() > 0.1)]
            pool_E = pool_E.sort_values('risk_corr', ascending=False)
            print(f"  Candidate pool E (HP-specific ∩ risk-correlated): {len(pool_E)} genes")
            return pool_E

    return hp_sig


def run_paga(adata):
    """PAGA topology validation."""
    print("  Running PAGA...")
    sc.tl.paga(adata, groups='celltype')
    # Check IM→EGC-like connection
    connectivities = adata.uns['paga']['connectivities'].toarray()
    cats = adata.obs['celltype'].cat.categories.tolist()
    im_types = [c for c in cats if 'IM' in c or 'Enterocyte' in c or 'Goblet' in c]
    prolif_types = [c for c in cats if 'Stem' in c or 'proliferative' in c.lower()]
    if im_types and prolif_types:
        for im_t in im_types:
            for pr_t in prolif_types:
                i, j = cats.index(im_t), cats.index(pr_t)
                conn = connectivities[i, j]
                print(f"    PAGA {im_t} → {pr_t}: connectivity={conn:.3f}")


def main():
    print("=" * 60)
    print("Step 4: TransitionRisk Score + HP Discovery")
    print("=" * 60)

    # [1] Load
    print("\n[1] Loading data...")
    adata = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
    epi_mask = adata.obs['is_epithelial']
    print(f"  {adata.n_obs} cells, {epi_mask.sum()} epithelial")

    # [2] DPT on epithelial subset
    print("\n[2] Diffusion pseudotime...")
    adata_epi = adata[epi_mask].copy()
    sc.pp.neighbors(adata_epi, use_rep="X_scVI", n_neighbors=30)
    compute_dpt(adata_epi)
    adata.obs.loc[epi_mask, 'dpt_pseudotime'] = adata_epi.obs['dpt_pseudotime'].values

    # [3] TransitionRisk (equal weights first)
    print("\n[3] Computing TransitionRisk (6 dimensions)...")
    os.makedirs(f"{BASE}/results", exist_ok=True)
    equal_weights = {d: 1.0/6 for d in SCORE_DIMS}
    scores_z, available = compute_transition_risk(adata, equal_weights)

    # Validate: TransitionRisk vs stage
    stage_map = {'NAG': 0, 'CAG': 1, 'IM': 2, 'EGC': 3, 'GC': 3, 'EGC_multi_region': 3}
    stage_num = adata.obs.loc[epi_mask, 'stage'].map(stage_map).values.astype(float)
    risk = adata.obs.loc[epi_mask, 'transition_risk'].values
    valid = ~np.isnan(stage_num) & ~np.isnan(risk)
    r, p = spearmanr(risk[valid], stage_num[valid])
    print(f"  TransitionRisk vs stage: Spearman r={r:.3f}, p={p:.2e}")

    # [4] Sensitivity analysis
    print("\n[4] Weight sensitivity analysis...")
    sens_results = sensitivity_analysis(adata, scores_z, available)

    # [4b] Linear model: TransitionRisk ~ stage + hp_status + stage:hp_status
    print("\n[4b] Linear model (HP as covariate)...")
    import statsmodels.formula.api as smf
    epi_df = adata.obs.loc[epi_mask, ['stage', 'hp_status', 'transition_risk']].copy()
    epi_df['stage_num'] = epi_df['stage'].map(stage_map)
    epi_df = epi_df.dropna(subset=['stage_num', 'transition_risk'])
    epi_df = epi_df[epi_df['hp_status'].isin(['HP+', 'HP-'])]
    if len(epi_df) > 100:
        model_lm = smf.ols('transition_risk ~ stage_num * hp_status', data=epi_df).fit()
        print(f"  R²={model_lm.rsquared:.3f}")
        hp_coef = model_lm.params.get('hp_status[T.HP+]', None)
        hp_p = model_lm.pvalues.get('hp_status[T.HP+]', None)
        int_coef = model_lm.params.get('stage_num:hp_status[T.HP+]', None)
        int_p = model_lm.pvalues.get('stage_num:hp_status[T.HP+]', None)
        print(f"  hp_status coef: {hp_coef if hp_coef is not None else 'N/A'}, "
              f"p={hp_p if hp_p is not None else 'N/A'}")
        print(f"  interaction coef: {int_coef if int_coef is not None else 'N/A'}, "
              f"p={int_p if int_p is not None else 'N/A'}")
    else:
        print("  Too few cells with HP annotation for linear model")

    # [5] PAGA topology
    print("\n[5] PAGA trajectory validation...")
    run_paga(adata_epi)

    # [5b] CytoTRACE2 differentiation direction
    print("\n[5b] CytoTRACE2 differentiation validation...")
    try:
        import cytotrace2
        cytotrace2.score(adata_epi)
        ct2_by_stage = adata_epi.obs.groupby('stage')['cytotrace2_score'].mean()
        print(f"  CytoTRACE2 by stage: {ct2_by_stage.to_dict()}")
    except ImportError:
        print("  cytotrace2 not installed, using gene-count proxy for differentiation...")
        # Proxy: number of expressed genes correlates with differentiation potential
        n_genes = (adata_epi.X > 0).sum(axis=1)
        if hasattr(n_genes, 'A1'):
            n_genes = n_genes.A1
        adata_epi.obs['diff_proxy'] = n_genes
        proxy_by_stage = adata_epi.obs.groupby('stage')['diff_proxy'].mean()
        print(f"  Gene-count differentiation proxy by stage: {proxy_by_stage.to_dict()}")

    # [6] Gene-TransitionRisk correlations
    print("\n[6] Gene-TransitionRisk correlations...")
    X_epi = adata[epi_mask].X
    if hasattr(X_epi, 'toarray'):
        X_epi = X_epi.toarray()
    risk_vals = adata.obs.loc[epi_mask, 'transition_risk'].values
    valid_r = ~np.isnan(risk_vals)

    correlations = []
    for i in range(X_epi.shape[1]):
        r_gene, p_gene = spearmanr(X_epi[valid_r, i], risk_vals[valid_r])
        correlations.append({'gene': adata.var_names[i], 'risk_corr': r_gene, 'pval': p_gene})
    corr_df = pd.DataFrame(correlations).sort_values('risk_corr', ascending=False)
    corr_df.to_csv(f"{BASE}/results/transition_risk_genes.csv", index=False)
    print(f"  Top risk-correlated genes: {corr_df.head(10)['gene'].tolist()}")
    print(f"  Genes with |r|>0.2: {(corr_df['risk_corr'].abs() > 0.2).sum()}")

    # [7] HP analysis → candidate pool E
    print("\n[7] HP infection analysis...")
    pool_E = hp_analysis(adata)
    if len(pool_E) > 0:
        pool_E[['gene', 'log2fc', 'padj', 'risk_corr']].to_csv(
            f"{BASE}/results/candidate_pool_E.csv", index=False)

    # [8] Save
    print("\n[8] Saving...")
    adata.write_h5ad(f"{BASE}/data/adata_integrated.h5ad")

    print(f"\n{'='*60}")
    print("Step 4 COMPLETE")
    print(f"  TransitionRisk vs stage: r={r:.3f}")
    print(f"  Risk-correlated genes (|r|>0.2): {(corr_df['risk_corr'].abs() > 0.2).sum()}")
    print(f"  Candidate pool E: {len(pool_E)} genes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
