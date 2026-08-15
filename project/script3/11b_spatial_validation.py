"""
Step 11b: Spatial Transcriptomics Validation (OMIX010346)
Input: Visium GP1-GP9 + Step 9 FINAL_PANEL
Output: script3/results/spatial_validation.csv, script3/figures/spatial_panel_score.png
Key: Patient-level paired statistics (n=9), NOT spot-level p-values.
     Region definition markers excluded from panel validation (anti-circular).
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import wilcoxon, kruskal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"

REGION_DEF_MARKERS = {'CDX2', 'MUC2', 'MUC5AC', 'KRT20', 'MKI67', 'VIM', 'EPCAM', 'TFF3'}


def define_regions(adata):
    """CDX2-based region definition (same as 11a, uses only region markers)."""
    def _get(gene):
        if gene in adata.var_names:
            v = adata[:, gene].X
            return np.asarray(v.todense()).flatten() if hasattr(v, 'todense') else np.asarray(v).flatten()
        return np.zeros(adata.n_obs)

    cdx2 = _get('CDX2')
    muc2 = _get('MUC2')
    krt20 = _get('KRT20')
    mki67 = _get('MKI67')
    epcam = _get('EPCAM')
    muc5ac = _get('MUC5AC')
    vim = _get('VIM')

    cdx2_thresh = np.percentile(cdx2[cdx2 > 0], 50) if (cdx2 > 0).sum() > 10 else 0.5
    regions = np.full(adata.n_obs, 'Unknown', dtype=object)
    regions[(epcam < 0.5) & (vim > np.percentile(vim, 70))] = 'Stroma'
    im_mask = (cdx2 > cdx2_thresh) | ((muc2 > np.percentile(muc2, 70)) & (krt20 > np.percentile(krt20, 70)))
    regions[im_mask] = 'IM'
    tumor_mask = (mki67 > np.percentile(mki67, 80)) & (epcam > np.percentile(epcam, 50)) & (muc5ac < np.percentile(muc5ac, 30))
    regions[tumor_mask] = 'Tumor'
    normal_mask = (cdx2 < cdx2_thresh) & (muc2 < np.percentile(muc2, 50)) & (epcam > np.percentile(epcam, 30))
    normal_mask = normal_mask & (regions == 'Unknown')
    regions[normal_mask] = 'Normal'
    adata.obs['region'] = regions


def compute_panel_score(adata, panel_genes, coefs):
    """Panel risk score = sum(coef_i * expr_i), only non-region-def genes."""
    valid_genes = [g for g in panel_genes if g in adata.var_names and g not in REGION_DEF_MARKERS]
    if not valid_genes:
        return None
    X = adata[:, valid_genes].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    valid_coefs = np.array([coefs.get(g, 1.0) for g in valid_genes])
    return X @ valid_coefs


def patient_level_paired_test(patient_scores):
    """Paired Wilcoxon on patient-level means: Tumor vs Normal."""
    tumor_means = []
    normal_means = []
    for gp, data in patient_scores.items():
        if 'Tumor' in data and 'Normal' in data:
            tumor_means.append(data['Tumor'])
            normal_means.append(data['Normal'])

    n_pairs = len(tumor_means)
    if n_pairs < 5:
        return {'n_pairs': n_pairs, 'p_value': np.nan, 'cohens_d': np.nan,
                'median_diff': np.nan, 'status': 'insufficient_pairs'}

    tumor_arr = np.array(tumor_means)
    normal_arr = np.array(normal_means)
    diff = tumor_arr - normal_arr
    median_diff = np.median(diff)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

    # Bootstrap 95% CI on median difference
    boot_medians = [np.median(np.random.choice(diff, size=n_pairs, replace=True)) for _ in range(2000)]
    ci_lower, ci_upper = np.percentile(boot_medians, [2.5, 97.5])

    try:
        stat, p = wilcoxon(tumor_arr, normal_arr, alternative='greater')
    except Exception:
        p = 1.0

    return {'n_pairs': n_pairs, 'p_value': p, 'cohens_d': cohens_d,
            'median_diff': median_diff, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'status': 'passed' if p < 0.05 else 'not_significant'}


def mixed_model_analysis(spot_data):
    """score ~ region + (1|patient), supplementary."""
    try:
        import statsmodels.formula.api as smf
        df = spot_data[spot_data['region'].isin(['Normal', 'IM', 'Tumor'])].copy()
        df['region'] = pd.Categorical(df['region'], categories=['Normal', 'IM', 'Tumor'], ordered=True)
        df['region_num'] = df['region'].cat.codes
        model = smf.mixedlm("score ~ region_num", df, groups=df["patient"])
        result = model.fit(reml=True)
        coef = result.params['region_num']
        p = result.pvalues['region_num']
        ci = result.conf_int().loc['region_num'].values
        return {'coef': coef, 'p': p, 'ci_lower': ci[0], 'ci_upper': ci[1]}
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        return None


def single_gene_spatial(adata_list, panel_genes):
    """Per-gene spatial distribution across regions (non-region-def genes only)."""
    valid_genes = [g for g in panel_genes if g not in REGION_DEF_MARKERS]
    results = []
    for gene in valid_genes:
        patient_im = []
        patient_normal = []
        for adata, gp in adata_list:
            if gene not in adata.var_names:
                continue
            v = adata[:, gene].X
            v = np.asarray(v.todense()).flatten() if hasattr(v, 'todense') else np.asarray(v).flatten()
            im_mask = adata.obs['region'] == 'IM'
            normal_mask = adata.obs['region'] == 'Normal'
            if im_mask.sum() >= 10 and normal_mask.sum() >= 10:
                patient_im.append(v[im_mask].mean())
                patient_normal.append(v[normal_mask].mean())

        if len(patient_im) >= 5:
            im_arr = np.array(patient_im)
            norm_arr = np.array(patient_normal)
            diff = im_arr - norm_arr
            d = np.mean(diff) / (np.std(diff) + 1e-10)
            try:
                _, p = wilcoxon(im_arr, norm_arr)
            except Exception:
                p = 1.0
            # Mark if gene was from spatial discovery (pool C)
            is_pool_c = False
            pool_c_path = f"{BASE}/results/spatial_gradient_genes.csv"
            if os.path.exists(pool_c_path):
                pool_c = pd.read_csv(pool_c_path)['gene'].tolist()
                is_pool_c = gene in pool_c
            results.append({'gene': gene, 'cohens_d': d, 'paired_p': p,
                            'n_patients': len(patient_im),
                            'spatial_discovery': is_pool_c,
                            'evidence_note': 'consistency_only' if is_pool_c else 'independent'})
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Step 11b: Spatial Transcriptomics Validation")
    print("  Statistical unit: Patient (n=9), NOT spot")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)
    os.makedirs(f"{BASE}/figures", exist_ok=True)

    # Load panel
    panel_path = f"{BASE}/results/FINAL_PANEL.csv"
    if not os.path.exists(panel_path):
        print("  FINAL_PANEL.csv not found")
        return
    panel = pd.read_csv(panel_path)
    panel_genes = panel['gene'].tolist()
    coefs = dict(zip(panel['gene'], panel['coef'])) if 'coef' in panel.columns else {g: 1.0 for g in panel_genes}

    # Exclude region definition markers from validation
    valid_panel = [g for g in panel_genes if g not in REGION_DEF_MARKERS]
    excluded = [g for g in panel_genes if g in REGION_DEF_MARKERS]
    print(f"  Panel: {len(panel_genes)} genes")
    print(f"  Valid for spatial validation: {len(valid_panel)} (excluded region markers: {excluded})")

    # [1] Load Visium samples
    print("\n[1] Loading Visium samples...")
    adata_list = []
    for gp in ['GP1', 'GP2', 'GP3', 'GP4', 'GP5', 'GP6', 'GP7', 'GP8', 'GP9']:
        path = f"{SPATIAL_DIR}/{gp}"
        if not os.path.exists(path):
            continue
        try:
            adata = sc.read_visium(path)
            adata.var_names_make_unique()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            define_regions(adata)
            print(f"  {gp}: {adata.n_obs} spots | Normal={( adata.obs['region']=='Normal').sum()} IM={(adata.obs['region']=='IM').sum()} Tumor={(adata.obs['region']=='Tumor').sum()}")
            adata_list.append((adata, gp))
        except Exception as e:
            print(f"  {gp}: failed ({e})")

    if not adata_list:
        print("  No spatial data loaded")
        return

    # [2] Compute panel score per spot
    print(f"\n[2] Computing panel risk score (non-region-def genes only)...")
    all_spot_data = []
    patient_scores = {}

    for adata, gp in adata_list:
        score = compute_panel_score(adata, panel_genes, coefs)
        if score is None:
            continue
        adata.obs['panel_score'] = score
        for region in ['Normal', 'IM', 'Tumor']:
            mask = adata.obs['region'] == region
            if mask.sum() >= 10:
                if gp not in patient_scores:
                    patient_scores[gp] = {}
                patient_scores[gp][region] = score[mask].mean()
                for idx in np.where(mask)[0]:
                    all_spot_data.append({'patient': gp, 'region': region, 'score': score[idx]})

    spot_df = pd.DataFrame(all_spot_data)

    # [3] Patient-level paired test (PRIMARY CONCLUSION)
    print(f"\n[3] Patient-level paired Wilcoxon (n={len(patient_scores)})...")
    paired_result = patient_level_paired_test(patient_scores)
    print(f"  Tumor vs Normal: p={paired_result.get('p_value', np.nan)}, Cohen's d={paired_result.get('cohens_d', np.nan)}")
    if 'ci_lower' in paired_result:
        print(f"  Median difference: {paired_result['median_diff']:.4f} (95% CI: [{paired_result['ci_lower']:.4f}, {paired_result['ci_upper']:.4f}])")
    else:
        print(f"  Insufficient pairs ({paired_result.get('n_pairs', 0)}) for paired test (need >=5)")
    if paired_result['p_value'] < 0.05:
        print(f"  PASS: Tumor > Normal (p<0.05)")
    elif paired_result['cohens_d'] > 0.5:
        print(f"  NOTE: p not significant but effect size d={paired_result['cohens_d']:.2f} > 0.5 (sample size limitation)")
    else:
        print(f"  NOTE: Not significant, report as-is")

    # IM vs Normal paired
    im_normal_scores = {}
    for gp, data in patient_scores.items():
        if 'IM' in data and 'Normal' in data:
            if gp not in im_normal_scores:
                im_normal_scores[gp] = {}
            im_normal_scores[gp] = {'IM': data['IM'], 'Normal': data['Normal']}
    if len(im_normal_scores) >= 5:
        im_arr = np.array([v['IM'] for v in im_normal_scores.values()])
        norm_arr = np.array([v['Normal'] for v in im_normal_scores.values()])
        try:
            _, p_im = wilcoxon(im_arr, norm_arr)
        except Exception:
            p_im = 1.0
        d_im = np.mean(im_arr - norm_arr) / (np.std(im_arr - norm_arr) + 1e-10)
        print(f"  IM vs Normal: p={p_im:.4f}, Cohen's d={d_im:.3f}")

    # [4] Mixed model (supplementary)
    print(f"\n[4] Mixed model: score ~ region + (1|patient)...")
    mm_result = mixed_model_analysis(spot_df)
    if mm_result:
        print(f"  Region effect: coef={mm_result['coef']:.4f}, p={mm_result['p']:.2e}")
        print(f"  95% CI: [{mm_result['ci_lower']:.4f}, {mm_result['ci_upper']:.4f}]")
        print("  NOTE: Spot-level mixed model may underestimate SE due to spatial autocorrelation")

    # [5] Spot-level KW (supplementary/visualization only)
    print(f"\n[5] Spot-level Kruskal-Wallis (supplementary, NOT primary)...")
    for region_pair in [('Tumor', 'Normal'), ('IM', 'Normal')]:
        r1, r2 = region_pair
        v1 = spot_df[spot_df['region'] == r1]['score'].values
        v2 = spot_df[spot_df['region'] == r2]['score'].values
        if len(v1) > 10 and len(v2) > 10:
            stat, p = kruskal(v1, v2)
            print(f"  {r1} vs {r2}: KW p={p:.2e} (n_spots={len(v1)}+{len(v2)}, inflated by spatial autocorrelation)")

    # [6] Single gene spatial distribution
    print(f"\n[6] Single gene spatial validation...")
    gene_spatial = single_gene_spatial(adata_list, panel_genes)
    if not gene_spatial.empty:
        gene_spatial.to_csv(f"{BASE}/results/spatial_gene_validation.csv", index=False)
        n_sig = (gene_spatial['paired_p'] < 0.05).sum()
        n_pool_c = gene_spatial['spatial_discovery'].sum()
        print(f"  Significant genes (patient-level p<0.05): {n_sig}/{len(gene_spatial)}")
        if n_pool_c > 0:
            print(f"  Pool C genes (consistency only, not independent evidence): {n_pool_c}")

    # [7] Spatial visualization
    print(f"\n[7] Spatial visualization...")
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for idx, (adata, gp) in enumerate(adata_list[:9]):
            ax = axes[idx // 3, idx % 3]
            if 'panel_score' in adata.obs.columns:
                sc.pl.spatial(adata, color='panel_score', ax=ax, show=False, title=gp)
        plt.tight_layout()
        plt.savefig(f"{BASE}/figures/spatial_panel_score.png", dpi=150)
        plt.close()
        print("  Saved spatial_panel_score.png")
    except Exception as e:
        print(f"  Visualization failed: {e}")

    # [8] Save results
    print(f"\n[8] Saving results...")
    validation_results = {
        'paired_test_p': paired_result['p_value'],
        'paired_cohens_d': paired_result['cohens_d'],
        'paired_median_diff': paired_result['median_diff'],
        'n_patients': paired_result['n_pairs'],
        'status': paired_result['status'],
    }
    if mm_result:
        validation_results['mixed_model_coef'] = mm_result['coef']
        validation_results['mixed_model_p'] = mm_result['p']
    pd.DataFrame([validation_results]).to_csv(f"{BASE}/results/spatial_validation.csv", index=False)

    print(f"\n{'='*60}")
    print("Step 11b COMPLETE")
    print(f"  Primary: Patient-level paired Wilcoxon (n={paired_result['n_pairs']})")
    print(f"  Tumor vs Normal: p={paired_result['p_value']:.4f}, d={paired_result['cohens_d']:.3f}")
    print(f"  Verification: d>0.5 = {'YES' if paired_result['cohens_d'] > 0.5 else 'NO'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
