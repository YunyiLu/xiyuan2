"""
Step 11a: Spatial Gradient Unbiased Discovery (before Step 8)
Input: OMIX010346 Visium 9 samples + CDX2-based region definition
Output: script3/results/spatial_gradient_genes.csv (pool C), spatial_niche_composition.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"

REGION_MARKERS = {'CDX2', 'MUC2', 'MUC5AC', 'KRT20', 'MKI67', 'VIM', 'EPCAM', 'TFF3'}


def define_regions(adata):
    """CDX2-based region definition (not using panel genes)."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

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

    # Stroma
    regions[(epcam < 0.5) & (vim > np.percentile(vim, 70))] = 'Stroma'
    # IM: CDX2-high or (MUC2-high & KRT20-high)
    im_mask = (cdx2 > cdx2_thresh) | ((muc2 > np.percentile(muc2, 70)) & (krt20 > np.percentile(krt20, 70)))
    regions[im_mask] = 'IM'
    # Tumor: MKI67-high & EPCAM-high & MUC5AC-low
    tumor_mask = (mki67 > np.percentile(mki67, 80)) & (epcam > np.percentile(epcam, 50)) & (muc5ac < np.percentile(muc5ac, 30))
    regions[tumor_mask] = 'Tumor'
    # Normal: CDX2-low & MUC2-low & EPCAM+
    normal_mask = (cdx2 < cdx2_thresh) & (muc2 < np.percentile(muc2, 50)) & (epcam > np.percentile(epcam, 30))
    normal_mask = normal_mask & (regions == 'Unknown')
    regions[normal_mask] = 'Normal'

    adata.obs['region'] = regions
    return adata


def compute_spatial_gradient(adata, patient_id):
    """Cohen's d for each gene: IM vs Normal within one patient."""
    im_mask = adata.obs['region'] == 'IM'
    normal_mask = adata.obs['region'] == 'Normal'
    if im_mask.sum() < 20 or normal_mask.sum() < 20:
        return pd.DataFrame()

    results = []
    genes = [g for g in adata.var_names if g not in REGION_MARKERS]
    X_im = adata[im_mask].X
    X_normal = adata[normal_mask].X
    if hasattr(X_im, 'toarray'):
        X_im = X_im.toarray()
        X_normal = X_normal.toarray()

    for i, gene in enumerate(genes):
        im_vals = X_im[:, i]
        norm_vals = X_normal[:, i]
        pooled_std = np.sqrt((np.var(im_vals) + np.var(norm_vals)) / 2) + 1e-10
        cohens_d = (np.mean(im_vals) - np.mean(norm_vals)) / pooled_std
        results.append({'gene': gene, 'cohens_d': cohens_d, 'patient': patient_id})
    return pd.DataFrame(results)


def niche_analysis(adata_list):
    """Neighborhood enrichment: which cell type combos at IM-Tumor boundary."""
    try:
        import squidpy as sq
    except ImportError:
        print("  squidpy not available, skipping niche analysis")
        return pd.DataFrame()

    niche_results = []
    for adata, gp in adata_list:
        if 'region' not in adata.obs.columns:
            continue
        try:
            sq.gr.spatial_neighbors(adata)
            sq.gr.nhood_enrichment(adata, cluster_key='region')
            zscore_matrix = adata.uns['region_nhood_enrichment']['zscore']
            categories = list(adata.obs['region'].cat.categories) if hasattr(adata.obs['region'], 'cat') else sorted(adata.obs['region'].unique())
            for i, r1 in enumerate(categories):
                for j, r2 in enumerate(categories):
                    if i < j:
                        niche_results.append({'patient': gp, 'region1': r1, 'region2': r2,
                                              'zscore': zscore_matrix[i, j]})
        except Exception as e:
            print(f"    {gp}: niche analysis failed ({e})")
    return pd.DataFrame(niche_results)


def main():
    print("=" * 60)
    print("Step 11a: Spatial Gradient Unbiased Discovery")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load and define regions for each patient
    print("\n[1] Loading Visium samples and defining regions...")
    all_gradients = []
    adata_list = []

    for gp in ['GP1', 'GP2', 'GP3', 'GP4', 'GP5', 'GP6', 'GP7', 'GP8', 'GP9']:
        path = f"{SPATIAL_DIR}/{gp}"
        if not os.path.exists(path):
            continue
        try:
            adata = sc.read_visium(path)
            adata.var_names_make_unique()
            define_regions(adata)
            n_im = (adata.obs['region'] == 'IM').sum()
            n_normal = (adata.obs['region'] == 'Normal').sum()
            print(f"  {gp}: {adata.n_obs} spots, IM={n_im}, Normal={n_normal}")
            adata_list.append((adata, gp))
        except Exception as e:
            print(f"  {gp}: failed ({e})")

    # [2] Compute spatial gradient per patient
    print(f"\n[2] Computing Normal→IM gradient (Cohen's d)...")
    for adata, gp in adata_list:
        grad = compute_spatial_gradient(adata, gp)
        if not grad.empty:
            all_gradients.append(grad)

    if not all_gradients:
        print("  No gradient data available")
        return

    grad_df = pd.concat(all_gradients)

    # [3] Filter: >=5/9 patients consistent direction (IM > Normal) & effect > 0.3
    print("\n[3] Filtering consistent gradient genes...")
    gene_stats = []
    for gene, gdf in grad_df.groupby('gene'):
        n_up = (gdf['cohens_d'] > 0.3).sum()
        n_patients = len(gdf)
        mean_d = gdf['cohens_d'].mean()
        if n_up >= 5 and n_patients >= 5:
            gene_stats.append({'gene': gene, 'n_consistent': n_up, 'n_patients': n_patients,
                               'mean_cohens_d': mean_d})

    gene_stats_df = pd.DataFrame(gene_stats)
    if gene_stats_df.empty:
        print("  No genes pass consistency filter (need >=5 patients with both regions)")
        print("  Relaxing to >=3 patients...")
        gene_stats = []
        for gene, gdf in grad_df.groupby('gene'):
            n_up = (gdf['cohens_d'] > 0.3).sum()
            n_patients = len(gdf)
            mean_d = gdf['cohens_d'].mean()
            if n_up >= 3 and n_patients >= 3:
                gene_stats.append({'gene': gene, 'n_consistent': n_up, 'n_patients': n_patients,
                                   'mean_cohens_d': mean_d})
        gene_stats_df = pd.DataFrame(gene_stats)
    if gene_stats_df.empty:
        print("  Still no genes pass filter. Saving empty results.")
        pd.DataFrame(columns=['gene','n_consistent','n_patients','mean_cohens_d']).to_csv(
            f"{BASE}/results/spatial_gradient_genes.csv", index=False)
        return
    gene_stats_df = gene_stats_df.sort_values('mean_cohens_d', ascending=False)
    print(f"  Genes with >=5/9 consistent direction & d>0.3: {len(gene_stats_df)}")

    # [4] Exclude region definition markers
    gene_stats_df = gene_stats_df[~gene_stats_df['gene'].isin(REGION_MARKERS)]
    print(f"  After excluding region markers: {len(gene_stats_df)}")

    # [5] Output pool C (top 20)
    pool_c = gene_stats_df.head(20)
    pool_c.to_csv(f"{BASE}/results/spatial_gradient_genes.csv", index=False)
    print(f"  Pool C (top 20): {pool_c['gene'].tolist()[:10]}...")

    # [6] Niche analysis (descriptive, not for scoring)
    print("\n[4] Spatial niche analysis (neighborhood enrichment)...")
    niche_df = niche_analysis(adata_list)
    if not niche_df.empty:
        # Filter: consistent in >=5/9 patients
        boundary_pairs = niche_df[niche_df['zscore'] > 2.0]
        pair_counts = boundary_pairs.groupby(['region1', 'region2']).size().reset_index(name='n_patients')
        consistent = pair_counts[pair_counts['n_patients'] >= 5]
        consistent.to_csv(f"{BASE}/results/spatial_niche_composition.csv", index=False)
        print(f"  Consistent niche pairs (>=5/9 patients): {len(consistent)}")
        for _, row in consistent.iterrows():
            print(f"    {row['region1']}-{row['region2']}: {row['n_patients']}/9 patients")
    else:
        print("  Niche analysis not available")

    # Moran's I for top gradient genes
    print("\n[5] Moran's I spatial autocorrelation (top gradient genes)...")
    try:
        import squidpy as sq
        top_genes = pool_c['gene'].tolist()[:10]
        moran_results = []
        for adata, gp in adata_list:
            avail = [g for g in top_genes if g in adata.var_names]
            if avail:
                try:
                    sq.gr.spatial_neighbors(adata)
                    sq.gr.spatial_autocorr(adata, genes=avail, mode='moran')
                    for gene in avail:
                        if gene in adata.uns['moranI'].index:
                            moran_results.append({'patient': gp, 'gene': gene,
                                                  'moran_I': adata.uns['moranI'].loc[gene, 'I'],
                                                  'pval': adata.uns['moranI'].loc[gene, 'pval_norm']})
                except Exception:
                    pass
        if moran_results:
            moran_df = pd.DataFrame(moran_results)
            n_sig = moran_df.groupby('gene').apply(lambda x: (x['pval'] < 0.05).sum())
            print(f"  Genes with significant Moran's I in >=5 patients: {(n_sig >= 5).sum()}")
    except ImportError:
        print("  squidpy not available")

    print(f"\n{'='*60}")
    print("Step 11a COMPLETE")
    print(f"  Pool C: {len(pool_c)} genes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
