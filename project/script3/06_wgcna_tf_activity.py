"""
Step 6: hdWGCNA (metacell-level) + decoupler TF activity + cNMF×TF cross + spatial TF.
Input: script3/data/adata_integrated.h5ad, script3/data/spatial_deconv.h5ad
Output: script3/results/wgcna_modules.csv, tf_activity.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"


def build_metacells(adata_epi, n_metacells=300):
    """Build metacells in scVI latent space using KNN aggregation."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import MiniBatchKMeans

    print(f"  Building {n_metacells} metacells from {adata_epi.n_obs} epithelial cells...")
    latent = adata_epi.obsm['X_scVI']

    # KMeans in latent space to define metacells
    km = MiniBatchKMeans(n_clusters=n_metacells, random_state=0, batch_size=1000)
    labels = km.fit_predict(latent)
    adata_epi.obs['metacell'] = labels

    # Aggregate expression per metacell
    X = adata_epi.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    metacell_expr = np.zeros((n_metacells, X.shape[1]))
    metacell_meta = []
    for mc in range(n_metacells):
        mask = labels == mc
        if mask.sum() < 5:
            continue
        metacell_expr[mc] = X[mask].mean(axis=0)
        # Metacell-level metadata: majority stage, mean TransitionRisk
        mc_obs = adata_epi.obs.loc[mask]
        stage = mc_obs['stage'].mode().iloc[0] if len(mc_obs['stage'].mode()) > 0 else 'unknown'
        risk = mc_obs['transition_risk'].mean() if 'transition_risk' in mc_obs.columns else np.nan
        metacell_meta.append({'metacell': mc, 'stage': stage, 'transition_risk': risk,
                              'n_cells': mask.sum()})

    meta_df = pd.DataFrame(metacell_meta)
    expr_df = pd.DataFrame(metacell_expr, columns=adata_epi.var_names)
    # Remove empty metacells
    valid = meta_df['n_cells'] >= 5
    expr_df = expr_df.loc[valid.values]
    meta_df = meta_df.loc[valid.values].reset_index(drop=True)
    print(f"  Valid metacells: {len(meta_df)}")
    return expr_df, meta_df


def run_hdwgcna(expr_df, meta_df):
    """Run WGCNA on metacell expression matrix."""
    # Select top 2000 variable genes
    gene_var = expr_df.var().sort_values(ascending=False)
    top_genes = gene_var.head(2000).index.tolist()
    expr_top = expr_df[top_genes]

    # Soft threshold selection
    print("  Soft threshold selection...")
    powers = [2, 4, 6, 8, 10, 12]
    best_power = 6
    for p in powers:
        corr = expr_top.corr().abs()
        adj = corr ** p
        np.fill_diagonal(adj.values, 0)
        k = adj.sum(axis=1)
        # Scale-free topology fit (R² of log(k) vs log(p(k)))
        hist, bin_edges = np.histogram(k, bins=20)
        mask = hist > 0
        if mask.sum() > 3:
            from scipy.stats import linregress
            x = np.log10(bin_edges[1:][mask])
            y = np.log10(hist[mask].astype(float))
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 3:
                r2 = linregress(x[valid], y[valid]).rvalue ** 2
                if r2 > 0.8:
                    best_power = p
                    break
    print(f"  Selected power: {best_power}")

    # Build adjacency and TOM
    corr = expr_top.corr().abs()
    adj = corr ** best_power
    np.fill_diagonal(adj.values, 0)

    # Module detection via hierarchical clustering
    dist = 1 - adj.values
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method='average')
    # Dynamic tree cut: try multiple thresholds, pick one giving 10-30 modules
    try:
        from dynamicTreeCut import cutreeHybrid
        modules = cutreeHybrid(Z, distM=dist, minClusterSize=30)['labels']
    except ImportError:
        best_t, best_n = 0.85, 0
        for t in [0.7, 0.75, 0.8, 0.85, 0.9]:
            m = fcluster(Z, t=t, criterion='distance')
            n = len(set(m))
            if 10 <= n <= 30:
                best_t, best_n = t, n
                break
        modules = fcluster(Z, t=best_t, criterion='distance')
        print(f"  (dynamicTreeCut not installed, using threshold={best_t})")

    module_df = pd.DataFrame({'gene': top_genes, 'module': modules})
    n_modules = module_df['module'].nunique()
    print(f"  Modules detected: {n_modules}")

    # Module-trait correlation (TransitionRisk)
    print("  Module-trait correlations...")
    trait = meta_df['transition_risk'].values
    valid_trait = ~np.isnan(trait)
    hub_genes = []

    module_results = []
    for mod in module_df['module'].unique():
        genes = module_df[module_df['module'] == mod]['gene'].tolist()
        if len(genes) < 5:
            continue
        # Module eigengene (mean expression of module genes)
        eigengene = expr_top[genes].mean(axis=1).values
        if valid_trait.sum() > 10:
            r, p = spearmanr(eigengene[valid_trait], trait[valid_trait])
        else:
            r, p = 0, 1
        module_results.append({'module': mod, 'n_genes': len(genes),
                               'trait_corr': r, 'trait_pval': p})

        # Hub genes: module membership > 0.8 AND gene significance > 0.3
        for g in genes:
            # Module membership: correlation of gene with eigengene
            mm = np.corrcoef(expr_top[g].values, eigengene)[0, 1]
            # Gene significance: correlation with trait
            if valid_trait.sum() > 10:
                gs, _ = spearmanr(expr_top[g].values[valid_trait], trait[valid_trait])
            else:
                gs = 0
            if abs(mm) > 0.8 and abs(gs) > 0.3:
                hub_genes.append({'gene': g, 'module': mod, 'membership': mm,
                                  'gene_significance': gs})

    module_trait_df = pd.DataFrame(module_results).sort_values('trait_pval')
    hub_df = pd.DataFrame(hub_genes)
    print(f"  Trait-correlated modules (p<0.05): {(module_trait_df['trait_pval'] < 0.05).sum()}")
    print(f"  Hub genes (MM>0.8, GS>0.3): {len(hub_df)}")

    # Module pathway enrichment (verify EMT/WNT/inflammatory)
    print("  Module pathway enrichment...")
    try:
        import gseapy as gp
        sig_modules = module_trait_df[module_trait_df['trait_pval'] < 0.05]['module'].tolist()
        enrich_results = []
        for mod in sig_modules[:5]:
            genes = module_df[module_df['module'] == mod]['gene'].tolist()
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets='MSigDB_Hallmark_2020',
                                 organism='human', no_plot=True, cutoff=0.05)
                for _, row in enr.results.head(3).iterrows():
                    enrich_results.append({'module': mod, 'pathway': row['Term'],
                                           'padj': row['Adjusted P-value']})
            except:
                pass
        if enrich_results:
            enrich_df = pd.DataFrame(enrich_results)
            enrich_df.to_csv(f"{BASE}/results/wgcna_module_pathways.csv", index=False)
            print(f"    Top pathways: {[r['pathway'] for r in enrich_results[:5]]}")
    except ImportError:
        print("  gseapy not available, skipping pathway enrichment")

    return module_df, module_trait_df, hub_df


def module_preservation_analysis(adata_epi, module_df):
    """Module preservation: run WGCNA independently on two datasets, check overlap."""
    print("  Module preservation analysis (cross-dataset)...")
    datasets = adata_epi.obs['dataset'].unique()
    if len(datasets) < 2:
        print("  Only 1 dataset, skipping preservation")
        return

    # Pick two largest datasets
    ds_sizes = adata_epi.obs['dataset'].value_counts()
    ds1, ds2 = ds_sizes.index[0], ds_sizes.index[1]

    module_genes = module_df['gene'].tolist()
    avail_genes = [g for g in module_genes if g in adata_epi.var_names]

    results = {}
    for ds in [ds1, ds2]:
        mask = adata_epi.obs['dataset'] == ds
        X = adata_epi[mask][:, avail_genes].X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        # Subsample to ~500 cells for speed
        if X.shape[0] > 500:
            idx = np.random.choice(X.shape[0], 500, replace=False)
            X = X[idx]
        corr = np.corrcoef(X.T)
        adj = np.abs(corr) ** 6
        np.fill_diagonal(adj, 0)
        dist = 1 - adj
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)
        dist[~np.isfinite(dist)] = 1.0
        dist_condensed = squareform(dist, checks=False)
        dist_condensed[~np.isfinite(dist_condensed)] = 1.0
        Z_ds = linkage(dist_condensed, method='average')
        modules_ds = fcluster(Z_ds, t=0.85, criterion='distance')
        results[ds] = dict(zip(avail_genes, modules_ds))

    # Check preservation: for each module in reference, what fraction of genes
    # co-cluster in the test dataset
    preservation_scores = []
    for mod in module_df['module'].unique():
        genes_in_mod = module_df[module_df['module'] == mod]['gene'].tolist()
        genes_in_mod = [g for g in genes_in_mod if g in results[ds2]]
        if len(genes_in_mod) < 5:
            continue
        # In ds2, check how many pairs from this module are in the same cluster
        ds2_labels = [results[ds2][g] for g in genes_in_mod]
        from collections import Counter
        label_counts = Counter(ds2_labels)
        max_overlap = label_counts.most_common(1)[0][1]
        preservation = max_overlap / len(genes_in_mod)
        preservation_scores.append({'module': mod, 'n_genes': len(genes_in_mod),
                                    'preservation': preservation})

    pres_df = pd.DataFrame(preservation_scores)
    if len(pres_df) > 0:
        mean_pres = pres_df['preservation'].mean()
        well_preserved = (pres_df['preservation'] > 0.5).sum()
        print(f"  Mean preservation: {mean_pres:.3f}")
        print(f"  Well-preserved modules (>50%): {well_preserved}/{len(pres_df)}")
        pres_df.to_csv(f"{BASE}/results/wgcna_module_preservation.csv", index=False)
    return pres_df


def run_decoupler_tf(adata_epi):
    """TF activity inference with decoupler + Dorothea (confidence A/B only)."""
    print("  Running decoupler TF activity (Dorothea A/B)...")
    net = dc.op.dorothea(organism='human', levels=['A', 'B'])
    print(f"  Dorothea regulons: {net['source'].nunique()} TFs, {len(net)} edges")

    dc.mt.ulm(data=adata_epi, net=net, verbose=False)

    if 'score_ulm' in adata_epi.obsm:
        tf_act = pd.DataFrame(adata_epi.obsm['score_ulm'], index=adata_epi.obs_names)
    elif 'ulm_estimate' in adata_epi.obsm:
        tf_act = pd.DataFrame(adata_epi.obsm['ulm_estimate'], index=adata_epi.obs_names)
    else:
        print("  WARNING: ULM results not found in obsm, skipping TF analysis")
        return pd.DataFrame(), net
    return tf_act, net


def differential_tf_pseudobulk(adata_epi, tf_act):
    """Differential TF activity: HighRisk vs LowRisk at sample-level."""
    print("  Differential TF activity (sample-level pseudobulk)...")
    if tf_act.empty:
        print("  Skipping (no TF activity data)")
        return pd.DataFrame()
    if 'transition_risk' not in adata_epi.obs.columns:
        print("  WARNING: No transition_risk, using stage as proxy")
        return pd.DataFrame()

    # Pseudobulk TF activity per sample
    samples = adata_epi.obs['sample_id'].unique()
    pb_tf = []
    pb_risk = []
    for sid in samples:
        mask = adata_epi.obs['sample_id'] == sid
        if mask.sum() < 10:
            continue
        mean_tf = tf_act.loc[mask].mean(axis=0)
        mean_risk = adata_epi.obs.loc[mask, 'transition_risk'].mean()
        pb_tf.append(mean_tf)
        pb_risk.append(mean_risk)

    if len(pb_tf) < 6:
        print("  Too few samples for differential TF")
        return pd.DataFrame()

    pb_tf_df = pd.DataFrame(pb_tf)
    pb_risk_arr = np.array(pb_risk)

    # Split by median risk
    high_mask = pb_risk_arr > np.median(pb_risk_arr)
    low_mask = ~high_mask

    tf_results = []
    for tf in pb_tf_df.columns:
        vals_h = pb_tf_df.loc[high_mask, tf].values
        vals_l = pb_tf_df.loc[low_mask, tf].values
        if len(vals_h) >= 3 and len(vals_l) >= 3:
            stat, p = mannwhitneyu(vals_h, vals_l, alternative='two-sided')
            tf_results.append({'TF': tf, 'diff_activity': vals_h.mean() - vals_l.mean(), 'pval': p})

    tf_df = pd.DataFrame(tf_results).sort_values('pval')
    if len(tf_df) > 0:
        from statsmodels.stats.multitest import multipletests
        tf_df['padj'] = multipletests(tf_df['pval'], method='fdr_bh')[1]
    return tf_df


def wgcna_tf_cross(hub_df, tf_df, net):
    """Cross: WGCNA hub genes ∩ active TF targets."""
    print("  WGCNA hub × TF targets cross...")
    if hub_df.empty or tf_df.empty:
        return pd.DataFrame()

    sig_tfs = tf_df[tf_df['padj'] < 0.05]['TF'].tolist() if 'padj' in tf_df.columns else []
    if not sig_tfs:
        sig_tfs = tf_df.head(10)['TF'].tolist()

    hub_genes_set = set(hub_df['gene'].tolist())
    cross_results = []
    for tf in sig_tfs:
        targets = set(net[net['source'] == tf]['target'].tolist())
        overlap = hub_genes_set & targets
        if overlap:
            cross_results.append({'TF': tf, 'n_hub_targets': len(overlap),
                                  'genes': ','.join(sorted(overlap))})

    cross_df = pd.DataFrame(cross_results)
    print(f"  TFs regulating hub genes: {len(cross_df)}")
    return cross_df


def cnmf_tf_cross(tf_df, net):
    """cNMF orphan programs × differential TF: Fisher exact test."""
    print("  cNMF × TF activity cross (Fisher exact)...")
    orphan_path = f"{BASE}/results/cnmf_orphan_programs.csv"
    program_path = f"{BASE}/results/cnmf_program_genes.csv"

    if not os.path.exists(orphan_path) or not os.path.exists(program_path):
        print("  cNMF results not available, skipping")
        return pd.DataFrame()

    orphan_df = pd.read_csv(orphan_path)
    orphans = orphan_df[orphan_df['is_orphan'] == True]['program'].tolist()
    if not orphans:
        print("  No orphan programs found")
        return pd.DataFrame()

    program_genes = pd.read_csv(program_path, index_col=0)
    sig_tfs = tf_df[tf_df['padj'] < 0.05]['TF'].tolist() if 'padj' in tf_df.columns else tf_df.head(10)['TF'].tolist()

    all_genes = set(program_genes.columns.tolist())
    results = []
    for prog_idx in orphans:
        prog_name = f"program_{prog_idx}"
        if prog_name not in program_genes.index:
            continue
        weights = program_genes.loc[prog_name]
        top50 = set(weights.nlargest(50).index.tolist())

        for tf in sig_tfs:
            targets = set(net[net['source'] == tf]['target'].tolist()) & all_genes
            if len(targets) < 5:
                continue
            # Fisher exact: program_genes ∩ TF_targets vs background
            a = len(top50 & targets)
            b = len(top50 - targets)
            c = len(targets - top50)
            d = len(all_genes - top50 - targets)
            if a >= 2:
                odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
                results.append({'program': prog_idx, 'TF': tf,
                                'overlap': a, 'odds_ratio': odds, 'pval': pval})

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values('pval')
        sig = result_df[result_df['pval'] < 0.05]
        print(f"  Significant TF-program associations: {len(sig)}")
    return result_df


def spatial_tf_validation():
    """Validate TF activity gradients in spatial data (CDX2, HNF4A, SNAI1)."""
    print("  Spatial TF activity validation...")
    key_tfs = ['CDX2', 'HNF4A', 'SNAI1']

    spatial_results = []
    for gp in ['GP1','GP2','GP3','GP4','GP5','GP6','GP7','GP8','GP9']:
        path = f"{SPATIAL_DIR}/{gp}"
        if not os.path.exists(path):
            continue
        try:
            sp = sc.read_visium(path)
            sp.var_names_make_unique()
            sc.pp.normalize_total(sp, target_sum=1e4)
            sc.pp.log1p(sp)

            # Run decoupler on spatial
            net = dc.op.dorothea(organism='human', levels=['A', 'B'])
            dc.run_ulm(mat=sp, net=net, source='source', target='target',
                       weight='weight', verbose=False, use_raw=False)
            tf_spatial = pd.DataFrame(sp.obsm['ulm_estimate'],
                                      index=sp.obs_names,
                                      columns=sp.uns['ulm_estimate_names'])

            # Moran's I for spatial autocorrelation of TF activity
            import squidpy as sq
            sq.gr.spatial_neighbors(sp)

            # Define IM region by CDX2 expression for direction check
            cdx2_expr = None
            if 'CDX2' in sp.var_names:
                cdx2_expr = sp[:, 'CDX2'].X
                if hasattr(cdx2_expr, 'toarray'):
                    cdx2_expr = cdx2_expr.toarray().flatten()
                else:
                    cdx2_expr = np.asarray(cdx2_expr).flatten()
                im_spots = cdx2_expr > np.percentile(cdx2_expr[cdx2_expr > 0], 50) if (cdx2_expr > 0).sum() > 10 else cdx2_expr > 0
                non_im_spots = ~im_spots

            for tf in key_tfs:
                if tf in tf_spatial.columns:
                    sp.obs[f'{tf}_activity'] = tf_spatial[tf].values
                    sq.gr.spatial_autocorr(sp, genes=[f'{tf}_activity'], mode='moran')
                    moran_i = sp.uns['moranI'].loc[f'{tf}_activity', 'I']
                    moran_p = sp.uns['moranI'].loc[f'{tf}_activity', 'pval_norm']

                    # Direction check: TF activity in IM vs non-IM spots
                    direction = np.nan
                    if cdx2_expr is not None and im_spots.sum() > 5 and non_im_spots.sum() > 5:
                        tf_vals = tf_spatial[tf].values
                        direction = tf_vals[im_spots].mean() - tf_vals[non_im_spots].mean()

                    spatial_results.append({'sample': gp, 'TF': tf,
                                           'moran_I': moran_i, 'pval': moran_p,
                                           'IM_vs_nonIM_diff': direction})
        except Exception as e:
            print(f"    {gp}: failed ({e})")

    spatial_df = pd.DataFrame(spatial_results)
    if len(spatial_df) > 0:
        for tf in key_tfs:
            tf_data = spatial_df[spatial_df['TF'] == tf]
            n_sig = (tf_data['pval'] < 0.05).sum()
            mean_I = tf_data['moran_I'].mean()
            mean_dir = tf_data['IM_vs_nonIM_diff'].mean()
            dir_str = "higher in IM" if mean_dir > 0 else "lower in IM"
            print(f"    {tf}: Moran's I={mean_I:.3f}, sig {n_sig}/{len(tf_data)}, {dir_str} ({mean_dir:.3f})")
    return spatial_df


def main():
    print("=" * 60)
    print("Step 6: hdWGCNA + TF Activity + Cross-analysis")
    print("=" * 60)

    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load and extract epithelial cells
    print("\n[1] Loading data...")
    adata = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
    epi_mask = adata.obs['is_epithelial']
    adata_epi = adata[epi_mask].copy()
    print(f"  Epithelial cells: {adata_epi.n_obs}")

    # [2] Build metacells in scVI latent space
    print("\n[2] Building metacells...")
    n_mc = min(300, adata_epi.n_obs // 50)
    expr_df, meta_df = build_metacells(adata_epi, n_metacells=n_mc)

    # [3] hdWGCNA on metacell expression
    print("\n[3] hdWGCNA on metacells...")
    module_df, module_trait_df, hub_df = run_hdwgcna(expr_df, meta_df)
    module_df.to_csv(f"{BASE}/results/wgcna_modules.csv", index=False)
    module_trait_df.to_csv(f"{BASE}/results/wgcna_module_trait.csv", index=False)
    if not hub_df.empty:
        hub_df.to_csv(f"{BASE}/results/wgcna_hub_genes.csv", index=False)

    # [3b] Module preservation across datasets
    print("\n[3b] Module preservation analysis...")
    module_preservation_analysis(adata_epi, module_df)

    # [4] decoupler TF activity
    print("\n[4] TF activity inference (Dorothea A/B)...")
    tf_act, net = run_decoupler_tf(adata_epi)

    # [5] Differential TF (sample-level pseudobulk)
    print("\n[5] Differential TF activity (pseudobulk)...")
    tf_df = differential_tf_pseudobulk(adata_epi, tf_act)
    if not tf_df.empty:
        tf_df.to_csv(f"{BASE}/results/tf_activity.csv", index=False)
        n_sig = (tf_df['padj'] < 0.05).sum() if 'padj' in tf_df.columns else 0
        print(f"  Significant TFs: {n_sig}")
        print(f"  Top TFs: {tf_df.head(5)['TF'].tolist()}")

    # [6] Cross: WGCNA hub ∩ TF targets
    print("\n[6] WGCNA hub × TF targets...")
    cross_df = wgcna_tf_cross(hub_df, tf_df, net)
    if not cross_df.empty:
        cross_df.to_csv(f"{BASE}/results/wgcna_tf_cross.csv", index=False)

    # [7] cNMF × TF cross (Fisher exact)
    print("\n[7] cNMF orphan × TF cross...")
    cnmf_tf_df = cnmf_tf_cross(tf_df, net)
    if not cnmf_tf_df.empty:
        cnmf_tf_df.to_csv(f"{BASE}/results/cnmf_tf_cross.csv", index=False)

    # [8] Spatial TF validation
    print("\n[8] Spatial TF activity validation...")
    spatial_tf_df = spatial_tf_validation()
    if not spatial_tf_df.empty:
        spatial_tf_df.to_csv(f"{BASE}/results/spatial_tf_validation.csv", index=False)

    print(f"\n{'='*60}")
    print("Step 6 COMPLETE")
    print(f"  Metacells: {len(meta_df)}")
    print(f"  WGCNA modules: {module_df['module'].nunique()}")
    print(f"  Hub genes: {len(hub_df)}")
    print(f"  Differential TFs: {(tf_df['padj'] < 0.05).sum() if not tf_df.empty and 'padj' in tf_df.columns else 0}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
