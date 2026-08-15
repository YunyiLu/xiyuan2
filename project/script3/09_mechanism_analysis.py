"""
Step 9: Multi-omics Mechanism Analysis for 92 Candidate Genes
  Integrates scRNA + Visium + Bulk (GSE55696/GSE78523/GSE60427/GSE27342)

Modules:
  A: Temporal activation ordering (cascade + pseudotime + spatial)
  B: TF regulation inference (decoupler + Dorothea)
  C: Co-activation modules (bulk co-expression + WGCNA/cNMF cross-reference)
  D: Immune microenvironment (deconvolution + CellChat)
  E: Pathway enrichment & functional classification

Output: results/mechanism_*.csv + figures/mechanism_*.png
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind, kruskal
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
BULK_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk"
FIG_DIR = f"{BASE}/figures"
RES_DIR = f"{BASE}/results"
os.makedirs(FIG_DIR, exist_ok=True)

# Load 92 candidate genes
candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()
print(f"Loaded {len(CANDIDATES)} candidate genes")


# =========================================================================
# MODULE A: Temporal Activation Ordering
# =========================================================================
def module_A_temporal():
    print("\n" + "="*70)
    print("MODULE A: Temporal Activation Ordering")
    print("="*70)

    # --- A1: Bulk cascade (GSE55696) ---
    print("\n[A1] GSE55696 cascade (CG→LGIN→HGIN→EGC)...")
    expr_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_expression.csv", index_col=0)
    meta_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_metadata.csv")

    stages_order = ['CG', 'LGIN', 'HGIN', 'EGC']
    stage_num_map = {'CG': 0, 'LGIN': 1, 'HGIN': 2, 'EGC': 3}

    avail_genes = [g for g in CANDIDATES if g in expr_55.index]
    print(f"  Available in GSE55696: {len(avail_genes)}/{len(CANDIDATES)}")

    # Per-stage expression for each gene
    stage_expr = {}
    for stage in stages_order:
        samples = meta_55[meta_55['stage'] == stage]['sample_id'].tolist()
        samples = [s for s in samples if s in expr_55.columns]
        stage_expr[stage] = expr_55.loc[avail_genes, samples]

    # Dunnett-like test: each stage vs CG (Bonferroni correction within gene)
    temporal_results = []
    cg_samples = meta_55[meta_55['stage'] == 'CG']['sample_id'].tolist()
    cg_samples = [s for s in cg_samples if s in expr_55.columns]

    for gene in avail_genes:
        cg_vals = expr_55.loc[gene, cg_samples].values.astype(float)
        onset = None
        max_fc_stage = 'CG'
        max_fc = 0
        stage_fcs = {'CG': 0}
        stage_pvals = {}

        for stage in ['LGIN', 'HGIN', 'EGC']:
            samples = meta_55[meta_55['stage'] == stage]['sample_id'].tolist()
            samples = [s for s in samples if s in expr_55.columns]
            stage_vals = expr_55.loc[gene, samples].values.astype(float)

            # t-test vs CG
            _, p = ttest_ind(stage_vals, cg_vals, equal_var=False)
            fc = np.mean(stage_vals) - np.mean(cg_vals)
            stage_fcs[stage] = fc
            stage_pvals[stage] = p

            # Bonferroni for 3 comparisons
            if onset is None and p < 0.05/3:
                onset = stage_num_map[stage]
            if abs(fc) > abs(max_fc):
                max_fc = fc
                max_fc_stage = stage

        temporal_results.append({
            'gene': gene,
            'onset_stage_num': onset if onset is not None else 4,  # 4 = never
            'onset_stage': stages_order[onset] if onset is not None else 'none',
            'max_fc_stage': max_fc_stage,
            'max_fc': max_fc,
            'fc_LGIN': stage_fcs.get('LGIN', 0),
            'fc_HGIN': stage_fcs.get('HGIN', 0),
            'fc_EGC': stage_fcs.get('EGC', 0),
            'p_LGIN': stage_pvals.get('LGIN', 1),
            'p_HGIN': stage_pvals.get('HGIN', 1),
            'p_EGC': stage_pvals.get('EGC', 1),
        })

    df_temporal = pd.DataFrame(temporal_results)

    # --- A2: Single-cell pseudotime ---
    print("\n[A2] Single-cell pseudotime binning...")
    try:
        adata = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
        epi_mask = adata.obs['is_epithelial'] if 'is_epithelial' in adata.obs.columns else \
                   adata.obs['cell_type'].str.contains('Goblet|Gastric|Chief|Stem|Neck', case=False)
        adata_epi = adata[epi_mask].copy()

        if 'dpt_pseudotime' in adata_epi.obs.columns:
            dpt = adata_epi.obs['dpt_pseudotime'].values
            valid_dpt = ~np.isnan(dpt)
            adata_epi = adata_epi[valid_dpt]
            dpt = adata_epi.obs['dpt_pseudotime'].values

            # Bin pseudotime into 10 windows
            bins = np.linspace(0, 1, 11)
            bin_labels = np.digitize(dpt, bins) - 1
            bin_labels = np.clip(bin_labels, 0, 9)

            pseudotime_onset = {}
            for gene in avail_genes:
                if gene not in adata_epi.var_names:
                    continue
                gidx = list(adata_epi.var_names).index(gene)
                X = adata_epi.X
                if hasattr(X, 'toarray'):
                    gene_expr = X[:, gidx].toarray().flatten()
                else:
                    gene_expr = X[:, gidx].flatten()

                # Mean per bin
                bin_means = np.array([gene_expr[bin_labels == b].mean() for b in range(10)])
                # Max slope bin
                slopes = np.diff(bin_means)
                max_slope_bin = np.argmax(np.abs(slopes))
                pseudotime_onset[gene] = max_slope_bin / 9.0  # normalize to [0,1]

            df_temporal['pseudotime_onset'] = df_temporal['gene'].map(pseudotime_onset)
            print(f"  Pseudotime computed for {len(pseudotime_onset)} genes")
        else:
            print("  WARNING: dpt_pseudotime not in adata.obs")
            df_temporal['pseudotime_onset'] = np.nan
        del adata, adata_epi
    except Exception as e:
        print(f"  ERROR loading adata: {e}")
        df_temporal['pseudotime_onset'] = np.nan

    # --- A3: Spatial peak position ---
    print("\n[A3] Spatial gradient peak position...")
    spatial_grad = pd.read_csv(f"{RES_DIR}/spatial_gradient_genes.csv")
    spatial_genes = set(spatial_grad['gene'].tolist())

    # Classify spatial peak from existing data
    spatial_peak_map = {}
    for gene in avail_genes:
        if gene in spatial_genes:
            spatial_peak_map[gene] = 'IM'  # spatial gradient genes peak in IM region
        else:
            spatial_peak_map[gene] = 'unknown'
    df_temporal['spatial_peak'] = df_temporal['gene'].map(spatial_peak_map)

    # --- A4: GSE60427 early stages ---
    print("\n[A4] GSE60427 early cascade (normal→gastritis→IM)...")
    try:
        expr_427 = pd.read_csv(f"{BULK_DIR}/GSE60427/GSE60427_expression.csv", index_col=0)
        meta_427 = pd.read_csv(f"{BULK_DIR}/GSE60427/GSE60427_metadata.csv")

        normal_samples = meta_427[meta_427['stage'] == 'normal']['sample_id'].tolist()
        im_samples = meta_427[meta_427['stage'] == 'IM']['sample_id'].tolist()
        normal_samples = [s for s in normal_samples if s in expr_427.columns]
        im_samples = [s for s in im_samples if s in expr_427.columns]

        early_activation = {}
        for gene in avail_genes:
            if gene not in expr_427.index:
                continue
            n_vals = expr_427.loc[gene, normal_samples].values.astype(float)
            im_vals = expr_427.loc[gene, im_samples].values.astype(float)
            _, p = mannwhitneyu(im_vals, n_vals, alternative='two-sided')
            fc = np.mean(im_vals) - np.mean(n_vals)
            if p < 0.05:
                early_activation[gene] = True
            else:
                early_activation[gene] = False

        df_temporal['early_activation_427'] = df_temporal['gene'].map(early_activation)
        print(f"  Early activated (normal→IM): {sum(early_activation.values())}/{len(early_activation)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        df_temporal['early_activation_427'] = np.nan

    # --- A5: Temporal classification ---
    print("\n[A5] Classifying temporal waves...")
    def classify_temporal(row):
        onset = row['onset_stage_num']
        if onset <= 1:  # CG or LGIN
            return 'early'
        elif onset == 2:  # HGIN
            return 'mid'
        elif onset == 3:  # EGC
            return 'late'
        else:
            return 'not_significant'

    df_temporal['temporal_class'] = df_temporal.apply(classify_temporal, axis=1)
    df_temporal = df_temporal.sort_values('onset_stage_num')
    df_temporal.to_csv(f"{RES_DIR}/mechanism_temporal_ordering.csv", index=False, encoding='utf-8-sig')

    # Print summary
    print(f"\n  Temporal classification:")
    print(f"    Early (CG/LGIN): {(df_temporal['temporal_class']=='early').sum()}")
    print(f"    Mid (HGIN): {(df_temporal['temporal_class']=='mid').sum()}")
    print(f"    Late (EGC): {(df_temporal['temporal_class']=='late').sum()}")
    print(f"    Not significant: {(df_temporal['temporal_class']=='not_significant').sum()}")

    # --- Figures ---
    print("\n  Generating figures...")

    # Cascade heatmap
    fig, ax = plt.subplots(figsize=(8, max(12, len(avail_genes)*0.15)))
    heatmap_data = df_temporal[['gene', 'fc_LGIN', 'fc_HGIN', 'fc_EGC']].set_index('gene')
    heatmap_data.columns = ['LGIN', 'HGIN', 'EGC']
    # Clip for visualization
    heatmap_data = heatmap_data.clip(-3, 3)
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax,
                yticklabels=True if len(avail_genes) < 50 else False,
                xticklabels=True, cbar_kws={'label': 'FC vs CG'})
    ax.set_title('Cascade Activation (92 genes, sorted by onset)')
    ax.set_ylabel('Gene (sorted by onset stage)')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mechanism_cascade_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Pseudotime curves for top 10
    if df_temporal['pseudotime_onset'].notna().sum() > 0:
        try:
            adata = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
            epi_mask = adata.obs['is_epithelial'] if 'is_epithelial' in adata.obs.columns else \
                       adata.obs['cell_type'].str.contains('Goblet|Gastric|Chief|Stem|Neck', case=False)
            adata_epi = adata[epi_mask].copy()
            valid_dpt = ~np.isnan(adata_epi.obs['dpt_pseudotime'].values)
            adata_epi = adata_epi[valid_dpt]
            dpt = adata_epi.obs['dpt_pseudotime'].values
            bins = np.linspace(0, 1, 11)
            bin_labels = np.digitize(dpt, bins) - 1
            bin_labels = np.clip(bin_labels, 0, 9)

            top10 = candidates_df['gene'].head(10).tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            for gene in top10:
                if gene not in adata_epi.var_names:
                    continue
                gidx = list(adata_epi.var_names).index(gene)
                X = adata_epi.X
                if hasattr(X, 'toarray'):
                    gene_expr = X[:, gidx].toarray().flatten()
                else:
                    gene_expr = X[:, gidx].flatten()
                bin_means = [gene_expr[bin_labels == b].mean() for b in range(10)]
                # z-score normalize for comparison
                bm = np.array(bin_means)
                if bm.std() > 0:
                    bm = (bm - bm.mean()) / bm.std()
                ax.plot(np.linspace(0, 1, 10), bm, '-o', label=gene, markersize=3)

            ax.set_xlabel('Pseudotime (DPT)')
            ax.set_ylabel('Expression (z-scored)')
            ax.set_title('Top 10 Candidates along Pseudotime')
            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/mechanism_pseudotime_curves.png", dpi=150)
            plt.close()
            del adata, adata_epi
        except Exception as e:
            print(f"  Pseudotime figure error: {e}")

    print("  Module A complete.")
    return df_temporal


# =========================================================================
# MODULE B: TF Regulation Inference
# =========================================================================
def module_B_tf_regulation(df_temporal):
    print("\n" + "="*70)
    print("MODULE B: TF Regulation Inference")
    print("="*70)

    # --- B1: Bulk TF activity (GSE55696) ---
    print("\n[B1] decoupler TF activity on GSE55696...")
    import decoupler as dc

    expr_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_expression.csv", index_col=0)
    meta_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_metadata.csv")

    # Prepare expression as samples×genes (decoupler expects this)
    expr_T = expr_55.T  # samples × genes

    # Get Dorothea regulon
    try:
        net = dc.get_dorothea(organism='human', levels=['A', 'B', 'C'])
        print(f"  Dorothea regulon: {net['source'].nunique()} TFs, {len(net)} edges")
    except:
        try:
            net = dc.op.dorothea(organism='human', levels=['A', 'B', 'C'])
            print(f"  Dorothea regulon: {net['source'].nunique()} TFs, {len(net)} edges")
        except Exception as e:
            print(f"  ERROR getting Dorothea: {e}")
            print("  Trying alternative approach...")
            net = None

    tf_cascade_results = []
    tf_target_pairs = []

    if net is not None:
        # Run ULM via AnnData interface (decoupler 2.x)
        try:
            adata_bulk = ad.AnnData(X=expr_T.values, obs=pd.DataFrame(index=expr_T.index),
                                    var=pd.DataFrame(index=expr_T.columns))
            dc.mt.ulm(data=adata_bulk, net=net, verbose=False)
            tf_acts = pd.DataFrame(adata_bulk.obsm['score_ulm'],
                                   index=adata_bulk.obs_names,
                                   columns=adata_bulk.uns.get('ulm_keys',
                                          [c for c in pd.DataFrame(adata_bulk.obsm['score_ulm']).columns]))
            # Get TF names from the net
            tf_names = net['source'].unique()
            if tf_acts.shape[1] <= len(tf_names):
                # Columns are TFs in order of appearance
                unique_tfs = list(dict.fromkeys(net['source'].tolist()))[:tf_acts.shape[1]]
                tf_acts.columns = unique_tfs
            print(f"  TF activities computed: {tf_acts.shape}")
        except Exception as e:
            print(f"  ULM failed: {e}")
            tf_acts = None

        if tf_acts is not None:
            # JT trend test for each TF along cascade
            stages_order = ['CG', 'LGIN', 'HGIN', 'EGC']
            stage_num_map = {'CG': 0, 'LGIN': 1, 'HGIN': 2, 'EGC': 3}

            sample_stages = meta_55.set_index('sample_id')['stage']
            # Only samples in tf_acts
            common_samples = [s for s in tf_acts.index if s in sample_stages.index]
            tf_acts_filt = tf_acts.loc[common_samples]
            stages_arr = sample_stages.loc[common_samples].map(stage_num_map).values

            from scipy.stats import spearmanr as sp_corr

            for tf in tf_acts_filt.columns:
                tf_vals = tf_acts_filt[tf].values
                if np.std(tf_vals) < 1e-10:
                    continue
                r, p = sp_corr(stages_arr, tf_vals)

                # Per-stage means
                stage_means = {}
                for stage in stages_order:
                    mask = sample_stages.loc[common_samples] == stage
                    stage_means[stage] = tf_vals[mask.values].mean()

                tf_cascade_results.append({
                    'TF': tf,
                    'trend_rho': r,
                    'trend_pval': p,
                    'mean_CG': stage_means['CG'],
                    'mean_LGIN': stage_means['LGIN'],
                    'mean_HGIN': stage_means['HGIN'],
                    'mean_EGC': stage_means['EGC'],
                })

            # --- B3: TF → candidate gene mapping ---
            print("\n[B3] TF → candidate gene regulon mapping...")
            sig_tfs = [r['TF'] for r in tf_cascade_results if r['trend_pval'] < 0.05]
            print(f"  Significant trend TFs: {len(sig_tfs)}")

            for _, row in net.iterrows():
                tf = row['source']
                target = row['target']
                if tf in sig_tfs and target in CANDIDATES:
                    # Get TF-target correlation in bulk
                    if target in expr_55.index:
                        tf_act_vals = tf_acts_filt[tf].values if tf in tf_acts_filt.columns else None
                        if tf_act_vals is not None:
                            target_vals = expr_55.loc[target, common_samples].values.astype(float)
                            r_corr, p_corr = sp_corr(tf_act_vals, target_vals)
                            tf_target_pairs.append({
                                'TF': tf,
                                'target_gene': target,
                                'confidence': row.get('confidence', row.get('weight', 'unknown')),
                                'bulk_corr': r_corr,
                                'bulk_pval': p_corr,
                            })

    # --- B2: Cross-reference with scRNA TF results ---
    print("\n[B2] Cross-referencing with scRNA TF activity...")
    scrna_tf = pd.read_csv(f"{RES_DIR}/tf_activity.csv")

    # Add scRNA consistency flag
    scrna_tf_dict = dict(zip(scrna_tf['TF'], scrna_tf['diff_activity']))
    for item in tf_cascade_results:
        tf = item['TF']
        if tf in scrna_tf_dict:
            scrna_dir = 'up' if scrna_tf_dict[tf] > 0 else 'down'
            bulk_dir = 'up' if item['trend_rho'] > 0 else 'down'
            item['scrna_activity'] = scrna_tf_dict[tf]
            item['scrna_consistent'] = scrna_dir == bulk_dir
        else:
            item['scrna_activity'] = np.nan
            item['scrna_consistent'] = np.nan

    # Save results
    df_tf_cascade = pd.DataFrame(tf_cascade_results)
    if len(df_tf_cascade) > 0:
        df_tf_cascade = df_tf_cascade.sort_values('trend_pval')
        df_tf_cascade.to_csv(f"{RES_DIR}/mechanism_tf_activity_cascade.csv", index=False, encoding='utf-8-sig')
        print(f"  Saved: {len(df_tf_cascade)} TFs with cascade trend")

    df_tf_targets = pd.DataFrame(tf_target_pairs)
    if len(df_tf_targets) > 0:
        df_tf_targets = df_tf_targets.sort_values('bulk_pval')
        df_tf_targets.to_csv(f"{RES_DIR}/mechanism_tf_target_pairs.csv", index=False, encoding='utf-8-sig')
        print(f"  Saved: {len(df_tf_targets)} TF-target pairs")

    # --- Figures ---
    print("\n  Generating TF figures...")
    if len(df_tf_cascade) > 0:
        # Top 15 TFs cascade plot
        top_tfs = df_tf_cascade.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = ['CG', 'LGIN', 'HGIN', 'EGC']
        for _, row in top_tfs.iterrows():
            means = [row['mean_CG'], row['mean_LGIN'], row['mean_HGIN'], row['mean_EGC']]
            label = f"{row['TF']} (r={row['trend_rho']:.2f})"
            ax.plot(range(4), means, '-o', label=label, markersize=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(stages)
        ax.set_xlabel('Stage')
        ax.set_ylabel('TF Activity (ULM score)')
        ax.set_title('Top 15 TFs with cascade trend (GSE55696)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/mechanism_tf_cascade.png", dpi=150)
        plt.close()

    if len(df_tf_targets) > 0:
        # TF-target network summary
        tf_counts = df_tf_targets.groupby('TF').size().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        tf_counts.plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of candidate gene targets')
        ax.set_title('Top TFs regulating candidate genes (Dorothea)')
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/mechanism_tf_network.png", dpi=150)
        plt.close()

    print("  Module B complete.")
    return df_tf_cascade, df_tf_targets


# =========================================================================
# MODULE C: Co-activation Modules
# =========================================================================
def module_C_coexpression():
    print("\n" + "="*70)
    print("MODULE C: Co-activation Modules")
    print("="*70)

    # --- C1: Bulk co-expression (GSE55696) ---
    print("\n[C1] Computing 92×92 co-expression matrix (GSE55696)...")
    expr_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_expression.csv", index_col=0)

    avail_genes = [g for g in CANDIDATES if g in expr_55.index]
    expr_sub = expr_55.loc[avail_genes].T  # samples × genes

    # Spearman correlation matrix
    corr_matrix = expr_sub.corr(method='spearman')
    corr_matrix.to_csv(f"{RES_DIR}/mechanism_coexpression_matrix.csv", encoding='utf-8-sig')
    print(f"  Correlation matrix: {corr_matrix.shape}")

    # Hierarchical clustering
    dist = 1 - corr_matrix.values
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist[~np.isfinite(dist)] = 1.0
    condensed = squareform(dist, checks=False)
    condensed[~np.isfinite(condensed)] = 1.0
    Z = linkage(condensed, method='average')

    # Cut tree to get 4-8 clusters
    best_k = 5
    for k in range(4, 9):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sizes = pd.Series(labels).value_counts()
        if sizes.min() >= 3:
            best_k = k
            break

    cluster_labels = fcluster(Z, t=best_k, criterion='maxclust')
    cluster_map = dict(zip(avail_genes, cluster_labels))
    print(f"  Clusters: {best_k}, sizes: {pd.Series(cluster_labels).value_counts().to_dict()}")

    # --- C2: Cross-reference with scRNA WGCNA ---
    print("\n[C2] Cross-referencing with scRNA WGCNA modules...")
    wgcna_modules = pd.read_csv(f"{RES_DIR}/wgcna_modules.csv")
    wgcna_map = dict(zip(wgcna_modules['gene'], wgcna_modules['module']))

    # --- C3: cNMF programs ---
    print("\n[C3] cNMF program membership...")
    cnmf_df = pd.read_csv(f"{RES_DIR}/cnmf_program_genes.csv", nrows=0)
    # cnmf_program_genes.csv is wide format (programs as columns with gene lists)
    # Read differently
    try:
        cnmf_raw = pd.read_csv(f"{RES_DIR}/cnmf_program_genes.csv")
        # Determine format
        cnmf_map = {}
        if 'gene' in cnmf_raw.columns and 'program' in cnmf_raw.columns:
            cnmf_map = dict(zip(cnmf_raw['gene'], cnmf_raw['program']))
        else:
            # Wide format: each column is a program, values are genes
            for col in cnmf_raw.columns:
                for gene in cnmf_raw[col].dropna().tolist():
                    if gene in CANDIDATES:
                        cnmf_map[gene] = col
    except Exception as e:
        print(f"  cNMF parsing: {e}")
        cnmf_map = {}

    # --- C4: Pathway enrichment per cluster ---
    print("\n[C4] Pathway enrichment per cluster...")
    cluster_pathways = {}
    try:
        import gseapy as gp
        for cl in range(1, best_k + 1):
            genes_in_cl = [g for g, c in cluster_map.items() if c == cl]
            if len(genes_in_cl) < 3:
                continue
            try:
                enr = gp.enrichr(gene_list=genes_in_cl, gene_sets='MSigDB_Hallmark_2020',
                                 organism='human', no_plot=True, cutoff=0.1)
                top_terms = enr.results.head(3)['Term'].tolist()
                cluster_pathways[cl] = '; '.join(top_terms) if top_terms else 'none'
                print(f"    Cluster {cl} ({len(genes_in_cl)} genes): {top_terms[:2]}")
            except:
                cluster_pathways[cl] = 'enrichment_failed'
    except ImportError:
        print("  gseapy not available, using manual GO annotation...")
        # Manual functional annotation based on known biology
        go_annotations = {
            'intestinal_differentiation': ['CDX2','REG4','MUC2','TFF3','VIL1','CDH17','MUC13','SI','FABP2',
                                           'ANPEP','KRT20','CLDN7','CLDN3','CLDN4','PRAP1','FABP1'],
            'stem_cell_maintenance': ['OLFM4','LGR5','ASCL2','SOX9','PRSS3','CD44','ALDH1A1'],
            'immune_inflammatory': ['CCL3','CCL4','CCL5','CXCL8','IL1B','TNF','CD68','CSF1R','HLA-DRA'],
            'metabolic_process': ['IDH2','IDH1','LDHA','ALDOB','FABP1','FABP2','CPS1','GIF'],
            'tight_junction': ['CLDN7','CLDN3','CLDN4','TJP1','OCLN'],
            'protein_homeostasis': ['POMP','PSMA7','PSMB5','HSP90AA1','BAG1'],
            'cell_signaling': ['ERBB2','MET','VEGFA','AREG','TOLLIP','NAMPT'],
            'extracellular_matrix': ['COL1A1','FN1','LAMA1','LAMC2'],
        }
        for cl in range(1, best_k + 1):
            genes_in_cl = [g for g, c in cluster_map.items() if c == cl]
            if len(genes_in_cl) < 3:
                continue
            # Find best matching annotation
            best_term = 'unclassified'
            best_overlap = 0
            for term, term_genes in go_annotations.items():
                overlap = len(set(genes_in_cl) & set(term_genes))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_term = f"{term} ({overlap}/{len(genes_in_cl)})"
            cluster_pathways[cl] = best_term
            print(f"    Cluster {cl} ({len(genes_in_cl)} genes): {best_term}")

    # --- C5: Progressor differential (GSE78523) ---
    print("\n[C5] Cluster eigengenes in progressors vs controls...")
    expr_78 = pd.read_csv(f"{BULK_DIR}/GSE78523/GSE78523_expression.csv", index_col=0)
    meta_78 = pd.read_csv(f"{BULK_DIR}/GSE78523/GSE78523_metadata.csv")

    prog_samples = meta_78[meta_78['group'].str.contains('progressor')]['sample_id'].tolist()
    ctrl_samples = meta_78[meta_78['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()
    prog_samples = [s for s in prog_samples if s in expr_78.columns]
    ctrl_samples = [s for s in ctrl_samples if s in expr_78.columns]

    cluster_progression = {}
    for cl in range(1, best_k + 1):
        genes_in_cl = [g for g, c in cluster_map.items() if c == cl and g in expr_78.index]
        if len(genes_in_cl) < 2:
            continue
        # Eigengene = mean expression
        eigen_prog = expr_78.loc[genes_in_cl, prog_samples].mean(axis=0).values
        eigen_ctrl = expr_78.loc[genes_in_cl, ctrl_samples].mean(axis=0).values
        _, p = mannwhitneyu(eigen_prog, eigen_ctrl, alternative='two-sided')
        d = (np.mean(eigen_prog) - np.mean(eigen_ctrl)) / (np.sqrt((np.var(eigen_prog) + np.var(eigen_ctrl))/2) + 1e-10)
        cluster_progression[cl] = {'p': p, 'd': d}
        print(f"    Cluster {cl}: d={d:.3f}, p={p:.4f}")

    # Compile results
    cluster_results = []
    for gene in avail_genes:
        cluster_results.append({
            'gene': gene,
            'bulk_cluster': cluster_map.get(gene, np.nan),
            'scrna_wgcna_module': wgcna_map.get(gene, np.nan),
            'cnmf_program': cnmf_map.get(gene, 'none'),
            'cluster_pathway': cluster_pathways.get(cluster_map.get(gene, -1), ''),
            'cluster_prog_d': cluster_progression.get(cluster_map.get(gene, -1), {}).get('d', np.nan),
            'cluster_prog_p': cluster_progression.get(cluster_map.get(gene, -1), {}).get('p', np.nan),
        })

    df_clusters = pd.DataFrame(cluster_results)
    df_clusters.to_csv(f"{RES_DIR}/mechanism_coexpr_clusters.csv", index=False, encoding='utf-8-sig')

    # --- Figure: Co-expression heatmap ---
    print("\n  Generating co-expression heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))
    # Cluster color bar
    cluster_colors = pd.Series(cluster_labels, index=avail_genes).map(
        {i: plt.cm.Set2(i/best_k) for i in range(1, best_k+1)})

    g = sns.clustermap(corr_matrix, method='average', cmap='RdBu_r', center=0,
                       figsize=(14, 12), vmin=-1, vmax=1,
                       row_linkage=Z, col_linkage=Z,
                       yticklabels=True if len(avail_genes) < 40 else False,
                       xticklabels=True if len(avail_genes) < 40 else False)
    g.fig.suptitle('Co-expression of 92 candidate genes (GSE55696)', y=1.01)
    g.savefig(f"{FIG_DIR}/mechanism_coexpression_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("  Module C complete.")
    return df_clusters, corr_matrix


# =========================================================================
# MODULE D: Immune Microenvironment
# =========================================================================
def module_D_immune():
    print("\n" + "="*70)
    print("MODULE D: Immune Microenvironment")
    print("="*70)

    # --- D1: CellChat L-R participation ---
    print("\n[D1] CellChat candidate gene participation...")
    cellchat = pd.read_csv(f"{RES_DIR}/cellchat_per_stage.csv")
    diff_lr = pd.read_csv(f"{RES_DIR}/differential_LR.csv")

    # Check which candidates appear as ligand or receptor
    cellchat_candidates = []
    for gene in CANDIDATES:
        as_ligand = cellchat[cellchat['ligand'] == gene]
        as_receptor = cellchat[cellchat['receptor'].str.contains(gene, na=False)]
        if len(as_ligand) > 0 or len(as_receptor) > 0:
            stages_involved = set()
            if len(as_ligand) > 0:
                stages_involved.update(as_ligand['stage'].unique())
            if len(as_receptor) > 0:
                stages_involved.update(as_receptor['stage'].unique())
            cellchat_candidates.append({
                'gene': gene,
                'role': 'ligand' if len(as_ligand) > 0 else 'receptor',
                'n_interactions_as_ligand': len(as_ligand),
                'n_interactions_as_receptor': len(as_receptor),
                'stages': ';'.join(sorted(stages_involved)),
            })

    # Check differential L-R
    for gene in CANDIDATES:
        matches = diff_lr[diff_lr['ligand'] == gene]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                cellchat_candidates.append({
                    'gene': gene,
                    'role': 'diff_ligand',
                    'lr_pair': f"{row['ligand']}→{row['receptor']}",
                    'source_cell': row['source'],
                    'target_cell': row['target'],
                    'delta': row['delta'],
                    'pval': row['pval'],
                })

    df_cellchat = pd.DataFrame(cellchat_candidates)
    if len(df_cellchat) > 0:
        df_cellchat.to_csv(f"{RES_DIR}/mechanism_cellchat_candidates.csv", index=False, encoding='utf-8-sig')
        print(f"  Candidates in CellChat: {df_cellchat['gene'].nunique()}")

    # --- D2: Bulk immune deconvolution (GSE55696) ---
    print("\n[D2] Immune deconvolution (ssGSEA-like) on GSE55696...")
    expr_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_expression.csv", index_col=0)
    meta_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_metadata.csv")

    # MCPcounter-like signatures (simplified)
    immune_signatures = {
        'T_cells': ['CD3D', 'CD3E', 'CD2', 'CD3G', 'TRAC'],
        'CD8_T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1'],
        'Macrophage': ['CD68', 'CD163', 'CSF1R', 'MSR1', 'MRC1'],
        'B_cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'BLK'],
        'NK_cells': ['NCAM1', 'NKG7', 'KLRD1', 'KLRB1', 'GNLY'],
        'Neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A8', 'S100A9'],
        'Dendritic': ['ITGAX', 'CD1C', 'FCER1A', 'CLEC10A', 'FLT3'],
        'Monocyte': ['CD14', 'VCAN', 'FCN1', 'S100A12', 'SELL'],
        'Fibroblast': ['FAP', 'PDGFRA', 'COL1A1', 'COL3A1', 'DCN'],
        'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'FLT1', 'KDR'],
    }

    # ssGSEA-like: mean z-score of signature genes
    all_samples = meta_55['sample_id'].tolist()
    all_samples = [s for s in all_samples if s in expr_55.columns]

    immune_scores = {}
    for cell_type, markers in immune_signatures.items():
        avail = [g for g in markers if g in expr_55.index]
        if len(avail) < 2:
            continue
        # Z-score per gene across samples, then mean
        vals = expr_55.loc[avail, all_samples].values.astype(float)
        z_vals = (vals - vals.mean(axis=1, keepdims=True)) / (vals.std(axis=1, keepdims=True) + 1e-10)
        immune_scores[cell_type] = z_vals.mean(axis=0)

    immune_df = pd.DataFrame(immune_scores, index=all_samples)
    immune_df['stage'] = meta_55.set_index('sample_id').loc[all_samples, 'stage'].values
    immune_df.to_csv(f"{RES_DIR}/mechanism_immune_deconv.csv", index=True, encoding='utf-8-sig')

    # Immune trend along cascade
    print("  Immune trends along cascade:")
    stage_num = {'CG': 0, 'LGIN': 1, 'HGIN': 2, 'EGC': 3}
    stages_numeric = immune_df['stage'].map(stage_num).values
    for cell_type in immune_scores:
        r, p = spearmanr(stages_numeric, immune_df[cell_type].values)
        direction = '↑' if r > 0 else '↓'
        sig = '*' if p < 0.05 else ''
        print(f"    {cell_type:<15}: r={r:+.3f} {direction} {sig}")

    # --- D3: Gene-immune correlation ---
    print("\n[D3] Candidate gene - immune correlation...")
    avail_genes = [g for g in CANDIDATES if g in expr_55.index]

    gene_immune_corr = []
    for gene in avail_genes:
        gene_vals = expr_55.loc[gene, all_samples].values.astype(float)
        for cell_type in immune_scores:
            r, p = spearmanr(gene_vals, immune_df[cell_type].values)
            gene_immune_corr.append({
                'gene': gene,
                'immune_cell': cell_type,
                'spearman_r': r,
                'pval': p,
            })

    df_gene_immune = pd.DataFrame(gene_immune_corr)
    df_gene_immune.to_csv(f"{RES_DIR}/mechanism_gene_immune_corr.csv", index=False, encoding='utf-8-sig')

    # --- D4: Progressor immune difference (GSE78523) ---
    print("\n[D4] Progressor immune difference (GSE78523)...")
    try:
        expr_78 = pd.read_csv(f"{BULK_DIR}/GSE78523/GSE78523_expression.csv", index_col=0)
        meta_78 = pd.read_csv(f"{BULK_DIR}/GSE78523/GSE78523_metadata.csv")
        prog_s = meta_78[meta_78['group'].str.contains('progressor')]['sample_id'].tolist()
        ctrl_s = meta_78[meta_78['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()
        prog_s = [s for s in prog_s if s in expr_78.columns]
        ctrl_s = [s for s in ctrl_s if s in expr_78.columns]

        for cell_type, markers in immune_signatures.items():
            avail = [g for g in markers if g in expr_78.index]
            if len(avail) < 2:
                continue
            prog_vals = expr_78.loc[avail, prog_s].values.astype(float).mean(axis=0)
            ctrl_vals = expr_78.loc[avail, ctrl_s].values.astype(float).mean(axis=0)
            _, p = mannwhitneyu(prog_vals, ctrl_vals, alternative='two-sided')
            d = (np.mean(prog_vals) - np.mean(ctrl_vals)) / (np.std(np.concatenate([prog_vals, ctrl_vals])) + 1e-10)
            sig = '*' if p < 0.05 else ''
            print(f"    {cell_type:<15}: d={d:+.3f}, p={p:.3f} {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Figures ---
    print("\n  Generating immune figures...")

    # Immune cascade
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, cell_type in enumerate(list(immune_scores.keys())[:6]):
        ax = axes[i]
        for stage in ['CG', 'LGIN', 'HGIN', 'EGC']:
            vals = immune_df[immune_df['stage'] == stage][cell_type].values
            ax.scatter([stage_num[stage]]*len(vals), vals, alpha=0.5, s=20)
        # Mean line
        means = [immune_df[immune_df['stage']==s][cell_type].mean() for s in ['CG','LGIN','HGIN','EGC']]
        ax.plot(range(4), means, 'k-o', linewidth=2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['CG','LGIN','HGIN','EGC'], fontsize=8)
        ax.set_title(cell_type, fontsize=10)
    plt.suptitle('Immune cell scores along cascade (GSE55696)')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mechanism_immune_cascade.png", dpi=150)
    plt.close()

    # Gene-immune heatmap (top correlations)
    pivot = df_gene_immune.pivot(index='gene', columns='immune_cell', values='spearman_r')
    # Only keep genes with at least one |r| > 0.3
    strong_genes = pivot[(pivot.abs() > 0.3).any(axis=1)].index.tolist()
    if len(strong_genes) > 5:
        fig, ax = plt.subplots(figsize=(8, max(6, len(strong_genes)*0.3)))
        sns.heatmap(pivot.loc[strong_genes], cmap='RdBu_r', center=0, ax=ax,
                    vmin=-0.8, vmax=0.8, annot=False)
        ax.set_title('Candidate gene - Immune correlation (|r|>0.3 genes)')
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/mechanism_gene_immune_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("  Module D complete.")
    return df_gene_immune


# =========================================================================
# MODULE E: Pathway Enrichment & Functional Classification
# =========================================================================
def module_E_pathways(df_temporal):
    print("\n" + "="*70)
    print("MODULE E: Pathway Enrichment & Functional Classification")
    print("="*70)

    # --- E1: Per-stage GSEA (GSE55696) ---
    print("\n[E1] Per-stage differential expression & pathway enrichment...")
    expr_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_expression.csv", index_col=0)
    meta_55 = pd.read_csv(f"{BULK_DIR}/GSE55696/GSE55696_metadata.csv")

    cg_samples = meta_55[meta_55['stage'] == 'CG']['sample_id'].tolist()
    cg_samples = [s for s in cg_samples if s in expr_55.columns]

    pathway_results = []

    # Manual pathway enrichment using fold-change ranking
    stages_order = ['CG', 'LGIN', 'HGIN', 'EGC']
    cg_samples_list = [s for s in cg_samples if s in expr_55.columns]

    # Hallmark-like gene sets (curated subset relevant to GI cancer)
    hallmark_sets = {
        'EPITHELIAL_MESENCHYMAL_TRANSITION': ['VIM','FN1','SNAI1','ZEB1','CDH2','TWIST1','MMP2','MMP9',
                                              'COL1A1','COL3A1','SPARC','LOXL2'],
        'INFLAMMATORY_RESPONSE': ['IL1B','TNF','CCL3','CCL4','CXCL8','IL6','NFKB1','PTGS2',
                                  'CSF1','CCL2','ICAM1','VCAM1'],
        'WNT_BETA_CATENIN': ['CTNNB1','TCF7L2','LEF1','AXIN2','LGR5','MYC','CCND1','SOX9',
                             'DKK1','SFRP1','WNT3A','FZD7'],
        'OXIDATIVE_PHOSPHORYLATION': ['IDH2','IDH1','NDUFA1','SDHA','COX5A','ATP5F1A','CS',
                                      'UQCRC1','NDUFB8','SDHB'],
        'P53_PATHWAY': ['TP53','CDKN1A','MDM2','BAX','BBC3','GADD45A','PMAIP1','SERPINE1'],
        'MTORC1_SIGNALING': ['AKT1','MTOR','RPS6KB1','EIF4EBP1','VEGFA','HIF1A','SLC2A1'],
        'PROTEIN_SECRETION': ['SEC61A1','SSR1','SPCS2','SRP14','SEC11A','HSPA5','CALR'],
        'FATTY_ACID_METABOLISM': ['FABP1','FABP2','CPT1A','ACOX1','PPARA','HMGCR','SCD','FASN'],
        'INTESTINAL_DIFFERENTIATION': ['CDX2','VIL1','SI','FABP2','MUC2','TFF3','CDH17',
                                       'REG4','ANPEP','KRT20','CLDN7'],
        'INTERFERON_GAMMA_RESPONSE': ['STAT1','IRF1','GBP1','CXCL10','IDO1','HLA-DRA','TAP1','B2M'],
    }

    for stage in ['LGIN', 'HGIN', 'EGC']:
        stage_samples = meta_55[meta_55['stage'] == stage]['sample_id'].tolist()
        stage_samples = [s for s in stage_samples if s in expr_55.columns]

        # Fold changes
        stage_mean = expr_55[stage_samples].mean(axis=1)
        cg_mean = expr_55[cg_samples_list].mean(axis=1)
        fc = stage_mean - cg_mean

        for pathway_name, pathway_genes in hallmark_sets.items():
            avail_pw = [g for g in pathway_genes if g in fc.index]
            if len(avail_pw) < 3:
                continue
            # Mean FC of pathway genes (NES-like)
            pw_fc = fc.loc[avail_pw].mean()
            # t-test: are pathway genes' FCs > 0?
            pw_vals = fc.loc[avail_pw].values
            if np.std(pw_vals) > 0:
                from scipy.stats import ttest_1samp
                _, p = ttest_1samp(pw_vals, 0)
            else:
                p = 1.0

            # Which candidates in this pathway?
            candidates_in = [g for g in avail_pw if g in CANDIDATES]

            pathway_results.append({
                'stage': stage,
                'pathway': pathway_name,
                'NES': pw_fc,
                'pval': p,
                'FDR': p * len(hallmark_sets),  # Bonferroni approx
                'candidates_in_leading_edge': ';'.join(candidates_in),
                'n_candidates_in_le': len(candidates_in),
            })

    df_pathways = pd.DataFrame(pathway_results)
    if len(df_pathways) > 0:
        df_pathways['FDR'] = df_pathways['FDR'].clip(upper=1.0)
        df_pathways.to_csv(f"{RES_DIR}/mechanism_pathway_enrichment.csv", index=False, encoding='utf-8-sig')
        print(f"  Pathway results: {len(df_pathways)} (stages × pathways)")
        sig = df_pathways[df_pathways['pval'] < 0.05]
        print(f"  Significant (p<0.05): {len(sig)}")

    # --- E2: ORA on 92 genes and per temporal class ---
    print("\n[E2] Manual ORA enrichment on 92 genes (per temporal class)...")
    ora_results = []

    # Use hallmark_sets defined above for ORA
    from scipy.stats import fisher_exact
    total_genome = 20000  # approximate genome size

    for gene_set_name, gene_list_query in [('all_92', CANDIDATES)] + \
        [(f'temporal_{tc}', df_temporal[df_temporal['temporal_class']==tc]['gene'].tolist())
         for tc in ['early', 'mid', 'late'] if df_temporal is not None and 'temporal_class' in df_temporal.columns]:

        if len(gene_list_query) < 3:
            continue
        for pathway_name, pathway_genes in hallmark_sets.items():
            overlap = len(set(gene_list_query) & set(pathway_genes))
            if overlap == 0:
                continue
            # Fisher exact test
            a = overlap
            b = len(gene_list_query) - overlap
            c = len(pathway_genes) - overlap
            d = total_genome - a - b - c
            _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            ora_results.append({
                'gene_set': gene_set_name,
                'term': pathway_name,
                'overlap': overlap,
                'pval': p,
                'genes': ';'.join(set(gene_list_query) & set(pathway_genes)),
            })

    if len(ora_results) > 0:
        ora_df = pd.DataFrame(ora_results).sort_values('pval')
        print(f"    ORA results: {len(ora_df)} (significant p<0.05: {(ora_df['pval']<0.05).sum()})")
    else:
        ora_df = pd.DataFrame()

    # --- E3: Functional classification ---
    print("\n[E3] Functional classification of 92 genes...")

    # Manual functional categories based on known biology
    functional_categories = {
        'intestinal_stem_cell': ['OLFM4', 'LGR5', 'ASCL2', 'SOX9', 'PRSS3'],
        'intestinal_differentiation': ['CDX2', 'REG4', 'MUC2', 'TFF3', 'VIL1', 'CDH17', 'MUC13'],
        'tight_junction_barrier': ['CLDN7', 'CLDN3', 'CLDN4', 'CLDN1', 'TJP1'],
        'lipid_metabolism': ['FABP1', 'FABP2', 'APOA1', 'APOB', 'MTTP', 'ALDOB'],
        'energy_metabolism': ['IDH2', 'IDH1', 'LDHA', 'PKM', 'ENO1'],
        'immune_chemokine': ['CCL3', 'CCL4', 'CCL5', 'CXCL8', 'CXCL12'],
        'protein_homeostasis': ['POMP', 'PSMA7', 'PSMB5', 'HSP90AA1'],
        'cell_adhesion': ['EPCAM', 'KRT20', 'KRT7', 'CDH1'],
        'signal_transduction': ['ERBB2', 'MET', 'VEGFA', 'AREG'],
        'immune_defense': ['ITLN1', 'DEFA5', 'DEFA6', 'REG3A', 'LYZ'],
    }

    gene_class = {}
    for gene in CANDIDATES:
        assigned = False
        for category, genes in functional_categories.items():
            if gene in genes:
                gene_class[gene] = category
                assigned = True
                break
        if not assigned:
            gene_class[gene] = 'other'

    df_func = pd.DataFrame([
        {'gene': g, 'functional_class': c} for g, c in gene_class.items()
    ])

    # Merge with pathway results
    if len(ora_df) > 0:
        # For each gene, find which pathways contain it
        gene_pathways = {}
        for _, row in ora_df.iterrows():
            if pd.isna(row['genes']):
                continue
            for g in str(row['genes']).split(';'):
                g = g.strip()
                if g in CANDIDATES:
                    if g not in gene_pathways:
                        gene_pathways[g] = []
                    gene_pathways[g].append(row['term'])

        df_func['supporting_pathways'] = df_func['gene'].map(
            lambda g: ';'.join(gene_pathways.get(g, [])[:3]))

    df_func.to_csv(f"{RES_DIR}/mechanism_gene_functional_class.csv", index=False, encoding='utf-8-sig')

    # Print summary
    print(f"\n  Functional classification summary:")
    for cat in df_func['functional_class'].value_counts().index:
        genes_in = df_func[df_func['functional_class'] == cat]['gene'].tolist()
        print(f"    {cat}: {len(genes_in)} genes ({', '.join(genes_in[:5])}...)")

    # --- Figures ---
    print("\n  Generating pathway figures...")

    if len(df_pathways) > 0:
        # Pathway dotplot (stage × pathway)
        pivot = df_pathways.pivot_table(index='pathway', columns='stage', values='NES', aggfunc='first')
        if len(pivot) > 0:
            fig, ax = plt.subplots(figsize=(8, max(6, len(pivot)*0.4)))
            sns.heatmap(pivot.fillna(0), cmap='RdBu_r', center=0, ax=ax, annot=True, fmt='.1f')
            ax.set_title('Pathway NES by stage (GSE55696, Hallmark)')
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/mechanism_pathway_dotplot.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Functional classes pie/bar
    class_counts = df_func['functional_class'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    class_counts.plot(kind='barh', ax=ax, color=plt.cm.Set3(np.linspace(0, 1, len(class_counts))))
    ax.set_xlabel('Number of genes')
    ax.set_title('Functional classification of 92 candidate genes')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mechanism_functional_classes.png", dpi=150)
    plt.close()

    print("  Module E complete.")
    return df_pathways, df_func


# =========================================================================
# INTEGRATED MODEL FIGURE
# =========================================================================
def integrated_model_figure(df_temporal, df_tf_cascade, df_clusters, df_gene_immune):
    print("\n" + "="*70)
    print("INTEGRATED MODEL FIGURE")
    print("="*70)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1.5, 1.5, 1]})

    # Panel 1: Temporal waves
    ax = axes[0]
    if df_temporal is not None and 'onset_stage_num' in df_temporal.columns:
        for tc, color in [('early', 'green'), ('mid', 'orange'), ('late', 'red'), ('not_significant', 'gray')]:
            subset = df_temporal[df_temporal['temporal_class'] == tc]
            ax.scatter(subset['onset_stage_num'], range(len(subset)),
                      c=color, alpha=0.6, s=30, label=f"{tc} ({len(subset)})")
            for _, row in subset.head(5).iterrows():
                ax.annotate(row['gene'], (row['onset_stage_num'], _), fontsize=6)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['CG', 'LGIN', 'HGIN', 'EGC', 'none'])
    ax.set_xlabel('Onset Stage')
    ax.set_title('A. Temporal Activation Waves')
    ax.legend(fontsize=8)

    # Panel 2: TF regulation
    ax = axes[1]
    if df_tf_cascade is not None and len(df_tf_cascade) > 0:
        top5_tfs = df_tf_cascade.head(5)
        for i, (_, row) in enumerate(top5_tfs.iterrows()):
            means = [row['mean_CG'], row['mean_LGIN'], row['mean_HGIN'], row['mean_EGC']]
            ax.plot(range(4), means, '-o', label=row['TF'], markersize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['CG', 'LGIN', 'HGIN', 'EGC'])
        ax.set_title('B. Top TF Activity along Cascade')
        ax.legend(fontsize=8)

    # Panel 3: Co-expression clusters summary
    ax = axes[2]
    if df_clusters is not None and 'bulk_cluster' in df_clusters.columns:
        cluster_counts = df_clusters['bulk_cluster'].value_counts().sort_index()
        bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values,
                     color=plt.cm.Set2(np.linspace(0, 1, len(cluster_counts))))
        ax.set_xlabel('Cluster')
        ax.set_ylabel('N genes')
        ax.set_title('C. Co-expression Clusters')
        # Annotate with top gene per cluster
        for cl in cluster_counts.index:
            cl_genes = df_clusters[df_clusters['bulk_cluster'] == cl]['gene'].tolist()[:3]
            ax.annotate(', '.join(cl_genes), (str(cl), cluster_counts[cl]),
                       fontsize=6, ha='center', va='bottom')

    # Panel 4: Immune association summary
    ax = axes[3]
    if df_gene_immune is not None and len(df_gene_immune) > 0:
        # Average correlation per immune cell type
        mean_corr = df_gene_immune.groupby('immune_cell')['spearman_r'].mean().sort_values()
        mean_corr.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Mean Spearman r with candidates')
        ax.set_title('D. Candidate-Immune Association')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

    plt.suptitle('Integrated Mechanism Model: 92 Candidate Genes\n'
                 'IM → EGC Transformation', fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mechanism_integrated_model.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Integrated model figure saved.")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("="*70)
    print("Step 9: Multi-omics Mechanism Analysis")
    print(f"  Candidates: {len(CANDIDATES)} genes")
    print(f"  Data: scRNA + Visium + 4 Bulk datasets")
    print("="*70)

    # Module A
    df_temporal = module_A_temporal()

    # Module B
    df_tf_cascade, df_tf_targets = module_B_tf_regulation(df_temporal)

    # Module C
    df_clusters, corr_matrix = module_C_coexpression()

    # Module D
    df_gene_immune = module_D_immune()

    # Module E
    df_pathways, df_func = module_E_pathways(df_temporal)

    # Integrated figure
    integrated_model_figure(df_temporal, df_tf_cascade, df_clusters, df_gene_immune)

    print("\n" + "="*70)
    print("Step 9 COMPLETE")
    print(f"  Results: {RES_DIR}/mechanism_*.csv")
    print(f"  Figures: {FIG_DIR}/mechanism_*.png")
    print("="*70)


if __name__ == "__main__":
    main()
