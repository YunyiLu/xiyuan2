"""
Step 2: scVI integration + cell type annotation + cNMF + spatial deconvolution.
Input: script3/data/adata_raw_unintegrated.h5ad
Output: script3/data/adata_integrated.h5ad, script3/data/spatial_deconv.h5ad
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.stats import spearmanr

sc.settings.verbosity = 1
BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"

# --- Marker definitions ---
MARKERS_LAYER1 = {
    "Gastric_mucous": ["MUC5AC", "TFF1", "GKN1"],
    "Chief_glandular": ["PGA3", "PGC", "MUC6"],
    "Enterocyte_IM": ["FABP1", "ALDOB", "CDX2"],
    "Goblet_IM": ["MUC2", "TFF3", "SPINK4"],
    "Stem_proliferative": ["MKI67", "TOP2A", "PCNA"],
    "T_NK": ["CD3D", "CD3E", "NKG7"],
    "B_Plasma": ["CD79A", "MS4A1", "JCHAIN"],
    "Macrophage": ["CD68", "CSF1R", "C1QA"],
    "Monocyte": ["CD14", "S100A8", "FCN1"],
    "DC": ["CLEC9A", "CD1C", "FCER1A"],
    "Mast": ["KIT", "TPSAB1"],
    "Fibroblast": ["DCN", "COL1A1", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
    "Pericyte": ["RGS5", "ACTA2", "PDGFRB"],
}

# Epithelial supertypes for grouping
EPITHELIAL_TYPES = {"Gastric_mucous", "Chief_glandular", "Enterocyte_IM",
                    "Goblet_IM", "Stem_proliferative"}

SCORES_LAYER2 = {
    "PMC_2_score": ["NAMPT", "ALDH1A1", "CD44", "SOX9", "OLFM4"],
    "PMC_P_score": ["AREG", "NAMPT", "PHLDA1", "ITGA2", "MYC"],
    "stemness_score": ["LGR5", "OLFM4", "SOX9", "ASCL2"],
    "proliferation_score": ["MKI67", "TOP2A", "PCNA", "CDK1"],
    "IM_score": ["CDX2", "MUC2", "TFF3", "VIL1"],
    "EMT_score": ["VIM", "SNAI1", "ZEB1", "CDH2"],
    "EGC_like_score": ["REG4", "CEACAM6", "MUC13", "CLDN3", "EPCAM", "KRT20", "ERBB2", "MET", "VEGFA"],
}



def annotate_clusters(adata, res=0.8):
    """Assign cell type per cluster based on mean marker expression."""
    key = f"leiden_{res}"
    cluster_labels = {}
    for cl in adata.obs[key].unique():
        mask = adata.obs[key] == cl
        scores = {}
        for ct, genes in MARKERS_LAYER1.items():
            avail = [g for g in genes if g in adata.var_names]
            if avail:
                expr = adata[mask][:, avail].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray()
                scores[ct] = np.mean(expr)
            else:
                scores[ct] = 0.0
        # Fibroblast/Pericyte require EPCAM-
        if "EPCAM" in adata.var_names:
            epcam = adata[mask][:, "EPCAM"].X
            if hasattr(epcam, 'toarray'):
                epcam = epcam.toarray()
            if np.mean(epcam) > 0.5:
                for stromal in ["Fibroblast", "Endothelial", "Pericyte"]:
                    scores[stromal] = 0.0
        best = max(scores, key=scores.get)
        # Low confidence → unassigned
        if scores[best] < 0.1:
            best = "Unassigned"
        cluster_labels[cl] = best
    adata.obs['celltype'] = adata.obs[key].map(cluster_labels).astype(str)
    adata.obs['is_epithelial'] = adata.obs['celltype'].isin(EPITHELIAL_TYPES)

    # Marker validation dotplot
    os.makedirs(f"{BASE}/results/figures", exist_ok=True)
    markers_avail = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS_LAYER1.items()}
    markers_avail = {k: v for k, v in markers_avail.items() if v}
    if markers_avail:
        sc.pl.dotplot(adata, markers_avail, groupby=key, save='_marker_validation.pdf', show=False)


def compute_layer2_scores(adata):
    """Compute research scores on epithelial cells only."""
    epi_mask = adata.obs['is_epithelial']
    for score_name, genes in SCORES_LAYER2.items():
        avail = [g for g in genes if g in adata.var_names]
        if avail:
            sc.tl.score_genes(adata, gene_list=avail, score_name=score_name)
            adata.obs.loc[~epi_mask, score_name] = np.nan

    # Incomplete IM: CDX2 AND MUC5AC co-expression
    if "CDX2" in adata.var_names and "MUC5AC" in adata.var_names:
        cdx2 = np.asarray(adata[:, "CDX2"].X.todense()).flatten() if hasattr(adata[:, "CDX2"].X, 'todense') else np.asarray(adata[:, "CDX2"].X).flatten()
        muc5ac = np.asarray(adata[:, "MUC5AC"].X.todense()).flatten() if hasattr(adata[:, "MUC5AC"].X, 'todense') else np.asarray(adata[:, "MUC5AC"].X).flatten()
        adata.obs['incomplete_IM_score'] = cdx2 * muc5ac
        adata.obs.loc[~epi_mask, 'incomplete_IM_score'] = np.nan

    # Complete IM: CDX2+ MUC2+ MUC5AC-
    if all(g in adata.var_names for g in ["CDX2", "MUC2", "MUC5AC"]):
        cdx2 = np.asarray(adata[:, "CDX2"].X.todense()).flatten() if hasattr(adata[:, "CDX2"].X, 'todense') else np.asarray(adata[:, "CDX2"].X).flatten()
        muc2 = np.asarray(adata[:, "MUC2"].X.todense()).flatten() if hasattr(adata[:, "MUC2"].X, 'todense') else np.asarray(adata[:, "MUC2"].X).flatten()
        muc5ac = np.asarray(adata[:, "MUC5AC"].X.todense()).flatten() if hasattr(adata[:, "MUC5AC"].X, 'todense') else np.asarray(adata[:, "MUC5AC"].X).flatten()
        adata.obs['complete_IM_score'] = cdx2 * muc2 * (1 - np.clip(muc5ac / (muc5ac.max() + 1e-6), 0, 1))
        adata.obs.loc[~epi_mask, 'complete_IM_score'] = np.nan


def high_risk_epithelial(adata):
    """Mark high-risk epithelial state: >=3/6 dimensions support."""
    epi_mask = adata.obs['is_epithelial']
    dims = []
    for s in ['proliferation_score', 'EGC_like_score', 'PMC_P_score',
              'stemness_score', 'incomplete_IM_score', 'cnv_score']:
        if s in adata.obs.columns:
            vals = adata.obs[s].copy()
            thresh = vals[epi_mask].quantile(0.75)
            dims.append((vals > thresh).astype(int))
    n_dims = len(dims)
    if n_dims >= 3:
        support = sum(dims)
        adata.obs['high_risk_epithelial'] = (support >= 3) & epi_mask
    else:
        adata.obs['high_risk_epithelial'] = False
    n_hr = adata.obs['high_risk_epithelial'].sum()
    print(f"  High-risk epithelial cells: {n_hr} ({n_hr/epi_mask.sum()*100:.1f}% of epithelial)")
    print(f"  Dimensions used: {n_dims}/6")


def run_infercnv(adata):
    """Estimate CNV scores using inferCNVpy (or simple moving-average approach)."""
    print("  Running CNV estimation (moving-average method)...")
    epi_mask = adata.obs['is_epithelial']
    # Use immune cells as reference (normal diploid)
    immune_types = {'T_NK', 'B_Plasma', 'Macrophage', 'Monocyte', 'DC', 'Mast'}
    ref_mask = adata.obs['celltype'].isin(immune_types)

    if ref_mask.sum() < 100:
        print("  WARNING: Too few reference cells for CNV estimation, skipping")
        adata.obs['cnv_score'] = np.nan
        return

    try:
        import infercnvpy as cnv
        cnv.tl.infercnv(adata, reference_key='celltype',
                        reference_cat=list(immune_types & set(adata.obs['celltype'].unique())),
                        window_size=100)
        cnv.tl.cnv_score(adata)
        adata.obs.loc[~epi_mask, 'cnv_score'] = np.nan
        print(f"  CNV scores computed for {epi_mask.sum()} epithelial cells")
    except (ImportError, Exception) as e:
        print(f"  infercnvpy not available ({e}), using simplified CNV estimation...")
        # Simplified: chromosome-arm level expression deviation from reference
        ref_expr = adata[ref_mask].X
        if hasattr(ref_expr, 'toarray'):
            ref_expr = ref_expr.toarray()
        ref_mean = ref_expr.mean(axis=0)

        epi_expr = adata[epi_mask].X
        if hasattr(epi_expr, 'toarray'):
            epi_expr = epi_expr.toarray()
        # CNV score = mean absolute deviation from reference (smoothed)
        deviation = np.abs(epi_expr - ref_mean).mean(axis=1)
        adata.obs['cnv_score'] = np.nan
        adata.obs.loc[epi_mask, 'cnv_score'] = deviation
        print(f"  Simplified CNV scores computed for {epi_mask.sum()} epithelial cells")


def run_cnmf(adata, scvi_model, adata_scvi):
    """Run consensus NMF on epithelial cells using scVI corrected expression."""
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans

    epi_mask = adata.obs['is_epithelial']
    epi_idx = adata.obs_names[epi_mask]
    # Use scVI normalized expression (batch-corrected)
    # Use boolean mask directly (obs_names may have duplicates across samples)
    epi_mask_bool = epi_mask.values
    print("  Getting scVI normalized expression for epithelial cells...")
    norm_expr = scvi_model.get_normalized_expression(adata_scvi[epi_mask_bool], library_size=1e4)
    X = np.log1p(norm_expr.values)
    X = np.clip(X, 0, None)
    gene_names = norm_expr.columns.tolist()

    # Select HVGs from this matrix
    var_genes = np.var(X, axis=0)
    top_idx = np.argsort(var_genes)[-2000:]
    X_hvg = X[:, top_idx]
    hvg_names = [gene_names[i] for i in top_idx]

    # Consensus NMF: run multiple times, cluster H matrices
    k = 20
    n_runs = 30
    print(f"  Running consensus NMF (k={k}, {n_runs} runs)...")
    all_H = []
    for i in range(n_runs):
        model_nmf = NMF(n_components=k, init='nndsvda', random_state=i, max_iter=400)
        model_nmf.fit(X_hvg)
        all_H.append(model_nmf.components_)  # k x genes

    # Stack all program vectors and cluster to find consensus
    all_programs = np.vstack(all_H)  # (k*n_runs) x genes
    # Normalize rows for clustering
    norms = np.linalg.norm(all_programs, axis=1, keepdims=True) + 1e-10
    all_programs_norm = all_programs / norms

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(all_programs_norm)

    # Consensus: average programs within each cluster
    H_consensus = np.zeros((k, X_hvg.shape[1]))
    program_stability = []
    for c in range(k):
        members = all_programs_norm[labels == c]
        H_consensus[c] = members.mean(axis=0)
        # Stability: mean pairwise cosine similarity within cluster
        if len(members) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(members)
            stability = (sim.sum() - len(members)) / (len(members) * (len(members) - 1))
        else:
            stability = 0.0
        program_stability.append(stability)

    # Filter unstable programs (stability < 0.3)
    stable_mask = np.array(program_stability) >= 0.3
    n_stable = stable_mask.sum()
    print(f"  Stable programs: {n_stable}/{k} (stability >= 0.3)")

    # Get cell scores (W) using final consensus H
    # Re-fit with consensus initialization
    W = X_hvg @ H_consensus.T  # approximate cell loadings
    print(f"  W shape: {W.shape}, epi cells: {epi_mask_bool.sum()}")

    # Save program scores back to full adata — use positional numpy assignment (obs_names have duplicates)
    for c in range(k):
        col = f"cNMF_program_{c}"
        arr = np.full(adata.n_obs, np.nan)
        arr[epi_mask_bool] = W[:, c]
        adata.obs[col] = arr
        adata.obs[f"{col}_stable"] = stable_mask[c]

    # Save program gene weights
    program_genes = pd.DataFrame(H_consensus, columns=hvg_names,
                                 index=[f"program_{c}" for c in range(k)])
    program_genes.to_csv(f"{BASE}/results/cnmf_program_genes.csv")

    # --- Stage correlation ---
    stage_map = {'NAG': 0, 'CAG': 1, 'IM': 2, 'EGC': 3, 'GC': 3, 'EGC_multi_region': 3}
    epi_obs = adata.obs.loc[epi_mask_bool]
    stage_numeric = epi_obs['stage'].map(stage_map).values.astype(float)
    valid = ~np.isnan(stage_numeric)

    program_stage_corr = []
    for c in range(k):
        if valid.sum() > 10:
            r, p = spearmanr(W[valid, c], stage_numeric[valid])
        else:
            r, p = 0, 1
        program_stage_corr.append({'program': c, 'spearman_r': r, 'pval': p,
                                   'stable': bool(stable_mask[c]),
                                   'stability_score': program_stability[c]})
    corr_df = pd.DataFrame(program_stage_corr)
    corr_df.to_csv(f"{BASE}/results/cnmf_stage_correlation.csv", index=False)

    # --- Orphan program identification (overlap coefficient, asymmetry-corrected) ---
    known_signatures = {
        'PMC_2': ["NAMPT", "ALDH1A1", "CD44", "SOX9", "OLFM4"],
        'PMC_P': ["AREG", "NAMPT", "PHLDA1", "ITGA2", "MYC"],
        'stemness': ["LGR5", "OLFM4", "SOX9", "ASCL2"],
        'proliferation': ["MKI67", "TOP2A", "PCNA", "CDK1"],
        'IM': ["CDX2", "MUC2", "TFF3", "VIL1"],
        'EMT': ["VIM", "SNAI1", "ZEB1", "CDH2"],
        'EGC_like': ["REG4", "CEACAM6", "MUC13", "CLDN3", "EPCAM", "KRT20", "ERBB2", "MET", "VEGFA"],
    }

    orphan_results = []
    candidate_pool_B = []
    for c in range(k):
        if not stable_mask[c]:
            continue
        top15_idx = np.argsort(H_consensus[c])[-15:]
        top15_genes = set(hvg_names[i] for i in top15_idx)

        max_overlap = 0
        max_sig = ""
        for sig_name, sig_genes in known_signatures.items():
            sig_set = set(sig_genes)
            intersection = len(top15_genes & sig_set)
            overlap_coef = intersection / min(len(top15_genes), len(sig_set)) if min(len(top15_genes), len(sig_set)) > 0 else 0
            if overlap_coef > max_overlap:
                max_overlap = overlap_coef
                max_sig = sig_name

        stage_r = corr_df.loc[c, 'spearman_r']
        stage_p = corr_df.loc[c, 'pval']
        is_orphan = (stage_p < 0.05) and (abs(stage_r) > 0.1) and (max_overlap < 0.4)
        orphan_results.append({
            'program': c, 'max_overlap_coef': max_overlap, 'best_match': max_sig,
            'stage_corr_p': stage_p, 'stage_corr_r': stage_r,
            'is_orphan': is_orphan
        })
        if is_orphan:
            top20_idx = np.argsort(H_consensus[c])[-20:]
            top20_loadings = H_consensus[c][top20_idx]
            for i, idx in enumerate(top20_idx):
                candidate_pool_B.append({
                    'gene': hvg_names[idx],
                    'program': c,
                    'loading': top20_loadings[i],
                    'stage_r': abs(stage_r),
                    'score': top20_loadings[i] * abs(stage_r)
                })

    orphan_df = pd.DataFrame(orphan_results)
    orphan_df.to_csv(f"{BASE}/results/cnmf_orphan_programs.csv", index=False)

    n_orphan = orphan_df['is_orphan'].sum() if len(orphan_df) > 0 else 0
    print(f"  Orphan programs (stage-correlated |r|>0.1, overlap<0.4): {n_orphan}")

    if candidate_pool_B:
        pool_B = pd.DataFrame(candidate_pool_B)
        pool_B = pool_B.sort_values('score', ascending=False).drop_duplicates('gene', keep='first')
        pool_B = pool_B.head(20)
        pool_B.to_csv(f"{BASE}/results/candidate_pool_B.csv", index=False)
        print(f"  Candidate pool B: {len(pool_B)} genes (ranked by loading * |stage_r|)")

    sig_programs = corr_df[(corr_df['pval'] < 0.05) & corr_df['stable']].shape[0]
    print(f"  Stage-correlated stable programs: {sig_programs}/{n_stable}")
    return program_genes


def generate_pseudobulk_by_celltype(adata):
    """Generate pseudobulk by (sample_id, celltype) for downstream DE."""
    pb_rows = []
    for (sid, ct), idx in adata.obs.groupby(['sample_id', 'celltype']).groups.items():
        if len(idx) < 10:
            continue
        X_sub = adata[idx].X
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        mean_expr = np.asarray(X_sub).mean(axis=0)
        row = dict(zip(adata.var_names, mean_expr))
        row['sample_id'] = sid
        row['celltype'] = ct
        row['dataset'] = adata.obs.loc[idx[0], 'dataset']
        row['stage'] = adata.obs.loc[idx[0], 'stage']
        row['hp_status'] = adata.obs.loc[idx[0], 'hp_status']
        row['n_cells'] = len(idx)
        pb_rows.append(row)
    pb_df = pd.DataFrame(pb_rows)
    pb_path = f"{BASE}/data/pseudobulk_by_sample_celltype.csv"
    pb_df.to_csv(pb_path, index=False)
    print(f"  Pseudobulk by (sample, celltype): {len(pb_rows)} entries saved")


def main():
    print("=" * 60)
    print("Step 2: scVI Integration + Annotation + cNMF")
    print("=" * 60)

    # [1] Load
    print("\n[1] Loading data...")
    adata = sc.read_h5ad(f"{BASE}/data/adata_raw_unintegrated.h5ad")
    adata.obs_names_make_unique()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # [2] Preprocessing
    print("\n[2] Preprocessing...")
    import gc
    from scipy.sparse import issparse, csr_matrix
    sc.pp.filter_genes(adata, min_cells=10)
    # Convert to float32 immediately to halve memory
    if issparse(adata.X):
        adata.X = csr_matrix(adata.X, dtype=np.float32)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    print(f"  {adata.n_vars} genes, {adata.var['highly_variable'].sum()} HVGs")
    gc.collect()

    # [3] scVI — load saved model or train
    print("\n[3] Training scVI...")
    import gc
    adata_scvi = adata[:, adata.var['highly_variable']].copy()
    gc.collect()
    scvi.model.SCVI.setup_anndata(adata_scvi, layer='counts', batch_key="dataset")
    model_path = f"{BASE}/data/scvi_model"
    if os.path.exists(model_path):
        print("  Loading saved model...")
        model = scvi.model.SCVI.load(model_path, adata=adata_scvi)
        print("  Model loaded from checkpoint.")
    else:
        model = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30, gene_likelihood="zinb")
        model.train(max_epochs=400, early_stopping=True, early_stopping_patience=20,
                    accelerator='gpu', batch_size=64,
                    datasplitter_kwargs={'pin_memory': False, 'num_workers': 0})
        model.save(model_path, overwrite=True)
        print("  Model saved.")

    # [4] Latent representation
    print("\n[4] Latent representation...")
    adata.obsm['X_scVI'] = model.get_latent_representation(adata_scvi)
    gc.collect()

    # [5] Neighbors + UMAP
    print("\n[5] Neighbors + UMAP...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30)
    sc.tl.umap(adata)

    # [6] Leiden clustering (0.8, 1.2, 2.0)
    print("\n[6] Leiden clustering...")
    for res in [0.8, 1.2, 2.0]:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
        print(f"  res={res}: {adata.obs[f'leiden_{res}'].nunique()} clusters")

    # [7] Cell type annotation (Layer 1)
    print("\n[7] Cell type annotation (Layer 1)...")
    annotate_clusters(adata, res=0.8)
    print("  Cell type distribution:")
    for ct, n in adata.obs['celltype'].value_counts().items():
        print(f"    {ct}: {n}")

    # [8] Research scores (Layer 2) on epithelial
    print("\n[8] Layer 2 research scores...")
    compute_layer2_scores(adata)
    print("  Scores: " + ", ".join(SCORES_LAYER2.keys()) + ", incomplete_IM, complete_IM")

    # [9] inferCNV
    print("\n[9] CNV estimation (inferCNV/simplified)...")
    run_infercnv(adata)

    # [10] High-risk epithelial state
    print("\n[10] High-risk epithelial state (>=3/6 dimensions)...")
    high_risk_epithelial(adata)

    # [11] cNMF (using scVI corrected expression)
    print("\n[11] Consensus NMF transcriptional programs...")
    os.makedirs(f"{BASE}/results", exist_ok=True)
    run_cnmf(adata, model, adata_scvi)

    # [12] Pseudobulk by (sample_id, celltype)
    print("\n[12] Generating pseudobulk by (sample, celltype)...")
    generate_pseudobulk_by_celltype(adata)

    # [13] Integration quality check (dataset entropy per cluster)
    print("\n[13] Integration quality check...")
    from scipy.stats import entropy as sp_entropy
    cluster_entropy = []
    for cl in adata.obs['leiden_0.8'].unique():
        mask = adata.obs['leiden_0.8'] == cl
        counts = adata.obs.loc[mask, 'dataset'].value_counts()
        props = counts / counts.sum()
        ent = sp_entropy(props, base=2) / np.log2(max(len(counts), 2))
        cluster_entropy.append({'cluster': cl, 'dataset_entropy': ent, 'n_datasets': len(counts)})
    ent_df = pd.DataFrame(cluster_entropy)
    mean_ent = ent_df['dataset_entropy'].mean()
    well_mixed = (ent_df['dataset_entropy'] > 0.5).sum()
    print(f"  Mean dataset entropy: {mean_ent:.3f}")
    print(f"  Well-mixed clusters (entropy>0.5): {well_mixed}/{len(ent_df)}")
    ent_df.to_csv(f"{BASE}/results/integration_quality.csv", index=False)

    # [14] Save
    print("\n[14] Saving adata_integrated.h5ad...")
    adata.write_h5ad(f"{BASE}/data/adata_integrated.h5ad")
    print(f"  Saved: {adata.n_obs} cells")

    print(f"\n{'='*60}")
    print("Step 2 COMPLETE")
    print(f"  Epithelial: {adata.obs['is_epithelial'].sum()}")
    print(f"  High-risk: {adata.obs['high_risk_epithelial'].sum()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
