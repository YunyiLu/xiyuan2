"""
Step 2b: Spatial deconvolution of OMIX010346 Visium data.
Uses scVI-annotated scRNA reference to deconvolve 9 Visium samples (GP1-GP9).
Input: script3/data/adata_integrated.h5ad (reference), OMIX010346 Visium samples
Output: script3/data/spatial_deconv.h5ad
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"

VISIUM_SAMPLES = ['GP1', 'GP2', 'GP3', 'GP4', 'GP5', 'GP6', 'GP7', 'GP8', 'GP9']


def load_visium_samples():
    """Load all 9 Visium samples (one ESD slice per patient)."""
    adatas = {}
    for sample in VISIUM_SAMPLES:
        path = f"{SPATIAL_DIR}/{sample}"
        if not os.path.exists(path):
            print(f"  WARNING: {sample} not found, skipping")
            continue
        try:
            adata_sp = sc.read_visium(path)
            adata_sp.var_names_make_unique()
            adata_sp.obs['sample_id'] = sample
            adata_sp.obs['patient'] = sample
            adatas[sample] = adata_sp
            print(f"  {sample}: {adata_sp.n_obs} spots, {adata_sp.n_vars} genes")
        except Exception as e:
            print(f"  ERROR loading {sample}: {e}")
    return adatas


def run_cell2location(adata_ref, adatas_spatial):
    """Run cell2location deconvolution using raw counts reference."""
    import cell2location
    from cell2location.models import RegressionModel

    print("\n[2] Training reference model (NB regression)...")
    ref = adata_ref[adata_ref.obs['celltype'] != 'Unassigned'].copy()
    # Use raw counts layer for cell2location
    if 'counts' in ref.layers:
        ref.X = ref.layers['counts'].copy()
    sc.pp.filter_genes(ref, min_cells=10)

    cell2location.models.RegressionModel.setup_anndata(ref, labels_key='celltype')
    ref_model = RegressionModel(ref)
    ref_model.train(max_epochs=250, accelerator='gpu')
    ref_model.export_posterior(ref, use_quantiles=True)
    inf_aver = ref_model.samples['post_sample_means']['per_cluster_mu_fg']
    inf_aver = pd.DataFrame(inf_aver.T, index=ref.var_names,
                            columns=ref.obs['celltype'].cat.categories)

    print("\n[3] Running cell2location on spatial data...")
    spatial_results = []
    for sample_name, adata_sp in adatas_spatial.items():
        print(f"  Deconvolving {sample_name}...")
        common = adata_sp.var_names.intersection(ref.var_names)
        adata_sp_sub = adata_sp[:, common].copy()

        cell2location.models.Cell2location.setup_anndata(adata_sp_sub)
        sp_model = cell2location.models.Cell2location(
            adata_sp_sub, cell_state_df=inf_aver.loc[common],
            N_cells_per_location=10, detection_alpha=20)
        sp_model.train(max_epochs=30000, accelerator='gpu')
        sp_model.export_posterior(adata_sp_sub, use_quantiles=True)
        adata_sp.obsm['cell2location'] = adata_sp_sub.obsm['q05_cell_abundance_w_sf']
        spatial_results.append(adata_sp)

    return spatial_results


def run_nnls_fallback(adata_ref, adatas_spatial):
    """NNLS deconvolution using marker genes as fallback."""
    from scipy.optimize import nnls

    print("\n[2] Building reference profiles (marker genes)...")
    ref = adata_ref[adata_ref.obs['celltype'] != 'Unassigned'].copy()
    if 'counts' in ref.layers:
        ref.X = ref.layers['counts'].copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    # Use HVGs for deconvolution (more informative than all genes)
    sc.pp.highly_variable_genes(ref, n_top_genes=2000)
    marker_genes = ref.var_names[ref.var['highly_variable']].tolist()

    celltypes = sorted(ref.obs['celltype'].unique())
    ref_sub = ref[:, marker_genes]
    profiles = np.zeros((len(celltypes), len(marker_genes)))
    for i, ct in enumerate(celltypes):
        mask = ref.obs['celltype'] == ct
        X = ref_sub[mask].X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        profiles[i] = X.mean(axis=0)

    print(f"  Reference: {len(celltypes)} cell types, {len(marker_genes)} marker genes")

    print("\n[3] NNLS deconvolution on spatial data...")
    spatial_results = []
    for sample_name, adata_sp in adatas_spatial.items():
        print(f"  Deconvolving {sample_name}...")
        common = [g for g in marker_genes if g in adata_sp.var_names]
        common_ref_idx = [marker_genes.index(g) for g in common]

        X_sp = adata_sp[:, common].X
        if hasattr(X_sp, 'toarray'):
            X_sp = X_sp.toarray()
        # Normalize spatial data
        X_sp = X_sp / (X_sp.sum(axis=1, keepdims=True) + 1e-6) * 1e4
        X_sp = np.log1p(X_sp)

        ref_matrix = profiles[:, common_ref_idx].T  # genes x celltypes

        proportions = np.zeros((adata_sp.n_obs, len(celltypes)))
        for j in range(adata_sp.n_obs):
            coef, _ = nnls(ref_matrix, X_sp[j])
            total = coef.sum()
            if total > 0:
                proportions[j] = coef / total

        adata_sp.obsm['deconv_proportions'] = pd.DataFrame(
            proportions, index=adata_sp.obs_names, columns=celltypes)
        spatial_results.append(adata_sp)
        print(f"    Done: {adata_sp.n_obs} spots")

    return spatial_results


def main():
    print("=" * 60)
    print("Step 2b: Spatial Deconvolution (OMIX010346 Visium)")
    print("=" * 60)

    # [1] Load reference (use raw counts layer)
    print("\n[1] Loading scRNA reference...")
    adata_ref = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
    print(f"  Reference: {adata_ref.n_obs} cells, {adata_ref.obs['celltype'].nunique()} cell types")
    if 'counts' in adata_ref.layers:
        print("  Raw counts layer available")
    else:
        print("  WARNING: No 'counts' layer found, using .X (may be log-normalized)")

    # Load spatial data
    print("\n  Loading Visium samples...")
    adatas_spatial = load_visium_samples()
    if not adatas_spatial:
        print("  ERROR: No spatial samples loaded. Exiting.")
        return

    # [2-3] Deconvolution
    try:
        import cell2location
        print("\n  Using cell2location for deconvolution...")
        spatial_results = run_cell2location(adata_ref, adatas_spatial)
    except ImportError:
        print("\n  cell2location not available, using NNLS fallback...")
        spatial_results = run_nnls_fallback(adata_ref, adatas_spatial)

    # [4] Save per-sample (preserving spatial coordinates)
    print("\n[4] Saving spatial results...")
    os.makedirs(f"{BASE}/data/spatial", exist_ok=True)
    for adata_sp in spatial_results:
        sid = adata_sp.obs['sample_id'].iloc[0]
        adata_sp.write_h5ad(f"{BASE}/data/spatial/{sid}_deconv.h5ad")

    # Also save combined (spatial coords in obsm['spatial'] preserved per-sample via obs index)
    adata_spatial = ad.concat(spatial_results, join='outer',
                              uns_merge='unique', merge='same')
    print(f"  Total: {adata_spatial.n_obs} spots across {len(spatial_results)} samples")

    out_path = f"{BASE}/data/spatial_deconv.h5ad"
    adata_spatial.write_h5ad(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Per-sample files: {BASE}/data/spatial/GP*_deconv.h5ad")

    print(f"\n{'='*60}")
    print("Step 2b COMPLETE")
    print(f"  Spots: {adata_spatial.n_obs}")
    print(f"  Samples: {len(spatial_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
