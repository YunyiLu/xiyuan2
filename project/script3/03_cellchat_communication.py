"""
Step 3: Cell-cell communication via LIANA (sample-level + spatial co-localization).
Input: script3/data/adata_integrated.h5ad, script3/data/spatial_deconv.h5ad
Output: script3/results/cellchat_per_stage.csv, differential_LR.csv, cellchat_candidates.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import liana as li
from scipy.stats import mannwhitneyu

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"

EARLY_STAGES = {'NAG', 'CAG'}
LATE_STAGES = {'IM', 'EGC', 'GC', 'EGC_multi_region'}

KNOWN_AXES = [
    ('Macrophage', 'Gastric_mucous', 'AREG'),
    ('Macrophage', 'Enterocyte_IM', 'AREG'),
    ('Gastric_mucous', 'Fibroblast', 'NAMPT'),
    ('Enterocyte_IM', 'Fibroblast', 'NAMPT'),
    ('Mast', 'Gastric_mucous', 'TPSAB1'),
    ('Pericyte', 'Stem_proliferative', 'PDGFRB'),
]


def run_liana_per_sample(adata):
    """Run LIANA independently per sample. Returns DataFrame with sample-level LR scores."""
    all_results = []
    samples = adata.obs['sample_id'].unique()

    for sid in samples:
        adata_s = adata[adata.obs['sample_id'] == sid].copy()
        ct_counts = adata_s.obs['celltype'].value_counts()
        valid_cts = ct_counts[ct_counts >= 10].index.tolist()
        if len(valid_cts) < 2:
            print(f"    {sid}: skipped (< 2 valid cell types)")
            continue

        adata_s = adata_s[adata_s.obs['celltype'].isin(valid_cts)].copy()
        try:
            li.mt.cellphonedb(adata_s, groupby='celltype', resource_name='consensus',
                             use_raw=False, verbose=False)
            res = adata_s.uns['liana_res'].copy()
            res['sample_id'] = sid
            res['stage'] = adata_s.obs['stage'].iloc[0]
            all_results.append(res)
            print(f"    {sid}: {len(res)} interactions")
        except Exception as e:
            print(f"    {sid}: LIANA failed ({e})")

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def sample_level_differential(liana_df):
    """Sample-level Wilcoxon test: late-stage vs early-stage LR scores."""
    score_col = None
    for col in ['lr_means', 'magnitude_rank', 'specificity_rank', 'cellphonedb_pvalue']:
        if col in liana_df.columns:
            score_col = col
            break
    if score_col is None:
        print("  WARNING: No score column found in LIANA output")
        return pd.DataFrame()

    # Create LR pair key
    liana_df['lr_pair'] = (liana_df['source'] + '|' + liana_df['target'] + '|' +
                           liana_df['ligand_complex'] + '|' + liana_df['receptor_complex'])

    early_samples = liana_df[liana_df['stage'].isin(EARLY_STAGES)]['sample_id'].unique()
    late_samples = liana_df[liana_df['stage'].isin(LATE_STAGES)]['sample_id'].unique()

    if len(early_samples) < 3 or len(late_samples) < 3:
        print(f"  WARNING: Too few samples (early={len(early_samples)}, late={len(late_samples)})")
        return pd.DataFrame()

    # Per LR pair: get sample-level mean score, then Wilcoxon
    results = []
    for lr_pair, grp in liana_df.groupby('lr_pair'):
        early_scores = grp[grp['stage'].isin(EARLY_STAGES)].groupby('sample_id')[score_col].mean()
        late_scores = grp[grp['stage'].isin(LATE_STAGES)].groupby('sample_id')[score_col].mean()

        if len(early_scores) < 3 or len(late_scores) < 3:
            continue

        stat, pval = mannwhitneyu(late_scores, early_scores, alternative='two-sided')
        parts = lr_pair.split('|')
        results.append({
            'source': parts[0], 'target': parts[1],
            'ligand': parts[2], 'receptor': parts[3],
            'lr_pair': lr_pair,
            'early_mean': early_scores.mean(), 'late_mean': late_scores.mean(),
            'delta': late_scores.mean() - early_scores.mean(),
            'pval': pval, 'n_early': len(early_scores), 'n_late': len(late_scores),
        })

    diff_df = pd.DataFrame(results)
    if len(diff_df) > 0:
        from statsmodels.stats.multitest import multipletests
        diff_df['padj'] = multipletests(diff_df['pval'], method='fdr_bh')[1]
        diff_df = diff_df.sort_values('padj')
    return diff_df


def spatial_colocalization(diff_df, n_perm=1000):
    """Validate top differential LR pairs via spatial co-localization in Visium data."""
    import squidpy as sq

    # Load spatial data
    spatial_path = f"{BASE}/data/spatial_deconv.h5ad"
    if not os.path.exists(spatial_path):
        print("  WARNING: spatial_deconv.h5ad not found, skipping spatial validation")
        return diff_df.assign(spatial_pval=np.nan, n_patients_coloc=0)

    adata_sp = sc.read_h5ad(spatial_path)

    # Need per-sample spatial files for coordinates
    spatial_dir = f"{BASE}/data/spatial"
    sample_files = [f for f in os.listdir(spatial_dir) if f.endswith('_deconv.h5ad')] if os.path.exists(spatial_dir) else []

    if not sample_files:
        # Try loading individual samples from source
        sample_adatas = {}
        for gp in ['GP1','GP2','GP3','GP4','GP5','GP6','GP7','GP8','GP9']:
            path = f"{SPATIAL_DIR}/{gp}"
            if os.path.exists(path):
                try:
                    sp = sc.read_visium(path)
                    sp.var_names_make_unique()
                    sp.obs['sample_id'] = gp
                    sample_adatas[gp] = sp
                except:
                    pass
    else:
        sample_adatas = {}
        for f in sample_files:
            sp = sc.read_h5ad(f"{spatial_dir}/{f}")
            sid = f.replace('_deconv.h5ad', '')
            sample_adatas[sid] = sp

    if not sample_adatas:
        print("  WARNING: No spatial samples with coordinates, skipping")
        return diff_df.assign(spatial_pval=np.nan, n_patients_coloc=0)

    # For top significant LR pairs, check spatial co-localization
    sig_pairs = diff_df[diff_df['padj'] < 0.05].head(50) if 'padj' in diff_df.columns else diff_df.head(50)

    coloc_results = []
    for _, row in sig_pairs.iterrows():
        ligand_gene = row['ligand'].split('_')[0] if '_' in row['ligand'] else row['ligand']
        receptor_gene = row['receptor'].split('_')[0] if '_' in row['receptor'] else row['receptor']

        patient_pvals = []
        for sid, sp in sample_adatas.items():
            if ligand_gene not in sp.var_names or receptor_gene not in sp.var_names:
                continue

            lig_expr = sp[:, ligand_gene].X
            rec_expr = sp[:, receptor_gene].X
            if hasattr(lig_expr, 'toarray'):
                lig_expr = lig_expr.toarray().flatten()
                rec_expr = rec_expr.toarray().flatten()
            else:
                lig_expr = np.asarray(lig_expr).flatten()
                rec_expr = np.asarray(rec_expr).flatten()

            lig_pos = lig_expr > np.percentile(lig_expr[lig_expr > 0], 50) if (lig_expr > 0).sum() > 10 else lig_expr > 0
            rec_pos = rec_expr > np.percentile(rec_expr[rec_expr > 0], 50) if (rec_expr > 0).sum() > 10 else rec_expr > 0

            if lig_pos.sum() < 5 or rec_pos.sum() < 5:
                continue

            # Compute spatial neighbors
            if 'spatial' in sp.obsm:
                coords = sp.obsm['spatial']
            else:
                continue

            # Mean distance between ligand+ and receptor+ spots
            lig_coords = coords[lig_pos]
            rec_coords = coords[rec_pos]
            from scipy.spatial.distance import cdist
            real_dist = cdist(lig_coords, rec_coords).min(axis=1).mean()

            # Permutation test
            perm_dists = []
            for _ in range(n_perm):
                perm_idx = np.random.permutation(len(coords))
                perm_rec = coords[perm_idx[:rec_pos.sum()]]
                perm_dists.append(cdist(lig_coords, perm_rec).min(axis=1).mean())
            perm_dists = np.array(perm_dists)
            p_coloc = (perm_dists <= real_dist).sum() / n_perm
            patient_pvals.append(p_coloc)

        n_sig_patients = sum(1 for p in patient_pvals if p < 0.05)
        mean_pval = np.mean(patient_pvals) if patient_pvals else 1.0
        coloc_results.append({
            'lr_pair': row['lr_pair'],
            'spatial_pval': mean_pval,
            'n_patients_coloc': n_sig_patients
        })

    coloc_df = pd.DataFrame(coloc_results)
    if len(coloc_df) > 0:
        diff_df = diff_df.merge(coloc_df, on='lr_pair', how='left')
        diff_df['spatial_pval'] = diff_df['spatial_pval'].fillna(1.0)
        diff_df['n_patients_coloc'] = diff_df['n_patients_coloc'].fillna(0).astype(int)
    else:
        diff_df['spatial_pval'] = np.nan
        diff_df['n_patients_coloc'] = 0
    return diff_df


def select_candidate_pool_D(diff_df):
    """Apply 3-constraint filter to select candidate genes for pool D."""
    # Constraint 1: sample-level significant (padj < 0.05)
    # Constraint 2: spatial co-localization (spatial_pval < 0.05)
    # Constraint 3: observed in >= 2/9 patients
    mask = (
        (diff_df['padj'] < 0.05) &
        (diff_df['spatial_pval'] < 0.05) &
        (diff_df['n_patients_coloc'] >= 2)
    )
    candidates = diff_df[mask].copy()

    # Extract ligand and receptor genes
    genes = set()
    for _, row in candidates.iterrows():
        lig = row['ligand'].split('_')[0] if '_' in row['ligand'] else row['ligand']
        rec = row['receptor'].split('_')[0] if '_' in row['receptor'] else row['receptor']
        genes.add(lig)
        genes.add(rec)

    # Cap at 10 genes (PLAN.md requirement)
    genes = sorted(genes)[:10]
    return candidates, genes


def check_known_axes(liana_df):
    """Verify known communication axes are detected."""
    print("\n  Known axis verification:")
    for source, target, gene in KNOWN_AXES:
        mask = (
            (liana_df['source'] == source) & (liana_df['target'] == target) &
            ((liana_df['ligand_complex'].str.contains(gene, na=False)) |
             (liana_df['receptor_complex'].str.contains(gene, na=False)))
        )
        n_found = mask.sum()
        status = "FOUND" if n_found > 0 else "NOT FOUND"
        print(f"    {source} → {target} ({gene}): {status} ({n_found} interactions)")


def main():
    print("=" * 60)
    print("Step 3: Cell Communication (LIANA, sample-level)")
    print("=" * 60)

    # [1] Load
    print("\n[1] Loading data...")
    adata = sc.read_h5ad(f"{BASE}/data/adata_integrated.h5ad")
    print(f"  {adata.n_obs} cells, {adata.obs['sample_id'].nunique()} samples")
    print(f"  Cell types: {adata.obs['celltype'].nunique()}")
    print(f"  Stages: {adata.obs['stage'].value_counts().to_dict()}")

    # [2] Run LIANA per sample
    print("\n[2] Running LIANA per sample (n={})...".format(adata.obs['sample_id'].nunique()))
    os.makedirs(f"{BASE}/results", exist_ok=True)
    liana_df = run_liana_per_sample(adata)

    if liana_df.empty:
        print("  ERROR: No LIANA results. Exiting.")
        return

    liana_df.to_csv(f"{BASE}/results/cellchat_per_stage.csv", index=False)
    print(f"  Total: {len(liana_df)} interactions across {liana_df['sample_id'].nunique()} samples")

    # Verify known axes
    check_known_axes(liana_df)

    # [3] Sample-level differential communication
    print("\n[3] Sample-level differential (late vs early)...")
    diff_df = sample_level_differential(liana_df)
    if diff_df.empty:
        print("  No differential results.")
        return

    n_sig = (diff_df['padj'] < 0.05).sum() if 'padj' in diff_df.columns else 0
    print(f"  Significant LR pairs (padj<0.05): {n_sig}")
    diff_df.to_csv(f"{BASE}/results/differential_LR.csv", index=False)

    # [4] Spatial co-localization validation
    print("\n[4] Spatial co-localization validation (permutation test)...")
    diff_df = spatial_colocalization(diff_df, n_perm=500)
    diff_df.to_csv(f"{BASE}/results/differential_LR.csv", index=False)

    # [5] Select candidate pool D (3-constraint filter)
    print("\n[5] Selecting candidate pool D (3-constraint filter)...")
    candidates, pool_D_genes = select_candidate_pool_D(diff_df)
    print(f"  LR pairs passing all 3 constraints: {len(candidates)}")
    print(f"  Candidate pool D genes (max 10): {pool_D_genes}")

    candidates.to_csv(f"{BASE}/results/cellchat_candidates.csv", index=False)
    if pool_D_genes:
        pd.DataFrame({'gene': pool_D_genes, 'source': 'CellChat_spatial'}).to_csv(
            f"{BASE}/results/candidate_pool_D.csv", index=False)

    # [6] Summary
    print(f"\n{'='*60}")
    print("Step 3 COMPLETE")
    print(f"  Samples analyzed: {liana_df['sample_id'].nunique()}")
    print(f"  Significant differential LR pairs: {n_sig}")
    print(f"  Spatially validated (>=2 patients): {(diff_df['n_patients_coloc'] >= 2).sum()}")
    print(f"  Candidate pool D: {len(pool_D_genes)} genes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
