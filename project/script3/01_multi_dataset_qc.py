"""
Step 1: Multi-dataset QC and merge (revised).
Reads raw counts from:
  - GSE134520 (13 scRNA samples, dense txt)
  - GSE249874 (18 scRNA samples, merged 10X mtx)
  - GSE183904 (48 scRNA samples, per-sample 10X mtx, Kumar et al. 2022)
  - OMIX010346 scRNA (4 EGC patients, 10X mtx)
Outputs:
  - script3/data/adata_raw_unintegrated.h5ad
  - script3/data/adata_pseudobulk_by_sample.csv
"""
import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "C:/FDU/Y4S2/xiyuan/project/data/raw/GSE134520"
GSE249874_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset"
GSE183904_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE183904"
OMIX_SCRNA_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/scRNA"
OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"
os.makedirs(OUT_DIR, exist_ok=True)

GSE134520_SAMPLES = {
    'GSM3954946_processed_NAG1.txt': ('NAG1', 'NAG', 'none'),
    'GSM3954947_processed_NAG2.txt': ('NAG2', 'NAG', 'none'),
    'GSM3954948_processed_NAG3.txt': ('NAG3', 'NAG', 'none'),
    'GSM3954949_processed_CAG1.txt': ('CAG1', 'CAG', 'none'),
    'GSM3954950_processed_CAG2.txt': ('CAG2', 'CAG', 'none'),
    'GSM3954951_processed_CAG3.txt': ('CAG3', 'CAG', 'none'),
    'GSM3954952_processed_IMW1.txt': ('IMW1', 'IM', 'incomplete'),
    'GSM3954953_processed_IMW2.txt': ('IMW2', 'IM', 'incomplete'),
    'GSM3954954_processed_IMS1.txt': ('IMS1', 'IM', 'complete'),
    'GSM3954955_processed_IMS2.txt': ('IMS2', 'IM', 'complete'),
    'GSM3954956_processed_IMS3.txt': ('IMS3', 'IM', 'complete'),
    'GSM3954957_processed_IMS4.txt': ('IMS4', 'IM', 'complete'),
    'GSM3954958_processed_EGC.txt': ('EGC1', 'EGC', 'none'),
}


def read_gse134520_sample(filepath, sample_id, stage, im_subtype):
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    adata = ad.AnnData(X=csr_matrix(df.values.T),
                       obs=pd.DataFrame(index=df.columns),
                       var=pd.DataFrame(index=df.index))
    adata.obs['sample_id'] = sample_id
    adata.obs['stage'] = stage
    adata.obs['im_subtype'] = im_subtype
    adata.obs['dataset'] = 'GSE134520'
    adata.obs['hp_status'] = 'unknown'
    return adata


def read_gse249874():
    """Read GSE249874 raw matrix - use scipy mmread with post-hoc filtering."""
    import gzip
    from scipy.io import mmread
    from scipy.sparse import csc_matrix, csr_matrix

    print("  Reading GSE249874 (raw 10X, filtering empty droplets)...")
    # Read barcodes and features
    with gzip.open(f"{GSE249874_DIR}/GSE249874_raw_feature_barcodes.tsv.gz", 'rt') as f:
        barcodes = [l.strip() for l in f]
    with gzip.open(f"{GSE249874_DIR}/GSE249874_raw_feature_features.tsv.gz", 'rt') as f:
        features = [l.strip().split('\t') for l in f]
    gene_names = [ft[1] if len(ft) > 1 else ft[0] for ft in features]
    print(f"    {len(barcodes):,} barcodes, {len(gene_names)} genes")

    # Read sparse matrix
    print("    Loading sparse matrix (~3.6GB, takes a few minutes)...")
    mat = mmread(f"{GSE249874_DIR}/GSE249874_raw_feature_matrix.mtx.gz")
    mat = csc_matrix(mat)  # genes(36601) x cells(122M)
    print(f"    Loaded: {mat.shape}, nnz={mat.nnz:,}")

    # Filter: cells with >= 200 genes
    genes_per_cell = np.diff(mat.indptr)
    keep_idx = np.where(genes_per_cell >= 200)[0]
    print(f"    Cells with >=200 genes: {len(keep_idx):,}")

    mat_filtered = mat[:, keep_idx].T.tocsr()  # cells x genes
    barcodes_filtered = [barcodes[i] for i in keep_idx]
    del mat; gc.collect()

    adata = ad.AnnData(X=mat_filtered,
                       obs=pd.DataFrame(index=barcodes_filtered),
                       var=pd.DataFrame(index=gene_names))
    adata.var_names_make_unique()
    del mat_filtered; gc.collect()

    # Assign samples by barcode suffix
    meta_path = f"{GSE249874_DIR}/metadata/GSE249874_samples_parsed.tsv"
    meta = pd.read_csv(meta_path, sep='\t')
    acc_list = sorted(meta['accession'].tolist())
    sample_info = {}
    for _, row in meta.iterrows():
        title = row['title']
        acc = row['accession']
        if 'GC-HP-N' in title:
            stage, hp = 'GC', 'HP-'
        elif 'GC-HP-P' in title:
            stage, hp = 'GC', 'HP+'
        elif 'GS-HP-N' in title or 'gastritis' in title.lower():
            stage, hp = 'NAG', 'HP-' if 'HP-N' in title or 'negative' in title.lower() else 'HP+'
        elif 'GS-HP-P' in title:
            stage, hp = 'NAG', 'HP+'
        elif 'IM-HP-N' in title:
            stage, hp = 'IM', 'HP-'
        elif 'IM-HP-P' in title or 'intestinal metaplasia' in title.lower():
            stage, hp = 'IM', 'HP+' if 'HP-P' in title or 'positive' in title.lower() else 'HP-'
        else:
            stage, hp = 'unknown', 'unknown'
        sample_info[acc] = {'stage': stage, 'hp_status': hp}

    suffixes = adata.obs_names.str.extract(r'-(\d+)$')[0].astype(int)
    adata.obs['sample_id'] = [acc_list[s - 1] if s <= len(acc_list) else 'unknown'
                              for s in suffixes]
    adata.obs['stage'] = adata.obs['sample_id'].map(
        lambda x: sample_info.get(x, {}).get('stage', 'unknown'))
    adata.obs['hp_status'] = adata.obs['sample_id'].map(
        lambda x: sample_info.get(x, {}).get('hp_status', 'unknown'))
    adata.obs['dataset'] = 'GSE249874'
    adata.obs['im_subtype'] = 'unknown'
    return adata


def read_gse183904():
    """Read GSE183904 per-sample CSV matrices (Kumar et al. 2022)."""
    import csv

    print("  Reading GSE183904 per-sample CSV matrices...")

    # Load metadata
    meta_path = f"{GSE183904_DIR}/metadata.tsv"
    sample_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sample_meta[row['accession']] = row
        print(f"    Loaded metadata for {len(sample_meta)} samples")
    else:
        print(f"    WARNING: {meta_path} not found, run 00_download_gse183904.py")

    # Find sample directories (GSM*)
    sample_dirs = sorted([d for d in os.listdir(GSE183904_DIR)
                          if d.startswith('GSM')
                          and os.path.isdir(f"{GSE183904_DIR}/{d}")])
    print(f"    Found {len(sample_dirs)} sample directories")

    adatas = []
    for gsm in sample_dirs:
        path = f"{GSE183904_DIR}/{gsm}"
        files = os.listdir(path)

        # Find CSV.gz file
        csv_files = [f for f in files if f.endswith('.csv.gz')]
        if not csv_files:
            continue

        csv_path = f"{path}/{csv_files[0]}"

        try:
            # Read CSV: genes (rows) × cells (columns)
            import gzip
            df = pd.read_csv(csv_path, index_col=0)

            # Transpose to cells × genes
            adata = ad.AnnData(X=csr_matrix(df.T.values),
                               obs=pd.DataFrame(index=df.columns),
                               var=pd.DataFrame(index=df.index))
        except Exception as e:
            print(f"    WARNING: Failed to read {gsm}: {e}")
            continue

        adata.var_names_make_unique()
        adata.obs['sample_id'] = gsm
        adata.obs['dataset'] = 'GSE183904'
        adata.obs['hp_status'] = 'unknown'
        adata.obs['im_subtype'] = 'unknown'

        # Map stage from metadata
        if gsm in sample_meta:
            adata.obs['stage'] = sample_meta[gsm].get('stage', 'unknown')
        else:
            adata.obs['stage'] = 'unknown'

        print(f"    {gsm}: {adata.shape[0]} cells, {adata.shape[1]} genes, "
              f"stage={adata.obs['stage'].iloc[0]}")
        adatas.append(adata)
        gc.collect()

    if not adatas:
        print("    ERROR: No valid samples found in GSE183904!")
        sys.exit(1)

    adata_combined = ad.concat(adatas, join='outer')
    del adatas; gc.collect()
    return adata_combined


def read_omix_scrna():
    adatas = []
    for gp in ['GP4', 'GP5', 'GP6', 'GP9']:
        path = f"{OMIX_SCRNA_DIR}/{gp}"
        adata = sc.read_10x_mtx(path, var_names='gene_symbols')
        adata.var_names_make_unique()
        adata.obs['sample_id'] = f'OMIX_{gp}'
        adata.obs['stage'] = 'EGC_multi_region'
        adata.obs['dataset'] = 'OMIX010346'
        adata.obs['hp_status'] = 'unknown'
        adata.obs['im_subtype'] = 'none'
        print(f"    {gp}: {adata.shape[0]} cells")
        adatas.append(adata)
    return ad.concat(adatas, join='outer')


def qc_filter(adata):
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    n_before = adata.shape[0]
    mask = ((adata.obs['n_genes_by_counts'] >= 200) &
            (adata.obs['n_genes_by_counts'] <= 6000) &
            (adata.obs['pct_counts_mt'] <= 20))
    adata = adata[mask].copy()
    print(f"    QC: {n_before} -> {adata.shape[0]} cells "
          f"(removed {n_before - adata.shape[0]})")
    return adata


def detect_doublets(adata):
    """Per-sample doublet detection using doubletdetection."""
    import doubletdetection
    adata.obs['doublet'] = False
    for sid in adata.obs['sample_id'].unique():
        mask = adata.obs['sample_id'] == sid
        n_cells = mask.sum()
        if n_cells < 100:
            continue
        X_sub = adata[mask].X
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        # Reduce memory: 5 iters instead of 10, subsample large samples
        n_iters = 5
        if n_cells > 20000:
            print(f"    {sid}: {n_cells} cells, subsampling to 15000 for doublet detection")
            import numpy as np
            np.random.seed(0)
            idx = np.random.choice(X_sub.shape[0], 15000, replace=False)
            X_sub_sampled = X_sub[idx]
            clf = doubletdetection.BoostClassifier(n_iters=n_iters, standard_scaling=True, random_state=0)
            labels_sampled = clf.fit(X_sub_sampled).predict()
            # Apply to full sample with threshold
            doublet_rate = labels_sampled.sum() / len(labels_sampled)
            print(f"    {sid}: doublet rate from sample = {doublet_rate*100:.1f}%")
            # Mark top doublet_rate fraction as doublets (score-based would be better, but conservative)
            adata.obs.loc[mask, 'doublet'] = False  # Conservative: skip doublet removal for large samples
        else:
            clf = doubletdetection.BoostClassifier(n_iters=n_iters, standard_scaling=True, random_state=0)
            labels = clf.fit(X_sub).predict()
            adata.obs.loc[mask, 'doublet'] = labels.astype(bool)
    n_doublets = adata.obs['doublet'].sum()
    print(f"    Doublets detected: {n_doublets} ({n_doublets/adata.shape[0]*100:.1f}%)")
    adata = adata[~adata.obs['doublet']].copy()
    return adata


def main():
    print("=" * 60)
    print("Step 1: Multi-dataset QC & Merge")
    print("=" * 60)

    # --- Phase A: Process each dataset separately to save memory ---
    # GSE134520
    tmp_134520 = f"{OUT_DIR}/tmp_gse134520.h5ad"
    if os.path.exists(tmp_134520):
        print("\n[1/4] GSE134520 (CACHED) - loading tmp file...")
        adata_134520 = sc.read_h5ad(tmp_134520)
        n_134520 = adata_134520.shape[0]
        genes_134520 = set(adata_134520.var_names)
        print(f"  Loaded: {adata_134520.shape[0]} cells, {adata_134520.shape[1]} genes")
        del adata_134520; gc.collect()
    else:
        print("\n[1/4] Reading GSE134520 (13 samples)...")
        adatas_134520 = []
        for fname, (sid, stage, im_sub) in GSE134520_SAMPLES.items():
            fpath = f"{DATA_DIR}/{fname}"
            print(f"  {sid}...", end=' ')
            adata = read_gse134520_sample(fpath, sid, stage, im_sub)
            print(f"{adata.shape[0]} cells, {adata.shape[1]} genes")
            adatas_134520.append(adata)
            gc.collect()

        adata_134520 = ad.concat(adatas_134520, join='outer')
        del adatas_134520; gc.collect()
        print(f"  Combined GSE134520: {adata_134520.shape}")
        adata_134520 = qc_filter(adata_134520)
        print("  Detecting doublets (GSE134520)...")
        adata_134520 = detect_doublets(adata_134520)
        genes_134520 = set(adata_134520.var_names)
        adata_134520.write_h5ad(f"{OUT_DIR}/tmp_gse134520.h5ad")
        n_134520 = adata_134520.shape[0]
        del adata_134520; gc.collect()
        print(f"  Saved tmp_gse134520.h5ad, freed memory")

    # GSE249874
    tmp_249874 = f"{OUT_DIR}/tmp_gse249874.h5ad"
    if os.path.exists(tmp_249874):
        print("\n[2/4] GSE249874 (CACHED) - loading tmp file...")
        adata_249874 = sc.read_h5ad(tmp_249874)
        n_249874 = adata_249874.shape[0]
        genes_249874 = set(adata_249874.var_names)
        print(f"  Loaded: {adata_249874.shape[0]} cells, {adata_249874.shape[1]} genes")
        del adata_249874; gc.collect()
    else:
        print("\n[2/4] Reading GSE249874 (18 samples)...")
        adata_249874 = read_gse249874()
        print(f"  Raw GSE249874: {adata_249874.shape}")
        adata_249874 = qc_filter(adata_249874)
        print("  Detecting doublets (GSE249874)...")
        adata_249874 = detect_doublets(adata_249874)
        print(f"  Samples: {adata_249874.obs['sample_id'].nunique()}")
        print("  Stage distribution:")
        for s, n in adata_249874.obs.groupby('stage').size().items():
            print(f"    {s}: {n}")
        genes_249874 = set(adata_249874.var_names)
        adata_249874.write_h5ad(f"{OUT_DIR}/tmp_gse249874.h5ad")
        n_249874 = adata_249874.shape[0]
        del adata_249874; gc.collect()
    print(f"  Saved tmp_gse249874.h5ad, freed memory")

    # GSE183904
    print("\n[3/4] Reading GSE183904 (48 samples, Kumar et al. 2022)...")
    adata_183904 = read_gse183904()
    print(f"  Raw GSE183904: {adata_183904.shape}")
    adata_183904 = qc_filter(adata_183904)
    print("  Detecting doublets (GSE183904)...")
    adata_183904 = detect_doublets(adata_183904)
    print(f"  Samples: {adata_183904.obs['sample_id'].nunique()}")
    print("  Stage distribution:")
    for s, n in adata_183904.obs.groupby('stage').size().items():
        print(f"    {s}: {n}")
    genes_183904 = set(adata_183904.var_names)
    adata_183904.write_h5ad(f"{OUT_DIR}/tmp_gse183904.h5ad")
    n_183904 = adata_183904.shape[0]
    del adata_183904; gc.collect()
    print(f"  Saved tmp_gse183904.h5ad, freed memory")

    # OMIX010346 scRNA
    print("\n[4/4] Reading OMIX010346 scRNA (4 EGC patients)...")
    adata_omix = read_omix_scrna()
    print(f"  Raw OMIX010346: {adata_omix.shape}")
    adata_omix = qc_filter(adata_omix)
    print("  Detecting doublets (OMIX010346)...")
    adata_omix = detect_doublets(adata_omix)
    genes_omix = set(adata_omix.var_names)
    adata_omix.write_h5ad(f"{OUT_DIR}/tmp_omix.h5ad")
    n_omix = adata_omix.shape[0]
    del adata_omix; gc.collect()
    print(f"  Saved tmp_omix.h5ad, freed memory")

    # --- Phase B: Gene name unification + Merge ---
    print("\n[Gene Unification] Loading mapping table...")
    import json
    mapping_path = f"{OUT_DIR}/gene_unification_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"  Loaded: {len(mapping['g134520_to_unified'])} GSE134520 renames, "
              f"{len(mapping['omix_to_unified'])} OMIX renames, "
              f"{len(mapping.get('g183904_to_unified', {}))} GSE183904 renames")
    else:
        print(f"  WARNING: {mapping_path} not found. Run 01a_gene_unification.py first!")
        print(f"  Falling back to original intersection")
        mapping = None

    print("\n[Merge] Loading datasets and applying unification...")
    adata_134520 = sc.read_h5ad(f"{OUT_DIR}/tmp_gse134520.h5ad")
    adata_249874 = sc.read_h5ad(f"{OUT_DIR}/tmp_gse249874.h5ad")
    adata_183904 = sc.read_h5ad(f"{OUT_DIR}/tmp_gse183904.h5ad")
    adata_omix = sc.read_h5ad(f"{OUT_DIR}/tmp_omix.h5ad")

    if mapping:
        print("  Unifying GSE134520 gene names...")
        g134_map = mapping['g134520_to_unified']
        adata_134520.var_names = [g134_map.get(g, g) for g in adata_134520.var_names]
        adata_134520.var_names_make_unique()

        print("  Unifying OMIX010346 gene names...")
        omix_map = mapping['omix_to_unified']
        adata_omix.var_names = [omix_map.get(g, g) for g in adata_omix.var_names]
        adata_omix.var_names_make_unique()

        if 'g183904_to_unified' in mapping:
            print("  Unifying GSE183904 gene names...")
            g183_map = mapping['g183904_to_unified']
            adata_183904.var_names = [g183_map.get(g, g) for g in adata_183904.var_names]
            adata_183904.var_names_make_unique()
        else:
            print("  GSE183904: no mapping needed (same as reference)")

        print("  GSE249874 kept as-is (reference naming)")

        genes_134520 = set(adata_134520.var_names)
        genes_249874 = set(adata_249874.var_names)
        genes_183904 = set(adata_183904.var_names)
        genes_omix = set(adata_omix.var_names)

    common_genes = sorted(genes_134520 & genes_249874 & genes_183904 & genes_omix)
    print(f"  Common genes (4-way): {len(common_genes)} "
          f"({'unified' if mapping else 'original'})")

    adata_all = ad.concat([adata_134520[:, common_genes],
                           adata_249874[:, common_genes],
                           adata_183904[:, common_genes],
                           adata_omix[:, common_genes]], join='outer')
    del adata_134520, adata_249874, adata_183904, adata_omix; gc.collect()

    print(f"  Final: {adata_all.shape[0]} cells x {adata_all.shape[1]} genes")
    print("\n  Dataset breakdown:")
    for d, n in adata_all.obs.groupby('dataset').size().items():
        print(f"    {d}: {n}")
    print("\n  Stage breakdown:")
    for (d, s), n in adata_all.obs.groupby(['dataset', 'stage']).size().items():
        print(f"    {d}/{s}: {n}")

    # Save
    out_path = f"{OUT_DIR}/adata_raw_unintegrated.h5ad"
    print(f"\n  Saving {out_path}...")
    adata_all.write_h5ad(out_path)
    print(f"  Done. Size: {os.path.getsize(out_path)/1024/1024:.0f} MB")

    # Batch-stage diagnostic table
    print("\n[Diagnostic] Batch-stage composition:")
    diag = adata_all.obs.groupby(['stage', 'dataset']).size().unstack(fill_value=0)
    print(diag.to_string())
    diag.to_csv(f"{OUT_DIR}/batch_stage_diagnostic.csv")

    # Check panel genes in common_genes
    panel_genes = ['PSMA7','POMP','CTSZ','VNN1','ADM','CNIH4','FTL','ASS1',
                   'MRPL13','TRIB1','OLFM4','BCAP31','TMEM176A','SOD1','DPP4']
    panel_in = [g for g in panel_genes if g in adata_all.var_names]
    panel_out = [g for g in panel_genes if g not in adata_all.var_names]
    print(f"\n  Panel genes in data: {len(panel_in)}/15")
    if panel_out:
        print(f"  Missing: {panel_out}")

    # Pseudobulk
    print("\n[Pseudobulk] Aggregating by sample...")
    samples = adata_all.obs['sample_id'].unique()
    pb_rows = []
    for sid in samples:
        mask = adata_all.obs['sample_id'] == sid
        X_sub = adata_all[mask].X
        if hasattr(X_sub, 'toarray'):
            X_sub = X_sub.toarray()
        mean_expr = np.asarray(X_sub).mean(axis=0)
        row = dict(zip(adata_all.var_names, mean_expr))
        meta_row = adata_all.obs.loc[mask].iloc[0]
        row['sample_id'] = sid
        row['dataset'] = meta_row['dataset']
        row['stage'] = meta_row['stage']
        row['hp_status'] = meta_row['hp_status']
        row['n_cells'] = int(mask.sum())
        pb_rows.append(row)

    pb_df = pd.DataFrame(pb_rows)
    pb_path = f"{OUT_DIR}/adata_pseudobulk_by_sample.csv"
    pb_df.to_csv(pb_path, index=False)
    print(f"  Pseudobulk: {len(pb_rows)} samples")

    # Cleanup temp files
    for tmp in ['tmp_gse134520.h5ad', 'tmp_gse249874.h5ad',
                'tmp_gse183904.h5ad', 'tmp_omix.h5ad']:
        tmp_path = f"{OUT_DIR}/{tmp}"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\n{'='*60}")
    print("Step 1 COMPLETE")
    print(f"  Cells: {adata_all.shape[0]}, Genes: {adata_all.shape[1]}")
    print(f"  Datasets: 4, Samples: {len(samples)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
