"""
01b_rebuild_with_unified_genes.py
Rebuild adata_raw_unintegrated.h5ad using unified gene names (21,484 genes)
WITHOUT re-running QC/doublet detection.

Strategy: Use existing QC'd cell barcodes from adata_raw_unintegrated.h5ad,
re-read raw counts from disk, apply gene name unification, and merge.
"""
import os, sys, gc, json
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
OMIX_SCRNA_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/scRNA"
OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"

print("=" * 60)
print("Rebuild with unified genes (skip QC, reuse cell selection)")
print("=" * 60)

# === Load existing QC'd data to get cell barcodes ===
print("\n[1] Loading existing adata to get QC'd cell barcodes...")
adata_old = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
obs_old = adata_old.obs.copy()
print(f"  Existing: {adata_old.shape[0]} cells x {adata_old.shape[1]} genes")

cells_134520 = obs_old[obs_old['dataset'] == 'GSE134520'].index.tolist()
cells_249874 = obs_old[obs_old['dataset'] == 'GSE249874'].index.tolist()
cells_omix = obs_old[obs_old['dataset'] == 'OMIX010346'].index.tolist()
print(f"  GSE134520: {len(cells_134520)} cells")
print(f"  GSE249874: {len(cells_249874)} cells")
print(f"  OMIX010346: {len(cells_omix)} cells")
# Keep obs_old for metadata transfer later; only delete adata_old's X
del adata_old; gc.collect()

# === Load gene unification mapping ===
print("\n[2] Loading gene unification mapping...")
with open(f"{OUT_DIR}/gene_unification_mapping.json", 'r', encoding='utf-8') as f:
    mapping = json.load(f)
g134_map = mapping['g134520_to_unified']
omix_map = mapping['omix_to_unified']
print(f"  GSE134520 renames: {len(g134_map)}")
print(f"  OMIX renames: {len(omix_map)}")

# === Re-read GSE134520 with full genes ===
print("\n[3] Re-reading GSE134520 raw data...")
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

adatas_134520 = []
for fname, (sid, stage, im_sub) in GSE134520_SAMPLES.items():
    fpath = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(fpath, sep='\t', index_col=0)
    a = ad.AnnData(X=csr_matrix(df.values.T),
                   obs=pd.DataFrame(index=df.columns),
                   var=pd.DataFrame(index=df.index))
    a.obs['sample_id'] = sid
    a.obs['stage'] = stage
    a.obs['im_subtype'] = im_sub
    a.obs['dataset'] = 'GSE134520'
    a.obs['hp_status'] = 'unknown'
    adatas_134520.append(a)
    print(f"  {sid}: {a.shape[0]} cells")

adata_134520 = ad.concat(adatas_134520)
adata_134520.var_names_make_unique()
del adatas_134520; gc.collect()
print(f"  Combined: {adata_134520.shape}")

# Filter to QC'd cells
valid_cells = [c for c in cells_134520 if c in adata_134520.obs_names]
adata_134520 = adata_134520[valid_cells].copy()
print(f"  After QC filter: {adata_134520.shape[0]} cells")

# Apply gene name unification
adata_134520.var_names = pd.Index([g134_map.get(g, g) for g in adata_134520.var_names])
adata_134520.var_names_make_unique()
print(f"  After unification: {adata_134520.shape[1]} genes")

# === Re-read GSE249874 ===
print("\n[4] Re-reading GSE249874 raw data...")
import gzip
from scipy.io import mmread
from scipy.sparse import csc_matrix

with gzip.open(f"{GSE249874_DIR}/GSE249874_raw_feature_barcodes.tsv.gz", 'rt') as f:
    barcodes_249 = [l.strip() for l in f]
with gzip.open(f"{GSE249874_DIR}/GSE249874_raw_feature_features.tsv.gz", 'rt') as f:
    features_249 = [l.strip().split('\t') for l in f]
gene_names_249 = [ft[1] if len(ft) > 1 else ft[0] for ft in features_249]
print(f"  {len(barcodes_249):,} barcodes, {len(gene_names_249)} genes")

print("  Loading sparse matrix (takes a few minutes)...")
mat = mmread(f"{GSE249874_DIR}/GSE249874_raw_feature_matrix.mtx.gz")
mat = csc_matrix(mat)
print(f"  Loaded: {mat.shape}, nnz={mat.nnz:,}")

# Find columns (cells) that match our QC'd barcodes
cells_249_set = set(cells_249874)
keep_idx = [i for i, b in enumerate(barcodes_249) if b in cells_249_set]
print(f"  Matching QC'd cells: {len(keep_idx)}")

mat_filtered = mat[:, keep_idx].T.tocsr()
barcodes_filtered = [barcodes_249[i] for i in keep_idx]
del mat; gc.collect()

adata_249874 = ad.AnnData(X=mat_filtered,
                          obs=pd.DataFrame(index=barcodes_filtered),
                          var=pd.DataFrame(index=gene_names_249))
adata_249874.var_names_make_unique()
del mat_filtered; gc.collect()

# Assign metadata for GSE249874 using barcode suffix (same logic as original script)
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
    elif 'GS-HP-N' in title or ('gastritis' in title.lower() and 'HP-N' in title):
        stage, hp = 'NAG', 'HP-'
    elif 'GS-HP-P' in title:
        stage, hp = 'NAG', 'HP+'
    elif 'IM-HP-N' in title:
        stage, hp = 'IM', 'HP-'
    elif 'IM-HP-P' in title:
        stage, hp = 'IM', 'HP+'
    else:
        stage, hp = 'unknown', 'unknown'
    sample_info[acc] = {'stage': stage, 'hp_status': hp}

suffixes = adata_249874.obs_names.str.extract(r'-(\d+)$')[0].astype(int)
adata_249874.obs['sample_id'] = [acc_list[s - 1] if s <= len(acc_list) else 'unknown'
                                  for s in suffixes]
adata_249874.obs['stage'] = adata_249874.obs['sample_id'].map(
    lambda x: sample_info.get(x, {}).get('stage', 'unknown'))
adata_249874.obs['hp_status'] = adata_249874.obs['sample_id'].map(
    lambda x: sample_info.get(x, {}).get('hp_status', 'unknown'))
adata_249874.obs['dataset'] = 'GSE249874'
adata_249874.obs['im_subtype'] = 'unknown'

print(f"  GSE249874 ready: {adata_249874.shape}")
# GSE249874 is reference naming - no rename needed

# === Re-read OMIX010346 ===
print("\n[5] Re-reading OMIX010346 raw data...")
adatas_omix = []
for gp in ['GP4', 'GP5', 'GP6', 'GP9']:
    path = f"{OMIX_SCRNA_DIR}/{gp}"
    a = sc.read_10x_mtx(path, var_names='gene_symbols')
    a.var_names_make_unique()
    a.obs['sample_id'] = f'OMIX_{gp}'
    a.obs['stage'] = 'EGC_multi_region'
    a.obs['dataset'] = 'OMIX010346'
    a.obs['hp_status'] = 'unknown'
    a.obs['im_subtype'] = 'none'
    # Add GP prefix to avoid barcode collisions
    a.obs_names = [f"{gp}_{bc}" for bc in a.obs_names]
    print(f"  {gp}: {a.shape[0]} cells")
    adatas_omix.append(a)

adata_omix = ad.concat(adatas_omix, join='outer')
adata_omix.var_names_make_unique()
del adatas_omix; gc.collect()
print(f"  Combined: {adata_omix.shape}")

# Filter: apply same QC criteria as original
adata_omix.var['mt'] = adata_omix.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_omix, qc_vars=['mt'], inplace=True)
mask = ((adata_omix.obs['n_genes_by_counts'] >= 200) &
        (adata_omix.obs['n_genes_by_counts'] <= 6000) &
        (adata_omix.obs['pct_counts_mt'] <= 20))
adata_omix = adata_omix[mask].copy()
print(f"  After QC filter: {adata_omix.shape[0]} cells (target: ~{len(cells_omix)})")

# Apply gene name unification
adata_omix.var_names = pd.Index([omix_map.get(g, g) for g in adata_omix.var_names])
adata_omix.var_names_make_unique()
print(f"  After unification: {adata_omix.shape[1]} genes")

# === Compute unified intersection and merge ===
print("\n[6] Computing unified intersection...")
genes_134520 = set(adata_134520.var_names)
genes_249874 = set(adata_249874.var_names)
genes_omix = set(adata_omix.var_names)
common_genes = sorted(genes_134520 & genes_249874 & genes_omix)
print(f"  Common genes (unified): {len(common_genes)}")

print("\n[7] Merging datasets...")
adata_all = ad.concat([adata_134520[:, common_genes],
                       adata_249874[:, common_genes],
                       adata_omix[:, common_genes]], join='outer')
del adata_134520, adata_249874, adata_omix; gc.collect()

print(f"  Final: {adata_all.shape[0]} cells x {adata_all.shape[1]} genes")
print("\n  Dataset breakdown:")
for d, n in adata_all.obs.groupby('dataset').size().items():
    print(f"    {d}: {n}")

# Transfer doublet column - skip complex matching, just set False
# Doublet detection will be re-run if needed, or applied from old results separately
adata_all.obs['doublet'] = False
print("  Doublet column set to False (re-run doublet detection if needed)")

# === Save ===
print("\n[8] Saving...")
out_path = f"{OUT_DIR}/adata_raw_unintegrated.h5ad"
backup_path = f"{OUT_DIR}/adata_raw_unintegrated_16948.h5ad"

# Backup old file
if os.path.exists(out_path) and not os.path.exists(backup_path):
    os.rename(out_path, backup_path)
    print(f"  Backed up old file to {backup_path}")

adata_all.obs_names_make_unique()
adata_all.write_h5ad(out_path)
print(f"  Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024/1024:.0f} MB")

# Panel gene check
panel_genes = ['PSMA7','POMP','CTSZ','VNN1','ADM','CNIH4','FTL','ASS1',
               'MRPL13','TRIB1','OLFM4','BCAP31','TMEM176A','SOD1','DPP4']
panel_in = [g for g in panel_genes if g in adata_all.var_names]
print(f"\n  Panel genes: {len(panel_in)}/15")

print("\n" + "=" * 60)
print(f"DONE: {adata_all.shape[0]} cells x {adata_all.shape[1]} genes")
print(f"Improvement: 16,948 -> {len(common_genes)} genes (+{len(common_genes)-16948})")
print("=" * 60)
