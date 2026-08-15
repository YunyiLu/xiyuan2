"""
A3: Doublet Detection on unified dataset
Run per-sample doublet detection using doubletdetection.BoostClassifier.
Operates on adata_raw_unintegrated.h5ad (190,200 cells x 21,486 genes).
"""
import os, sys, gc
import numpy as np
import scanpy as sc
import anndata as ad
import doubletdetection
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"

print("=" * 60)
print("A3: Doublet Detection (per-sample)")
print("=" * 60)

# Load unified data
print("\n[1] Loading unified data...")
adata = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
adata.obs_names_make_unique()
print(f"  Shape: {adata.shape}")
print(f"  Samples: {adata.obs['sample_id'].nunique()}")

# Per-sample doublet detection
print("\n[2] Running doublet detection per sample...")
adata.obs['doublet_score'] = 0.0
adata.obs['doublet'] = False

samples = adata.obs['sample_id'].unique()
total_doublets = 0

for i, sid in enumerate(samples):
    mask = adata.obs['sample_id'] == sid
    n_cells = mask.sum()
    if n_cells < 100:
        print(f"  [{i+1}/{len(samples)}] {sid}: {n_cells} cells (SKIP, too few)")
        continue

    X_sub = adata[mask].X
    if hasattr(X_sub, 'toarray'):
        X_sub = X_sub.toarray()

    try:
        clf = doubletdetection.BoostClassifier(
            n_iters=10,
            standard_scaling=True,
            random_state=0,
            n_jobs=1  # Disable multiprocessing to avoid pagefile issues
        )
        labels = clf.fit(X_sub).predict()
        n_doublets = int(labels.sum())
        total_doublets += n_doublets
        pct = n_doublets / n_cells * 100

        adata.obs.loc[mask, 'doublet'] = labels.astype(bool)
        adata.obs.loc[mask, 'doublet_score'] = clf.doublet_score()

        print(f"  [{i+1}/{len(samples)}] {sid}: {n_cells} cells, "
              f"{n_doublets} doublets ({pct:.1f}%)")
    except Exception as e:
        print(f"  [{i+1}/{len(samples)}] {sid}: ERROR - {str(e)[:80]}")

    del X_sub
    gc.collect()

# Summary
print(f"\n[3] Summary:")
print(f"  Total cells: {adata.shape[0]}")
print(f"  Total doublets: {total_doublets} ({total_doublets/adata.shape[0]*100:.1f}%)")

# Remove doublets
adata_clean = adata[~adata.obs['doublet']].copy()
print(f"  After removal: {adata_clean.shape[0]} cells")

# Save
print("\n[4] Saving...")
out_path = f"{OUT_DIR}/adata_raw_unintegrated.h5ad"
adata_clean.write_h5ad(out_path)
print(f"  Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024/1024:.0f} MB")

# Also save doublet scores for reference
adata.obs[['sample_id', 'dataset', 'doublet', 'doublet_score']].to_csv(
    f"{OUT_DIR}/doublet_scores.csv")
print(f"  Saved: {OUT_DIR}/doublet_scores.csv")

print("\n" + "=" * 60)
print(f"DONE: {adata.shape[0]} -> {adata_clean.shape[0]} cells "
      f"(removed {total_doublets} doublets)")
print("=" * 60)
