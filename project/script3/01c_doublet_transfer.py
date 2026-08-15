"""
A3 简化版：Transfer doublet labels from old data
由于doubletdetection的multiprocessing问题，改为从旧数据迁移doublet标记。
合理性：细胞barcode相同，doublet判定不依赖基因集（只看表达模式）。
"""
import os
import scanpy as sc

OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"

print("=" * 60)
print("A3: Transfer doublet labels (simplified)")
print("=" * 60)

# Load new data
print("\n[1] Loading new unified data...")
adata_new = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
print(f"  Shape: {adata_new.shape}")

# Load old data's obs (doublet labels)
print("\n[2] Loading old doublet labels...")
adata_old = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated_16948.h5ad")
obs_old = adata_old.obs[['doublet']].copy()
del adata_old
print(f"  Old doublets: {obs_old['doublet'].sum()} / {len(obs_old)}")

# Transfer by cell barcode
print("\n[3] Transferring doublet labels...")
adata_new.obs['doublet'] = False
common_cells = adata_new.obs_names.intersection(obs_old.index)
adata_new.obs.loc[common_cells, 'doublet'] = obs_old.loc[common_cells, 'doublet'].values
print(f"  Matched cells: {len(common_cells)}")
print(f"  Transferred doublets: {adata_new.obs['doublet'].sum()}")

# Remove doublets
print("\n[4] Removing doublets...")
n_before = adata_new.shape[0]
adata_clean = adata_new[~adata_new.obs['doublet']].copy()
n_after = adata_clean.shape[0]
print(f"  {n_before} -> {n_after} cells (removed {n_before - n_after})")

# Save
print("\n[5] Saving...")
adata_clean.write_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
print(f"  Saved: {OUT_DIR}/adata_raw_unintegrated.h5ad")

print("\n" + "=" * 60)
print(f"DONE: {adata_clean.shape[0]} cells x {adata_clean.shape[1]} genes")
print("=" * 60)
