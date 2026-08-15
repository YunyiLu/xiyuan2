"""
B1: scVI training - Optimized v2
Changes from v1: fewer epochs (100), larger batch (256), Tensor Core, checkpoint
"""
import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

torch.set_float32_matmul_precision('medium')

OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"

print("=" * 60, flush=True)
print("B1: scVI Training v2 (100 epochs, batch=256)", flush=True)
print("=" * 60, flush=True)

# Step 1: HVG selection on subsample
print("\n[1] Selecting HVGs on subsample...", flush=True)
adata = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad", backed='r')
print(f"  Full shape: {adata.shape}", flush=True)

np.random.seed(42)
idx = np.sort(np.random.choice(adata.shape[0], 20000, replace=False))
adata_sub = adata[idx].to_memory()
adata.file.close()
del adata; gc.collect()

sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)
sc.pp.highly_variable_genes(adata_sub, n_top_genes=4000, batch_key='dataset')
hvg = adata_sub.var_names[adata_sub.var['highly_variable']].tolist()
print(f"  HVGs selected: {len(hvg)}", flush=True)
del adata_sub; gc.collect()

# Step 2: Load full data, subset to HVGs
print("\n[2] Loading full data (HVG subset)...", flush=True)
adata_full = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
adata_full.obs_names_make_unique()
print(f"  Loaded: {adata_full.shape}", flush=True)

adata_hvg = adata_full[:, hvg].copy()
del adata_full; gc.collect()
print(f"  HVG subset: {adata_hvg.shape}", flush=True)

adata_hvg.layers['counts'] = adata_hvg.X.copy()

# Save HVG list now (in case training crashes)
with open(f"{OUT_DIR}/hvg_list.txt", 'w') as f:
    f.write('\n'.join(hvg))
print("  HVG list saved", flush=True)

# Step 3: Setup scVI
print("\n[3] Setting up scVI...", flush=True)
scvi.model.SCVI.setup_anndata(adata_hvg, layer='counts', batch_key='dataset')
model = scvi.model.SCVI(
    adata_hvg, n_latent=30, gene_likelihood='zinb',
    n_layers=2, n_hidden=128,
)
print(f"  Model params: {sum(p.numel() for p in model.module.parameters()):,}", flush=True)

# Step 4: Train
print("\n[4] Training (100 epochs, batch_size=256)...", flush=True)
model.train(
    max_epochs=100,
    batch_size=256,
    early_stopping=True,
    early_stopping_patience=15,
    plan_kwargs={'lr': 1e-3},
    check_val_every_n_epoch=5,
)
print("  Training complete!", flush=True)

# Step 5: Save model + latent
print("\n[5] Saving model and latent...", flush=True)
model.save(f"{OUT_DIR}/scvi_model", overwrite=True)

latent = model.get_latent_representation()
latent_df = pd.DataFrame(latent, index=adata_hvg.obs_names,
                         columns=[f'scVI_{i}' for i in range(30)])
latent_df.to_csv(f"{OUT_DIR}/scvi_latent.csv")
print(f"  Latent: {latent.shape}", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"DONE: {adata_hvg.shape[0]} cells x {len(hvg)} HVGs", flush=True)
print("=" * 60, flush=True)
