"""
B2-B3: Cell annotation + continuous scores
Loads scVI latent from 02a output, builds UMAP, annotates, scores.
"""
import os, sys, gc, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

sc.settings.verbosity = 1
BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
OUT_DIR = f"{BASE}/data"
os.makedirs(f"{BASE}/results/figures", exist_ok=True)

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

EPITHELIAL_TYPES = {"Gastric_mucous", "Chief_glandular", "Enterocyte_IM",
                    "Goblet_IM", "Stem_proliferative"}

SCORES_LAYER2 = {
    "PMC_2_score": ["NAMPT", "ALDH1A1", "CD44", "SOX9", "OLFM4"],
    "PMC_P_score": ["AREG", "NAMPT", "PHLDA1", "ITGA2", "MYC"],
    "stemness_score": ["LGR5", "OLFM4", "SOX9", "ASCL2"],
    "proliferation_score": ["MKI67", "TOP2A", "PCNA", "CDK1"],
    "IM_score": ["CDX2", "MUC2", "TFF3", "VIL1"],
    "EMT_score": ["VIM", "SNAI1", "ZEB1", "CDH2"],
    "EGC_like_score": ["REG4","CEACAM6","MUC13","CLDN3","EPCAM","KRT20",
                       "ERBB2","MET","VEGFA"],
}

print("=" * 60, flush=True)
print("B2-B3: Cell Annotation + Continuous Scores", flush=True)
print("=" * 60, flush=True)

# Step 1: Load data + scVI latent
print("\n[1] Loading data...", flush=True)
adata = sc.read_h5ad(f"{OUT_DIR}/adata_raw_unintegrated.h5ad")
adata.obs_names_make_unique()
print(f"  {adata.shape[0]} cells x {adata.shape[1]} genes", flush=True)

latent_df = pd.read_csv(f"{OUT_DIR}/scvi_latent.csv", index_col=0)
common = adata.obs_names.intersection(latent_df.index)
adata = adata[common].copy()
adata.obsm['X_scVI'] = latent_df.loc[common].values
print(f"  Latent loaded: {adata.obsm['X_scVI'].shape}", flush=True)

# Step 2: Normalize for marker scoring
print("\n[2] Normalizing for scoring...", flush=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Step 3: Neighbors + UMAP + Leiden
print("\n[3] Neighbors + UMAP + Leiden...", flush=True)
sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=30)
sc.tl.umap(adata)
for res in [0.8, 1.2, 2.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
n_cl = adata.obs['leiden_0.8'].nunique()
print(f"  Leiden 0.8: {n_cl} clusters", flush=True)

# Step 4: Cell type annotation
print("\n[4] Cell type annotation...", flush=True)
key = 'leiden_0.8'
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
    if "EPCAM" in adata.var_names:
        epcam = adata[mask][:, "EPCAM"].X
        if hasattr(epcam, 'toarray'):
            epcam = epcam.toarray()
        if np.mean(epcam) > 0.5:
            for stromal in ["Fibroblast", "Endothelial", "Pericyte"]:
                scores[stromal] = 0.0
    best = max(scores, key=scores.get)
    if scores[best] < 0.1:
        best = "Unassigned"
    cluster_labels[cl] = best

adata.obs['celltype'] = adata.obs[key].map(cluster_labels).astype(str)
adata.obs['is_epithelial'] = adata.obs['celltype'].isin(EPITHELIAL_TYPES)
print(f"  Cell types:", flush=True)
for ct, n in adata.obs['celltype'].value_counts().items():
    print(f"    {ct}: {n}", flush=True)
