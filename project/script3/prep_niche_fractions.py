"""
Pre-compute niche fractions from the full adata_integrated.h5ad.
Saves a lightweight CSV so Phase 22 never needs to touch the 2.7GB file.
"""
import pandas as pd
import h5py
from pathlib import Path

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
OUT = DATA / "niche_fractions.csv"

print("Reading obs metadata from adata_integrated.h5ad ...")
h5_path = DATA / "adata_integrated.h5ad"

with h5py.File(h5_path, "r") as f:
    obs = f["obs"]

    # Read sample_id (categorical)
    sid_grp = obs["sample_id"]
    if "categories" in sid_grp:
        cats = [c.decode() if isinstance(c, bytes) else c for c in sid_grp["categories"][...]]
        codes = sid_grp["codes"][...]
        sample_ids = [cats[c] for c in codes]
    else:
        sample_ids = [x.decode() if isinstance(x, bytes) else x for x in sid_grp[...]]

    # Read celltype (categorical)
    ct_grp = obs["celltype"]
    if "categories" in ct_grp:
        cats = [c.decode() if isinstance(c, bytes) else c for c in ct_grp["categories"][...]]
        codes = ct_grp["codes"][...]
        celltypes = [cats[c] if c >= 0 else "unknown" for c in codes]
    else:
        celltypes = [x.decode() if isinstance(x, bytes) else x for x in ct_grp[...]]

print(f"  Read {len(sample_ids)} cells")

# Build sample x celltype fraction table
ct_df = pd.DataFrame({"sample_id": sample_ids, "celltype": celltypes})
ct_counts = ct_df.groupby(["sample_id", "celltype"]).size().unstack(fill_value=0)
ct_fracs = ct_counts.div(ct_counts.sum(axis=1), axis=0)

# Aggregate into 3 niche categories
myeloid_cols = [c for c in ct_fracs.columns
                if any(k in c.lower() for k in ["mono", "macro", "dc", "mast", "myeloid"])]
fibro_cols = [c for c in ct_fracs.columns
              if any(k in c.lower() for k in ["fibro", "stroma"])]
tcell_cols = [c for c in ct_fracs.columns
              if any(k in c.lower() for k in ["t_cell", "t cell", "nk", "cd8", "cd4"])]

print(f"  Myeloid cols: {myeloid_cols}")
print(f"  Fibro cols: {fibro_cols}")
print(f"  T-cell cols: {tcell_cols}")

niche_df = pd.DataFrame({
    "sample_id": ct_fracs.index,
    "myeloid_fraction": ct_fracs[myeloid_cols].sum(axis=1).values if myeloid_cols else 0,
    "fibroblast_fraction": ct_fracs[fibro_cols].sum(axis=1).values if fibro_cols else 0,
    "T_cell_fraction": ct_fracs[tcell_cols].sum(axis=1).values if tcell_cols else 0,
})

niche_df.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(niche_df.describe())
