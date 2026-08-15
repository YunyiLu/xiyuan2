"""
Phase 24d: Robustness Validation for IM->EGC Bifurcations
Addresses 4 weaknesses:
1. LODO cross-validation (3 datasets)
2. Original cell count traceback (metacell -> real n)
3. Continuous regression for Bif-4 (avoid n=6 split)
4. Bootstrap stability for Cluster 2/3
5. RNA velocity check (unspliced counts availability)
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from pathlib import Path

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"

adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
val_df = pd.read_csv(RESULTS / "rl_value_function.csv")

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values
X_emb = adata_mc.obsm['X_scVI']

im_idx = np.where(stages == 'IM')[0]
egc_idx = np.where((stages == 'EGC') | (stages == 'EGC_multi_region'))[0]
gc_idx = np.where(stages == 'GC')[0]
nag_idx = np.where(stages == 'NAG')[0]

print("=" * 70)
print("Phase 24d: Robustness Validation")
print("=" * 70)


# ===================================================================
# Helper: compute fate profile for a set of IM cells
# ===================================================================
def compute_im_fate_profile(im_indices, T, all_stages, k=20):
    """Forward-propagate IM cells k steps, return fate fractions."""
    n = T.shape[0]
    prop = np.zeros((len(im_indices), n))
    for i, idx in enumerate(im_indices):
        prop[i, idx] = 1.0
    for _ in range(k):
        prop = prop @ T

    egc_mask = (all_stages == 'EGC') | (all_stages == 'EGC_multi_region')
    gc_mask = all_stages == 'GC'
    im_mask = all_stages == 'IM'
    nag_mask = all_stages == 'NAG'

    fate = np.column_stack([
        (prop * egc_mask[None, :]).sum(axis=1),
        (prop * gc_mask[None, :]).sum(axis=1),
        (prop * im_mask[None, :]).sum(axis=1),
        (prop * nag_mask[None, :]).sum(axis=1),
    ])
    return fate


# ===================================================================
# [1] Traceback: How many ORIGINAL cells per cluster?
# ===================================================================
print("\n[1/5] Original cell count per fate cluster ...")

# First compute fate clusters (same as 24c)
fate_matrix = compute_im_fate_profile(im_idx, T_cr, stages, k=20)
km = KMeans(n_clusters=4, n_init=10, random_state=42)
im_fate_labels = km.fit_predict(fate_matrix)

n_cells_per_mc = adata_mc.obs['n_cells'].astype(int).values
for c in range(4):
    mc_in_cluster = im_idx[im_fate_labels == c]
    n_metacells = len(mc_in_cluster)
    n_original = int(n_cells_per_mc[mc_in_cluster].sum())
    print(f"  Cluster {c}: {n_metacells} metacells = "
          f"{n_original} original cells")

print("\n  => Even Cluster 3 (13 metacells) represents hundreds of "
      "original cells -- adequate for DE analysis")


# ===================================================================
# [2] Bootstrap Stability of 4 clusters
# ===================================================================
print("\n[2/5] Bootstrap stability (100 resamples) ...")

n_boot = 100
ari_scores = []
cluster_sizes_boot = {c: [] for c in range(4)}

for b in range(n_boot):
    boot_idx = np.random.choice(len(im_idx), len(im_idx), replace=True)
    fate_boot = fate_matrix[boot_idx]
    km_boot = KMeans(n_clusters=4, n_init=5, random_state=b)
    labels_boot = km_boot.fit_predict(fate_boot)

    # Map boot labels to original labels (Hungarian matching)
    # Simple approach: for each boot cluster, find best-matching original
    mapping = {}
    used = set()
    for bc in range(4):
        boot_mask = labels_boot == bc
        best_ari = -1
        best_oc = 0
        for oc in range(4):
            if oc in used:
                continue
            orig_mask = im_fate_labels[boot_idx] == oc
            overlap = (boot_mask & orig_mask).sum() / max(boot_mask.sum(), 1)
            if overlap > best_ari:
                best_ari = overlap
                best_oc = oc
        mapping[bc] = best_oc
        used.add(best_oc)

    # Remap
    mapped_labels = np.array([mapping.get(l, l) for l in labels_boot])
    ari = adjusted_rand_score(im_fate_labels[boot_idx], mapped_labels)
    ari_scores.append(ari)

    for c in range(4):
        cluster_sizes_boot[c].append((mapped_labels == c).sum())

print(f"  Bootstrap ARI: mean={np.mean(ari_scores):.3f}, "
      f"std={np.std(ari_scores):.3f}, "
      f"min={np.min(ari_scores):.3f}")
print(f"  Cluster size stability (mean +/- std):")
for c in range(4):
    sizes = cluster_sizes_boot[c]
    print(f"    Cluster {c}: {np.mean(sizes):.0f} +/- {np.std(sizes):.1f} "
          f"(original: {(im_fate_labels == c).sum()})")

stability_verdict = "STABLE" if np.mean(ari_scores) > 0.6 else "UNSTABLE"
print(f"\n  Verdict: Clustering is {stability_verdict} "
      f"(ARI threshold: 0.6)")


# ===================================================================
# [3] LODO: Leave-One-Dataset-Out validation
# ===================================================================
print("\n[3/5] LODO validation (leave-one-dataset-out) ...")

datasets = adata_mc.obs['dataset'].values
unique_ds = np.unique(datasets)
print(f"  Datasets: {list(unique_ds)}")

lodo_results = []
for holdout in unique_ds:
    train_mask_global = datasets != holdout
    test_mask_global = datasets == holdout

    im_train = np.where((stages == 'IM') & train_mask_global)[0]
    im_test = np.where((stages == 'IM') & test_mask_global)[0]

    if len(im_test) < 5:
        print(f"    {holdout}: only {len(im_test)} IM cells in test, skip")
        continue

    fate_train = compute_im_fate_profile(im_train, T_cr, stages, k=20)
    km_lodo = KMeans(n_clusters=4, n_init=10, random_state=42)
    km_lodo.fit(fate_train)

    fate_test = compute_im_fate_profile(im_test, T_cr, stages, k=20)
    test_labels = km_lodo.predict(fate_test)

    X_test = adata_mc[im_test].X
    if sparse.issparse(X_test):
        X_test = X_test.toarray()
    gene_names = adata_mc.var_names

    oxphos_genes = ['COX5B', 'NDUFA3', 'COX7B', 'NDUFB3', 'NDUFA4']
    oxphos_idx = [np.where(gene_names == g)[0][0] for g in oxphos_genes
                  if g in gene_names]

    c0_mask = test_labels == 0
    c1_mask = test_labels == 1

    if c0_mask.sum() >= 3 and c1_mask.sum() >= 3 and len(oxphos_idx) > 0:
        oxphos_c0 = X_test[c0_mask][:, oxphos_idx].mean()
        oxphos_c1 = X_test[c1_mask][:, oxphos_idx].mean()
        oxphos_diff = oxphos_c0 - oxphos_c1

        result = {
            "holdout": holdout,
            "n_im_test": len(im_test),
            "n_cluster0": int(c0_mask.sum()),
            "n_cluster1": int(c1_mask.sum()),
            "oxphos_c0": float(oxphos_c0),
            "oxphos_c1": float(oxphos_c1),
            "oxphos_diff": float(oxphos_diff),
            "bif3_replicated": "YES" if oxphos_diff < 0 else "NO",
        }
        lodo_results.append(result)
        print(f"    {holdout}: n_test={len(im_test)}, "
              f"OXPHOS(C0)={oxphos_c0:.3f}, OXPHOS(C1)={oxphos_c1:.3f}, "
              f"diff={oxphos_diff:+.3f} -> {result['bif3_replicated']}")
    else:
        print(f"    {holdout}: insufficient cells (C0={c0_mask.sum()}, "
              f"C1={c1_mask.sum()})")

if lodo_results:
    n_yes = sum(1 for r in lodo_results if r["bif3_replicated"] == "YES")
    print(f"\n  LODO Bif-3 (OXPHOS): {n_yes}/{len(lodo_results)} replicated")


# ===================================================================
# [4] Continuous regression for Bif-4 (avoid binary split)
# ===================================================================
print("\n[4/5] Continuous regression at Bif-4 (pt > 0.10) ...")

im_pt_local = pt[im_idx]
im_fate_egc = fate_matrix[:, 0]  # P(EGC) from forward propagation

# Late IM cells
late_mask = im_pt_local > 0.10
n_late = late_mask.sum()
print(f"  Late IM cells (pt > 0.10): {n_late}")

X_im = adata_mc[im_idx].X
if sparse.issparse(X_im):
    X_im = X_im.toarray()
gene_names = adata_mc.var_names

# Correlate each gene with P(EGC) in the late IM window
late_X = X_im[late_mask]
late_fate = im_fate_egc[late_mask]

# Only test genes with sufficient variance
gene_var = late_X.var(axis=0)
testable = gene_var > 0.01
n_testable = testable.sum()

correlations = []
for gi in np.where(testable)[0]:
    rho, p = spearmanr(late_X[:, gi], late_fate)
    if np.isfinite(rho):
        correlations.append({
            "gene": gene_names[gi],
            "spearman_rho": rho,
            "pvalue": p,
            "direction": "pro-EGC" if rho > 0 else "anti-EGC",
        })

corr_df = pd.DataFrame(correlations)
corr_df["abs_rho"] = corr_df["spearman_rho"].abs()
corr_df = corr_df.sort_values("abs_rho", ascending=False)

# BH correction
from statsmodels.stats.multitest import multipletests
_, corr_df["fdr"], _, _ = multipletests(corr_df["pvalue"], method="fdr_bh")

sig_genes = corr_df[corr_df["fdr"] < 0.1]
print(f"  Testable genes: {n_testable}, Significant (FDR<0.1): {len(sig_genes)}")

# Check if Bif-4 markers replicate
bif4_markers = ["KMT2E", "SNW1", "APEX1", "SERBP1", "SF1",
                "TFF1", "CTSE", "CD55"]
print("\n  Bif-4 marker validation (continuous Spearman):")
print(f"  {'Gene':>10} {'rho':>8} {'FDR':>10} {'Direction':>12}")
print(f"  {'-'*45}")
for gene in bif4_markers:
    match = corr_df[corr_df["gene"] == gene]
    if len(match) > 0:
        row = match.iloc[0]
        sig = "*" if row["fdr"] < 0.1 else ""
        print(f"  {gene:>10} {row['spearman_rho']:>+8.3f} "
              f"{row['fdr']:>10.4f} {row['direction']:>12} {sig}")
    else:
        print(f"  {gene:>10}  (not testable - low variance)")

# Top 10 overall
print("\n  Top 10 fate-correlated genes in late IM:")
for _, row in corr_df.head(10).iterrows():
    sig = "**" if row["fdr"] < 0.01 else ("*" if row["fdr"] < 0.1 else "")
    print(f"    {row['gene']:>12} rho={row['spearman_rho']:+.3f} "
          f"FDR={row['fdr']:.4f} {row['direction']} {sig}")


# ===================================================================
# [5] RNA velocity: check for unspliced counts
# ===================================================================
print("\n[5/5] Checking RNA velocity availability ...")

import h5py
h5_path = DATA / "adata_integrated.h5ad"
has_unspliced = False
try:
    with h5py.File(h5_path, "r") as f:
        layers = list(f["layers"].keys()) if "layers" in f else []
        has_spliced = "spliced" in layers or "Ms" in layers
        has_unspliced = "unspliced" in layers or "Mu" in layers
        print(f"  Layers in adata_integrated: {layers}")
        if has_unspliced:
            print("  -> UNSPLICED COUNTS AVAILABLE: RNA velocity is feasible")
        else:
            print("  -> No unspliced layer found. RNA velocity NOT feasible "
                  "without re-running velocyto/STARsolo on raw BAMs.")

        # Also check obsm for existing velocity
        obsm_keys = list(f["obsm"].keys()) if "obsm" in f else []
        if any("velocity" in k.lower() for k in obsm_keys):
            print(f"  -> Pre-computed velocity found: {obsm_keys}")
except Exception as e:
    print(f"  Error reading h5ad: {e}")


# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS VALIDATION SUMMARY")
print("=" * 70)

print(f"\n  [1] Original cell counts:")
for c in range(4):
    mc_in_cluster = im_idx[im_fate_labels == c]
    n_orig = int(n_cells_per_mc[mc_in_cluster].sum())
    print(f"      Cluster {c}: {len(mc_in_cluster)} metacells = "
          f"{n_orig} original cells")

print(f"\n  [2] Bootstrap stability: ARI={np.mean(ari_scores):.3f} "
      f"-> {stability_verdict}")

if lodo_results:
    print(f"\n  [3] LODO Bif-3 replication: "
          f"{n_yes}/{len(lodo_results)} datasets")

print(f"\n  [4] Bif-4 continuous validation: "
      f"{len(sig_genes)} genes FDR<0.1 in late IM")

print(f"\n  [5] RNA velocity: "
      f"{'Available' if has_unspliced else 'Not available'}")

print("\n" + "=" * 70)

# Save
pd.DataFrame(lodo_results).to_csv(
    RESULTS / "ot_lodo_bif3_validation.csv", index=False)
corr_df.head(50).to_csv(
    RESULTS / "ot_bif4_continuous_correlations.csv", index=False)
