"""
Focused: IM->EGC bifurcation via sub-trajectory divergence.
Key insight: within IM, all cells progress toward EGC,
but they may take DIFFERENT ROUTES (sub-trajectories).
Bifurcation = where initially similar IM cells diverge into
distinct phenotypic sub-trajectories.

Strategy:
1. Cluster IM cells into sub-populations using embedding
2. Trace forward fate of each sub-pop via T matrix
3. Where sub-pops diverge in their downstream targets = bifurcation
4. Also: use multi-fate model (EGC vs GC vs CAG-like stasis)
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

adata_mc = sc.read_h5ad('data/rl_metacells.h5ad')
T_sparse = sparse.load_npz('results/rl_transition_matrix.npz')
T_cr = T_sparse.toarray()
val_df = pd.read_csv('results/rl_value_function.csv')

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values
X_emb = adata_mc.obsm['X_scVI']

print("=" * 70)
print("IM->EGC Bifurcation Analysis via Sub-trajectory Divergence")
print("=" * 70)

# Define terminal fates (using ALL stages, not just IM+EGC)
egc_idx = np.where((stages == 'EGC') | (stages == 'EGC_multi_region'))[0]
gc_idx = np.where(stages == 'GC')[0]
nag_idx = np.where(stages == 'NAG')[0]
cag_idx = np.where(stages == 'CAG')[0]
im_idx = np.where(stages == 'IM')[0]

print(f"IM: {len(im_idx)}, EGC: {len(egc_idx)}, GC: {len(gc_idx)}, "
      f"NAG: {len(nag_idx)}, CAG: {len(cag_idx)}")

# ===================================================================
# Method 1: Multi-step forward propagation from IM cells
# After k steps of T, where does each IM cell end up?
# ===================================================================
print("\n[1] Multi-step forward propagation from IM cells ...")

# Propagate IM cells forward through T for k steps
# At each step, record the "stage distribution" each cell reaches
k_steps = [5, 10, 15, 20, 30]
im_start = np.zeros((len(im_idx), T_cr.shape[0]))
for i, idx in enumerate(im_idx):
    im_start[i, idx] = 1.0  # delta distribution at start

# Stage membership vectors
stage_membership = {}
for s in ['NAG', 'CAG', 'IM', 'EGC', 'EGC_multi_region', 'GC']:
    mask = stages == s
    stage_membership[s] = mask.astype(float)

print("  Propagating IM cells forward ...")
prop = im_start.copy()
fate_trajectories = {}

for k in range(1, max(k_steps) + 1):
    prop = prop @ T_cr
    if k in k_steps:
        # For each IM cell, what fraction of its probability mass
        # is in each stage after k steps?
        stage_fracs = {}
        for s, mem in stage_membership.items():
            stage_fracs[s] = (prop * mem[None, :]).sum(axis=1)
        fate_trajectories[k] = pd.DataFrame(stage_fracs, index=range(len(im_idx)))

        total_egc = stage_fracs['EGC'] + stage_fracs['EGC_multi_region']
        total_gc = stage_fracs['GC']
        total_im = stage_fracs['IM']
        print(f"    k={k:2d}: mean P(EGC)={total_egc.mean():.3f}, "
              f"P(GC)={total_gc.mean():.3f}, P(IM)={total_im.mean():.3f}")

# Use k=20 as the "fate horizon"
fate_k20 = fate_trajectories[20]
im_fate_egc = fate_k20['EGC'].values + fate_k20['EGC_multi_region'].values
im_fate_gc = fate_k20['GC'].values
im_fate_im = fate_k20['IM'].values
im_fate_nag = fate_k20['NAG'].values

# ===================================================================
# Method 2: Cluster IM cells by their k=20 fate profile
# Different clusters = different sub-trajectories
# ===================================================================
print("\n[2] Clustering IM cells by fate profile ...")

fate_matrix = np.column_stack([im_fate_egc, im_fate_gc, im_fate_im, im_fate_nag])
# How many distinct fate clusters?
from sklearn.metrics import silhouette_score

best_k, best_sil = 2, -1
for nc in range(2, 7):
    km = KMeans(n_clusters=nc, n_init=10, random_state=42)
    labels = km.fit_predict(fate_matrix)
    sil = silhouette_score(fate_matrix, labels)
    if sil > best_sil:
        best_k, best_sil = nc, sil
    print(f"    k={nc}: silhouette={sil:.3f}")

print(f"  Best: {best_k} clusters (silhouette={best_sil:.3f})")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
im_fate_labels = km.fit_predict(fate_matrix)

print(f"\n  Fate cluster profiles (k=20 forward prop):")
print(f"  {'Cluster':>8} {'n':>4} {'P(EGC)':>8} {'P(GC)':>8} {'P(IM)':>8} "
      f"{'P(NAG)':>8} {'meanPT':>8} {'meanV':>8}")
print("  " + "-" * 65)

im_pt_local = pt[im_idx]
im_V_local = V[im_idx]

cluster_profiles = []
for c in range(best_k):
    mask = im_fate_labels == c
    profile = {
        "cluster": c,
        "n": int(mask.sum()),
        "P_EGC": im_fate_egc[mask].mean(),
        "P_GC": im_fate_gc[mask].mean(),
        "P_IM": im_fate_im[mask].mean(),
        "P_NAG": im_fate_nag[mask].mean(),
        "mean_pt": im_pt_local[mask].mean(),
        "mean_V": im_V_local[mask].mean(),
    }
    cluster_profiles.append(profile)
    print(f"  {c:>8} {mask.sum():>4} {profile['P_EGC']:>8.3f} "
          f"{profile['P_GC']:>8.3f} {profile['P_IM']:>8.3f} "
          f"{profile['P_NAG']:>8.3f} {profile['mean_pt']:>8.4f} "
          f"{profile['mean_V']:>8.3f}")

# ===================================================================
# Method 3: Pseudotime-resolved divergence within IM
# At each pseudotime bin, how diverse are the fate profiles?
# ===================================================================
print("\n[3] Pseudotime-resolved fate divergence within IM ...")

N_BINS = 10
bin_edges = np.percentile(im_pt_local, np.linspace(0, 100, N_BINS + 1))
bin_edges[-1] += 1e-10
bin_labels = np.digitize(im_pt_local, bin_edges) - 1
bin_labels = np.clip(bin_labels, 0, N_BINS - 1)

print(f"\n  {'Bin':>4} {'PT range':>22} {'n':>4} {'P(EGC)var':>10} "
      f"{'ClustDiv':>9} {'V_std':>6}")
print("  " + "-" * 60)

bin_records = []
for b in range(N_BINS):
    mask = bin_labels == b
    if mask.sum() < 5:
        continue

    # Fate variance: how much cells in this bin disagree about destination
    egc_var = im_fate_egc[mask].var()
    gc_var = im_fate_gc[mask].var()
    total_var = egc_var + gc_var

    # Cluster diversity: entropy of cluster label distribution
    cluster_counts = np.bincount(im_fate_labels[mask], minlength=best_k)
    cluster_probs = cluster_counts / cluster_counts.sum()
    cluster_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

    v_std = im_V_local[mask].std()
    pt_lo = im_pt_local[mask].min()
    pt_hi = im_pt_local[mask].max()

    bin_records.append({
        "bin": b, "pt_lo": pt_lo, "pt_hi": pt_hi,
        "pt_mean": im_pt_local[mask].mean(),
        "n": int(mask.sum()),
        "fate_variance": total_var,
        "cluster_entropy": cluster_entropy,
        "V_std": v_std,
        "mean_P_EGC": im_fate_egc[mask].mean(),
        "mean_P_GC": im_fate_gc[mask].mean(),
    })

    print(f"  {b:>4} [{pt_lo:.4f},{pt_hi:.4f}] {mask.sum():>4} "
          f"{total_var:>10.4f} {cluster_entropy:>9.3f} {v_std:>6.3f}")

bin_df = pd.DataFrame(bin_records)

# Bifurcation = where fate_variance AND cluster_entropy are both high
bin_df["bif_score"] = bin_df["fate_variance"] * bin_df["cluster_entropy"] * (1 + bin_df["V_std"])

# Find peaks
scores = bin_df["bif_score"].values
peaks = []
for i in range(1, len(scores) - 1):
    if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
        if scores[i] > np.percentile(scores, 40):
            peaks.append(i)
if len(scores) > 0 and scores[0] > np.percentile(scores, 60):
    peaks.insert(0, 0)
if len(scores) > 1 and scores[-1] > np.percentile(scores, 60):
    peaks.append(len(scores) - 1)
peaks = sorted(set(peaks))

print(f"\n{'='*70}")
print(f"BIFURCATIONS WITHIN IM->EGC: {len(peaks)}")
print(f"{'='*70}")
for rank, pi in enumerate(peaks):
    row = bin_df.iloc[pi]
    print(f"\n  Bif-{rank+1}: pseudotime [{row['pt_lo']:.4f}, {row['pt_hi']:.4f}]")
    print(f"    Fate variance={row['fate_variance']:.4f}, "
          f"Cluster entropy={row['cluster_entropy']:.3f}")
    print(f"    V(s) std={row['V_std']:.3f}, score={row['bif_score']:.5f}")
    print(f"    P(EGC)={row['mean_P_EGC']:.3f}, P(GC)={row['mean_P_GC']:.3f}")

# ===================================================================
# Method 4: Gene markers at each IM bifurcation
# ===================================================================
print(f"\n{'='*70}")
print("MOLECULAR MARKERS at IM bifurcations")
print(f"{'='*70}")

X_im = adata_mc[im_idx].X
if sparse.issparse(X_im):
    X_im = X_im.toarray()
gene_names = adata_mc.var_names

for rank, pi in enumerate(peaks):
    row = bin_df.iloc[pi]
    # Cells in this bin + neighbors
    window = np.abs(bin_labels - row["bin"]) <= 1
    if window.sum() < 10:
        window = np.abs(bin_labels - row["bin"]) <= 2

    # Split by fate cluster
    window_clusters = im_fate_labels[window]
    window_X = X_im[window]
    unique_cl = np.unique(window_clusters)

    if len(unique_cl) < 2:
        print(f"\n  Bif-{rank+1}: insufficient cluster diversity")
        continue

    # Compare two most common clusters
    cl_counts = [(c, (window_clusters == c).sum()) for c in unique_cl]
    cl_counts.sort(key=lambda x: -x[1])
    c1, c2 = cl_counts[0][0], cl_counts[1][0]
    mask1 = window_clusters == c1
    mask2 = window_clusters == c2

    mean1 = window_X[mask1].mean(axis=0)
    mean2 = window_X[mask2].mean(axis=0)
    std_pool = np.sqrt(
        (window_X[mask1].var(axis=0) + window_X[mask2].var(axis=0)) / 2 + 1e-10
    )
    d = (np.asarray(mean1) - np.asarray(mean2)).flatten()
    std_f = np.asarray(std_pool).flatten()
    cohen_d = d / std_f

    # Filter out nan/inf
    valid = np.isfinite(cohen_d)
    if valid.sum() == 0:
        continue

    top_up = np.argsort(np.where(valid, cohen_d, 0))[-10:][::-1]
    top_down = np.argsort(np.where(valid, cohen_d, 0))[:10]

    # Cluster fate profiles
    p1_egc = im_fate_egc[window][mask1].mean()
    p1_gc = im_fate_gc[window][mask1].mean()
    p2_egc = im_fate_egc[window][mask2].mean()
    p2_gc = im_fate_gc[window][mask2].mean()

    print(f"\n  Bif-{rank+1} (pt~{row['pt_mean']:.4f}):")
    print(f"    Cluster {c1} (n={mask1.sum()}): P(EGC)={p1_egc:.3f}, P(GC)={p1_gc:.3f}")
    print(f"    Cluster {c2} (n={mask2.sum()}): P(EGC)={p2_egc:.3f}, P(GC)={p2_gc:.3f}")
    print(f"    Top genes distinguishing sub-trajectories:")
    print(f"    {'Gene':>15} {'cohen_d':>8}  Direction")
    print(f"    {'-'*40}")
    for gi in top_up[:5]:
        if np.isfinite(cohen_d[gi]):
            print(f"    {gene_names[gi]:>15} {cohen_d[gi]:>+8.3f}  Cluster{c1} > Cluster{c2}")
    for gi in top_down[:5]:
        if np.isfinite(cohen_d[gi]):
            print(f"    {gene_names[gi]:>15} {cohen_d[gi]:>+8.3f}  Cluster{c2} > Cluster{c1}")

# Save full results
bin_df.to_csv("results/ot_im_subtraj_bifurcations.csv", index=False)
cluster_df = pd.DataFrame(cluster_profiles)
cluster_df.to_csv("results/ot_im_fate_clusters.csv", index=False)

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  IM sub-trajectory clusters: {best_k}")
print(f"  Bifurcation points detected: {len(peaks)}")
print(f"  Key insight: IM cells diverge into {best_k} fate-distinct groups")
print(f"  that separate at specific pseudotime windows.")
print(f"{'='*70}")

