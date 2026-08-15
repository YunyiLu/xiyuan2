"""
Phase 25b: CellRank Kernel Sensitivity Analysis
Test whether bifurcation structure depends on kernel choice / pseudotime prior.

Models:
  M1: ConnectivityKernel only (no directional bias)
  M2: Current model (loaded from saved T matrix) [reference]
  M3: Manual pseudotime kernel (100% directional, no connectivity)
  M4: Shuffled pseudotime (negative control)

Bypass CellRank's parallelize (Windows multiprocessing issue) by
implementing pseudotime biasing manually on the kNN graph.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25b: CellRank Kernel Sensitivity Analysis")
print("=" * 70)

# Load metacell data
print("\n[1/4] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values

# Ensure neighbor graph
if 'connectivities' not in adata_mc.obsp:
    sc.pp.neighbors(adata_mc, use_rep="X_scVI", n_neighbors=30, random_state=42)

conn = adata_mc.obsp['connectivities']
n_cells = conn.shape[0]
im_idx = np.where(stages == 'IM')[0]
print(f"  {n_cells} metacells, {len(im_idx)} IM metacells")

# Load reference T matrix (M2)
T_ref = sparse.load_npz(RESULTS / "rl_transition_matrix.npz").toarray()
print(f"  Reference T loaded: {T_ref.shape}")

# ===================================================================
# [2/4] Build kernels manually (avoid CellRank multiprocessing)
# ===================================================================
print("\n[2/4] Building transition kernels manually ...")

def connectivity_kernel(conn_matrix):
    """M1: Row-normalize connectivity matrix -> symmetric random walk."""
    T = conn_matrix.toarray().astype(np.float64)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums
    return T

def pseudotime_kernel(conn_matrix, pseudotime, frac_to_keep=0.95):
    """Manual implementation of pseudotime-biased kernel.
    For each cell i, among its kNN neighbors j:
      - weight_ij = conn_ij * softmax(pt_j - pt_i)
      - This biases transitions toward increasing pseudotime
    frac_to_keep: fraction of weight going forward (vs backward)
    """
    T = conn_matrix.toarray().astype(np.float64)
    n = T.shape[0]
    for i in range(n):
        neighbors = np.where(T[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        # Pseudotime difference: positive = forward
        dt = pseudotime[neighbors] - pseudotime[i]
        # Soft threshold: exponential bias toward forward
        # Scale factor controls how strongly we bias forward
        scale = 1.0 / (np.std(dt) + 1e-10)
        weights = np.exp(scale * dt)
        # Apply connectivity as base weight
        weights = weights * T[i, neighbors]
        # Normalize
        weights = weights / (weights.sum() + 1e-10)
        T[i, :] = 0
        T[i, neighbors] = weights
    return T

def combined_kernel(conn_matrix, pseudotime, alpha=0.7, frac=0.95):
    """alpha * pseudotime_kernel + (1-alpha) * connectivity_kernel"""
    T_pt = pseudotime_kernel(conn_matrix, pseudotime, frac)
    T_ck = connectivity_kernel(conn_matrix)
    return alpha * T_pt + (1 - alpha) * T_ck

print("  Building M1 (connectivity only) ...")
T_M1 = connectivity_kernel(conn)

print("  M2 = reference (loaded from disk)")
T_M2 = T_ref

print("  Building M3 (pure pseudotime) ...")
T_M3 = pseudotime_kernel(conn, pt)

print("  Building M4 (shuffled pseudotime control) ...")
rng = np.random.default_rng(123)
pt_shuffled = rng.permutation(pt)
T_M4 = combined_kernel(conn, pt_shuffled, alpha=0.7)

T_matrices = {
    'M1_connectivity': T_M1,
    'M2_current': T_M2,
    'M3_pseudotime': T_M3,
    'M4_shuffled': T_M4,
}

for name, T in T_matrices.items():
    print(f"    {name}: shape={T.shape}, "
          f"row_sum_check={T.sum(axis=1).mean():.4f}")

# ===================================================================
# [3/4] Fate profiles and clustering for each model
# ===================================================================
print("\n[3/4] Computing fate profiles and clustering ...")

def compute_fate(indices, T, all_stages, k=20):
    n = T.shape[0]
    prop = np.zeros((len(indices), n))
    for i, idx in enumerate(indices):
        prop[i, idx] = 1.0
    for _ in range(k):
        prop = prop @ T
    egc_m = (all_stages == 'EGC') | (all_stages == 'EGC_multi_region')
    gc_m = all_stages == 'GC'
    im_m = all_stages == 'IM'
    nag_m = all_stages == 'NAG'
    return np.column_stack([
        (prop * egc_m).sum(1), (prop * gc_m).sum(1),
        (prop * im_m).sum(1), (prop * nag_m).sum(1)])

# Reference labels from M2
fate_M2 = compute_fate(im_idx, T_M2, stages)
km_ref = KMeans(n_clusters=4, n_init=10, random_state=42)
labels_ref = km_ref.fit_predict(fate_M2)

model_results = {}
for name, T in T_matrices.items():
    fate = compute_fate(im_idx, T, stages)
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = km.fit_predict(fate)
    sil = silhouette_score(fate, labels) if len(np.unique(labels)) > 1 else 0
    ari = adjusted_rand_score(labels_ref, labels)
    p_egc = fate[:, 0]
    model_results[name] = {
        'labels': labels, 'fate': fate, 'silhouette': sil,
        'ARI_vs_M2': ari, 'P_EGC_range': (p_egc.min(), p_egc.max()),
        'P_EGC_std': p_egc.std(),
    }
    print(f"  {name:20s}: sil={sil:.3f}, ARI_vs_M2={ari:.3f}, "
          f"P(EGC)=[{p_egc.min():.3f},{p_egc.max():.3f}]")

# ===================================================================
# [4/4] Cross-model comparison
# ===================================================================
print("\n[4/4] Cross-model comparison ...")

model_names = list(model_results.keys())
n_models = len(model_names)
ari_matrix = np.zeros((n_models, n_models))
for i in range(n_models):
    for j in range(n_models):
        ari_matrix[i, j] = adjusted_rand_score(
            model_results[model_names[i]]['labels'],
            model_results[model_names[j]]['labels'])

print("\n  ARI pairwise matrix:")
header = f"  {'':>18}" + "".join(f"{m:>18}" for m in model_names)
print(header)
for i, m in enumerate(model_names):
    row = "".join(f"{ari_matrix[i,j]:>18.3f}" for j in range(n_models))
    print(f"  {m:>18}{row}")

ari_m1_m2 = ari_matrix[0, 1]
ari_m4_m2 = ari_matrix[3, 1]

# P(EGC) correlation between models
fate_m1 = model_results['M1_connectivity']['fate'][:, 0]
fate_m2 = model_results['M2_current']['fate'][:, 0]
fate_m3 = model_results['M3_pseudotime']['fate'][:, 0]
fate_m4 = model_results['M4_shuffled']['fate'][:, 0]

rho_12, _ = spearmanr(fate_m1, fate_m2)
rho_13, _ = spearmanr(fate_m1, fate_m3)
rho_14, _ = spearmanr(fate_m1, fate_m4)

print(f"\n  P(EGC) Spearman correlations:")
print(f"    M1 vs M2: rho={rho_12:.3f}")
print(f"    M1 vs M3: rho={rho_13:.3f}")
print(f"    M1 vs M4: rho={rho_14:.3f}")

# OXPHOS-fate correlation under each model
X = adata_mc.X
if sparse.issparse(X):
    X = X.toarray()
gene_names = adata_mc.var_names
ox_genes = ['COX5B', 'NDUFA3', 'COX7B', 'NDUFB3', 'NDUFA4']
ox_idx = [np.where(gene_names == g)[0][0] for g in ox_genes if g in gene_names]
ox_im = X[im_idx][:, ox_idx].mean(axis=1)

print(f"\n  OXPHOS vs P(EGC) under each model:")
for name, res in model_results.items():
    p_egc_model = res['fate'][:, 0]
    rho, p = spearmanr(ox_im, p_egc_model)
    print(f"    {name:>18}: rho={rho:+.3f} (p={p:.2e})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, (name, res) in enumerate(model_results.items()):
    ax = axes[i // 2, i % 2]
    fate = res['fate']
    sc_plt = ax.scatter(fate[:, 0], fate[:, 1], c=res['labels'],
                        cmap='tab10', s=15, alpha=0.7)
    ax.set_xlabel("P(EGC)")
    ax.set_ylabel("P(GC)")
    ax.set_title(f"{name}\nsil={res['silhouette']:.3f}, "
                 f"ARI={res['ARI_vs_M2']:.3f}")
plt.tight_layout()
plt.savefig(FIGURES / "kernel_sensitivity_fate.png", dpi=150,
            bbox_inches='tight')
plt.close()

# Save
summary = []
for name, res in model_results.items():
    summary.append({
        'model': name, 'silhouette': res['silhouette'],
        'ARI_vs_M2': res['ARI_vs_M2'],
        'P_EGC_std': res['P_EGC_std'],
    })
pd.DataFrame(summary).to_csv(
    RESULTS / "kernel_sensitivity_summary.csv", index=False)

print("\n" + "=" * 70)
print("Phase 25b COMPLETE")
print("=" * 70)
print(f"  M1 (no prior) vs M2: ARI={ari_m1_m2:.3f}, P(EGC) rho={rho_12:.3f}")
print(f"  M4 (shuffled) vs M2: ARI={ari_m4_m2:.3f}")
if ari_m1_m2 > 0.5:
    print("  -> Bifurcation structure ROBUST to kernel choice")
elif ari_m1_m2 > 0.3:
    print("  -> Partial structure preserved without prior")
else:
    print("  -> Structure depends on pseudotime prior (needs discussion)")
print("=" * 70)
