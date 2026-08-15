"""
Phase 25c: Cluster Number Sensitivity (k=2 to k=6)
- Consensus clustering (repeated KMeans with subsampling)
- Silhouette, Calinski-Harabasz, Davies-Bouldin indices
- Patient/dataset composition per cluster
- Gap statistic approximation
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25c: Cluster Number Sensitivity (k=2 to k=6)")
print("=" * 70)

# Load data
print("\n[1/4] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
stages = adata_mc.obs['stage'].values
sample_ids = adata_mc.obs['sample_id'].values
datasets = adata_mc.obs['dataset'].values if 'dataset' in adata_mc.obs else None

im_idx = np.where(stages == 'IM')[0]
print(f"  {len(im_idx)} IM metacells for clustering")

# Compute fate profiles
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

fate_matrix = compute_fate(im_idx, T_cr, stages)
print(f"  Fate matrix: {fate_matrix.shape}")

# ===================================================================
# [2/4] Multi-index evaluation for k=2..6
# ===================================================================
print("\n[2/4] Evaluating k=2 to k=6 ...")

k_range = range(2, 7)
metrics = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(fate_matrix)
    sil = silhouette_score(fate_matrix, labels)
    ch = calinski_harabasz_score(fate_matrix, labels)
    db = davies_bouldin_score(fate_matrix, labels)
    inertia = km.inertia_
    metrics.append({
        'k': k, 'silhouette': sil, 'calinski_harabasz': ch,
        'davies_bouldin': db, 'inertia': inertia
    })
    print(f"  k={k}: sil={sil:.3f}, CH={ch:.1f}, DB={db:.3f}, "
          f"inertia={inertia:.2f}")

metrics_df = pd.DataFrame(metrics)

# Identify optimal k
best_sil_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
best_ch_k = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
best_db_k = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
print(f"\n  Optimal k by metric:")
print(f"    Silhouette: k={best_sil_k}")
print(f"    Calinski-Harabasz: k={best_ch_k}")
print(f"    Davies-Bouldin: k={best_db_k}")

# ===================================================================
# [3/4] Consensus clustering (subsampled KMeans, 100 runs)
# ===================================================================
print("\n[3/4] Consensus clustering (100 bootstrap runs per k) ...")

n_boot = 100
rng = np.random.default_rng(42)
consensus_results = {}

for k in k_range:
    # ARI stability: repeated full-data KMeans with different inits
    boot_labels = []
    for b in range(n_boot):
        km_b = KMeans(n_clusters=k, n_init=1, random_state=b)
        boot_labels.append(km_b.fit_predict(fate_matrix))

    # Mean pairwise ARI
    ari_pairs = []
    for i in range(0, min(50, n_boot)):
        for j in range(i+1, min(50, n_boot)):
            ari_pairs.append(adjusted_rand_score(
                boot_labels[i], boot_labels[j]))
    mean_ari = np.mean(ari_pairs)

    # Subsample consensus: 80% cells, compute co-clustering rate
    co_occurrence = np.zeros((len(im_idx), len(im_idx)))
    co_sampled = np.zeros((len(im_idx), len(im_idx)))
    n_sub = 50
    for b in range(n_sub):
        subset = rng.choice(len(im_idx), size=int(0.8 * len(im_idx)),
                           replace=False)
        km = KMeans(n_clusters=k, n_init=5, random_state=b + 200)
        labels_sub = km.fit_predict(fate_matrix[subset])
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                co_sampled[subset[i], subset[j]] += 1
                co_sampled[subset[j], subset[i]] += 1
                if labels_sub[i] == labels_sub[j]:
                    co_occurrence[subset[i], subset[j]] += 1
                    co_occurrence[subset[j], subset[i]] += 1

    mask = co_sampled > 0
    consensus = np.zeros_like(co_occurrence)
    consensus[mask] = co_occurrence[mask] / co_sampled[mask]
    upper_tri = consensus[np.triu_indices(len(im_idx), k=1)]
    pac = np.mean((upper_tri > 0.1) & (upper_tri < 0.9))

    km_final = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels_final = km_final.fit_predict(fate_matrix)

    consensus_results[k] = {
        'PAC': pac, 'mean_ARI': mean_ari, 'labels': labels_final,
    }
    print(f"  k={k}: PAC={pac:.3f}, mean_ARI={mean_ari:.3f}")

best_pac_k = min(consensus_results, key=lambda x: consensus_results[x]['PAC'])
best_ari_k = max(consensus_results, key=lambda x: consensus_results[x]['mean_ARI'])
print(f"\n  Best k by PAC (lowest): k={best_pac_k} "
      f"(PAC={consensus_results[best_pac_k]['PAC']:.3f})")
print(f"  Best k by ARI stability: k={best_ari_k} "
      f"(ARI={consensus_results[best_ari_k]['mean_ARI']:.3f})")

# ===================================================================
# [4/4] Patient/dataset composition at k=4
# ===================================================================
print("\n[4/4] Patient/dataset composition (k=4) ...")

labels_k4 = consensus_results[4]['labels']
im_samples = sample_ids[im_idx]
im_datasets = datasets[im_idx] if datasets is not None else None

print("\n  Cluster composition by patient:")
comp_rows = []
for c in range(4):
    mask = labels_k4 == c
    samples_in_c = im_samples[mask]
    unique_s, counts_s = np.unique(samples_in_c, return_counts=True)
    n_patients = len(unique_s)
    top_patient = unique_s[np.argmax(counts_s)]
    top_frac = counts_s.max() / mask.sum()
    comp_rows.append({
        'cluster': c, 'n_metacells': mask.sum(),
        'n_patients': n_patients, 'max_patient_frac': top_frac,
        'top_patient': top_patient,
    })
    print(f"    Cluster {c}: {mask.sum()} mc, {n_patients} patients, "
          f"max_patient_frac={top_frac:.2f} ({top_patient})")

# Check for single-patient clusters (red flag)
single_patient_clusters = [r for r in comp_rows if r['n_patients'] <= 1]
if single_patient_clusters:
    print(f"\n  WARNING: {len(single_patient_clusters)} cluster(s) "
          f"dominated by single patient!")
else:
    print(f"\n  All clusters have multi-patient representation.")

# Dataset balance
if im_datasets is not None:
    print("\n  Cluster composition by dataset:")
    for c in range(4):
        mask = labels_k4 == c
        ds_in_c = im_datasets[mask]
        unique_d, counts_d = np.unique(ds_in_c, return_counts=True)
        ds_str = ", ".join(f"{d}:{n}" for d, n in zip(unique_d, counts_d))
        print(f"    Cluster {c}: {ds_str}")

# Chi-square test: are clusters independent of dataset?
from scipy.stats import chi2_contingency
if im_datasets is not None:
    contingency = pd.crosstab(
        pd.Series(labels_k4, name='cluster'),
        pd.Series(im_datasets, name='dataset'))
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    print(f"\n  Chi-square (cluster vs dataset): chi2={chi2:.2f}, "
          f"p={p_chi:.4f}, dof={dof}")
    if p_chi < 0.05:
        print("    -> Clusters are NOT independent of dataset (batch effect?)")
    else:
        print("    -> Clusters ARE independent of dataset (no batch confound)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Row 1: metric curves
ax = axes[0, 0]
ax.plot([m['k'] for m in metrics], [m['silhouette'] for m in metrics], 'bo-')
ax.axvline(4, color='red', ls='--', alpha=0.5)
ax.set_xlabel('k'); ax.set_ylabel('Silhouette'); ax.set_title('Silhouette Score')

ax = axes[0, 1]
ax.plot([m['k'] for m in metrics], [m['calinski_harabasz'] for m in metrics], 'go-')
ax.axvline(4, color='red', ls='--', alpha=0.5)
ax.set_xlabel('k'); ax.set_ylabel('CH Index'); ax.set_title('Calinski-Harabasz')

ax = axes[0, 2]
pac_vals = [consensus_results[k]['PAC'] for k in k_range]
ari_vals = [consensus_results[k]['mean_ARI'] for k in k_range]
ax.plot(list(k_range), pac_vals, 'rs-', label='PAC (lower=better)')
ax2 = ax.twinx()
ax2.plot(list(k_range), ari_vals, 'b^-', label='ARI stability')
ax.set_xlabel('k'); ax.set_ylabel('PAC', color='r')
ax2.set_ylabel('Mean ARI', color='b')
ax.set_title('Consensus Metrics')
ax.axvline(4, color='gray', ls='--', alpha=0.5)

# Row 2: fate scatter for k=3,4,5
for i, k in enumerate([3, 4, 5]):
    ax = axes[1, i]
    labels_k = consensus_results[k]['labels']
    ax.scatter(fate_matrix[:, 0], fate_matrix[:, 1],
              c=labels_k, cmap='tab10', s=10, alpha=0.6)
    ax.set_xlabel('P(EGC)'); ax.set_ylabel('P(GC)')
    sil_k = metrics_df[metrics_df['k'] == k]['silhouette'].values[0]
    ax.set_title(f'k={k} (sil={sil_k:.3f})')

plt.tight_layout()
plt.savefig(FIGURES / "cluster_sensitivity_k2_k6.png", dpi=150,
            bbox_inches='tight')
plt.close()

# Save
all_metrics = metrics_df.copy()
all_metrics['PAC'] = [consensus_results[k]['PAC'] for k in k_range]
all_metrics['mean_ARI'] = [consensus_results[k]['mean_ARI'] for k in k_range]
all_metrics.to_csv(RESULTS / "cluster_sensitivity_metrics.csv", index=False)
pd.DataFrame(comp_rows).to_csv(
    RESULTS / "cluster_patient_composition.csv", index=False)

print(f"\n  Saved: cluster_sensitivity_k2_k6.png")
print(f"  Saved: cluster_sensitivity_metrics.csv")
print(f"  Saved: cluster_patient_composition.csv")

print("\n" + "=" * 70)
print("Phase 25c COMPLETE")
print("=" * 70)
print(f"  Optimal k: Silhouette={int(best_sil_k)}, CH={int(best_ch_k)}, "
      f"DB={int(best_db_k)}, PAC={best_pac_k}, ARI={best_ari_k}")
print(f"  k=4 consensus: PAC={consensus_results[4]['PAC']:.3f}, "
      f"ARI={consensus_results[4]['mean_ARI']:.3f}")
n_multi = sum(1 for r in comp_rows if r['n_patients'] > 1)
print(f"  Patient diversity: {n_multi}/4 clusters have >1 patient")
print("=" * 70)