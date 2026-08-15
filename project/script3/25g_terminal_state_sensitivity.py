"""
Phase 25g: Terminal State Definition Sensitivity Analysis
Tests whether absorption probabilities are robust to threshold choices.

Sweeps: EGC percentile [70,80,90], GC percentile [60,70,80], NAG percentile [20,30,40]
Reports: pairwise Spearman correlation of P(EGC) across all settings.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25g: Terminal State Sensitivity Analysis")
print("=" * 70)

# Load data
print("\n[1/3] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()

n = T_cr.shape[0]
stages = adata_mc.obs["stage"].values
pt = adata_mc.obs["dpt_pseudotime"].values
print(f"  {n} metacells")

# ===================================================================
# [2/3] Sweep thresholds
# ===================================================================
print("\n[2/3] Sweeping terminal state thresholds ...")

egc_thresholds = [70, 80, 90]
gc_thresholds = [60, 70, 80]
nag_thresholds = [20, 30, 40]

all_combos = list(product(egc_thresholds, gc_thresholds, nag_thresholds))
print(f"  {len(all_combos)} threshold combinations to test")


def compute_absorption(egc_pct, gc_pct, nag_pct):
    """Compute P(EGC) for given threshold combo."""
    egc_mask = ((stages == "EGC") |
                ((stages == "EGC_multi_region") &
                 (pt > np.percentile(pt, egc_pct))))
    gc_mask = ((stages == "GC") &
               (pt > np.percentile(pt[stages == "GC"], gc_pct)))
    stasis_mask = ((stages == "NAG") &
                   (pt < np.percentile(pt[stages == "NAG"], nag_pct)))

    absorbing_idx = np.where(egc_mask | gc_mask | stasis_mask)[0]
    transient_idx = np.where(~(egc_mask | gc_mask | stasis_mask))[0]

    if len(absorbing_idx) < 3 or len(transient_idx) < 10:
        return None, (egc_mask.sum(), gc_mask.sum(), stasis_mask.sum())

    order = np.concatenate([transient_idx, absorbing_idx])
    T_reordered = T_cr[order][:, order]

    n_t = len(transient_idx)
    n_a = len(absorbing_idx)

    Q = T_reordered[:n_t, :n_t]
    R = T_reordered[:n_t, n_t:]

    I_minus_Q = np.eye(n_t) - Q
    try:
        B = np.linalg.solve(I_minus_Q, R)
    except np.linalg.LinAlgError:
        return None, (egc_mask.sum(), gc_mask.sum(), stasis_mask.sum())

    # Map back: which absorbing cells are EGC?
    absorbing_stages = []
    for j in range(n_a):
        orig_idx = absorbing_idx[j]
        if egc_mask[orig_idx]:
            absorbing_stages.append(0)
        elif gc_mask[orig_idx]:
            absorbing_stages.append(1)
        else:
            absorbing_stages.append(2)
    absorbing_stages = np.array(absorbing_stages)

    fate_probs = np.zeros((n, 3))
    for fi in range(3):
        fate_cols = np.where(absorbing_stages == fi)[0]
        if len(fate_cols) > 0:
            fate_probs[transient_idx, fi] = B[:, fate_cols].sum(axis=1)

    fate_probs[egc_mask, 0] = 1.0
    fate_probs[gc_mask, 1] = 1.0
    fate_probs[stasis_mask, 2] = 1.0

    row_sums = fate_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    fate_probs = fate_probs / row_sums

    return fate_probs[:, 0], (egc_mask.sum(), gc_mask.sum(), stasis_mask.sum())


# Run all combinations
results = {}
print(f"\n  {'EGC%':>5} {'GC%':>5} {'NAG%':>5} {'nEGC':>5} {'nGC':>5} "
      f"{'nNAG':>5} {'mean_P(EGC)':>12}")
print(f"  {'-'*55}")

for egc_p, gc_p, nag_p in all_combos:
    p_egc, counts = compute_absorption(egc_p, gc_p, nag_p)
    key = f"E{egc_p}_G{gc_p}_N{nag_p}"
    if p_egc is not None:
        results[key] = p_egc
        print(f"  {egc_p:>5} {gc_p:>5} {nag_p:>5} {counts[0]:>5} "
              f"{counts[1]:>5} {counts[2]:>5} {p_egc.mean():>12.4f}")
    else:
        print(f"  {egc_p:>5} {gc_p:>5} {nag_p:>5} -- FAILED --")

# ===================================================================
# [3/3] Pairwise correlations
# ===================================================================
print(f"\n[3/3] Computing pairwise Spearman correlations ...")

keys = list(results.keys())
n_k = len(keys)
corr_matrix = np.ones((n_k, n_k))

for i in range(n_k):
    for j in range(i + 1, n_k):
        rho, _ = spearmanr(results[keys[i]], results[keys[j]])
        corr_matrix[i, j] = rho
        corr_matrix[j, i] = rho

min_rho = corr_matrix[np.triu_indices(n_k, k=1)].min()
mean_rho = corr_matrix[np.triu_indices(n_k, k=1)].mean()
median_rho = np.median(corr_matrix[np.triu_indices(n_k, k=1)])

print(f"\n  Pairwise Spearman rho across {n_k} threshold settings:")
print(f"    Min:    {min_rho:.4f}")
print(f"    Median: {median_rho:.4f}")
print(f"    Mean:   {mean_rho:.4f}")

# Reference setting (original: E80_G70_N30)
ref_key = "E80_G70_N30"
if ref_key in results:
    ref_idx = keys.index(ref_key)
    print(f"\n  Correlations with reference ({ref_key}):")
    for i, k in enumerate(keys):
        if k != ref_key:
            rho = corr_matrix[ref_idx, i]
            print(f"    {k}: rho={rho:.4f}")

# IM-specific: does the RANKING within IM cells change?
im_mask = stages == "IM"
im_corrs = []
for i in range(n_k):
    for j in range(i + 1, n_k):
        rho, _ = spearmanr(results[keys[i]][im_mask],
                           results[keys[j]][im_mask])
        im_corrs.append(rho)

print(f"\n  IM-cell ranking stability (most relevant for bifurcation):")
print(f"    Min rho:  {min(im_corrs):.4f}")
print(f"    Mean rho: {np.mean(im_corrs):.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel A: correlation heatmap
ax = axes[0]
im = ax.imshow(corr_matrix, vmin=0.8, vmax=1.0, cmap='RdYlGn')
ax.set_xticks(range(n_k))
ax.set_yticks(range(n_k))
ax.set_xticklabels(keys, rotation=90, fontsize=6)
ax.set_yticklabels(keys, fontsize=6)
ax.set_title(f"Pairwise rho (min={min_rho:.3f})")
plt.colorbar(im, ax=ax)

# Panel B: P(EGC) distribution across settings (IM cells only)
ax = axes[1]
for k in keys:
    ax.hist(results[k][im_mask], bins=30, alpha=0.2, label=k)
ax.set_xlabel("P(EGC)")
ax.set_ylabel("Count (IM cells)")
ax.set_title("P(EGC) distribution across threshold settings")

# Panel C: reference vs most different setting
ax = axes[2]
if ref_key in results:
    ref_corrs = corr_matrix[keys.index(ref_key)]
    worst_idx = np.argmin(ref_corrs)
    worst_key = keys[worst_idx]
    ax.scatter(results[ref_key], results[worst_key], s=3, alpha=0.3)
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    rho_worst = ref_corrs[worst_idx]
    ax.set_xlabel(f"P(EGC) [{ref_key}]")
    ax.set_ylabel(f"P(EGC) [{worst_key}]")
    ax.set_title(f"Most different pair (rho={rho_worst:.3f})")

plt.tight_layout()
plt.savefig(FIGURES / "terminal_state_sensitivity.png", dpi=150,
            bbox_inches='tight')
plt.close()

# Save summary
summary = {
    'n_settings': n_k,
    'min_rho_all': min_rho,
    'mean_rho_all': mean_rho,
    'min_rho_IM': min(im_corrs),
    'mean_rho_IM': np.mean(im_corrs),
}
pd.DataFrame([summary]).to_csv(
    RESULTS / "terminal_state_sensitivity.csv", index=False)

print(f"\n  Saved: terminal_state_sensitivity.png")
print(f"  Saved: terminal_state_sensitivity.csv")

print("\n" + "=" * 70)
print("Phase 25g COMPLETE")
print("=" * 70)
if min_rho > 0.95:
    print(f"  CONCLUSION: ROBUST (min rho={min_rho:.3f} > 0.95)")
    print(f"  Terminal state thresholds do NOT materially affect results.")
elif min_rho > 0.85:
    print(f"  CONCLUSION: MOSTLY ROBUST (min rho={min_rho:.3f})")
    print(f"  Rankings preserved, absolute values shift slightly.")
else:
    print(f"  CONCLUSION: SENSITIVE (min rho={min_rho:.3f} < 0.85)")
    print(f"  Consider using data-driven terminal state identification.")
print("=" * 70)


