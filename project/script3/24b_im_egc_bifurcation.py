"""Focused IM->EGC bifurcation analysis."""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

adata_mc = sc.read_h5ad('data/rl_metacells.h5ad')
T_sparse = sparse.load_npz('results/rl_transition_matrix.npz')
T_cr = T_sparse.toarray()
val_df = pd.read_csv('results/rl_value_function.csv')

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values

# Relevant corridor: IM + EGC + EGC_multi_region
relevant_mask = np.isin(stages, ['IM', 'EGC', 'EGC_multi_region'])
relevant_idx = np.where(relevant_mask)[0]
print(f"Relevant cells (IM+EGC+EGC_multi): {len(relevant_idx)}")

# Terminal EGC: top 30% pseudotime EGC cells
egc_all_mask = (stages == 'EGC') | (stages == 'EGC_multi_region')
egc_terminal = np.where(
    egc_all_mask & (pt > np.percentile(pt[egc_all_mask], 70))
)[0]

# IM non-progressing: bottom 25% in P(->EGC)
im_idx = np.where(stages == 'IM')[0]
egc_idx_all = np.where(egc_all_mask)[0]
p_to_egc = T_cr[im_idx][:, egc_idx_all].sum(axis=1)
im_nonprog = im_idx[p_to_egc < np.percentile(p_to_egc, 25)]

print(f"Terminal EGC: {len(egc_terminal)}")
print(f"IM non-progressing: {len(im_nonprog)}")

# Absorption probabilities within corridor
absorbing = np.concatenate([egc_terminal, im_nonprog])
absorbing_labels = np.array([0]*len(egc_terminal) + [1]*len(im_nonprog))

transient_mask = np.ones(adata_mc.n_obs, dtype=bool)
transient_mask[absorbing] = False
transient_mask[~relevant_mask] = False
transient_idx = np.where(transient_mask)[0]
print(f"Transient: {len(transient_idx)}")

all_idx = np.concatenate([transient_idx, absorbing])
T_sub = T_cr[all_idx][:, all_idx]
row_sums = T_sub.sum(axis=1, keepdims=True)
T_sub = np.where(row_sums > 0, T_sub / row_sums, 0)

n_t = len(transient_idx)
Q = T_sub[:n_t, :n_t]
R = T_sub[:n_t, n_t:]

print("Solving absorption system ...")
I_Q = np.eye(n_t) - Q
B = np.linalg.solve(I_Q, R)

# Map back
fate_probs = np.zeros((adata_mc.n_obs, 2))
for fi in range(2):
    cols = np.where(absorbing_labels == fi)[0]
    fate_probs[transient_idx, fi] = B[:, cols].sum(axis=1)
fate_probs[egc_terminal, 0] = 1.0
fate_probs[im_nonprog, 1] = 1.0
rs = fate_probs.sum(axis=1, keepdims=True)
rs = np.where(rs > 0, rs, 1.0)
fate_probs = fate_probs / rs

# Focus on IM cells
im_fate_egc = fate_probs[im_idx, 0]
im_fate_stay = fate_probs[im_idx, 1]
im_pt = pt[im_idx]
im_V = V[im_idx]

print(f"\nIM cells: mean P(EGC)={im_fate_egc.mean():.3f}, "
      f"mean P(Stay)={im_fate_stay.mean():.3f}")

# ===== Bifurcation detection within IM (quantile bins) =====
N_BINS = 12
bin_edges = np.percentile(im_pt, np.linspace(0, 100, N_BINS + 1))
bin_edges[-1] += 1e-10
bin_labels = np.digitize(im_pt, bin_edges) - 1
bin_labels = np.clip(bin_labels, 0, N_BINS - 1)

print("\n" + "=" * 70)
print("IM internal bifurcation analysis (12 quantile bins)")
print("=" * 70)
print(f"{'Bin':>4} {'PT_range':>22} {'n':>4} {'P(EGC)':>8} {'P(Stay)':>8} "
      f"{'FateVar':>8} {'Undecid':>7}")
print("-" * 70)

bif_records = []
for b in range(N_BINS):
    mask = bin_labels == b
    if mask.sum() < 3:
        continue
    fe = im_fate_egc[mask]
    fs = im_fate_stay[mask]
    fate_var = fe.var() + fs.var()
    undecided = ((fe < 0.7) & (fs < 0.7)).mean()
    pt_lo = im_pt[mask].min()
    pt_hi = im_pt[mask].max()

    bif_records.append({
        "bin": b, "pt_lo": pt_lo, "pt_hi": pt_hi,
        "pt_mean": im_pt[mask].mean(), "n": int(mask.sum()),
        "mean_P_EGC": fe.mean(), "mean_P_Stay": fs.mean(),
        "fate_var": fate_var, "frac_undecided": undecided,
        "mean_V": im_V[mask].mean(), "std_V": im_V[mask].std(),
    })
    print(f"{b:>4} [{pt_lo:.4f},{pt_hi:.4f}] {mask.sum():>4} "
          f"{fe.mean():>8.3f} {fs.mean():>8.3f} "
          f"{fate_var:>8.4f} {undecided:>7.2f}")

bif_df = pd.DataFrame(bif_records)
bif_df["score"] = bif_df["fate_var"] * (1 + bif_df["frac_undecided"])

# Detect peaks in score
scores = bif_df["score"].values
peaks = []
for i in range(1, len(scores) - 1):
    if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
        if scores[i] > np.percentile(scores, 50):
            peaks.append(i)
# Check edges
if len(scores) > 2:
    if scores[0] > scores[1] and scores[0] > np.percentile(scores, 50):
        peaks.insert(0, 0)
    if scores[-1] > scores[-2] and scores[-1] > np.percentile(scores, 50):
        peaks.append(len(scores) - 1)

# Also detect based on entropy gradient (sharp transitions)
entropy = -(bif_df["mean_P_EGC"] * np.log(bif_df["mean_P_EGC"] + 1e-10) +
            bif_df["mean_P_Stay"] * np.log(bif_df["mean_P_Stay"] + 1e-10))
bif_df["entropy"] = entropy.values
ent_grad = np.gradient(entropy.values)

# Where entropy drops most sharply = decision point
sharp_drops = np.where(ent_grad < np.percentile(ent_grad, 25))[0]
for sd in sharp_drops:
    if sd not in peaks and scores[sd] > np.percentile(scores, 30):
        peaks.append(sd)
peaks = sorted(set(peaks))

print("\n" + "=" * 70)
print(f"BIFURCATIONS WITHIN IM->EGC: {len(peaks)}")
print("=" * 70)
for rank, pi in enumerate(peaks):
    row = bif_df.iloc[pi]
    print(f"\n  Bifurcation IM-{rank+1}:")
    print(f"    Pseudotime window: [{row['pt_lo']:.4f}, {row['pt_hi']:.4f}]")
    print(f"    Mean P(EGC)={row['mean_P_EGC']:.3f}, P(Stay)={row['mean_P_Stay']:.3f}")
    print(f"    Fate variance={row['fate_var']:.4f}, Undecided={row['frac_undecided']:.0%}")
    print(f"    Score={row['score']:.4f}")
    print(f"    IRL V(s): mean={row['mean_V']:.3f}, std={row['std_V']:.3f}")

# Molecular markers at each bifurcation
print("\n" + "=" * 70)
print("MOLECULAR MARKERS at each IM bifurcation")
print("=" * 70)

X_im = adata_mc[im_idx].X
if sparse.issparse(X_im):
    X_im = X_im.toarray()
gene_names = adata_mc.var_names

for rank, pi in enumerate(peaks):
    row = bif_df.iloc[pi]
    bin_mask = bin_labels == row["bin"]
    # Expand window slightly
    neighbors = np.abs(bin_labels - row["bin"]) <= 1
    window_mask = bin_mask | neighbors

    fe_window = im_fate_egc[window_mask]
    fs_window = im_fate_stay[window_mask]
    X_window = X_im[window_mask]

    # Split: EGC-fated (P>0.6) vs Stay-fated (P_stay>0.6)
    egc_fated = fe_window > 0.6
    stay_fated = fs_window > 0.6

    if egc_fated.sum() < 3 or stay_fated.sum() < 3:
        # Use median split instead
        egc_fated = fe_window > np.median(fe_window)
        stay_fated = ~egc_fated

    mean_egc = X_window[egc_fated].mean(axis=0)
    mean_stay = X_window[stay_fated].mean(axis=0)
    std_pool = np.sqrt(
        (X_window[egc_fated].var(axis=0) + X_window[stay_fated].var(axis=0)) / 2 + 1e-10
    )
    cohen_d = (np.asarray(mean_egc) - np.asarray(mean_stay)).flatten() / np.asarray(std_pool).flatten()

    top_up = np.argsort(cohen_d)[-5:][::-1]  # EGC > Stay
    top_down = np.argsort(cohen_d)[:5]        # Stay > EGC

    print(f"\n  Bifurcation IM-{rank+1} (pt~{row['pt_mean']:.4f}):")
    print(f"    Genes UP in EGC-fated cells:")
    for gi in top_up:
        print(f"      {gene_names[gi]:15s} d={cohen_d[gi]:+.3f}")
    print(f"    Genes UP in Stay-fated cells:")
    for gi in top_down:
        print(f"      {gene_names[gi]:15s} d={cohen_d[gi]:+.3f}")

bif_df.to_csv("results/ot_im_egc_bifurcations.csv", index=False)
print("\n\nSaved: results/ot_im_egc_bifurcations.csv")

