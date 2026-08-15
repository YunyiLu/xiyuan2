"""
Phase 24: Waddington Optimal Transport & Bifurcation Analysis
Identifies bifurcation points in gastric cancer progression using OT.

Strategy: Stages overlap in pseudotime, so we use two complementary approaches:
  A) Stage-level OT: transport between NAG->CAG->IM->EGC/GC (macro fate coupling)
  B) Absorption-based fate analysis: from CellRank T matrix, compute absorption
     probabilities to terminal states, detect where fate diverges.

Input:  script3/data/rl_metacells.h5ad
        script3/results/rl_transition_matrix.npz
        script3/results/rl_value_function.csv
Output: script3/results/ot_fate_probabilities.csv
        script3/results/ot_bifurcations.csv
        script3/results/ot_bifurcation_markers.csv
        script3/results/ot_growth_rates.csv
        script3/figures/ot_fate_landscape.png
        script3/figures/ot_bifurcation_tree.png
        script3/figures/ot_growth_vs_value.png

Requires: numpy, scipy, pandas, scanpy, ot (POT), matplotlib
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import ot as pot
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

STAGE_ORDER = {"NAG": 0, "CAG": 1, "IM": 2, "EGC": 3, "EGC_multi_region": 3, "GC": 4}
EPSILON = 0.05
N_QUANTILE_BINS = 20


# ===========================================================================
# Section 1: Stage-level Optimal Transport
# ===========================================================================

def stage_ot(adata_mc, embedding_key="X_scVI"):
    """
    Compute OT between consecutive disease stages.
    Unbalanced OT to estimate growth/death between stages.
    """
    print("[1/7] Computing stage-level OT transport maps ...")
    X = adata_mc.obsm[embedding_key]
    stages = adata_mc.obs["stage"].values

    # Define temporal ordering (parallel fates branch at IM)
    stage_sequence = [
        ("NAG", "CAG"),
        ("CAG", "IM"),
        ("IM", "EGC"),
        ("IM", "EGC_multi_region"),
        ("IM", "GC"),
        ("EGC_multi_region", "GC"),
    ]

    ot_results = {}
    for s_from, s_to in stage_sequence:
        idx_from = np.where(stages == s_from)[0]
        idx_to = np.where(stages == s_to)[0]
        if len(idx_from) < 2 or len(idx_to) < 2:
            continue

        C = cdist(X[idx_from], X[idx_to], metric="sqeuclidean")
        C = C / (C.max() + 1e-10)

        a = np.ones(len(idx_from)) / len(idx_from)
        b = np.ones(len(idx_to)) / len(idx_to)

        # Unbalanced OT (allows mass creation/destruction = growth/death)
        T_ot = pot.unbalanced.sinkhorn_unbalanced(
            a, b, C, reg=EPSILON, reg_m=0.5, numItermax=1000
        )

        # Growth: mass sent per source cell (relative to uniform)
        growth_per_cell = T_ot.sum(axis=1) * len(idx_from)

        ot_results[(s_from, s_to)] = {
            "T": T_ot, "idx_from": idx_from, "idx_to": idx_to,
            "cost": float(np.sum(T_ot * C)),
            "growth": growth_per_cell,
            "total_mass": float(T_ot.sum()),
        }

        print(f"    {s_from:20s} -> {s_to:20s}: "
              f"n={len(idx_from)}x{len(idx_to)}, "
              f"cost={ot_results[(s_from,s_to)]['cost']:.4f}, "
              f"mass={ot_results[(s_from,s_to)]['total_mass']:.3f}")

    return ot_results


# ===========================================================================
# Section 2: Absorption Probability (from CellRank transition matrix)
# ===========================================================================

def compute_absorption_probs(adata_mc, T_cr):
    """
    Compute absorption probabilities from CellRank's transition matrix.
    Terminal states: EGC (high V), GC (low V), Stasis (NAG).
    Uses fundamental matrix approach: F = (I - Q)^{-1}, B = F * R
    """
    print("[2/7] Computing absorption probabilities to terminal fates ...")

    n = T_cr.shape[0]
    stages = adata_mc.obs["stage"].values
    pt = adata_mc.obs["dpt_pseudotime"].values

    # Define absorbing states (terminal fates)
    # Use the highest-pseudotime cells in each terminal stage
    egc_mask = (stages == "EGC") | ((stages == "EGC_multi_region") & (pt > np.percentile(pt, 80)))
    gc_mask = (stages == "GC") & (pt > np.percentile(pt[stages == "GC"], 70))
    stasis_mask = (stages == "NAG") & (pt < np.percentile(pt[stages == "NAG"], 30))

    absorbing_idx = np.where(egc_mask | gc_mask | stasis_mask)[0]
    transient_idx = np.where(~(egc_mask | gc_mask | stasis_mask))[0]

    print(f"    Absorbing: EGC={egc_mask.sum()}, GC={gc_mask.sum()}, "
          f"Stasis={stasis_mask.sum()} (total={len(absorbing_idx)})")
    print(f"    Transient: {len(transient_idx)}")

    # Reorder: transient first, absorbing last
    order = np.concatenate([transient_idx, absorbing_idx])
    T_reordered = T_cr[order][:, order]

    n_t = len(transient_idx)
    n_a = len(absorbing_idx)

    Q = T_reordered[:n_t, :n_t]  # transient -> transient
    R = T_reordered[:n_t, n_t:]  # transient -> absorbing

    # Fundamental matrix F = (I - Q)^{-1}
    # Use iterative approach to avoid memory issues with 1000+ matrix inverse
    # B = (I - Q)^{-1} R via solving (I-Q) B = R
    I_minus_Q = np.eye(n_t) - Q
    print("    Solving absorption system (may take a moment) ...")
    B = np.linalg.solve(I_minus_Q, R)

    # Map absorption probs back to original cell indices
    # B[i, j] = prob that transient cell i is absorbed by absorbing cell j
    # Aggregate by fate type
    absorbing_stages = np.array(
        [0 if egc_mask[absorbing_idx[j]] else
         (1 if gc_mask[absorbing_idx[j]] else 2)
         for j in range(n_a)]
    )

    fate_probs = np.zeros((n, 3))
    for fi in range(3):
        fate_cols = np.where(absorbing_stages == fi)[0]
        if len(fate_cols) > 0:
            fate_probs[transient_idx, fi] = B[:, fate_cols].sum(axis=1)

    # Absorbing cells have probability 1 for their own fate
    fate_probs[egc_mask.nonzero()[0], 0] = 1.0
    fate_probs[gc_mask.nonzero()[0], 1] = 1.0
    fate_probs[stasis_mask.nonzero()[0], 2] = 1.0

    # Normalize
    row_sums = fate_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    fate_probs = fate_probs / row_sums

    # Fate entropy
    eps = 1e-10
    fate_entropy = -np.sum(fate_probs * np.log(fate_probs + eps), axis=1)

    adata_mc.obs["fate_EGC"] = fate_probs[:, 0]
    adata_mc.obs["fate_GC"] = fate_probs[:, 1]
    adata_mc.obs["fate_Stasis"] = fate_probs[:, 2]
    adata_mc.obs["fate_entropy"] = fate_entropy

    print(f"  Absorption prob means: EGC={fate_probs[:,0].mean():.3f}, "
          f"GC={fate_probs[:,1].mean():.3f}, Stasis={fate_probs[:,2].mean():.3f}")
    print(f"  Fate entropy range: [{fate_entropy.min():.3f}, {fate_entropy.max():.3f}]")

    return fate_probs, fate_entropy


# ===========================================================================
# Section 3: Bifurcation Detection via Fate Entropy Gradient
# ===========================================================================

def detect_bifurcations(adata_mc, fate_probs, fate_entropy):
    """
    Detect bifurcation points where fate entropy drops sharply.
    Uses quantile-based pseudotime bins for uniform cell density per bin.
    A bifurcation = region where nearby cells in embedding space
    have divergent fate probabilities.
    """
    print("[3/7] Detecting bifurcation points ...")

    pt = adata_mc.obs["dpt_pseudotime"].values
    stages = adata_mc.obs["stage"].values

    # Quantile binning for uniform density
    bin_edges = np.percentile(pt, np.linspace(0, 100, N_QUANTILE_BINS + 1))
    bin_edges[-1] += 1e-10
    bin_labels = np.digitize(pt, bin_edges) - 1
    bin_labels = np.clip(bin_labels, 0, N_QUANTILE_BINS - 1)

    records = []
    for b in range(N_QUANTILE_BINS):
        mask = bin_labels == b
        if mask.sum() < 5:
            continue

        bin_fates = fate_probs[mask]
        bin_ent = fate_entropy[mask]
        bin_pt_mean = pt[mask].mean()
        bin_stage = pd.Series(stages[mask]).mode().iloc[0]

        # Fate divergence: variance of fate probs across cells in this bin
        fate_var = bin_fates.var(axis=0).sum()

        # Within-bin entropy stats
        mean_ent = bin_ent.mean()
        max_ent = bin_ent.max()

        # Bimodality: fraction of cells that are "undecided" (no fate > 0.6)
        undecided = (bin_fates.max(axis=1) < 0.6).mean()

        # Dominant fate composition
        dominant_per_cell = np.argmax(bin_fates, axis=1)
        fate_composition = np.bincount(dominant_per_cell, minlength=3) / mask.sum()

        records.append({
            "bin": b,
            "mean_pseudotime": bin_pt_mean,
            "dominant_stage": bin_stage,
            "n_cells": int(mask.sum()),
            "mean_fate_entropy": mean_ent,
            "max_fate_entropy": max_ent,
            "fate_variance": fate_var,
            "frac_undecided": undecided,
            "frac_EGC_fate": fate_composition[0],
            "frac_GC_fate": fate_composition[1],
            "frac_Stasis_fate": fate_composition[2],
            "bifurcation_score": fate_var * (1 + undecided),
        })

    bif_df = pd.DataFrame(records)

    # Detect entropy gradient drops
    ent_smooth = bif_df["mean_fate_entropy"].rolling(3, center=True, min_periods=1).mean()
    bif_df["entropy_gradient"] = np.gradient(ent_smooth)

    # Bifurcation = where score is high AND entropy is dropping
    # Top bifurcation candidates
    bif_df["combined_score"] = (
        bif_df["bifurcation_score"] *
        np.maximum(-bif_df["entropy_gradient"], 0) * 10 +
        bif_df["bifurcation_score"]
    )

    # Find peaks in combined_score
    scores = bif_df["combined_score"].values
    peaks = []
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
            if scores[i] > np.percentile(scores, 60):
                peaks.append(i)

    # If no peaks found, take top N by score
    if len(peaks) == 0:
        peaks = list(bif_df.nlargest(3, "combined_score").index)

    bifurcations = bif_df.iloc[peaks].sort_values("mean_pseudotime").reset_index(drop=True)

    print(f"\n  Detected {len(bifurcations)} bifurcation events:")
    for i, bif in bifurcations.iterrows():
        print(f"    Bif {i+1}: pt={bif['mean_pseudotime']:.4f} "
              f"({bif['dominant_stage']}), "
              f"score={bif['combined_score']:.4f}, "
              f"undecided={bif['frac_undecided']:.2f}, "
              f"fate_var={bif['fate_variance']:.4f}")
        print(f"           EGC:{bif['frac_EGC_fate']:.2f} "
              f"GC:{bif['frac_GC_fate']:.2f} "
              f"Stasis:{bif['frac_Stasis_fate']:.2f}")

    return bif_df, bifurcations


# ===========================================================================
# Section 4: Growth Rate from Unbalanced OT
# ===========================================================================

def compute_growth_rates(adata_mc, ot_results):
    """Aggregate per-cell growth from stage-level unbalanced OT."""
    print("[4/7] Computing growth rates from unbalanced OT ...")

    n = adata_mc.n_obs
    growth = np.ones(n)
    n_contributions = np.zeros(n)

    for (s_from, s_to), res in ot_results.items():
        idx_from = res["idx_from"]
        gr = res["growth"]
        growth[idx_from] += gr
        n_contributions[idx_from] += 1

    # Average growth for cells with multiple contributions
    multi = n_contributions > 0
    growth[multi] = growth[multi] / (n_contributions[multi] + 1)

    adata_mc.obs["ot_growth_rate"] = growth

    stage_gr = adata_mc.obs.groupby("stage")["ot_growth_rate"].agg(["mean", "std"])
    print("\n  Growth rates by stage (unbalanced OT):")
    for stage in ["NAG", "CAG", "IM", "EGC", "EGC_multi_region", "GC"]:
        if stage in stage_gr.index:
            row = stage_gr.loc[stage]
            symbol = "+" if row["mean"] > 1.1 else ("-" if row["mean"] < 0.9 else "=")
            print(f"    {symbol} {stage:20s}: growth={row['mean']:.3f} +/- {row['std']:.3f}")

    return growth


# ===========================================================================
# Section 5: Molecular Markers at Bifurcations
# ===========================================================================

def characterize_bifurcations(adata_mc, bifurcations, fate_probs):
    """Find genes distinguishing cells with different fates at each bifurcation."""
    print("[5/7] Characterizing bifurcation markers ...")

    pt = adata_mc.obs["dpt_pseudotime"].values
    all_markers = []
    fate_names = ["EGC_fate", "GC_fate", "Stasis_fate"]

    for bif_idx, bif in bifurcations.iterrows():
        bif_pt = bif["mean_pseudotime"]
        # Window around bifurcation: +/- 10% of pseudotime range
        pt_range = pt.max() - pt.min()
        window = 0.1 * pt_range
        mask = np.abs(pt - bif_pt) <= window
        if mask.sum() < 20:
            window = 0.2 * pt_range
            mask = np.abs(pt - bif_pt) <= window

        bif_fates = fate_probs[mask]
        dominant = np.argmax(bif_fates, axis=1)
        unique_fates = np.unique(dominant)

        if len(unique_fates) < 2:
            continue

        bif_adata = adata_mc[mask]
        X_dense = bif_adata.X
        if sparse.issparse(X_dense):
            X_dense = X_dense.toarray()

        # Compare each pair of fates
        for fi in range(len(unique_fates)):
            for fj in range(fi + 1, len(unique_fates)):
                f1, f2 = unique_fates[fi], unique_fates[fj]
                mask1 = dominant == f1
                mask2 = dominant == f2
                if mask1.sum() < 5 or mask2.sum() < 5:
                    continue

                mean1 = X_dense[mask1].mean(axis=0)
                mean2 = X_dense[mask2].mean(axis=0)
                std_pooled = np.sqrt(
                    (X_dense[mask1].var(axis=0) + X_dense[mask2].var(axis=0)) / 2 + 1e-10
                )
                effects = (np.asarray(mean1) - np.asarray(mean2)).flatten()
                std_flat = np.asarray(std_pooled).flatten()
                cohen_d = effects / std_flat

                # Top 20 genes by absolute effect size
                top_idx = np.argsort(np.abs(cohen_d))[-20:]
                gene_names = bif_adata.var_names[top_idx]

                for gi, gname in enumerate(gene_names):
                    all_markers.append({
                        "bifurcation_id": bif_idx + 1,
                        "pseudotime": bif_pt,
                        "stage": bif["dominant_stage"],
                        "fate_A": fate_names[f1],
                        "fate_B": fate_names[f2],
                        "gene": gname,
                        "mean_A": float(mean1.flat[top_idx[gi]]),
                        "mean_B": float(mean2.flat[top_idx[gi]]),
                        "cohen_d": float(cohen_d[top_idx[gi]]),
                    })

    markers_df = pd.DataFrame(all_markers)
    if len(markers_df) > 0:
        markers_df = markers_df.sort_values("cohen_d", key=abs, ascending=False)
        print(f"\n  Found {len(markers_df)} marker-fate associations")
        print(f"  Top markers per bifurcation:")
        for bif_id in markers_df["bifurcation_id"].unique():
            sub = markers_df[markers_df["bifurcation_id"] == bif_id].head(5)
            print(f"    Bif {bif_id} ({sub.iloc[0]['stage']}):")
            for _, row in sub.iterrows():
                direction = "A>B" if row["cohen_d"] > 0 else "B>A"
                print(f"      {row['gene']:15s} d={row['cohen_d']:+.3f} "
                      f"({row['fate_A']} vs {row['fate_B']}, {direction})")

    return markers_df


# ===========================================================================
# Section 6: Cross-validation with IRL
# ===========================================================================

def cross_validate_irl(adata_mc, fate_probs, growth, val_df):
    """Compare OT results with IRL value function and policy entropy."""
    print("[6/7] Cross-validating OT with IRL ...")

    V = val_df["V_value"].values
    irl_entropy = val_df["policy_entropy"].values
    fate_entropy = adata_mc.obs["fate_entropy"].values

    # Hypothesis 1: High V cells should have high EGC fate probability
    rho_V_egc, p_V_egc = spearmanr(V, fate_probs[:, 0])
    # Hypothesis 2: IRL policy entropy correlates with OT fate entropy
    rho_ent, p_ent = spearmanr(irl_entropy, fate_entropy)
    # Hypothesis 3: Growth rate correlates with V (fitter cells proliferate)
    rho_gr_V, p_gr_V = spearmanr(growth, V)
    # Hypothesis 4: EGC-fated cells have higher proliferation score
    if "proliferation_score" in adata_mc.obs.columns:
        prolif = adata_mc.obs["proliferation_score"].values
        rho_egc_prolif, p_egc_prolif = spearmanr(fate_probs[:, 0], prolif)
    else:
        rho_egc_prolif, p_egc_prolif = 0, 1

    results = {
        "V_vs_EGC_fate": (rho_V_egc, p_V_egc),
        "IRL_entropy_vs_fate_entropy": (rho_ent, p_ent),
        "growth_vs_V": (rho_gr_V, p_gr_V),
        "EGC_fate_vs_proliferation": (rho_egc_prolif, p_egc_prolif),
    }

    print("\n  Cross-validation (Spearman correlations):")
    for name, (rho, p) in results.items():
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {name:35s}: rho={rho:+.3f}, p={p:.2e} {sig}")

    return results


# ===========================================================================
# Section 7: Visualization
# ===========================================================================

def plot_all(adata_mc, fate_probs, fate_entropy, bif_df, bifurcations,
             growth, val_df):
    """Generate Phase 24 figures."""
    import matplotlib.pyplot as plt

    print("[7/7] Generating plots ...")
    pt = adata_mc.obs["dpt_pseudotime"].values
    V = val_df["V_value"].values
    stages = adata_mc.obs["stage"].values
    stage_colors = {"NAG": "#4575b4", "CAG": "#91bfdb", "IM": "#fee090",
                    "EGC": "#d73027", "EGC_multi_region": "#fc8d59", "GC": "#1a1a1a"}
    colors = [stage_colors.get(s, "#999999") for s in stages]

    # --- Figure 1: Fate landscape ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.scatter(pt, fate_probs[:, 0], c="#d73027", s=5, alpha=0.3, label="EGC fate")
    ax.scatter(pt, fate_probs[:, 1], c="#1a1a1a", s=5, alpha=0.3, label="GC fate")
    ax.scatter(pt, fate_probs[:, 2], c="#4575b4", s=5, alpha=0.3, label="Stasis")
    ax.set_ylabel("Absorption probability")
    ax.set_title("Waddington Landscape: Fate Probabilities (from CellRank T)")
    ax.legend(loc="upper right")
    for _, bif in bifurcations.iterrows():
        ax.axvline(bif["mean_pseudotime"], color="gray", ls="--", lw=1.5)

    ax = axes[1]
    sc_plot = ax.scatter(pt, fate_entropy, c=colors, s=8, alpha=0.6)
    ax.set_ylabel("Fate entropy")
    ax.set_title("Fate Uncertainty (high = pre-bifurcation, multiple fates possible)")
    for i, bif in bifurcations.iterrows():
        ax.axvline(bif["mean_pseudotime"], color="gray", ls="--", lw=1.5)
        ax.text(bif["mean_pseudotime"], fate_entropy.max() * 0.95,
                f"Bif{i+1}", fontsize=9, ha="center", color="red")

    ax = axes[2]
    ax.scatter(pt, V, c=colors, s=8, alpha=0.6)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("V(s) from IRL")
    ax.set_title("IRL Fitness Landscape (cross-reference)")
    for _, bif in bifurcations.iterrows():
        ax.axvline(bif["mean_pseudotime"], color="gray", ls="--", lw=1.5)

    plt.tight_layout()
    plt.savefig(FIGURES / "ot_fate_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Bifurcation detection ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    bif_sorted = bif_df.sort_values("mean_pseudotime")
    ax.plot(bif_sorted["mean_pseudotime"], bif_sorted["fate_variance"],
            "o-", color="#2ca02c", ms=4, label="Fate variance")
    ax.fill_between(bif_sorted["mean_pseudotime"], 0,
                    bif_sorted["fate_variance"], alpha=0.15, color="#2ca02c")
    ax.set_ylabel("Fate variance (divergence)")
    ax.set_title("Bifurcation Signatures along Progression")
    ax.legend(loc="upper left")
    for i, bif in bifurcations.iterrows():
        ax.axvline(bif["mean_pseudotime"], color="red", ls="--", lw=2)

    ax = axes[1]
    ax.bar(bif_sorted["mean_pseudotime"],
           bif_sorted["frac_undecided"], width=0.003,
           color="#ff7f0e", alpha=0.7, label="Fraction undecided")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Fraction undecided cells")
    ax.set_title("Undecided Cells (fate prob < 0.6 for all fates)")
    ax.legend()
    for i, bif in bifurcations.iterrows():
        ax.axvline(bif["mean_pseudotime"], color="red", ls="--", lw=2)
        ax.text(bif["mean_pseudotime"], ax.get_ylim()[1] * 0.9,
                f"Bif{i+1}\n{bif['dominant_stage']}", fontsize=8,
                ha="center", color="red")

    plt.tight_layout()
    plt.savefig(FIGURES / "ot_bifurcation_tree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 3: Growth vs Value ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(pt, growth, c=colors, s=8, alpha=0.5)
    ax.axhline(1.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("OT Growth Rate")
    ax.set_title("Unbalanced OT Growth (>1=proliferating, <1=dying)")
    # Add stage legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s)
                       for s, c in stage_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    ax = axes[1]
    scatter = ax.scatter(V, growth, c=fate_probs[:, 0], cmap="Reds", s=8, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="P(EGC fate)")
    ax.axhline(1.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("V(s) from IRL (fitness)")
    ax.set_ylabel("OT Growth Rate")
    ax.set_title("IRL Fitness vs OT Proliferation")

    plt.tight_layout()
    plt.savefig(FIGURES / "ot_growth_vs_value.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: ot_fate_landscape.png, ot_bifurcation_tree.png, ot_growth_vs_value.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Phase 24: Waddington OT & Bifurcation Analysis")
    print("=" * 70)

    print("[0] Loading data ...")
    adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
    T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
    T_cr = T_sparse.toarray()
    val_df = pd.read_csv(RESULTS / "rl_value_function.csv")
    print(f"  {adata_mc.n_obs} metacells, T={T_cr.shape}, embedding=X_scVI")

    # Step 1: Stage-level OT (unbalanced -> growth rates)
    ot_results = stage_ot(adata_mc)

    # Step 2: Absorption probabilities (from CellRank T -> fate probs)
    fate_probs, fate_entropy = compute_absorption_probs(adata_mc, T_cr)

    # Step 3: Bifurcation detection
    bif_df, bifurcations = detect_bifurcations(adata_mc, fate_probs, fate_entropy)

    # Step 4: Growth rates
    growth = compute_growth_rates(adata_mc, ot_results)

    # Step 5: Molecular markers at bifurcations
    markers_df = characterize_bifurcations(adata_mc, bifurcations, fate_probs)

    # Step 6: Cross-validation with IRL
    xval = cross_validate_irl(adata_mc, fate_probs, growth, val_df)

    # Step 7: Plots
    plot_all(adata_mc, fate_probs, fate_entropy, bif_df, bifurcations,
             growth, val_df)

    # Save
    print("\n[SAVE] Writing outputs ...")
    bif_df.to_csv(RESULTS / "ot_bifurcation_scores.csv", index=False)
    bifurcations.to_csv(RESULTS / "ot_bifurcations.csv", index=False)
    if len(markers_df) > 0:
        markers_df.to_csv(RESULTS / "ot_bifurcation_markers.csv", index=False)

    fate_out = adata_mc.obs[["sample_id", "stage", "dpt_pseudotime",
                              "fate_EGC", "fate_GC", "fate_Stasis",
                              "fate_entropy", "ot_growth_rate"]].copy()
    fate_out.to_csv(RESULTS / "ot_fate_probabilities.csv", index=False)

    gr_summary = adata_mc.obs.groupby("stage").agg(
        mean_growth=("ot_growth_rate", "mean"),
        std_growth=("ot_growth_rate", "std"),
        mean_fate_EGC=("fate_EGC", "mean"),
        mean_fate_GC=("fate_GC", "mean"),
        mean_fate_entropy=("fate_entropy", "mean"),
    ).reset_index()
    gr_summary.to_csv(RESULTS / "ot_growth_rates.csv", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("Phase 24 COMPLETE")
    print("-" * 70)
    print(f"\n  BIFURCATIONS DETECTED: {len(bifurcations)}")
    for i, bif in bifurcations.iterrows():
        print(f"    [{i+1}] pseudotime={bif['mean_pseudotime']:.4f}, "
              f"stage={bif['dominant_stage']}, "
              f"undecided={bif['frac_undecided']:.0%}")
        print(f"        Fate split: EGC={bif['frac_EGC_fate']:.0%} | "
              f"GC={bif['frac_GC_fate']:.0%} | "
              f"Stasis={bif['frac_Stasis_fate']:.0%}")

    print(f"\n  CROSS-VALIDATION:")
    for name, (rho, p) in xval.items():
        print(f"    {name}: rho={rho:+.3f} (p={p:.2e})")

    print("=" * 70)


if __name__ == "__main__":
    main()