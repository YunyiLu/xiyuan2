"""
Phase 23: Cooperation & Strategy Identification
Part of the IRL pipeline (PLAN_RL.md Layer 3)

Input:  script3/results/rl_reward_weights.csv
        script3/results/rl_value_function.csv
        script3/results/rl_transition_matrix.npz
        script3/data/rl_metacells.h5ad
Output: script3/results/rl_cooperation_ranking.csv
        script3/results/rl_critical_transitions.csv
        script3/results/rl_strategy_decomposition.csv
        script3/figures/rl_cooperation_shapley.png
        script3/figures/rl_strategy_timeline.png
        script3/figures/rl_entropy_curve.png

Requires: numpy, scipy, pandas, matplotlib, itertools
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")

# --- Paths ---
BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# Feature names (must match Phase 22)
REWARD_FEATURES = [
    "proliferation", "stemness", "apoptosis_resistance",
    "differentiation_loss", "inflammatory_NFkB", "metabolic_shift",
    "myeloid_niche", "fibroblast_niche", "T_cell_pressure", "spatial_border",
]

# Intrinsic vs niche partition
INTRINSIC_IDX = [0, 1, 2, 3, 4, 5]  # first 6 = cell-intrinsic
NICHE_IDX = [6, 7, 8, 9]             # last 4 = microenvironment
NICHE_NAMES = ["myeloid_niche", "fibroblast_niche", "T_cell_pressure", "spatial_border"]


# ===========================================================================
# Section 1: Shapley Value Decomposition of Niche Contributions
# ===========================================================================

def shapley_niche_contribution(theta: np.ndarray, phi: np.ndarray,
                                V: np.ndarray) -> pd.DataFrame:
    """
    Compute Shapley values for niche features' contribution to total reward.

    For each niche feature i, Shapley value = average marginal contribution
    of feature i across all possible subsets of niche features.

    This answers: "Who contributes most to the cancer cell's fitness?"
    """
    print("[1/4] Computing Shapley values for niche contributions ...")

    n_niche = len(NICHE_IDX)
    niche_theta = theta[NICHE_IDX]
    niche_phi = phi[:, NICHE_IDX]

    # Full niche contribution per metacell
    full_niche_reward = niche_phi @ niche_theta

    # Shapley: for each feature, average marginal contribution
    shapley_values = np.zeros(n_niche)

    for i in range(n_niche):
        marginals = []
        # All subsets S not containing i
        others = [j for j in range(n_niche) if j != i]
        for size in range(n_niche):
            for subset in combinations(others, size):
                subset_list = list(subset)
                # v(S ∪ {i}) - v(S)
                with_i = subset_list + [i]
                val_with = np.mean(niche_phi[:, with_i] @ niche_theta[with_i])
                val_without = np.mean(niche_phi[:, subset_list] @ niche_theta[subset_list]) \
                    if subset_list else 0.0
                marginals.append(val_with - val_without)

        shapley_values[i] = np.mean(marginals)

    # Normalize to sum to total niche contribution
    total_niche = np.mean(full_niche_reward)
    if abs(total_niche) > 1e-10:
        shapley_normalized = shapley_values / np.sum(np.abs(shapley_values)) * total_niche
    else:
        shapley_normalized = shapley_values

    # Build result
    shapley_df = pd.DataFrame({
        "niche_component": NICHE_NAMES,
        "shapley_value": shapley_values,
        "shapley_normalized": shapley_normalized,
        "theta": niche_theta,
        "mean_phi": niche_phi.mean(axis=0),
        "interpretation": [
            "Cooperator (positive = helps cancer)" if sv > 0 else "Suppressor (negative = resists)"
            for sv in shapley_values
        ],
    }).sort_values("shapley_value", ascending=False)

    print("\n  Niche Shapley values (who cooperates?):")
    for _, row in shapley_df.iterrows():
        symbol = "+" if row["shapley_value"] > 0 else "-"
        print(f"    {symbol} {row['niche_component']:20s}: "
              f"Shapley={row['shapley_value']:+.4f} "
              f"(theta={row['theta']:+.3f}, mean_phi={row['mean_phi']:.3f})")

    return shapley_df


# ===========================================================================
# Section 2: Critical Transition Identification
# ===========================================================================

def identify_critical_transitions(T: np.ndarray, V: np.ndarray, phi: np.ndarray,
                                   theta: np.ndarray, adata_mc: pd.DataFrame,
                                   top_k: int = 10) -> pd.DataFrame:
    """
    Find transitions with highest value gradient: ΔV = V(s') - V(s).
    For each, decompose which φ features changed most.

    This answers: "What strategy switches are critical?"
    """
    print("[2/4] Identifying critical transitions ...")

    n = T.shape[0]
    records = []

    for s in range(n):
        for sp in range(n):
            if T[s, sp] < 0.01:
                continue
            delta_V = V[sp] - V[s]
            if delta_V <= 0:
                continue

            # Feature changes
            delta_phi = phi[sp] - phi[s]
            # Weighted contribution to reward change
            delta_reward = delta_phi * theta

            records.append({
                "from_mc": s,
                "to_mc": sp,
                "transition_prob": T[s, sp],
                "delta_V": delta_V,
                "from_stage": adata_mc.obs.iloc[s]["stage"],
                "to_stage": adata_mc.obs.iloc[sp]["stage"],
                "from_pseudotime": adata_mc.obs.iloc[s]["dpt_pseudotime"],
                "to_pseudotime": adata_mc.obs.iloc[sp]["dpt_pseudotime"],
                **{f"delta_{REWARD_FEATURES[j]}": delta_phi[j] for j in range(len(REWARD_FEATURES))},
                **{f"reward_contrib_{REWARD_FEATURES[j]}": delta_reward[j] for j in range(len(REWARD_FEATURES))},
            })

    trans_df = pd.DataFrame(records)
    trans_df = trans_df.sort_values("delta_V", ascending=False).head(top_k * 5)

    # For top transitions, identify dominant strategy shift
    trans_df["dominant_feature"] = trans_df[[
        f"reward_contrib_{f}" for f in REWARD_FEATURES
    ]].apply(lambda row: REWARD_FEATURES[np.argmax(np.abs(row.values))], axis=1)

    top_transitions = trans_df.head(top_k)

    print(f"\n  Top {top_k} critical transitions:")
    for _, row in top_transitions.iterrows():
        print(f"    {row['from_stage']}->{row['to_stage']} "
              f"(dV={row['delta_V']:.3f}, p(t)={row['transition_prob']:.3f}): "
              f"dominant={row['dominant_feature']}")

    return trans_df


# ===========================================================================
# Section 3: Strategy Decomposition per Macrostate
# ===========================================================================

def strategy_decomposition(phi: np.ndarray, theta: np.ndarray, V: np.ndarray,
                            entropy: np.ndarray, adata_mc) -> pd.DataFrame:
    """
    For each macrostate, compute the average strategy profile (phi * theta contribution).
    This answers: "What is the dominant strategy at each disease stage?"
    """
    print("[3/4] Decomposing strategy per macrostate ...")

    macrostates = adata_mc.obs["macrostate"].dropna().unique()
    records = []

    for ms in macrostates:
        mask = (adata_mc.obs["macrostate"] == ms).values
        if mask.sum() == 0:
            continue

        phi_ms = phi[mask].mean(axis=0)
        reward_contrib = phi_ms * theta
        total_reward = reward_contrib.sum()
        intrinsic_reward = reward_contrib[INTRINSIC_IDX].sum()
        niche_reward = reward_contrib[NICHE_IDX].sum()

        record = {
            "macrostate": ms,
            "n_metacells": int(mask.sum()),
            "mean_V": float(V[mask].mean()),
            "mean_entropy": float(entropy[mask].mean()),
            "total_reward": float(total_reward),
            "intrinsic_reward": float(intrinsic_reward),
            "niche_reward": float(niche_reward),
            "dominant_stage": adata_mc.obs.loc[mask, "stage"].mode().iloc[0],
            "mean_pseudotime": float(adata_mc.obs.loc[mask, "dpt_pseudotime"].mean()),
        }

        # Per-feature contributions
        for j, feat in enumerate(REWARD_FEATURES):
            record[f"reward_{feat}"] = float(reward_contrib[j])
            record[f"phi_{feat}"] = float(phi_ms[j])

        # Dominant strategy = feature with highest absolute reward contribution
        record["dominant_strategy"] = REWARD_FEATURES[np.argmax(np.abs(reward_contrib))]

        records.append(record)

    strat_df = pd.DataFrame(records).sort_values("mean_pseudotime")

    print("\n  Strategy decomposition per macrostate:")
    for _, row in strat_df.iterrows():
        print(f"    {row['macrostate']} ({row['dominant_stage']}): "
              f"V={row['mean_V']:.3f}, H={row['mean_entropy']:.3f}, "
              f"strategy={row['dominant_strategy']}")

    return strat_df


# ===========================================================================
# Section 4: Visualization
# ===========================================================================

def plot_all(shapley_df, trans_df, strat_df, V, entropy, adata_mc):
    """Generate all Layer 3 figures."""
    import matplotlib.pyplot as plt

    print("[4/4] Generating plots ...")

    # --- Plot 1: Shapley bar ---
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#d73027' if v > 0 else '#4575b4' for v in shapley_df["shapley_value"]]
    ax.barh(range(len(shapley_df)), shapley_df["shapley_value"].values, color=colors)
    ax.set_yticks(range(len(shapley_df)))
    ax.set_yticklabels(shapley_df["niche_component"].values)
    ax.set_xlabel("Shapley Value (+ = cooperator, - = suppressor)")
    ax.set_title("Who Helps Cancer? Niche Cooperation Ranking")
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES / "rl_cooperation_shapley.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Strategy timeline ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel A: Reward contribution stacked bar
    ax = axes[0]
    strat_sorted = strat_df.sort_values("mean_pseudotime")
    x = range(len(strat_sorted))
    labels = strat_sorted["macrostate"].values

    feature_colors = plt.cm.Set3(np.linspace(0, 1, len(REWARD_FEATURES)))
    bottom_pos = np.zeros(len(strat_sorted))
    bottom_neg = np.zeros(len(strat_sorted))

    for j, feat in enumerate(REWARD_FEATURES):
        vals = strat_sorted[f"reward_{feat}"].values
        pos_vals = np.maximum(vals, 0)
        neg_vals = np.minimum(vals, 0)
        ax.bar(x, pos_vals, bottom=bottom_pos, color=feature_colors[j],
               label=feat, width=0.7)
        ax.bar(x, neg_vals, bottom=bottom_neg, color=feature_colors[j],
               width=0.7, alpha=0.5)
        bottom_pos += pos_vals
        bottom_neg += neg_vals

    ax.set_ylabel("Reward contribution")
    ax.set_title("Strategy Evolution along Progression")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel B: Entropy curve
    ax = axes[1]
    ax.bar(x, strat_sorted["mean_entropy"].values, color="#2ca02c", alpha=0.7)
    ax.set_xlabel("Macrostate (ordered by pseudotime)")
    ax.set_ylabel("Policy entropy H(s)")
    ax.set_title("Strategy Certainty (lower = more determined path)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES / "rl_strategy_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Entropy vs pseudotime scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    pt = adata_mc.obs["dpt_pseudotime"].values
    scatter = ax.scatter(pt, entropy, c=V, cmap="RdYlBu_r", s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="V(s) value")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Policy Entropy H(s)")
    ax.set_title("From Disorder to Determination\n(lower entropy = cancer has 'chosen' its path)")
    plt.tight_layout()
    plt.savefig(FIGURES / "rl_entropy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: rl_cooperation_shapley.png, rl_strategy_timeline.png, rl_entropy_curve.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Phase 23: Cooperation & Strategy Identification (PLAN_RL.md Layer 3)")
    print("=" * 70)

    # Load inputs
    print("[0] Loading Phase 21-22 outputs ...")
    adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
    T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
    T = T_sparse.toarray()

    theta_df = pd.read_csv(RESULTS / "rl_reward_weights.csv")
    theta = theta_df["theta"].values

    val_df = pd.read_csv(RESULTS / "rl_value_function.csv")
    V = val_df["V_value"].values
    entropy = val_df["policy_entropy"].values

    # Recompute phi (must match Phase 22 exactly)
    from importlib import import_module
    phase22 = import_module("22_fitness_landscape_irl")
    phi = phase22.compute_reward_features(adata_mc)

    print(f"  Loaded: {adata_mc.n_obs} metacells, theta shape={theta.shape}, "
          f"T shape={T.shape}")

    # Step 1: Shapley
    shapley_df = shapley_niche_contribution(theta, phi, V)
    shapley_df.to_csv(RESULTS / "rl_cooperation_ranking.csv", index=False)

    # Step 2: Critical transitions
    trans_df = identify_critical_transitions(T, V, phi, theta, adata_mc)
    trans_df.to_csv(RESULTS / "rl_critical_transitions.csv", index=False)

    # Step 3: Strategy decomposition
    strat_df = strategy_decomposition(phi, theta, V, entropy, adata_mc)
    strat_df.to_csv(RESULTS / "rl_strategy_decomposition.csv", index=False)

    # Step 4: Plots
    plot_all(shapley_df, trans_df, strat_df, V, entropy, adata_mc)

    # Final summary
    print("\n" + "=" * 70)
    print("Phase 23 COMPLETE -- Key Findings:")
    print("-" * 70)

    # Who cooperates?
    top_coop = shapley_df.iloc[0]
    print(f"\n  Top cooperator: {top_coop['niche_component']} "
          f"(Shapley = {top_coop['shapley_value']:+.4f})")

    # What strategy at what stage?
    for _, row in strat_df.iterrows():
        print(f"  {row['macrostate']} ({row['dominant_stage']}): "
              f"dominant strategy = {row['dominant_strategy']}")

    # Entropy trend (disorder -> determination)
    early_ent = entropy[adata_mc.obs["dpt_pseudotime"].values < 0.3].mean()
    late_ent = entropy[adata_mc.obs["dpt_pseudotime"].values > 0.7].mean()
    trend = "CONFIRMED" if early_ent > late_ent * 1.2 else "NOT confirmed"
    print(f"\n  'Disorder->Determination' hypothesis: {trend}")
    print(f"    Early entropy: {early_ent:.3f}")
    print(f"    Late entropy:  {late_ent:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
