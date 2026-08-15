"""
Phase 22: Fitness Landscape Recovery via MaxEnt State-Only IRL
Part of the IRL pipeline (PLAN_RL.md Layer 2)

Input:  script3/data/rl_metacells.h5ad
        script3/results/rl_transition_matrix.npz
Output: script3/results/rl_reward_weights.csv
        script3/results/rl_value_function.csv
        script3/results/rl_policy_entropy.csv
        script3/results/rl_irl_diagnostics.csv
        script3/figures/rl_fitness_landscape.png
        script3/figures/rl_reward_bar.png

Requires: numpy, scipy, scanpy, anndata, matplotlib
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Paths ---
BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# --- IRL Hyperparameters ---
CONFIG = {
    "gamma": 0.95,            # discount: [0.80, 0.90, 0.95, 0.99]
    "tau": 1.0,               # temperature: [0.1, 0.3, 1.0, 3.0]
    "lambda_l1": 0.01,        # L1 reg: [0, 1e-4, 1e-3, 1e-2, 0.1]
    "lambda_l2": 0.01,        # L2 reg: [1e-4, 1e-3, 1e-2, 0.1, 1]
    "learning_rate": 0.02,
    "max_iter": 200,
    "convergence_tol": 1e-3,
    "n_seeds": 3,
    "random_seed": 42,
}

# Pre-frozen reward features (10 dimensions, MUST NOT change after seeing results)
REWARD_FEATURES = [
    "proliferation",        # MKI67 + TOP2A + PCNA
    "stemness",             # LGR5 + OLFM4 + SOX9 + ASCL2
    "apoptosis_resistance", # BCL2 + MCL1 - BAX - CASP3
    "differentiation_loss", # -(GKN1 + PGC + TFF1 + MUC5AC)
    "inflammatory_NFkB",    # RELA regulon activity (decoupler, Phase 6)
    "metabolic_shift",      # glycolysis_score - oxphos_score
    "myeloid_niche",        # Mono+Macro fraction in same sample
    "fibroblast_niche",     # Fibroblast fraction in same sample
    "T_cell_pressure",      # CD8+T fraction (negative = evasion)
    "spatial_border",       # inverse distance to Tumor region (Visium only)
]

# Gene sets for feature computation
GENE_SETS = {
    "proliferation_pos": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1"],
    "stemness_pos": ["LGR5", "OLFM4", "SOX9", "ASCL2", "CD44"],
    "apoptosis_pos": ["BCL2", "MCL1", "BIRC5"],
    "apoptosis_neg": ["BAX", "CASP3", "CASP9", "BID"],
    "differentiation_pos": ["GKN1", "PGC", "TFF1", "MUC5AC", "GIF"],
    "glycolysis": ["HK2", "PKM", "LDHA", "ENO1", "GAPDH", "PGK1"],
    "oxphos": ["MT-CO1", "MT-CO2", "MT-ND1", "COX5A", "NDUFA1", "ATP5F1A"],
}


# ===========================================================================
# Section 1: Compute reward features φ(s) for each metacell
# ===========================================================================

def compute_reward_features(adata_mc: ad.AnnData) -> np.ndarray:
    """
    Compute the 10-dimensional reward feature vector for each metacell.
    Features are z-scored across metacells for comparability.
    """
    print("[1/5] Computing reward features phi(s) ...")

    n_mc = adata_mc.n_obs
    phi = np.zeros((n_mc, len(REWARD_FEATURES)))
    var_names = list(adata_mc.var_names)

    def gene_set_score(genes_pos, genes_neg=None):
        """Mean expression of positive genes minus negative genes."""
        pos_idx = [var_names.index(g) for g in genes_pos if g in var_names]
        score = np.zeros(n_mc)
        if pos_idx:
            X = adata_mc.X
            if sparse.issparse(X):
                X = X.toarray()
            score = X[:, pos_idx].mean(axis=1)
        if genes_neg:
            neg_idx = [var_names.index(g) for g in genes_neg if g in var_names]
            if neg_idx:
                score -= X[:, neg_idx].mean(axis=1)
        return score

    # Feature 0: proliferation
    phi[:, 0] = gene_set_score(GENE_SETS["proliferation_pos"])

    # Feature 1: stemness
    phi[:, 1] = gene_set_score(GENE_SETS["stemness_pos"])

    # Feature 2: apoptosis resistance (pos - neg)
    phi[:, 2] = gene_set_score(GENE_SETS["apoptosis_pos"], GENE_SETS["apoptosis_neg"])

    # Feature 3: differentiation loss (negate differentiation score)
    phi[:, 3] = -gene_set_score(GENE_SETS["differentiation_pos"])

    # Feature 4: inflammatory NF-κB (from RELA regulon activity if available)
    if "RELA_activity" in adata_mc.obs.columns:
        phi[:, 4] = adata_mc.obs["RELA_activity"].values
    elif "inflammatory_score" in adata_mc.obs.columns:
        phi[:, 4] = adata_mc.obs["inflammatory_score"].values
    else:
        # Fallback: NF-κB target genes (CCL3, CXCL8, IL6, TNFAIP3)
        phi[:, 4] = gene_set_score(["CCL3", "CXCL8", "IL6", "TNFAIP3", "NFKBIA"])

    # Feature 5: metabolic shift (glycolysis - OXPHOS)
    phi[:, 5] = gene_set_score(GENE_SETS["glycolysis"]) - gene_set_score(GENE_SETS["oxphos"])

    # Features 6-8: niche composition from sample-level cell fractions
    niche_data = _compute_niche_fractions(adata_mc)
    phi[:, 6] = niche_data["myeloid_fraction"]
    phi[:, 7] = niche_data["fibroblast_fraction"]
    phi[:, 8] = niche_data["T_cell_fraction"]  # will be negated in reward (pressure)

    # Feature 9: spatial border (only for OMIX010346 patients)
    phi[:, 9] = _compute_spatial_border(adata_mc)

    # Z-score normalize each feature
    from scipy.stats import zscore
    phi_z = np.zeros_like(phi)
    for j in range(phi.shape[1]):
        col = phi[:, j]
        std = col.std()
        if std > 1e-10:
            phi_z[:, j] = (col - col.mean()) / std
        else:
            phi_z[:, j] = 0.0

    print(f"  Feature matrix: {phi_z.shape} (z-scored)")
    return phi_z


def _compute_niche_fractions(adata_mc: ad.AnnData) -> dict:
    """
    Load pre-computed niche fractions per sample (from prep_niche_fractions.py).
    Maps sample_id to myeloid/fibroblast/T-cell fractions for each metacell.
    """
    n_mc = adata_mc.n_obs
    myeloid_frac = np.zeros(n_mc)
    fibro_frac = np.zeros(n_mc)
    tcell_frac = np.zeros(n_mc)

    niche_csv = DATA / "niche_fractions.csv"
    if not niche_csv.exists():
        print("    WARNING: niche_fractions.csv not found. Run prep_niche_fractions.py first.")
        return {"myeloid_fraction": myeloid_frac,
                "fibroblast_fraction": fibro_frac,
                "T_cell_fraction": tcell_frac}

    niche_df = pd.read_csv(niche_csv).set_index("sample_id")
    print(f"    Loaded niche fractions for {len(niche_df)} samples")

    for i, sample in enumerate(adata_mc.obs["sample_id"]):
        if sample in niche_df.index:
            myeloid_frac[i] = niche_df.loc[sample, "myeloid_fraction"]
            fibro_frac[i] = niche_df.loc[sample, "fibroblast_fraction"]
            tcell_frac[i] = niche_df.loc[sample, "T_cell_fraction"]

    return {"myeloid_fraction": myeloid_frac,
            "fibroblast_fraction": fibro_frac,
            "T_cell_fraction": tcell_frac}


def _compute_spatial_border(adata_mc: ad.AnnData) -> np.ndarray:
    """
    Spatial border feature: only valid for OMIX010346 patients.
    For others, returns 0 (missing data handled in IRL gradient).
    """
    n_mc = adata_mc.n_obs
    border = np.zeros(n_mc)
    omix_mask = adata_mc.obs["dataset"].str.contains("OMIX", na=False)
    if omix_mask.sum() > 0 and "spatial_border_score" in adata_mc.obs.columns:
        border[omix_mask.values] = adata_mc.obs.loc[omix_mask, "spatial_border_score"].values
    return border


# ===========================================================================
# Section 2: Construct demonstration trajectories from stage-ordered metacells
# ===========================================================================

def build_trajectories(adata_mc: ad.AnnData, T: sparse.csr_matrix) -> list:
    """
    Build 'demonstration' trajectories from stage-ordered metacell sequences.
    Each trajectory = sequence of metacell indices (within one patient if possible).
    """
    print("[2/5] Building demonstration trajectories ...")

    trajectories = []
    n_mc = adata_mc.n_obs

    # Per-patient: sort metacells by pseudotime → trajectory
    for sample in adata_mc.obs["sample_id"].unique():
        mask = adata_mc.obs["sample_id"] == sample
        idx = np.where(mask)[0]
        if len(idx) < 3:
            continue

        sub = adata_mc.obs.iloc[idx]
        order = np.argsort(sub["dpt_pseudotime"].values)
        traj = idx[order].tolist()
        trajectories.append(traj)

    # Cross-patient trajectories (NAG→IM→EGC ordering)
    stage_groups = {}
    for stage in ["NAG", "CAG", "IM", "EGC", "GC"]:
        mask = adata_mc.obs["stage"] == stage
        stage_groups[stage] = np.where(mask)[0]

    # Build cross-patient trajectories by chaining stage representatives
    for s1, s2, s3 in [("NAG", "IM", "EGC"), ("NAG", "IM", "GC")]:
        if all(len(stage_groups.get(s, [])) >= 2 for s in [s1, s2, s3]):
            # Sample random paths through each stage
            rng = np.random.default_rng(42)
            for _ in range(20):
                traj = []
                for stage in [s1, s2, s3]:
                    choices = stage_groups[stage]
                    traj.append(int(rng.choice(choices)))
                trajectories.append(traj)

    print(f"  Built {len(trajectories)} trajectories "
          f"(per-patient: {sum(1 for t in trajectories if len(t) > 5)}, "
          f"cross-patient: {sum(1 for t in trajectories if len(t) <= 5)})")
    return trajectories


# ===========================================================================
# Section 3: MaxEnt State-Only IRL
# ===========================================================================

def soft_value_iteration(log_T: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                          gamma: float, tau: float, max_iter: int = 50,
                          tol: float = 0.01, V_init: np.ndarray = None) -> np.ndarray:
    """
    Soft (entropy-regularized) value iteration with warm-starting.
    V(s) = r(s) + gamma*tau*logsumexp_{s'}[log P(s'|s) + V(s')/tau]
    """
    n = log_T.shape[0]
    r = phi @ theta
    V = V_init.copy() if V_init is not None else r.copy()

    for _ in range(max_iter):
        Q = log_T + V[np.newaxis, :] / tau
        Q_max = Q.max(axis=1, keepdims=True)
        log_sum_exp = Q_max.squeeze() + np.log(np.exp(Q - Q_max).sum(axis=1))
        V_new = r + gamma * tau * log_sum_exp

        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    return V


def compute_state_visitation(log_T: np.ndarray, V: np.ndarray, phi: np.ndarray,
                              trajectories: list, tau: float) -> np.ndarray:
    """
    Compute expected state visitation under soft-optimal policy.
    Uses discounted visitation truncated at L=20 steps.
    """
    n = log_T.shape[0]

    # Soft policy
    log_pi = log_T + V[np.newaxis, :] / tau
    log_pi -= log_pi.max(axis=1, keepdims=True)
    pi = np.exp(log_pi)
    pi /= pi.sum(axis=1, keepdims=True) + 1e-300

    # Initial state distribution from trajectories
    d0 = np.zeros(n)
    for traj in trajectories:
        d0[traj[0]] += 1.0
    d0 /= d0.sum() + 1e-300

    # Discounted state visitation: mu = d0 + gamma*d0@pi + gamma^2*d0@pi^2 + ...
    # Truncate at L=20 steps (sufficient for gamma=0.95: gamma^20 = 0.36)
    gamma_trunc = 0.95
    L = 20
    mu = d0.copy()
    visit = d0.copy()
    for t in range(L):
        visit = visit @ pi * gamma_trunc
        mu += visit

    mu /= (mu.sum() + 1e-300)
    return mu  # (n,) state visitation frequencies


def maxent_irl(T: np.ndarray, phi: np.ndarray, trajectories: list,
               config: dict, seed: int = 42) -> dict:
    """
    MaxEnt State-Only IRL main loop.

    Returns dict with: theta, V, convergence history, diagnostics.
    """
    rng = np.random.default_rng(seed)
    n, p = phi.shape
    gamma = config["gamma"]
    tau = config["tau"]
    lr = config["learning_rate"]
    l1 = config["lambda_l1"]
    l2 = config["lambda_l2"]
    max_iter = config["max_iter"]
    tol = config["convergence_tol"]

    # Precompute log_T once
    log_T = np.log(T + 1e-300)

    # Initialize theta
    theta = rng.normal(0, 0.01, size=p)

    # Empirical feature expectations from demonstrations
    mu_demo = np.zeros(p)
    total_steps = 0
    for traj in trajectories:
        for s in traj:
            mu_demo += phi[s]
            total_steps += 1
    mu_demo /= total_steps

    history = []
    V_prev = None
    best_theta = theta.copy()
    best_grad_norm = np.inf
    for it in range(max_iter):
        # Value iteration (warm-started from previous V)
        V = soft_value_iteration(log_T, theta, phi, gamma, tau, V_init=V_prev)
        V_prev = V

        # Model state visitation -> model feature expectations
        mu_model = compute_state_visitation(log_T, V, phi, trajectories, tau)
        mu_theta = phi.T @ mu_model
        mu_theta /= (np.abs(mu_theta).sum() + 1e-300)

        # Gradient
        grad = mu_demo - mu_theta
        grad -= l1 * np.sign(theta)
        grad -= 2 * l2 * theta

        # Track best theta (minimum grad_norm)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < best_grad_norm:
            best_grad_norm = grad_norm
            best_theta = theta.copy()

        # Update with gradient clipping
        grad_clip = np.clip(grad, -0.5, 0.5)
        theta = theta + lr * grad_clip

        history.append(grad_norm)
        if grad_norm < tol:
            print(f"    Converged at iteration {it}")
            break

        if (it + 1) % 50 == 0 or it < 3:
            print(f"    iter {it+1}: grad_norm={grad_norm:.6f}, "
                  f"theta_norm={np.linalg.norm(theta):.4f}")

    # Use best theta found
    theta = best_theta
    print(f"    Using best theta (grad_norm={best_grad_norm:.6f})")

    # Final V and policy (use precomputed log_T)
    V_final = soft_value_iteration(log_T, theta, phi, gamma, tau)

    # Policy entropy at each state
    log_pi = log_T + V_final[np.newaxis, :] / tau
    log_pi -= log_pi.max(axis=1, keepdims=True)
    pi = np.exp(log_pi)
    pi /= pi.sum(axis=1, keepdims=True) + 1e-300
    entropy = -(pi * np.log(pi + 1e-300)).sum(axis=1)

    return {
        "theta": theta,
        "V": V_final,
        "policy_entropy": entropy,
        "pi": pi,
        "convergence_history": history,
        "mu_demo": mu_demo,
    }


# ===========================================================================
# Section 4: Stability analysis (multi-seed + LODO)
# ===========================================================================

def stability_analysis(T: np.ndarray, phi: np.ndarray, trajectories: list,
                        adata_mc: ad.AnnData, config: dict) -> pd.DataFrame:
    """
    Run IRL with multiple seeds and leave-one-dataset-out.
    Report theta stability.
    """
    print("[4/5] Running stability analysis ...")

    results = []

    # Multi-seed
    for seed_i in range(config["n_seeds"]):
        s = config["random_seed"] + seed_i * 100
        res = maxent_irl(T, phi, trajectories, config, seed=s)
        results.append({
            "run": f"seed_{seed_i}",
            **{f"theta_{REWARD_FEATURES[j]}": res["theta"][j]
               for j in range(len(REWARD_FEATURES))},
        })

    # Leave-one-dataset-out (LODO)
    datasets = adata_mc.obs["dataset"].unique()
    for ds_held in datasets:
        mask_train = (adata_mc.obs["dataset"] != ds_held).values
        idx_train = np.where(mask_train)[0]
        if len(idx_train) < 50:
            continue

        # Subset transition matrix and features
        T_sub = T[np.ix_(idx_train, idx_train)]
        T_sub /= (T_sub.sum(axis=1, keepdims=True) + 1e-300)
        phi_sub = phi[idx_train]

        # Rebuild trajectories for subset
        idx_map = {old: new for new, old in enumerate(idx_train)}
        trajs_sub = []
        for traj in trajectories:
            traj_mapped = [idx_map[s] for s in traj if s in idx_map]
            if len(traj_mapped) >= 3:
                trajs_sub.append(traj_mapped)

        if len(trajs_sub) < 5:
            continue

        res = maxent_irl(T_sub, phi_sub, trajs_sub, config, seed=config["random_seed"])
        results.append({
            "run": f"LODO_without_{ds_held}",
            **{f"theta_{REWARD_FEATURES[j]}": res["theta"][j]
               for j in range(len(REWARD_FEATURES))},
        })

    df_stability = pd.DataFrame(results)

    # Compute sign consistency: for each feature, fraction of runs with same sign
    theta_cols = [c for c in df_stability.columns if c.startswith("theta_")]
    sign_consistency = {}
    for col in theta_cols:
        vals = df_stability[col].values
        pos_frac = (vals > 0).mean()
        sign_consistency[col.replace("theta_", "")] = max(pos_frac, 1 - pos_frac)

    print("  Sign consistency across runs:")
    for feat, cons in sorted(sign_consistency.items(), key=lambda x: -x[1]):
        status = "PASS" if cons >= 0.8 else "WARN"
        print(f"    {feat}: {cons:.2f} {status}")

    return df_stability


# ===========================================================================
# Section 5: Negative control — permuted stage labels
# ===========================================================================

def negative_control(T: np.ndarray, phi: np.ndarray, adata_mc: ad.AnnData,
                      config: dict, n_perms: int = 10) -> pd.DataFrame:
    """
    Negative control: shuffle stage assignments, rebuild trajectories, run IRL.
    Under null, theta should have no consistent direction.
    """
    print("[4b/5] Running negative control (permuted stages) ...")

    rng = np.random.default_rng(config["random_seed"] + 7777)
    null_thetas = []

    for perm_i in range(n_perms):
        # Shuffle pseudotime
        perm_pt = rng.permutation(adata_mc.obs["dpt_pseudotime"].values)
        adata_mc.obs["_perm_pt"] = perm_pt

        # Build fake trajectories from shuffled pseudotime
        fake_trajs = []
        for sample in adata_mc.obs["sample_id"].unique():
            mask = adata_mc.obs["sample_id"] == sample
            idx = np.where(mask)[0]
            if len(idx) < 3:
                continue
            order = np.argsort(perm_pt[idx])
            fake_trajs.append(idx[order].tolist())

        if len(fake_trajs) < 5:
            continue

        res = maxent_irl(T, phi, fake_trajs, config, seed=config["random_seed"] + perm_i)
        null_thetas.append(res["theta"])

    null_thetas = np.array(null_thetas)
    null_df = pd.DataFrame(null_thetas, columns=REWARD_FEATURES)
    null_df["run_type"] = "null_permutation"

    # Check: under null, all theta CIs should cross 0
    for j, feat in enumerate(REWARD_FEATURES):
        vals = null_thetas[:, j]
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        contains_zero = ci_low <= 0 <= ci_high
        print(f"  Null {feat}: [{ci_low:.3f}, {ci_high:.3f}] "
              f"{'PASS crosses 0' if contains_zero else 'WARN does NOT cross 0'}")

    return null_df


# ===========================================================================
# Section 6: Visualization and output
# ===========================================================================

def plot_results(theta: np.ndarray, V: np.ndarray, entropy: np.ndarray,
                 adata_mc: ad.AnnData, stability_df: pd.DataFrame):
    """Generate IRL result plots."""
    import matplotlib.pyplot as plt

    print("[5/5] Generating plots ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reward weights bar plot
    ax = axes[0, 0]
    colors = ['#d73027' if t > 0 else '#4575b4' for t in theta]
    ax.barh(range(len(REWARD_FEATURES)), theta, color=colors)
    ax.set_yticks(range(len(REWARD_FEATURES)))
    ax.set_yticklabels(REWARD_FEATURES, fontsize=9)
    ax.set_xlabel("Learned reward weight θ")
    ax.set_title("IRL Reward Weights (θ*)")
    ax.axvline(0, color='k', linewidth=0.5)

    # Plot 2: Value function vs pseudotime
    ax = axes[0, 1]
    pt = adata_mc.obs["dpt_pseudotime"].values
    ax.scatter(pt, V, c=V, cmap="RdYlBu_r", s=8, alpha=0.7)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("V(s)")
    ax.set_title("Value Function along Pseudotime")

    # Plot 3: Policy entropy vs pseudotime
    ax = axes[1, 0]
    stage_order = {"NAG": 0, "CAG": 1, "IM": 2, "EGC": 3, "GC": 4}
    stage_colors = [stage_order.get(s, 2) for s in adata_mc.obs["stage"]]
    ax.scatter(pt, entropy, c=stage_colors,
               cmap="viridis", s=8, alpha=0.7)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Policy entropy H(s)")
    ax.set_title("Strategy Certainty (lower = more determined)")

    # Plot 4: Theta stability across runs
    ax = axes[1, 1]
    theta_cols = [c for c in stability_df.columns if c.startswith("theta_")]
    data = stability_df[theta_cols].values
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_xticklabels([c.replace("theta_", "") for c in theta_cols],
                       rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("θ value")
    ax.set_title("θ Stability (multi-seed + LODO)")

    plt.tight_layout()
    plt.savefig(FIGURES / "rl_fitness_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES / 'rl_fitness_landscape.png'}")


def save_outputs(theta, V, entropy, adata_mc, stability_df, null_df, irl_result):
    """Save all Layer 2 outputs."""
    print("\n[SAVE] Writing outputs ...")

    # Reward weights with feature names
    theta_df = pd.DataFrame({
        "feature": REWARD_FEATURES,
        "theta": theta,
        "abs_theta": np.abs(theta),
    }).sort_values("abs_theta", ascending=False)
    theta_df.to_csv(RESULTS / "rl_reward_weights.csv", index=False)

    # Value function per metacell
    val_df = adata_mc.obs[["sample_id", "dataset", "stage", "celltype",
                            "dpt_pseudotime"]].copy()
    val_df["V_value"] = V
    val_df["policy_entropy"] = entropy
    val_df.to_csv(RESULTS / "rl_value_function.csv", index=False)

    # Policy entropy summary per stage
    ent_df = val_df.groupby("stage").agg(
        mean_V=("V_value", "mean"),
        mean_entropy=("policy_entropy", "mean"),
        std_entropy=("policy_entropy", "std"),
        n_metacells=("V_value", "count"),
    ).reset_index()
    ent_df.to_csv(RESULTS / "rl_policy_entropy.csv", index=False)

    # Stability
    stability_df.to_csv(RESULTS / "rl_irl_stability.csv", index=False)

    # Negative control
    null_df.to_csv(RESULTS / "rl_irl_null_control.csv", index=False)

    print(f"  Saved: rl_reward_weights.csv, rl_value_function.csv, "
          f"rl_policy_entropy.csv, rl_irl_stability.csv, rl_irl_null_control.csv")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Phase 22: Fitness Landscape Recovery (MaxEnt State-Only IRL)")
    print("=" * 70)

    # Load metacells and transition matrix
    print("[0/5] Loading Phase 21 outputs ...")
    adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
    T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
    T = T_sparse.toarray()
    print(f"  Metacells: {adata_mc.n_obs}, Transition matrix: {T.shape}")

    # Step 1: Compute features
    phi = compute_reward_features(adata_mc)

    # Step 2: Build trajectories
    trajectories = build_trajectories(adata_mc, T_sparse)

    # Step 3: Run IRL (primary)
    print("[3/5] Running MaxEnt IRL (primary) ...")
    irl_result = maxent_irl(T, phi, trajectories, CONFIG, seed=CONFIG["random_seed"])
    theta = irl_result["theta"]
    V = irl_result["V"]
    entropy = irl_result["policy_entropy"]

    print("\n  Learned theta*:")
    for j, feat in enumerate(REWARD_FEATURES):
        print(f"    {feat:25s}: {theta[j]:+.4f}")

    # Step 4: Stability
    stability_df = stability_analysis(T, phi, trajectories, adata_mc, CONFIG)

    # Step 4b: Negative control
    null_df = negative_control(T, phi, adata_mc, CONFIG)

    # Step 5: Plots
    plot_results(theta, V, entropy, adata_mc, stability_df)

    # Save
    save_outputs(theta, V, entropy, adata_mc, stability_df, null_df, irl_result)

    # Final summary
    print("\n" + "=" * 70)
    print("Phase 22 COMPLETE")
    top3 = np.argsort(np.abs(theta))[::-1][:3]
    print(f"  Top reward features: "
          f"{REWARD_FEATURES[top3[0]]} ({theta[top3[0]]:+.3f}), "
          f"{REWARD_FEATURES[top3[1]]} ({theta[top3[1]]:+.3f}), "
          f"{REWARD_FEATURES[top3[2]]} ({theta[top3[2]]:+.3f})")
    print(f"  Mean entropy early (pt<0.3): {entropy[adata_mc.obs['dpt_pseudotime'] < 0.3].mean():.3f}")
    print(f"  Mean entropy late  (pt>0.7): {entropy[adata_mc.obs['dpt_pseudotime'] > 0.7].mean():.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
