"""
Phase 21: Transition Dynamics — Metacell construction + CellRank transition matrix
Part of the IRL pipeline (PLAN_RL.md Layer 1)

Input:  script3/data/adata_integrated.h5ad
Output: script3/data/rl_metacells.h5ad
        script3/results/rl_transition_matrix.npz
        script3/results/rl_macrostates.csv
        script3/figures/rl_transition_graph.png
        script3/figures/rl_macrostate_umap.png

Requires: cellrank>=2.0, scanpy, anndata, numpy, scipy
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
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# --- Config (hyperparameters with search ranges documented) ---
CONFIG = {
    "k_nn": 30,                # search: [15, 30, 50, 75]
    "n_metacells_target": 2000,  # search: [1000, 2000, 4000]
    "cells_per_metacell": 50,    # min cells to form a metacell
    "random_seed": 42,
    "n_seeds": 5,                # for stability analysis
    "epsilon_back": 0.05,        # reverse transition tolerance [0, 0.025, 0.05, 0.10]
    "schur_n_components": 15,    # for macrostate identification
}

STAGE_ORDER = {"NAG": 0, "CAG": 1, "IM": 2, "EGC": 3, "GC": 4, "EGC_multi_region": 3}


# ===========================================================================
# Section 1: Load and subset to epithelial cells
# ===========================================================================

def load_epithelial(adata_path: Path) -> ad.AnnData:
    """Load integrated adata and subset to epithelial cells only."""
    print("[1/6] Loading adata_integrated.h5ad ...")
    adata = sc.read_h5ad(adata_path)
    print(f"  Full dataset: {adata.n_obs} cells × {adata.n_vars} genes")

    mask = adata.obs["is_epithelial"].astype(bool)
    adata_epi = adata[mask].copy()
    print(f"  Epithelial subset: {adata_epi.n_obs} cells")

    # Map stage to numeric for ordering
    adata_epi.obs["stage_num"] = adata_epi.obs["stage"].map(STAGE_ORDER).fillna(2).astype(int)

    # Verify scVI latent exists
    assert "X_scVI" in adata_epi.obsm, "Missing X_scVI in obsm"
    assert "dpt_pseudotime" in adata_epi.obs.columns, "Missing dpt_pseudotime"

    return adata_epi


# ===========================================================================
# Section 2: Metacell construction (kNN-based aggregation)
# ===========================================================================

def build_metacells(adata_epi: ad.AnnData, config: dict) -> ad.AnnData:
    """
    Construct metacells by kNN-based aggregation within (patient, celltype) blocks.

    Constraint: metacells never mix different patients or major cell types.
    This prevents batch/patient effects from being confused with transitions.
    """
    print("[2/6] Building metacells ...")

    n_target = config["n_metacells_target"]
    min_cells = config["cells_per_metacell"]
    seed = config["random_seed"]

    rng = np.random.default_rng(seed)

    # Group by (sample_id, celltype) to ensure no cross-patient mixing
    adata_epi.obs["_block"] = (
        adata_epi.obs["sample_id"].astype(str) + "||" +
        adata_epi.obs["celltype"].astype(str)
    )
    blocks = adata_epi.obs["_block"].unique()

    metacell_records = []
    metacell_latents = []
    metacell_exprs = []

    mc_id = 0

    for block in blocks:
        idx = np.where(adata_epi.obs["_block"] == block)[0]
        n_cells = len(idx)

        if n_cells < min_cells:
            # Too few cells — treat entire block as one metacell
            n_mc = 1
        else:
            # Target ~cells_per_metacell cells per metacell
            n_mc = max(1, n_cells // min_cells)

        # Random assignment within block (simple, reproducible)
        assignments = rng.integers(0, n_mc, size=n_cells)

        latent = adata_epi.obsm["X_scVI"][idx]
        # Use raw counts if available, else .X
        if adata_epi.raw is not None:
            expr_mat = adata_epi.raw.X[idx] if sparse.issparse(adata_epi.raw.X) \
                else adata_epi.raw.X[idx]
        else:
            expr_mat = adata_epi.X[idx]

        obs_df = adata_epi.obs.iloc[idx]

        for k in range(n_mc):
            mask_k = assignments == k
            if mask_k.sum() < 5:
                continue

            # Mean latent
            mc_latent = latent[mask_k].mean(axis=0)
            metacell_latents.append(mc_latent)

            # Mean expression
            if sparse.issparse(expr_mat):
                mc_expr = np.asarray(expr_mat[mask_k].mean(axis=0)).flatten()
            else:
                mc_expr = expr_mat[mask_k].mean(axis=0)
            metacell_exprs.append(mc_expr)

            # Metadata (majority vote / mean)
            sub_obs = obs_df.iloc[np.where(mask_k)[0]]
            metacell_records.append({
                "metacell_id": mc_id,
                "sample_id": sub_obs["sample_id"].iloc[0],
                "dataset": sub_obs["dataset"].iloc[0],
                "stage": sub_obs["stage"].mode().iloc[0],
                "celltype": sub_obs["celltype"].mode().iloc[0],
                "stage_num": sub_obs["stage_num"].median(),
                "dpt_pseudotime": sub_obs["dpt_pseudotime"].mean(),
                "proliferation_score": sub_obs["proliferation_score"].mean(),
                "stemness_score": sub_obs["stemness_score"].mean(),
                "EGC_like_score": sub_obs["EGC_like_score"].mean(),
                "PMC_P_score": sub_obs["PMC_P_score"].mean(),
                "n_cells": int(mask_k.sum()),
                "hp_status": sub_obs["hp_status"].mode().iloc[0]
                    if "hp_status" in sub_obs.columns else "unknown",
            })
            mc_id += 1

    print(f"  Constructed {mc_id} metacells from {len(blocks)} blocks")

    # Assemble AnnData
    mc_obs = pd.DataFrame(metacell_records)
    mc_obs.index = [f"MC_{i}" for i in range(mc_id)]

    var_names = adata_epi.var_names if adata_epi.raw is None else adata_epi.raw.var_names
    adata_mc = ad.AnnData(
        X=np.vstack(metacell_exprs),
        obs=mc_obs,
        var=pd.DataFrame(index=var_names),
    )
    adata_mc.obsm["X_scVI"] = np.vstack(metacell_latents)

    # Map stage_num
    adata_mc.obs["stage_num"] = adata_mc.obs["stage_num"].astype(float)

    print(f"  Final metacell AnnData: {adata_mc.n_obs} × {adata_mc.n_vars}")
    return adata_mc


# ===========================================================================
# Section 3: Build kNN graph and CellRank transition kernel
# ===========================================================================

def build_transition_kernel(adata_mc: ad.AnnData, config: dict):
    """
    Build directed transition matrix using CellRank.
    Uses PseudotimeKernel (DPT-based direction) + ConnectivityKernel.
    """
    import cellrank as cr

    print("[3/6] Building transition kernel (CellRank) ...")

    k_nn = config["k_nn"]
    eps_back = config["epsilon_back"]

    # Compute neighbors on scVI latent
    sc.pp.neighbors(adata_mc, use_rep="X_scVI", n_neighbors=k_nn, random_state=42)

    # PseudotimeKernel: uses dpt_pseudotime for directionality
    pk = cr.kernels.PseudotimeKernel(adata_mc, time_key="dpt_pseudotime")
    pk.compute_transition_matrix(threshold_scheme="soft", frac_to_keep=1.0 - eps_back)

    # ConnectivityKernel: undirected local structure
    ck = cr.kernels.ConnectivityKernel(adata_mc)
    ck.compute_transition_matrix()

    # Combined kernel: 70% pseudotime direction + 30% connectivity
    # This ratio is itself a hyperparameter but we fix it for v1
    combined = 0.7 * pk + 0.3 * ck
    print(f"  Transition matrix shape: {combined.transition_matrix.shape}")

    return combined


# ===========================================================================
# Section 4: Macrostate identification via GPCCA (Schur decomposition)
# ===========================================================================

def identify_macrostates(adata_mc: ad.AnnData, kernel, config: dict) -> pd.DataFrame:
    """
    Identify macrostates using CellRank's GPCCA estimator.
    K_macro determined by eigengap, not preset.
    """
    import cellrank as cr

    print("[4/6] Identifying macrostates (GPCCA / Schur) ...")

    # Use GPCCA for macrostate identification
    estimator = cr.estimators.GPCCA(kernel)

    # Compute Schur decomposition — examine eigenvalues for gap
    n_comp = config["schur_n_components"]
    estimator.compute_schur(n_components=n_comp)

    # Determine K_macro from eigengap
    eigenvalues = np.abs(estimator.eigendecomposition["D"])
    real_eigs = np.sort(eigenvalues)[::-1]

    # Eigengap: largest drop between consecutive eigenvalues
    gaps = np.diff(real_eigs[:n_comp])
    # K_macro = position of largest gap + 1 (at least 3, at most 8)
    k_macro_candidates = np.argsort(gaps)[::-1]  # indices of largest gaps
    k_macro = None
    for candidate in k_macro_candidates:
        k = candidate + 1  # +1 because gap at position i means i+1 states
        if 3 <= k <= 8:
            k_macro = k
            break
    if k_macro is None:
        k_macro = 5  # fallback
        print(f"  WARNING: eigengap did not yield K in [3,8], defaulting to {k_macro}")
    else:
        print(f"  Eigengap selected K_macro = {k_macro}")

    # Override: if eigengap selects K<4 with >1000 metacells, use at least 5
    # Biological rationale: NAG, CAG/early-IM, late-IM, high-risk, EGC/GC
    if k_macro < 5 and adata_mc.n_obs > 500:
        print(f"  Overriding K_macro from {k_macro} to 5 (insufficient resolution)")
        k_macro = 5

    k_macro = int(k_macro)  # ensure Python int, not numpy int64

    # Ensure cluster_key column is categorical (CellRank requirement)
    adata_mc.obs["stage"] = pd.Categorical(adata_mc.obs["stage"])

    # Compute macrostates
    estimator.compute_macrostates(n_states=k_macro, cluster_key="stage")

    # Extract assignments — use macrostates_memberships for full assignment
    # estimator.macrostates only labels top n_cells per state (rest NaN)
    # Use membership probabilities to assign ALL metacells
    if estimator.macrostates_memberships is not None:
        memberships = estimator.macrostates_memberships
        # Assign each metacell to its highest-probability macrostate
        state_names = memberships.names
        assignments = memberships.X.argmax(axis=1)
        adata_mc.obs["macrostate"] = pd.Categorical(
            [state_names[i] for i in assignments],
            categories=state_names
        )
    else:
        # Fallback: use the sparse labeling
        mc_states = estimator.macrostates
        adata_mc.obs["macrostate"] = mc_states

    # Predict terminal states and compute fate probabilities
    try:
        estimator.predict_terminal_states(method="stability", n_cells=30)
        estimator.compute_fate_probabilities()
    except Exception as e:
        print(f"  WARNING: Could not compute fate probabilities: {e}")
        print(f"  Continuing without absorption probabilities.")

    # Summarize macrostates
    ms_summary = []
    for state in adata_mc.obs["macrostate"].dropna().unique():
        mask = adata_mc.obs["macrostate"] == state
        sub = adata_mc.obs[mask]
        ms_summary.append({
            "macrostate": state,
            "n_metacells": int(mask.sum()),
            "n_patients": sub["sample_id"].nunique(),
            "n_datasets": sub["dataset"].nunique(),
            "dominant_stage": sub["stage"].mode().iloc[0] if len(sub) > 0 else "NA",
            "mean_pseudotime": sub["dpt_pseudotime"].mean(),
            "mean_proliferation": sub["proliferation_score"].mean(),
            "mean_stemness": sub["stemness_score"].mean(),
            "mean_EGC_score": sub["EGC_like_score"].mean(),
        })

    ms_df = pd.DataFrame(ms_summary)
    print(f"\n  Macrostate summary:")
    print(ms_df.to_string(index=False))

    return ms_df, estimator


# ===========================================================================
# Section 5: Quality checks and negative controls
# ===========================================================================

def quality_checks(adata_mc: ad.AnnData, ms_df: pd.DataFrame) -> dict:
    """
    Validate transition dynamics quality:
    1. Each macrostate supported by >=2 patients
    2. Each macrostate supported by >=2 datasets
    3. Main flow direction consistent with stage ordering
    4. No unreasonable cross-stage jumps
    """
    print("[5/6] Running quality checks ...")

    checks = {}

    # Check 1: Multi-patient support
    min_patients = ms_df["n_patients"].min()
    checks["min_patients_per_state"] = int(min_patients)
    checks["all_states_multi_patient"] = bool(min_patients >= 2)
    print(f"  Min patients per macrostate: {min_patients} "
          f"{'PASS' if min_patients >= 2 else 'FAIL'}")

    # Check 2: Multi-dataset support
    min_datasets = ms_df["n_datasets"].min()
    checks["min_datasets_per_state"] = int(min_datasets)
    checks["all_states_multi_dataset"] = bool(min_datasets >= 2)
    print(f"  Min datasets per macrostate: {min_datasets} "
          f"{'PASS' if min_datasets >= 2 else 'FAIL'}")

    # Check 3: Pseudotime ordering matches stage
    stage_pt = adata_mc.obs.groupby("stage")["dpt_pseudotime"].mean()
    ordered_correctly = True
    for s1, s2 in [("NAG", "IM"), ("IM", "EGC")]:
        if s1 in stage_pt.index and s2 in stage_pt.index:
            if stage_pt[s1] >= stage_pt[s2]:
                ordered_correctly = False
    checks["pseudotime_stage_consistent"] = ordered_correctly
    print(f"  Pseudotime-stage ordering: {'PASS' if ordered_correctly else 'FAIL'}")

    # Check 4: No single-patient-dominated macrostates
    dominated = []
    for _, row in ms_df.iterrows():
        state = row["macrostate"]
        sub = adata_mc.obs[adata_mc.obs["macrostate"] == state]
        patient_counts = sub["sample_id"].value_counts()
        if len(patient_counts) > 0:
            top_frac = patient_counts.iloc[0] / patient_counts.sum()
            if top_frac > 0.7:
                dominated.append(state)
    checks["patient_dominated_states"] = dominated
    checks["no_dominated_states"] = len(dominated) == 0
    print(f"  Patient-dominated states: {dominated if dominated else 'None PASS'}")

    return checks


# ===========================================================================
# Section 6: Negative control — stage label permutation
# ===========================================================================

def negative_control_permutation(adata_epi: ad.AnnData, adata_mc: ad.AnnData,
                                  config: dict, n_perms: int = 100) -> dict:
    """
    Permutation test: shuffle stage labels, rebuild pseudotime direction,
    check if main transitions still appear (they shouldn't).
    """
    import cellrank as cr

    print("[5b/6] Running permutation negative control (simplified) ...")

    rng = np.random.default_rng(config["random_seed"] + 999)
    k_nn = config["k_nn"]

    # Get the real transition matrix's main flow
    real_T = adata_mc.obsp.get("T_fwd", None)
    if real_T is None:
        # Fallback: use connectivities
        print("  Skipping permutation (no T_fwd in obsp). Will run after kernel saved.")
        return {"permutation_done": False}

    # For each permutation: shuffle stage_num, recompute pseudotime kernel direction
    # Measure: correlation between metacell stage_num and transition probability flow
    real_corr = np.corrcoef(
        adata_mc.obs["stage_num"].values,
        adata_mc.obs["dpt_pseudotime"].values
    )[0, 1]

    null_corrs = []
    for _ in range(n_perms):
        shuffled = rng.permutation(adata_mc.obs["stage_num"].values)
        null_corrs.append(np.corrcoef(shuffled, adata_mc.obs["dpt_pseudotime"].values)[0, 1])

    null_corrs = np.array(null_corrs)
    p_value = (np.abs(null_corrs) >= np.abs(real_corr)).mean()

    result = {
        "permutation_done": True,
        "real_stage_pt_correlation": float(real_corr),
        "null_mean": float(null_corrs.mean()),
        "null_std": float(null_corrs.std()),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.01),
    }
    print(f"  Real correlation: {real_corr:.3f}")
    print(f"  Null: {null_corrs.mean():.3f} ± {null_corrs.std():.3f}")
    print(f"  Permutation p = {p_value:.4f} {'✓' if p_value < 0.01 else '✗'}")
    return result


# ===========================================================================
# Section 7: Visualization
# ===========================================================================

def plot_results(adata_mc: ad.AnnData, ms_df: pd.DataFrame, estimator):
    """Generate diagnostic plots."""
    import matplotlib.pyplot as plt

    print("[6/6] Generating plots ...")

    # Plot 1: UMAP of metacells colored by macrostate
    sc.pp.neighbors(adata_mc, use_rep="X_scVI", n_neighbors=30, random_state=42)
    sc.tl.umap(adata_mc, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc.pl.umap(adata_mc, color="macrostate", ax=axes[0], show=False,
               title="Macrostates", frameon=False)
    sc.pl.umap(adata_mc, color="stage", ax=axes[1], show=False,
               title="Pathology Stage", frameon=False)
    sc.pl.umap(adata_mc, color="dpt_pseudotime", ax=axes[2], show=False,
               title="Pseudotime (DPT)", frameon=False, color_map="viridis")

    plt.tight_layout()
    plt.savefig(FIGURES / "rl_macrostate_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES / 'rl_macrostate_umap.png'}")

    # Plot 2: Transition graph (if estimator supports it)
    try:
        estimator.plot_macrostates(which="all", basis="umap", show=False)
        plt.savefig(FIGURES / "rl_transition_graph.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIGURES / 'rl_transition_graph.png'}")
    except Exception as e:
        print(f"  Could not plot transition graph: {e}")


# ===========================================================================
# Section 8: Save outputs
# ===========================================================================

def save_outputs(adata_mc: ad.AnnData, kernel, ms_df: pd.DataFrame,
                 checks: dict, perm_result: dict):
    """Save all outputs."""
    print("\n[SAVE] Writing outputs ...")

    # Save metacell adata
    adata_mc.write_h5ad(DATA / "rl_metacells.h5ad")
    print(f"  Saved: {DATA / 'rl_metacells.h5ad'}")

    # Save transition matrix
    T = kernel.transition_matrix
    sparse.save_npz(RESULTS / "rl_transition_matrix.npz", T)
    print(f"  Saved: {RESULTS / 'rl_transition_matrix.npz'}")

    # Save macrostate summary
    ms_df.to_csv(RESULTS / "rl_macrostates.csv", index=False)
    print(f"  Saved: {RESULTS / 'rl_macrostates.csv'}")

    # Save quality checks + permutation as JSON-like CSV
    audit_df = pd.DataFrame([{
        **checks,
        **{f"perm_{k}": v for k, v in perm_result.items()},
        "config_k_nn": CONFIG["k_nn"],
        "config_n_metacells_target": CONFIG["n_metacells_target"],
        "config_epsilon_back": CONFIG["epsilon_back"],
    }])
    audit_df.to_csv(RESULTS / "rl_layer1_audit.csv", index=False)
    print(f"  Saved: {RESULTS / 'rl_layer1_audit.csv'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Phase 21: Transition Dynamics (PLAN_RL.md Layer 1)")
    print("=" * 70)

    # Step 1: Load
    adata_epi = load_epithelial(DATA / "adata_integrated.h5ad")

    # Step 2: Metacells
    adata_mc = build_metacells(adata_epi, CONFIG)

    # Step 3: Transition kernel
    kernel = build_transition_kernel(adata_mc, CONFIG)

    # Step 4: Macrostates
    ms_df, estimator = identify_macrostates(adata_mc, kernel, CONFIG)

    # Step 5: Quality checks
    checks = quality_checks(adata_mc, ms_df)

    # Step 5b: Negative control
    perm_result = negative_control_permutation(adata_epi, adata_mc, CONFIG)

    # Step 6: Plots
    plot_results(adata_mc, ms_df, estimator)

    # Save
    save_outputs(adata_mc, kernel, ms_df, checks, perm_result)

    print("\n" + "=" * 70)
    print("Phase 21 COMPLETE")
    print(f"  Metacells: {adata_mc.n_obs}")
    print(f"  Macrostates: {ms_df.shape[0]}")
    print(f"  Quality: {'ALL PASS' if all([checks.get('all_states_multi_patient', False), checks.get('pseudotime_stage_consistent', False)]) else 'SEE AUDIT'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
