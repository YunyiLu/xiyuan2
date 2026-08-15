"""
Step 12D: GSE78523 Progressor队列 LOOCV诊断效能 - 核心模型对比
M1 = OLFM4 alone
M2 = REG4 alone
M3 = OLFM4 + REG4 (已有文献基线)
M4 = OLFM4 + REG4 + ITLN1
M5 = OLFM4 + REG4 + ITLN1 + PRAP1 (核心创新模型)
M6 = ITLN1 + PRAP1 (纯新候选)
M7 = Tier1 + selected Tier2 (扩展)
M_ext1 = M5 + FABP1
M_ext2 = M5 + ANPEP

Input: script3/data/gse78523_gene_expr.csv
Output: script3/results/circulating_panel_roc.csv
        script3/figures/circulating_panel_roc.png
        script3/figures/panel_size_vs_auc.png
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
os.makedirs(f"{BASE}/figures", exist_ok=True)

# ============================================================
# Model definitions (pre-specified, NOT data-driven)
# ============================================================
MODELS = {
    "M1_OLFM4": ["OLFM4"],
    "M2_REG4": ["REG4"],
    "M3_OLFM4_REG4": ["OLFM4", "REG4"],
    "M4_add_ITLN1": ["OLFM4", "REG4", "ITLN1"],
    "M5_add_PRAP1": ["OLFM4", "REG4", "ITLN1", "PRAP1"],
    "M6_new_only": ["ITLN1", "PRAP1"],
    "M7_extended": ["OLFM4", "REG4", "ITLN1", "PRAP1", "ANPEP", "PSCA"],
    "M_ext1_FABP1": ["OLFM4", "REG4", "ITLN1", "PRAP1", "FABP1"],
    "M_ext2_ANPEP": ["OLFM4", "REG4", "ITLN1", "PRAP1", "ANPEP"],
}


def load_data():
    """Load GSE78523 expression matrix with progressor/non-progressor labels."""
    df = pd.read_csv(f"{BASE}/data/gse78523_gene_expr.csv", index_col=0)

    # Binary label: progressor vs non-progressor (include healthy as non-progressor)
    y = (df["progression_status"] == "Progressor").astype(int).values
    meta = df[["group", "im_type", "progression_status"]].copy()

    # Drop metadata columns to get expression
    expr_cols = [c for c in df.columns if c not in ["group", "im_type", "progression_status"]]
    X = df[expr_cols]

    return X, y, meta


def loocv_evaluate(X, y, gene_list):
    """LOOCV with fold-internal z-score normalization. Returns predicted probabilities."""
    available = [g for g in gene_list if g in X.columns]
    if len(available) == 0:
        return None, available

    n = len(y)
    y_pred_prob = np.zeros(n)

    for i in range(n):
        # Leave one out
        train_idx = np.concatenate([np.arange(0, i), np.arange(i+1, n)])
        test_idx = np.array([i])

        X_train = X.iloc[train_idx][available].values
        X_test = X.iloc[test_idx][available].values
        y_train = y[train_idx]

        # Fold-internal standardization
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Logistic regression (small regularization to handle collinearity)
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        y_pred_prob[i] = clf.predict_proba(X_test_s)[0, 1]

    return y_pred_prob, available


def bootstrap_auc_ci(y_true, y_pred_prob, n_bootstrap=2000, ci=0.95):
    """Bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(42)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred_prob[idx]))

    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.percentile(aucs, alpha * 100), np.percentile(aucs, (1 - alpha) * 100)


def bootstrap_delta_auc(y_true, prob_new, prob_base, n_bootstrap=2000):
    """Bootstrap CI for ΔAUC (new - base)."""
    rng = np.random.default_rng(42)
    deltas = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_new = roc_auc_score(y_true[idx], prob_new[idx])
        auc_base = roc_auc_score(y_true[idx], prob_base[idx])
        deltas.append(auc_new - auc_base)

    deltas = np.array(deltas)
    return np.median(deltas), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def sensitivity_at_specificity(y_true, y_pred_prob, target_spec=0.90):
    """Find sensitivity at given specificity threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    spec = 1 - fpr
    # Find point closest to target specificity (from above)
    valid = spec >= target_spec
    if valid.any():
        idx = np.where(valid)[0][-1]  # highest TPR at spec >= target
        return tpr[idx]
    return 0.0


def specificity_at_sensitivity(y_true, y_pred_prob, target_sens=0.90):
    """Find specificity at given sensitivity threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    spec = 1 - fpr
    valid = tpr >= target_sens
    if valid.any():
        idx = np.where(valid)[0][0]  # highest specificity at sens >= target
        return spec[idx]
    return 0.0


def subsample_stability(X, y, gene_list, n_repeats=100, frac=0.8):
    """Assess coefficient direction stability across 80% subsamples."""
    available = [g for g in gene_list if g in X.columns]
    if len(available) == 0:
        return {}

    rng = np.random.default_rng(42)
    n = len(y)
    coef_signs = {g: [] for g in available}

    for _ in range(n_repeats):
        idx = rng.choice(n, int(n * frac), replace=False)
        X_sub = X.iloc[idx][available].values
        y_sub = y[idx]

        if len(np.unique(y_sub)) < 2:
            continue

        scaler = StandardScaler()
        X_sub_s = scaler.fit_transform(X_sub)

        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_sub_s, y_sub)

        for j, g in enumerate(available):
            coef_signs[g].append(np.sign(clf.coef_[0][j]))

    # Compute direction consistency (fraction positive)
    stability = {}
    for g in available:
        signs = np.array(coef_signs[g])
        if len(signs) > 0:
            stability[g] = max(np.mean(signs > 0), np.mean(signs < 0))
        else:
            stability[g] = 0.5
    return stability


def main():
    print("=" * 70)
    print("Step 12D: LOOCV Model Comparison - GSE78523 Progressor Cohort")
    print("=" * 70)

    X, y, meta = load_data()
    print(f"\nData: {len(y)} samples ({y.sum()} progressors, {(1-y).sum()} non-progressors)")
    print(f"Groups: {meta['group'].value_counts().to_dict()}")

    # Run LOOCV for each model
    results = []
    predictions = {}

    print(f"\n{'Model':<20} {'Genes':<6} {'AUC':>6} {'95% CI':>16} {'Sens@Sp90':>10} {'Sp@Sn90':>8}")
    print("-" * 70)

    for model_name, genes in MODELS.items():
        y_pred, available = loocv_evaluate(X, y, genes)
        if y_pred is None:
            print(f"{model_name:<20} — genes not available")
            continue

        predictions[model_name] = y_pred
        auc = roc_auc_score(y, y_pred)
        ci_low, ci_high = bootstrap_auc_ci(y, y_pred)
        sens90 = sensitivity_at_specificity(y, y_pred, 0.90)
        spec90 = specificity_at_sensitivity(y, y_pred, 0.90)

        results.append({
            "model": model_name,
            "genes": ";".join(available),
            "n_genes": len(available),
            "AUC": round(auc, 4),
            "AUC_CI_low": round(ci_low, 4),
            "AUC_CI_high": round(ci_high, 4),
            "sensitivity_at_spec90": round(sens90, 4),
            "specificity_at_sens90": round(spec90, 4),
        })

        print(f"{model_name:<20} {len(available):<6} {auc:>6.3f} [{ci_low:.3f}-{ci_high:.3f}] {sens90:>10.3f} {spec90:>8.3f}")

    # Delta AUC comparisons (vs M3 baseline)
    print("\n" + "=" * 70)
    print("Incremental Analysis (vs M3 = OLFM4+REG4 baseline)")
    print("-" * 70)

    if "M3_OLFM4_REG4" in predictions:
        base_pred = predictions["M3_OLFM4_REG4"]
        for model_name in ["M4_add_ITLN1", "M5_add_PRAP1", "M7_extended"]:
            if model_name in predictions:
                delta_med, delta_low, delta_high = bootstrap_delta_auc(
                    y, predictions[model_name], base_pred
                )
                print(f"  {model_name} vs M3: ΔAUC = {delta_med:+.4f} [{delta_low:+.4f}, {delta_high:+.4f}]")

                # Add to results
                for r in results:
                    if r["model"] == model_name:
                        r["delta_AUC_vs_M3"] = round(delta_med, 4)
                        r["delta_CI_low"] = round(delta_low, 4)
                        r["delta_CI_high"] = round(delta_high, 4)

    # Stability analysis for M5 (core innovation model)
    print("\n" + "=" * 70)
    print("Coefficient Direction Stability (M5, 100x 80% subsample)")
    print("-" * 70)

    m5_genes = [g for g in MODELS["M5_add_PRAP1"] if g in X.columns]
    stability = subsample_stability(X, y, m5_genes)
    for g, s in stability.items():
        status = "STABLE" if s >= 0.70 else "UNSTABLE"
        print(f"  {g:<10}: {s:.1%} direction consistency [{status}]")
        for r in results:
            if r["model"] == "M5_add_PRAP1":
                r[f"stability_{g}"] = round(s, 3)

    # Save results
    df_results = pd.DataFrame(results)
    out_path = f"{BASE}/results/circulating_panel_roc.csv"
    df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] {out_path}")

    # ============================================================
    # Plot 1: ROC curves comparison
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = {
        "M1_OLFM4": "#aaaaaa",
        "M2_REG4": "#cccccc",
        "M3_OLFM4_REG4": "#1f77b4",
        "M4_add_ITLN1": "#ff7f0e",
        "M5_add_PRAP1": "#d62728",
        "M6_new_only": "#9467bd",
        "M7_extended": "#2ca02c",
    }
    linewidths = {"M3_OLFM4_REG4": 2.0, "M5_add_PRAP1": 2.5}

    for model_name in ["M1_OLFM4", "M2_REG4", "M3_OLFM4_REG4",
                       "M4_add_ITLN1", "M5_add_PRAP1", "M6_new_only", "M7_extended"]:
        if model_name not in predictions:
            continue
        fpr, tpr, _ = roc_curve(y, predictions[model_name])
        auc = roc_auc_score(y, predictions[model_name])
        label = f"{model_name} (AUC={auc:.3f})"
        lw = linewidths.get(model_name, 1.2)
        ls = '--' if model_name in ["M1_OLFM4", "M2_REG4"] else '-'
        ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'), lw=lw, ls=ls, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel("1 - Specificity (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
    ax.set_title("LOOCV ROC: IM→EGC Progressor Prediction\n(GSE78523, n=45: 14 progressors vs 31 non-progressors)", fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/circulating_panel_roc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {BASE}/figures/circulating_panel_roc.png")

    # ============================================================
    # Plot 2: Panel size vs AUC
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for r in results:
        if r["model"].startswith("M_ext"):
            marker = 's'
            color = 'gray'
            alpha = 0.6
        else:
            marker = 'o'
            color = colors.get(r["model"], 'black')
            alpha = 0.9

        ax.errorbar(r["n_genes"], r["AUC"],
                    yerr=[[r["AUC"] - r["AUC_CI_low"]], [r["AUC_CI_high"] - r["AUC"]]],
                    fmt=marker, color=color, alpha=alpha, markersize=8, capsize=4)
        ax.annotate(r["model"].replace("_", "\n"), (r["n_genes"] + 0.1, r["AUC"]),
                    fontsize=7, alpha=0.7)

    ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel("Number of Genes in Model", fontsize=11)
    ax.set_ylabel("LOOCV AUC", fontsize=11)
    ax.set_title("Panel Size vs Diagnostic Performance", fontsize=11)
    ax.set_xlim(0.5, max(r["n_genes"] for r in results) + 1)
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/panel_size_vs_auc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {BASE}/figures/panel_size_vs_auc.png")

    # ============================================================
    # Summary interpretation
    # ============================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    m3_auc = next((r["AUC"] for r in results if r["model"] == "M3_OLFM4_REG4"), None)
    m5_auc = next((r["AUC"] for r in results if r["model"] == "M5_add_PRAP1"), None)
    if m3_auc and m5_auc:
        if m5_auc > m3_auc + 0.03:
            print("  M5 shows meaningful increment over M3 baseline.")
            print("  Innovation: ITLN1+PRAP1 provide additional discriminative information.")
        elif m5_auc > m3_auc:
            print("  M5 shows marginal increment over M3.")
            print("  Innovation primarily in mechanism-driven discovery, not raw performance.")
        else:
            print("  M5 does not improve over M3. Innovation lies in mechanistic understanding,")
            print("  not in panel performance.")

    print("\n  NOTE: These are tissue RNA-level LOOCV results.")
    print("  They estimate the discriminative capacity of these secreted protein")
    print("  candidates at the transcriptional level in IM tissue.")
    print("  Blood protein concentrations may differ due to:")
    print("    - Post-translational regulation")
    print("    - Dilution in systemic circulation")
    print("    - Clearance kinetics")
    print("    - Non-gastric sources (especially FABP1/liver, ITLN1/adipose)")


if __name__ == "__main__":
    main()
