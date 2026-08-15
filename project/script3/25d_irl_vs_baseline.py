"""
Phase 25d: IRL vs Baseline Model Comparison
Compare IRL value function against simpler alternatives:
  1. Ordinal logistic regression (stage -> progression score)
  2. Direct fate probability from T matrix (CellRank absorption)
  3. Simple pseudotime (already available)

Key question: Does IRL's V(s) provide information beyond what simpler models capture?
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25d: IRL vs Baseline Model Comparison")
print("=" * 70)

# Load data
print("\n[1/4] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
val_df = pd.read_csv(RESULTS / "rl_value_function.csv")

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values

X = adata_mc.X
if sparse.issparse(X):
    X = X.toarray()
gene_names = adata_mc.var_names
n_cells = len(adata_mc)

print(f"  {n_cells} metacells, V range=[{V.min():.2f}, {V.max():.2f}]")

# ===================================================================
# [2/4] Build baseline models
# ===================================================================
print("\n[2/4] Building baseline models ...")

# --- Baseline 1: Ordinal stage score ---
stage_order = {'NAG': 0, 'CAG': 1, 'IM': 2, 'EGC': 3,
               'EGC_multi_region': 3, 'GC': 4}
ordinal_score = np.array([stage_order.get(s, 2) for s in stages])
ordinal_score_norm = ordinal_score / ordinal_score.max()
print(f"  B1 (ordinal stage): range=[{ordinal_score.min()}, {ordinal_score.max()}]")

# --- Baseline 2: Logistic regression on gene features ---
# Predict "progression" (IM/EGC/GC vs NAG/CAG) using top variable genes
is_progressed = ((stages == 'IM') | (stages == 'EGC') |
                 (stages == 'EGC_multi_region') | (stages == 'GC')).astype(int)

# Use scVI latent space as features (already dimensionality-reduced)
if 'X_scVI' in adata_mc.obsm:
    features = adata_mc.obsm['X_scVI']
else:
    features = X[:, :50]  # fallback: top 50 genes

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(features_scaled, is_progressed)
logistic_prob = lr.predict_proba(features_scaled)[:, 1]
print(f"  B2 (logistic): AUC={roc_auc_score(is_progressed, logistic_prob):.3f}")

# --- Baseline 3: Pseudotime (already computed) ---
print(f"  B3 (pseudotime): range=[{pt.min():.3f}, {pt.max():.3f}]")

# --- Baseline 4: Absorption probability to EGC/GC ---
# Compute P(reach EGC|start=i) via iterating T many times
print("  B4 (absorption probability): computing ...")
egc_mask = (stages == 'EGC') | (stages == 'EGC_multi_region')
gc_mask = stages == 'GC'
absorb_mask = egc_mask | gc_mask

# Make absorbing states: zero out their transition rows, self-loop
T_abs = T_cr.copy()
for i in np.where(absorb_mask)[0]:
    T_abs[i, :] = 0
    T_abs[i, i] = 1.0

# Iterate to get absorption probabilities
state = np.eye(n_cells)
for _ in range(50):
    state = state @ T_abs
absorption_prob = state[:, absorb_mask].sum(axis=1)
print(f"  B4 (absorption): range=[{absorption_prob.min():.3f}, "
      f"{absorption_prob.max():.3f}]")

# ===================================================================
# [3/4] Compare all models
# ===================================================================
print("\n[3/4] Comparing models ...")

models = {
    'IRL_V(s)': V,
    'B1_ordinal_stage': ordinal_score_norm,
    'B2_logistic': logistic_prob,
    'B3_pseudotime': pt,
    'B4_absorption': absorption_prob,
}

# Pairwise Spearman correlations
print("\n  Spearman correlation matrix:")
model_names = list(models.keys())
corr_matrix = np.zeros((len(model_names), len(model_names)))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        corr_matrix[i, j], _ = spearmanr(models[m1], models[m2])

header = f"  {'':>16}" + "".join(f"{m:>16}" for m in model_names)
print(header)
for i, m in enumerate(model_names):
    row = "".join(f"{corr_matrix[i,j]:>16.3f}" for j in range(len(model_names)))
    print(f"  {m:>16}{row}")

# Key comparisons: IRL vs each baseline
print(f"\n  IRL V(s) vs baselines:")
for bname in ['B1_ordinal_stage', 'B2_logistic', 'B3_pseudotime', 'B4_absorption']:
    rho, p = spearmanr(V, models[bname])
    print(f"    vs {bname:>18}: rho={rho:+.3f} (p={p:.2e})")

# Prediction task: can each model predict fate cluster membership?
# Use IM metacells, predict which fate cluster they belong to
im_idx = np.where(stages == 'IM')[0]

def compute_fate(indices, T, all_stages, k=20):
    n = T.shape[0]
    prop = np.zeros((len(indices), n))
    for i, idx in enumerate(indices):
        prop[i, idx] = 1.0
    for _ in range(k):
        prop = prop @ T
    egc_m = (all_stages == 'EGC') | (all_stages == 'EGC_multi_region')
    return (prop * egc_m).sum(1)

p_egc_im = compute_fate(im_idx, T_cr, stages)
# Binary: high-fate (top 50%) vs low-fate
fate_binary = (p_egc_im >= np.median(p_egc_im)).astype(int)

print(f"\n  Predicting EGC fate (binary, IM cells only, n={len(im_idx)}):")
for mname, mvals in models.items():
    vals_im = mvals[im_idx]
    if vals_im.std() > 0:
        auc = roc_auc_score(fate_binary, vals_im)
        # Flip if anti-correlated
        if auc < 0.5:
            auc = 1 - auc
        rho, _ = spearmanr(vals_im, p_egc_im)
        print(f"    {mname:>18}: AUC={auc:.3f}, rho(P_EGC)={rho:+.3f}")

# Unique information in IRL: residual after regressing out baselines
from sklearn.linear_model import LinearRegression
print(f"\n  Residual analysis (IRL info beyond baselines):")
baseline_features = np.column_stack([
    models['B1_ordinal_stage'][im_idx],
    models['B3_pseudotime'][im_idx],
    models['B4_absorption'][im_idx],
])
lr_res = LinearRegression()
lr_res.fit(baseline_features, V[im_idx])
V_predicted = lr_res.predict(baseline_features)
V_residual = V[im_idx] - V_predicted
r2 = lr_res.score(baseline_features, V[im_idx])
print(f"    R² of baselines predicting V(s): {r2:.3f}")
print(f"    -> {(1-r2)*100:.1f}% of V(s) variance is UNIQUE to IRL")

# Does residual correlate with fate?
rho_res, p_res = spearmanr(V_residual, p_egc_im)
print(f"    V_residual vs P(EGC): rho={rho_res:+.3f} (p={p_res:.2e})")
if abs(rho_res) > 0.1 and p_res < 0.05:
    print("    -> IRL captures fate-relevant info BEYOND simple baselines")
else:
    print("    -> IRL residual does not add predictive power for fate")

# ===================================================================
# [4/4] Visualization & Save
# ===================================================================
print("\n[4/4] Visualization ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Scatter: V(s) vs each baseline
for i, (bname, bvals) in enumerate([
    ('Ordinal Stage', ordinal_score_norm),
    ('Logistic Prob', logistic_prob),
    ('Pseudotime', pt),
    ('Absorption P', absorption_prob),
]):
    ax = axes[i // 3, i % 3]
    rho, _ = spearmanr(V, bvals)
    ax.scatter(bvals, V, c=ordinal_score, cmap='RdYlBu_r', s=5, alpha=0.4)
    ax.set_xlabel(bname)
    ax.set_ylabel('IRL V(s)')
    ax.set_title(f'V(s) vs {bname}\n(rho={rho:.3f})')

# Panel 5: V(s) vs P(EGC) in IM
ax = axes[1, 1]
ax.scatter(V[im_idx], p_egc_im, c='steelblue', s=10, alpha=0.5)
rho_vf, _ = spearmanr(V[im_idx], p_egc_im)
ax.set_xlabel('V(s)')
ax.set_ylabel('P(EGC)')
ax.set_title(f'IM cells: V(s) vs Fate\n(rho={rho_vf:.3f})')

# Panel 6: Residual vs P(EGC)
ax = axes[1, 2]
ax.scatter(V_residual, p_egc_im, c='coral', s=10, alpha=0.5)
ax.set_xlabel('V(s) residual')
ax.set_ylabel('P(EGC)')
ax.set_title(f'Unique IRL info vs Fate\n(rho={rho_res:.3f}, p={p_res:.3f})')
ax.axvline(0, color='gray', ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "irl_vs_baseline.png", dpi=150, bbox_inches='tight')
plt.close()

# Save summary
summary = {
    'model': model_names,
    'corr_with_V': [corr_matrix[0, i] for i in range(len(model_names))],
}
pd.DataFrame(summary).to_csv(RESULTS / "irl_baseline_comparison.csv", index=False)

print(f"\n  Saved: irl_vs_baseline.png")
print(f"  Saved: irl_baseline_comparison.csv")

print("\n" + "=" * 70)
print("Phase 25d COMPLETE")
print("=" * 70)
print(f"  IRL V(s) correlation with baselines:")
print(f"    Ordinal stage: {corr_matrix[0,1]:.3f}")
print(f"    Logistic: {corr_matrix[0,2]:.3f}")
print(f"    Pseudotime: {corr_matrix[0,3]:.3f}")
print(f"    Absorption: {corr_matrix[0,4]:.3f}")
print(f"  Unique IRL variance: {(1-r2)*100:.1f}%")
print(f"  Residual-fate correlation: rho={rho_res:+.3f} (p={p_res:.3f})")
print("=" * 70)
