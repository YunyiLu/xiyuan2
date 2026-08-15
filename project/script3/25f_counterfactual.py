"""
Phase 25f: IRL Counterfactual Simulation & Drug Target Prioritization
Make IRL indispensable by demonstrating two unique capabilities:

1. Counterfactual: "What if we block OXPHOS-loss / restore T-cell pressure?"
   -> Re-solve V(s) with modified theta -> predict fate shift
2. Drug targets: rank features by theta* × druggability × effect size

These are things ONLY an IRL framework can do (not absorption prob, not pseudotime).
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25f: IRL Counterfactual & Drug Target Prioritization")
print("=" * 70)

# Load data
print("\n[1/5] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
val_df = pd.read_csv(RESULTS / "rl_value_function.csv")
theta_df = pd.read_csv(RESULTS / "rl_reward_weights.csv")

stages = adata_mc.obs['stage'].values
V_original = val_df['V_value'].values
n_cells = len(adata_mc)
im_idx = np.where(stages == 'IM')[0]

# Get theta and feature names
theta = theta_df.set_index('feature')['theta'].to_dict()
feature_names = list(theta.keys())
theta_vec = np.array([theta[f] for f in feature_names])
print(f"  {n_cells} metacells, {len(feature_names)} features")
print(f"  Theta: {dict(zip(feature_names, np.round(theta_vec, 4)))}")

# ===================================================================
# [2/5] Reconstruct feature matrix and V(s) solver
# ===================================================================
print("\n[2/5] Reconstructing feature matrix ...")

X = adata_mc.X
if sparse.issparse(X):
    X = X.toarray()
gene_names = list(adata_mc.var_names)

def get_gene_mean(genes):
    """Mean expression of gene set."""
    valid = [g for g in genes if g in gene_names]
    if not valid:
        return np.zeros(n_cells)
    idx = [gene_names.index(g) for g in valid]
    return X[:, idx].mean(axis=1)

# Reconstruct 10 features (matching 22_fitness_landscape_irl.py)
phi = np.zeros((n_cells, 10))
phi[:, 0] = get_gene_mean(['MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1'])
phi[:, 1] = get_gene_mean(['LGR5', 'OLFM4', 'SOX9', 'ASCL2', 'CD44'])
phi[:, 2] = (get_gene_mean(['BCL2', 'MCL1', 'BIRC5']) -
             get_gene_mean(['BAX', 'CASP3', 'CASP9', 'BID']))
phi[:, 3] = -get_gene_mean(['GKN1', 'PGC', 'TFF1', 'MUC5AC', 'GIF'])
# NFkB: try obs column first
if 'RELA_activity' in adata_mc.obs:
    phi[:, 4] = adata_mc.obs['RELA_activity'].values
else:
    phi[:, 4] = get_gene_mean(['CCL3', 'CXCL8', 'IL6', 'TNFAIP3', 'NFKBIA'])
phi[:, 5] = (get_gene_mean(['HK2', 'PKM', 'LDHA', 'ENO1', 'GAPDH', 'PGK1']) -
             get_gene_mean(['MT-CO1', 'MT-CO2', 'MT-ND1', 'COX5A', 'NDUFA1',
                           'ATP5F1A']))
# Niche fractions
niche_path = DATA / "niche_fractions.csv"
if niche_path.exists():
    niche = pd.read_csv(niche_path, index_col=0)
    sample_ids = adata_mc.obs['sample_id'].values
    for i, sid in enumerate(sample_ids):
        if sid in niche.index:
            phi[i, 6] = niche.loc[sid].get('myeloid_fraction', 0)
            phi[i, 7] = niche.loc[sid].get('fibroblast_fraction', 0)
            phi[i, 8] = niche.loc[sid].get('T_cell_fraction', 0)
phi[:, 9] = 0  # spatial_border (only for Visium samples)

# Z-score normalize (matching original: set to 0 if std < 1e-10)
phi_z = np.zeros_like(phi)
for j in range(10):
    col = phi[:, j]
    std = col.std()
    if std > 1e-10:
        phi_z[:, j] = (col - col.mean()) / std
print(f"  Feature matrix: {phi_z.shape}")

# Soft value iteration
def soft_value_iteration(theta_vec, phi_z, T, gamma=0.95, tau=1.0,
                         max_iter=50, tol=0.01):
    """Solve V(s) = r(s) + gamma * tau * logsumexp(log T + V/tau)."""
    r = phi_z @ theta_vec
    log_T = np.log(T + 1e-300)
    V = r.copy()
    for _ in range(max_iter):
        Q = log_T + V[np.newaxis, :] / tau
        Q_max = Q.max(axis=1, keepdims=True)
        log_sum_exp = Q_max.squeeze() + np.log(
            np.exp(Q - Q_max).sum(axis=1) + 1e-300)
        V_new = r + gamma * tau * log_sum_exp
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V

def compute_soft_policy(V, T, tau=1.0):
    """Derive Boltzmann policy pi(s'|s) from V and base T."""
    log_T = np.log(T + 1e-300)
    Q = log_T + V[np.newaxis, :] / tau  # shape: (n, n)
    # Boltzmann: pi(s'|s) proportional to T(s'|s) * exp(V(s')/tau)
    # = exp(log T(s'|s) + V(s')/tau)
    Q_max = Q.max(axis=1, keepdims=True)
    pi = np.exp(Q - Q_max)
    # Normalize rows
    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pi = pi / row_sums
    return pi

# Verify reconstruction matches original V(s)
V_recon = soft_value_iteration(theta_vec, phi_z, T_cr)
rho_check, _ = spearmanr(V_recon, V_original)
print(f"  V(s) reconstruction check: rho={rho_check:.4f} vs original")
if abs(rho_check) < 0.5:
    print("  NOTE: Using original V(s) as reference for counterfactual")
    V_recon = V_original

# ===================================================================
# [3/5] Counterfactual simulations
# ===================================================================
print("\n[3/5] Counterfactual simulations ...")

def compute_fate_from_V(V, im_indices, T, stages, tau=1.0, k=20):
    """Compute P(EGC) using soft-policy-induced transitions from V."""
    # Derive policy from V
    T_policy = compute_soft_policy(V, T, tau)
    n = T_policy.shape[0]
    prop = np.zeros((len(im_indices), n))
    for i, idx in enumerate(im_indices):
        prop[i, idx] = 1.0
    for _ in range(k):
        prop = prop @ T_policy
    egc_m = (stages == 'EGC') | (stages == 'EGC_multi_region')
    return (prop * egc_m).sum(1)

# Baseline fate
p_egc_baseline = compute_fate_from_V(V_recon, im_idx, T_cr, stages)

# Define counterfactual interventions
interventions = {
    'Block metabolic shift (set theta_metabolic=0)': {
        'feature_idx': 5, 'new_weight': 0.0,
        'rationale': 'What if OXPHOS-loss confers no fitness advantage?'
    },
    'Restore T-cell pressure (flip sign)': {
        'feature_idx': 8, 'new_weight': +0.084,
        'rationale': 'What if T-cell presence helps rather than hinders?'
    },
    'Block inflammatory NFkB (set to 0)': {
        'feature_idx': 4, 'new_weight': 0.0,
        'rationale': 'Anti-inflammatory intervention'
    },
    'Double metabolic penalty': {
        'feature_idx': 5, 'new_weight': theta_vec[5] * 2,
        'rationale': 'What if metabolic shift is even more rewarded?'
    },
    'Restore differentiation (flip sign)': {
        'feature_idx': 3, 'new_weight': +0.024,
        'rationale': 'What if differentiation is rewarded instead of penalized?'
    },
}

cf_results = []
print(f"\n  {'Intervention':>45} {'dP(EGC)':>9} {'% change':>9}")
print(f"  {'-'*68}")

for name, params in interventions.items():
    theta_cf = theta_vec.copy()
    theta_cf[params['feature_idx']] = params['new_weight']
    V_cf = soft_value_iteration(theta_cf, phi_z, T_cr)
    p_egc_cf = compute_fate_from_V(V_cf, im_idx, T_cr, stages)

    delta_p = p_egc_cf.mean() - p_egc_baseline.mean()
    pct_change = 100 * delta_p / (p_egc_baseline.mean() + 1e-10)

    cf_results.append({
        'intervention': name,
        'feature': feature_names[params['feature_idx']],
        'original_theta': theta_vec[params['feature_idx']],
        'new_theta': params['new_weight'],
        'mean_P_EGC_baseline': p_egc_baseline.mean(),
        'mean_P_EGC_counterfactual': p_egc_cf.mean(),
        'delta_P_EGC': delta_p,
        'pct_change': pct_change,
        'rationale': params['rationale'],
    })
    print(f"  {name:>45} {delta_p:>+9.4f} {pct_change:>+8.1f}%")

# ===================================================================
# [4/5] Drug target prioritization
# ===================================================================
print("\n[4/5] Drug target prioritization ...")

# Score = |theta| * druggability * direction_actionability
drug_targets = {
    'T_cell_pressure': {
        'druggable': True, 'drugs': 'anti-PD1, anti-CTLA4, IL-2',
        'direction': 'increase (checkpoint inhibitors)',
        'score_mult': 1.5,  # well-established drug class
    },
    'inflammatory_NFkB': {
        'druggable': True, 'drugs': 'Bortezomib, Aspirin, BAY11-7082',
        'direction': 'decrease (NFkB inhibitors)',
        'score_mult': 1.2,
    },
    'proliferation': {
        'druggable': True, 'drugs': 'CDK4/6i (Palbociclib), mTOR inhibitors',
        'direction': 'decrease (anti-proliferative)',
        'score_mult': 1.0,
    },
    'metabolic_shift': {
        'druggable': True, 'drugs': 'Metformin, 2-DG, OXPHOS activators',
        'direction': 'decrease (restore OXPHOS / block glycolysis)',
        'score_mult': 1.3,
    },
    'apoptosis_resistance': {
        'druggable': True, 'drugs': 'Venetoclax (BCL2i), SMAC mimetics',
        'direction': 'increase apoptosis (BCL2 inhibitors)',
        'score_mult': 1.4,
    },
    'differentiation_loss': {
        'druggable': False, 'drugs': 'Vitamin A/retinoids (indirect)',
        'direction': 'restore differentiation',
        'score_mult': 0.6,
    },
    'fibroblast_niche': {
        'druggable': True, 'drugs': 'anti-TGFb, Nintedanib',
        'direction': 'decrease (block CAF support)',
        'score_mult': 0.8,
    },
    'stemness': {
        'druggable': False, 'drugs': 'Wnt inhibitors (experimental)',
        'direction': 'decrease',
        'score_mult': 0.5,
    },
    'myeloid_niche': {
        'druggable': True, 'drugs': 'CSF1R inhibitors, anti-CCL2',
        'direction': 'reprogram (M2->M1)',
        'score_mult': 0.9,
    },
    'spatial_border': {
        'druggable': False, 'drugs': 'N/A (structural)',
        'direction': 'N/A',
        'score_mult': 0.0,
    },
}

# Compute priority score
target_scores = []
for i, feat in enumerate(feature_names):
    info = drug_targets.get(feat, {})
    priority = abs(theta_vec[i]) * info.get('score_mult', 0.5) * 100
    target_scores.append({
        'feature': feat, 'theta': theta_vec[i],
        'abs_theta': abs(theta_vec[i]),
        'druggable': info.get('druggable', False),
        'candidate_drugs': info.get('drugs', 'N/A'),
        'intervention_direction': info.get('direction', 'N/A'),
        'priority_score': priority,
    })

target_df = pd.DataFrame(target_scores).sort_values('priority_score',
                                                     ascending=False)
print(f"\n  Drug Target Priority Ranking:")
print(f"  {'Rank':>4} {'Feature':>20} {'theta':>7} {'Priority':>8} "
      f"{'Drugs':>35}")
print(f"  {'-'*80}")
for rank, (_, row) in enumerate(target_df.iterrows(), 1):
    if row['priority_score'] > 0:
        print(f"  {rank:>4} {row['feature']:>20} {row['theta']:>+7.4f} "
              f"{row['priority_score']:>8.2f} {row['candidate_drugs']:>35}")

# ===================================================================
# [5/5] Visualization and summary
# ===================================================================
print("\n[5/5] Visualization ...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: Counterfactual bar chart
ax = axes[0, 0]
cf_df = pd.DataFrame(cf_results)
colors = ['green' if d < 0 else 'red' for d in cf_df['pct_change']]
bars = ax.barh(range(len(cf_df)), cf_df['pct_change'], color=colors, alpha=0.7)
ax.set_yticks(range(len(cf_df)))
ax.set_yticklabels([r['feature'] for r in cf_results], fontsize=8)
ax.set_xlabel('% change in P(EGC)')
ax.set_title('Counterfactual: Effect on EGC Fate')
ax.axvline(0, color='black', lw=0.5)

# Panel B: Drug target priority
ax = axes[0, 1]
top_targets = target_df[target_df['priority_score'] > 0].head(7)
ax.barh(range(len(top_targets)), top_targets['priority_score'],
        color=['#d73027' if t > 0 else '#4575b4'
               for t in top_targets['theta']], alpha=0.7)
ax.set_yticks(range(len(top_targets)))
ax.set_yticklabels(top_targets['feature'], fontsize=9)
ax.set_xlabel('Priority Score')
ax.set_title('Drug Target Priority (|theta| x druggability)')

# Panel C: V(s) landscape shift under best intervention
ax = axes[1, 0]
# Best intervention = most negative delta_P_EGC
best_cf = cf_df.loc[cf_df['delta_P_EGC'].idxmin()]
theta_best = theta_vec.copy()
best_idx = feature_names.index(best_cf['feature'])
best_params = list(interventions.values())[cf_df['delta_P_EGC'].idxmin()]
theta_best[best_params['feature_idx']] = best_params['new_weight']
V_best = soft_value_iteration(theta_best, phi_z, T_cr)

pt = adata_mc.obs['dpt_pseudotime'].values
ax.scatter(pt, V_recon, c='gray', s=3, alpha=0.3, label='Original V(s)')
ax.scatter(pt[im_idx], V_best[im_idx], c='blue', s=5, alpha=0.5,
           label=f'CF: {best_cf["feature"]}')
ax.set_xlabel('Pseudotime')
ax.set_ylabel('V(s)')
ax.set_title(f'Best Intervention: {best_cf["intervention"][:30]}...')
ax.legend(fontsize=8)

# Panel D: Fate distribution shift
ax = axes[1, 1]
ax.hist(p_egc_baseline, bins=30, alpha=0.5, color='red',
        label=f'Baseline (mean={p_egc_baseline.mean():.3f})')
p_best = compute_fate_from_V(V_best, im_idx, T_cr, stages)
ax.hist(p_best, bins=30, alpha=0.5, color='blue',
        label=f'After intervention (mean={p_best.mean():.3f})')
ax.set_xlabel('P(EGC)')
ax.set_ylabel('Count')
ax.set_title('IM Cell Fate Distribution Shift')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / "irl_counterfactual.png", dpi=150, bbox_inches='tight')
plt.close()

# Save
cf_df.to_csv(RESULTS / "irl_counterfactual_results.csv", index=False)
target_df.to_csv(RESULTS / "irl_drug_target_priority.csv", index=False)

print(f"\n  Saved: irl_counterfactual.png")
print(f"  Saved: irl_counterfactual_results.csv")
print(f"  Saved: irl_drug_target_priority.csv")

print("\n" + "=" * 70)
print("Phase 25f COMPLETE")
print("=" * 70)
print(f"\n  Counterfactual results:")
for _, row in cf_df.iterrows():
    direction = "REDUCES" if row['pct_change'] < 0 else "INCREASES"
    print(f"    {row['feature']:>20}: {direction} EGC risk by "
          f"{abs(row['pct_change']):.1f}%")
print(f"\n  Top drug targets:")
for _, row in target_df.head(3).iterrows():
    print(f"    {row['feature']}: {row['candidate_drugs']}")
print(f"\n  KEY: These analyses are ONLY possible with IRL framework")
print(f"       (cannot be done with pseudotime/absorption/CellRank alone)")
print("=" * 70)
