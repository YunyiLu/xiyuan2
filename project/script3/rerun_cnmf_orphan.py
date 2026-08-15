"""Re-run only the cNMF orphan identification with fixed overlap coefficient logic."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"

# Load existing cNMF results (no need to re-run NMF itself)
program_genes = pd.read_csv(f"{BASE}/results/cnmf_program_genes.csv", index_col=0)
corr_df = pd.read_csv(f"{BASE}/results/cnmf_stage_correlation.csv")

H_consensus = program_genes.values  # k x genes
hvg_names = program_genes.columns.tolist()
k = H_consensus.shape[0]
stable_mask = corr_df['stable'].values

print(f"Loaded: {k} programs, {len(hvg_names)} genes")
print(f"Stable programs: {stable_mask.sum()}/{k}")

# --- Fixed orphan identification (overlap coefficient) ---
known_signatures = {
    'PMC_2': ["NAMPT", "ALDH1A1", "CD44", "SOX9", "OLFM4"],
    'PMC_P': ["AREG", "NAMPT", "PHLDA1", "ITGA2", "MYC"],
    'stemness': ["LGR5", "OLFM4", "SOX9", "ASCL2"],
    'proliferation': ["MKI67", "TOP2A", "PCNA", "CDK1"],
    'IM': ["CDX2", "MUC2", "TFF3", "VIL1"],
    'EMT': ["VIM", "SNAI1", "ZEB1", "CDH2"],
    'EGC_like': ["REG4", "CEACAM6", "MUC13", "CLDN3", "EPCAM", "KRT20", "ERBB2", "MET", "VEGFA"],
}

orphan_results = []
candidate_pool_B = []

for c in range(k):
    if not stable_mask[c]:
        continue
    top15_idx = np.argsort(H_consensus[c])[-15:]
    top15_genes = set(hvg_names[i] for i in top15_idx)

    max_overlap = 0
    max_sig = ""
    for sig_name, sig_genes in known_signatures.items():
        sig_set = set(sig_genes)
        intersection = len(top15_genes & sig_set)
        overlap_coef = intersection / min(len(top15_genes), len(sig_set)) if min(len(top15_genes), len(sig_set)) > 0 else 0
        if overlap_coef > max_overlap:
            max_overlap = overlap_coef
            max_sig = sig_name

    stage_r = corr_df.loc[c, 'spearman_r']
    stage_p = corr_df.loc[c, 'pval']
    is_orphan = (stage_p < 0.05) and (abs(stage_r) > 0.1) and (max_overlap < 0.4)
    orphan_results.append({
        'program': c, 'max_overlap_coef': max_overlap, 'best_match': max_sig,
        'stage_corr_p': stage_p, 'stage_corr_r': stage_r,
        'is_orphan': is_orphan
    })
    if is_orphan:
        top20_idx = np.argsort(H_consensus[c])[-20:]
        top20_loadings = H_consensus[c][top20_idx]
        for i, idx in enumerate(top20_idx):
            candidate_pool_B.append({
                'gene': hvg_names[idx],
                'program': c,
                'loading': top20_loadings[i],
                'stage_r': abs(stage_r),
                'score': top20_loadings[i] * abs(stage_r)
            })

orphan_df = pd.DataFrame(orphan_results)
orphan_df.to_csv(f"{BASE}/results/cnmf_orphan_programs.csv", index=False)

n_orphan = orphan_df['is_orphan'].sum()
print(f"\n--- Results (fixed) ---")
print(f"Orphan programs (|r|>0.1, overlap_coef<0.4): {n_orphan}/{stable_mask.sum()}")
print(f"\nAll programs:")
for _, row in orphan_df.iterrows():
    status = "ORPHAN" if row['is_orphan'] else "KNOWN"
    print(f"  P{int(row['program']):2d}: overlap={row['max_overlap_coef']:.3f} "
          f"match={row['best_match']:15s} r={row['stage_corr_r']:+.3f} p={row['stage_corr_p']:.2e} [{status}]")

if candidate_pool_B:
    pool_B = pd.DataFrame(candidate_pool_B)
    pool_B = pool_B.sort_values('score', ascending=False).drop_duplicates('gene', keep='first')
    pool_B = pool_B.head(20)
    pool_B.to_csv(f"{BASE}/results/candidate_pool_B.csv", index=False)
    print(f"\nCandidate pool B: {len(pool_B)} genes (ranked by loading * |stage_r|)")
    print(pool_B[['gene', 'program', 'loading', 'stage_r', 'score']].to_string(index=False))
else:
    print("\nNo orphan programs found → Pool B empty")
