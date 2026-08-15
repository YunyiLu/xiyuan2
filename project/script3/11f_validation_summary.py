"""
Step 11f: Compile Multi-Dataset Validation Summary
  Integrates all external validation results into a unified table and updates final synthesis.
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
print(f"Loaded {len(candidates_df)} candidates")

# ===== Load all validation results =====
validation_files = {
    'GSE29272': f"{RES_DIR}/gse29272_validation.csv",
    'GSE13911': f"{RES_DIR}/gse13911_validation.csv",
    'GSE54129': f"{RES_DIR}/gse54129_validation.csv",
    'GSE62254': f"{RES_DIR}/gse62254_acrg_validation.csv",
    'GSE116312': f"{RES_DIR}/gse116312_validation.csv",
    'GSE249874': f"{RES_DIR}/gse249874_validation_summary.csv",
}

all_val = {}
for name, path in validation_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_val[name] = df
        print(f"  {name}: {len(df)} genes")
    else:
        print(f"  {name}: NOT FOUND")

# ===== Build cross-dataset summary =====
print("\n" + "="*70)
print("CROSS-DATASET VALIDATION SUMMARY")
print("="*70)

summary_rows = []
for _, row in candidates_df.iterrows():
    gene = row['gene']
    entry = {
        'gene': gene,
        'TransformationScore': row['TransformationScore'],
        'n_datasets_tested': 0,
        'n_datasets_significant': 0,
        'n_datasets_up': 0,
        'n_datasets_down': 0,
        'datasets_sig': [],
        'max_logFC': np.nan,
        'min_fdr': np.nan,
    }

    logfcs = []
    fdrs = []

    # GSE29272 (Cancer vs Normal, 134 paired Chinese)
    if 'GSE29272' in all_val:
        df = all_val['GSE29272']
        match = df[df['gene'] == gene]
        if len(match) > 0:
            entry['n_datasets_tested'] += 1
            r = match.iloc[0]
            if r.get('fdr', 1) < 0.05:
                entry['n_datasets_significant'] += 1
                entry['datasets_sig'].append('GSE29272')
                if r['cancer_vs_normal_logFC'] > 0:
                    entry['n_datasets_up'] += 1
                else:
                    entry['n_datasets_down'] += 1
                logfcs.append(r['cancer_vs_normal_logFC'])
                fdrs.append(r['fdr'])
            entry['GSE29272_logFC'] = r.get('cancer_vs_normal_logFC', np.nan)
            entry['GSE29272_fdr'] = r.get('fdr', np.nan)

    # GSE13911 (Cancer vs Normal, 69 GC)
    if 'GSE13911' in all_val:
        df = all_val['GSE13911']
        match = df[df['gene'] == gene]
        if len(match) > 0:
            entry['n_datasets_tested'] += 1
            r = match.iloc[0]
            if r.get('fdr', 1) < 0.05:
                entry['n_datasets_significant'] += 1
                entry['datasets_sig'].append('GSE13911')
                if r['cancer_vs_normal_logFC'] > 0:
                    entry['n_datasets_up'] += 1
                else:
                    entry['n_datasets_down'] += 1
                logfcs.append(r['cancer_vs_normal_logFC'])
                fdrs.append(r['fdr'])
            entry['GSE13911_logFC'] = r.get('cancer_vs_normal_logFC', np.nan)
            entry['GSE13911_fdr'] = r.get('fdr', np.nan)

    # GSE54129 (111 Chinese GC + 21 Normal, GPL570)
    if 'GSE54129' in all_val:
        df = all_val['GSE54129']
        match = df[df['gene'] == gene]
        if len(match) > 0:
            entry['n_datasets_tested'] += 1
            r = match.iloc[0]
            if r.get('fdr', 1) < 0.05:
                entry['n_datasets_significant'] += 1
                entry['datasets_sig'].append('GSE54129')
                if r['cancer_vs_normal_logFC'] > 0:
                    entry['n_datasets_up'] += 1
                else:
                    entry['n_datasets_down'] += 1
                logfcs.append(r['cancer_vs_normal_logFC'])
                fdrs.append(r['fdr'])
            entry['GSE54129_logFC'] = r.get('cancer_vs_normal_logFC', np.nan)
            entry['GSE54129_fdr'] = r.get('fdr', np.nan)

    # GSE116312 (Precancerous, 13 samples)
    if 'GSE116312' in all_val:
        df = all_val['GSE116312']
        match = df[df['gene'] == gene]
        if len(match) > 0 and 'cancer_vs_normal_p' in df.columns:
            r = match.iloc[0]
            if pd.notna(r.get('cancer_vs_normal_p')):
                entry['n_datasets_tested'] += 1
                if r.get('fdr', 1) < 0.05:
                    entry['n_datasets_significant'] += 1
                    entry['datasets_sig'].append('GSE116312')
                    if r.get('cancer_vs_normal_logFC', 0) > 0:
                        entry['n_datasets_up'] += 1
                    else:
                        entry['n_datasets_down'] += 1
                    logfcs.append(r['cancer_vs_normal_logFC'])
                    fdrs.append(r['fdr'])

    # GSE249874 (scRNA-seq, 796K cells)
    if 'GSE249874' in all_val:
        df = all_val['GSE249874']
        match = df[df['gene'] == gene]
        if len(match) > 0:
            entry['n_datasets_tested'] += 1
            r = match.iloc[0]
            # Check for significance
            pval_col = [c for c in df.columns if 'pval' in c.lower() or 'p_val' in c.lower() or 'fdr' in c.lower()]
            fc_col = [c for c in df.columns if 'logfc' in c.lower() or 'log2fc' in c.lower() or 'logFC' in c]
            if pval_col:
                p = r.get(pval_col[0], 1)
                if p < 0.05:
                    entry['n_datasets_significant'] += 1
                    entry['datasets_sig'].append('GSE249874')
                    if fc_col and r.get(fc_col[0], 0) > 0:
                        entry['n_datasets_up'] += 1
                    else:
                        entry['n_datasets_down'] += 1

    # GSE62254 ACRG (tumor-only — expression confirmation)
    if 'GSE62254' in all_val:
        df = all_val['GSE62254']
        match = df[df['gene'] == gene]
        if len(match) > 0:
            r = match.iloc[0]
            entry['ACRG_mean_expr'] = r.get('mean_expr', r.get('mean_all', np.nan))

    if logfcs:
        entry['max_logFC'] = max(logfcs, key=abs)
        entry['min_fdr'] = min(fdrs)

    entry['datasets_sig'] = ';'.join(entry['datasets_sig'])
    summary_rows.append(entry)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('n_datasets_significant', ascending=False)
summary_df.to_csv(f"{RES_DIR}/multi_dataset_validation_summary.csv", index=False)

# ===== Report =====
print(f"\nTotal genes: {len(summary_df)}")
print(f"Tested in ≥3 datasets: {(summary_df['n_datasets_tested'] >= 3).sum()}")
print(f"Significant in ≥3 datasets: {(summary_df['n_datasets_significant'] >= 3).sum()}")
print(f"Significant in ≥4 datasets: {(summary_df['n_datasets_significant'] >= 4).sum()}")
print(f"Significant in all tested: {(summary_df['n_datasets_significant'] == summary_df['n_datasets_tested']).sum()}")

print(f"\n{'='*70}")
print("TOP VALIDATED GENES (significant in ≥3 datasets):")
print(f"{'='*70}")
top = summary_df[summary_df['n_datasets_significant'] >= 3].head(30)
for _, r in top.iterrows():
    print(f"  {r['gene']:12s} | TS={r['TransformationScore']:.3f} | Sig in {r['n_datasets_significant']}/{r['n_datasets_tested']} | Up:{r['n_datasets_up']} Down:{r['n_datasets_down']} | {r['datasets_sig']}")

# Direction consistency
print(f"\n{'='*70}")
print("DIRECTION CONSISTENCY (consistently up or down in cancer):")
print(f"{'='*70}")
consistent_up = summary_df[(summary_df['n_datasets_up'] >= 3) & (summary_df['n_datasets_down'] == 0)]
consistent_down = summary_df[(summary_df['n_datasets_down'] >= 3) & (summary_df['n_datasets_up'] == 0)]
print(f"\nConsistently UP in cancer (≥3 datasets, never down): {len(consistent_up)}")
for _, r in consistent_up.sort_values('TransformationScore', ascending=False).iterrows():
    print(f"  {r['gene']:12s} | TS={r['TransformationScore']:.3f} | Up in {r['n_datasets_up']} datasets")

print(f"\nConsistently DOWN in cancer (≥3 datasets, never up): {len(consistent_down)}")
for _, r in consistent_down.sort_values('TransformationScore', ascending=False).iterrows():
    print(f"  {r['gene']:12s} | TS={r['TransformationScore']:.3f} | Down in {r['n_datasets_down']} datasets")

# ===== Update FinalScore with validation weight =====
print(f"\n{'='*70}")
print("UPDATED FINAL RANKING (incorporating multi-dataset validation)")
print(f"{'='*70}")

# Validation score: fraction of datasets where significant
summary_df['validation_score'] = summary_df['n_datasets_significant'] / summary_df['n_datasets_tested'].clip(lower=1)
# Consistency bonus: all in same direction
summary_df['direction_consistent'] = (
    (summary_df['n_datasets_up'] == summary_df['n_datasets_significant']) |
    (summary_df['n_datasets_down'] == summary_df['n_datasets_significant'])
).astype(int)

# Updated FinalScore: original TS * 0.70 + validation_score * 0.25 + consistency * 0.05
summary_df['UpdatedFinalScore'] = (
    summary_df['TransformationScore'] * 0.70 +
    summary_df['validation_score'] * 0.25 +
    summary_df['direction_consistent'] * 0.05
)

final_ranked = summary_df.sort_values('UpdatedFinalScore', ascending=False)
final_ranked.to_csv(f"{RES_DIR}/final_validated_ranking.csv", index=False)

print(f"\nTop 20 (UpdatedFinalScore = 0.70*TS + 0.25*Validation + 0.05*Consistency):")
for i, (_, r) in enumerate(final_ranked.head(20).iterrows()):
    print(f"  {i+1:2d}. {r['gene']:12s} | UFS={r['UpdatedFinalScore']:.3f} | TS={r['TransformationScore']:.3f} | Val={r['validation_score']:.2f} | Sig {r['n_datasets_significant']}/{r['n_datasets_tested']}")

print(f"\nFiles saved:")
print(f"  multi_dataset_validation_summary.csv")
print(f"  final_validated_ranking.csv")
print("\nDone!")
