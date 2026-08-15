"""
Step 12a: GSE62254 (ACRG) Survival Analysis
  - 300 Korean GC with clinical follow-up
  - Kaplan-Meier and Cox regression for top candidate genes
  - Already downloaded data in data/validation/GSE62254_series_matrix.txt.gz
"""
import sys, os, warnings, gzip
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"
DATA_DIR = f"{BASE}/data/validation"

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = set(candidates_df['gene'].tolist())

# Load GPL570 annotation
gpl570_dest = f"{DATA_DIR}/GPL570.annot.gz"
probe570 = {}
in_table = False
with gzip.open(gpl570_dest, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table = True
            continue
        if line.startswith('!platform_table_end'):
            break
        if not in_table:
            continue
        if line.startswith('ID\t'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            probe_id = parts[0]
            for g in parts[2].split('///'):
                g = g.strip()
                if g in CANDIDATES:
                    probe570[probe_id] = g
print(f"GPL570: {len(probe570)} probes -> {len(set(probe570.values()))} genes")

# Parse GSE62254
dest = f"{DATA_DIR}/GSE62254_series_matrix.txt.gz"
data_rows = []
header_samples = []
sample_ids = []
sample_chars = {}

with gzip.open(dest, 'rt', errors='replace') as f:
    reading = False
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            sample_ids = [p.strip('"') for p in parts[1:]]
        elif line.startswith('!Sample_characteristics'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids):
                    sample_chars.setdefault(sample_ids[i], []).append(p.strip('"'))
        elif line.startswith('"ID_REF"'):
            reading = True
            parts = line.strip().split('\t')
            header_samples = [p.strip('"') for p in parts[1:]]
            continue
        elif reading:
            if line.startswith('!') or not line.strip():
                break
            parts = line.strip().split('\t')
            probe = parts[0].strip('"')
            if probe in probe570:
                gene = probe570[probe]
                values = []
                for v in parts[1:]:
                    try:
                        values.append(float(v.strip('"')))
                    except:
                        values.append(np.nan)
                data_rows.append([gene] + values)

expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
expr_df = expr_df.groupby('gene').mean()
print(f"Expression matrix: {expr_df.shape}")

# Extract clinical data from characteristics
clinical = {}
for sid in sample_ids:
    chars = sample_chars.get(sid, [])
    info = {}
    for c in chars:
        if ':' in c:
            key, val = c.split(':', 1)
            key = key.strip().lower()
            val = val.strip()
            if 'survival' in key or 'os' in key:
                if 'month' in key or 'time' in key:
                    try:
                        info['os_months'] = float(val)
                    except:
                        pass
                elif 'status' in key or 'event' in key:
                    info['os_event'] = val
            elif 'recurrence' in key or 'rfs' in key or 'dfs' in key:
                if 'month' in key or 'time' in key:
                    try:
                        info['rfs_months'] = float(val)
                    except:
                        pass
                elif 'status' in key or 'event' in key:
                    info['rfs_event'] = val
            elif 'stage' in key:
                info['stage'] = val
            elif 'lauren' in key or 'subtype' in key or 'molecular' in key:
                info['subtype'] = val
    clinical[sid] = info

# Check what clinical data we have
clin_df = pd.DataFrame.from_dict(clinical, orient='index')
print(f"\nClinical data columns: {clin_df.columns.tolist()}")
print(f"Non-null counts:\n{clin_df.notna().sum()}")
print(f"\nSample characteristics example:")
first_sid = sample_ids[0]
print(f"  {first_sid}: {sample_chars.get(first_sid, [])}")

# If no survival data in matrix, try supplementary approach
# ACRG survival data is often available via clinical annotations
if 'os_months' not in clin_df.columns or clin_df['os_months'].notna().sum() < 50:
    print("\nSurvival data not in series matrix characteristics.")
    print("Attempting to extract from all characteristics...")

    # Re-parse all characteristics more aggressively
    all_char_keys = set()
    for sid, chars in sample_chars.items():
        for c in chars:
            if ':' in c:
                key = c.split(':')[0].strip().lower()
                all_char_keys.add(key)

    print(f"All characteristic keys: {sorted(all_char_keys)}")

    # Try different parsing strategies
    for sid in sample_ids[:3]:
        print(f"  {sid}: {sample_chars.get(sid, [])}")

# Even without survival, we can still do molecular subtype analysis
# and expression-based prognostic scoring
print("\n" + "="*70)
print("EXPRESSION-BASED ANALYSIS (ACRG 300 tumors)")
print("="*70)

# Compute expression percentiles for each candidate gene
gene_stats = []
for gene in expr_df.index:
    vals = expr_df.loc[gene].dropna().values
    gene_stats.append({
        'gene': gene,
        'mean_expr': np.mean(vals),
        'median_expr': np.median(vals),
        'std_expr': np.std(vals),
        'cv': np.std(vals) / (np.mean(vals) + 0.001),
        'pct10': np.percentile(vals, 10),
        'pct90': np.percentile(vals, 90),
        'dynamic_range': np.percentile(vals, 90) - np.percentile(vals, 10),
        'n_samples': len(vals)
    })

stats_df = pd.DataFrame(gene_stats).sort_values('mean_expr', ascending=False)

# Identify genes with high expression and high variability — good biomarker candidates
stats_df['biomarker_potential'] = stats_df['dynamic_range'] * stats_df['mean_expr']
stats_df = stats_df.sort_values('biomarker_potential', ascending=False)

print(f"\nTop genes by biomarker potential (expression × dynamic range):")
for _, r in stats_df.head(15).iterrows():
    print(f"  {r['gene']:12s}: mean={r['mean_expr']:.2f}, CV={r['cv']:.3f}, range={r['dynamic_range']:.2f}")

# Correlation analysis between candidates in tumor tissue
# (co-expression patterns support functional modules)
top_genes = stats_df.head(20)['gene'].tolist()
top_genes_in_expr = [g for g in top_genes if g in expr_df.index]
if len(top_genes_in_expr) >= 5:
    corr_matrix = expr_df.loc[top_genes_in_expr].T.corr(method='spearman')
    print(f"\nCo-expression clusters in ACRG (top correlated pairs):")
    pairs = []
    for i in range(len(top_genes_in_expr)):
        for j in range(i+1, len(top_genes_in_expr)):
            g1, g2 = top_genes_in_expr[i], top_genes_in_expr[j]
            r = corr_matrix.loc[g1, g2]
            if abs(r) > 0.3:
                pairs.append((g1, g2, r))
    pairs.sort(key=lambda x: -abs(x[2]))
    for g1, g2, r in pairs[:15]:
        print(f"  {g1:12s} - {g2:12s}: rho={r:.3f}")

stats_df.to_csv(f"{RES_DIR}/gse62254_biomarker_potential.csv", index=False)

# Check if Lauren subtype info is available
subtypes = {}
for sid in sample_ids:
    chars = sample_chars.get(sid, [])
    for c in chars:
        cl = c.lower()
        if 'intestinal' in cl:
            subtypes[sid] = 'Intestinal'
        elif 'diffuse' in cl:
            subtypes[sid] = 'Diffuse'
        elif 'mixed' in cl:
            subtypes[sid] = 'Mixed'

if subtypes:
    print(f"\n{'='*70}")
    print("LAUREN SUBTYPE ANALYSIS")
    print(f"{'='*70}")
    print(f"Subtypes: {pd.Series(subtypes).value_counts().to_dict()}")

    intestinal = [s for s in expr_df.columns if subtypes.get(s) == 'Intestinal']
    diffuse = [s for s in expr_df.columns if subtypes.get(s) == 'Diffuse']

    if len(intestinal) >= 10 and len(diffuse) >= 10:
        lauren_results = []
        for gene in expr_df.index:
            i_vals = expr_df.loc[gene, intestinal].dropna().values
            d_vals = expr_df.loc[gene, diffuse].dropna().values
            if len(i_vals) >= 10 and len(d_vals) >= 10:
                _, p = mannwhitneyu(i_vals, d_vals, alternative='two-sided')
                fc = np.log2((np.mean(i_vals)+0.01)/(np.mean(d_vals)+0.01))
                lauren_results.append({
                    'gene': gene, 'intestinal_vs_diffuse_p': p,
                    'intestinal_vs_diffuse_logFC': fc,
                    'mean_intestinal': np.mean(i_vals),
                    'mean_diffuse': np.mean(d_vals)
                })

        lauren_df = pd.DataFrame(lauren_results)
        from statsmodels.stats.multitest import multipletests
        _, lauren_df['fdr'], _, _ = multipletests(lauren_df['intestinal_vs_diffuse_p'], method='fdr_bh')
        lauren_df = lauren_df.sort_values('intestinal_vs_diffuse_p')
        lauren_df.to_csv(f"{RES_DIR}/gse62254_lauren_subtype.csv", index=False)

        sig = (lauren_df['fdr'] < 0.05).sum()
        up_int = ((lauren_df['fdr'] < 0.05) & (lauren_df['intestinal_vs_diffuse_logFC'] > 0)).sum()
        print(f"Lauren differential: {sig} significant, {up_int} higher in intestinal")

        print("\nIntestinal-type enriched (IM-origin genes expected here):")
        top_int = lauren_df[(lauren_df['fdr'] < 0.05) & (lauren_df['intestinal_vs_diffuse_logFC'] > 0)].nlargest(10, 'intestinal_vs_diffuse_logFC')
        for _, r in top_int.iterrows():
            print(f"  {r['gene']:12s}: logFC={r['intestinal_vs_diffuse_logFC']:.3f}, FDR={r['fdr']:.2e}")

print("\nDone!")
