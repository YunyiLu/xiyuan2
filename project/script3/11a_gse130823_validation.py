"""
Step 11a: GSE130823 Validation — LGIN/HGIN/EGC progression microarray (94 samples)
  J Pathol 2020, 47 paired samples covering dysplasia→cancer progression

  Validates: 92 candidate genes in an independent dysplasia cohort
  Key advantage: larger than GSE55696, covers LGIN→HGIN→EGC stages

Output:
  - results/gse130823_validation.csv
  - results/gse130823_stage_comparison.csv
"""
import sys, os, warnings, gzip, urllib.request, ssl
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"
DATA_DIR = f"{BASE}/data/validation"
os.makedirs(DATA_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Load candidates
candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()
print(f"Loaded {len(CANDIDATES)} candidate genes")

# ============================================================
# Download GSE130823
# ============================================================
print("\n" + "="*70)
print("DOWNLOADING GSE130823")
print("="*70)

dest = f"{DATA_DIR}/GSE130823_series_matrix.txt.gz"
url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130823/matrix/GSE130823_series_matrix.txt.gz"

if not os.path.exists(dest):
    print("  Downloading series matrix (~22MB)...")
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    resp = urllib.request.urlopen(req, timeout=300, context=ctx)

    downloaded = 0
    with open(dest, 'wb') as f:
        while True:
            chunk = resp.read(512*1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (5*1024*1024) < 512*1024:
                print(f"    {downloaded/1024/1024:.1f} MB")
    print(f"  Done: {downloaded/1024/1024:.1f} MB")
else:
    print(f"  Already exists: {os.path.getsize(dest)/1024/1024:.1f} MB")

# ============================================================
# Parse metadata
# ============================================================
print("\n" + "="*70)
print("PARSING METADATA")
print("="*70)

sample_info = {}
sample_ids = []

with gzip.open(dest, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            sample_ids = [p.strip('"') for p in parts[1:]]
        elif line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            titles = [p.strip('"') for p in parts[1:]]
            for i, t in enumerate(titles):
                if i < len(sample_ids):
                    sample_info.setdefault(sample_ids[i], {})['title'] = t
        elif line.startswith('!Sample_characteristics_ch1') and 'tissue' in line.lower():
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids):
                    sample_info.setdefault(sample_ids[i], {})['tissue'] = p.strip('"')
        elif line.startswith('!Sample_characteristics_ch1') and ('histol' in line.lower() or 'stage' in line.lower() or 'grade' in line.lower()):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids):
                    sample_info.setdefault(sample_ids[i], {})['histology'] = p.strip('"')
        elif line.startswith('"ID_REF"'):
            break

print(f"  Samples: {len(sample_ids)}")
if sample_info:
    # Show first few
    for sid in list(sample_info.keys())[:5]:
        print(f"    {sid}: {sample_info[sid]}")

# Classify samples by stage
stage_map = {}
for sid, info in sample_info.items():
    title = info.get('title', '').lower()
    histology = info.get('histology', '').lower()
    combined = title + ' ' + histology

    if 'normal' in combined or 'adjacent' in combined or 'non-tumor' in combined:
        stage_map[sid] = 'Normal'
    elif 'high-grade' in combined or 'hgin' in combined or 'high grade' in combined:
        stage_map[sid] = 'HGIN'
    elif 'low-grade' in combined or 'lgin' in combined or 'low grade' in combined:
        stage_map[sid] = 'LGIN'
    elif 'intestinal metaplasia' in combined or ' im ' in combined:
        stage_map[sid] = 'IM'
    elif 'dysplasia' in combined:
        stage_map[sid] = 'Dysplasia'
    elif 'cancer' in combined or 'carcinoma' in combined or 'tumor' in combined or 'egc' in combined:
        stage_map[sid] = 'Cancer'
    elif 'gastritis' in combined or 'inflammation' in combined:
        stage_map[sid] = 'Gastritis'
    else:
        stage_map[sid] = 'Unknown'

stage_counts = pd.Series(stage_map).value_counts()
print(f"\n  Stage distribution:")
print(stage_counts)

# ============================================================
# Parse expression matrix
# ============================================================
print("\n" + "="*70)
print("PARSING EXPRESSION MATRIX")
print("="*70)

# Read expression data - look for our 92 genes
data_rows = []
gene_col = None
reading_data = False
header_samples = []

with gzip.open(dest, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('"ID_REF"'):
            reading_data = True
            parts = line.strip().split('\t')
            header_samples = [p.strip('"') for p in parts[1:]]
            continue
        if reading_data:
            if line.startswith('!') or not line.strip():
                break
            parts = line.strip().split('\t')
            probe_id = parts[0].strip('"')
            # For microarray, probe_id might be gene symbol or probe ID
            # Check if it's one of our candidates
            if probe_id in CANDIDATES:
                values = []
                for v in parts[1:]:
                    try:
                        values.append(float(v.strip('"')))
                    except:
                        values.append(np.nan)
                data_rows.append([probe_id] + values)

print(f"  Direct gene matches: {len(data_rows)}")

# If no direct matches, this is likely a probe-based array
# Need to check if ID_REF contains gene symbols or probe IDs
if len(data_rows) < 10:
    print("  Few direct matches — checking if array uses probe IDs...")
    # Read first 20 IDs to check format
    probe_samples = []
    with gzip.open(dest, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('"ID_REF"'):
                reading_data = True
                continue
            if reading_data:
                if line.startswith('!') or not line.strip():
                    break
                parts = line.strip().split('\t')
                probe_samples.append(parts[0].strip('"'))
                if len(probe_samples) >= 20:
                    break

    print(f"  First few IDs: {probe_samples[:5]}")

    # If numeric or starts with numbers, it's likely Agilent/Illumina probe IDs
    # Check GPL platform
    platform = None
    with gzip.open(dest, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!Series_platform_id') or line.startswith('!Sample_platform_id'):
                parts = line.strip().split('\t')
                platform = parts[1].strip('"') if len(parts) > 1 else None
                break
    print(f"  Platform: {platform}")

    # For Agilent arrays, the GENE_SYMBOL might be in another column
    # Or we need the GPL annotation file
    # Alternative: read ALL data and try to match gene symbols from titles

    # Let's try reading all rows and check if any probe looks like a gene
    all_genes_found = set()
    gene_to_probes = {}
    n_rows = 0

    with gzip.open(dest, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('"ID_REF"'):
                reading_data = True
                parts = line.strip().split('\t')
                header_samples = [p.strip('"') for p in parts[1:]]
                continue
            if reading_data:
                if line.startswith('!') or not line.strip():
                    break
                parts = line.strip().split('\t')
                probe_id = parts[0].strip('"')
                n_rows += 1

                # Check if this probe ID IS a gene symbol
                if probe_id in CANDIDATES:
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    data_rows.append([probe_id] + values)
                    all_genes_found.add(probe_id)

    print(f"  Total probes/rows: {n_rows}")
    print(f"  Candidate genes found by symbol: {len(all_genes_found)}")

if len(data_rows) >= 5:
    expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
    # If multiple probes per gene, take mean
    expr_df = expr_df.groupby('gene').mean()
    print(f"  Expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")

    # Add stage info
    stages = pd.Series({s: stage_map.get(s, 'Unknown') for s in expr_df.columns})

    # ============================================================
    # Stage comparison
    # ============================================================
    print("\n" + "="*70)
    print("STAGE COMPARISON")
    print("="*70)

    results = []
    for gene in expr_df.index:
        row = {'gene': gene}

        gene_vals = expr_df.loc[gene]

        # Get values per stage
        for stage in ['Normal', 'Gastritis', 'IM', 'LGIN', 'HGIN', 'Dysplasia', 'Cancer']:
            stage_samples = [s for s in expr_df.columns if stages[s] == stage]
            if stage_samples:
                vals = gene_vals[stage_samples].dropna().values
                row[f'mean_{stage}'] = np.mean(vals) if len(vals) > 0 else np.nan
                row[f'n_{stage}'] = len(vals)

        # Key comparisons
        # Cancer vs Normal
        normal_samps = [s for s in expr_df.columns if stages[s] == 'Normal']
        cancer_samps = [s for s in expr_df.columns if stages[s] == 'Cancer']

        if normal_samps and cancer_samps:
            n_vals = gene_vals[normal_samps].dropna().values
            c_vals = gene_vals[cancer_samps].dropna().values
            if len(n_vals) >= 3 and len(c_vals) >= 3:
                _, p = mannwhitneyu(c_vals, n_vals, alternative='two-sided')
                row['cancer_vs_normal_p'] = p
                row['cancer_vs_normal_fc'] = np.log2((np.mean(c_vals)+0.01)/(np.mean(n_vals)+0.01))

        # HGIN vs LGIN (progression)
        lgin_samps = [s for s in expr_df.columns if stages[s] == 'LGIN']
        hgin_samps = [s for s in expr_df.columns if stages[s] == 'HGIN']

        if lgin_samps and hgin_samps:
            l_vals = gene_vals[lgin_samps].dropna().values
            h_vals = gene_vals[hgin_samps].dropna().values
            if len(l_vals) >= 3 and len(h_vals) >= 3:
                _, p = mannwhitneyu(h_vals, l_vals, alternative='two-sided')
                row['hgin_vs_lgin_p'] = p
                row['hgin_vs_lgin_fc'] = np.log2((np.mean(h_vals)+0.01)/(np.mean(l_vals)+0.01))

        # Kruskal across all available stages
        all_groups = []
        for stage in ['Normal', 'LGIN', 'HGIN', 'Cancer']:
            samps = [s for s in expr_df.columns if stages[s] == stage]
            if samps:
                vals = gene_vals[samps].dropna().values
                if len(vals) >= 3:
                    all_groups.append(vals)

        if len(all_groups) >= 3:
            _, kw_p = kruskal(*all_groups)
            row['kruskal_p'] = kw_p

        results.append(row)

    results_df = pd.DataFrame(results)

    # FDR correction
    if 'kruskal_p' in results_df.columns:
        valid_p = results_df['kruskal_p'].dropna()
        if len(valid_p) >= 2:
            _, fdr, _, _ = multipletests(valid_p, method='fdr_bh')
            results_df.loc[valid_p.index, 'kruskal_fdr'] = fdr

    results_df = results_df.sort_values('kruskal_p' if 'kruskal_p' in results_df.columns else 'cancer_vs_normal_p')
    results_df.to_csv(f"{RES_DIR}/gse130823_validation.csv", index=False)

    print(f"  Genes analyzed: {len(results_df)}")
    if 'kruskal_fdr' in results_df.columns:
        print(f"  Significant stage effect (FDR<0.05): {(results_df['kruskal_fdr'] < 0.05).sum()}")
    if 'cancer_vs_normal_p' in results_df.columns:
        sig_cn = results_df['cancer_vs_normal_p'] < 0.05
        up_cn = results_df['cancer_vs_normal_fc'] > 0
        print(f"  Cancer > Normal (p<0.05): {(sig_cn & up_cn).sum()}")
        print(f"  Cancer < Normal (p<0.05): {(sig_cn & ~up_cn).sum()}")
    if 'hgin_vs_lgin_p' in results_df.columns:
        sig_hl = results_df['hgin_vs_lgin_p'] < 0.05
        print(f"  HGIN vs LGIN significant: {sig_hl.sum()}")

    print("\n  Top progression genes (Cancer vs Normal):")
    if 'cancer_vs_normal_fc' in results_df.columns:
        top = results_df.nlargest(10, 'cancer_vs_normal_fc')
        for _, r in top.iterrows():
            print(f"    {r['gene']}: FC={r.get('cancer_vs_normal_fc',0):.2f}, p={r.get('cancer_vs_normal_p',1):.4f}")
else:
    print(f"\n  Only {len(data_rows)} genes matched directly.")
    print("  This array likely uses probe IDs — need GPL annotation for gene mapping.")
    print("  Attempting to use GEOparse or direct GPL download...")

    # Save what we have
    if data_rows:
        expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
        expr_df.to_csv(f"{RES_DIR}/gse130823_partial.csv", index=False)
        print(f"  Saved partial: {len(data_rows)} genes")

print("\nDone!")
