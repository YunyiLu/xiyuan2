"""
Step 11c: GSE29272 Validation — 134 paired Chinese GC (GPL96)
Large paired tumor/normal cohort for cancer vs normal validation.
"""
import sys, os, warnings, gzip, urllib.request, ssl
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"
DATA_DIR = f"{BASE}/data/validation"
os.makedirs(DATA_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = set(candidates_df['gene'].tolist())
print(f"Loaded {len(CANDIDATES)} candidates")

# ===== GPL96 annotation =====
gpl96_dest = f"{DATA_DIR}/GPL96.annot.gz"
if not os.path.exists(gpl96_dest):
    print("Downloading GPL96...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    resp = urllib.request.urlopen(req, timeout=60, context=ctx)
    with open(gpl96_dest, 'wb') as f:
        f.write(resp.read())
    print(f"  {os.path.getsize(gpl96_dest)/1024:.0f} KB")

probe_to_gene = {}
in_table = False
with gzip.open(gpl96_dest, 'rt', errors='replace') as f:
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
                    probe_to_gene[probe_id] = g

print(f"GPL96: {len(probe_to_gene)} probes -> {len(set(probe_to_gene.values()))} genes")

# ===== Download GSE29272 =====
dest = f"{DATA_DIR}/GSE29272_series_matrix.txt.gz"
if not os.path.exists(dest):
    print("Downloading GSE29272 (~26MB)...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE29nnn/GSE29272/matrix/GSE29272_series_matrix.txt.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    resp = urllib.request.urlopen(req, timeout=300, context=ctx)
    with open(dest, 'wb') as f:
        while True:
            chunk = resp.read(512*1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  {os.path.getsize(dest)/1024/1024:.1f} MB")
else:
    print(f"  Exists: {os.path.getsize(dest)/1024/1024:.1f} MB")

# ===== Parse =====
data_rows = []
header_samples = []
sample_titles = {}
sample_ids = []
sample_chars = {}

with gzip.open(dest, 'rt', errors='replace') as f:
    reading = False
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            sample_ids = [p.strip('"') for p in parts[1:]]
        elif line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids):
                    sample_titles[sample_ids[i]] = p.strip('"')
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
            if probe in probe_to_gene:
                gene = probe_to_gene[probe]
                values = []
                for v in parts[1:]:
                    try:
                        values.append(float(v.strip('"')))
                    except:
                        values.append(np.nan)
                data_rows.append([gene] + values)

print(f"\nSamples: {len(sample_ids)}")
print(f"Genes matched: {len(set(r[0] for r in data_rows))}")
print(f"Sample titles (first 3): {list(sample_titles.values())[:3]}")

# Classify
stages = {}
for sid in sample_ids:
    title = sample_titles.get(sid, '').lower()
    chars = ' '.join(sample_chars.get(sid, [])).lower()
    combined = title + ' ' + chars
    if 'normal' in combined or 'adjacent' in combined:
        stages[sid] = 'Normal'
    elif 'tumor' in combined or 'cancer' in combined:
        stages[sid] = 'Cancer'
    else:
        stages[sid] = 'Unknown'

print(f"Stages: {pd.Series(stages).value_counts().to_dict()}")

# ===== Analysis =====
if data_rows:
    expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
    expr_df = expr_df.groupby('gene').mean()
    print(f"Matrix: {expr_df.shape}")

    normal_samps = [s for s in expr_df.columns if stages.get(s) == 'Normal']
    cancer_samps = [s for s in expr_df.columns if stages.get(s) == 'Cancer']
    print(f"Normal: {len(normal_samps)}, Cancer: {len(cancer_samps)}")

    results = []
    for gene in expr_df.index:
        n_vals = expr_df.loc[gene, normal_samps].dropna().values
        c_vals = expr_df.loc[gene, cancer_samps].dropna().values
        if len(n_vals) >= 5 and len(c_vals) >= 5:
            _, p = mannwhitneyu(c_vals, n_vals, alternative='two-sided')
            fc = np.log2((np.mean(c_vals) + 0.01) / (np.mean(n_vals) + 0.01))
            results.append({
                'gene': gene,
                'cancer_vs_normal_p': p,
                'cancer_vs_normal_logFC': fc,
                'mean_normal': np.mean(n_vals),
                'mean_cancer': np.mean(c_vals),
                'dataset': 'GSE29272',
                'n_normal': len(n_vals),
                'n_cancer': len(c_vals)
            })

    res_df = pd.DataFrame(results)
    _, res_df['fdr'], _, _ = multipletests(res_df['cancer_vs_normal_p'], method='fdr_bh')
    res_df = res_df.sort_values('cancer_vs_normal_p')
    res_df.to_csv(f"{RES_DIR}/gse29272_validation.csv", index=False)

    sig = (res_df['fdr'] < 0.05).sum()
    up = ((res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] > 0)).sum()
    down = ((res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] < 0)).sum()
    print(f"\nResults: {len(res_df)} genes")
    print(f"Significant (FDR<0.05): {sig}/{len(res_df)}")
    print(f"  Up in cancer: {up}")
    print(f"  Down in cancer: {down}")

    print(f"\nTop upregulated in cancer:")
    top_up = res_df[(res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] > 0)].nlargest(10, 'cancer_vs_normal_logFC')
    for _, r in top_up.iterrows():
        print(f"  {r['gene']}: logFC={r['cancer_vs_normal_logFC']:.3f}, FDR={r['fdr']:.2e}")

    print(f"\nTop downregulated in cancer:")
    top_down = res_df[(res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] < 0)].nsmallest(10, 'cancer_vs_normal_logFC')
    for _, r in top_down.iterrows():
        print(f"  {r['gene']}: logFC={r['cancer_vs_normal_logFC']:.3f}, FDR={r['fdr']:.2e}")

# ===== Also do GSE13911 (69 pairs, GPL96) =====
print("\n" + "="*70)
print("GSE13911 — 69 paired GC vs Normal (GPL96)")
print("="*70)

dest2 = f"{DATA_DIR}/GSE13911_series_matrix.txt.gz"
if not os.path.exists(dest2):
    print("Downloading GSE13911 (~8MB)...")
    url2 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13911/matrix/GSE13911_series_matrix.txt.gz"
    req = urllib.request.Request(url2)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    resp = urllib.request.urlopen(req, timeout=300, context=ctx)
    with open(dest2, 'wb') as f:
        while True:
            chunk = resp.read(512*1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  {os.path.getsize(dest2)/1024/1024:.1f} MB")

# Parse GSE13911
data_rows2 = []
header_samples2 = []
sample_titles2 = {}
sample_ids2 = []

with gzip.open(dest2, 'rt', errors='replace') as f:
    reading = False
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            sample_ids2 = [p.strip('"') for p in parts[1:]]
        elif line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids2):
                    sample_titles2[sample_ids2[i]] = p.strip('"')
        elif line.startswith('"ID_REF"'):
            reading = True
            parts = line.strip().split('\t')
            header_samples2 = [p.strip('"') for p in parts[1:]]
            continue
        elif reading:
            if line.startswith('!') or not line.strip():
                break
            parts = line.strip().split('\t')
            probe = parts[0].strip('"')
            if probe in probe_to_gene:
                gene = probe_to_gene[probe]
                values = []
                for v in parts[1:]:
                    try:
                        values.append(float(v.strip('"')))
                    except:
                        values.append(np.nan)
                data_rows2.append([gene] + values)

print(f"Samples: {len(sample_ids2)}")
print(f"Genes matched: {len(set(r[0] for r in data_rows2))}")

stages2 = {}
sample_chars2 = {}
with gzip.open(dest2, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('!Sample_characteristics'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                if i < len(sample_ids2):
                    sample_chars2.setdefault(sample_ids2[i], []).append(p.strip('"'))
        elif line.startswith('"ID_REF"'):
            break

for sid in sample_ids2:
    title = sample_titles2.get(sid, '').lower()
    chars = ' '.join(sample_chars2.get(sid, [])).lower()
    combined = title + ' ' + chars
    if 'normal' in combined or 'control' in combined or 'non-tumor' in combined or 'adjacent' in combined or 'healthy' in combined:
        stages2[sid] = 'Normal'
    elif 'tumor' in combined or 'cancer' in combined or 'carcinoma' in combined or 'malignant' in combined:
        stages2[sid] = 'Cancer'
    else:
        stages2[sid] = 'Unknown'

print(f"Stages: {pd.Series(stages2).value_counts().to_dict()}")

if data_rows2:
    expr_df2 = pd.DataFrame(data_rows2, columns=['gene'] + header_samples2)
    expr_df2 = expr_df2.groupby('gene').mean()

    normal2 = [s for s in expr_df2.columns if stages2.get(s) == 'Normal']
    cancer2 = [s for s in expr_df2.columns if stages2.get(s) == 'Cancer']

    results2 = []
    for gene in expr_df2.index:
        n_vals = expr_df2.loc[gene, normal2].dropna().values
        c_vals = expr_df2.loc[gene, cancer2].dropna().values
        if len(n_vals) >= 5 and len(c_vals) >= 5:
            _, p = mannwhitneyu(c_vals, n_vals, alternative='two-sided')
            fc = np.log2((np.mean(c_vals) + 0.01) / (np.mean(n_vals) + 0.01))
            results2.append({
                'gene': gene,
                'cancer_vs_normal_p': p,
                'cancer_vs_normal_logFC': fc,
                'mean_normal': np.mean(n_vals),
                'mean_cancer': np.mean(c_vals),
                'dataset': 'GSE13911',
                'n_normal': len(n_vals),
                'n_cancer': len(c_vals)
            })

    res_df2 = pd.DataFrame(results2)
    _, res_df2['fdr'], _, _ = multipletests(res_df2['cancer_vs_normal_p'], method='fdr_bh')
    res_df2 = res_df2.sort_values('cancer_vs_normal_p')
    res_df2.to_csv(f"{RES_DIR}/gse13911_validation.csv", index=False)

    sig2 = (res_df2['fdr'] < 0.05).sum()
    print(f"\nResults: {len(res_df2)} genes")
    print(f"Significant (FDR<0.05): {sig2}/{len(res_df2)}")
    up2 = ((res_df2['fdr'] < 0.05) & (res_df2['cancer_vs_normal_logFC'] > 0)).sum()
    down2 = ((res_df2['fdr'] < 0.05) & (res_df2['cancer_vs_normal_logFC'] < 0)).sum()
    print(f"  Up: {up2}, Down: {down2}")

print("\nAll done!")
