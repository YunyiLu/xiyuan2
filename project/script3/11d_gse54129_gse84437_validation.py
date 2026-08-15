"""
Step 11d: GSE84437 + GSE54129 Validation
  - GSE84437: 433 Korean GC tumors (Illumina HumanHT-12 V3, gene-symbol indexed)
  - GSE54129: 132 Chinese (111 GC + 21 normal, GPL570)
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


def download_file(url, dest, label=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"  {label} exists: {os.path.getsize(dest)/1024/1024:.1f} MB")
        return True
    print(f"  Downloading {label}...")
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    try:
        resp = urllib.request.urlopen(req, timeout=300, context=ctx)
        with open(dest, 'wb') as f:
            while True:
                chunk = resp.read(512*1024)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  {os.path.getsize(dest)/1024/1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ===== GPL570 annotation (for GSE54129) =====
gpl570_dest = f"{DATA_DIR}/GPL570.annot.gz"
if not os.path.exists(gpl570_dest):
    print("Downloading GPL570 annotation...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    try:
        resp = urllib.request.urlopen(req, timeout=120, context=ctx)
        with open(gpl570_dest, 'wb') as f:
            f.write(resp.read())
        print(f"  {os.path.getsize(gpl570_dest)/1024:.0f} KB")
    except Exception as e:
        print(f"  GPL570 download failed: {e}")

probe570 = {}
if os.path.exists(gpl570_dest):
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


# ===== GSE54129 (132 Chinese, GPL570) =====
print("\n" + "="*70)
print("GSE54129 — 111 GC + 21 Normal (Chinese, GPL570)")
print("="*70)

dest54 = f"{DATA_DIR}/GSE54129_series_matrix.txt.gz"
download_file(
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54129/matrix/GSE54129_series_matrix.txt.gz",
    dest54, "GSE54129"
)

if os.path.exists(dest54) and probe570:
    data_rows = []
    header_samples = []
    sample_ids = []
    sample_titles = {}
    sample_chars = {}

    with gzip.open(dest54, 'rt', errors='replace') as f:
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
                if probe in probe570:
                    gene = probe570[probe]
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    data_rows.append([gene] + values)

    print(f"Samples: {len(sample_ids)}")
    print(f"Genes matched: {len(set(r[0] for r in data_rows))}")

    stages = {}
    for sid in sample_ids:
        title = sample_titles.get(sid, '').lower()
        chars = ' '.join(sample_chars.get(sid, [])).lower()
        combined = title + ' ' + chars
        if 'normal' in combined or 'non-cancer' in combined or 'healthy' in combined or 'noncancer' in combined:
            stages[sid] = 'Normal'
        elif 'tumor' in combined or 'cancer' in combined or 'carcinoma' in combined:
            stages[sid] = 'Cancer'
        else:
            stages[sid] = 'Unknown'

    print(f"Stages: {pd.Series(stages).value_counts().to_dict()}")

    if data_rows:
        expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
        expr_df = expr_df.groupby('gene').mean()
        normal_s = [s for s in expr_df.columns if stages.get(s) == 'Normal']
        cancer_s = [s for s in expr_df.columns if stages.get(s) == 'Cancer']
        print(f"Normal: {len(normal_s)}, Cancer: {len(cancer_s)}")

        results = []
        for gene in expr_df.index:
            n_vals = expr_df.loc[gene, normal_s].dropna().values
            c_vals = expr_df.loc[gene, cancer_s].dropna().values
            if len(n_vals) >= 5 and len(c_vals) >= 5:
                _, p = mannwhitneyu(c_vals, n_vals, alternative='two-sided')
                fc = np.log2((np.mean(c_vals)+0.01)/(np.mean(n_vals)+0.01))
                results.append({
                    'gene': gene, 'cancer_vs_normal_p': p,
                    'cancer_vs_normal_logFC': fc,
                    'mean_normal': np.mean(n_vals), 'mean_cancer': np.mean(c_vals),
                    'dataset': 'GSE54129', 'n_normal': len(n_vals), 'n_cancer': len(c_vals)
                })

        res_df = pd.DataFrame(results)
        _, res_df['fdr'], _, _ = multipletests(res_df['cancer_vs_normal_p'], method='fdr_bh')
        res_df = res_df.sort_values('cancer_vs_normal_p')
        res_df.to_csv(f"{RES_DIR}/gse54129_validation.csv", index=False)

        sig = (res_df['fdr'] < 0.05).sum()
        up = ((res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] > 0)).sum()
        down = ((res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] < 0)).sum()
        print(f"\nResults: {len(res_df)} genes tested")
        print(f"Significant (FDR<0.05): {sig}/{len(res_df)}")
        print(f"  Up: {up}, Down: {down}")

        top_up = res_df[(res_df['fdr'] < 0.05) & (res_df['cancer_vs_normal_logFC'] > 0)].nlargest(10, 'cancer_vs_normal_logFC')
        print("\nTop up in cancer:")
        for _, r in top_up.iterrows():
            print(f"  {r['gene']}: logFC={r['cancer_vs_normal_logFC']:.3f}, FDR={r['fdr']:.2e}")


# ===== GSE84437 (483 Korean GC, GPL6947 Illumina HumanHT-12 V3) =====
print("\n" + "="*70)
print("GSE84437 — 433 Korean GC (Illumina HumanHT-12, tumor-only for survival)")
print("="*70)

# This is tumor-only (no normal), so we'll extract expression and compute
# percentile ranking (top 25% vs bottom 25% — proxy for differential)
# Actually let's check if it has normal first

dest84 = f"{DATA_DIR}/GSE84437_series_matrix.txt.gz"
download_file(
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84437/matrix/GSE84437_series_matrix.txt.gz",
    dest84, "GSE84437"
)

# GPL6947 annotation
gpl6947_dest = f"{DATA_DIR}/GPL6947.annot.gz"
if not os.path.exists(gpl6947_dest):
    print("Downloading GPL6947 annotation...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL6947/annot/GPL6947.annot.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    try:
        resp = urllib.request.urlopen(req, timeout=120, context=ctx)
        with open(gpl6947_dest, 'wb') as f:
            f.write(resp.read())
        print(f"  {os.path.getsize(gpl6947_dest)/1024:.0f} KB")
    except Exception as e:
        print(f"  Failed: {e}")

probe6947 = {}
if os.path.exists(gpl6947_dest):
    in_table = False
    with gzip.open(gpl6947_dest, 'rt', errors='replace') as f:
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
                        probe6947[probe_id] = g
    print(f"GPL6947: {len(probe6947)} probes -> {len(set(probe6947.values()))} genes")

if os.path.exists(dest84) and probe6947:
    data_rows = []
    header_samples = []
    sample_ids = []

    with gzip.open(dest84, 'rt', errors='replace') as f:
        reading = False
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                parts = line.strip().split('\t')
                sample_ids = [p.strip('"') for p in parts[1:]]
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
                if probe in probe6947:
                    gene = probe6947[probe]
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    data_rows.append([gene] + values)

    print(f"Samples: {len(sample_ids)}")
    print(f"Genes matched: {len(set(r[0] for r in data_rows))}")

    if data_rows:
        expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
        expr_df = expr_df.groupby('gene').mean()
        print(f"Matrix: {expr_df.shape}")

        # Tumor-only dataset: compute expression statistics (mean, median, variance)
        # This tells us which genes are highly expressed / variable in GC
        stats = []
        for gene in expr_df.index:
            vals = expr_df.loc[gene].dropna().values
            stats.append({
                'gene': gene,
                'mean_expr': np.mean(vals),
                'median_expr': np.median(vals),
                'std_expr': np.std(vals),
                'cv': np.std(vals)/np.mean(vals) if np.mean(vals) > 0 else 0,
                'pct75': np.percentile(vals, 75),
                'pct25': np.percentile(vals, 25),
                'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
                'n_samples': len(vals),
                'dataset': 'GSE84437'
            })

        stats_df = pd.DataFrame(stats).sort_values('mean_expr', ascending=False)
        stats_df.to_csv(f"{RES_DIR}/gse84437_expression_stats.csv", index=False)
        print(f"\nResults: {len(stats_df)} genes in 433 tumors")
        print(f"Top expressed:")
        for _, r in stats_df.head(10).iterrows():
            print(f"  {r['gene']}: mean={r['mean_expr']:.2f}, CV={r['cv']:.3f}")

        print(f"\nMost variable (high CV):")
        top_var = stats_df.nlargest(10, 'cv')
        for _, r in top_var.iterrows():
            print(f"  {r['gene']}: CV={r['cv']:.3f}, mean={r['mean_expr']:.2f}")

print("\nAll done!")
