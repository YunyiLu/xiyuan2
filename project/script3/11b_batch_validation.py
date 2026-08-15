"""
Step 11b: Batch Validation with Additional GEO Datasets
  Downloads and validates 92 candidates in 3 new datasets:
  1. GSE33335 — Gastric intestinal metaplasia microarray (IM-specific)
  2. GSE116312 — Gastric precancerous lesions progression
  3. GSE87666 — Gastric premalignant lesions (ACRG-related)

  For each: download series matrix, find gene matches, stage comparison.

Output:
  - results/batch_validation_summary.csv (combined)
  - results/gse33335_validation.csv
  - results/gse116312_validation.csv
  - results/gse87666_validation.csv
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

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = set(candidates_df['gene'].tolist())
print(f"Loaded {len(CANDIDATES)} candidate genes")


def download_geo(gse, dest):
    """Download series matrix."""
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"  Already exists: {os.path.getsize(dest)/1024/1024:.1f} MB")
        return True
    prefix = gse[:5] + 'nnn' if len(gse) <= 8 else gse[:6] + 'nnn'
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    try:
        resp = urllib.request.urlopen(req, timeout=300, context=ctx)
        downloaded = 0
        with open(dest, 'wb') as f:
            while True:
                chunk = resp.read(512*1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        print(f"  Downloaded: {downloaded/1024/1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def parse_series_matrix(path, candidates):
    """Parse series matrix, extract metadata and expression for candidate genes."""
    sample_ids = []
    sample_titles = {}
    sample_chars = {}
    expression_data = []
    header_samples = []

    with gzip.open(path, 'rt', errors='replace') as f:
        reading_data = False
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                parts = line.strip().split('\t')
                sample_ids = [p.strip('"') for p in parts[1:]]
            elif line.startswith('!Sample_title'):
                parts = line.strip().split('\t')
                for i, p in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        sample_titles[sample_ids[i]] = p.strip('"')
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.strip().split('\t')
                for i, p in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        sample_chars.setdefault(sample_ids[i], []).append(p.strip('"'))
            elif line.startswith('"ID_REF"'):
                reading_data = True
                parts = line.strip().split('\t')
                header_samples = [p.strip('"') for p in parts[1:]]
                continue
            elif reading_data:
                if line.startswith('!') or not line.strip():
                    break
                parts = line.strip().split('\t')
                gene_id = parts[0].strip('"')
                if gene_id in candidates:
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    expression_data.append([gene_id] + values)

    return {
        'sample_ids': sample_ids,
        'titles': sample_titles,
        'chars': sample_chars,
        'expression': expression_data,
        'header_samples': header_samples
    }


def classify_samples(titles, chars):
    """Classify samples by disease stage."""
    stages = {}
    for sid in titles:
        title = titles.get(sid, '').lower()
        char_str = ' '.join(chars.get(sid, [])).lower()
        combined = title + ' ' + char_str

        if 'normal' in combined or 'healthy' in combined or 'non-atrophic' in combined:
            stages[sid] = 'Normal'
        elif 'intestinal metaplasia' in combined or ' im ' in combined or 'metaplasia' in combined:
            stages[sid] = 'IM'
        elif 'high-grade' in combined or 'hgin' in combined or 'high grade dysplasia' in combined:
            stages[sid] = 'HGIN'
        elif 'low-grade' in combined or 'lgin' in combined or 'low grade dysplasia' in combined:
            stages[sid] = 'LGIN'
        elif 'dysplasia' in combined:
            stages[sid] = 'Dysplasia'
        elif 'cancer' in combined or 'tumor' in combined or 'carcinoma' in combined or 'malignant' in combined:
            stages[sid] = 'Cancer'
        elif 'atrophic' in combined or 'atrophy' in combined:
            stages[sid] = 'Atrophic'
        elif 'gastritis' in combined or 'inflammation' in combined:
            stages[sid] = 'Gastritis'
        else:
            stages[sid] = 'Unknown'
    return stages


def validate_genes(expr_df, stages, dataset_name):
    """Run validation analysis on expression dataframe."""
    results = []
    stage_list = list(set(stages.values()) - {'Unknown'})

    for gene in expr_df.index:
        row = {'gene': gene, 'dataset': dataset_name}
        gene_vals = expr_df.loc[gene]

        for stage in stage_list:
            samps = [s for s in expr_df.columns if stages.get(s) == stage]
            if samps:
                vals = gene_vals[samps].dropna().values
                row[f'mean_{stage}'] = np.mean(vals) if len(vals) > 0 else np.nan
                row[f'n_{stage}'] = len(vals)

        # Key comparison: disease vs normal
        normal_samps = [s for s in expr_df.columns if stages.get(s) == 'Normal']
        disease_groups = ['IM', 'Cancer', 'LGIN', 'HGIN', 'Dysplasia']

        for disease in disease_groups:
            dis_samps = [s for s in expr_df.columns if stages.get(s) == disease]
            if len(normal_samps) >= 3 and len(dis_samps) >= 3:
                n_vals = gene_vals[normal_samps].dropna().values
                d_vals = gene_vals[dis_samps].dropna().values
                if len(n_vals) >= 3 and len(d_vals) >= 3:
                    _, p = mannwhitneyu(d_vals, n_vals, alternative='two-sided')
                    fc = np.log2((np.mean(d_vals)+0.01)/(np.mean(n_vals)+0.01))
                    row[f'{disease}_vs_Normal_p'] = p
                    row[f'{disease}_vs_Normal_fc'] = fc

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# DATASET 1: GSE33335 — Intestinal Metaplasia
# ============================================================
print("\n" + "="*70)
print("DATASET 1: GSE33335 — Gastric Intestinal Metaplasia")
print("="*70)

dest1 = f"{DATA_DIR}/GSE33335_series_matrix.txt.gz"
if download_geo('GSE33335', dest1):
    data = parse_series_matrix(dest1, CANDIDATES)
    print(f"  Samples: {len(data['sample_ids'])}")
    print(f"  Genes matched: {len(data['expression'])}")

    if data['expression']:
        expr_df = pd.DataFrame(data['expression'], columns=['gene'] + data['header_samples'])
        expr_df = expr_df.groupby('gene').mean()
        stages = classify_samples(data['titles'], data['chars'])
        stage_counts = pd.Series(stages).value_counts()
        print(f"  Stage distribution: {dict(stage_counts)}")
        print(f"  Titles sample: {list(data['titles'].values())[:3]}")

        if len(expr_df) >= 5:
            results1 = validate_genes(expr_df, stages, 'GSE33335')
            results1.to_csv(f"{RES_DIR}/gse33335_validation.csv", index=False)
            print(f"  Validated {len(results1)} genes")

            # Show key results
            for col in results1.columns:
                if col.endswith('_vs_Normal_p'):
                    sig = (results1[col] < 0.05).sum()
                    if sig > 0:
                        print(f"    {col}: {sig} significant")
    else:
        print("  No gene-symbol matches (probe-based array)")
        results1 = pd.DataFrame()


# ============================================================
# DATASET 2: GSE116312 — Precancerous Progression
# ============================================================
print("\n" + "="*70)
print("DATASET 2: GSE116312 — Gastric Precancerous Lesions")
print("="*70)

dest2 = f"{DATA_DIR}/GSE116312_series_matrix.txt.gz"
if download_geo('GSE116312', dest2):
    data = parse_series_matrix(dest2, CANDIDATES)
    print(f"  Samples: {len(data['sample_ids'])}")
    print(f"  Genes matched: {len(data['expression'])}")

    if data['expression']:
        expr_df = pd.DataFrame(data['expression'], columns=['gene'] + data['header_samples'])
        expr_df = expr_df.groupby('gene').mean()
        stages = classify_samples(data['titles'], data['chars'])
        stage_counts = pd.Series(stages).value_counts()
        print(f"  Stage distribution: {dict(stage_counts)}")
        print(f"  Titles sample: {list(data['titles'].values())[:3]}")

        if len(expr_df) >= 5:
            results2 = validate_genes(expr_df, stages, 'GSE116312')
            results2.to_csv(f"{RES_DIR}/gse116312_validation.csv", index=False)
            print(f"  Validated {len(results2)} genes")

            for col in results2.columns:
                if col.endswith('_vs_Normal_p'):
                    sig = (results2[col] < 0.05).sum()
                    if sig > 0:
                        print(f"    {col}: {sig} significant")
    else:
        print("  No gene-symbol matches")
        results2 = pd.DataFrame()


# ============================================================
# DATASET 3: GSE87666 — Premalignant Lesions
# ============================================================
print("\n" + "="*70)
print("DATASET 3: GSE87666 — Gastric Premalignant Lesions")
print("="*70)

dest3 = f"{DATA_DIR}/GSE87666_series_matrix.txt.gz"
if download_geo('GSE87666', dest3):
    data = parse_series_matrix(dest3, CANDIDATES)
    print(f"  Samples: {len(data['sample_ids'])}")
    print(f"  Genes matched: {len(data['expression'])}")

    if data['expression']:
        expr_df = pd.DataFrame(data['expression'], columns=['gene'] + data['header_samples'])
        expr_df = expr_df.groupby('gene').mean()
        stages = classify_samples(data['titles'], data['chars'])
        stage_counts = pd.Series(stages).value_counts()
        print(f"  Stage distribution: {dict(stage_counts)}")
        print(f"  Titles sample: {list(data['titles'].values())[:3]}")

        if len(expr_df) >= 5:
            results3 = validate_genes(expr_df, stages, 'GSE87666')
            results3.to_csv(f"{RES_DIR}/gse87666_validation.csv", index=False)
            print(f"  Validated {len(results3)} genes")

            for col in results3.columns:
                if col.endswith('_vs_Normal_p'):
                    sig = (results3[col] < 0.05).sum()
                    if sig > 0:
                        print(f"    {col}: {sig} significant")
    else:
        print("  No gene-symbol matches")
        results3 = pd.DataFrame()


# ============================================================
# COMBINE: Batch Validation Summary
# ============================================================
print("\n" + "="*70)
print("BATCH VALIDATION SUMMARY")
print("="*70)

all_results = []
for name, df in [('GSE33335', results1 if 'results1' in dir() else pd.DataFrame()),
                  ('GSE116312', results2 if 'results2' in dir() else pd.DataFrame()),
                  ('GSE87666', results3 if 'results3' in dir() else pd.DataFrame())]:
    if len(df) > 0:
        all_results.append(df)
        print(f"  {name}: {len(df)} genes validated")

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{RES_DIR}/batch_validation_summary.csv", index=False)
    print(f"\n  Combined: {len(combined)} gene-dataset pairs")

    # Cross-dataset consistency: how many genes significant in multiple datasets?
    gene_sig_count = {}
    for gene in CANDIDATES:
        gene_rows = combined[combined['gene'] == gene]
        n_sig = 0
        for _, r in gene_rows.iterrows():
            for col in r.index:
                if col.endswith('_vs_Normal_p') and r[col] < 0.05:
                    n_sig += 1
                    break
        gene_sig_count[gene] = n_sig

    sig_series = pd.Series(gene_sig_count)
    print(f"\n  Genes significant in multiple new datasets:")
    print(f"    3/3 datasets: {(sig_series >= 3).sum()}")
    print(f"    2/3 datasets: {(sig_series >= 2).sum()}")
    print(f"    1/3 datasets: {(sig_series >= 1).sum()}")
    print(f"    0 datasets: {(sig_series == 0).sum()}")

    # Top genes by cross-dataset significance
    top_validated = sig_series[sig_series >= 2].index.tolist()
    if top_validated:
        print(f"\n  Genes validated in ≥2 new datasets: {top_validated[:20]}")

print("\nDone!")
