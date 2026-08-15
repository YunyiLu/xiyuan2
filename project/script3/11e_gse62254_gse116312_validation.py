"""
Step 11e: GSE62254 (ACRG, 300 Korean GC, GPL570) + GSE116312 (GPL6244)
  Uses existing GPL570 and GPL6244 annotation files.
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

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = set(candidates_df['gene'].tolist())
print(f"Loaded {len(CANDIDATES)} candidates")

# Load GPL570 (already downloaded in 11d)
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

# Load GPL6244 (already downloaded from 11b)
gpl6244_dest = f"{DATA_DIR}/GPL6244.annot.gz"
probe6244 = {}
if os.path.exists(gpl6244_dest):
    in_table = False
    with gzip.open(gpl6244_dest, 'rt', errors='replace') as f:
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
                        probe6244[probe_id] = g
    print(f"GPL6244: {len(probe6244)} probes -> {len(set(probe6244.values()))} genes")


# ===== GSE62254 — ACRG 300 Korean GC (GPL570) =====
print("\n" + "="*70)
print("GSE62254 — ACRG 300 Korean GC (GPL570)")
print("="*70)

dest_acrg = f"{DATA_DIR}/GSE62254_series_matrix.txt.gz"
if not os.path.exists(dest_acrg) or os.path.getsize(dest_acrg) < 10000:
    print("Downloading GSE62254...")
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62254/matrix/GSE62254_series_matrix.txt.gz"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Accept-Encoding', 'identity')
    resp = urllib.request.urlopen(req, timeout=600, context=ctx)
    with open(dest_acrg, 'wb') as f:
        while True:
            chunk = resp.read(512*1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  {os.path.getsize(dest_acrg)/1024/1024:.1f} MB")
else:
    print(f"  Exists: {os.path.getsize(dest_acrg)/1024/1024:.1f} MB")

# Parse — ACRG is tumor-only with molecular subtypes
data_rows = []
header_samples = []
sample_ids = []
sample_titles = {}
sample_chars = {}

with gzip.open(dest_acrg, 'rt', errors='replace') as f:
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

# Classify subtypes from characteristics
subtypes = {}
for sid in sample_ids:
    chars_list = sample_chars.get(sid, [])
    chars = ' '.join(chars_list).lower()
    title = sample_titles.get(sid, '').lower()
    combined = chars + ' ' + title
    if 'msi' in combined or 'microsatellite instab' in combined:
        subtypes[sid] = 'MSI'
    elif 'mss/emt' in combined or 'mesenchymal' in combined or 'emt' in combined:
        subtypes[sid] = 'MSS/EMT'
    elif 'mss/tp53+' in combined or 'tp53 active' in combined or 'tp53+' in combined:
        subtypes[sid] = 'MSS/TP53+'
    elif 'mss/tp53-' in combined or 'tp53 inactive' in combined or 'tp53-' in combined or 'tp53 loss' in combined:
        subtypes[sid] = 'MSS/TP53-'
    elif 'intestinal' in combined:
        subtypes[sid] = 'Intestinal'
    elif 'diffuse' in combined:
        subtypes[sid] = 'Diffuse'
    else:
        subtypes[sid] = 'Unknown'

print(f"Subtypes: {pd.Series(subtypes).value_counts().to_dict()}")
if sample_chars:
    first_sid = list(sample_chars.keys())[0]
    print(f"Example chars: {sample_chars[first_sid]}")
print(f"Titles sample: {list(sample_titles.values())[:3]}")

if data_rows:
    expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
    expr_df = expr_df.groupby('gene').mean()
    print(f"Matrix: {expr_df.shape}")

    # For tumor-only: expression stats + subtype differences
    # MSI subtype is intestinal-type enriched — compare MSI vs MSS/EMT
    msi_samps = [s for s in expr_df.columns if subtypes.get(s) == 'MSI']
    emt_samps = [s for s in expr_df.columns if subtypes.get(s) == 'MSS/EMT']
    tp53n_samps = [s for s in expr_df.columns if subtypes.get(s) == 'MSS/TP53-']

    print(f"MSI: {len(msi_samps)}, MSS/EMT: {len(emt_samps)}, MSS/TP53-: {len(tp53n_samps)}")

    results = []
    for gene in expr_df.index:
        row = {'gene': gene, 'dataset': 'GSE62254_ACRG'}
        row['mean_all'] = expr_df.loc[gene].mean()
        row['std_all'] = expr_df.loc[gene].std()

        if msi_samps and emt_samps:
            msi_vals = expr_df.loc[gene, msi_samps].dropna().values
            emt_vals = expr_df.loc[gene, emt_samps].dropna().values
            if len(msi_vals) >= 5 and len(emt_vals) >= 5:
                _, p = mannwhitneyu(msi_vals, emt_vals, alternative='two-sided')
                fc = np.log2((np.mean(msi_vals)+0.01)/(np.mean(emt_vals)+0.01))
                row['MSI_vs_EMT_p'] = p
                row['MSI_vs_EMT_logFC'] = fc
                row['mean_MSI'] = np.mean(msi_vals)
                row['mean_EMT'] = np.mean(emt_vals)

        results.append(row)

    res_df = pd.DataFrame(results)
    if 'MSI_vs_EMT_p' in res_df.columns:
        valid_p = res_df['MSI_vs_EMT_p'].dropna()
        if len(valid_p) > 1:
            _, fdr, _, _ = multipletests(valid_p, method='fdr_bh')
            res_df.loc[valid_p.index, 'MSI_vs_EMT_fdr'] = fdr

        res_df = res_df.sort_values('MSI_vs_EMT_p')
        res_df.to_csv(f"{RES_DIR}/gse62254_acrg_validation.csv", index=False)

        if 'MSI_vs_EMT_fdr' in res_df.columns:
            sig = (res_df['MSI_vs_EMT_fdr'] < 0.05).sum()
            up_msi = ((res_df['MSI_vs_EMT_fdr'] < 0.05) & (res_df['MSI_vs_EMT_logFC'] > 0)).sum()
            print(f"\nMSI vs EMT: {sig} significant")
            print(f"  Up in MSI (intestinal): {up_msi}")

            print("\nTop MSI-enriched (intestinal-type):")
            top = res_df[(res_df['MSI_vs_EMT_fdr'] < 0.05) & (res_df['MSI_vs_EMT_logFC'] > 0)].nlargest(10, 'MSI_vs_EMT_logFC')
            for _, r in top.iterrows():
                print(f"  {r['gene']}: logFC={r['MSI_vs_EMT_logFC']:.3f}, FDR={r['MSI_vs_EMT_fdr']:.2e}")
    else:
        # No subtype comparison possible — just save expression stats
        res_df['mean_expr'] = [expr_df.loc[g].mean() for g in res_df['gene']]
        res_df['std_expr'] = [expr_df.loc[g].std() for g in res_df['gene']]
        res_df = res_df.sort_values('mean_expr', ascending=False)
        res_df.to_csv(f"{RES_DIR}/gse62254_acrg_validation.csv", index=False)
        print(f"\nSaved expression stats for {len(res_df)} genes (no subtype comparison)")
        print("Top expressed in GC tumors:")
        for _, r in res_df.head(10).iterrows():
            print(f"  {r['gene']}: mean={r['mean_expr']:.2f}")


# ===== GSE116312 with GPL6244 annotation =====
print("\n" + "="*70)
print("GSE116312 — Precancerous Lesions (GPL6244, 13 samples)")
print("="*70)

dest116 = f"{DATA_DIR}/GSE116312_series_matrix.txt.gz"
if os.path.exists(dest116) and probe6244:
    data_rows2 = []
    header_samples2 = []
    sample_ids2 = []
    sample_titles2 = {}
    sample_chars2 = {}

    with gzip.open(dest116, 'rt', errors='replace') as f:
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
            elif line.startswith('!Sample_characteristics'):
                parts = line.strip().split('\t')
                for i, p in enumerate(parts[1:]):
                    if i < len(sample_ids2):
                        sample_chars2.setdefault(sample_ids2[i], []).append(p.strip('"'))
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
                if probe in probe6244:
                    gene = probe6244[probe]
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    data_rows2.append([gene] + values)

    print(f"Samples: {len(sample_ids2)}")
    print(f"Genes matched: {len(set(r[0] for r in data_rows2))}")
    print(f"Titles: {list(sample_titles2.values())}")

    stages2 = {}
    for sid in sample_ids2:
        title = sample_titles2.get(sid, '').lower()
        chars = ' '.join(sample_chars2.get(sid, [])).lower()
        combined = title + ' ' + chars
        if 'follicular' in combined or 'normal' in combined:
            stages2[sid] = 'Normal'
        elif 'atrophic' in combined or 'cag' in combined:
            stages2[sid] = 'Atrophic'
        elif 'cancer' in combined or 'gc' in combined or 'carcinoma' in combined:
            stages2[sid] = 'Cancer'
        else:
            stages2[sid] = 'Unknown'

    print(f"Stages: {pd.Series(stages2).value_counts().to_dict()}")

    if data_rows2:
        expr_df2 = pd.DataFrame(data_rows2, columns=['gene'] + header_samples2)
        expr_df2 = expr_df2.groupby('gene').mean()

        normal2 = [s for s in expr_df2.columns if stages2.get(s) == 'Normal']
        cancer2 = [s for s in expr_df2.columns if stages2.get(s) == 'Cancer']
        atrophic2 = [s for s in expr_df2.columns if stages2.get(s) == 'Atrophic']

        print(f"Normal: {len(normal2)}, Atrophic: {len(atrophic2)}, Cancer: {len(cancer2)}")

        results2 = []
        for gene in expr_df2.index:
            row = {'gene': gene, 'dataset': 'GSE116312'}
            if len(normal2) >= 3 and len(cancer2) >= 3:
                n_vals = expr_df2.loc[gene, normal2].dropna().values
                c_vals = expr_df2.loc[gene, cancer2].dropna().values
                if len(n_vals) >= 3 and len(c_vals) >= 3:
                    _, p = mannwhitneyu(c_vals, n_vals, alternative='two-sided')
                    fc = np.log2((np.mean(c_vals)+0.01)/(np.mean(n_vals)+0.01))
                    row['cancer_vs_normal_p'] = p
                    row['cancer_vs_normal_logFC'] = fc
            results2.append(row)

        res_df2 = pd.DataFrame(results2)
        if 'cancer_vs_normal_p' in res_df2.columns:
            valid_p = res_df2['cancer_vs_normal_p'].dropna()
            if len(valid_p) > 1:
                _, fdr, _, _ = multipletests(valid_p, method='fdr_bh')
                res_df2.loc[valid_p.index, 'fdr'] = fdr
            res_df2.to_csv(f"{RES_DIR}/gse116312_validation.csv", index=False)
            sig2 = (res_df2['fdr'] < 0.05).sum() if 'fdr' in res_df2.columns else 0
            print(f"\nResults: {len(valid_p)} genes tested, {sig2} significant")
else:
    print("  Data or annotation not available")

print("\nAll done!")
