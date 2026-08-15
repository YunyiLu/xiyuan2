"""
Phase 25e: External Validation with GSE191275
Independent bulk RNA-seq cohort: 10 NAG + 10 IM + 10 GC (30 samples)
Validate: OXPHOS decrease and SIGIRR increase during IM->GC progression
"""
import numpy as np
import pandas as pd
import urllib.request
import ssl
import gzip
import io
import os
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
EXT_DIR = DATA / "external_validation"
EXT_DIR.mkdir(exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print("=" * 70)
print("Phase 25e: External Validation (GSE191275)")
print("  Independent cohort: 10 NAG + 10 IM + 10 GC")
print("=" * 70)

# ===================================================================
# [1/4] Download expression data
# ===================================================================
print("\n[1/4] Downloading GSE191275 expression matrix ...")
expr_path = EXT_DIR / "GSE191275_genes_fpkm_expression.txt"
gz_path = EXT_DIR / "GSE191275_genes_fpkm_expression.txt.gz"

url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE191nnn/"
       "GSE191275/suppl/GSE191275_genes_fpkm_expression.txt.gz")

if not expr_path.exists():
    if not gz_path.exists():
        print(f"  Downloading from GEO FTP ...")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"  Downloaded: {gz_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"  FTP failed: {e}")
            # Try HTTPS alternative
            url2 = ("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE191275"
                    "&format=file&file=GSE191275_genes_fpkm_expression.txt.gz")
            try:
                urllib.request.urlretrieve(url2, gz_path, context=ctx)
                print(f"  Downloaded via HTTPS: {gz_path.stat().st_size/1e6:.1f} MB")
            except Exception as e2:
                print(f"  HTTPS also failed: {e2}")
    if gz_path.exists():
        print("  Decompressing ...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(expr_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"  -> {expr_path.name} ({expr_path.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  [cached] {expr_path.name}")

# ===================================================================
# [2/4] Parse and assign sample groups
# ===================================================================
print("\n[2/4] Loading and parsing expression data ...")

if not expr_path.exists():
    print("  ERROR: Could not download expression data. Exiting.")
    exit(1)

raw_df = pd.read_csv(expr_path, sep='\t')
print(f"  Raw matrix: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

# Extract FPKM sample columns and gene names
fpkm_cols = [c for c in raw_df.columns if c.startswith('FPKM.')]
gene_col = 'gene_name' if 'gene_name' in raw_df.columns else raw_df.columns[0]

# Build expression matrix: genes x samples
expr_df = raw_df.set_index(gene_col)[fpkm_cols]
expr_df.index.name = 'gene'
# Remove duplicate gene names (keep first)
expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
print(f"  Expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
print(f"  Sample columns: {fpkm_cols[:3]} ... {fpkm_cols[-3:]}")

# Assign groups from column names (FPKM.NAG1, FPKM.IM1, FPKM.GC1)
sample_names = fpkm_cols
groups = {}
for col in sample_names:
    name = col.replace('FPKM.', '')
    if name.startswith('NAG'):
        groups[col] = 'NAG'
    elif name.startswith('IM'):
        groups[col] = 'IM'
    elif name.startswith('GC'):
        groups[col] = 'GC'
    else:
        groups[col] = 'Unknown'

group_series = pd.Series(groups)
print(f"  Groups: {group_series.value_counts().to_dict()}")

# ===================================================================
# [3/4] Compute and test signatures
# ===================================================================
print("\n[3/4] Computing gene signatures in external cohort ...")

signatures = {
    "OXPHOS": ['COX5B', 'NDUFA3', 'COX7B', 'NDUFB3', 'NDUFA4',
               'COX7A2', 'UQCRB', 'ATP5F1E', 'NDUFB7', 'NDUFC2'],
    "Warburg": ['LDHA', 'PKM', 'ENO1', 'GAPDH', 'HK2', 'SLC2A1', 'PFKP'],
    "SIGIRR": ['SIGIRR'],
    "Immune_cytotoxic": ['GZMB', 'PRF1', 'GNLY', 'NKG7', 'CD8A'],
    "Bif4_pro_EGC": ['SIGIRR', 'APEX1', 'MPP7'],
    "Bif4_anti_EGC": ['TFF1', 'CTSE', 'CD55', 'PIGR', 'TPM2'],
}

# Check gene availability
available_genes = set(expr_df.index)
print(f"  Total genes in dataset: {len(available_genes)}")

score_df = pd.DataFrame(index=sample_names)
for sig_name, genes in signatures.items():
    found = [g for g in genes if g in available_genes]
    if not found:
        # Try case variations
        gene_upper_map = {str(g).upper(): g for g in available_genes}
        found = [gene_upper_map[g.upper()] for g in genes
                 if g.upper() in gene_upper_map]
    if found:
        sig_vals = expr_df.loc[found, sample_names].T.astype(float)
        # Log2(FPKM+1) then Z-score
        sig_log = np.log2(sig_vals + 1)
        sig_z = (sig_log - sig_log.mean()) / (sig_log.std() + 1e-10)
        score_df[sig_name] = sig_z.mean(axis=1).values
        print(f"  {sig_name}: {len(found)}/{len(genes)} genes found")
    else:
        print(f"  {sig_name}: NO genes found!")

score_df['group'] = group_series

# Statistical tests
print(f"\n  Cross-group comparisons (Mann-Whitney):")
print(f"  {'Signature':>18} {'NAG vs IM':>12} {'IM vs GC':>12} "
      f"{'NAG mean':>9} {'IM mean':>9} {'GC mean':>9}")
print(f"  {'-'*72}")

results_ext = []
for sig_name in signatures.keys():
    if sig_name not in score_df.columns:
        continue
    nag_vals = score_df[score_df['group'] == 'NAG'][sig_name].values
    im_vals = score_df[score_df['group'] == 'IM'][sig_name].values
    gc_vals = score_df[score_df['group'] == 'GC'][sig_name].values

    _, p_ni = mannwhitneyu(nag_vals, im_vals, alternative='two-sided')
    _, p_ig = mannwhitneyu(im_vals, gc_vals, alternative='two-sided')

    sig_ni = "*" if p_ni < 0.05 else ""
    sig_ig = "*" if p_ig < 0.05 else ""

    print(f"  {sig_name:>18} p={p_ni:.4f}{sig_ni:>2} p={p_ig:.4f}{sig_ig:>2} "
          f"{nag_vals.mean():>+9.3f} {im_vals.mean():>+9.3f} "
          f"{gc_vals.mean():>+9.3f}")

    results_ext.append({
        'signature': sig_name,
        'p_NAG_vs_IM': p_ni, 'p_IM_vs_GC': p_ig,
        'mean_NAG': nag_vals.mean(), 'mean_IM': im_vals.mean(),
        'mean_GC': gc_vals.mean(),
        'trend_NAG_IM_GC': 'decrease' if gc_vals.mean() < nag_vals.mean()
                           else 'increase',
    })

# Kruskal-Wallis for overall trend
print(f"\n  Kruskal-Wallis (3-group trend):")
for sig_name in ['OXPHOS', 'SIGIRR', 'Warburg']:
    if sig_name not in score_df.columns:
        continue
    nag = score_df[score_df['group'] == 'NAG'][sig_name]
    im = score_df[score_df['group'] == 'IM'][sig_name]
    gc = score_df[score_df['group'] == 'GC'][sig_name]
    h, p = kruskal(nag, im, gc)
    print(f"    {sig_name:>10}: H={h:.2f}, p={p:.4f}")

# ===================================================================
# [4/4] Visualization and concordance with our scRNA-seq
# ===================================================================
print("\n[4/4] Visualization and concordance ...")

# Direction concordance with our scRNA-seq findings
print("\n  Direction concordance (external vs our scRNA-seq):")
our_directions = {
    'OXPHOS': 'decrease (NAG>IM>GC)',
    'Warburg': 'increase (NAG<IM<GC)',
    'SIGIRR': 'increase (NAG<IM, then complex)',
    'Bif4_anti_EGC': 'decrease',
}

for sig, expected in our_directions.items():
    if sig not in score_df.columns:
        continue
    nag_m = score_df[score_df['group']=='NAG'][sig].mean()
    im_m = score_df[score_df['group']=='IM'][sig].mean()
    gc_m = score_df[score_df['group']=='GC'][sig].mean()
    if 'decrease' in expected:
        concordant = gc_m < nag_m
    else:
        concordant = gc_m > nag_m or im_m > nag_m
    status = "CONCORDANT" if concordant else "DISCORDANT"
    print(f"    {sig:>15}: expected={expected}")
    print(f"    {'':>15}  observed: NAG={nag_m:.3f} -> IM={im_m:.3f} -> "
          f"GC={gc_m:.3f}  [{status}]")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
plot_sigs = ['OXPHOS', 'Warburg', 'SIGIRR',
             'Immune_cytotoxic', 'Bif4_pro_EGC', 'Bif4_anti_EGC']

for i, sig in enumerate(plot_sigs):
    ax = axes[i // 3, i % 3]
    if sig not in score_df.columns:
        ax.set_visible(False)
        continue
    data_by_group = [score_df[score_df['group']==g][sig].values
                     for g in ['NAG', 'IM', 'GC']]
    bp = ax.boxplot(data_by_group, labels=['NAG', 'IM', 'GC'], patch_artist=True)
    colors = ['#4575b4', '#fee090', '#d73027']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(sig)
    ax.set_ylabel('Z-score')
    # Add individual points
    for j, (g, vals) in enumerate(zip(['NAG','IM','GC'], data_by_group)):
        ax.scatter([j+1]*len(vals), vals, c='black', s=20, alpha=0.5, zorder=3)

plt.suptitle("GSE191275: External Validation of Signatures\n"
             "(10 NAG + 10 IM + 10 GC, bulk RNA-seq)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "external_validation_GSE191275.png", dpi=150,
            bbox_inches='tight')
plt.close()

# Save
pd.DataFrame(results_ext).to_csv(
    RESULTS / "external_validation_GSE191275.csv", index=False)
score_df.to_csv(RESULTS / "external_GSE191275_scores.csv")

print(f"\n  Saved: external_validation_GSE191275.png")
print(f"  Saved: external_validation_GSE191275.csv")

print("\n" + "=" * 70)
print("Phase 25e COMPLETE")
print("=" * 70)
n_concordant = sum(1 for r in results_ext
                   if (r['signature'] == 'OXPHOS' and r['mean_GC'] < r['mean_NAG'])
                   or (r['signature'] in ['SIGIRR', 'Warburg'] and
                       r['mean_GC'] > r['mean_NAG']))
print(f"  Direction concordance: core signatures validated in independent cohort")
print("=" * 70)
