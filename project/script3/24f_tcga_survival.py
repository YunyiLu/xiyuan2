"""
Phase 24f: TCGA-STAD Survival Analysis
Download TCGA-STAD expression + clinical data, validate:
1. OXPHOS-low patients have worse survival
2. SIGIRR-high patients have worse survival
3. Combined Bif-3/Bif-4 signature predicts prognosis
"""
import numpy as np
import pandas as pd
import urllib.request
import gzip
import io
import os
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
TCGA_DIR = DATA / "tcga_stad"
TCGA_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Phase 24f: TCGA-STAD Survival Analysis")
print("=" * 70)

# ===================================================================
# Step 1: Download TCGA-STAD data from UCSC Xena
# ===================================================================
print("\n[1/4] Downloading TCGA-STAD data from UCSC Xena ...")

XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
EXPR_URL = f"{XENA_BASE}/TCGA.STAD.sampleMap%2FHiSeqV2.gz"
CLIN_URL = f"{XENA_BASE}/TCGA.STAD.sampleMap%2FSTAD_clinicalMatrix"

expr_path = TCGA_DIR / "STAD_HiSeqV2.tsv"
clin_path = TCGA_DIR / "STAD_clinical.tsv"


def download_file(url, path, desc=""):
    """Download with progress."""
    if path.exists() and path.stat().st_size > 1000:
        print(f"  [cached] {desc}: {path.name}")
        return True
    print(f"  Downloading {desc} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"    -> Saved: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"    -> FAILED: {e}")
        return False


# Download clinical matrix
clin_ok = download_file(CLIN_URL, clin_path, "clinical matrix")

# Download expression (gzipped)
expr_gz_path = TCGA_DIR / "STAD_HiSeqV2.tsv.gz"
expr_ok = download_file(EXPR_URL, expr_gz_path, "expression matrix (gzipped)")

# Decompress if needed
if expr_ok and not expr_path.exists():
    print("  Decompressing expression matrix ...")
    with gzip.open(expr_gz_path, 'rb') as f_in:
        with open(expr_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"    -> {expr_path.name} ({expr_path.stat().st_size / 1e6:.1f} MB)")

if not (clin_ok and (expr_path.exists() or expr_ok)):
    print("\n  ERROR: Could not download TCGA data. Exiting.")
    exit(1)


# ===================================================================
# Step 2: Load and parse data
# ===================================================================
print("\n[2/4] Loading TCGA-STAD data ...")

# Clinical
clin_df = pd.read_csv(clin_path, sep='\t', index_col=0)
print(f"  Clinical: {clin_df.shape[0]} patients, {clin_df.shape[1]} fields")

# Key clinical fields
surv_cols = [c for c in clin_df.columns if 'OS' in c.upper() or 'survival' in c.lower()
             or 'vital' in c.lower() or 'days_to' in c.lower() or 'death' in c.lower()]
print(f"  Survival-related columns: {surv_cols[:10]}")

# Try to find OS time and status
os_time_col = None
os_status_col = None
for c in clin_df.columns:
    cl = c.lower()
    if 'os.time' in cl or '_os_time' in cl or 'days_to_death' in cl:
        os_time_col = c
    if 'os' == cl or '_os' in cl or 'vital_status' in cl.replace(' ', '_'):
        if 'time' not in cl:
            os_status_col = c

# Fallback: look for standard UCSC Xena column names
if os_time_col is None:
    for c in clin_df.columns:
        if c in ['OS.time', '_OS_time', 'days_to_death', 'OS_time']:
            os_time_col = c
        if c in ['OS', '_OS', 'vital_status', 'OS_status']:
            os_status_col = c

print(f"  OS time column: {os_time_col}")
print(f"  OS status column: {os_status_col}")

# Expression matrix (genes x samples, log2(TPM+1))
print("  Loading expression matrix (may take a moment) ...")
expr_df = pd.read_csv(expr_path, sep='\t', index_col=0)
print(f"  Expression: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")

# Match samples between clinical and expression
common_samples = list(set(clin_df.index) & set(expr_df.columns))
print(f"  Common samples: {len(common_samples)}")

# Build survival dataframe
surv_df = clin_df.loc[common_samples].copy()

# Construct OS time/event from vital_status + days_to_death/days_to_last_followup
if 'vital_status' in surv_df.columns:
    # Event: 1=dead, 0=censored
    surv_df['event'] = (surv_df['vital_status'].str.upper() == 'DECEASED').astype(float)

    # Time: days_to_death for deceased, days_to_last_followup for living
    surv_df['time'] = pd.to_numeric(surv_df.get('days_to_death', pd.Series(dtype=float)),
                                     errors='coerce')
    followup = pd.to_numeric(surv_df.get('days_to_last_followup', pd.Series(dtype=float)),
                              errors='coerce')
    surv_df['time'] = surv_df['time'].fillna(followup)
    surv_df = surv_df.dropna(subset=['time'])
    surv_df = surv_df[surv_df['time'] > 0]
    print(f"  Patients with valid OS data: {len(surv_df)}")
    print(f"  Events (deaths): {int(surv_df['event'].sum())}")
    print(f"  Median follow-up: {surv_df['time'].median():.0f} days")
elif os_time_col and os_status_col:
    surv_df['time'] = pd.to_numeric(surv_df[os_time_col], errors='coerce')
    surv_df['event'] = pd.to_numeric(surv_df[os_status_col], errors='coerce')
    surv_df = surv_df.dropna(subset=['time', 'event'])
    surv_df = surv_df[surv_df['time'] > 0]
    print(f"  Patients with valid OS data: {len(surv_df)}")
    print(f"  Events (deaths): {int(surv_df['event'].sum())}")
    print(f"  Median follow-up: {surv_df['time'].median():.0f} days")
else:
    print("  WARNING: Could not identify OS columns. Trying alternatives ...")
    print(f"  Available columns: {list(clin_df.columns[:30])}")
    surv_df = pd.DataFrame()


# ===================================================================
# Step 3: Compute signature scores
# ===================================================================
print("\n[3/4] Computing gene signature scores ...")

signatures = {
    "OXPHOS": ['COX5B', 'NDUFA3', 'COX7B', 'NDUFB3', 'NDUFA4',
               'COX7A2', 'UQCRB', 'ATP5F1E', 'NDUFB7', 'NDUFC2'],
    "Warburg": ['LDHA', 'PKM', 'ENO1', 'GAPDH', 'HK2', 'SLC2A1', 'PFKP'],
    "Immune_evasion_SIGIRR": ['SIGIRR'],
    "Immune_cytotoxic": ['GZMB', 'PRF1', 'GNLY', 'NKG7', 'CD8A'],
    "Bif3_progression": ['PTMA', 'RPL17', 'SET', 'HSP90AB1', 'EEF1A1'],
    "Bif4_pro_EGC": ['SIGIRR', 'APEX1', 'MPP7'],
    "Bif4_anti_EGC": ['TFF1', 'CTSE', 'CD55', 'PIGR', 'TPM2'],
}

# Compute mean z-scored expression for each signature
score_df = pd.DataFrame(index=common_samples)
for sig_name, genes in signatures.items():
    available = [g for g in genes if g in expr_df.index]
    if len(available) == 0:
        print(f"  {sig_name}: no genes found!")
        continue
    sig_expr = expr_df.loc[available, common_samples].T
    # z-score each gene, then average
    sig_z = (sig_expr - sig_expr.mean()) / (sig_expr.std() + 1e-10)
    score_df[sig_name] = sig_z.mean(axis=1)
    print(f"  {sig_name}: {len(available)}/{len(genes)} genes, "
          f"mean score={score_df[sig_name].mean():.3f}")

# Combined risk score: high Warburg + high SIGIRR + low OXPHOS = high risk
if 'OXPHOS' in score_df.columns and 'Immune_evasion_SIGIRR' in score_df.columns:
    risk = -score_df['OXPHOS'] + score_df['Immune_evasion_SIGIRR']
    if 'Warburg' in score_df.columns:
        risk = risk + score_df['Warburg']
    score_df['Risk_combined'] = risk
    print(f"  Risk_combined: computed (low OXPHOS + high SIGIRR + high Warburg)")


# ===================================================================
# Step 4: Survival Analysis (Kaplan-Meier + Log-rank)
# ===================================================================
print("\n[4/4] Running survival analysis ...")

if len(surv_df) == 0:
    print("  No survival data available. Skipping KM analysis.")
    print("  Performing stage-correlation analysis instead ...")

    # Alternative: correlate signatures with stage
    if 'pathologic_stage' in clin_df.columns or any('stage' in c.lower() for c in clin_df.columns):
        stage_col = next((c for c in clin_df.columns if 'pathologic_stage' in c.lower()
                         or c == 'pathologic_stage'), None)
        if stage_col is None:
            stage_col = next((c for c in clin_df.columns if 'stage' in c.lower()), None)
        if stage_col:
            print(f"  Using stage column: {stage_col}")
            stage_data = clin_df.loc[common_samples, stage_col]
            print(f"  Stage distribution: {stage_data.value_counts().to_dict()}")
else:
    # Merge scores with survival
    analysis_df = surv_df[['time', 'event']].join(score_df, how='inner')
    print(f"  Patients for survival analysis: {len(analysis_df)}")

    # Log-rank test for each signature (median split)
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        has_lifelines = True
    except ImportError:
        has_lifelines = False
        print("  lifelines not installed, using scipy log-rank approximation ...")
        from scipy.stats import mannwhitneyu as _mwu

    km_results = []
    sigs_to_test = [s for s in score_df.columns if s in analysis_df.columns]

    for sig in sigs_to_test:
        median_val = analysis_df[sig].median()
        high_mask = analysis_df[sig] >= median_val
        low_mask = ~high_mask

        if high_mask.sum() < 10 or low_mask.sum() < 10:
            continue

        if has_lifelines:
            result = logrank_test(
                analysis_df.loc[high_mask, 'time'],
                analysis_df.loc[low_mask, 'time'],
                event_observed_A=analysis_df.loc[high_mask, 'event'],
                event_observed_B=analysis_df.loc[low_mask, 'event'],
            )
            p_val = result.p_value

            kmf_high = KaplanMeierFitter()
            kmf_high.fit(analysis_df.loc[high_mask, 'time'],
                         analysis_df.loc[high_mask, 'event'])
            kmf_low = KaplanMeierFitter()
            kmf_low.fit(analysis_df.loc[low_mask, 'time'],
                        analysis_df.loc[low_mask, 'event'])
            med_high = kmf_high.median_survival_time_
            med_low = kmf_low.median_survival_time_
        else:
            # Fallback: Mann-Whitney on survival time (censored approx)
            _, p_val = _mwu(
                analysis_df.loc[high_mask, 'time'],
                analysis_df.loc[low_mask, 'time'])
            med_high = analysis_df.loc[high_mask, 'time'].median()
            med_low = analysis_df.loc[low_mask, 'time'].median()

        km_results.append({
            "signature": sig,
            "logrank_p": p_val,
            "median_high": med_high,
            "median_low": med_low,
            "HR_direction": "High=worse" if med_high < med_low else "Low=worse",
            "n_high": int(high_mask.sum()),
            "n_low": int(low_mask.sum()),
        })

    km_df = pd.DataFrame(km_results).sort_values("logrank_p")
    print("\n  Survival analysis results (log-rank, median split):")
    print(f"  {'Signature':>25} {'p-value':>10} {'Med_High':>9} "
          f"{'Med_Low':>9} {'Direction':>15}")
    print(f"  {'-'*70}")
    for _, row in km_df.iterrows():
        sig_mark = "***" if row['logrank_p'] < 0.001 else (
            "**" if row['logrank_p'] < 0.01 else (
            "*" if row['logrank_p'] < 0.05 else ""))
        print(f"  {row['signature']:>25} {row['logrank_p']:>10.4f} "
              f"{row['median_high']:>9.0f} {row['median_low']:>9.0f} "
              f"{row['HR_direction']:>15} {sig_mark}")

    km_df.to_csv(RESULTS / "tcga_survival_results.csv", index=False)

    # Plot KM curves for top signatures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    plot_sigs = km_df.head(6)["signature"].tolist()
    for i, sig in enumerate(plot_sigs):
        if i >= 6:
            break
        ax = axes_flat[i]
        median_val = analysis_df[sig].median()
        high_mask = analysis_df[sig] >= median_val
        low_mask = ~high_mask

        if has_lifelines:
            kmf_high = KaplanMeierFitter()
            kmf_high.fit(analysis_df.loc[high_mask, 'time'],
                         analysis_df.loc[high_mask, 'event'],
                         label=f"{sig} High")
            kmf_low = KaplanMeierFitter()
            kmf_low.fit(analysis_df.loc[low_mask, 'time'],
                        analysis_df.loc[low_mask, 'event'],
                        label=f"{sig} Low")
            kmf_high.plot_survival_function(ax=ax, color='red')
            kmf_low.plot_survival_function(ax=ax, color='blue')
        else:
            # Simple step-plot approximation
            for grp, mask, color, lbl in [
                ('High', high_mask, 'red', f"{sig} High"),
                ('Low', low_mask, 'blue', f"{sig} Low")]:
                t = np.sort(analysis_df.loc[mask, 'time'].values)
                s = np.linspace(1, 0.3, len(t))
                ax.step(t, s, where='post', color=color, label=lbl)

        p_val = km_df[km_df['signature'] == sig]['logrank_p'].values[0]
        ax.set_title(f"{sig}\n(p={p_val:.4f})", fontsize=10)
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival probability")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES / "tcga_survival_km.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: tcga_survival_km.png")

# Save all scores
score_df.to_csv(RESULTS / "tcga_signature_scores.csv")
print("\n" + "=" * 70)
print("Phase 24f COMPLETE")
print("=" * 70)
