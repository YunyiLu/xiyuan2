"""
Step 13A: GSE27342 Cancer Endpoint Validation
80 paired gastric cancer + adjacent normal tissues (Affymetrix Human Exon 1.0 ST, GPL5175)

Purpose: Validate candidate genes show differential expression in GC vs Normal
(Level 5 evidence: cancer endpoint, not progression)

Input: GSE27342_family.soft.gz (downloaded from GEO)
Output: results/gse27342_cancer_validation.csv
        figures/gse27342_paired_expression.png
"""
import os, sys, gzip, re, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DATA_DIR = f"{BASE}/data/gse27342"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{BASE}/figures", exist_ok=True)
os.makedirs(f"{BASE}/results", exist_ok=True)

CANDIDATES = ["OLFM4", "REG4", "ITLN1", "PRAP1", "ANPEP", "PSCA", "FABP1", "CPS1", "MUC13", "CLDN4"]

SOFT_PATH = f"{DATA_DIR}/GSE27342_family.soft.gz"


def parse_gene_assignment(gene_assignment_str):
    """Parse Affymetrix gene_assignment field to extract gene symbol.
    Format: 'NR_024005 // DDX11L2 // description // cytoband // entrezid /// ...'
    Returns first gene symbol found.
    """
    if not gene_assignment_str or gene_assignment_str == '---':
        return ''
    parts = gene_assignment_str.split('///')
    for part in parts:
        fields = [f.strip() for f in part.split('//')]
        if len(fields) >= 2:
            gene_sym = fields[1].strip()
            if gene_sym and gene_sym != '---':
                return gene_sym
    return ''


def parse_soft_file():
    """Parse the GSE27342 SOFT file to extract platform, samples, and expression."""
    cache_expr = f"{DATA_DIR}/expression_gene_level.csv"
    cache_meta = f"{DATA_DIR}/metadata.csv"

    if os.path.exists(cache_expr) and os.path.exists(cache_meta):
        expr = pd.read_csv(cache_expr, index_col=0)
        meta = pd.read_csv(cache_meta)
        if len(expr) > 100 and len(meta) > 0:
            print(f"  [CACHED] {expr.shape[0]} genes × {expr.shape[1]} samples")
            return expr, meta

    print("  Parsing SOFT file (this may take a minute)...")

    # Phase 1: Parse platform table (probe ID → gene)
    probe2gene = {}
    platform_parsed = False

    with gzip.open(SOFT_PATH, 'rt', encoding='utf-8', errors='replace') as f:
        in_platform_table = False
        for line in f:
            if line.startswith('!platform_table_begin'):
                header = next(f).strip().split('\t')
                # Find gene_assignment column
                gene_col_idx = -1
                for i, col in enumerate(header):
                    if 'gene_assignment' in col.lower():
                        gene_col_idx = i
                        break
                if gene_col_idx == -1:
                    for i, col in enumerate(header):
                        if 'symbol' in col.lower() or 'gene' in col.lower():
                            gene_col_idx = i
                            break
                print(f"  Platform gene column: [{gene_col_idx}] {header[gene_col_idx] if gene_col_idx >= 0 else 'NOT FOUND'}")
                in_platform_table = True
                continue

            if line.startswith('!platform_table_end'):
                in_platform_table = False
                platform_parsed = True
                continue

            if in_platform_table:
                parts = line.strip().split('\t')
                if len(parts) > gene_col_idx and gene_col_idx >= 0:
                    probe_id = parts[0]
                    gene_sym = parse_gene_assignment(parts[gene_col_idx])
                    if gene_sym:
                        probe2gene[probe_id] = gene_sym

            if platform_parsed and line.startswith('^SAMPLE'):
                break

    print(f"  Platform probes → genes: {len(probe2gene)}")

    # Phase 2: Parse sample metadata and expression
    samples_meta = []
    samples_expr = {}

    with gzip.open(SOFT_PATH, 'rt', encoding='utf-8', errors='replace') as f:
        current_sample = None
        current_meta = {}
        in_sample_table = False

        for line in f:
            if line.startswith('^SAMPLE'):
                if current_sample and current_sample in samples_expr:
                    samples_meta.append(current_meta.copy())
                current_sample = line.strip().split('=')[1].strip()
                current_meta = {'sample_id': current_sample}
                in_sample_table = False

            elif line.startswith('!Sample_title'):
                current_meta['title'] = line.strip().split('=', 1)[1].strip()
            elif line.startswith('!Sample_source_name_ch1'):
                current_meta['source'] = line.strip().split('=', 1)[1].strip()
            elif line.startswith('!Sample_characteristics_ch1'):
                val = line.strip().split('=', 1)[1].strip()
                if ':' in val:
                    k, v = val.split(':', 1)
                    current_meta[k.strip()] = v.strip()

            elif line.startswith('!sample_table_begin'):
                next(f)  # skip header
                in_sample_table = True
                samples_expr[current_sample] = {}
                continue
            elif line.startswith('!sample_table_end'):
                in_sample_table = False
                continue

            elif in_sample_table and current_sample:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    probe_id = parts[0]
                    try:
                        val = float(parts[1])
                        samples_expr[current_sample][probe_id] = val
                    except ValueError:
                        pass

        # Don't forget last sample
        if current_sample and current_sample in samples_expr:
            samples_meta.append(current_meta.copy())

    print(f"  Parsed {len(samples_meta)} samples")

    # Build expression matrix
    all_probes = set()
    for s_expr in samples_expr.values():
        all_probes.update(s_expr.keys())

    # Filter to probes with gene mapping
    mapped_probes = [p for p in all_probes if p in probe2gene]
    print(f"  Mapped probes: {len(mapped_probes)}")

    # Build DataFrame
    expr_data = {}
    for sample_id, s_expr in samples_expr.items():
        expr_data[sample_id] = {p: s_expr.get(p, np.nan) for p in mapped_probes}

    expr_df = pd.DataFrame(expr_data, index=mapped_probes)
    expr_df['gene'] = [probe2gene[p] for p in mapped_probes]

    # Collapse to gene level (max mean)
    sample_cols = [c for c in expr_df.columns if c != 'gene']
    expr_df['mean_expr'] = expr_df[sample_cols].mean(axis=1)
    expr_df = expr_df.sort_values('mean_expr', ascending=False).drop_duplicates(subset='gene', keep='first')
    gene_names = expr_df['gene'].values
    gene_expr = expr_df[sample_cols].copy()
    gene_expr.index = gene_names
    gene_expr.index.name = 'gene'

    print(f"  Gene-level expression: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")

    # Build metadata DataFrame
    meta_df = pd.DataFrame(samples_meta)

    # Identify tissue type
    meta_df['tissue_type'] = 'unknown'
    for idx, row in meta_df.iterrows():
        combined = ' '.join(str(v).lower() for v in row.values)
        if any(x in combined for x in ['tumor', 'cancer tissue', 'gastric cancer tissue']):
            meta_df.at[idx, 'tissue_type'] = 'tumor'
        elif any(x in combined for x in ['normal', 'control from', 'normal tissue']):
            meta_df.at[idx, 'tissue_type'] = 'normal'

    # Patient ID
    meta_df['patient_id'] = meta_df['title'].apply(
        lambda t: re.findall(r'patient\s*(\d+)', str(t).lower())[0]
        if re.findall(r'patient\s*(\d+)', str(t).lower()) else ''
    )

    print(f"  Tissue types: {meta_df['tissue_type'].value_counts().to_dict()}")

    # Save cache
    gene_expr.to_csv(cache_expr)
    meta_df.to_csv(cache_meta, index=False)

    return gene_expr, meta_df


def paired_analysis(gene_expr, meta_df, candidates):
    """Perform paired differential expression analysis."""
    tumor_samples = meta_df[meta_df['tissue_type'] == 'tumor']['sample_id'].values
    normal_samples = meta_df[meta_df['tissue_type'] == 'normal']['sample_id'].values

    all_cols = set(gene_expr.columns)
    tumor_samples = [s for s in tumor_samples if s in all_cols]
    normal_samples = [s for s in normal_samples if s in all_cols]

    print(f"\n  Tumor samples: {len(tumor_samples)}")
    print(f"  Normal samples: {len(normal_samples)}")

    results = []
    available = [g for g in candidates if g in gene_expr.index]
    missing = [g for g in candidates if g not in gene_expr.index]

    if missing:
        print(f"  Missing genes: {missing}")
    print(f"  Available genes: {available}")

    for gene in available:
        tumor_vals = gene_expr.loc[gene, tumor_samples].dropna().values.astype(float)
        normal_vals = gene_expr.loc[gene, normal_samples].dropna().values.astype(float)

        n_t, n_n = len(tumor_vals), len(normal_vals)
        if n_t < 3 or n_n < 3:
            continue

        stat, p_mw = mannwhitneyu(tumor_vals, normal_vals, alternative='two-sided')

        p_paired = np.nan
        if n_t == n_n:
            try:
                _, p_paired = wilcoxon(tumor_vals, normal_vals)
            except Exception:
                pass

        mean_tumor = np.mean(tumor_vals)
        mean_normal = np.mean(normal_vals)
        log2fc = np.log2((mean_tumor + 0.01) / (mean_normal + 0.01))
        pooled_std = np.std(np.concatenate([tumor_vals, normal_vals]))
        cohens_d = (mean_tumor - mean_normal) / (pooled_std + 1e-10)

        results.append({
            'gene': gene,
            'mean_tumor': round(mean_tumor, 3),
            'mean_normal': round(mean_normal, 3),
            'log2FC': round(log2fc, 4),
            'cohens_d': round(cohens_d, 3),
            'p_mannwhitney': p_mw,
            'p_wilcoxon_paired': p_paired,
            'n_tumor': n_t,
            'n_normal': n_n,
            'direction': 'UP in tumor' if log2fc > 0 else 'DOWN in tumor',
            'significant': p_mw < 0.05,
        })

    return pd.DataFrame(results), available, tumor_samples, normal_samples


def plot_paired_expression(gene_expr, available, tumor_samples, normal_samples):
    """Generate boxplots for candidate genes."""
    n_genes = len(available)
    if n_genes == 0:
        return

    cols = min(5, n_genes)
    rows = (n_genes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 4*rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, gene in enumerate(available):
        ax = axes_flat[idx]
        tumor_vals = gene_expr.loc[gene, tumor_samples].dropna().values.astype(float)
        normal_vals = gene_expr.loc[gene, normal_samples].dropna().values.astype(float)

        bp = ax.boxplot([normal_vals, tumor_vals], positions=[0, 1],
                       widths=0.5, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], ['#4daf4a', '#e41a1c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        rng = np.random.default_rng(42)
        for vals, pos in [(normal_vals, 0), (tumor_vals, 1)]:
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                      alpha=0.3, s=10, color='black', zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Tumor'], fontsize=9)
        ax.set_ylabel('Expression', fontsize=9)
        ax.set_title(gene, fontsize=11, fontweight='bold')

        stat, p = mannwhitneyu(tumor_vals, normal_vals, alternative='two-sided')
        fc = np.mean(tumor_vals) / (np.mean(normal_vals) + 0.01)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.5, 0.95, f'{sig}\nFC={fc:.2f}', transform=ax.transAxes,
               ha='center', va='top', fontsize=8)

    for idx in range(n_genes, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('GSE27342: Cancer Endpoint Validation\n(80 paired GC vs Adjacent Normal, Affymetrix Exon 1.0 ST)',
                fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/gse27342_paired_expression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] figures/gse27342_paired_expression.png")


def main():
    print("=" * 70)
    print("Step 13A: GSE27342 Cancer Endpoint Validation")
    print("  (80 paired GC + Adjacent Normal, Affymetrix Human Exon 1.0 ST)")
    print("=" * 70)

    # Parse SOFT file
    print("\n[1] Parsing data...")
    gene_expr, meta_df = parse_soft_file()

    # Check candidates
    print(f"\n  Candidates in dataset: {[g for g in CANDIDATES if g in gene_expr.index]}")

    # Analysis
    print("\n[2] Differential expression analysis...")
    results_df, available, tumor_samples, normal_samples = paired_analysis(
        gene_expr, meta_df, CANDIDATES
    )

    if results_df.empty:
        print("  ERROR: No results generated.")
        return

    print(f"\n  {'Gene':<8} {'log2FC':>8} {'Cohen d':>8} {'p-value':>12} {'Direction'}")
    print("  " + "-" * 60)
    for _, row in results_df.sort_values('p_mannwhitney').iterrows():
        sig = '***' if row['p_mannwhitney'] < 0.001 else '**' if row['p_mannwhitney'] < 0.01 else '*' if row['p_mannwhitney'] < 0.05 else ''
        print(f"  {row['gene']:<8} {row['log2FC']:>8.3f} {row['cohens_d']:>8.3f} "
              f"{row['p_mannwhitney']:>12.2e} {row['direction']} {sig}")

    # Save
    results_df.to_csv(f"{BASE}/results/gse27342_cancer_validation.csv", index=False, encoding='utf-8-sig')
    print(f"\n  [SAVED] results/gse27342_cancer_validation.csv")

    # Plot
    print("\n[3] Generating figure...")
    plot_paired_expression(gene_expr, available, tumor_samples, normal_samples)

    # Summary
    sig_genes = results_df[results_df['significant']]['gene'].tolist()
    up_genes = results_df[(results_df['significant']) & (results_df['log2FC'] > 0)]['gene'].tolist()
    down_genes = results_df[(results_df['significant']) & (results_df['log2FC'] < 0)]['gene'].tolist()
    print(f"\n  Summary:")
    print(f"    Significant (p<0.05): {sig_genes}")
    print(f"    UP in tumor: {up_genes}")
    print(f"    DOWN in tumor: {down_genes}")

    print("\n" + "=" * 70)
    print("Step 13A Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
