"""
Step 14: Loss Marker Systematic Screen
"What disappears" — proteins lost during IM/cancer transformation

Strategy:
  1. GSE78523 (Correa cascade): Healthy vs CIM/IIM → genes DOWN in IM
  2. GSE27342 (Cancer endpoint): Normal vs Tumor → genes DOWN in cancer
  3. Intersection + secretion filter → candidate loss markers
  4. Rank by effect size, secretability, and blood detectability

Output: results/loss_markers_screen.csv
        results/loss_markers_top_candidates.csv
        figures/loss_markers_volcano.png
        figures/loss_markers_heatmap.png
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
os.makedirs(f"{BASE}/results", exist_ok=True)
os.makedirs(f"{BASE}/figures", exist_ok=True)


def screen_gse78523_loss():
    """Screen for genes downregulated in IM vs Healthy (Correa cascade)."""
    print("\n[1] GSE78523: Healthy vs IM (loss in IM)")
    df = pd.read_csv(f"{BASE}/data/gse78523_gene_expr.csv")

    healthy = df[df['group'] == 'Healthy']
    im_all = df[df['group'].isin(['CIM_control', 'CIM_GC_progressor', 'IIM_control', 'IIM_GC_progressor'])]

    gene_cols = [c for c in df.columns if c not in ['sample_id', 'group', 'im_type', 'progression_status']]
    print(f"  Healthy: {len(healthy)}, IM: {len(im_all)}, Genes: {len(gene_cols)}")

    results = []
    for gene in gene_cols:
        h_vals = healthy[gene].values.astype(float)
        im_vals = im_all[gene].values.astype(float)

        h_mean = np.mean(h_vals)
        im_mean = np.mean(im_vals)

        # Skip very low expression genes
        if h_mean < 1.0 and im_mean < 1.0:
            continue

        # Log2 fold change (IM vs Healthy) — negative = loss in IM
        log2fc = np.log2((im_mean + 0.01) / (h_mean + 0.01))

        try:
            _, p = mannwhitneyu(h_vals, im_vals, alternative='two-sided')
        except:
            continue

        results.append({
            'gene': gene,
            'mean_healthy': h_mean,
            'mean_IM': im_mean,
            'log2FC_IM_vs_Healthy': log2fc,
            'pvalue_78523': p,
        })

    res_df = pd.DataFrame(results)

    # FDR correction
    _, fdr, _, _ = multipletests(res_df['pvalue_78523'], method='fdr_bh')
    res_df['fdr_78523'] = fdr

    # Filter: significantly DOWN in IM (log2FC < -0.5, FDR < 0.1)
    loss_in_im = res_df[(res_df['log2FC_IM_vs_Healthy'] < -0.5) & (res_df['fdr_78523'] < 0.1)]
    loss_in_im = loss_in_im.sort_values('log2FC_IM_vs_Healthy')

    print(f"  Total tested: {len(res_df)}")
    print(f"  Significantly DOWN in IM (log2FC<-0.5, FDR<0.1): {len(loss_in_im)}")
    print(f"  Top 10 losses:")
    for _, row in loss_in_im.head(10).iterrows():
        print(f"    {row['gene']:<12} log2FC={row['log2FC_IM_vs_Healthy']:.2f}  "
              f"Healthy={row['mean_healthy']:.1f} → IM={row['mean_IM']:.1f}  FDR={row['fdr_78523']:.3f}")

    return res_df, loss_in_im


def screen_gse27342_loss():
    """Screen for genes downregulated in Tumor vs Normal (cancer endpoint)."""
    print("\n[2] GSE27342: Normal vs Tumor (loss in cancer)")
    expr = pd.read_csv(f"{BASE}/data/gse27342/expression_gene_level.csv", index_col=0)
    meta = pd.read_csv(f"{BASE}/data/gse27342/metadata.csv")

    normal_ids = meta[meta['tissue_type'] == 'normal']['sample_id'].values
    tumor_ids = meta[meta['tissue_type'] == 'tumor']['sample_id'].values

    normal_ids = [s for s in normal_ids if s in expr.columns]
    tumor_ids = [s for s in tumor_ids if s in expr.columns]

    print(f"  Normal: {len(normal_ids)}, Tumor: {len(tumor_ids)}, Genes: {len(expr)}")

    results = []
    for gene in expr.index:
        n_vals = expr.loc[gene, normal_ids].values.astype(float)
        t_vals = expr.loc[gene, tumor_ids].values.astype(float)

        n_mean = np.mean(n_vals)
        t_mean = np.mean(t_vals)

        if n_mean < 3.0 and t_mean < 3.0:
            continue

        log2fc = np.log2((t_mean + 0.01) / (n_mean + 0.01))

        try:
            _, p = mannwhitneyu(n_vals, t_vals, alternative='two-sided')
        except:
            continue

        results.append({
            'gene': gene,
            'mean_normal': n_mean,
            'mean_tumor': t_mean,
            'log2FC_tumor_vs_normal': log2fc,
            'pvalue_27342': p,
        })

    res_df = pd.DataFrame(results)
    _, fdr, _, _ = multipletests(res_df['pvalue_27342'], method='fdr_bh')
    res_df['fdr_27342'] = fdr

    loss_in_cancer = res_df[(res_df['log2FC_tumor_vs_normal'] < -0.5) & (res_df['fdr_27342'] < 0.05)]
    loss_in_cancer = loss_in_cancer.sort_values('log2FC_tumor_vs_normal')

    print(f"  Total tested: {len(res_df)}")
    print(f"  Significantly DOWN in tumor (log2FC<-0.5, FDR<0.05): {len(loss_in_cancer)}")
    print(f"  Top 10 losses:")
    for _, row in loss_in_cancer.head(10).iterrows():
        print(f"    {row['gene']:<12} log2FC={row['log2FC_tumor_vs_normal']:.2f}  "
              f"Normal={row['mean_normal']:.1f} → Tumor={row['mean_tumor']:.1f}  FDR={row['fdr_27342']:.4f}")

    return res_df, loss_in_cancer


def annotate_secretion(gene_list):
    """Annotate genes for secretion potential using UniProt keywords and known databases."""
    # Known secreted / blood-detectable proteins (from our prior research + databases)
    # This is a curated list of known secreted gastric proteins
    known_secreted = {
        'PSCA', 'GKN1', 'GKN2', 'PGA3', 'PGA4', 'PGA5', 'PGC',
        'LIPF', 'GIF', 'TFF1', 'TFF2', 'MUC5AC', 'MUC6', 'MUC1',
        'GAST', 'SST', 'GHRL', 'ATP4A', 'ATP4B',
        'WFDC2', 'LTF', 'CXCL17', 'CCL28', 'DMBT1',
        'VSIG1', 'CLDN18', 'AQP5', 'KCNE2', 'CA2',
        'CTSE', 'SPINK1', 'TFF3', 'LGALS4', 'REG1A', 'REG3A',
        'PGA', 'SERPINA5', 'CLU', 'C3', 'SERPINA1',
        'AGR2', 'FCGBP', 'ZG16', 'CLCA1', 'ITLN1',
    }

    # Signal peptide / extracellular annotation keywords
    secretion_keywords = {
        'GKN1': 'signal_peptide, secreted gastrokine',
        'GKN2': 'signal_peptide, secreted gastrokine',
        'PGA3': 'signal_peptide, pepsinogen A (zymogen secreted)',
        'PGA4': 'signal_peptide, pepsinogen A',
        'PGA5': 'signal_peptide, pepsinogen A',
        'PGC': 'signal_peptide, pepsinogen C (progastricsin)',
        'LIPF': 'signal_peptide, gastric lipase (secreted into lumen)',
        'GIF': 'signal_peptide, intrinsic factor (secreted by parietal cells)',
        'TFF1': 'signal_peptide, trefoil factor (mucus-associated secreted peptide)',
        'TFF2': 'signal_peptide, trefoil factor',
        'MUC5AC': 'secreted gel-forming mucin',
        'MUC6': 'secreted gel-forming mucin',
        'GAST': 'signal_peptide, gastrin (endocrine, enters blood)',
        'SST': 'signal_peptide, somatostatin (endocrine)',
        'GHRL': 'signal_peptide, ghrelin (endocrine, blood-detectable)',
        'PSCA': 'GPI-anchored, shed into blood (confirmed)',
        'CXCL17': 'signal_peptide, chemokine (secreted)',
        'DMBT1': 'signal_peptide, glycoprotein (secreted into lumen)',
        'CTSE': 'type II transmembrane, cathepsin E (shed ectodomain)',
        'WFDC2': 'signal_peptide, HE4 (FDA-approved serum biomarker)',
        'VSIG1': 'type I transmembrane (potentially shed)',
        'LTF': 'signal_peptide, lactoferrin (secreted)',
        'AGR2': 'signal_peptide, secreted ER-resident protein',
        'SPINK1': 'signal_peptide, serine protease inhibitor (secreted)',
        'CLU': 'signal_peptide, clusterin (secreted, blood protein)',
        'AQP5': 'integral membrane (water channel)',
        'CA2': 'cytoplasmic (carbonic anhydrase, leaked on damage)',
        'ATP4A': 'integral membrane (H+/K+ ATPase alpha)',
        'ATP4B': 'integral membrane (H+/K+ ATPase beta)',
        'KCNE2': 'integral membrane (K+ channel)',
        'CLDN18': 'integral membrane (tight junction)',
    }

    # Olink / SomaScan coverage
    olink_covered = {
        'PSCA', 'GKN1', 'GKN2', 'WFDC2', 'CLU', 'GAST', 'GHRL',
        'LTF', 'CXCL17', 'SPINK1', 'TFF3', 'AGR2', 'CTSE',
        'REG1A', 'C3', 'SERPINA1', 'MUC1', 'DMBT1',
    }

    somascan_covered = {
        'PSCA', 'GKN1', 'GKN2', 'PGC', 'WFDC2', 'CLU', 'GAST',
        'GHRL', 'LTF', 'GIF', 'LIPF', 'TFF1', 'TFF2', 'SPINK1',
    }

    elisa_available = {
        'PSCA', 'GKN1', 'GKN2', 'PGC', 'PGA5', 'GAST', 'GHRL',
        'WFDC2', 'TFF1', 'TFF2', 'CLU', 'LTF', 'SPINK1', 'GIF',
    }

    annotations = []
    for gene in gene_list:
        is_secreted = gene in known_secreted
        mechanism = secretion_keywords.get(gene, '')
        has_signal = 'signal_peptide' in mechanism.lower() if mechanism else False

        annotations.append({
            'gene': gene,
            'is_known_secreted': is_secreted,
            'has_signal_peptide': has_signal,
            'secretion_annotation': mechanism,
            'olink_covered': gene in olink_covered,
            'somascan_covered': gene in somascan_covered,
            'elisa_available': gene in elisa_available,
            'blood_detectable': is_secreted or has_signal or gene in olink_covered or gene in elisa_available,
        })

    return pd.DataFrame(annotations)


def combine_and_rank(im_full, im_loss, cancer_full, cancer_loss):
    """Find genes lost in BOTH IM and cancer, rank by detectability."""
    print("\n[3] Combining screens: genes lost in IM AND cancer")

    # Genes significantly down in IM
    im_loss_genes = set(im_loss['gene'].values)
    # Genes significantly down in cancer
    cancer_loss_genes = set(cancer_loss['gene'].values)

    # Intersection: consistently lost across Correa cascade
    both_lost = im_loss_genes & cancer_loss_genes
    print(f"  Lost in IM (GSE78523): {len(im_loss_genes)}")
    print(f"  Lost in Cancer (GSE27342): {len(cancer_loss_genes)}")
    print(f"  Lost in BOTH: {len(both_lost)}")

    if not both_lost:
        # Relax criteria: use union with preference for overlap
        print("  Relaxing to union (prioritizing overlap)...")
        all_loss = im_loss_genes | cancer_loss_genes
        # Score by presence in both
        both_lost = all_loss

    # Build combined table
    combined = []
    for gene in both_lost:
        row = {'gene': gene}

        # IM data
        im_row = im_full[im_full['gene'] == gene]
        if not im_row.empty:
            row['log2FC_IM'] = im_row.iloc[0]['log2FC_IM_vs_Healthy']
            row['fdr_IM'] = im_row.iloc[0]['fdr_78523']
            row['mean_healthy'] = im_row.iloc[0]['mean_healthy']
            row['mean_IM'] = im_row.iloc[0]['mean_IM']
        else:
            row['log2FC_IM'] = np.nan
            row['fdr_IM'] = np.nan
            row['mean_healthy'] = np.nan
            row['mean_IM'] = np.nan

        # Cancer data
        ca_row = cancer_full[cancer_full['gene'] == gene]
        if not ca_row.empty:
            row['log2FC_cancer'] = ca_row.iloc[0]['log2FC_tumor_vs_normal']
            row['fdr_cancer'] = ca_row.iloc[0]['fdr_27342']
            row['mean_normal'] = ca_row.iloc[0]['mean_normal']
            row['mean_tumor'] = ca_row.iloc[0]['mean_tumor']
        else:
            row['log2FC_cancer'] = np.nan
            row['fdr_cancer'] = np.nan
            row['mean_normal'] = np.nan
            row['mean_tumor'] = np.nan

        # Consistency score
        in_im = gene in im_loss_genes
        in_cancer = gene in cancer_loss_genes
        row['lost_in_IM'] = in_im
        row['lost_in_cancer'] = in_cancer
        row['consistent_loss'] = in_im and in_cancer

        combined.append(row)

    combined_df = pd.DataFrame(combined)

    # Annotate secretion
    annot_df = annotate_secretion(combined_df['gene'].tolist())
    combined_df = combined_df.merge(annot_df, on='gene', how='left')

    # Rank score: prioritize large effect + blood detectability + consistency
    combined_df['rank_score'] = (
        combined_df['consistent_loss'].astype(float) * 3 +
        combined_df['blood_detectable'].astype(float) * 2 +
        combined_df['has_signal_peptide'].astype(float) * 1 +
        combined_df['elisa_available'].astype(float) * 1 +
        (-combined_df['log2FC_IM'].fillna(0)) * 0.5 +
        (-combined_df['log2FC_cancer'].fillna(0)) * 0.5
    )

    combined_df = combined_df.sort_values('rank_score', ascending=False)

    return combined_df


def plot_volcano_dual(im_full, cancer_full, top_candidates):
    """Dual volcano plot showing loss markers in both datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    top_genes = set(top_candidates['gene'].head(15).values)

    # GSE78523 volcano
    ax = ax1
    x = im_full['log2FC_IM_vs_Healthy'].values
    y = -np.log10(im_full['fdr_78523'].values + 1e-300)
    colors = ['red' if g in top_genes else ('#aaaaaa' if fc > -0.5 else '#3498db')
              for g, fc in zip(im_full['gene'], x)]
    ax.scatter(x, y, c=colors, s=8, alpha=0.5, rasterized=True)

    for _, row in im_full[im_full['gene'].isin(top_genes)].iterrows():
        ax.annotate(row['gene'], (row['log2FC_IM_vs_Healthy'], -np.log10(row['fdr_78523'] + 1e-300)),
                   fontsize=7, fontweight='bold', color='red', alpha=0.9,
                   textcoords='offset points', xytext=(5, 3))

    ax.axvline(-0.5, color='gray', ls='--', lw=0.8)
    ax.axhline(-np.log10(0.1), color='gray', ls='--', lw=0.8)
    ax.set_xlabel('log2FC (IM vs Healthy)', fontsize=10)
    ax.set_ylabel('-log10(FDR)', fontsize=10)
    ax.set_title('GSE78523: Loss in Intestinal Metaplasia\n(n=15 Healthy vs n=30 IM)', fontsize=11)

    # GSE27342 volcano
    ax = ax2
    x = cancer_full['log2FC_tumor_vs_normal'].values
    y = -np.log10(cancer_full['fdr_27342'].values + 1e-300)
    colors = ['red' if g in top_genes else ('#aaaaaa' if fc > -0.5 else '#3498db')
              for g, fc in zip(cancer_full['gene'], x)]
    ax.scatter(x, y, c=colors, s=8, alpha=0.5, rasterized=True)

    for _, row in cancer_full[cancer_full['gene'].isin(top_genes)].iterrows():
        ax.annotate(row['gene'], (row['log2FC_tumor_vs_normal'], -np.log10(row['fdr_27342'] + 1e-300)),
                   fontsize=7, fontweight='bold', color='red', alpha=0.9,
                   textcoords='offset points', xytext=(5, 3))

    ax.axvline(-0.5, color='gray', ls='--', lw=0.8)
    ax.axhline(-np.log10(0.05), color='gray', ls='--', lw=0.8)
    ax.set_xlabel('log2FC (Tumor vs Normal)', fontsize=10)
    ax.set_ylabel('-log10(FDR)', fontsize=10)
    ax.set_title('GSE27342: Loss in Gastric Cancer\n(n=80 Normal vs n=80 Tumor, paired)', fontsize=11)

    plt.suptitle('Loss Marker Screen: Genes Downregulated During IM→Cancer Progression',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/loss_markers_volcano.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] figures/loss_markers_volcano.png")


def plot_candidate_heatmap(top_df, im_full, cancer_full):
    """Heatmap of top loss marker candidates."""
    genes = top_df['gene'].head(20).tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Build data matrix: [gene] × [mean_healthy, mean_IM, mean_normal, mean_tumor]
    data = []
    labels = []
    for gene in genes:
        row = []
        im_row = im_full[im_full['gene'] == gene]
        ca_row = cancer_full[cancer_full['gene'] == gene]

        h = im_row.iloc[0]['mean_healthy'] if not im_row.empty else np.nan
        im = im_row.iloc[0]['mean_IM'] if not im_row.empty else np.nan
        n = ca_row.iloc[0]['mean_normal'] if not ca_row.empty else np.nan
        t = ca_row.iloc[0]['mean_tumor'] if not ca_row.empty else np.nan
        row = [h, im, n, t]
        data.append(row)
        labels.append(gene)

    data = np.array(data)

    # Normalize each row (gene) to [0, 1]
    row_max = np.nanmax(data, axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    data_norm = data / row_max

    im_plot = ax.imshow(data_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Healthy\n(GSE78523)', 'IM\n(GSE78523)',
                       'Normal\n(GSE27342)', 'Tumor\n(GSE27342)'], fontsize=10)
    ax.set_yticks(range(len(labels)))

    # Annotate with secretion info
    ytick_labels = []
    for gene in labels:
        t_row = top_df[top_df['gene'] == gene]
        if not t_row.empty and t_row.iloc[0].get('blood_detectable', False):
            ytick_labels.append(f"★ {gene}")
        else:
            ytick_labels.append(f"  {gene}")

    ax.set_yticklabels(ytick_labels, fontsize=9, fontfamily='monospace')

    # Add value annotations
    for i in range(len(labels)):
        for j in range(4):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if data_norm[i, j] > 0.7 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

    cbar = plt.colorbar(im_plot, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Relative Expression (row-normalized)', fontsize=9)

    ax.set_title('Top 20 Loss Markers: Expression Across Correa Cascade\n'
                '(★ = blood-detectable / has signal peptide / ELISA available)',
                fontsize=11, fontweight='bold')

    # Add arrows
    ax.annotate('', xy=(1, -1.5), xytext=(0, -1.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               annotation_clip=False)
    ax.text(0.5, -2.0, 'LOST in IM', ha='center', fontsize=9, color='red',
           transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/loss_markers_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] figures/loss_markers_heatmap.png")


def generate_dual_panel_summary(top_loss_df):
    """Generate final summary comparing gain vs loss panel."""
    print("\n" + "=" * 70)
    print("DUAL-DIRECTION PANEL CONCEPT")
    print("=" * 70)

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  GAIN MARKERS (↑ in IM/cancer → appears in blood)              │
  │    OLFM4  — IM stem cell marker, secreted via MVs              │
  │    REG4   — IM goblet cell secretory protein                   │
  │    ITLN1  — CIM-specific goblet cell marker                    │
  ├─────────────────────────────────────────────────────────────────┤
  │  LOSS MARKERS (↓ in IM/cancer → disappears from blood)         │""")

    blood_detectable = top_loss_df[top_loss_df['blood_detectable'] == True].head(5)
    for _, row in blood_detectable.iterrows():
        fc_im = row.get('log2FC_IM', np.nan)
        fc_ca = row.get('log2FC_cancer', np.nan)
        fc_str = f"IM:{fc_im:.1f}" if not np.isnan(fc_im) else ""
        fc_str += f" Cancer:{fc_ca:.1f}" if not np.isnan(fc_ca) else ""
        annot = row.get('secretion_annotation', '')[:40]
        print(f"  │    {row['gene']:<8} — {annot:<40} ({fc_str}) │")

    print("""  └─────────────────────────────────────────────────────────────────┘

  CLINICAL LOGIC:
    Gain ↑ + Loss ↓ = High confidence IM transformation
    - High OLFM4/REG4 + Low PGC/GKN1 = very likely IM present
    - Adding ITLN1 discriminates CIM (high risk) from IIM (low risk)
    - Ratio-based: OLFM4/PGC or REG4/GKN1 may be more robust than
      absolute levels (controls for inter-individual variation)
""")


def main():
    print("=" * 70)
    print("Step 14: Loss Marker Systematic Screen")
    print("  'What disappears during IM → Cancer transformation'")
    print("=" * 70)

    # Screen 1: GSE78523 (Correa cascade)
    im_full, im_loss = screen_gse78523_loss()

    # Screen 2: GSE27342 (Cancer endpoint)
    cancer_full, cancer_loss = screen_gse27342_loss()

    # Combine and rank
    combined_df = combine_and_rank(im_full, im_loss, cancer_full, cancer_loss)

    # Save full results
    combined_df.to_csv(f"{BASE}/results/loss_markers_screen.csv", index=False, encoding='utf-8-sig')
    print(f"\n  [SAVED] results/loss_markers_screen.csv ({len(combined_df)} genes)")

    # Top candidates (blood-detectable + consistently lost)
    top = combined_df[
        (combined_df['blood_detectable'] == True) |
        (combined_df['consistent_loss'] == True)
    ].head(30)
    top.to_csv(f"{BASE}/results/loss_markers_top_candidates.csv", index=False, encoding='utf-8-sig')
    print(f"  [SAVED] results/loss_markers_top_candidates.csv ({len(top)} candidates)")

    # Print top candidates
    print(f"\n  {'Rank':<5} {'Gene':<10} {'FC_IM':>7} {'FC_Ca':>7} {'Secreted':>9} {'ELISA':>6} {'Score':>6}")
    print("  " + "-" * 65)
    for i, (_, row) in enumerate(top.head(15).iterrows()):
        fc_im = f"{row['log2FC_IM']:.2f}" if not np.isnan(row.get('log2FC_IM', np.nan)) else "N/A"
        fc_ca = f"{row['log2FC_cancer']:.2f}" if not np.isnan(row.get('log2FC_cancer', np.nan)) else "N/A"
        sec = "Yes" if row.get('blood_detectable', False) else "No"
        elisa = "Yes" if row.get('elisa_available', False) else "No"
        print(f"  {i+1:<5} {row['gene']:<10} {fc_im:>7} {fc_ca:>7} {sec:>9} {elisa:>6} {row['rank_score']:>6.1f}")

    # Figures
    print("\n[4] Generating figures...")
    plot_volcano_dual(im_full, cancer_full, top)
    plot_candidate_heatmap(top, im_full, cancer_full)

    # Summary
    generate_dual_panel_summary(top)

    print("\n" + "=" * 70)
    print("Step 14 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
