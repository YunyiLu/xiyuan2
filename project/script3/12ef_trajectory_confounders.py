"""
Step 12E: ITLN1 Complete Trajectory Analysis (Correa Cascade)
Step 12F: Clinical Confounder Assessment

12E validates ITLN1 as a progression marker (not just IM marker):
  - GSE55696 Correa cascade trajectory: CG → LGIN → HGIN → EGC
  - GSE78523 IIM vs CIM subtype analysis
  - Correlation with OLFM4/REG4 (redundancy check)

12F evaluates confounders for Tier 1 new candidates:
  - Tissue expression specificity from HPA
  - scRNA cell type origin from adata_integrated.h5ad

Output:
  - figures/itln1_trajectory.png
  - figures/gene_tissue_specificity.png
  - results/itln1_trajectory_stats.csv
  - results/confounder_assessment.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DATASET = "C:/FDU/Y4S2/xiyuan/project/dataset"
os.makedirs(f"{BASE}/figures", exist_ok=True)

TIER1_NEW = ["ITLN1", "PRAP1"]
TIER1_ALL = ["OLFM4", "REG4", "ITLN1", "PRAP1"]


# ============================================================
# 12E-1: ITLN1 trajectory in GSE55696 (Correa cascade)
# ============================================================
def itln1_correa_trajectory():
    """Plot ITLN1 expression across Correa cascade stages."""
    print("\n[12E-1] ITLN1 Correa Cascade Trajectory (GSE55696)")
    print("-" * 50)

    expr = pd.read_csv(f"{DATASET}/GEO_bulk/GSE55696/GSE55696_expression.csv", index_col=0)
    meta = pd.read_csv(f"{DATASET}/GEO_bulk/GSE55696/GSE55696_metadata.csv")

    # Check gene availability
    genes_to_plot = ["ITLN1", "OLFM4", "REG4", "PRAP1"]
    available = [g for g in genes_to_plot if g in expr.index]
    missing = [g for g in genes_to_plot if g not in expr.index]
    print(f"  Available: {available}")
    if missing:
        print(f"  Missing: {missing}")

    # Stages in order
    stage_order = ["CG", "LGIN", "HGIN", "EGC"]
    stage_labels = ["Chronic\nGastritis", "Low-grade\nIN", "High-grade\nIN", "Early\nGC"]
    meta_valid = meta[meta["stage"].isin(stage_order)]
    print(f"  Samples by stage: {meta_valid['stage'].value_counts().to_dict()}")

    # Plot
    fig, axes = plt.subplots(1, len(available), figsize=(4*len(available), 4.5), squeeze=False)
    axes = axes[0]

    stats_rows = []

    for idx, gene in enumerate(available):
        ax = axes[idx]
        gene_expr = expr.loc[gene]

        data_by_stage = []
        for stage in stage_order:
            samples = meta_valid[meta_valid["stage"] == stage]["sample_id"]
            vals = gene_expr[samples.values].dropna().values
            data_by_stage.append(vals)

        # Box plot
        bp = ax.boxplot(data_by_stage, positions=range(len(stage_order)),
                       widths=0.6, patch_artist=True, showfliers=True)
        colors = ['#a8dadc', '#457b9d', '#e63946', '#d62828']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, vals in enumerate(data_by_stage):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                      alpha=0.4, s=15, color='black', zorder=3)

        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(stage_labels, fontsize=9)
        ax.set_ylabel("Expression (log2)", fontsize=10)
        ax.set_title(gene, fontsize=12, fontweight='bold')

        # Kruskal-Wallis test
        stat, p = kruskal(*[d for d in data_by_stage if len(d) > 0])
        ax.text(0.05, 0.95, f"KW p={p:.2e}", transform=ax.transAxes,
               fontsize=8, va='top', ha='left')

        # Trend: median values
        medians = [np.median(d) if len(d) > 0 else np.nan for d in data_by_stage]
        stats_rows.append({
            "gene": gene,
            "CG_median": medians[0] if len(data_by_stage[0]) > 0 else np.nan,
            "LGIN_median": medians[1] if len(data_by_stage[1]) > 0 else np.nan,
            "HGIN_median": medians[2] if len(data_by_stage[2]) > 0 else np.nan,
            "EGC_median": medians[3] if len(data_by_stage[3]) > 0 else np.nan,
            "kruskal_p": p,
            "trend": "up" if medians[-1] > medians[0] else "down" if medians[-1] < medians[0] else "flat",
        })

        print(f"  {gene}: medians = {[f'{m:.2f}' for m in medians]}, KW p={p:.2e}")

    plt.suptitle("Correa Cascade Trajectory (GSE55696)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/itln1_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] figures/itln1_trajectory.png")

    # Save stats
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(f"{BASE}/results/itln1_trajectory_stats.csv", index=False, encoding='utf-8-sig')
    print(f"  [SAVED] results/itln1_trajectory_stats.csv")
    return df_stats


# ============================================================
# 12E-2: ITLN1 in GSE78523 by IM subtype (IIM vs CIM)
# ============================================================
def itln1_im_subtype_analysis():
    """Check if ITLN1 progressor effect is IIM-specific or universal."""
    print("\n[12E-2] ITLN1 by IM Subtype (GSE78523)")
    print("-" * 50)

    df = pd.read_csv(f"{BASE}/data/gse78523_gene_expr.csv", index_col=0)
    df_im = df[df["group"] != "Healthy"]

    genes = ["ITLN1", "OLFM4", "REG4", "PRAP1"]
    available = [g for g in genes if g in df_im.columns]

    print(f"  {'Gene':<8} {'IIM prog vs ctrl':>18} {'CIM prog vs ctrl':>18} {'Interpretation'}")
    print("  " + "-" * 70)

    for gene in available:
        # IIM: progressor vs control
        iim_prog = df_im[df_im["group"] == "IIM_GC_progressor"][gene].values
        iim_ctrl = df_im[df_im["group"] == "IIM_control"][gene].values
        # CIM: progressor vs control
        cim_prog = df_im[df_im["group"] == "CIM_GC_progressor"][gene].values
        cim_ctrl = df_im[df_im["group"] == "CIM_control"][gene].values

        # Mann-Whitney U
        if len(iim_prog) > 1 and len(iim_ctrl) > 1:
            stat_iim, p_iim = mannwhitneyu(iim_prog, iim_ctrl, alternative='greater')
            d_iim = (np.mean(iim_prog) - np.mean(iim_ctrl)) / (np.std(np.concatenate([iim_prog, iim_ctrl])) + 1e-10)
        else:
            p_iim, d_iim = 1.0, 0.0

        if len(cim_prog) > 1 and len(cim_ctrl) > 1:
            stat_cim, p_cim = mannwhitneyu(cim_prog, cim_ctrl, alternative='greater')
            d_cim = (np.mean(cim_prog) - np.mean(cim_ctrl)) / (np.std(np.concatenate([cim_prog, cim_ctrl])) + 1e-10)
        else:
            p_cim, d_cim = 1.0, 0.0

        # Interpretation
        if d_iim > 0.5 and d_cim > 0.5:
            interp = "Both subtypes (universal)"
        elif d_iim > 0.5:
            interp = "IIM-specific"
        elif d_cim > 0.5:
            interp = "CIM-specific"
        else:
            interp = "Weak in both"

        print(f"  {gene:<8} d={d_iim:+.2f} p={p_iim:.3f}   d={d_cim:+.2f} p={p_cim:.3f}   {interp}")


# ============================================================
# 12E-3: Correlation with OLFM4/REG4 (redundancy check)
# ============================================================
def correlation_analysis():
    """Check independence of ITLN1/PRAP1 from OLFM4/REG4."""
    print("\n[12E-3] Correlation Analysis (Redundancy Check)")
    print("-" * 50)

    df = pd.read_csv(f"{BASE}/data/gse78523_gene_expr.csv", index_col=0)
    df_im = df[df["group"] != "Healthy"]  # Only IM patients

    genes = ["OLFM4", "REG4", "ITLN1", "PRAP1"]
    available = [g for g in genes if g in df_im.columns]

    print(f"  Spearman correlations (IM patients only, n={len(df_im)}):")
    print(f"  {'Pair':<20} {'rho':>6} {'p':>10} {'Interpretation'}")
    print("  " + "-" * 50)

    for i in range(len(available)):
        for j in range(i+1, len(available)):
            g1, g2 = available[i], available[j]
            x = df_im[g1].values
            y_val = df_im[g2].values
            rho, p = spearmanr(x, y_val)

            if abs(rho) > 0.7:
                interp = "HIGH redundancy"
            elif abs(rho) > 0.4:
                interp = "Moderate"
            else:
                interp = "Low (good for panel)"

            print(f"  {g1+' vs '+g2:<20} {rho:>6.3f} {p:>10.4f} {interp}")


# ============================================================
# 12F: Confounder Assessment
# ============================================================
def confounder_assessment():
    """Assess clinical confounders using HPA tissue data."""
    print("\n[12F] Clinical Confounder Assessment")
    print("-" * 50)

    # Load HPA tissue data
    hpa = pd.read_csv(f"{BASE}/data/step12_databases/hpa_annotations.csv")

    confounders = []
    for _, row in hpa.iterrows():
        gene = row["gene"]
        stomach = row.get("stomach_nTPM", 0)
        liver = row.get("liver_nTPM", 0)
        gi_spec = row.get("gi_specificity", 0)
        max_tissue = str(row.get("max_tissue", ""))

        # Assess confounders
        risks = []
        if liver > 100:
            risks.append(f"Liver expression high ({liver:.0f} nTPM)")
        if "liver" in max_tissue.lower():
            risks.append("Max tissue = liver")
        if gi_spec < 0.5:
            risks.append(f"Low GI specificity ({gi_spec:.2f})")
        if gene == "ITLN1":
            risks.append("Known adipose/metabolic marker (omentin-1)")
            risks.append("BMI/insulin resistance confounder")
        if gene == "PRAP1":
            if liver > 500:
                risks.append("Substantial liver expression")
            risks.append("Lipoprotein-associated, fasting state variable")

        confounders.append({
            "gene": gene,
            "stomach_nTPM": stomach,
            "liver_nTPM": liver,
            "gi_specificity": gi_spec,
            "max_tissue": max_tissue,
            "confounder_risks": "; ".join(risks) if risks else "Low risk",
            "severity": "HIGH" if liver > 500 or gi_spec < 0.3 else
                       "MODERATE" if liver > 100 or gi_spec < 0.5 else "LOW",
        })

    df_conf = pd.DataFrame(confounders)
    df_conf.to_csv(f"{BASE}/results/confounder_assessment.csv", index=False, encoding='utf-8-sig')

    print("\n  Tier 1 Confounder Summary:")
    for _, row in df_conf[df_conf["gene"].isin(TIER1_ALL)].iterrows():
        print(f"    {row['gene']:<8} [{row['severity']:<8}] {row['confounder_risks']}")

    print(f"\n  [SAVED] results/confounder_assessment.csv")

    # Plot: tissue specificity heatmap for key genes
    plot_tissue_specificity()

    return df_conf


def plot_tissue_specificity():
    """Create tissue expression comparison plot for Tier 1 genes."""
    tissue_df = pd.read_csv(
        f"{BASE}/data/step12_databases/hpa/rna_tissue_consensus.tsv", sep='\t'
    )

    genes = ["OLFM4", "REG4", "ITLN1", "PRAP1", "FABP1", "CPS1"]
    tissues = ["stomach", "small intestine", "colon", "liver", "adipose tissue",
               "bone marrow", "kidney", "pancreas"]

    # Build matrix
    matrix = np.zeros((len(genes), len(tissues)))
    for i, gene in enumerate(genes):
        gene_data = tissue_df[tissue_df["Gene name"] == gene]
        for j, tissue in enumerate(tissues):
            match = gene_data[gene_data["Tissue"].str.contains(tissue, case=False, na=False)]
            if not match.empty:
                matrix[i, j] = match["nTPM"].max()

    # Log transform for visualization
    matrix_log = np.log2(matrix + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix_log, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels([t.capitalize() for t in tissues], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=10)

    # Add text values
    for i in range(len(genes)):
        for j in range(len(tissues)):
            val = matrix[i, j]
            color = 'white' if matrix_log[i, j] > matrix_log.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='log2(nTPM + 1)')
    ax.set_title("Tissue Expression (nTPM) - Confounder Assessment", fontsize=11)

    # Add tier labels
    tier_labels = ["Tier1", "Tier1", "Tier1", "Tier1", "Tier3", "Tier3"]
    for i, label in enumerate(tier_labels):
        ax.text(-0.8, i, label, ha='right', va='center', fontsize=8,
               color='green' if 'Tier1' in label else 'orange')

    plt.tight_layout()
    fig.savefig(f"{BASE}/figures/gene_tissue_specificity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] figures/gene_tissue_specificity.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Step 12E: ITLN1 Trajectory + 12F: Confounder Assessment")
    print("=" * 70)

    # 12E
    itln1_correa_trajectory()
    itln1_im_subtype_analysis()
    correlation_analysis()

    # 12F
    confounder_assessment()

    print("\n" + "=" * 70)
    print("Step 12E/F Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
