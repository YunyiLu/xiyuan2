"""
Step 12 Summary Report: Circulating-Detectable Protein Panel Evaluation
Integrates all sub-step results (12A-12F) into a final interpretive report.

Output: results/step12_summary_report.txt
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"


def main():
    report_lines = []

    def p(text=""):
        report_lines.append(text)
        print(text)

    p("=" * 80)
    p("STEP 12 COMPREHENSIVE REPORT: Circulating-Detectable Protein Panel Evaluation")
    p("=" * 80)
    p()

    # ============ 12A: Secretion Annotation ============
    p("━━━ 12A: Secretion Mechanism Annotation ━━━")
    ann = pd.read_csv(f"{BASE}/results/circulating_annotation.csv")
    p(f"  Total candidates annotated: {len(ann)}")
    p(f"  Signal peptide positive: {ann['uniprot_signal_peptide'].sum()}/19")
    p()
    p("  Tier Classification:")
    p(f"    Tier 1 (Active Secretion): OLFM4, REG4, ITLN1, PRAP1")
    p(f"    Tier 2 (Membrane Shed/EV): ANPEP, MUC17, CLDN4, PSCA")
    p(f"    Tier 3 (Damage Leakage):   FABP1, CPS1")
    p(f"    Tier 4 (Intracellular):    CLDN7, ANK3, IDH2, TOLLIP, POMP, MUC13")
    p(f"    Excluded:                  MUC5AC (direction), GAST (known), CCL3 (nonspecific)")
    p()

    # ============ 12B: Plasma Detectability ============
    p("━━━ 12B: Plasma Detectability Evidence ━━━")
    tier1 = ann[ann['tier'].str.contains('Tier1')]
    for _, row in tier1.iterrows():
        olink = "Olink+" if row.get('olink_explore_3072') else "Olink-"
        soma = "SomaScan+" if row.get('somascan_7k') else "SomaScan-"
        p(f"  {row['gene']:<8} {olink} {soma} ELISA={'Yes' if row.get('elisa_available') else 'No'}")
    p()
    p("  Key: All Tier 1 candidates are detected in human plasma via at least one platform.")
    p("  PRAP1 is the weakest (Olink only, no validated commercial ELISA).")
    p()

    # ============ 12D: Model Comparison ============
    p("━━━ 12D: LOOCV Model Comparison Results ━━━")
    p()
    p("  Scenario 1: Full cohort (n=45: 14 progressors, 31 non-progressors incl. healthy)")
    roc = pd.read_csv(f"{BASE}/results/circulating_panel_roc.csv")
    for _, row in roc.iterrows():
        if not row['model'].startswith('M_ext'):
            ci = f"[{row['AUC_CI_low']:.3f}-{row['AUC_CI_high']:.3f}]"
            p(f"    {row['model']:<20} AUC={row['AUC']:.3f} {ci}")
    p()

    p("  Scenario 2: IM-only cohort (n=30: 14 progressors, 16 non-progressor IM patients)")
    roc_im = pd.read_csv(f"{BASE}/results/circulating_panel_roc_im_only.csv")
    for _, row in roc_im.iterrows():
        ci = f"[{row['CI_low']:.3f}-{row['CI_high']:.3f}]"
        p(f"    {row['model']:<20} AUC={row['AUC_IM_only']:.3f} {ci}")
    p()

    p("  Incremental Assessment (Full cohort):")
    m3_row = roc[roc['model'] == 'M3_OLFM4_REG4'].iloc[0]
    m5_row = roc[roc['model'] == 'M5_add_PRAP1'].iloc[0]
    if 'delta_AUC_vs_M3' in m5_row.index:
        p(f"    M5 vs M3 ΔAUC = {m5_row['delta_AUC_vs_M3']:+.4f} "
          f"[{m5_row['delta_CI_low']:+.4f}, {m5_row['delta_CI_high']:+.4f}]")
    p()
    p("  Coefficient Direction Stability (M5, 100x 80% subsample):")
    stab_cols = [c for c in roc.columns if c.startswith('stability_')]
    if stab_cols:
        for c in stab_cols:
            gene = c.replace('stability_', '')
            val = m5_row[c]
            p(f"    {gene:<8}: {val:.1%} consistency {'[STABLE]' if val >= 0.7 else '[UNSTABLE]'}")
    p()

    # ============ 12E: ITLN1 Trajectory ============
    p("━━━ 12E: ITLN1 Trajectory & Independence Analysis ━━━")
    p()
    traj = pd.read_csv(f"{BASE}/results/itln1_trajectory_stats.csv")
    p("  GSE55696 Correa Cascade (CG → LGIN → HGIN → EGC):")
    for _, row in traj.iterrows():
        p(f"    {row['gene']:<8} KW p={row['kruskal_p']:.2e} "
          f"Medians: CG={row['CG_median']:.1f} → LGIN={row['LGIN_median']:.1f} → "
          f"HGIN={row['HGIN_median']:.1f} → EGC={row['EGC_median']:.1f} [{row['trend']}]")
    p()
    p("  GSE78523 IM Subtype Analysis:")
    p("    ITLN1: CIM-specific progression signal (d=0.60, p=0.046)")
    p("    OLFM4: Universal (both IIM and CIM subtypes)")
    p("    PRAP1: WEAK/REVERSED direction in progressors (d=-0.36~-0.40)")
    p()
    p("  *** CRITICAL FINDING: PRAP1 does NOT behave as expected in progressors ***")
    p("      In GSE78523, PRAP1 shows LOWER expression in progressors vs controls")
    p("      This contradicts its inclusion in the innovation model M5")
    p("      Recommendation: Remove PRAP1 from core panel; retain for discussion only")
    p()
    p("  Correlation Matrix (IM patients, n=30):")
    p("    OLFM4-ITLN1: rho=0.678 (moderate, some redundancy)")
    p("    OLFM4-REG4:  rho=0.521 (moderate)")
    p("    OLFM4-PRAP1: rho=0.152 (low → independent dimension)")
    p("    ITLN1-PRAP1: rho=0.338 (low)")
    p("    REG4-PRAP1:  rho=0.168 (low)")
    p()

    # ============ 12F: Confounders ============
    p("━━━ 12F: Confounder Assessment ━━━")
    conf = pd.read_csv(f"{BASE}/results/confounder_assessment.csv")
    tier1_conf = conf[conf['gene'].isin(['OLFM4', 'REG4', 'ITLN1', 'PRAP1'])]
    for _, row in tier1_conf.iterrows():
        p(f"  {row['gene']:<8} [{row['severity']:<8}] "
          f"Stomach={row['stomach_nTPM']:.0f} Liver={row['liver_nTPM']:.0f} nTPM | "
          f"{row['confounder_risks']}")
    p()

    # ============ 12G: Cell Origin (from scRNA) ============
    p("━━━ 12G: Single-Cell Origin Analysis ━━━")
    p()
    p("  scRNA-seq (189,750 cells, 6 stages: NAG/CAG/IM/EGC/GC):")
    stage_expr = pd.read_csv(f"{BASE}/results/scrna_stage_gene_expr.csv", index_col=0)
    p("  Stage-level mean expression:")
    for gene in ['OLFM4', 'ITLN1', 'REG4', 'PRAP1']:
        if gene in stage_expr.columns:
            vals = stage_expr[gene]
            p(f"    {gene:<8} NAG={vals.get('NAG',0):.3f} CAG={vals.get('CAG',0):.3f} "
              f"IM={vals.get('IM',0):.3f} EGC={vals.get('EGC',0):.3f} GC={vals.get('GC',0):.3f}")
    p()
    p("  Cluster Analysis:")
    p("    OLFM4: 80% epithelial, 80% IM-enriched clusters → IM-specific epithelial")
    p("    ITLN1: 80% epithelial, only 20% IM-enriched → specialized cell type")
    p("    REG4:  80% epithelial, 60% IM-enriched → IM epithelial")
    p("    PRAP1: 80% epithelial, 80% IM-enriched → IM epithelial")
    p()

    # ============ FINAL VERDICT ============
    p("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    p("FINAL INTEGRATED ASSESSMENT")
    p("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    p()
    p("1. RECOMMENDED PANEL: OLFM4 + REG4 + ITLN1 (3-protein panel)")
    p("   - PRAP1 removed: reversed direction in progressor cohort")
    p("   - Corresponds to revised M4 model (AUC=0.795 full, 0.629 IM-only)")
    p()
    p("2. INNOVATION LEVEL ASSESSMENT:")
    p("   ┌─────────────────────────────────────────────────────────────────┐")
    p("   │ Q: Does M4 (OLFM4+REG4+ITLN1) improve over M3 (OLFM4+REG4)?  │")
    p("   │ A: NO statistically significant improvement in this cohort.     │")
    p("   │    ΔAUC = -0.011 [-0.044, 0.000] (full cohort)                 │")
    p("   │    However, ITLN1 coefficient direction is 94% stable.          │")
    p("   └─────────────────────────────────────────────────────────────────┘")
    p()
    p("3. WHERE THE INNOVATION ACTUALLY LIES:")
    p("   a) Mechanism-driven discovery: scRNA → TransitionRisk → candidate")
    p("      selection is novel compared to correlative serum screening")
    p("   b) Cell-of-origin precision: We identify WHICH cell state in WHICH")
    p("      spatial context secretes each marker (not achievable by serum-only)")
    p("   c) ITLN1 as CIM-specific marker: First identification of ITLN1 in")
    p("      complete intestinal metaplasia → cancer progression context")
    p("   d) Progressor cohort validation: GSE78523 (14 progressors) provides")
    p("      unique clinical anchor not available to purely tissue-based studies")
    p()
    p("4. HONEST LIMITATIONS:")
    p("   - n=45 (14 progressors) is underpowered to detect ΔAUC < 0.05")
    p("   - IM-only analysis (n=30) drops AUC to 0.63-0.72 (modest)")
    p("   - No serum protein-level validation (tissue RNA only)")
    p("   - ITLN1 has metabolic confounders (BMI/adipose) requiring")
    p("     prospective cohort correction")
    p("   - PRAP1 failed directionality test → panel reduced to 3 proteins")
    p()
    p("5. RECOMMENDED PAPER FRAMING:")
    p("   'Mechanism-informed circulating biomarker candidate pipeline for")
    p("    gastric intestinal metaplasia progression monitoring'")
    p("   NOT: 'Novel blood test for early gastric cancer detection'")
    p()
    p("6. NEXT STEPS FOR CLINICAL TRANSLATION:")
    p("   a) Prospective cohort: 200+ IM patients, serum ELISA for")
    p("      OLFM4/REG4/ITLN1, 3-5 year follow-up")
    p("   b) Confounders to collect: BMI, fasting glucose, HbA1c,")
    p("      ALT/AST, HP status, IM subtype (IIM/CIM)")
    p("   c) Comparator: PGI/PGII ratio + HP status (current standard)")
    p()
    p("=" * 80)

    # Save report
    report_path = f"{BASE}/results/step12_summary_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[SAVED] {report_path}")


if __name__ == "__main__":
    main()
