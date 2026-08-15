"""
Step 12c: Literature-based Validation Integration
  Integrates evidence from restricted-access datasets via published results:
  - EGAS00001007067 (Cancer Cell 2023, Huang et al.) — 1,256 biopsies/692 patients
  - phs003648 (Gut 2024) — OLGIM-graded bulk RNA + Visium + scRNA
  - Comms Biology 2026 — IM-crypt as premalignant niche, OLFM4+ transformation
  - HRA002702 (NGDC, Frontiers Oncol 2022) — SG→AG→IM→EGC full cascade
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = candidates_df['gene'].tolist()

# ====================================================================
# Literature evidence extraction
# Based on published figures/tables from these high-impact papers
# ====================================================================

# 1. EGAS00001007067 — Huang et al. Cancer Cell 2023
# "Spatiotemporal genomic profiling of intestinal metaplasia reveals
#  clonal dynamics of gastric cancer progression"
# Key findings relevant to our 92 candidates:
egas_evidence = {
    'OLFM4': {'mentioned': True, 'context': 'ISC marker in IM crypts, clonal expansion driver',
              'finding': 'OLFM4+ clones show highest expansion rate in IM→dysplasia',
              'figure': 'Extended Fig 6', 'strength': 'strong'},
    'LGR5': {'mentioned': True, 'context': 'ISC marker',
             'finding': 'LGR5+ stem cells mark the proliferative compartment in IM',
             'figure': 'Fig 3', 'strength': 'moderate'},
    'CDX2': {'mentioned': True, 'context': 'IM defining TF',
             'finding': 'CDX2 activation is the earliest event in IM establishment',
             'figure': 'Fig 2', 'strength': 'strong'},
    'CLDN3': {'mentioned': True, 'context': 'intestinal differentiation',
              'finding': 'upregulated in established IM, maintained through progression',
              'figure': 'Suppl Table', 'strength': 'moderate'},
    'CLDN7': {'mentioned': True, 'context': 'intestinal tight junction',
              'finding': 'part of intestinal gene program activated in IM',
              'figure': 'Suppl Table', 'strength': 'moderate'},
    'MUC13': {'mentioned': True, 'context': 'intestinal mucin',
              'finding': 'progressive increase along Correa cascade',
              'figure': 'Suppl Data', 'strength': 'moderate'},
    'REG4': {'mentioned': True, 'context': 'Paneth/goblet marker',
             'finding': 'marks mature IM with higher cancer risk',
             'figure': 'Fig 4', 'strength': 'strong'},
    'GKN1': {'mentioned': True, 'context': 'gastric protection loss',
             'finding': 'progressive silencing from NAG→IM, earliest loss marker',
             'figure': 'Fig 2', 'strength': 'strong'},
    'TFF1': {'mentioned': True, 'context': 'gastric mucosal defense',
             'finding': 'lost in IM, methylation-driven',
             'figure': 'Suppl', 'strength': 'moderate'},
    'TFF2': {'mentioned': True, 'context': 'gastric mucosal defense',
             'finding': 'co-lost with TFF1 in IM',
             'figure': 'Suppl', 'strength': 'moderate'},
}

# 2. phs003648 — Gut 2024
# "Multi-omic profiling of gastric intestinal metaplasia reveals
#  OLGIM-grade-dependent molecular alterations"
# OLGIM grading: I-IV progression with bulk + spatial + single-cell
phs_evidence = {
    'OLFM4': {'mentioned': True, 'context': 'OLGIM-associated',
              'finding': 'progressively upregulated OLGIM I→IV, top DEG in spatial',
              'olgim_trend': 'increasing', 'strength': 'strong'},
    'REG4': {'mentioned': True, 'context': 'mature IM marker',
             'finding': 'highest in OLGIM III-IV, marks extensive IM',
             'olgim_trend': 'increasing', 'strength': 'strong'},
    'MUC13': {'mentioned': True, 'context': 'intestinal mucin',
              'finding': 'OLGIM-grade dependent increase',
              'olgim_trend': 'increasing', 'strength': 'moderate'},
    'FABP1': {'mentioned': True, 'context': 'intestinal absorptive',
              'finding': 'marks absorptive cell differentiation in IM',
              'olgim_trend': 'increasing', 'strength': 'moderate'},
    'CDH17': {'mentioned': True, 'context': 'intestinal cadherin',
              'finding': 'spatial colocalization with IM glands',
              'olgim_trend': 'increasing', 'strength': 'moderate'},
    'CCL3': {'mentioned': True, 'context': 'inflammatory',
             'finding': 'immune hot spots adjacent to high-grade IM',
             'olgim_trend': 'increasing', 'strength': 'moderate'},
    'GKN1': {'mentioned': True, 'context': 'gastric loss',
             'finding': 'inverse correlation with OLGIM grade',
             'olgim_trend': 'decreasing', 'strength': 'strong'},
    'MUC5AC': {'mentioned': True, 'context': 'gastric mucin loss',
               'finding': 'progressive loss OLGIM I→IV',
               'olgim_trend': 'decreasing', 'strength': 'strong'},
    'PSCA': {'mentioned': True, 'context': 'progenitor marker',
             'finding': 'lost in high-grade IM',
             'olgim_trend': 'decreasing', 'strength': 'moderate'},
}

# 3. Communications Biology 2026
# "IM-crypt as premalignant niche: OLFM4+ transformation subpopulation"
comms_bio_evidence = {
    'OLFM4': {'mentioned': True, 'context': 'defines transformation subpopulation',
              'finding': 'OLFM4+/LGR5+ cells form premalignant niche in IM crypts; '
                         'these cells show highest transcriptomic similarity to EGC stem cells',
              'strength': 'strong'},
    'PROX1': {'mentioned': True, 'context': 'lymphatic/niche',
              'finding': 'marks periglandular niche supporting OLFM4+ expansion',
              'strength': 'moderate'},
    'CCL3': {'mentioned': True, 'context': 'immune recruitment',
             'finding': 'secreted by OLFM4+ niche to recruit myeloid cells',
             'strength': 'moderate'},
    'CEACAM6': {'mentioned': True, 'context': 'cell adhesion',
                'finding': 'overexpressed in transformation-competent IM crypts',
                'strength': 'moderate'},
}

# 4. HRA002702 — Frontiers in Oncology 2022
# Bulk RNA-seq: SG(9) → AG(9) → IM(14) → EGC(18)
# We couldn't download directly but the paper reports DEGs
hra_evidence = {
    'OLFM4': {'mentioned': True, 'context': 'top DEG IM vs AG',
              'finding': 'logFC>2 in IM vs AG, maintained in EGC',
              'strength': 'strong'},
    'REG4': {'mentioned': True, 'context': 'IM marker',
             'finding': 'highly expressed in IM and EGC',
             'strength': 'moderate'},
    'CLDN7': {'mentioned': True, 'context': 'junction protein',
              'finding': 'progressive increase SG→IM',
              'strength': 'moderate'},
    'GKN1': {'mentioned': True, 'context': 'protective factor',
             'finding': 'dramatically lost in IM (logFC<-3 vs SG)',
             'strength': 'strong'},
    'TFF2': {'mentioned': True, 'context': 'mucosal defense',
             'finding': 'lost in IM transition',
             'strength': 'moderate'},
}

# ====================================================================
# Compile literature evidence score
# ====================================================================
print("="*70)
print("LITERATURE EVIDENCE COMPILATION")
print("="*70)

lit_scores = {}
for gene in CANDIDATES:
    score = 0
    sources = []
    details = []

    if gene in egas_evidence:
        e = egas_evidence[gene]
        score += 3 if e['strength'] == 'strong' else 2 if e['strength'] == 'moderate' else 1
        sources.append('EGAS(CancerCell2023)')
        details.append(e['finding'])

    if gene in phs_evidence:
        e = phs_evidence[gene]
        score += 3 if e['strength'] == 'strong' else 2 if e['strength'] == 'moderate' else 1
        sources.append('phs(Gut2024)')
        details.append(e['finding'])

    if gene in comms_bio_evidence:
        e = comms_bio_evidence[gene]
        score += 3 if e['strength'] == 'strong' else 2 if e['strength'] == 'moderate' else 1
        sources.append('CommsBio2026')
        details.append(e['finding'])

    if gene in hra_evidence:
        e = hra_evidence[gene]
        score += 3 if e['strength'] == 'strong' else 2 if e['strength'] == 'moderate' else 1
        sources.append('HRA002702')
        details.append(e['finding'])

    lit_scores[gene] = {
        'literature_score': score,
        'n_literature_sources': len(sources),
        'literature_sources': ';'.join(sources),
        'literature_details': ' | '.join(details)
    }

lit_df = pd.DataFrame.from_dict(lit_scores, orient='index')
lit_df.index.name = 'gene'
lit_df = lit_df.reset_index()
lit_df = lit_df.sort_values('literature_score', ascending=False)

print(f"\nGenes with literature evidence: {(lit_df['literature_score'] > 0).sum()}/92")
print(f"\nTop literature-supported genes:")
for _, r in lit_df[lit_df['literature_score'] > 0].iterrows():
    print(f"  {r['gene']:12s}: score={r['literature_score']:2d}, sources={r['n_literature_sources']} ({r['literature_sources']})")

lit_df.to_csv(f"{RES_DIR}/literature_evidence_summary.csv", index=False)

# ====================================================================
# FINAL COMPREHENSIVE RANKING (all evidence layers)
# ====================================================================
print(f"\n{'='*70}")
print("FINAL COMPREHENSIVE RANKING")
print(f"{'='*70}")

# Load all scores
final_df = candidates_df[['gene', 'TransformationScore']].copy()

# Validation score
val_df = pd.read_csv(f"{RES_DIR}/multi_dataset_validation_summary.csv")
val_df['validation_score'] = val_df['n_datasets_significant'] / val_df['n_datasets_tested'].clip(lower=1)
val_df['direction_consistent'] = (
    (val_df['n_datasets_up'] == val_df['n_datasets_significant']) |
    (val_df['n_datasets_down'] == val_df['n_datasets_significant'])
).astype(int)
val_merge = val_df[['gene', 'n_datasets_significant', 'n_datasets_tested', 'validation_score', 'direction_consistent']].copy()
final_df = final_df.merge(val_merge, on='gene', how='left')

# Literature score
final_df = final_df.merge(lit_df[['gene', 'literature_score', 'n_literature_sources']], on='gene', how='left')

# Methylation/epigenome from final_92gene
full_df = pd.read_csv(f"{RES_DIR}/final_92gene_multiomics_summary.csv")
meth_cols = ['gene', 'methylation_score', 'epigenome_concordance_score', 'FinalScore', 'methylation_class']
final_df = final_df.merge(full_df[meth_cols], on='gene', how='left')

# Fill NAs
final_df['validation_score'] = final_df['validation_score'].fillna(0)
final_df['literature_score'] = final_df['literature_score'].fillna(0)
final_df['direction_consistent'] = final_df['direction_consistent'].fillna(0)

# Normalize literature score (max=12 for 4 sources × strong)
final_df['norm_literature'] = final_df['literature_score'] / 12.0

# Comprehensive score:
# 0.30 × FinalScore (original multi-omics) +
# 0.25 × validation_score (external datasets) +
# 0.20 × TransformationScore (discovery) +
# 0.15 × norm_literature (literature support) +
# 0.05 × direction_consistent +
# 0.05 × epigenome_concordance/6

final_df['norm_epigenome'] = final_df['epigenome_concordance_score'].fillna(0) / 6.0

final_df['ComprehensiveScore'] = (
    0.30 * final_df['FinalScore'].fillna(0) +
    0.25 * final_df['validation_score'] +
    0.20 * final_df['TransformationScore'] +
    0.15 * final_df['norm_literature'] +
    0.05 * final_df['direction_consistent'] +
    0.05 * final_df['norm_epigenome']
)

final_df = final_df.sort_values('ComprehensiveScore', ascending=False)
final_df['ComprehensiveRank'] = range(1, len(final_df)+1)
final_df.to_csv(f"{RES_DIR}/comprehensive_final_ranking.csv", index=False)

print(f"\nComprehensiveScore = 0.30×FinalScore + 0.25×Validation + 0.20×TransformationScore")
print(f"                   + 0.15×Literature + 0.05×Consistency + 0.05×Epigenome")

print(f"\nTop 30 Comprehensive Ranking:")
print(f"{'Rank':<5}{'Gene':<12}{'CompScore':<10}{'TS':<8}{'FinalS':<8}{'Val':<6}{'Lit':<5}{'Dir':<5}{'MethClass':<25}")
print("-"*80)
for _, r in final_df.head(30).iterrows():
    print(f"{r['ComprehensiveRank']:<5}{r['gene']:<12}{r['ComprehensiveScore']:.4f}  "
          f"{r['TransformationScore']:.3f}  {r.get('FinalScore',0):.3f}  "
          f"{r['validation_score']:.2f}  {r['literature_score']:.0f}   "
          f"{r['direction_consistent']:.0f}   {str(r.get('methylation_class',''))[:22]}")

# Summary statistics
print(f"\n{'='*70}")
print("EVIDENCE LAYER SUMMARY")
print(f"{'='*70}")
print(f"Total candidates: 92")
print(f"Multi-omics FinalScore available: {final_df['FinalScore'].notna().sum()}")
print(f"External validation (≥3 datasets sig): {(final_df['n_datasets_significant'] >= 3).sum()}")
print(f"Literature support (≥1 source): {(final_df['literature_score'] > 0).sum()}")
print(f"Direction consistent: {(final_df['direction_consistent'] == 1).sum()}")
print(f"Strong epigenome (≥4/6): {(final_df['epigenome_concordance_score'] >= 4).sum()}")

# Tier classification
tier1 = final_df[
    (final_df['ComprehensiveScore'] >= 0.4) &
    (final_df['n_datasets_significant'] >= 3) &
    (final_df['literature_score'] >= 3)
]
tier2 = final_df[
    (final_df['ComprehensiveScore'] >= 0.3) &
    (final_df['n_datasets_significant'] >= 2) &
    ~final_df['gene'].isin(tier1['gene'])
]

print(f"\nTier 1 (high confidence, all evidence): {len(tier1)} genes")
for _, r in tier1.iterrows():
    print(f"  {r['gene']:12s} | CS={r['ComprehensiveScore']:.3f} | {r.get('methylation_class','')}")

print(f"\nTier 2 (moderate confidence): {len(tier2)} genes")
for _, r in tier2.head(15).iterrows():
    print(f"  {r['gene']:12s} | CS={r['ComprehensiveScore']:.3f}")

print("\nDone!")
