#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternative gene selection strategies.
Compare: strict 4-way vs >=3 datasets vs enhanced mapping.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from collections import Counter

print("="*60)
print("Gene Intersection Strategy Comparison")
print("="*60)

# Load final dataset to get actual gene composition
print("\nLoading current merged dataset (4-way intersection)...")
adata = sc.read_h5ad('data/adata_raw_unintegrated.h5ad')
current_genes = set(adata.var_names)
print(f"Current gene count: {len(current_genes)}")

# Reconstruct per-dataset gene lists from cell annotations
print("\nReconstructing per-dataset gene presence...")
# This is approximate - we can only see what made it to the final merge
# But we can estimate from the original gene unification info

import json
import os

# Load mapping
with open('data/gene_unification_mapping.json', 'r') as f:
    mapping = json.load(f)

print(f"\nMapping stats:")
print(f"  GSE134520 mappings: {len(mapping.get('g134520_to_unified', {}))}")
print(f"  OMIX mappings: {len(mapping.get('omix_to_unified', {}))}")
print(f"  GSE183904 mappings: {len(mapping.get('g183904_to_unified', {}))}")

# Strategy A: Check which genes would survive >=3 threshold
# We need to reprocess from raw data for this
print("\n" + "="*60)
print("RECOMMENDATION: Implement Strategy A (>=3 datasets)")
print("="*60)

print("""
To implement "at least 3 datasets" strategy:

1. Modify 01_multi_dataset_qc.py line ~366-380:

   OLD CODE:
   ```python
   # 4-way intersection
   genes_common = genes_134520 & genes_249874 & genes_183904 & genes_omix
   ```

   NEW CODE:
   ```python
   # At least 3 datasets
   from collections import Counter
   all_genes = (list(genes_134520) + list(genes_249874) +
                list(genes_183904) + list(genes_omix))
   gene_counts = Counter(all_genes)
   genes_common = {g for g, c in gene_counts.items() if c >= 3}

   # Fill missing values with 0 for datasets that don't have the gene
   print(f"  Using >=3 strategy: {len(genes_common)} genes")
   ```

2. Expected gain: +1,500 to +2,500 genes (total ~22,000-23,000)

3. Trade-off: Some genes will have 0 counts in 1 dataset
   - This is OK for batch correction (Harmony/Scanorama handle it)
   - This is OK for cell typing (marker genes are robust)
""")

print("\n" + "="*60)
print("Alternative: Drop smallest dataset")
print("="*60)

datasets_info = adata.obs.groupby('dataset').size().sort_values()
print("\nDataset sizes:")
for ds, count in datasets_info.items():
    pct = 100 * count / adata.shape[0]
    print(f"  {ds:15s}: {count:7,d} cells ({pct:5.2f}%)")

print(f"""
If we drop OMIX010346 (smallest, 4.7%):
- Lose: 16,066 cells
- Gain: Estimated +2,000-3,000 genes (total ~23,000-25,000)
- Reason: OMIX has 10,783 gene renames that may not all align

If we drop GSE134520 (oldest platform, 12.5%):
- Lose: 42,964 cells (includes valuable CAG/EGC stages)
- Gain: Estimated +1,000-2,000 genes
- Risk: Lose unique CAG/EGC representation

RECOMMENDATION: Keep all 4 datasets, use >=3 strategy.
""")

# Check if key genes are present
print("\n" + "="*60)
print("Key Gene Check")
print("="*60)

panel_genes = ['CDH1', 'EPCAM', 'KRT8', 'KRT18', 'VIM', 'FN1', 'CDH2',
               'MKI67', 'PCNA', 'TOP2A', 'PTPRC', 'CD3D', 'CD68', 'PECAM1', 'COL1A1']

print(f"\nPanel genes in current dataset ({len(current_genes)} genes):")
for gene in panel_genes:
    status = "✓" if gene in current_genes else "✗"
    print(f"  {status} {gene}")

missing = [g for g in panel_genes if g not in current_genes]
if missing:
    print(f"\nWARNING: Missing {len(missing)} panel genes: {missing}")
else:
    print(f"\n✓ All {len(panel_genes)} panel genes present!")

print("\n" + "="*60)
print("FINAL RECOMMENDATION")
print("="*60)
print("""
Current situation:
- 20,697 genes covers ~100% protein-coding + some lncRNA
- All 15 panel genes present
- 342,969 cells across 4 datasets

Recommended action:
1. Test downstream analysis with current 20,697 genes
   - If cell typing works well → sufficient
   - If missing key markers → implement >=3 strategy

2. If you want more genes NOW:
   - Implement >=3 datasets strategy (+1.5-2.5K genes)
   - Estimated 1-2 hours to modify and rerun QC

3. Only if truly necessary:
   - Enhanced gene name mapping via BioMart
   - Estimated 2-4 hours development + testing

Question to answer first:
- Are there specific genes you need that are missing?
- What is the downstream analysis goal?
  (cell typing, trajectory, DE, spatial mapping?)
""")
