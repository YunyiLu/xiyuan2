#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Gene Unification - Quick Analysis
Analyze current gene intersection and check for potential improvements
"""

import pandas as pd
import numpy as np
import sys
import os

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Gene Intersection Analysis - Current State")
print("=" * 70)
print()

# ===========================================================================
# Step 1: Analyze current gene mapping status
# ===========================================================================

print("Step 1: Checking existing gene unification results...")
print("-" * 70)

mapping_file = "data/gene_unification_mapping.json"
if os.path.exists(mapping_file):
    import json
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    print(f"Found existing mapping: {mapping_file}")
    print()

    for dataset, genes in mapping.items():
        print(f"  {dataset}: {len(genes):,} gene mappings")
    print()
else:
    print("No existing mapping file found.")
    print()

# ===========================================================================
# Step 2: Check current merged data
# ===========================================================================

print("Step 2: Analyzing current merged dataset...")
print("-" * 70)

try:
    import scanpy as sc

    adata = sc.read_h5ad('data/adata_raw_unintegrated.h5ad')

    print(f"Current dataset:")
    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Total genes: {adata.n_vars:,}")
    print(f"  Datasets: {adata.obs['dataset'].nunique()}")
    print()

    # Check gene expression statistics
    mean_expr = adata.X.mean(axis=0)
    if hasattr(mean_expr, 'A1'):
        mean_expr = mean_expr.A1

    zero_genes = (mean_expr == 0).sum()
    low_expr_genes = (mean_expr < 0.001).sum()

    print(f"Gene expression quality:")
    print(f"  Zero expression genes: {zero_genes} ({zero_genes/adata.n_vars*100:.2f}%)")
    print(f"  Low expression genes (<0.001): {low_expr_genes} ({low_expr_genes/adata.n_vars*100:.2f}%)")
    print()

    # ===========================================================================
    # Step 3: Check for missing panel genes
    # ===========================================================================

    print("Step 3: Panel gene completeness check...")
    print("-" * 70)

    panel_genes = {
        'Core markers': ['OLFM4', 'REG4', 'CDX2', 'MUC5AC'],
        'IM markers': ['MUC2', 'TFF3', 'SPINK4', 'KRT20'],
        'Epithelial': ['CDH1', 'EPCAM', 'KRT8', 'KRT18'],
        'Proliferation': ['MKI67', 'PCNA', 'TOP2A'],
        'Mesenchymal': ['VIM', 'FN1', 'CDH2'],
        'Immune': ['PTPRC', 'CD3D', 'CD68'],
        'Endothelial': ['PECAM1', 'VWF']
    }

    total_panel = 0
    missing_panel = []

    for category, genes in panel_genes.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]

        total_panel += len(genes)
        missing_panel.extend(missing)

        status = f"{len(present)}/{len(genes)}"
        if missing:
            print(f"  {category:15s}: {status:6s} - Missing: {', '.join(missing)}")
        else:
            print(f"  {category:15s}: {status:6s} - All present")

    print()
    print(f"Overall panel completeness: {total_panel - len(missing_panel)}/{total_panel} " +
          f"({(total_panel - len(missing_panel))/total_panel*100:.1f}%)")
    print()

    if missing_panel:
        print(f"CRITICAL MISSING GENES: {', '.join(set(missing_panel))}")
        print()

    # ===========================================================================
    # Step 4: Gene name pattern analysis
    # ===========================================================================

    print("Step 4: Gene naming pattern analysis...")
    print("-" * 70)

    gene_names = adata.var_names.tolist()

    patterns = {
        'Standard (e.g., TP53)': sum(1 for g in gene_names if g.isupper() and '-' not in g and '.' not in g),
        'With dash (e.g., HLA-A)': sum(1 for g in gene_names if '-' in g),
        'With dot (e.g., MARCH1.1)': sum(1 for g in gene_names if '.' in g and not g.startswith('.')),
        'Ensembl-like (ENSG...)': sum(1 for g in gene_names if g.startswith('ENS')),
        'LOC/C#orf pattern': sum(1 for g in gene_names if g.startswith('LOC') or 'orf' in g.lower()),
        'Others': 0
    }

    patterns['Others'] = len(gene_names) - sum(patterns.values())

    for pattern, count in patterns.items():
        print(f"  {pattern:30s}: {count:6,} ({count/len(gene_names)*100:5.1f}%)")

    print()

    # ===========================================================================
    # Step 5: Recommendations
    # ===========================================================================

    print("=" * 70)
    print("Analysis Summary & Recommendations")
    print("=" * 70)
    print()

    if missing_panel:
        print("ISSUE FOUND: Critical panel genes are missing")
        print("-" * 70)
        print(f"Missing genes: {', '.join(set(missing_panel))}")
        print()
        print("Recommended Actions:")
        print()
        print("Option A: Try >=3 dataset strategy (RECOMMENDED)")
        print("  - Relax from 4-way intersection to >=3 datasets")
        print("  - May recover MUC2 and other missing genes")
        print("  - Expected gene count: ~22,000-23,000")
        print()
        print("Option B: Use Ensembl ID unification (BEST but requires internet)")
        print("  - Install: pip install mygene")
        print("  - Unify gene names via Ensembl ID")
        print("  - Expected recovery: 500-1,500 additional genes")
        print()
        print("Option C: Substitute missing markers")
        print("  - MUC2 -> Use TFF3 + SPINK4 + KRT20 combination")
        print("  - PECAM1 -> Use VWF (also endothelial marker)")
        print()
    else:
        print("GOOD NEWS: All critical panel genes are present!")
        print()
        print("Optimization Options (optional):")
        print()
        print("1. Flexible gene set strategy:")
        print("   - Phase 2A/4: Keep using 4-way (20,697 genes)")
        print("   - Phase 2B/8: Use per-dataset full gene sets")
        print("   - Phase 11: Use OMIX full genes for spatial")
        print()
        print("2. Marker discovery enhancement:")
        print("   - Discover in >=2 datasets")
        print("   - Validate in 4-way intersection")
        print("   - Maximize discovery power")
        print()

    # ===========================================================================
    # Step 6: Generate gene set files for flexible strategy
    # ===========================================================================

    if not missing_panel:
        print("Step 6: Generating files for flexible gene set strategy...")
        print("-" * 70)

        # Save current 4-way genes
        gene_df = pd.DataFrame({
            'gene_symbol': adata.var_names,
            'mean_expression': mean_expr,
            'in_4way': True
        })

        os.makedirs('data', exist_ok=True)
        gene_df.to_csv('data/genes_4way_current.csv', index=False)

        print(f"Saved: data/genes_4way_current.csv ({len(gene_df)} genes)")
        print()
        print("Next steps:")
        print("1. This file can be used as reference for 4-way intersection")
        print("2. For >=3 or >=2 strategies, need to re-process from raw data")
        print("3. Or continue with current setup (already very good)")
        print()

except FileNotFoundError:
    print("ERROR: data/adata_raw_unintegrated.h5ad not found")
    print()
    print("Please ensure Phase 1 (01_multi_dataset_qc.py) has been completed.")
    print()

except Exception as e:
    print(f"ERROR: {e}")
    print()
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("Analysis Complete")
print("=" * 70)
