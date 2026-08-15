#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze why gene intersection is low and propose solutions."""

import scanpy as sc
import pandas as pd
import numpy as np

print("Loading final merged dataset...")
adata = sc.read_h5ad('data/adata_raw_unintegrated.h5ad')

print(f"\n=== Final dataset stats ===")
print(f"Cells: {adata.shape[0]}")
print(f"Genes: {adata.shape[1]}")
print(f"\nDataset breakdown:")
print(adata.obs['dataset'].value_counts().sort_index())

# Check what genes are present
print(f"\n=== Gene annotation check ===")
print(f"Gene names (first 20):")
print(list(adata.var_names[:20]))
print(f"\nGene names (last 20):")
print(list(adata.var_names[-20:]))

# Check if genes start with ENSG (Ensembl IDs)
ensembl_genes = sum(1 for g in adata.var_names if g.startswith('ENSG'))
print(f"\nGenes starting with ENSG: {ensembl_genes}")

# Load the original datasets to compare
print("\n=== Reading original dataset metadata ===")

# GSE134520
import os
gse134520_path = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE134520"
sample_dirs = [d for d in os.listdir(gse134520_path) if os.path.isdir(os.path.join(gse134520_path, d))]
if sample_dirs:
    sample_path = os.path.join(gse134520_path, sample_dirs[0])
    genes_file = os.path.join(sample_path, "genes.tsv")
    if os.path.exists(genes_file):
        genes_134520 = pd.read_csv(genes_file, sep='\t', header=None)
        print(f"\nGSE134520 genes file sample (first 10):")
        print(genes_134520.head(10))
        print(f"Total genes in GSE134520: {len(genes_134520)}")
        print(f"Column 0 (Ensembl): {genes_134520[0].iloc[0]}")
        print(f"Column 1 (Symbol): {genes_134520[1].iloc[0]}")

# GSE183904
gse183904_path = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE183904"
sample_dirs_183904 = [d for d in os.listdir(gse183904_path) if d.startswith('GSM')]
if sample_dirs_183904:
    sample_path_183904 = os.path.join(gse183904_path, sample_dirs_183904[0])
    features_file = os.path.join(sample_path_183904, "features.tsv.gz")
    if os.path.exists(features_file):
        genes_183904 = pd.read_csv(features_file, sep='\t', header=None, compression='gzip')
        print(f"\nGSE183904 features file sample (first 10):")
        print(genes_183904.head(10))
        print(f"Total genes in GSE183904: {len(genes_183904)}")

# Check gene unification mapping
mapping_path = "data/gene_unification_mapping.json"
if os.path.exists(mapping_path):
    import json
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    print(f"\n=== Gene unification mapping stats ===")
    print(f"GSE134520 renames: {len(mapping.get('g134520_to_unified', {}))}")
    print(f"OMIX renames: {len(mapping.get('omix_to_unified', {}))}")
    print(f"GSE183904 renames: {len(mapping.get('g183904_to_unified', {}))}")

    if mapping.get('g134520_to_unified'):
        print(f"\nSample GSE134520 mappings (first 10):")
        for k, v in list(mapping['g134520_to_unified'].items())[:10]:
            print(f"  {k} -> {v}")
