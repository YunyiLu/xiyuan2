#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep analysis of gene intersection across 4 datasets."""

import scanpy as sc
import json
from collections import Counter

# Load gene counts from temp files
print('=== Gene counts per dataset ===')
adata_134520 = sc.read_h5ad('data/tmp_gse134520.h5ad')
print(f'GSE134520: {len(adata_134520.var_names)} genes')
genes_134520 = set(adata_134520.var_names)

adata_249874 = sc.read_h5ad('data/tmp_gse249874.h5ad')
print(f'GSE249874: {len(adata_249874.var_names)} genes')
genes_249874 = set(adata_249874.var_names)

adata_183904 = sc.read_h5ad('data/tmp_gse183904.h5ad')
print(f'GSE183904: {len(adata_183904.var_names)} genes')
genes_183904 = set(adata_183904.var_names)

adata_omix = sc.read_h5ad('data/tmp_omix.h5ad')
print(f'OMIX010346: {len(adata_omix.var_names)} genes')
genes_omix = set(adata_omix.var_names)

print('\n=== 2-way intersections ===')
print(f'134520 & 249874: {len(genes_134520 & genes_249874)}')
print(f'134520 & 183904: {len(genes_134520 & genes_183904)}')
print(f'134520 & OMIX: {len(genes_134520 & genes_omix)}')
print(f'249874 & 183904: {len(genes_249874 & genes_183904)}')
print(f'249874 & OMIX: {len(genes_249874 & genes_omix)}')
print(f'183904 & OMIX: {len(genes_183904 & genes_omix)}')

print('\n=== 3-way intersections ===')
print(f'134520 & 249874 & 183904: {len(genes_134520 & genes_249874 & genes_183904)}')
print(f'134520 & 249874 & OMIX: {len(genes_134520 & genes_249874 & genes_omix)}')
print(f'134520 & 183904 & OMIX: {len(genes_134520 & genes_183904 & genes_omix)}')
print(f'249874 & 183904 & OMIX: {len(genes_249874 & genes_183904 & genes_omix)}')

print('\n=== 4-way intersection ===')
intersection_4way = genes_134520 & genes_249874 & genes_183904 & genes_omix
print(f'All 4 datasets: {len(intersection_4way)}')

print('\n=== Dataset-specific genes ===')
print(f'GSE134520 only: {len(genes_134520 - genes_249874 - genes_183904 - genes_omix)}')
print(f'GSE249874 only: {len(genes_249874 - genes_134520 - genes_183904 - genes_omix)}')
print(f'GSE183904 only: {len(genes_183904 - genes_134520 - genes_249874 - genes_omix)}')
print(f'OMIX only: {len(genes_omix - genes_134520 - genes_249874 - genes_183904)}')

print('\n=== Union-based stats ===')
all_genes = list(genes_134520) + list(genes_249874) + list(genes_183904) + list(genes_omix)
gene_counts = Counter(all_genes)
at_least_3 = [g for g, c in gene_counts.items() if c >= 3]
at_least_2 = [g for g, c in gene_counts.items() if c >= 2]
print(f'Present in >=3 datasets: {len(at_least_3)} genes')
print(f'Present in >=2 datasets: {len(at_least_2)} genes')
print(f'Total gene union: {len(gene_counts)} genes')

print('\n=== Lost genes analysis ===')
# What genes are lost from GSE134520?
lost_from_134520 = genes_134520 - intersection_4way
print(f'Genes lost from GSE134520: {len(lost_from_134520)}')
print(f'  Missing in GSE249874: {len(lost_from_134520 & (genes_134520 - genes_249874))}')
print(f'  Missing in GSE183904: {len(lost_from_134520 & (genes_134520 - genes_183904))}')
print(f'  Missing in OMIX: {len(lost_from_134520 & (genes_134520 - genes_omix))}')

print('\nFirst 20 lost genes from GSE134520:')
print(list(lost_from_134520)[:20])
