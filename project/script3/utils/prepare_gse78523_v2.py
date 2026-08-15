"""
Preprocess GSE78523: GPL18990 series matrix → gene-level expression + metadata.
Input: dataset/GSE78523_series_matrix.txt.gz + dataset/GPL18990_probe2gene.json
Output: dataset/GEO_bulk/GSE78523/GSE78523_expression.csv (gene × sample)
        dataset/GEO_bulk/GSE78523/GSE78523_metadata.csv
"""
import os, sys, gzip, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATASET = "C:/FDU/Y4S2/xiyuan/project/dataset"
OUT_DIR = f"{DATASET}/GEO_bulk/GSE78523"
META_PATH = f"{DATASET}/metadata/GSE78523_samples_parsed.tsv"


def main():
    print("Preprocessing GSE78523...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load probe→gene from JSON
    probe2gene = json.load(open(f"{DATASET}/GPL18990_probe2gene.json"))
    print(f"  Probe→gene mappings: {len(probe2gene)}")

    # Parse series matrix
    matrix_path = f"{DATASET}/GSE78523_series_matrix.txt.gz"
    sample_ids = []
    probe_ids = []
    expr_rows = []

    with gzip.open(matrix_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('"ID_REF"'):
                continue
            elif line.startswith('!series_matrix_table_end'):
                break
            elif not line.startswith('!') and not line.startswith('^') and line.strip():
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    probe_id = parts[0].strip('"')
                    vals = []
                    for v in parts[1:]:
                        v = v.strip('"').strip()
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(np.nan)
                    probe_ids.append(probe_id)
                    expr_rows.append(vals)

    expr_df = pd.DataFrame(expr_rows, index=probe_ids, columns=sample_ids)
    print(f"  Raw: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")

    # Probe → gene (max mean probe per gene)
    expr_df['gene'] = expr_df.index.map(lambda x: probe2gene.get(x, ''))
    expr_df = expr_df[expr_df['gene'] != '']
    expr_df['mean_expr'] = expr_df[sample_ids].mean(axis=1)
    expr_df = expr_df.sort_values('mean_expr', ascending=False).drop_duplicates(subset='gene', keep='first')
    gene_expr = expr_df[sample_ids].copy()
    gene_expr.index = expr_df['gene'].values
    gene_expr.index.name = 'gene'
    print(f"  Gene-level: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")

    # Metadata from parsed TSV
    meta = pd.read_csv(META_PATH, sep='\t')
    meta = meta.rename(columns={'accession': 'sample_id', 'status': 'group'})
    # Standardize group names for downstream scripts
    group_map = {
        'Healthy Control': 'Healthy',
        'IIM Control': 'IIM_ctrl',
        'IIM-GC': 'IIM_GC_progressor',
        'CIM Control': 'CIM_ctrl',
        'CIM-GC': 'CIM_GC_progressor',
    }
    meta['group'] = meta['group'].map(group_map).fillna(meta['group'])
    meta.to_csv(f"{OUT_DIR}/GSE78523_metadata.csv", index=False)

    # Save expression
    gene_expr.to_csv(f"{OUT_DIR}/GSE78523_expression.csv")

    print(f"  Groups: {meta['group'].value_counts().to_dict()}")
    print(f"  Saved to {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
