"""
Preprocess GSE62254 (ACRG): GPL570 series matrix → gene-level expression + survival.
Input: dataset/GEO_bulk/GSE62254_series_matrix.txt.gz + dataset/GPL570_data.txt
Output: dataset/GSE62254/GSE62254_expression.csv (gene × sample)
        dataset/GSE62254/GSE62254_survival.csv (sample_id, OS.time, OS)
Note: 300 gastric tumors, Affymetrix HG-U133 Plus 2.0
      Survival data must be extracted from sample characteristics or supplementary files.
"""
import os, sys, gzip
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATASET = "C:/FDU/Y4S2/xiyuan/project/dataset"
OUT_DIR = f"{DATASET}/GSE62254"


def load_gpl570_probe2gene():
    """Parse GPL570_data.txt: ID (col0) → Gene Symbol (col10)."""
    path = f"{DATASET}/GPL570_data.txt"
    mapping = {}
    in_table = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!platform_table_begin'):
                in_table = True
                next(f)  # skip header
                continue
            if line.startswith('!platform_table_end'):
                break
            if in_table:
                parts = line.strip().split('\t')
                if len(parts) > 10:
                    probe_id = parts[0]
                    gene_sym = parts[10].strip()
                    if gene_sym and gene_sym != '---':
                        # Handle "GENE1 /// GENE2" → take first
                        if ' /// ' in gene_sym:
                            gene_sym = gene_sym.split(' /// ')[0]
                        mapping[probe_id] = gene_sym
    return mapping


def main():
    print("Preprocessing GSE62254 (ACRG)...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load probe→gene
    print("  Loading GPL570 annotation...")
    probe2gene = load_gpl570_probe2gene()
    print(f"  Probe→gene mappings: {len(probe2gene)}")

    # Parse series matrix
    matrix_path = f"{DATASET}/GEO_bulk/GSE62254_series_matrix.txt.gz"
    sample_ids = []
    sample_titles = []
    char_lines = []
    probe_ids = []
    expr_rows = []

    with gzip.open(matrix_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_title'):
                sample_titles = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                char_lines.append([s.strip('"') for s in line.strip().split('\t')[1:]])
            elif line.startswith('"ID_REF"'):
                continue
            elif line.startswith('!series_matrix_table_end'):
                break
            elif not line.startswith('!') and not line.startswith('^') and line.strip():
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    probe_ids.append(parts[0].strip('"'))
                    vals = []
                    for v in parts[1:]:
                        try:
                            vals.append(float(v.strip('"')))
                        except ValueError:
                            vals.append(np.nan)
                    expr_rows.append(vals)

    expr_df = pd.DataFrame(expr_rows, index=probe_ids, columns=sample_ids)
    print(f"  Raw: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")

    # Probe → gene
    expr_df['gene'] = expr_df.index.map(lambda x: probe2gene.get(x, ''))
    expr_df = expr_df[expr_df['gene'] != '']
    expr_df['mean_expr'] = expr_df[sample_ids].mean(axis=1)
    expr_df = expr_df.sort_values('mean_expr', ascending=False).drop_duplicates(subset='gene', keep='first')
    gene_expr = expr_df[sample_ids].copy()
    gene_expr.index = expr_df['gene'].values
    gene_expr.index.name = 'gene'
    print(f"  Gene-level: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")

    # Extract survival from characteristics
    # Parse all characteristic lines to find OS/survival info
    sample_meta = {sid: {} for sid in sample_ids}
    for char_line in char_lines:
        for i, val in enumerate(char_line):
            if i < len(sample_ids) and ':' in val:
                key, value = val.split(':', 1)
                sample_meta[sample_ids[i]][key.strip()] = value.strip()

    # Try to extract survival (OS.time, OS event)
    surv_rows = []
    for sid in sample_ids:
        meta = sample_meta[sid]
        os_time = None
        os_event = None
        for k, v in meta.items():
            k_lower = k.lower()
            if 'survival' in k_lower and 'month' in k_lower:
                try:
                    os_time = float(v) * 30.44  # months → days
                except ValueError:
                    pass
            elif 'survival' in k_lower and 'time' in k_lower:
                try:
                    os_time = float(v)
                except ValueError:
                    pass
            elif 'vital' in k_lower or 'status' in k_lower or 'alive' in k_lower:
                v_lower = v.lower()
                if 'dead' in v_lower or 'deceased' in v_lower:
                    os_event = 1
                elif 'alive' in v_lower or 'living' in v_lower:
                    os_event = 0
                else:
                    try:
                        os_event = int(v)
                    except ValueError:
                        pass
        if os_time is not None and os_event is not None:
            surv_rows.append({'sample_id': sid, 'OS.time': os_time, 'OS': os_event})

    # Save expression
    gene_expr.to_csv(f"{OUT_DIR}/GSE62254_expression.csv")

    # Save survival
    if surv_rows:
        surv_df = pd.DataFrame(surv_rows).set_index('sample_id')
        surv_df.to_csv(f"{OUT_DIR}/GSE62254_survival.csv")
        print(f"  Survival data: {len(surv_df)} samples")
    else:
        print("  WARNING: No survival data found in series matrix characteristics")
        print("  Survival may need to be extracted from supplementary files or GEO")
        # Create placeholder with sample_titles as index for downstream compatibility
        surv_df = pd.DataFrame({'OS.time': [np.nan]*len(sample_ids),
                                'OS': [np.nan]*len(sample_ids)}, index=sample_ids)
        surv_df.index.name = 'sample_id'
        surv_df.to_csv(f"{OUT_DIR}/GSE62254_survival.csv")

    print(f"  Saved to {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
