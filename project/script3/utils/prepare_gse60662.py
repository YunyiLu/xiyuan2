"""
Preprocess GSE60662: GPL13497 series matrix → gene-level expression + metadata.
Input: dataset/GEO_bulk/GSE60662_series_matrix.txt.gz + dataset/GPL13497_probe2gene.json
Output: dataset/GEO_bulk/GSE60662/GSE60662_expression.csv (gene × sample)
        dataset/GEO_bulk/GSE60662/GSE60662_metadata.csv
Groups from titles: control/mild gastritis/severe gastritis/intestinal metaplasia (4 each)
"""
import os, sys, gzip, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATASET = "C:/FDU/Y4S2/xiyuan/project/dataset"
OUT_DIR = f"{DATASET}/GEO_bulk/GSE60662"


def main():
    print("Preprocessing GSE60662...")
    os.makedirs(OUT_DIR, exist_ok=True)

    probe2gene = json.load(open(f"{DATASET}/GPL13497_probe2gene.json"))
    print(f"  Probe→gene mappings: {len(probe2gene)}")

    matrix_path = f"{DATASET}/GEO_bulk/GSE60662_series_matrix.txt.gz"
    sample_ids = []
    sample_titles = []
    probe_ids = []
    expr_rows = []

    with gzip.open(matrix_path, 'rt', encoding='utf-8', errors='replace') as f:
        try:
            for line in f:
                if line.startswith('!Sample_geo_accession'):
                    sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
                elif line.startswith('!Sample_title'):
                    sample_titles = [s.strip('"') for s in line.strip().split('\t')[1:]]
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
        except EOFError:
            print("  NOTE: Truncated gz file, using data read so far")

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

    # Metadata from titles: "control rep1", "mild gastritis rep1", etc.
    meta_rows = []
    for i, sid in enumerate(sample_ids):
        title = sample_titles[i] if i < len(sample_titles) else ''
        title_lower = title.lower()
        if 'control' in title_lower:
            stage = 'normal'
        elif 'mild' in title_lower:
            stage = 'gastritis'
        elif 'severe' in title_lower:
            stage = 'gastritis'
        elif 'metaplasia' in title_lower or 'im' in title_lower:
            stage = 'IM'
        else:
            stage = 'unknown'
        stage_num_map = {'normal': 0, 'gastritis': 1, 'IM': 2}
        meta_rows.append({
            'sample_id': sid, 'title': title, 'stage': stage,
            'stage_num': stage_num_map.get(stage, -1),
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(f"{OUT_DIR}/GSE60662_metadata.csv", index=False)
    gene_expr.to_csv(f"{OUT_DIR}/GSE60662_expression.csv")

    print(f"  Stages: {meta_df['stage'].value_counts().to_dict()}")
    print(f"  Saved to {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
