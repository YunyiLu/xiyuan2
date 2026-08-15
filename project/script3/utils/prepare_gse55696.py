"""
Preprocess GSE55696: Agilent GPL6480 series matrix → gene-level expression + metadata.
Input: dataset/GSE55696_series_matrix.txt.gz + dataset/GPL6480.annot.gz
Output: dataset/GEO_bulk/GSE55696/GSE55696_expression.csv (gene × sample)
        dataset/GEO_bulk/GSE55696/GSE55696_metadata.csv (sample_id, stage, stage_num)
"""
import os, sys, gzip
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATASET = "C:/FDU/Y4S2/xiyuan/project/dataset"
OUT_DIR = f"{DATASET}/GEO_bulk/GSE55696"


def load_gpl6480_annotation():
    """Load GPL6480 probe → gene symbol mapping."""
    path = f"{DATASET}/GPL6480.annot.gz"
    # Skip comment lines (start with ! or #), find table after !platform_table_begin
    rows = []
    in_table = False
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!platform_table_end'):
                break
            if in_table:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] != 'ID':
                    probe_id = parts[0]
                    gene_symbol = parts[2].strip() if parts[2].strip() else ''
                    if gene_symbol and gene_symbol != '---':
                        rows.append((probe_id, gene_symbol))
            if line.startswith('!platform_table_begin'):
                in_table = True
                next_line = f.readline()  # header line
                in_table = True
    return dict(rows)


def load_series_matrix():
    """Load GSE55696 series matrix: extract metadata + expression."""
    path = f"{DATASET}/GSE55696_series_matrix.txt.gz"
    sample_ids = []
    sample_titles = []
    disease_states = []
    expr_rows = []
    probe_ids = []

    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_title'):
                sample_titles = [s.strip('"') for s in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1') and 'disease state' in line:
                disease_states = [s.strip('"').replace('disease state: ', '') for s in line.strip().split('\t')[1:]]
            elif line.startswith('"ID_REF"'):
                # Data header - skip, already have sample_ids
                continue
            elif line.startswith('!series_matrix_table_end'):
                break
            elif not line.startswith('!') and not line.startswith('^'):
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    probe_id = parts[0].strip('"')
                    values = []
                    for v in parts[1:]:
                        v = v.strip('"').strip()
                        try:
                            values.append(float(v))
                        except ValueError:
                            values.append(np.nan)
                    probe_ids.append(probe_id)
                    expr_rows.append(values)

    expr_df = pd.DataFrame(expr_rows, index=probe_ids, columns=sample_ids)

    # Build metadata
    stage_map = {
        'chronic gastritis': 'CG',
        'gastric low-grade intraepithelial neoplasia': 'LGIN',
        'gastric high-grade intraepithelial neoplasia': 'HGIN',
        'gastric early-stage adenocarcinoma': 'EGC',
    }
    stage_num_map = {'CG': 0, 'LGIN': 1, 'HGIN': 2, 'EGC': 3}

    meta_rows = []
    for i, sid in enumerate(sample_ids):
        raw_state = disease_states[i] if i < len(disease_states) else ''
        stage = stage_map.get(raw_state, raw_state)
        meta_rows.append({
            'sample_id': sid,
            'title': sample_titles[i] if i < len(sample_titles) else '',
            'stage': stage,
            'stage_num': stage_num_map.get(stage, -1),
        })
    meta_df = pd.DataFrame(meta_rows)
    return expr_df, meta_df


def probe_to_gene(expr_df, probe2gene):
    """Collapse probes to genes: take max mean expression probe per gene."""
    expr_df = expr_df.copy()
    expr_df['gene'] = expr_df.index.map(lambda x: probe2gene.get(x, ''))
    expr_df = expr_df[expr_df['gene'] != '']

    # For multi-probe genes, keep the probe with highest mean expression
    expr_df['mean_expr'] = expr_df.drop(columns='gene').mean(axis=1)
    expr_df = expr_df.sort_values('mean_expr', ascending=False).drop_duplicates(subset='gene', keep='first')
    expr_df = expr_df.drop(columns=['gene', 'mean_expr'])
    expr_df.index = expr_df.index.map(lambda x: probe2gene[x])
    expr_df.index.name = 'gene'
    return expr_df


def main():
    print("Preprocessing GSE55696...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load annotation
    print("  Loading GPL6480 annotation...")
    probe2gene = load_gpl6480_annotation()
    print(f"  Probes with gene symbol: {len(probe2gene)}")

    # Load series matrix
    print("  Loading series matrix...")
    expr_df, meta_df = load_series_matrix()
    print(f"  Raw: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")

    # Probe → gene
    print("  Collapsing probes to genes...")
    gene_expr = probe_to_gene(expr_df, probe2gene)
    print(f"  Gene-level: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")

    # Verify stage distribution
    print(f"  Stage distribution:")
    for stage, n in meta_df['stage'].value_counts().items():
        print(f"    {stage}: {n}")

    # Save
    gene_expr.to_csv(f"{OUT_DIR}/GSE55696_expression.csv")
    meta_df.to_csv(f"{OUT_DIR}/GSE55696_metadata.csv", index=False)
    print(f"  Saved to {OUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
