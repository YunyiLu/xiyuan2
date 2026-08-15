"""Prepare GSE78523 gene-level expression matrix with sample metadata."""
import gzip
import pandas as pd
import numpy as np
from pathlib import Path

SOFT_PATH = Path(r"C:/FDU/Y4S2/xiyuan/project/dataset/GPL18990_family.soft.gz")
MATRIX_PATH = Path(r"C:/FDU/Y4S2/xiyuan/project/dataset/GSE78523_series_matrix.txt.gz")
OUT_PATH = Path(r"C:/FDU/Y4S2/xiyuan/project/script3/data/gse78523_gene_expr.csv")


def parse_probe_to_gene(soft_path):
    """Parse GPL18990 SOFT file to get probe ID -> gene symbol mapping."""
    mapping = {}
    in_table = False
    with gzip.open(soft_path, "rt") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                next(f)  # skip header
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                cols = line.strip().split("\t")
                if len(cols) > 14:
                    probe_id = cols[0]
                    gene_sym = cols[14].strip()
                    if not gene_sym or gene_sym == "---":
                        continue
                    if "///" in gene_sym:
                        continue  # skip ambiguous multi-mapped probes
                    mapping[probe_id] = gene_sym
    return mapping


def parse_series_matrix(matrix_path):
    """Parse series matrix file, return expression df and sample metadata."""
    metadata_lines = {}
    data_lines = []
    header = None
    with gzip.open(matrix_path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_"):
                key = line.split("\t")[0]
                metadata_lines.setdefault(key, []).append(line.strip().split("\t")[1:])
            elif line.startswith('"ID_REF"'):
                header = [h.strip('"') for h in line.strip().split("\t")]
            elif header and not line.startswith("!") and line.strip():
                data_lines.append(line.strip().split("\t"))

    samples = header[1:]
    expr_data = {}
    for row in data_lines:
        probe = row[0].strip('"')
        vals = [float(v) if v not in ("", "null", "NA") else np.nan for v in row[1:]]
        expr_data[probe] = vals

    expr_df = pd.DataFrame(expr_data, index=samples)

    # Parse sample characteristics for group info
    meta = {}
    title_key = "!Sample_title"
    char_key = "!Sample_characteristics_ch1"

    if title_key in metadata_lines:
        titles = [t[0].strip('"') for t in metadata_lines[title_key]]
        for i, sample in enumerate(samples):
            meta[sample] = {"title": titles[i] if i < len(titles) else ""}

    if char_key in metadata_lines:
        for row in metadata_lines[char_key]:
            for i, val in enumerate(row):
                val = val.strip('"')
                if i < len(samples):
                    meta.setdefault(samples[i], {})
                    if ":" in val:
                        k, v = val.split(":", 1)
                        meta[samples[i]][k.strip()] = v.strip()

    return expr_df, meta, samples


def assign_groups(meta, samples):
    """Assign group, im_type, progression_status from metadata."""
    groups = []
    for s in samples:
        info = meta.get(s, {})
        title = info.get("title", "").lower()
        disease = info.get("disease state", info.get("disease", "")).lower()
        tissue = info.get("tissue", "").lower()

        # Determine group from available metadata
        group = "Unknown"
        im_type = "Unknown"
        progression = "Unknown"

        combined = f"{title} {disease} {tissue}"

        if "healthy" in combined or "normal" in combined:
            group = "Healthy"
            im_type = "None"
            progression = "Non-progressor"
        elif "iim" in combined or "intestinal" in combined:
            im_type = "IIM"
            if "progressor" in combined or "gc" in combined or "cancer" in combined:
                group = "IIM_GC_progressor"
                progression = "Progressor"
            else:
                group = "IIM_control"
                progression = "Non-progressor"
        elif "cim" in combined or "colonic" in combined or "complete" in combined:
            im_type = "CIM"
            if "progressor" in combined or "gc" in combined or "cancer" in combined:
                group = "CIM_GC_progressor"
                progression = "Progressor"
            else:
                group = "CIM_control"
                progression = "Non-progressor"

        groups.append({"group": group, "im_type": im_type, "progression_status": progression})
    return pd.DataFrame(groups, index=samples)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: probe -> gene mapping
    probe2gene = parse_probe_to_gene(SOFT_PATH)
    print(f"Parsed {len(probe2gene)} probe-gene mappings")

    # Step 2: expression matrix
    expr_df, meta, samples = parse_series_matrix(MATRIX_PATH)
    print(f"Expression matrix: {expr_df.shape[0]} samples x {expr_df.shape[1]} probes")

    # Step 3: filter to mapped probes and collapse by gene (mean)
    mapped_probes = [p for p in expr_df.columns if p in probe2gene]
    expr_mapped = expr_df[mapped_probes].copy()
    expr_mapped.columns = [probe2gene[p] for p in mapped_probes]

    # Collapse duplicate genes by mean (transpose, groupby, transpose back)
    gene_expr = expr_mapped.T.groupby(level=0).mean().T
    print(f"Gene-level matrix: {gene_expr.shape[0]} samples x {gene_expr.shape[1]} genes")

    # Step 4: metadata
    meta_df = assign_groups(meta, samples)

    # Step 5: combine and save
    result = pd.concat([meta_df, gene_expr], axis=1)
    result.index.name = "sample_id"
    result.to_csv(OUT_PATH)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
