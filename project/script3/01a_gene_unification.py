"""
Phase A Step 1: Gene Name Unification
Resolves naming inconsistencies across four datasets before taking intersection.

Results:
  Original 3-way intersection: 16,948 genes
  After unification (3-way):   21,484 genes (+4,536, +26.8%)
  After adding GSE183904 (4-way): TBD

Strategy:
  1. Ensembl ID bridge: OMIX, GSE249874, GSE183904 share Ensembl IDs
     -> detect symbol renames (RP11->AL/AC, old->new GENCODE)
  2. GSE134520 shares naming convention -> apply same Ensembl bridge
  3. mygene alias query: C*orf*, old HGNC symbols -> current official symbols
"""

import gzip
import json
import os
import sys

import mygene

DATA_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset"
OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/script3/data"
os.makedirs(OUT_DIR, exist_ok=True)

mg = mygene.MyGeneInfo()

# === 1. Read features with Ensembl IDs ===
print("=== Step 1: Reading features.tsv.gz files ===")

ens_to_sym_249874 = {}
with gzip.open(f"{DATA_DIR}/GSE249874_raw_feature_features.tsv.gz", "rt") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            ens_to_sym_249874[parts[0]] = parts[1]

ens_to_sym_omix = {}
with gzip.open(
    f"{DATA_DIR}/OMIX010346/Stomach_cancer/scRNA/GP4/features.tsv.gz", "rt"
) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            ens_to_sym_omix[parts[0]] = parts[1]

# GSE183904: 10X per-sample directories, read first sample for features
ens_to_sym_183904 = {}
gse183_dir = f"{DATA_DIR}/GSE183904"
if os.path.exists(gse183_dir):
    sample_dirs = sorted([d for d in os.listdir(gse183_dir)
                          if d.startswith('GSM')
                          and os.path.isdir(f"{gse183_dir}/{d}")])
    if sample_dirs:
        feat_path = f"{gse183_dir}/{sample_dirs[0]}/features.tsv.gz"
        if os.path.exists(feat_path):
            with gzip.open(feat_path, "rt") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ens_to_sym_183904[parts[0]] = parts[1]
        print(f"  GSE183904: {len(ens_to_sym_183904)} genes (Ensembl-mapped)")
    else:
        print("  GSE183904: no sample directories found, run 00_download_gse183904.py")
else:
    print(f"  GSE183904 not found at {gse183_dir}, run 00_download_gse183904.py first")

# GSE134520: text matrix with gene names as row index (no Ensembl IDs)
import csv

genes_134520 = set()
raw_dir = "C:/FDU/Y4S2/xiyuan/project/data/raw/GSE134520"
for fname in os.listdir(raw_dir):
    if fname.endswith(".txt"):
        fpath = os.path.join(raw_dir, fname)
        with open(fpath, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if row:
                    genes_134520.add(row[0])
        break  # all files have same gene set

print(f"  GSE249874: {len(ens_to_sym_249874)} genes (Ensembl-mapped)")
print(f"  OMIX010346: {len(ens_to_sym_omix)} genes (Ensembl-mapped)")
print(f"  GSE183904: {len(ens_to_sym_183904)} genes (Ensembl-mapped)")
print(f"  GSE134520: {len(genes_134520)} genes (symbol only)")

# === 2. Build OMIX -> Unified mapping via Ensembl bridge ===
print("\n=== Step 2: Ensembl ID bridge (OMIX -> GSE249874 naming) ===")

omix_to_unified = {}
shared_ens = set(ens_to_sym_249874.keys()) & set(ens_to_sym_omix.keys())
for ens in shared_ens:
    sym249 = ens_to_sym_249874[ens]
    sym_omix = ens_to_sym_omix[ens]
    if sym249 != sym_omix:
        omix_to_unified[sym_omix] = sym249

print(f"  Shared Ensembl IDs: {len(shared_ens)}")
print(f"  Symbol mismatches (renamed): {len(omix_to_unified)}")

# Build GSE183904 -> Unified mapping
g183904_to_unified = {}
if ens_to_sym_183904:
    shared_ens_183 = set(ens_to_sym_249874.keys()) & set(ens_to_sym_183904.keys())
    for ens in shared_ens_183:
        sym249 = ens_to_sym_249874[ens]
        sym_183 = ens_to_sym_183904[ens]
        if sym249 != sym_183:
            g183904_to_unified[sym_183] = sym249
    print(f"  GSE183904 shared Ensembl IDs: {len(shared_ens_183)}")
    print(f"  GSE183904 symbol mismatches (renamed): {len(g183904_to_unified)}")
else:
    print(f"  GSE183904 mapping skipped (data not available)")

# === 3. GSE134520 -> Unified (shares naming with OMIX) ===
print("\n=== Step 3: GSE134520 -> Unified (via OMIX bridge) ===")

g134520_to_unified = {g: omix_to_unified[g] for g in genes_134520 if g in omix_to_unified}
print(f"  Direct Ensembl-bridge rescues: {len(g134520_to_unified)}")

# === 4. mygene alias resolution for remaining genes ===
print("\n=== Step 4: mygene alias resolution ===")

genes_249874 = set(ens_to_sym_249874.values())
genes_omix_set = set(ens_to_sym_omix.values())
unified_omix = set(omix_to_unified.get(g, g) for g in genes_omix_set)
unified_134520_v1 = set(g134520_to_unified.get(g, g) for g in genes_134520)
intersection_v1 = unified_134520_v1 & genes_249874 & unified_omix

still_unmapped = unified_134520_v1 - intersection_v1
candidates = [g for g in sorted(still_unmapped)
              if not any(g.startswith(p) for p in
                         ["RP11-", "RP5-", "RP4-", "RP1-", "AC0", "AL",
                          "AP0", "AF", "CTD", "CTC", "CTB", "XXbac", "LINC"])]

print(f"  Protein-coding candidates for mygene: {len(candidates)}")
results = mg.querymany(candidates, scopes="symbol,alias", fields="symbol",
                       species="human", returnall=True)

extra_mapped = {}
for r in results["out"]:
    if "symbol" in r and "query" in r:
        official = r["symbol"]
        query = r["query"]
        if official != query and official in genes_249874 and official in unified_omix:
            extra_mapped[query] = official

print(f"  mygene alias rescues: {len(extra_mapped)}")
g134520_to_unified.update(extra_mapped)

# === 5. Final intersection ===
print("\n=== Step 5: Final Results ===")

unified_134520_final = set(g134520_to_unified.get(g, g) for g in genes_134520)
genes_183904_set = set(ens_to_sym_183904.values()) if ens_to_sym_183904 else set()
unified_183904 = set(g183904_to_unified.get(g, g) for g in genes_183904_set) if genes_183904_set else set()

# 3-way intersection (original)
intersection_3way = sorted(unified_134520_final & genes_249874 & unified_omix)

# 4-way intersection (with GSE183904)
if unified_183904:
    intersection_4way = sorted(unified_134520_final & genes_249874 & unified_183904 & unified_omix)
    print(f"  Original 3-way intersection:  16,948 genes")
    print(f"  Unified 3-way intersection:   {len(intersection_3way):,} genes")
    print(f"  Unified 4-way intersection:   {len(intersection_4way):,} genes")
    print(f"  Loss from adding GSE183904:   {len(intersection_3way) - len(intersection_4way):,} genes "
          f"({(len(intersection_3way) - len(intersection_4way)) / len(intersection_3way) * 100:.1f}%)")
    final_intersection = intersection_4way
else:
    print(f"  Original 3-way intersection:  16,948 genes")
    print(f"  Unified 3-way intersection:   {len(intersection_3way):,} genes")
    print(f"  Recovered:                    +{len(intersection_3way)-16948:,} "
          f"({(len(intersection_3way)-16948)/16948*100:.1f}%)")
    final_intersection = intersection_3way

# === 6. Save outputs ===
print("\n=== Step 6: Saving outputs ===")

with open(f"{OUT_DIR}/unified_intersection_genes.txt", "w") as f:
    f.write("\n".join(final_intersection))

mapping_output = {
    "omix_to_unified": omix_to_unified,
    "g134520_to_unified": g134520_to_unified,
    "g183904_to_unified": g183904_to_unified,
    "stats": {
        "original_3way_intersection": 16948,
        "unified_3way_intersection": len(intersection_3way),
        "unified_4way_intersection": len(final_intersection) if unified_183904 else None,
        "recovered_3way": len(intersection_3way) - 16948,
        "improvement_pct": round((len(intersection_3way) - 16948) / 16948 * 100, 1),
        "loss_from_gse183904": (len(intersection_3way) - len(final_intersection)) if unified_183904 else 0,
        "ensembl_bridge_renames_omix": len(omix_to_unified),
        "ensembl_bridge_renames_183904": len(g183904_to_unified),
        "mygene_alias_renames": len(extra_mapped),
    },
}

with open(f"{OUT_DIR}/gene_unification_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping_output, f, ensure_ascii=False)

print(f"  Saved: {OUT_DIR}/unified_intersection_genes.txt")
print(f"  Saved: {OUT_DIR}/gene_unification_mapping.json")
print("\nDone. Use unified_intersection_genes.txt in 01_multi_dataset_qc.py")
