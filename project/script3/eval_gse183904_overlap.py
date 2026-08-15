"""
Evaluate GSE183904 gene overlap with existing 3 datasets.
Goal: Determine if adding GSE183904 to Phase 1 is feasible
      without significant gene loss.

Checks:
  1. Download GSE183904 feature list (gene names only, not full matrix)
  2. Compare with existing unified intersection (21,484 genes)
  3. Compute 4-way intersection size
  4. Report stage/sample metadata compatibility
"""
import os, sys, gzip, json, urllib.request, ssl
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DATA_DIR = f"{BASE}/data"
EVAL_DIR = f"{BASE}/data/eval_gse183904"
os.makedirs(EVAL_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_with_fallback(url, target):
    """Download with SSL fallback."""
    if os.path.exists(target):
        print(f"  [CACHED] {os.path.basename(target)}")
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            with open(target, 'wb') as f:
                f.write(resp.read())
        print(f"  Downloaded: {os.path.basename(target)} "
              f"({os.path.getsize(target)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def get_existing_gene_sets():
    """Load gene sets from existing datasets."""
    print("\n[1/4] Loading existing gene sets...")

    # Unified intersection (current)
    unified_path = f"{DATA_DIR}/unified_intersection_genes.txt"
    if os.path.exists(unified_path):
        with open(unified_path, 'r') as f:
            unified_genes = set(line.strip() for line in f if line.strip())
        print(f"  Unified intersection (current): {len(unified_genes):,} genes")
    else:
        unified_genes = None
        print(f"  WARNING: {unified_path} not found")

    # Individual dataset gene sets from features files
    dataset_dir = "C:/FDU/Y4S2/xiyuan/project/dataset"

    # GSE249874 genes
    genes_249874 = set()
    feat_249 = f"{dataset_dir}/GSE249874_raw_feature_features.tsv.gz"
    if os.path.exists(feat_249):
        with gzip.open(feat_249, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    genes_249874.add(parts[1])
        print(f"  GSE249874: {len(genes_249874):,} genes")
    else:
        print(f"  WARNING: GSE249874 features not found")

    # OMIX010346 genes
    genes_omix = set()
    feat_omix = f"{dataset_dir}/OMIX010346/Stomach_cancer/scRNA/GP4/features.tsv.gz"
    if os.path.exists(feat_omix):
        with gzip.open(feat_omix, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    genes_omix.add(parts[1])
        print(f"  OMIX010346: {len(genes_omix):,} genes")
    else:
        print(f"  WARNING: OMIX features not found")

    # GSE134520 genes (from first txt file)
    genes_134520 = set()
    raw_dir = "C:/FDU/Y4S2/xiyuan/project/data/raw/GSE134520"
    if os.path.exists(raw_dir):
        import csv
        for fname in os.listdir(raw_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(raw_dir, fname), 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = next(reader)
                    for row in reader:
                        if row:
                            genes_134520.add(row[0])
                break
        print(f"  GSE134520: {len(genes_134520):,} genes")
    else:
        print(f"  WARNING: GSE134520 raw dir not found")

    return unified_genes, genes_134520, genes_249874, genes_omix


def get_gse183904_genes():
    """Try to get GSE183904 gene list without downloading full matrix."""
    print("\n[2/4] Getting GSE183904 gene list...")

    # Strategy 1: Try GEO supplementary features file
    supp_base = ("https://ftp.ncbi.nlm.nih.gov/geo/series/"
                 "GSE183nnn/GSE183904/suppl/")

    # GSE183904 from Kumar et al. provides processed h5ad
    # Try to get just the gene list from supplementary
    possible_feature_files = [
        "GSE183904_GC_all_genes.tsv.gz",
        "GSE183904_genes.tsv.gz",
        "GSE183904_features.tsv.gz",
    ]

    genes_183904 = set()

    for fname in possible_feature_files:
        target = f"{EVAL_DIR}/{fname}"
        url = f"{supp_base}{fname}"
        if download_with_fallback(url, target):
            with gzip.open(target, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        genes_183904.add(parts[1])
                    elif parts:
                        genes_183904.add(parts[0])
            if genes_183904:
                print(f"  Got {len(genes_183904):,} genes from {fname}")
                return genes_183904

    # Strategy 2: Try to list supplementary files and find any gene/feature file
    print("  Direct feature files not found. Listing supplementary...")
    try:
        req = urllib.request.Request(supp_base,
                                    headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            listing = resp.read().decode('utf-8')
        import re
        files = re.findall(r'href="([^"]*GSE183904[^"]*)"', listing)
        if not files:
            files = re.findall(r'>(GSE183904[^<]+)<', listing)
        print(f"  Found {len(files)} supplementary files:")
        for f in files[:15]:
            print(f"    {f}")

        # Look for small files that might contain gene names
        for f in files:
            if any(k in f.lower() for k in ['gene', 'feature', 'var_name']):
                target = f"{EVAL_DIR}/{f}"
                if download_with_fallback(f"{supp_base}{f}", target):
                    try:
                        opener = gzip.open if f.endswith('.gz') else open
                        with opener(target, 'rt') as fh:
                            for line in fh:
                                parts = line.strip().split('\t')
                                if len(parts) >= 2:
                                    genes_183904.add(parts[1])
                                elif parts:
                                    genes_183904.add(parts[0])
                        if genes_183904:
                            print(f"  Got {len(genes_183904):,} genes")
                            return genes_183904
                    except Exception:
                        pass
    except Exception as e:
        print(f"  Could not list supplementary: {e}")

    # Strategy 3: Fallback — assume standard 10X V3 gene panel
    # If we can't download, use GSE249874 as proxy (same platform)
    print("\n  [FALLBACK] Cannot download GSE183904 gene list directly.")
    print("  Using GSE249874 as proxy (both are 10X V3, similar GENCODE version).")
    print("  This gives a reasonable UPPER BOUND on overlap.")
    return None


def compute_overlaps(unified, g134520, g249874, g_omix, g183904):
    """Compute all intersection scenarios."""
    print("\n[3/4] Computing overlaps...")

    # If we couldn't get GSE183904 genes, simulate with GSE249874
    if g183904 is None:
        print("  Using GSE249874 as proxy for GSE183904 (same platform)")
        g183904 = g249874.copy()
        is_proxy = True
    else:
        is_proxy = False

    print(f"\n  Individual dataset sizes:")
    print(f"    GSE134520:  {len(g134520):,} genes")
    print(f"    GSE249874:  {len(g249874):,} genes")
    print(f"    OMIX010346: {len(g_omix):,} genes")
    print(f"    GSE183904:  {len(g183904):,} genes"
          f"{' (PROXY)' if is_proxy else ''}")

    # Current 3-way intersection
    intersect_3 = g134520 & g249874 & g_omix
    print(f"\n  Current 3-way intersection: {len(intersect_3):,} genes")

    # New 4-way intersection
    intersect_4 = g134520 & g249874 & g_omix & g183904
    print(f"  New 4-way intersection:     {len(intersect_4):,} genes")
    loss = len(intersect_3) - len(intersect_4)
    pct = loss / len(intersect_3) * 100 if intersect_3 else 0
    print(f"  Gene LOSS from adding GSE183904: {loss:,} ({pct:.1f}%)")

    # What genes are lost?
    lost_genes = intersect_3 - intersect_4
    if lost_genes:
        print(f"\n  Sample of lost genes (first 30):")
        for g in sorted(lost_genes)[:30]:
            print(f"    {g}")

    # Pairwise intersections (for context)
    print(f"\n  Pairwise intersections:")
    pairs = [
        ("GSE134520 ∩ GSE249874", g134520 & g249874),
        ("GSE134520 ∩ OMIX", g134520 & g_omix),
        ("GSE134520 ∩ GSE183904", g134520 & g183904),
        ("GSE249874 ∩ OMIX", g249874 & g_omix),
        ("GSE249874 ∩ GSE183904", g249874 & g183904),
        ("OMIX ∩ GSE183904", g_omix & g183904),
    ]
    for name, pset in pairs:
        print(f"    {name}: {len(pset):,}")

    # Check panel genes
    panel_genes = ['PSMA7', 'POMP', 'CTSZ', 'VNN1', 'ADM', 'CNIH4',
                   'FTL', 'ASS1', 'MRPL13', 'TRIB1', 'OLFM4', 'BCAP31',
                   'TMEM176A', 'SOD1', 'DPP4']
    panel_in_4way = [g for g in panel_genes if g in intersect_4]
    panel_lost = [g for g in panel_genes if g in intersect_3
                  and g not in intersect_4]
    print(f"\n  Panel genes in 4-way intersection: "
          f"{len(panel_in_4way)}/{len(panel_genes)}")
    if panel_lost:
        print(f"  Panel genes LOST by adding GSE183904: {panel_lost}")

    # Extended candidate genes from the project
    extended = ['OLFM4', 'REG4', 'ITLN1', 'PRAP1', 'ANPEP', 'PSCA',
                'FABP1', 'CPS1', 'MUC13', 'CLDN4', 'CDX2', 'TFF3',
                'MUC2', 'VIL1', 'CDH17', 'GPA33', 'KRT20']
    ext_in = [g for g in extended if g in intersect_4]
    ext_lost = [g for g in extended if g in intersect_3
                and g not in intersect_4]
    print(f"  Extended markers in 4-way: {len(ext_in)}/{len(extended)}")
    if ext_lost:
        print(f"  Extended markers LOST: {ext_lost}")

    return intersect_4, is_proxy


def assess_metadata_compatibility():
    """Check stage annotation compatibility."""
    print("\n[4/4] Metadata compatibility assessment...")
    print("""
  GSE183904 (Kumar et al. 2022, Cancer Discovery):
  ─────────────────────────────────────────────────
  Paper reports: 48 samples, 31 patients
  Stages reported in paper:
    - Normal adjacent mucosa
    - Intestinal metaplasia (IM)
    - Gastric cancer (GC) — intestinal, diffuse, mixed subtypes

  Your current stage system:
    NAG → CAG → IM → EGC → GC

  Mapping feasibility:
    Kumar "Normal"  →  NAG (reasonable)
    Kumar "IM"      →  IM  (direct match)
    Kumar "GC"      →  GC  (direct match, but no EGC distinction)

  KEY ISSUE: Kumar et al. does NOT distinguish EGC from advanced GC.
  Their "GC" includes both early and advanced stages.
  This means:
    - You CANNOT use GSE183904 to add EGC cells
    - You CAN use it to add NAG/IM/GC cells
    - GC from GSE183904 + GC from GSE249874 = better batch-biology separation

  BENEFIT for batch-stage confounding:
    Before: GC only from GSE249874 (100% confounded)
    After:  GC from GSE249874 + GSE183904 (partially deconfounded!)
    Before: IM from GSE134520 + GSE249874 (already OK)
    After:  IM from 3 datasets (even better)

  Stage coverage improvement:
""")

    # Current confounding
    print("  Current batch-stage matrix:")
    print("    stage         GSE134520  GSE249874  OMIX010346  GSE183904(new)")
    print("    ──────────────────────────────────────────────────────────────")
    print("    NAG           6,088      32,292     0           ~??K (Normal)")
    print("    CAG           19,560     0          0           0")
    print("    IM            14,746     57,060     0           ~??K")
    print("    EGC           2,731      0          16,041      0")
    print("    GC            0          41,232     0           ~??K")
    print("    ──────────────────────────────────────────────────────────────")
    print("    GC now from 2 datasets → batch-biology partially deconfounded")
    print("    CAG still only from GSE134520 → remains confounded")
    print("    EGC still only from GSE134520+OMIX → no change")


def print_verdict(intersect_4, is_proxy):
    """Final recommendation."""
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if is_proxy:
        print("""
  [Based on PROXY estimate — actual result depends on download]

  Since GSE183904 is also 10X V3 (same as GSE249874), the gene
  overlap is likely very high. Expected scenarios:

  BEST CASE:  4-way intersection ≈ 3-way intersection (loss < 100 genes)
              → GSE183904 uses same GENCODE as GSE249874

  LIKELY:     Loss of 200-500 genes (1-2%)
              → Minor version differences, negligible impact

  WORST CASE: Loss of 1000-2000 genes (5-10%)
              → Different GENCODE major version or heavy pre-filtering
              → Still manageable with gene unification script

  RECOMMENDATION:
  1. Download GSE183904 h5ad/mtx (will be large, ~5-15 GB)
  2. Extract var_names only (don't need to load full matrix yet)
  3. Run this script again with real gene list
  4. If loss < 5%: proceed with integration
  5. If loss > 5%: run gene unification (01a) for 4 datasets first
""")
    else:
        n = len(intersect_4) if intersect_4 else 0
        print(f"""
  4-way intersection: {n:,} genes

  Decision criteria:
    Loss < 2%  (< ~430 genes):  ✓ Go ahead, minimal impact
    Loss 2-5%  (430-1074):      ✓ Acceptable, run gene unification
    Loss 5-10% (1074-2148):     ⚠ Marginal, weigh against benefits
    Loss > 10% (> 2148):        ✗ Too costly, keep as external validation
""")

    print("""
  NEXT STEPS if you decide to add GSE183904:
  ──────────────────────────────────────────
  1. Download full matrix: ~5-15 GB, need sufficient disk space
  2. Extend 01a_gene_unification.py to handle 4th dataset
  3. Modify 01_multi_dataset_qc.py to include GSE183904
  4. Map Kumar et al. stage labels → your NAG/IM/GC system
  5. Re-run entire pipeline (Phase 1 → Phase 25)
     Estimated time: depends on your hardware, likely 2-4 days
""")


def main():
    print("=" * 60)
    print("GSE183904 Integration Feasibility Assessment")
    print("=" * 60)

    unified, g134520, g249874, g_omix = get_existing_gene_sets()
    g183904 = get_gse183904_genes()
    intersect_4, is_proxy = compute_overlaps(
        unified, g134520, g249874, g_omix, g183904)
    assess_metadata_compatibility()
    print_verdict(intersect_4, is_proxy)

    print("\nDone.")


if __name__ == "__main__":
    main()
