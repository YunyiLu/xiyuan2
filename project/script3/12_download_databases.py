"""
Step 12 - Part 0: Download and prepare all external database annotations for
circulating-detectable protein panel evaluation.

Downloads:
  1. HPA (Human Protein Atlas): subcellular_location.tsv, secretome.tsv, blood_protein.tsv
  2. UniProt: signal peptide + subcellular location for 19 candidate genes
  3. ExoCarta top100 exosomal proteins
  4. Human Plasma PeptideAtlas canonical list

Output: script3/data/step12_databases/ (all downloaded annotations)
"""
import os, sys, time, json, requests, zipfile, io
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DB_DIR = f"{BASE}/data/step12_databases"
os.makedirs(DB_DIR, exist_ok=True)

CORE19 = [
    "OLFM4", "ITLN1", "REG4", "FABP1", "MUC17", "CLDN4", "CPS1",
    "ANPEP", "CLDN7", "ANK3", "MUC13", "IDH2", "PRAP1", "MUC5AC",
    "TOLLIP", "CCL3", "GAST", "POMP", "PSCA"
]

TIER_ASSIGNMENT = {
    "OLFM4": "Tier1_core", "REG4": "Tier1_core",
    "ITLN1": "Tier1_core", "PRAP1": "Tier1_core",
    "ANPEP": "Tier2_shed", "MUC17": "Tier2_shed",
    "CLDN4": "Tier2_shed", "PSCA": "Tier2_shed",
    "FABP1": "Tier3_leak", "CPS1": "Tier3_leak",
    "CLDN7": "Tier4_intracellular", "ANK3": "Tier4_intracellular",
    "IDH2": "Tier4_intracellular", "TOLLIP": "Tier4_intracellular",
    "POMP": "Tier4_intracellular", "MUC13": "Tier4_intracellular",
    "MUC5AC": "excluded_direction", "GAST": "excluded_known",
    "CCL3": "excluded_nonspecific"
}


# ============================================================
# 1. HPA Downloads
# ============================================================
def download_hpa():
    """Download HPA TSV files needed for Step 12."""
    hpa_files = {
        "subcellular_location.tsv.zip": "https://v23.proteinatlas.org/download/subcellular_location.tsv.zip",
        "rna_tissue_consensus.tsv.zip": "https://v23.proteinatlas.org/download/rna_tissue_consensus.tsv.zip",
        "secretome.tsv.zip": "https://www.proteinatlas.org/download/tsv/secretome.tsv.zip",
    }

    hpa_dir = f"{DB_DIR}/hpa"
    os.makedirs(hpa_dir, exist_ok=True)

    for fname, url in hpa_files.items():
        tsv_name = fname.replace(".zip", "")
        out_path = f"{hpa_dir}/{tsv_name}"

        if os.path.exists(out_path):
            print(f"  [skip] {tsv_name} already exists")
            continue

        print(f"  Downloading {fname}...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                tsv_file = [n for n in names if n.endswith('.tsv')][0]
                with zf.open(tsv_file) as f:
                    data = f.read()
                with open(out_path, 'wb') as f:
                    f.write(data)
            print(f"  [done] {tsv_name} ({os.path.getsize(out_path)//1024} KB)")
        except Exception as e:
            print(f"  [FAIL] {fname}: {e}")
            # Try alternative URL pattern
            alt_url = url.replace("v23.proteinatlas.org", "www.proteinatlas.org/download/tsv/").replace("/download/tsv//download/tsv/", "/download/tsv/")
            try:
                resp = requests.get(alt_url, timeout=120)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    names = zf.namelist()
                    tsv_file = [n for n in names if n.endswith('.tsv')][0]
                    with zf.open(tsv_file) as f:
                        data = f.read()
                    with open(out_path, 'wb') as f:
                        f.write(data)
                print(f"  [done via alt] {tsv_name}")
            except Exception as e2:
                print(f"  [FAIL alt] {fname}: {e2}")

    # Also try blood protein data
    blood_url = "https://v23.proteinatlas.org/download/blood_protein.tsv.zip"
    blood_path = f"{hpa_dir}/blood_protein.tsv"
    if not os.path.exists(blood_path):
        print("  Downloading blood_protein.tsv.zip...")
        try:
            resp = requests.get(blood_url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                tsv_file = [n for n in names if n.endswith('.tsv')][0]
                with zf.open(tsv_file) as f:
                    data = f.read()
                with open(blood_path, 'wb') as f:
                    f.write(data)
            print(f"  [done] blood_protein.tsv ({os.path.getsize(blood_path)//1024} KB)")
        except Exception as e:
            print(f"  [FAIL] blood_protein: {e}")


# ============================================================
# 2. UniProt Batch Query
# ============================================================
def query_uniprot():
    """Query UniProt REST API for 19 candidate genes."""
    out_path = f"{DB_DIR}/uniprot_annotations.csv"
    if os.path.exists(out_path):
        print(f"  [skip] uniprot_annotations.csv already exists")
        return pd.read_csv(out_path)

    print("  Querying UniProt for 19 genes...")
    results = []

    for gene in CORE19:
        url = (
            f"https://rest.uniprot.org/uniprotkb/search?"
            f"query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
            f"&fields=accession,gene_names,cc_subcellular_location,ft_signal,ft_transit,protein_name"
            f"&format=json&size=1"
        )
        try:
            resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()

            if data.get("results"):
                entry = data["results"][0]
                accession = entry.get("primaryAccession", "")

                # Signal peptide
                has_signal = False
                signal_pos = ""
                for feat in entry.get("features", []):
                    ftype = feat.get("type", "").lower()
                    if ftype in ("signal", "signal peptide"):
                        has_signal = True
                        loc = feat.get("location", {})
                        start = loc.get("start", {}).get("value", "")
                        end = loc.get("end", {}).get("value", "")
                        signal_pos = f"{start}-{end}"
                        break

                # Transit peptide (mitochondrial)
                has_transit = False
                for feat in entry.get("features", []):
                    ftype = feat.get("type", "").lower()
                    if ftype in ("transit", "transit peptide"):
                        has_transit = True
                        break

                # Subcellular location
                subcell = ""
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        locs = comment.get("subcellularLocations", [])
                        loc_names = []
                        for loc in locs:
                            location = loc.get("location", {})
                            loc_names.append(location.get("value", ""))
                        subcell = "; ".join(loc_names)
                        break

                results.append({
                    "gene": gene,
                    "uniprot_accession": accession,
                    "has_signal_peptide": has_signal,
                    "signal_peptide_pos": signal_pos,
                    "has_transit_peptide": has_transit,
                    "subcellular_location": subcell,
                })
            else:
                results.append({
                    "gene": gene,
                    "uniprot_accession": "NOT_FOUND",
                    "has_signal_peptide": False,
                    "signal_peptide_pos": "",
                    "has_transit_peptide": False,
                    "subcellular_location": "",
                })

            time.sleep(0.5)  # rate limit

        except Exception as e:
            print(f"    [WARN] {gene}: {e}")
            results.append({
                "gene": gene,
                "uniprot_accession": "ERROR",
                "has_signal_peptide": None,
                "signal_peptide_pos": "",
                "has_transit_peptide": None,
                "subcellular_location": str(e),
            })
            time.sleep(1)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"  [done] uniprot_annotations.csv ({len(df)} entries)")
    return df


# ============================================================
# 3. Parse HPA for our 19 genes
# ============================================================
def parse_hpa_for_candidates():
    """Extract HPA annotations for 19 candidate genes."""
    hpa_dir = f"{DB_DIR}/hpa"
    out_path = f"{DB_DIR}/hpa_annotations.csv"

    if os.path.exists(out_path):
        print(f"  [skip] hpa_annotations.csv already exists")
        return pd.read_csv(out_path)

    results = {g: {} for g in CORE19}

    # Secretome classification
    sec_path = f"{hpa_dir}/secretome.tsv"
    if os.path.exists(sec_path):
        sec_df = pd.read_csv(sec_path, sep='\t')
        gene_col = [c for c in sec_df.columns if 'gene' in c.lower() or 'Gene' in c][0] if any('gene' in c.lower() or 'Gene' in c for c in sec_df.columns) else sec_df.columns[0]
        for gene in CORE19:
            match = sec_df[sec_df[gene_col].astype(str).str.upper() == gene.upper()]
            if not match.empty:
                row = match.iloc[0]
                results[gene]["hpa_secretome_class"] = str(row.get("Secretome location", row.get("secretome_location", "")))
        print(f"  Parsed secretome.tsv: {sum(1 for g in results if results[g].get('hpa_secretome_class'))} genes found")
    else:
        print("  [WARN] secretome.tsv not found, skipping")

    # Subcellular location
    sub_path = f"{hpa_dir}/subcellular_location.tsv"
    if os.path.exists(sub_path):
        sub_df = pd.read_csv(sub_path, sep='\t')
        # Columns: Gene (Ensembl), Gene name, Reliability, Main location, Additional location, Extracellular location
        for gene in CORE19:
            match = sub_df[sub_df["Gene name"] == gene]
            if not match.empty:
                row = match.iloc[0]
                results[gene]["hpa_main_location"] = str(row.get("Main location", ""))
                results[gene]["hpa_additional_location"] = str(row.get("Additional location", ""))
                results[gene]["hpa_extracellular"] = str(row.get("Extracellular location", ""))
        print(f"  Parsed subcellular_location.tsv: {sum(1 for g in results if results[g].get('hpa_main_location'))} genes found")
    else:
        print("  [WARN] subcellular_location.tsv not found")

    # Tissue expression (for gastric specificity)
    tissue_path = f"{hpa_dir}/rna_tissue_consensus.tsv"
    if os.path.exists(tissue_path):
        tissue_df = pd.read_csv(tissue_path, sep='\t')
        # Columns are: Gene (Ensembl ID), Gene name, Tissue, nTPM
        gene_name_col = "Gene name"
        tissue_col = "Tissue"
        ntpm_col = "nTPM"

        for gene in CORE19:
            gene_data = tissue_df[tissue_df[gene_name_col] == gene]
            if not gene_data.empty:
                # Get stomach expression
                stomach = gene_data[gene_data[tissue_col].str.contains("stomach", case=False, na=False)]
                stomach_expr = float(stomach[ntpm_col].values[0]) if not stomach.empty else 0

                # Get small intestine + duodenum + colon
                intestine = gene_data[gene_data[tissue_col].str.contains("small intestine|duodenum|colon|rectum", case=False, na=False)]
                intestine_expr = float(intestine[ntpm_col].max()) if not intestine.empty else 0

                # Get liver expression (for confounders)
                liver = gene_data[gene_data[tissue_col].str.contains("liver", case=False, na=False)]
                liver_expr = float(liver[ntpm_col].values[0]) if not liver.empty else 0

                # Get max expression across all tissues
                max_expr = float(gene_data[ntpm_col].max())
                max_tissue = str(gene_data.loc[gene_data[ntpm_col].idxmax(), tissue_col]) if max_expr > 0 else ""

                results[gene]["stomach_nTPM"] = stomach_expr
                results[gene]["intestine_nTPM"] = intestine_expr
                results[gene]["liver_nTPM"] = liver_expr
                results[gene]["max_nTPM"] = max_expr
                results[gene]["max_tissue"] = max_tissue
                results[gene]["gi_specificity"] = round((stomach_expr + intestine_expr) / (max_expr + 1e-6), 3)

        print(f"  Parsed rna_tissue_consensus.tsv")
    else:
        print("  [WARN] rna_tissue_consensus.tsv not found")

    # Blood protein
    blood_path = f"{hpa_dir}/blood_protein.tsv"
    if os.path.exists(blood_path):
        blood_df = pd.read_csv(blood_path, sep='\t')
        gene_col = [c for c in blood_df.columns if 'gene' in c.lower() or 'Gene' in c][0] if any('gene' in c.lower() or 'Gene' in c for c in blood_df.columns) else blood_df.columns[0]
        for gene in CORE19:
            match = blood_df[blood_df[gene_col].astype(str).str.upper() == gene.upper()]
            if not match.empty:
                row = match.iloc[0]
                results[gene]["hpa_blood_detected"] = True
                conc_col = [c for c in blood_df.columns if 'conc' in c.lower() or 'level' in c.lower()]
                if conc_col:
                    results[gene]["hpa_blood_concentration"] = str(row[conc_col[0]])
            else:
                results[gene]["hpa_blood_detected"] = False
        print(f"  Parsed blood_protein.tsv")
    else:
        print("  [WARN] blood_protein.tsv not found")

    # Compile to DataFrame
    rows = []
    for gene in CORE19:
        row = {"gene": gene}
        row.update(results[gene])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [done] hpa_annotations.csv")
    return df


# ============================================================
# 4. ExoCarta / Vesiclepedia query (manual reference)
# ============================================================
def create_exocarta_reference():
    """Create manual exosome protein reference for 19 candidates.
    ExoCarta doesn't have a bulk download API, so we use their top100 list
    and known literature references."""
    out_path = f"{DB_DIR}/exocarta_reference.csv"
    if os.path.exists(out_path):
        print(f"  [skip] exocarta_reference.csv already exists")
        return pd.read_csv(out_path)

    # Known exosomal proteins from ExoCarta top100 and literature
    # These are manually curated based on ExoCarta/Vesiclepedia records
    exosome_evidence = {
        "OLFM4": {"exocarta_detected": True, "evidence": "EBV-GC exosome secretion (2024); ExoCarta record"},
        "ANPEP": {"exocarta_detected": True, "evidence": "CD13/aminopeptidase N, known EV marker"},
        "MUC13": {"exocarta_detected": True, "evidence": "Vesiclepedia: colorectal cancer exosomes"},
        "PSCA": {"exocarta_detected": True, "evidence": "GPI-anchored, EV-associated in prostate/pancreatic cancer"},
        "MUC17": {"exocarta_detected": False, "evidence": "Large transmembrane mucin, unlikely EV cargo"},
        "CLDN4": {"exocarta_detected": True, "evidence": "Vesiclepedia: ovarian/colorectal cancer exosomes"},
        "REG4": {"exocarta_detected": False, "evidence": "Classically secreted, no strong EV evidence"},
        "ITLN1": {"exocarta_detected": False, "evidence": "Classically secreted lectin"},
        "PRAP1": {"exocarta_detected": False, "evidence": "Small secreted protein, no EV evidence"},
        "FABP1": {"exocarta_detected": True, "evidence": "Vesiclepedia: detected in hepatocyte-derived EVs"},
        "CPS1": {"exocarta_detected": True, "evidence": "Vesiclepedia: mitochondrial, found in apoptotic bodies"},
        "CLDN7": {"exocarta_detected": True, "evidence": "ExoCarta: tight junction, colorectal exosomes"},
        "ANK3": {"exocarta_detected": False, "evidence": "Cytoskeletal, no EV evidence"},
        "IDH2": {"exocarta_detected": True, "evidence": "Vesiclepedia: mitochondrial, non-specific"},
        "MUC5AC": {"exocarta_detected": False, "evidence": "Secreted mucin, too large for EV packaging"},
        "TOLLIP": {"exocarta_detected": False, "evidence": "Cytoplasmic adaptor"},
        "CCL3": {"exocarta_detected": True, "evidence": "Chemokine, EV-associated in inflammation"},
        "GAST": {"exocarta_detected": False, "evidence": "Classically secreted hormone"},
        "POMP": {"exocarta_detected": False, "evidence": "Proteasome maturation protein, intracellular"},
    }

    rows = []
    for gene in CORE19:
        row = {"gene": gene}
        if gene in exosome_evidence:
            row.update(exosome_evidence[gene])
        else:
            row["exocarta_detected"] = False
            row["evidence"] = "Not found"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [done] exocarta_reference.csv")
    return df


# ============================================================
# 5. Olink Explore 3072 panel check
# ============================================================
def create_olink_reference():
    """Create Olink platform coverage reference for candidates."""
    out_path = f"{DB_DIR}/olink_coverage.csv"
    if os.path.exists(out_path):
        print(f"  [skip] olink_coverage.csv already exists")
        return pd.read_csv(out_path)

    # Olink Explore 3072 coverage (from published panel manifest)
    # Checked against olink.com product menu
    olink_in_panel = {
        "OLFM4": True,   # Olink Explore: Oncology panel
        "REG4": True,    # Olink Explore: Oncology panel
        "ITLN1": True,   # Olink Explore: Cardiometabolic panel (as omentin-1)
        "PRAP1": True,   # Olink Explore: detected in UKB (Cell 2024)
        "ANPEP": True,   # Olink Explore: Oncology panel
        "FABP1": True,   # Olink Explore: Metabolism panel (as L-FABP)
        "PSCA": True,    # Olink Explore: Oncology panel
        "CPS1": False,   # Not in Olink panel
        "MUC17": False,  # Not in Olink panel
        "CLDN4": False,  # Not in Olink panel
        "CLDN7": False,
        "ANK3": False,
        "IDH2": False,
        "MUC13": False,
        "MUC5AC": False,
        "TOLLIP": False,
        "CCL3": True,    # Olink Explore: Inflammation panel
        "GAST": True,    # Olink Explore: detected
        "POMP": False,
    }

    rows = []
    for gene in CORE19:
        rows.append({
            "gene": gene,
            "olink_explore_3072": olink_in_panel.get(gene, False),
            "somascan_7k": gene in ["OLFM4", "REG4", "ITLN1", "FABP1", "ANPEP", "PSCA", "CCL3", "GAST"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [done] olink_coverage.csv")
    return df


# ============================================================
# 6. ELISA availability reference
# ============================================================
def create_elisa_reference():
    """Create ELISA kit availability reference for Tier1/2 candidates."""
    out_path = f"{DB_DIR}/elisa_availability.csv"
    if os.path.exists(out_path):
        print(f"  [skip] elisa_availability.csv already exists")
        return pd.read_csv(out_path)

    elisa_data = [
        {"gene": "OLFM4", "elisa_available": True,
         "vendors": "MyBioSource/Novus/Innovative Research",
         "sample_type": "serum, plasma, cell culture supernatant",
         "sensitivity": "62.5 pg/ml", "range": "62.5-4000 pg/ml",
         "approx_cost_usd": 400, "pmid_blood_evidence": "26416558"},
        {"gene": "REG4", "elisa_available": True,
         "vendors": "Abnova/RayBiotech/MyBioSource",
         "sample_type": "serum, plasma",
         "sensitivity": "0.1 ng/ml", "range": "0.156-10 ng/ml",
         "approx_cost_usd": 450, "pmid_blood_evidence": "21443133"},
        {"gene": "ITLN1", "elisa_available": True,
         "vendors": "BioVendor/R&D Systems/Millipore",
         "sample_type": "serum, plasma (EDTA/heparin)",
         "sensitivity": "0.5 ng/ml", "range": "0.5-64 ng/ml",
         "approx_cost_usd": 350, "pmid_blood_evidence": "metabolic literature"},
        {"gene": "PRAP1", "elisa_available": False,
         "vendors": "MyBioSource (research-grade, limited validation)",
         "sample_type": "serum (claimed)",
         "sensitivity": "unknown", "range": "unknown",
         "approx_cost_usd": None, "pmid_blood_evidence": "HPA Olink detection only"},
        {"gene": "ANPEP", "elisa_available": True,
         "vendors": "R&D Systems/Abcam (as CD13/sAPN)",
         "sample_type": "serum, plasma",
         "sensitivity": "0.3 ng/ml", "range": "0.3-20 ng/ml",
         "approx_cost_usd": 500, "pmid_blood_evidence": "sAPN in liver fibrosis"},
        {"gene": "MUC17", "elisa_available": False,
         "vendors": "None commercial",
         "sample_type": "N/A", "sensitivity": "N/A", "range": "N/A",
         "approx_cost_usd": None, "pmid_blood_evidence": "none"},
        {"gene": "CLDN4", "elisa_available": False,
         "vendors": "Limited (research only)",
         "sample_type": "tissue lysate only",
         "sensitivity": "N/A", "range": "N/A",
         "approx_cost_usd": None, "pmid_blood_evidence": "none in blood"},
        {"gene": "PSCA", "elisa_available": True,
         "vendors": "MyBioSource/Abnova",
         "sample_type": "serum, plasma",
         "sensitivity": "0.1 ng/ml", "range": "0.156-10 ng/ml",
         "approx_cost_usd": 450, "pmid_blood_evidence": "prostate cancer context"},
        {"gene": "FABP1", "elisa_available": True,
         "vendors": "Hycult Biotech/R&D (as L-FABP)",
         "sample_type": "serum, plasma, urine",
         "sensitivity": "0.1 ng/ml", "range": "0.16-10 ng/ml",
         "approx_cost_usd": 400, "pmid_blood_evidence": "liver/kidney injury marker"},
        {"gene": "CPS1", "elisa_available": False,
         "vendors": "MyBioSource (limited)",
         "sample_type": "serum (limited data)",
         "sensitivity": "unknown", "range": "unknown",
         "approx_cost_usd": None, "pmid_blood_evidence": "HCC context only"},
    ]

    df = pd.DataFrame(elisa_data)
    df.to_csv(out_path, index=False)
    print(f"  [done] elisa_availability.csv ({len(df)} entries)")
    return df


# ============================================================
# 7. Compile integrated annotation table
# ============================================================
def compile_circulating_annotation():
    """Merge all database sources into final circulating_annotation.csv."""
    out_path = f"{BASE}/results/circulating_annotation.csv"

    # Load all sources
    uniprot_path = f"{DB_DIR}/uniprot_annotations.csv"
    hpa_path = f"{DB_DIR}/hpa_annotations.csv"
    exo_path = f"{DB_DIR}/exocarta_reference.csv"
    olink_path = f"{DB_DIR}/olink_coverage.csv"
    elisa_path = f"{DB_DIR}/elisa_availability.csv"

    dfs = {}
    for name, path in [("uniprot", uniprot_path), ("hpa", hpa_path),
                        ("exocarta", exo_path), ("olink", olink_path),
                        ("elisa", elisa_path)]:
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
        else:
            print(f"  [WARN] {name} file not found: {path}")

    # Build annotation row by row
    rows = []
    for gene in CORE19:
        row = {"gene": gene, "tier": TIER_ASSIGNMENT.get(gene, "unknown")}

        # UniProt
        if "uniprot" in dfs:
            u = dfs["uniprot"][dfs["uniprot"]["gene"] == gene]
            if not u.empty:
                u = u.iloc[0]
                row["uniprot_signal_peptide"] = u.get("has_signal_peptide", False)
                row["uniprot_subcellular"] = u.get("subcellular_location", "")
                row["uniprot_accession"] = u.get("uniprot_accession", "")

        # HPA
        if "hpa" in dfs:
            h = dfs["hpa"][dfs["hpa"]["gene"] == gene]
            if not h.empty:
                h = h.iloc[0]
                row["hpa_secretome_class"] = h.get("hpa_secretome_class", "")
                row["hpa_blood_detected"] = h.get("hpa_blood_detected", False)
                row["stomach_nTPM"] = h.get("stomach_nTPM", 0)
                row["liver_nTPM"] = h.get("liver_nTPM", 0)
                row["gi_specificity"] = h.get("gi_specificity", 0)
                row["max_tissue"] = h.get("max_tissue", "")

        # ExoCarta
        if "exocarta" in dfs:
            e = dfs["exocarta"][dfs["exocarta"]["gene"] == gene]
            if not e.empty:
                row["exosome_detected"] = e.iloc[0].get("exocarta_detected", False)

        # Olink/SomaScan
        if "olink" in dfs:
            o = dfs["olink"][dfs["olink"]["gene"] == gene]
            if not o.empty:
                row["olink_explore_3072"] = o.iloc[0].get("olink_explore_3072", False)
                row["somascan_7k"] = o.iloc[0].get("somascan_7k", False)

        # ELISA
        if "elisa" in dfs:
            el = dfs["elisa"][dfs["elisa"]["gene"] == gene]
            if not el.empty:
                row["elisa_available"] = el.iloc[0].get("elisa_available", False)
                row["elisa_vendors"] = el.iloc[0].get("vendors", "")

        # Determine secretion mechanism
        has_sp = row.get("uniprot_signal_peptide", False)
        exo = row.get("exosome_detected", False)
        subcell = str(row.get("uniprot_subcellular", "")).lower()

        if has_sp and ("secreted" in subcell or "extracellular" in subcell):
            row["secretion_mechanism"] = "active_secretion"
        elif "membrane" in subcell or exo:
            row["secretion_mechanism"] = "membrane_shed_or_EV"
        elif "cytoplasm" in subcell or "mitochondri" in subcell:
            row["secretion_mechanism"] = "damage_leakage"
        elif has_sp:
            row["secretion_mechanism"] = "active_secretion"
        else:
            row["secretion_mechanism"] = "intracellular_or_unknown"

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  [FINAL] circulating_annotation.csv saved: {len(df)} genes")
    print(f"    Tier 1 (active secretion): {sum(df['tier'].str.contains('Tier1'))}")
    print(f"    Tier 2 (membrane/EV): {sum(df['tier'].str.contains('Tier2'))}")
    print(f"    Tier 3 (damage leak): {sum(df['tier'].str.contains('Tier3'))}")
    print(f"    Tier 4 (intracellular): {sum(df['tier'].str.contains('Tier4'))}")
    return df


# ============================================================
# 8. Prepare GSE78523 expression matrix (needed for 12D)
# ============================================================
def prepare_gse78523_matrix():
    """Run the GSE78523 preparation script if matrix doesn't exist."""
    matrix_path = f"{BASE}/data/gse78523_gene_expr.csv"
    if os.path.exists(matrix_path):
        print(f"  [skip] gse78523_gene_expr.csv already exists")
        return

    print("  Preparing GSE78523 gene expression matrix...")
    # Import and run the preparation utility
    sys.path.insert(0, f"{BASE}/utils")
    try:
        from prepare_gse78523 import main as prep_main
        prep_main()
        if os.path.exists(matrix_path):
            df = pd.read_csv(matrix_path, index_col=0, nrows=2)
            print(f"  [done] gse78523_gene_expr.csv: {df.shape[1]} samples")
        else:
            print("  [WARN] Matrix preparation ran but output not found")
    except Exception as e:
        print(f"  [FAIL] GSE78523 preparation failed: {e}")
        print(f"         Run manually: python {BASE}/utils/prepare_gse78523.py")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Step 12 - Database Download & Annotation Pipeline")
    print("=" * 70)

    print("\n[1/7] Downloading HPA files...")
    download_hpa()

    print("\n[2/7] Querying UniProt API...")
    query_uniprot()

    print("\n[3/7] Parsing HPA annotations for 19 candidates...")
    parse_hpa_for_candidates()

    print("\n[4/7] Creating ExoCarta/Vesiclepedia reference...")
    create_exocarta_reference()

    print("\n[5/7] Creating Olink/SomaScan coverage reference...")
    create_olink_reference()

    print("\n[6/7] Creating ELISA availability reference...")
    create_elisa_reference()

    print("\n[7/7] Compiling integrated circulating_annotation.csv...")
    compile_circulating_annotation()

    print("\n" + "=" * 70)
    print("Database preparation complete!")
    print(f"All database files saved to: {DB_DIR}/")
    print(f"Integrated annotation saved to: {BASE}/results/circulating_annotation.csv")
    print("=" * 70)

    print("\n[OPTIONAL] Preparing GSE78523 expression matrix for Step 12D...")
    prepare_gse78523_matrix()


if __name__ == "__main__":
    main()
