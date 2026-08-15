"""
Step 10c: Download and Process External Methylation + Epigenome Datasets
  Downloads: GSE103186, GSE220511, GSE141660, GSE178925, GSE150290

  Each dataset is downloaded as processed data (beta matrix or supplementary)
  from GEO, then filtered to 92 candidate genes.

Output:
  data/methylation/GSE103186_beta.csv (191 samples × gene promoter beta)
  data/methylation/GSE220511_beta.csv (26 samples × gene promoter beta)
  data/methylation/GSE141660_beta.csv (organoid methylation)
  data/methylation/GSE178925_beta.csv (inflammation stage methylation)
"""
import sys, os, warnings, gzip, urllib.request, ssl
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
METHYL_DIR = f"{BASE}/data/methylation"
RES_DIR = f"{BASE}/results"
os.makedirs(METHYL_DIR, exist_ok=True)

# SSL context for downloads
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Load probe-gene mapping
manifest = pd.read_csv(f"{METHYL_DIR}/HM450_manifest_genes.csv")
print(f"Manifest: {len(manifest)} probe-gene pairs for {manifest['gene'].nunique()} genes")

# Promoter probes
promoter_regions = ['TSS1500', 'TSS200', '1stExon', "5'UTR"]
promoter_manifest = manifest[manifest['region'].isin(promoter_regions)]
print(f"Promoter probes: {len(promoter_manifest)} for {promoter_manifest['gene'].nunique()} genes")


def download_file(url, dest, description=""):
    """Download with progress and retry."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest} ({os.path.getsize(dest)/1024/1024:.1f} MB)")
        return True

    print(f"  Downloading {description}...")
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

    try:
        response = urllib.request.urlopen(req, timeout=300, context=ctx)
        total = int(response.headers.get('Content-Length', 0))

        downloaded = 0
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (20*1024*1024) < 1024*1024:
                    print(f"    {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB")

        print(f"  Done: {downloaded/1024/1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ============================================================
# Dataset 1: GSE103186 (450K, 191 samples, IM progression)
# ============================================================
print("\n" + "="*70)
print("DATASET 1: GSE103186 — 450K methylation, IM progression cohort")
print("="*70)

# GSE103186 supplementary files contain processed beta values
# URL pattern: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE103nnn/GSE103186/suppl/
gse103186_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE103nnn/GSE103186/suppl/GSE103186_RAW.tar"
gse103186_matrix_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE103nnn/GSE103186/matrix/GSE103186_series_matrix.txt.gz"

dest_matrix = f"{METHYL_DIR}/GSE103186_series_matrix.txt.gz"
success = download_file(gse103186_matrix_url, dest_matrix, "GSE103186 series matrix")

if success and os.path.exists(dest_matrix):
    print("  Parsing series matrix for metadata...")
    try:
        # Extract sample metadata from series matrix
        metadata_lines = []
        data_start = False
        with gzip.open(dest_matrix, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('!Sample_'):
                    metadata_lines.append(line.strip())
                if line.startswith('"ID_REF"'):
                    data_start = True
                    header = line.strip()
                    break

        # Parse metadata
        sample_chars = {}
        for line in metadata_lines:
            if 'characteristics' in line.lower() or 'title' in line.lower():
                parts = line.split('\t')
                key = parts[0].strip('"!')
                values = [p.strip('"') for p in parts[1:]]
                sample_chars[key] = values

        if sample_chars:
            print(f"  Metadata keys: {list(sample_chars.keys())[:5]}")
            if 'Sample_title' in sample_chars:
                print(f"  First 3 titles: {sample_chars['Sample_title'][:3]}")
            n_samples = len(sample_chars.get('Sample_title', []))
            print(f"  Number of samples in metadata: {n_samples}")

        # For GSE103186, the beta values are in the series matrix itself
        # Read the actual data (probe × sample beta matrix)
        print("  Reading beta matrix from series matrix...")

        # Re-read to get data
        data_rows = []
        reading_data = False
        with gzip.open(dest_matrix, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('"ID_REF"'):
                    reading_data = True
                    samples = [s.strip('"') for s in line.strip().split('\t')[1:]]
                    continue
                if reading_data:
                    if line.startswith('!') or not line.strip():
                        break
                    parts = line.strip().split('\t')
                    probe = parts[0].strip('"')
                    # Only keep probes for our genes
                    if probe in set(manifest['probe']):
                        values = []
                        for v in parts[1:]:
                            try:
                                values.append(float(v.strip('"')))
                            except:
                                values.append(np.nan)
                        data_rows.append([probe] + values)

        if data_rows:
            beta_df = pd.DataFrame(data_rows, columns=['probe'] + samples)
            beta_df = beta_df.set_index('probe')
            print(f"  Beta matrix (our probes): {beta_df.shape}")

            # Aggregate to gene-level (promoter probes)
            gene_beta = {}
            for gene in manifest['gene'].unique():
                gene_probes = promoter_manifest[promoter_manifest['gene'] == gene]['probe'].tolist()
                avail = [p for p in gene_probes if p in beta_df.index]
                if avail:
                    gene_beta[gene] = beta_df.loc[avail].mean(axis=0)

            gene_beta_df = pd.DataFrame(gene_beta).T
            gene_beta_df.index.name = 'gene'
            gene_beta_df.to_csv(f"{METHYL_DIR}/GSE103186_gene_promoter_beta.csv")
            print(f"  Gene-level promoter beta: {gene_beta_df.shape}")
            print(f"  Saved: GSE103186_gene_promoter_beta.csv")
        else:
            print("  WARNING: No beta data found in series matrix")
            print("  This dataset may require RAW IDAT processing")

    except Exception as e:
        print(f"  Error parsing: {e}")


# ============================================================
# Dataset 2: GSE220511 (EPIC, IM crypts, scATAC, H3K27ac)
# ============================================================
print("\n" + "="*70)
print("DATASET 2: GSE220511 — EPIC methylation + scATAC, IM crypts")
print("="*70)

gse220511_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE220nnn/GSE220511/matrix/GSE220511_series_matrix.txt.gz"
dest_220511 = f"{METHYL_DIR}/GSE220511_series_matrix.txt.gz"
success = download_file(gse220511_url, dest_220511, "GSE220511 series matrix")

if success and os.path.exists(dest_220511):
    print("  Parsing GSE220511...")
    try:
        data_rows = []
        samples = []
        reading_data = False
        with gzip.open(dest_220511, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('"ID_REF"'):
                    reading_data = True
                    samples = [s.strip('"') for s in line.strip().split('\t')[1:]]
                    continue
                if reading_data:
                    if line.startswith('!') or not line.strip():
                        break
                    parts = line.strip().split('\t')
                    probe = parts[0].strip('"')
                    if probe in set(manifest['probe']):
                        values = []
                        for v in parts[1:]:
                            try:
                                values.append(float(v.strip('"')))
                            except:
                                values.append(np.nan)
                        data_rows.append([probe] + values)

        if data_rows:
            beta_df = pd.DataFrame(data_rows, columns=['probe'] + samples).set_index('probe')
            print(f"  Beta matrix (our probes): {beta_df.shape}")

            gene_beta = {}
            for gene in manifest['gene'].unique():
                gene_probes = promoter_manifest[promoter_manifest['gene'] == gene]['probe'].tolist()
                avail = [p for p in gene_probes if p in beta_df.index]
                if avail:
                    gene_beta[gene] = beta_df.loc[avail].mean(axis=0)

            if gene_beta:
                gene_beta_df = pd.DataFrame(gene_beta).T
                gene_beta_df.to_csv(f"{METHYL_DIR}/GSE220511_gene_promoter_beta.csv")
                print(f"  Gene-level: {gene_beta_df.shape}")
        else:
            print("  No matching probes in series matrix (may be EPIC-only probes)")
            print("  Note: GSE220511 uses EPIC array - need EPIC manifest for additional probes")

    except Exception as e:
        print(f"  Error: {e}")


# ============================================================
# Dataset 3: GSE141660 (Organoid methylation + expression)
# ============================================================
print("\n" + "="*70)
print("DATASET 3: GSE141660 — Organoid methylation + expression")
print("="*70)

# GSE141660 is a SuperSeries, methylation is in GSE141659
gse141659_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141659/matrix/GSE141659_series_matrix.txt.gz"
dest_141659 = f"{METHYL_DIR}/GSE141659_series_matrix.txt.gz"
success = download_file(gse141659_url, dest_141659, "GSE141659 (methylation subseries)")

if success and os.path.exists(dest_141659):
    print("  Parsing GSE141659...")
    try:
        data_rows = []
        samples = []
        reading_data = False
        with gzip.open(dest_141659, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('"ID_REF"'):
                    reading_data = True
                    samples = [s.strip('"') for s in line.strip().split('\t')[1:]]
                    continue
                if reading_data:
                    if line.startswith('!') or not line.strip():
                        break
                    parts = line.strip().split('\t')
                    probe = parts[0].strip('"')
                    if probe in set(manifest['probe']):
                        values = []
                        for v in parts[1:]:
                            try:
                                values.append(float(v.strip('"')))
                            except:
                                values.append(np.nan)
                        data_rows.append([probe] + values)

        if data_rows:
            beta_df = pd.DataFrame(data_rows, columns=['probe'] + samples).set_index('probe')
            print(f"  Beta matrix: {beta_df.shape}")

            gene_beta = {}
            for gene in manifest['gene'].unique():
                gene_probes = promoter_manifest[promoter_manifest['gene'] == gene]['probe'].tolist()
                avail = [p for p in gene_probes if p in beta_df.index]
                if avail:
                    gene_beta[gene] = beta_df.loc[avail].mean(axis=0)

            if gene_beta:
                gene_beta_df = pd.DataFrame(gene_beta).T
                gene_beta_df.to_csv(f"{METHYL_DIR}/GSE141659_gene_promoter_beta.csv")
                print(f"  Gene-level: {gene_beta_df.shape}")
        else:
            print("  No data rows found (may need supplementary files)")

    except Exception as e:
        print(f"  Error: {e}")


# ============================================================
# Dataset 4: GSE178925 (Early inflammation methylation)
# ============================================================
print("\n" + "="*70)
print("DATASET 4: GSE178925 — EPIC 850K, early gastritis methylation")
print("="*70)

gse178925_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178925/matrix/GSE178925_series_matrix.txt.gz"
dest_178925 = f"{METHYL_DIR}/GSE178925_series_matrix.txt.gz"
success = download_file(gse178925_url, dest_178925, "GSE178925 series matrix")

if success and os.path.exists(dest_178925):
    print("  Parsing GSE178925...")
    try:
        data_rows = []
        samples = []
        reading_data = False
        with gzip.open(dest_178925, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('"ID_REF"'):
                    reading_data = True
                    samples = [s.strip('"') for s in line.strip().split('\t')[1:]]
                    continue
                if reading_data:
                    if line.startswith('!') or not line.strip():
                        break
                    parts = line.strip().split('\t')
                    probe = parts[0].strip('"')
                    if probe in set(manifest['probe']):
                        values = []
                        for v in parts[1:]:
                            try:
                                values.append(float(v.strip('"')))
                            except:
                                values.append(np.nan)
                        data_rows.append([probe] + values)

        if data_rows:
            beta_df = pd.DataFrame(data_rows, columns=['probe'] + samples).set_index('probe')
            print(f"  Beta matrix: {beta_df.shape}")

            gene_beta = {}
            for gene in manifest['gene'].unique():
                gene_probes = promoter_manifest[promoter_manifest['gene'] == gene]['probe'].tolist()
                avail = [p for p in gene_probes if p in beta_df.index]
                if avail:
                    gene_beta[gene] = beta_df.loc[avail].mean(axis=0)

            if gene_beta:
                gene_beta_df = pd.DataFrame(gene_beta).T
                gene_beta_df.to_csv(f"{METHYL_DIR}/GSE178925_gene_promoter_beta.csv")
                print(f"  Gene-level: {gene_beta_df.shape}")
        else:
            print("  No matching probes (EPIC uses different probe set)")

    except Exception as e:
        print(f"  Error: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)

for f in os.listdir(METHYL_DIR):
    if f.endswith('.csv') and 'gene_promoter' in f:
        df = pd.read_csv(f"{METHYL_DIR}/{f}", index_col=0)
        print(f"  {f}: {df.shape[0]} genes × {df.shape[1]} samples")

print("\nDone!")
