"""
Step 0: Download and extract GSE183904 (Kumar et al. 2022, Cancer Discovery).
~200,000 cells, 48 samples, 31 patients, 10X Genomics scRNA-seq.

Downloads GSE183904_RAW.tar from GEO, extracts per-sample 10X directories,
and parses sample metadata for stage mapping.

Output:
  dataset/GSE183904/<GSM_accession>/  (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
  dataset/GSE183904/metadata.tsv      (accession, title, stage, subtype)
"""
import os, sys, tarfile, gzip, ssl, urllib.request, shutil
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE183904"
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

GEO_FTP = ("https://ftp.ncbi.nlm.nih.gov/geo/series/"
           "GSE183nnn/GSE183904/suppl/GSE183904_RAW.tar")

# Fallback: NCBI GEO download gateway
GEO_ALT = ("https://www.ncbi.nlm.nih.gov/geo/download/"
           "?acc=GSE183904&format=file")


def download_raw_tar():
    """Download GSE183904_RAW.tar (~10-30 GB)."""
    tar_path = f"{OUT_DIR}/GSE183904_RAW.tar"
    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 1e9:
        print(f"  [CACHED] {tar_path} ({os.path.getsize(tar_path)/1e9:.1f} GB)")
        return tar_path

    print(f"  Downloading GSE183904_RAW.tar (this may take 30-60 min)...")
    for url in [GEO_FTP, GEO_ALT]:
        try:
            print(f"  Trying: {url[:80]}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=3600) as resp:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                with open(tar_path, 'wb') as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r    {downloaded/1e9:.2f}/{total/1e9:.2f} GB "
                                  f"({pct:.1f}%)", end='', flush=True)
                        else:
                            print(f"\r    {downloaded/1e9:.2f} GB", end='', flush=True)
            print(f"\n  Done: {os.path.getsize(tar_path)/1e9:.2f} GB")
            return tar_path
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            continue

    print("\n  ERROR: Could not download from any source.")
    print("  Manual download instructions:")
    print(f"    1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183904")
    print(f"    2. Click 'Custom download' or find supplementary files")
    print(f"    3. Download GSE183904_RAW.tar")
    print(f"    4. Place at: {tar_path}")
    sys.exit(1)


def extract_samples(tar_path):
    """Extract per-sample 10X directories from RAW.tar."""
    print("\n[2/3] Extracting per-sample archives...")
    extracted_dirs = []

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"  {len(members)} files in archive")

        # GEO RAW.tar typically contains per-sample .tar.gz or individual files
        # Pattern: GSM######_<samplename>_barcodes.tsv.gz, etc.
        # Or: GSM######.tar.gz containing a 10X directory
        sample_files = {}
        for m in members:
            name = os.path.basename(m.name)
            # Extract GSM accession
            if name.startswith('GSM'):
                parts = name.split('_', 1)
                gsm = parts[0]
                if gsm not in sample_files:
                    sample_files[gsm] = []
                sample_files[gsm].append(m)

        print(f"  Found {len(sample_files)} samples")

        for gsm, files in sorted(sample_files.items()):
            sample_dir = f"{OUT_DIR}/{gsm}"
            os.makedirs(sample_dir, exist_ok=True)

            # Check if already extracted
            expected = ['barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz']
            if all(os.path.exists(f"{sample_dir}/{e}") for e in expected):
                extracted_dirs.append(sample_dir)
                continue

            for m in files:
                fname = os.path.basename(m.name)
                if fname.endswith('.tar.gz') or fname.endswith('.tgz'):
                    # Nested tarball per sample
                    tar.extract(m, OUT_DIR)
                    nested_path = f"{OUT_DIR}/{m.name}"
                    with tarfile.open(nested_path, 'r:gz') as nested:
                        nested.extractall(sample_dir)
                    os.remove(nested_path)
                else:
                    # Direct files: rename to standard 10X names
                    tar.extract(m, sample_dir)
                    src = f"{sample_dir}/{m.name}"
                    # Normalize filename
                    lower = fname.lower()
                    if 'barcode' in lower:
                        dst = f"{sample_dir}/barcodes.tsv.gz"
                    elif 'feature' in lower or 'gene' in lower:
                        dst = f"{sample_dir}/features.tsv.gz"
                    elif 'matrix' in lower:
                        dst = f"{sample_dir}/matrix.mtx.gz"
                    else:
                        dst = f"{sample_dir}/{fname}"
                    if src != dst and os.path.exists(src):
                        shutil.move(src, dst)

            extracted_dirs.append(sample_dir)

        print(f"  Extracted {len(extracted_dirs)} sample directories")
    return extracted_dirs


def parse_metadata():
    """Parse sample metadata from GEO to get stage annotations.

    Kumar et al. 2022 sample categories (from Cancer Discovery paper):
    - NAM: Normal adjacent mucosa
    - IM: Intestinal metaplasia
    - EAC/GC: Gastric cancer (intestinal/diffuse/mixed subtypes)
    """
    print("\n[3/3] Parsing sample metadata...")

    # Try to download series matrix for metadata
    matrix_url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/"
                  "GSE183nnn/GSE183904/matrix/GSE183904_series_matrix.txt.gz")
    matrix_path = f"{OUT_DIR}/series_matrix.txt.gz"

    if not os.path.exists(matrix_path):
        try:
            req = urllib.request.Request(matrix_url,
                                        headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                with open(matrix_path, 'wb') as f:
                    f.write(resp.read())
            print(f"  Downloaded series_matrix.txt.gz")
        except Exception as e:
            print(f"  Could not download series matrix: {e}")

    metadata = []

    if os.path.exists(matrix_path):
        # Parse series matrix for sample titles and characteristics
        titles = {}
        characteristics = {}
        geo_accessions = []

        with gzip.open(matrix_path, 'rt', errors='replace') as f:
            for line in f:
                if line.startswith('!Sample_geo_accession'):
                    geo_accessions = [x.strip('"') for x in
                                      line.strip().split('\t')[1:]]
                elif line.startswith('!Sample_title'):
                    vals = [x.strip('"') for x in line.strip().split('\t')[1:]]
                    for i, v in enumerate(vals):
                        if i < len(geo_accessions):
                            titles[geo_accessions[i]] = v
                elif line.startswith('!Sample_characteristics_ch1'):
                    vals = [x.strip('"') for x in line.strip().split('\t')[1:]]
                    for i, v in enumerate(vals):
                        if i < len(geo_accessions):
                            gsm = geo_accessions[i]
                            if gsm not in characteristics:
                                characteristics[gsm] = []
                            characteristics[gsm].append(v)

        print(f"  Found {len(geo_accessions)} samples in series matrix")

        # Map to stages based on title/characteristics
        for gsm in geo_accessions:
            title = titles.get(gsm, '')
            chars = characteristics.get(gsm, [])
            chars_str = ' '.join(chars).lower()
            title_lower = title.lower()
            combined = f"{title_lower} {chars_str}"

            # Stage mapping logic
            if any(k in combined for k in ['normal', 'adjacent', 'nam']):
                stage = 'NAG'
            elif any(k in combined for k in ['intestinal metaplasia', ' im ',
                                             'metaplasia']):
                stage = 'IM'
            elif any(k in combined for k in ['tumor', 'cancer', 'carcinoma',
                                             'gc', 'diffuse', 'intestinal type']):
                stage = 'GC'
            else:
                stage = 'unknown'

            # Subtype (for GC samples)
            subtype = 'none'
            if stage == 'GC':
                if 'diffuse' in combined:
                    subtype = 'diffuse'
                elif 'intestinal' in combined:
                    subtype = 'intestinal'
                elif 'mixed' in combined:
                    subtype = 'mixed'

            metadata.append({
                'accession': gsm,
                'title': title,
                'stage': stage,
                'gc_subtype': subtype,
                'characteristics': '; '.join(chars)
            })

    # Save metadata
    import csv
    meta_path = f"{OUT_DIR}/metadata.tsv"
    if metadata:
        with open(meta_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys(),
                                    delimiter='\t')
            writer.writeheader()
            writer.writerows(metadata)
        print(f"  Saved: {meta_path} ({len(metadata)} samples)")

        # Summary
        from collections import Counter
        stage_counts = Counter(m['stage'] for m in metadata)
        print(f"  Stage distribution:")
        for stage, count in sorted(stage_counts.items()):
            print(f"    {stage}: {count} samples")
    else:
        print("  WARNING: Could not parse metadata. Manual annotation needed.")
        print(f"  Please create {meta_path} with columns:")
        print(f"    accession\\ttitle\\tstage\\tgc_subtype\\tcharacteristics")

    return metadata


def main():
    print("=" * 60)
    print("Step 0: Download GSE183904 (Kumar et al. 2022)")
    print("=" * 60)

    print("\n[1/3] Downloading RAW.tar from GEO...")
    tar_path = download_raw_tar()

    extract_samples(tar_path)
    parse_metadata()

    # Verify: check a sample directory has expected files
    print("\n[Verify] Checking extracted data...")
    sample_dirs = [d for d in os.listdir(OUT_DIR)
                   if d.startswith('GSM') and os.path.isdir(f"{OUT_DIR}/{d}")]
    valid = 0
    for sd in sample_dirs[:5]:
        path = f"{OUT_DIR}/{sd}"
        has_mtx = any('matrix' in f for f in os.listdir(path))
        has_feat = any('feature' in f or 'gene' in f for f in os.listdir(path))
        has_bc = any('barcode' in f for f in os.listdir(path))
        if has_mtx and has_feat and has_bc:
            valid += 1
    print(f"  Validated {valid}/{min(5, len(sample_dirs))} sample directories")

    if valid == 0 and sample_dirs:
        print("\n  WARNING: Standard 10X structure not found.")
        print("  Listing first sample contents:")
        first = f"{OUT_DIR}/{sample_dirs[0]}"
        for f in os.listdir(first)[:10]:
            print(f"    {f}")
        print("  You may need to adjust extract_samples() for this format.")

    print(f"\n{'='*60}")
    print(f"Step 0 COMPLETE")
    print(f"  Samples: {len(sample_dirs)}")
    print(f"  Location: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
