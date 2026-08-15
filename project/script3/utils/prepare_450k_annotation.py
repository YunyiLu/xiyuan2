"""Utility: Generate promoter CpG list for 450K array from Illumina manifest."""
import pandas as pd
import os

TCGA = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"
OUT = f"{TCGA}/HM450K_promoter_cpgs.csv"

if os.path.exists(OUT):
    print(f"Already exists: {OUT}")
else:
    # Download manifest (skip header rows)
    url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
    print("Downloading 450K manifest (~50MB)...")
    manifest = pd.read_csv(url, skiprows=7, low_memory=False,
                           usecols=['IlsmnID', 'UCSC_RefGene_Group', 'UCSC_RefGene_Name'])
    manifest = manifest.rename(columns={'IlsmnID': 'cpg'})
    # Filter promoter: TSS200 or TSS1500
    promoter = manifest[manifest['UCSC_RefGene_Group'].str.contains('TSS200|TSS1500', na=False)]
    promoter[['cpg', 'UCSC_RefGene_Name']].to_csv(OUT, index=False)
    print(f"Saved {len(promoter)} promoter CpGs to {OUT}")
