import pandas as pd
import numpy as np

path = 'C:/FDU/Y4S2/xiyuan/project/dataset/GSE62254/41591_2015_BFnm3850_MOESM31_ESM.xls'
out_dir = 'C:/FDU/Y4S2/xiyuan/project/dataset/GSE62254'

df = pd.read_excel(path, sheet_name='ACRG')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3).to_string())
print(f"\nOS censor unique: {df['OS censor'].unique()}")

# Build survival CSV
# OS censor: 0=alive(censored), 1=dead(event) — verify
# Standard: OS=1 means event (death)
mapping = pd.read_csv(f"{out_dir}/GSE62254_sample_mapping.csv")

# Match Sample ID to GSM accession via patient_id
# Sample IDs in supplement are like "107", "108" matching patient IDs
surv = df[['Sample ID', 'OS (mos)', 'OS censor']].copy()
surv.columns = ['patient_id', 'OS_months', 'OS']
surv['patient_id'] = surv['patient_id'].astype(str)
surv['OS.time'] = surv['OS_months'] * 30.44  # months to days
surv = surv.dropna(subset=['OS.time', 'OS'])
surv['OS'] = surv['OS'].astype(int)

# Map patient_id to sample_id (GSM accession)
mapping['patient_id'] = mapping['patient_id'].astype(str)
surv_merged = surv.merge(mapping[['sample_id', 'patient_id']], on='patient_id', how='inner')
surv_final = surv_merged[['sample_id', 'OS.time', 'OS']].set_index('sample_id')

print(f"\nMatched survival: {len(surv_final)} samples")
print(f"Events (deaths): {surv_final['OS'].sum()}")
print(f"OS.time range: {surv_final['OS.time'].min():.0f} - {surv_final['OS.time'].max():.0f} days")

surv_final.to_csv(f"{out_dir}/GSE62254_survival.csv")
print(f"Saved: {out_dir}/GSE62254_survival.csv")
