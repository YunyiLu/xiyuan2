import requests, os
import pandas as pd

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
session.verify = False

OUT = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE62254"

# cBioPortal datahub on GitHub has ACRG clinical data
urls = [
    "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/stad_asian/data_clinical_patient.txt",
    "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/egc_tmucih_2015/data_clinical_patient.txt",
]

for url in urls:
    try:
        r = session.get(url, timeout=30)
        fname = url.split("/")[-2]
        print(f"{fname}: status={r.status_code}, size={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 100:
            out_path = f"{OUT}/cbioportal_{fname}_clinical.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            # Parse
            lines = [l for l in r.text.split('\n') if not l.startswith('#')]
            if lines:
                from io import StringIO
                df = pd.read_csv(StringIO('\n'.join(lines)), sep='\t')
                print(f"  Columns: {df.columns.tolist()[:10]}")
                print(f"  Rows: {len(df)}")
                os_cols = [c for c in df.columns if 'OS' in c.upper() or 'SURV' in c.upper() or 'MONTH' in c.upper()]
                print(f"  OS-related: {os_cols}")
                if os_cols:
                    print(df[os_cols].head(3).to_string())
    except Exception as e:
        print(f"  Error: {e}")
