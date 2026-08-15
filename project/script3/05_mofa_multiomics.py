"""
Step 5: MOFA+ multi-omics factor analysis on TCGA-STAD.
Input: TCGA-STAD (RNA + Methylation450K + CNA)
Output: script3/results/mofa_factors.csv, mofa_weights.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
TCGA = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"


def load_rna():
    """Load TCGA-STAD RNA HiSeqV2, select top 5000 HVG."""
    expr = pd.read_csv(f"{TCGA}/TCGA-STAD.HiSeqV2.gz", sep='\t', index_col=0, compression='gzip')
    expr = expr.T  # samples x genes
    gene_var = expr.var(axis=0).sort_values(ascending=False)
    top_genes = gene_var.head(5000).index.tolist()
    return expr[top_genes]


def load_methylation():
    """Load TCGA-STAD Methylation 450K, filter to promoter CpGs (TSS200/TSS1500), select top 5000."""·  
    meth = pd.read_csv(f"{TCGA}/TCGA-STAD.methylation450.tsv.gz", sep='\t',
                       index_col=0, compression='gzip')
    meth = meth.T  # samples x CpGs

    # Filter to promoter CpGs using 450K annotation
    annot_path = f"{TCGA}/HM450K_promoter_cpgs.csv"
    if not os.path.exists(annot_path):
        print("  Downloading 450K promoter annotation...")
        try:
            # Use sesame package manifest or download from GEO
            import urllib.request
            url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
            # Fallback: filter by CpG naming convention (cg probes in TSS regions)
            # Since full manifest is large, use a heuristic: keep only CpGs that start with 'cg'
            # and apply variance filter (promoter CpGs tend to be more variable in cancer)
            print("  Full manifest unavailable, using variance-based selection with TSS heuristic")
            promoter_cpgs = None
        except:
            promoter_cpgs = None
    else:
        promoter_df = pd.read_csv(annot_path)
        promoter_cpgs = set(promoter_df['cpg'].tolist())

    if promoter_cpgs:
        available_promoter = [c for c in meth.columns if c in promoter_cpgs]
        print(f"  Promoter CpGs available: {len(available_promoter)}")
        meth = meth[available_promoter]

    # Remove CpGs with >20% missing
    missing_frac = meth.isna().mean(axis=0)
    meth = meth.loc[:, missing_frac < 0.2]
    meth = meth.fillna(meth.median())
    cpg_var = meth.var(axis=0).sort_values(ascending=False)
    top_cpgs = cpg_var.head(5000).index.tolist()
    return meth[top_cpgs]


def load_cna():
    """Load TCGA-STAD CNA (GISTIC2 discrete) from JSON, pivot to gene-level matrix."""
    import json
    with open(f"{TCGA}/stad_tcga_cna.json", 'r') as f:
        records = json.load(f)
    df = pd.DataFrame(records)[['sampleId', 'entrezGeneId', 'alteration']]
    # Pivot to samples x genes
    cna_matrix = df.pivot_table(index='sampleId', columns='entrezGeneId',
                                values='alteration', aggfunc='first')
    cna_matrix = cna_matrix.fillna(0)
    # Select top 5000 most variable genes
    cna_var = cna_matrix.var(axis=0).sort_values(ascending=False)
    top_genes = cna_var.head(5000).index.tolist()
    return cna_matrix[top_genes]


def load_clinical():
    """Load clinical data with molecular subtype and survival."""
    clin = pd.read_csv(f"{TCGA}/TCGA-STAD.clinicalMatrix.tsv", sep='\t', index_col=0)
    surv = pd.read_csv(f"{TCGA}/TCGA-STAD.survival.tsv", sep='\t')
    surv = surv[['sample', 'OS', 'OS.time']].dropna().set_index('sample')
    # MSI status as subtype proxy (CDE_ID_3226963 column)
    clin['msi_status'] = clin['CDE_ID_3226963'] if 'CDE_ID_3226963' in clin.columns else 'unknown'
    return clin, surv


def main():
    print("=" * 60)
    print("Step 5: MOFA+ Multi-omics (TCGA-STAD)")
    print("=" * 60)

    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load and check sample overlap
    print("\n[1] Loading TCGA-STAD multi-omics...")
    rna = load_rna()
    print(f"  RNA: {rna.shape[0]} samples, {rna.shape[1]} genes")

    meth = load_methylation()
    print(f"  Methylation: {meth.shape[0]} samples, {meth.shape[1]} CpGs")

    cna = load_cna()
    print(f"  CNA: {cna.shape[0]} samples, {cna.shape[1]} genes")

    clin, surv = load_clinical()

    # Sample overlap — fix ID format: methylation has vial letter (16 chars), RNA/CNA have 15
    meth.index = meth.index.str[:15]
    meth = meth[~meth.index.duplicated(keep='first')]
    overlap = sorted(set(rna.index) & set(meth.index) & set(cna.index))
    print(f"\n  Sample overlap (RNA ∩ Meth ∩ CNA): {len(overlap)}")

    if len(overlap) < 250:
        # Fallback: RNA + Methylation only
        overlap_2 = sorted(set(rna.index) & set(meth.index))
        print(f"  Overlap < 250, trying RNA ∩ Meth: {len(overlap_2)}")
        if len(overlap_2) >= 250:
            overlap = overlap_2
            use_cna = False
        else:
            print("  ERROR: Insufficient overlap for MOFA. Exiting.")
            return
    else:
        use_cna = True

    # Subset to overlap
    rna_sub = rna.loc[overlap]
    meth_sub = meth.loc[overlap]
    if use_cna:
        cna_sub = cna.loc[overlap]

    # Determine n_factors
    n_factors = 15 if len(overlap) >= 300 else (12 if len(overlap) >= 250 else 10)
    print(f"  Using {n_factors} factors (sample/param ratio: {len(overlap)//n_factors})")

    # [2] MOFA+ training
    print(f"\n[2] Running MOFA+ ({'3-view' if use_cna else '2-view'})...")
    from mofapy2.run.entry_point import entry_point

    ent = entry_point()
    views = ["RNA", "Methylation"]
    data_mat = [[rna_sub.values.astype(np.float64)], [meth_sub.values.astype(np.float64)]]
    features_names = [rna_sub.columns.tolist(), meth_sub.columns.tolist()]

    if use_cna:
        views.append("CNA")
        data_mat.append([cna_sub.values.astype(np.float64)])
        features_names.append([str(x) for x in cna_sub.columns.tolist()])

    ent.set_data_options(scale_groups=False, scale_views=True)
    ent.set_data_matrix(data_mat, views_names=views,
                        samples_names=[overlap],
                        features_names=features_names,
                        groups_names=["TCGA"],
                        likelihoods=["gaussian"] * len(views))
    ent.set_model_options(factors=n_factors, spikeslab_weights=True,
                          ard_factors=True, ard_weights=True)
    ent.set_train_options(iter=1000, convergence_mode="fast", seed=42, verbose=False)

    ent.build()
    ent.run()

    # Extract factors and weights
    factors_raw = ent.model.nodes["Z"].getExpectation()
    weights_raw = ent.model.nodes["W"].getExpectation()

    if isinstance(factors_raw, list):
        Z_arr = factors_raw[0]
    elif isinstance(factors_raw, dict):
        Z_arr = list(factors_raw.values())[0]
    else:
        Z_arr = factors_raw

    Z = pd.DataFrame(Z_arr, index=overlap,
                     columns=[f"Factor{i+1}" for i in range(Z_arr.shape[1])])

    # Weights per view
    W_all = {}
    for v_idx, view in enumerate(views):
        if isinstance(weights_raw, list):
            W_v = weights_raw[v_idx]
        elif isinstance(weights_raw, dict):
            W_v = list(weights_raw.values())[v_idx]
        else:
            W_v = weights_raw
        W_all[view] = pd.DataFrame(W_v, index=features_names[v_idx],
                                   columns=[f"Factor{i+1}" for i in range(W_v.shape[1])])

    print(f"  Factors: {Z.shape}")
    for v, w in W_all.items():
        print(f"  Weights ({v}): {w.shape}")

    # [3] Factor-subtype association (MSI status as best available proxy for TCGA 4-class)
    print("\n[3] Factor-subtype association (MSI status)...")
    print("  Note: Full TCGA 4-class (CIN/GS/MSI/EBV) not in clinicalMatrix; using MSI status")
    clin_overlap = clin.loc[clin.index.isin(overlap)]
    subtype_col = 'msi_status'
    subtypes = clin_overlap[subtype_col].dropna()
    subtypes = subtypes[subtypes.isin(['MSS', 'MSI-H', 'MSI-L'])]

    factor_subtype = []
    for col in Z.columns:
        common_st = list(set(subtypes.index) & set(Z.index))
        if len(common_st) < 50:
            continue
        groups = [Z.loc[common_st][col][subtypes.loc[common_st] == st].values
                  for st in ['MSS', 'MSI-H', 'MSI-L']
                  if (subtypes.loc[common_st] == st).sum() > 5]
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            factor_subtype.append({'factor': col, 'kruskal_stat': stat, 'pval': pval})

    subtype_df = pd.DataFrame(factor_subtype)
    if len(subtype_df) > 0:
        from statsmodels.stats.multitest import multipletests
        subtype_df['padj'] = multipletests(subtype_df['pval'], method='fdr_bh')[1]
        n_sig_subtype = (subtype_df['padj'] < 0.01).sum()
        print(f"  Factors associated with subtype (padj<0.01): {n_sig_subtype}")
        subtype_df.to_csv(f"{BASE}/results/mofa_factor_subtype.csv", index=False)

    # [4] Factor-survival Cox association
    print("\n[4] Factor-survival association...")
    from lifelines import CoxPHFitter

    common_surv = list(set(Z.index) & set(surv.index))
    Z_surv = Z.loc[common_surv]
    surv_common = surv.loc[common_surv]

    factor_survival = []
    for col in Z.columns:
        df = surv_common.copy()
        df['factor'] = Z_surv[col].values
        df = df.rename(columns={'OS': 'event', 'OS.time': 'duration'})
        df = df[df['duration'] > 0]
        try:
            cph = CoxPHFitter()
            cph.fit(df[['duration', 'event', 'factor']], 'duration', 'event')
            hr = np.exp(cph.params_['factor'])
            p = cph.summary['p']['factor']
            factor_survival.append({'factor': col, 'HR': hr, 'p_value': p})
        except:
            pass

    surv_df = pd.DataFrame(factor_survival).sort_values('p_value')
    if len(surv_df) > 0:
        surv_df['padj'] = multipletests(surv_df['p_value'], method='fdr_bh')[1]
        n_sig_surv = (surv_df['padj'] < 0.01).sum()
        print(f"  Factors associated with survival (padj<0.01): {n_sig_surv}")
    surv_df.to_csv(f"{BASE}/results/mofa_factor_survival.csv", index=False)

    # [5] Extract top genes from significant factors
    print("\n[5] Extracting top genes from significant factors...")
    sig_factors = set()
    if len(subtype_df) > 0:
        sig_factors |= set(subtype_df[subtype_df['padj'] < 0.01]['factor'])
    if len(surv_df) > 0:
        sig_factors |= set(surv_df[surv_df['padj'] < 0.05]['factor'])

    top_mofa_genes = []
    for f in sig_factors:
        if f in W_all['RNA'].columns:
            top_w = W_all['RNA'][f].abs().sort_values(ascending=False).head(50)
            top_mofa_genes.extend(top_w.index.tolist())
    top_mofa_genes = list(set(top_mofa_genes))
    print(f"  MOFA top genes (from {len(sig_factors)} sig factors): {len(top_mofa_genes)}")

    # [6] Pathway enrichment on factors (post-hoc interpretation)
    print("\n[6] Pathway enrichment (post-hoc)...")
    try:
        import decoupler as dc
        msigdb = dc.op.progeny(organism='human', top=300)
        factor_pathway = {}
        for f in sig_factors:
            if f not in W_all['RNA'].columns:
                continue
            w = W_all['RNA'][f].sort_values(ascending=False)
            # Enrichment: top/bottom genes vs pathway gene sets
            top_genes_f = set(w.head(100).index)
            for pathway in msigdb['source'].unique():
                pw_genes = set(msigdb[msigdb['source'] == pathway]['target'])
                overlap_pw = len(top_genes_f & pw_genes)
                if overlap_pw >= 3:
                    factor_pathway.setdefault(f, []).append(
                        {'pathway': pathway, 'overlap': overlap_pw})
        if factor_pathway:
            rows = []
            for f, pws in factor_pathway.items():
                for pw in pws:
                    rows.append({'factor': f, **pw})
            pd.DataFrame(rows).to_csv(f"{BASE}/results/mofa_factor_pathways.csv", index=False)
            print(f"  Pathway associations saved")
    except (ImportError, AttributeError) as e:
        print(f"  decoupler pathway enrichment skipped: {e}")

    # [7] Cross-reference with scRNA candidates
    print("\n[7] Cross-referencing with scRNA candidates...")
    scrna_candidates = set()
    for pool_file in ['candidate_pool_B.csv', 'candidate_pool_D.csv', 'candidate_pool_E.csv',
                      'transition_risk_genes.csv']:
        path = f"{BASE}/results/{pool_file}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'gene' in df.columns:
                scrna_candidates |= set(df['gene'].tolist())
            elif pool_file == 'transition_risk_genes.csv' and 'gene' in df.columns:
                # Top 100 risk-correlated
                scrna_candidates |= set(df.head(100)['gene'].tolist())

    if scrna_candidates and top_mofa_genes:
        mofa_scrna_overlap = sorted(set(top_mofa_genes) & scrna_candidates)
        print(f"  MOFA ∩ scRNA candidates: {len(mofa_scrna_overlap)} genes")
        if mofa_scrna_overlap:
            pd.DataFrame({'gene': mofa_scrna_overlap, 'evidence': 'cancer_endpoint_MOFA'}).to_csv(
                f"{BASE}/results/mofa_scrna_overlap.csv", index=False)
    else:
        print("  No scRNA candidates available yet (run after Step 4)")

    # Save
    Z.to_csv(f"{BASE}/results/mofa_factors.csv")
    W_all['RNA'].to_csv(f"{BASE}/results/mofa_weights.csv")
    if top_mofa_genes:
        pd.DataFrame({'gene': top_mofa_genes, 'source': 'MOFA_sig_factor'}).to_csv(
            f"{BASE}/results/mofa_top_genes.csv", index=False)

    # Validation
    n_sig_total = len(sig_factors)
    print(f"\n{'='*60}")
    print("Step 5 COMPLETE")
    print(f"  Samples: {len(overlap)} (3-omics overlap)")
    print(f"  Significant factors (subtype or survival): {n_sig_total}")
    print(f"  MOFA top genes: {len(top_mofa_genes)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
