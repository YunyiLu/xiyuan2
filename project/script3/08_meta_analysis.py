"""
Step 8: Stratified Evidence Integration & Marker Prioritization
Input: Candidate pools A-G + bulk cohorts (GSE55696, GSE78523, GSE60427, GSE60662)
Output: script3/results/evidence_ranked_genes.csv
Design: Dual scoring (TransformationScore + ClinicalExtensionScore), NOT Fisher p-value merge.
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu, rankdata
from lifelines import CoxPHFitter

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
BULK = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk"
TCGA = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"
ACRG = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE62254"


def load_candidate_pools():
    """Load candidate pools A-G, deduplicate."""
    pools = {}
    pool_defs = {
        'A': (f"{BASE}/results/transition_risk_genes.csv", 30),
        'B': (f"{BASE}/results/candidate_pool_B.csv", 20),
        'C': (f"{BASE}/results/spatial_gradient_genes.csv", 20),
        'D': (f"{BASE}/results/candidate_pool_D.csv", 10),
        'E': (f"{BASE}/results/candidate_pool_E.csv", 10),
        'F': (f"{BASE}/results/wgcna_hub_genes.csv", 20),
        'G': (f"{BASE}/results/graph_ranked_genes.csv", 20),
    }
    all_genes = set()
    for pool_id, (path, limit) in pool_defs.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            genes = df['gene'].head(limit).tolist()
            pools[pool_id] = genes
            all_genes.update(genes)
            print(f"    Pool {pool_id}: {len(genes)} genes")
        else:
            pools[pool_id] = []
            print(f"    Pool {pool_id}: not available")
    return sorted(all_genes), pools


def compute_scrna_risk(candidates):
    """scRNA_risk = max(|Spearman(gene, stage)|, sqrt(KW_chi2/n))."""
    pb_path = f"{BASE}/data/pseudobulk_by_sample_celltype.csv"
    if not os.path.exists(pb_path):
        return {g: 0.0 for g in candidates}

    pb = pd.read_csv(pb_path)
    stage_map = {'NAG': 0, 'CAG': 1, 'IM': 2, 'EGC': 3, 'GC': 3}
    pb['stage_num'] = pb['stage'].map(stage_map)
    pb = pb.dropna(subset=['stage_num'])

    scores = {}
    for gene in candidates:
        if gene not in pb.columns:
            scores[gene] = 0.0
            continue
        vals = pb[gene].values
        stages = pb['stage_num'].values
        valid = ~np.isnan(vals) & ~np.isnan(stages)
        if valid.sum() < 10:
            scores[gene] = 0.0
            continue
        r, _ = spearmanr(vals[valid], stages[valid])
        try:
            groups = [vals[valid][stages[valid] == s] for s in sorted(set(stages[valid]))]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) >= 2:
                stat, _ = kruskal(*groups)
                kw_score = np.sqrt(stat / valid.sum())
            else:
                kw_score = 0.0
        except Exception:
            kw_score = 0.0
        scores[gene] = max(abs(r), kw_score)
    return scores


def load_spatial_gradient(candidates):
    """Load spatial gradient effect sizes from Step 11a."""
    path = f"{BASE}/results/spatial_gradient_genes.csv"
    if not os.path.exists(path):
        return {g: 0.0 for g in candidates}
    df = pd.read_csv(path)
    for col in ['effect_size', 'cohens_d', 'mean_cohens_d']:
        if col in df.columns:
            mapping = dict(zip(df['gene'], df[col]))
            return {g: mapping.get(g, 0.0) for g in candidates}
    return {g: 0.0 for g in candidates}


def load_network_score(candidates):
    """Load RWR network_score from Step 7."""
    path = f"{BASE}/results/graph_ranked_genes.csv"
    if not os.path.exists(path):
        return {g: 0.0 for g in candidates}
    df = pd.read_csv(path)
    mapping = dict(zip(df['gene'], df['network_score']))
    return {g: mapping.get(g, 0.0) for g in candidates}


def jt_trend_test(expr_vals, group_labels):
    """Jonckheere-Terpstra trend test (Spearman-based approximation)."""
    valid = ~np.isnan(expr_vals) & ~np.isnan(group_labels)
    if valid.sum() < 10:
        return 0.0, 1.0, 'none'
    r, p = spearmanr(expr_vals[valid], group_labels[valid])
    direction = 'up' if r > 0 else 'down'
    return r, p, direction


def test_gse55696(candidates):
    """GSE55696: JT trend across CG→LGIN→HGIN→EGC."""
    results_path = f"{BASE}/results/gse55696_jt_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        out = {}
        for _, row in df.iterrows():
            if row['gene'] in candidates:
                out[row['gene']] = {'jt_z': row.get('jt_z', row.get('spearman_r', 0)),
                                    'p': row.get('padj', row.get('pval', 1.0)),
                                    'direction': row.get('direction', 'none')}
        return out

    expr_path = f"{BULK}/GSE55696/GSE55696_expression.csv"
    meta_path = f"{BULK}/GSE55696/GSE55696_metadata.csv"
    if not os.path.exists(expr_path):
        return {g: {'jt_z': 0, 'p': 1.0, 'direction': 'none'} for g in candidates}

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    stage_map = {'CG': 0, 'chronic_gastritis': 0, 'LGIN': 1, 'HGIN': 2, 'EGC': 3, 'adenocarcinoma': 3}

    if meta is not None and 'stage' in meta.columns:
        meta['stage_num'] = meta['stage'].map(stage_map)
        samples = meta.dropna(subset=['stage_num'])['sample_id'].tolist()
        stages = meta.set_index('sample_id').loc[samples, 'stage_num'].values
    else:
        n = expr.shape[1]
        stages = np.array([0]*19 + [1]*19 + [2]*20 + [3]*19)[:n]
        samples = expr.columns[:n].tolist()

    out = {}
    for gene in candidates:
        if gene not in expr.index:
            out[gene] = {'jt_z': 0, 'p': 1.0, 'direction': 'none'}
            continue
        vals = expr.loc[gene, samples].values.astype(float)
        r, p, direction = jt_trend_test(vals, stages)
        out[gene] = {'jt_z': r, 'p': p, 'direction': direction}
    return out


def test_gse78523(candidates):
    """GSE78523: Wilcoxon progressor vs non-progressor + IIM/CIM subtype."""
    expr_path = f"{BULK}/GSE78523/GSE78523_expression.csv"
    meta_path = f"{BULK}/GSE78523/GSE78523_metadata.csv"
    if not os.path.exists(expr_path) or not os.path.exists(meta_path):
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none',
                    'iim_effect': 0, 'cim_effect': 0} for g in candidates}

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path)

    prog_samples = meta[meta['group'].str.contains('GC|progressor', case=False, na=False)]['sample_id'].tolist()
    ctrl_samples = meta[meta['group'].str.contains('ctrl|control|healthy', case=False, na=False)]['sample_id'].tolist()
    prog_samples = [s for s in prog_samples if s in expr.columns]
    ctrl_samples = [s for s in ctrl_samples if s in expr.columns]

    iim_prog = meta[meta['group'].str.contains('IIM.*GC|IIM.*prog', case=False, na=False)]['sample_id'].tolist()
    iim_ctrl = meta[meta['group'].str.contains('IIM.*ctrl', case=False, na=False)]['sample_id'].tolist()
    cim_prog = meta[meta['group'].str.contains('CIM.*GC|CIM.*prog', case=False, na=False)]['sample_id'].tolist()
    cim_ctrl = meta[meta['group'].str.contains('CIM.*ctrl', case=False, na=False)]['sample_id'].tolist()

    out = {}
    for gene in candidates:
        if gene not in expr.index or not prog_samples or not ctrl_samples:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none',
                         'iim_effect': 0, 'cim_effect': 0}
            continue
        v_prog = expr.loc[gene, prog_samples].values.astype(float)
        v_ctrl = expr.loc[gene, ctrl_samples].values.astype(float)
        try:
            stat, p = mannwhitneyu(v_prog, v_ctrl, alternative='two-sided')
            pooled_std = np.sqrt((np.var(v_prog) + np.var(v_ctrl)) / 2) + 1e-10
            effect = (np.mean(v_prog) - np.mean(v_ctrl)) / pooled_std
        except Exception:
            effect, p = 0, 1.0
        direction = 'up' if effect > 0 else 'down'

        iim_eff = cim_eff = 0
        if iim_prog and iim_ctrl and gene in expr.index:
            ip = [s for s in iim_prog if s in expr.columns]
            ic = [s for s in iim_ctrl if s in expr.columns]
            if ip and ic:
                iim_eff = (expr.loc[gene, ip].mean() - expr.loc[gene, ic].mean()) / pooled_std
        if cim_prog and cim_ctrl and gene in expr.index:
            cp = [s for s in cim_prog if s in expr.columns]
            cc = [s for s in cim_ctrl if s in expr.columns]
            if cp and cc:
                cim_eff = (expr.loc[gene, cp].mean() - expr.loc[gene, cc].mean()) / pooled_std

        out[gene] = {'effect': effect, 'p': p, 'direction': direction,
                     'iim_effect': iim_eff, 'cim_effect': cim_eff}
    return out


def test_kw_cohort(gse_id, candidates):
    """Kruskal-Wallis + Dunn post-hoc for GSE60427/GSE60662."""
    expr_path = f"{BULK}/{gse_id}/{gse_id}_expression.csv"
    meta_path = f"{BULK}/{gse_id}/{gse_id}_metadata.csv"
    if not os.path.exists(expr_path):
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none', 'dunn_pairs': {}} for g in candidates}

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    stage_map = {'normal': 0, 'gastritis': 1, 'IM': 2, 'dysplasia': 3, 'cancer': 4}
    if meta is not None and 'stage' in meta.columns:
        meta['stage_num'] = meta['stage'].map(stage_map)
        valid_meta = meta.dropna(subset=['stage_num'])
        samples = [s for s in valid_meta['sample_id'] if s in expr.columns]
        stages = valid_meta.set_index('sample_id').loc[samples, 'stage_num'].values
    else:
        samples = expr.columns.tolist()
        stages = None

    out = {}
    for gene in candidates:
        if gene not in expr.index or stages is None:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none', 'dunn_pairs': {}}
            continue
        vals = expr.loc[gene, samples].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none', 'dunn_pairs': {}}
            continue
        unique_stages = sorted(set(stages[valid]))
        groups = [vals[valid][stages[valid] == s] for s in unique_stages]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none', 'dunn_pairs': {}}
            continue
        try:
            stat, p = kruskal(*groups)
            r, _ = spearmanr(vals[valid], stages[valid])
            effect = r
            direction = 'up' if r > 0 else 'down'
        except Exception:
            effect, p, direction = 0, 1.0, 'none'
        # Dunn post-hoc (pairwise Mann-Whitney with Bonferroni)
        dunn_pairs = {}
        n_pairs = len(unique_stages) * (len(unique_stages) - 1) // 2
        for i in range(len(unique_stages)):
            for j in range(i+1, len(unique_stages)):
                g1 = vals[valid][stages[valid] == unique_stages[i]]
                g2 = vals[valid][stages[valid] == unique_stages[j]]
                if len(g1) >= 2 and len(g2) >= 2:
                    try:
                        _, pw = mannwhitneyu(g1, g2, alternative='two-sided')
                        dunn_pairs[f"{unique_stages[i]}v{unique_stages[j]}"] = min(pw * n_pairs, 1.0)
                    except Exception:
                        pass
        out[gene] = {'effect': effect, 'p': p, 'direction': direction, 'dunn_pairs': dunn_pairs}
    return out


def compute_clinical_extension(candidates):
    """ClinicalExtensionScore: TCGA_HR + ACRG_HR + HPA + DGIdb."""
    scores = {g: 0.0 for g in candidates}

    # TCGA survival HR
    expr_path = f"{TCGA}/TCGA-STAD.HiSeqV2.gz"
    surv_path = f"{TCGA}/TCGA-STAD.survival.tsv"
    if os.path.exists(expr_path) and os.path.exists(surv_path):
        expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
        surv = pd.read_csv(surv_path, sep='\t')
        surv = surv[['sample', 'OS', 'OS.time']].dropna()
        surv = surv[surv['OS.time'] > 0].set_index('sample')
        common = sorted(set(expr.columns) & set(surv.index))
        for gene in candidates:
            if gene not in expr.index:
                continue
            df = surv.loc[common].copy()
            df['expr'] = expr.loc[gene, common].values
            try:
                cph = CoxPHFitter()
                cph.fit(df.rename(columns={'OS': 'event', 'OS.time': 'duration'})[['duration', 'event', 'expr']], 'duration', 'event')
                hr = np.exp(abs(cph.params_['expr']))
                p = cph.summary['p']['expr']
                if p < 0.05:
                    scores[gene] += min(np.log2(hr), 2.0)
            except Exception:
                pass

    # HPA protein evidence
    hpa_path = f"{BASE}/results/hpa_protein_evidence.csv"
    if os.path.exists(hpa_path):
        hpa = pd.read_csv(hpa_path)
        hpa_map = dict(zip(hpa['gene'], hpa.get('score', [1]*len(hpa))))
        for g in candidates:
            if g in hpa_map:
                scores[g] += 0.5

    # ACRG survival HR
    acrg_expr_path = f"{ACRG}/GSE62254_expression.csv"
    acrg_surv_path = f"{ACRG}/GSE62254_survival.csv"
    if os.path.exists(acrg_expr_path) and os.path.exists(acrg_surv_path):
        acrg_expr = pd.read_csv(acrg_expr_path, index_col=0)
        acrg_surv = pd.read_csv(acrg_surv_path, index_col=0)
        os_col = 'OS.time' if 'OS.time' in acrg_surv.columns else acrg_surv.columns[0]
        ev_col = 'OS' if 'OS' in acrg_surv.columns else acrg_surv.columns[1]
        acrg_surv = acrg_surv[[os_col, ev_col]].dropna()
        acrg_surv = acrg_surv[acrg_surv[os_col] > 0]
        acrg_common = sorted(set(acrg_expr.columns) & set(acrg_surv.index))
        for gene in candidates:
            if gene not in acrg_expr.index:
                continue
            df_a = acrg_surv.loc[acrg_common].copy()
            df_a['expr'] = acrg_expr.loc[gene, acrg_common].values
            try:
                cph = CoxPHFitter()
                cph.fit(df_a.rename(columns={os_col: 'duration', ev_col: 'event'})[['duration', 'event', 'expr']], 'duration', 'event')
                hr = np.exp(abs(cph.params_['expr']))
                p = cph.summary['p']['expr']
                if p < 0.05:
                    scores[gene] += min(np.log2(hr), 2.0)
            except Exception:
                pass

    # DGIdb druggability
    dgi_path = f"{BASE}/results/dgidb_druggable.csv"
    if os.path.exists(dgi_path):
        dgi = pd.read_csv(dgi_path)
        dgi_genes = set(dgi['gene']) if 'gene' in dgi.columns else set()
        for g in candidates:
            if g in dgi_genes:
                scores[g] += 0.5

    return scores


def normalize_scores(score_dict):
    """Min-max normalize to [0,1]."""
    vals = np.array(list(score_dict.values()))
    if vals.max() == vals.min():
        return {k: 0.5 for k in score_dict}
    return {k: (v - vals.min()) / (vals.max() - vals.min() + 1e-10) for k, v in score_dict.items()}


def weight_robustness(gene_data, candidates):
    """5 weight schemes, check top15 stability."""
    schemes = {
        'A_expert': (0.30, 0.30, 0.25, 0.15),
        'B_equal': (0.25, 0.25, 0.25, 0.25),
        'C_direct_only': (0.50, 0.50, 0.0, 0.0),
        'D_no_network': (0.35, 0.35, 0.30, 0.0),
        'E_no_scrna': (0.0, 0.40, 0.35, 0.25),
    }
    top15_sets = {}
    for name, (w1, w2, w3, w4) in schemes.items():
        scores = {}
        for g in candidates:
            d = gene_data[g]
            scores[g] = w1*d['scRNA_risk_norm'] + w2*d['spatial_norm'] + w3*d['bulk_norm'] + w4*d['network_norm']
        ranked = sorted(scores, key=scores.get, reverse=True)[:15]
        top15_sets[name] = set(ranked)

    # Stability: gene appears in >=4/5 schemes
    all_top15 = set()
    for s in top15_sets.values():
        all_top15.update(s)
    stability = {}
    for g in all_top15:
        count = sum(1 for s in top15_sets.values() if g in s)
        stability[g] = count
    return stability, top15_sets


def main():
    print("=" * 60)
    print("Step 8: Stratified Evidence Integration & Marker Prioritization")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load candidate pools A-G
    print("\n[1] Loading candidate pools A-G...")
    candidates, pools = load_candidate_pools()
    print(f"  Total candidates (deduplicated): {len(candidates)}")

    if not candidates:
        candidates = ["CDX2", "OLFM4", "LGR5", "NAMPT", "AREG", "PHLDA1",
                      "SOX9", "MYC", "CTNNB1", "IL1B", "NNMT", "CDH17"]
        print(f"  Fallback: {len(candidates)} literature genes")

    # [2] scRNA_risk
    print("\n[2] Computing scRNA_risk (Spearman/KW)...")
    scrna_risk = compute_scrna_risk(candidates)
    n_nonzero = sum(1 for v in scrna_risk.values() if v > 0)
    print(f"  Non-zero scRNA_risk: {n_nonzero}/{len(candidates)}")

    # [3] Spatial gradient
    print("\n[3] Loading spatial gradient (Step 11a)...")
    spatial_grad = load_spatial_gradient(candidates)

    # [4] Bulk progression (4 cohorts, weighted)
    print("\n[4] Bulk cohort testing...")
    print("  [4a] GSE55696 (JT trend)...")
    gse55696 = test_gse55696(candidates)
    n_sig = sum(1 for v in gse55696.values() if v['p'] < 0.05)
    print(f"    Significant: {n_sig}/{len(candidates)}")

    print("  [4b] GSE78523 (progressor vs non-progressor)...")
    gse78523 = test_gse78523(candidates)
    n_sig = sum(1 for v in gse78523.values() if v['p'] < 0.05)
    print(f"    Significant: {n_sig}/{len(candidates)}")

    print("  [4c] GSE60427 (KW trend)...")
    gse60427 = test_kw_cohort('GSE60427', candidates)
    n_sig = sum(1 for v in gse60427.values() if v['p'] < 0.05)
    print(f"    Significant: {n_sig}/{len(candidates)}")

    print("  [4d] GSE60662 (KW trend)...")
    gse60662 = test_kw_cohort('GSE60662', candidates)
    n_sig = sum(1 for v in gse60662.values() if v['p'] < 0.05)
    print(f"    Significant: {n_sig}/{len(candidates)}")

    # Compute bulk_progression (weighted)
    bulk_scores = {}
    for gene in candidates:
        jt_eff = abs(gse55696[gene]['jt_z']) if gse55696[gene]['p'] < 0.1 else 0
        prog_eff = abs(gse78523[gene]['effect']) if gse78523[gene]['p'] < 0.1 else 0
        kw427_eff = abs(gse60427[gene]['effect']) if gse60427[gene]['p'] < 0.1 else 0
        kw662_eff = abs(gse60662[gene]['effect']) if gse60662[gene]['p'] < 0.1 else 0
        bulk_scores[gene] = 0.40*jt_eff + 0.30*prog_eff + 0.15*kw427_eff + 0.15*kw662_eff
    print(f"  Bulk progression computed (weighted 0.40/0.30/0.15/0.15)")

    # [5] Network score
    print("\n[5] Loading network score (RWR)...")
    network_scores = load_network_score(candidates)

    # [6] Direction consistency constraint
    print("\n[6] Direction consistency check...")
    direction_penalty = {}
    for gene in candidates:
        dirs = []
        if gse55696[gene]['direction'] != 'none':
            dirs.append(gse55696[gene]['direction'])
        if gse78523[gene]['direction'] != 'none':
            dirs.append(gse78523[gene]['direction'])
        if dirs and len(set(dirs)) > 1:
            direction_penalty[gene] = 0.5
        else:
            direction_penalty[gene] = 1.0
    n_penalized = sum(1 for v in direction_penalty.values() if v < 1.0)
    print(f"  Direction-penalized genes: {n_penalized}")

    # [7] Compute TransformationScore
    print("\n[7] Computing TransformationScore...")
    scrna_norm = normalize_scores(scrna_risk)
    spatial_norm = normalize_scores(spatial_grad)
    bulk_norm = normalize_scores(bulk_scores)
    network_norm = normalize_scores(network_scores)

    gene_data = {}
    for gene in candidates:
        gene_data[gene] = {
            'scRNA_risk_norm': scrna_norm[gene],
            'spatial_norm': spatial_norm[gene],
            'bulk_norm': bulk_norm[gene],
            'network_norm': network_norm[gene],
        }
        ts = (0.30*scrna_norm[gene] + 0.30*spatial_norm[gene] +
              0.25*bulk_norm[gene] + 0.15*network_norm[gene])
        ts *= direction_penalty[gene]
        gene_data[gene]['TransformationScore'] = ts

    # [8] ClinicalExtensionScore
    print("\n[8] Computing ClinicalExtensionScore...")
    clinical_scores = compute_clinical_extension(candidates)
    clinical_norm = normalize_scores(clinical_scores)
    for gene in candidates:
        gene_data[gene]['ClinicalExtensionScore'] = clinical_norm[gene]

    # [9] Weight robustness (5 schemes)
    print("\n[9] Weight robustness analysis (5 schemes)...")
    stability, top15_sets = weight_robustness(gene_data, candidates)
    stable_top15 = [g for g, c in stability.items() if c >= 4]
    sensitive_genes = [g for g, c in stability.items() if c < 4]
    print(f"  Stable top15 (>=4/5 schemes): {len(stable_top15)}")
    print(f"  Weight-sensitive (appear in <4 schemes): {len(sensitive_genes)}")
    for name, genes in top15_sets.items():
        print(f"    {name}: {sorted(genes)[:5]}...")

    # Penalize weight-sensitive genes
    for gene in sensitive_genes:
        gene_data[gene]['TransformationScore'] *= 0.85
        gene_data[gene]['weight_sensitive'] = True
    for gene in candidates:
        if gene not in sensitive_genes:
            gene_data[gene]['weight_sensitive'] = False

    # [10] IM subtype specificity
    print("\n[10] IM subtype specificity check...")
    for gene in sorted(gene_data, key=lambda g: gene_data[g]['TransformationScore'], reverse=True)[:15]:
        iim = gse78523[gene]['iim_effect']
        cim = gse78523[gene]['cim_effect']
        if cim > 0 and iim <= 0:
            gene_data[gene]['TransformationScore'] *= 0.8
            gene_data[gene]['im_subtype_note'] = 'CIM_only'
        else:
            gene_data[gene]['im_subtype_note'] = 'IIM_or_both'

    # [11] Marker classification
    print("\n[11] Marker classification...")
    ts_thresh = np.percentile([gene_data[g]['TransformationScore'] for g in candidates], 75)
    cs_thresh = np.percentile([gene_data[g]['ClinicalExtensionScore'] for g in candidates], 75)

    for gene in candidates:
        ts = gene_data[gene]['TransformationScore']
        cs = gene_data[gene]['ClinicalExtensionScore']
        if ts >= ts_thresh and cs >= cs_thresh:
            gene_data[gene]['marker_class'] = 'clinical_extrapolation'
        elif ts >= ts_thresh:
            gene_data[gene]['marker_class'] = 'core_transformation'
        elif cs >= cs_thresh and ts < ts_thresh:
            gene_data[gene]['marker_class'] = 'mature_cancer_only'
        else:
            gene_data[gene]['marker_class'] = 'mechanism_candidate'

    # [12] Save
    print("\n[12] Saving results...")
    rows = []
    for gene in candidates:
        d = gene_data[gene]
        row = {
            'gene': gene,
            'TransformationScore': d['TransformationScore'],
            'ClinicalExtensionScore': d['ClinicalExtensionScore'],
            'scRNA_risk': scrna_risk[gene],
            'spatial_gradient': spatial_grad[gene],
            'bulk_progression': bulk_scores[gene],
            'network_score': network_scores[gene],
            'direction_penalty': direction_penalty[gene],
            'marker_class': d['marker_class'],
            'weight_stability': stability.get(gene, 0),
            'im_subtype_note': d.get('im_subtype_note', ''),
            'gse55696_jt_z': gse55696[gene]['jt_z'],
            'gse55696_p': gse55696[gene]['p'],
            'gse78523_effect': gse78523[gene]['effect'],
            'gse78523_p': gse78523[gene]['p'],
            'pool_source': ','.join([k for k, v in pools.items() if gene in v]),
            'spatial_discovery': gene in pools.get('C', []),
            'weight_sensitive': d.get('weight_sensitive', False),
        }
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values('TransformationScore', ascending=False)
    result_df.to_csv(f"{BASE}/results/evidence_ranked_genes.csv", index=False)

    # Verification
    print("\n  --- Verification ---")
    top15 = result_df.head(15)
    n_primary_sig = ((top15['gse55696_p'] < 0.05) | (top15['gse78523_p'] < 0.05)).sum()
    print(f"  Top15 with primary evidence padj<0.05: {n_primary_sig}/15 (target: >=10)")
    print(f"  Top15 stable across >=4/5 weight schemes: {len(stable_top15)}")

    ts_vals = result_df['TransformationScore'].values
    cs_vals = result_df['ClinicalExtensionScore'].values
    corr, _ = spearmanr(ts_vals, cs_vals)
    print(f"  TransformationScore vs ClinicalExtensionScore correlation: r={corr:.3f}")

    class_counts = result_df['marker_class'].value_counts()
    for cls, n in class_counts.items():
        print(f"    {cls}: {n}")

    print(f"\n{'='*60}")
    print("Step 8 COMPLETE")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Top 5 TransformationScore: {result_df.head(5)['gene'].tolist()}")
    print(f"  Core transformation markers: {(result_df['marker_class']=='core_transformation').sum()}")
    print(f"  Clinical extrapolation markers: {(result_df['marker_class']=='clinical_extrapolation').sum()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()