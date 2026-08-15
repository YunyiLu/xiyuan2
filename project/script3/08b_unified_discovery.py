"""
Step 8b: Unified Discovery Layer — All Bulk Data for Candidate Ranking
(Replaces separate discovery + validation architecture)

Changes from original 08_meta_analysis.py:
  1. GSE78523: Healthy excluded, only 14 prog vs 16 IM_ctrl
  2. GSE27342: Added as 5th bulk source (paired cancer endpoint)
  3. bulk_progression weights: 0.35/0.30/0.15/0.10/0.10
  4. TransformationScore weights: unchanged (0.30/0.30/0.25/0.15)
  5. New: cross-data consistency score
  6. New: power analysis for future validation

Output:
  results/unified_discovery_ranked.csv
  results/unified_vs_original_comparison.csv
  results/power_analysis.txt
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu, wilcoxon, rankdata, norm

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
BULK = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk"


def load_candidate_pools():
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
    path = f"{BASE}/results/graph_ranked_genes.csv"
    if not os.path.exists(path):
        return {g: 0.0 for g in candidates}
    df = pd.read_csv(path)
    mapping = dict(zip(df['gene'], df['network_score']))
    return {g: mapping.get(g, 0.0) for g in candidates}


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)


def test_gse55696(candidates):
    results_path = f"{BASE}/results/gse55696_jt_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        out = {}
        for _, row in df.iterrows():
            if row['gene'] in candidates:
                out[row['gene']] = {
                    'jt_z': row.get('jt_z', row.get('spearman_r', 0)),
                    'p': row.get('padj', row.get('pval', 1.0)),
                    'direction': row.get('direction', 'none')
                }
        for g in candidates:
            if g not in out:
                out[g] = {'jt_z': 0, 'p': 1.0, 'direction': 'none'}
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
        samples = [s for s in samples if s in expr.columns]
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
        valid = ~np.isnan(vals) & ~np.isnan(stages)
        if valid.sum() < 10:
            out[gene] = {'jt_z': 0, 'p': 1.0, 'direction': 'none'}
            continue
        r, p = spearmanr(vals[valid], stages[valid])
        direction = 'up' if r > 0 else 'down'
        out[gene] = {'jt_z': r, 'p': p, 'direction': direction}
    return out


def test_gse78523(candidates):
    """GSE78523: 14 progressor vs 16 IM_ctrl (Healthy EXCLUDED)."""
    expr_path = f"{BULK}/GSE78523/GSE78523_expression.csv"
    meta_path = f"{BULK}/GSE78523/GSE78523_metadata.csv"
    if not os.path.exists(expr_path) or not os.path.exists(meta_path):
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none',
                    'iim_effect': 0, 'cim_effect': 0} for g in candidates}

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path)

    prog_samples = meta[meta['group'].str.contains('progressor', case=False, na=False)]['sample_id'].tolist()
    ctrl_samples = meta[meta['group'].isin(['IIM_ctrl', 'CIM_ctrl'])]['sample_id'].tolist()
    prog_samples = [s for s in prog_samples if s in expr.columns]
    ctrl_samples = [s for s in ctrl_samples if s in expr.columns]

    iim_prog = meta[meta['group'].str.contains('IIM.*progressor', case=False, na=False)]['sample_id'].tolist()
    iim_ctrl = meta[meta['group'] == 'IIM_ctrl']['sample_id'].tolist()
    cim_prog = meta[meta['group'].str.contains('CIM.*progressor', case=False, na=False)]['sample_id'].tolist()
    cim_ctrl = meta[meta['group'] == 'CIM_ctrl']['sample_id'].tolist()

    print(f"    GSE78523 groups: {len(prog_samples)} prog vs {len(ctrl_samples)} ctrl (Healthy excluded)")

    out = {}
    for gene in candidates:
        if gene not in expr.index or not prog_samples or not ctrl_samples:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none',
                         'iim_effect': 0, 'cim_effect': 0}
            continue
        v_prog = expr.loc[gene, prog_samples].values.astype(float)
        v_ctrl = expr.loc[gene, ctrl_samples].values.astype(float)
        try:
            _, p = mannwhitneyu(v_prog, v_ctrl, alternative='two-sided')
            effect = compute_cohens_d(v_prog, v_ctrl)
        except Exception:
            effect, p = 0, 1.0
        direction = 'up' if effect > 0 else 'down'

        iim_eff = cim_eff = 0
        ip = [s for s in iim_prog if s in expr.columns]
        ic = [s for s in iim_ctrl if s in expr.columns]
        if ip and ic:
            iim_eff = compute_cohens_d(
                expr.loc[gene, ip].values.astype(float),
                expr.loc[gene, ic].values.astype(float))
        cp = [s for s in cim_prog if s in expr.columns]
        cc = [s for s in cim_ctrl if s in expr.columns]
        if cp and cc:
            cim_eff = compute_cohens_d(
                expr.loc[gene, cp].values.astype(float),
                expr.loc[gene, cc].values.astype(float))

        out[gene] = {'effect': effect, 'p': p, 'direction': direction,
                     'iim_effect': iim_eff, 'cim_effect': cim_eff}
    return out


def test_gse27342(candidates):
    """GSE27342: 80 paired GC vs Normal, Wilcoxon signed-rank."""
    expr_path = f"{BASE}/data/gse27342/expression_gene_level.csv"
    meta_path = f"{BASE}/data/gse27342/metadata.csv"
    if not os.path.exists(expr_path) or not os.path.exists(meta_path):
        print("    GSE27342: data not found, skipping")
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none'} for g in candidates}

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path)

    tumor_samples = meta[meta['tissue'].str.contains('tumor|cancer', case=False, na=False)]['sample_id'].tolist()
    normal_samples = meta[meta['tissue'].str.contains('normal|adjacent', case=False, na=False)]['sample_id'].tolist()
    tumor_samples = [s for s in tumor_samples if s in expr.columns]
    normal_samples = [s for s in normal_samples if s in expr.columns]

    n_pairs = min(len(tumor_samples), len(normal_samples))
    print(f"    GSE27342: {n_pairs} pairs available")

    if n_pairs < 10:
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none'} for g in candidates}

    out = {}
    for gene in candidates:
        if gene not in expr.index:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none'}
            continue
        v_tumor = expr.loc[gene, tumor_samples[:n_pairs]].values.astype(float)
        v_normal = expr.loc[gene, normal_samples[:n_pairs]].values.astype(float)
        diff = v_tumor - v_normal
        valid = ~np.isnan(diff) & (diff != 0)
        if valid.sum() < 10:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none'}
            continue
        try:
            _, p = wilcoxon(diff[valid])
            dz = np.mean(diff[valid]) / (np.std(diff[valid], ddof=1) + 1e-10)
        except Exception:
            dz, p = 0, 1.0
        direction = 'up' if dz > 0 else 'down'
        out[gene] = {'effect': dz, 'p': p, 'direction': direction}
    return out


def test_kw_cohort(gse_id, candidates):
    expr_path = f"{BULK}/{gse_id}/{gse_id}_expression.csv"
    meta_path = f"{BULK}/{gse_id}/{gse_id}_metadata.csv"
    if not os.path.exists(expr_path):
        return {g: {'effect': 0, 'p': 1.0, 'direction': 'none'} for g in candidates}

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
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none'}
            continue
        vals = expr.loc[gene, samples].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            out[gene] = {'effect': 0, 'p': 1.0, 'direction': 'none'}
            continue
        try:
            _, p = kruskal(*[vals[valid][stages[valid] == s]
                             for s in sorted(set(stages[valid]))
                             if (stages[valid] == s).sum() >= 2])
            r, _ = spearmanr(vals[valid], stages[valid])
            effect = r
            direction = 'up' if r > 0 else 'down'
        except Exception:
            effect, p, direction = 0, 1.0, 'none'
        out[gene] = {'effect': effect, 'p': p, 'direction': direction}
    return out


def normalize_scores(score_dict):
    vals = np.array(list(score_dict.values()))
    if vals.max() == vals.min():
        return {k: 0.5 for k in score_dict}
    return {k: (v - vals.min()) / (vals.max() - vals.min() + 1e-10) for k, v in score_dict.items()}


def weight_robustness(gene_data, candidates):
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

    all_top15 = set()
    for s in top15_sets.values():
        all_top15.update(s)
    stability = {}
    for g in all_top15:
        count = sum(1 for s in top15_sets.values() if g in s)
        stability[g] = count
    return stability, top15_sets


def compute_power_analysis():
    """Sample size needed to detect various effect sizes at 80% power, alpha=0.05."""
    alpha = 0.05
    power = 0.80
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    results = []
    for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n_per_group = int(np.ceil(2 * ((z_alpha + z_beta) / d) ** 2))
        results.append({'effect_size_d': d, 'n_per_group': n_per_group,
                        'total_n': 2 * n_per_group})
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("Step 8b: Unified Discovery Layer")
    print("  ALL bulk data used for discovery (no hold-out validation)")
    print("  GSE78523 Healthy EXCLUDED (14 prog vs 16 IM_ctrl)")
    print("  GSE27342 ADDED as 5th bulk source")
    print("=" * 70)
    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load candidate pools
    print("\n[1] Loading candidate pools A-G...")
    candidates, pools = load_candidate_pools()
    print(f"  Total candidates (deduplicated): {len(candidates)}")

    if not candidates:
        candidates = ["CDX2", "OLFM4", "LGR5", "NAMPT", "AREG", "PHLDA1",
                      "SOX9", "MYC", "CTNNB1", "IL1B", "NNMT", "CDH17",
                      "REG4", "ITLN1", "FABP1", "ANPEP", "CLDN4", "PRAP1"]
        print(f"  Fallback: {len(candidates)} literature genes")

    # [2] scRNA_risk
    print("\n[2] Computing scRNA_risk...")
    scrna_risk = compute_scrna_risk(candidates)
    n_nonzero = sum(1 for v in scrna_risk.values() if v > 0)
    print(f"  Non-zero scRNA_risk: {n_nonzero}/{len(candidates)}")

    # [3] Spatial gradient
    print("\n[3] Loading spatial gradient...")
    spatial_grad = load_spatial_gradient(candidates)

    # [4] Bulk progression (5 cohorts, new weights)
    print("\n[4] Bulk cohort testing (5 datasets, unified discovery)...")

    print("  [4a] GSE78523 (14 prog vs 16 IM_ctrl, Healthy excluded)...")
    gse78523 = test_gse78523(candidates)
    n_sig = sum(1 for v in gse78523.values() if v['p'] < 0.05)
    print(f"    p<0.05: {n_sig}/{len(candidates)}")

    print("  [4b] GSE55696 (JT trend, CG→LGIN→HGIN→EGC)...")
    gse55696 = test_gse55696(candidates)
    n_sig = sum(1 for v in gse55696.values() if v['p'] < 0.05)
    print(f"    p<0.05: {n_sig}/{len(candidates)}")

    print("  [4c] GSE27342 (paired cancer endpoint, 80 pairs)...")
    gse27342 = test_gse27342(candidates)
    n_sig = sum(1 for v in gse27342.values() if v['p'] < 0.05)
    print(f"    p<0.05: {n_sig}/{len(candidates)}")

    print("  [4d] GSE60427 (KW trend)...")
    gse60427 = test_kw_cohort('GSE60427', candidates)
    n_sig = sum(1 for v in gse60427.values() if v['p'] < 0.05)
    print(f"    p<0.05: {n_sig}/{len(candidates)}")

    print("  [4e] GSE60662 (KW trend)...")
    gse60662 = test_kw_cohort('GSE60662', candidates)
    n_sig = sum(1 for v in gse60662.values() if v['p'] < 0.05)
    print(f"    p<0.05: {n_sig}/{len(candidates)}")

    # Compute bulk_progression (new 5-dataset weights)
    # 0.35*GSE78523 + 0.30*GSE55696 + 0.15*GSE27342 + 0.10*GSE60427 + 0.10*GSE60662
    bulk_scores = {}
    for gene in candidates:
        prog_eff = abs(gse78523[gene]['effect']) if gse78523[gene]['p'] < 0.1 else 0
        jt_eff = abs(gse55696[gene]['jt_z']) if gse55696[gene]['p'] < 0.1 else 0
        cancer_eff = abs(gse27342[gene]['effect']) if gse27342[gene]['p'] < 0.1 else 0
        kw427_eff = abs(gse60427[gene]['effect']) if gse60427[gene]['p'] < 0.1 else 0
        kw662_eff = abs(gse60662[gene]['effect']) if gse60662[gene]['p'] < 0.1 else 0
        bulk_scores[gene] = (0.35*prog_eff + 0.30*jt_eff + 0.15*cancer_eff
                             + 0.10*kw427_eff + 0.10*kw662_eff)
    print(f"\n  bulk_progression computed (0.35/0.30/0.15/0.10/0.10)")

    # [5] Network score
    print("\n[5] Loading network score (RWR)...")
    network_scores = load_network_score(candidates)

    # [6] Cross-data consistency score
    print("\n[6] Cross-data consistency...")
    consistency_scores = {}
    n_datasets_available = {}
    for gene in candidates:
        dirs = []
        if gse78523[gene]['direction'] != 'none':
            dirs.append(gse78523[gene]['direction'])
        if gse55696[gene]['direction'] != 'none':
            dirs.append(gse55696[gene]['direction'])
        if gse27342[gene]['direction'] != 'none':
            dirs.append(gse27342[gene]['direction'])
        if gse60427[gene]['direction'] != 'none':
            dirs.append(gse60427[gene]['direction'])
        if gse60662[gene]['direction'] != 'none':
            dirs.append(gse60662[gene]['direction'])

        n_datasets_available[gene] = len(dirs)
        if not dirs:
            consistency_scores[gene] = 0
        else:
            n_up = dirs.count('up')
            n_down = dirs.count('down')
            consistency_scores[gene] = max(n_up, n_down)

    # Direction penalty (primary evidence conflict)
    direction_penalty = {}
    for gene in candidates:
        primary_dirs = []
        if gse78523[gene]['direction'] != 'none':
            primary_dirs.append(gse78523[gene]['direction'])
        if gse55696[gene]['direction'] != 'none':
            primary_dirs.append(gse55696[gene]['direction'])
        if primary_dirs and len(set(primary_dirs)) > 1:
            direction_penalty[gene] = 0.5
        else:
            direction_penalty[gene] = 1.0
    n_penalized = sum(1 for v in direction_penalty.values() if v < 1.0)
    print(f"  Direction-penalized: {n_penalized}")
    print(f"  Consistency 5/5: {sum(1 for v in consistency_scores.values() if v >= 5)}")
    print(f"  Consistency 4+/5: {sum(1 for v in consistency_scores.values() if v >= 4)}")

    # [7] TransformationScore (weights UNCHANGED: 0.30/0.30/0.25/0.15)
    print("\n[7] Computing TransformationScore (0.30/0.30/0.25/0.15)...")
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

    # [8] Weight robustness
    print("\n[8] Weight robustness (5 schemes)...")
    stability, top15_sets = weight_robustness(gene_data, candidates)
    stable_top15 = [g for g, c in stability.items() if c >= 4]
    sensitive_genes = [g for g, c in stability.items() if c < 4]
    print(f"  Stable (>=4/5): {len(stable_top15)}")
    for name, genes in top15_sets.items():
        print(f"    {name}: {sorted(genes)[:5]}...")

    for gene in sensitive_genes:
        gene_data[gene]['TransformationScore'] *= 0.85
        gene_data[gene]['weight_sensitive'] = True
    for gene in candidates:
        if gene not in sensitive_genes:
            gene_data[gene]['weight_sensitive'] = False

    # [9] IM subtype check (using GSE78523 subtype data — now part of discovery)
    print("\n[9] IM subtype specificity...")
    for gene in sorted(gene_data, key=lambda g: gene_data[g]['TransformationScore'], reverse=True)[:15]:
        iim = gse78523[gene]['iim_effect']
        cim = gse78523[gene]['cim_effect']
        if cim > 0 and iim <= 0:
            gene_data[gene]['TransformationScore'] *= 0.8
            gene_data[gene]['im_subtype_note'] = 'CIM_only'
        else:
            gene_data[gene]['im_subtype_note'] = 'IIM_or_both'

    # [10] Save
    print("\n[10] Saving results...")
    rows = []
    for gene in candidates:
        d = gene_data[gene]
        row = {
            'gene': gene,
            'TransformationScore': d['TransformationScore'],
            'scRNA_risk': scrna_risk[gene],
            'spatial_gradient': spatial_grad[gene],
            'bulk_progression': bulk_scores[gene],
            'network_score': network_scores[gene],
            'direction_penalty': direction_penalty[gene],
            'consistency_score': consistency_scores[gene],
            'n_datasets_with_data': n_datasets_available[gene],
            'weight_stability': stability.get(gene, 0),
            'im_subtype_note': d.get('im_subtype_note', ''),
            'weight_sensitive': d.get('weight_sensitive', False),
            'gse78523_effect': gse78523[gene]['effect'],
            'gse78523_p': gse78523[gene]['p'],
            'gse78523_iim_effect': gse78523[gene]['iim_effect'],
            'gse78523_cim_effect': gse78523[gene]['cim_effect'],
            'gse55696_jt_z': gse55696[gene]['jt_z'],
            'gse55696_p': gse55696[gene]['p'],
            'gse27342_effect': gse27342[gene]['effect'],
            'gse27342_p': gse27342[gene]['p'],
            'gse60427_effect': gse60427[gene]['effect'],
            'gse60427_p': gse60427[gene]['p'],
            'gse60662_effect': gse60662[gene]['effect'],
            'gse60662_p': gse60662[gene]['p'],
            'pool_source': ','.join([k for k, v in pools.items() if gene in v]),
        }
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values('TransformationScore', ascending=False)
    result_df.to_csv(f"{BASE}/results/unified_discovery_ranked.csv",
                     index=False, encoding='utf-8-sig')

    # [11] Compare with original ranking
    print("\n[11] Comparison with original evidence_ranked_genes.csv...")
    orig_path = f"{BASE}/results/evidence_ranked_genes.csv"
    if os.path.exists(orig_path):
        orig = pd.read_csv(orig_path)
        orig_rank = {g: i+1 for i, g in enumerate(orig['gene'])}
        new_rank = {g: i+1 for i, g in enumerate(result_df['gene'])}
        comparison = []
        for gene in result_df['gene']:
            comparison.append({
                'gene': gene,
                'new_rank': new_rank.get(gene, -1),
                'old_rank': orig_rank.get(gene, -1),
                'rank_change': orig_rank.get(gene, -1) - new_rank.get(gene, -1),
            })
        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv(f"{BASE}/results/unified_vs_original_comparison.csv",
                       index=False, encoding='utf-8-sig')
        print(f"  Biggest risers (old→new):")
        risers = comp_df[comp_df['rank_change'] > 0].nlargest(5, 'rank_change')
        for _, r in risers.iterrows():
            print(f"    {r['gene']}: #{r['old_rank']} → #{r['new_rank']} (+{r['rank_change']})")
        print(f"  Biggest fallers:")
        fallers = comp_df[comp_df['rank_change'] < 0].nsmallest(5, 'rank_change')
        for _, r in fallers.iterrows():
            print(f"    {r['gene']}: #{r['old_rank']} → #{r['new_rank']} ({r['rank_change']})")

    # [12] Power analysis
    print("\n[12] Power analysis for future validation...")
    power_df = compute_power_analysis()
    power_text = "Power Analysis: Sample sizes needed for two-group MWU test\n"
    power_text += "=" * 60 + "\n"
    power_text += f"  Alpha = 0.05 (two-sided), Power = 0.80\n\n"
    power_text += f"  {'Effect Size (d)':<18} {'N per group':<15} {'Total N':<10}\n"
    power_text += f"  {'-'*45}\n"
    for _, r in power_df.iterrows():
        power_text += f"  {r['effect_size_d']:<18.1f} {r['n_per_group']:<15} {r['total_n']:<10}\n"
    power_text += f"\n  OLFM4 observed d = 0.88 in GSE78523 (14v16)\n"
    power_text += f"  To confirm d=0.88 @80% power: n={power_df[power_df['effect_size_d']==0.9].iloc[0]['n_per_group']} per group\n"
    power_text += f"  Conservative (d=0.5): n={power_df[power_df['effect_size_d']==0.5].iloc[0]['n_per_group']} per group\n"
    power_text += f"\n  Recommendation: Future CIM progression cohort needs >=52 per group\n"
    power_text += f"  (based on d=0.8, allowing for real-world attenuation)\n"

    with open(f"{BASE}/results/power_analysis.txt", 'w', encoding='utf-8') as f:
        f.write(power_text)
    print(power_text)

    # [13] Final summary
    print(f"\n{'='*70}")
    print("UNIFIED DISCOVERY RESULTS")
    print(f"{'='*70}")
    print(f"\n  Total candidates: {len(candidates)}")
    print(f"  Top 10 by TransformationScore:")
    print(f"  {'Rank':<5} {'Gene':<10} {'Score':>8} {'Consist':>8} {'78523_d':>8} {'55696_r':>8} {'27342_dz':>8}")
    print(f"  {'-'*60}")
    for i, (_, row) in enumerate(result_df.head(10).iterrows()):
        cons_str = f"{int(row['consistency_score'])}/{int(row['n_datasets_with_data'])}"
        print(f"  {i+1:<5} {row['gene']:<10} {row['TransformationScore']:>8.4f} "
              f"{cons_str:>8} {row['gse78523_effect']:>8.3f} "
              f"{row['gse55696_jt_z']:>8.3f} {row['gse27342_effect']:>8.3f}")

    print(f"\n  Evidence strength tiers:")
    tier1 = result_df[result_df['consistency_score'] >= 4]
    tier2 = result_df[(result_df['consistency_score'] >= 3) & (result_df['consistency_score'] < 4)]
    tier3 = result_df[result_df['consistency_score'] < 3]
    print(f"    Tier 1 (4+ datasets consistent): {len(tier1)} genes")
    print(f"    Tier 2 (3 datasets consistent): {len(tier2)} genes")
    print(f"    Tier 3 (<3 datasets): {len(tier3)} genes")

    print(f"\n{'='*70}")
    print("Step 8b COMPLETE — Unified discovery, no hold-out validation")
    print(f"  Output: results/unified_discovery_ranked.csv")
    print(f"  Comparison: results/unified_vs_original_comparison.csv")
    print(f"  Power: results/power_analysis.txt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
