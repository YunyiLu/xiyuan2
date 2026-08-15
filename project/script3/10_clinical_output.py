"""
Step 10: Interpretability (SHAP) + DCA + Clinical Translation Evidence
Input: FINAL_PANEL + GSE78523/GSE55696/TCGA + HPA + DGIdb + TCGA methylation/CNA
Output: SHAP values, DCA, calibration, drug targets, pathway enrichment, multi-omics annotation
"""
import os, sys, warnings, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
TCGA = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"
BULK = "C:/FDU/Y4S2/xiyuan/project/dataset/GEO_bulk"
DB = "C:/FDU/Y4S2/xiyuan/project/dataset/databases"


def load_panel():
    panel = pd.read_csv(f"{BASE}/results/FINAL_PANEL.csv")
    return panel['gene'].tolist()


def shap_analysis(panel_genes):
    """SHAP with label priority: GSE78523 > GSE55696 > TCGA survival."""
    import shap

    # Priority 1: GSE78523 progressor vs non-progressor
    gse78_expr = f"{BULK}/GSE78523/GSE78523_expression.csv"
    gse78_meta = f"{BULK}/GSE78523/GSE78523_metadata.csv"
    if os.path.exists(gse78_expr) and os.path.exists(gse78_meta):
        expr = pd.read_csv(gse78_expr, index_col=0)
        meta = pd.read_csv(gse78_meta)
        avail = [g for g in panel_genes if g in expr.index]
        if len(avail) >= 3:
            prog = meta[meta['group'].str.contains('GC|progressor', case=False, na=False)]['sample_id'].tolist()
            ctrl = meta[meta['group'].str.contains('ctrl|control|healthy', case=False, na=False)]['sample_id'].tolist()
            prog = [s for s in prog if s in expr.columns]
            ctrl = [s for s in ctrl if s in expr.columns]
            if len(prog) >= 5 and len(ctrl) >= 5:
                samples = prog + ctrl
                X = expr.loc[avail, samples].T.values
                y = np.array([1]*len(prog) + [0]*len(ctrl))
                X_scaled = StandardScaler().fit_transform(X)
                # Small sample: Logistic Regression + LOOCV
                model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                model.fit(X_scaled, y)
                loo = LeaveOneOut()
                y_pred = cross_val_predict(model, X_scaled, y, cv=loo, method='predict_proba')[:, 1]
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y, y_pred)
                explainer = shap.LinearExplainer(model, X_scaled)
                shap_vals = explainer.shap_values(X_scaled)
                print(f"  SHAP label: GSE78523 progressor (n={len(prog)+len(ctrl)}, LOOCV AUC={auc:.3f})")
                return shap_vals, avail, 'GSE78523_progressor', auc

    # Priority 2: GSE55696 HGIN+EGC vs CG+LGIN
    gse55_expr = f"{BULK}/GSE55696/GSE55696_expression.csv"
    gse55_meta = f"{BULK}/GSE55696/GSE55696_metadata.csv"
    if os.path.exists(gse55_expr):
        expr = pd.read_csv(gse55_expr, index_col=0)
        avail = [g for g in panel_genes if g in expr.index]
        if len(avail) >= 3:
            n = expr.shape[1]
            # CG(19) + LGIN(19) = 0, HGIN(20) + EGC(19) = 1
            y = np.array([0]*19 + [0]*19 + [1]*20 + [1]*min(19, n-58))[:n]
            samples = expr.columns[:len(y)].tolist()
            X = expr.loc[avail, samples].T.values
            X_scaled = StandardScaler().fit_transform(X)
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model.fit(X_scaled, y)
            from sklearn.metrics import roc_auc_score
            y_pred = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            auc = roc_auc_score(y, y_pred)
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_vals = explainer.shap_values(X_scaled)
            print(f"  SHAP label: GSE55696 HGIN+EGC vs CG+LGIN (n={len(y)}, 5-fold AUC={auc:.3f})")
            return shap_vals, avail, 'GSE55696_progression', auc

    # Priority 3: TCGA survival (median Cox risk split)
    expr_path = f"{TCGA}/TCGA-STAD.HiSeqV2.gz"
    surv_path = f"{TCGA}/TCGA-STAD.survival.tsv"
    if os.path.exists(expr_path) and os.path.exists(surv_path):
        expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
        surv = pd.read_csv(surv_path, sep='\t')
        surv = surv[['sample', 'OS', 'OS.time']].dropna()
        surv = surv[surv['OS.time'] > 0].set_index('sample')
        common = sorted(set(expr.columns) & set(surv.index))
        avail = [g for g in panel_genes if g in expr.index]
        if len(avail) >= 3:
            X = expr.loc[avail, common].T.values
            X_scaled = StandardScaler().fit_transform(X)
            y = ((surv.loc[common, 'OS'] == 1) & (surv.loc[common, 'OS.time'] < 1095)).astype(int).values
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model.fit(X_scaled, y)
            from sklearn.metrics import roc_auc_score
            y_pred = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            auc = roc_auc_score(y, y_pred)
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_vals = explainer.shap_values(X_scaled)
            print(f"  SHAP label: TCGA 3yr mortality (n={len(common)}, 5-fold AUC={auc:.3f})")
            print("  NOTE: This is prognostic SHAP, not transformation prediction SHAP")
            return shap_vals, avail, 'TCGA_survival', auc

    return None, panel_genes, 'none', 0


def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """Net benefit for DCA."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.80, 0.01)
    results = []
    n = len(y_true)
    for t in thresholds:
        tp = ((y_prob >= t) & (y_true == 1)).sum()
        fp = ((y_prob >= t) & (y_true == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        results.append({'threshold': t, 'net_benefit': nb})
    return pd.DataFrame(results)


def run_dca(panel_genes):
    """DCA with priority: GSE78523 > TCGA. Compare panel vs CDX2 vs random."""
    os.makedirs(f"{BASE}/figures", exist_ok=True)

    # Try GSE78523 first
    gse78_expr = f"{BULK}/GSE78523/GSE78523_expression.csv"
    gse78_meta = f"{BULK}/GSE78523/GSE78523_metadata.csv"
    dca_label = None

    if os.path.exists(gse78_expr) and os.path.exists(gse78_meta):
        expr = pd.read_csv(gse78_expr, index_col=0)
        meta = pd.read_csv(gse78_meta)
        avail = [g for g in panel_genes if g in expr.index]
        prog = meta[meta['group'].str.contains('GC|progressor', case=False, na=False)]['sample_id'].tolist()
        ctrl = meta[meta['group'].str.contains('ctrl|control|healthy', case=False, na=False)]['sample_id'].tolist()
        prog = [s for s in prog if s in expr.columns]
        ctrl = [s for s in ctrl if s in expr.columns]
        if len(prog) >= 5 and len(ctrl) >= 5 and len(avail) >= 3:
            samples = prog + ctrl
            X = StandardScaler().fit_transform(expr.loc[avail, samples].T.values)
            y = np.array([1]*len(prog) + [0]*len(ctrl))
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            y_prob = cross_val_predict(model.fit(X, y), X, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
            # CDX2 single gene
            cdx2_prob = None
            if 'CDX2' in expr.index:
                x_cdx2 = StandardScaler().fit_transform(expr.loc[['CDX2'], samples].T.values)
                m2 = LogisticRegression(max_iter=1000, random_state=42).fit(x_cdx2, y)
                cdx2_prob = cross_val_predict(m2, x_cdx2, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
            dca_label = 'GSE78523 (transformation prediction DCA)'
            _plot_dca(y, y_prob, cdx2_prob, dca_label)
            return dca_label

    # Fallback: TCGA survival
    expr_path = f"{TCGA}/TCGA-STAD.HiSeqV2.gz"
    surv_path = f"{TCGA}/TCGA-STAD.survival.tsv"
    if os.path.exists(expr_path) and os.path.exists(surv_path):
        expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
        surv = pd.read_csv(surv_path, sep='\t')
        surv = surv[['sample', 'OS', 'OS.time']].dropna()
        surv = surv[surv['OS.time'] > 0].set_index('sample')
        common = sorted(set(expr.columns) & set(surv.index))
        avail = [g for g in panel_genes if g in expr.index]
        if len(avail) >= 3:
            X = StandardScaler().fit_transform(expr.loc[avail, common].T.values)
            y = ((surv.loc[common, 'OS'] == 1) & (surv.loc[common, 'OS.time'] < 1095)).astype(int).values
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model.fit(X, y)
            y_prob = model.predict_proba(X)[:, 1]
            cdx2_prob = None
            if 'CDX2' in expr.index:
                x_cdx2 = StandardScaler().fit_transform(expr.loc[['CDX2'], common].T.values)
                m2 = LogisticRegression(max_iter=1000, random_state=42).fit(x_cdx2, y)
                cdx2_prob = m2.predict_proba(x_cdx2)[:, 1]
            dca_label = 'TCGA survival (prognostic DCA, NOT transformation prediction)'
            _plot_dca(y, y_prob, cdx2_prob, dca_label)
            return dca_label

    print("  WARNING: No suitable cohort for DCA")
    return None


def _plot_dca(y, y_prob_panel, y_prob_cdx2, title):
    dca_panel = decision_curve_analysis(y, y_prob_panel)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca_panel['threshold'], dca_panel['net_benefit'], label='Panel', linewidth=2)
    if y_prob_cdx2 is not None:
        dca_cdx2 = decision_curve_analysis(y, y_prob_cdx2)
        ax.plot(dca_cdx2['threshold'], dca_cdx2['net_benefit'], label='CDX2 alone', linewidth=1.5, linestyle='-.')
    prevalence = y.mean()
    ax.axhline(prevalence, color='gray', linestyle='--', label='Treat All')
    ax.axhline(0, color='black', linestyle=':', label='Treat None')
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 0.8)
    plt.tight_layout()
    plt.savefig(f"{BASE}/figures/10_dca.png", dpi=150)
    plt.close()
    # Check panel > treat-all
    panel_better = (dca_panel['net_benefit'] > prevalence).any()
    print(f"  Panel net benefit > Treat All: {panel_better}")


def calibration_curve(panel_genes):
    """Calibration: predicted probability vs observed outcome."""
    expr_path = f"{TCGA}/TCGA-STAD.HiSeqV2.gz"
    surv_path = f"{TCGA}/TCGA-STAD.survival.tsv"
    if not os.path.exists(expr_path):
        return
    expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
    surv = pd.read_csv(surv_path, sep='\t')[['sample', 'OS', 'OS.time']].dropna()
    surv = surv[surv['OS.time'] > 0].set_index('sample')
    common = sorted(set(expr.columns) & set(surv.index))
    avail = [g for g in panel_genes if g in expr.index]
    if len(avail) < 3:
        return

    X = StandardScaler().fit_transform(expr.loc[avail, common].T.values)
    y = ((surv.loc[common, 'OS'] == 1) & (surv.loc[common, 'OS.time'] < 1095)).astype(int).values
    model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    y_prob = cross_val_predict(model.fit(X, y), X, y, cv=5, method='predict_proba')[:, 1]

    from sklearn.calibration import calibration_curve as sk_cal
    prob_true, prob_pred = sk_cal(y, y_prob, n_bins=5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, 'o-', label='Panel')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration Curve (TCGA 3yr mortality)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{BASE}/figures/10_calibration.png", dpi=150)
    plt.close()
    print(f"  Calibration curve saved")


def hpa_validation(panel_genes):
    """HPA: compare normal stomach vs gastric cancer protein expression."""
    import zipfile
    results = []
    normal_path = f"{DB}/HPA/normal_tissue.tsv.zip"
    cancer_path = f"{DB}/HPA/pathology.tsv.zip"

    normal_data = {}
    if os.path.exists(normal_path):
        with zipfile.ZipFile(normal_path) as z:
            hpa = pd.read_csv(z.open(z.namelist()[0]), sep='\t')
        stomach = hpa[hpa['Tissue'].str.contains('stomach', case=False, na=False)]
        for gene in panel_genes:
            rows = stomach[stomach['Gene name'] == gene]
            normal_data[gene] = rows['Level'].values[0] if len(rows) > 0 else 'Not found'

    cancer_data = {}
    if os.path.exists(cancer_path):
        with zipfile.ZipFile(cancer_path) as z:
            path_df = pd.read_csv(z.open(z.namelist()[0]), sep='\t')
        gc = path_df[path_df['Cancer'].str.contains('stomach', case=False, na=False)]
        for gene in panel_genes:
            rows = gc[gc['Gene name'] == gene]
            if len(rows) > 0:
                cancer_data[gene] = rows.iloc[0].get('High', 'N/A')
            else:
                cancer_data[gene] = 'Not found'

    for gene in panel_genes:
        results.append({'gene': gene, 'normal_stomach': normal_data.get(gene, 'N/A'),
                        'cancer_stomach': cancer_data.get(gene, 'N/A')})
        print(f"    {gene}: normal={normal_data.get(gene, 'N/A')}, cancer={cancer_data.get(gene, 'N/A')}")
    return pd.DataFrame(results)


def drug_prediction(panel_genes):
    """DGIdb + OpenTargets drug target check."""
    results = []
    dgi_path = f"{DB}/DGIdb/interactions.tsv"
    if os.path.exists(dgi_path):
        dgi = pd.read_csv(dgi_path, sep='\t')
        gene_col = 'gene_name' if 'gene_name' in dgi.columns else dgi.columns[0]
        drug_col = 'drug_name' if 'drug_name' in dgi.columns else dgi.columns[1]
        for gene in panel_genes:
            hits = dgi[dgi[gene_col] == gene]
            if len(hits) > 0:
                drugs = hits[drug_col].unique()[:5].tolist()
                results.append({'gene': gene, 'n_drugs': len(hits), 'top_drugs': '; '.join(drugs)})
            else:
                results.append({'gene': gene, 'n_drugs': 0, 'top_drugs': ''})
    else:
        # Try DGIdb API fallback or local file
        for gene in panel_genes:
            results.append({'gene': gene, 'n_drugs': 0, 'top_drugs': 'DGIdb not available'})

    ot_path = f"{DB}/OpenTargets/targets.csv"
    if os.path.exists(ot_path):
        ot = pd.read_csv(ot_path)
        for r in results:
            gene = r['gene']
            hits = ot[ot['symbol'] == gene] if 'symbol' in ot.columns else pd.DataFrame()
            r['opentargets_score'] = hits['score'].values[0] if len(hits) > 0 else 0

    return pd.DataFrame(results)


def pathway_enrichment(panel_genes):
    """GSEA Hallmarks + KEGG on panel genes."""
    try:
        import gseapy as gp
        enr = gp.enrichr(gene_list=panel_genes, gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                         organism='human', outdir=None, no_plot=True)
        res = enr.results
        sig = res[res['Adjusted P-value'] < 0.05].head(20)
        sig.to_csv(f"{BASE}/results/panel_pathway_enrichment.csv", index=False)
        print(f"  Significant pathways (padj<0.05): {len(sig)}")
        for _, row in sig.head(5).iterrows():
            print(f"    {row['Term']}: p={row['Adjusted P-value']:.2e}")
        return sig
    except ImportError:
        print("  gseapy not available, skipping pathway enrichment")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Pathway enrichment failed: {e}")
        return pd.DataFrame()


def methylation_annotation(panel_genes):
    """TCGA 450K: promoter methylation difference (tumor vs normal)."""
    meth_path = f"{TCGA}/TCGA-STAD.methylation450.tsv.gz"
    annot_path = f"{BASE}/data/promoter_cpg_annotation.csv"
    if not os.path.exists(meth_path):
        print("  Methylation data not available")
        return pd.DataFrame()

    # Load promoter CpG annotation
    if os.path.exists(annot_path):
        annot = pd.read_csv(annot_path)
    else:
        print("  Promoter CpG annotation not available")
        return pd.DataFrame()

    # Filter to panel gene promoter CpGs
    panel_cpgs = annot[annot['gene'].isin(panel_genes)]
    if panel_cpgs.empty:
        return pd.DataFrame()

    cpg_ids = panel_cpgs['cpg_id'].tolist()
    # Read methylation (large file, only needed CpGs)
    meth = pd.read_csv(meth_path, sep='\t', index_col=0, compression='gzip', nrows=0)
    all_samples = meth.columns.tolist()
    # Tumor vs Normal (TCGA barcode: -01 = tumor, -11 = normal)
    tumor_samples = [s for s in all_samples if len(s) >= 15 and s[13:15] == '01']
    normal_samples = [s for s in all_samples if len(s) >= 15 and s[13:15] == '11']

    if not normal_samples:
        print("  No normal samples in methylation data")
        return pd.DataFrame()

    # Read only needed rows
    meth_full = pd.read_csv(meth_path, sep='\t', index_col=0, compression='gzip')
    avail_cpgs = [c for c in cpg_ids if c in meth_full.index]

    results = []
    for gene in panel_genes:
        gene_cpgs = panel_cpgs[panel_cpgs['gene'] == gene]['cpg_id'].tolist()
        gene_cpgs = [c for c in gene_cpgs if c in avail_cpgs]
        if not gene_cpgs:
            results.append({'gene': gene, 'mean_diff': np.nan, 'status': 'no_cpg'})
            continue
        tumor_beta = meth_full.loc[gene_cpgs, tumor_samples].mean(axis=0).mean()
        normal_beta = meth_full.loc[gene_cpgs, normal_samples].mean(axis=0).mean()
        diff = tumor_beta - normal_beta
        status = 'hypermethylated' if diff > 0.1 else ('hypomethylated' if diff < -0.1 else 'unchanged')
        results.append({'gene': gene, 'mean_diff': diff, 'status': status,
                        'tumor_beta': tumor_beta, 'normal_beta': normal_beta})
    return pd.DataFrame(results)


def cna_annotation(panel_genes):
    """TCGA CNA: amplification/deletion frequency per panel gene."""
    cna_path = f"{TCGA}/stad_tcga_cna.json"
    if not os.path.exists(cna_path):
        print("  CNA data not available")
        return pd.DataFrame()

    with open(cna_path, 'r') as f:
        cna_data = json.load(f)

    # Map entrezGeneId to gene symbol
    gene_info_path = f"{DB}/gene_info/Homo_sapiens.gene_info.gz"
    entrez_to_symbol = {}
    if os.path.exists(gene_info_path):
        import gzip
        with gzip.open(gene_info_path, 'rt') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entrez_to_symbol[int(parts[1])] = parts[2]

    # Count amplification/deletion per gene
    gene_alterations = {g: {'amp': 0, 'del': 0, 'total': 0} for g in panel_genes}
    n_samples = len(set(r.get('sampleId', '') for r in cna_data))

    for record in cna_data:
        symbol = entrez_to_symbol.get(record.get('entrezGeneId', 0), '')
        if symbol in gene_alterations:
            alt = record.get('alteration', 0)
            gene_alterations[symbol]['total'] += 1
            if alt >= 1:
                gene_alterations[symbol]['amp'] += 1
            elif alt <= -1:
                gene_alterations[symbol]['del'] += 1

    results = []
    for gene in panel_genes:
        d = gene_alterations[gene]
        amp_freq = d['amp'] / n_samples if n_samples > 0 else 0
        del_freq = d['del'] / n_samples if n_samples > 0 else 0
        status = 'frequent_amp' if amp_freq > 0.2 else ('frequent_del' if del_freq > 0.2 else 'stable')
        results.append({'gene': gene, 'amp_freq': amp_freq, 'del_freq': del_freq, 'status': status})
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Step 10: Interpretability + Clinical Translation Evidence")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)
    os.makedirs(f"{BASE}/figures", exist_ok=True)

    panel_genes = load_panel()
    print(f"  Panel: {len(panel_genes)} genes")

    # [1] SHAP (label priority: GSE78523 > GSE55696 > TCGA)
    print("\n[1] SHAP analysis (real clinical labels)...")
    shap_vals, shap_genes, shap_source, shap_auc = shap_analysis(panel_genes)
    if shap_vals is not None:
        import shap
        shap_df = pd.DataFrame(shap_vals, columns=shap_genes)
        mean_shap = shap_df.abs().mean().sort_values(ascending=False)
        print(f"  SHAP importance ranking:")
        for gene, val in mean_shap.items():
            print(f"    {gene}: {val:.4f}")
        shap_df.to_csv(f"{BASE}/results/shap_values.csv", index=False)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, features=pd.DataFrame(
            StandardScaler().fit_transform(np.zeros((shap_vals.shape[0], len(shap_genes)))),
            columns=shap_genes), show=False)
        plt.tight_layout()
        plt.savefig(f"{BASE}/figures/10_shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Verification: SHAP rank vs TransformationScore rank
        evidence_path = f"{BASE}/results/evidence_ranked_genes.csv"
        if os.path.exists(evidence_path):
            ev = pd.read_csv(evidence_path)
            ts_rank = {g: i for i, g in enumerate(ev['gene'].tolist())}
            shap_rank = {g: i for i, g in enumerate(mean_shap.index)}
            common_g = [g for g in shap_rank if g in ts_rank]
            if len(common_g) >= 3:
                r, p = spearmanr([shap_rank[g] for g in common_g], [ts_rank[g] for g in common_g])
                print(f"  SHAP vs TransformationScore rank: Spearman r={r:.3f}, p={p:.3f}")
    else:
        print("  No suitable data for SHAP")

    # [2] DCA (priority: GSE78523 > TCGA, compare panel vs CDX2)
    print("\n[2] Decision Curve Analysis...")
    dca_label = run_dca(panel_genes)
    if dca_label is None:
        print("  DCA not feasible: no cohort with progression outcome available")

    # [3] Calibration curve
    print("\n[3] Calibration curve...")
    calibration_curve(panel_genes)

    # [4] HPA protein validation (normal vs cancer)
    print("\n[4] HPA protein validation...")
    hpa_df = hpa_validation(panel_genes)
    if not hpa_df.empty:
        hpa_df.to_csv(f"{BASE}/results/hpa_validation.csv", index=False)

    # [5] Drug prediction (DGIdb + OpenTargets)
    print("\n[5] Drug target prediction...")
    drug_df = drug_prediction(panel_genes)
    if not drug_df.empty:
        drug_df.to_csv(f"{BASE}/results/drug_targets.csv", index=False)
        n_druggable = (drug_df['n_drugs'] > 0).sum()
        print(f"  Druggable panel genes: {n_druggable}/{len(panel_genes)}")

    # [6] Pathway enrichment (Hallmarks + KEGG)
    print("\n[6] Pathway enrichment (GSEA)...")
    pathway_enrichment(panel_genes)

    # [7] Multi-omics annotation (post-hoc, not for selection)
    print("\n[7] Multi-omics annotation (post-hoc)...")
    print("  [7a] Methylation (TCGA 450K promoter)...")
    meth_df = methylation_annotation(panel_genes)
    if not meth_df.empty:
        meth_df.to_csv(f"{BASE}/results/panel_methylation.csv", index=False)
        for _, row in meth_df.iterrows():
            print(f"    {row['gene']}: {row.get('status', 'N/A')} (diff={row.get('mean_diff', 0):.3f})")

    print("  [7b] CNA (TCGA amplification/deletion)...")
    cna_df = cna_annotation(panel_genes)
    if not cna_df.empty:
        cna_df.to_csv(f"{BASE}/results/panel_cna.csv", index=False)
        for _, row in cna_df.iterrows():
            print(f"    {row['gene']}: amp={row['amp_freq']:.2%}, del={row['del_freq']:.2%} ({row['status']})")

    print(f"\n{'='*60}")
    print("Step 10 COMPLETE")
    print(f"  SHAP source: {shap_source} (AUC={shap_auc:.3f})")
    print(f"  DCA: {dca_label or 'not available'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()