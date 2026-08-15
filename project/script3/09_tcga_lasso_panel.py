"""
Step 9: TCGA/ACRG Cancer Endpoint Validation + LASSO-Cox (Prognostic Extrapolation)
Input: TCGA-STAD + GSE62254/ACRG + Step 8 TransformationScore top candidates
Output: script3/results/FINAL_PANEL.csv, survival_metrics.csv
Note: This is clinical extrapolation, NOT early cancer prediction.
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
TCGA = "C:/FDU/Y4S2/xiyuan/project/dataset/TCGA_STAD"
ACRG = "C:/FDU/Y4S2/xiyuan/project/dataset/GSE62254"


def load_candidates():
    """Load Step 8 TransformationScore top 20-30 candidates."""
    path = f"{BASE}/results/evidence_ranked_genes.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'TransformationScore' in df.columns:
            df = df.sort_values('TransformationScore', ascending=False)
        return df.head(30)['gene'].tolist()
    fallback = f"{BASE}/results/meta_validated_genes.csv"
    if os.path.exists(fallback):
        return pd.read_csv(fallback).head(30)['gene'].tolist()
    return ["CDX2", "OLFM4", "LGR5", "NAMPT", "AREG", "PHLDA1",
            "SOX9", "MYC", "CTNNB1", "IL1B", "NNMT", "CDH17"]


def load_tcga():
    """Load TCGA-STAD expression + survival."""
    expr = pd.read_csv(f"{TCGA}/TCGA-STAD.HiSeqV2.gz", sep='\t', index_col=0, compression='gzip')
    surv = pd.read_csv(f"{TCGA}/TCGA-STAD.survival.tsv", sep='\t')
    surv = surv[['sample', 'OS', 'OS.time']].dropna()
    surv = surv[surv['OS.time'] > 0].set_index('sample')
    return expr, surv


def load_acrg():
    """Load GSE62254/ACRG expression + survival."""
    expr_path = f"{ACRG}/GSE62254_expression.csv"
    surv_path = f"{ACRG}/GSE62254_survival.csv"
    if not os.path.exists(expr_path) or not os.path.exists(surv_path):
        alt_expr = f"{ACRG}/GSE62254_series_matrix.txt.gz"
        if os.path.exists(alt_expr):
            expr = pd.read_csv(alt_expr, sep='\t', comment='!', index_col=0)
        else:
            return None, None
        surv = pd.read_csv(surv_path) if os.path.exists(surv_path) else None
        return expr, surv
    expr = pd.read_csv(expr_path, index_col=0)
    surv = pd.read_csv(surv_path, index_col=0)
    return expr, surv


def single_gene_cox(X_scaled, genes, duration, event):
    """Single-gene Cox for each candidate: HR, logrank p, C-index."""
    results = []
    median_split = np.median(X_scaled, axis=0)
    for i, gene in enumerate(genes):
        df = pd.DataFrame({'expr': X_scaled[:, i], 'duration': duration, 'event': event})
        try:
            cph = CoxPHFitter()
            cph.fit(df, 'duration', 'event')
            hr = np.exp(cph.params_['expr'])
            p = cph.summary['p']['expr']
            ci = cph.concordance_index_
            high = X_scaled[:, i] > median_split[i]
            lr = logrank_test(duration[high], duration[~high], event[high], event[~high])
            results.append({'gene': gene, 'HR': hr, 'cox_p': p,
                            'c_index': ci, 'logrank_p': lr.p_value})
        except Exception:
            results.append({'gene': gene, 'HR': np.nan, 'cox_p': 1.0,
                            'c_index': 0.5, 'logrank_p': 1.0})
    return pd.DataFrame(results)


def stepwise_cox_backward(X_scaled, genes, duration, event):
    """Backward elimination Cox (AIC) for <=15 candidates."""
    current_genes = list(range(len(genes)))
    while len(current_genes) > 1:
        df = pd.DataFrame(X_scaled[:, current_genes], columns=[genes[i] for i in current_genes])
        df['duration'] = duration
        df['event'] = event
        cph = CoxPHFitter()
        cph.fit(df, 'duration', 'event')
        base_aic = -2 * cph.log_likelihood_ + 2 * len(current_genes)
        worst_idx, best_aic = None, base_aic
        for j in range(len(current_genes)):
            subset = [current_genes[k] for k in range(len(current_genes)) if k != j]
            df_sub = pd.DataFrame(X_scaled[:, subset], columns=[genes[i] for i in subset])
            df_sub['duration'] = duration
            df_sub['event'] = event
            try:
                cph_sub = CoxPHFitter()
                cph_sub.fit(df_sub, 'duration', 'event')
                aic = -2 * cph_sub.log_likelihood_ + 2 * len(subset)
                if aic < best_aic:
                    best_aic = aic
                    worst_idx = j
            except Exception:
                continue
        if worst_idx is None:
            break
        current_genes.pop(worst_idx)
    return [genes[i] for i in current_genes]


def lasso_cox_repeated_cv(X_scaled, genes, duration, event, n_repeats=100):
    """LASSO-Cox with repeated 10-fold CV (100 repeats) for lambda selection."""
    alphas = np.logspace(-3, 1, 30)
    alpha_scores = {a: [] for a in alphas}

    rkf = RepeatedKFold(n_splits=10, n_repeats=n_repeats, random_state=42)
    for train_idx, test_idx in rkf.split(X_scaled):
        for alpha in alphas:
            df_train = pd.DataFrame(X_scaled[train_idx], columns=genes)
            df_train['duration'] = duration[train_idx]
            df_train['event'] = event[train_idx]
            df_test = pd.DataFrame(X_scaled[test_idx], columns=genes)
            df_test['duration'] = duration[test_idx]
            df_test['event'] = event[test_idx]
            try:
                cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.0)
                cph.fit(df_train, 'duration', 'event')
                c = cph.score(df_test, scoring_method='concordance_index')
                alpha_scores[alpha].append(c)
            except Exception:
                pass

    best_alpha = max(alphas, key=lambda a: np.mean(alpha_scores[a]) if alpha_scores[a] else 0)
    best_c = np.mean(alpha_scores[best_alpha])
    return best_alpha, best_c


def elastic_net_cox_cv(X_scaled, genes, duration, event, n_repeats=100):
    """Elastic Net Cox (alpha=0.5) for >30 candidates."""
    alphas = np.logspace(-3, 1, 30)
    alpha_scores = {a: [] for a in alphas}

    rkf = RepeatedKFold(n_splits=10, n_repeats=n_repeats, random_state=42)
    for train_idx, test_idx in rkf.split(X_scaled):
        for alpha in alphas:
            df_train = pd.DataFrame(X_scaled[train_idx], columns=genes)
            df_train['duration'] = duration[train_idx]
            df_train['event'] = event[train_idx]
            df_test = pd.DataFrame(X_scaled[test_idx], columns=genes)
            df_test['duration'] = duration[test_idx]
            df_test['event'] = event[test_idx]
            try:
                cph = CoxPHFitter(penalizer=alpha, l1_ratio=0.5)
                cph.fit(df_train, 'duration', 'event')
                c = cph.score(df_test, scoring_method='concordance_index')
                alpha_scores[alpha].append(c)
            except Exception:
                pass

    best_alpha = max(alphas, key=lambda a: np.mean(alpha_scores[a]) if alpha_scores[a] else 0)
    best_c = np.mean(alpha_scores[best_alpha])
    return best_alpha, best_c


def bootstrap_stability(X_scaled, genes, duration, event, best_alpha, l1_ratio=1.0, n_boot=1000):
    """Bootstrap 1000x: report gene selection frequency."""
    n = len(duration)
    gene_counts = {g: 0 for g in genes}
    for b in range(n_boot):
        idx = resample(np.arange(n), random_state=b)
        df_boot = pd.DataFrame(X_scaled[idx], columns=genes)
        df_boot['duration'] = duration[idx]
        df_boot['event'] = event[idx]
        try:
            cph = CoxPHFitter(penalizer=best_alpha, l1_ratio=l1_ratio)
            cph.fit(df_boot, 'duration', 'event')
            for g in genes:
                if abs(cph.params_.get(g, 0)) > 0.01:
                    gene_counts[g] += 1
        except Exception:
            continue
    freq = {g: gene_counts[g] / n_boot for g in genes}
    return freq


def time_dependent_auc(X_panel, coefs, duration, event, times=[365, 1095, 1825]):
    """Time-dependent AUC at 1, 3, 5 years."""
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
        risk = X_panel @ coefs
        surv_arr = Surv.from_arrays(event.astype(bool), duration)
        aucs, mean_auc = cumulative_dynamic_auc(surv_arr, surv_arr, risk, times)
        return dict(zip([f"AUC_{t//365}yr" for t in times], aucs))
    except ImportError:
        from lifelines.utils import concordance_index
        risk = X_panel @ coefs
        result = {}
        for t in times:
            mask = (duration <= t) | (event == 0)
            if mask.sum() > 10:
                ci = concordance_index(duration[mask], -risk[mask], event[mask])
                result[f"AUC_{t//365}yr"] = ci
        return result


def calibration_analysis(risk_score, duration, event, n_groups=5):
    """Calibration: predicted vs observed survival by risk quintile."""
    from lifelines import KaplanMeierFitter
    quintiles = pd.qcut(risk_score, n_groups, labels=False, duplicates='drop')
    cal_results = []
    for q in sorted(set(quintiles)):
        mask = quintiles == q
        kmf = KaplanMeierFitter()
        kmf.fit(duration[mask], event[mask])
        obs_3yr = kmf.predict(1095) if 1095 <= duration[mask].max() else np.nan
        cal_results.append({'quintile': q, 'mean_risk': risk_score[mask].mean(),
                            'observed_3yr_surv': obs_3yr, 'n': mask.sum()})
    return pd.DataFrame(cal_results)


def validate_acrg(panel_genes, coefs_dict, tcga_expr, acrg_expr, acrg_surv):
    """GSE62254/ACRG independent external validation."""
    if acrg_expr is None or acrg_surv is None:
        print("  ACRG data not available, skipping external validation")
        return None

    common_genes = [g for g in panel_genes if g in acrg_expr.index and g in tcga_expr.index]
    if len(common_genes) < 3:
        print(f"  Only {len(common_genes)} genes overlap with ACRG platform, skipping")
        return None

    print(f"  ACRG gene overlap: {len(common_genes)}/{len(panel_genes)}")

    # Each cohort independent z-score
    acrg_samples = sorted(set(acrg_expr.columns) & set(acrg_surv.index))
    if len(acrg_samples) < 50:
        print(f"  ACRG samples too few ({len(acrg_samples)}), skipping")
        return None

    X_acrg = acrg_expr.loc[common_genes, acrg_samples].T.values
    scaler_acrg = StandardScaler()
    X_acrg_z = scaler_acrg.fit_transform(X_acrg)

    # Direction consistency check
    tcga_samples_check = tcga_expr.columns[:100]
    X_tcga_check = tcga_expr.loc[common_genes, tcga_samples_check].T.values
    tcga_means = X_tcga_check.mean(axis=0)
    acrg_means = X_acrg.mean(axis=0)
    direction_consistent = np.sign(tcga_means - np.median(tcga_means)) == np.sign(acrg_means - np.median(acrg_means))
    n_consistent = direction_consistent.sum()
    print(f"  Direction consistency: {n_consistent}/{len(common_genes)} genes")

    # Risk score using TCGA coefficients
    coefs_arr = np.array([coefs_dict.get(g, 0) for g in common_genes])
    risk_acrg = X_acrg_z @ coefs_arr

    # Survival metrics
    dur_acrg = acrg_surv.loc[acrg_samples, 'OS.time'].values if 'OS.time' in acrg_surv.columns else acrg_surv.loc[acrg_samples].iloc[:, 0].values
    evt_acrg = acrg_surv.loc[acrg_samples, 'OS'].values.astype(int) if 'OS' in acrg_surv.columns else acrg_surv.loc[acrg_samples].iloc[:, 1].values.astype(int)

    from lifelines.utils import concordance_index
    ci_acrg = concordance_index(dur_acrg, -risk_acrg, evt_acrg)

    # KM
    median_risk = np.median(risk_acrg)
    high = risk_acrg > median_risk
    lr = logrank_test(dur_acrg[high], dur_acrg[~high], evt_acrg[high], evt_acrg[~high])

    # Time-dependent AUC
    td_auc = time_dependent_auc(X_acrg_z, coefs_arr, dur_acrg, evt_acrg)

    result = {
        'c_index': ci_acrg,
        'logrank_p': lr.p_value,
        'n_samples': len(acrg_samples),
        'n_genes_used': len(common_genes),
        'direction_consistent': n_consistent,
        **td_auc
    }

    # Diagnosis if validation fails
    if ci_acrg < 0.55:
        print(f"  WARNING: ACRG C-index={ci_acrg:.3f} < 0.55, validation weak")
        print(f"    Direction inconsistent genes: {[common_genes[i] for i in range(len(common_genes)) if not direction_consistent[i]]}")
        # Fallback: rank-based transform (percentile rank, platform-agnostic)
        print("  Trying rank-based transform (percentile rank)...")
        from scipy.stats import rankdata
        X_acrg_rank = np.apply_along_axis(lambda x: rankdata(x) / len(x), 0, X_acrg)
        risk_rank = X_acrg_rank @ coefs_arr
        ci_rank = concordance_index(dur_acrg, -risk_rank, evt_acrg)
        print(f"    Rank-based C-index: {ci_rank:.4f}")
        result['rank_c_index'] = ci_rank
        result['validation_status'] = 'weak_zscore' if ci_rank >= 0.55 else 'weak_both'
    else:
        result['validation_status'] = 'passed'

    return result


def main():
    print("=" * 60)
    print("Step 9: TCGA/ACRG Cancer Endpoint Validation + LASSO-Cox")
    print("  (Clinical extrapolation, NOT early cancer prediction)")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load candidates (Step 8 TransformationScore top 20-30)
    print("\n[1] Loading candidates from Step 8...")
    candidates = load_candidates()
    n_cand = len(candidates)
    print(f"  Candidates: {n_cand} genes")

    # [2] Load TCGA
    print("\n[2] Loading TCGA-STAD...")
    expr, surv = load_tcga()
    common_samples = sorted(set(expr.columns) & set(surv.index))
    available_genes = [g for g in candidates if g in expr.index]
    print(f"  Samples: {len(common_samples)}, Available genes: {len(available_genes)}/{n_cand}")

    if len(available_genes) < 3 or len(common_samples) < 50:
        print("  ERROR: Insufficient data for survival analysis")
        return

    X = expr.loc[available_genes, common_samples].T.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    duration = surv.loc[common_samples, 'OS.time'].values
    event = surv.loc[common_samples, 'OS'].values.astype(int)
    n_cand_avail = len(available_genes)

    # [3] Single-gene Cox
    print(f"\n[3] Single-gene Cox (n={n_cand_avail})...")
    uni_df = single_gene_cox(X_scaled, available_genes, duration, event)
    uni_df.to_csv(f"{BASE}/results/single_gene_cox.csv", index=False)
    n_sig = (uni_df['cox_p'] < 0.05).sum()
    print(f"  Significant (p<0.05): {n_sig}/{n_cand_avail}")

    # [4] Model selection based on candidate count
    print(f"\n[4] Model selection (n_candidates={n_cand_avail})...")
    if n_cand_avail <= 15:
        print("  Strategy: Stepwise Cox (backward, AIC)")
        panel_genes = stepwise_cox_backward(X_scaled, available_genes, duration, event)
        best_alpha = None
        l1_ratio = None
        method = "stepwise_cox"
    elif n_cand_avail <= 30:
        print("  Strategy: LASSO-Cox (repeated 10-fold CV, 100 repeats)")
        best_alpha, best_c = lasso_cox_repeated_cv(X_scaled, available_genes, duration, event)
        print(f"  Best lambda: {best_alpha:.4f}, mean CV C-index: {best_c:.4f}")
        l1_ratio = 1.0
        method = "lasso_cox"
    else:
        print("  Strategy: Elastic Net Cox (alpha=0.5, repeated CV)")
        best_alpha, best_c = elastic_net_cox_cv(X_scaled, available_genes, duration, event)
        print(f"  Best lambda: {best_alpha:.4f}, mean CV C-index: {best_c:.4f}")
        l1_ratio = 0.5
        method = "elastic_net_cox"

    # Fit final penalized model (for LASSO/EN)
    if method != "stepwise_cox":
        df_full = pd.DataFrame(X_scaled, columns=available_genes)
        df_full['duration'] = duration
        df_full['event'] = event
        cph_final = CoxPHFitter(penalizer=best_alpha, l1_ratio=l1_ratio)
        cph_final.fit(df_full, 'duration', 'event')
        coefs = cph_final.params_
        panel_genes = [g for g in available_genes if abs(coefs.get(g, 0)) > 0.01]
    else:
        # Fit full model on selected genes
        panel_idx = [available_genes.index(g) for g in panel_genes]
        df_panel = pd.DataFrame(X_scaled[:, panel_idx], columns=panel_genes)
        df_panel['duration'] = duration
        df_panel['event'] = event
        cph_final = CoxPHFitter()
        cph_final.fit(df_panel, 'duration', 'event')
        coefs = cph_final.params_

    print(f"  Panel genes: {len(panel_genes)}")

    # [5] Bootstrap stability (1000x)
    print("\n[5] Bootstrap stability (1000 iterations)...")
    boot_freq = bootstrap_stability(X_scaled, available_genes, duration, event,
                                    best_alpha if best_alpha else 0.1,
                                    l1_ratio if l1_ratio else 1.0)
    stable_genes = [g for g, f in boot_freq.items() if f > 0.8]
    marginal_genes = [g for g, f in boot_freq.items() if 0.5 <= f <= 0.8]
    print(f"  Stable (>80%): {len(stable_genes)}")
    print(f"  Marginal (50-80%): {len(marginal_genes)}")
    if len(stable_genes) >= 5:
        print(f"  PASS: {len(stable_genes)} stable genes >= 5")
    else:
        print(f"  WARNING: only {len(stable_genes)} stable genes < 5 threshold")
    for g in sorted(boot_freq, key=boot_freq.get, reverse=True)[:10]:
        print(f"    {g}: {boot_freq[g]*100:.1f}%")

    # [6] Risk score
    print("\n[6] Computing risk score...")
    if not panel_genes:
        print("  No panel genes selected, using top stable genes")
        panel_genes = stable_genes[:8] if stable_genes else available_genes[:5]
        panel_idx = [available_genes.index(g) for g in panel_genes if g in available_genes]
        df_panel = pd.DataFrame(X_scaled[:, panel_idx], columns=panel_genes)
        df_panel['duration'] = duration
        df_panel['event'] = event
        cph_final = CoxPHFitter()
        cph_final.fit(df_panel, 'duration', 'event')
        coefs = cph_final.params_

    coefs_dict = {g: float(coefs.get(g, 0)) for g in panel_genes}
    panel_idx = [available_genes.index(g) for g in panel_genes]
    X_panel = X_scaled[:, panel_idx]
    coefs_arr = np.array([coefs_dict[g] for g in panel_genes])
    risk_score = X_panel @ coefs_arr

    # [7] TCGA internal validation
    print("\n[7] TCGA internal validation...")
    # C-index + bootstrap CI
    from lifelines.utils import concordance_index
    ci_full = concordance_index(duration, -risk_score, event)
    ci_boots = []
    for b in range(200):
        idx = resample(np.arange(len(duration)), random_state=b)
        ci_b = concordance_index(duration[idx], -risk_score[idx], event[idx])
        ci_boots.append(ci_b)
    ci_lower, ci_upper = np.percentile(ci_boots, [2.5, 97.5])
    print(f"  C-index: {ci_full:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")

    # KM (median split)
    median_risk = np.median(risk_score)
    high_risk = risk_score > median_risk
    lr = logrank_test(duration[high_risk], duration[~high_risk],
                      event[high_risk], event[~high_risk])
    print(f"  KM logrank p: {lr.p_value:.2e}")

    # Time-dependent AUC
    td_auc = time_dependent_auc(X_panel, coefs_arr, duration, event)
    for k, v in td_auc.items():
        print(f"  {k}: {v:.4f}")

    # Calibration
    cal_df = calibration_analysis(risk_score, duration, event)
    cal_df.to_csv(f"{BASE}/results/calibration_tcga.csv", index=False)
    print(f"  Calibration saved (5 quintiles)")

    # Threshold check
    if ci_full > 0.65:
        print(f"  PASS: C-index {ci_full:.3f} > 0.65")
    else:
        print(f"  NOTE: C-index {ci_full:.3f} < 0.65 threshold")

    # [8] GSE62254/ACRG external validation
    print("\n[8] GSE62254/ACRG external validation...")
    acrg_expr, acrg_surv = load_acrg()
    acrg_result = validate_acrg(panel_genes, coefs_dict, expr, acrg_expr, acrg_surv)
    if acrg_result:
        print(f"  ACRG C-index: {acrg_result['c_index']:.4f}")
        print(f"  ACRG logrank p: {acrg_result['logrank_p']:.2e}")
        if acrg_result['c_index'] > 0.58:
            print(f"  PASS: ACRG C-index > 0.58")
        else:
            print(f"  NOTE: ACRG C-index < 0.58")

    # [9] Save results
    print("\n[9] Saving results...")
    panel_df = pd.DataFrame({
        'gene': panel_genes,
        'coef': [coefs_dict[g] for g in panel_genes],
        'HR': [np.exp(coefs_dict[g]) for g in panel_genes],
        'bootstrap_freq': [boot_freq.get(g, 0) for g in panel_genes],
        'stability': ['stable' if boot_freq.get(g, 0) > 0.8 else
                      'marginal' if boot_freq.get(g, 0) > 0.5 else 'unstable'
                      for g in panel_genes],
    })
    panel_df = panel_df.sort_values('bootstrap_freq', ascending=False)
    panel_df.to_csv(f"{BASE}/results/FINAL_PANEL.csv", index=False)

    metrics = {
        'method': method,
        'n_candidates': n_cand_avail,
        'n_panel_genes': len(panel_genes),
        'n_stable_genes': len(stable_genes),
        'best_alpha': best_alpha,
        'tcga_c_index': ci_full,
        'tcga_c_index_lower': ci_lower,
        'tcga_c_index_upper': ci_upper,
        'tcga_logrank_p': lr.p_value,
        **td_auc,
    }
    if acrg_result:
        metrics['acrg_c_index'] = acrg_result['c_index']
        metrics['acrg_logrank_p'] = acrg_result['logrank_p']
        metrics['acrg_validation_status'] = acrg_result['validation_status']
    pd.DataFrame([metrics]).to_csv(f"{BASE}/results/survival_metrics.csv", index=False)

    print(f"\n{'='*60}")
    print("Step 9 COMPLETE (Clinical Extrapolation)")
    print(f"  Method: {method}")
    print(f"  Panel: {len(panel_genes)} genes ({len(stable_genes)} stable)")
    print(f"  TCGA C-index: {ci_full:.4f} [{ci_lower:.4f}-{ci_upper:.4f}]")
    if acrg_result:
        print(f"  ACRG C-index: {acrg_result['c_index']:.4f} ({acrg_result['validation_status']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

