"""
Step 12b: Multi-gene LASSO Panel
  Uses GSE29272 (134 paired, largest paired dataset) to build a diagnostic model.
  5-fold CV LASSO logistic regression for Cancer vs Normal classification.
"""
import sys, os, warnings, gzip
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
RES_DIR = f"{BASE}/results"
DATA_DIR = f"{BASE}/data/validation"
FIG_DIR = f"{BASE}/results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

candidates_df = pd.read_csv(f"{RES_DIR}/unified_discovery_ranked.csv")
CANDIDATES = set(candidates_df['gene'].tolist())

# Load GPL570 for GSE54129 (111 GC + 21 Normal — larger difference, better for LASSO)
gpl570_dest = f"{DATA_DIR}/GPL570.annot.gz"
probe570 = {}
in_table = False
with gzip.open(gpl570_dest, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table = True
            continue
        if line.startswith('!platform_table_end'):
            break
        if not in_table:
            continue
        if line.startswith('ID\t'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            for g in parts[2].split('///'):
                g = g.strip()
                if g in CANDIDATES:
                    probe570[parts[0]] = g

# Also load GPL96 for GSE29272
gpl96_dest = f"{DATA_DIR}/GPL96.annot.gz"
probe96 = {}
in_table = False
with gzip.open(gpl96_dest, 'rt', errors='replace') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table = True
            continue
        if line.startswith('!platform_table_end'):
            break
        if not in_table:
            continue
        if line.startswith('ID\t'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            for g in parts[2].split('///'):
                g = g.strip()
                if g in CANDIDATES:
                    probe96[parts[0]] = g

print(f"GPL570: {len(set(probe570.values()))} genes, GPL96: {len(set(probe96.values()))} genes")


def parse_dataset(path, probe_map):
    """Parse a GEO series matrix with probe→gene mapping."""
    data_rows = []
    header_samples = []
    sample_ids = []
    sample_titles = {}
    sample_chars = {}

    with gzip.open(path, 'rt', errors='replace') as f:
        reading = False
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                parts = line.strip().split('\t')
                sample_ids = [p.strip('"') for p in parts[1:]]
            elif line.startswith('!Sample_title'):
                parts = line.strip().split('\t')
                for i, p in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        sample_titles[sample_ids[i]] = p.strip('"')
            elif line.startswith('!Sample_characteristics'):
                parts = line.strip().split('\t')
                for i, p in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        sample_chars.setdefault(sample_ids[i], []).append(p.strip('"'))
            elif line.startswith('"ID_REF"'):
                reading = True
                parts = line.strip().split('\t')
                header_samples = [p.strip('"') for p in parts[1:]]
                continue
            elif reading:
                if line.startswith('!') or not line.strip():
                    break
                parts = line.strip().split('\t')
                probe = parts[0].strip('"')
                if probe in probe_map:
                    gene = probe_map[probe]
                    values = []
                    for v in parts[1:]:
                        try:
                            values.append(float(v.strip('"')))
                        except:
                            values.append(np.nan)
                    data_rows.append([gene] + values)

    if not data_rows:
        return None, None, None

    expr_df = pd.DataFrame(data_rows, columns=['gene'] + header_samples)
    expr_df = expr_df.groupby('gene').mean()

    stages = {}
    for sid in sample_ids:
        title = sample_titles.get(sid, '').lower()
        chars = ' '.join(sample_chars.get(sid, [])).lower()
        combined = title + ' ' + chars
        if any(w in combined for w in ['normal', 'adjacent', 'non-cancer', 'noncancer', 'healthy', 'non-tumor']):
            stages[sid] = 0
        elif any(w in combined for w in ['tumor', 'cancer', 'carcinoma', 'malignant']):
            stages[sid] = 1
        else:
            stages[sid] = -1

    return expr_df, stages, sample_ids


# ===== MODEL 1: GSE29272 (134 paired, GPL96) =====
print("="*70)
print("MODEL 1: LASSO Logistic Regression — GSE29272 (134 paired)")
print("="*70)

dest29 = f"{DATA_DIR}/GSE29272_series_matrix.txt.gz"
expr29, stages29, sids29 = parse_dataset(dest29, probe96)

if expr29 is not None:
    valid_samps = [s for s in expr29.columns if stages29.get(s, -1) >= 0]
    X = expr29[valid_samps].T.values
    y = np.array([stages29[s] for s in valid_samps])
    gene_names = expr29.index.tolist()

    print(f"Samples: {len(valid_samps)} (Normal:{(y==0).sum()}, Cancer:{(y==1).sum()})")
    print(f"Features (genes): {len(gene_names)}")

    # Remove NaN
    nan_mask = ~np.any(np.isnan(X), axis=1)
    X = X[nan_mask]
    y = y[nan_mask]
    print(f"After NaN removal: {X.shape[0]} samples")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO logistic regression with 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Find optimal C
    lasso_cv = LogisticRegressionCV(
        Cs=20, cv=cv, penalty='l1', solver='saga',
        max_iter=5000, random_state=42, scoring='roc_auc'
    )
    lasso_cv.fit(X_scaled, y)
    best_C = lasso_cv.C_[0]
    print(f"\nOptimal C: {best_C:.4f}")

    # Get selected features
    coefs = lasso_cv.coef_[0]
    selected = [(gene_names[i], coefs[i]) for i in range(len(gene_names)) if abs(coefs[i]) > 1e-6]
    selected.sort(key=lambda x: -abs(x[1]))

    print(f"Selected features: {len(selected)}/{len(gene_names)}")
    print("\nLASSO Panel (non-zero coefficients):")
    for gene, coef in selected:
        direction = "↑Cancer" if coef > 0 else "↓Cancer"
        print(f"  {gene:12s}: coef={coef:+.4f} ({direction})")

    # Cross-validated predictions for AUC
    y_prob = cross_val_predict(lasso_cv, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"\n5-fold CV AUC: {auc:.4f}")

    # Per-gene AUC (univariate)
    print("\nUnivariate AUC for top genes:")
    gene_aucs = []
    for i, gene in enumerate(gene_names):
        try:
            g_auc = roc_auc_score(y, X_scaled[:, i])
            gene_aucs.append((gene, g_auc))
        except:
            pass
    gene_aucs.sort(key=lambda x: -abs(x[1] - 0.5))
    for gene, g_auc in gene_aucs[:15]:
        direction = "Up in Cancer" if g_auc > 0.5 else "Down in Cancer"
        print(f"  {gene:12s}: AUC={max(g_auc, 1-g_auc):.4f} ({direction})")

    # Save results
    panel_df = pd.DataFrame(selected, columns=['gene', 'lasso_coef'])
    panel_df['abs_coef'] = panel_df['lasso_coef'].abs()
    panel_df = panel_df.sort_values('abs_coef', ascending=False)
    panel_df['direction'] = panel_df['lasso_coef'].apply(lambda x: 'up_in_cancer' if x > 0 else 'down_in_cancer')
    panel_df.to_csv(f"{RES_DIR}/lasso_panel_gse29272.csv", index=False)

    # ROC curve figure
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'LASSO Panel (AUC={auc:.3f})')
    axes[0].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'GSE29272 (n=268) — {len(selected)}-gene LASSO Panel')
    axes[0].legend()

    # Coefficient plot
    if selected:
        genes_plot = [x[0] for x in selected[:20]]
        coefs_plot = [x[1] for x in selected[:20]]
        colors = ['red' if c > 0 else 'blue' for c in coefs_plot]
        axes[1].barh(range(len(genes_plot)), coefs_plot, color=colors, alpha=0.7)
        axes[1].set_yticks(range(len(genes_plot)))
        axes[1].set_yticklabels(genes_plot, fontsize=8)
        axes[1].set_xlabel('LASSO Coefficient')
        axes[1].set_title('Gene Panel Coefficients (Red=Up, Blue=Down)')
        axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/lasso_panel_roc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: figures/lasso_panel_roc.png")


# ===== MODEL 2: GSE54129 (132, GPL570, more genes) =====
print("\n" + "="*70)
print("MODEL 2: LASSO — GSE54129 (111 GC + 21 Normal, 88 genes)")
print("="*70)

dest54 = f"{DATA_DIR}/GSE54129_series_matrix.txt.gz"
expr54, stages54, sids54 = parse_dataset(dest54, probe570)

if expr54 is not None:
    valid_samps = [s for s in expr54.columns if stages54.get(s, -1) >= 0]
    X2 = expr54[valid_samps].T.values
    y2 = np.array([stages54[s] for s in valid_samps])
    gene_names2 = expr54.index.tolist()

    print(f"Samples: {len(valid_samps)} (Normal:{(y2==0).sum()}, Cancer:{(y2==1).sum()})")
    print(f"Features: {len(gene_names2)}")

    nan_mask2 = ~np.any(np.isnan(X2), axis=1)
    X2 = X2[nan_mask2]
    y2 = y2[nan_mask2]

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)

    cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lasso_cv2 = LogisticRegressionCV(
        Cs=20, cv=cv2, penalty='l1', solver='saga',
        max_iter=5000, random_state=42, scoring='roc_auc'
    )
    lasso_cv2.fit(X2_scaled, y2)

    coefs2 = lasso_cv2.coef_[0]
    selected2 = [(gene_names2[i], coefs2[i]) for i in range(len(gene_names2)) if abs(coefs2[i]) > 1e-6]
    selected2.sort(key=lambda x: -abs(x[1]))

    y_prob2 = cross_val_predict(lasso_cv2, X2_scaled, y2, cv=cv2, method='predict_proba')[:, 1]
    auc2 = roc_auc_score(y2, y_prob2)

    print(f"Selected: {len(selected2)} genes")
    print(f"5-fold CV AUC: {auc2:.4f}")
    print("\nPanel genes:")
    for gene, coef in selected2[:20]:
        direction = "↑Cancer" if coef > 0 else "↓Cancer"
        print(f"  {gene:12s}: coef={coef:+.4f} ({direction})")

    panel_df2 = pd.DataFrame(selected2, columns=['gene', 'lasso_coef'])
    panel_df2['direction'] = panel_df2['lasso_coef'].apply(lambda x: 'up_in_cancer' if x > 0 else 'down_in_cancer')
    panel_df2.to_csv(f"{RES_DIR}/lasso_panel_gse54129.csv", index=False)

    # Cross-dataset validation: train on GSE54129, test on GSE29272 (overlapping genes)
    common_genes = list(set(gene_names) & set(gene_names2))
    if len(common_genes) >= 10:
        print(f"\n{'='*70}")
        print(f"CROSS-DATASET VALIDATION (train GSE54129 → test GSE29272)")
        print(f"{'='*70}")
        print(f"Common genes: {len(common_genes)}")

        # Train on GSE54129
        idx_train = [gene_names2.index(g) for g in common_genes]
        X_train = X2_scaled[:, idx_train]
        lasso_cross = LogisticRegression(C=lasso_cv2.C_[0], penalty='l1', solver='saga', max_iter=5000)
        lasso_cross.fit(X_train, y2)

        # Test on GSE29272
        idx_test = [gene_names.index(g) for g in common_genes]
        X_test = X_scaled[:, idx_test]
        y_pred_cross = lasso_cross.predict_proba(X_test)[:, 1]
        auc_cross = roc_auc_score(y, y_pred_cross)
        print(f"Cross-dataset AUC (train:GSE54129 → test:GSE29272): {auc_cross:.4f}")

        # Reverse direction
        lasso_rev = LogisticRegression(C=lasso_cv.C_[0], penalty='l1', solver='saga', max_iter=5000)
        lasso_rev.fit(X_scaled[:, idx_test], y)
        y_pred_rev = lasso_rev.predict_proba(X2_scaled[:, idx_train])[:, 1]
        auc_rev = roc_auc_score(y2, y_pred_rev)
        print(f"Cross-dataset AUC (train:GSE29272 → test:GSE54129): {auc_rev:.4f}")


# ===== Consensus Panel =====
print(f"\n{'='*70}")
print("CONSENSUS PANEL (selected in both datasets)")
print(f"{'='*70}")

if 'selected' in dir() and 'selected2' in dir():
    genes1 = set(g for g, _ in selected)
    genes2 = set(g for g, _ in selected2)
    consensus = genes1 & genes2
    print(f"GSE29272 panel: {len(genes1)} genes")
    print(f"GSE54129 panel: {len(genes2)} genes")
    print(f"Consensus (both): {len(consensus)} genes")

    if consensus:
        print("\nConsensus genes:")
        for gene in sorted(consensus):
            c1 = next(c for g, c in selected if g == gene)
            c2 = next(c for g, c in selected2 if g == gene)
            consistent = "CONSISTENT" if (c1 > 0) == (c2 > 0) else "INCONSISTENT"
            print(f"  {gene:12s}: GSE29272={c1:+.3f}, GSE54129={c2:+.3f} [{consistent}]")

print("\nDone!")
