"""
Step 7: Graph diffusion (RWR, main) + GAT (supplementary) + spatial validation.
Input: STRING PPI + Dorothea + seed genes (MOFA + WGCNA hub + CellChat LR)
Output: script3/results/graph_ranked_genes.csv
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import gzip
from scipy.stats import spearmanr

BASE = "C:/FDU/Y4S2/xiyuan/project/script3"
DB = "C:/FDU/Y4S2/xiyuan/project/dataset/databases"
SPATIAL_DIR = "C:/FDU/Y4S2/xiyuan/project/dataset/OMIX010346/Stomach_cancer/Spatial_Omics"


def load_string_ppi(candidates):
    """Load STRING PPI edges (score>700) for candidate genes."""
    info_path = f"{DB}/STRING/9606.protein.info.v12.0.txt.gz"
    ensp_to_gene = {}
    with gzip.open(info_path, 'rt') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ensp_to_gene[parts[0]] = parts[1]

    cand_set = set(candidates)
    edges = []
    links_path = f"{DB}/STRING/9606.protein.links.v12.0.txt.gz"
    try:
        with gzip.open(links_path, 'rt') as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3 and int(parts[2]) >= 700:
                    g1 = ensp_to_gene.get(parts[0])
                    g2 = ensp_to_gene.get(parts[1])
                    if g1 and g2 and (g1 in cand_set or g2 in cand_set):
                        edges.append((g1, g2, int(parts[2]) / 1000.0))
    except EOFError:
        pass
    return edges


def load_dorothea_edges(candidates):
    """Load Dorothea TF-target edges."""
    doro_path = f"{DB}/Dorothea/dorothea_ABC_omnipath.tsv"
    if not os.path.exists(doro_path):
        return []
    doro = pd.read_csv(doro_path, sep='\t')
    cand_set = set(candidates)
    edges = []
    for _, row in doro.iterrows():
        src = str(row.get('source_genesymbol', ''))
        tgt = str(row.get('target_genesymbol', ''))
        if src in cand_set and tgt in cand_set:
            edges.append((src, tgt, 0.7))
    return edges


def random_walk_restart(G, seeds, alpha=0.3, tol=1e-6, max_iter=200):
    """RWR from seed nodes with convergence threshold."""
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    D = A.sum(axis=1)
    D[D == 0] = 1
    T = A / D[:, None]

    r = np.zeros(n)
    seed_idx = [node_idx[s] for s in seeds if s in node_idx]
    if not seed_idx:
        return pd.Series(np.zeros(n), index=nodes)
    r[seed_idx] = 1.0 / len(seed_idx)

    p = r.copy()
    for _ in range(max_iter):
        p_new = (1 - alpha) * T.T @ p + alpha * r
        if np.abs(p_new - p).sum() < tol:
            break
        p = p_new
    return pd.Series(p, index=nodes)


def rwr_sensitivity(G, seeds):
    """Run RWR at multiple restart probs, select stable top30."""
    alphas = [0.1, 0.3, 0.5, 0.7]
    all_rankings = {}
    for alpha in alphas:
        scores = random_walk_restart(G, seeds, alpha=alpha)
        ranked = scores.sort_values(ascending=False)
        all_rankings[alpha] = ranked

    top30_sets = {a: set(r.head(30).index) for a, r in all_rankings.items()}
    gene_counts = {}
    for s in top30_sets.values():
        for g in s:
            gene_counts[g] = gene_counts.get(g, 0) + 1
    stable_genes = [g for g, c in gene_counts.items() if c >= 3]

    print(f"  RWR sensitivity: alphas={alphas}")
    for a in alphas:
        print(f"    alpha={a}: top={all_rankings[a].index[0]}, max={all_rankings[a].iloc[0]:.6f}")
    print(f"  Stable top30 (>=3/4 alphas): {len(stable_genes)}")
    return all_rankings[0.3], stable_genes, all_rankings


def shortest_path_traceability(G, seeds, top_genes, n_top=20):
    """Shortest path from top genes back to nearest seed for interpretability."""
    paths = []
    seed_set = set(seeds)
    for gene in top_genes[:n_top]:
        if gene in seed_set:
            paths.append({'gene': gene, 'nearest_seed': gene, 'distance': 0, 'path': gene})
            continue
        min_dist = float('inf')
        best_seed, best_path = None, None
        for seed in seeds:
            if seed not in G or gene not in G:
                continue
            try:
                sp = nx.shortest_path(G, seed, gene)
                if len(sp) - 1 < min_dist:
                    min_dist = len(sp) - 1
                    best_seed = seed
                    best_path = '→'.join(sp)
            except nx.NetworkXNoPath:
                continue
        if best_seed:
            paths.append({'gene': gene, 'nearest_seed': best_seed,
                          'distance': min_dist, 'path': best_path})
    return pd.DataFrame(paths)


def run_gat(G, seeds, candidates):
    """GAT: 2-layer, 64 hidden, 4 heads. Regression on GSE55696 JT z-score."""
    print("  Running GAT (supplementary)...")
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GATConv
    except ImportError:
        print("  torch_geometric not available, skipping GAT")
        return None

    jt_path = f"{BASE}/results/gse55696_jt_results.csv"
    if not os.path.exists(jt_path):
        print("  GSE55696 JT results not available, skipping GAT")
        return None
    jt_df = pd.read_csv(jt_path)
    if 'gene' not in jt_df.columns or 'jt_z' not in jt_df.columns:
        print("  JT results missing columns, skipping GAT")
        return None

    # Node features: expression mean + variance only (no bulk-derived info)
    pb_path = f"{BASE}/data/pseudobulk_by_sample_celltype.csv"
    if not os.path.exists(pb_path):
        print("  Pseudobulk not available, skipping GAT")
        return None
    pb = pd.read_csv(pb_path)
    gene_cols = [c for c in pb.columns if c not in
                 ['sample_id','celltype','dataset','stage','hp_status','n_cells']]
    expr_mean = pb[gene_cols].mean().to_dict()
    expr_var = pb[gene_cols].var().to_dict()

    nodes = list(G.nodes())
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)

    x = torch.zeros(n, 2)
    for i, nd in enumerate(nodes):
        x[i, 0] = expr_mean.get(nd, 0)
        x[i, 1] = expr_var.get(nd, 0)

    edge_list = []
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            edge_list.append([node_idx[u], node_idx[v]])
            edge_list.append([node_idx[v], node_idx[u]])
    if not edge_list:
        print("  No edges for GAT, skipping")
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    jt_map = dict(zip(jt_df['gene'], jt_df['jt_z']))
    y = torch.full((n,), float('nan'))
    train_mask = torch.zeros(n, dtype=torch.bool)
    for i, nd in enumerate(nodes):
        if nd in jt_map:
            y[i] = jt_map[nd]
            padj = jt_df.loc[jt_df['gene'] == nd, 'padj'].values
            if len(padj) > 0:
                if padj[0] < 0.05 or padj[0] > 0.5:
                    train_mask[i] = True

    if train_mask.sum() < 20:
        print("  Too few training genes for GAT")
        return None

    data = Data(x=x, edge_index=edge_index, y=y)

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(2, 64, heads=4, concat=True)
            self.conv2 = GATConv(256, 1, heads=1, concat=False)
        def forward(self, d):
            h = F.elu(self.conv1(d.x, d.edge_index))
            return self.conv2(h, d.edge_index).squeeze()

    model = GAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data).numpy()
    return pd.Series(pred, index=nodes)


def evaluate_gat_increment(rwr_scores, gat_scores):
    """If Spearman(GAT, RWR) > 0.7, GAT adds no value."""
    if gat_scores is None:
        return None
    common = list(set(rwr_scores.index) & set(gat_scores.index))
    r, p = spearmanr(rwr_scores[common], gat_scores[common])
    print(f"  GAT vs RWR: Spearman r={r:.3f}")
    report = r <= 0.7
    if not report:
        print("  → GAT redundant with RWR (supplement only)")
    else:
        print("  → GAT provides additional non-linear info")
    return {'spearman_r': r, 'report_gat': report}


def spatial_validation_rwr(top_genes, n_top=50):
    """Moran's I on RWR top genes in Visium data."""
    print("  Spatial validation of RWR top genes...")
    import scanpy as sc
    try:
        import squidpy as sq
    except ImportError:
        print("  squidpy not available, skipping")
        return pd.DataFrame()

    results = []
    for gp in ['GP1','GP2','GP3','GP4','GP5','GP6','GP7','GP8','GP9']:
        path = f"{SPATIAL_DIR}/{gp}"
        if not os.path.exists(path):
            continue
        try:
            sp = sc.read_visium(path)
            sp.var_names_make_unique()
            sc.pp.normalize_total(sp, target_sum=1e4)
            sc.pp.log1p(sp)
            sq.gr.spatial_neighbors(sp)
            avail = [g for g in top_genes[:n_top] if g in sp.var_names]
            if avail:
                sq.gr.spatial_autocorr(sp, genes=avail, mode='moran')
                for gene in avail:
                    if gene in sp.uns['moranI'].index:
                        results.append({'sample': gp, 'gene': gene,
                                        'moran_I': sp.uns['moranI'].loc[gene, 'I'],
                                        'pval': sp.uns['moranI'].loc[gene, 'pval_norm']})
        except Exception as e:
            print(f"    {gp}: failed ({e})")

    spatial_df = pd.DataFrame(results)
    if len(spatial_df) > 0:
        gene_agg = spatial_df.groupby('gene').agg(
            mean_moran=('moran_I', 'mean'), n_sig=('pval', lambda x: (x < 0.05).sum()))
        n_valid = (gene_agg['n_sig'] >= 2).sum()
        print(f"  Spatially significant (>=2 samples): {n_valid}/{len(gene_agg)}")
    return spatial_df


def main():
    print("=" * 60)
    print("Step 7: Graph Diffusion (RWR) + GAT")
    print("=" * 60)
    os.makedirs(f"{BASE}/results", exist_ok=True)

    # [1] Load seeds: MOFA top + WGCNA hub + CellChat LR
    print("\n[1] Loading seed genes...")
    seeds = set()

    mofa_path = f"{BASE}/results/mofa_top_genes.csv"
    if os.path.exists(mofa_path):
        mg = pd.read_csv(mofa_path)['gene'].tolist()
        seeds.update(mg[:30])
        print(f"  MOFA: {min(len(mg), 30)} seeds")

    hub_path = f"{BASE}/results/wgcna_hub_genes.csv"
    if os.path.exists(hub_path):
        hg = pd.read_csv(hub_path)['gene'].tolist()
        seeds.update(hg[:20])
        print(f"  WGCNA hub: {min(len(hg), 20)} seeds")

    cellchat_path = f"{BASE}/results/candidate_pool_D.csv"
    if os.path.exists(cellchat_path):
        cg = pd.read_csv(cellchat_path)['gene'].tolist()
        seeds.update(cg)
        print(f"  CellChat LR: {len(cg)} seeds")

    if not seeds:
        seeds = {"CDX2", "OLFM4", "LGR5", "NAMPT", "AREG", "PHLDA1",
                 "SOX9", "MYC", "CTNNB1", "IL1B", "NNMT", "CDH17"}
        print(f"  Fallback literature seeds: {len(seeds)}")

    # Candidate universe from TransitionRisk genes
    risk_path = f"{BASE}/results/transition_risk_genes.csv"
    if os.path.exists(risk_path):
        risk_genes = pd.read_csv(risk_path)['gene'].head(2000).tolist()
    else:
        risk_genes = list(seeds)
    candidates = list(set(risk_genes) | seeds)
    print(f"  Total candidate universe: {len(candidates)}")

    # [2] Build graph
    print("\n[2] Building gene graph (STRING + Dorothea)...")
    ppi_edges = load_string_ppi(candidates)
    doro_edges = load_dorothea_edges(candidates)
    print(f"  STRING edges (score>700): {len(ppi_edges)}")
    print(f"  Dorothea edges: {len(doro_edges)}")

    G = nx.Graph()
    G.add_nodes_from(candidates)
    for g1, g2, w in ppi_edges:
        G.add_edge(g1, g2, weight=w)
    for g1, g2, w in doro_edges:
        G.add_edge(g1, g2, weight=w)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # [3] RWR with sensitivity analysis
    print("\n[3] RWR with restart sensitivity (0.1, 0.3, 0.5, 0.7)...")
    rwr_scores, stable_genes, all_rankings = rwr_sensitivity(G, list(seeds))

    # [4] Traceability: shortest path from top genes to seeds
    print("\n[4] Path traceability (top 20)...")
    top_genes = rwr_scores.sort_values(ascending=False).head(50).index.tolist()
    path_df = shortest_path_traceability(G, list(seeds), top_genes)
    if not path_df.empty:
        path_df.to_csv(f"{BASE}/results/rwr_path_traceability.csv", index=False)
        print(f"  Traceable paths: {len(path_df)}")

    # [5] GAT (supplementary)
    print("\n[5] GAT supplementary model...")
    gat_scores = run_gat(G, list(seeds), candidates)

    # [6] Evaluate GAT increment
    print("\n[6] GAT increment evaluation...")
    gat_eval = evaluate_gat_increment(rwr_scores, gat_scores)

    # [7] Spatial validation
    print("\n[7] Spatial validation (Moran's I)...")
    spatial_df = spatial_validation_rwr(top_genes)
    if not spatial_df.empty:
        spatial_df.to_csv(f"{BASE}/results/rwr_spatial_validation.csv", index=False)

    # [8] Save final output
    print("\n[8] Saving results...")
    rwr_df = pd.DataFrame({'gene': rwr_scores.index, 'network_score': rwr_scores.values})
    rwr_df = rwr_df.sort_values('network_score', ascending=False).reset_index(drop=True)
    rwr_df['rank'] = range(1, len(rwr_df) + 1)
    rwr_df['is_seed'] = rwr_df['gene'].isin(seeds)
    rwr_df['stable_top30'] = rwr_df['gene'].isin(stable_genes)
    if gat_scores is not None:
        rwr_df['gat_score'] = rwr_df['gene'].map(gat_scores)
    rwr_df.to_csv(f"{BASE}/results/graph_ranked_genes.csv", index=False)

    print(f"\n{'='*60}")
    print("Step 7 COMPLETE")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Seeds: {len(seeds)}")
    print(f"  Stable top30: {len(stable_genes)}")
    print(f"  Top 5: {top_genes[:5]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
