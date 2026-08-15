#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Gene Unification - via Ensembl ID and HGNC synonyms
Resolve the issue of same genes with different names across datasets

Goal: Increase from 20,697 to 21,500-22,000 genes, possibly recover MUC2
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("Deep Gene Name Unification Analysis")
print("=" * 60)
print()

# ============================================================================
# Step 1: 加载各数据集原始基因列表 (QC前)
# ============================================================================

print("Step 1: 提取各数据集原始基因列表...")
print("-" * 60)

# GSE134520: 从txt文件提取基因名 (第一列)
genes_134520 = []
data_dir = Path("../dataset/GSE134520")
for txt_file in data_dir.glob("GSM*.txt"):
    df = pd.read_csv(txt_file, sep='\t', nrows=0)  # 只读表头
    genes_134520 = df.columns.tolist()[1:]  # 跳过第一列(细胞名)
    break

print(f"✓ GSE134520: {len(genes_134520):,} genes (从{txt_file.name})")

# GSE249874: 从10X features.tsv提取
features_249874 = pd.read_csv("../dataset/GSE249874/features.tsv.gz",
                              sep='\t', header=None, compression='gzip')
genes_249874 = features_249874[1].tolist()  # 第二列是gene symbol
print(f"✓ GSE249874: {len(genes_249874):,} genes (10X features)")

# GSE183904: 从任一样本的features.csv提取
sample_dir = Path("../dataset/GSE183904").glob("GSM*")
first_sample = next(sample_dir)
features_183904 = pd.read_csv(first_sample / "features.csv")
genes_183904 = features_183904.iloc[:, 0].tolist()  # 第一列通常是gene
print(f"✓ GSE183904: {len(genes_183904):,} genes (从{first_sample.name})")

# OMIX010346: 从10X features.tsv提取
features_omix = pd.read_csv("../dataset/OMIX010346/Stomach_cancer/scRNA/GP4/filtered_feature_bc_matrix/features.tsv.gz",
                            sep='\t', header=None, compression='gzip')
genes_omix = features_omix[1].tolist()
print(f"✓ OMIX010346: {len(genes_omix):,} genes (10X features, GP4)")

print()

# ============================================================================
# Step 2: 使用mygene.info通过Ensembl ID统一 (最可靠方法)
# ============================================================================

print("Step 2: 通过mygene.info统一到Ensembl ID...")
print("-" * 60)
print("提示: 这需要联网查询，约需3-10分钟")
print()

try:
    import mygene
    mg = mygene.MyGeneInfo()

    def batch_query_ensembl(gene_list, dataset_name, batch_size=1000):
        """分批查询避免超时"""
        results = []
        total_batches = (len(gene_list) + batch_size - 1) // batch_size

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i+batch_size]
            batch_num = i // batch_size + 1

            print(f"  {dataset_name}: batch {batch_num}/{total_batches}...", end='\r')

            res = mg.querymany(batch,
                              scopes='symbol,alias,ensembl.gene',
                              fields='ensembl.gene,symbol',
                              species='human',
                              verbose=False)
            results.extend(res)

        print(f"  {dataset_name}: {len(results):,} queries completed" + " " * 20)
        return results

    # 查询各数据集
    print("开始批量查询...")
    results_134520 = batch_query_ensembl(genes_134520, "GSE134520")
    results_249874 = batch_query_ensembl(genes_249874, "GSE249874")
    results_183904 = batch_query_ensembl(genes_183904, "GSE183904")
    results_omix = batch_query_ensembl(genes_omix, "OMIX010346")

    print()
    print("✓ 所有数据集查询完成")
    print()

    # 构建映射
    print("构建Ensembl ID映射...")

    ensembl_to_symbol = {}  # Ensembl ID -> 官方symbol
    symbol_to_ensembl = {}  # 原始symbol -> Ensembl ID

    all_results = results_134520 + results_249874 + results_183904 + results_omix

    unmapped_count = 0
    mapped_count = 0

    for res in all_results:
        query = res['query']

        if 'ensembl' in res and res['ensembl'] is not None:
            # 处理单个或多个Ensembl ID
            if isinstance(res['ensembl'], list):
                ensembl_id = res['ensembl'][0]['gene']
            else:
                ensembl_id = res['ensembl']['gene'] if 'gene' in res['ensembl'] else None

            if ensembl_id:
                symbol_to_ensembl[query] = ensembl_id
                mapped_count += 1

                # 优先使用官方symbol
                if 'symbol' in res and res['symbol']:
                    ensembl_to_symbol[ensembl_id] = res['symbol']
                else:
                    if ensembl_id not in ensembl_to_symbol:
                        ensembl_to_symbol[ensembl_id] = query
        else:
            unmapped_count += 1

    print(f"  映射成功: {mapped_count:,} symbols")
    print(f"  未映射: {unmapped_count:,} symbols ({unmapped_count/len(all_results)*100:.1f}%)")
    print()

    # 基于Ensembl求交集
    print("计算基于Ensembl ID的交集...")

    ensembl_134520 = set([symbol_to_ensembl[g] for g in genes_134520
                          if g in symbol_to_ensembl])
    ensembl_249874 = set([symbol_to_ensembl[g] for g in genes_249874
                          if g in symbol_to_ensembl])
    ensembl_183904 = set([symbol_to_ensembl[g] for g in genes_183904
                          if g in symbol_to_ensembl])
    ensembl_omix = set([symbol_to_ensembl[g] for g in genes_omix
                       if g in symbol_to_ensembl])

    print(f"  GSE134520: {len(ensembl_134520):,} Ensembl IDs")
    print(f"  GSE249874: {len(ensembl_249874):,} Ensembl IDs")
    print(f"  GSE183904: {len(ensembl_183904):,} Ensembl IDs")
    print(f"  OMIX010346: {len(ensembl_omix):,} Ensembl IDs")
    print()

    # 多版本交集
    common_4way = ensembl_134520 & ensembl_249874 & ensembl_183904 & ensembl_omix
    common_3plus = set()
    for ens_id in ensembl_134520 | ensembl_249874 | ensembl_183904 | ensembl_omix:
        count = sum([ens_id in s for s in [ensembl_134520, ensembl_249874,
                                            ensembl_183904, ensembl_omix]])
        if count >= 3:
            common_3plus.add(ens_id)

    common_2plus = set()
    for ens_id in ensembl_134520 | ensembl_249874 | ensembl_183904 | ensembl_omix:
        count = sum([ens_id in s for s in [ensembl_134520, ensembl_249874,
                                            ensembl_183904, ensembl_omix]])
        if count >= 2:
            common_2plus.add(ens_id)

    print("=" * 60)
    print("基于Ensembl ID的交集结果:")
    print("=" * 60)
    print(f"4路严格交集: {len(common_4way):,} genes")
    print(f"≥3数据集:    {len(common_3plus):,} genes (+{len(common_3plus)-len(common_4way):,}, +{(len(common_3plus)-len(common_4way))/len(common_4way)*100:.1f}%)")
    print(f"≥2数据集:    {len(common_2plus):,} genes (+{len(common_2plus)-len(common_4way):,}, +{(len(common_2plus)-len(common_4way))/len(common_4way)*100:.1f}%)")
    print()
    print(f"vs 当前方法 (简单symbol匹配): 20,697 genes")
    print(f"增加: {len(common_4way) - 20697:+,} genes ({(len(common_4way) - 20697)/20697*100:+.1f}%)")
    print()

    MYGENE_SUCCESS = True

except ImportError:
    print("⚠ mygene包未安装，跳过Ensembl ID统一")
    print("  安装: pip install mygene")
    print()
    MYGENE_SUCCESS = False

except Exception as e:
    print(f"⚠ mygene查询失败: {e}")
    print("  可能是网络问题，将使用备用方法")
    print()
    MYGENE_SUCCESS = False

# ============================================================================
# Step 3: 检查关键Panel基因恢复情况
# ============================================================================

if MYGENE_SUCCESS:
    print("Step 3: 检查关键Panel基因恢复情况...")
    print("-" * 60)

    panel_genes = ['MUC2', 'PECAM1', 'ITLN1', 'PRAP1', 'FABP1',
                   'OLFM4', 'REG4', 'CDX2', 'MUC5AC']

    for pg in panel_genes:
        # 检查是否被映射
        if pg in symbol_to_ensembl:
            ensembl_id = symbol_to_ensembl[pg]

            # 检查在哪些数据集出现
            in_datasets = []
            if ensembl_id in ensembl_134520: in_datasets.append("GSE134520")
            if ensembl_id in ensembl_249874: in_datasets.append("GSE249874")
            if ensembl_id in ensembl_183904: in_datasets.append("GSE183904")
            if ensembl_id in ensembl_omix: in_datasets.append("OMIX010346")

            n_datasets = len(in_datasets)
            official = ensembl_to_symbol.get(ensembl_id, pg)

            if n_datasets == 4:
                status = "✓ 4路交集"
            elif n_datasets >= 3:
                status = f"⚠ {n_datasets}数据集 (可用≥3策略)"
            elif n_datasets >= 2:
                status = f"⚠ {n_datasets}数据集 (可用≥2策略)"
            else:
                status = "✗ 单数据集"

            print(f"  {pg:10s} → {official:10s} | {status} | {', '.join(in_datasets)}")
        else:
            print(f"  {pg:10s} → (未映射)   | ✗ 无Ensembl ID")

    print()

# ============================================================================
# Step 4: 输出增强映射表
# ============================================================================

if MYGENE_SUCCESS:
    print("Step 4: 生成增强基因映射表...")
    print("-" * 60)

    # 构建完整映射表
    mapping_data = {
        'dataset': [],
        'original_symbol': [],
        'ensembl_id': [],
        'official_symbol': [],
        'in_4way': [],
        'in_3plus': [],
        'in_2plus': []
    }

    for dataset, genes in [('GSE134520', genes_134520),
                           ('GSE249874', genes_249874),
                           ('GSE183904', genes_183904),
                           ('OMIX010346', genes_omix)]:
        for g in genes:
            if g in symbol_to_ensembl:
                ensembl_id = symbol_to_ensembl[g]
                official = ensembl_to_symbol.get(ensembl_id, g)

                mapping_data['dataset'].append(dataset)
                mapping_data['original_symbol'].append(g)
                mapping_data['ensembl_id'].append(ensembl_id)
                mapping_data['official_symbol'].append(official)
                mapping_data['in_4way'].append(ensembl_id in common_4way)
                mapping_data['in_3plus'].append(ensembl_id in common_3plus)
                mapping_data['in_2plus'].append(ensembl_id in common_2plus)

    df_mapping = pd.DataFrame(mapping_data)

    # 保存
    os.makedirs('data', exist_ok=True)
    df_mapping.to_csv('data/gene_mapping_enhanced.csv', index=False)

    print(f"✓ 映射表已保存: data/gene_mapping_enhanced.csv")
    print(f"  总映射数: {len(df_mapping):,}")
    print()

    # 输出各版本基因列表
    for version, gene_set in [('4way', common_4way),
                              ('3plus', common_3plus),
                              ('2plus', common_2plus)]:
        genes_list = [ensembl_to_symbol[ens] for ens in gene_set]
        pd.DataFrame({'ensembl_id': list(gene_set),
                     'symbol': genes_list}).to_csv(
            f'data/genes_{version}.csv', index=False)
        print(f"✓ {version}基因列表已保存: data/genes_{version}.csv ({len(gene_set):,} genes)")

    print()

# ============================================================================
# Step 5: 检查潜在同义词 (简单方法，不需联网)
# ============================================================================

print("Step 5: 检查潜在同义词 (基于编辑距离)...")
print("-" * 60)

from difflib import SequenceMatcher

def similar(a, b, threshold=0.85):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio() >= threshold

# 只检查在原始交集外的基因
genes_set_134520 = set(genes_134520)
genes_set_249874 = set(genes_249874)
genes_set_183904 = set(genes_183904)
genes_set_omix = set(genes_omix)

# 当前简单交集 (symbol完全匹配)
simple_intersect = genes_set_134520 & genes_set_249874 & genes_set_183904 & genes_set_omix

# 找潜在同义词
potential_synonyms = []

print("检查GSE134520 vs GSE249874的潜在同义词...")
for g1 in genes_set_134520:
    if g1 in simple_intersect:
        continue
    for g2 in genes_set_249874:
        if g2 in simple_intersect:
            continue
        # 高相似度且不完全相同
        if g1 != g2 and similar(g1, g2, 0.85):
            potential_synonyms.append((g1, g2, 'GSE134520', 'GSE249874'))

print(f"  发现 {len(potential_synonyms)} 对潜在同义词")

# 输出前20对供人工审核
if potential_synonyms:
    print("\n前20对潜在同义词 (需人工审核):")
    print("-" * 60)
    for i, (g1, g2, ds1, ds2) in enumerate(potential_synonyms[:20]):
        sim = SequenceMatcher(None, g1, g2).ratio()
        print(f"  {i+1:2d}. {g1:20s} ({ds1}) ↔ {g2:20s} ({ds2}) | 相似度:{sim:.2f}")

    # 保存完整列表
    df_synonyms = pd.DataFrame(potential_synonyms,
                               columns=['gene1', 'gene2', 'dataset1', 'dataset2'])
    df_synonyms['similarity'] = df_synonyms.apply(
        lambda row: SequenceMatcher(None, row['gene1'], row['gene2']).ratio(),
        axis=1)
    df_synonyms.to_csv('data/potential_synonyms.csv', index=False)
    print(f"\n✓ 完整列表已保存: data/potential_synonyms.csv ({len(potential_synonyms)} pairs)")
else:
    print("  未发现高相似度的潜在同义词")

print()

# ============================================================================
# 总结
# ============================================================================

print("=" * 60)
print("分析完成")
print("=" * 60)
print()

if MYGENE_SUCCESS:
    print("✓ Ensembl ID统一成功")
    print()
    print("建议下一步:")
    print("1. 检查 data/gene_mapping_enhanced.csv 确认映射正确")
    print("2. 运行 01c_rebuild_with_enhanced_genes.py 重建数据集")
    print("3. 选择基因集策略:")
    print("   - Phase 2A/4 (scVI/TransitionRisk): 用4way")
    print("   - Phase 2B/6/8 (注释/WGCNA/Marker): 用3plus或单数据集full")
    print("   - Phase 11 (空间): 用OMIX full")
else:
    print("⚠ Ensembl ID统一失败，请:")
    print("1. 安装mygene: pip install mygene")
    print("2. 检查网络连接")
    print("3. 重新运行本脚本")

print()
print("重要发现:")
if MYGENE_SUCCESS:
    print(f"- 基于Ensembl的4路交集比简单匹配多/少 {len(common_4way) - 20697:+,} 基因")
    print(f"- 使用≥3数据集策略可获得 {len(common_3plus):,} 基因 (+{len(common_3plus)-20697:,})")
    print(f"- 使用≥2数据集策略可获得 {len(common_2plus):,} 基因 (+{len(common_2plus)-20697:,})")
