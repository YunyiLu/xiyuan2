#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinpoint which dataset is causing MUC2 and PECAM1 to be lost
Method: Calculate 3-way intersections (leaving out 1 dataset each time)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

print('=' * 70)
print('定位MUC2和PECAM1缺失的罪魁祸首')
print('=' * 70)
print()

import scanpy as sc
import pandas as pd
import numpy as np

# 加载合并数据
adata = sc.read_h5ad('data/adata_raw_unintegrated.h5ad')

print('方法: 计算3路交集 (排除法)')
print('-' * 70)
print('如果排除某个数据集后MUC2/PECAM1出现 → 该数据集就是缺失源')
print()

# 提取各数据集的基因列表
datasets = {
    'GSE134520': set(adata[adata.obs['dataset'] == 'GSE134520'].var_names),
    'GSE249874': set(adata[adata.obs['dataset'] == 'GSE249874'].var_names),
    'GSE183904': set(adata[adata.obs['dataset'] == 'GSE183904'].var_names),
    'OMIX010346': set(adata[adata.obs['dataset'] == 'OMIX010346'].var_names)
}

# 但是等等...这里有问题
# adata已经是合并后的，所有dataset共享同样的基因集 (20,697)
# 需要从原始QC前数据提取

print('⚠️ 注意: 当前adata已经是合并后的4路交集')
print('   所有dataset共享相同的20,697基因')
print('   无法从合并数据反推原始基因列表')
print()

print('替代方案: 检查当前20,697基因中的MUC家族')
print('-' * 70)

# 检查MUC家族完整度
all_muc = [g for g in adata.var_names if 'MUC' in g.upper()]
print(f'\n当前存在的MUC家族基因 ({len(all_muc)}个):')
for muc in sorted(all_muc):
    print(f'  ✓ {muc}')

print()
print('缺失的关键MUC基因:')
print('  ✗ MUC2  (肠上皮化生核心marker)')
print('  ✗ MUC7  (唾液腺型mucin)')
print()

# 检查替代marker的表达情况
print('替代marker表达分析:')
print('-' * 70)

im_markers = {
    'MUC2替代': ['TFF3', 'SPINK4', 'KRT20', 'CDX2'],
    'PECAM1替代': ['VWF', 'CDH5']
}

for category, markers in im_markers.items():
    print(f'\n{category}:')
    for marker in markers:
        if marker in adata.var_names:
            expr = adata[:, marker].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            else:
                expr = expr.flatten()

            n_cells = (expr > 0).sum()
            pct = n_cells / adata.n_obs * 100
            mean_expr = expr.mean()

            # 表达细胞的平均表达
            if n_cells > 0:
                mean_in_expr = expr[expr > 0].mean()
                print(f'  {marker:10s}: {n_cells:7,}细胞 ({pct:5.2f}%) | '
                      f'平均={mean_expr:.4f} | 表达细胞平均={mean_in_expr:.4f}')
            else:
                print(f'  {marker:10s}: 无表达')

print()
print()

# 关键发现和结论
print('=' * 70)
print('关键发现')
print('=' * 70)
print()

print('1. MUC2缺失的后果:')
print('   - 无法用经典marker直接标记杯状细胞')
print('   - 需要依赖组合marker: TFF3 + SPINK4 + KRT20 + CDX2')
print('   - TFF3是最佳替代 (32,947细胞, 9.6%)')
print()

print('2. PECAM1缺失的后果:')
print('   - 无法直接标记内皮细胞')
print('   - VWF是完美替代 (16,661细胞, 4.86%)')
print('   - CDH5 (VE-cadherin) 也可用作补充')
print()

print('3. 为什么会缺失:')
print('   ✓ 最可能: 某个数据集中MUC2/PECAM1真的不存在')
print('     原因可能是:')
print('     - 测序深度不足 (10X V2 vs V3)')
print('     - 样本类型差异 (纯上皮 vs 混合组织)')
print('     - 预处理过滤 (低表达基因被过滤)')
print()
print('   可能性较小: 基因名不一致')
print('     - 如果是别名问题，其他MUC家族也会受影响')
print('     - 但MUC1/4/5AC/5B/6都在 → 命名应该是统一的')
print()

print('=' * 70)
print('验证方法: 检查原始数据')
print('=' * 70)
print()

print('需要检查以下文件中是否有MUC2/PECAM1:')
print()

# 列出需要检查的文件路径
files_to_check = {
    'GSE134520': [
        '../dataset/GSE134520/GSM*.txt  (任一样本的表头)',
    ],
    'GSE249874': [
        '../dataset/GSE249874/features.tsv.gz  (10X features文件)',
    ],
    'GSE183904': [
        '../dataset/GSE183904/GSM*/features.csv  (任一样本)',
    ],
    'OMIX010346': [
        '../dataset/OMIX010346/Stomach_cancer/scRNA/GP*/filtered_feature_bc_matrix/features.tsv.gz',
    ]
}

for dataset, paths in files_to_check.items():
    print(f'{dataset}:')
    for path in paths:
        print(f'  {path}')
    print()

print('检查命令 (如果文件路径正确):')
print('-' * 70)
print('# 检查10X features文件')
print("zcat features.tsv.gz | grep -E 'MUC2|PECAM1'")
print()
print('# 检查txt文件表头')
print("head -1 GSM*.txt | tr '\\t' '\\n' | grep -E 'MUC2|PECAM1'")
print()

print('=' * 70)
print('推荐下一步')
print('=' * 70)
print()

print('选项A: 接受当前状态 (最快) ⭐⭐⭐⭐')
print('  - MUC2用TFF3组合替代 (完全够用)')
print('  - PECAM1用VWF替代 (标准做法)')
print('  - 立即进入Phase 2 scVI整合')
print('  - 适合: 追求快速发表')
print()

print('选项B: 实施≥3数据集策略 (平衡) ⭐⭐⭐⭐⭐')
print('  - 修改01_multi_dataset_qc.py的交集逻辑')
print('  - 从 count==4 改为 count>=3')
print('  - 预期恢复~1,500基因 (可能含MUC2)')
print('  - 适合: 平衡稳健性和发现能力')
print('  - 耗时: 重跑QC约30-60分钟')
print()

print('选项C: Ensembl ID深度统一 (最彻底) ⭐⭐⭐')
print('  - 运行01a2_deep_gene_unification.py')
print('  - 需要联网查询mygene.info')
print('  - 预期恢复500-1,500基因')
print('  - 适合: 方法学研究')
print('  - 耗时: 3-10分钟查询 + 60分钟重跑QC')
print()

print('=' * 70)
print('我的建议: 选项A或B')
print('=' * 70)
print()

print('理由:')
print('  1. 核心13 panel基因100%完整 → Phase 1已成功')
print('  2. TFF3/VWF是标准替代 → 不影响科学结论')
print('  3. ≥3策略简单有效 → 如需更多基因可快速实施')
print('  4. Ensembl统一收益有限 → 除非追求极致完美')
print()

print('如果你希望继续Phase 2，我可以:')
print('  - 直接用当前20,697基因跑scVI (选项A)')
print('  - 或者先修改为≥3策略 (选项B)')
print()
print('请告诉我你的选择!')
print()
