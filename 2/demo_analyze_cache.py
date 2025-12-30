#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_cache.py 快速演示脚本

自动分析最近的ROI分析缓存文件并生成完整报告
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

def find_latest_cache_file():
    """查找最新的缓存文件"""
    cache_files = glob.glob('export/roi_analysis_cache_*.jsonl')
    if not cache_files:
        print("错误: 未找到任何 ROI 分析缓存文件")
        print("请确保 export/ 文件夹中存在 roi_analysis_cache_*.jsonl 文件")
        sys.exit(1)

    # 按修改时间排序，获取最新的
    latest = max(cache_files, key=os.path.getmtime)
    return latest

def main():
    print("="*80)
    print("ROI Analysis Cache Analyzer - 快速演示")
    print("="*80)

    # 查找最新的缓存文件
    cache_file = find_latest_cache_file()
    print(f"\n使用缓存文件: {cache_file}")
    print(f"文件大小: {os.path.getsize(cache_file) / 1024:.2f} KB")

    # 创建输出目录
    output_dir = Path('cache_analysis_output')
    output_dir.mkdir(exist_ok=True)

    base_name = Path(cache_file).stem
    timestamp = subprocess.check_output(['powershell', '-Command',
                                        'Get-Date -Format "yyyyMMdd_HHmmss"'],
                                       text=True).strip()

    print(f"\n输出目录: {output_dir}")

    # 1. 生成摘要
    print("\n" + "-"*80)
    print("步骤 1/4: 生成统计摘要...")
    print("-"*80)
    subprocess.run([sys.executable, 'analyze_cache.py', cache_file, '--summary'])

    # 2. 波峰详细分析
    print("\n" + "-"*80)
    print("步骤 2/4: 分析波峰检测...")
    print("-"*80)
    subprocess.run([sys.executable, 'analyze_cache.py', cache_file, '--peaks'])

    # 3. 交点分析
    print("\n" + "-"*80)
    print("步骤 3/4: 分析绿线交点...")
    print("-"*80)
    subprocess.run([sys.executable, 'analyze_cache.py', cache_file, '--intersection'])

    # 4. 生成波形图
    print("\n" + "-"*80)
    print("步骤 4/4: 生成波形图...")
    print("-"*80)
    waveform_file = output_dir / f'waveform_{base_name}_{timestamp}.png'
    subprocess.run([sys.executable, 'analyze_cache.py', cache_file,
                   '--waveform', '--output-waveform', str(waveform_file)])

    # 5. 导出CSV
    print("\n" + "-"*80)
    print("额外步骤: 导出到CSV...")
    print("-"*80)
    csv_file = output_dir / f'data_{base_name}_{timestamp}.csv'
    subprocess.run([sys.executable, 'analyze_cache.py', cache_file,
                   '--export-csv', str(csv_file)])

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - 波形图: {waveform_file}")
    print(f"  - CSV数据: {csv_file}")
    print(f"\n可以使用 Excel 或其他工具打开 CSV 文件进行进一步分析。")

if __name__ == '__main__':
    main()
