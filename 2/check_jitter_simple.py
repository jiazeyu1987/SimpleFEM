#!/usr/bin/env python3
"""
简单的ROI2抖动检查工具
"""

import json
import os
import sys

def analyze_jsonl(file_path):
    try:
        roi2_regions = []
        frame_indices = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('type') == 'frame' and 'roi2_region' in data:
                        region = data['roi2_region']
                        if region and len(region) == 4:
                            roi2_regions.append(region)
                            frame_indices.append(data.get('frame_index', line_num))
                except json.JSONDecodeError:
                    continue

        print(f"文件: {os.path.basename(file_path)}")
        print(f"找到 {len(roi2_regions)} 个ROI2区域")

        if len(roi2_regions) < 2:
            print("数据不足，无法分析变化")
            return

        # 分析变化
        changes = []
        for i in range(1, len(roi2_regions)):
            prev = roi2_regions[i-1]
            curr = roi2_regions[i]

            # 计算中心点变化
            prev_center_x = (prev[0] + prev[2]) / 2
            prev_center_y = (prev[1] + prev[3]) / 2
            curr_center_x = (curr[0] + curr[2]) / 2
            curr_center_y = (curr[1] + curr[3]) / 2

            distance = ((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)**0.5
            changes.append(distance)

        if not changes:
            print("没有计算到变化数据")
            return

        import math
        avg_change = sum(changes) / len(changes)
        max_change = max(changes)
        min_change = min(changes)

        # 统计不同范围的变化
        small = sum(1 for c in changes if c <= 5)
        medium = sum(1 for c in changes if 5 < c <= 15)
        large = sum(1 for c in changes if c > 15)

        print(f"平均变化: {avg_change:.2f} 像素")
        print(f"最大变化: {max_change:.2f} 像素")
        print(f"最小变化: {min_change:.2f} 像素")
        print(f"小变化(≤5px): {small} 次 ({small/len(changes)*100:.1f}%)")
        print(f"中等变化(5-15px): {medium} 次 ({medium/len(changes)*100:.1f}%)")
        print(f"大变化(>15px): {large} 次 ({large/len(changes)*100:.1f}%)")

        # 效果评估
        if avg_change <= 3:
            print("✅ 防抖动效果优秀")
        elif avg_change <= 6:
            print("🟡 防抖动效果良好")
        elif avg_change <= 10:
            print("🟠 防抖动效果一般")
        else:
            print("❌ 防抖动效果需要改进")

        # 显示前10个变化作为样本
        print("\n前10个变化样本:")
        for i in range(min(10, len(changes))):
            print(f"  帧{frame_indices[i]}->帧{frame_indices[i+1]}: {changes[i]:.2f}px")

    except Exception as e:
        print(f"分析文件时出错: {e}")

if __name__ == "__main__":
    export_dir = os.path.join(os.path.dirname(__file__), 'export')

    # 查找jsonl文件
    jsonl_files = []
    if os.path.exists(export_dir):
        for filename in os.listdir(export_dir):
            if filename.startswith('roi_analysis_cache_') and filename.endswith('.jsonl'):
                jsonl_files.append(os.path.join(export_dir, filename))

    if not jsonl_files:
        print("没有找到分析缓存文件")
        sys.exit(1)

    print("ROI2防抖动效果分析")
    print("=" * 50)

    for file_path in jsonl_files:
        analyze_jsonl(file_path)
        print("-" * 50)