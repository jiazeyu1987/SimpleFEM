#!/usr/bin/env python3
"""
诊断ROI2目视无变化但波形有明显变化的问题
"""

import json
import os
import numpy as np
from PIL import Image, ImageStat

def diagnose_roi2_wave_issue():
    cache_file = "export/roi_analysis_cache_测试3_20251219_091445_aeeca85d5b22.jsonl"

    if not os.path.exists(cache_file):
        print(f"缓存文件不存在: {cache_file}")
        return

    print("诊断ROI2-波形不匹配问题")
    print("=" * 60)

    # 读取数据
    frames_data = []
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get('type') == 'frame':
                    frame_idx = data.get('frame_index', 0)
                    if 15 <= frame_idx <= 35:  # 扩大范围，查看波峰前后的情况
                        frames_data.append(data)

    print(f"找到 {len(frames_data)} 帧数据 (15-35帧)\n")

    # 分析ROI2数据一致性
    print("🔍 ROI2数据一致性检查:")
    print("-" * 50)

    for frame in frames_data:
        frame_idx = frame.get('frame_index', 0)
        roi2_region = frame.get('roi2_region')
        roi2_gray = frame.get('roi2_gray')
        intersection = frame.get('intersection', {}).get('used')

        if roi2_region and len(roi2_region) == 4:
            # 计算ROI2区域大小
            width = roi2_region[2] - roi2_region[0]
            height = roi2_region[3] - roi2_region[1]
            size = width * height

            # 计算ROI2中心点
            center_x = (roi2_region[0] + roi2_region[2]) / 2
            center_y = (roi2_region[1] + roi2_region[3]) / 2

            print(f"帧{frame_idx:2d}: ROI2({roi2_region[0]:3d},{roi2_region[1]:3d},{roi2_region[2]:3d},{roi2_region[3]:3d}) "
                  f"大小{width:2d}x{height:2d}={size:4d}px 中心({center_x:5.1f},{center_y:5.1f}) "
                  f"灰度{roi2_gray:6.1f}")

    # 检查ROI2区域稳定性
    print(f"\n📊 ROI2区域稳定性分析:")
    print("-" * 50)

    roi2_regions = []
    gray_values = []
    intersection_points = []

    for frame in frames_data:
        frame_idx = frame.get('frame_index', 0)
        roi2_region = frame.get('roi2_region')
        roi2_gray = frame.get('roi2_gray')
        intersection = frame.get('intersection', {}).get('used')

        if roi2_region and len(roi2_region) == 4 and roi2_gray is not None:
            roi2_regions.append((frame_idx, roi2_region))
            gray_values.append((frame_idx, roi2_gray))
            if intersection and len(intersection) == 2:
                intersection_points.append((frame_idx, intersection[0], intersection[1]))

    # 分析ROI2区域变化
    roi2_changes = []
    for i in range(1, len(roi2_regions)):
        prev_idx, prev_region = roi2_regions[i-1]
        curr_idx, curr_region = roi2_regions[i]

        # 计算区域变化
        x1_change = abs(curr_region[0] - prev_region[0])
        y1_change = abs(curr_region[1] - prev_region[1])
        x2_change = abs(curr_region[2] - prev_region[2])
        y2_change = abs(curr_region[3] - prev_region[3])

        total_change = x1_change + y1_change + x2_change + y2_change

        # 计算中心点变化
        prev_center_x = (prev_region[0] + prev_region[2]) / 2
        prev_center_y = (prev_region[1] + prev_region[3]) / 2
        curr_center_x = (curr_region[0] + curr_region[2]) / 2
        curr_center_y = (curr_region[1] + curr_region[3]) / 2
        center_change = ((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)**0.5

        roi2_changes.append({
            'frame': curr_idx,
            'total_change': total_change,
            'center_change': center_change,
            'region': curr_region
        })

        print(f"帧{curr_idx:2d}: 区域变化{total_change:2d}px, 中心变化{center_change:5.2f}px")

    # 分析灰度值变化
    print(f"\n📈 灰度值变化分析:")
    print("-" * 50)

    gray_changes = []
    for i in range(1, len(gray_values)):
        prev_idx, prev_gray = gray_values[i-1]
        curr_idx, curr_gray = gray_values[i]
        change = abs(curr_gray - prev_gray)
        gray_changes.append((curr_idx, change))

        if change > 5:  # 显示明显变化
            print(f"帧{curr_idx:2d}: 灰度变化{change:6.1f} ({prev_gray:6.1f} → {curr_gray:6.1f})")

    # 关键分析：ROI2稳定性 vs 灰度变化
    print(f"\n🎯 关键问题分析:")
    print("-" * 50)

    max_roi2_change = max([c['center_change'] for c in roi2_changes]) if roi2_changes else 0
    max_gray_change = max([c[1] for c in gray_changes]) if gray_changes else 0

    print(f"最大ROI2中心变化: {max_roi2_change:.2f} px")
    print(f"最大灰度值变化: {max_gray_change:.1f}")

    # 查找波峰帧
    peak_frames = []
    for frame in frames_data:
        frame_idx = frame.get('frame_index', 0)
        peaks = frame.get('peaks', {})
        if len(peaks.get('green', [])) > 0 or len(peaks.get('red', [])) > 0:
            peak_frames.append(frame_idx)

    print(f"检测到峰值的帧: {peak_frames}")

    # 检查波峰帧的ROI2和灰度变化
    print(f"\n🔍 波峰帧详细分析:")
    print("-" * 50)

    for peak_frame in peak_frames:
        # 找到该帧的数据
        peak_data = None
        for frame in frames_data:
            if frame.get('frame_index', 0) == peak_frame:
                peak_data = frame
                break

        if peak_data:
            roi2_region = peak_data.get('roi2_region')
            roi2_gray = peak_data.get('roi2_gray')
            threshold = peak_data.get('threshold', {}).get('used', 0)

            # 计算与前一帧的ROI2变化
            roi2_change = "N/A"
            gray_change = "N/A"

            if roi2_region and len(roi2_region) == 4:
                # 查找前一帧
                prev_frame = None
                for frame in frames_data:
                    if frame.get('frame_index', 0) == peak_frame - 1:
                        prev_frame = frame
                        break

                if prev_frame:
                    prev_roi2 = prev_frame.get('roi2_region')
                    prev_gray = prev_frame.get('roi2_gray')

                    if prev_roi2 and len(prev_roi2) == 4:
                        prev_center_x = (prev_roi2[0] + prev_roi2[2]) / 2
                        prev_center_y = (prev_roi2[1] + prev_roi2[3]) / 2
                        curr_center_x = (roi2_region[0] + roi2_region[2]) / 2
                        curr_center_y = (roi2_region[1] + roi2_region[3]) / 2
                        center_change = ((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)**0.5
                        roi2_change = f"{center_change:.2f}px"

                    if prev_gray is not None:
                        gray_change = f"{abs(roi2_gray - prev_gray):.1f}"

            print(f"帧{peak_frame:2d}: 灰度{roi2_gray:6.1f} (变化{gray_change}), 阈值{threshold:6.1f}, ROI2变化{roi2_change}")

    # 问题诊断结论
    print(f"\n🔍 问题诊断结论:")
    print("-" * 50)

    if max_roi2_change < 2.0 and max_gray_change > 10.0:
        print("⚠️  发现问题：ROI2几乎无变化但灰度值变化明显")
        print("\n可能原因:")
        print("1. ROI2灰度计算错误")
        print("2. ROI2区域定义错误")
        print("3. 图像处理管道中有问题")
        print("4. 数据记录或缓存错误")

        print("\n建议检查:")
        print("1. 检查compute_average_gray()函数")
        print("2. 验证ROI2区域坐标是否正确")
        print("3. 查看实际的ROI2图像文件")
        print("4. 检查绿色线检测和交点计算")

    elif max_roi2_change < 5.0 and peak_frames:
        print("📋 ROI2变化较小，可能是:")
        print("1. 防抖动作用正常")
        print("2. 实际的图像变化很小")
        print("3. 需要检查ROI2图像确认")

    else:
        print("✅ ROI2变化和灰度变化基本一致")

    # 检查ROI2图像文件
    print(f"\n📁 建议检查以下文件:")
    print("-" * 50)

    # 查找测试3的临时文件夹
    for root, dirs, files in os.walk("."):
        if "测试3" in root and "roi2" in dirs:
            roi2_dir = os.path.join(root, "roi2")
            print(f"ROI2图像目录: {roi2_dir}")

            # 查找20-30帧的ROI2图像
            for frame_idx in range(20, 31):
                possible_names = [
                    f"roi2_{frame_idx:06d}.png",
                    f"roi2_{frame_idx:06d}_0000.00s.png"
                ]

                for name in possible_names:
                    file_path = os.path.join(roi2_dir, name)
                    if os.path.exists(file_path):
                        print(f"  帧{frame_idx:2d}: {file_path}")
                        break

            break

if __name__ == "__main__":
    diagnose_roi2_wave_issue()