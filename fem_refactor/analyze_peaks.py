#!/usr/bin/env python3
"""分析波峰未检测到的原因"""
import json

cache_file = r"D:\ProjectPackage\SimpleFEM\fem_refactor\external\export\roi_analysis_cache_20260112_162856_e11a966f9374.jsonl"

target_frames = [25, 68, 117, 126, 167, 238]
results = []

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if data.get('type') == 'frame':
                frame_idx = data.get('frame_index')
                if frame_idx in target_frames:
                    results.append(data)
        except:
            pass

print("=" * 80)
print("波峰检测失败分析")
print("=" * 80)

for data in results:
    frame_idx = data['frame_index']
    print(f"\n【第 {frame_idx} 帧】")
    print("-" * 80)

    # ROI2 灰度值
    roi2_gray = data.get('roi2_gray', 'N/A')
    print(f"ROI2 灰度值: {roi2_gray}")

    # 阈值信息
    threshold = data.get('threshold', {})
    print(f"\n阈值状态:")
    print(f"  - 固定阈值: {threshold.get('fixed', 'N/A')}")
    print(f"  - 最小阈值: {threshold.get('minimum', 'N/A')}")
    print(f"  - 使用阈值: {threshold.get('used', 'N/A')}")
    print(f"  - 自适应启用: {threshold.get('adaptive_enabled', 'N/A')}")
    print(f"  - 计算的背景均值: {threshold.get('calculated_bg_mean', 'N/A')}")
    print(f"  - 实际背景均值: {threshold.get('bg_mean', 'N/A')}")
    print(f"  - 背景计数: {threshold.get('bg_count', 'N/A')}")
    print(f"  - ⚠️ 保护激活: {threshold.get('protection_active', 'N/A')}")
    print(f"  - 连续低于阈值帧数: {threshold.get('consecutive_below_threshold', 'N/A')}")

    # 检测到的波峰
    peaks = data.get('peaks', {})
    print(f"\n波峰检测:")
    print(f"  - 绿色原始波峰: {len(peaks.get('green_raw', []))} 个")
    print(f"  - 红色原始波峰: {len(peaks.get('red_raw', []))} 个")
    print(f"  - 最终绿色波峰: {len(peaks.get('green', []))} 个")
    print(f"  - 最终红色波峰: {len(peaks.get('red', []))} 个")

    # 检测参数
    detect_params = data.get('detect_params', {})
    print(f"\n检测参数:")
    print(f"  - margin_frames: {detect_params.get('margin_frames', 'N/A')}")
    print(f"  - silence_frames: {detect_params.get('silence_frames', 'N/A')}")
    print(f"  - difference_threshold: {detect_params.get('difference_threshold', 'N/A')}")
    print(f"  - pre_post_avg_frames: {detect_params.get('pre_post_avg_frames', 'N/A')}")
    print(f"  - min_region_length: {detect_params.get('min_region_length', 'N/A')}")

    # ROI3 数据
    roi3 = data.get('roi3', {})
    if roi3:
        print(f"\nROI3 数据:")
        print(f"  - g1: {roi3.get('g1', 'N/A')}")
        print(f"  - g2: {roi3.get('g2', 'N/A')}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

# 统计分析
protection_active_count = sum(1 for r in results if r.get('threshold', {}).get('protection_active', False))
print(f"\n1. 阈值保护状态:")
print(f"   - {protection_active_count}/{len(results)} 帧处于保护激活状态")

if protection_active_count == len(results):
    print(f"   ⚠️ 所有目标帧都处于阈值保护状态！")
    print(f"   这是波峰无法检测的主要原因。")

print(f"\n2. 背景均值计算:")
bg_mean_available = sum(1 for r in results if r.get('threshold', {}).get('calculated_bg_mean') is not None)
print(f"   - {bg_mean_available}/{len(results)} 帧有背景均值")

print(f"\n3. 波峰检测情况:")
total_peaks = sum(len(r.get('peaks', {}).get('green_raw', [])) + len(r.get('peaks', {}).get('red_raw', [])) for r in results)
print(f"   - 总共检测到 {total_peaks} 个原始波峰")
if total_peaks == 0:
    print(f"   ⚠️ 没有检测到任何波峰！")

print(f"\n4. 可能原因:")
print(f"   ✓ 阈值保护机制一直激活（从早期的高值触发后未恢复）")
print(f"   ✓ 背景均值无法更新（因为保护机制阻止）")
print(f"   ✓ 检测阈值可能过高或数据质量不足")
