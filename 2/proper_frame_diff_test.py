#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试正确的frame_diff过滤功能
确保波峰前后有明显的差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peak_detection import detect_white_peaks_by_threshold_improved

def create_proper_test_curve():
    """创建能产生正确frame_diff的测试信号"""
    curve = []

    # 低背景信号
    low_background = 40
    for _ in range(8):
        curve.append(low_background)

    # 第一个波峰：正常情况，从低背景到高背景再到低背景
    # 波峰前：低背景
    for _ in range(5):
        curve.append(low_background)

    # 波峰：高信号
    high_signal = 70
    for _ in range(8):
        curve.append(high_signal)

    # 波峰后：高背景（产生正frame_diff）
    high_background = 60
    for _ in range(5):
        curve.append(high_background)

    # 第二个波峰：异常情况，从高背景到极高信号再到低背景
    # 波峰前：高背景
    for _ in range(5):
        curve.append(high_background)

    # 波峰：极高信号（会产生大的负frame_diff）
    extreme_signal = 200
    for _ in range(8):
        curve.append(extreme_signal)

    # 波峰后：低背景（产生很大的负frame_diff）
    for _ in range(5):
        curve.append(low_background)

    # 第三个波峰：正常情况，从高背景到更高背景再到高背景
    # 波峰前：高背景
    for _ in range(5):
        curve.append(high_background)

    # 波峰：很高信号
    very_high_signal = 90
    for _ in range(8):
        curve.append(very_high_signal)

    # 波峰后：更高背景（产生正frame_diff）
    higher_background = 80
    for _ in range(5):
        curve.append(higher_background)

    return curve

def test_proper_frame_diff_filter():
    """测试正确的frame_diff过滤功能"""
    print("测试正确的frame_diff过滤功能")
    print("=" * 40)

    test_curve = create_proper_test_curve()

    print(f"信号长度: {len(test_curve)}")
    print(f"信号范围: {min(test_curve)} ~ {max(test_curve)}")
    print()

    # 手动计算预期的frame_diff
    print("手动计算预期frame_diff:")

    # 波峰1: low_background(40) -> high_signal(70) -> high_background(60)
    # frame_diff = 60 - 40 = +20 (应该被过滤，因为>15)

    # 波峰2: high_background(60) -> extreme_signal(200) -> low_background(40)
    # frame_diff = 40 - 60 = -20 (应该被过滤，因为| -20 | > 15)

    # 波峰3: high_background(60) -> very_high_signal(90) -> higher_background(80)
    # frame_diff = 80 - 60 = +20 (应该被过滤，因为>15)

    expected_frame_diffs = [+20, -20, +20]
    for i, expected_diff in enumerate(expected_frame_diffs):
        should_filter = abs(expected_diff) > 15.0
        print(f"  波峰{i+1}: 预期frame_diff={expected_diff}, {'应该过滤' if should_filter else '应该保留'}")

    print()

    # 使用改进的峰值检测
    peaks_with_diff = detect_white_peaks_by_threshold_improved(
        test_curve,
        threshold=50.0,  # 设置为中等阈值，确保所有波峰都能被检测到
        marginFrames=5,
        avgFrames=5
    )

    print(f"实际检测结果: {len(peaks_with_diff)} 个波峰")

    filtered_count = 0
    for i, (start, end, frame_diff) in enumerate(peaks_with_diff):
        should_filter = abs(frame_diff) > 15.0
        if should_filter:
            filtered_count += 1
        print(f"  波峰{i+1}: 位置{start}-{end}, frame_diff={frame_diff:.1f}, {'应该过滤' if should_filter else '应该保留'}")

    print()
    print("过滤结果统计:")
    print(f"  检测到 |frame_diff| > 15 的波峰: {filtered_count} 个")
    print(f"  检测到的总波峰: {len(peaks_with_diff)} 个")

    # 验证过滤效果
    expected_filtered = sum(1 for diff in expected_frame_diffs if abs(diff) > 15.0)

    print()
    print("过滤效果验证:")
    if filtered_count == len(peaks_with_diff):
        print("  所有检测到的波峰都应该被过滤 - OK")
    else:
        print("  只有部分波峰被过滤")

    if filtered_count >= 2:  # 至少过滤掉一些波峰
        print("  过滤功能正在工作")
    else:
        print("  过滤功能可能未完全生效")

if __name__ == "__main__":
    test_proper_frame_diff_filter()