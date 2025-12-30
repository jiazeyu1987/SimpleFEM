#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试极端frame_diff值的过滤功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peak_detection import detect_white_peaks_by_threshold_improved

def create_extreme_test_curve():
    """创建包含极端frame_diff值的测试信号"""
    curve = []

    # 背景信号
    background = 40
    for _ in range(10):
        curve.append(background)

    # 第一个波峰：正常变化
    for i in range(13):
        curve.append(background + 20 + i)

    # 恢复背景
    for _ in range(5):
        curve.append(background)

    # 第二个波峰：极端正向变化 (frame_diff > 15)
    for i in range(10):
        curve.append(background + 150 + i * 10)  # 从190到280

    # 恢复背景
    for _ in range(5):
        curve.append(background)

    # 第三个波峰：极端负向变化 (frame_diff < -15)
    for i in range(8):
        curve.append(background - 100 - i * 8)  # 从-60到-116

    # 恢复背景
    for _ in range(5):
        curve.append(background)

    # 第四个波峰：正常变化
    for i in range(12):
        curve.append(background + 15 + i)

    # 结束背景
    for _ in range(5):
        curve.append(background)

    return curve

def test_extreme_frame_diff_filter():
    """测试极端frame_diff过滤功能"""
    print("测试极端frame_diff错误数据过滤功能")
    print("=" * 40)

    test_curve = create_extreme_test_curve()

    print(f"信号长度: {len(test_curve)}")
    print(f"信号范围: {min(test_curve)} ~ {max(test_curve)}")
    print()

    # 预期的波峰位置（手动估算）
    expected_peaks = [
        (10, 22),   # 第一个波峰（正常）
        (28, 37),   # 第二个波峰（极端正向，应该被过滤）
        (43, 50),   # 第三个波峰（极端负向，应该被过滤）
        (56, 67),   # 第四个波峰（正常）
    ]

    print("预期的波峰位置:")
    for i, (start, end) in enumerate(expected_peaks):
        # 手动计算frame_diff
        if start > 0 and end < len(test_curve):
            pre_avg = sum(test_curve[start-5:start]) / 5 if start >= 5 else test_curve[start]
            post_avg = sum(test_curve[end+1:end+6]) / 5 if end+5 < len(test_curve) else test_curve[end]
            frame_diff = post_avg - pre_avg
            should_filter = abs(frame_diff) > 15.0
            print(f"  波峰{i+1}: 位置{start}-{end}, frame_diff≈{frame_diff:.1f}, {'应该过滤' if should_filter else '应该保留'}")
    print()

    # 使用改进的峰值检测
    peaks_with_diff = detect_white_peaks_by_threshold_improved(
        test_curve,
        threshold=50.0,
        marginFrames=5,
        avgFrames=5
    )

    print(f"实际检测结果: {len(peaks_with_diff)} 个波峰")

    for i, (start, end, frame_diff) in enumerate(peaks_with_diff):
        print(f"  波峰: 位置{start}-{end}, frame_diff={frame_diff:.1f}")

    print()
    print("过滤效果验证:")

    # 统计应该被过滤的波峰
    filtered_count = 0
    for _, _, frame_diff in peaks_with_diff:
        if abs(frame_diff) > 15.0:
            filtered_count += 1

    print(f"检测到 |frame_diff| > 15 的波峰: {filtered_count} 个")
    print(f"总检测波峰: {len(peaks_with_diff)} 个")

    if len(peaks_with_diff) < len(expected_peaks):
        print("✅ 过滤功能生效 - 一些波峰被正确过滤")
    else:
        print("❌ 过滤功能未生效 - 所有波峰都被检测到")

if __name__ == "__main__":
    test_extreme_frame_diff_filter()