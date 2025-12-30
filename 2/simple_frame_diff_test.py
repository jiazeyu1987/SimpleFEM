#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试frame_diff过滤功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peak_detection import detect_white_peaks_by_threshold_improved
from peak_detection import classify_peak_color

def test_frame_diff_filter():
    """测试frame_diff过滤功能"""
    print("测试frame_diff错误数据过滤功能")
    print("=" * 40)

    # 创建包含正常和异常frame_diff的测试信号
    test_curve = [
        40, 41, 42, 40, 39, 41, 42, 40, 41, 43,  # 背景
        # 第一个波峰：正常变化
        50, 52, 55, 58, 60, 62, 65, 63, 60, 58, 55, 52, 50,
        # 背景
        45, 42, 41, 40, 39, 41, 42, 43, 42, 40,
        # 第二个波峰：异常变化 (frame_diff > 15)
        50, 100, 150, 180, 200, 180, 150, 100, 50, 45,
        # 背景
        42, 40, 41, 43, 42, 40, 39, 41, 42, 40,
        # 第三个波峰：正常变化
        55, 58, 60, 62, 63, 64, 62, 60, 58, 56, 54, 52, 50,
    ]

    print(f"信号长度: {len(test_curve)}")
    print(f"信号范围: {min(test_curve)} ~ {max(test_curve)}")
    print()

    # 使用改进的峰值检测函数（包含frame_diff过滤）
    peaks_with_diff = detect_white_peaks_by_threshold_improved(
        test_curve,
        threshold=45.0,
        marginFrames=5,
        avgFrames=5
    )

    print(f"检测到 {len(peaks_with_diff)} 个波峰（应用过滤后）:")

    for i, (start, end, frame_diff) in enumerate(peaks_with_diff):
        color = classify_peak_color(frame_diff, 1.1)
        print(f"  波峰{i+1}: 位置{start}-{end}, frame_diff={frame_diff:.1f}, 颜色={color}")

    print()

    # 手动计算预期的frame_diff值进行对比
    print("手动计算验证:")
    expected_peaks = [
        (10, 22),   # 第一个波峰
        (34, 43),   # 第二个波峰（异常）
        (54, 67),   # 第三个波峰
    ]

    for i, (start, end) in enumerate(expected_peaks):
        if end < len(test_curve):
            # 简化的frame_diff计算
            pre_avg = sum(test_curve[max(0, start-5):start]) / min(5, start)
            post_avg = sum(test_curve[end+1:min(len(test_curve), end+6)]) / min(5, len(test_curve)-end-1)
            frame_diff = post_avg - pre_avg
            should_filter = abs(frame_diff) > 15.0

            print(f"  位置{start}-{end}: frame_diff={frame_diff:.1f}, "
                  f"{'应该被过滤' if should_filter else '应该保留'}")

    print()
    print("结论:")
    if len(peaks_with_diff) < 3:  # 如果异常波峰被过滤
        print("✅ frame_diff过滤功能正常工作")
    else:
        print("❌ frame_diff过滤功能未生效")

if __name__ == "__main__":
    test_frame_diff_filter()