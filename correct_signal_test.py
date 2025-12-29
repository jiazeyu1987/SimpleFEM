#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

使用正确信号设计测试frame_diff过滤功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peak_detection import detect_white_peaks_by_threshold_improved

def test_correct_signal():
    """使用正确信号设计测试frame_diff过滤功能"""
    print("使用正确信号设计测试frame_diff过滤功能")
    print("=" * 40)

    # 创建测试信号 - 在波峰之间确保有足够的低值区域
    curve = []

    # 背景1: 20
    for _ in range(8):
        curve.append(20)

    # 波峰1: 50
    for _ in range(6):
        curve.append(50)

    # 背景2: 30 (应该产生 frame_diff = 30-20 = +10 < 15，应该保留)
    for _ in range(6):
        curve.append(30)

    # 背景3: 20
    for _ in range(8):
        curve.append(20)

    # 波峰2: 50
    for _ in range(6):
        curve.append(50)

    # 背景4: 0 (应该产生 frame_diff = 0-20 = -20 > 15，应该被过滤)
    for _ in range(6):
        curve.append(0)

    # 背景5: 20
    for _ in range(8):
        curve.append(20)

    print("测试信号设计:")
    print("  位置0-7:   背景 (20)")
    print("  位置8-13:  波峰1 (50)")
    print("  位置14-19: 背景 (30)")
    print("  位置20-27: 背景 (20)")
    print("  位置28-33: 波峰2 (50)")
    print("  位置34-39: 背景 (0)")
    print("  位置40-47: 背景 (20)")

    print(f"信号长度: {len(curve)}")
    print()

    # 预期结果
    print("预期结果:")
    print("  波峰1: frame_diff ≈ +10 (应该保留)")
    print("  波峰2: frame_diff ≈ -20 (应该被过滤)")
    print()

    # 使用算法检测
    peaks = detect_white_peaks_by_threshold_improved(
        curve,
        threshold=30.0,  # 设置在背景(20)和波峰(50)之间
        marginFrames=3,
        avgFrames=5
    )

    print(f"算法检测结果: {len(peaks)} 个波峰")
    for i, (start, end, frame_diff) in enumerate(peaks):
        should_filter = abs(frame_diff) > 15.0
        print(f"  波峰{i+1}: 位置{start}-{end}, frame_diff={frame_diff:.1f}, {'应该过滤' if should_filter else '应该保留'}")

    print()

    # 验证过滤效果
    filtered_count = sum(1 for _, _, frame_diff in peaks if abs(frame_diff) > 15.0)
    total_count = len(peaks)

    print("过滤结果验证:")
    print(f"  总波峰数: {total_count}")
    print(f"  被过滤的波峰: {filtered_count}")
    print(f"  保留的波峰: {total_count - filtered_count}")

    if filtered_count > 0:
        print("  过滤功能正在工作")
    else:
        print("  过滤功能未检测到需要过滤的波峰")

    if total_count == 2 and filtered_count == 1:
        print("  完美: 检测到2个波峰，过滤掉1个异常波峰")
    else:
        print(f"  结果: 检测到{total_count}个波峰，过滤{filtered_count}个")

    print()
    print("测试完成")

if __name__ == "__main__":
    test_correct_signal()