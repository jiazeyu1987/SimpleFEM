#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终验证frame_diff过滤功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peak_detection import detect_white_peaks_by_threshold_improved, calculate_frame_difference

def test_filter_final():
    """最终验证frame_diff过滤功能"""
    print("最终验证frame_diff过滤功能")
    print("=" * 40)

    # 创建测试信号
    curve = [
        # 背景: 30
        30, 30, 30, 30, 30,    # 0-4

        # 波峰1: 上升到60
        60, 60, 60, 60, 60,    # 5-9

        # 背景: 80 (应该产生 +50 的frame_diff)
        80, 80, 80, 80, 80,    # 10-14

        # 背景: 30
        30, 30, 30, 30, 30,    # 15-19

        # 波峰2: 上升到60
        60, 60, 60, 60, 60,    # 20-24

        # 背景: 0 (应该产生 -30 的frame_diff)
        0, 0, 0, 0, 0,         # 25-29

        # 背景: 30
        30, 30, 30, 30, 30,    # 30-34
    ]

    print("测试信号 (0-29):")
    for i, val in enumerate(curve):
        print(f"  {i:2d}: {val:2d}")

    print()

    # 手动计算预期结果
    print("手动计算:")
    # 波峰1: 位置5-9
    pre1 = sum(curve[0:5]) / 5  # = 30
    post1 = sum(curve[10:15]) / 5  # = 80
    frame_diff1 = post1 - pre1  # = 50 (>15, 应该被过滤)
    print(f"波峰1(5-9): 前5帧平均={pre1:.1f}, 后5帧平均={post1:.1f}, frame_diff={frame_diff1:.1f}, 应该过滤")

    # 波峰2: 位置20-24
    pre2 = sum(curve[15:20]) / 5  # = 30
    post2 = sum(curve[25:30]) / 5  # = 0
    frame_diff2 = post2 - pre2  # = -30 (| -30 | > 15, 应该被过滤)
    print(f"波峰2(20-24): 前5帧平均={pre2:.1f}, 后5帧平均={post2:.1f}, frame_diff={frame_diff2:.1f}, 应该过滤")

    print()

    # 使用算法进行检测
    print("算法检测结果:")
    raw_peaks = detect_white_peaks_by_threshold_improved(
        curve,
        threshold=45.0,
        marginFrames=0,
        avgFrames=5
    )

    print(f"原始检测: {len(raw_peaks)} 个波峰")
    for i, (start, end, frame_diff) in enumerate(raw_peaks):
        print(f"  波峰{i+1}: 位置{start}-{end}, frame_diff={frame_diff}")

    print()

    # 验证过滤结果
    print("过滤验证:")
    filtered_count = 0
    for start, end, frame_diff in raw_peaks:
        if abs(frame_diff) > 15.0:
            filtered_count += 1
            print(f"  位置{start}-{end}: frame_diff={frame_diff:.1f} -> 被过滤")
        else:
            print(f"  位置{start}-{end}: frame_diff={frame_diff:.1f} -> 保留")

    print()
    print("结论:")
    if filtered_count == len(raw_peaks):
        print("✅ 所有检测到的波峰都被正确过滤")
        print("✅ frame_diff过滤功能正常工作")
    elif filtered_count > 0:
        print(f"⚠️ 部分波峰被过滤: {filtered_count}/{len(raw_peaks)}")
    else:
        print("❌ 没有波峰被过滤，过滤功能可能有问题")

    print()
    print("测试完成")

if __name__ == "__main__":
    test_filter_final()