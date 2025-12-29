#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 ROI3 peak max frame < 110 强制设置为 RED 的规则
"""

from datetime import datetime
from safe_peak_statistics import SafePeakStatistics

def test_roi3_frame_rule():
    """测试 ROI3 帧索引规则"""

    stats = SafePeakStatistics(video_name="test_roi3_frame_rule")

    # 模拟数据
    timestamp = datetime.now()
    frame_index = 100
    intersection = (1600, 500)
    roi2_info = {"x1": 1550, "y1": 450, "x2": 1650, "y2": 550}

    # 测试场景1：ROI3 peak max frame = 105 (< 110)，应该被强制为 RED
    print("\n=== 测试场景1: ROI3 peak max frame = 105 (< 110) ===")

    # 创建一个模拟的 curve
    curve = [30.0] * 50 + [100.0] * 10 + [30.0] * 40  # 50-60是波峰区域

    # ROI3 curve
    roi3_curve = [40.0] * 50 + [120.0] * 10 + [40.0] * 40  # 峰值在50-60，对应全局帧105

    peak_data = stats._create_peak_data(
        timestamp=timestamp,
        frame_index=frame_index,
        peak_type="green",  # 初始为 GREEN
        start_frame=50,
        end_frame=60,
        curve=curve,
        intersection=intersection,
        roi2_info=roi2_info,
        gray_value=100.0,
        difference_threshold=1.5,
        pre_post_avg_frames=5,
        threshold_used=40.0,
        bg_mean=30.0,
        roi3_curve=roi3_curve,
        roi3_override_enabled=True,
        roi3_override_threshold=115.0
    )

    print(f"初始 peak_type: green")
    print(f"ROI3 peak max value: {peak_data['roi3_peak_max_value']}")
    print(f"ROI3 peak max frame: {peak_data['roi3_peak_max_frame']}")
    print(f"最终 peak_type: {peak_data['peak_type']}")
    print(f"ROI3 override applied: {peak_data['roi3_override_applied']}")

    assert peak_data['peak_type'] == "red", f"期望 peak_type='red', 实际得到 '{peak_data['peak_type']}'"
    print("[PASS] 测试通过: ROI3 peak max frame < 110 时强制为 RED")

    # 测试场景2：ROI3 peak max frame = 150 (>= 110)，不受新规则影响
    print("\n=== 测试场景2: ROI3 peak max frame = 150 (>= 110) ===")

    frame_index = 200  # 调整当前帧索引
    roi3_curve2 = [40.0] * 90 + [120.0] * 10 + [40.0] * 10  # 峰值在90-100，对应全局帧150

    peak_data2 = stats._create_peak_data(
        timestamp=timestamp,
        frame_index=frame_index,
        peak_type="red",  # 初始为 RED
        start_frame=90,
        end_frame=100,
        curve=curve,
        intersection=intersection,
        roi2_info=roi2_info,
        gray_value=100.0,
        difference_threshold=1.5,
        pre_post_avg_frames=5,
        threshold_used=40.0,
        bg_mean=30.0,
        roi3_curve=roi3_curve2,
        roi3_override_enabled=True,
        roi3_override_threshold=115.0
    )

    print(f"初始 peak_type: red")
    print(f"ROI3 peak max value: {peak_data2['roi3_peak_max_value']}")
    print(f"ROI3 peak max frame: {peak_data2['roi3_peak_max_frame']}")
    print(f"最终 peak_type: {peak_data2['peak_type']}")
    print(f"ROI3 override applied: {peak_data2['roi3_override_applied']}")

    # 这里应该应用原有的 ROI3 override (RED -> GREEN)，因为 peak max value = 120 > 115
    assert peak_data2['peak_type'] == "green", f"期望 peak_type='green' (原有ROI3 override), 实际得到 '{peak_data2['peak_type']}'"
    print("[PASS] 测试通过: ROI3 peak max frame >= 110 时，原有ROI3 override规则生效")

    # 测试场景3：ROI3 peak max frame = 105 (< 110)，即使原有规则要改成 GREEN，新规则也会强制为 RED
    print("\n=== 测试场景3: ROI3 peak max frame = 105 且 peak max value = 120 (> threshold) ===")

    frame_index = 100
    roi3_curve3 = [40.0] * 50 + [120.0] * 10 + [40.0] * 40  # 峰值在50-60，对应全局帧105

    peak_data3 = stats._create_peak_data(
        timestamp=timestamp,
        frame_index=frame_index,
        peak_type="red",  # 初始为 RED
        start_frame=50,
        end_frame=60,
        curve=curve,
        intersection=intersection,
        roi2_info=roi2_info,
        gray_value=100.0,
        difference_threshold=1.5,
        pre_post_avg_frames=5,
        threshold_used=40.0,
        bg_mean=30.0,
        roi3_curve=roi3_curve3,
        roi3_override_enabled=True,
        roi3_override_threshold=115.0
    )

    print(f"初始 peak_type: red")
    print(f"ROI3 peak max value: {peak_data3['roi3_peak_max_value']} (>= 115)")
    print(f"ROI3 peak max frame: {peak_data3['roi3_peak_max_frame']} (< 110)")
    print(f"最终 peak_type: {peak_data3['peak_type']}")
    print(f"ROI3 override applied: {peak_data3['roi3_override_applied']}")

    # 新规则优先级更高：即使 peak max value = 120 > 115，但 frame = 105 < 110，应该是 RED
    assert peak_data3['peak_type'] == "red", f"期望 peak_type='red' (新规则优先), 实际得到 '{peak_data3['peak_type']}'"
    print("[PASS] 测试通过: 新规则优先级更高，ROI3 peak max frame < 110 时强制为 RED")

    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)

if __name__ == "__main__":
    test_roi3_frame_rule()
