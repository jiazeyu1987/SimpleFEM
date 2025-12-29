"""
测试ROI3早期波峰覆盖逻辑
"""
import sys
from datetime import datetime

# 导入统计模块
from safe_peak_statistics import SafePeakStatistics

# 创建测试实例
stats = SafePeakStatistics()

# 模拟数据
frame_index = 100  # 当前帧索引
start_frame = 80   # 波峰在curve中的起始位置
end_frame = 90     # 波峰在curve中的结束位置

# 创建测试曲线（150帧）
curve = [40.0] * 150
curve[80:91] = [100, 105, 110, 115, 118, 120, 118, 115, 110, 105, 100]  # 波峰

# 创建ROI3曲线（峰值在早期位置）
roi3_curve = [40.0] * 150
roi3_curve[85] = 130.0  # ROI3峰值在第85帧（全局帧 = 100 - 150 + 1 + 85 = 36）

print("=== 测试ROI3早期波峰覆盖 ===")
print(f"当前帧索引: {frame_index}")
print(f"波峰位置: curve[{start_frame}:{end_frame}]")
print(f"ROI3峰值位置: curve[85] = 130.0")
print(f"ROI3峰值全局帧索引: {frame_index - len(curve) + 1 + 85}")

# 调用_create_peak_data
peak_data = stats._create_peak_data(
    timestamp=datetime.now(),
    frame_index=frame_index,
    peak_type="green",  # 初始判定为绿色
    start_frame=start_frame,
    end_frame=end_frame,
    curve=curve,
    intersection=(300, 400),
    roi2_info={'x1': 280, 'y1': 400, 'x2': 360, 'y2': 500},
    gray_value=100.0,
    difference_threshold=1.5,
    pre_post_avg_frames=5,
    threshold_used=40.0,
    bg_mean=35.0,
    roi3_curve=roi3_curve,
    roi3_override_enabled=True,
    roi3_override_threshold=115.0
)

print(f"\n结果:")
print(f"  初始颜色: green")
print(f"  最终颜色: {peak_data['peak_type']}")
print(f"  ROI3峰值: {peak_data['roi3_peak_max_value']}")
print(f"  ROI3峰值帧: {peak_data['roi3_peak_max_frame']}")
print(f"  ROI3覆盖应用: {peak_data['roi3_override_applied']}")

if peak_data['peak_type'] == 'red' and peak_data['roi3_peak_max_frame'] < 110:
    print("\n✓ 覆盖逻辑生效：绿色 -> 红色")
else:
    print("\n✗ 覆盖逻辑未生效！")
