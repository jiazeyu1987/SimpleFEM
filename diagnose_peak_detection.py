"""
诊断波峰检测问题
"""
import json
import sys

# 读取配置文件
with open("simple_fem_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print("=== 配置检查 ===")
peak_conf = config.get("peak_detection", {})
print(f"threshold: {peak_conf.get('threshold')}")
print(f"margin_frames: {peak_conf.get('margin_frames')}")
print(f"margin_frames_enabled: {peak_conf.get('margin_frames_enabled')}")
print(f"silence_frames: {peak_conf.get('silence_frames')}")
print(f"silence_frames_enabled: {peak_conf.get('silence_frames_enabled')}")

roi1_conf = config.get("roi1_peak_detection", {})
print(f"\nROI1 threshold: {roi1_conf.get('threshold')}")
print(f"ROI1 margin_frames: {roi1_conf.get('margin_frames')}")
print(f"ROI1 margin_frames_enabled: {roi1_conf.get('margin_frames_enabled')}")
print(f"ROI1 silence_frames: {roi1_conf.get('silence_frames')}")
print(f"ROI1 silence_frames_enabled: {roi1_conf.get('silence_frames_enabled')}")

print("\n=== 测试波峰检测 ===")
from peak_detection import detect_peaks

# 构造一个简单的测试曲线
test_curve = [
    40, 42, 45, 48, 52,  # 上升
    95, 98, 100, 102, 100, 98, 95,  # 波峰
    55, 50, 45, 42, 40,  # 下降
    40, 40, 40, 40, 40,  # 平稳
    45, 50, 55, 60, 65,
    90, 95, 97, 99, 97, 95, 90,  # 第二个波峰
    65, 60, 55, 50, 45,
    40, 40, 40, 40, 40,
]

print(f"测试曲线长度: {len(test_curve)}")
print(f"测试曲线: {test_curve}")

# 测试1: 不使用任何过滤
print("\n--- 测试1: 不使用过滤 (margin_frames_enabled=False, silence_frames_enabled=False) ---")
green, red = detect_peaks(
    test_curve,
    threshold=50.0,
    marginFrames=5,
    silenceFrames=5,
    margin_frames_enabled=False,
    silence_frames_enabled=False,
    differenceThreshold=2.0,
)
print(f"检测到绿色波峰: {green}")
print(f"检测到红色波峰: {red}")

# 测试2: 使用过滤
print("\n--- 测试2: 使用过滤 (margin_frames_enabled=True, silence_frames_enabled=True) ---")
green2, red2 = detect_peaks(
    test_curve,
    threshold=50.0,
    marginFrames=5,
    silenceFrames=5,
    margin_frames_enabled=True,
    silence_frames_enabled=True,
    differenceThreshold=2.0,
)
print(f"检测到绿色波峰: {green2}")
print(f"检测到红色波峰: {red2}")

# 测试3: 检查函数默认值
print("\n--- 测试3: 使用默认参数 ---")
green3, red3 = detect_peaks(
    test_curve,
    threshold=50.0,
    marginFrames=5,
    silenceFrames=5,
    differenceThreshold=2.0,
)
print(f"检测到绿色波峰: {green3}")
print(f"检测到红色波峰: {red3}")

if not green and not red:
    print("\n❌ 错误: 使用默认参数没有检测到任何波峰!")
    print("   这可能说明函数内部有其他过滤逻辑在起作用")
else:
    print("\n✓ 检测功能正常")
