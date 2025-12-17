#!/usr/bin/env python3
import json

def quick_diagnosis():
    print("=== Mask生成问题快速诊断 ===")

    # 读取当前配置
    with open('vein_detection_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("1. 当前ROI坐标:")
    roi_config = config.get("roi_capture", {}).get("roi2_config", {})
    print(f"   x1: {roi_config.get('x1')}")
    print(f"   y1: {roi_config.get('y1')}")
    print(f"   x2: {roi_config.get('x2')}")
    print(f"   y2: {roi_config.get('y2')}")
    width = roi_config.get('x2', 0) - roi_config.get('x1', 0)
    height = roi_config.get('y2', 0) - roi_config.get('y1', 0)
    print(f"   尺寸: {width} x {height} 像素")

    print("\n2. 当前检测参数:")
    vein_config = config.get("vein_detection", {})
    print(f"   阈值范围: {vein_config.get('threshold_min')} - {vein_config.get('threshold_max')}")
    print(f"   最小组件大小: {vein_config.get('min_component_size')} 像素")
    print(f"   连通性: {vein_config.get('connectivity')}")

    print("\n3. 问题分析:")

    # 检查阈值范围
    threshold_range = vein_config.get('threshold_max', 0) - vein_config.get('threshold_min', 0)
    if threshold_range < 20:
        print("   [问题] 阈值范围太小！当前范围 =", threshold_range)
        print("   建议: 扩展到 0-30 或 0-50")

    # 检查组件大小
    roi_area = width * height
    min_size = vein_config.get('min_component_size', 0)
    if min_size > roi_area * 0.3:
        print("   [问题] 最小组件尺寸过大！")
        print(f"   当前: {min_size}, ROI面积: {roi_area}")
        print("   建议: 设置为 20-30")

    # 检查ROI位置
    if roi_config.get('x2', 0) > 1920:
        print("   [问题] ROI X坐标可能超出1920宽度视频边界！")
    if roi_config.get('y2', 0) > 1080:
        print("   [问题] ROI Y坐标可能超出1080高度视频边界！")

    print("\n4. 建议的修复配置:")

    # 创建宽松配置
    loose_config = config.copy()
    loose_config['vein_detection']['threshold_min'] = 0
    loose_config['vein_detection']['threshold_max'] = 30
    loose_config['vein_detection']['min_component_size'] = 20

    with open('vein_detection_config_loose.json', 'w', encoding='utf-8') as f:
        json.dump(loose_config, f, indent=2, ensure_ascii=False)

    print("   已创建宽松配置: vein_detection_config_loose.json")
    print("   阈值: 0-30, 最小组件: 20")

    # 创建原始坐标配置
    original_config = config.copy()
    original_config['roi_capture']['roi2_config'] = {
        "x1": 1480, "y1": 480, "x2": 1580, "y2": 580,
        "_comment": "原始ROI2坐标"
    }
    original_config['vein_detection']['threshold_min'] = 0
    original_config['vein_detection']['threshold_max'] = 30
    original_config['vein_detection']['min_component_size'] = 20

    with open('vein_detection_config_original.json', 'w', encoding='utf-8') as f:
        json.dump(original_config, f, indent=2, ensure_ascii=False)

    print("   已创建原始坐标配置: vein_detection_config_original.json")

    print("\n5. 测试方法:")
    print("   方法1: python auto_vein_detector.py --config vein_detection_config_loose.json")
    print("   方法2: python auto_vein_detector.py --config vein_detection_config_original.json")
    print("   方法3: 检查视频帧内容，确保ROI区域确实包含静脉")

if __name__ == "__main__":
    quick_diagnosis()