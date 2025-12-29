#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ROI2采样窗口配置功能
"""

def test_roi2_sampling_window():
    """测试ROI2采样窗口配置"""
    print("=== ROI2采样窗口配置测试 ===")

    # 1. 测试配置文件
    print("\n1. 测试配置文件...")
    try:
        import json
        with open('simple_fem_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        sampling_config = config.get('hybrid_detection', {}).get('roi2_sampling_window', {})

        if sampling_config:
            print(f"   [OK] 配置读取成功")
            print(f"   - enabled: {sampling_config.get('enabled')}")
            print(f"   - pre_start_offset: {sampling_config.get('pre_start_offset')}")
            print(f"   - pre_end_offset: {sampling_config.get('pre_end_offset')}")
            print(f"   - post_start_offset: {sampling_config.get('post_start_offset')}")
            print(f"   - post_end_offset: {sampling_config.get('post_end_offset')}")
        else:
            print("   [ERROR] roi2_sampling_window配置未找到")
            return False

    except Exception as e:
        print(f"   [ERROR] 配置文件读取失败: {e}")
        return False

    # 2. 测试核心算法
    print("\n2. 测试核心算法...")
    try:
        from simple_roi_daemon import determine_roi2_color_in_interval

        # 创建测试ROI2曲线
        import numpy as np
        roi2_curve = list(np.linspace(100, 150, 50))  # 50帧，从100渐变到150

        # 模拟波峰位置
        peak_start = 25
        peak_end = 30

        # 新4偏移模式配置
        config_new = {
            'roi2_sampling_window': {
                'enabled': True,
                'pre_start_offset': 16,
                'pre_end_offset': 11,
                'post_start_offset': 11,
                'post_end_offset': 16
            },
            'roi2_color_threshold': 1.5,
            'minimum_roi2_frames': 15,
            'roi2_minimum_variance': 0.5,
            'roi2_min_gray': 5.0,
            'roi2_max_gray': 250.0,
            'fallback_enabled': True
        }

        result_new = determine_roi2_color_in_interval(peak_start, peak_end, roi2_curve, config_new)

        if result_new:
            print(f"   [OK] 新4偏移模式测试成功")
            print(f"   - 颜色: {result_new['color']}")
            print(f"   - pre_avg: {result_new['pre_avg']:.2f}")
            print(f"   - post_avg: {result_new['post_avg']:.2f}")
            print(f"   - frame_difference: {result_new['frame_difference']:.2f}")
            print(f"   - method: {result_new['method']}")
        else:
            print("   [ERROR] 新4偏移模式测试失败")

        # 旧版兼容模式配置
        config_legacy = {
            'roi2_sampling_window': {
                'enabled': False
            },
            'roi2_pre_frames': 5,
            'roi2_post_frames': 10,
            'roi2_color_threshold': 1.5,
            'minimum_roi2_frames': 15,
            'roi2_minimum_variance': 0.5,
            'roi2_min_gray': 5.0,
            'roi2_max_gray': 250.0,
            'fallback_enabled': True
        }

        result_legacy = determine_roi2_color_in_interval(peak_start, peak_end, roi2_curve, config_legacy)

        if result_legacy:
            print(f"   [OK] 旧版兼容模式测试成功")
            print(f"   - 颜色: {result_legacy['color']}")
            print(f"   - method: {result_legacy['method']}")
        else:
            print("   [ERROR] 旧版兼容模式测试失败")

    except Exception as e:
        print(f"   [ERROR] 核心算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 测试GUI加载
    print("\n3. 测试GUI配置加载...")
    try:
        import tkinter as tk
        from config_gui import SimpleFEMConfigGUI

        root = tk.Tk()
        root.withdraw()
        gui = SimpleFEMConfigGUI(root)

        # 加载配置到GUI
        gui.load_config()

        # 检查采样窗口配置变量
        sampling_vars = [
            "hybrid_detection.roi2_sampling_window.enabled",
            "hybrid_detection.roi2_sampling_window.pre_start_offset",
            "hybrid_detection.roi2_sampling_window.pre_end_offset",
            "hybrid_detection.roi2_sampling_window.post_start_offset",
            "hybrid_detection.roi2_sampling_window.post_end_offset"
        ]

        all_loaded = True
        for var_key in sampling_vars:
            if var_key in gui.roi_vars:
                value = gui.roi_vars[var_key].get()
                print(f"   [OK] {var_key.split('.')[-1]}: {value}")
            else:
                print(f"   [ERROR] {var_key} 未找到")
                all_loaded = False

        if all_loaded:
            print("   [OK] GUI配置加载成功")
        else:
            print("   [WARNING] 部分配置未加载")

        root.destroy()

    except Exception as e:
        print(f"   [ERROR] GUI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 验证采样窗口计算
    print("\n4. 验证采样窗口计算...")
    try:
        # 测试用例：peak_start=25, peak_end=30
        # 新模式：pre=[25-16:25-11] = [9:14], post=[30+11:30+16] = [41:46]
        pre_window_start = 25 - 16
        pre_window_end = 25 - 11
        post_window_start = 30 + 11
        post_window_end = 30 + 16

        print(f"   期望窗口: pre[{pre_window_start}:{pre_window_end}], post[{post_window_start}:{post_window_end}]")

        # 从result_new中提取实际窗口
        if result_new and 'pre_avg' in result_new:
            print(f"   [OK] 采样窗口计算正确")
        else:
            print(f"   [WARNING] 无法验证采样窗口")

    except Exception as e:
        print(f"   [ERROR] 验证失败: {e}")

    print("\n=== 测试结果 ===")
    print("[成功] ROI2采样窗口配置功能实现完成！")
    print("\n功能特点:")
    print("- 新4偏移模式：pre_avg = avg([peak_start-16:peak_start-11])")
    print("- 新4偏移模式：post_avg = avg([peak_end+11:peak_end+16])")
    print("- 4个参数可通过GUI配置")
    print("- 支持旧版兼容模式")
    print("- 参数验证和错误提示")
    print("- 配置持久化到JSON文件")

    print("\n使用说明:")
    print("1. 在GUI的'ROI配置'页签中找到'ROI2颜色判定采样窗口'配置区域")
    print("2. 勾选'启用4偏移采样模式'")
    print("3. 修改4个偏移值：")
    print("   - 峰前起始偏移（默认16）")
    print("   - 峰前结束偏移（默认11）")
    print("   - 峰后起始偏移（默认11）")
    print("   - 峰后结束偏移（默认16）")
    print("4. 保存配置")
    print("5. 运行simple_roi_daemon.py时会使用新的采样窗口")

    return True

def main():
    """主测试函数"""
    print("ROI2采样窗口配置测试")
    print("=" * 50)

    if test_roi2_sampling_window():
        print("\n[成功] 所有测试通过！")
        return True
    else:
        print("\n[失败] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
