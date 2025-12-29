"""
重构版本测试脚本

SimpleFEM Refactored Version Test
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactor.config_manager import ConfigManager
from refactor.threshold_protection_manager import ThresholdProtectionManager
from refactor.roi_capture_manager import ROICaptureManager


def test_config_manager():
    """测试配置管理器"""
    print("=" * 60)
    print("测试 ConfigManager")
    print("=" * 60)

    try:
        config = ConfigManager()
        print(f"[OK] 配置加载成功")
        print(f"  - 处理模式: {config.processing_mode}")
        print(f"  - 帧率: {config.frame_rate}")
        print(f"  - 阈值: {config.peak_detection_threshold}")
        print(f"  - ROI1: {config.roi1_config}")
        return True
    except Exception as e:
        print(f"[FAIL] 配置加载失败: {e}")
        return False


def test_threshold_protection_manager():
    """测试阈值保护管理器"""
    print("\n" + "=" * 60)
    print("测试 ThresholdProtectionManager")
    print("=" * 60)

    try:
        config = ConfigManager()
        protection = ThresholdProtectionManager(config)
        print(f"[OK] 阈值保护管理器创建成功")

        # 测试更新
        should_protect, frames = protection.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )
        print(f"  - 保护激活: {should_protect}")
        print(f"  - 距结束帧数: {frames}")
        return True
    except Exception as e:
        print(f"[FAIL] 阈值保护管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_capture_manager():
    """测试ROI捕获管理器"""
    print("\n" + "=" * 60)
    print("测试 ROICaptureManager")
    print("=" * 60)

    try:
        config = ConfigManager()
        capture = ROICaptureManager(config)
        print(f"[OK] ROI捕获管理器创建成功")
        print(f"  - 处理模式: {capture._processing_mode}")
        print(f"  - 屏幕尺寸: {capture._screen_width}x{capture._screen_height}")
        print(f"  - ROI1缓冲区大小: {len(capture.roi1_buffer)}")
        print(f"  - ROI2缓冲区大小: {len(capture.roi2_buffer)}")
        print(f"  - ROI3缓冲区大小: {len(capture.roi3_buffer)}")
        return True
    except Exception as e:
        print(f"[FAIL] ROI捕获管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_green_line_manager():
    """测试绿线检测管理器"""
    print("\n" + "=" * 60)
    print("测试 GreenLineManager")
    print("=" * 60)

    try:
        config = ConfigManager()
        from refactor.green_line_manager import GreenLineManager

        green_line = GreenLineManager(config)
        print(f"[OK] 绿线检测管理器创建成功")
        print(f"  - 防抖动算法: {config.roi2_anti_jitter_algorithm}")
        print(f"  - 移动阈值: {config.roi2_movement_threshold}")
        return True
    except Exception as e:
        print(f"[FAIL] 绿线检测管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("SimpleFEM 重构版本测试")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("ConfigManager", test_config_manager()))
    results.append(("ThresholdProtectionManager", test_threshold_protection_manager()))
    results.append(("ROICaptureManager", test_roi_capture_manager()))
    results.append(("GreenLineManager", test_green_line_manager()))

    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 通过")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
