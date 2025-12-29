"""
重构版本集成测试 - 测试所有功能

SimpleFEM Refactored Version Integration Test
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactor.config_manager import ConfigManager
from refactor.threshold_protection_manager import ThresholdProtectionManager
from refactor.roi_capture_manager import ROICaptureManager
from refactor.green_line_manager import GreenLineManager
from refactor.hybrid_detection_manager import HybridDetectionManager
from refactor.roi3_statistics import ROI3Statistics


def test_roi3_statistics():
    """测试ROI3统计计算"""
    print("=" * 60)
    print("测试 ROI3Statistics")
    print("=" * 60)

    try:
        from PIL import Image
        import numpy as np

        # 创建测试图像
        test_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        test_image = Image.fromarray(test_array)

        # 计算统计值
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(test_image)
        column_diff = ROI3Statistics.compute_column_mean_diff(test_image)
        normalized = ROI3Statistics.compute_normalized_80_160(test_image)
        all_stats = ROI3Statistics.compute_all(test_image)

        print(f"[OK] ROI3统计计算成功")
        print(f"  - G1百分比: {g1:.2f}%")
        print(f"  - G2百分比: {g2:.2f}%")
        print(f"  - 列差值: {column_diff:.2f}")
        print(f"  - 归一化值: {normalized:.2f}")
        print(f"  - 完整统计: {all_stats}")
        return True
    except Exception as e:
        print(f"[FAIL] ROI3统计测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_detection():
    """测试混合检测"""
    print("\n" + "=" * 60)
    print("测试 HybridDetectionManager")
    print("=" * 60)

    try:
        config = ConfigManager()
        hybrid = HybridDetectionManager(config)
        print(f"[OK] 混合检测管理器创建成功")

        # 创建测试曲线
        roi1_curve = [50.0] * 50 + [100.0] * 20 + [50.0] * 30  # 中间有波峰
        roi2_curve = [50.0] * 50 + [80.0] * 20 + [50.0] * 30

        # 执行混合检测
        green_peaks, red_peaks, hybrid_info = hybrid.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 0, (100, 100)
        )

        print(f"  - 检测到绿色波峰: {len(green_peaks)}")
        print(f"  - 检测到红色波峰: {len(red_peaks)}")
        print(f"  - 混合波峰信息: {len(hybrid_info)}")
        return True
    except Exception as e:
        print(f"[FAIL] 混合检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi2_color_determination():
    """测试ROI2颜色判定"""
    print("\n" + "=" * 60)
    print("测试 ROI2 颜色判定")
    print("=" * 60)

    try:
        config = ConfigManager()
        hybrid = HybridDetectionManager(config)

        # 创建测试曲线 - post > pre 应该是绿色
        roi2_curve = [50.0] * 50 + [80.0] * 20 + [50.0] * 30

        # 判定颜色
        color_info = hybrid.determine_roi2_color_in_interval(roi2_curve, 50, 70)

        print(f"[OK] ROI2颜色判定成功")
        print(f"  - 前均值: {color_info['pre_avg']:.2f}")
        print(f"  - 后均值: {color_info['post_avg']:.2f}")
        print(f"  - 帧差: {color_info['frame_diff']:.2f}")
        print(f"  - 判定颜色: {color_info['color']}")
        print(f"  - 差值阈值: {color_info['difference_threshold']}")
        return True
    except Exception as e:
        print(f"[FAIL] ROI2颜色判定测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality():
    """测试数据质量计算"""
    print("\n" + "=" * 60)
    print("测试 ROI2 数据质量计算")
    print("=" * 60)

    try:
        config = ConfigManager()
        hybrid = HybridDetectionManager(config)

        # 创建测试曲线
        roi2_curve = [50.0 + i * 0.1 for i in range(100)]

        # 计算数据质量
        quality = hybrid.calculate_roi2_data_quality(roi2_curve, 50, 70)

        print(f"[OK] 数据质量计算成功")
        print(f"  - 有效帧数: {quality['valid_frames']}")
        print(f"  - 方差: {quality['variance']:.2f}")
        print(f"  - 最小需要帧数: {quality['minimum_required_frames']}")
        print(f"  - 最小方差: {quality['minimum_variance']}")
        return True
    except Exception as e:
        print(f"[FAIL] 数据质量计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("SimpleFEM 重构版本集成测试")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("ROI3Statistics", test_roi3_statistics()))
    results.append(("HybridDetectionManager", test_hybrid_detection()))
    results.append(("ROI2 Color Determination", test_roi2_color_determination()))
    results.append(("Data Quality Calculation", test_data_quality()))

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
