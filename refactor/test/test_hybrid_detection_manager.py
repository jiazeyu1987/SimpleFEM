"""
HybridDetectionManager 单元测试

SimpleFEM Refactored Version
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager
from refactor.hybrid_detection_manager import HybridDetectionManager


class TestHybridDetectionManager(unittest.TestCase):
    """测试 HybridDetectionManager 混合检测管理器"""

    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
        self.manager = HybridDetectionManager(self.config)

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager._config)
        self.assertIsInstance(self.manager._roi1_peak_counter, dict)
        self.assertEqual(self.manager._peak_id_counter, 0)

    def test_reset(self):
        """测试重置方法"""
        # 添加一些数据
        self.manager._roi1_peak_counter['test_peak'] = 1
        self.manager._peak_id_counter = 10

        # 重置
        self.manager.reset()

        # 验证重置
        self.assertEqual(len(self.manager._roi1_peak_counter), 0)
        self.assertEqual(self.manager._peak_id_counter, 0)

    def test_roi1_peak_counter_property(self):
        """测试 roi1_peak_counter 属性"""
        counter = self.manager.roi1_peak_counter
        self.assertIsInstance(counter, dict)

    def test_detect_hybrid_peaks_disabled(self):
        """测试禁用混合检测"""
        self.config._config['hybrid_detection']['enabled'] = False

        roi1_curve = [50.0] * 100
        roi2_curve = [50.0] * 100

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 0, None
        )

        # 应该返回空结果
        self.assertEqual(len(green), 0)
        self.assertEqual(len(red), 0)
        self.assertEqual(len(info), 0)

    def test_detect_hybrid_peaks_roi1_disabled(self):
        """测试禁用 ROI1 检测"""
        self.config._config['roi1_peak_detection']['enabled'] = False

        roi1_curve = [50.0] * 100
        roi2_curve = [50.0] * 100

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 0, None
        )

        # 应该返回空结果
        self.assertEqual(len(green), 0)
        self.assertEqual(len(red), 0)
        self.assertEqual(len(info), 0)

    def test_detect_hybrid_peaks_no_peaks(self):
        """测试无波峰情况"""
        roi1_curve = [50.0] * 100  # 平坦曲线，无波峰
        roi2_curve = [50.0] * 100

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 0, None
        )

        # 应该没有检测到波峰
        self.assertEqual(len(green), 0)
        self.assertEqual(len(red), 0)
        self.assertEqual(len(info), 0)

    def test_detect_hybrid_peaks_with_peak(self):
        """测试有波峰情况"""
        # 创建有波峰的曲线
        roi1_curve = [50.0] * 40 + [150.0] * 20 + [50.0] * 40
        roi2_curve = [50.0] * 30 + [100.0] * 20 + [60.0] * 50  # 后均值高于前均值

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 100, None
        )

        # 应该检测到波峰
        self.assertGreater(len(green) + len(red), 0)

        # 检查波峰信息
        if len(info) > 0:
            peak_info = info[0]
            self.assertIn('roi1_peak_id', peak_info)
            self.assertIn('start', peak_info)
            self.assertIn('end', peak_info)
            self.assertIn('color', peak_info)
            self.assertIn('detection_method', peak_info)

    def test_determine_roi2_color_green(self):
        """测试 ROI2 颜色判定为绿色"""
        roi2_curve = [50.0] * 20 + [100.0] * 20 + [50.0] * 60

        color = self.manager._determine_roi2_color(roi2_curve, 20, 39)

        # 后均值应该高于前均值，判定为绿色
        self.assertEqual(color, 'green')

    def test_determine_roi2_color_red(self):
        """测试 ROI2 颜色判定为红色"""
        roi2_curve = [100.0] * 20 + [50.0] * 20 + [50.0] * 60

        color = self.manager._determine_roi2_color(roi2_curve, 20, 39)

        # 后均值应该低于前均值，判定为红色
        self.assertEqual(color, 'red')

    def test_determine_roi2_color_in_interval(self):
        """测试详细的颜色判定信息"""
        roi2_curve = [50.0] * 20 + [100.0] * 20 + [50.0] * 60

        info = self.manager.determine_roi2_color_in_interval(roi2_curve, 20, 39)

        # 检查返回的字典结构
        self.assertIn('pre_avg', info)
        self.assertIn('post_avg', info)
        self.assertIn('frame_diff', info)
        self.assertIn('color', info)
        self.assertIn('difference_threshold', info)

        # 检查值
        self.assertGreater(info['post_avg'], info['pre_avg'])
        self.assertEqual(info['color'], 'green')

    def test_calculate_roi2_data_quality(self):
        """测试 ROI2 数据质量计算"""
        roi2_curve = [50.0] * 100

        quality = self.manager.calculate_roi2_data_quality(roi2_curve, 20, 39)

        # 检查返回的字典结构
        self.assertIn('valid_frames', quality)
        self.assertIn('variance', quality)
        self.assertIn('minimum_required_frames', quality)
        self.assertIn('minimum_variance', quality)

        # 检查值
        self.assertEqual(quality['valid_frames'], 100)
        self.assertEqual(quality['variance'], 0.0)  # 平坦曲线方差为0

    def test_calculate_roi2_data_quality_with_variance(self):
        """测试有方差的数据质量计算"""
        roi2_curve = [float(i) for i in range(100)]  # 有方差

        quality = self.manager.calculate_roi2_data_quality(roi2_curve, 20, 39)

        # 方差应该大于0
        self.assertGreater(quality['variance'], 0)

    def test_roi1_peak_id_generation(self):
        """测试 ROI1 波峰 ID 生成"""
        roi1_curve = [50.0] * 40 + [150.0] * 20 + [50.0] * 40
        roi2_curve = [50.0] * 30 + [100.0] * 20 + [60.0] * 50

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 100, None
        )

        # 检查 ID 格式
        if len(info) > 0:
            peak_id = info[0]['roi1_peak_id']
            self.assertIn('roi1_', peak_id)
            # ID 格式应该是 roi1_{frame_index}_{start_position}
            # frame_index=100, 峰从位置40开始
            self.assertEqual(peak_id, 'roi1_100_40')

    def test_roi1_peak_counter_increments(self):
        """测试波峰计数器递增"""
        roi1_curve = [50.0] * 40 + [150.0] * 20 + [50.0] * 40
        roi2_curve = [50.0] * 30 + [100.0] * 20 + [60.0] * 50

        initial_count = self.manager._peak_id_counter

        self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 100, None
        )

        # 计数器应该增加
        self.assertGreater(self.manager._peak_id_counter, initial_count)

    def test_multiple_peaks_detection(self):
        """测试多个波峰检测"""
        # 创建多个波峰
        roi1_curve = (
            [50.0] * 20 +
            [150.0] * 10 +
            [50.0] * 20 +
            [150.0] * 10 +
            [50.0] * 40
        )
        roi2_curve = [50.0] * 100

        green, red, info = self.manager.detect_hybrid_peaks(
            roi1_curve, roi2_curve, 100, None
        )

        # 应该检测到多个波峰
        total_peaks = len(green) + len(red)
        self.assertGreater(total_peaks, 1)


if __name__ == '__main__':
    unittest.main()
