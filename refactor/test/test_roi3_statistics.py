"""
ROI3Statistics 单元测试

SimpleFEM Refactored Version
"""

import unittest
import os
import sys
from PIL import Image
import numpy as np

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.roi3_statistics import ROI3Statistics


class TestROI3Statistics(unittest.TestCase):
    """测试 ROI3Statistics ROI3统计计算器"""

    def setUp(self):
        """测试前准备"""
        # 创建测试用的 ROI3 图像
        self.test_image_80_255 = self._create_test_image(
            high_value=200, low_value=50, high_ratio=0.8
        )
        self.test_image_150_255 = self._create_test_image(
            high_value=200, low_value=50, high_ratio=0.3
        )
        self.test_image_column_diff = self._create_column_diff_image()

    def _create_test_image(self, high_value=200, low_value=50, high_ratio=0.5):
        """创建测试图像"""
        # 创建 100x100 的灰度图像
        width, height = 100, 100

        # 计算高值像素数量
        total_pixels = width * height
        high_pixels = int(total_pixels * high_ratio)

        # 创建数组
        array = np.full((height, width), low_value, dtype=np.uint8)

        # 随机设置高值像素
        indices = np.random.choice(total_pixels, high_pixels, replace=False)
        flat_array = array.flatten()
        flat_array[indices] = high_value
        array = flat_array.reshape((height, width))

        return Image.fromarray(array)

    def _create_column_diff_image(self):
        """创建有列差异的测试图像"""
        width, height = 100, 100

        # 左半部分低值，右半部分高值
        array = np.zeros((height, width), dtype=np.uint8)
        array[:, :50] = 50   # 左半部分
        array[:, 50:] = 200  # 右半部分

        return Image.fromarray(array)

    def test_compute_g1_g2_ranges(self):
        """测试 G1/G2 范围计算"""
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(self.test_image_80_255)

        # G1: [80, 255] 范围应该包含大部分像素
        self.assertGreater(g1, 70)  # 至少 70%
        self.assertLessEqual(g1, 100)

        # G2: [150, 255] 范围应该包含较少像素
        self.assertGreaterEqual(g2, 0)
        self.assertLessEqual(g2, 100)

    def test_compute_g1_g2_ranges_values(self):
        """测试 G1/G2 计算值的正确性"""
        # 创建全部为 200 的图像
        all_high = Image.fromarray(np.full((50, 50), 200, dtype=np.uint8))
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(all_high)

        # 应该 100% 在范围内
        self.assertEqual(g1, 100.0)
        self.assertEqual(g2, 100.0)

        # 创建全部为 50 的图像
        all_low = Image.fromarray(np.full((50, 50), 50, dtype=np.uint8))
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(all_low)

        # G1 应该 0%（低于 80）
        self.assertEqual(g1, 0.0)
        # G2 应该 0%（低于 150）
        self.assertEqual(g2, 0.0)

    def test_compute_column_mean_diff(self):
        """测试列灰度差值计算"""
        diff = ROI3Statistics.compute_column_mean_diff(self.test_image_column_diff)

        # 左右差异应该很大
        self.assertGreater(diff, 100)

    def test_compute_column_mean_diff_uniform(self):
        """测试均匀图像的列差值"""
        # 创建均匀图像
        uniform = Image.fromarray(np.full((100, 100), 128, dtype=np.uint8))
        diff = ROI3Statistics.compute_column_mean_diff(uniform)

        # 差值应该接近 0
        self.assertAlmostEqual(diff, 0.0, places=1)

    def test_compute_normalized_80_160(self):
        """测试归一化灰度值计算"""
        normalized = ROI3Statistics.compute_normalized_80_160(self.test_image_80_255)

        # 归一化值应该在 [0, 160] 范围内
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 160.0)

    def test_compute_normalized_80_160_clipping(self):
        """测试归一化时的裁剪"""
        # 创建包含超过 160 值的图像
        high_values = Image.fromarray(np.full((50, 50), 250, dtype=np.uint8))
        normalized = ROI3Statistics.compute_normalized_80_160(high_values)

        # 应该被裁剪到 160
        self.assertAlmostEqual(normalized, 160.0, places=1)

    def test_compute_all(self):
        """测试计算所有统计值"""
        all_stats = ROI3Statistics.compute_all(self.test_image_80_255)

        # 检查返回的字典包含所有键
        self.assertIn('g1_percent', all_stats)
        self.assertIn('g2_percent', all_stats)
        self.assertIn('column_diff', all_stats)
        self.assertIn('normalized_80_160', all_stats)

        # 检查值的类型
        self.assertIsInstance(all_stats['g1_percent'], float)
        self.assertIsInstance(all_stats['g2_percent'], float)
        self.assertIsInstance(all_stats['column_diff'], float)
        self.assertIsInstance(all_stats['normalized_80_160'], float)

    def test_compute_all_consistency(self):
        """测试 compute_all 结果与单独计算一致"""
        all_stats = ROI3Statistics.compute_all(self.test_image_80_255)

        # 单独计算
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(self.test_image_80_255)
        diff = ROI3Statistics.compute_column_mean_diff(self.test_image_80_255)
        normalized = ROI3Statistics.compute_normalized_80_160(self.test_image_80_255)

        # 应该一致
        self.assertAlmostEqual(all_stats['g1_percent'], g1, places=5)
        self.assertAlmostEqual(all_stats['g2_percent'], g2, places=5)
        self.assertAlmostEqual(all_stats['column_diff'], diff, places=5)
        self.assertAlmostEqual(all_stats['normalized_80_160'], normalized, places=5)

    def test_static_methods(self):
        """测试所有方法都是静态方法"""
        # 不需要实例化就可以调用
        self.assertTrue(callable(ROI3Statistics.compute_g1_g2_ranges))
        self.assertTrue(callable(ROI3Statistics.compute_column_mean_diff))
        self.assertTrue(callable(ROI3Statistics.compute_normalized_80_160))
        self.assertTrue(callable(ROI3Statistics.compute_all))

    def test_g1_boundary_values(self):
        """测试 G1 边界值"""
        # 测试 80 正好在边界上
        image = Image.fromarray(np.full((10, 10), 80, dtype=np.uint8))
        g1, _ = ROI3Statistics.compute_g1_g2_ranges(image)
        self.assertEqual(g1, 100.0)  # 80 应该被包含

        # 测试 79 刚好在边界外
        image = Image.fromarray(np.full((10, 10), 79, dtype=np.uint8))
        g1, _ = ROI3Statistics.compute_g1_g2_ranges(image)
        self.assertEqual(g1, 0.0)  # 79 应该不被包含

    def test_g2_boundary_values(self):
        """测试 G2 边界值"""
        # 测试 150 正好在边界上
        image = Image.fromarray(np.full((10, 10), 150, dtype=np.uint8))
        _, g2 = ROI3Statistics.compute_g1_g2_ranges(image)
        self.assertEqual(g2, 100.0)  # 150 应该被包含

        # 测试 149 刚好在边界外
        image = Image.fromarray(np.full((10, 10), 149, dtype=np.uint8))
        _, g2 = ROI3Statistics.compute_g1_g2_ranges(image)
        self.assertLess(g2, 100.0)  # 149 应该不在 G2 范围内


if __name__ == '__main__':
    unittest.main()
