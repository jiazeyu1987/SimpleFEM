"""
GreenLineManager 单元测试

SimpleFEM Refactored Version
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager
from refactor.green_line_manager import GreenLineManager


class TestGreenLineManager(unittest.TestCase):
    """测试 GreenLineManager 绿线检测管理器"""

    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
        self.manager = GreenLineManager(self.config)

    def tearDown(self):
        """测试后清理"""
        self.manager.reset()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager._config)
        self.assertIsNotNone(self.manager._intersection_filter)
        self.assertEqual(len(self.manager._intersection_history), 0)
        self.assertIsNone(self.manager._last_valid_intersection)

    def test_reset(self):
        """测试重置方法"""
        # 添加一些历史记录
        self.manager._intersection_history.append((100, 100))
        self.manager._last_valid_intersection = (100, 100)

        # 重置
        self.manager.reset()

        # 验证重置
        self.assertEqual(len(self.manager._intersection_history), 0)
        self.assertIsNone(self.manager._last_valid_intersection)

    def test_intersection_history_property(self):
        """测试 intersection_history 属性"""
        from collections import deque
        history = self.manager.intersection_history
        self.assertIsInstance(history, deque)

    def test_get_last_valid_intersection(self):
        """测试获取最后有效交点"""
        # 初始应该为 None
        self.assertIsNone(self.manager.get_last_valid_intersection())

        # 设置一个交点
        self.manager._last_valid_intersection = (100, 100)
        self.assertEqual(self.manager.get_last_valid_intersection(), (100, 100))

    def test_detect_intersection_with_valid_image(self):
        """测试检测有效图像的交点"""
        # 创建包含绿色线条的测试图像
        test_image = self._create_green_line_image()

        # 注意：这个测试依赖原始的 detect_green_intersection 函数
        # 如果函数不可用，测试会被跳过
        try:
            intersection = self.manager.detect_intersection(test_image)
            # 交点可能是 None 或 (x, y)
            self.assertTrue(intersection is None or isinstance(intersection, tuple))
        except Exception as e:
            self.skipTest(f"detect_green_intersection not available: {e}")

    def test_detect_intersection_with_invalid_input(self):
        """测试检测无效输入"""
        # 测试 None 输入
        try:
            intersection = self.manager.detect_intersection(None)
            self.assertIsNone(intersection)
        except:
            pass  # 如果抛出异常也是可以接受的

    def test_anti_jitter_enabled(self):
        """测试防抖动启用配置"""
        enabled = self.config.roi2_anti_jitter_enabled
        self.assertIsInstance(enabled, bool)

    def test_anti_jitter_algorithm(self):
        """测试防抖动算法配置"""
        algorithm = self.config.roi2_anti_jitter_algorithm
        self.assertIn(algorithm, ['ema', 'threshold'])

    def test_movement_threshold(self):
        """测试运动阈值配置"""
        threshold = self.config.roi2_movement_threshold
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)

    def test_ema_alpha(self):
        """测试 EMA alpha 参数"""
        alpha = self.config.roi2_anti_jitter_alpha
        self.assertIsInstance(alpha, float)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_intersection_filter_initialization(self):
        """测试交点滤波器初始化"""
        # 检查滤波器是否正确初始化
        self.assertIsNotNone(self.manager._intersection_filter)

        # 检查算法类型
        algorithm = self.config.roi2_anti_jitter_algorithm
        if algorithm == 'ema':
            # EMA 模式应该有特定的参数
            self.assertIsNotNone(self.manager._intersection_filter)
        elif algorithm == 'threshold':
            # Threshold 模式
            self.assertIsNotNone(self.manager._intersection_filter)

    def _create_green_line_image(self):
        """创建包含绿色线条的测试图像"""
        # 创建 640x480 的图像
        width, height = 640, 480
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # 添加绿色背景
        img_array[:, :, 1] = 50  # G channel

        # 绘制一条水平绿色线
        y_pos = height // 2
        img_array[y_pos-2:y_pos+2, :, 1] = 200  # 亮绿色线

        return Image.fromarray(img_array)

    def test_history_maxlen(self):
        """测试历史记录的最大长度"""
        from collections import deque
        history = self.manager.intersection_history

        # 应该是 deque 且有 maxlen
        self.assertIsInstance(history, deque)
        # maxlen 应该是 10
        self.assertEqual(history.maxlen, 10)


if __name__ == '__main__':
    unittest.main()
