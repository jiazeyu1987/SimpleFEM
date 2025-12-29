"""
ThresholdProtectionManager 单元测试

SimpleFEM Refactored Version
"""

import unittest
import os
import sys

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager
from refactor.threshold_protection_manager import ThresholdProtectionManager


class TestThresholdProtectionManager(unittest.TestCase):
    """测试 ThresholdProtectionManager 阈值保护管理器"""

    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
        self.manager = ThresholdProtectionManager(self.config)

    def test_initialization(self):
        """测试初始化"""
        self.assertFalse(self.manager._protection_active)
        self.assertEqual(self.manager._protection_end_time, 0.0)
        self.assertEqual(self.manager._consecutive_below, 0)
        self.assertEqual(self.manager._last_waveform_time, 0.0)

    def test_reset(self):
        """测试重置方法"""
        # 先激活保护
        self.manager._protection_active = True
        self.manager._consecutive_below = 5

        # 重置
        self.manager.reset()

        # 验证重置
        self.assertFalse(self.manager._protection_active)
        self.assertEqual(self.manager._consecutive_below, 0)

    def test_is_active_property(self):
        """测试 is_active 属性"""
        self.assertFalse(self.manager.is_active)
        self.manager._protection_active = True
        self.assertTrue(self.manager.is_active)

    def test_frames_since_end_property(self):
        """测试 frames_since_end 属性"""
        self.manager._frames_since_end = 10
        self.assertEqual(self.manager.frames_since_end, 10)

    def test_update_with_protection_disabled(self):
        """测试禁用保护时的更新"""
        # 禁用保护
        self.config._config['peak_detection']['threshold_protection']['enabled'] = False

        # 更新
        should_protect, frames = self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 不应该保护
        self.assertFalse(should_protect)
        self.assertEqual(frames, 0)

    def test_waveform_trigger(self):
        """测试波形触发保护"""
        # 启用保护
        self.config._config['peak_detection']['threshold_protection']['enabled'] = True
        self.config._config['peak_detection']['threshold_protection']['waveform_trigger_enabled'] = True

        # 当前灰度超过阈值，应该触发保护
        should_protect, frames = self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 应该保护
        self.assertTrue(should_protect)
        self.assertTrue(self.manager.is_active)

    def test_peak_trigger(self):
        """测试波峰触发保护"""
        # 启用保护
        self.config._config['peak_detection']['threshold_protection']['enabled'] = True

        # 检测到波峰，应该触发保护
        should_protect, frames = self.manager.update(
            current_gray=90.0,  # 低于阈值
            current_threshold=95.0,
            has_peaks=True,  # 有波峰
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 应该保护
        self.assertTrue(should_protect)
        self.assertTrue(self.manager.is_active)

    def test_protection_continues_when_gray_above_threshold(self):
        """测试灰度值持续高于阈值时保护继续"""
        # 先触发保护
        self.config._config['peak_detection']['threshold_protection']['enabled'] = True
        self.config._config['peak_detection']['threshold_protection']['waveform_trigger_enabled'] = True

        # 触发保护
        self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 继续高于阈值
        should_protect, _ = self.manager.update(
            current_gray=98.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.5,  # 0.5秒后
            frame_index=5,
            fps=10.0
        )

        # 应该继续保护
        self.assertTrue(should_protect)

    def test_protection_releases_with_time_and_stability(self):
        """测试满足时间和稳定性条件时解除保护"""
        # 配置短的恢复延迟
        self.config._config['peak_detection']['threshold_protection']['enabled'] = True
        self.config._config['peak_detection']['threshold_protection']['recovery_delay_seconds'] = 0.1
        self.config._config['peak_detection']['threshold_protection']['stability_frames'] = 3
        self.config._config['peak_detection']['threshold_protection']['waveform_trigger_enabled'] = True

        # 触发保护
        self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 等待恢复时间并保持低于阈值
        for i in range(5):
            should_protect, _ = self.manager.update(
                current_gray=90.0,  # 低于阈值
                current_threshold=95.0,
                has_peaks=False,
                frame_time=1.0 + (i + 1) * 0.1,  # 时间推进
                frame_index=i + 1,
                fps=10.0
            )

        # 应该解除保护
        self.assertFalse(should_protect)
        self.assertFalse(self.manager.is_active)

    def test_stability_condition_resets(self):
        """测试稳定性条件在灰度高于阈值时重置"""
        self.config._config['peak_detection']['threshold_protection']['enabled'] = True
        self.config._config['peak_detection']['threshold_protection']['waveform_trigger_enabled'] = True

        # 触发保护
        self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.0,
            frame_index=0,
            fps=10.0
        )

        # 降低一次
        self.manager.update(
            current_gray=90.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.1,
            frame_index=1,
            fps=10.0
        )
        self.assertEqual(self.manager._consecutive_below, 1)

        # 再次升高
        self.manager.update(
            current_gray=100.0,
            current_threshold=95.0,
            has_peaks=False,
            frame_time=1.2,
            frame_index=2,
            fps=10.0
        )
        # 稳定性计数应该重置
        self.assertEqual(self.manager._consecutive_below, 0)


if __name__ == '__main__':
    unittest.main()
