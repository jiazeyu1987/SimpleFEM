"""
ConfigManager 单元测试

SimpleFEM Refactored Version
"""

import json
import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """测试 ConfigManager 配置管理器"""

    def setUp(self):
        """测试前准备"""
        # 创建临时配置文件
        self.temp_config = {
            "processing_mode": "video",
            "roi_capture": {
                "frame_rate": 10,
                "default_config": {
                    "x1": 1280, "y1": 80, "x2": 1920, "y2": 980
                },
                "roi2_config": {
                    "extension_params": {
                        "left": 20, "right": 30, "top": 60, "bottom": 20
                    }
                },
                "roi3_config": {
                    "extension_params": {
                        "left": 20, "right": 30, "top": 80, "bottom": 40
                    }
                }
            },
            "peak_detection": {
                "threshold": 95.0,
                "threshold_minimum": 80.0,
                "margin_frames": 5,
                "silence_frames": 15,
                "difference_threshold": 2.1,
                "min_region_length": 5,
                "pre_post_avg_frames": 5,
                "adaptive_threshold_enabled": True,
                "threshold_over_mean_ratio": 0.15,
                "adaptive_window_seconds": 3.0,
                "threshold_protection": {
                    "enabled": True,
                    "recovery_delay_seconds": 1.0,
                    "stability_frames": 5,
                    "waveform_trigger_enabled": True
                },
                "g1_g2_override": {
                    "enabled": True,
                    "g1_threshold": 98.0,
                    "g2_threshold": 20.0,
                    "use_peak_max": True
                },
                "roi3_column_diff_override": {
                    "enabled": True,
                    "threshold": 15.0,
                    "use_peak_max": True
                }
            },
            "roi1_peak_detection": {
                "enabled": True,
                "threshold": 120.0,
                "threshold_minimum": 110.0
            },
            "hybrid_detection": {
                "enabled": True,
                "detection_strategy": "roi1_peaks_roi2_color",
                "roi2_color_frames": {
                    "pre_peak": 5,
                    "post_peak": 10
                }
            },
            "roi2_anti_jitter": {
                "enabled": True,
                "algorithm": "ema",
                "movement_threshold": 20.0,
                "ema": {
                    "alpha": 0.25,
                    "stability_threshold": 8.0,
                    "initialization_frames": 3
                }
            },
            "video_processing": {
                "video_path": "test_video.mp4",
                "loop_enabled": False
            },
            "data_processing": {
                "save_roi1": True,
                "save_roi2": True,
                "save_roi3": True,
                "save_wave": True,
                "save_roi1_wave": True,
                "only_delect": False
            },
            "analysis_cache": {
                "enabled": True,
                "flush_every": 50
            },
            "deduplication": {
                "consecutive_frame_window": 40,
                "color_priority": ["green", "red"],
                "cross_color_deduplication_enabled": True
            },
            "startup_cleanup": {
                "enabled": True,
                "cleanup_export": True,
                "cleanup_tmp": True,
                "cleanup_logs": False
            }
        }

        # 创建临时文件
        self.temp_fd, self.temp_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(self.temp_fd, 'w') as f:
            json.dump(self.temp_config, f)

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_load_config(self):
        """测试加载配置文件"""
        manager = ConfigManager(self.temp_path)
        self.assertIsNotNone(manager._config)
        self.assertEqual(manager._config['processing_mode'], 'video')

    def test_load_config_file_not_found(self):
        """测试加载不存在的配置文件"""
        with self.assertRaises(FileNotFoundError):
            ConfigManager('nonexistent.json')

    def test_get_config_value(self):
        """测试获取配置值"""
        manager = ConfigManager(self.temp_path)

        # 测试获取简单值
        self.assertEqual(manager.get('processing_mode'), 'video')

        # 测试获取嵌套值
        self.assertEqual(manager.get('roi_capture', 'frame_rate'), 10)

        # 测试默认值
        self.assertEqual(manager.get('nonexistent', 'key', default='default_value'), 'default_value')

    def test_processing_mode_property(self):
        """测试 processing_mode 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.processing_mode, 'video')

    def test_frame_rate_property(self):
        """测试 frame_rate 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.frame_rate, 10)

    def test_roi1_config_property(self):
        """测试 roi1_config 属性"""
        manager = ConfigManager(self.temp_path)
        roi1 = manager.roi1_config
        self.assertEqual(roi1['x1'], 1280)
        self.assertEqual(roi1['y1'], 80)
        self.assertEqual(roi1['x2'], 1920)
        self.assertEqual(roi1['y2'], 980)

    def test_roi2_extension_params_property(self):
        """测试 roi2_extension_params 属性"""
        manager = ConfigManager(self.temp_path)
        params = manager.roi2_extension_params
        self.assertEqual(params['left'], 20)
        self.assertEqual(params['right'], 30)
        self.assertEqual(params['top'], 60)
        self.assertEqual(params['bottom'], 20)

    def test_roi3_extension_params_property(self):
        """测试 roi3_extension_params 属性"""
        manager = ConfigManager(self.temp_path)
        params = manager.roi3_extension_params
        self.assertEqual(params['left'], 20)
        self.assertEqual(params['right'], 30)
        self.assertEqual(params['top'], 80)
        self.assertEqual(params['bottom'], 40)

    def test_peak_detection_threshold_property(self):
        """测试 peak_detection_threshold 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.peak_detection_threshold, 95.0)

    def test_adaptive_threshold_enabled_property(self):
        """测试 adaptive_threshold_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.adaptive_threshold_enabled)

    def test_threshold_minimum_property(self):
        """测试 threshold_minimum 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.threshold_minimum, 80.0)

    def test_threshold_over_mean_ratio_property(self):
        """测试 threshold_over_mean_ratio 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertAlmostEqual(manager.threshold_over_mean_ratio, 0.15, places=2)

    def test_difference_threshold_property(self):
        """测试 difference_threshold 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertAlmostEqual(manager.difference_threshold, 2.1, places=1)

    def test_margin_frames_property(self):
        """测试 margin_frames 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.margin_frames, 5)

    def test_silence_frames_property(self):
        """测试 silence_frames 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.silence_frames, 15)

    def test_min_region_length_property(self):
        """测试 min_region_length 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.min_region_length, 5)

    def test_threshold_protection_enabled_property(self):
        """测试 threshold_protection_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.threshold_protection_enabled)

    def test_threshold_protection_recovery_delay_property(self):
        """测试 threshold_protection_recovery_delay 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.threshold_protection_recovery_delay, 1.0)

    def test_g1_g2_override_enabled_property(self):
        """测试 g1_g2_override_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.g1_g2_override_enabled)

    def test_g1_threshold_property(self):
        """测试 g1_threshold 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.g1_threshold, 98.0)

    def test_g2_threshold_property(self):
        """测试 g2_threshold 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.g2_threshold, 20.0)

    def test_hybrid_detection_enabled_property(self):
        """测试 hybrid_detection_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.hybrid_detection_enabled)

    def test_roi2_anti_jitter_enabled_property(self):
        """测试 roi2_anti_jitter_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.roi2_anti_jitter_enabled)

    def test_roi2_anti_jitter_algorithm_property(self):
        """测试 roi2_anti_jitter_algorithm 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.roi2_anti_jitter_algorithm, 'ema')

    def test_roi2_movement_threshold_property(self):
        """测试 roi2_movement_threshold 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.roi2_movement_threshold, 20.0)

    def test_video_path_property(self):
        """测试 video_path 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.video_path, 'test_video.mp4')

    def test_save_roi1_property(self):
        """测试 save_roi1 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.save_roi1)

    def test_save_roi2_property(self):
        """测试 save_roi2 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.save_roi2)

    def test_save_roi3_property(self):
        """测试 save_roi3 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.save_roi3)

    def test_save_wave_property(self):
        """测试 save_wave 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.save_wave)

    def test_analysis_cache_enabled_property(self):
        """测试 analysis_cache_enabled 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertTrue(manager.analysis_cache_enabled)

    def test_analysis_cache_flush_every_property(self):
        """测试 analysis_cache_flush_every 属性"""
        manager = ConfigManager(self.temp_path)
        self.assertEqual(manager.analysis_cache_flush_every, 50)

    def test_get_full_config(self):
        """测试 get_full_config 方法"""
        manager = ConfigManager(self.temp_path)
        full_config = manager.get_full_config()
        self.assertIsInstance(full_config, dict)
        self.assertIn('processing_mode', full_config)
        # 确保返回的是副本
        full_config['processing_mode'] = 'modified'
        self.assertEqual(manager._config['processing_mode'], 'video')

    @patch.dict(os.environ, {'NHEM_PROCESSING_MODE': 'screen'})
    def test_env_override(self):
        """测试环境变量覆盖"""
        manager = ConfigManager(self.temp_path)
        # 环境变量应该覆盖配置文件
        self.assertEqual(manager.processing_mode, 'screen')


if __name__ == '__main__':
    unittest.main()
