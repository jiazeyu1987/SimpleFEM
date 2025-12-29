"""
DataExportManager 单元测试

SimpleFEM Refactored Version
"""

import os
import sys
import tempfile
import unittest
from collections import deque
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager
from refactor.data_export_manager import DataExportManager


class TestDataExportManager(unittest.TestCase):
    """测试 DataExportManager 数据导出管理器"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConfigManager()

    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_image(self, width=100, height=100, color=(128, 128, 128)):
        """创建测试图像"""
        arr = np.full((height, width, 3), color, dtype=np.uint8)
        return Image.fromarray(arr)

    def test_initialization_video_mode(self):
        """测试视频模式初始化"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        self.assertIsNotNone(manager._tmp_root)
        self.assertIn("test_video", manager._tmp_root)

    def test_initialization_screen_mode(self):
        """测试屏幕模式初始化"""
        self.config._config['processing_mode'] = 'screen'
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path=None
        )

        self.assertIsNotNone(manager._tmp_root)

    def test_directory_creation(self):
        """测试目录创建"""
        self.config._config['data_processing']['save_roi1'] = True
        self.config._config['data_processing']['save_roi2'] = True
        self.config._config['data_processing']['save_roi3'] = True
        self.config._config['data_processing']['save_wave'] = True
        self.config._config['data_processing']['save_roi1_wave'] = True

        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 检查目录是否创建
        self.assertTrue(os.path.exists(manager._roi1_dir))
        self.assertTrue(os.path.exists(manager._roi2_dir))
        self.assertTrue(os.path.exists(manager._roi3_dir))
        self.assertTrue(os.path.exists(manager._wave_dir))
        self.assertTrue(os.path.exists(manager._wave1_dir))

    def test_save_roi1(self):
        """测试保存 ROI1"""
        self.config._config['data_processing']['save_roi1'] = True
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 创建测试图像
        test_image = self._create_test_image()

        # 保存
        manager.save_roi1(test_image, 100, 10.5)

        # 验证文件创建
        expected_path = os.path.join(manager._roi1_dir, "roi1_000100_010.50s.png")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_roi1_disabled(self):
        """测试禁用保存 ROI1"""
        self.config._config['data_processing']['save_roi1'] = False
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        test_image = self._create_test_image()

        # 不应该抛出异常
        manager.save_roi1(test_image, 100, 10.5)

    def test_save_roi2(self):
        """测试保存 ROI2"""
        self.config._config['data_processing']['save_roi2'] = True
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        test_image = self._create_test_image()
        manager.save_roi2(test_image, 100)

        expected_path = os.path.join(manager._roi2_dir, "roi2_000100.png")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_roi3(self):
        """测试保存 ROI3"""
        self.config._config['data_processing']['save_roi3'] = True
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        test_image = self._create_test_image()
        manager.save_roi3(test_image, 100)

        expected_path = os.path.join(manager._roi3_dir, "roi3_000100.png")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_roi1_waveform(self):
        """测试保存 ROI1 波形"""
        self.config._config['data_processing']['save_roi1_wave'] = True
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 创建测试缓冲
        buffer = deque([50.0 + i * 0.1 for i in range(100)], maxlen=100)

        manager.save_roi1_waveform(buffer, 95.0, 100)

        expected_path = os.path.join(manager._wave1_dir, "roi1_wave_000100.png")
        self.assertTrue(os.path.exists(expected_path))

    def test_save_waveform(self):
        """测试保存波形图"""
        self.config._config['data_processing']['save_wave'] = True
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 创建测试数据
        buffer = deque([50.0] * 100, maxlen=100)
        green_peaks = [(20, 30)]
        red_peaks = [(50, 60)]

        manager.save_waveform(
            buffer,
            green_peaks,
            red_peaks,
            95.0,
            100,
            10.5
        )

        expected_path = os.path.join(manager._wave_dir, "wave_000100_010.50s.png")
        self.assertTrue(os.path.exists(expected_path))

    def test_sanitize_video_name(self):
        """测试视频名称清理"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 测试非法字符
        sanitized = manager._sanitize_video_name('video<>:"/\\|?*.mp4')
        self.assertNotIn('<', sanitized)
        self.assertNotIn('>', sanitized)
        self.assertNotIn(':', sanitized)
        self.assertNotIn('"', sanitized)
        self.assertNotIn('/', sanitized)
        self.assertNotIn('\\', sanitized)
        self.assertNotIn('|', sanitized)
        self.assertNotIn('?', sanitized)
        self.assertNotIn('*', sanitized)

    def test_sanitize_video_name_length_limit(self):
        """测试视频名称长度限制"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 创建很长的名称
        long_name = 'a' * 100
        sanitized = manager._sanitize_video_name(long_name)

        # 应该被限制在 50 字符
        self.assertLessEqual(len(sanitized), 50)

    def test_get_filename_with_video_time(self):
        """测试带视频时间的文件名生成"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        filename = manager._get_filename(100, 10.5, "roi")

        self.assertEqual(filename, "roi_000100_010.50s.png")

    def test_get_filename_without_video_time(self):
        """测试不带视频时间的文件名生成"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        filename = manager._get_filename(100, None, "roi")

        self.assertEqual(filename, "roi_000100.png")

    def test_tmp_root_property(self):
        """测试 tmp_root 属性"""
        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        self.assertIsNotNone(manager.tmp_root)
        self.assertTrue(os.path.exists(manager.tmp_root))

    def test_waveform_with_roi2_annotation(self):
        """测试带 ROI2 标注的波形图"""
        self.config._config['data_processing']['save_wave'] = True
        self.config._config['data_processing']['save_roi2'] = True

        manager = DataExportManager(
            self.config,
            session_id="test_session",
            video_path="test_video.mp4"
        )

        # 先保存 ROI2 图像
        roi2_image = self._create_test_image(color=(200, 100, 50))
        manager.save_roi2(roi2_image, 100)

        # 保存波形图（应该标注 ROI2）
        buffer = deque([50.0] * 100, maxlen=100)
        roi2_path = os.path.join(manager._roi2_dir, "roi2_000100.png")

        manager.save_waveform(
            buffer,
            [],
            [],
            95.0,
            100,
            None,
            roi2_path
        )

        # 验证波形图创建
        expected_path = os.path.join(manager._wave_dir, "wave_000100.png")
        self.assertTrue(os.path.exists(expected_path))


if __name__ == '__main__':
    unittest.main()
