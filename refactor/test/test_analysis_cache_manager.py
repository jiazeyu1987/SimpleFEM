"""
AnalysisCacheManager 单元测试

SimpleFEM Refactored Version
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime

# 添加父目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from refactor.config_manager import ConfigManager
from refactor.analysis_cache_manager import AnalysisCacheManager


class TestAnalysisCacheManager(unittest.TestCase):
    """测试 AnalysisCacheManager 分析缓存管理器"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConfigManager()
        self.manager = AnalysisCacheManager(self.config, self.temp_dir)

    def tearDown(self):
        """测试后清理"""
        self.manager.close(reason="test_cleanup")
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager._export_dir, self.temp_dir)
        self.assertIsNotNone(self.manager._run_id)
        self.assertEqual(len(self.manager._run_id), 12)  # UUID 前12位
        self.assertIsNone(self.manager._fh)

    def test_path_property(self):
        """测试 path 属性"""
        self.assertIsNone(self.manager.path)  # 未启动会话时为 None

    def test_start_session(self):
        """测试开始会话"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 检查文件是否创建
        self.assertIsNotNone(self.manager.path)
        self.assertTrue(os.path.exists(self.manager.path))

        # 检查文件名格式
        filename = os.path.basename(self.manager.path)
        self.assertTrue(filename.startswith("roi_analysis_cache_"))
        self.assertTrue(filename.endswith(".jsonl"))

    def test_start_session_creates_meta_entry(self):
        """测试开始会话创建元数据条目"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 读取文件并检查元数据
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            meta = json.loads(first_line)

            self.assertEqual(meta['type'], 'meta')
            self.assertEqual(meta['session_id'], 'test_session')
            self.assertEqual(meta['processing_mode'], 'video')
            self.assertIn('cache_version', meta)
            self.assertIn('created_at', meta)

    def test_record_frame(self):
        """测试记录帧数据"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录帧数据
        payload = {
            'frame_index': 100,
            'roi1_avg': 50.0,
            'roi2_avg': 100.0,
            'green_peaks': [(10, 15)],
            'red_peaks': []
        }

        self.manager.record_frame(payload)

        # 刷新以确保数据写入
        self.manager._flush()

        # 读取文件并验证
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是元数据，第二行是帧数据
            self.assertGreaterEqual(len(lines), 2)

            frame_data = json.loads(lines[1])
            self.assertEqual(frame_data['type'], 'frame')
            self.assertEqual(frame_data['frame_index'], 100)

    def test_record_frame_auto_type(self):
        """测试记录帧数据时自动添加 type"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 不提供 type 的 payload
        payload = {'frame_index': 100}
        self.manager.record_frame(payload)

        # 刷新以确保数据写入
        self.manager._flush()

        # 读取并验证
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            frame_data = json.loads(lines[1])
            self.assertEqual(frame_data['type'], 'frame')  # 应该自动添加

    def test_close_writes_session_end(self):
        """测试关闭时写入会话结束标记"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录一些帧
        self.manager.record_frame({'frame_index': 100})

        # 关闭
        self.manager.close(reason="test_complete")

        # 读取文件验证
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_line = lines[-1]
            end_data = json.loads(last_line)

            self.assertEqual(end_data['type'], 'session_end')
            self.assertEqual(end_data['reason'], 'test_complete')
            self.assertIn('ended_at', end_data)

    def test_close_disabled_cache(self):
        """测试禁用缓存时的关闭"""
        # 创建禁用缓存的管理器
        self.config._config['analysis_cache']['enabled'] = False
        disabled_manager = AnalysisCacheManager(self.config, self.temp_dir)

        # 启动会话（不会创建文件）
        disabled_manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 关闭（不应该抛出异常）
        disabled_manager.close(reason="test")

        # 不应该创建文件
        self.assertIsNone(disabled_manager.path)

    def test_record_frame_disabled_cache(self):
        """测试禁用缓存时记录帧"""
        self.config._config['analysis_cache']['enabled'] = False
        disabled_manager = AnalysisCacheManager(self.config, self.temp_dir)

        disabled_manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录帧（不应该抛出异常）
        disabled_manager.record_frame({'frame_index': 100})

        # 不应该创建文件
        self.assertIsNone(disabled_manager.path)

    def test_json_default_numpy(self):
        """测试 numpy 值的 JSON 序列化"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录包含 numpy 值的数据
        import numpy as np
        payload = {
            'frame_index': 100,
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3])
        }

        self.manager.record_frame(payload)

        # 刷新以确保数据写入
        self.manager._flush()

        # 读取并验证序列化
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            frame_data = json.loads(lines[1])

            self.assertEqual(frame_data['numpy_int'], 42)
            self.assertAlmostEqual(frame_data['numpy_float'], 3.14, places=2)
            self.assertEqual(frame_data['numpy_array'], [1, 2, 3])

    def test_json_default_datetime(self):
        """测试 datetime 值的 JSON 序列化"""
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录包含 datetime 的数据
        payload = {
            'frame_index': 100,
            'timestamp': datetime(2025, 12, 28, 12, 30, 45)
        }

        self.manager.record_frame(payload)

        # 刷新以确保数据写入
        self.manager._flush()

        # 读取并验证
        with open(self.manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            frame_data = json.loads(lines[1])

            self.assertIn('timestamp', frame_data)
            # datetime 应该被转换为 ISO 格式字符串

    def test_switch_session(self):
        """测试切换会话"""
        # 开始第一个会话
        self.manager.start_session(
            session_id="session_1",
            processing_mode="video",
            video_path="video1.mp4",
            config={}
        )

        self.manager.record_frame({'frame_index': 100})

        first_path = self.manager.path

        # 开始第二个会话（应该关闭第一个）
        self.manager.start_session(
            session_id="session_2",
            processing_mode="video",
            video_path="video2.mp4",
            config={}
        )

        second_path = self.manager.path

        # 应该有两个不同的文件
        self.assertNotEqual(first_path, second_path)

        # 第一个文件应该包含 session_end
        with open(first_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_data = json.loads(lines[-1])
            self.assertEqual(last_data['type'], 'session_end')
            self.assertEqual(last_data['reason'], 'switch_session')

    def test_flush_every(self):
        """测试自动刷新机制"""
        # 设置较小的刷新间隔
        self.config._config['analysis_cache']['flush_every'] = 2
        manager = AnalysisCacheManager(self.config, self.temp_dir)

        manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        # 记录多帧（应该触发多次刷新）
        for i in range(10):
            manager.record_frame({'frame_index': i})

        # 关闭
        manager.close(reason="test")

        # 验证所有帧都被写入
        with open(manager.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 元数据 + 10 帧数据 + session_end
            self.assertEqual(len(lines), 12)

    def test_current_session_id_property(self):
        """测试 current_session_id 属性"""
        # 未启动会话时
        self.assertIsNone(self.manager.current_session_id)

        # 启动会话后
        self.manager.start_session(
            session_id="test_session",
            processing_mode="video",
            video_path="test.mp4",
            config={}
        )

        self.assertEqual(self.manager.current_session_id, "test_session")


if __name__ == '__main__':
    unittest.main()
