"""
统计数据管理器 - 封装SafePeakStatistics

SimpleFEM Refactored Version
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 添加父目录到路径以导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_peak_statistics import SafePeakStatistics
from refactor.config_manager import ConfigManager


class StatisticsManager:
    """
    统计数据管理器

    功能:
    - 管理每视频的统计实例
    - 批量模式支持
    - 添加波峰数据
    - 导出CSV
    """

    def __init__(self, config: ConfigManager):
        """
        初始化统计管理器

        Args:
            config: 配置管理器
        """
        self._config = config
        self._current_statistics: Optional[SafePeakStatistics] = None
        self._all_statistics: List[SafePeakStatistics] = []
        self._is_batch_mode = False
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialize_for_video(self, video_path: str, is_batch: bool = False) -> SafePeakStatistics:
        """
        为视频初始化新的统计实例

        Args:
            video_path: 视频路径
            is_batch: 是否批量模式

        Returns:
            SafePeakStatistics实例
        """
        # 关闭之前的统计
        if self._current_statistics:
            self._current_statistics.export_final_csv()
            self._all_statistics.append(self._current_statistics)

        # 创建新的统计实例
        self._is_batch_mode = is_batch
        video_name = os.path.basename(video_path) if video_path else None
        self._current_statistics = SafePeakStatistics(
            video_name=video_name,
            is_batch_mode=is_batch
        )

        return self._current_statistics

    def add_peaks(
        self,
        frame_index: int,
        green_peaks: List[tuple],
        red_peaks: List[tuple],
        curve_data: List[float],
        intersection: Optional[tuple],
        roi2_info: Dict[str, Any],
        gray_value: float,
        threshold_used: float,
        bg_mean: float,
        roi3_curve: Optional[List[float]] = None,
        hybrid_enabled: bool = False,
        hybrid_peaks: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加波峰数据

        Args:
            frame_index: 帧索引
            green_peaks: 绿色波峰列表
            red_peaks: 红色波峰列表
            curve_data: 曲线数据
            intersection: 交点坐标
            roi2_info: ROI2信息
            gray_value: 灰度值
            threshold_used: 使用的阈值
            bg_mean: 背景均值
            roi3_curve: ROI3曲线（可选）
            hybrid_enabled: 混合检测是否启用
            hybrid_peaks: 混合检测波峰

        Returns:
            统计写入结果
        """
        if self._current_statistics is None:
            return {}

        return self._current_statistics.add_peaks_from_daemon(
            frame_index=frame_index,
            green_peaks=green_peaks,
            red_peaks=red_peaks,
            hybrid_enabled=hybrid_enabled,
            hybrid_peaks=hybrid_peaks or [],
            roi3_curve=roi3_curve or [],
            curve=curve_data,
            intersection=intersection,
            roi2_info=roi2_info,
            gray_value=gray_value,
            difference_threshold=self._config.difference_threshold,
            pre_post_avg_frames=self._config.pre_post_avg_frames,
            threshold_used=threshold_used,
            bg_mean=bg_mean
        )

    def export_final_csv(self) -> None:
        """导出最终CSV"""
        if self._current_statistics:
            self._current_statistics.export_final_csv()
            self._all_statistics.append(self._current_statistics)
            self._current_statistics = None

    def get_global_summary(self) -> Dict[str, Any]:
        """
        聚合所有视频的汇总信息

        Returns:
            汇总信息字典
        """
        if not self._all_statistics:
            return {
                'total_videos_processed': 0,
                'total_peaks': 0,
                'total_green_peaks': 0,
                'total_red_peaks': 0,
                'session_duration': '00:00:00',
                'videos_processed': []
            }

        total_peaks = sum(len(s.stats_data) for s in self._all_statistics)
        total_green = sum(len([p for p in s.stats_data if p['peak_type'] == 'green'])
                         for s in self._all_statistics)
        total_red = sum(len([p for p in s.stats_data if p['peak_type'] == 'red'])
                       for s in self._all_statistics)

        session_start_dt = datetime.strptime(self._session_start, "%Y%m%d_%H%M%S")
        session_duration = str(datetime.now() - session_start_dt).split('.')[0]

        return {
            'total_videos_processed': len(self._all_statistics),
            'total_peaks': total_peaks,
            'total_green_peaks': total_green,
            'total_red_peaks': total_red,
            'session_duration': session_duration,
            'videos_processed': [s.video_name for s in self._all_statistics]
        }

    @property
    def current_statistics(self) -> Optional[SafePeakStatistics]:
        """当前统计实例"""
        return self._current_statistics
