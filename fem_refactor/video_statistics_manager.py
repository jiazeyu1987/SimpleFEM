from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .external.safe_peak_statistics import SafePeakStatistics


class VideoStatisticsManager:
    """管理每视频的统计实例"""

    def __init__(self):
        self.current_statistics: Optional[SafePeakStatistics] = None
        self.all_statistics: List[SafePeakStatistics] = []
        self.is_batch_mode = False
        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialize_for_video(self, video_path: str, is_batch: bool = False):
        """为视频初始化新的统计实例"""
        # 关闭之前的统计
        if self.current_statistics:
            self.current_statistics.export_final_csv()
            self.all_statistics.append(self.current_statistics)

        # 创建新的统计实例
        self.is_batch_mode = is_batch
        video_name = os.path.basename(video_path) if video_path else None
        self.current_statistics = SafePeakStatistics(
            video_name=video_name,
            is_batch_mode=is_batch,
        )

        return self.current_statistics

    def get_global_summary(self) -> Dict[str, Any]:
        """聚合所有视频的汇总信息"""
        if not self.all_statistics:
            return {
                "total_videos_processed": 0,
                "total_peaks": 0,
                "total_green_peaks": 0,
                "total_red_peaks": 0,
                "session_duration": "00:00:00",
                "videos_processed": [],
            }

        total_peaks = sum(len(s.stats_data) for s in self.all_statistics)
        total_green = sum(len([p for p in s.stats_data if p["peak_type"] == "green"]) for s in self.all_statistics)
        total_red = sum(len([p for p in s.stats_data if p["peak_type"] == "red"]) for s in self.all_statistics)

        session_start_dt = datetime.strptime(self.session_start, "%Y%m%d_%H%M%S")
        session_duration = str(datetime.now() - session_start_dt).split(".")[0]

        return {
            "total_videos_processed": len(self.all_statistics),
            "total_peaks": total_peaks,
            "total_green_peaks": total_green,
            "total_red_peaks": total_red,
            "session_duration": session_duration,
            "videos_processed": [s.video_name for s in self.all_statistics],
        }


# 全局统计管理器实例
statistics_manager = VideoStatisticsManager()

# 为了向后兼容，保持原有的safe_statistics全局变量
safe_statistics = statistics_manager.current_statistics
