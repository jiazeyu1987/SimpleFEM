from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from .video_source import discover_video_files, initialize_video_capture


def initialize_processing_mode(
    config: dict,
    statistics_manager: Any,
) -> Tuple[str, Any, List[str], int, Optional[Any]]:
    """
    Initialize screen/video mode resources.

    Returns:
        (processing_mode, video_cap, video_files, current_video_index, safe_statistics)
    """
    # 检测处理模式
    processing_mode = config.get("processing_mode", "screen")
    video_cap = None
    video_files: List[str] = []  # 存储要处理的视频文件列表
    current_video_index = 0  # 当前处理的视频索引
    safe_statistics = None

    # 为屏幕模式初始化统计实例
    if processing_mode == "screen":
        statistics_manager.initialize_for_video(None, is_batch=False)
        safe_statistics = statistics_manager.current_statistics

    if processing_mode == "video":
        video_config = config.get("video_processing", {})
        video_path = video_config.get("video_path", "")
        if not video_path:
            raise ValueError("Video mode enabled but no video_path specified in config")

        # 检查是单个文件还是文件夹
        if os.path.isfile(video_path):
            # 单个视频文件
            video_files = [video_path]
            print(f"视频模式: 单个视频文件 {video_path}")
        elif os.path.isdir(video_path):
            # 视频文件夹
            video_files = discover_video_files(video_path)
            print(f"视频模式: 文件夹 {video_path}")
            print(f"发现 {len(video_files)} 个视频文件:")
            for i, video_file in enumerate(video_files, 1):
                print(f"  {i}. {os.path.basename(video_file)}")
        else:
            raise ValueError(f"Video path does not exist: {video_path}")

        # 初始化第一个视频
        if video_files:
            # 为第一个视频初始化统计
            statistics_manager.initialize_for_video(video_files[0], is_batch=True)
            video_cap = initialize_video_capture(video_files[0])

    return processing_mode, video_cap, video_files, current_video_index, safe_statistics

