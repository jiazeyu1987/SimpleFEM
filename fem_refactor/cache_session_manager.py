from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def start_analysis_cache_session(
    *,
    analysis_cache: Any,
    processing_mode: str,
    video_files: Optional[List[str]],
    current_video_index: int,
    config: Dict[str, Any],
    statistics_manager: Any,
    roi_frame_rate: float,
    effective_frame_rate: float,
    video_fps: float,
    video_frame_step: int,
    adaptive_window_frames: int,
    gray_buffer_maxlen: int = 100,
) -> None:
    current_stats = statistics_manager.current_statistics
    session_id = current_stats.session_id if current_stats else datetime.now().strftime("%Y%m%d_%H%M%S")

    video_path_for_meta = None
    if processing_mode == "video" and video_files:
        if current_video_index < len(video_files):
            video_path_for_meta = video_files[current_video_index]
        else:
            video_path_for_meta = video_files[0]

    analysis_cache.start_session(
        session_id,
        processing_mode=processing_mode,
        video_path=video_path_for_meta,
        config=config,
        extra_meta={
            "roi_frame_rate": roi_frame_rate,
            "effective_frame_rate": effective_frame_rate,
            "video_fps": video_fps,
            "video_frame_step": video_frame_step,
            "adaptive_window_frames": adaptive_window_frames,
            "gray_buffer_maxlen": gray_buffer_maxlen,
        },
    )
    if getattr(analysis_cache, "path", None):
        print(f"[cache] analysis_cache={analysis_cache.path}")

