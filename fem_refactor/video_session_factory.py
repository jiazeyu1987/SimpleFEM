from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fem_refactor.models import DaemonContext
from fem_refactor.video_session_manager import VideoSessionManager


def maybe_create_video_session_manager(
    *,
    processing_mode: str,
    ctx: DaemonContext,
    config: Dict[str, Any],
    statistics_manager: Any,
    analysis_cache: Any,
    create_video_folders: Callable[..., str],
    intersection_filter: Any,
    roi_frame_rate: float,
    adaptive_window_frames: int,
    save_roi1: bool,
    save_roi2: bool,
    save_roi3: bool,
    save_wave: bool,
    save_roi1_wave: bool,
    video_files: List[str],
) -> Optional[VideoSessionManager]:
    if processing_mode != "video":
        return None

    return VideoSessionManager(
        ctx=ctx,
        config=config,
        statistics_manager=statistics_manager,
        analysis_cache=analysis_cache,
        create_video_folders=create_video_folders,
        intersection_filter=intersection_filter,
        roi_frame_rate=float(roi_frame_rate),
        adaptive_window_frames=int(adaptive_window_frames),
        save_roi1=save_roi1,
        save_roi2=save_roi2,
        save_roi3=save_roi3,
        save_wave=save_wave,
        save_roi1_wave=save_roi1_wave,
        video_files=list(video_files) if video_files else [],
    )

