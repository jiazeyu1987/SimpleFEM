from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fem_refactor.video_source import get_video_fps


@dataclass(frozen=True)
class TimingState:
    video_fps: float
    video_frame_step: int
    first_video_frame: bool
    effective_frame_rate: float
    interval_seconds: float
    adaptive_window_frames: int
    recovery_delay_frames: int


def compute_timing_state(
    *,
    processing_mode: str,
    video_cap: Any,
    roi_frame_rate: float,
    adaptive_window_seconds: float,
    recovery_delay_seconds: float,
) -> TimingState:
    video_fps = 0.0
    video_frame_step = 1
    first_video_frame = True
    effective_frame_rate = roi_frame_rate

    if processing_mode == "video" and video_cap is not None:
        video_fps = get_video_fps(video_cap)
        if video_fps > 0:
            effective_frame_rate = min(roi_frame_rate, video_fps)
            if effective_frame_rate > 0:
                video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))

    interval_seconds = 1.0 / max(1e-6, effective_frame_rate)
    if processing_mode == "video" and video_fps > 0:
        print(
            f"[video] source_fps={video_fps:.2f} target_fps={effective_frame_rate:.2f} frame_step={video_frame_step}"
        )

    print(f"[帧率配置] 配置帧率: {roi_frame_rate} fps")
    print(f"[帧率配置] 计算间隔: {interval_seconds:.3f} 秒/帧")
    print(f"[帧率配置] 预期7秒视频处理: {7 * roi_frame_rate} 帧")

    adaptive_window_frames = int(adaptive_window_seconds * effective_frame_rate)
    adaptive_window_frames = max(1, min(adaptive_window_frames, 100))

    recovery_delay_frames = int(recovery_delay_seconds * effective_frame_rate)
    recovery_delay_frames = max(1, recovery_delay_frames)

    return TimingState(
        video_fps=float(video_fps),
        video_frame_step=int(video_frame_step),
        first_video_frame=bool(first_video_frame),
        effective_frame_rate=float(effective_frame_rate),
        interval_seconds=float(interval_seconds),
        adaptive_window_frames=int(adaptive_window_frames),
        recovery_delay_frames=int(recovery_delay_frames),
    )
