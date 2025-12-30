from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import cv2

from fem_refactor.frame_step import process_frame
from fem_refactor.models import DaemonContext
from fem_refactor.screen_source import capture_screen
from fem_refactor.video_session_manager import VideoSessionManager


def get_video_seconds(*, processing_mode: str, video_cap: Any) -> Optional[float]:
    if processing_mode != "video" or video_cap is None:
        return None
    try:
        video_pos_msec = float(video_cap.get(cv2.CAP_PROP_POS_MSEC))
        if video_pos_msec >= 0:
            return video_pos_msec / 1000.0
    except Exception:
        return None
    return None


def capture_frame_for_iteration(
    *,
    processing_mode: str,
    ctx: DaemonContext,
    video_session_manager: Optional[VideoSessionManager],
    loop_start: float,
    interval_seconds: float,
    frame_index: int,
) -> Tuple[str, Optional[Any], Optional[int], Optional[int]]:
    """
    Returns: (action, screen, screen_width, screen_height)
      - action in {"ok","continue","break"}
    """
    if processing_mode == "video":
        if video_session_manager is None:
            raise RuntimeError("video_session_manager is not initialized")

        capture_result = video_session_manager.capture_next(
            loop_start=loop_start,
            interval_seconds=interval_seconds,
            frame_index=frame_index,
        )
        if capture_result.should_break:
            return ("break", None, None, None)
        if capture_result.should_continue:
            return ("continue", None, None, None)
        screen = capture_result.screen
        if screen is None:
            raise RuntimeError("video capture returned no frame")
        screen_width, screen_height = screen.size
        return ("ok", screen, int(screen_width), int(screen_height))

    screen = capture_screen()
    screen_width, screen_height = screen.size
    return ("ok", screen, int(screen_width), int(screen_height))


def process_iteration_step(
    *,
    ctx: DaemonContext,
    screen: Any,
    screen_width: int,
    screen_height: int,
    loop_start: float,
    ts: str,
) -> Optional[str]:
    video_seconds = get_video_seconds(
        processing_mode=ctx.video.processing_mode,
        video_cap=ctx.video.video_cap,
    )
    step_result = process_frame(
        ctx=ctx,
        screen=screen,
        screen_width=screen_width,
        screen_height=screen_height,
        loop_start=loop_start,
        ts=ts,
        video_seconds=video_seconds,
    )
    return step_result.log_line


def log_and_sleep(
    *,
    logger: logging.Logger,
    log_line: Optional[str],
    loop_start: float,
    interval_seconds: float,
    frame_index: int,
) -> None:
    if log_line is not None:
        logger.info(log_line)

    # Maintain ~1-second interval between iterations
    elapsed = time.time() - loop_start
    sleep_time = max(0.0, interval_seconds - elapsed)

    # 调试信息：每10帧打印一次帧率控制信息
    if frame_index % 10 == 0:
        print(
            f"[帧率调试] 帧{frame_index}: 目标间隔={interval_seconds:.3f}s, 实际耗时={elapsed:.3f}s, 睡眠时间={sleep_time:.3f}s"
        )

    time.sleep(sleep_time)

