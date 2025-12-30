"""
Simple ROI daemon:
 - Every second, capture ROI1 from screen using PIL.ImageGrab
 - Detect green line intersection inside ROI1 using existing green_detector
 - Around the latest intersection point, extract ROI2 according to roi2_config.extension_params
 - Compute ROI2 average gray value and push into a fixed-length buffer (length 100)
 - Run peak detection using backend.app.peak_detection.detect_peaks with fem_config parameters
 - Log per-second summary to a daily-rotating log file (backend/logs/roi_peak_daemon.log)

Usage:
    python simple_roi_daemon.py
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional, Tuple

from fem_refactor.analysis_cache import RoiAnalysisCache
from fem_refactor.paths import get_base_dir
from fem_refactor.daemon_bootstrap import DaemonBootstrap
from fem_refactor.loop_iteration import capture_frame_for_iteration, log_and_sleep, process_iteration_step
from fem_refactor.models import DaemonContext
from fem_refactor.shutdown_manager import shutdown_daemon
from fem_refactor.video_session_manager import VideoSessionManager
from fem_refactor.video_statistics_manager import statistics_manager, safe_statistics
from fem_refactor.video_folders import create_video_folders


def _get_base_dir() -> str:
    """
    Resolve base directory both for source (.py) and frozen (.exe) modes.

    When packaged with PyInstaller, sys.frozen is True and sys.executable
    points to the .exe location. In source mode, use this file's directory.
    """
    return get_base_dir(__file__)


BASE_DIR = _get_base_dir()
EXTERNAL_BASE_DIR = os.path.join(BASE_DIR, "fem_refactor", "external")


def _create_video_folders(video_path: str, session_id: str, processing_mode: str, save_roi1: bool, save_roi2: bool, save_roi3: bool, save_wave: bool, save_roi1_wave: bool = False) -> str:
    return create_video_folders(
        base_dir=EXTERNAL_BASE_DIR,
        video_path=video_path,
        session_id=session_id,
        processing_mode=processing_mode,
        save_roi1=save_roi1,
        save_roi2=save_roi2,
        save_roi3=save_roi3,
        save_wave=save_wave,
        save_roi1_wave=save_roi1_wave,
    )


def run_daemon() -> None:
    """
    Main loop:
      - capture ROI1
      - detect/update line intersection
      - extract ROI2
      - update gray buffer and run peak detection
      - log results at configured frame_rate
    """
    ctx: Optional[DaemonContext] = None
    analysis_cache: Optional[RoiAnalysisCache] = None
    intersection_filter: Any = None
    processing_mode: str = "screen"
    interval_seconds: float = 1.0
    logger: Optional[logging.Logger] = None
    video_session_manager: Optional[VideoSessionManager] = None

    try:
        boot = DaemonBootstrap(
            base_dir=BASE_DIR,
            statistics_manager=statistics_manager,
            create_video_folders=_create_video_folders,
        ).bootstrap()

        ctx = boot.ctx
        analysis_cache = boot.analysis_cache
        intersection_filter = boot.intersection_filter
        processing_mode = boot.processing_mode
        interval_seconds = boot.interval_seconds
        logger = boot.logger
        video_session_manager = boot.video_session_manager

        while True:
            loop_start = time.time()
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            log_line: Optional[str] = None

            try:
                if ctx is None:
                    raise RuntimeError("ctx is not initialized")
                if logger is None:
                    raise RuntimeError("logger is not initialized")

                ctx.state.frame_index += 1
                frame_index = ctx.state.frame_index

                action, screen, screen_width, screen_height = capture_frame_for_iteration(
                    processing_mode=processing_mode,
                    ctx=ctx,
                    video_session_manager=video_session_manager,
                    loop_start=loop_start,
                    interval_seconds=interval_seconds,
                    frame_index=frame_index,
                )
                if action == "break":
                    break
                if action == "continue":
                    continue
                if screen is None or screen_width is None or screen_height is None:
                    raise RuntimeError("capture returned incomplete frame")

                log_line = process_iteration_step(
                    ctx=ctx,
                    screen=screen,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    loop_start=loop_start,
                    ts=ts,
                )
            except KeyboardInterrupt:
                if logger is not None:
                    logger.info(f"{ts} INFO=daemon_stopped_by_user")
                break
            except Exception as e:
                # Log unexpected error but keep daemon alive
                log_line = f"{ts} ERROR={repr(e)}"

            log_and_sleep(
                logger=logger if logger is not None else logging.getLogger(__name__),
                log_line=log_line,
                loop_start=loop_start,
                interval_seconds=interval_seconds,
                frame_index=frame_index,
            )

    finally:
        shutdown_daemon(
            analysis_cache=analysis_cache,
            ctx=ctx,
            intersection_filter=intersection_filter,
        )


if __name__ == "__main__":
    run_daemon()
