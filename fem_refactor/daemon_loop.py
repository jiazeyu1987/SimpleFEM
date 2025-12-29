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
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2

from fem_refactor.analysis_cache import RoiAnalysisCache
from fem_refactor.paths import get_base_dir
from fem_refactor.screen_source import capture_screen
from fem_refactor.frame_step import process_frame
from fem_refactor.daemon_bootstrap import DaemonBootstrap
from fem_refactor.models import DaemonContext
from fem_refactor.video_session_manager import VideoSessionManager


def _get_video_seconds(*, processing_mode: str, video_cap: Any) -> Optional[float]:
    if processing_mode != "video" or video_cap is None:
        return None
    try:
        video_pos_msec = float(video_cap.get(cv2.CAP_PROP_POS_MSEC))
        if video_pos_msec >= 0:
            return video_pos_msec / 1000.0
    except Exception:
        return None
    return None


def _get_base_dir() -> str:
    """
    Resolve base directory both for source (.py) and frozen (.exe) modes.

    When packaged with PyInstaller, sys.frozen is True and sys.executable
    points to the .exe location. In source mode, use this file's directory.
    """
    return get_base_dir(__file__)


BASE_DIR = _get_base_dir()


def _setup_import_paths() -> None:
    """
    Ensure we can import local peak_detection and green_detector modules.

    All required files (simple_roi_daemon.py, peak_detection.py, green_detector.py,
    simple_fem_config.json) are expected to be in the same SimpleFEM directory.
    """
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)


_setup_import_paths()

from safe_peak_statistics import SafePeakStatistics  # type: ignore  # noqa: E402


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
            is_batch_mode=is_batch
        )

        return self.current_statistics

    def get_global_summary(self) -> Dict[str, Any]:
        """聚合所有视频的汇总信息"""
        if not self.all_statistics:
            return {
                'total_videos_processed': 0,
                'total_peaks': 0,
                'total_green_peaks': 0,
                'total_red_peaks': 0,
                'session_duration': '00:00:00',
                'videos_processed': []
            }

        total_peaks = sum(len(s.stats_data) for s in self.all_statistics)
        total_green = sum(len([p for p in s.stats_data if p['peak_type'] == 'green'])
                         for s in self.all_statistics)
        total_red = sum(len([p for p in s.stats_data if p['peak_type'] == 'red'])
                       for s in self.all_statistics)

        session_start_dt = datetime.strptime(self.session_start, "%Y%m%d_%H%M%S")
        session_duration = str(datetime.now() - session_start_dt).split('.')[0]

        return {
            'total_videos_processed': len(self.all_statistics),
            'total_peaks': total_peaks,
            'total_green_peaks': total_green,
            'total_red_peaks': total_red,
            'session_duration': session_duration,
            'videos_processed': [s.video_name for s in self.all_statistics]
        }


# 全局统计管理器实例
statistics_manager = VideoStatisticsManager()

# 为了向后兼容，保持原有的safe_statistics全局变量
safe_statistics = statistics_manager.current_statistics


def _sanitize_video_name(video_name: str) -> str:
    """清理视频名称用于文件夹创建"""
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', video_name)
    sanitized = sanitized.strip('._')[:50]
    return sanitized or f"video_{int(time.time())}"


def _create_video_folders(video_path: str, session_id: str, processing_mode: str, save_roi1: bool, save_roi2: bool, save_roi3: bool, save_wave: bool, save_roi1_wave: bool = False) -> str:
    """创建每视频的文件夹结构"""
    if processing_mode == "video":
        # 批量模式：使用视频名称
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sanitized_name = _sanitize_video_name(video_name)
        tmp_root = os.path.join(BASE_DIR, "tmp", sanitized_name)
    else:
        # 屏幕模式：使用基于会话的命名（原有行为）
        session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_root = os.path.join(BASE_DIR, "tmp", session_start)

    # 创建子文件夹
    roi1_dir = os.path.join(tmp_root, "roi1")
    roi2_dir = os.path.join(tmp_root, "roi2")
    roi3_dir = os.path.join(tmp_root, "roi3")
    wave_dir = os.path.join(tmp_root, "wave")
    wave1_dir = os.path.join(tmp_root, "wave1")

    # 根据配置创建目录
    if save_roi1 or save_roi2 or save_roi3 or save_wave or save_roi1_wave:
        os.makedirs(tmp_root, exist_ok=True)
    if save_roi1:
        os.makedirs(roi1_dir, exist_ok=True)
    if save_roi2:
        os.makedirs(roi2_dir, exist_ok=True)
    if save_roi3:
        os.makedirs(roi3_dir, exist_ok=True)
    if save_wave:
        os.makedirs(wave_dir, exist_ok=True)
    if save_roi1_wave:
        os.makedirs(wave1_dir, exist_ok=True)

    return tmp_root


def run_daemon() -> None:
    """
    Main loop:
      - capture ROI1
      - detect/update line intersection
      - extract ROI2
      - update gray buffer and run peak detection
      - log results at configured frame_rate
    """
    try:
        ctx: Optional[DaemonContext] = None
        analysis_cache: Optional[RoiAnalysisCache] = None
        intersection_filter: Any = None
        processing_mode: str = "screen"
        interval_seconds: float = 1.0
        logger: Optional[logging.Logger] = None
        video_session_manager: Optional[VideoSessionManager] = None

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

        def _capture_frame_for_iteration(
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

        def _process_iteration_step(
            *,
            ctx: DaemonContext,
            screen: Any,
            screen_width: int,
            screen_height: int,
            loop_start: float,
            ts: str,
        ) -> Optional[str]:
            video_seconds = _get_video_seconds(
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

        def _log_and_sleep(
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

                action, screen, screen_width, screen_height = _capture_frame_for_iteration(
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

                log_line = _process_iteration_step(
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

            _log_and_sleep(
                logger=logger if logger is not None else logging.getLogger(__name__),
                log_line=log_line,
                loop_start=loop_start,
                interval_seconds=interval_seconds,
                frame_index=frame_index,
            )

    finally:
        if 'analysis_cache' in locals() and analysis_cache is not None:
            try:
                analysis_cache.close(reason="shutdown")
            except Exception:
                pass
        # 释放视频资源
        if 'ctx' in locals() and ctx is not None and ctx.video.video_cap is not None:
            ctx.video.video_cap.release()
            print("视频资源已释放")

        # 输出防抖动滤波器最终统计信息
        if 'intersection_filter' in locals() and intersection_filter:
            try:
                debug_info = intersection_filter.get_debug_info()
                print(f"\n防抖动滤波器最终统计:")
                print(f"  总处理帧数: {debug_info['frame_count']}")

                # 根据滤波器类型显示不同信息
                if 'update_count' in debug_info:
                    # 阈值式滤波器
                    print(f"  更新次数: {debug_info['update_count']}")
                    print(f"  忽略次数: {debug_info['ignore_count']}")
                    print(f"  稳定率: {debug_info.get('stability_rate', 0):.1f}%")
                    print(f"  大运动事件: {debug_info['large_movement_count']}次")
                    print(f"  阈值参数: threshold={debug_info['parameters']['movement_threshold']}px")
                else:
                    # EMA滤波器
                    print(f"  大运动事件: {debug_info['large_movement_count']}次")
                    print(f"  边界限制事件: {debug_info['boundary_clamp_count']}次")
                    print(f"  稳定事件: {debug_info['stability_count']}次")
                    print(f"  EMA参数: alpha={debug_info['parameters']['alpha']}, "
                          f"threshold={debug_info['parameters']['movement_threshold']}px")
            except Exception as e:
                print(f"获取防抖动统计信息失败: {e}")


if __name__ == "__main__":
    run_daemon()
