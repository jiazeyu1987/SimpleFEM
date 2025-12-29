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

import json
import logging
import logging.handlers
import os
import platform
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob

from fem_refactor.analysis_cache import RoiAnalysisCache
from fem_refactor.anti_jitter_manager import AntiJitterManager
from fem_refactor.cleanup_manager import cleanup_directories
from fem_refactor.config_loader import load_fem_config
from fem_refactor.image_metrics import (
    compute_average_gray,
    compute_roi3_80_160_normalized,
    compute_roi3_column_mean_diff,
    compute_roi3_g1_g2_ranges,
)
from fem_refactor.logging_manager import setup_logging, setup_peak_logger
from fem_refactor.paths import get_base_dir
from fem_refactor.processing_mode_manager import initialize_processing_mode
from fem_refactor.roi_math import adjust_roi1_to_screen, compute_roi2_region
from fem_refactor.screen_source import capture_screen
from fem_refactor.intersection_manager import IntersectionManager
from fem_refactor.video_source import (
    discover_video_files,
    get_video_fps,
    get_video_frame,
    initialize_video_capture,
)
from fem_refactor.detection_pipeline import run_peak_detection_step
from fem_refactor.artifact_saver import save_frame_artifacts
from fem_refactor.stats_sink import add_peaks_to_statistics
from fem_refactor.frame_step import process_frame
from fem_refactor.models import (
    Buffers,
    ConfigValues,
    DaemonContext,
    Managers,
    Paths,
    Roi1ThresholdState,
    RuntimeState,
    ThresholdState,
    VideoState,
)
from fem_refactor.threshold_manager import update_roi1_threshold_state, update_roi2_threshold_state
from fem_refactor.threshold_protection import manage_threshold_protection
from fem_refactor.signal_buffers import (
    create_signal_buffers,
    reset_roi1_state,
    reset_video_state_variables,
)
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

from peak_detection import detect_peaks  # type: ignore  # noqa: E402
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
    # 配置日志系统（在清理之前，以便记录清理过程）
    log_file = setup_logging()
    logging.info("SimpleFEM ROI Daemon 启动...")
    print("SimpleFEM ROI Daemon 启动...")

    # 清理现有的数据文件夹
    cleanup_directories()

    config = load_fem_config()

    anti_jitter_config, intersection_filter = AntiJitterManager().build(config)

    # Optional: write a per-frame cache for later analysis / root-cause debugging
    analysis_cache_conf = config.get("analysis_cache", {})
    if not isinstance(analysis_cache_conf, dict):
        analysis_cache_conf = {}
    analysis_cache = RoiAnalysisCache(
        os.path.join(BASE_DIR, "export"),
        enabled=bool(analysis_cache_conf.get("enabled", True)),
        flush_every=int(analysis_cache_conf.get("flush_every", 50)),
    )

    processing_mode, video_cap, video_files, current_video_index, safe_statistics = initialize_processing_mode(
        config, statistics_manager
    )

    try:
        roi_default = config.get("roi_capture", {}).get("default_config", {})
        roi2_config = config.get("roi_capture", {}).get("roi2_config", {})
        extension_params = roi2_config.get("extension_params", {})

        # Load ROI3 configuration
        roi3_config = config.get("roi_capture", {}).get("roi3_config", {})
        roi3_extension_params = roi3_config.get("extension_params", {})

        data_processing = config.get("data_processing", {})
        save_roi1 = bool(data_processing.get("save_roi1", False))
        save_roi2 = bool(data_processing.get("save_roi2", False))
        save_roi3 = bool(data_processing.get("save_roi3", False))
        save_wave = bool(data_processing.get("save_wave", False))
        save_roi1_wave = bool(data_processing.get("save_roi1_wave", False))
        # only_delect == True: save ROI1/ROI2/wave only when peaks are detected
        only_delect = bool(data_processing.get("only_delect", False))

        peak_conf = config.get("peak_detection", {})
        threshold = float(peak_conf.get("threshold", 105.0))
        threshold_minimum = float(peak_conf.get("threshold_minimum", 80.0))
        margin_frames = int(peak_conf.get("margin_frames", 5))
        diff_threshold = float(peak_conf.get("difference_threshold", 0.5))
        # 新增：阈值前后"静默"帧数要求（升阈值前 X 帧和降阈值后 X 帧都不能超过阈值）
        silence_frames = int(peak_conf.get("silence_frames", 0))
        pre_post_avg_frames = int(peak_conf.get("pre_post_avg_frames", 5))
        # 自适应阈值参数
        adaptive_threshold_enabled = bool(peak_conf.get("adaptive_threshold_enabled", False))
        threshold_over_mean_ratio = float(peak_conf.get("threshold_over_mean_ratio", 0.15))
        adaptive_window_seconds = float(peak_conf.get("adaptive_window_seconds", 3.0))

        # 阈值保护参数
        protection_conf = peak_conf.get("threshold_protection", {})
        protection_enabled = bool(protection_conf.get("enabled", False))
        recovery_delay_seconds = float(protection_conf.get("recovery_delay_seconds", 1.0))
        stability_frames = int(protection_conf.get("stability_frames", 5))
        waveform_trigger_enabled = bool(protection_conf.get("waveform_trigger_enabled", True))

        min_region_length = int(peak_conf.get("min_region_length", 1))

        # ROI1 configuration parameters (independent from ROI2)
        roi1_peak_conf = config.get("roi1_peak_detection", {})
        roi1_enabled = bool(roi1_peak_conf.get("enabled", False))
        roi1_threshold = float(roi1_peak_conf.get("threshold", 120.0))
        roi1_threshold_minimum = float(roi1_peak_conf.get("threshold_minimum", 110.0))
        roi1_margin_frames = int(roi1_peak_conf.get("margin_frames", 5))
        roi1_silence_frames = int(roi1_peak_conf.get("silence_frames", 5))
        roi1_pre_post_avg_frames = int(roi1_peak_conf.get("pre_post_avg_frames", 5))
        roi1_difference_threshold = float(roi1_peak_conf.get("difference_threshold", 2.0))
        roi1_min_region_length = int(roi1_peak_conf.get("min_region_length", 5))

        # ROI1 adaptive threshold parameters
        roi1_adaptive_threshold_enabled = bool(roi1_peak_conf.get("adaptive_threshold_enabled", True))
        roi1_threshold_over_mean_ratio = float(roi1_peak_conf.get("threshold_over_mean_ratio", 0.08))
        roi1_adaptive_window_seconds = float(roi1_peak_conf.get("adaptive_window_seconds", 3.0))

        # ROI1 threshold protection parameters
        roi1_protection_conf = roi1_peak_conf.get("threshold_protection", {})
        roi1_protection_enabled = bool(roi1_protection_conf.get("enabled", True))
        roi1_recovery_delay_seconds = float(roi1_protection_conf.get("recovery_delay_seconds", 1.0))
        roi1_stability_frames = int(roi1_protection_conf.get("stability_frames", 5))
        roi1_waveform_trigger_enabled = bool(roi1_protection_conf.get("waveform_trigger_enabled", True))

        # 混合检测配置参数读取
        hybrid_conf = config.get("hybrid_detection", {})
        hybrid_enabled = bool(hybrid_conf.get("enabled", False))
        detection_strategy = hybrid_conf.get("detection_strategy", "roi1_peaks_roi2_color")
        fusion_strategy = hybrid_conf.get("fusion_strategy", "roi2_priority")

        # ROI2颜色判定配置
        roi2_color_config = hybrid_conf.get("roi2_color_frames", {})
        roi2_pre_frames = int(roi2_color_config.get("pre_peak", 5))
        roi2_post_frames = int(roi2_color_config.get("post_peak", 10))

        # ROI1波峰宽度验证配置
        peak_width_config = hybrid_conf.get("roi1_peak_width_range", [30, 40])
        min_peak_width = int(peak_width_config[0])
        max_peak_width = int(peak_width_config[1])

        # 数据质量检查配置
        data_quality_conf = hybrid_conf.get("data_quality", {})
        min_roi2_frames = int(data_quality_conf.get("minimum_roi2_frames", 15))
        roi2_min_variance = float(data_quality_conf.get("roi2_minimum_variance", 0.5))
        fallback_enabled = bool(hybrid_conf.get("fallback_enabled", True))

        # G1/G2 覆盖配置（新增）
        g1_g2_conf = peak_conf.get("g1_g2_override", {})
        g1_g2_override_enabled = bool(g1_g2_conf.get("enabled", True))
        g1_threshold = float(g1_g2_conf.get("g1_threshold", 98.0))
        g2_threshold = float(g1_g2_conf.get("g2_threshold", 20.0))
        use_peak_max_g1_g2 = bool(g1_g2_conf.get("use_peak_max", True))

        print(f"[G1/G2覆盖] 配置: enabled={g1_g2_override_enabled}, "
              f"G1>{g1_threshold}%, G2>{g2_threshold}%, "
              f"use_peak_max={use_peak_max_g1_g2}")

        logger = setup_peak_logger()
        # Store only the latest 100 gray values for waveform / peak detection
        gray_buffer: Deque[float]
        # Track a session-wide "background mean" using a gated incremental mean:
        # only update the mean when the current gray value is below the current
        # (mean-based) threshold, so peak frames do not contaminate the baseline.
        bg_count: int = 0
        bg_mean: float = 0.0
        last_intersection_roi: Optional[Tuple[int, int]] = None
        intersection_manager = IntersectionManager()
        frames_since_protection_end: int = 0

        # Threshold protection state management
        threshold_protection_active: bool = False
        protection_end_time: float = 0.0
        consecutive_below_threshold: int = 0
        last_waveform_time: float = 0.0

        # ROI1 independent buffer and state (parallel to ROI2)
        roi1_gray_buffer: Deque[float]
        roi1_bg_count: int = 0
        roi1_bg_mean: float = 0.0
        roi1_threshold_protection_active: bool = False
        roi1_protection_end_time: float = 0.0
        roi1_consecutive_below_threshold: int = 0
        roi1_last_waveform_time: float = 0.0

        # ROI3 independent buffer (same structure as ROI2)
        roi3_gray_buffer: Deque[float]
        roi3_80_160_buffer: Deque[float]
        roi3_g1_buffer: Deque[float]  # G1值缓冲区
        roi3_g2_buffer: Deque[float]  # G2值缓冲区
        roi3_column_diff_buffer: Deque[float]  # ROI3列灰度差值缓冲区

        (
            gray_buffer,
            roi1_gray_buffer,
            roi3_gray_buffer,
            roi3_80_160_buffer,
            roi3_g1_buffer,
            roi3_g2_buffer,
            roi3_column_diff_buffer,
        ) = create_signal_buffers(maxlen=100)

        # Initialize ROI1 threshold used so hybrid detection can reference it
        # before the per-frame ROI1 adaptive-threshold block runs.
        roi1_threshold_used: float = max(roi1_threshold, roi1_threshold_minimum)

        # ROI1波峰唯一ID管理机制 - 防止重复记录
        processed_roi1_peaks: Dict[Tuple[int, int], str] = {}  # {(start, end): peak_id}
        roi1_peak_counter: int = 0  # 唯一ID计数器

        # Prepare per-video image save directories if enabled
        if processing_mode == "video" and video_files:
            # Video mode: Use first video for initial folder creation
            current_stats = statistics_manager.current_statistics
            if current_stats and current_stats.video_name:
                tmp_root = _create_video_folders(
                    video_files[0],
                    current_stats.session_id,
                    processing_mode,
                    save_roi1,
                    save_roi2,
                    save_roi3,
                    save_wave,
                    save_roi1_wave
                )
                # 关键修复：更新ROI保存路径变量
                roi1_dir = os.path.join(tmp_root, "roi1")
                roi2_dir = os.path.join(tmp_root, "roi2")
                roi3_dir = os.path.join(tmp_root, "roi3")
                wave_dir = os.path.join(tmp_root, "wave")
                wave1_dir = os.path.join(tmp_root, "wave1")
            else:
                # Fallback for screen mode or if video stats not initialized
                session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
                tmp_root = os.path.join(BASE_DIR, "tmp", session_start)
                if save_roi1 or save_roi2 or save_wave:
                    os.makedirs(tmp_root, exist_ok=True)
                if save_roi1:
                    os.makedirs(os.path.join(tmp_root, "roi1"), exist_ok=True)
                if save_roi2:
                    os.makedirs(os.path.join(tmp_root, "roi2"), exist_ok=True)
                if save_wave:
                    os.makedirs(os.path.join(tmp_root, "wave"), exist_ok=True)
        else:
            # Screen mode: Use original session-based naming
            session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp_root = os.path.join(BASE_DIR, "tmp", session_start)
            roi1_dir = os.path.join(tmp_root, "roi1")
            roi2_dir = os.path.join(tmp_root, "roi2")
            roi3_dir = os.path.join(tmp_root, "roi3")
            wave_dir = os.path.join(tmp_root, "wave")

            if save_roi1 or save_roi2 or save_roi3 or save_wave:
                os.makedirs(tmp_root, exist_ok=True)
            if save_roi1:
                os.makedirs(roi1_dir, exist_ok=True)
            if save_roi2:
                os.makedirs(roi2_dir, exist_ok=True)
            if save_roi3:
                os.makedirs(roi3_dir, exist_ok=True)
            if save_wave:
                os.makedirs(wave_dir, exist_ok=True)

        frame_index = 0

        # Use roi_capture.frame_rate as loop frequency
        roi_frame_rate = config.get("roi_capture", {}).get("frame_rate", 1)
        try:
            roi_frame_rate = float(roi_frame_rate)
        except Exception:
            roi_frame_rate = 1.0
        if roi_frame_rate <= 0:
            roi_frame_rate = 1.0
        # Video mode: make roi_capture.frame_rate control sampling on the video timeline
        # (skip frames based on source FPS), while keeping screen-capture mode unchanged.
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

        # 调试信息：打印帧率配置
        print(f"[帧率配置] 配置帧率: {roi_frame_rate} fps")
        print(f"[帧率配置] 计算间隔: {interval_seconds:.3f} 秒/帧")
        print(f"[帧率配置] 预期7秒视频处理: {7 * roi_frame_rate} 帧")

        # Calculate adaptive window frame count based on time window and frame rate
        adaptive_window_frames = int(adaptive_window_seconds * effective_frame_rate)
        # Ensure at least 1 frame and not exceed buffer size
        adaptive_window_frames = max(1, min(adaptive_window_frames, 100))

        # Calculate recovery delay in frames
        recovery_delay_frames = int(recovery_delay_seconds * effective_frame_rate)
        recovery_delay_frames = max(1, recovery_delay_frames)

        # Start cache session (one file per SafePeakStatistics session/video)
        try:
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
                    "gray_buffer_maxlen": 100,
                },
            )
            if analysis_cache.path:
                print(f"[cache] analysis_cache={analysis_cache.path}")
        except Exception:
            pass

        ctx = DaemonContext(
            cfg=ConfigValues(
                raw=config,
                roi_default=roi_default,
                extension_params=extension_params,
                roi3_extension_params=roi3_extension_params,
                save_roi1=save_roi1,
                save_roi2=save_roi2,
                save_roi3=save_roi3,
                save_wave=save_wave,
                save_roi1_wave=save_roi1_wave,
                only_delect=only_delect,
                threshold=threshold,
                threshold_minimum=threshold_minimum,
                margin_frames=margin_frames,
                diff_threshold=diff_threshold,
                silence_frames=silence_frames,
                pre_post_avg_frames=pre_post_avg_frames,
                min_region_length=min_region_length,
                adaptive_threshold_enabled=adaptive_threshold_enabled,
                threshold_over_mean_ratio=threshold_over_mean_ratio,
                adaptive_window_seconds=adaptive_window_seconds,
                adaptive_window_frames=adaptive_window_frames,
                protection_enabled=protection_enabled,
                recovery_delay_seconds=recovery_delay_seconds,
                recovery_delay_frames=recovery_delay_frames,
                stability_frames=stability_frames,
                waveform_trigger_enabled=waveform_trigger_enabled,
                roi1_enabled=roi1_enabled,
                roi1_threshold=roi1_threshold,
                roi1_threshold_minimum=roi1_threshold_minimum,
                roi1_margin_frames=roi1_margin_frames,
                roi1_silence_frames=roi1_silence_frames,
                roi1_pre_post_avg_frames=roi1_pre_post_avg_frames,
                roi1_difference_threshold=roi1_difference_threshold,
                roi1_min_region_length=roi1_min_region_length,
                roi1_adaptive_threshold_enabled=roi1_adaptive_threshold_enabled,
                roi1_threshold_over_mean_ratio=roi1_threshold_over_mean_ratio,
                roi1_adaptive_window_seconds=roi1_adaptive_window_seconds,
                roi1_protection_enabled=roi1_protection_enabled,
                roi1_recovery_delay_seconds=roi1_recovery_delay_seconds,
                roi1_stability_frames=roi1_stability_frames,
                roi1_waveform_trigger_enabled=roi1_waveform_trigger_enabled,
                hybrid_enabled=hybrid_enabled,
                roi2_pre_frames=roi2_pre_frames,
                roi2_post_frames=roi2_post_frames,
                min_roi2_frames=min_roi2_frames,
                roi2_min_variance=roi2_min_variance,
                fallback_enabled=fallback_enabled,
                max_peak_width=max_peak_width,
                data_quality_conf=data_quality_conf,
                hybrid_conf=hybrid_conf,
                g1_g2_override_enabled=g1_g2_override_enabled,
                g1_threshold=g1_threshold,
                g2_threshold=g2_threshold,
                use_peak_max_g1_g2=use_peak_max_g1_g2,
            ),
            video=VideoState(
                processing_mode=processing_mode,
                video_cap=video_cap,
                video_files=list(video_files) if video_files else [],
                current_video_index=int(current_video_index),
                video_fps=float(video_fps),
                video_frame_step=int(video_frame_step),
                first_video_frame=bool(first_video_frame),
                effective_frame_rate=float(effective_frame_rate),
                interval_seconds=float(interval_seconds),
            ),
            paths=Paths(
                base_dir=BASE_DIR,
                tmp_root=str(tmp_root),
                roi1_dir=str(roi1_dir),
                roi2_dir=str(roi2_dir),
                roi3_dir=str(roi3_dir),
                wave_dir=str(wave_dir),
                wave1_dir=str(wave1_dir),
            ),
            buffers=Buffers(
                gray_buffer=gray_buffer,
                roi1_gray_buffer=roi1_gray_buffer,
                roi3_gray_buffer=roi3_gray_buffer,
                roi3_80_160_buffer=roi3_80_160_buffer,
                roi3_g1_buffer=roi3_g1_buffer,
                roi3_g2_buffer=roi3_g2_buffer,
                roi3_column_diff_buffer=roi3_column_diff_buffer,
            ),
            thr=ThresholdState(
                bg_count=int(bg_count),
                bg_mean=float(bg_mean),
                frames_since_protection_end=int(frames_since_protection_end),
                threshold_protection_active=bool(threshold_protection_active),
                protection_end_time=float(protection_end_time),
                consecutive_below_threshold=int(consecutive_below_threshold),
                last_waveform_time=float(last_waveform_time),
            ),
            roi1_thr=Roi1ThresholdState(
                bg_count=int(roi1_bg_count),
                bg_mean=float(roi1_bg_mean),
                threshold_protection_active=bool(roi1_threshold_protection_active),
                protection_end_time=float(roi1_protection_end_time),
                consecutive_below_threshold=int(roi1_consecutive_below_threshold),
                last_waveform_time=float(roi1_last_waveform_time),
                threshold_used=float(roi1_threshold_used),
            ),
            managers=Managers(
                statistics_manager=statistics_manager,
                analysis_cache=analysis_cache,
                intersection_manager=intersection_manager,
                intersection_filter=intersection_filter,
                anti_jitter_config=anti_jitter_config,
                logger=logger,
            ),
            state=RuntimeState(
                frame_index=int(frame_index),
                last_intersection_roi=last_intersection_roi,
                processed_roi1_peaks=processed_roi1_peaks,
                roi1_peak_counter=int(roi1_peak_counter),
            ),
        )

        while True:
            loop_start = time.time()
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            log_line: Optional[str] = None

            try:
                frame_index += 1

                # 1. Capture image (screen or video frame)
                if processing_mode == "video":
                    video_config = config.get("video_processing", {})
                    loop_enabled = video_config.get("loop_enabled", False)
                    # First frame uses step=1 to avoid skipping the very beginning.
                    step = 1 if first_video_frame else video_frame_step
                    first_video_frame = False
                    screen = get_video_frame(video_cap, loop_enabled, frame_step=step)
                    if screen is None:
                        # 当前视频播放结束
                        current_video_index += 1

                        # 释放当前视频资源
                        video_cap.release()

                        if current_video_index < len(video_files):
                            # 切换到下一个视频
                            next_video_path = video_files[current_video_index]
                            try:
                                # 为此视频初始化新的统计
                                current_stats = statistics_manager.initialize_for_video(
                                    next_video_path,
                                    is_batch=True
                                )

                                # 重置防抖动滤波器状态（用于新视频）
                                if intersection_filter:
                                    # 保存当前调试信息
                                    old_debug_info = intersection_filter.get_debug_info()
                                    intersection_filter.reset()
                                    print(f"已重置ROI2防抖动滤波器，切换到新视频: {os.path.basename(next_video_path)}")

                                    # 根据滤波器类型显示不同的统计信息
                                    if 'update_count' in old_debug_info:
                                        # 阈值式滤波器
                                        print(f"上一个视频的阈值防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                                              f"更新{old_debug_info['update_count']}次, "
                                              f"稳定率{old_debug_info.get('stability_rate', 0):.1f}%")
                                    else:
                                        # EMA滤波器
                                        print(f"上一个视频的EMA防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                                              f"大运动{old_debug_info['large_movement_count']}次, "
                                              f"边界限制{old_debug_info.get('boundary_clamp_count', 0)}次")

                                # 重置全局状态变量（防止数据污染）
                                gray_buffer.clear()
                                roi1_gray_buffer.clear()
                                roi3_gray_buffer.clear()
                                roi3_80_160_buffer.clear()
                                roi3_g1_buffer.clear()  # 清空G1缓冲区
                                roi3_g2_buffer.clear()  # 清空G2缓冲区
                                roi3_column_diff_buffer.clear()  # 清空列灰度差值缓冲区
                                reset_values = reset_video_state_variables(gray_buffer)
                                (bg_count, bg_mean, last_intersection_roi, frames_since_protection_end,
                                 threshold_protection_active, protection_end_time, consecutive_below_threshold,
                                 last_waveform_time, frame_index, first_video_frame) = reset_values

                                # 重置ROI1状态变量
                                (
                                    roi1_bg_count,
                                    roi1_bg_mean,
                                    roi1_threshold_protection_active,
                                    roi1_protection_end_time,
                                    roi1_consecutive_below_threshold,
                                    roi1_last_waveform_time,
                                    roi1_threshold_used,
                                ) = reset_roi1_state(
                                    roi1_threshold=roi1_threshold,
                                    roi1_threshold_minimum=roi1_threshold_minimum,
                                )

                                # 重置ROI1波峰ID管理机制
                                processed_roi1_peaks.clear()
                                roi1_peak_counter = 0

                                print(f"已重置全局和ROI1状态变量，确保数据隔离")

                                video_cap = initialize_video_capture(next_video_path)
                                print(f"\n" + "="*50)
                                print(f"开始处理下一个视频 ({current_video_index + 1}/{len(video_files)}):")
                                print(f"文件名: {os.path.basename(next_video_path)}")
                                print(f"统计会话: {current_stats.session_id}")

                                # 重新计算新视频的帧率参数
                                video_fps = get_video_fps(video_cap)
                                if video_fps > 0:
                                    effective_frame_rate = min(roi_frame_rate, video_fps)
                                    if effective_frame_rate > 0:
                                        video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))

                                # 创建每视频文件夹结构
                                tmp_root = _create_video_folders(
                                    next_video_path,
                                    current_stats.session_id,
                                    processing_mode,
                                    save_roi1,
                                    save_roi2,
                                    save_roi3,
                                    save_wave,
                                    save_roi1_wave
                                )

                                # 关键修复：更新ROI保存路径变量
                                roi1_dir = os.path.join(tmp_root, "roi1")
                                roi2_dir = os.path.join(tmp_root, "roi2")
                                roi3_dir = os.path.join(tmp_root, "roi3")
                                wave_dir = os.path.join(tmp_root, "wave")
                                wave1_dir = os.path.join(tmp_root, "wave1")

                                print(f"[video] source_fps={video_fps:.2f} target_fps={effective_frame_rate:.2f} frame_step={video_frame_step}")
                                print(f"[folders] tmp_root={tmp_root}")
                                print(f"[folders] roi1_dir={roi1_dir}")
                                print(f"[folders] roi2_dir={roi2_dir}")
                                print(f"[folders] wave_dir={wave_dir}")
                                print("="*50)

                                # 重置帧索引和首帧标志
                                frame_index = 0
                                first_video_frame = True

                                # Keep ctx in sync after switching videos
                                ctx.state.frame_index = int(frame_index)
                                ctx.state.last_intersection_roi = last_intersection_roi
                                ctx.state.processed_roi1_peaks = processed_roi1_peaks
                                ctx.state.roi1_peak_counter = int(roi1_peak_counter)

                                ctx.thr.bg_count = int(bg_count)
                                ctx.thr.bg_mean = float(bg_mean)
                                ctx.thr.frames_since_protection_end = int(frames_since_protection_end)
                                ctx.thr.threshold_protection_active = bool(threshold_protection_active)
                                ctx.thr.protection_end_time = float(protection_end_time)
                                ctx.thr.consecutive_below_threshold = int(consecutive_below_threshold)
                                ctx.thr.last_waveform_time = float(last_waveform_time)

                                ctx.roi1_thr.bg_count = int(roi1_bg_count)
                                ctx.roi1_thr.bg_mean = float(roi1_bg_mean)
                                ctx.roi1_thr.threshold_protection_active = bool(roi1_threshold_protection_active)
                                ctx.roi1_thr.protection_end_time = float(roi1_protection_end_time)
                                ctx.roi1_thr.consecutive_below_threshold = int(roi1_consecutive_below_threshold)
                                ctx.roi1_thr.last_waveform_time = float(roi1_last_waveform_time)
                                ctx.roi1_thr.threshold_used = float(roi1_threshold_used)

                                ctx.video.video_cap = video_cap
                                ctx.video.current_video_index = int(current_video_index)
                                ctx.video.video_fps = float(video_fps)
                                ctx.video.video_frame_step = int(video_frame_step)
                                ctx.video.first_video_frame = bool(first_video_frame)
                                ctx.video.effective_frame_rate = float(effective_frame_rate)

                                ctx.paths.tmp_root = tmp_root
                                ctx.paths.roi1_dir = roi1_dir
                                ctx.paths.roi2_dir = roi2_dir
                                ctx.paths.roi3_dir = roi3_dir
                                ctx.paths.wave_dir = wave_dir
                                ctx.paths.wave1_dir = wave1_dir

                                # Start a new cache session for the new video/statistics session
                                try:
                                    analysis_cache.start_session(
                                        current_stats.session_id,
                                        processing_mode=processing_mode,
                                        video_path=next_video_path,
                                        config=config,
                                        extra_meta={
                                            "roi_frame_rate": roi_frame_rate,
                                            "effective_frame_rate": effective_frame_rate,
                                            "video_fps": video_fps,
                                            "video_frame_step": video_frame_step,
                                            "adaptive_window_frames": adaptive_window_frames,
                                            "gray_buffer_maxlen": 100,
                                        },
                                    )
                                    if analysis_cache.path:
                                        print(f"[cache] analysis_cache={analysis_cache.path}")
                                except Exception:
                                    pass

                                # 继续处理下一个视频，不break
                                continue
                            except Exception as e:
                                print(f"无法打开下一个视频 {next_video_path}: {e}")
                                print("继续处理下一个视频...")
                                continue
                        else:
                            # 所有视频都处理完毕
                            total_time = time.time() - (loop_start - (frame_index * interval_seconds))
                            actual_fps = frame_index / total_time if total_time > 0 else 0
                            msg = f"\n" + "="*50
                            print(msg)
                            logging.info(msg)
                            msg = f"所有视频处理完成！"
                            print(msg)
                            logging.info(msg)
                            msg = f"[统计] 总处理时间: {total_time:.2f} 秒"
                            print(msg)
                            logging.info(msg)
                            msg = f"[统计] 总处理视频数: {len(video_files)}"
                            print(msg)
                            logging.info(msg)
                            msg = f"[统计] 总处理帧数: {frame_index}"
                            print(msg)
                            logging.info(msg)
                            msg = f"[统计] 平均帧率: {actual_fps:.2f} fps"
                            print(msg)
                            logging.info(msg)
                            msg = f"[统计] 配置帧率: {roi_frame_rate:.2f} fps"
                            print(msg)
                            logging.info(msg)
                            msg = "="*50
                            print(msg)
                            logging.info(msg)
                            break
                    screen_width, screen_height = screen.size
                else:
                    screen = capture_screen()
                    screen_width, screen_height = screen.size

                video_seconds: Optional[float] = None
                if processing_mode == "video" and video_cap is not None:
                    try:
                        video_pos_msec = float(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                        if video_pos_msec >= 0:
                            video_seconds = video_pos_msec / 1000.0
                    except Exception:
                        video_seconds = None

                # Sync live locals into ctx (ctx is the source-of-truth for frame processing)
                ctx.state.frame_index = frame_index
                ctx.video.processing_mode = processing_mode
                ctx.video.video_cap = video_cap
                ctx.video.current_video_index = current_video_index
                ctx.video.video_fps = video_fps
                ctx.video.video_frame_step = video_frame_step
                ctx.video.first_video_frame = first_video_frame
                ctx.video.effective_frame_rate = effective_frame_rate
                ctx.video.interval_seconds = interval_seconds
                ctx.paths.tmp_root = tmp_root
                ctx.paths.roi1_dir = roi1_dir
                ctx.paths.roi2_dir = roi2_dir
                ctx.paths.roi3_dir = roi3_dir
                ctx.paths.wave_dir = wave_dir
                ctx.paths.wave1_dir = wave1_dir

                step_result = process_frame(
                    ctx=ctx,
                    screen=screen,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    loop_start=loop_start,
                    ts=ts,
                    video_seconds=video_seconds,
                )
                log_line = step_result.log_line
            except KeyboardInterrupt:
                logger.info(f"{ts} INFO=daemon_stopped_by_user")
                break
            except Exception as e:
                # Log unexpected error but keep daemon alive
                log_line = f"{ts} ERROR={repr(e)}"

            if log_line is not None:
                logger.info(log_line)

            # Maintain ~1-second interval between iterations
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, interval_seconds - elapsed)

            # 调试信息：每10帧打印一次帧率控制信息
            if frame_index % 10 == 0:
                print(f"[帧率调试] 帧{frame_index}: 目标间隔={interval_seconds:.3f}s, 实际耗时={elapsed:.3f}s, 睡眠时间={sleep_time:.3f}s")

            time.sleep(sleep_time)

    finally:
        try:
            analysis_cache.close(reason="shutdown")
        except Exception:
            pass
        # 释放视频资源
        if video_cap is not None:
            video_cap.release()
            print("视频资源已释放")

        # 输出防抖动滤波器最终统计信息
        if intersection_filter:
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
    try:
        run_daemon()
    except KeyboardInterrupt:
        # 程序结束时导出最终CSV文件（task要求）
        msg = "\n数据处理完成，CSV文件已保存..."
        print(msg)
        logging.info(msg)
        try:
            # 获取当前统计文件路径
            current_stats = statistics_manager.current_statistics
            if current_stats:
                export_path = current_stats.export_final_csv()
                if export_path:
                    msg = f"✅ 当前视频CSV文件已保存至: {export_path}"
                    print(msg)
                    logging.info(msg)

            # 显示所有视频的统计摘要
            global_summary = statistics_manager.get_global_summary()
            msg = f"📊 批量处理统计摘要:"
            print(msg)
            logging.info(msg)
            msg = f"   总处理视频数: {global_summary.get('total_videos_processed', 0)}"
            print(msg)
            logging.info(msg)
            msg = f"   总波峰数: {global_summary.get('total_peaks', 0)}"
            print(msg)
            logging.info(msg)
            msg = f"   绿色波峰: {global_summary.get('total_green_peaks', 0)}"
            print(msg)
            logging.info(msg)
            msg = f"   红色波峰: {global_summary.get('total_red_peaks', 0)}"
            print(msg)
            logging.info(msg)
            msg = f"   会话时长: {global_summary.get('session_duration', 'N/A')}"
            print(msg)
            logging.info(msg)

            # 显示每个视频的详细信息
            videos_processed = global_summary.get('videos_processed', [])
            if videos_processed:
                msg = f"   处理的视频: {', '.join(videos_processed)}"
                print(msg)
                logging.info(msg)

        except Exception as e:
            msg = f"❌ 处理CSV文件时发生错误: {e}"
            print(msg)
            logging.error(msg)

        msg = "守护进程已停止"
        print(msg)
        logging.info(msg)
    except Exception as e:
        print(f"❌ 守护进程运行时发生错误: {e}")
        # 即使出错也尝试保存数据
        try:
            current_stats = statistics_manager.current_statistics
            if current_stats:
                export_path = current_stats.export_final_csv()
                if export_path:
                    print(f"✅ 异常停止前数据已保存至: {export_path}")
        except Exception:
            pass
