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

                # 2. Get ROI1 region and crop
                x1, y1, x2, y2 = adjust_roi1_to_screen(
                    (screen_width, screen_height),
                    roi_default,
                )
                roi1_image = screen.crop((x1, y1, x2, y2))
                roi1_width, roi1_height = roi1_image.size

                # Initialize ROI3 statistics variables
                roi3_g1: Optional[float] = None
                roi3_g2: Optional[float] = None

                # 3. Detect green line intersection in ROI1
                intersection, (center_x, center_y) = intersection_manager.detect_and_get_center(
                    roi1_image=roi1_image,
                    anti_jitter_config=anti_jitter_config,
                    intersection_filter=intersection_filter,
                )
                last_intersection_roi = intersection_manager.last_intersection_roi

                # 4. Compute ROI2 region and crop
                roi2_region = compute_roi2_region(
                    (roi1_width, roi1_height),
                    (center_x, center_y),
                    extension_params,
                )

                roi2_gray: Optional[float] = None
                roi2_image: Optional[Image.Image] = None

                if roi2_region is not None:
                    rx1, ry1, rx2, ry2 = roi2_region
                    roi2_image = roi1_image.crop((rx1, ry1, rx2, ry2))
                    roi2_gray = compute_average_gray(roi2_image)
                    gray_buffer.append(roi2_gray)

                    # ROI3 extraction (independent from ROI2)
                    roi3_gray: Optional[float] = None
                    roi3_image: Optional[Image.Image] = None
                    if roi3_extension_params:
                        roi3_region = compute_roi2_region(
                            (roi1_width, roi1_height),
                            (center_x, center_y),
                            roi3_extension_params,
                        )
                        if roi3_region is not None:
                            r3x1, r3y1, r3x2, r3y2 = roi3_region
                            roi3_image = roi1_image.crop((r3x1, r3y1, r3x2, r3y2))
                            roi3_gray = compute_average_gray(roi3_image)
                            roi3_gray_buffer.append(roi3_gray)
                            print(f"[DEBUG] ROI3 captured: frame={frame_index}, gray={roi3_gray:.2f}, buffer_len={len(roi3_gray_buffer)}")
                            print(f"[DEBUG] ROI3 coords: ({r3x1}, {r3y1}, {r3x2}, {r3y2}), size={r3x2-r3x1}x{r3y2-r3y1}, center=({center_x}, {center_y})")

                            # Compute normalized pixel count for range [80, 160]
                            roi3_80_160_normalized = compute_roi3_80_160_normalized(roi3_image)
                            roi3_80_160_buffer.append(roi3_80_160_normalized)
                            print(f"[DEBUG] ROI3(80-160)%: frame={frame_index}, percentage={roi3_80_160_normalized:.2f}%, buffer_len={len(roi3_80_160_buffer)}")

                            # Compute G1 and G2 ranges
                            g1, g2 = compute_roi3_g1_g2_ranges(roi3_image)
                            roi3_g1 = g1  # Save for cache recording
                            roi3_g2 = g2  # Save for cache recording
                            roi3_g1_buffer.append(g1)  # 存入G1缓冲区
                            roi3_g2_buffer.append(g2)  # 存入G2缓冲区
                            msg = f"[STAT] 帧{frame_index} G1(80-255)={g1:.2f}%, G2(150-255)={g2:.2f}%"
                            logging.debug(msg)
                            print(msg)

                            # 计算ROI3列灰度差值
                            roi3_column_diff = compute_roi3_column_mean_diff(roi3_image)
                            roi3_column_diff_buffer.append(roi3_column_diff)
                            msg = f"[STAT] 帧{frame_index} ROI3列灰度差值: {roi3_column_diff:.2f}"
                            logging.debug(msg)
                            print(msg)
                        else:
                            print(f"[DEBUG] ROI3 extraction failed: frame={frame_index}, intersection={intersection}, roi3_extension_params={roi3_extension_params}")
                    else:
                        print(f"[DEBUG] ROI3 extension params not available")

                    # ROI1 gray value calculation (independent from ROI2)
                    roi1_gray: Optional[float] = None
                    if roi1_enabled:
                        roi1_gray = compute_average_gray(roi1_image)
                        roi1_gray_buffer.append(roi1_gray)

                # 5. Run peak detection on current gray buffer
                green_peaks: List[Tuple[int, int]] = []
                red_peaks: List[Tuple[int, int]] = []
                green_peaks_raw: List[Tuple[int, int]] = []
                red_peaks_raw: List[Tuple[int, int]] = []
                (
                    threshold_used,
                    recent_frames_count,
                    calculated_bg_mean,
                    bg_mean,
                    bg_count,
                    threshold_protection_active,
                    protection_end_time,
                    consecutive_below_threshold,
                    frames_since_protection_end,
                    last_waveform_time,
                ) = update_roi2_threshold_state(
                    gray_buffer=gray_buffer,
                    adaptive_threshold_enabled=adaptive_threshold_enabled,
                    adaptive_window_frames=adaptive_window_frames,
                    threshold=threshold,
                    threshold_minimum=threshold_minimum,
                    threshold_over_mean_ratio=threshold_over_mean_ratio,
                    roi2_gray=roi2_gray,
                    frame_index=frame_index,
                    protection_enabled=protection_enabled,
                    recovery_delay_frames=recovery_delay_frames,
                    stability_frames=stability_frames,
                    waveform_trigger_enabled=waveform_trigger_enabled,
                    threshold_protection_active=threshold_protection_active,
                    protection_end_time=protection_end_time,
                    consecutive_below_threshold=consecutive_below_threshold,
                    frames_since_protection_end=frames_since_protection_end,
                    last_waveform_time=last_waveform_time,
                    bg_mean=bg_mean,
                    bg_count=bg_count,
                 )

                (detection_mode,
                 hybrid_peaks,
                 green_peaks_raw,
                 red_peaks_raw,
                 green_peaks,
                 red_peaks,
                ) = run_peak_detection_step(
                    frame_index=frame_index,
                    hybrid_enabled=hybrid_enabled,
                    roi1_enabled=roi1_enabled,
                    roi1_gray_buffer=roi1_gray_buffer,
                    gray_buffer=gray_buffer,
                    roi1_threshold_used=roi1_threshold_used,
                    roi1_margin_frames=roi1_margin_frames,
                    roi1_silence_frames=roi1_silence_frames,
                    roi1_pre_post_avg_frames=roi1_pre_post_avg_frames,
                    roi1_min_region_length=roi1_min_region_length,
                    max_peak_width=max_peak_width,
                    roi2_pre_frames=roi2_pre_frames,
                    roi2_post_frames=roi2_post_frames,
                    min_roi2_frames=min_roi2_frames,
                    roi2_min_variance=roi2_min_variance,
                    diff_threshold=diff_threshold,
                    fallback_enabled=fallback_enabled,
                    hybrid_conf=hybrid_conf,
                    data_quality_conf=data_quality_conf,
                    intersection=intersection,
                    g1_g2_override_enabled=g1_g2_override_enabled,
                    g1_threshold=g1_threshold,
                    g2_threshold=g2_threshold,
                    use_peak_max_g1_g2=use_peak_max_g1_g2,
                    roi3_g1_buffer=roi3_g1_buffer,
                    roi3_g2_buffer=roi3_g2_buffer,
                    roi3_column_diff_buffer=roi3_column_diff_buffer,
                    processed_roi1_peaks=processed_roi1_peaks,
                    roi1_peak_counter=roi1_peak_counter,
                    threshold_used=threshold_used,
                    margin_frames=margin_frames,
                    silence_frames=silence_frames,
                    pre_post_avg_frames=pre_post_avg_frames,
                    min_region_length=min_region_length,
                )

                # Re-check threshold protection with actual peak detection results
                if protection_enabled and roi2_gray is not None:
                    has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
                    current_time = time.time()

                    (threshold_protection_active, protection_end_time,
                     consecutive_below_threshold, frames_since_protection_end,
                     last_waveform_time) = manage_threshold_protection(
                        current_gray=roi2_gray,
                        current_threshold=threshold_used,
                        has_peaks=has_peaks,
                        frame_time=current_time,
                        frame_index=frame_index,
                        protection_active=threshold_protection_active,
                        protection_end_time=protection_end_time,
                        consecutive_below=consecutive_below_threshold,
                        last_waveform_time=last_waveform_time,
                        enabled=protection_enabled,
                        recovery_delay_frames=recovery_delay_frames,
                        stability_frames=stability_frames,
                        waveform_trigger=waveform_trigger_enabled,
                        threshold_minimum=threshold_minimum,
                    )

                  
                # ROI1 adaptive threshold calculation (independent from ROI2)
                roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)
                roi1_curve = list(roi1_gray_buffer) if roi1_gray_buffer else []
                roi1_threshold_used, roi1_bg_mean, roi1_bg_count = update_roi1_threshold_state(
                    roi1_enabled=roi1_enabled,
                    roi1_gray_buffer=roi1_gray_buffer,
                    roi1_gray=roi1_gray,
                    frame_index=frame_index,
                    effective_frame_rate=effective_frame_rate,
                    roi1_threshold=roi1_threshold,
                    roi1_threshold_minimum=roi1_threshold_minimum,
                    roi1_threshold_over_mean_ratio=roi1_threshold_over_mean_ratio,
                    roi1_adaptive_threshold_enabled=roi1_adaptive_threshold_enabled,
                    roi1_adaptive_window_seconds=roi1_adaptive_window_seconds,
                    roi1_threshold_protection_active=roi1_threshold_protection_active,
                    roi1_bg_mean=roi1_bg_mean,
                    roi1_bg_count=roi1_bg_count,
                )

                green_count = len(green_peaks)
                red_count = len(red_peaks)
                last_green = green_peaks[-1] if green_peaks else None
                last_green_repr = (
                    f"[{last_green[0]},{last_green[1]}]" if last_green else "[]"
                )

                gray_str = (
                    f"{roi2_gray:.1f}" if roi2_gray is not None else "nan"
                )

                # Add peaks to statistics for Excel data collection (task requirement)
                stats_write_results: List[Dict[str, Any]] = []
                try:
                    # Prepare ROI2 information for statistics
                    roi2_info = None
                    if roi2_region is not None:
                        rx1, ry1, rx2, ry2 = roi2_region
                        roi2_info = {
                            'x1': rx1, 'y1': ry1, 'x2': rx2, 'y2': ry2,
                            'width': rx2 - rx1, 'height': ry2 - ry1
                        }

                    # Add detected peaks to statistics with deduplication
                    current_stats = statistics_manager.current_statistics
                    if current_stats:
                        # 准备ROI1曲线数据用于混合检测统计
                        roi1_curve_for_stats = list(roi1_gray_buffer) if roi1_gray_buffer else []

                        # 调用扩展的add_peaks_from_daemon方法，支持混合检测
                        stats_write_results = current_stats.add_peaks_from_daemon(
                            frame_index=frame_index,
                            green_peaks=green_peaks,
                            red_peaks=red_peaks,
                            curve=list(gray_buffer) if gray_buffer else [],
                            intersection=last_intersection_roi,
                            roi2_info=roi2_info,
                            gray_value=roi2_gray,
                            difference_threshold=diff_threshold,
                            pre_post_avg_frames=pre_post_avg_frames,
                            threshold_used=threshold_used,
                            bg_mean=(bg_mean if bg_count > 0 else None),
                            # 混合检测参数
                            hybrid_enabled=hybrid_enabled,
                            hybrid_peaks=hybrid_peaks,
                            roi1_curve=roi1_curve_for_stats,
                            roi1_threshold_used=roi1_threshold_used,
                            # ROI3数据（用于统计）
                            roi3_curve=list(roi3_gray_buffer) if roi3_gray_buffer else []
                        )

                except Exception as e:
                    # Keep daemon running even if statistics collection fails
                    print(f"Statistics collection error: {e}")

                # Decide whether to save images/wave for this frame
                has_peak = (green_count > 0) or (red_count > 0)
                should_save = (not only_delect) or has_peak

                # For ROI1, save waveforms when data is available (independent of ROI2 peaks)
                roi1_should_save = (not only_delect) or (len(roi1_gray_buffer) > 0)

                # Write a per-frame cache record for later Q&A / root cause analysis
                try:
                    buffer_len = len(gray_buffer)
                    buffer_start_frame = max(0, frame_index - buffer_len + 1)

                    def _peaks_to_abs(peaks: List[Tuple[int, int]]) -> List[Dict[str, int]]:
                        out: List[Dict[str, int]] = []
                        for s, e in peaks:
                            out.append(
                                {
                                    "buffer_start": int(s),
                                    "buffer_end": int(e),
                                    "abs_start": int(buffer_start_frame + s),
                                    "abs_end": int(buffer_start_frame + e),
                                }
                            )
                        return out

                    analysis_cache.record_frame(
                        {
                            "ts_wall": loop_start,
                            "ts_local": ts,
                            "frame_index": int(frame_index),
                            "video_seconds": video_seconds,
                            "screen_size": [int(screen_width), int(screen_height)],
                            "roi1": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                            "intersection": {"current": intersection, "used": last_intersection_roi},
                            "roi2_region": roi2_region,
                            "roi2_gray": roi2_gray,
                            "roi3": {
                                "g1": float(roi3_g1) if roi3_g1 is not None else None,
                                "g2": float(roi3_g2) if roi3_g2 is not None else None,
                                "gray": float(roi3_gray) if roi3_gray is not None else None,
                                "column_diff": float(roi3_column_diff) if roi3_column_diff is not None else None,
                            },
                            "buffer": {
                                "len": int(buffer_len),
                                "start_frame_index": int(buffer_start_frame),
                                "maxlen": 100,
                            },
                            "threshold": {
                                "fixed": float(threshold),
                                "minimum": float(threshold_minimum),
                                "used": float(threshold_used),
                                "adaptive_enabled": bool(adaptive_threshold_enabled),
                                "adaptive_window_frames": int(adaptive_window_frames),
                                "recent_frames_count": recent_frames_count,
                                "calculated_bg_mean": calculated_bg_mean,
                                "bg_mean": (float(bg_mean) if bg_count > 0 else None),
                                "bg_count": int(bg_count),
                                "protection_active": bool(threshold_protection_active),
                                "consecutive_below_threshold": int(consecutive_below_threshold),
                                "frames_since_protection_end": int(frames_since_protection_end),
                            },
                            "detect_params": {
                                "margin_frames": int(margin_frames),
                                "silence_frames": int(silence_frames),
                                "difference_threshold": float(diff_threshold),
                                "pre_post_avg_frames": int(pre_post_avg_frames),
                                "min_region_length": int(min_region_length),
                            },
                            "detection": {
                                "mode": str(detection_mode),
                                "hybrid_enabled": bool(hybrid_enabled),
                                "roi1_enabled": bool(roi1_enabled),
                            },
                            "peaks": {
                                "green_raw": _peaks_to_abs(green_peaks_raw),
                                "red_raw": _peaks_to_abs(red_peaks_raw),
                                "green": _peaks_to_abs(green_peaks),
                                "red": _peaks_to_abs(red_peaks),
                            },
                            "stats_write": stats_write_results,
                        }
                    )
                except Exception:
                    pass

            # Optionally save ROI1 image
                if should_save and save_roi1:
                    roi1_path = os.path.join(roi1_dir, f"roi1_{frame_index:06d}.png")
                    try:
                        roi1_image.save(roi1_path)
                        # 调试：每保存10张图像输出一次日志
                        if frame_index % 10 == 1:
                            print(f"[DEBUG] ROI1 saved: {roi1_path}")
                    except Exception as e:
                        # 调试：输出保存失败的错误信息
                        print(f"[ERROR] Failed to save ROI1 {roi1_path}: {e}")
                        # Ignore individual save errors to keep daemon running
                        pass

                # Optionally save ROI2 image (align index with ROI1 saves)
                if should_save and save_roi2 and roi2_image is not None:
                    # Calculate video time in seconds if in video mode
                    video_time_str = ""
                    if processing_mode == "video" and video_cap is not None:
                        try:
                            # Get current video position in milliseconds
                            video_pos_msec = video_cap.get(cv2.CAP_PROP_POS_MSEC)
                            video_seconds = video_pos_msec / 1000.0
                            video_time_str = f"_{video_seconds:06.2f}s"
                        except Exception:
                            video_time_str = "_0000.00s"

                    roi2_path = os.path.join(roi2_dir, f"roi2_{frame_index:06d}{video_time_str}.png")
                    try:
                        roi2_image.save(roi2_path)
                    except Exception:
                        pass

                # Save ROI3 image if enabled and available
                if should_save and save_roi3 and roi3_image is not None and roi3_dir:
                    try:
                        roi3_path = os.path.join(roi3_dir, f"roi3_{frame_index:06d}{video_time_str}.png")
                        roi3_image.save(roi3_path)
                    except Exception:
                        pass

                # Save wave plot (curve before detection, but annotated with detection result)
                if should_save and save_wave and gray_buffer:
                    try:
                        wave_path = os.path.join(
                            wave_dir,
                            f"wave_{frame_index:06d}.png",
                        )

                        # Save wave plot (curve before detection, but annotated with detection result)
                        curve = list(gray_buffer) if gray_buffer else []
                        fig, ax = plt.subplots(figsize=(8, 3))
                        x = list(range(len(curve)))
                        ax.plot(x, curve, color="black", linewidth=1)

                        # Add ROI3 purple curve if buffer has data
                        if roi3_gray_buffer:
                            x3 = list(range(len(roi3_gray_buffer)))
                            ax.plot(x3, list(roi3_gray_buffer), color="purple", linewidth=1, label="ROI3")
                            ax.legend()

                        # Draw session-wide background mean (adaptive threshold baseline)
                        if bg_count > 0:
                            ax.axhline(
                                bg_mean,
                                color="blue",
                                linestyle="--",
                                linewidth=1,
                                label="bg_mean",
                            )
                        else:
                            # 调试：输出为什么没有黄线
                            print(f"[DEBUG] No bg_mean line: bg_count={bg_count}, buffer_len={len(gray_buffer)}, adaptive_frames={adaptive_window_frames}, adaptive_enabled={adaptive_threshold_enabled}")
                            print(f"[DEBUG] protection_active={threshold_protection_active}, bg_mean={bg_mean}")

                        # Draw current threshold used for peak detection
                        threshold_color = "red" if threshold_protection_active else "orange"
                        threshold_style = "--" if threshold_protection_active else "-"
                        ax.axhline(
                            threshold_used,
                            color=threshold_color,
                            linestyle=threshold_style,
                            linewidth=1.5,
                            label=f"threshold ({threshold_used:.1f}{'[PROTECTED]' if threshold_protection_active else ''})",
                        )

                        # Highlight green and red regions (slightly expanded for readability)
                        for start, end in green_peaks:
                            s = max(0, start - 1)
                            e = min(len(curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = curve[s : e + 1]
                            ax.plot(xs, ys, color="green", linewidth=2)

                        for start, end in red_peaks:
                            s = max(0, start - 1)
                            e = min(len(curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = curve[s : e + 1]
                            ax.plot(xs, ys, color="red", linewidth=2)

                        # Add ROI2 frame information if available
                        if roi2_dir and os.path.exists(roi2_dir):
                            # Look for ROI2 files to display frame information
                            roi2_files = []
                            buffer_start = max(0, frame_index - len(curve) + 1)
                            buffer_end = frame_index

                            # Search for ROI2 files with the new naming pattern (frame_xxxxxx_XXXX.XXs.png)
                            roi2_pattern = os.path.join(roi2_dir, "roi2_*.png")
                            all_roi2_files = glob.glob(roi2_pattern)

                            for actual_frame_num in range(buffer_start, buffer_end + 1):
                                # Try to find file with new pattern first
                                found_file = None
                                for roi2_file in all_roi2_files:
                                    basename = os.path.basename(roi2_file)
                                    # Check if filename starts with the current frame number
                                    if basename.startswith(f"roi2_{actual_frame_num:06d}_"):
                                        found_file = roi2_file
                                        break

                                # Fallback to old pattern if new pattern not found
                                if found_file is None:
                                    old_path = os.path.join(roi2_dir, f"roi2_{actual_frame_num:06d}.png")
                                    if os.path.exists(old_path):
                                        found_file = old_path

                                if found_file:
                                    # Extract frame number and time from filename
                                    basename = os.path.basename(found_file)
                                    try:
                                        if "_" in basename:
                                            parts = basename.replace("roi2_", "").replace(".png", "").split("_")
                                            frame_num = int(parts[0])
                                            if len(parts) > 1 and parts[1].endswith("s"):
                                                time_str = parts[1]
                                                roi2_files.append(f"{frame_num}({time_str})")
                                            else:
                                                roi2_files.append(str(frame_num))
                                        else:
                                            frame_num = int(basename.replace("roi2_", "").replace(".png", ""))
                                            roi2_files.append(str(frame_num))
                                    except Exception:
                                        roi2_files.append(str(actual_frame_num))

                                    if len(roi2_files) >= 3:  # Limit to 3 examples
                                        break

                            if roi2_files:
                                sample_text = "ROI2: " + ", ".join(roi2_files)
                                ax.text(0.02, 0.98, sample_text, transform=ax.transAxes,
                                       fontsize=8, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                        ax.set_xlabel("Frame index in buffer")
                        ax.set_ylabel("Gray value")
                        ax.set_title("ROI2 gray waveform with peaks")
                        ax.set_ylim(50, 150)
                        ax.grid(True, linestyle="--", alpha=0.3)
                        ax.legend(loc="best", fontsize=8)
                        fig.tight_layout()
                        fig.savefig(wave_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    except Exception:
                        # Ignore individual plotting/saving errors
                        pass

                # ROI1 waveform visualization (if enabled)
                roi1_green_peaks: List[Tuple[int, int]] = []
                roi1_red_peaks: List[Tuple[int, int]] = []
                # Note: ROI1 peak detection will be implemented in a future phase
                # For now, we just visualize the ROI1 gray values without peak detection

                # Save ROI1 wave plot
                if roi1_should_save and save_roi1_wave and roi1_enabled and roi1_curve:
                    try:
                        roi1_wave_path = os.path.join(
                            wave1_dir,
                            f"roi1_wave_{frame_index:06d}.png",
                        )

                        # Create ROI1 waveform plot
                        fig, ax = plt.subplots(figsize=(8, 3))
                        x = list(range(len(roi1_curve)))
                        ax.plot(x, roi1_curve, color="darkblue", linewidth=1, label="ROI1")

                        # Draw ROI1 background mean
                        if roi1_bg_count > 0:
                            ax.axhline(
                                roi1_bg_mean,
                                color="blue",
                                linestyle="--",
                                linewidth=1,
                                label="bg_mean",
                            )

                        # Draw ROI1 threshold
                        roi1_threshold_color = "red" if roi1_threshold_protection_active else "orange"
                        roi1_threshold_style = "--" if roi1_threshold_protection_active else "-"
                        ax.axhline(
                            roi1_threshold_used,
                            color=roi1_threshold_color,
                            linestyle=roi1_threshold_style,
                            linewidth=1.5,
                            label=f"threshold ({roi1_threshold_used:.1f}{'[PROTECTED]' if roi1_threshold_protection_active else ''})",
                        )

                        # Add ROI3 (80-160) percentage red curve if buffer has data
                        if roi3_80_160_buffer:
                            x3_80_160 = list(range(len(roi3_80_160_buffer)))
                            ax.plot(x3_80_160, list(roi3_80_160_buffer), color="red", linewidth=1, label="ROI3(80-160)%")

                        # Highlight ROI1 peaks regions (placeholder for future peak detection)
                        for start, end in roi1_green_peaks:
                            s = max(0, start - 1)
                            e = min(len(roi1_curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = roi1_curve[s : e + 1]
                            ax.plot(xs, ys, color="green", linewidth=2)

                        for start, end in roi1_red_peaks:
                            s = max(0, start - 1)
                            e = min(len(roi1_curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = roi1_curve[s : e + 1]
                            ax.plot(xs, ys, color="red", linewidth=2)

                        
                        # Set plot title and labels
                        ax.set_title(f"ROI1 Waveform - Frame {frame_index} (len={len(roi1_curve)})")
                        ax.set_xlabel("Frame Index (relative)")
                        ax.set_ylabel("Gray Value (0-255)")
                        ax.set_ylim(0, 100)
                        ax.legend(loc='upper right', fontsize=8)
                        ax.grid(True, alpha=0.3)

                        fig.tight_layout()
                        fig.savefig(roi1_wave_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    except Exception:
                        # Ignore ROI1 plotting/saving errors
                        pass

            # Build log line; when only_delect is True, only log frames with peaks
                if (not only_delect) or has_peak:
                    log_line = (
                        f"{ts} gray={gray_str} "
                        f"green_peaks={green_count} red_peaks={red_count} "
                        f"last_green={last_green_repr}"
                    )
                else:
                    log_line = None
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
