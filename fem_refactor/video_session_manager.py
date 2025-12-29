from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from .models import DaemonContext
from .signal_buffers import reset_roi1_state, reset_video_state_variables
from .video_source import get_video_fps, get_video_frame, initialize_video_capture


@dataclass
class VideoCaptureResult:
    screen: Optional[Image.Image]
    should_continue: bool
    should_break: bool


class VideoSessionManager:
    """
    Encapsulate video-mode frame capture + end-of-video switching.

    This is a refactor-only extraction: all decision logic and prints should
    remain identical to the legacy inline block in `daemon_loop.run_daemon()`.
    """

    def __init__(
        self,
        *,
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
    ) -> None:
        self._ctx = ctx
        self._config = config
        self._statistics_manager = statistics_manager
        self._analysis_cache = analysis_cache
        self._create_video_folders = create_video_folders
        self._intersection_filter = intersection_filter
        self._roi_frame_rate = roi_frame_rate
        self._adaptive_window_frames = adaptive_window_frames
        self._save_roi1 = save_roi1
        self._save_roi2 = save_roi2
        self._save_roi3 = save_roi3
        self._save_wave = save_wave
        self._save_roi1_wave = save_roi1_wave
        self._video_files = video_files

    def capture_next(self, *, loop_start: float, interval_seconds: float, frame_index: int) -> VideoCaptureResult:
        """
        Capture a frame from the current video, or switch videos when current ends.

        Returns:
            VideoCaptureResult:
              - screen: captured frame (PIL.Image) or None
              - should_continue: True when caller should `continue` the main loop
              - should_break: True when caller should `break` the main loop
        """
        ctx = self._ctx
        video_config = self._config.get("video_processing", {})
        loop_enabled = video_config.get("loop_enabled", False)

        # First frame uses step=1 to avoid skipping the very beginning.
        step = 1 if ctx.video.first_video_frame else ctx.video.video_frame_step
        ctx.video.first_video_frame = False

        screen = get_video_frame(ctx.video.video_cap, loop_enabled, frame_step=step)
        if screen is not None:
            return VideoCaptureResult(screen=screen, should_continue=False, should_break=False)

        # 当前视频播放结束
        ctx.video.current_video_index += 1

        # 释放当前视频资源
        try:
            ctx.video.video_cap.release()
        except Exception:
            pass

        if ctx.video.current_video_index < len(self._video_files):
            # 切换到下一个视频
            next_video_path = self._video_files[ctx.video.current_video_index]
            try:
                # 为此视频初始化新的统计
                current_stats = self._statistics_manager.initialize_for_video(
                    next_video_path,
                    is_batch=True,
                )

                # 重置防抖动滤波器状态（用于新视频）
                if self._intersection_filter:
                    # 保存当前调试信息
                    old_debug_info = self._intersection_filter.get_debug_info()
                    self._intersection_filter.reset()
                    print(f"已重置ROI2防抖动滤波器，切换到新视频: {os.path.basename(next_video_path)}")

                    # 根据滤波器类型显示不同的统计信息
                    if "update_count" in old_debug_info:
                        # 阈值式滤波器
                        print(
                            f"上一个视频的阈值防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                            f"更新{old_debug_info['update_count']}次, "
                            f"稳定率{old_debug_info.get('stability_rate', 0):.1f}%"
                        )
                    else:
                        # EMA滤波器
                        print(
                            f"上一个视频的EMA防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                            f"大运动{old_debug_info['large_movement_count']}次, "
                            f"边界限制{old_debug_info.get('boundary_clamp_count', 0)}次"
                        )

                # 重置全局状态变量（防止数据污染）
                ctx.buffers.gray_buffer.clear()
                ctx.buffers.roi1_gray_buffer.clear()
                ctx.buffers.roi3_gray_buffer.clear()
                ctx.buffers.roi3_80_160_buffer.clear()
                ctx.buffers.roi3_g1_buffer.clear()  # 清空G1缓冲区
                ctx.buffers.roi3_g2_buffer.clear()  # 清空G2缓冲区
                ctx.buffers.roi3_column_diff_buffer.clear()  # 清空列灰度差值缓冲区

                reset_values = reset_video_state_variables(ctx.buffers.gray_buffer)
                (
                    bg_count,
                    bg_mean,
                    last_intersection_roi,
                    frames_since_protection_end,
                    threshold_protection_active,
                    protection_end_time,
                    consecutive_below_threshold,
                    last_waveform_time,
                    reset_frame_index,
                    first_video_frame,
                ) = reset_values

                ctx.thr.bg_count = int(bg_count)
                ctx.thr.bg_mean = float(bg_mean)
                ctx.state.last_intersection_roi = last_intersection_roi
                ctx.thr.frames_since_protection_end = int(frames_since_protection_end)
                ctx.thr.threshold_protection_active = bool(threshold_protection_active)
                ctx.thr.protection_end_time = float(protection_end_time)
                ctx.thr.consecutive_below_threshold = int(consecutive_below_threshold)
                ctx.thr.last_waveform_time = float(last_waveform_time)
                ctx.state.frame_index = int(reset_frame_index)
                ctx.video.first_video_frame = bool(first_video_frame)

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
                    roi1_threshold=ctx.cfg.roi1_threshold,
                    roi1_threshold_minimum=ctx.cfg.roi1_threshold_minimum,
                )

                ctx.roi1_thr.bg_count = int(roi1_bg_count)
                ctx.roi1_thr.bg_mean = float(roi1_bg_mean)
                ctx.roi1_thr.threshold_protection_active = bool(roi1_threshold_protection_active)
                ctx.roi1_thr.protection_end_time = float(roi1_protection_end_time)
                ctx.roi1_thr.consecutive_below_threshold = int(roi1_consecutive_below_threshold)
                ctx.roi1_thr.last_waveform_time = float(roi1_last_waveform_time)
                ctx.roi1_thr.threshold_used = float(roi1_threshold_used)

                # 重置ROI1波峰ID管理机制
                ctx.state.processed_roi1_peaks.clear()
                ctx.state.roi1_peak_counter = 0

                print("已重置全局和ROI1状态变量，确保数据隔离")

                ctx.video.video_cap = initialize_video_capture(next_video_path)
                print("\n" + "=" * 50)
                print(f"开始处理下一个视频 ({ctx.video.current_video_index + 1}/{len(self._video_files)}):")
                print(f"文件名: {os.path.basename(next_video_path)}")
                print(f"统计会话: {current_stats.session_id}")

                # 重新计算新视频的帧率参数
                ctx.video.video_fps = get_video_fps(ctx.video.video_cap)
                if ctx.video.video_fps > 0:
                    ctx.video.effective_frame_rate = min(self._roi_frame_rate, ctx.video.video_fps)
                    if ctx.video.effective_frame_rate > 0:
                        ctx.video.video_frame_step = max(
                            1, int(round(ctx.video.video_fps / ctx.video.effective_frame_rate))
                        )

                # 创建每视频文件夹结构
                ctx.paths.tmp_root = self._create_video_folders(
                    next_video_path,
                    current_stats.session_id,
                    ctx.video.processing_mode,
                    self._save_roi1,
                    self._save_roi2,
                    self._save_roi3,
                    self._save_wave,
                    self._save_roi1_wave,
                )

                # 关键修复：更新ROI保存路径变量
                ctx.paths.roi1_dir = os.path.join(ctx.paths.tmp_root, "roi1")
                ctx.paths.roi2_dir = os.path.join(ctx.paths.tmp_root, "roi2")
                ctx.paths.roi3_dir = os.path.join(ctx.paths.tmp_root, "roi3")
                ctx.paths.wave_dir = os.path.join(ctx.paths.tmp_root, "wave")
                ctx.paths.wave1_dir = os.path.join(ctx.paths.tmp_root, "wave1")

                print(
                    f"[video] source_fps={ctx.video.video_fps:.2f} "
                    f"target_fps={ctx.video.effective_frame_rate:.2f} frame_step={ctx.video.video_frame_step}"
                )
                print(f"[folders] tmp_root={ctx.paths.tmp_root}")
                print(f"[folders] roi1_dir={ctx.paths.roi1_dir}")
                print(f"[folders] roi2_dir={ctx.paths.roi2_dir}")
                print(f"[folders] wave_dir={ctx.paths.wave_dir}")
                print("=" * 50)

                # 重置帧索引和首帧标志
                ctx.state.frame_index = 0
                ctx.video.first_video_frame = True

                # Start a new cache session for the new video/statistics session
                try:
                    self._analysis_cache.start_session(
                        current_stats.session_id,
                        processing_mode=ctx.video.processing_mode,
                        video_path=next_video_path,
                        config=self._config,
                        extra_meta={
                            "roi_frame_rate": self._roi_frame_rate,
                            "effective_frame_rate": ctx.video.effective_frame_rate,
                            "video_fps": ctx.video.video_fps,
                            "video_frame_step": ctx.video.video_frame_step,
                            "adaptive_window_frames": self._adaptive_window_frames,
                            "gray_buffer_maxlen": 100,
                        },
                    )
                    if self._analysis_cache.path:
                        print(f"[cache] analysis_cache={self._analysis_cache.path}")
                except Exception:
                    pass

                # 继续处理下一个视频，不break
                return VideoCaptureResult(screen=None, should_continue=True, should_break=False)
            except Exception as e:
                print(f"无法打开下一个视频 {next_video_path}: {e}")
                print("继续处理下一个视频...")
                return VideoCaptureResult(screen=None, should_continue=True, should_break=False)

        # 所有视频都处理完毕
        total_time = time.time() - (loop_start - (frame_index * interval_seconds))
        actual_fps = frame_index / total_time if total_time > 0 else 0
        msg = "\n" + "=" * 50
        print(msg)
        logging.info(msg)
        msg = "所有视频处理完成！"
        print(msg)
        logging.info(msg)
        msg = f"[统计] 总处理时间: {total_time:.2f} 秒"
        print(msg)
        logging.info(msg)
        msg = f"[统计] 总处理视频数: {len(self._video_files)}"
        print(msg)
        logging.info(msg)
        msg = f"[统计] 总处理帧数: {frame_index}"
        print(msg)
        logging.info(msg)
        msg = f"[统计] 平均帧率: {actual_fps:.2f} fps"
        print(msg)
        logging.info(msg)
        msg = f"[统计] 配置帧率: {self._roi_frame_rate:.2f} fps"
        print(msg)
        logging.info(msg)
        msg = "=" * 50
        print(msg)
        logging.info(msg)
        return VideoCaptureResult(screen=None, should_continue=False, should_break=True)

