from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fem_refactor.analysis_cache import RoiAnalysisCache
from fem_refactor.anti_jitter_manager import AntiJitterManager
from fem_refactor.artifact_directories import prepare_artifact_dirs
from fem_refactor.cache_session_manager import start_analysis_cache_session
from fem_refactor.cleanup_manager import cleanup_directories
from fem_refactor.config_loader import load_fem_config
from fem_refactor.config_extractors import (
    extract_data_processing_config,
    extract_hybrid_detection_config,
    extract_peak_detection_config,
    extract_roi1_peak_detection_config,
    extract_roi_capture_config,
)
from fem_refactor.config_values_factory import build_config_values
from fem_refactor.initial_state_builder import build_initial_state
from fem_refactor.intersection_manager import IntersectionManager
from fem_refactor.logging_manager import (
    resolve_master_logging_enabled,
    set_master_logging_enabled,
    setup_logging,
    setup_peak_logger,
)
from fem_refactor.models import (
    DaemonContext,
    Managers,
    Paths,
    RuntimeState,
    VideoState,
)
from fem_refactor.processing_mode_manager import initialize_processing_mode
from fem_refactor.video_session_manager import VideoSessionManager
from fem_refactor.timing_manager import compute_timing_state
from fem_refactor.video_session_factory import maybe_create_video_session_manager


@dataclass
class BootstrappedDaemon:
    ctx: DaemonContext
    processing_mode: str
    interval_seconds: float
    logger: logging.Logger
    analysis_cache: RoiAnalysisCache
    intersection_filter: Any
    video_session_manager: Optional[VideoSessionManager]


class DaemonBootstrap:
    def __init__(
        self,
        *,
        base_dir: str,
        statistics_manager: Any,
        create_video_folders: Any,
    ) -> None:
        self._base_dir = base_dir
        self._statistics_manager = statistics_manager
        self._create_video_folders = create_video_folders

    def bootstrap(self) -> BootstrappedDaemon:
        """
        Build and return all runtime objects needed by the daemon main loop.

        Refactor-only: behavior and decision logic must remain identical to the
        legacy inline setup in `daemon_loop.run_daemon()`.
        """
        config = load_fem_config()

        master_logging_enabled = resolve_master_logging_enabled(config)
        set_master_logging_enabled(master_logging_enabled)

        # 配置日志系统（在清理之前，以便记录清理过程）
        setup_logging(enabled=master_logging_enabled, config=config)
        logging.info("SimpleFEM ROI Daemon 启动...")
        print("SimpleFEM ROI Daemon 启动...")

        # 清理现有的数据文件夹
        cleanup_directories()

        anti_jitter_config, intersection_filter = AntiJitterManager().build(config)

        # Optional: write a per-frame cache for later analysis / root-cause debugging
        analysis_cache_conf = config.get("analysis_cache", {})
        if not isinstance(analysis_cache_conf, dict):
            analysis_cache_conf = {}

        external_base_dir = os.path.join(self._base_dir, "fem_refactor", "external")
        os.makedirs(external_base_dir, exist_ok=True)
        export_dir = os.path.join(external_base_dir, "export")
        os.makedirs(export_dir, exist_ok=True)

        analysis_cache = RoiAnalysisCache(
            export_dir,
            enabled=bool(analysis_cache_conf.get("enabled", True)),
            flush_every=int(analysis_cache_conf.get("flush_every", 50)),
        )

        processing_mode, video_cap, video_files, current_video_index, _safe_statistics = initialize_processing_mode(
            config,
            self._statistics_manager,
        )

        roi_capture_cfg = extract_roi_capture_config(config)
        roi_default = roi_capture_cfg.roi_default
        extension_params = roi_capture_cfg.extension_params
        roi3_extension_params = roi_capture_cfg.roi3_extension_params

        data_processing_cfg = extract_data_processing_config(config)
        save_roi1 = data_processing_cfg.save_roi1
        save_roi2 = data_processing_cfg.save_roi2
        save_roi3 = data_processing_cfg.save_roi3
        save_wave = data_processing_cfg.save_wave
        save_roi1_wave = data_processing_cfg.save_roi1_wave
        # only_delect == True: save ROI1/ROI2/wave only when peaks are detected
        only_delect = data_processing_cfg.only_delect

        peak_cfg, peak_conf = extract_peak_detection_config(config)
        threshold = peak_cfg.threshold
        threshold_minimum = peak_cfg.threshold_minimum
        margin_frames = peak_cfg.margin_frames
        diff_threshold = peak_cfg.diff_threshold
        # 新增：阈值前后"静默"帧数要求（升阈值前 X 帧和降阈值后 X 帧都不能超过阈值）
        silence_frames = peak_cfg.silence_frames
        pre_post_avg_frames = peak_cfg.pre_post_avg_frames
        # 自适应阈值参数
        adaptive_threshold_enabled = peak_cfg.adaptive_threshold_enabled
        threshold_over_mean_ratio = peak_cfg.threshold_over_mean_ratio
        adaptive_window_seconds = peak_cfg.adaptive_window_seconds

        # 阈值保护参数
        protection_enabled = peak_cfg.protection_enabled
        recovery_delay_seconds = peak_cfg.recovery_delay_seconds
        stability_frames = peak_cfg.stability_frames
        waveform_trigger_enabled = peak_cfg.waveform_trigger_enabled

        min_region_length = peak_cfg.min_region_length

        # ROI1 configuration parameters (independent from ROI2)
        roi1_peak_cfg = extract_roi1_peak_detection_config(config)
        roi1_enabled = roi1_peak_cfg.roi1_enabled
        roi1_threshold = roi1_peak_cfg.roi1_threshold
        roi1_threshold_minimum = roi1_peak_cfg.roi1_threshold_minimum
        roi1_margin_frames = roi1_peak_cfg.roi1_margin_frames
        roi1_silence_frames = roi1_peak_cfg.roi1_silence_frames
        roi1_pre_post_avg_frames = roi1_peak_cfg.roi1_pre_post_avg_frames
        roi1_difference_threshold = roi1_peak_cfg.roi1_difference_threshold
        roi1_min_region_length = roi1_peak_cfg.roi1_min_region_length

        # ROI1 adaptive threshold parameters
        roi1_adaptive_threshold_enabled = roi1_peak_cfg.roi1_adaptive_threshold_enabled
        roi1_threshold_over_mean_ratio = roi1_peak_cfg.roi1_threshold_over_mean_ratio
        roi1_adaptive_window_seconds = roi1_peak_cfg.roi1_adaptive_window_seconds

        # ROI1 threshold protection parameters
        roi1_protection_enabled = roi1_peak_cfg.roi1_protection_enabled
        roi1_recovery_delay_seconds = roi1_peak_cfg.roi1_recovery_delay_seconds
        roi1_stability_frames = roi1_peak_cfg.roi1_stability_frames
        roi1_waveform_trigger_enabled = roi1_peak_cfg.roi1_waveform_trigger_enabled

        # 混合检测配置参数读取
        hybrid_cfg = extract_hybrid_detection_config(config)
        hybrid_conf = hybrid_cfg.hybrid_conf
        hybrid_enabled = hybrid_cfg.hybrid_enabled

        # ROI2颜色判定配置
        roi2_pre_frames = hybrid_cfg.roi2_pre_frames
        roi2_post_frames = hybrid_cfg.roi2_post_frames

        # ROI1波峰宽度验证配置
        max_peak_width = hybrid_cfg.max_peak_width

        # 数据质量检查配置
        data_quality_conf = hybrid_cfg.data_quality_conf
        min_roi2_frames = hybrid_cfg.min_roi2_frames
        roi2_min_variance = hybrid_cfg.roi2_min_variance
        fallback_enabled = hybrid_cfg.fallback_enabled

        # G1/G2 覆盖配置（新增）
        g1_g2_override_enabled = peak_cfg.g1_g2_override_enabled
        g1_threshold = peak_cfg.g1_threshold
        g2_threshold = peak_cfg.g2_threshold
        use_peak_max_g1_g2 = peak_cfg.use_peak_max_g1_g2

        print(
            f"[G1/G2覆盖] 配置: enabled={g1_g2_override_enabled}, "
            f"G1>{g1_threshold}%, G2>{g2_threshold}%, "
            f"use_peak_max={use_peak_max_g1_g2}"
        )

        logger = setup_peak_logger(enabled=master_logging_enabled)

        last_intersection_roi: Optional[Tuple[int, int]] = None
        intersection_manager = IntersectionManager()
        buffers_obj, thr_state, roi1_thr_state, runtime_state = build_initial_state(
            roi1_threshold=float(roi1_threshold),
            roi1_threshold_minimum=float(roi1_threshold_minimum),
        )

        artifact_dirs = prepare_artifact_dirs(
            base_dir=self._base_dir,
            processing_mode=processing_mode,
            video_files=list(video_files) if video_files else None,
            statistics_manager=self._statistics_manager,
            create_video_folders=self._create_video_folders,
            save_roi1=save_roi1,
            save_roi2=save_roi2,
            save_roi3=save_roi3,
            save_wave=save_wave,
            save_roi1_wave=save_roi1_wave,
        )
        tmp_root = artifact_dirs.tmp_root
        roi1_dir = artifact_dirs.roi1_dir
        roi2_dir = artifact_dirs.roi2_dir
        roi3_dir = artifact_dirs.roi3_dir
        wave_dir = artifact_dirs.wave_dir
        wave1_dir = artifact_dirs.wave1_dir

        frame_index = 0

        # Use roi_capture.frame_rate as loop frequency
        roi_frame_rate = roi_capture_cfg.roi_frame_rate

        timing = compute_timing_state(
            processing_mode=processing_mode,
            video_cap=video_cap,
            roi_frame_rate=float(roi_frame_rate),
            adaptive_window_seconds=float(adaptive_window_seconds),
            recovery_delay_seconds=float(recovery_delay_seconds),
        )
        video_fps = timing.video_fps
        video_frame_step = timing.video_frame_step
        first_video_frame = timing.first_video_frame
        effective_frame_rate = timing.effective_frame_rate
        interval_seconds = timing.interval_seconds
        adaptive_window_frames = timing.adaptive_window_frames
        recovery_delay_frames = timing.recovery_delay_frames

        # Start cache session (one file per SafePeakStatistics session/video)
        try:
            start_analysis_cache_session(
                analysis_cache=analysis_cache,
                processing_mode=processing_mode,
                video_files=list(video_files) if video_files else None,
                current_video_index=int(current_video_index),
                config=config,
                statistics_manager=self._statistics_manager,
                roi_frame_rate=float(roi_frame_rate),
                effective_frame_rate=float(effective_frame_rate),
                video_fps=float(video_fps),
                video_frame_step=int(video_frame_step),
                adaptive_window_frames=int(adaptive_window_frames),
                gray_buffer_maxlen=100,
            )
        except Exception:
            pass

        cfg_values = build_config_values(
            config=config,
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
        )

        video_state = VideoState(
            processing_mode=processing_mode,
            video_cap=video_cap,
            video_files=list(video_files) if video_files else [],
            current_video_index=int(current_video_index),
            video_fps=float(video_fps),
            video_frame_step=int(video_frame_step),
            first_video_frame=bool(first_video_frame),
            effective_frame_rate=float(effective_frame_rate),
            interval_seconds=float(interval_seconds),
        )

        paths_obj = Paths(
            base_dir=self._base_dir,
            tmp_root=str(tmp_root),
            roi1_dir=str(roi1_dir),
            roi2_dir=str(roi2_dir),
            roi3_dir=str(roi3_dir),
            wave_dir=str(wave_dir),
            wave1_dir=str(wave1_dir),
        )

        managers = Managers(
            statistics_manager=self._statistics_manager,
            analysis_cache=analysis_cache,
            intersection_manager=intersection_manager,
            intersection_filter=intersection_filter,
            anti_jitter_config=anti_jitter_config,
            logger=logger,
        )

        runtime_state.frame_index = int(frame_index)
        runtime_state.last_intersection_roi = last_intersection_roi

        ctx = DaemonContext(
            cfg=cfg_values,
            video=video_state,
            paths=paths_obj,
            buffers=buffers_obj,
            thr=thr_state,
            roi1_thr=roi1_thr_state,
            managers=managers,
            state=runtime_state,
        )

        video_session_manager: Optional[VideoSessionManager] = maybe_create_video_session_manager(
            processing_mode=processing_mode,
            ctx=ctx,
            config=config,
            statistics_manager=self._statistics_manager,
            analysis_cache=analysis_cache,
            create_video_folders=self._create_video_folders,
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

        return BootstrappedDaemon(
            ctx=ctx,
            processing_mode=processing_mode,
            interval_seconds=float(interval_seconds),
            logger=logger,
            analysis_cache=analysis_cache,
            intersection_filter=intersection_filter,
            video_session_manager=video_session_manager,
        )
