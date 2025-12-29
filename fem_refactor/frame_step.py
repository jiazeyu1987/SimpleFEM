from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .artifact_saver import save_frame_artifacts
from .detection_pipeline import run_peak_detection_step
from .image_metrics import (
    compute_average_gray,
    compute_roi3_80_160_normalized,
    compute_roi3_column_mean_diff,
    compute_roi3_g1_g2_ranges,
)
from .models import DaemonContext, StepResult
from .roi_math import adjust_roi1_to_screen, compute_roi2_region
from .stats_sink import add_peaks_to_statistics
from .threshold_manager import update_roi1_threshold_state, update_roi2_threshold_state
from .threshold_protection import manage_threshold_protection


def process_frame(
    *,
    ctx: DaemonContext,
    screen: Image.Image,
    screen_width: int,
    screen_height: int,
    loop_start: float,
    ts: str,
    video_seconds: Optional[float],
) -> StepResult:
    """
    Process a single frame after a screen/video image has been captured.

    This function is a relocation of the legacy per-frame logic from
    `daemon_loop.run_daemon()` (starting at ROI1 crop), with state persisted in `ctx`.
    Decision logic must not change.
    """
    frame_index = ctx.state.frame_index

    # 2. Get ROI1 region and crop
    x1, y1, x2, y2 = adjust_roi1_to_screen(
        (screen_width, screen_height),
        ctx.cfg.roi_default,
    )
    roi1_image = screen.crop((x1, y1, x2, y2))
    roi1_width, roi1_height = roi1_image.size

    # Initialize ROI3 statistics variables
    roi3_g1: Optional[float] = None
    roi3_g2: Optional[float] = None
    roi3_gray: Optional[float] = None
    roi3_column_diff: Optional[float] = None
    roi3_image: Optional[Image.Image] = None

    # 3. Detect green line intersection in ROI1
    intersection, (center_x, center_y) = ctx.managers.intersection_manager.detect_and_get_center(
        roi1_image=roi1_image,
        anti_jitter_config=ctx.managers.anti_jitter_config,
        intersection_filter=ctx.managers.intersection_filter,
    )
    ctx.state.last_intersection_roi = ctx.managers.intersection_manager.last_intersection_roi

    # 4. Compute ROI2 region and crop
    roi2_region = compute_roi2_region(
        (roi1_width, roi1_height),
        (center_x, center_y),
        ctx.cfg.extension_params,
    )

    roi2_gray: Optional[float] = None
    roi2_image: Optional[Image.Image] = None
    roi1_gray: Optional[float] = None

    if roi2_region is not None:
        rx1, ry1, rx2, ry2 = roi2_region
        roi2_image = roi1_image.crop((rx1, ry1, rx2, ry2))
        roi2_gray = compute_average_gray(roi2_image)
        ctx.buffers.gray_buffer.append(roi2_gray)

        # ROI3 extraction (independent from ROI2)
        if ctx.cfg.roi3_extension_params:
            roi3_region = compute_roi2_region(
                (roi1_width, roi1_height),
                (center_x, center_y),
                ctx.cfg.roi3_extension_params,
            )
            if roi3_region is not None:
                r3x1, r3y1, r3x2, r3y2 = roi3_region
                roi3_image = roi1_image.crop((r3x1, r3y1, r3x2, r3y2))
                roi3_gray = compute_average_gray(roi3_image)
                ctx.buffers.roi3_gray_buffer.append(roi3_gray)
                print(
                    f"[DEBUG] ROI3 captured: frame={frame_index}, gray={roi3_gray:.2f}, buffer_len={len(ctx.buffers.roi3_gray_buffer)}"
                )
                print(
                    f"[DEBUG] ROI3 coords: ({r3x1}, {r3y1}, {r3x2}, {r3y2}), size={r3x2-r3x1}x{r3y2-r3y1}, center=({center_x}, {center_y})"
                )

                # Compute normalized pixel count for range [80, 160]
                roi3_80_160_normalized = compute_roi3_80_160_normalized(roi3_image)
                ctx.buffers.roi3_80_160_buffer.append(roi3_80_160_normalized)
                print(
                    f"[DEBUG] ROI3(80-160)%: frame={frame_index}, percentage={roi3_80_160_normalized:.2f}%, buffer_len={len(ctx.buffers.roi3_80_160_buffer)}"
                )

                # Compute G1 and G2 ranges
                g1, g2 = compute_roi3_g1_g2_ranges(roi3_image)
                roi3_g1 = g1  # Save for cache recording
                roi3_g2 = g2  # Save for cache recording
                ctx.buffers.roi3_g1_buffer.append(g1)  # 存入G1缓冲区
                ctx.buffers.roi3_g2_buffer.append(g2)  # 存入G2缓冲区
                msg = f"[STAT] 帧{frame_index} G1(80-255)={g1:.2f}%, G2(150-255)={g2:.2f}%"
                logging.debug(msg)
                print(msg)

                # 计算ROI3列灰度差值
                roi3_column_diff = compute_roi3_column_mean_diff(roi3_image)
                ctx.buffers.roi3_column_diff_buffer.append(roi3_column_diff)
                msg = f"[STAT] 帧{frame_index} ROI3列灰度差值: {roi3_column_diff:.2f}"
                logging.debug(msg)
                print(msg)
            else:
                print(
                    f"[DEBUG] ROI3 extraction failed: frame={frame_index}, intersection={intersection}, roi3_extension_params={ctx.cfg.roi3_extension_params}"
                )
        else:
            print("[DEBUG] ROI3 extension params not available")

        # ROI1 gray value calculation (independent from ROI2)
        if ctx.cfg.roi1_enabled:
            roi1_gray = compute_average_gray(roi1_image)
            ctx.buffers.roi1_gray_buffer.append(roi1_gray)

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
        gray_buffer=ctx.buffers.gray_buffer,
        adaptive_threshold_enabled=ctx.cfg.adaptive_threshold_enabled,
        adaptive_window_frames=ctx.cfg.adaptive_window_frames,
        threshold=ctx.cfg.threshold,
        threshold_minimum=ctx.cfg.threshold_minimum,
        threshold_over_mean_ratio=ctx.cfg.threshold_over_mean_ratio,
        roi2_gray=roi2_gray,
        frame_index=frame_index,
        protection_enabled=ctx.cfg.protection_enabled,
        recovery_delay_frames=ctx.cfg.recovery_delay_frames,
        stability_frames=ctx.cfg.stability_frames,
        waveform_trigger_enabled=ctx.cfg.waveform_trigger_enabled,
        threshold_protection_active=ctx.thr.threshold_protection_active,
        protection_end_time=ctx.thr.protection_end_time,
        consecutive_below_threshold=ctx.thr.consecutive_below_threshold,
        frames_since_protection_end=ctx.thr.frames_since_protection_end,
        last_waveform_time=ctx.thr.last_waveform_time,
        bg_mean=ctx.thr.bg_mean,
        bg_count=ctx.thr.bg_count,
    )

    ctx.thr.bg_mean = bg_mean
    ctx.thr.bg_count = bg_count
    ctx.thr.threshold_protection_active = threshold_protection_active
    ctx.thr.protection_end_time = protection_end_time
    ctx.thr.consecutive_below_threshold = consecutive_below_threshold
    ctx.thr.frames_since_protection_end = frames_since_protection_end
    ctx.thr.last_waveform_time = last_waveform_time

    (
        detection_mode,
        hybrid_peaks,
        green_peaks_raw,
        red_peaks_raw,
        green_peaks,
        red_peaks,
    ) = run_peak_detection_step(
        frame_index=frame_index,
        hybrid_enabled=ctx.cfg.hybrid_enabled,
        roi1_enabled=ctx.cfg.roi1_enabled,
        roi1_gray_buffer=ctx.buffers.roi1_gray_buffer,
        gray_buffer=ctx.buffers.gray_buffer,
        roi1_threshold_used=ctx.roi1_thr.threshold_used,
        roi1_margin_frames=ctx.cfg.roi1_margin_frames,
        roi1_silence_frames=ctx.cfg.roi1_silence_frames,
        roi1_pre_post_avg_frames=ctx.cfg.roi1_pre_post_avg_frames,
        roi1_min_region_length=ctx.cfg.roi1_min_region_length,
        max_peak_width=ctx.cfg.max_peak_width,
        roi2_pre_frames=ctx.cfg.roi2_pre_frames,
        roi2_post_frames=ctx.cfg.roi2_post_frames,
        min_roi2_frames=ctx.cfg.min_roi2_frames,
        roi2_min_variance=ctx.cfg.roi2_min_variance,
        diff_threshold=ctx.cfg.diff_threshold,
        fallback_enabled=ctx.cfg.fallback_enabled,
        hybrid_conf=ctx.cfg.hybrid_conf,
        data_quality_conf=ctx.cfg.data_quality_conf,
        intersection=intersection,
        g1_g2_override_enabled=ctx.cfg.g1_g2_override_enabled,
        g1_threshold=ctx.cfg.g1_threshold,
        g2_threshold=ctx.cfg.g2_threshold,
        use_peak_max_g1_g2=ctx.cfg.use_peak_max_g1_g2,
        roi3_g1_buffer=ctx.buffers.roi3_g1_buffer,
        roi3_g2_buffer=ctx.buffers.roi3_g2_buffer,
        roi3_column_diff_buffer=ctx.buffers.roi3_column_diff_buffer,
        processed_roi1_peaks=ctx.state.processed_roi1_peaks,
        roi1_peak_counter=ctx.state.roi1_peak_counter,
        threshold_used=threshold_used,
        margin_frames=ctx.cfg.margin_frames,
        silence_frames=ctx.cfg.silence_frames,
        pre_post_avg_frames=ctx.cfg.pre_post_avg_frames,
        min_region_length=ctx.cfg.min_region_length,
    )

    # Re-check threshold protection with actual peak detection results
    if ctx.cfg.protection_enabled and roi2_gray is not None:
        has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
        current_time = time.time()

        (
            ctx.thr.threshold_protection_active,
            ctx.thr.protection_end_time,
            ctx.thr.consecutive_below_threshold,
            ctx.thr.frames_since_protection_end,
            ctx.thr.last_waveform_time,
        ) = manage_threshold_protection(
            current_gray=roi2_gray,
            current_threshold=threshold_used,
            has_peaks=has_peaks,
            frame_time=current_time,
            frame_index=frame_index,
            protection_active=ctx.thr.threshold_protection_active,
            protection_end_time=ctx.thr.protection_end_time,
            consecutive_below=ctx.thr.consecutive_below_threshold,
            last_waveform_time=ctx.thr.last_waveform_time,
            enabled=ctx.cfg.protection_enabled,
            recovery_delay_frames=ctx.cfg.recovery_delay_frames,
            stability_frames=ctx.cfg.stability_frames,
            waveform_trigger=ctx.cfg.waveform_trigger_enabled,
            threshold_minimum=ctx.cfg.threshold_minimum,
        )

    # ROI1 adaptive threshold calculation (independent from ROI2)
    roi1_threshold_used = max(ctx.cfg.roi1_threshold, ctx.cfg.roi1_threshold_minimum)
    roi1_curve = list(ctx.buffers.roi1_gray_buffer) if ctx.buffers.roi1_gray_buffer else []
    roi1_threshold_used, roi1_bg_mean, roi1_bg_count = update_roi1_threshold_state(
        roi1_enabled=ctx.cfg.roi1_enabled,
        roi1_gray_buffer=ctx.buffers.roi1_gray_buffer,
        roi1_gray=roi1_gray,
        frame_index=frame_index,
        effective_frame_rate=ctx.video.effective_frame_rate,
        roi1_threshold=ctx.cfg.roi1_threshold,
        roi1_threshold_minimum=ctx.cfg.roi1_threshold_minimum,
        roi1_threshold_over_mean_ratio=ctx.cfg.roi1_threshold_over_mean_ratio,
        roi1_adaptive_threshold_enabled=ctx.cfg.roi1_adaptive_threshold_enabled,
        roi1_adaptive_window_seconds=ctx.cfg.roi1_adaptive_window_seconds,
        roi1_threshold_protection_active=ctx.roi1_thr.threshold_protection_active,
        roi1_bg_mean=ctx.roi1_thr.bg_mean,
        roi1_bg_count=ctx.roi1_thr.bg_count,
    )
    ctx.roi1_thr.bg_mean = roi1_bg_mean
    ctx.roi1_thr.bg_count = roi1_bg_count
    ctx.roi1_thr.threshold_used = roi1_threshold_used

    green_count = len(green_peaks)
    red_count = len(red_peaks)
    last_green = green_peaks[-1] if green_peaks else None
    last_green_repr = f"[{last_green[0]},{last_green[1]}]" if last_green else "[]"

    gray_str = f"{roi2_gray:.1f}" if roi2_gray is not None else "nan"

    stats_write_results = add_peaks_to_statistics(
        statistics_manager=ctx.managers.statistics_manager,
        frame_index=frame_index,
        green_peaks=green_peaks,
        red_peaks=red_peaks,
        gray_buffer=ctx.buffers.gray_buffer,
        last_intersection_roi=ctx.state.last_intersection_roi,
        roi2_region=roi2_region,
        roi2_gray=roi2_gray,
        diff_threshold=ctx.cfg.diff_threshold,
        pre_post_avg_frames=ctx.cfg.pre_post_avg_frames,
        threshold_used=threshold_used,
        bg_mean=ctx.thr.bg_mean,
        bg_count=ctx.thr.bg_count,
        hybrid_enabled=ctx.cfg.hybrid_enabled,
        hybrid_peaks=hybrid_peaks,
        roi1_gray_buffer=ctx.buffers.roi1_gray_buffer,
        roi1_threshold_used=roi1_threshold_used,
        roi3_gray_buffer=ctx.buffers.roi3_gray_buffer,
    )

    # Decide whether to save images/wave for this frame
    has_peak = (green_count > 0) or (red_count > 0)
    should_save = (not ctx.cfg.only_delect) or has_peak

    # For ROI1, save waveforms when data is available (independent of ROI2 peaks)
    roi1_should_save = (not ctx.cfg.only_delect) or (len(ctx.buffers.roi1_gray_buffer) > 0)

    # Write a per-frame cache record for later Q&A / root cause analysis
    try:
        buffer_len = len(ctx.buffers.gray_buffer)
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

        ctx.managers.analysis_cache.record_frame(
            {
                "ts_wall": loop_start,
                "ts_local": ts,
                "frame_index": int(frame_index),
                "video_seconds": video_seconds,
                "screen_size": [int(screen_width), int(screen_height)],
                "roi1": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "intersection": {"current": intersection, "used": ctx.state.last_intersection_roi},
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
                    "fixed": float(ctx.cfg.threshold),
                    "minimum": float(ctx.cfg.threshold_minimum),
                    "used": float(threshold_used),
                    "adaptive_enabled": bool(ctx.cfg.adaptive_threshold_enabled),
                    "adaptive_window_frames": int(ctx.cfg.adaptive_window_frames),
                    "recent_frames_count": recent_frames_count,
                    "calculated_bg_mean": calculated_bg_mean,
                    "bg_mean": (float(ctx.thr.bg_mean) if ctx.thr.bg_count > 0 else None),
                    "bg_count": int(ctx.thr.bg_count),
                    "protection_active": bool(ctx.thr.threshold_protection_active),
                    "consecutive_below_threshold": int(ctx.thr.consecutive_below_threshold),
                    "frames_since_protection_end": int(ctx.thr.frames_since_protection_end),
                },
                "detect_params": {
                    "margin_frames": int(ctx.cfg.margin_frames),
                    "silence_frames": int(ctx.cfg.silence_frames),
                    "difference_threshold": float(ctx.cfg.diff_threshold),
                    "pre_post_avg_frames": int(ctx.cfg.pre_post_avg_frames),
                    "min_region_length": int(ctx.cfg.min_region_length),
                },
                "detection": {
                    "mode": str(detection_mode),
                    "hybrid_enabled": bool(ctx.cfg.hybrid_enabled),
                    "roi1_enabled": bool(ctx.cfg.roi1_enabled),
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

    save_frame_artifacts(
        frame_index=frame_index,
        should_save=should_save,
        roi1_should_save=roi1_should_save,
        save_roi1=ctx.cfg.save_roi1,
        save_roi2=ctx.cfg.save_roi2,
        save_roi3=ctx.cfg.save_roi3,
        save_wave=ctx.cfg.save_wave,
        save_roi1_wave=ctx.cfg.save_roi1_wave,
        roi1_enabled=ctx.cfg.roi1_enabled,
        processing_mode=ctx.video.processing_mode,
        video_cap=ctx.video.video_cap,
        roi1_dir=ctx.paths.roi1_dir,
        roi2_dir=ctx.paths.roi2_dir,
        roi3_dir=ctx.paths.roi3_dir,
        wave_dir=ctx.paths.wave_dir,
        wave1_dir=ctx.paths.wave1_dir,
        roi1_image=roi1_image,
        roi2_image=roi2_image,
        roi3_image=roi3_image,
        roi2_region=roi2_region,
        gray_buffer=ctx.buffers.gray_buffer,
        roi3_gray_buffer=ctx.buffers.roi3_gray_buffer,
        roi3_80_160_buffer=ctx.buffers.roi3_80_160_buffer,
        green_peaks=green_peaks,
        red_peaks=red_peaks,
        bg_count=ctx.thr.bg_count,
        bg_mean=ctx.thr.bg_mean,
        adaptive_window_frames=ctx.cfg.adaptive_window_frames,
        adaptive_threshold_enabled=ctx.cfg.adaptive_threshold_enabled,
        threshold_protection_active=ctx.thr.threshold_protection_active,
        threshold_used=threshold_used,
        roi1_curve=roi1_curve,
        roi1_bg_count=ctx.roi1_thr.bg_count,
        roi1_bg_mean=ctx.roi1_thr.bg_mean,
        roi1_threshold_protection_active=ctx.roi1_thr.threshold_protection_active,
        roi1_threshold_used=roi1_threshold_used,
    )

    # Build log line; when only_delect is True, only log frames with peaks
    if (not ctx.cfg.only_delect) or has_peak:
        log_line = (
            f"{ts} gray={gray_str} "
            f"green_peaks={green_count} red_peaks={red_count} "
            f"last_green={last_green_repr}"
        )
    else:
        log_line = None

    return StepResult(log_line=log_line)

