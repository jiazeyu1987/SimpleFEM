from __future__ import annotations

import time
from typing import Deque, Optional, Tuple

from .threshold_protection import manage_threshold_protection


def update_roi2_threshold_state(
    *,
    gray_buffer: Deque[float],
    adaptive_threshold_enabled: bool,
    adaptive_window_frames: int,
    threshold: float,
    threshold_minimum: float,
    threshold_over_mean_ratio: float,
    roi2_gray: Optional[float],
    frame_index: int,
    protection_enabled: bool,
    recovery_delay_frames: int,
    stability_frames: int,
    waveform_trigger_enabled: bool,
    threshold_protection_active: bool,
    protection_end_time: float,
    consecutive_below_threshold: int,
    frames_since_protection_end: int,
    last_waveform_time: float,
    bg_mean: float,
    bg_count: int,
) -> Tuple[float, Optional[int], Optional[float], float, int, bool, float, int, int, float]:
    """
    ROI2 threshold calculation + protection gating.

    Returns:
        (threshold_used, recent_frames_count, calculated_bg_mean,
         bg_mean, bg_count,
         threshold_protection_active, protection_end_time,
         consecutive_below_threshold, frames_since_protection_end, last_waveform_time)
    """
    threshold_used = max(threshold, threshold_minimum)
    recent_frames_count: Optional[int] = None
    calculated_bg_mean: Optional[float] = None

    if gray_buffer:
        curve = list(gray_buffer)
        _ = curve  # keep side-effect parity: list() materialization

        print(
            f"[DEBUG] Buffer status: len={len(gray_buffer)}, adaptive_frames={adaptive_window_frames}, enabled={adaptive_threshold_enabled}"
        )

        # Calculate adaptive threshold if enabled and enough history is available
        if adaptive_threshold_enabled and len(gray_buffer) >= adaptive_window_frames:
            # Calculate recent mean (last adaptive_window_frames from gray_buffer)
            recent_frames_count = min(len(gray_buffer), adaptive_window_frames)
            recent_frames = list(gray_buffer)[-recent_frames_count:]
            calculated_bg_mean = sum(recent_frames) / len(recent_frames)

            # First, check if we're already in protection mode and need to extend it
            current_time = time.time()
            if threshold_protection_active:
                # Check protection status with current gray value
                (
                    threshold_protection_active,
                    protection_end_time,
                    consecutive_below_threshold,
                    frames_since_protection_end,
                    last_waveform_time,
                ) = manage_threshold_protection(
                    current_gray=roi2_gray if roi2_gray is not None else 0,
                    current_threshold=threshold_used,
                    has_peaks=False,  # Will check again after detection
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

            # Only update background mean when protection is not active
            if not threshold_protection_active:
                bg_mean = calculated_bg_mean
                bg_count = int(recent_frames_count)
                print(
                    f"[DEBUG] bg_mean updated: {bg_mean:.2f}, bg_count={bg_count}, buffer_len={len(gray_buffer)}"
                )
                if adaptive_threshold_enabled and bg_mean > 0:
                    threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                    # Apply minimum threshold constraint
                    threshold_used = max(threshold_used, threshold_minimum)
            else:
                # Use last known background mean during protection
                if bg_mean > 0:
                    threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                    # Apply minimum threshold constraint even during protection
                    threshold_used = max(threshold_used, threshold_minimum)

    return (
        float(threshold_used),
        recent_frames_count,
        calculated_bg_mean,
        float(bg_mean),
        int(bg_count),
        bool(threshold_protection_active),
        float(protection_end_time),
        int(consecutive_below_threshold),
        int(frames_since_protection_end),
        float(last_waveform_time),
    )


def update_roi1_threshold_state(
    *,
    roi1_enabled: bool,
    roi1_gray_buffer: Deque[float],
    roi1_gray: Optional[float],
    frame_index: int,
    effective_frame_rate: float,
    roi1_threshold: float,
    roi1_threshold_minimum: float,
    roi1_threshold_over_mean_ratio: float,
    roi1_adaptive_threshold_enabled: bool,
    roi1_adaptive_window_seconds: float,
    roi1_threshold_protection_active: bool,
    roi1_bg_mean: float,
    roi1_bg_count: int,
) -> Tuple[float, float, int]:
    """
    ROI1 adaptive threshold calculation (independent from ROI2).

    Returns:
        (roi1_threshold_used, roi1_bg_mean, roi1_bg_count)
    """
    roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)

    if roi1_enabled and roi1_gray_buffer:
        # 每50帧打印一次ROI1阈值使用情况
        if frame_index % 50 == 0:
            print(f"[ROI1阈值] 配置值={roi1_threshold:.1f}, 下限={roi1_threshold_minimum:.1f}, 使用={roi1_threshold_used:.1f}")
            if roi1_adaptive_threshold_enabled and roi1_bg_count > 0:
                print(f"[ROI1阈值] 自适应背景均值={roi1_bg_mean:.1f}, 比例={roi1_threshold_over_mean_ratio:.2f}")
            else:
                print(f"[ROI1阈值] 使用固定阈值")

        roi1_adaptive_window_frames = int(roi1_adaptive_window_seconds * effective_frame_rate)
        roi1_adaptive_window_frames = max(1, min(roi1_adaptive_window_frames, 100))

        if roi1_adaptive_threshold_enabled and len(roi1_gray_buffer) >= roi1_adaptive_window_frames:
            # Calculate ROI1 recent mean
            roi1_recent_frames_count = min(len(roi1_gray_buffer), roi1_adaptive_window_frames)
            roi1_recent_frames = list(roi1_gray_buffer)[-roi1_recent_frames_count:]
            roi1_calculated_bg_mean = sum(roi1_recent_frames) / len(roi1_recent_frames)
            _ = roi1_calculated_bg_mean  # preserve computation

            # Check ROI1 threshold protection status
            current_time = time.time()
            _ = current_time  # preserve side-effect parity: time.time() call
            if roi1_threshold_protection_active:
                # For now, use last known background mean during protection
                if roi1_bg_mean > 0:
                    roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
                    roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)
            else:
                # Update ROI1 background mean if current value is below threshold
                if roi1_gray < roi1_threshold_used:
                    roi1_bg_count += 1
                    # Incremental mean update: new_mean = old_mean + (new_value - old_mean) / count
                    roi1_bg_mean = roi1_bg_mean + (roi1_gray - roi1_bg_mean) / roi1_bg_count

                # Calculate ROI1 adaptive threshold if we have enough background samples
                if roi1_adaptive_threshold_enabled and roi1_bg_mean > 0:
                    roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
                    roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)

    return float(roi1_threshold_used), float(roi1_bg_mean), int(roi1_bg_count)
