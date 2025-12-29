from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple


def create_signal_buffers(
    *,
    maxlen: int = 100,
) -> Tuple[
    Deque[float],
    Deque[float],
    Deque[float],
    Deque[float],
    Deque[float],
    Deque[float],
    Deque[float],
]:
    """
    Create ROI signal buffers (all deques) with the same maxlen as legacy code.

    Returns:
        (gray_buffer, roi1_gray_buffer, roi3_gray_buffer, roi3_80_160_buffer,
         roi3_g1_buffer, roi3_g2_buffer, roi3_column_diff_buffer)
    """
    gray_buffer: Deque[float] = deque(maxlen=maxlen)
    roi1_gray_buffer: Deque[float] = deque(maxlen=maxlen)
    roi3_gray_buffer: Deque[float] = deque(maxlen=maxlen)
    roi3_80_160_buffer: Deque[float] = deque(maxlen=maxlen)
    roi3_g1_buffer: Deque[float] = deque(maxlen=maxlen)
    roi3_g2_buffer: Deque[float] = deque(maxlen=maxlen)
    roi3_column_diff_buffer: Deque[float] = deque(maxlen=maxlen)

    return (
        gray_buffer,
        roi1_gray_buffer,
        roi3_gray_buffer,
        roi3_80_160_buffer,
        roi3_g1_buffer,
        roi3_g2_buffer,
        roi3_column_diff_buffer,
    )


def reset_video_state_variables(
    gray_buffer=None,
    bg_count=None,
    bg_mean=None,
    last_intersection_roi=None,
    frames_since_protection_end=None,
    threshold_protection_active=None,
    protection_end_time=None,
    consecutive_below_threshold=None,
    last_waveform_time=None,
    frame_index=None,
    first_video_frame=None,
) -> Tuple[int, float, Optional[Tuple[int, int]], int, bool, float, int, float, int, bool]:
    """
    Reset video-processing state variables to prevent cross-video contamination.

    This is a pure relocation of the legacy helper (behavior must not change).
    """
    if gray_buffer is not None:
        gray_buffer.clear()

    return (
        0,  # bg_count
        0.0,  # bg_mean
        None,  # last_intersection_roi
        0,  # frames_since_protection_end
        False,  # threshold_protection_active
        0.0,  # protection_end_time
        0,  # consecutive_below_threshold
        0.0,  # last_waveform_time
        0,  # frame_index
        True,  # first_video_frame
    )


def reset_roi1_state(
    *,
    roi1_threshold: float,
    roi1_threshold_minimum: float,
) -> Tuple[int, float, bool, float, int, float, float]:
    """
    Reset ROI1 background/threshold-protection state for a new video.

    Returns:
        (roi1_bg_count, roi1_bg_mean,
         roi1_threshold_protection_active, roi1_protection_end_time,
         roi1_consecutive_below_threshold, roi1_last_waveform_time,
         roi1_threshold_used)
    """
    roi1_bg_count = 0
    roi1_bg_mean = 0.0
    roi1_threshold_protection_active = False
    roi1_protection_end_time = 0.0
    roi1_consecutive_below_threshold = 0
    roi1_last_waveform_time = 0.0
    roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)

    return (
        roi1_bg_count,
        roi1_bg_mean,
        roi1_threshold_protection_active,
        roi1_protection_end_time,
        roi1_consecutive_below_threshold,
        roi1_last_waveform_time,
        roi1_threshold_used,
    )

