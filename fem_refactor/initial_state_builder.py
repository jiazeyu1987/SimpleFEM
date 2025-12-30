from __future__ import annotations

from typing import Any, Dict, Tuple

from fem_refactor.models import Buffers, Roi1ThresholdState, RuntimeState, ThresholdState
from fem_refactor.signal_buffers import create_signal_buffers


def build_initial_state(*, roi1_threshold: float, roi1_threshold_minimum: float) -> Tuple[Buffers, ThresholdState, Roi1ThresholdState, RuntimeState]:
    bg_count: int = 0
    bg_mean: float = 0.0
    frames_since_protection_end: int = 0
    threshold_protection_active: bool = False
    protection_end_time: float = 0.0
    consecutive_below_threshold: int = 0
    last_waveform_time: float = 0.0

    roi1_bg_count: int = 0
    roi1_bg_mean: float = 0.0
    roi1_threshold_protection_active: bool = False
    roi1_protection_end_time: float = 0.0
    roi1_consecutive_below_threshold: int = 0
    roi1_last_waveform_time: float = 0.0

    (
        gray_buffer,
        roi1_gray_buffer,
        roi3_gray_buffer,
        roi3_80_160_buffer,
        roi3_g1_buffer,
        roi3_g2_buffer,
        roi3_column_diff_buffer,
    ) = create_signal_buffers(maxlen=100)

    buffers_obj = Buffers(
        gray_buffer=gray_buffer,
        roi1_gray_buffer=roi1_gray_buffer,
        roi3_gray_buffer=roi3_gray_buffer,
        roi3_80_160_buffer=roi3_80_160_buffer,
        roi3_g1_buffer=roi3_g1_buffer,
        roi3_g2_buffer=roi3_g2_buffer,
        roi3_column_diff_buffer=roi3_column_diff_buffer,
    )

    thr_state = ThresholdState(
        bg_count=int(bg_count),
        bg_mean=float(bg_mean),
        frames_since_protection_end=int(frames_since_protection_end),
        threshold_protection_active=bool(threshold_protection_active),
        protection_end_time=float(protection_end_time),
        consecutive_below_threshold=int(consecutive_below_threshold),
        last_waveform_time=float(last_waveform_time),
    )

    roi1_threshold_used: float = max(float(roi1_threshold), float(roi1_threshold_minimum))
    roi1_thr_state = Roi1ThresholdState(
        bg_count=int(roi1_bg_count),
        bg_mean=float(roi1_bg_mean),
        threshold_protection_active=bool(roi1_threshold_protection_active),
        protection_end_time=float(roi1_protection_end_time),
        consecutive_below_threshold=int(roi1_consecutive_below_threshold),
        last_waveform_time=float(roi1_last_waveform_time),
        threshold_used=float(roi1_threshold_used),
    )

    processed_roi1_peaks: Dict[Any, Any] = {}
    roi1_peak_counter: int = 0

    runtime_state = RuntimeState(
        frame_index=0,
        last_intersection_roi=None,
        processed_roi1_peaks=processed_roi1_peaks,
        roi1_peak_counter=int(roi1_peak_counter),
    )

    return buffers_obj, thr_state, roi1_thr_state, runtime_state
