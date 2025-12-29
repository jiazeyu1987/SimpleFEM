from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class Buffers:
    gray_buffer: Deque[float]
    roi1_gray_buffer: Deque[float]
    roi3_gray_buffer: Deque[float]
    roi3_80_160_buffer: Deque[float]
    roi3_g1_buffer: Deque[float]
    roi3_g2_buffer: Deque[float]
    roi3_column_diff_buffer: Deque[float]


@dataclass
class ThresholdState:
    bg_count: int
    bg_mean: float
    frames_since_protection_end: int
    threshold_protection_active: bool
    protection_end_time: float
    consecutive_below_threshold: int
    last_waveform_time: float


@dataclass
class Roi1ThresholdState:
    bg_count: int
    bg_mean: float
    threshold_protection_active: bool
    protection_end_time: float
    consecutive_below_threshold: int
    last_waveform_time: float
    threshold_used: float


@dataclass
class VideoState:
    processing_mode: str
    video_cap: Any
    video_files: List[str]
    current_video_index: int
    video_fps: float
    video_frame_step: int
    first_video_frame: bool
    effective_frame_rate: float
    interval_seconds: float


@dataclass
class Paths:
    base_dir: str
    tmp_root: str
    roi1_dir: str
    roi2_dir: str
    roi3_dir: str
    wave_dir: str
    wave1_dir: str


@dataclass
class ConfigValues:
    raw: Dict[str, Any]
    roi_default: Dict[str, Any]
    extension_params: Dict[str, Any]
    roi3_extension_params: Dict[str, Any]

    # saving flags
    save_roi1: bool
    save_roi2: bool
    save_roi3: bool
    save_wave: bool
    save_roi1_wave: bool
    only_delect: bool

    # roi2 peak detection
    threshold: float
    threshold_minimum: float
    margin_frames: int
    diff_threshold: float
    silence_frames: int
    pre_post_avg_frames: int
    min_region_length: int

    # adaptive threshold
    adaptive_threshold_enabled: bool
    threshold_over_mean_ratio: float
    adaptive_window_seconds: float
    adaptive_window_frames: int

    # threshold protection
    protection_enabled: bool
    recovery_delay_seconds: float
    recovery_delay_frames: int
    stability_frames: int
    waveform_trigger_enabled: bool

    # ROI1 peak detection configuration
    roi1_enabled: bool
    roi1_threshold: float
    roi1_threshold_minimum: float
    roi1_margin_frames: int
    roi1_silence_frames: int
    roi1_pre_post_avg_frames: int
    roi1_difference_threshold: float
    roi1_min_region_length: int
    roi1_adaptive_threshold_enabled: bool
    roi1_threshold_over_mean_ratio: float
    roi1_adaptive_window_seconds: float

    # ROI1 threshold protection parameters
    roi1_protection_enabled: bool
    roi1_recovery_delay_seconds: float
    roi1_stability_frames: int
    roi1_waveform_trigger_enabled: bool

    # hybrid detection
    hybrid_enabled: bool
    roi2_pre_frames: int
    roi2_post_frames: int
    min_roi2_frames: int
    roi2_min_variance: float
    fallback_enabled: bool
    max_peak_width: int
    data_quality_conf: Dict[str, Any]
    hybrid_conf: Dict[str, Any]

    # g1/g2 override
    g1_g2_override_enabled: bool
    g1_threshold: float
    g2_threshold: float
    use_peak_max_g1_g2: bool


@dataclass
class Managers:
    statistics_manager: Any
    analysis_cache: Any
    intersection_manager: Any
    intersection_filter: Any
    anti_jitter_config: Dict[str, Any]
    logger: Any


@dataclass
class RuntimeState:
    frame_index: int
    last_intersection_roi: Optional[Tuple[int, int]]
    processed_roi1_peaks: Dict[Any, Any]
    roi1_peak_counter: int


@dataclass
class DaemonContext:
    cfg: ConfigValues
    video: VideoState
    paths: Paths
    buffers: Buffers
    thr: ThresholdState
    roi1_thr: Roi1ThresholdState
    managers: Managers
    state: RuntimeState


@dataclass
class StepResult:
    log_line: Optional[str]

