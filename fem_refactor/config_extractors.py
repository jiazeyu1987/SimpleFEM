from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class RoiCaptureConfig:
    roi_default: Dict[str, Any]
    extension_params: Dict[str, Any]
    roi3_extension_params: Dict[str, Any]
    roi_frame_rate: float


@dataclass(frozen=True)
class DataProcessingConfig:
    save_roi1: bool
    save_roi2: bool
    save_roi3: bool
    save_wave: bool
    save_roi1_wave: bool
    only_delect: bool


@dataclass(frozen=True)
class PeakDetectionConfig:
    threshold: float
    threshold_minimum: float
    margin_frames: int
    diff_threshold: float
    silence_frames: int
    pre_post_avg_frames: int
    adaptive_threshold_enabled: bool
    threshold_over_mean_ratio: float
    adaptive_window_seconds: float
    protection_enabled: bool
    recovery_delay_seconds: float
    stability_frames: int
    waveform_trigger_enabled: bool
    min_region_length: int

    g1_g2_override_enabled: bool
    g1_threshold: float
    g2_threshold: float
    use_peak_max_g1_g2: bool


@dataclass(frozen=True)
class Roi1PeakDetectionConfig:
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
    roi1_protection_enabled: bool
    roi1_recovery_delay_seconds: float
    roi1_stability_frames: int
    roi1_waveform_trigger_enabled: bool


@dataclass(frozen=True)
class HybridDetectionConfig:
    hybrid_conf: Dict[str, Any]
    data_quality_conf: Dict[str, Any]
    hybrid_enabled: bool
    roi2_pre_frames: int
    roi2_post_frames: int
    min_roi2_frames: int
    roi2_min_variance: float
    fallback_enabled: bool
    max_peak_width: int


def extract_roi_capture_config(config: Dict[str, Any]) -> RoiCaptureConfig:
    roi_capture = config.get("roi_capture", {}) if isinstance(config, dict) else {}
    roi_default = roi_capture.get("default_config", {})
    roi2_config = roi_capture.get("roi2_config", {})
    extension_params = roi2_config.get("extension_params", {})

    roi3_config = roi_capture.get("roi3_config", {})
    roi3_extension_params = roi3_config.get("extension_params", {})

    roi_frame_rate = roi_capture.get("frame_rate", 1)
    try:
        roi_frame_rate = float(roi_frame_rate)
    except Exception:
        roi_frame_rate = 1.0
    if roi_frame_rate <= 0:
        roi_frame_rate = 1.0

    return RoiCaptureConfig(
        roi_default=roi_default,
        extension_params=extension_params,
        roi3_extension_params=roi3_extension_params,
        roi_frame_rate=float(roi_frame_rate),
    )


def extract_data_processing_config(config: Dict[str, Any]) -> DataProcessingConfig:
    data_processing = config.get("data_processing", {}) if isinstance(config, dict) else {}
    save_roi1 = bool(data_processing.get("save_roi1", False))
    save_roi2 = bool(data_processing.get("save_roi2", False))
    save_roi3 = bool(data_processing.get("save_roi3", False))
    save_wave = bool(data_processing.get("save_wave", False))
    save_roi1_wave = bool(data_processing.get("save_roi1_wave", False))
    only_delect = bool(data_processing.get("only_delect", False))
    return DataProcessingConfig(
        save_roi1=save_roi1,
        save_roi2=save_roi2,
        save_roi3=save_roi3,
        save_wave=save_wave,
        save_roi1_wave=save_roi1_wave,
        only_delect=only_delect,
    )


def extract_peak_detection_config(config: Dict[str, Any]) -> Tuple[PeakDetectionConfig, Dict[str, Any]]:
    peak_conf = config.get("peak_detection", {}) if isinstance(config, dict) else {}
    threshold = float(peak_conf.get("threshold", 105.0))
    threshold_minimum = float(peak_conf.get("threshold_minimum", 80.0))
    margin_frames = int(peak_conf.get("margin_frames", 5))
    diff_threshold = float(peak_conf.get("difference_threshold", 0.5))
    silence_frames = int(peak_conf.get("silence_frames", 0))
    pre_post_avg_frames = int(peak_conf.get("pre_post_avg_frames", 5))

    adaptive_threshold_enabled = bool(peak_conf.get("adaptive_threshold_enabled", False))
    threshold_over_mean_ratio = float(peak_conf.get("threshold_over_mean_ratio", 0.15))
    adaptive_window_seconds = float(peak_conf.get("adaptive_window_seconds", 3.0))

    protection_conf = peak_conf.get("threshold_protection", {})
    protection_enabled = bool(protection_conf.get("enabled", False))
    recovery_delay_seconds = float(protection_conf.get("recovery_delay_seconds", 1.0))
    stability_frames = int(protection_conf.get("stability_frames", 5))
    waveform_trigger_enabled = bool(protection_conf.get("waveform_trigger_enabled", True))

    min_region_length = int(peak_conf.get("min_region_length", 1))

    g1_g2_conf = peak_conf.get("g1_g2_override", {})
    g1_g2_override_enabled = bool(g1_g2_conf.get("enabled", True))
    g1_threshold = float(g1_g2_conf.get("g1_threshold", 98.0))
    g2_threshold = float(g1_g2_conf.get("g2_threshold", 20.0))
    use_peak_max_g1_g2 = bool(g1_g2_conf.get("use_peak_max", True))

    return (
        PeakDetectionConfig(
            threshold=threshold,
            threshold_minimum=threshold_minimum,
            margin_frames=margin_frames,
            diff_threshold=diff_threshold,
            silence_frames=silence_frames,
            pre_post_avg_frames=pre_post_avg_frames,
            adaptive_threshold_enabled=adaptive_threshold_enabled,
            threshold_over_mean_ratio=threshold_over_mean_ratio,
            adaptive_window_seconds=adaptive_window_seconds,
            protection_enabled=protection_enabled,
            recovery_delay_seconds=recovery_delay_seconds,
            stability_frames=stability_frames,
            waveform_trigger_enabled=waveform_trigger_enabled,
            min_region_length=min_region_length,
            g1_g2_override_enabled=g1_g2_override_enabled,
            g1_threshold=g1_threshold,
            g2_threshold=g2_threshold,
            use_peak_max_g1_g2=use_peak_max_g1_g2,
        ),
        peak_conf,
    )


def extract_roi1_peak_detection_config(config: Dict[str, Any]) -> Roi1PeakDetectionConfig:
    roi1_peak_conf = config.get("roi1_peak_detection", {}) if isinstance(config, dict) else {}
    roi1_enabled = bool(roi1_peak_conf.get("enabled", False))
    roi1_threshold = float(roi1_peak_conf.get("threshold", 120.0))
    roi1_threshold_minimum = float(roi1_peak_conf.get("threshold_minimum", 110.0))
    roi1_margin_frames = int(roi1_peak_conf.get("margin_frames", 5))
    roi1_silence_frames = int(roi1_peak_conf.get("silence_frames", 5))
    roi1_pre_post_avg_frames = int(roi1_peak_conf.get("pre_post_avg_frames", 5))
    roi1_difference_threshold = float(roi1_peak_conf.get("difference_threshold", 2.0))
    roi1_min_region_length = int(roi1_peak_conf.get("min_region_length", 5))

    roi1_adaptive_threshold_enabled = bool(roi1_peak_conf.get("adaptive_threshold_enabled", True))
    roi1_threshold_over_mean_ratio = float(roi1_peak_conf.get("threshold_over_mean_ratio", 0.08))
    roi1_adaptive_window_seconds = float(roi1_peak_conf.get("adaptive_window_seconds", 3.0))

    roi1_protection_conf = roi1_peak_conf.get("threshold_protection", {})
    roi1_protection_enabled = bool(roi1_protection_conf.get("enabled", True))
    roi1_recovery_delay_seconds = float(roi1_protection_conf.get("recovery_delay_seconds", 1.0))
    roi1_stability_frames = int(roi1_protection_conf.get("stability_frames", 5))
    roi1_waveform_trigger_enabled = bool(roi1_protection_conf.get("waveform_trigger_enabled", True))

    return Roi1PeakDetectionConfig(
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
    )


def extract_hybrid_detection_config(config: Dict[str, Any]) -> HybridDetectionConfig:
    hybrid_conf = config.get("hybrid_detection", {}) if isinstance(config, dict) else {}
    hybrid_enabled = bool(hybrid_conf.get("enabled", False))

    roi2_color_config = hybrid_conf.get("roi2_color_frames", {})
    roi2_pre_frames = int(roi2_color_config.get("pre_peak", 5))
    roi2_post_frames = int(roi2_color_config.get("post_peak", 10))

    peak_width_config = hybrid_conf.get("roi1_peak_width_range", [30, 40])
    max_peak_width = int(peak_width_config[1])

    data_quality_conf = hybrid_conf.get("data_quality", {})
    min_roi2_frames = int(data_quality_conf.get("minimum_roi2_frames", 15))
    roi2_min_variance = float(data_quality_conf.get("roi2_minimum_variance", 0.5))
    fallback_enabled = bool(hybrid_conf.get("fallback_enabled", True))

    return HybridDetectionConfig(
        hybrid_conf=hybrid_conf,
        data_quality_conf=data_quality_conf,
        hybrid_enabled=hybrid_enabled,
        roi2_pre_frames=roi2_pre_frames,
        roi2_post_frames=roi2_post_frames,
        min_roi2_frames=min_roi2_frames,
        roi2_min_variance=roi2_min_variance,
        fallback_enabled=fallback_enabled,
        max_peak_width=max_peak_width,
    )

