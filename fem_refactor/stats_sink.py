from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def add_peaks_to_statistics(
    *,
    statistics_manager: Any,
    frame_index: int,
    green_peaks: List[Tuple[int, int]],
    red_peaks: List[Tuple[int, int]],
    gray_buffer: Any,
    last_intersection_roi: Optional[Tuple[int, int]],
    roi2_region: Optional[Tuple[int, int, int, int]],
    roi2_gray: Optional[float],
    diff_threshold: float,
    pre_post_avg_frames: int,
    threshold_used: float,
    bg_mean: float,
    bg_count: int,
    hybrid_enabled: bool,
    hybrid_peaks: List[Dict[str, Any]],
    roi1_gray_buffer: Any,
    roi1_threshold_used: float,
    roi3_gray_buffer: Any,
) -> List[Dict[str, Any]]:
    """
    Write peaks to SafePeakStatistics via statistics_manager.current_statistics.

    This is a pure relocation of the legacy 'Add peaks to statistics' block.
    All parameters are passed through without logic changes.
    """
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

    return stats_write_results

