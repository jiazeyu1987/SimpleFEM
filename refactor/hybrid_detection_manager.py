"""
混合检测管理器 - ROI1 波峰检测 + ROI2 颜色分类

SimpleFEM Refactored Version
"""

import sys
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# 添加父目录到路径以导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_detection import detect_peaks
from refactor.config_manager import ConfigManager


class HybridDetectionManager:
    """
    混合检测管理器

    功能:
    - ROI1 检测波峰发生时机
    - ROI2 确定波峰颜色（绿/红）
    - 支持 G1/G2 覆盖和列灰度差值覆盖
    """

    def __init__(self, config: ConfigManager):
        """
        初始化混合检测管理器

        Args:
            config: 配置管理器
        """
        self._config = config
        self._roi1_peak_counter: Dict[str, int] = {}
        self._peak_id_counter = 0

    def detect_hybrid_peaks(
        self,
        roi1_curve: List[float],
        roi2_curve: List[float],
        frame_index: int,
        roi2_intersection: Optional[Tuple[int, int]] = None,
        roi3_g1_curve: Optional[List[float]] = None,
        roi3_g2_curve: Optional[List[float]] = None,
        roi3_column_diff_curve: Optional[List[float]] = None
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Dict[str, Any]]]:
        """
        混合检测波峰

        Args:
            roi1_curve: ROI1 灰度值曲线
            roi2_curve: ROI2 灰度值曲线
            frame_index: 当前帧索引
            roi2_intersection: ROI2 交点坐标（可选）
            roi3_g1_curve: ROI3 G1值曲线（与roi2_curve同步）- 可选
            roi3_g2_curve: ROI3 G2值曲线（与roi2_curve同步）- 可选
            roi3_column_diff_curve: ROI3列灰度差值曲线（与roi2_curve同步）- 可选

        Returns:
            (green_peaks, red_peaks, roi1_green_peaks, roi1_red_peaks, hybrid_peaks_info)
                green_peaks: 绿色波峰列表 [(start, end), ...] (用于ROI2统计)
                red_peaks: 红色波峰列表 [(start, end), ...] (用于ROI2统计)
                roi1_green_peaks: ROI1绿色波峰列表 [(start, end), ...] (用于ROI1波形图)
                roi1_red_peaks: ROI1红色波峰列表 [(start, end), ...] (用于ROI1波形图)
                hybrid_peaks_info: 混合检测波峰信息列表
        """
        if not self._config.hybrid_detection_enabled or not self._config.roi1_peak_detection_enabled:
            return [], [], [], [], []

        # 1. ROI1 波峰检测（检测波峰发生时机）
        # 使用 difference_threshold=999.0 让ROI1只检测波峰，不做颜色分类（与原始代码保持一致）
        roi1_green_raw, roi1_red_raw = detect_peaks(
            roi1_curve,
            threshold=self._config.roi1_threshold,
            difference_threshold=999.0,  # ← 关键：与原始代码保持一致
            margin_frames=self._config.margin_frames,
            silence_frames=self._config.silence_frames,
            avgFrames=self._config.pre_post_avg_frames,  # 添加缺失的参数
            min_region_length=self._config.min_region_length
        )

        # 合并 ROI1 检测到的所有波峰（不论颜色）
        roi1_peaks = roi1_green_raw + roi1_red_raw

        # 2. ROI2 颜色判定（确定每个波峰的颜色，包含G1/G2覆盖）
        hybrid_peaks_info = []
        green_peaks = []
        red_peaks = []

        for start, end in roi1_peaks:
            # 生成唯一的 ROI1 波峰 ID
            peak_id = f"roi1_{frame_index}_{start}"

            # 使用 ROI2 曲线判定颜色（包含ROI3 G1/G2覆盖）
            color_result = self._determine_roi2_color(
                roi2_curve, start, end,
                roi3_g1_curve, roi3_g2_curve, roi3_column_diff_curve
            )

            # 构建混合检测结果（与原始代码safe_peak_statistics.py期望的格式一致）
            peak_info = {
                'peak_interval': (start, end),
                'roi1_peak_id': peak_id,
                'color': color_result['color'],
                'detection_method': color_result['method'],
                # G1/G2覆盖字段
                'g1_value': color_result.get('g1_value', None),
                'g2_value': color_result.get('g2_value', None),
                'g1_g2_override_applied': color_result.get('g1_g2_override_applied', False),
                'g1_g2_override_frame_idx': color_result.get('g1_g2_override_frame_idx', None),
                # 列灰度差值字段
                'column_diff_value': color_result.get('column_diff_value', None),
                'column_diff_override_applied': color_result.get('column_diff_override_applied', False),
                'column_diff_override_frame_idx': color_result.get('column_diff_override_frame_idx', None),
            }

            hybrid_peaks_info.append(peak_info)

            if color_result['color'] == 'green':
                green_peaks.append((start, end))
            else:
                red_peaks.append((start, end))

            # 更新计数器
            self._roi1_peak_counter[peak_id] = self._peak_id_counter
            self._peak_id_counter += 1

        return green_peaks, red_peaks, roi1_green_raw, roi1_red_raw, hybrid_peaks_info

    def _determine_roi2_color(
        self,
        roi2_curve: List[float],
        peak_start: int,
        peak_end: int,
        roi3_g1_curve: Optional[List[float]] = None,
        roi3_g2_curve: Optional[List[float]] = None,
        roi3_column_diff_curve: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        基于 ROI2 曲线判定波峰颜色（包含G1/G2覆盖和列灰度差值覆盖）

        Args:
            roi2_curve: ROI2 灰度值曲线
            peak_start: 波峰起始位置
            peak_end: 波峰结束位置
            roi3_g1_curve: ROI3 G1值曲线（与roi2_curve同步）- 可选
            roi3_g2_curve: ROI3 G2值曲线（与roi2_curve同步）- 可选
            roi3_column_diff_curve: ROI3列灰度差值曲线（与roi2_curve同步）- 可选

        Returns:
            颜色判定结果字典，包含:
            - color: 'green' or 'red'
            - method: 判定方法
            - g1_g2_override_applied: G1/G2覆盖是否应用
            - g1_value: 使用的G1值
            - g2_value: 使用的G2值
            - column_diff_override_applied: 列灰度差值覆盖是否应用
            - column_diff_value: 使用的列灰度差值
        """
        # 配置参数
        pre_frames = self._config.pre_post_avg_frames
        post_frames = self._config.pre_post_avg_frames * 2  # 与原始代码保持一致
        color_threshold = self._config.difference_threshold
        min_frames = 15  # 与原始代码保持一致

        # ROI3覆盖阈值
        g1_threshold = self._config.get('peak_detection', 'g1_g2_override', 'g1_threshold', default=99.0)
        g2_threshold = self._config.get('peak_detection', 'g1_g2_override', 'g2_threshold', default=5.0)
        column_diff_threshold = self._config.get('peak_detection', 'roi3_column_diff_override', 'threshold', default=20.0)
        g1_g2_override_enabled = self._config.get('peak_detection', 'g1_g2_override', 'enabled', default=False)
        column_diff_override_enabled = self._config.get('peak_detection', 'roi3_column_diff_override', 'enabled', default=False)

        try:
            # 检查ROI2数据是否充足
            roi2_interval_length = len(roi2_curve)
            if roi2_interval_length < min_frames:
                # ROI2数据不足，回退到红色
                return {
                    'color': 'red',
                    'method': 'roi2_fallback',
                    'confidence': 0.0,
                    'frame_difference': 0.0,
                    'roi2_valid': False,
                    'error': f'ROI2数据不足({roi2_interval_length} < {min_frames})',
                    # G1/G2字段（默认值）
                    'g1_g2_override_applied': False,
                    'g1_value': None,
                    'g2_value': None,
                    # 列灰度差值字段（默认值）
                    'column_diff_override_applied': False,
                    'column_diff_value': None,
                }

            # 计算ROI2在波峰区间前的平均值
            pre_start = max(0, peak_start - pre_frames)
            pre_values = roi2_curve[pre_start:peak_start]
            pre_avg = sum(pre_values) / len(pre_values) if pre_values else roi2_curve[peak_start] if peak_start < len(roi2_curve) else 0.0

            # 计算ROI2在波峰区间后的平均值
            post_end = min(len(roi2_curve), peak_end + post_frames + 1)
            post_values = roi2_curve[peak_end + 1:post_end]
            post_avg = sum(post_values) / len(post_values) if post_values else roi2_curve[peak_end] if peak_end < len(roi2_curve) else 0.0

            # 颜色判定：基于前后差异
            frame_difference = post_avg - pre_avg

            # 过滤错误数据：如果|frame_diff| > 15，认为是噪声/信号错误
            if abs(frame_difference) > 15.0:
                return {
                    'color': 'red',
                    'method': 'error_filter',
                    'confidence': 0.0,
                    'frame_difference': frame_difference,
                    'roi2_valid': True,
                    'error': f'帧差过大({frame_difference:.2f} > 15.0)',
                    # G1/G2字段（默认值）
                    'g1_g2_override_applied': False,
                    'g1_value': None,
                    'g2_value': None,
                    # 列灰度差值字段（默认值）
                    'column_diff_override_applied': False,
                    'column_diff_value': None,
                }

            # 初始颜色判定
            initial_color = 'green' if frame_difference >= color_threshold else 'red'
            final_color = initial_color

            # G1/G2 覆盖逻辑
            g1_g2_override_applied = False
            g1_value_used = None
            g2_value_used = None
            g1_g2_override_frame_idx = None

            if g1_g2_override_enabled and roi3_g1_curve and roi3_g2_curve:
                # 提取波峰区间内的G1/G2值
                g1_values = []
                g2_values = []
                valid_frame_indices = []

                for frame_idx in range(peak_start, min(peak_end + 1, len(roi3_g1_curve))):
                    if frame_idx < len(roi3_g1_curve) and frame_idx < len(roi3_g2_curve):
                        g1_val = roi3_g1_curve[frame_idx]
                        g2_val = roi3_g2_curve[frame_idx]
                        if g1_val > 0:  # 只收集有效值
                            g1_values.append(g1_val)
                            g2_values.append(g2_val)
                            valid_frame_indices.append(frame_idx)

                # 如果有足够的G1/G2数据
                if len(g1_values) > 0 and len(g2_values) > 0:
                    # 找到G1最大的位置（波峰最亮的帧）
                    max_g1_idx = int(max(range(len(g1_values)), key=lambda i: g1_values[i]))
                    g1_value = g1_values[max_g1_idx]
                    g2_value = g2_values[max_g1_idx]

                    # 始终记录实际的G1/G2值（无论是否应用覆盖）
                    g1_value_used = g1_value
                    g2_value_used = g2_value
                    if valid_frame_indices:
                        g1_g2_override_frame_idx = valid_frame_indices[max_g1_idx]

                    # 检查是否满足G1/G2覆盖条件
                    if g1_value >= g1_threshold and g2_value >= g2_threshold:
                        # 强制将红色改为绿色
                        if initial_color == 'red':
                            final_color = 'green'
                            g1_g2_override_applied = True

            # 列灰度差值覆盖逻辑
            column_diff_override_applied = False
            column_diff_value_used = None
            column_diff_override_frame_idx = None

            if column_diff_override_enabled and roi3_column_diff_curve:
                # 提取波峰区间内的列灰度差值
                column_diff_values = []
                valid_frame_indices = []

                for frame_idx in range(peak_start, min(peak_end + 1, len(roi3_column_diff_curve))):
                    col_diff = roi3_column_diff_curve[frame_idx]
                    if col_diff > 0:  # 只收集有效值
                        column_diff_values.append(float(col_diff))
                        valid_frame_indices.append(frame_idx)

                # 如果有足够的列灰度差值数据
                if len(column_diff_values) > 0:
                    max_column_diff = max(column_diff_values)
                    max_idx = column_diff_values.index(max_column_diff)

                    # 始终记录实际的列灰度差值（无论是否应用覆盖）
                    column_diff_value_used = max_column_diff
                    if valid_frame_indices:
                        column_diff_override_frame_idx = valid_frame_indices[max_idx]

                    # 检查是否满足列灰度差值覆盖条件
                    if max_column_diff >= column_diff_threshold:
                        # 强制将红色改为绿色
                        if final_color == 'red':
                            final_color = 'green'
                            column_diff_override_applied = True

            # 返回完整的颜色判定结果
            return {
                'color': final_color,
                'method': 'roi2_frame_diff_with_overrides',
                'confidence': 1.0,
                'frame_difference': frame_difference,
                'roi2_valid': True,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                # G1/G2字段
                'g1_g2_override_applied': g1_g2_override_applied,
                'g1_value': g1_value_used,
                'g2_value': g2_value_used,
                'g1_g2_override_frame_idx': g1_g2_override_frame_idx,
                # 列灰度差值字段
                'column_diff_override_applied': column_diff_override_applied,
                'column_diff_value': column_diff_value_used,
                'column_diff_override_frame_idx': column_diff_override_frame_idx,
            }

        except Exception as e:
            # 发生错误，返回红色
            return {
                'color': 'red',
                'method': 'error',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'roi2_valid': False,
                'error': str(e),
                # G1/G2字段（默认值）
                'g1_g2_override_applied': False,
                'g1_value': None,
                'g2_value': None,
                # 列灰度差值字段（默认值）
                'column_diff_override_applied': False,
                'column_diff_value': None,
            }

    @property
    def roi1_peak_counter(self) -> Dict[str, int]:
        """ROI1 波峰计数器"""
        return self._roi1_peak_counter

    def reset(self) -> None:
        """重置检测状态"""
        self._roi1_peak_counter.clear()
        self._peak_id_counter = 0
