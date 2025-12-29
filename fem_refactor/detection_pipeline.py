from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple


def hybrid_peak_detection(roi1_curve: List[float], roi2_curve: List[float],
                          config: Dict[str, Any],
                          processed_peaks: Dict[Tuple[int, int], str] = None,
                          peak_counter: int = 0) -> List[Dict[str, Any]]:
    """
    混合波峰检测：ROI1检测波峰区间，ROI2在相同区间内判定颜色

    Args:
        roi1_curve: ROI1灰度曲线数据
        roi2_curve: ROI2灰度曲线数据（与ROI1完全同步）
        config: 混合检测配置参数
        processed_peaks: 已处理的ROI1波峰字典 {(start, end): peak_id}
        peak_counter: ROI1波峰ID计数器

    Returns:
        hybrid_peaks: 混合检测结果列表（包含唯一ID）
    """
    from peak_detection import detect_peaks

    hybrid_peaks = []

    # 初始化ROI1波峰管理
    if processed_peaks is None:
        processed_peaks = {}

    require_intersection = bool(config.get("require_intersection", True))
    intersection_detected = bool(config.get("intersection_detected", True))
    if require_intersection and not intersection_detected:
        print("[混合检测] 当前帧未检测到绿线交点，跳过ROI1波峰检测（避免ROI2定位失效导致误报）")
        return []

    buffer_start_frame_index = int(config.get("buffer_start_frame_index", 1))

    # 1. 使用ROI1数据进行波峰检测（ROI1独立阈值）
    try:
        roi1_green_peaks_raw, roi1_red_peaks_raw = detect_peaks(
            roi1_curve,
            threshold=config['roi1_threshold'],
            marginFrames=config['margin_frames'],
            silenceFrames=config['silence_frames'],
            avgFrames=config['pre_post_avg_frames'],
            differenceThreshold=999.0,  # 设为很大的值，让ROI1只检测波峰，不做颜色分类
        )
    except Exception as e:
        logging.error(f"[混合检测] ROI1波峰检测失败: {e}")
        print(f"[混合检测] ROI1波峰检测失败: {e}")
        return []

    # 合并ROI1检测到的所有波峰（不区分颜色）
    roi1_all_peaks = roi1_green_peaks_raw + roi1_red_peaks_raw

    # 过滤最小宽度的波峰
    min_width = config.get('min_peak_width', 5)
    max_width = config.get('max_peak_width', 100)
    new_peaks: List[Tuple[int, int, str, int]] = []
    duplicate_count = 0
    width_filtered_count = 0

    current_frame_index = config.get('frame_index', 0)
    logging.info(f"[混合检测-ROI1过滤] 帧{current_frame_index} 开始过滤ROI1检测到的{len(roi1_all_peaks)}个波峰 "
                f"(宽度范围: {min_width}-{max_width}帧)")

    for peak_start, peak_end in roi1_all_peaks:
        peak_width = peak_end - peak_start + 1

        # 宽度过滤
        if peak_width < min_width or peak_width > max_width:
            width_filtered_count += 1
            logging.debug(f"[混合检测-ROI1宽度过滤] 波峰[{peak_start}-{peak_end}] 宽度={peak_width}帧 "
                         f"不在范围[{min_width}-{max_width}]内，被过滤")
            continue

        # Use absolute peak max position as a stable dedup key so the same
        # physical peak is not re-detected when the sliding buffer shifts.
        peak_slice = roi1_curve[peak_start : peak_end + 1]
        local_max_offset = 0
        if peak_slice:
            local_max_offset = max(range(len(peak_slice)), key=lambda i: peak_slice[i])
        abs_peak_max = buffer_start_frame_index + peak_start + local_max_offset
        peak_key = abs_peak_max

        # 检查是否已经处理过这个ROI1波峰
        if peak_key in processed_peaks:
            duplicate_count += 1
            existing_id = processed_peaks[peak_key]
            logging.debug(f"[混合检测-ROI1去重] 波峰[{peak_start}-{peak_end}] (peak_max={abs_peak_max}) "
                         f"已处理过(ID:{existing_id})，跳过")
            continue

        # 新的ROI1波峰，生成唯一ID
        peak_counter += 1
        peak_id = f"ROI1_MAX_{abs_peak_max:06d}"
        processed_peaks[peak_key] = peak_id

        new_peaks.append((peak_start, peak_end, peak_id, abs_peak_max))
        logging.info(f"[混合检测-ROI1新波峰] 帧{current_frame_index} 波峰[{peak_start}-{peak_end}] {peak_width}帧 -> ID: {peak_id}")

    # 获取当前帧索引
    current_frame_index = config.get('frame_index', 0)
    logging.info(f"[混合检测-ROI1过滤完成] 帧{current_frame_index} ROI1原始波峰{len(roi1_all_peaks)}个 -> "
                f"宽度过滤{width_filtered_count}个 + 重复过滤{duplicate_count}个 = 保留{len(new_peaks)}个新波峰")

    # 2. 对每个新的ROI1波峰，使用ROI2数据进行颜色判定
    for peak_start, peak_end, peak_id, abs_peak_max in new_peaks:
        peak_width = peak_end - peak_start + 1

        # 使用ROI2数据进行颜色判定
        # 从配置中获取G1/G2曲线和列灰度差值曲线
        roi3_g1_curve = config.get('roi3_g1_curve', None)
        roi3_g2_curve = config.get('roi3_g2_curve', None)
        roi3_column_diff_curve = config.get('roi3_column_diff_curve', None)

        color_result = determine_roi2_color_in_interval(
            peak_start, peak_end, roi2_curve, config,
            roi3_g1_curve=roi3_g1_curve,
            roi3_g2_curve=roi3_g2_curve,
            roi3_column_diff_curve=roi3_column_diff_curve
        )

        # 检查是否被frame_diff过滤掉
        if color_result.get("method") == "error_filtered":
            current_frame_index = config.get('frame_index', 0)
            error_msg = color_result.get('error', '未知错误')
            logging.warning(f"[混合检测-ROI2过滤] 帧{current_frame_index} 波峰[{peak_start}-{peak_end}] (ID:{peak_id}) "
                          f"被过滤: frame_diff异常 - {error_msg}")
            continue

        # 检查ROI2数据是否有效
        roi2_valid = bool(color_result.get("roi2_valid", True))
        skip_when_invalid = bool(config.get("skip_when_roi2_invalid", True))

        if not roi2_valid and skip_when_invalid:
            current_frame_index = config.get('frame_index', 0)
            # 尝试获取更多无效原因信息
            variance = color_result.get('roi2_variance', 0)
            frames_count = color_result.get('roi2_frames_count', 0)
            logging.warning(f"[混合检测-ROI2过滤] 帧{current_frame_index} 波峰[{peak_start}-{peak_end}] (ID:{peak_id}) "
                          f"被过滤: ROI2数据无效 (方差:{variance:.3f}, 帧数:{frames_count})")
            continue

        if not roi2_valid:
            current_frame_index = config.get('frame_index', 0)
            logging.info(f"[混合检测-ROI2回退] 帧{current_frame_index} 波峰[{peak_start}-{peak_end}] (ID:{peak_id}) "
                        f"ROI2数据无效但skip_when_roi2_invalid=False，继续处理")

        # 创建混合检测结果（包含ROI1唯一ID）
        hybrid_peak = {
            'peak_interval': (peak_start, peak_end),
            'width': peak_width,
            'color': color_result['color'],
            'method': color_result['method'],
            'confidence': color_result['confidence'],
            'roi1_frame_diff': 0.0,  # ROI1不做前后差异计算
            'roi2_frame_diff': color_result['frame_difference'],
            'pre_avg': color_result.get('pre_avg', 0.0),
            'post_avg': color_result.get('post_avg', 0.0),
            'quality_score': color_result.get('quality_score', 0.0),
            # ROI1波峰唯一ID信息
            'roi1_peak_id': peak_id,
            'roi1_peak_key': abs_peak_max,
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

        hybrid_peaks.append(hybrid_peak)

        # 获取当前帧索引
        current_frame_index = config.get('frame_index', 0)
        logging.info(f"[混合检测] 帧{current_frame_index} 波峰[{peak_start}-{peak_end}] {peak_width}帧: {color_result['color']}色 "
              f"(ID:{peak_id}, 方法:{color_result['method']}, 置信度:{color_result['confidence']:.2f})")

    # 返回结果和更新后的状态
    return hybrid_peaks


def determine_roi2_color_in_interval(peak_start: int, peak_end: int,
                                   roi2_curve: List[float],
                                   config: Dict[str, Any],
                                   roi3_g1_curve: Optional[List[float]] = None,
                                   roi3_g2_curve: Optional[List[float]] = None,
                                   roi3_column_diff_curve: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    在ROI1检测的波峰区间内，使用ROI2数据进行颜色判定（支持G1/G2覆盖和列灰度差值覆盖）

    Args:
        peak_start: 波峰开始帧
        peak_end: 波峰结束帧
        roi2_curve: ROI2灰度曲线
        config: 配置参数
        roi3_g1_curve: ROI3的G1值曲线（与roi2_curve同步）- 可选
        roi3_g2_curve: ROI3的G2值曲线（与roi2_curve同步）- 可选
        roi3_column_diff_curve: ROI3的列灰度差值曲线（与roi2_curve同步）- 可选

    Returns:
        颜色判定结果字典，包含:
        - color: 'green' or 'red'
        - method: 判定方法
        - confidence: 置信度
        - g1_g2_override_applied: G1/G2覆盖是否应用
        - g1_value: 使用的G1值
        - g2_value: 使用的G2值
        - column_diff_override_applied: 列灰度差值覆盖是否应用（新增）
        - column_diff_value: 使用的列灰度差值（新增）
    """
    pre_frames = config.get('roi2_pre_frames', 5)
    post_frames = config.get('roi2_post_frames', 10)
    color_threshold = config.get('roi2_color_threshold', 1.5)
    min_frames = config.get('minimum_roi2_frames', 15)
    min_variance = config.get('roi2_minimum_variance', 0.5)
    roi2_min_gray = float(config.get("roi2_min_gray", 5.0))
    roi2_max_gray = float(config.get("roi2_max_gray", 250.0))
    fallback_enabled = config.get('fallback_enabled', True)

    # 调试：打印传入的曲线长度
    print(f"[DEBUG] determine_roi2_color_in_interval - peak[{peak_start}-{peak_end}], "
          f"roi2_curve={len(roi2_curve)}, "
          f"roi3_g1_curve={len(roi3_g1_curve) if roi3_g1_curve else 0}, "
          f"roi3_g2_curve={len(roi3_g2_curve) if roi3_g2_curve else 0}, "
          f"roi3_column_diff_curve={len(roi3_column_diff_curve) if roi3_column_diff_curve else 0}")

    try:
        # 检查ROI2数据是否充足
        roi2_interval_length = len(roi2_curve)
        if roi2_interval_length < min_frames:
            if fallback_enabled:
                return {
                    'color': 'red',
                    'method': 'roi1_fallback',
                    'confidence': 0.0,
                    'frame_difference': 0.0,
                    'roi2_valid': False,
                    'error': f'ROI2数据不足({roi2_interval_length} < {min_frames})，回退到ROI1',
                    'roi2_variance': 0.0,
                    'roi2_frames_count': roi2_interval_length,
                    # G1/G2字段（默认值）
                    'g1_g2_override_applied': False,
                    'g1_value': None,
                    'g2_value': None,
                    # 列灰度差值字段（默认值）
                    'column_diff_override_applied': False,
                    'column_diff_value': None,
                }
            else:
                return {
                    'color': 'red',
                    'method': 'error',
                    'confidence': 0.0,
                    'frame_difference': 0.0,
                    'roi2_valid': False,
                    'error': f'ROI2数据不足且未启用回退',
                    'roi2_variance': 0.0,
                    'roi2_frames_count': roi2_interval_length,
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

        # Filter out error data: if |frame_diff| > 15, consider it as noise/signal error
        if abs(frame_difference) > 15.0:
            return {
                'color': 'red',  # 标记为红色但会被后续过滤
                'method': 'error_filtered',
                'confidence': 0.0,
                'frame_difference': frame_difference,
                'threshold': color_threshold,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'roi2_valid': False,
                'error': f'frame_difference异常(|{frame_difference:.1f}| > 15)，判定为错误数据',
                'roi2_variance': 0.0,
                'roi2_frames_count': roi2_interval_length,
                # G1/G2字段（默认值）
                'g1_g2_override_applied': False,
                'g1_value': None,
                'g2_value': None,
                'g1_g2_override_frame_idx': None,
                # 列灰度差值字段（默认值）
                'column_diff_override_applied': False,
                'column_diff_value': None,
                'column_diff_override_frame_idx': None,
            }

        color = "green" if frame_difference >= color_threshold else "red"

        # G1/G2 覆盖逻辑（新增）
        g1_g2_override_applied = False
        g1_value_used = None
        g2_value_used = None
        g1_g2_override_frame_idx = None  # 记录覆盖帧索引

        # G1/G2覆盖逻辑
        if roi3_g1_curve and roi3_g2_curve:
            # 读取G1/G2配置
            g1_g2_conf = config.get("g1_g2_override", {})
            g1_g2_enabled = bool(g1_g2_conf.get("enabled", True))
            g1_threshold = float(g1_g2_conf.get("g1_threshold", 98.0))
            g2_threshold = float(g1_g2_conf.get("g2_threshold", 20.0))
            use_peak_max = bool(g1_g2_conf.get("use_peak_max", True))

            # 提取波峰区间的G1/G2值（无论颜色是什么都计算）
            g1_values = []
            g2_values = []

            for frame_idx in range(peak_start, min(peak_end + 1, len(roi3_g1_curve))):
                if frame_idx < len(roi3_g1_curve) and frame_idx < len(roi3_g2_curve):
                    g1_values.append(roi3_g1_curve[frame_idx])
                    g2_values.append(roi3_g2_curve[frame_idx])

            # 根据配置选择G1/G2值（无论颜色是什么都记录）
            if g1_values and g2_values:
                if use_peak_max:
                    # 找到G1最大值对应的索引（使用G1为基准）
                    import numpy as np
                    max_g1_idx = int(np.argmax(g1_values))
                    # 使用同一帧的G1和G2值
                    g1_value = g1_values[max_g1_idx]
                    g2_value = g2_values[max_g1_idx]
                    g1_g2_override_frame_idx = max_g1_idx  # 记录覆盖帧索引
                else:
                    # 使用波峰结束帧的值（同一帧）
                    g1_value = g1_values[-1] if peak_end < len(roi3_g1_curve) else g1_values[0]
                    g2_value = g2_values[-1] if peak_end < len(roi3_g2_curve) else g2_values[0]
                    g1_g2_override_frame_idx = len(g1_values) - 1  # 记录覆盖帧索引

                g1_value_used = g1_value
                g2_value_used = g2_value

                # 检查是否满足覆盖条件（只对红色波峰生效）
                if g1_g2_enabled and color == "red":
                    if g1_value > g1_threshold and g2_value > g2_threshold:
                        color = "green"
                        g1_g2_override_applied = True

                        # 获取当前帧索引
                        current_frame_index = config.get('frame_index', 0)
                        buffer_start_frame_index = config.get('buffer_start_frame_index', 0)
                        absolute_frame_idx = buffer_start_frame_index + g1_g2_override_frame_idx

                        # 输出覆盖日志（包含帧索引）
                        msg = (f"[G1/G2覆盖] 帧{absolute_frame_idx} 波峰[{peak_start}-{peak_end}] RED→GREEN: "
                              f"G1={g1_value:.2f}%, G2={g2_value:.2f}% "
                              f"(阈值: G1>{g1_threshold}%, G2>{g2_threshold}%)")
                        logging.info(msg)
                        print(msg)

        # ROI3列灰度差值覆盖逻辑（新增）
        column_diff_override_applied = False
        column_diff_used = None
        column_diff_override_frame_idx = None  # 记录覆盖帧索引

        if roi3_column_diff_curve and roi3_g1_curve:
            # 读取列灰度差值配置
            column_diff_conf = config.get("roi3_column_diff_override", {})
            column_diff_enabled = bool(column_diff_conf.get("enabled", True))
            column_diff_threshold = float(column_diff_conf.get("threshold", 15.0))
            use_peak_max = bool(column_diff_conf.get("use_peak_max", True))

            # 记录配置信息
            logging.debug(f"[列灰度差值覆盖] 配置: enabled={column_diff_enabled}, threshold={column_diff_threshold}, use_peak_max={use_peak_max}")

            # 提取波峰区间的列灰度差值（无论颜色是什么都计算）
            column_diff_values = []
            for frame_idx in range(peak_start, min(peak_end + 1, len(roi3_column_diff_curve))):
                if frame_idx < len(roi3_column_diff_curve):
                    column_diff_values.append(roi3_column_diff_curve[frame_idx])

            # 提取波峰区间的G1值（用于列灰度差值覆盖判定）
            g1_values_for_column_diff = []
            for frame_idx in range(peak_start, min(peak_end + 1, len(roi3_g1_curve))):
                if frame_idx < len(roi3_g1_curve):
                    g1_values_for_column_diff.append(roi3_g1_curve[frame_idx])

            # 根据配置选择差值和G1值（无论颜色是什么都记录）
            if column_diff_values and g1_values_for_column_diff:
                if use_peak_max:
                    # 找到G1最大值对应的索引（使用G1为基准，因为覆盖条件是基于G1的）
                    import numpy as np
                    max_g1_idx = int(np.argmax(g1_values_for_column_diff))
                    # 使用同一帧的G1值和列灰度差值
                    g1_for_column_diff = g1_values_for_column_diff[max_g1_idx]
                    column_diff = column_diff_values[max_g1_idx]
                    column_diff_override_frame_idx = max_g1_idx  # 记录覆盖帧索引（使用G1最大值的帧）
                else:
                    # 使用波峰结束帧的值（同一帧）
                    column_diff = column_diff_values[-1] if peak_end < len(roi3_column_diff_curve) else column_diff_values[0]
                    g1_for_column_diff = g1_values_for_column_diff[-1] if peak_end < len(roi3_g1_curve) else g1_values_for_column_diff[0]
                    column_diff_override_frame_idx = len(column_diff_values) - 1  # 记录覆盖帧索引

                column_diff_used = column_diff

                # 调试输出（改为logging.debug）
                msg = (f"[DEBUG] 列灰度差值覆盖判定 - 波峰[{peak_start}-{peak_end}]: "
                      f"g1_for_column_diff={g1_for_column_diff:.2f}%, column_diff={column_diff:.2f}, "
                      f"阈值: G1>99.00%, column_diff>{column_diff_threshold}, "
                      f"当前color={color}, column_diff_enabled={column_diff_enabled}")
                logging.debug(msg)
                print(msg)

                # 检查是否满足覆盖条件：G1 > 99 并且 列灰度差值大于阈值（只对红色波峰生效）
                logging.debug(f"[DEBUG] 列灰度差值覆盖条件检查: column_diff_enabled={column_diff_enabled}, color={color}")
                if column_diff_enabled and color == "red":
                    logging.debug(f"[DEBUG] 满足前置条件，检查数值: g1_for_column_diff={g1_for_column_diff:.2f} > 99.0? {g1_for_column_diff > 99.0}, column_diff={column_diff:.2f} > {column_diff_threshold}? {column_diff > column_diff_threshold}")
                    if g1_for_column_diff > 99.0 and column_diff > column_diff_threshold:
                        color = "green"
                        column_diff_override_applied = True

                        # 获取当前帧索引
                        buffer_start_frame_index = config.get('buffer_start_frame_index', 0)
                        absolute_frame_idx = buffer_start_frame_index + column_diff_override_frame_idx

                        # 输出覆盖日志（包含帧索引）
                        msg = (f"[列灰度差值覆盖] 帧{absolute_frame_idx} 波峰[{peak_start}-{peak_end}] RED→GREEN: "
                              f"G1={g1_for_column_diff:.2f}%, 列灰度差值={column_diff:.2f} "
                              f"(条件: G1>99.00% 且 列灰度差值>{column_diff_threshold})")
                        logging.info(msg)
                        print(msg)
                    else:
                        logging.debug(f"[DEBUG] 列灰度差值覆盖条件未满足，不执行覆盖")
                else:
                    logging.debug(f"[DEBUG] 列灰度差值覆盖跳过: column_diff_enabled={column_diff_enabled}, color={color}")



        # 计算置信度
        confidence = min(abs(frame_difference) / max(color_threshold, abs(frame_difference)), 1.0)

        # 计算数据质量评分
        quality_info = calculate_roi2_data_quality(peak_start, peak_end, roi2_curve)
        variance_val = float(quality_info.get("variance", 0.0))
        mean_val = float(quality_info.get("mean_val", 0.0))
        if variance_val < float(min_variance) or mean_val < roi2_min_gray or mean_val > roi2_max_gray:
            return {
                'color': 'red',
                'method': 'roi2_invalid',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'threshold': color_threshold,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'roi2_valid': False,
                'quality_score': quality_info.get('quality_score', 0.0),
                'variance': variance_val,
                'data_range': quality_info.get('data_range', 0.0),
                'roi2_frames_count': roi2_interval_length,
                'error': f'ROI2无效: mean={mean_val:.2f} (min={roi2_min_gray:.2f}, max={roi2_max_gray:.2f}), variance={variance_val:.4f} (min={float(min_variance):.4f})',
                # G1/G2字段（默认值）
                'g1_g2_override_applied': False,
                'g1_value': None,
                'g2_value': None,
                'g1_g2_override_frame_idx': None,
                # 列灰度差值字段（默认值）
                'column_diff_override_applied': False,
                'column_diff_value': None,
                'column_diff_override_frame_idx': None,
            }

        return {
            'color': color,
            'method': 'roi2' if not (g1_g2_override_applied or column_diff_override_applied) else
                      ('g1_g2_override' if g1_g2_override_applied else 'column_diff_override'),
            'frame_difference': frame_difference,
            'threshold': color_threshold,
            'pre_avg': pre_avg,
            'post_avg': post_avg,
            'confidence': confidence,
            'roi2_valid': True,
            'quality_score': quality_info['quality_score'],
            'variance': quality_info.get('variance', 0.0),
            'data_range': quality_info.get('data_range', 0.0),
            # G1/G2字段
            'g1_g2_override_applied': g1_g2_override_applied,
            'g1_value': g1_value_used,
            'g2_value': g2_value_used,
            'g1_g2_override_frame_idx': g1_g2_override_frame_idx,  # 新增：覆盖帧索引
            # 列灰度差值字段（新增）
            'column_diff_override_applied': column_diff_override_applied,
            'column_diff_value': column_diff_used,
            'column_diff_override_frame_idx': column_diff_override_frame_idx,  # 新增：覆盖帧索引
        }

    except Exception as e:
        if fallback_enabled:
            return {
                'color': 'red',
                'method': 'roi1_fallback',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'roi2_valid': False,
                'error': f'ROI2计算错误({str(e)})，回退到ROI1',
                'roi2_variance': 0.0,
                'roi2_frames_count': 0,
                # G1/G2字段（默认值）
                'g1_g2_override_applied': False,
                'g1_value': None,
                'g2_value': None,
                'g1_g2_override_frame_idx': None,
                # 列灰度差值字段（默认值）
                'column_diff_override_applied': False,
                'column_diff_value': None,
                'column_diff_override_frame_idx': None,
            }
        else:
            return {
                'color': 'red',
                'method': 'error',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'roi2_valid': False,
                'error': f'ROI2计算错误且未启用回退: {str(e)}',
                'roi2_variance': 0.0,
                'roi2_frames_count': 0,
                # G1/G2字段（默认值）
                'g1_g2_override_applied': False,
                'g1_value': None,
                'g2_value': None,
                'g1_g2_override_frame_idx': None,
                # 列灰度差值字段（默认值）
                'column_diff_override_applied': False,
                'column_diff_value': None,
                'column_diff_override_frame_idx': None,
            }


def calculate_roi2_data_quality(peak_start: int, peak_end: int,
                                roi2_curve: List[float]) -> Dict[str, float]:
    """
    计算ROI2在波峰区间内的数据质量评分

    Args:
        peak_start: 波峰开始帧
        peak_end: 波峰结束帧
        roi2_curve: ROI2灰度曲线

    Returns:
        数据质量指标
    """
    try:
        # 提取ROI2在波峰区间内的数据
        if peak_start >= len(roi2_curve) or peak_end >= len(roi2_curve):
            return {'quality_score': 0.0, 'error': '波峰区间超出ROI2数据范围'}

        interval_values = roi2_curve[peak_start:peak_end + 1]

        if not interval_values:
            return {'quality_score': 0.0, 'error': '无ROI2数据'}

        # 计算基本统计指标
        import math
        mean_val = sum(interval_values) / len(interval_values)
        variance_val = sum((x - mean_val) ** 2 for x in interval_values) / len(interval_values)
        std_dev = math.sqrt(variance_val)

        # 计算数据范围
        data_max = max(interval_values)
        data_min = min(interval_values)
        data_range = data_max - data_min

        # 计算数据稳定性（标准差相对于数据范围的比率）
        stability_score = max(0, 1.0 - std_dev / max(10.0, data_range))

        # 计算数据一致性（避免过度波动）
        consistency = 1.0 - min(1.0, std_dev / mean_val) if mean_val > 0 else 0.0

        # 综合质量评分
        quality_score = (stability_score + consistency) / 2.0

        return {
            'quality_score': quality_score,
            'stability_score': stability_score,
            'consistency': consistency,
            'variance': variance_val,
            'std_dev': std_dev,
            'data_range': data_range,
            'frame_count': len(interval_values),
            'mean_val': mean_val
        }

    except Exception as e:
        return {'quality_score': 0.0, 'error': f'质量计算错误: {str(e)}'}


def run_peak_detection_step(
    *,
    frame_index: int,
    hybrid_enabled: bool,
    roi1_enabled: bool,
    roi1_gray_buffer: Any,
    gray_buffer: Any,
    roi1_threshold_used: float,
    roi1_margin_frames: int,
    roi1_silence_frames: int,
    roi1_pre_post_avg_frames: int,
    roi1_min_region_length: int,
    max_peak_width: int,
    roi2_pre_frames: int,
    roi2_post_frames: int,
    min_roi2_frames: int,
    roi2_min_variance: float,
    diff_threshold: float,
    fallback_enabled: bool,
    hybrid_conf: Dict[str, Any],
    data_quality_conf: Dict[str, Any],
    intersection: Any,
    g1_g2_override_enabled: bool,
    g1_threshold: float,
    g2_threshold: float,
    use_peak_max_g1_g2: bool,
    roi3_g1_buffer: Any,
    roi3_g2_buffer: Any,
    roi3_column_diff_buffer: Any,
    processed_roi1_peaks: Dict[Any, Any],
    roi1_peak_counter: int,
    threshold_used: float,
    margin_frames: int,
    silence_frames: int,
    pre_post_avg_frames: int,
    min_region_length: int,
) -> Tuple[str, List[Dict[str, Any]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Run one-frame peak detection step (hybrid ROI1->ROI2 color or legacy ROI2).

    This is a pure relocation of the legacy in-loop block; decision logic must not change.
    """
    from peak_detection import detect_peaks  # type: ignore

    detection_mode = "roi2_legacy"
    hybrid_peaks: List[Dict[str, Any]] = []
    green_peaks_raw: List[Tuple[int, int]] = []
    red_peaks_raw: List[Tuple[int, int]] = []
    green_peaks: List[Tuple[int, int]] = []
    red_peaks: List[Tuple[int, int]] = []

    if hybrid_enabled and roi1_enabled and len(roi1_gray_buffer) > 0:
        # 混合检测模式：ROI1检测波峰，ROI2判定颜色
        roi1_curve = list(roi1_gray_buffer)
        roi2_curve = list(gray_buffer) if gray_buffer else []

        hybrid_config = {
            'roi1_threshold': roi1_threshold_used,
            'margin_frames': roi1_margin_frames,
            'silence_frames': roi1_silence_frames,
            'pre_post_avg_frames': roi1_pre_post_avg_frames,
            'min_peak_width': roi1_min_region_length,
            'max_peak_width': max_peak_width,
            'roi2_pre_frames': roi2_pre_frames,
            'roi2_post_frames': roi2_post_frames,
            'minimum_roi2_frames': min_roi2_frames,
            'roi2_minimum_variance': roi2_min_variance,
            'roi2_color_threshold': diff_threshold,
            'fallback_enabled': fallback_enabled,
            'require_intersection': bool(hybrid_conf.get("require_intersection", True)),
            'intersection_detected': bool(intersection is not None),
            'skip_when_roi2_invalid': bool(data_quality_conf.get("skip_peaks_when_roi2_invalid", True)),
            'roi2_min_gray': float(data_quality_conf.get("roi2_min_gray", 5.0)),
            'roi2_max_gray': float(data_quality_conf.get("roi2_max_gray", 250.0)),
            # 新增G1/G2配置
            'g1_g2_override': {
                'enabled': g1_g2_override_enabled,
                'g1_threshold': g1_threshold,
                'g2_threshold': g2_threshold,
                'use_peak_max': use_peak_max_g1_g2,
            }
        }

        logging.info(f"[混合检测] 帧{frame_index} 开始分析 - ROI1曲线长度:{len(roi1_curve)}, ROI2曲线长度:{len(roi2_curve)}")

        # 执行混合检测（传递ROI1波峰管理参数）
        try:
            hybrid_config_with_frame = {**hybrid_config, 'frame_index': frame_index}
            hybrid_config_with_frame["buffer_start_frame_index"] = frame_index - len(roi1_curve) + 1
            # 新增：传递G1/G2曲线
            hybrid_config_with_frame["roi3_g1_curve"] = list(roi3_g1_buffer) if roi3_g1_buffer else []
            hybrid_config_with_frame["roi3_g2_curve"] = list(roi3_g2_buffer) if roi3_g2_buffer else []
            # 传递列灰度差值曲线（新增）
            hybrid_config_with_frame["roi3_column_diff_curve"] = list(roi3_column_diff_buffer) if roi3_column_diff_buffer else []

            hybrid_peaks = hybrid_peak_detection(
                roi1_curve, roi2_curve, hybrid_config_with_frame,
                processed_roi1_peaks, roi1_peak_counter
            )
            detection_mode = "hybrid_roi1_peaks_roi2_color"

            # 转换为传统格式以保持兼容性
            green_peaks = []
            red_peaks = []
            for peak in hybrid_peaks:
                if peak['color'] == 'green':
                    green_peaks.append(peak['peak_interval'])
                else:
                    red_peaks.append(peak['peak_interval'])

            # 统计颜色数量和质量
            green_count = len(green_peaks)
            red_count = len(red_peaks)
            avg_quality = sum(peak.get('quality_score', 0) for peak in hybrid_peaks) / len(hybrid_peaks) if hybrid_peaks else 0

            logging.info(f"[混合检测] 帧{frame_index} 结果统计: 绿色{green_count}个, 红色{red_count}个, 平均质量{avg_quality:.2f}")

            # 详细日志输出
            for i, peak in enumerate(hybrid_peaks[:5]):  # 只显示前5个
                start, end = peak['peak_interval']
                width = end - start + 1
                method = peak['method']
                color = peak['color']
                confidence = peak.get('confidence', 0)
                logging.info(f"  帧{frame_index} 波峰{i+1}: [{start}-{end}] {width}帧, {color}色, 方法:{method}, 置信度:{confidence:.2f}")

        except Exception as e:
            logging.error(f"[混合检测] 帧{frame_index} 执行失败: {e}")
            # 回退到传统ROI2检测
            hybrid_peaks = []
            green_peaks, red_peaks = [], []
            detection_mode = "hybrid_failed"

    else:
        # 保持原有的ROI2独立检测逻辑作为后备
        if hybrid_enabled:
            logging.info(f"[传统检测] 帧{frame_index} 混合检测未启用或数据不足，使用ROI2独立检测模式")

        if hybrid_enabled and roi1_enabled:
            if frame_index % 50 == 0:
                print("[?бдбд????бъА?ж╠?] ROI1??бу??????ии?3???иибд3ии???3бщ?3бу?бъА?ж╠??????????ижАА??буROI2?3бщ?3бу?бъА?ж╠???бы")
            green_peaks_raw, red_peaks_raw = [], []
            green_peaks, red_peaks = [], []
            detection_mode = "hybrid_roi1_insufficient"
        else:
            # Now run actual ROI2 peak detection with the determined threshold
            curve = list(gray_buffer) if gray_buffer else []
            try:
                green_peaks_raw, red_peaks_raw = detect_peaks(
                    curve,
                    threshold=threshold_used,
                    marginFrames=margin_frames,
                    differenceThreshold=diff_threshold,
                    silenceFrames=silence_frames,
                    avgFrames=pre_post_avg_frames,
                )
            except Exception:
                green_peaks_raw, red_peaks_raw = [], []

            # Apply min_region_length filter
            green_peaks = [
                (start, end)
                for start, end in green_peaks_raw
                if (end - start + 1) >= min_region_length
            ]
            red_peaks = [
                (start, end)
                for start, end in red_peaks_raw
                if (end - start + 1) >= min_region_length
            ]
            detection_mode = "roi2_legacy"

    return (
        detection_mode,
        hybrid_peaks,
        green_peaks_raw,
        red_peaks_raw,
        green_peaks,
        red_peaks,
    )

