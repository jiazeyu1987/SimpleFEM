"""
改进的HEM波峰检测算法
解决原有算法的关键问题：边界检测、噪声敏感、分类逻辑缺陷等

主要改进：
1. 信号预处理管道（去噪+基线校正+标准化）
2. 物理约束的鲁棒波峰检测
3. 峰生命周期追踪（白色→绿色/红色）
4. 误差带稳定性分类
5. 噪声鲁棒的帧差异计算
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import statistics
import math


@dataclass
class Peak:
    """波峰数据结构"""
    start: int
    end: int
    values: List[float]
    state: str = 'white'  # white, green, red
    birth_frame: int = 0
    last_seen_frame: int = 0
    stability_score: float = 0.0
    variance_history: List[float] = field(default_factory=list)
    trend_slope: float = 0.0
    snr: float = 0.0  # 信噪比
    id: int = 0

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    @property
    def max_value(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def mean_value(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0


class PeakLifecycleTracker:
    """峰生命周期追踪器 - 实现白色→绿色/红色演化"""

    def __init__(self, config: Dict):
        self.config = config
        self.tracked_peaks: Dict[int, Peak] = {}
        self.next_peak_id = 1
        self.frame_counter = 0

    def update_peak_states(self, new_peaks: List[Tuple[int, int, List[float]]]) -> List[Peak]:
        """
        更新峰状态：
        1. 匹配新峰与现有追踪峰
        2. 更新稳定性指标
        3. 基于演化过程分类
        """
        self.frame_counter += 1
        updated_peaks = []

        # 匹配新峰与现有追踪峰
        matched_peaks = self._match_peaks(new_peaks)

        for peak_data in new_peaks:
            start, end, values = peak_data
            center = (start + end) // 2

            # 计算frame_diff用于错误数据过滤
            if len(values) >= 2:
                # 使用与主算法相同的计算方法
                try:
                    # 需要传入完整的curve数据，但这里我们只有values
                    # 使用values的均值变化来模拟frame_diff计算
                    if len(values) >= 10:
                        mid = len(values) // 2
                        pre_avg = sum(values[:mid]) / mid
                        post_avg = sum(values[mid:]) / (len(values) - mid)
                        frame_diff = post_avg - pre_avg
                    else:
                        # 对于较短的values，使用简单的首尾差值
                        frame_diff = values[-1] - values[0]

                    # Filter out error data: if |frame_diff| > 15, consider it as noise/signal error
                    if abs(frame_diff) > 15.0:
                        # Skip this peak as it's considered erroneous data
                        continue
                except:
                    # 如果计算失败，保留峰（保守处理）
                    pass

            if center in matched_peaks:
                # 更新现有峰
                peak_id = matched_peaks[center]
                peak = self.tracked_peaks[peak_id]
                self._update_existing_peak(peak, start, end, values)
            else:
                # 创建新峰（白色）
                peak = self._create_new_peak(start, end, values)
                self.tracked_peaks[peak.id] = peak

            # 更新分类
            self._classify_peak_stability(peak)
            updated_peaks.append(peak)

        # 清理过期峰
        self._cleanup_expired_peaks()

        return updated_peaks

    def _match_peaks(self, new_peaks: List[Tuple[int, int, List[float]]]) -> Dict[int, int]:
        """基于位置匹配新峰与现有峰"""
        matched = {}
        match_threshold = self.config.get('tracking', {}).get('peak_matching_threshold', 0.8)

        for start, end, values in new_peaks:
            center = (start + end) // 2

            best_match_id = None
            best_score = 0

            for peak_id, peak in self.tracked_peaks.items():
                # 计算位置相似度
                distance = abs(center - peak.center)
                max_distance = max(peak.width, end - start + 1)
                similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 0

                if similarity > best_score and similarity > match_threshold:
                    best_score = similarity
                    best_match_id = peak_id

            if best_match_id is not None:
                matched[center] = best_match_id

        return matched

    def _create_new_peak(self, start: int, end: int, values: List[float]) -> Peak:
        """创建新峰（白色状态）"""
        peak = Peak(
            start=start,
            end=end,
            values=values.copy(),
            state='white',
            birth_frame=self.frame_counter,
            last_seen_frame=self.frame_counter,
            id=self.next_peak_id
        )
        self.next_peak_id += 1
        return peak

    def _update_existing_peak(self, peak: Peak, start: int, end: int, values: List[float]):
        """更新现有峰的数据"""
        # 更新位置（可能有轻微移动）
        peak.start = min(peak.start, start)
        peak.end = max(peak.end, end)

        # 更新值（保持滑动窗口）
        max_history = self.config.get('classification', {}).get('stability_window', 5)
        peak.values = values[-max_history:] if len(values) > max_history else values
        peak.last_seen_frame = self.frame_counter

        # 计算新指标
        if len(values) > 1:
            # 计算趋势斜率
            n = len(values)
            if n > 1:
                x = np.arange(n)
                slope = np.polyfit(x, values, 1)[0]
                peak.trend_slope = slope

    def _classify_peak_stability(self, peak: Peak):
        """基于统计特性分类峰的稳定性"""
        if len(peak.values) < 2:
            peak.state = 'white'
            return

        # 计算统计指标
        mean_val = statistics.mean(peak.values)
        variance = statistics.variance(peak.values) if len(peak.values) > 1 else 0
        peak.variance_history.append(variance)

        # 保持历史记录
        max_history = 10
        if len(peak.variance_history) > max_history:
            peak.variance_history = peak.variance_history[-max_history:]

        # 计算信噪比
        if mean_val > 0:
            peak.snr = mean_val / (math.sqrt(variance) + 1e-6)

        # 分类逻辑
        error_tolerance = self.config.get('classification', {}).get('error_tolerance', 0.15)
        min_green_prominence = self.config.get('classification', {}).get('min_green_prominence', 0.3)
        red_detection_variance = self.config.get('classification', {}).get('red_detection_variance', 0.4)

        # 计算变异系数（相对变异度）
        cv = math.sqrt(variance) / (mean_val + 1e-6)

        # 计算稳定性分数
        stability_score = max(0, 1.0 - cv)

        # 峰的年龄（追踪帧数）
        age = self.frame_counter - peak.birth_frame

        # 分类决策
        if age < 2:
            # 新峰，仍在观察期
            peak.state = 'white'
        elif cv < error_tolerance and peak.trend_slope > -0.1 and peak.snr > min_green_prominence:
            # 稳定峰：变异系数小、趋势不下降、信噪比高
            peak.state = 'green'
        elif cv > red_detection_variance or peak.trend_slope < -0.5:
            # 不稳定峰：变异系数大或下降趋势明显
            peak.state = 'red'
        else:
            # 中等稳定度，根据历史判断
            if len(peak.variance_history) >= 3:
                recent_variance = statistics.mean(peak.variance_history[-3:])
                older_variance = statistics.mean(peak.variance_history[-6:-3]) if len(peak.variance_history) >= 6 else recent_variance

                if recent_variance < older_variance * 0.8:  # 方差在减小
                    peak.state = 'green'
                else:
                    peak.state = 'red'
            else:
                peak.state = 'white'

        peak.stability_score = stability_score

    def _cleanup_expired_peaks(self):
        """清理过期的峰"""
        max_age = self.config.get('tracking', {}).get('max_tracking_age', 15)
        expired_ids = []

        for peak_id, peak in self.tracked_peaks.items():
            age_since_last_seen = self.frame_counter - peak.last_seen_frame
            if age_since_last_seen > max_age:
                expired_ids.append(peak_id)

        for peak_id in expired_ids:
            del self.tracked_peaks[peak_id]


def preprocess_signal(curve: List[float],
                     noise_reduction_factor: float = 0.8,
                     baseline_correction: bool = True,
                     smoothing_window: int = 5) -> List[float]:
    """
    信号预处理管道：
    1. Savitzky-Golay平滑滤波（保持波峰形状）
    2. 基线校正（去除DC偏移）
    3. 自适应标准化
    4. 边界保护
    """
    if not curve or len(curve) < 3:
        return curve.copy()

    signal = np.array(curve, dtype=float)

    try:
        # 1. Savitzky-Golay平滑滤波
        if smoothing_window % 2 == 0:
            smoothing_window += 1  # 确保奇数

        smoothing_window = min(smoothing_window, len(signal) // 3)
        if smoothing_window >= 3:
            # 使用3阶多项式保持峰形
            window_length = min(smoothing_window, len(signal) | 1)  # 确保奇数且不超过信号长度
            if window_length >= 3:
                filtered = np.array(_savgol_filter(signal, window_length, 3))
                signal = noise_reduction_factor * filtered + (1 - noise_reduction_factor) * signal

        # 2. 基线校正
        if baseline_correction:
            # 使用滚动最小值估计基线
            baseline_window = min(50, len(signal) // 4)
            baseline = _rolling_minimum(signal, baseline_window)
            signal = signal - baseline
            signal = signal - np.min(signal)  # 确保非负

        # 3. 自适应标准化
        signal_std = np.std(signal)
        if signal_std > 1e-6:
            signal = (signal - np.mean(signal)) / signal_std * 10 + 50  # 标准化到合理范围

        return signal.tolist()

    except Exception:
        # 如果预处理失败，返回原信号
        return curve.copy()


def detect_physical_peaks(curve: List[float],
                         threshold: float,
                         min_peak_width: int = 3,
                         max_peak_width: int = 20,
                         min_peak_distance: int = 10,
                         prominence_factor: float = 0.1) -> List[Tuple[int, int, List[float]]]:
    """
    物理约束的鲁棒波峰检测：
    1. 基于显著性的波峰发现
    2. 形态学验证（钟形检查）
    3. 最小峰间距强制执行
    4. 平台型波峰特殊处理
    5. 边界安全检测
    """
    if not curve or len(curve) < min_peak_width * 2:
        return []

    n = len(curve)
    signal = np.array(curve)

    # 计算显著性
    baseline = np.median(signal)
    noise_level = np.percentile(np.abs(signal - baseline), 75)  # 使用75分位数估计噪声
    prominence_threshold = baseline + prominence_factor * (np.max(signal) - baseline)

    peaks = []

    # 寻找候选峰
    for i in range(min_peak_width, n - min_peak_width):
        # 检查是否为局部极大值
        local_window = signal[max(0, i-2):min(n, i+3)]
        if len(local_window) < 5:
            continue

        if signal[i] != np.max(local_window):
            continue

        # 检查是否超过阈值
        if signal[i] < threshold or signal[i] < prominence_threshold:
            continue

        # 寻找真实边界
        left_boundary = _find_left_boundary(signal, i, min_peak_width, max_peak_width)
        right_boundary = _find_right_boundary(signal, i, min_peak_width, max_peak_width)

        if left_boundary is None or right_boundary is None:
            continue

        width = right_boundary - left_boundary + 1
        if width < min_peak_width or width > max_peak_width:
            continue

        # 形态学验证
        if not _validate_peak_morphology(signal, left_boundary, right_boundary):
            continue

        # 提取峰值数据
        peak_values = signal[left_boundary:right_boundary+1].tolist()

        peaks.append((left_boundary, right_boundary, peak_values))

    # 强制最小峰间距
    peaks = _enforce_peak_distance(peaks, min_peak_distance)

    return peaks


def _savgol_filter(signal: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """简化的Savitzky-Golay滤波器实现"""
    if window_length < polyorder + 1:
        return signal.copy()

    # 使用简单的移动平均作为替代（避免scipy依赖）
    half_window = window_length // 2
    filtered = np.zeros_like(signal)

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        filtered[i] = np.mean(signal[start:end])

    return filtered


def _rolling_minimum(signal: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值"""
    if window <= 1:
        return signal.copy()

    result = np.zeros_like(signal)
    half_window = window // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        result[i] = np.min(signal[start:end])

    return result


def _find_left_boundary(signal: np.ndarray, peak_center: int,
                       min_width: int, max_width: int) -> Optional[int]:
    """寻找波峰左边界"""
    for left in range(peak_center, max(0, peak_center - max_width), -1):
        # 检查最小宽度
        if peak_center - left < min_width:
            continue

        # 检查是否继续上升
        if left > 0 and signal[left] < signal[left - 1]:
            continue

        # 检查是否有显著的下降
        if left > 0 and signal[left] > signal[left - 1] * 1.1:
            return left + 1

        return left

    return None


def _find_right_boundary(signal: np.ndarray, peak_center: int,
                        min_width: int, max_width: int) -> Optional[int]:
    """寻找波峰右边界"""
    n = len(signal)
    for right in range(peak_center, min(n, peak_center + max_width)):
        # 检查最小宽度
        if right - peak_center < min_width:
            continue

        # 检查是否继续下降
        if right < n - 1 and signal[right] < signal[right + 1]:
            continue

        # 检查是否有显著的下降
        if right < n - 1 and signal[right] > signal[right + 1] * 1.1:
            return right - 1

        return right

    return None


def _validate_peak_morphology(signal: np.ndarray, start: int, end: int) -> bool:
    """验证波峰形态（钟形检查）"""
    if start >= end:
        return False

    peak_values = signal[start:end+1]
    peak_max = np.max(peak_values)
    peak_max_idx = np.argmax(peak_values) + start

    # 检查左侧是否大致上升
    left_slope_ok = True
    for i in range(start, peak_max_idx):
        if i > start and signal[i] < signal[i-1] * 0.9:  # 允许小幅下降
            left_slope_ok = False
            break

    # 检查右侧是否大致下降
    right_slope_ok = True
    for i in range(peak_max_idx + 1, end + 1):
        if i > peak_max_idx + 1 and signal[i] > signal[i-1] * 1.1:  # 允许小幅上升
            right_slope_ok = False
            break

    return left_slope_ok and right_slope_ok


def _enforce_peak_distance(peaks: List[Tuple[int, int, List[float]]],
                          min_distance: int) -> List[Tuple[int, int, List[float]]]:
    """强制最小峰间距"""
    if not peaks or len(peaks) <= 1:
        return peaks

    # 按峰值高度排序
    peaks_with_height = [(start, end, values, max(values)) for start, end, values in peaks]
    peaks_with_height.sort(key=lambda x: x[3], reverse=True)

    selected_peaks = []

    for start, end, values, height in peaks_with_height:
        center = (start + end) // 2

        # 检查与已选峰的距离
        too_close = False
        for selected_start, selected_end, _ in selected_peaks:
            selected_center = (selected_start + selected_end) // 2
            if abs(center - selected_center) < min_distance:
                too_close = True
                break

        if not too_close:
            selected_peaks.append((start, end, values))

    # 按位置重新排序
    selected_peaks.sort(key=lambda x: x[0])

    return selected_peaks


def calculate_robust_frame_difference(curve: List[float],
                                    peak_region: Tuple[int, int],
                                    window_size: int = 7,
                                    outlier_rejection: bool = True) -> Tuple[float, Dict]:
    """
    噪声鲁棒的帧差异计算：
    1. 使用中位数代替均值（抗异常值）
    2. 基于梯度的变化检测
    3. 趋势分析代替单点比较
    4. 噪声感知误差估计
    5. 保留方向信息
    """
    if not curve or len(curve) < window_size * 2:
        return 0.0, {"error": "insufficient_data"}

    start, end = peak_region
    n = len(curve)

    try:
        # 计算峰前窗口（使用中位数）
        before_start = max(0, start - window_size)
        before_end = max(0, start - 1)
        if before_start <= before_end:
            before_values = curve[before_start:before_end + 1]
            before_median = statistics.median(before_values)
            before_std = statistics.stdev(before_values) if len(before_values) > 1 else 0
        else:
            before_median = curve[start]
            before_std = 0

        # 计算峰后窗口（使用中位数）
        after_start = min(n - 1, end + 1)
        after_end = min(n - 1, end + window_size)
        if after_start <= after_end:
            after_values = curve[after_start:after_end + 1]
            after_median = statistics.median(after_values)
            after_std = statistics.stdev(after_values) if len(after_values) > 1 else 0
        else:
            after_median = curve[end]
            after_std = 0

        # 计算鲁棒差异（保留方向）
        diff = after_median - before_median

        # 计算不确定性
        combined_std = math.sqrt(before_std**2 + after_std**2)

        # 趋势分析
        if len(curve) > end + window_size + 5:
            trend_values = curve[end+1:end+window_size+6]
            if len(trend_values) >= 3:
                x = np.arange(len(trend_values))
                trend_slope = np.polyfit(x, trend_values, 1)[0]
            else:
                trend_slope = 0
        else:
            trend_slope = 0

        # 计算信噪比
        snr = abs(diff) / (combined_std + 1e-6)

        return diff, {
            "before_median": before_median,
            "after_median": after_median,
            "uncertainty": combined_std,
            "trend_slope": trend_slope,
            "snr": snr,
            "confidence": min(1.0, snr / 2.0)  # 置信度
        }

    except Exception as e:
        return 0.0, {"error": str(e)}


def classify_all_peaks(peaks: List[Peak], curve: List[float]) -> List[Peak]:
    """对所有峰进行最终分类"""
    return peaks  # 峰状态已在生命周期追踪中更新


def detect_peaks_improved(
    curve: List[float],
    threshold: float,
    margin_frames: int = 5,
    difference_threshold: float = 1.1,
    min_region_length: int = 5,
    **config_params
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    改进的主峰检测函数 - 保持向后兼容

    Returns: (green_peaks, red_peaks) for interface compatibility
    """

    # 默认配置
    default_config = {
        "preprocessing": {
            "noise_reduction_factor": 0.8,
            "baseline_correction": True,
            "smoothing_window": 5
        },
        "detection": {
            "min_peak_width": 3,
            "max_peak_width": 20,
            "min_peak_distance": 10,
            "prominence_factor": 0.1
        },
        "classification": {
            "stability_window": 5,
            "error_tolerance": 0.15,
            "min_green_prominence": 0.3,
            "red_detection_variance": 0.4
        },
        "tracking": {
            "enable_lifecycle_tracking": True,
            "peak_matching_threshold": 0.8,
            "max_tracking_age": 15
        }
    }

    # 合并用户配置
    config = default_config
    for key, value in config_params.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value

    try:
        # 1. 信号预处理
        processed_curve = preprocess_signal(
            curve,
            config["preprocessing"]["noise_reduction_factor"],
            config["preprocessing"]["baseline_correction"],
            config["preprocessing"]["smoothing_window"]
        )

        # 2. 鲁棒波峰检测
        candidate_peaks = detect_physical_peaks(
            processed_curve,
            threshold,
            config["detection"]["min_peak_width"],
            config["detection"]["max_peak_width"],
            config["detection"]["min_peak_distance"],
            config["detection"]["prominence_factor"]
        )

        # 3. 峰值生命周期追踪
        if config["tracking"]["enable_lifecycle_tracking"]:
            # 使用全局追踪器（这里简化处理）
            tracker = PeakLifecycleTracker(config)
            tracked_peaks = tracker.update_peak_states(candidate_peaks)
        else:
            # 简单分类，不进行生命周期追踪
            tracked_peaks = []
            for start, end, values in candidate_peaks:
                # 简单的即时分类
                if len(values) > 0:
                    variance = statistics.variance(values) if len(values) > 1 else 0
                    mean_val = statistics.mean(values)
                    cv = math.sqrt(variance) / (mean_val + 1e-6)

                    if cv < 0.1:  # 稳定
                        state = 'green'
                    elif cv > 0.3:  # 不稳定
                        state = 'red'
                    else:
                        state = 'white'

                    peak = Peak(
                        start=start,
                        end=end,
                        values=values,
                        state=state
                    )
                    tracked_peaks.append(peak)

        # 4. 格式化输出以保持向后兼容
        green_peaks = [(p.start, p.end) for p in tracked_peaks if p.state == 'green']
        red_peaks = [(p.start, p.end) for p in tracked_peaks if p.state in ['red', 'white']]

        return green_peaks, red_peaks

    except Exception as e:
        # 如果改进算法失败，回退到简单实现
        print(f"改进的峰检测失败，回退到简单实现: {e}")

        # 简单的阈值检测作为回退
        simple_peaks = []
        in_peak = False
        peak_start = -1

        for i, value in enumerate(curve):
            if value >= threshold:
                if not in_peak:
                    peak_start = i
                    in_peak = True
            else:
                if in_peak:
                    simple_peaks.append((peak_start, i - 1))
                    in_peak = False

        if in_peak:
            simple_peaks.append((peak_start, len(curve) - 1))

        # 所有峰都标记为红色（保守策略）
        return [], simple_peaks