from typing import Optional, Tuple, Dict

import cv2
import numpy as np


class IntersectionFilter:
    """
    ROI2交点防抖动滤波器

    使用指数移动平均(EMA)算法平滑交点坐标，减少小幅随机抖动，
    同时保持对大运动的响应性。
    """

    def __init__(self, alpha: float = 0.25, movement_threshold: float = 6.0, initialization_frames: int = 3, stability_threshold: float = 20.0):
        """
        初始化交点滤波器

        Args:
            alpha: EMA平滑因子 (0.05-0.5)。值越小越平滑，值越大响应性越好
            movement_threshold: 运动阈值(像素)。小于此值的运动被视为抖动，大于此值的被视为真实运动
            initialization_frames: 初始化帧数。收集足够的初始数据后才开始滤波
            stability_threshold: 稳定阈值(像素)。小于此值的运动会被强力平滑
        """
        # 参数验证和标准化
        self.alpha = max(0.05, min(0.95, float(alpha)))  # 限制在0.05-0.95范围内
        self.movement_threshold = max(1.0, float(movement_threshold))  # 最小1像素
        self.initialization_frames = max(1, int(initialization_frames))  # 最少1帧
        self.stability_threshold = max(1.0, float(stability_threshold))  # 稳定阈值

        # 滤波状态变量
        self.filtered_x = None
        self.filtered_y = None
        self.frame_count = 0
        self.history_x = []
        self.history_y = []
        self.image_width = None
        self.image_height = None

        # 调试计数器
        self.large_movement_count = 0
        self.boundary_clamp_count = 0
        self.stability_count = 0
        self.small_movements_sum = 0.0

    def set_image_bounds(self, width: int, height: int):
        """设置图像边界，用于坐标限制"""
        self.image_width = max(1, int(width))
        self.image_height = max(1, int(height))

    def filter_intersection(self, current_x: int, current_y: int) -> Tuple[int, int]:
        """
        应用EMA滤波到交点坐标

        Args:
            current_x: 当前帧检测到的x坐标
            current_y: 当前帧检测到的y坐标

        Returns:
            滤波后的坐标 (x, y)
        """
        self.frame_count += 1

        # 边界检查：确保输入坐标在合理范围内
        if self.image_width is not None and self.image_height is not None:
            current_x = max(0, min(self.image_width - 1, current_x))
            current_y = max(0, min(self.image_height - 1, current_y))

        # 初始化阶段：收集数据并开始滤波
        if self.frame_count <= self.initialization_frames:
            self.history_x.append(current_x)
            self.history_y.append(current_y)

            if self.frame_count == self.initialization_frames:
                # 使用收集到的数据初始化滤波器
                self.filtered_x = sum(self.history_x) / len(self.history_x)
                self.filtered_y = sum(self.history_y) / len(self.history_y)
                # 初始化完成后立即应用一次滤波以保持连续性
                return self._apply_ema_filter(current_x, current_y)

            # 初始化期间就开始应用轻量滤波以避免跳跃
            if len(self.history_x) >= 2:
                # 使用简单的平均作为临时滤波
                temp_x = sum(self.history_x) / len(self.history_x)
                temp_y = sum(self.history_y) / len(self.history_y)
                return int(round(temp_x)), int(round(temp_y))

            return current_x, current_y

        # 计算从滤波位置的运动幅度
        movement = 0.0
        if self.filtered_x is not None and self.filtered_y is not None:
            movement = ((current_x - self.filtered_x) ** 2 + (current_y - self.filtered_y) ** 2) ** 0.5

            # 检测到大运动：重置滤波器以快速跟随
            if movement > self.movement_threshold:
                self.large_movement_count += 1
                self.filtered_x = float(current_x)
                self.filtered_y = float(current_y)
                return current_x, current_y  # 大运动直接通过

            # 小运动稳定性增强：在稳定阈值内使用更强的平滑
            if movement <= self.stability_threshold:
                self.stability_count += 1
                self.small_movements_sum += movement
                # 使用更小的alpha值进行强力平滑
                return self._apply_stability_filter(current_x, current_y)

        # 应用标准EMA滤波
        return self._apply_ema_filter(current_x, current_y)

    def _apply_ema_filter(self, current_x: int, current_y: int) -> Tuple[int, int]:
        """应用标准EMA滤波的内部方法"""
        if self.filtered_x is None:  # 第一次滤波
            self.filtered_x = float(current_x)
            self.filtered_y = float(current_y)
        else:
            # EMA公式: filtered_value = alpha * new_value + (1 - alpha) * old_filtered_value
            self.filtered_x = self.alpha * current_x + (1 - self.alpha) * self.filtered_x
            self.filtered_y = self.alpha * current_y + (1 - self.alpha) * self.filtered_y

        # 边界限制：确保滤波结果也在图像边界内
        if self.image_width is not None and self.image_height is not None:
            original_filtered_x = self.filtered_x
            original_filtered_y = self.filtered_y

            self.filtered_x = max(0, min(self.image_width - 1, self.filtered_x))
            self.filtered_y = max(0, min(self.image_height - 1, self.filtered_y))

            # 检查是否发生了边界限制
            if (abs(original_filtered_x - self.filtered_x) > 0.1 or
                abs(original_filtered_y - self.filtered_y) > 0.1):
                self.boundary_clamp_count += 1

        # 返回整数坐标
        return int(round(self.filtered_x)), int(round(self.filtered_y))

    def _apply_stability_filter(self, current_x: int, current_y: int) -> Tuple[int, int]:
        """应用强力稳定性滤波的内部方法（用于小运动）"""
        # 使用更小的alpha值进行强力平滑（alpha的1/3）
        stability_alpha = self.alpha * 0.33

        if self.filtered_x is None:  # 第一次滤波
            self.filtered_x = float(current_x)
            self.filtered_y = float(current_y)
        else:
            # 强力EMA滤波：给新值更小的权重
            self.filtered_x = stability_alpha * current_x + (1 - stability_alpha) * self.filtered_x
            self.filtered_y = stability_alpha * current_y + (1 - stability_alpha) * self.filtered_y

        # 边界限制：确保滤波结果也在图像边界内
        if self.image_width is not None and self.image_height is not None:
            self.filtered_x = max(0, min(self.image_width - 1, self.filtered_x))
            self.filtered_y = max(0, min(self.image_height - 1, self.filtered_y))

        # 返回整数坐标
        return int(round(self.filtered_x)), int(round(self.filtered_y))

    def reset(self):
        """重置滤波器状态"""
        self.filtered_x = None
        self.filtered_y = None
        self.frame_count = 0
        self.history_x = []
        self.history_y = []
        # 保留图像边界设置
        # 重置调试计数器
        self.large_movement_count = 0
        self.boundary_clamp_count = 0
        self.stability_count = 0
        self.small_movements_sum = 0.0

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        avg_small_movement = (self.small_movements_sum / max(1, self.stability_count)) if self.stability_count > 0 else 0.0
        return {
            "frame_count": self.frame_count,
            "filtered_position": (self.filtered_x, self.filtered_y),
            "large_movement_count": self.large_movement_count,
            "boundary_clamp_count": self.boundary_clamp_count,
            "stability_count": self.stability_count,
            "avg_small_movement": round(avg_small_movement, 2),
            "parameters": {
                "alpha": self.alpha,
                "movement_threshold": self.movement_threshold,
                "stability_threshold": self.stability_threshold,
                "initialization_frames": self.initialization_frames
            }
        }


def _detect_green_lines(
    image: np.ndarray,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    Detect two main green straight lines in the image.

    Returns two lines in (x1, y1, x2, y2) format, or None if not found.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rough green range for the lines in the sample images.
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Edge detection and Hough transform to find line segments.
    edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=80,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 2:
        return None

    # Sort lines by length (descending).
    def line_length_sq(l):
        x1, y1, x2, y2 = l
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    lines_list = [tuple(l[0]) for l in lines]
    lines_list.sort(key=line_length_sq, reverse=True)

    # Select two non-parallel lines with sufficient angle difference.
    def line_angle(l):
        x1, y1, x2, y2 = l
        return np.arctan2(y2 - y1, x2 - x1)

    first_line = lines_list[0]
    first_angle = line_angle(first_line)

    chosen_second = None
    for candidate in lines_list[1:]:
        angle = line_angle(candidate)
        diff = abs(angle - first_angle)
        # Normalize to [0, pi].
        diff = min(diff, np.pi - diff)
        if diff > np.deg2rad(10):  # Not almost parallel.
            chosen_second = candidate
            break

    if chosen_second is None:
        return None

    return first_line, chosen_second


def _compute_intersection(
    line1: Tuple[int, int, int, int],
    line2: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two infinite lines.
    Lines are given as (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    px = px_num / denom
    py = py_num / denom
    return float(px), float(py)


def detect_green_intersection(image: np.ndarray,
                           anti_jitter_config: Optional[Dict] = None,
                           filter_instance: Optional[IntersectionFilter] = None) -> Optional[Tuple[int, int]]:
    """
    检测绿色线交点，支持可选的防抖动滤波

    Args:
        image: BGR 图像数组 (OpenCV 读取的格式)
        anti_jitter_config: 防抖动配置参数
        filter_instance: 交点滤波器实例(用于跨帧状态保持)

    Returns:
        交点相对图像左上角的坐标 (x, y)，单位为像素；
        如果检测失败则返回 None。

    注意:
        - 当启用防抖动时，返回的坐标可能经过滤波平滑处理
        - filter_instance参数用于在连续帧间保持滤波状态
        - 本函数不负责文件的读写，由外部调用者处理 IO
    """
    if image is None:
        raise ValueError("Input image is None.")

    h, w = image.shape[:2]

    detected = _detect_green_lines(image)
    if detected is None:
        return None

    line1, line2 = detected
    intersection = _compute_intersection(line1, line2)
    if intersection is None:
        return None

    x, y = intersection
    cx = int(round(x))
    cy = int(round(y))

    # 设置滤波器的图像边界（如果有滤波器实例）
    if filter_instance is not None:
        filter_instance.set_image_bounds(w, h)

    # 应用防抖动滤波（如果启用）
    if (anti_jitter_config and filter_instance and
        anti_jitter_config.get("enabled", False)):
        try:
            # 原始坐标用于调试
            raw_x, raw_y = cx, cy
            cx, cy = filter_instance.filter_intersection(cx, cy)

            # 调试信息（每100帧输出一次）
            if hasattr(filter_instance, 'frame_count') and filter_instance.frame_count % 100 == 0:
                debug_info = filter_instance.get_debug_info()
                if 'large_movement_count' in debug_info:
                    # EMA滤波器
                    print(f"[防抖动调试] 帧{filter_instance.frame_count}: "
                          f"原始({raw_x},{raw_y}) -> 滤波后({cx},{cy}), "
                          f"大运动次数: {debug_info['large_movement_count']}, "
                          f"边界限制次数: {debug_info.get('boundary_clamp_count', 0)}")
                else:
                    # 阈值滤波器
                    print(f"[阈值防抖动调试] 帧{filter_instance.frame_count}: "
                          f"原始({raw_x},{raw_y}) -> 滤波后({cx},{cy}), "
                          f"更新次数: {debug_info['update_count']}, "
                          f"稳定率: {debug_info.get('stability_rate', 0):.1f}%")

        except Exception as e:
            # 如果滤波失败，记录警告并返回原始坐标
            print(f"Warning: Anti-jitter filtering failed: {e}, using raw intersection")
            # 尝试重置滤波器状态
            try:
                if filter_instance:
                    filter_instance.reset()
                    filter_instance.set_image_bounds(w, h)
            except:
                pass

    # 最终边界检查：确保返回的坐标在图像范围内
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    return cx, cy


def detect_green_intersection_legacy(image: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    向后兼容的交点检测函数（不包含防抖动功能）

    Args:
        image: BGR 图像数组 (OpenCV 读取的格式)

    Returns:
        交点相对图像左上角的坐标 (x, y)，单位为像素；
        如果检测失败则返回 None。
    """
    return detect_green_intersection(image, anti_jitter_config=None, filter_instance=None)

