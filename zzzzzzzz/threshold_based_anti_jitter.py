from typing import Optional, Tuple, Dict
import numpy as np


class ThresholdIntersectionFilter:
    """
    基于阈值的ROI2交点防抖动滤波器

    策略：当交点变化小于阈值时，ROI2区域完全不动；
         只有当变化超过阈值时，才更新ROI2位置。
    """

    def __init__(self, movement_threshold: float = 20.0, initialization_frames: int = 3):
        """
        初始化阈值式滤波器

        Args:
            movement_threshold: 运动阈值(像素)。当交点移动超过此值时才更新ROI2
            initialization_frames: 初始化帧数。前N帧收集初始稳定位置
        """
        self.movement_threshold = max(1.0, float(movement_threshold))
        self.initialization_frames = max(1, int(initialization_frames))

        # 滤波状态变量
        self.stable_x = None
        self.stable_y = None
        self.frame_count = 0
        self.initial_positions = []
        self.image_width = None
        self.image_height = None

        # 调试计数器
        self.update_count = 0
        self.ignore_count = 0
        self.large_movement_count = 0

    def set_image_bounds(self, width: int, height: int):
        """设置图像边界，用于坐标限制"""
        self.image_width = max(1, int(width))
        self.image_height = max(1, int(height))

    def filter_intersection(self, current_x: int, current_y: int) -> Tuple[int, int]:
        """
        应用阈值滤波到交点坐标

        Args:
            current_x: 当前帧检测到的x坐标
            current_y: 当前帧检测到的y坐标

        Returns:
            滤波后的坐标 (x, y) - 小于阈值时返回稳定位置，超过阈值时返回新位置
        """
        self.frame_count += 1

        # 边界检查
        if self.image_width is not None and self.image_height is not None:
            current_x = max(0, min(self.image_width - 1, current_x))
            current_y = max(0, min(self.image_height - 1, current_y))

        # 初始化阶段：收集初始位置
        if self.frame_count <= self.initialization_frames:
            self.initial_positions.append((current_x, current_y))

            if self.frame_count == self.initialization_frames:
                # 计算初始稳定位置（平均值）
                avg_x = sum(pos[0] for pos in self.initial_positions) / len(self.initial_positions)
                avg_y = sum(pos[1] for pos in self.initial_positions) / len(self.initial_positions)
                self.stable_x = avg_x
                self.stable_y = avg_y

                print(f"[阈值防抖动] 初始化完成，稳定位置: ({self.stable_x:.1f}, {self.stable_y:.1f})")
                return int(round(self.stable_x)), int(round(self.stable_y))

            # 初始化期间使用第一个位置作为稳定位置
            if self.frame_count == 1:
                self.stable_x = float(current_x)
                self.stable_y = float(current_y)

            return int(round(self.stable_x)), int(round(self.stable_y))

        # 计算从稳定位置的距离
        if self.stable_x is not None and self.stable_y is not None:
            distance = ((current_x - self.stable_x) ** 2 + (current_y - self.stable_y) ** 2) ** 0.5

            # 判断是否超过阈值
            if distance > self.movement_threshold:
                # 超过阈值，更新稳定位置
                old_x, old_y = self.stable_x, self.stable_y
                self.stable_x = float(current_x)
                self.stable_y = float(current_y)
                self.update_count += 1
                self.large_movement_count += 1

                # 调试输出（每10次更新输出一次）
                if self.update_count % 10 == 1:
                    print(f"[阈值防抖动] 大幅移动更新: ({old_x:.1f},{old_y:.1f}) → ({self.stable_x:.1f},{self.stable_y:.1f}), 距离: {distance:.1f}px")

                return current_x, current_y
            else:
                # 小于阈值，保持稳定位置不变
                self.ignore_count += 1
                return int(round(self.stable_x)), int(round(self.stable_y))

        # 默认返回原始坐标
        return current_x, current_y

    def reset(self):
        """重置滤波器状态"""
        self.stable_x = None
        self.stable_y = None
        self.frame_count = 0
        self.initial_positions = []
        self.update_count = 0
        self.ignore_count = 0
        self.large_movement_count = 0
        print("[阈值防抖动] 滤波器已重置")

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        total_processed = self.update_count + self.ignore_count
        stability_rate = (self.ignore_count / total_processed * 100) if total_processed > 0 else 0

        return {
            "frame_count": self.frame_count,
            "filtered_position": (self.stable_x, self.stable_y),
            "update_count": self.update_count,
            "ignore_count": self.ignore_count,
            "large_movement_count": self.large_movement_count,
            "boundary_clamp_count": 0,  # 阈值式滤波器不需要边界限制
            "stability_count": self.ignore_count,  # 兼容性
            "small_movements_sum": 0.0,  # 阈值式滤波器没有小运动求和
            "avg_small_movement": 0.0,
            "stability_rate": round(stability_rate, 1),
            "parameters": {
                "movement_threshold": self.movement_threshold,
                "initialization_frames": self.initialization_frames,
                "alpha": 0.0,  # 阈值式滤波器没有alpha参数
                "stability_threshold": self.movement_threshold
            }
        }


def detect_green_intersection_threshold(image: np.ndarray,
                                       threshold_config: Optional[Dict] = None,
                                       filter_instance: Optional[ThresholdIntersectionFilter] = None) -> Optional[Tuple[int, int]]:
    """
    检测绿色线交点，使用阈值式防抖动

    Args:
        image: BGR 图像数组
        threshold_config: 防抖动配置参数
        filter_instance: 阈值滤波器实例

    Returns:
        交点坐标 (x, y)，或 None
    """
    if image is None:
        raise ValueError("Input image is None.")

    # 导入原有的检测函数
    from green_detector import _detect_green_lines, _compute_intersection

    h, w = image.shape[:2]

    # 检测绿色线
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

    # 设置滤波器边界
    if filter_instance is not None:
        filter_instance.set_image_bounds(w, h)

    # 应用阈值防抖动
    if (threshold_config and filter_instance and
        threshold_config.get("enabled", False)):
        try:
            raw_x, raw_y = cx, cy
            cx, cy = filter_instance.filter_intersection(cx, cy)

            # 调试信息（每50帧输出一次）
            if filter_instance.frame_count % 50 == 0:
                debug_info = filter_instance.get_debug_info()
                print(f"[阈值防抖动] 帧{filter_instance.frame_count}: "
                      f"原始({raw_x},{raw_y}) → 阈值滤波后({cx},{cy}), "
                      f"更新次数: {debug_info['update_count']}, "
                      f"稳定率: {debug_info['stability_rate']:.1f}%")

        except Exception as e:
            print(f"Warning: 阈值防抖动失败: {e}, using raw intersection")
            if filter_instance:
                filter_instance.reset()
                filter_instance.set_image_bounds(w, h)

    # 最终边界检查
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    return cx, cy