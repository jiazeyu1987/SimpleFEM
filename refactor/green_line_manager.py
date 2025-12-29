"""
绿线检测管理器 - 绿线交点检测和滤波

SimpleFEM Refactored Version
"""

import sys
import os
from typing import Optional, Tuple
from collections import deque

import numpy as np
from PIL import Image

# 添加父目录到路径以导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_detector import detect_green_intersection, IntersectionFilter
from refactor.config_manager import ConfigManager


class GreenLineManager:
    """
    绿线检测管理器

    功能:
    - 绿线交点检测
    - EMA/Velocity/Threshold 滤波
    - 防抖动处理
    """

    def __init__(self, config: ConfigManager):
        """
        初始化绿线检测管理器

        Args:
            config: 配置管理器
        """
        self._config = config

        # 交点滤波器
        self._intersection_filter: Optional[IntersectionFilter] = None

        # 最近交点历史
        self._intersection_history: deque = deque(maxlen=10)

        # 上次有效交点
        self._last_valid_intersection: Optional[Tuple[int, int]] = None

        self._initialize_filter()

    def _initialize_filter(self) -> None:
        """初始化交点滤波器"""
        algorithm = self._config.roi2_anti_jitter_algorithm

        if algorithm == "threshold":
            self._intersection_filter = IntersectionFilter(
                alpha=0.1,  # threshold模式下使用较小的alpha
                movement_threshold=self._config.roi2_movement_threshold,
                initialization_frames=3,
                stability_threshold=self._config.roi2_movement_threshold
            )
        else:  # ema 或默认
            self._intersection_filter = IntersectionFilter(
                alpha=self._config.roi2_anti_jitter_alpha,
                movement_threshold=20.0,  # 默认运动阈值
                initialization_frames=3,
                stability_threshold=8.0
            )

    def detect_intersection(self, roi1_image: Image.Image) -> Optional[Tuple[int, int]]:
        """
        检测绿线交点

        Args:
            roi1_image: ROI1图像

        Returns:
            交点坐标 (x, y) 或 None
        """
        # 设置图像边界
        if self._intersection_filter is not None:
            width, height = roi1_image.size
            self._intersection_filter.set_image_bounds(width, height)

        # 转换PIL图像到numpy数组
        roi1_array = np.array(roi1_image)

        # 调用原始检测函数
        intersection = detect_green_intersection(roi1_array)

        if intersection is not None:
            # 应用滤波
            if self._config.roi2_anti_jitter_enabled and self._intersection_filter is not None:
                x, y = intersection
                intersection = self._intersection_filter.filter_intersection(x, y)

            # 更新历史
            if intersection is not None:
                self._intersection_history.append(intersection)
                self._last_valid_intersection = intersection

        return intersection

    def get_last_valid_intersection(self) -> Optional[Tuple[int, int]]:
        """
        获取上次有效交点

        Returns:
            上次有效交点 (x, y) 或 None
        """
        return self._last_valid_intersection

    def reset(self) -> None:
        """重置检测状态"""
        self._intersection_history.clear()
        self._last_valid_intersection = None
        if self._intersection_filter is not None:
            # 重新初始化滤波器
            self._initialize_filter()

    @property
    def intersection_history(self) -> deque:
        """交点历史"""
        return self._intersection_history
