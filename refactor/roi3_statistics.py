"""
ROI3 统计计算 - G1/G2 范围和列差值分析

SimpleFEM Refactored Version
"""

import numpy as np
from PIL import Image
from typing import Tuple


class ROI3Statistics:
    """
    ROI3 统计计算器

    功能:
    - 计算 G1/G2 像素百分比（用于绿/红覆盖判定）
    - 计算列灰度差值（用于绿/红覆盖判定）
    - 计算归一化灰度值（0-160范围）
    """

    @staticmethod
    def compute_g1_g2_ranges(image: Image.Image) -> Tuple[float, float]:
        """
        计算 G1/G2 像素百分比

        Args:
            image: ROI3 图像

        Returns:
            (g1_percent, g2_percent)
                g1_percent: [80, 255] 范围的像素百分比
                g2_percent: [150, 255] 范围的像素百分比
        """
        # 转换为灰度图
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)

        # 计算 G1: [80, 255] 范围
        g1_mask = (gray_array >= 80) & (gray_array <= 255)
        g1_percent = (np.sum(g1_mask) / gray_array.size) * 100

        # 计算 G2: [150, 255] 范围
        g2_mask = (gray_array >= 150) & (gray_array <= 255)
        g2_percent = (np.sum(g2_mask) / gray_array.size) * 100

        return float(g1_percent), float(g2_percent)

    @staticmethod
    def compute_column_mean_diff(image: Image.Image) -> float:
        """
        计算列灰度差值

        Args:
            image: ROI3 图像

        Returns:
            列灰度差值（最大列均值 - 最小列均值）
        """
        # 转换为灰度图
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)

        # 计算每列的平均灰度值
        column_means = np.mean(gray_array, axis=0)

        # 计算最大值和最小值的差
        column_diff = float(np.max(column_means) - np.min(column_means))

        return column_diff

    @staticmethod
    def compute_normalized_80_160(image: Image.Image) -> float:
        """
        计算归一化灰度值（0-160范围）

        Args:
            image: ROI3 图像

        Returns:
            归一化后的平均灰度值
        """
        # 转换为灰度图
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)

        # 限制在 [0, 160] 范围
        normalized = np.clip(gray_array, 0, 160)

        # 计算平均值
        avg_value = float(np.mean(normalized))

        return avg_value

    @staticmethod
    def compute_80_160_percentage(image: Image.Image) -> float:
        """
        计算像素在[80, 160]范围内的百分比（与原始代码保持一致）

        Args:
            image: ROI3 图像

        Returns:
            百分比 (0-100)
        """
        # 转换为灰度图
        gray_image = image.convert('L')
        histogram = gray_image.histogram()
        width, height = gray_image.size
        total_pixels = width * height

        if total_pixels <= 0:
            return 0.0

        # 计算在[80, 160]范围内的像素数量
        pixel_count = sum(histogram[80:161])  # 161因为上限是独占的

        # 返回百分比 (0-100)
        percentage = float((pixel_count / total_pixels) * 100)

        return percentage

    @staticmethod
    def compute_all(image: Image.Image) -> dict:
        """
        计算所有 ROI3 统计值

        Args:
            image: ROI3 图像

        Returns:
            包含所有统计值的字典
        """
        g1, g2 = ROI3Statistics.compute_g1_g2_ranges(image)
        column_diff = ROI3Statistics.compute_column_mean_diff(image)
        normalized = ROI3Statistics.compute_normalized_80_160(image)
        percentage_80_160 = ROI3Statistics.compute_80_160_percentage(image)

        return {
            'g1_percent': g1,
            'g2_percent': g2,
            'column_diff': column_diff,
            'normalized_80_160': normalized,
            'percentage_80_160': percentage_80_160
        }
