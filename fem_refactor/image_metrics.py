from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def compute_average_gray(image: Image.Image) -> float:
    """Compute average gray value (0-255) of a PIL image."""
    gray = image.convert("L")
    histogram = gray.histogram()
    width, height = gray.size
    total_pixels = width * height
    if total_pixels <= 0:
        return 0.0

    total_sum = 0
    for value, count in enumerate(histogram):
        if count:
            total_sum += value * count
    return float(total_sum / total_pixels)


def compute_roi3_80_160_normalized(image: Image.Image) -> float:
    """
    Compute percentage of pixels with grayscale values in range [80, 160].

    Returns:
        Percentage of pixels in range [80, 160] (0-100)
    """
    gray = image.convert("L")
    histogram = gray.histogram()
    width, height = gray.size
    total_pixels = width * height

    if total_pixels <= 0:
        return 0.0

    # Sum pixel counts in range [80, 160]
    pixel_count = sum(histogram[80:161])  # 161 because upper bound is exclusive

    # Return as percentage (0-100)
    percentage = float((pixel_count / total_pixels) * 100)

    # Debug output
    print(
        f"[DEBUG] ROI3 image size: {width}x{height}={total_pixels} pixels, "
        f"80-160 count={pixel_count}, percentage={percentage:.2f}%"
    )

    return percentage


def compute_roi3_g1_g2_ranges(image: Image.Image) -> Tuple[float, float]:
    """
    Compute G1 and G2 grayscale range percentages for ROI3 image.

    Returns:
        (G1, G2) percentages:
        - G1: Percentage of pixels in range [80, 255] (0-100)
        - G2: Percentage of pixels in range [150, 255] (0-100)
    """
    gray = image.convert("L")
    histogram = gray.histogram()
    width, height = gray.size
    total_pixels = width * height

    if total_pixels <= 0:
        return 0.0, 0.0

    g1_count = sum(histogram[80:256])  # 256 because upper bound is exclusive
    g1_percentage = float((g1_count / total_pixels) * 100)

    g2_count = sum(histogram[150:256])  # 256 because upper bound is exclusive
    g2_percentage = float((g2_count / total_pixels) * 100)

    # Debug output with histogram distribution
    print(f"[DEBUG] ROI3 histogram: total={total_pixels}, G1(80-255)={g1_count}, G2(150-255)={g2_count}")

    return g1_percentage, g2_percentage


def compute_roi3_column_mean_diff(image: Image.Image) -> float:
    """
    计算ROI3图像每一列的平均灰度值的最大值与最小值之差
    """
    try:
        if image.mode != "L":
            image = image.convert("L")
        roi3_array = np.array(image)
        column_means = np.mean(roi3_array, axis=0)
        max_mean = float(np.max(column_means))
        min_mean = float(np.min(column_means))
        return max_mean - min_mean
    except Exception as e:
        print(f"[ERROR] 计算ROI3列灰度差值失败: {e}")
        return 0.0
