from typing import Optional, Tuple

import cv2
import numpy as np


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


def detect_green_intersection(image: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    纯检测函数：
    - 输入: BGR 图像数组 image (OpenCV 读取的格式)
    - 输出: 交点相对图像左上角的坐标 (x, y)，单位为像素；
      如果检测失败则返回 None。

    本函数不负责文件的读写，由外部调用者处理 IO。
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

    # 允许交点在图像外部，此时仍然返回其坐标。
    # 如果只关心图像内部，可以在这里加范围判断。
    return cx, cy

