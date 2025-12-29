from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from green_detector import detect_green_intersection


class IntersectionManager:
    """
    Detect green-line intersection and provide stable fallback center.

    Maintains `last_intersection_roi` state, and resets filter on detection failure.
    """

    def __init__(self) -> None:
        self.last_intersection_roi: Optional[Tuple[int, int]] = None

    def detect_and_get_center(
        self,
        *,
        roi1_image: Image.Image,
        anti_jitter_config: Dict[str, Any],
        intersection_filter: Any,
    ) -> Tuple[Optional[Tuple[int, int]], Tuple[int, int]]:
        roi1_width, roi1_height = roi1_image.size

        roi_cv_image = cv2.cvtColor(np.array(roi1_image), cv2.COLOR_RGB2BGR)
        try:
            intersection = detect_green_intersection(roi_cv_image, anti_jitter_config, intersection_filter)
        except Exception as e:
            # Keep daemon running even if detection fails on this frame
            print(f"Warning: Green intersection detection failed: {e}")
            intersection = None
            # 如果检测失败，尝试重置防抖动滤波器
            if intersection_filter:
                try:
                    intersection_filter.reset()
                    intersection_filter.set_image_bounds(roi1_width, roi1_height)
                except Exception:
                    pass

        if intersection is not None:
            self.last_intersection_roi = intersection

        # Fallback for very first frames: use ROI1 center if we never had a hit
        if self.last_intersection_roi is not None:
            center_x, center_y = self.last_intersection_roi
        else:
            center_x = roi1_width // 2
            center_y = roi1_height // 2

        return intersection, (center_x, center_y)

