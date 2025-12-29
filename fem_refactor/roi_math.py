from __future__ import annotations

from typing import Dict, Optional, Tuple


def adjust_roi1_to_screen(
    screen_size: Tuple[int, int],
    roi_default: Dict[str, int],
) -> Tuple[int, int, int, int]:
    """
    Ensure ROI1 coordinates are within screen bounds.

    Returns (x1,y1,x2,y2).
    """
    screen_width, screen_height = screen_size
    x1 = roi_default.get("x1", 0)
    y1 = roi_default.get("y1", 0)
    x2 = roi_default.get("x2", screen_width)
    y2 = roi_default.get("y2", screen_height)

    if x2 > screen_width or y2 > screen_height or x1 < 0 or y1 < 0:
        x1 = max(0, min(x1, screen_width - 1))
        y1 = max(0, min(y1, screen_height - 1))
        x2 = max(x1 + 1, min(x2, screen_width))
        y2 = max(y1 + 1, min(y2, screen_height))

    return x1, y1, x2, y2


def compute_roi2_region(
    roi1_size: Tuple[int, int],
    center: Tuple[int, int],
    extension_params: Dict[str, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute ROI2/ROI3 region inside ROI1 based on intersection center and extension params.

    Returns (rx1,ry1,rx2,ry2) in ROI1-local coordinates, or None.
    """
    roi_width, roi_height = roi1_size
    cx, cy = center

    # Clamp center to ROI1 bounds for safety
    cx = max(0, min(roi_width - 1, cx))
    cy = max(0, min(roi_height - 1, cy))

    left = int(extension_params.get("left", 0))
    right = int(extension_params.get("right", 0))
    top = int(extension_params.get("top", 0))
    bottom = int(extension_params.get("bottom", 0))

    x1 = cx - left
    x2 = cx + right
    y1 = cy - top
    y2 = cy + bottom

    # Clamp to ROI1 bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(roi_width, x2)
    y2 = min(roi_height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2
