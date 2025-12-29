from __future__ import annotations

from PIL import Image, ImageGrab


def capture_screen() -> Image.Image:
    return ImageGrab.grab()

