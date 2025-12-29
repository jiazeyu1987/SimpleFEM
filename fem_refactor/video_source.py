from __future__ import annotations

import glob
import os
from typing import List

import cv2
from PIL import Image


def discover_video_files(video_path: str) -> List[str]:
    """发现文件夹中的所有视频文件"""
    if not os.path.exists(video_path):
        raise ValueError(f"Video directory does not exist: {video_path}")

    # 支持的视频文件扩展名
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm"]

    video_files = []
    for ext in video_extensions:
        # 搜索文件夹中的匹配文件
        pattern = os.path.join(video_path, ext)
        video_files.extend(glob.glob(pattern))
        # 也搜索大写扩展名
        pattern = os.path.join(video_path, ext.upper())
        video_files.extend(glob.glob(pattern))

    # 去重并排序
    video_files = sorted(list(set(video_files)))

    if not video_files:
        raise ValueError(f"No video files found in directory: {video_path}")

    return video_files


def initialize_video_capture(video_path: str):
    """初始化视频捕获器"""
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
    return video_cap


def get_video_fps(video_cap, default_fps: float = 30.0) -> float:
    try:
        fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        fps = 0.0
    if not fps or fps <= 0:
        return float(default_fps)
    return float(fps)


def get_video_frame(video_cap, loop_enabled: bool = False, frame_step: int = 1):
    """
    从视频获取帧，返回PIL图像或None。

    frame_step>1 时，会在视频时间轴上“跳帧取样”：每次返回 1 帧，并将读取位置前进约 frame_step 帧。
    这让 roi_capture.frame_rate 在 video 模式下真正对应“每秒采样多少帧”，而不是仅仅降低处理速度。
    """
    if frame_step is None:
        frame_step = 1
    try:
        frame_step = int(frame_step)
    except Exception:
        frame_step = 1
    frame_step = max(1, frame_step)

    frame = None
    for _ in range(frame_step):
        ret, frame = video_cap.read()
        if ret:
            continue
        if not loop_enabled:
            return None
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_cap.read()
        if not ret:
            return None

    if frame is None:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

