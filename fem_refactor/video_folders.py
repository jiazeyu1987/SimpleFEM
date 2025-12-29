from __future__ import annotations

import os
import re
import time
from datetime import datetime


def sanitize_video_name(video_name: str) -> str:
    """清理视频名称用于文件夹创建"""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", video_name)
    sanitized = sanitized.strip("._")[:50]
    return sanitized or f"video_{int(time.time())}"


def create_video_folders(
    *,
    base_dir: str,
    video_path: str,
    session_id: str,
    processing_mode: str,
    save_roi1: bool,
    save_roi2: bool,
    save_roi3: bool,
    save_wave: bool,
    save_roi1_wave: bool = False,
) -> str:
    """创建每视频的文件夹结构"""
    if processing_mode == "video":
        # 批量模式：使用视频名称
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sanitized_name = sanitize_video_name(video_name)
        tmp_root = os.path.join(base_dir, "tmp", sanitized_name)
    else:
        # 屏幕模式：使用基于会话的命名（原有行为）
        session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_root = os.path.join(base_dir, "tmp", session_start)

    # 创建子文件夹
    roi1_dir = os.path.join(tmp_root, "roi1")
    roi2_dir = os.path.join(tmp_root, "roi2")
    roi3_dir = os.path.join(tmp_root, "roi3")
    wave_dir = os.path.join(tmp_root, "wave")
    wave1_dir = os.path.join(tmp_root, "wave1")

    # 根据配置创建目录
    if save_roi1 or save_roi2 or save_roi3 or save_wave or save_roi1_wave:
        os.makedirs(tmp_root, exist_ok=True)
    if save_roi1:
        os.makedirs(roi1_dir, exist_ok=True)
    if save_roi2:
        os.makedirs(roi2_dir, exist_ok=True)
    if save_roi3:
        os.makedirs(roi3_dir, exist_ok=True)
    if save_wave:
        os.makedirs(wave_dir, exist_ok=True)
    if save_roi1_wave:
        os.makedirs(wave1_dir, exist_ok=True)

    return tmp_root

