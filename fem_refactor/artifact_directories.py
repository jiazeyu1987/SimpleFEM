from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional


@dataclass(frozen=True)
class ArtifactDirs:
    tmp_root: str
    roi1_dir: str
    roi2_dir: str
    roi3_dir: str
    wave_dir: str
    wave1_dir: str


def prepare_artifact_dirs(
    *,
    base_dir: str,
    processing_mode: str,
    video_files: Optional[List[str]],
    statistics_manager: Any,
    create_video_folders: Callable[..., str],
    save_roi1: bool,
    save_roi2: bool,
    save_roi3: bool,
    save_wave: bool,
    save_roi1_wave: bool,
) -> ArtifactDirs:
    # Prepare per-video image save directories if enabled
    if processing_mode == "video" and video_files:
        current_stats = statistics_manager.current_statistics
        if current_stats and getattr(current_stats, "video_name", None):
            tmp_root = create_video_folders(
                video_files[0],
                current_stats.session_id,
                processing_mode,
                save_roi1,
                save_roi2,
                save_roi3,
                save_wave,
                save_roi1_wave,
            )
            roi1_dir = os.path.join(tmp_root, "roi1")
            roi2_dir = os.path.join(tmp_root, "roi2")
            roi3_dir = os.path.join(tmp_root, "roi3")
            wave_dir = os.path.join(tmp_root, "wave")
            wave1_dir = os.path.join(tmp_root, "wave1")
            return ArtifactDirs(
                tmp_root=str(tmp_root),
                roi1_dir=str(roi1_dir),
                roi2_dir=str(roi2_dir),
                roi3_dir=str(roi3_dir),
                wave_dir=str(wave_dir),
                wave1_dir=str(wave1_dir),
            )

        session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_root = os.path.join(base_dir, "tmp", session_start)
        if save_roi1 or save_roi2 or save_wave:
            os.makedirs(tmp_root, exist_ok=True)
        if save_roi1:
            os.makedirs(os.path.join(tmp_root, "roi1"), exist_ok=True)
        if save_roi2:
            os.makedirs(os.path.join(tmp_root, "roi2"), exist_ok=True)
        if save_wave:
            os.makedirs(os.path.join(tmp_root, "wave"), exist_ok=True)
        if save_roi1_wave:
            os.makedirs(os.path.join(tmp_root, "wave1"), exist_ok=True)
        roi1_dir = os.path.join(tmp_root, "roi1")
        roi2_dir = os.path.join(tmp_root, "roi2")
        roi3_dir = os.path.join(tmp_root, "roi3")
        wave_dir = os.path.join(tmp_root, "wave")
        wave1_dir = os.path.join(tmp_root, "wave1")
        return ArtifactDirs(
            tmp_root=str(tmp_root),
            roi1_dir=str(roi1_dir),
            roi2_dir=str(roi2_dir),
            roi3_dir=str(roi3_dir),
            wave_dir=str(wave_dir),
            wave1_dir=str(wave1_dir),
        )

    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_root = os.path.join(base_dir, "tmp", session_start)
    roi1_dir = os.path.join(tmp_root, "roi1")
    roi2_dir = os.path.join(tmp_root, "roi2")
    roi3_dir = os.path.join(tmp_root, "roi3")
    wave_dir = os.path.join(tmp_root, "wave")
    wave1_dir = os.path.join(tmp_root, "wave1")

    if save_roi1 or save_roi2 or save_roi3 or save_wave:
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

    return ArtifactDirs(
        tmp_root=str(tmp_root),
        roi1_dir=str(roi1_dir),
        roi2_dir=str(roi2_dir),
        roi3_dir=str(roi3_dir),
        wave_dir=str(wave_dir),
        wave1_dir=str(wave1_dir),
    )

