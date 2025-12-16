"""
Simple ROI daemon:
 - Every second, capture ROI1 from screen using PIL.ImageGrab
 - Detect green line intersection inside ROI1 using existing green_detector
 - Around the latest intersection point, extract ROI2 according to roi2_config.extension_params
 - Compute ROI2 average gray value and push into a fixed-length buffer (length 100)
 - Run peak detection using backend.app.peak_detection.detect_peaks with fem_config parameters
 - Log per-second summary to a daily-rotating log file (backend/logs/roi_peak_daemon.log)

Usage:
    python simple_roi_daemon.py
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageGrab
import cv2
import matplotlib.pyplot as plt


def _get_base_dir() -> str:
    """
    Resolve base directory both for source (.py) and frozen (.exe) modes.

    When packaged with PyInstaller, sys.frozen is True and sys.executable
    points to the .exe location. In source mode, use this file's directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _get_base_dir()


def _setup_import_paths() -> None:
    """
    Ensure we can import local peak_detection and green_detector modules.

    All required files (simple_roi_daemon.py, peak_detection.py, green_detector.py,
    simple_fem_config.json) are expected to be in the same SimpleFEM directory.
    """
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)


_setup_import_paths()

from peak_detection import detect_peaks  # type: ignore  # noqa: E402
from green_detector import detect_green_intersection  # type: ignore  # noqa: E402
from safe_peak_statistics import safe_statistics  # type: ignore  # noqa: E402


def initialize_video_capture(video_path: str):
    """初始化视频捕获器"""
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
    return video_cap


def get_video_frame(video_cap, loop_enabled=False):
    """从视频获取帧，返回PIL图像或None"""
    ret, frame = video_cap.read()
    if not ret:
        if loop_enabled:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = video_cap.read()
            if not ret:
                return None
        else:
            return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def load_fem_config() -> Dict:
    """Load simple_fem_config.json from backend/app (only fields needed by this script)."""
    config_path = os.path.join(BASE_DIR,  "simple_fem_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def setup_peak_logger() -> logging.Logger:
    """Create a logger that writes plain text lines and rotates daily."""
    logger = logging.getLogger("roi_peak_daemon")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Keep logs local to SimpleFEM project directory
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "roi_peak_daemon.log")

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def adjust_roi1_to_screen(
    screen_size: Tuple[int, int],
    roi_default: Dict[str, int],
) -> Tuple[int, int, int, int]:
    """
    Ensure ROI1 coordinates stay inside screen bounds.
    Mirrors the safety checks used in RoiCaptureService.
    """
    screen_width, screen_height = screen_size
    x1 = roi_default.get("x1", 0)
    y1 = roi_default.get("y1", 0)
    x2 = roi_default.get("x2", screen_width)
    y2 = roi_default.get("y2", screen_height)

    if (
        x2 > screen_width
        or y2 > screen_height
        or x1 < 0
        or y1 < 0
    ):
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
    Compute ROI2 region inside ROI1 using extension_params around the given center.

    center is in ROI1-local coordinates (0,0) at ROI1 top-left.
    Returns (x1, y1, x2, y2) in ROI1-local coordinates, or None if invalid.
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


def run_daemon() -> None:
    """
    Main loop:
      - capture ROI1
      - detect/update line intersection
      - extract ROI2
      - update gray buffer and run peak detection
      - log results at configured frame_rate
    """
    config = load_fem_config()

    # 检测处理模式
    processing_mode = config.get("processing_mode", "screen")
    video_cap = None

    if processing_mode == "video":
        video_config = config.get("video_processing", {})
        video_path = video_config.get("video_path", "")
        if not video_path:
            raise ValueError("Video mode enabled but no video_path specified in config")
        video_cap = initialize_video_capture(video_path)
        print(f"视频模式: {video_path}")

    try:
        roi_default = config.get("roi_capture", {}).get("default_config", {})
        roi2_config = config.get("roi_capture", {}).get("roi2_config", {})
        extension_params = roi2_config.get("extension_params", {})

        data_processing = config.get("data_processing", {})
        save_roi1 = bool(data_processing.get("save_roi1", False))
        save_roi2 = bool(data_processing.get("save_roi2", False))
        save_wave = bool(data_processing.get("save_wave", False))
        # only_delect == True: save ROI1/ROI2/wave only when peaks are detected
        only_delect = bool(data_processing.get("only_delect", False))

        peak_conf = config.get("peak_detection", {})
        threshold = float(peak_conf.get("threshold", 105.0))
        margin_frames = int(peak_conf.get("margin_frames", 5))
        diff_threshold = float(peak_conf.get("difference_threshold", 0.5))
        # 新增：阈值前后“静默”帧数要求（升阈值前 X 帧和降阈值后 X 帧都不能超过阈值）
        silence_frames = int(peak_conf.get("silence_frames", 0))
        pre_post_avg_frames = int(peak_conf.get("pre_post_avg_frames", 5))
        # 新增：自适应阈值（基于 ROI2 灰度历史均值）
        adaptive_threshold_enabled = bool(peak_conf.get("adaptive_threshold_enabled", False))
        threshold_over_mean_ratio = float(peak_conf.get("threshold_over_mean_ratio", 0.1))
        history_mean_min_samples = int(peak_conf.get("history_mean_min_samples", 30))
        min_region_length = int(peak_conf.get("min_region_length", 1))

        logger = setup_peak_logger()
        # Store only the latest 100 gray values for waveform / peak detection
        gray_buffer: Deque[float] = deque(maxlen=100)
        # Track a session-wide "background mean" using a gated incremental mean:
        # only update the mean when the current gray value is below the current
        # (mean-based) threshold, so peak frames do not contaminate the baseline.
        bg_count: int = 0
        bg_mean: float = 0.0
        last_intersection_roi: Optional[Tuple[int, int]] = None

        # Prepare per-session image save directories if enabled
        session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_root = os.path.join(BASE_DIR, "tmp", session_start)
        roi1_dir = os.path.join(tmp_root, "roi1")
        roi2_dir = os.path.join(tmp_root, "roi2")
        wave_dir = os.path.join(tmp_root, "wave")

        if save_roi1 or save_roi2 or save_wave:
            os.makedirs(tmp_root, exist_ok=True)
        if save_roi1:
            os.makedirs(roi1_dir, exist_ok=True)
        if save_roi2:
            os.makedirs(roi2_dir, exist_ok=True)
        if save_wave:
            os.makedirs(wave_dir, exist_ok=True)

        frame_index = 0

        # Use roi_capture.frame_rate as loop frequency
        roi_frame_rate = config.get("roi_capture", {}).get("frame_rate", 1)
        try:
            roi_frame_rate = float(roi_frame_rate)
        except Exception:
            roi_frame_rate = 1.0
        if roi_frame_rate <= 0:
            roi_frame_rate = 1.0
        interval_seconds = 1.0 / roi_frame_rate

        while True:
            loop_start = time.time()
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            log_line: Optional[str] = None

            try:
                frame_index += 1

                # 1. Capture image (screen or video frame)
                if processing_mode == "video":
                    video_config = config.get("video_processing", {})
                    loop_enabled = video_config.get("loop_enabled", False)
                    screen = get_video_frame(video_cap, loop_enabled)
                    if screen is None:
                        print("视频播放结束")
                        break
                    screen_width, screen_height = screen.size
                else:
                    screen = ImageGrab.grab()
                    screen_width, screen_height = screen.size

                # 2. Get ROI1 region and crop
                x1, y1, x2, y2 = adjust_roi1_to_screen(
                    (screen_width, screen_height),
                    roi_default,
                )
                roi1_image = screen.crop((x1, y1, x2, y2))
                roi1_width, roi1_height = roi1_image.size

                # 3. Detect green line intersection in ROI1
                roi_cv_image = cv2.cvtColor(
                    np.array(roi1_image),
                    cv2.COLOR_RGB2BGR,
                )
                try:
                    intersection = detect_green_intersection(roi_cv_image)
                except Exception:
                    # Keep daemon running even if detection fails on this frame
                    intersection = None

                if intersection is not None:
                    last_intersection_roi = intersection

                # Fallback for very first frames: use ROI1 center if we never had a hit
                if last_intersection_roi is not None:
                    center_x, center_y = last_intersection_roi
                else:
                    center_x = roi1_width // 2
                    center_y = roi1_height // 2

                # 4. Compute ROI2 region and crop
                roi2_region = compute_roi2_region(
                    (roi1_width, roi1_height),
                    (center_x, center_y),
                    extension_params,
                )

                roi2_gray: Optional[float] = None
                roi2_image: Optional[Image.Image] = None

                if roi2_region is not None:
                    rx1, ry1, rx2, ry2 = roi2_region
                    roi2_image = roi1_image.crop((rx1, ry1, rx2, ry2))
                    roi2_gray = compute_average_gray(roi2_image)
                    gray_buffer.append(roi2_gray)

                    # Update background mean with gating (exclude peak/high frames).
                    if bg_count == 0:
                        bg_count = 1
                        bg_mean = roi2_gray
                    else:
                        if adaptive_threshold_enabled and bg_mean > 0:
                            gate_threshold = bg_mean * (1.0 + threshold_over_mean_ratio)
                        else:
                            gate_threshold = threshold

                        if roi2_gray < gate_threshold:
                            bg_count += 1
                            bg_mean += (roi2_gray - bg_mean) / bg_count

                # 5. Run peak detection on current gray buffer
                green_peaks: List[Tuple[int, int]] = []
                red_peaks: List[Tuple[int, int]] = []
                threshold_used = threshold

                if gray_buffer:
                    curve = list(gray_buffer)
                    # Compute adaptive threshold if enabled and enough history is available.
                    # threshold_used = historical_mean * (1 + ratio)
                    if (
                        adaptive_threshold_enabled
                        and bg_count >= history_mean_min_samples
                        and bg_mean > 0
                    ):
                        threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                    try:
                        green_peaks_raw, red_peaks_raw = detect_peaks(
                            curve,
                            threshold=threshold_used,
                            marginFrames=margin_frames,
                            differenceThreshold=diff_threshold,
                            silenceFrames=silence_frames,
                            avgFrames=pre_post_avg_frames,
                        )
                    except Exception:
                        green_peaks_raw, red_peaks_raw = [], []

                    # Apply min_region_length filter
                    green_peaks = [
                        (start, end)
                        for (start, end) in green_peaks_raw
                        if (end - start + 1) >= min_region_length
                    ]
                    red_peaks = [
                        (start, end)
                        for (start, end) in red_peaks_raw
                        if (end - start + 1) >= min_region_length
                    ]

                green_count = len(green_peaks)
                red_count = len(red_peaks)
                last_green = green_peaks[-1] if green_peaks else None
                last_green_repr = (
                    f"[{last_green[0]},{last_green[1]}]" if last_green else "[]"
                )

                gray_str = (
                    f"{roi2_gray:.1f}" if roi2_gray is not None else "nan"
                )

                # Add peaks to statistics for Excel data collection (task requirement)
                try:
                    # Prepare ROI2 information for statistics
                    roi2_info = None
                    if roi2_region is not None:
                        rx1, ry1, rx2, ry2 = roi2_region
                        roi2_info = {
                            'x1': rx1, 'y1': ry1, 'x2': rx2, 'y2': ry2,
                            'width': rx2 - rx1, 'height': ry2 - ry1
                        }

                    # Add detected peaks to statistics with deduplication
                    safe_statistics.add_peaks_from_daemon(
                        frame_index=frame_index,
                        green_peaks=green_peaks,
                        red_peaks=red_peaks,
                        curve=list(gray_buffer) if gray_buffer else [],
                        intersection=last_intersection_roi,
                        roi2_info=roi2_info,
                        gray_value=roi2_gray,
                        difference_threshold=diff_threshold,
                        pre_post_avg_frames=pre_post_avg_frames,
                        threshold_used=threshold_used,
                        bg_mean=(bg_mean if bg_count > 0 else None),
                    )

                except Exception as e:
                    # Keep daemon running even if statistics collection fails
                    print(f"Statistics collection error: {e}")

                # Decide whether to save images/wave for this frame
                has_peak = (green_count > 0) or (red_count > 0)
                should_save = (not only_delect) or has_peak

            # Optionally save ROI1 image
                if should_save and save_roi1:
                    roi1_path = os.path.join(roi1_dir, f"roi1_{frame_index:06d}.png")
                    try:
                        roi1_image.save(roi1_path)
                    except Exception:
                        # Ignore individual save errors to keep daemon running
                        pass

                # Optionally save ROI2 image (align index with ROI1 saves)
                if should_save and save_roi2 and roi2_image is not None:
                    roi2_path = os.path.join(roi2_dir, f"roi2_{frame_index:06d}.png")
                    try:
                        roi2_image.save(roi2_path)
                    except Exception:
                        pass

                # Save wave plot (curve before detection, but annotated with detection result)
                if should_save and save_wave and gray_buffer:
                    try:
                        wave_path = os.path.join(
                            wave_dir,
                            f"wave_{frame_index:06d}.png",
                        )

                        fig, ax = plt.subplots(figsize=(8, 3))
                        x = list(range(len(curve)))
                        ax.plot(x, curve, color="black", linewidth=1)

                        # Draw session-wide background mean (adaptive threshold baseline)
                        if bg_count > 0:
                            ax.axhline(
                                bg_mean,
                                color="blue",
                                linestyle="--",
                                linewidth=1,
                                label="bg_mean",
                            )

                        # Highlight green and red regions (slightly expanded for readability)
                        for start, end in green_peaks:
                            s = max(0, start - 1)
                            e = min(len(curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = curve[s : e + 1]
                            ax.plot(xs, ys, color="green", linewidth=2)

                        for start, end in red_peaks:
                            s = max(0, start - 1)
                            e = min(len(curve) - 1, end + 1)
                            xs = list(range(s, e + 1))
                            ys = curve[s : e + 1]
                            ax.plot(xs, ys, color="red", linewidth=2)

                        ax.set_xlabel("Frame index in buffer")
                        ax.set_ylabel("Gray value")
                        ax.set_title("ROI2 gray waveform with peaks")
                        ax.grid(True, linestyle="--", alpha=0.3)
                        if bg_count > 0:
                            ax.legend(loc="best", fontsize=8)
                        fig.tight_layout()
                        fig.savefig(wave_path)
                        plt.close(fig)
                    except Exception:
                        # Ignore individual plotting/saving errors
                        pass

            # Build log line; when only_delect is True, only log frames with peaks
                if (not only_delect) or has_peak:
                    log_line = (
                        f"{ts} gray={gray_str} "
                        f"green_peaks={green_count} red_peaks={red_count} "
                        f"last_green={last_green_repr}"
                    )
                else:
                    log_line = None
            except KeyboardInterrupt:
                logger.info(f"{ts} INFO=daemon_stopped_by_user")
                break
            except Exception as e:
                # Log unexpected error but keep daemon alive
                log_line = f"{ts} ERROR={repr(e)}"

            if log_line is not None:
                logger.info(log_line)

            # Maintain ~1-second interval between iterations
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, interval_seconds - elapsed)
            time.sleep(sleep_time)

    finally:
        # 释放视频资源
        if video_cap is not None:
            video_cap.release()
            print("视频资源已释放")


if __name__ == "__main__":
    try:
        run_daemon()
    except KeyboardInterrupt:
        # 程序结束时导出最终CSV文件（task要求）
        print("\n正在导出最终CSV文件...")
        try:
            export_path = safe_statistics.export_final_csv()
            if export_path:
                print(f"✅ 最终CSV文件已导出至: {export_path}")

                # 显示统计摘要
                summary = safe_statistics.get_statistics_summary()
                print(f"📊 统计摘要:")
                print(f"   总波峰数: {summary.get('total_peaks', 0)}")
                print(f"   绿色波峰: {summary.get('green_peaks', 0)}")
                print(f"   红色波峰: {summary.get('red_peaks', 0)}")
                print(f"   会话时长: {summary.get('session_duration', 'N/A')}")
                print(f"   会话ID: {summary.get('session_id', 'N/A')}")
            else:
                print("ℹ️ 没有数据可导出")
        except Exception as e:
            print(f"❌ 导出CSV文件时发生错误: {e}")

        print("守护进程已停止")
    except Exception as e:
        print(f"❌ 守护进程运行时发生错误: {e}")
        # 即使出错也尝试导出数据
        try:
            export_path = safe_statistics.export_final_csv()
            if export_path:
                print(f"✅ 异常停止前数据已导出至: {export_path}")
        except Exception:
            pass
