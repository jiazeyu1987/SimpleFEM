from __future__ import annotations

import glob
import os
from typing import Any, Deque, Dict, List, Optional, Tuple


def save_frame_artifacts(
    *,
    frame_index: int,
    should_save: bool,
    roi1_should_save: bool,
    save_roi1: bool,
    save_roi2: bool,
    save_roi3: bool,
    save_wave: bool,
    save_roi1_wave: bool,
    roi1_enabled: bool,
    processing_mode: str,
    video_cap: Any,
    roi1_dir: str,
    roi2_dir: str,
    roi3_dir: str,
    wave_dir: str,
    wave1_dir: str,
    roi1_image: Any,
    roi2_image: Any,
    roi3_image: Any,
    roi2_region: Optional[Tuple[int, int, int, int]],
    gray_buffer: Any,
    roi3_gray_buffer: Any,
    roi3_80_160_buffer: Any,
    green_peaks: List[Tuple[int, int]],
    red_peaks: List[Tuple[int, int]],
    bg_count: int,
    bg_mean: float,
    adaptive_window_frames: int,
    adaptive_threshold_enabled: bool,
    threshold_protection_active: bool,
    threshold_used: float,
    roi1_curve: List[float],
    roi1_bg_count: int,
    roi1_bg_mean: float,
    roi1_threshold_protection_active: bool,
    roi1_threshold_used: float,
) -> None:
    """
    Save per-frame artifacts (ROI1/ROI2/ROI3 images + wave plots).

    This function is a pure relocation of the legacy block; behavior and
    error-handling must remain identical.
    """
    import cv2  # local import to keep module import light
    import matplotlib.pyplot as plt

    # Optionally save ROI1 image
    if should_save and save_roi1:
        roi1_path = os.path.join(roi1_dir, f"roi1_{frame_index:06d}.png")
        try:
            roi1_image.save(roi1_path)
            # 调试：每保存10张图像输出一次日志
            if frame_index % 10 == 1:
                print(f"[DEBUG] ROI1 saved: {roi1_path}")
        except Exception as e:
            # 调试：输出保存失败的错误信息
            print(f"[ERROR] Failed to save ROI1 {roi1_path}: {e}")
            # Ignore individual save errors to keep daemon running
            pass

    # Optionally save ROI2 image (align index with ROI1 saves)
    if should_save and save_roi2 and roi2_image is not None:
        # Calculate video time in seconds if in video mode
        video_time_str = ""
        if processing_mode == "video" and video_cap is not None:
            try:
                # Get current video position in milliseconds
                video_pos_msec = video_cap.get(cv2.CAP_PROP_POS_MSEC)
                video_seconds = video_pos_msec / 1000.0
                video_time_str = f"_{video_seconds:06.2f}s"
            except Exception:
                video_time_str = "_0000.00s"

        roi2_path = os.path.join(roi2_dir, f"roi2_{frame_index:06d}{video_time_str}.png")
        try:
            roi2_image.save(roi2_path)
        except Exception:
            pass

    # Save ROI3 image if enabled and available
    if should_save and save_roi3 and roi3_image is not None and roi3_dir:
        try:
            roi3_path = os.path.join(roi3_dir, f"roi3_{frame_index:06d}{video_time_str}.png")
            roi3_image.save(roi3_path)
        except Exception:
            pass

    # Save wave plot (curve before detection, but annotated with detection result)
    if should_save and save_wave and gray_buffer:
        try:
            wave_path = os.path.join(
                wave_dir,
                f"wave_{frame_index:06d}.png",
            )

            # Save wave plot (curve before detection, but annotated with detection result)
            curve = list(gray_buffer) if gray_buffer else []
            fig, ax = plt.subplots(figsize=(8, 3))
            x = list(range(len(curve)))
            ax.plot(x, curve, color="black", linewidth=1)

            # Add ROI3 purple curve if buffer has data
            if roi3_gray_buffer:
                x3 = list(range(len(roi3_gray_buffer)))
                ax.plot(x3, list(roi3_gray_buffer), color="purple", linewidth=1, label="ROI3")
                ax.legend()

            # Draw session-wide background mean (adaptive threshold baseline)
            if bg_count > 0:
                ax.axhline(
                    bg_mean,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    label="bg_mean",
                )
            else:
                # 调试：输出为什么没有黄线
                print(
                    f"[DEBUG] No bg_mean line: bg_count={bg_count}, buffer_len={len(gray_buffer)}, adaptive_frames={adaptive_window_frames}, adaptive_enabled={adaptive_threshold_enabled}"
                )
                print(f"[DEBUG] protection_active={threshold_protection_active}, bg_mean={bg_mean}")

            # Draw current threshold used for peak detection
            threshold_color = "red" if threshold_protection_active else "orange"
            threshold_style = "--" if threshold_protection_active else "-"
            ax.axhline(
                threshold_used,
                color=threshold_color,
                linestyle=threshold_style,
                linewidth=1.5,
                label=f"threshold ({threshold_used:.1f}{'[PROTECTED]' if threshold_protection_active else ''})",
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

            # Add ROI2 frame information if available
            if roi2_dir and os.path.exists(roi2_dir):
                # Look for ROI2 files to display frame information
                roi2_files = []
                buffer_start = max(0, frame_index - len(curve) + 1)
                buffer_end = frame_index

                # Search for ROI2 files with the new naming pattern (frame_xxxxxx_XXXX.XXs.png)
                roi2_pattern = os.path.join(roi2_dir, "roi2_*.png")
                all_roi2_files = glob.glob(roi2_pattern)

                for actual_frame_num in range(buffer_start, buffer_end + 1):
                    # Try to find file with new pattern first
                    found_file = None
                    for roi2_file in all_roi2_files:
                        basename = os.path.basename(roi2_file)
                        # Check if filename starts with the current frame number
                        if basename.startswith(f"roi2_{actual_frame_num:06d}_"):
                            found_file = roi2_file
                            break

                    # Fallback to old pattern if new pattern not found
                    if found_file is None:
                        old_path = os.path.join(roi2_dir, f"roi2_{actual_frame_num:06d}.png")
                        if os.path.exists(old_path):
                            found_file = old_path

                    if found_file:
                        # Extract frame number and time from filename
                        basename = os.path.basename(found_file)
                        try:
                            if "_" in basename:
                                parts = basename.replace("roi2_", "").replace(".png", "").split("_")
                                frame_num = int(parts[0])
                                if len(parts) > 1 and parts[1].endswith("s"):
                                    time_str = parts[1]
                                    roi2_files.append(f"{frame_num}({time_str})")
                                else:
                                    roi2_files.append(str(frame_num))
                            else:
                                frame_num = int(basename.replace("roi2_", "").replace(".png", ""))
                                roi2_files.append(str(frame_num))
                        except Exception:
                            roi2_files.append(str(actual_frame_num))

                        if len(roi2_files) >= 3:  # Limit to 3 examples
                            break

                if roi2_files:
                    sample_text = "ROI2: " + ", ".join(roi2_files)
                    ax.text(
                        0.02,
                        0.98,
                        sample_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    )

            ax.set_xlabel("Frame index in buffer")
            ax.set_ylabel("Gray value")
            ax.set_title("ROI2 gray waveform with peaks")
            ax.set_ylim(50, 150)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(wave_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            # Ignore individual plotting/saving errors
            pass

    # ROI1 waveform visualization (if enabled)
    roi1_green_peaks: List[Tuple[int, int]] = []
    roi1_red_peaks: List[Tuple[int, int]] = []
    # Note: ROI1 peak detection will be implemented in a future phase
    # For now, we just visualize the ROI1 gray values without peak detection

    # Save ROI1 wave plot
    if roi1_should_save and save_roi1_wave and roi1_enabled and roi1_curve:
        try:
            roi1_wave_path = os.path.join(
                wave1_dir,
                f"roi1_wave_{frame_index:06d}.png",
            )

            # Create ROI1 waveform plot
            fig, ax = plt.subplots(figsize=(8, 3))
            x = list(range(len(roi1_curve)))
            ax.plot(x, roi1_curve, color="darkblue", linewidth=1, label="ROI1")

            # Draw ROI1 background mean
            if roi1_bg_count > 0:
                ax.axhline(
                    roi1_bg_mean,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    label="bg_mean",
                )

            # Draw ROI1 threshold
            roi1_threshold_color = "red" if roi1_threshold_protection_active else "orange"
            roi1_threshold_style = "--" if roi1_threshold_protection_active else "-"
            ax.axhline(
                roi1_threshold_used,
                color=roi1_threshold_color,
                linestyle=roi1_threshold_style,
                linewidth=1.5,
                label=f"threshold ({roi1_threshold_used:.1f}{'[PROTECTED]' if roi1_threshold_protection_active else ''})",
            )

            # Add ROI3 (80-160) percentage red curve if buffer has data
            if roi3_80_160_buffer:
                x3_80_160 = list(range(len(roi3_80_160_buffer)))
                ax.plot(
                    x3_80_160,
                    list(roi3_80_160_buffer),
                    color="red",
                    linewidth=1,
                    label="ROI3(80-160)%",
                )

            # Highlight ROI1 peaks regions (placeholder for future peak detection)
            for start, end in roi1_green_peaks:
                s = max(0, start - 1)
                e = min(len(roi1_curve) - 1, end + 1)
                xs = list(range(s, e + 1))
                ys = roi1_curve[s : e + 1]
                ax.plot(xs, ys, color="green", linewidth=2)

            for start, end in roi1_red_peaks:
                s = max(0, start - 1)
                e = min(len(roi1_curve) - 1, end + 1)
                xs = list(range(s, e + 1))
                ys = roi1_curve[s : e + 1]
                ax.plot(xs, ys, color="red", linewidth=2)

            # Set plot title and labels
            ax.set_title(f"ROI1 Waveform - Frame {frame_index} (len={len(roi1_curve)})")
            ax.set_xlabel("Frame Index (relative)")
            ax.set_ylabel("Gray Value (0-255)")
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(roi1_wave_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            # Ignore ROI1 plotting/saving errors
            pass

