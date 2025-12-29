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
import platform
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageGrab
import cv2
import matplotlib.pyplot as plt
import glob


def _json_default(obj: Any) -> Any:
    """json.dumps fallback for numpy / datetime / other non-serializable values."""
    try:
        import numpy as _np  # local import to avoid hard dependency in helper

        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, datetime):
        return obj.isoformat()

    return str(obj)


class RoiAnalysisCache:
    """
    Write a lightweight per-frame cache to `export/` for later analysis.

    Format: JSONL (one JSON object per line), with `type` in {"meta","frame","session_end"}.
    """

    def __init__(self, export_dir: str, enabled: bool = True, flush_every: int = 50) -> None:
        self.export_dir = export_dir
        self.enabled = bool(enabled)
        self.flush_every = max(1, int(flush_every))
        self._fh: Optional[Any] = None
        self._path: Optional[str] = None
        self._run_id = uuid.uuid4().hex[:12]
        self._write_count = 0
        os.makedirs(self.export_dir, exist_ok=True)

    @property
    def path(self) -> Optional[str]:
        return self._path

    def start_session(
        self,
        session_id: str,
        *,
        processing_mode: str,
        video_path: Optional[str],
        config: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        self.close(reason="switch_session")

        safe_session = str(session_id or "unknown")
        filename = f"roi_analysis_cache_{safe_session}_{self._run_id}.jsonl"
        self._path = os.path.join(self.export_dir, filename)
        self._fh = open(self._path, "a", encoding="utf-8", newline="\n")
        self._write_count = 0

        meta: Dict[str, Any] = {
            "type": "meta",
            "cache_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": safe_session,
            "processing_mode": processing_mode,
            "video_path": video_path,
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "config": config,
        }
        if extra_meta:
            meta["extra"] = extra_meta

        self._write_line(meta)
        try:
            self._fh.flush()
        except Exception:
            pass

    def record_frame(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        payload = dict(payload)
        payload.setdefault("type", "frame")
        self._write_line(payload)

    def close(self, reason: str = "normal") -> None:
        if not self.enabled or self._fh is None:
            self._fh = None
            self._path = self._path  # keep last path for reference
            return
        try:
            self._write_line(
                {
                    "type": "session_end",
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "reason": reason,
                }
            )
            self._fh.flush()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None

    def _write_line(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        line = json.dumps(obj, ensure_ascii=False, default=_json_default)
        self._fh.write(line + "\n")
        self._write_count += 1
        if self._write_count % self.flush_every == 0:
            try:
                self._fh.flush()
            except Exception:
                pass



def manage_threshold_protection(
    current_gray: float,
    current_threshold: float,
    has_peaks: bool,
    frame_time: float,
    # State variables (passed by reference)
    protection_active: bool,
    protection_end_time: float,
    consecutive_below: int,
    last_waveform_time: float,
    # Configuration
    enabled: bool = True,
    recovery_delay_frames: int = 10,
    stability_frames: int = 5,
    waveform_trigger: bool = True,
    threshold_minimum: float = 80.0,
) -> Tuple[bool, float, int, int, float]:
    """
    管理阈值保护状态

    Args:
        current_gray: 当前灰度值
        current_threshold: 当前阈值
        has_peaks: 当前帧是否检测到波峰
        frame_time: 当前帧的时间戳
        protection_active: 保护状态是否激活
        protection_end_time: 保护结束时间
        consecutive_below: 连续低于阈值的帧数
        last_waveform_time: 上次波形时间
        enabled: 是否启用保护机制
        recovery_delay_frames: 恢复延迟帧数
        stability_frames: 稳定性帧数要求
        waveform_trigger: 是否启用波形触发
        threshold_minimum: 阈值下限，确保阈值不会低于此值

    Returns:
        Tuple[bool, float, int, int, float]:
            (should_protect, new_end_time, new_consecutive_below, frames_since_end, new_waveform_time)
    """
    current_time = frame_time
    frames_since_end = max(0, int((current_time - protection_end_time) / (1.0/10)))  # 假设10fps

    if not enabled:
        return False, protection_end_time, consecutive_below, frames_since_end, last_waveform_time

    # 检查是否需要触发保护
    should_protect = protection_active

    # 1. 波形触发：当前灰度超过阈值时立即触发保护
    if waveform_trigger and current_gray >= current_threshold:
        should_protect = True
        last_waveform_time = current_time
        if not protection_active:
            print(f"[阈值保护] 波形触发保护: 灰度={current_gray:.1f} >= 阈值={current_threshold:.1f}")

    # 2. 波峰结果触发：检测到波峰时激活保护
    elif has_peaks and not protection_active:
        should_protect = True
        last_waveform_time = current_time
        print(f"[阈值保护] 波峰触发保护: 检测到波峰")

    # 3. 检查是否可以解除保护
    if should_protect:
        # 计算应该的结束时间
        planned_end_time = last_waveform_time + (recovery_delay_frames * 0.1)  # 0.1秒每帧

        # 检查稳定性条件：连续多帧低于阈值
        if current_gray < current_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        # 智能退出：满足延迟时间和稳定性条件
        time_condition = current_time >= planned_end_time
        stability_condition = consecutive_below >= stability_frames

        if time_condition and stability_condition:
            should_protect = False
            consecutive_below = 0
            frames_since_end = 0
            print(f"[阈值保护] 解除保护: 满足时间延迟({recovery_delay_frames}帧)和稳定性({stability_frames}帧)条件")
        else:
            # 更新结束时间
            protection_end_time = planned_end_time

    return should_protect, protection_end_time, consecutive_below, frames_since_end, last_waveform_time


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


def reset_video_state_variables(
    gray_buffer=None,
    bg_count=None,
    bg_mean=None,
    last_intersection_roi=None,
    frames_since_protection_end=None,
    threshold_protection_active=None,
    protection_end_time=None,
    consecutive_below_threshold=None,
    last_waveform_time=None,
    frame_index=None,
    first_video_frame=None
) -> None:
    """
    重置视频处理相关的状态变量，防止多视频分析时数据污染

    这个函数返回重置后的值，调用者需要重新赋值
    """
    if gray_buffer is not None:
        gray_buffer.clear()

    # 返回重置后的值
    return (
        0,  # bg_count
        0.0,  # bg_mean
        None,  # last_intersection_roi
        0,  # frames_since_protection_end
        False,  # threshold_protection_active
        0.0,  # protection_end_time
        0,  # consecutive_below_threshold
        0.0,  # last_waveform_time
        0,  # frame_index
        True  # first_video_frame
    )


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
from green_detector import detect_green_intersection, IntersectionFilter  # type: ignore  # noqa: E402
from safe_peak_statistics import SafePeakStatistics  # type: ignore  # noqa: E402


class VideoStatisticsManager:
    """管理每视频的统计实例"""

    def __init__(self):
        self.current_statistics: Optional[SafePeakStatistics] = None
        self.all_statistics: List[SafePeakStatistics] = []
        self.is_batch_mode = False
        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialize_for_video(self, video_path: str, is_batch: bool = False):
        """为视频初始化新的统计实例"""
        # 关闭之前的统计
        if self.current_statistics:
            self.current_statistics.export_final_csv()
            self.all_statistics.append(self.current_statistics)

        # 创建新的统计实例
        self.is_batch_mode = is_batch
        video_name = os.path.basename(video_path) if video_path else None
        self.current_statistics = SafePeakStatistics(
            video_name=video_name,
            is_batch_mode=is_batch
        )

        return self.current_statistics

    def get_global_summary(self) -> Dict[str, Any]:
        """聚合所有视频的汇总信息"""
        if not self.all_statistics:
            return {
                'total_videos_processed': 0,
                'total_peaks': 0,
                'total_green_peaks': 0,
                'total_red_peaks': 0,
                'session_duration': '00:00:00',
                'videos_processed': []
            }

        total_peaks = sum(len(s.stats_data) for s in self.all_statistics)
        total_green = sum(len([p for p in s.stats_data if p['peak_type'] == 'green'])
                         for s in self.all_statistics)
        total_red = sum(len([p for p in s.stats_data if p['peak_type'] == 'red'])
                       for s in self.all_statistics)

        session_start_dt = datetime.strptime(self.session_start, "%Y%m%d_%H%M%S")
        session_duration = str(datetime.now() - session_start_dt).split('.')[0]

        return {
            'total_videos_processed': len(self.all_statistics),
            'total_peaks': total_peaks,
            'total_green_peaks': total_green,
            'total_red_peaks': total_red,
            'session_duration': session_duration,
            'videos_processed': [s.video_name for s in self.all_statistics]
        }


# 全局统计管理器实例
statistics_manager = VideoStatisticsManager()

# 为了向后兼容，保持原有的safe_statistics全局变量
safe_statistics = statistics_manager.current_statistics


def _sanitize_video_name(video_name: str) -> str:
    """清理视频名称用于文件夹创建"""
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', video_name)
    sanitized = sanitized.strip('._')[:50]
    return sanitized or f"video_{int(time.time())}"


def _create_video_folders(video_path: str, session_id: str, processing_mode: str, save_roi1: bool, save_roi2: bool, save_roi3: bool, save_wave: bool, save_roi1_wave: bool = False) -> str:
    """创建每视频的文件夹结构"""
    if processing_mode == "video":
        # 批量模式：使用视频名称
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sanitized_name = _sanitize_video_name(video_name)
        tmp_root = os.path.join(BASE_DIR, "tmp", sanitized_name)
    else:
        # 屏幕模式：使用基于会话的命名（原有行为）
        session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_root = os.path.join(BASE_DIR, "tmp", session_start)

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


def discover_video_files(video_path: str) -> List[str]:
    """发现文件夹中的所有视频文件"""
    if not os.path.exists(video_path):
        raise ValueError(f"Video directory does not exist: {video_path}")

    # 支持的视频文件扩展名
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']

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


def _get_video_fps(video_cap, default_fps: float = 30.0) -> float:
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


def hybrid_peak_detection(roi1_curve: List[float], roi2_curve: List[float],
                          config: Dict[str, Any],
                          processed_peaks: Dict[Tuple[int, int], str] = None,
                          peak_counter: int = 0) -> List[Dict[str, Any]]:
    """
    混合波峰检测：ROI1检测波峰区间，ROI2在相同区间内判定颜色

    Args:
        roi1_curve: ROI1灰度曲线数据
        roi2_curve: ROI2灰度曲线数据（与ROI1完全同步）
        config: 混合检测配置参数
        processed_peaks: 已处理的ROI1波峰字典 {(start, end): peak_id}
        peak_counter: ROI1波峰ID计数器

    Returns:
        hybrid_peaks: 混合检测结果列表（包含唯一ID）
    """
    from peak_detection import detect_peaks

    hybrid_peaks = []

    # 初始化ROI1波峰管理
    if processed_peaks is None:
        processed_peaks = {}

    require_intersection = bool(config.get("require_intersection", True))
    intersection_detected = bool(config.get("intersection_detected", True))
    if require_intersection and not intersection_detected:
        print("[混合检测] 当前帧未检测到绿线交点，跳过ROI1波峰检测（避免ROI2定位失效导致误报）")
        return []

    buffer_start_frame_index = int(config.get("buffer_start_frame_index", 1))

    # 1. 使用ROI1数据进行波峰检测（ROI1独立阈值）
    try:
        roi1_green_peaks_raw, roi1_red_peaks_raw = detect_peaks(
            roi1_curve,
            threshold=config['roi1_threshold'],
            marginFrames=config['margin_frames'],
            silenceFrames=config['silence_frames'],
            avgFrames=config['pre_post_avg_frames'],
            differenceThreshold=999.0,  # 设为很大的值，让ROI1只检测波峰，不做颜色分类
        )
    except Exception as e:
        print(f"[混合检测] ROI1波峰检测失败: {e}")
        return []

    # 合并ROI1检测到的所有波峰（不区分颜色）
    roi1_all_peaks = roi1_green_peaks_raw + roi1_red_peaks_raw

    # 过滤最小宽度的波峰
    min_width = config.get('min_peak_width', 5)
    max_width = config.get('max_peak_width', 100)
    new_peaks: List[Tuple[int, int, str, int]] = []
    duplicate_count = 0

    for peak_start, peak_end in roi1_all_peaks:
        peak_width = peak_end - peak_start + 1
        if min_width <= peak_width <= max_width:
            # Use absolute peak max position as a stable dedup key so the same
            # physical peak is not re-detected when the sliding buffer shifts.
            peak_slice = roi1_curve[peak_start : peak_end + 1]
            local_max_offset = 0
            if peak_slice:
                local_max_offset = max(range(len(peak_slice)), key=lambda i: peak_slice[i])
            abs_peak_max = buffer_start_frame_index + peak_start + local_max_offset
            peak_key = abs_peak_max

            # 检查是否已经处理过这个ROI1波峰
            if peak_key in processed_peaks:
                duplicate_count += 1
                print(f"[混合检测] ROI1波峰[{peak_start}-{peak_end}]已处理过(peak_max={abs_peak_max})，跳过")
                continue

            # 新的ROI1波峰，生成唯一ID
            peak_counter += 1
            peak_id = f"ROI1_MAX_{abs_peak_max:06d}"
            processed_peaks[peak_key] = peak_id

            new_peaks.append((peak_start, peak_end, peak_id, abs_peak_max))
            print(f"[混合检测] 新ROI1波峰[{peak_start}-{peak_end}] -> ID: {peak_id}")

    print(f"[混合检测] ROI1检测到{len(roi1_all_peaks)}个波峰，过滤后新增{len(new_peaks)}个，重复{duplicate_count}个")

    # 2. 对每个新的ROI1波峰，使用ROI2数据进行颜色判定
    for peak_start, peak_end, peak_id, abs_peak_max in new_peaks:
        peak_width = peak_end - peak_start + 1

        # 使用ROI2数据进行颜色判定
        color_result = determine_roi2_color_in_interval(
            peak_start, peak_end, roi2_curve, config
        )

        # 检查是否被frame_diff过滤掉
        if color_result.get("method") == "error_filtered":
            print(f"[混合检测] frame_diff异常被过滤，跳过波峰[{peak_start}-{peak_end}] (ID:{peak_id}): {color_result.get('error', '未知错误')}")
            continue

        if not bool(color_result.get("roi2_valid", True)) and bool(config.get("skip_when_roi2_invalid", True)):
            print(f"[混合检测] ROI2数据无效，跳过波峰[{peak_start}-{peak_end}] (ID:{peak_id})")
            continue

        # 创建混合检测结果（包含ROI1唯一ID）
        hybrid_peak = {
            'peak_interval': (peak_start, peak_end),
            'width': peak_width,
            'color': color_result['color'],
            'method': color_result['method'],
            'confidence': color_result['confidence'],
            'roi1_frame_diff': 0.0,  # ROI1不做前后差异计算
            'roi2_frame_diff': color_result['frame_difference'],
            'pre_avg': color_result.get('pre_avg', 0.0),
            'post_avg': color_result.get('post_avg', 0.0),
            'quality_score': color_result.get('quality_score', 0.0),
            # ROI1波峰唯一ID信息
            'roi1_peak_id': peak_id,
            'roi1_peak_key': abs_peak_max
        }

        hybrid_peaks.append(hybrid_peak)

        print(f"[混合检测] 波峰[{peak_start}-{peak_end}] {peak_width}帧: {color_result['color']}色 "
              f"(ID:{peak_id}, 方法:{color_result['method']}, 置信度:{color_result['confidence']:.2f})")

    # 返回结果和更新后的状态
    return hybrid_peaks


def determine_roi2_color_in_interval(peak_start: int, peak_end: int,
                                   roi2_curve: List[float],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    在ROI1检测的波峰区间内，使用ROI2数据进行颜色判定

    Args:
        peak_start: 波峰开始帧
        peak_end: 波峰结束帧
        roi2_curve: ROI2灰度曲线
        config: 配置参数

    Returns:
        颜色判定结果
    """
    pre_frames = config.get('roi2_pre_frames', 5)
    post_frames = config.get('roi2_post_frames', 10)
    color_threshold = config.get('roi2_color_threshold', 1.5)
    min_frames = config.get('minimum_roi2_frames', 15)
    min_variance = config.get('roi2_minimum_variance', 0.5)
    roi2_min_gray = float(config.get("roi2_min_gray", 5.0))
    roi2_max_gray = float(config.get("roi2_max_gray", 250.0))
    fallback_enabled = config.get('fallback_enabled', True)

    try:
        # 检查ROI2数据是否充足
        roi2_interval_length = len(roi2_curve)
        if roi2_interval_length < min_frames:
            if fallback_enabled:
                return {
                    'color': 'red',
                    'method': 'roi1_fallback',
                    'confidence': 0.0,
                    'frame_difference': 0.0,
                    'roi2_valid': False,
                    'error': f'ROI2数据不足({roi2_interval_length} < {min_frames})，回退到ROI1'
                }
            else:
                return {
                    'color': 'red',
                    'method': 'error',
                    'confidence': 0.0,
                    'frame_difference': 0.0,
                    'roi2_valid': False,
                    'error': f'ROI2数据不足且未启用回退'
                }

        # 计算ROI2在波峰区间前的平均值
        pre_start = max(0, peak_start - pre_frames)
        pre_values = roi2_curve[pre_start:peak_start]
        pre_avg = sum(pre_values) / len(pre_values) if pre_values else roi2_curve[peak_start] if peak_start < len(roi2_curve) else 0.0

        # 计算ROI2在波峰区间后的平均值
        post_end = min(len(roi2_curve), peak_end + post_frames + 1)
        post_values = roi2_curve[peak_end + 1:post_end]
        post_avg = sum(post_values) / len(post_values) if post_values else roi2_curve[peak_end] if peak_end < len(roi2_curve) else 0.0

        # 颜色判定：基于前后差异
        frame_difference = post_avg - pre_avg

        # Filter out error data: if |frame_diff| > 15, consider it as noise/signal error
        if abs(frame_difference) > 15.0:
            return {
                'color': 'red',  # 标记为红色但会被后续过滤
                'method': 'error_filtered',
                'confidence': 0.0,
                'frame_difference': frame_difference,
                'threshold': color_threshold,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'roi2_valid': False,
                'error': f'frame_difference异常(|{frame_difference:.1f}| > 15)，判定为错误数据',
            }

        color = "green" if frame_difference >= color_threshold else "red"

        # 计算置信度
        confidence = min(abs(frame_difference) / max(color_threshold, abs(frame_difference)), 1.0)

        # 计算数据质量评分
        quality_info = calculate_roi2_data_quality(peak_start, peak_end, roi2_curve)
        variance_val = float(quality_info.get("variance", 0.0))
        mean_val = float(quality_info.get("mean_val", 0.0))
        if variance_val < float(min_variance) or mean_val < roi2_min_gray or mean_val > roi2_max_gray:
            return {
                'color': 'red',
                'method': 'roi2_invalid',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'threshold': color_threshold,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'roi2_valid': False,
                'quality_score': quality_info.get('quality_score', 0.0),
                'variance': variance_val,
                'data_range': quality_info.get('data_range', 0.0),
                'error': f'ROI2无效: mean={mean_val:.2f} (min={roi2_min_gray:.2f}, max={roi2_max_gray:.2f}), variance={variance_val:.4f} (min={float(min_variance):.4f})'
            }

        return {
            'color': color,
            'method': 'roi2',
            'frame_difference': frame_difference,
            'threshold': color_threshold,
            'pre_avg': pre_avg,
            'post_avg': post_avg,
            'confidence': confidence,
            'roi2_valid': True,
            'quality_score': quality_info['quality_score'],
            'variance': quality_info.get('variance', 0.0),
            'data_range': quality_info.get('data_range', 0.0)
        }

    except Exception as e:
        if fallback_enabled:
            return {
                'color': 'red',
                'method': 'roi1_fallback',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'roi2_valid': False,
                'error': f'ROI2计算错误({str(e)})，回退到ROI1'
            }
        else:
            return {
                'color': 'red',
                'method': 'error',
                'confidence': 0.0,
                'frame_difference': 0.0,
                'roi2_valid': False,
                'error': f'ROI2计算错误且未启用回退: {str(e)}'
            }


def calculate_roi2_data_quality(peak_start: int, peak_end: int,
                               roi2_curve: List[float]) -> Dict[str, float]:
    """
    计算ROI2在波峰区间内的数据质量评分

    Args:
        peak_start: 波峰开始帧
        peak_end: 波峰结束帧
        roi2_curve: ROI2灰度曲线

    Returns:
        数据质量指标
    """
    try:
        # 提取ROI2在波峰区间内的数据
        if peak_start >= len(roi2_curve) or peak_end >= len(roi2_curve):
            return {'quality_score': 0.0, 'error': '波峰区间超出ROI2数据范围'}

        interval_values = roi2_curve[peak_start:peak_end + 1]

        if not interval_values:
            return {'quality_score': 0.0, 'error': '无ROI2数据'}

        # 计算基本统计指标
        import math
        mean_val = sum(interval_values) / len(interval_values)
        variance_val = sum((x - mean_val) ** 2 for x in interval_values) / len(interval_values)
        std_dev = math.sqrt(variance_val)

        # 计算数据范围
        data_max = max(interval_values)
        data_min = min(interval_values)
        data_range = data_max - data_min

        # 计算数据稳定性（标准差相对于数据范围的比率）
        stability_score = max(0, 1.0 - std_dev / max(10.0, data_range))

        # 计算数据一致性（避免过度波动）
        consistency = 1.0 - min(1.0, std_dev / mean_val) if mean_val > 0 else 0.0

        # 综合质量评分
        quality_score = (stability_score + consistency) / 2.0

        return {
            'quality_score': quality_score,
            'stability_score': stability_score,
            'consistency': consistency,
            'variance': variance_val,
            'std_dev': std_dev,
            'data_range': data_range,
            'frame_count': len(interval_values),
            'mean_val': mean_val
        }

    except Exception as e:
        return {'quality_score': 0.0, 'error': f'质量计算错误: {str(e)}'}


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


def cleanup_directories():
    """根据配置文件清理指定文件夹下的所有内容"""
    try:
        config = load_fem_config()
        cleanup_config = config.get("startup_cleanup", {})

        # 检查是否启用清理功能
        if not cleanup_config.get("enabled", True):
            print("启动时清理功能已禁用（配置文件中 startup_cleanup.enabled = false）")
            return

        # 获取要清理的目录列表
        directories_to_clean = cleanup_config.get("directories_to_clean", ["export", "tmp", "logs"])

        # 检查各个目录的清理开关
        cleanup_switches = {
            "export": cleanup_config.get("cleanup_export", True),
            "tmp": cleanup_config.get("cleanup_tmp", True),
            "logs": cleanup_config.get("cleanup_logs", True)
        }

        base_dir = _get_base_dir()
        cleaned_count = 0

        print("开始启动时清理...")

        for dir_name in directories_to_clean:
            # 检查该目录是否被标记为可清理
            if dir_name not in cleanup_switches or not cleanup_switches[dir_name]:
                print(f"跳过目录 {dir_name}（配置文件中已禁用）")
                continue

            dir_path = os.path.join(base_dir, dir_name)

            if os.path.exists(dir_path):
                try:
                    # 统计要删除的项目
                    items_to_delete = os.listdir(dir_path)
                    if not items_to_delete:
                        print(f"  目录 {dir_name} 为空，无需清理")
                        continue

                    print(f"清理文件夹: {dir_path}（包含 {len(items_to_delete)} 个项目）")

                    # 遍历文件夹并删除所有文件和子文件夹
                    deleted_files = 0
                    deleted_dirs = 0
                    for item_name in items_to_delete:
                        item_path = os.path.join(dir_path, item_name)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                                print(f"  删除文件: {item_name}")
                                deleted_files += 1
                            elif os.path.isdir(item_path):
                                import shutil
                                shutil.rmtree(item_path)
                                print(f"  删除文件夹: {item_name}")
                                deleted_dirs += 1
                        except Exception as item_error:
                            print(f"  删除失败 {item_name}: {item_error}")

                    print(f"  清理完成: {dir_name}（删除 {deleted_files} 个文件，{deleted_dirs} 个文件夹）")
                    cleaned_count += 1

                except Exception as e:
                    print(f"  清理文件夹失败: {e}")
            else:
                print(f"  文件夹不存在，跳过: {dir_name}")

        if cleaned_count == 0:
            print("没有需要清理的目录或所有目录都为空")
        else:
            print(f"清理完成：共清理了 {cleaned_count} 个目录")

    except Exception as e:
        print(f"读取清理配置时发生错误: {e}")
        # 如果配置读取失败，使用默认行为（不清理）
        print("由于配置读取失败，跳过启动时清理")


def run_daemon() -> None:
    """
    Main loop:
      - capture ROI1
      - detect/update line intersection
      - extract ROI2
      - update gray buffer and run peak detection
      - log results at configured frame_rate
    """
    print("SimpleFEM ROI Daemon 启动...")

    # 清理现有的数据文件夹
    cleanup_directories()

    config = load_fem_config()

    # Initialize anti-jitter filter if enabled
    anti_jitter_config = config.get("roi2_anti_jitter", {})
    intersection_filter = None
    if anti_jitter_config.get("enabled", False):
        # 参数验证和标准化
        try:
            algorithm = anti_jitter_config.get("algorithm", "ema")
            movement_threshold = float(anti_jitter_config.get("movement_threshold", 20.0))
            initialization_frames = int(anti_jitter_config.get("initialization_frames", 3))

            if algorithm == "threshold":
                # 阈值式防抖动
                from threshold_based_anti_jitter import ThresholdIntersectionFilter
                intersection_filter = ThresholdIntersectionFilter(movement_threshold, initialization_frames)
                print(f"ROI2阈值式防抖动已启用:")
                print(f"  - algorithm: threshold (阈值式)")
                print(f"  - movement_threshold: {movement_threshold}px (小于此值ROI2完全不动)")
                print(f"  - initialization_frames: {initialization_frames} (前N帧初始化稳定位置)")
                print(f"  - 策略: 小于{movement_threshold}px变化时ROI2完全静止，超过才更新")
            else:
                # EMA平滑式防抖动
                ema_config = anti_jitter_config.get("ema", {})
                alpha = float(ema_config.get("alpha", 0.25))
                stability_threshold = float(anti_jitter_config.get("stability_threshold", 8.0))

                # 参数范围验证
                if not (0.05 <= alpha <= 0.95):
                    print(f"Warning: alpha={alpha} 超出推荐范围[0.05, 0.95]，将自动调整")
                if movement_threshold < 1.0:
                    print(f"Warning: movement_threshold={movement_threshold} 过小，建议设置为1.0以上")
                if stability_threshold < 1.0:
                    print(f"Warning: stability_threshold={stability_threshold} 过小，建议设置为1.0以上")
                if not (stability_threshold < movement_threshold):
                    print(f"Warning: stability_threshold({stability_threshold}) 应该小于 movement_threshold({movement_threshold})")
                if initialization_frames < 1 or initialization_frames > 20:
                    print(f"Warning: initialization_frames={initialization_frames} 可能不合适，推荐范围[1, 20]")

                intersection_filter = IntersectionFilter(alpha, movement_threshold, initialization_frames, stability_threshold)
                print(f"ROI2平滑式防抖动已启用:")
                print(f"  - algorithm: ema (指数移动平均平滑)")
                print(f"  - alpha (平滑因子): {alpha} (值越小越平滑)")
                print(f"  - movement_threshold (运动阈值): {movement_threshold}px (大于此值直接通过)")
                print(f"  - stability_threshold (稳定阈值): {stability_threshold}px (小于此值强力平滑)")
                print(f"  - initialization_frames (初始化帧数): {initialization_frames}")

        except (ValueError, TypeError) as e:
            print(f"Error: 防抖动配置参数无效: {e}")
            print("使用默认参数启用EMA防抖动")
            intersection_filter = IntersectionFilter()  # 使用默认参数
    else:
        print("ROI2防抖动已禁用")

    # Optional: write a per-frame cache for later analysis / root-cause debugging
    analysis_cache_conf = config.get("analysis_cache", {})
    if not isinstance(analysis_cache_conf, dict):
        analysis_cache_conf = {}
    analysis_cache = RoiAnalysisCache(
        os.path.join(BASE_DIR, "export"),
        enabled=bool(analysis_cache_conf.get("enabled", True)),
        flush_every=int(analysis_cache_conf.get("flush_every", 50)),
    )

    # 检测处理模式
    processing_mode = config.get("processing_mode", "screen")
    video_cap = None
    video_files = []  # 存储要处理的视频文件列表
    current_video_index = 0  # 当前处理的视频索引

    # 为屏幕模式初始化统计实例
    if processing_mode == "screen":
        statistics_manager.initialize_for_video(None, is_batch=False)
        safe_statistics = statistics_manager.current_statistics

    if processing_mode == "video":
        video_config = config.get("video_processing", {})
        video_path = video_config.get("video_path", "")
        if not video_path:
            raise ValueError("Video mode enabled but no video_path specified in config")

        # 检查是单个文件还是文件夹
        if os.path.isfile(video_path):
            # 单个视频文件
            video_files = [video_path]
            print(f"视频模式: 单个视频文件 {video_path}")
        elif os.path.isdir(video_path):
            # 视频文件夹
            video_files = discover_video_files(video_path)
            print(f"视频模式: 文件夹 {video_path}")
            print(f"发现 {len(video_files)} 个视频文件:")
            for i, video_file in enumerate(video_files, 1):
                print(f"  {i}. {os.path.basename(video_file)}")
        else:
            raise ValueError(f"Video path does not exist: {video_path}")

        # 初始化第一个视频
        if video_files:
            # 为第一个视频初始化统计
            statistics_manager.initialize_for_video(video_files[0], is_batch=True)
            video_cap = initialize_video_capture(video_files[0])

    try:
        roi_default = config.get("roi_capture", {}).get("default_config", {})
        roi2_config = config.get("roi_capture", {}).get("roi2_config", {})
        extension_params = roi2_config.get("extension_params", {})

        # Load ROI3 configuration
        roi3_config = config.get("roi_capture", {}).get("roi3_config", {})
        roi3_extension_params = roi3_config.get("extension_params", {})

        data_processing = config.get("data_processing", {})
        save_roi1 = bool(data_processing.get("save_roi1", False))
        save_roi2 = bool(data_processing.get("save_roi2", False))
        save_roi3 = bool(data_processing.get("save_roi3", False))
        save_wave = bool(data_processing.get("save_wave", False))
        save_roi1_wave = bool(data_processing.get("save_roi1_wave", False))
        # only_delect == True: save ROI1/ROI2/wave only when peaks are detected
        only_delect = bool(data_processing.get("only_delect", False))

        peak_conf = config.get("peak_detection", {})
        threshold = float(peak_conf.get("threshold", 105.0))
        threshold_minimum = float(peak_conf.get("threshold_minimum", 80.0))
        margin_frames = int(peak_conf.get("margin_frames", 5))
        diff_threshold = float(peak_conf.get("difference_threshold", 0.5))
        # 新增：阈值前后"静默"帧数要求（升阈值前 X 帧和降阈值后 X 帧都不能超过阈值）
        silence_frames = int(peak_conf.get("silence_frames", 0))
        pre_post_avg_frames = int(peak_conf.get("pre_post_avg_frames", 5))
        # 自适应阈值参数
        adaptive_threshold_enabled = bool(peak_conf.get("adaptive_threshold_enabled", False))
        threshold_over_mean_ratio = float(peak_conf.get("threshold_over_mean_ratio", 0.15))
        adaptive_window_seconds = float(peak_conf.get("adaptive_window_seconds", 3.0))

        # 阈值保护参数
        protection_conf = peak_conf.get("threshold_protection", {})
        protection_enabled = bool(protection_conf.get("enabled", False))
        recovery_delay_seconds = float(protection_conf.get("recovery_delay_seconds", 1.0))
        stability_frames = int(protection_conf.get("stability_frames", 5))
        waveform_trigger_enabled = bool(protection_conf.get("waveform_trigger_enabled", True))

        # ROI3 override parameters
        roi3_override_conf = peak_conf.get("roi3_override", {})
        roi3_override_enabled = bool(roi3_override_conf.get("enabled", False))
        roi3_override_threshold = float(roi3_override_conf.get("threshold", 115.0))
        require_roi3_data = bool(roi3_override_conf.get("require_roi3_data", True))

        min_region_length = int(peak_conf.get("min_region_length", 1))

        # ROI1 configuration parameters (independent from ROI2)
        roi1_peak_conf = config.get("roi1_peak_detection", {})
        roi1_enabled = bool(roi1_peak_conf.get("enabled", False))
        roi1_threshold = float(roi1_peak_conf.get("threshold", 120.0))
        roi1_threshold_minimum = float(roi1_peak_conf.get("threshold_minimum", 110.0))
        roi1_margin_frames = int(roi1_peak_conf.get("margin_frames", 5))
        roi1_silence_frames = int(roi1_peak_conf.get("silence_frames", 5))
        roi1_pre_post_avg_frames = int(roi1_peak_conf.get("pre_post_avg_frames", 5))
        roi1_difference_threshold = float(roi1_peak_conf.get("difference_threshold", 2.0))
        roi1_min_region_length = int(roi1_peak_conf.get("min_region_length", 5))

        # ROI1 adaptive threshold parameters
        roi1_adaptive_threshold_enabled = bool(roi1_peak_conf.get("adaptive_threshold_enabled", True))
        roi1_threshold_over_mean_ratio = float(roi1_peak_conf.get("threshold_over_mean_ratio", 0.08))
        roi1_adaptive_window_seconds = float(roi1_peak_conf.get("adaptive_window_seconds", 3.0))

        # ROI1 threshold protection parameters
        roi1_protection_conf = roi1_peak_conf.get("threshold_protection", {})
        roi1_protection_enabled = bool(roi1_protection_conf.get("enabled", True))
        roi1_recovery_delay_seconds = float(roi1_protection_conf.get("recovery_delay_seconds", 1.0))
        roi1_stability_frames = int(roi1_protection_conf.get("stability_frames", 5))
        roi1_waveform_trigger_enabled = bool(roi1_protection_conf.get("waveform_trigger_enabled", True))

        # 混合检测配置参数读取
        hybrid_conf = config.get("hybrid_detection", {})
        hybrid_enabled = bool(hybrid_conf.get("enabled", False))
        detection_strategy = hybrid_conf.get("detection_strategy", "roi1_peaks_roi2_color")
        fusion_strategy = hybrid_conf.get("fusion_strategy", "roi2_priority")

        # ROI2颜色判定配置
        roi2_color_config = hybrid_conf.get("roi2_color_frames", {})
        roi2_pre_frames = int(roi2_color_config.get("pre_peak", 5))
        roi2_post_frames = int(roi2_color_config.get("post_peak", 10))

        # ROI1波峰宽度验证配置
        peak_width_config = hybrid_conf.get("roi1_peak_width_range", [30, 40])
        min_peak_width = int(peak_width_config[0])
        max_peak_width = int(peak_width_config[1])

        # 数据质量检查配置
        data_quality_conf = hybrid_conf.get("data_quality", {})
        min_roi2_frames = int(data_quality_conf.get("minimum_roi2_frames", 15))
        roi2_min_variance = float(data_quality_conf.get("roi2_minimum_variance", 0.5))
        fallback_enabled = bool(hybrid_conf.get("fallback_enabled", True))

        logger = setup_peak_logger()
        # Store only the latest 100 gray values for waveform / peak detection
        gray_buffer: Deque[float] = deque(maxlen=100)
        # Track a session-wide "background mean" using a gated incremental mean:
        # only update the mean when the current gray value is below the current
        # (mean-based) threshold, so peak frames do not contaminate the baseline.
        bg_count: int = 0
        bg_mean: float = 0.0
        last_intersection_roi: Optional[Tuple[int, int]] = None
        frames_since_protection_end: int = 0

        # Threshold protection state management
        threshold_protection_active: bool = False
        protection_end_time: float = 0.0
        consecutive_below_threshold: int = 0
        last_waveform_time: float = 0.0

        # ROI1 independent buffer and state (parallel to ROI2)
        roi1_gray_buffer: Deque[float] = deque(maxlen=100)
        roi1_bg_count: int = 0
        roi1_bg_mean: float = 0.0
        roi1_threshold_protection_active: bool = False
        roi1_protection_end_time: float = 0.0
        roi1_consecutive_below_threshold: int = 0
        roi1_last_waveform_time: float = 0.0

        # ROI3 independent buffer (same structure as ROI2)
        roi3_gray_buffer: Deque[float] = deque(maxlen=100)

        # Initialize ROI1 threshold used so hybrid detection can reference it
        # before the per-frame ROI1 adaptive-threshold block runs.
        roi1_threshold_used: float = max(roi1_threshold, roi1_threshold_minimum)

        # ROI1波峰唯一ID管理机制 - 防止重复记录
        processed_roi1_peaks: Dict[Tuple[int, int], str] = {}  # {(start, end): peak_id}
        roi1_peak_counter: int = 0  # 唯一ID计数器

        # Prepare per-video image save directories if enabled
        if processing_mode == "video" and video_files:
            # Video mode: Use first video for initial folder creation
            current_stats = statistics_manager.current_statistics
            if current_stats and current_stats.video_name:
                tmp_root = _create_video_folders(
                    video_files[0],
                    current_stats.session_id,
                    processing_mode,
                    save_roi1,
                    save_roi2,
                    save_roi3,
                    save_wave,
                    save_roi1_wave
                )
                # 关键修复：更新ROI保存路径变量
                roi1_dir = os.path.join(tmp_root, "roi1")
                roi2_dir = os.path.join(tmp_root, "roi2")
                roi3_dir = os.path.join(tmp_root, "roi3")
                wave_dir = os.path.join(tmp_root, "wave")
                wave1_dir = os.path.join(tmp_root, "wave1")
            else:
                # Fallback for screen mode or if video stats not initialized
                session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
                tmp_root = os.path.join(BASE_DIR, "tmp", session_start)
                if save_roi1 or save_roi2 or save_wave:
                    os.makedirs(tmp_root, exist_ok=True)
                if save_roi1:
                    os.makedirs(os.path.join(tmp_root, "roi1"), exist_ok=True)
                if save_roi2:
                    os.makedirs(os.path.join(tmp_root, "roi2"), exist_ok=True)
                if save_wave:
                    os.makedirs(os.path.join(tmp_root, "wave"), exist_ok=True)
        else:
            # Screen mode: Use original session-based naming
            session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp_root = os.path.join(BASE_DIR, "tmp", session_start)
            roi1_dir = os.path.join(tmp_root, "roi1")
            roi2_dir = os.path.join(tmp_root, "roi2")
            roi3_dir = os.path.join(tmp_root, "roi3")
            wave_dir = os.path.join(tmp_root, "wave")

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

        frame_index = 0

        # Use roi_capture.frame_rate as loop frequency
        roi_frame_rate = config.get("roi_capture", {}).get("frame_rate", 1)
        try:
            roi_frame_rate = float(roi_frame_rate)
        except Exception:
            roi_frame_rate = 1.0
        if roi_frame_rate <= 0:
            roi_frame_rate = 1.0
        # Video mode: make roi_capture.frame_rate control sampling on the video timeline
        # (skip frames based on source FPS), while keeping screen-capture mode unchanged.
        video_fps = 0.0
        video_frame_step = 1
        first_video_frame = True
        effective_frame_rate = roi_frame_rate
        if processing_mode == "video" and video_cap is not None:
            video_fps = _get_video_fps(video_cap)
            if video_fps > 0:
                effective_frame_rate = min(roi_frame_rate, video_fps)
                if effective_frame_rate > 0:
                    video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))

        interval_seconds = 1.0 / max(1e-6, effective_frame_rate)
        if processing_mode == "video" and video_fps > 0:
            print(
                f"[video] source_fps={video_fps:.2f} target_fps={effective_frame_rate:.2f} frame_step={video_frame_step}"
            )

        # 调试信息：打印帧率配置
        print(f"[帧率配置] 配置帧率: {roi_frame_rate} fps")
        print(f"[帧率配置] 计算间隔: {interval_seconds:.3f} 秒/帧")
        print(f"[帧率配置] 预期7秒视频处理: {7 * roi_frame_rate} 帧")

        # Calculate adaptive window frame count based on time window and frame rate
        adaptive_window_frames = int(adaptive_window_seconds * effective_frame_rate)
        # Ensure at least 1 frame and not exceed buffer size
        adaptive_window_frames = max(1, min(adaptive_window_frames, 100))

        # Calculate recovery delay in frames
        recovery_delay_frames = int(recovery_delay_seconds * effective_frame_rate)
        recovery_delay_frames = max(1, recovery_delay_frames)

        # Start cache session (one file per SafePeakStatistics session/video)
        try:
            current_stats = statistics_manager.current_statistics
            session_id = current_stats.session_id if current_stats else datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path_for_meta = None
            if processing_mode == "video" and video_files:
                if current_video_index < len(video_files):
                    video_path_for_meta = video_files[current_video_index]
                else:
                    video_path_for_meta = video_files[0]
            analysis_cache.start_session(
                session_id,
                processing_mode=processing_mode,
                video_path=video_path_for_meta,
                config=config,
                extra_meta={
                    "roi_frame_rate": roi_frame_rate,
                    "effective_frame_rate": effective_frame_rate,
                    "video_fps": video_fps,
                    "video_frame_step": video_frame_step,
                    "adaptive_window_frames": adaptive_window_frames,
                    "gray_buffer_maxlen": 100,
                },
            )
            if analysis_cache.path:
                print(f"[cache] analysis_cache={analysis_cache.path}")
        except Exception:
            pass

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
                    # First frame uses step=1 to avoid skipping the very beginning.
                    step = 1 if first_video_frame else video_frame_step
                    first_video_frame = False
                    screen = get_video_frame(video_cap, loop_enabled, frame_step=step)
                    if screen is None:
                        # 当前视频播放结束
                        current_video_index += 1

                        # 释放当前视频资源
                        video_cap.release()

                        if current_video_index < len(video_files):
                            # 切换到下一个视频
                            next_video_path = video_files[current_video_index]
                            try:
                                # 为此视频初始化新的统计
                                current_stats = statistics_manager.initialize_for_video(
                                    next_video_path,
                                    is_batch=True
                                )

                                # 重置防抖动滤波器状态（用于新视频）
                                if intersection_filter:
                                    # 保存当前调试信息
                                    old_debug_info = intersection_filter.get_debug_info()
                                    intersection_filter.reset()
                                    print(f"已重置ROI2防抖动滤波器，切换到新视频: {os.path.basename(next_video_path)}")

                                    # 根据滤波器类型显示不同的统计信息
                                    if 'update_count' in old_debug_info:
                                        # 阈值式滤波器
                                        print(f"上一个视频的阈值防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                                              f"更新{old_debug_info['update_count']}次, "
                                              f"稳定率{old_debug_info.get('stability_rate', 0):.1f}%")
                                    else:
                                        # EMA滤波器
                                        print(f"上一个视频的EMA防抖动统计: 处理{old_debug_info['frame_count']}帧, "
                                              f"大运动{old_debug_info['large_movement_count']}次, "
                                              f"边界限制{old_debug_info.get('boundary_clamp_count', 0)}次")

                                # 重置全局状态变量（防止数据污染）
                                gray_buffer.clear()
                                roi1_gray_buffer.clear()
                                reset_values = reset_video_state_variables(gray_buffer)
                                (bg_count, bg_mean, last_intersection_roi, frames_since_protection_end,
                                 threshold_protection_active, protection_end_time, consecutive_below_threshold,
                                 last_waveform_time, frame_index, first_video_frame) = reset_values

                                # 重置ROI1状态变量
                                roi1_bg_count = 0
                                roi1_bg_mean = 0.0
                                roi1_threshold_protection_active = False
                                roi1_protection_end_time = 0.0
                                roi1_consecutive_below_threshold = 0
                                roi1_last_waveform_time = 0.0
                                roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)

                                # 重置ROI1波峰ID管理机制
                                processed_roi1_peaks.clear()
                                roi1_peak_counter = 0

                                print(f"已重置全局和ROI1状态变量，确保数据隔离")

                                video_cap = initialize_video_capture(next_video_path)
                                print(f"\n" + "="*50)
                                print(f"开始处理下一个视频 ({current_video_index + 1}/{len(video_files)}):")
                                print(f"文件名: {os.path.basename(next_video_path)}")
                                print(f"统计会话: {current_stats.session_id}")

                                # 重新计算新视频的帧率参数
                                video_fps = _get_video_fps(video_cap)
                                if video_fps > 0:
                                    effective_frame_rate = min(roi_frame_rate, video_fps)
                                    if effective_frame_rate > 0:
                                        video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))

                                # 创建每视频文件夹结构
                                tmp_root = _create_video_folders(
                                    next_video_path,
                                    current_stats.session_id,
                                    processing_mode,
                                    save_roi1,
                                    save_roi2,
                                    save_roi3,
                                    save_wave,
                                    save_roi1_wave
                                )

                                # 关键修复：更新ROI保存路径变量
                                roi1_dir = os.path.join(tmp_root, "roi1")
                                roi2_dir = os.path.join(tmp_root, "roi2")
                                roi3_dir = os.path.join(tmp_root, "roi3")
                                wave_dir = os.path.join(tmp_root, "wave")
                                wave1_dir = os.path.join(tmp_root, "wave1")

                                print(f"[video] source_fps={video_fps:.2f} target_fps={effective_frame_rate:.2f} frame_step={video_frame_step}")
                                print(f"[folders] tmp_root={tmp_root}")
                                print(f"[folders] roi1_dir={roi1_dir}")
                                print(f"[folders] roi2_dir={roi2_dir}")
                                print(f"[folders] wave_dir={wave_dir}")
                                print("="*50)

                                # 重置帧索引和首帧标志
                                frame_index = 0
                                first_video_frame = True

                                # Start a new cache session for the new video/statistics session
                                try:
                                    analysis_cache.start_session(
                                        current_stats.session_id,
                                        processing_mode=processing_mode,
                                        video_path=next_video_path,
                                        config=config,
                                        extra_meta={
                                            "roi_frame_rate": roi_frame_rate,
                                            "effective_frame_rate": effective_frame_rate,
                                            "video_fps": video_fps,
                                            "video_frame_step": video_frame_step,
                                            "adaptive_window_frames": adaptive_window_frames,
                                            "gray_buffer_maxlen": 100,
                                        },
                                    )
                                    if analysis_cache.path:
                                        print(f"[cache] analysis_cache={analysis_cache.path}")
                                except Exception:
                                    pass

                                # 继续处理下一个视频，不break
                                continue
                            except Exception as e:
                                print(f"无法打开下一个视频 {next_video_path}: {e}")
                                print("继续处理下一个视频...")
                                continue
                        else:
                            # 所有视频都处理完毕
                            total_time = time.time() - (loop_start - (frame_index * interval_seconds))
                            actual_fps = frame_index / total_time if total_time > 0 else 0
                            print(f"\n" + "="*50)
                            print(f"所有视频处理完成！")
                            print(f"[统计] 总处理时间: {total_time:.2f} 秒")
                            print(f"[统计] 总处理视频数: {len(video_files)}")
                            print(f"[统计] 总处理帧数: {frame_index}")
                            print(f"[统计] 平均帧率: {actual_fps:.2f} fps")
                            print(f"[统计] 配置帧率: {roi_frame_rate:.2f} fps")
                            print("="*50)
                            break
                    screen_width, screen_height = screen.size
                else:
                    screen = ImageGrab.grab()
                    screen_width, screen_height = screen.size

                video_seconds: Optional[float] = None
                if processing_mode == "video" and video_cap is not None:
                    try:
                        video_pos_msec = float(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                        if video_pos_msec >= 0:
                            video_seconds = video_pos_msec / 1000.0
                    except Exception:
                        video_seconds = None

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
                    intersection = detect_green_intersection(
                        roi_cv_image,
                        anti_jitter_config,
                        intersection_filter
                    )
                except Exception as e:
                    # Keep daemon running even if detection fails on this frame
                    print(f"Warning: Green intersection detection failed: {e}")
                    intersection = None
                    # 如果检测失败，尝试重置防抖动滤波器
                    if intersection_filter:
                        try:
                            intersection_filter.reset()
                            intersection_filter.set_image_bounds(roi1_width, roi1_height)
                        except:
                            pass

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

                    # ROI3 extraction (independent from ROI2)
                    roi3_gray: Optional[float] = None
                    roi3_image: Optional[Image.Image] = None
                    if roi3_extension_params:
                        roi3_region = compute_roi2_region(
                            (roi1_width, roi1_height),
                            (center_x, center_y),
                            roi3_extension_params,
                        )
                        if roi3_region is not None:
                            r3x1, r3y1, r3x2, r3y2 = roi3_region
                            roi3_image = roi1_image.crop((r3x1, r3y1, r3x2, r3y2))
                            roi3_gray = compute_average_gray(roi3_image)
                            roi3_gray_buffer.append(roi3_gray)
                            print(f"[DEBUG] ROI3 captured: frame={frame_index}, gray={roi3_gray:.2f}, buffer_len={len(roi3_gray_buffer)}")
                        else:
                            print(f"[DEBUG] ROI3 extraction failed: frame={frame_index}, intersection={intersection}, roi3_extension_params={roi3_extension_params}")
                    else:
                        print(f"[DEBUG] ROI3 extension params not available")

                    # ROI1 gray value calculation (independent from ROI2)
                    roi1_gray: Optional[float] = None
                    if roi1_enabled:
                        roi1_gray = compute_average_gray(roi1_image)
                        roi1_gray_buffer.append(roi1_gray)

                # 5. Run peak detection on current gray buffer
                green_peaks: List[Tuple[int, int]] = []
                red_peaks: List[Tuple[int, int]] = []
                green_peaks_raw: List[Tuple[int, int]] = []
                red_peaks_raw: List[Tuple[int, int]] = []
                threshold_used = max(threshold, threshold_minimum)
                recent_frames_count: Optional[int] = None
                calculated_bg_mean: Optional[float] = None

                if gray_buffer:
                    curve = list(gray_buffer)
                    # 调试：输出缓冲区状态
                    print(f"[DEBUG] Buffer status: len={len(gray_buffer)}, adaptive_frames={adaptive_window_frames}, enabled={adaptive_threshold_enabled}")

                    # Calculate adaptive threshold if enabled and enough history is available
                    if (
                        adaptive_threshold_enabled
                        and len(gray_buffer) >= adaptive_window_frames
                    ):
                        # Calculate recent mean (last adaptive_window_frames from gray_buffer)
                        recent_frames_count = min(len(gray_buffer), adaptive_window_frames)
                        recent_frames = list(gray_buffer)[-recent_frames_count:]
                        calculated_bg_mean = sum(recent_frames) / len(recent_frames)

                        # First, check if we're already in protection mode and need to extend it
                        current_time = time.time()
                        if threshold_protection_active:
                            # Check protection status with current gray value
                            (threshold_protection_active, protection_end_time,
                             consecutive_below_threshold, frames_since_protection_end,
                             last_waveform_time) = manage_threshold_protection(
                                current_gray=roi2_gray if roi2_gray is not None else 0,
                                current_threshold=threshold_used,
                                has_peaks=False,  # Will check again after detection
                                frame_time=current_time,
                                protection_active=threshold_protection_active,
                                protection_end_time=protection_end_time,
                                consecutive_below=consecutive_below_threshold,
                                last_waveform_time=last_waveform_time,
                                enabled=protection_enabled,
                                recovery_delay_frames=recovery_delay_frames,
                                stability_frames=stability_frames,
                                waveform_trigger=waveform_trigger_enabled,
                                threshold_minimum=threshold_minimum,
                            )

                        # Only update background mean when protection is not active
                        if not threshold_protection_active:
                            bg_mean = calculated_bg_mean
                            bg_count = recent_frames_count
                            # 调试：输出背景均值更新信息
                            print(f"[DEBUG] bg_mean updated: {bg_mean:.2f}, bg_count={bg_count}, buffer_len={len(gray_buffer)}")
                            if adaptive_threshold_enabled and bg_mean > 0:
                                threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                                # Apply minimum threshold constraint
                                threshold_used = max(threshold_used, threshold_minimum)
                        else:
                            # Use last known background mean during protection
                            if bg_mean > 0:
                                threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                                # Apply minimum threshold constraint even during protection
                                threshold_used = max(threshold_used, threshold_minimum)
                            #print(f"[阈值保护] 保护期间使用冻结阈值: {threshold_used:.1f} (下限: {threshold_minimum:.1f})")

                    # 混合检测模式集成
                    detection_mode = "roi2_legacy"
                    hybrid_peaks: List[Dict[str, Any]] = []
                    if hybrid_enabled and roi1_enabled and len(roi1_gray_buffer) > 0:
                        # 混合检测模式：ROI1检测波峰，ROI2判定颜色
                        roi1_curve = list(roi1_gray_buffer)
                        roi2_curve = list(gray_buffer) if gray_buffer else []

                        hybrid_config = {
                            'roi1_threshold': roi1_threshold_used,
                            'margin_frames': roi1_margin_frames,
                            'silence_frames': roi1_silence_frames,
                            'pre_post_avg_frames': roi1_pre_post_avg_frames,
                            'min_peak_width': roi1_min_region_length,
                            'max_peak_width': max_peak_width,
                            'roi2_pre_frames': roi2_pre_frames,
                            'roi2_post_frames': roi2_post_frames,
                            'minimum_roi2_frames': min_roi2_frames,
                            'roi2_minimum_variance': roi2_min_variance,
                            'roi2_color_threshold': diff_threshold,
                            'fallback_enabled': fallback_enabled,
                            'require_intersection': bool(hybrid_conf.get("require_intersection", True)),
                            'intersection_detected': bool(intersection is not None),
                            'skip_when_roi2_invalid': bool(data_quality_conf.get("skip_peaks_when_roi2_invalid", True)),
                            'roi2_min_gray': float(data_quality_conf.get("roi2_min_gray", 5.0)),
                            'roi2_max_gray': float(data_quality_conf.get("roi2_max_gray", 250.0)),
                        }

                        print(f"[混合检测] 开始分析 - ROI1曲线长度:{len(roi1_curve)}, ROI2曲线长度:{len(roi2_curve)}")

                        # 执行混合检测（传递ROI1波峰管理参数）
                        try:
                            hybrid_config_with_frame = {**hybrid_config, 'frame_index': frame_index}
                            hybrid_config_with_frame["buffer_start_frame_index"] = frame_index - len(roi1_curve) + 1
                            hybrid_peaks = hybrid_peak_detection(
                                roi1_curve, roi2_curve, hybrid_config_with_frame,
                                processed_roi1_peaks, roi1_peak_counter
                            )
                            detection_mode = "hybrid_roi1_peaks_roi2_color"

                            # 转换为传统格式以保持兼容性
                            green_peaks = []
                            red_peaks = []
                            for peak in hybrid_peaks:
                                if peak['color'] == 'green':
                                    green_peaks.append(peak['peak_interval'])
                                else:
                                    red_peaks.append(peak['peak_interval'])

                            # 统计颜色数量和质量
                            green_count = len(green_peaks)
                            red_count = len(red_peaks)
                            avg_quality = sum(peak.get('quality_score', 0) for peak in hybrid_peaks) / len(hybrid_peaks) if hybrid_peaks else 0

                            print(f"[混合检测] 结果统计: 绿色{green_count}个, 红色{red_count}个, 平均质量{avg_quality:.2f}")

                            # 详细日志输出
                            for i, peak in enumerate(hybrid_peaks[:5]):  # 只显示前5个
                                start, end = peak['peak_interval']
                                width = end - start + 1
                                method = peak['method']
                                color = peak['color']
                                confidence = peak.get('confidence', 0)
                                print(f"  波峰{i+1}: [{start}-{end}] {width}帧, {color}色, 方法:{method}, 置信度:{confidence:.2f}")

                        except Exception as e:
                            print(f"[混合检测] 执行失败: {e}")
                            # 回退到传统ROI2检测
                            hybrid_peaks = []
                            green_peaks, red_peaks = [], []
                            detection_mode = "hybrid_failed"

                    else:
                        # 保持原有的ROI2独立检测逻辑作为后备
                        if hybrid_enabled:
                            print(f"[传统检测] 混合检测未启用或数据不足，使用ROI2独立检测模式")

                        if hybrid_enabled and roi1_enabled:
                            if frame_index % 50 == 0:
                                print("[æ··åˆæ£€æµ‹] ROI1æ•°æ®ä¸è¶³ï¼Œè·³è¿‡æ³¢å³°æ£€æµ‹ï¼ˆä¸å›žé€€åˆ°ROI2æ³¢å³°æ£€æµ‹ï¼‰")
                            green_peaks_raw, red_peaks_raw = [], []
                            green_peaks, red_peaks = [], []
                            detection_mode = "hybrid_roi1_insufficient"
                        else:
                            # Now run actual ROI2 peak detection with the determined threshold
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
                                for start, end in green_peaks_raw
                                if (end - start + 1) >= min_region_length
                            ]
                            red_peaks = [
                                (start, end)
                                for start, end in red_peaks_raw
                                if (end - start + 1) >= min_region_length
                            ]
                            detection_mode = "roi2_legacy"

                    # Re-check threshold protection with actual peak detection results
                    if protection_enabled and roi2_gray is not None:
                        has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
                        current_time = time.time()

                        (threshold_protection_active, protection_end_time,
                         consecutive_below_threshold, frames_since_protection_end,
                         last_waveform_time) = manage_threshold_protection(
                            current_gray=roi2_gray,
                            current_threshold=threshold_used,
                            has_peaks=has_peaks,
                            frame_time=current_time,
                            protection_active=threshold_protection_active,
                            protection_end_time=protection_end_time,
                            consecutive_below=consecutive_below_threshold,
                            last_waveform_time=last_waveform_time,
                            enabled=protection_enabled,
                            recovery_delay_frames=recovery_delay_frames,
                            stability_frames=stability_frames,
                            waveform_trigger=waveform_trigger_enabled,
                            threshold_minimum=threshold_minimum,
                        )

  
                # ROI1 adaptive threshold calculation (independent from ROI2)
                roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)
                roi1_curve = list(roi1_gray_buffer) if roi1_gray_buffer else []
                if roi1_enabled and roi1_gray_buffer:
                    # 每50帧打印一次ROI1阈值使用情况
                    if frame_index % 50 == 0:
                        print(f"[ROI1阈值] 配置值={roi1_threshold:.1f}, 下限={roi1_threshold_minimum:.1f}, 使用={roi1_threshold_used:.1f}")
                        if roi1_adaptive_threshold_enabled and roi1_bg_count > 0:
                            print(f"[ROI1阈值] 自适应背景均值={roi1_bg_mean:.1f}, 比例={roi1_threshold_over_mean_ratio:.2f}")
                        else:
                            print(f"[ROI1阈值] 使用固定阈值")
                    roi1_adaptive_window_frames = int(roi1_adaptive_window_seconds * effective_frame_rate)
                    roi1_adaptive_window_frames = max(1, min(roi1_adaptive_window_frames, 100))

                    if (roi1_adaptive_threshold_enabled and
                        len(roi1_gray_buffer) >= roi1_adaptive_window_frames):

                        # Calculate ROI1 recent mean
                        roi1_recent_frames_count = min(len(roi1_gray_buffer), roi1_adaptive_window_frames)
                        roi1_recent_frames = list(roi1_gray_buffer)[-roi1_recent_frames_count:]
                        roi1_calculated_bg_mean = sum(roi1_recent_frames) / len(roi1_recent_frames)

                        # Check ROI1 threshold protection status
                        current_time = time.time()
                        if roi1_threshold_protection_active:
                            # Manage ROI1 threshold protection with current gray value
                            # For now, use last known background mean during protection
                            if roi1_bg_mean > 0:
                                roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
                                roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)
                        else:
                            # Update ROI1 background mean if current value is below threshold
                            if roi1_gray < roi1_threshold_used:
                                roi1_bg_count += 1
                                # Incremental mean update: new_mean = old_mean + (new_value - old_mean) / count
                                roi1_bg_mean = roi1_bg_mean + (roi1_gray - roi1_bg_mean) / roi1_bg_count

                            # Calculate ROI1 adaptive threshold if we have enough background samples
                            if roi1_adaptive_threshold_enabled and roi1_bg_mean > 0:
                                roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
                                roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)

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
                stats_write_results: List[Dict[str, Any]] = []
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
                    current_stats = statistics_manager.current_statistics
                    if current_stats:
                        # 准备ROI1曲线数据用于混合检测统计
                        roi1_curve_for_stats = list(roi1_gray_buffer) if roi1_gray_buffer else []

                        # 调用扩展的add_peaks_from_daemon方法，支持混合检测
                        stats_write_results = current_stats.add_peaks_from_daemon(
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
                            # 混合检测参数
                            hybrid_enabled=hybrid_enabled,
                            hybrid_peaks=hybrid_peaks,
                            roi1_curve=roi1_curve_for_stats,
                            roi1_threshold_used=roi1_threshold_used,
                            # ROI3 override参数
                            roi3_curve=list(roi3_gray_buffer) if roi3_gray_buffer else [],
                            roi3_override_enabled=roi3_override_enabled,
                            roi3_override_threshold=roi3_override_threshold
                        )

                except Exception as e:
                    # Keep daemon running even if statistics collection fails
                    print(f"Statistics collection error: {e}")

                # Decide whether to save images/wave for this frame
                has_peak = (green_count > 0) or (red_count > 0)
                should_save = (not only_delect) or has_peak

                # For ROI1, save waveforms when data is available (independent of ROI2 peaks)
                roi1_should_save = (not only_delect) or (len(roi1_gray_buffer) > 0)

                # Write a per-frame cache record for later Q&A / root cause analysis
                try:
                    buffer_len = len(gray_buffer)
                    buffer_start_frame = max(0, frame_index - buffer_len + 1)

                    def _peaks_to_abs(peaks: List[Tuple[int, int]]) -> List[Dict[str, int]]:
                        out: List[Dict[str, int]] = []
                        for s, e in peaks:
                            out.append(
                                {
                                    "buffer_start": int(s),
                                    "buffer_end": int(e),
                                    "abs_start": int(buffer_start_frame + s),
                                    "abs_end": int(buffer_start_frame + e),
                                }
                            )
                        return out

                    analysis_cache.record_frame(
                        {
                            "ts_wall": loop_start,
                            "ts_local": ts,
                            "frame_index": int(frame_index),
                            "video_seconds": video_seconds,
                            "screen_size": [int(screen_width), int(screen_height)],
                            "roi1": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                            "intersection": {"current": intersection, "used": last_intersection_roi},
                            "roi2_region": roi2_region,
                            "roi2_gray": roi2_gray,
                            "buffer": {
                                "len": int(buffer_len),
                                "start_frame_index": int(buffer_start_frame),
                                "maxlen": 100,
                            },
                            "threshold": {
                                "fixed": float(threshold),
                                "minimum": float(threshold_minimum),
                                "used": float(threshold_used),
                                "adaptive_enabled": bool(adaptive_threshold_enabled),
                                "adaptive_window_frames": int(adaptive_window_frames),
                                "recent_frames_count": recent_frames_count,
                                "calculated_bg_mean": calculated_bg_mean,
                                "bg_mean": (float(bg_mean) if bg_count > 0 else None),
                                "bg_count": int(bg_count),
                                "protection_active": bool(threshold_protection_active),
                                "consecutive_below_threshold": int(consecutive_below_threshold),
                                "frames_since_protection_end": int(frames_since_protection_end),
                            },
                            "detect_params": {
                                "margin_frames": int(margin_frames),
                                "silence_frames": int(silence_frames),
                                "difference_threshold": float(diff_threshold),
                                "pre_post_avg_frames": int(pre_post_avg_frames),
                                "min_region_length": int(min_region_length),
                            },
                            "detection": {
                                "mode": str(detection_mode),
                                "hybrid_enabled": bool(hybrid_enabled),
                                "roi1_enabled": bool(roi1_enabled),
                            },
                            "peaks": {
                                "green_raw": _peaks_to_abs(green_peaks_raw),
                                "red_raw": _peaks_to_abs(red_peaks_raw),
                                "green": _peaks_to_abs(green_peaks),
                                "red": _peaks_to_abs(red_peaks),
                            },
                            "stats_write": stats_write_results,
                        }
                    )
                except Exception:
                    pass

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
                            print(f"[DEBUG] No bg_mean line: bg_count={bg_count}, buffer_len={len(gray_buffer)}, adaptive_frames={adaptive_window_frames}, adaptive_enabled={adaptive_threshold_enabled}")
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
                                ax.text(0.02, 0.98, sample_text, transform=ax.transAxes,
                                       fontsize=8, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

            # 调试信息：每10帧打印一次帧率控制信息
            if frame_index % 10 == 0:
                print(f"[帧率调试] 帧{frame_index}: 目标间隔={interval_seconds:.3f}s, 实际耗时={elapsed:.3f}s, 睡眠时间={sleep_time:.3f}s")

            time.sleep(sleep_time)

    finally:
        try:
            analysis_cache.close(reason="shutdown")
        except Exception:
            pass
        # 释放视频资源
        if video_cap is not None:
            video_cap.release()
            print("视频资源已释放")

        # 输出防抖动滤波器最终统计信息
        if intersection_filter:
            try:
                debug_info = intersection_filter.get_debug_info()
                print(f"\n防抖动滤波器最终统计:")
                print(f"  总处理帧数: {debug_info['frame_count']}")

                # 根据滤波器类型显示不同信息
                if 'update_count' in debug_info:
                    # 阈值式滤波器
                    print(f"  更新次数: {debug_info['update_count']}")
                    print(f"  忽略次数: {debug_info['ignore_count']}")
                    print(f"  稳定率: {debug_info.get('stability_rate', 0):.1f}%")
                    print(f"  大运动事件: {debug_info['large_movement_count']}次")
                    print(f"  阈值参数: threshold={debug_info['parameters']['movement_threshold']}px")
                else:
                    # EMA滤波器
                    print(f"  大运动事件: {debug_info['large_movement_count']}次")
                    print(f"  边界限制事件: {debug_info['boundary_clamp_count']}次")
                    print(f"  稳定事件: {debug_info['stability_count']}次")
                    print(f"  EMA参数: alpha={debug_info['parameters']['alpha']}, "
                          f"threshold={debug_info['parameters']['movement_threshold']}px")
            except Exception as e:
                print(f"获取防抖动统计信息失败: {e}")


if __name__ == "__main__":
    try:
        run_daemon()
    except KeyboardInterrupt:
        # 程序结束时导出最终CSV文件（task要求）
        print("\n数据处理完成，CSV文件已保存...")
        try:
            # 获取当前统计文件路径
            current_stats = statistics_manager.current_statistics
            if current_stats:
                export_path = current_stats.export_final_csv()
                if export_path:
                    print(f"✅ 当前视频CSV文件已保存至: {export_path}")

            # 显示所有视频的统计摘要
            global_summary = statistics_manager.get_global_summary()
            print(f"📊 批量处理统计摘要:")
            print(f"   总处理视频数: {global_summary.get('total_videos_processed', 0)}")
            print(f"   总波峰数: {global_summary.get('total_peaks', 0)}")
            print(f"   绿色波峰: {global_summary.get('total_green_peaks', 0)}")
            print(f"   红色波峰: {global_summary.get('total_red_peaks', 0)}")
            print(f"   会话时长: {global_summary.get('session_duration', 'N/A')}")

            # 显示每个视频的详细信息
            videos_processed = global_summary.get('videos_processed', [])
            if videos_processed:
                print(f"   处理的视频: {', '.join(videos_processed)}")

        except Exception as e:
            print(f"❌ 处理CSV文件时发生错误: {e}")

        print("守护进程已停止")
    except Exception as e:
        print(f"❌ 守护进程运行时发生错误: {e}")
        # 即使出错也尝试保存数据
        try:
            current_stats = statistics_manager.current_statistics
            if current_stats:
                export_path = current_stats.export_final_csv()
                if export_path:
                    print(f"✅ 异常停止前数据已保存至: {export_path}")
        except Exception:
            pass
