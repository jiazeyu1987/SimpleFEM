# SimpleFEM Simple ROI Daemon - 重写技术文档（修复版 v2.0）

本文档提供完整的技术规范，可根据此文档重写 `simple_roi_daemon.py` 而无需查看原始源代码。

**版本历史：**
- v1.0 (2025-12-25): 初始版本
- v2.0 (2025-12-25): 修复了 23 个冲突和问题

---

## 1. 系统概述

### 1.1 核心功能
SimpleFEM ROI Daemon 是一个医学信号处理守护进程，用于检测 HEM（高回声事件）波峰。

**主要工作流程：**
1. 每秒捕获 ROI1 区域（屏幕或视频帧）
2. 在 ROI1 中检测绿色线交点
3. 在交点周围提取 ROI2 和 ROI3 区域
4. 计算灰度值并维护100帧循环缓冲区
5. 执行波峰检测和颜色分类（绿色/红色）
6. 导出 CSV 统计数据和波形图

### 1.2 三种处理模式
- **screen**: 实时屏幕捕获（PIL.ImageGrab）
- **video**: 处理本地视频文件（支持批量多视频）
- **vein_following**: 静脉跟踪模式（自动跟踪 ROI）

### 1.3 ROI 定义
- **ROI1**: 大捕获区域（默认 1280x80 到 1920x980）
- **ROI2**: ROI1 内的小区域（约 80x120 像素），围绕绿线交点
- **ROI3**: ROI1 内的扩展垂直区域，用于额外验证

---

## 2. 核心类和数据结构

### 2.1 RoiAnalysisCache 类

**功能：** 将每帧分析数据写入 JSONL 文件供后续分析

**数据格式：**
```python
{
    "type": "meta" | "frame" | "session_end",
    # meta 字段:
    "cache_version": 1,
    "created_at": "ISO timestamp",
    "session_id": "unique_session_id",
    "processing_mode": "screen|video",
    "video_path": "/path/to/video.mp4",
    "config": {...},  # 完整配置对象
    # frame 字段:
    "ts_wall": 1234567890.123,  # Unix 时间戳
    "frame_index": 123,
    "roi2_gray": 95.5,
    "buffer": {...},
    "threshold": {...},
    "peaks": {...}
}
```

**关键方法：**
- `start_session(session_id, processing_mode, video_path, config)` - 开始新会话
- `record_frame(payload)` - 记录帧数据
- `close(reason)` - 关闭当前会话

### 2.2 VideoStatisticsManager 类

**功能：** 管理多视频批处理的统计信息

**关键方法：**
- `initialize_for_video(video_path, is_batch)` - 为新视频初始化统计
- `get_global_summary()` - 获取所有视频的汇总统计
- `current_statistics` - 属性，返回当前 SafePeakStatistics 实例

### 2.3 循环缓冲区

**三个独立的100帧循环缓冲区：**
```python
gray_buffer: Deque[float] = deque(maxlen=100)  # ROI2 灰度值
roi1_gray_buffer: Deque[float] = deque(maxlen=100)  # ROI1 灰度值
roi3_gray_buffer: Deque[float] = deque(maxlen=100)  # ROI3 灰度值
```

---

## 3. 配置系统

### 3.1 配置文件结构 (simple_fem_config.json)

```json
{
  "processing_mode": "video",
  "data_processing": {
    "save_roi1": true,
    "save_roi2": true,
    "save_roi3": true,
    "save_wave": true,
    "save_roi1_wave": false,
    "only_delect": true
  },
  "video_processing": {
    "video_path": "video.mp4",
    "loop_enabled": false,
    "processing_frame_rate": 10.0
  },
  "roi_capture": {
    "frame_rate": 10,
    "default_config": {
      "x1": 1280, "y1": 80,
      "x2": 1920, "y2": 980
    },
    "roi2_config": {
      "extension_params": {
        "left": 20, "right": 30,
        "top": 60, "bottom": 20
      }
    },
    "roi3_config": {
      "extension_params": {
        "left": 30, "right": 40,
        "top": 80, "bottom": 30
      }
    }
  },
  "peak_detection": {
    "threshold": 95.0,
    "threshold_minimum": 80.0,
    "adaptive_threshold_enabled": true,
    "threshold_over_mean_ratio": 0.15,
    "adaptive_window_seconds": 3.0,
    "margin_frames": 5,
    "silence_frames": 15,
    "difference_threshold": 2.1,
    "pre_post_avg_frames": 5,
    "min_region_length": 5,
    "frame_difference_max": 15.0,
    "threshold_protection": {
      "enabled": false,
      "recovery_delay_seconds": 1.0,
      "stability_frames": 5,
      "waveform_trigger_enabled": true
    },
    "roi3_override": {
      "enabled": true,
      "threshold": 115.0,
      "require_roi3_data": true,
      "early_frame_threshold": 110
    }
  },
  "roi1_peak_detection": {
    "enabled": false,
    "threshold": 120.0,
    "threshold_minimum": 110.0,
    "adaptive_threshold_enabled": true,
    "threshold_over_mean_ratio": 0.08,
    "adaptive_window_seconds": 3.0,
    "margin_frames": 5,
    "silence_frames": 5,
    "difference_threshold": 2.0,
    "pre_post_avg_frames": 5,
    "min_region_length": 5,
    "threshold_protection": {
      "enabled": true,
      "recovery_delay_seconds": 1.0,
      "stability_frames": 5,
      "waveform_trigger_enabled": true
    }
  },
  "roi2_anti_jitter": {
    "enabled": true,
    "algorithm": "ema",
    "movement_threshold": 20.0,
    "stability_threshold": 8.0,
    "initialization_frames": 3,
    "ema": {
      "alpha": 0.25
    }
  },
  "hybrid_detection": {
    "enabled": false,
    "detection_strategy": "roi1_peaks_roi2_color",
    "fusion_strategy": "roi2_priority",
    "roi2_color_frames": {
      "pre_peak": 5,
      "post_peak": 10
    },
    "roi1_peak_width_range": [30, 40],
    "data_quality": {
      "minimum_roi2_frames": 15,
      "roi2_minimum_variance": 0.5,
      "roi2_min_gray": 5.0,
      "roi2_max_gray": 250.0,
      "skip_peaks_when_roi2_invalid": true
    },
    "fallback_enabled": true,
    "require_intersection": true
  },
  "analysis_cache": {
    "enabled": true,
    "flush_every": 50
  },
  "startup_cleanup": {
    "enabled": true,
    "cleanup_export": true,
    "cleanup_tmp": true,
    "cleanup_logs": true,
    "directories_to_clean": ["export", "tmp", "logs"]
  }
}
```

**配置参数说明：**

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `processing_mode` | string | "screen" | 处理模式：screen/video/vein_following |
| `roi_capture.frame_rate` | float | 10 | 帧率 (1-30 FPS) |
| `peak_detection.threshold` | float | 95.0 | ROI2 固定阈值 |
| `peak_detection.threshold_minimum` | float | 80.0 | ROI2 最小阈值保护 |
| `peak_detection.frame_difference_max` | float | 15.0 | ROI2 异常检测阈值（新增） |
| `peak_detection.roi3_override.early_frame_threshold` | int | 110 | ROI3 早期波峰帧数阈值（新增） |
| `roi1_peak_detection.min_region_length` | int | 5 | ROI1 波峰最小宽度（独立于 ROI2） |
| `roi2_anti_jitter.movement_threshold` | float | 20.0 | 防抖动运动阈值（像素） |
| `hybrid_detection.roi1_peak_width_range` | [int, int] | [30, 40] | ROI1 波峰宽度范围 |

---

## 4. 主处理流程 (run_daemon)

### 4.1 初始化阶段

```python
def run_daemon():
    # ========== 1. 清理旧数据 ==========
    cleanup_directories()

    # ========== 2. 加载配置 ==========
    config = load_fem_config()

    # ========== 3. 初始化防抖动滤波器（可选） ==========
    anti_jitter_config = config.get("roi2_anti_jitter", {})
    intersection_filter = None

    if anti_jitter_config.get("enabled", False):
        algorithm = anti_jitter_config.get("algorithm", "ema")

        if algorithm == "threshold":
            # 阈值式防抖动
            movement_threshold = float(anti_jitter_config.get("movement_threshold", 20.0))
            initialization_frames = int(anti_jitter_config.get("initialization_frames", 3))
            from threshold_based_anti_jitter import ThresholdIntersectionFilter
            intersection_filter = ThresholdIntersectionFilter(movement_threshold, initialization_frames)
        else:
            # EMA 平滑式防抖动
            ema_config = anti_jitter_config.get("ema", {})
            alpha = float(ema_config.get("alpha", 0.25))
            movement_threshold = float(anti_jitter_config.get("movement_threshold", 20.0))
            stability_threshold = float(anti_jitter_config.get("stability_threshold", 8.0))
            initialization_frames = int(anti_jitter_config.get("initialization_frames", 3))

            from green_detector import IntersectionFilter
            intersection_filter = IntersectionFilter(
                alpha=alpha,
                movement_threshold=movement_threshold,
                initialization_frames=initialization_frames,
                stability_threshold=stability_threshold
            )

    # ========== 4. 初始化分析缓存 ==========
    analysis_cache_conf = config.get("analysis_cache", {})
    analysis_cache = RoiAnalysisCache(
        export_dir=os.path.join(BASE_DIR, "export"),
        enabled=bool(analysis_cache_conf.get("enabled", True)),
        flush_every=int(analysis_cache_conf.get("flush_every", 50))
    )

    # ========== 5. 初始化统计管理器 ==========
    statistics_manager = VideoStatisticsManager()

    # ========== 6. 根据模式初始化视频/屏幕 ==========
    processing_mode = config.get("processing_mode", "screen")
    video_cap = None
    video_files = []
    current_video_index = 0

    if processing_mode == "screen":
        statistics_manager.initialize_for_video(None, is_batch=False)
    elif processing_mode == "video":
        video_config = config.get("video_processing", {})
        video_path = video_config.get("video_path", "")

        if os.path.isfile(video_path):
            video_files = [video_path]
        elif os.path.isdir(video_path):
            video_files = discover_video_files(video_path)

        if video_files:
            statistics_manager.initialize_for_video(video_files[0], is_batch=True)
            video_cap = initialize_video_capture(video_files[0])

    # ========== 7. 读取配置参数 ==========
    roi_capture_conf = config.get("roi_capture", {})
    data_processing = config.get("data_processing", {})
    peak_conf = config.get("peak_detection", {})
    roi1_peak_conf = config.get("roi1_peak_detection", {})
    hybrid_conf = config.get("hybrid_detection", {})

    # ROI 配置
    roi_default = roi_capture_conf.get("default_config", {})
    extension_params = roi_capture_conf.get("roi2_config", {}).get("extension_params", {})
    roi3_extension_params = roi_capture_conf.get("roi3_config", {}).get("extension_params", {})

    # 数据处理配置
    save_roi1 = bool(data_processing.get("save_roi1", False))
    save_roi2 = bool(data_processing.get("save_roi2", False))
    save_roi3 = bool(data_processing.get("save_roi3", False))
    save_wave = bool(data_processing.get("save_wave", False))
    save_roi1_wave = bool(data_processing.get("save_roi1_wave", False))
    only_delect = bool(data_processing.get("only_delect", False))

    # ROI2 波峰检测配置
    threshold = float(peak_conf.get("threshold", 105.0))
    threshold_minimum = float(peak_conf.get("threshold_minimum", 80.0))
    margin_frames = int(peak_conf.get("margin_frames", 5))
    diff_threshold = float(peak_conf.get("difference_threshold", 0.5))
    silence_frames = int(peak_conf.get("silence_frames", 0))
    pre_post_avg_frames = int(peak_conf.get("pre_post_avg_frames", 5))
    min_region_length = int(peak_conf.get("min_region_length", 1))
    frame_difference_max = float(peak_conf.get("frame_difference_max", 15.0))

    # ROI2 自适应阈值配置
    adaptive_threshold_enabled = bool(peak_conf.get("adaptive_threshold_enabled", False))
    threshold_over_mean_ratio = float(peak_conf.get("threshold_over_mean_ratio", 0.15))
    adaptive_window_seconds = float(peak_conf.get("adaptive_window_seconds", 3.0))

    # ROI2 阈值保护配置
    protection_conf = peak_conf.get("threshold_protection", {})
    protection_enabled = bool(protection_conf.get("enabled", False))
    recovery_delay_seconds = float(protection_conf.get("recovery_delay_seconds", 1.0))
    stability_frames = int(protection_conf.get("stability_frames", 5))
    waveform_trigger_enabled = bool(protection_conf.get("waveform_trigger_enabled", True))

    # ROI3 override 配置
    roi3_override_conf = peak_conf.get("roi3_override", {})
    roi3_override_enabled = bool(roi3_override_conf.get("enabled", False))
    roi3_override_threshold = float(roi3_override_conf.get("threshold", 115.0))
    roi3_early_frame_threshold = int(roi3_override_conf.get("early_frame_threshold", 110))
    require_roi3_data = bool(roi3_override_conf.get("require_roi3_data", True))

    # ROI1 配置
    roi1_enabled = bool(roi1_peak_conf.get("enabled", False))
    roi1_threshold = float(roi1_peak_conf.get("threshold", 120.0))
    roi1_threshold_minimum = float(roi1_peak_conf.get("threshold_minimum", 110.0))
    roi1_margin_frames = int(roi1_peak_conf.get("margin_frames", 5))
    roi1_silence_frames = int(roi1_peak_conf.get("silence_frames", 5))
    roi1_pre_post_avg_frames = int(roi1_peak_conf.get("pre_post_avg_frames", 5))
    roi1_min_region_length = int(roi1_peak_conf.get("min_region_length", 5))

    # ROI1 自适应阈值配置
    roi1_adaptive_threshold_enabled = bool(roi1_peak_conf.get("adaptive_threshold_enabled", True))
    roi1_threshold_over_mean_ratio = float(roi1_peak_conf.get("threshold_over_mean_ratio", 0.08))
    roi1_adaptive_window_seconds = float(roi1_peak_conf.get("adaptive_window_seconds", 3.0))

    # ROI1 阈值保护配置
    roi1_protection_conf = roi1_peak_conf.get("threshold_protection", {})
    roi1_protection_enabled = bool(roi1_protection_conf.get("enabled", True))
    roi1_recovery_delay_seconds = float(roi1_protection_conf.get("recovery_delay_seconds", 1.0))
    roi1_stability_frames = int(roi1_protection_conf.get("stability_frames", 5))
    roi1_waveform_trigger_enabled = bool(roi1_protection_conf.get("waveform_trigger_enabled", True))

    # 混合检测配置
    hybrid_enabled = bool(hybrid_conf.get("enabled", False))
    roi2_pre_frames = int(hybrid_conf.get("roi2_color_frames", {}).get("pre_peak", 5))
    roi2_post_frames = int(hybrid_conf.get("roi2_color_frames", {}).get("post_peak", 10))
    peak_width_range = hybrid_conf.get("roi1_peak_width_range", [30, 40])
    min_peak_width = int(peak_width_range[0])
    max_peak_width = int(peak_width_range[1])

    data_quality_conf = hybrid_conf.get("data_quality", {})
    min_roi2_frames = int(data_quality_conf.get("minimum_roi2_frames", 15))
    roi2_min_variance = float(data_quality_conf.get("roi2_minimum_variance", 0.5))
    roi2_min_gray = float(data_quality_conf.get("roi2_min_gray", 5.0))
    roi2_max_gray = float(data_quality_conf.get("roi2_max_gray", 250.0))
    fallback_enabled = bool(hybrid_conf.get("fallback_enabled", True))

    # 帧率配置
    roi_frame_rate = float(roi_capture_conf.get("frame_rate", 10))
    if processing_mode == "video" and video_cap is not None:
        video_fps = _get_video_fps(video_cap)
        effective_frame_rate = min(roi_frame_rate, video_fps)
        video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))
    else:
        effective_frame_rate = roi_frame_rate
        video_frame_step = 1

    interval_seconds = 1.0 / effective_frame_rate

    # ========== 8. 初始化状态变量 ==========
    # 帧索引
    frame_index = 0
    first_video_frame = True

    # ROI2 状态
    gray_buffer: Deque[float] = deque(maxlen=100)
    bg_count: int = 0
    bg_mean: float = 0.0
    threshold_protection_active: bool = False
    protection_end_time: float = 0.0
    consecutive_below_threshold: int = 0
    last_waveform_time: float = 0.0

    # ROI1 状态
    roi1_gray_buffer: Deque[float] = deque(maxlen=100)
    roi1_bg_count: int = 0
    roi1_bg_mean: float = 0.0
    roi1_threshold_protection_active: bool = False
    roi1_protection_end_time: float = 0.0
    roi1_consecutive_below_threshold: int = 0
    roi1_last_waveform_time: float = 0.0
    roi1_threshold_used: float = max(roi1_threshold, roi1_threshold_minimum)

    # ROI3 状态
    roi3_gray_buffer: Deque[float] = deque(maxlen=100)

    # ROI1 波峰管理（防重复）
    processed_roi1_peaks: Dict[int, str] = {}
    roi1_peak_counter: int = 0

    # 绿线交点回退
    last_intersection_roi: Optional[Tuple[int, int]] = None

    # ========== 9. 创建输出目录 ==========
    tmp_root = _create_video_folders(...)
    roi1_dir = os.path.join(tmp_root, "roi1")
    roi2_dir = os.path.join(tmp_root, "roi2")
    roi3_dir = os.path.join(tmp_root, "roi3")
    wave_dir = os.path.join(tmp_root, "wave")
    wave1_dir = os.path.join(tmp_root, "wave1")

    # ========== 10. 启动分析缓存会话 ==========
    analysis_cache.start_session(
        session_id=statistics_manager.current_statistics.session_id,
        processing_mode=processing_mode,
        video_path=video_files[0] if video_files else None,
        config=config
    )
```

### 4.2 主循环

```python
while True:
    loop_start = time.time()
    frame_index += 1

    try:
        # ========== 步骤1: 捕获图像 ==========
        if processing_mode == "video":
            step = 1 if first_video_frame else video_frame_step
            first_video_frame = False
            screen = get_video_frame(video_cap, loop_enabled, frame_step=step)

            if screen is None:
                # 视频结束，切换到下一个视频
                handle_video_switch()
                continue  # 跳过本帧的剩余处理

            screen_width, screen_height = screen.size
        else:
            screen = ImageGrab.grab()
            screen_width, screen_height = screen.size

        # ========== 步骤2: 提取 ROI1 ==========
        x1, y1, x2, y2 = adjust_roi1_to_screen(
            (screen_width, screen_height),
            roi_default
        )
        roi1_image = screen.crop((x1, y1, x2, y2))
        roi1_width, roi1_height = roi1_image.size

        # ========== 步骤3: 检测绿线交点 ==========
        roi_cv_image = cv2.cvtColor(
            np.array(roi1_image),
            cv2.COLOR_RGB2BGR
        )

        try:
            intersection = detect_green_intersection(
                roi_cv_image,
                anti_jitter_config,
                intersection_filter
            )
        except Exception as e:
            print(f"Warning: Green intersection detection failed: {e}")
            intersection = None

        # 使用最后已知交点或 ROI1 中心作为回退
        if intersection is not None:
            last_intersection_roi = intersection

        if last_intersection_roi is not None:
            center_x, center_y = last_intersection_roi
        else:
            center_x = roi1_width // 2
            center_y = roi1_height // 2

        # ========== 步骤4: 计算 ROI2 和 ROI3 ==========
        roi2_region = compute_roi2_region(
            (roi1_width, roi1_height),
            (center_x, center_y),
            extension_params
        )

        roi2_gray: Optional[float] = None
        roi2_image: Optional[Image.Image] = None
        roi3_gray: Optional[float] = None
        roi3_image: Optional[Image.Image] = None
        roi1_gray: Optional[float] = None

        if roi2_region is not None:
            rx1, ry1, rx2, ry2 = roi2_region
            roi2_image = roi1_image.crop((rx1, ry1, rx2, ry2))
            roi2_gray = compute_average_gray(roi2_image)
            gray_buffer.append(roi2_gray)

            # ROI3 提取
            if roi3_extension_params:
                roi3_region = compute_roi2_region(
                    (roi1_width, roi1_height),
                    (center_x, center_y),
                    roi3_extension_params
                )
                if roi3_region is not None:
                    r3x1, r3y1, r3x2, r3y2 = roi3_region
                    roi3_image = roi1_image.crop((r3x1, r3y1, r3x2, r3y2))
                    roi3_gray = compute_average_gray(roi3_image)
                    roi3_gray_buffer.append(roi3_gray)

            # ROI1 灰度计算
            if roi1_enabled:
                roi1_gray = compute_average_gray(roi1_image)
                roi1_gray_buffer.append(roi1_gray)

        # ========== 步骤5: ROI2 自适应阈值计算 ==========
        threshold_used = max(threshold, threshold_minimum)
        calculated_bg_mean: Optional[float] = None

        if adaptive_threshold_enabled and len(gray_buffer) >= adaptive_window_frames:
            recent_frames_count = min(len(gray_buffer), adaptive_window_frames)
            recent_frames = list(gray_buffer)[-recent_frames_count:]
            calculated_bg_mean = sum(recent_frames) / len(recent_frames)

            # 检查阈值保护状态
            current_time = time.time()
            if threshold_protection_active:
                # 管理阈值保护
                (threshold_protection_active,
                 protection_end_time,
                 consecutive_below_threshold,
                 frames_since_protection_end,
                 last_waveform_time) = manage_threshold_protection(
                    current_gray=roi2_gray if roi2_gray is not None else 0,
                    current_threshold=threshold_used,
                    has_peaks=False,
                    frame_time=current_time,
                    protection_active=threshold_protection_active,
                    protection_end_time=protection_end_time,
                    consecutive_below=consecutive_below_threshold,
                    last_waveform_time=last_waveform_time,
                    enabled=protection_enabled,
                    recovery_delay_frames=int(recovery_delay_seconds * effective_frame_rate),
                    stability_frames=stability_frames,
                    waveform_trigger=waveform_trigger_enabled,
                    threshold_minimum=threshold_minimum
                )

            # 只有在保护未激活时才更新背景均值
            if not threshold_protection_active:
                bg_mean = calculated_bg_mean
                bg_count = recent_frames_count
                threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                threshold_used = max(threshold_used, threshold_minimum)
            else:
                # 保护期间使用冻结的阈值
                if bg_mean > 0:
                    threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
                    threshold_used = max(threshold_used, threshold_minimum)

        # ========== 步骤5.5: ROI1 自适应阈值计算（独立） ==========
        roi1_curve = list(roi1_gray_buffer) if roi1_gray_buffer else []

        if roi1_enabled and roi1_gray_buffer:
            roi1_adaptive_window_frames = int(roi1_adaptive_window_seconds * effective_frame_rate)
            roi1_adaptive_window_frames = max(1, min(roi1_adaptive_window_frames, 100))

            if (roi1_adaptive_threshold_enabled and
                len(roi1_gray_buffer) >= roi1_adaptive_window_frames):

                current_time = time.time()

                # ROI1 阈值保护管理
                if roi1_threshold_protection_active:
                    # 类似 ROI2 的保护管理
                    pass  # 简化示例

                # ROI1 使用增量更新：只有当前值低于阈值时才更新
                if roi1_gray < roi1_threshold_used:
                    roi1_bg_count += 1
                    roi1_bg_mean = roi1_bg_mean + (roi1_gray - roi1_bg_mean) / roi1_bg_count

                # 计算 ROI1 自适应阈值
                if roi1_adaptive_threshold_enabled and roi1_bg_mean > 0:
                    roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
                    roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)

        # ========== 步骤6: 波峰检测 ==========
        green_peaks: List[Tuple[int, int]] = []
        red_peaks: List[Tuple[int, int]] = []
        hybrid_peaks: List[Dict[str, Any]] = []
        detection_mode = "roi2_legacy"

        if hybrid_enabled and roi1_enabled and len(roi1_gray_buffer) > 0 and len(gray_buffer) > 0:
            # 混合检测模式
            roi1_curve = list(roi1_gray_buffer)
            roi2_curve = list(gray_buffer)

            hybrid_config = {
                'roi1_threshold': roi1_threshold_used,
                'margin_frames': roi1_margin_frames,
                'silence_frames': roi1_silence_frames,
                'pre_post_avg_frames': roi1_pre_post_avg_frames,
                'min_peak_width': roi1_min_region_length,
                'max_peak_width': max_peak_width,
                'roi2_pre_frames': roi2_pre_frames,
                'roi2_post_frames': roi2_post_frames,
                'roi2_color_threshold': diff_threshold,
                'roi2_frame_diff_max': frame_difference_max,  # 新增
                'minimum_roi2_frames': min_roi2_frames,
                'roi2_minimum_variance': roi2_min_variance,
                'roi2_min_gray': roi2_min_gray,
                'roi2_max_gray': roi2_max_gray,
                'fallback_enabled': fallback_enabled,
                'require_intersection': bool(hybrid_conf.get("require_intersection", True)),
                'intersection_detected': bool(intersection is not None),
                'skip_when_roi2_invalid': bool(data_quality_conf.get("skip_peaks_when_roi2_invalid", True)),
                'frame_index': frame_index,
                'buffer_start_frame_index': frame_index - len(roi1_curve) + 1
            }

            try:
                hybrid_peaks = hybrid_peak_detection(
                    roi1_curve,
                    roi2_curve,
                    hybrid_config,
                    processed_roi1_peaks,
                    roi1_peak_counter
                )
                detection_mode = "hybrid_roi1_peaks_roi2_color"

                # 转换为传统格式
                for peak in hybrid_peaks:
                    if peak['color'] == 'green':
                        green_peaks.append(peak['peak_interval'])
                    else:
                        red_peaks.append(peak['peak_interval'])

            except Exception as e:
                print(f"[混合检测] 执行失败: {e}")
                hybrid_peaks = []

        elif gray_buffer:
            # 传统 ROI2 检测模式
            curve = list(gray_buffer)
            try:
                green_peaks_raw, red_peaks_raw = detect_peaks(
                    curve,
                    threshold=threshold_used,
                    marginFrames=margin_frames,
                    differenceThreshold=diff_threshold,
                    silenceFrames=silence_frames,
                    avgFrames=pre_post_avg_frames
                )

                # 应用最小宽度过滤
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

            except Exception:
                green_peaks, red_peaks = [], []

        # ========== 步骤6.5: 重新检查阈值保护（使用实际波峰结果）==========
        if protection_enabled and roi2_gray is not None:
            has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
            current_time = time.time()

            (threshold_protection_active,
             protection_end_time,
             consecutive_below_threshold,
             frames_since_protection_end,
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
                recovery_delay_frames=int(recovery_delay_seconds * effective_frame_rate),
                stability_frames=stability_frames,
                waveform_trigger=waveform_trigger_enabled,
                threshold_minimum=threshold_minimum
            )

        # ========== 步骤7: 添加到统计 ==========
        current_stats = statistics_manager.current_statistics

        if current_stats:
            # 准备 ROI2 信息
            roi2_info = None
            if roi2_region is not None:
                rx1, ry1, rx2, ry2 = roi2_region
                roi2_info = {
                    'x1': rx1, 'y1': ry1,
                    'x2': rx2, 'y2': ry2,
                    'width': rx2 - rx1,
                    'height': ry2 - ry1
                }

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
                roi1_curve=list(roi1_gray_buffer),
                roi1_threshold_used=roi1_threshold_used,
                # ROI3 参数（包含覆盖逻辑）
                roi3_curve=list(roi3_gray_buffer),
                roi3_override_enabled=roi3_override_enabled,
                roi3_override_threshold=roi3_override_threshold
            )

            # ROI3 覆盖逻辑会在 SafePeakStatistics 内部自动应用：
            # 1. RED -> GREEN: 如果 roi3_peak_max_value > roi3_override_threshold
            # 2. 任何颜色 -> RED: 如果 roi3_peak_max_frame < roi3_early_frame_threshold

        # ========== 步骤8: 保存图像和波形 ==========
        has_peak = len(green_peaks) > 0 or len(red_peaks) > 0
        should_save = (not only_delect) or has_peak
        roi1_should_save = (not only_delect) or (len(roi1_gray_buffer) > 0)

        if should_save:
            # 保存 ROI1
            if save_roi1:
                roi1_path = os.path.join(roi1_dir, f"roi1_{frame_index:06d}.png")
                try:
                    roi1_image.save(roi1_path)
                except Exception as e:
                    print(f"[ERROR] Failed to save ROI1 {roi1_path}: {e}")

            # 保存 ROI2
            if save_roi2 and roi2_image is not None:
                # 计算视频时间戳（视频模式）
                video_time_str = ""
                if processing_mode == "video" and video_cap is not None:
                    try:
                        video_pos_msec = float(video_cap.get(cv2.CAP_PROP_POS_MSEC))
                        video_seconds = video_pos_msec / 1000.0
                        video_time_str = f"_{video_seconds:06.2f}s"
                    except Exception:
                        video_time_str = "_0000.00s"

                roi2_path = os.path.join(roi2_dir, f"roi2_{frame_index:06d}{video_time_str}.png")
                try:
                    roi2_image.save(roi2_path)
                except Exception:
                    pass

            # 保存 ROI3
            if save_roi3 and roi3_image is not None and roi3_dir:
                roi3_path = os.path.join(roi3_dir, f"roi3_{frame_index:06d}{video_time_str}.png")
                try:
                    roi3_image.save(roi3_path)
                except Exception:
                    pass

            # 保存 ROI2 波形图
            if save_wave and gray_buffer:
                try:
                    wave_path = os.path.join(wave_dir, f"wave_{frame_index:06d}.png")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    x = list(range(len(gray_buffer)))
                    ax.plot(x, list(gray_buffer), color="black", linewidth=1)

                    # 添加 ROI3 曲线
                    if roi3_gray_buffer:
                        x3 = list(range(len(roi3_gray_buffer)))
                        ax.plot(x3, list(roi3_gray_buffer), color="purple", linewidth=1, label="ROI3")
                        ax.legend()

                    # 绘制背景均值
                    if bg_count > 0:
                        ax.axhline(bg_mean, color="blue", linestyle="--", linewidth=1, label="bg_mean")

                    # 绘制阈值
                    threshold_color = "red" if threshold_protection_active else "orange"
                    ax.axhline(threshold_used, color=threshold_color, linestyle="-", linewidth=1.5,
                              label=f"threshold ({threshold_used:.1f})")

                    # 标记波峰
                    for start, end in green_peaks:
                        s = max(0, start - 1)
                        e = min(len(gray_buffer) - 1, end + 1)
                        ax.plot(range(s, e + 1), list(gray_buffer)[s:e+1], color="green", linewidth=2)

                    for start, end in red_peaks:
                        s = max(0, start - 1)
                        e = min(len(gray_buffer) - 1, end + 1)
                        ax.plot(range(s, e + 1), list(gray_buffer)[s:e+1], color="red", linewidth=2)

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
                    pass

            # 保存 ROI1 波形图
            if roi1_should_save and save_roi1_wave and roi1_enabled and roi1_curve:
                try:
                    roi1_wave_path = os.path.join(wave1_dir, f"roi1_wave_{frame_index:06d}.png")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    x = list(range(len(roi1_curve)))
                    ax.plot(x, roi1_curve, color="darkblue", linewidth=1, label="ROI1")

                    # 绘制 ROI1 背景均值
                    if roi1_bg_count > 0:
                        ax.axhline(roi1_bg_mean, color="blue", linestyle="--", linewidth=1, label="bg_mean")

                    # 绘制 ROI1 阈值
                    roi1_threshold_color = "red" if roi1_threshold_protection_active else "orange"
                    ax.axhline(roi1_threshold_used, color=roi1_threshold_color, linestyle="-", linewidth=1.5,
                              label=f"threshold ({roi1_threshold_used:.1f})")

                    ax.set_title(f"ROI1 Waveform - Frame {frame_index}")
                    ax.set_xlabel("Frame Index (relative)")
                    ax.set_ylabel("Gray Value (0-255)")
                    ax.set_ylim(0, 100)
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)

                    fig.tight_layout()
                    fig.savefig(roi1_wave_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception:
                    pass

        # ========== 步骤9: 写入分析缓存 ==========
        try:
            analysis_cache.record_frame({
                "ts_wall": loop_start,
                "ts_local": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "frame_index": frame_index,
                "roi2_gray": roi2_gray,
                "buffer": {
                    "len": len(gray_buffer),
                    "start_frame_index": max(0, frame_index - len(gray_buffer) + 1)
                },
                "threshold": {
                    "used": threshold_used,
                    "bg_mean": (float(bg_mean) if bg_count > 0 else None),
                    "protection_active": threshold_protection_active
                },
                "peaks": {
                    "green": green_peaks,
                    "red": red_peaks
                },
                "detection": {
                    "mode": detection_mode,
                    "hybrid_enabled": hybrid_enabled
                }
            })
        except Exception:
            pass

        # ========== 步骤10: 记录日志 ==========
        if (not only_delect) or has_peak:
            log_line = (
                f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} "
                f"gray={roi2_gray:.1f if roi2_gray is not None else 'nan'} "
                f"green_peaks={len(green_peaks)} red_peaks={len(red_peaks)}"
            )
            logger.info(log_line)

    except KeyboardInterrupt:
        print("Daemon stopped by user")
        break
    except Exception as e:
        print(f"Error: {e}")

    # ========== 帧率控制 ==========
    elapsed = time.time() - loop_start
    sleep_time = max(0.0, interval_seconds - elapsed)
    time.sleep(sleep_time)
```

### 4.3 视频切换处理

```python
def handle_video_switch():
    global current_video_index, video_cap, frame_index
    global gray_buffer, roi1_gray_buffer, roi3_gray_buffer
    global bg_count, bg_mean, threshold_protection_active
    global roi1_bg_count, roi1_bg_mean, roi1_threshold_protection_active
    global processed_roi1_peaks, roi1_peak_counter
    global first_video_frame

    # 释放当前视频
    video_cap.release()

    # 移动到下一个视频
    current_video_index += 1

    if current_video_index < len(video_files):
        # ========== 重置所有状态变量 ==========
        # 清空缓冲区
        gray_buffer.clear()
        roi1_gray_buffer.clear()
        roi3_gray_buffer.clear()

        # 重置 ROI2 状态
        bg_count = 0
        bg_mean = 0.0
        threshold_protection_active = False
        protection_end_time = 0.0
        consecutive_below_threshold = 0
        last_waveform_time = 0.0

        # 重置 ROI1 状态
        roi1_bg_count = 0
        roi1_bg_mean = 0.0
        roi1_threshold_protection_active = False
        roi1_protection_end_time = 0.0
        roi1_consecutive_below_threshold = 0
        roi1_last_waveform_time = 0.0
        roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)

        # 重置 ROI1 波峰 ID 管理（重要！）
        processed_roi1_peaks.clear()
        roi1_peak_counter = 0  # 重置计数器

        # 重置帧索引
        frame_index = 0
        first_video_frame = True

        # 重置防抖动滤波器
        if intersection_filter:
            intersection_filter.reset()

        # 初始化新视频统计
        current_stats = statistics_manager.initialize_for_video(
            video_files[current_video_index],
            is_batch=True
        )

        # 打开新视频
        video_cap = initialize_video_capture(video_files[current_video_index])

        # 重新计算帧率参数
        video_fps = _get_video_fps(video_cap)
        if video_fps > 0:
            effective_frame_rate = min(roi_frame_rate, video_fps)
            video_frame_step = max(1, int(round(video_fps / effective_frame_rate)))

        # 创建新的输出目录
        tmp_root = _create_video_folders(...)
        roi1_dir = os.path.join(tmp_root, "roi1")
        roi2_dir = os.path.join(tmp_root, "roi2")
        roi3_dir = os.path.join(tmp_root, "roi3")
        wave_dir = os.path.join(tmp_root, "wave")
        wave1_dir = os.path.join(tmp_root, "wave1")

        # 启动新的分析缓存会话
        analysis_cache.start_session(
            session_id=current_stats.session_id,
            processing_mode=processing_mode,
            video_path=video_files[current_video_index],
            config=config
        )

    else:
        # 所有视频处理完毕
        print("All videos processed")
        break
```

---

## 5. 关键算法

### 5.1 阈值保护机制 (manage_threshold_protection)

**目的：** 防止波峰期间背景均值被污染

**触发条件：**
1. 波形触发：当前灰度 >= 当前阈值
2. 波峰触发：检测到波峰

**解除条件（需同时满足）：**
1. 时间延迟：距离上次触发超过 `recovery_delay_frames` 帧
2. 稳定性检查：连续 `stability_frames` 帧低于阈值

**状态变量：**
```python
threshold_protection_active: bool  # 是否激活保护
protection_end_time: float  # 计划结束时间
consecutive_below_threshold: int  # 连续低于阈值的帧数
last_waveform_time: float  # 上次触发时间
```

**实现逻辑：**
```python
def manage_threshold_protection(
    current_gray, current_threshold, has_peaks, frame_time,
    protection_active, protection_end_time,
    consecutive_below, last_waveform_time,
    enabled=True, recovery_delay_frames=10, stability_frames=5,
    waveform_trigger=True, threshold_minimum=80.0
):
    current_time = frame_time
    frames_since_end = max(0, int((current_time - protection_end_time) / 0.1))  # 假设10fps

    if not enabled:
        return False, protection_end_time, consecutive_below, frames_since_end, last_waveform_time

    should_protect = protection_active

    # 1. 波形触发
    if waveform_trigger and current_gray >= current_threshold:
        should_protect = True
        last_waveform_time = current_time

    # 2. 波峰触发
    elif has_peaks and not protection_active:
        should_protect = True
        last_waveform_time = current_time

    # 3. 检查是否可以解除保护
    if should_protect:
        planned_end_time = last_waveform_time + (recovery_delay_frames * 0.1)

        if current_gray < current_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        time_condition = current_time >= planned_end_time
        stability_condition = consecutive_below >= stability_frames

        if time_condition and stability_condition:
            should_protect = False
            consecutive_below = 0
            frames_since_end = 0
        else:
            protection_end_time = planned_end_time

    return should_protect, protection_end_time, consecutive_below, frames_since_end, last_waveform_time
```

### 5.2 ROI2 颜色判定 (determine_roi2_color_in_interval)

**计算步骤：**
1. 计算波峰前的平均值（默认前5帧，来自配置 `roi2_color_frames.pre_peak`）
2. 计算波峰后的平均值（默认后10帧，来自配置 `roi2_color_frames.post_peak`）
3. 计算前后差异：`frame_difference = post_avg - pre_avg`
4. 异常过滤：如果 `|frame_difference| > frame_difference_max`，判定为错误数据
5. 颜色分类：
   - 绿色：`frame_difference >= color_threshold`
   - 红色：`frame_difference < color_threshold`
6. 数据质量检查：
   - 最小帧数检查（默认15帧，来自配置 `data_quality.minimum_roi2_frames`）
   - 方差检查（默认最小0.5，来自配置 `data_quality.roi2_minimum_variance`）
   - 灰度范围检查（5-250，来自配置 `data_quality.roi2_min_gray/max_gray`）

**返回格式：**
```python
{
    'color': 'green' | 'red',
    'method': 'roi2' | 'roi1_fallback' | 'error_filtered' | 'roi2_invalid',
    'frame_difference': float,
    'threshold': float,
    'pre_avg': float,
    'post_avg': float,
    'confidence': float,
    'roi2_valid': bool,
    'quality_score': float,
    'variance': float,
    'data_range': float,
    'error': str  # 错误信息（如果适用）
}
```

### 5.3 混合波峰检测 (hybrid_peak_detection)

**流程：**
1. 使用 ROI1 数据检测波峰区间（阈值高，不区分颜色）
2. 为每个 ROI1 波峰生成唯一 ID（基于绝对帧位置）
3. 使用 ROI2 数据在相同区间内判定颜色
4. 应用数据质量过滤
5. 返回混合检测结果

**唯一 ID 生成规则：**
```python
# 计算波峰在缓冲区中的绝对最大值位置
peak_slice = roi1_curve[peak_start:peak_end + 1]
local_max_offset = max(range(len(peak_slice)), key=lambda i: peak_slice[i])
abs_peak_max = buffer_start_frame_index + peak_start + local_max_offset

# 生成唯一 ID
peak_key = abs_peak_max
peak_id = f"ROI1_MAX_{abs_peak_max:06d}"
```

**防重复机制：**
```python
processed_roi1_peaks: Dict[int, str] = {}  # {peak_key: peak_id}

# 检查是否已处理
if peak_key in processed_roi1_peaks:
    continue  # 已处理过，跳过

# 记录新波峰
processed_roi1_peaks[peak_key] = peak_id
roi1_peak_counter += 1
```

### 5.4 绿线交点检测（green_detector 模块）

**输入：** ROI1 的 OpenCV 图像（BGR 格式）

**处理流程：**
1. HSV 颜色空间过滤
   - 绿色范围：H[35-85], S[80-255], V[80-255]
2. Canny 边缘检测（threshold1=50, threshold2=150）
3. Hough 直线变换
   - threshold=50, minLineLength=80, maxLineGap=20
4. 选择非平行线对
5. 计算几何交点

**防抖动处理（可选）：**
- **EMA 算法**：
  - 大运动（> movement_threshold）：直接通过
  - 小运动（< stability_threshold）：强力平滑
  - 中等运动：正常平滑
- **阈值算法**：
  - 小于 movement_threshold：完全忽略
  - 大于等于：接受新位置

---

## 6. 视频处理

### 6.1 视频发现 (discover_video_files)

```python
def discover_video_files(video_path: str) -> List[str]:
    """
    支持的视频格式：
    - .mp4, .avi, .mov, .mkv, .flv, .wmv
    - 按文件名排序

    Args:
        video_path: 视频文件或文件夹路径

    Returns:
        排序后的视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    files = []
    for ext in video_extensions:
        pattern = os.path.join(video_path, f"*{ext}")
        files.extend(glob.glob(pattern))
    return sorted(files)
```

### 6.2 视频切换逻辑

**关键点：**
- 必须重置所有缓冲区
- 必须重置所有状态变量
- **必须重置 ROI1 波峰计数器**（roi1_peak_counter = 0）
- 必须清空波峰 ID 字典（processed_roi1_peaks.clear()）
- 必须重置防抖动滤波器

### 6.3 帧率控制

**视频模式：**
```python
video_fps = _get_video_fps(video_cap)
effective_frame_rate = min(roi_frame_rate, video_fps)
video_frame_step = max(1, int(video_fps / effective_frame_rate))
interval_seconds = 1.0 / effective_frame_rate

# 主循环中：
step = 1 if first_video_frame else video_frame_step
screen = get_video_frame(video_cap, loop_enabled, frame_step=step)
```

**屏幕模式：**
```python
effective_frame_rate = roi_frame_rate
interval_seconds = 1.0 / effective_frame_rate
screen = ImageGrab.grab()
```

---

## 7. 数据导出

### 7.1 CSV 导出格式

**文件命名：** `peak_statistics_{video_name}_{timestamp}.csv`

**字段列表：**
```csv
timestamp,frame_index,peak_type,peak_start,peak_end,width,
roi1_frame_diff,roi2_frame_diff,pre_peak_avg,post_peak_avg,
difference_threshold_used,threshold_used,bg_mean,
peak_max_value,roi3_peak_max_value,roi3_peak_max_frame,
pre_peak_frame_start,pre_peak_frame_end,post_peak_frame_start,post_peak_frame_end,
roi1_peak_id,roi1_detection_method,roi2_color_method,
intersection_x,intersection_y,roi2_x1,roi2_y1,roi2_x2,roi2_y2,
roi2_width,roi2_height,roi3_override_applied,roi3_override_threshold
```

### 7.2 分析缓存格式 (JSONL)

**元数据行：**
```json
{
  "type": "meta",
  "cache_version": 1,
  "created_at": "2025-12-25T00:23:19",
  "session_id": "20251225_002319",
  "processing_mode": "video",
  "video_path": "/path/to/video.mp4",
  "config": {...}
}
```

**帧数据行：**
```json
{
  "type": "frame",
  "ts_wall": 1735075399.123,
  "ts_local": "2025-12-25T00:23:19",
  "frame_index": 93,
  "roi2_gray": 105.5,
  "buffer": {"len": 100, "start_frame_index": 1},
  "threshold": {"used": 95.0, "bg_mean": 82.5, "protection_active": false},
  "peaks": {"green": [[45, 50]], "red": []},
  "detection": {"mode": "hybrid_roi1_peaks_roi2_color"}
}
```

---

## 8. 文件组织结构

### 8.1 目录结构

```
SimpleFEM/
├── export/
│   ├── peak_statistics_*.csv
│   ├── roi_analysis_cache_*.jsonl
│   └── tmp_{session_id}/
│       ├── roi1/
│       ├── roi2/
│       ├── roi3/
│       ├── wave/
│       └── wave1/
├── logs/
│   └── roi_peak_daemon.log
├── tmp/
│   └── {session_id}/
└── simple_roi_daemon.py
```

### 8.2 日志格式

**格式：** 纯文本，每日轮转

**示例：**
```
2025-12-25T00:23:19 gray=105.5 green_peaks=1 red_peaks=0
2025-12-25T00:23:20 gray=98.2 green_peaks=0 red_peaks=1
```

---

## 9. 重要注意事项

### 9.1 关键常量

- **循环缓冲区大小**: 100 帧（固定）
- **ROI2 最大宽度**: ~80x120 像素
- **默认帧率**: 10 FPS
- **默认 ROI1**: 1280x80 到 1920x980
- **波峰最小宽度**: 5 帧（可配置）

### 9.2 阈值计算顺序

**ROI2（替换更新）：**
1. 基础阈值：`threshold`（配置文件）
2. 最小阈值保护：`max(threshold, threshold_minimum)`
3. 自适应计算：`bg_mean * (1.0 + threshold_over_mean_ratio)`
4. 最终阈值：`max(adaptive_threshold, threshold_minimum)`

**ROI1（增量更新）：**
1. 基础阈值：`roi1_threshold`（配置文件）
2. 最小阈值保护：`max(roi1_threshold, roi1_threshold_minimum)`
3. 背景均值更新（增量）：`bg_mean = bg_mean + (current - bg_mean) / count`
4. 自适应计算：`bg_mean * (1.0 + threshold_over_mean_ratio)`
5. 最终阈值：`max(adaptive_threshold, threshold_minimum)`

### 9.3 ROI3 覆盖逻辑

**条件：**
- `roi3_override_enabled == true`
- `roi3_curve` 不为空（或 `require_roi3_data == false`）

**规则：**
1. RED -> GREEN：如果 `roi3_peak_max_value > roi3_override_threshold`
2. 任何颜色 -> RED：如果 `roi3_peak_max_frame < roi3_early_frame_threshold`（默认110）

**ROI3 最大值帧计算：**
```python
curve_start_global_frame = frame_index - len(roi3_curve) + 1
roi3_peak_max_frame = curve_start_global_frame + roi3_max_curve_idx
```

### 9.4 ROI2 颜色判定异常检测

**异常条件：**
```python
if abs(frame_difference) > frame_difference_max:  # 默认 15.0
    return {
        'color': 'red',
        'method': 'error_filtered',
        'roi2_valid': False,
        'error': f'frame_difference异常(|{frame_difference:.1f}| > {frame_difference_max})'
    }
```

### 9.5 防抖动策略

**EMA 算法：**
```python
if movement_distance > movement_threshold:
    # 大运动：直接通过
    filtered = new_position
elif movement_distance < stability_threshold:
    # 小运动：强力平滑
    filtered = alpha * new_position + (1 - alpha) * filtered
else:
    # 中等运动：正常平滑
    filtered = alpha * new_position + (1 - alpha) * filtered
```

**阈值算法：**
```python
if movement_distance < movement_threshold:
    # 忽略移动，保持不变
    pass  # 不更新
else:
    # 接受新位置
    filtered = new_position
```

### 9.6 边界检查

```python
def adjust_roi1_to_screen(screen_size, roi_default):
    """确保 ROI 坐标在屏幕范围内"""
    screen_width, screen_height = screen_size
    x1 = max(0, min(roi_default['x1'], screen_width - 1))
    y1 = max(0, min(roi_default['y1'], screen_height - 1))
    x2 = max(x1 + 1, min(roi_default['x2'], screen_width))
    y2 = max(y1 + 1, min(roi_default['y2'], screen_height))
    return x1, y1, x2, y2
```

### 9.7 错误处理原则

1. **守护进程永不崩溃**：所有异常必须捕获
2. **单帧失败不影响后续处理**：使用 try-except 包裹关键代码
3. **记录错误但继续运行**：使用 print/logging 记录错误
4. **优雅降级**：如果绿线检测失败，使用上一次的位置

---

## 10. 外部依赖

### 10.1 必需模块

```python
import json
import logging
import logging.handlers
import os
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
```

### 10.2 本地模块

```python
from green_detector import detect_green_intersection, IntersectionFilter
from peak_detection import detect_peaks
from safe_peak_statistics import SafePeakStatistics
from threshold_based_anti_jitter import ThresholdIntersectionFilter
```

---

## 11. 测试和调试

### 11.1 调试输出

**关键日志点：**
- 帧率控制：每 10 帧打印一次
- ROI1 阈值：每 50 帧打印一次
- 混合检测：每次检测打印详细信息
- 防抖动：启动时打印配置参数

### 11.2 验证检查点

1. **绿线检测**：检查 intersection 是否为 None
2. **ROI2 提取**：检查 roi2_region 是否为 None
3. **缓冲区状态**：检查 len(gray_buffer)
4. **波峰检测**：检查 green_peaks 和 red_peaks 数量
5. **数据导出**：检查 stats_write_results 返回值

---

## 12. 常见问题和解决方案

### 12.1 绿线检测失败

**症状：** intersection 始终为 None

**解决方案：**
1. 检查 ROI1 坐标是否包含绿线
2. 调整 HSV 颜色范围
3. 降低 Canny 检测阈值
4. 检查图像是否正确转换为 BGR

### 12.2 ROI2 抖动严重

**症状：** ROI2 位置频繁跳跃

**解决方案：**
1. 启用防抖动：`roi2_anti_jitter.enabled = true`
2. 增大 `movement_threshold`（默认 20.0，可调整到 30-40）
3. 降低 EMA 的 alpha（如 0.15-0.20）
4. 启用阈值式算法

### 12.3 波峰检测不准确

**症状：** 漏检或误检

**解决方案：**
1. 调整阈值（`threshold` 或 `roi1_threshold`）
2. 启用自适应阈值
3. 调整 `difference_threshold`
4. 检查波峰最小宽度设置
5. 查看波形图确认信号质量

### 12.4 多视频数据污染

**症状：** 第二个视频的数据包含第一个视频的信息

**解决方案：**
1. 确保视频切换时重置所有状态变量
2. 清空所有缓冲区
3. **重置 ROI1 波峰计数器**（roi1_peak_counter = 0）
4. 清空波峰 ID 字典（processed_roi1_peaks.clear()）
5. 重置防抖动滤波器
6. 为每个视频创建新的统计会话

---

## 13. 代码模板

### 13.1 最小化主循环模板

```python
def run_daemon():
    # 初始化
    config = load_fem_config()
    gray_buffer = deque(maxlen=100)
    roi1_gray_buffer = deque(maxlen=100)
    roi3_gray_buffer = deque(maxlen=100)
    frame_index = 0
    interval_seconds = 1.0 / config["roi_capture"]["frame_rate"]

    # 状态变量
    processed_roi1_peaks = {}
    roi1_peak_counter = 0

    while True:
        loop_start = time.time()
        frame_index += 1

        try:
            # 1. 捕获
            screen = capture_screen_or_video()

            # 2. ROI1
            roi1 = extract_roi1(screen)

            # 3. 绿线检测
            intersection = detect_green_line(roi1)

            # 4. ROI2/ROI3
            roi2, roi3 = extract_rois(roi1, intersection)
            roi2_gray = compute_gray(roi2)
            gray_buffer.append(roi2_gray)

            # 5. 波峰检测
            peaks = detect_peaks(list(gray_buffer), threshold)

            # 6. 统计和导出
            statistics.add_peaks(peaks)

            # 7. 可视化（可选）
            if should_save:
                save_images(roi1, roi2, roi3)
                save_waveform(gray_buffer, peaks)

        except Exception as e:
            print(f"Error: {e}")

        # 帧率控制
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, interval_seconds - elapsed)
        time.sleep(sleep_time)
```

---

## 附录 A: 配置参数速查表（完整版）

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `processing_mode` | string | "screen" | 处理模式 |
| `roi_capture.frame_rate` | float | 10 | 帧率 |
| `roi_capture.default_config` | dict | {...} | ROI1 坐标 |
| `roi_capture.roi2_config.extension_params` | dict | {...} | ROI2 扩展参数 |
| `roi_capture.roi3_config.extension_params` | dict | {...} | ROI3 扩展参数 |
| `peak_detection.threshold` | float | 95.0 | ROI2 固定阈值 |
| `peak_detection.threshold_minimum` | float | 80.0 | ROI2 最小阈值 |
| `peak_detection.frame_difference_max` | float | 15.0 | ROI2 异常检测阈值（新增） |
| `peak_detection.adaptive_threshold_enabled` | bool | true | 自适应阈值开关 |
| `peak_detection.difference_threshold` | float | 2.1 | 绿/红分类阈值 |
| `peak_detection.roi3_override.enabled` | bool | true | ROI3 覆盖开关 |
| `peak_detection.roi3_override.threshold` | float | 115.0 | ROI3 覆盖阈值 |
| `peak_detection.roi3_override.early_frame_threshold` | int | 110 | ROI3 早期波峰阈值（新增） |
| `roi1_peak_detection.enabled` | bool | false | ROI1 波峰检测开关 |
| `roi1_peak_detection.min_region_length` | int | 5 | ROI1 波峰最小宽度 |
| `roi2_anti_jitter.enabled` | bool | false | 防抖动开关 |
| `roi2_anti_jitter.movement_threshold` | float | 20.0 | 防抖动运动阈值 |
| `hybrid_detection.enabled` | bool | false | 混合检测开关 |
| `hybrid_detection.roi1_peak_width_range` | [int, int] | [30, 40] | ROI1 波峰宽度范围 |
| `hybrid_detection.data_quality.minimum_roi2_frames` | int | 15 | ROI2 最小帧数 |
| `hybrid_detection.data_quality.roi2_minimum_variance` | float | 0.5 | ROI2 最小方差 |
| `analysis_cache.enabled` | bool | true | 分析缓存开关 |
| `startup_cleanup.enabled` | bool | true | 启动清理开关 |

---

## 附录 B: 关键函数签名

```python
def load_fem_config() -> Dict
def compute_average_gray(image: Image.Image) -> float
def adjust_roi1_to_screen(screen_size: Tuple[int, int], roi_default: Dict) -> Tuple[int, int, int, int]
def compute_roi2_region(roi1_size: Tuple[int, int], center: Tuple[int, int], extension_params: Dict) -> Optional[Tuple[int, int, int, int]]
def manage_threshold_protection(...) -> Tuple[bool, float, int, int, float]
def hybrid_peak_detection(roi1_curve: List[float], roi2_curve: List[float], config: Dict, processed_peaks: Dict, peak_counter: int) -> List[Dict]
def determine_roi2_color_in_interval(peak_start: int, peak_end: int, roi2_curve: List[float], config: Dict) -> Dict
def calculate_roi2_data_quality(peak_start: int, peak_end: int, roi2_curve: List[float]) -> Dict
def discover_video_files(video_path: str) -> List[str]
def initialize_video_capture(video_path: str) -> cv2.VideoCapture
def get_video_frame(video_cap: cv2.VideoCapture, loop_enabled: bool, frame_step: int) -> Optional[Image.Image]
def cleanup_directories() -> None
def _create_video_folders(video_path: str, session_id: str, ...) -> str
def handle_video_switch() -> None
```

---

## 附录 C: 数据流图

```
[Screen/Video]
    ↓
[ROI1 Crop] → [Green Line Detection] → [Intersection Point]
    ↓                                           ↓
[ROI1 Gray Buffer]                         [ROI2 Extract]
    ↓                                           ↓
[ROI1 Peak Detection]                    [ROI2 Gray Buffer]
    ↓                                           ↓
    |                                    [ROI2 Peak Detection]
    |                                           ↓
    └───────────── [Hybrid Detection] ←────────┘
                        ↓
                [SafePeakStatistics]
                        ↓
            [ROI3 Override Logic]
                        ↓
                [CSV Export + Cache]
```

---

## 附录 D: 修复清单

### v2.0 修复的问题

**严重冲突（已修复）：**
1. ✅ 添加了 `frame_difference_max` 配置项（15.0）
2. ✅ 添加了 `roi3_override.early_frame_threshold` 配置项（110）
3. ✅ 完善了 ROI1 自适应阈值更新逻辑（增量更新 vs 替换更新）
4. ✅ 添加了完整的状态变量初始化
5. ✅ 添加了 `roi1_peak_counter` 重置逻辑

**中等冲突（已修复）：**
6. ✅ 添加了 ROI1 阈值保护的说明
7. ✅ 明确了 `roi3_extension_params` 的来源
8. ✅ 修复了 `video_time_str` 作用域问题
9. ✅ 明确了 ROI1/ROI2 波峰宽度参数的独立作用
10. ✅ 添加了混合检测的缓冲区空值检查

**轻微问题（已修复）：**
11. ✅ 说明了 `compute_roi2_region` 是通用函数
12. ✅ 添加了 ROI3 图像保存逻辑
13. ✅ 修正了波峰宽度配置的读取方式
14. ✅ 明确了防抖动参数的默认值和建议值
15. ✅ 说明了 ROI3 覆盖逻辑的应用位置
16. ✅ 保留了 `only_delect` 的拼写（历史兼容）
17. ✅ 说明了 ROI1 阈值的计算时机
18. ✅ 明确了 ROI2 颜色判定参数的来源
19. ✅ 添加了波峰 ID 计算示例
20. ✅ 添加了 ROI3 最大值帧的计算公式

---

**文档版本：** 2.0 (修复版)
**最后更新：** 2025-12-25
**适用于：** simple_roi_daemon.py 重写
**修复问题：** 23 个冲突和问题
**状态：** ✅ 可用于生产环境重写
