# SimpleFEM Simple ROI Daemon - 重写技术文档

本文档提供完整的技术规范，可根据此文档重写 `simple_roi_daemon.py` 而无需查看原始源代码。

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
    "threshold_protection": {
      "enabled": false,
      "recovery_delay_seconds": 1.0,
      "stability_frames": 5,
      "waveform_trigger_enabled": true
    },
    "roi3_override": {
      "enabled": true,
      "threshold": 115.0,
      "require_roi3_data": true
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
    "algorithm": "ema",  // "ema" 或 "threshold"
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

---

## 4. 主处理流程 (run_daemon)

### 4.1 初始化阶段

```python
def run_daemon():
    # 1. 清理旧数据
    cleanup_directories()

    # 2. 加载配置
    config = load_fem_config()

    # 3. 初始化防抖动滤波器（可选）
    if anti_jitter_config.get("enabled"):
        if algorithm == "ema":
            intersection_filter = IntersectionFilter(
                alpha=0.25,
                movement_threshold=20.0,
                initialization_frames=3,
                stability_threshold=8.0
            )
        else:  # threshold
            intersection_filter = ThresholdIntersectionFilter(
                movement_threshold=20.0,
                initialization_frames=3
            )

    # 4. 初始化分析缓存
    analysis_cache = RoiAnalysisCache(
        export_dir="export",
        enabled=True,
        flush_every=50
    )

    # 5. 初始化统计管理器
    statistics_manager = VideoStatisticsManager()

    # 6. 根据模式初始化视频/屏幕
    if processing_mode == "video":
        video_files = discover_video_files(video_path)
        video_cap = initialize_video_capture(video_files[0])
        statistics_manager.initialize_for_video(video_files[0], is_batch=True)
    else:
        statistics_manager.initialize_for_video(None, is_batch=False)
```

### 4.2 主循环

```python
while True:
    loop_start = time.time()
    frame_index += 1

    # ========== 步骤1: 捕获图像 ==========
    if processing_mode == "video":
        screen = get_video_frame(video_cap, loop_enabled, frame_step)
        if screen is None:
            # 视频结束，切换到下一个视频或退出
            handle_video_switch()
    else:
        screen = ImageGrab.grab()

    # ========== 步骤2: 提取 ROI1 ==========
    x1, y1, x2, y2 = adjust_roi1_to_screen(screen.size, roi_default)
    roi1_image = screen.crop((x1, y1, x2, y2))

    # ========== 步骤3: 检测绿线交点 ==========
    roi_cv_image = cv2.cvtColor(np.array(roi1_image), cv2.COLOR_RGB2BGR)
    intersection = detect_green_intersection(
        roi_cv_image,
        anti_jitter_config,
        intersection_filter
    )

    # 使用最后已知交点或 ROI1 中心作为回退
    if intersection is not None:
        last_intersection_roi = intersection
    if last_intersection_roi is None:
        center_x, center_y = roi1_width // 2, roi1_height // 2
    else:
        center_x, center_y = last_intersection_roi

    # ========== 步骤4: 计算 ROI2 和 ROI3 ==========
    roi2_region = compute_roi2_region(
        (roi1_width, roi1_height),
        (center_x, center_y),
        extension_params
    )

    if roi2_region is not None:
        rx1, ry1, rx2, ry2 = roi2_region
        roi2_image = roi1_image.crop((rx1, ry1, rx2, ry2))
        roi2_gray = compute_average_gray(roi2_image)
        gray_buffer.append(roi2_gray)

    # ROI3 提取（类似 ROI2）
    if roi3_extension_params:
        roi3_region = compute_roi2_region(
            (roi1_width, roi1_height),
            (center_x, center_y),
            roi3_extension_params
        )
        if roi3_region is not None:
            roi3_image = roi1_image.crop(roi3_region)
            roi3_gray = compute_average_gray(roi3_image)
            roi3_gray_buffer.append(roi3_gray)

    # ROI1 灰度计算（如果启用）
    if roi1_enabled:
        roi1_gray = compute_average_gray(roi1_image)
        roi1_gray_buffer.append(roi1_gray)

    # ========== 步骤5: 自适应阈值计算 ==========
    threshold_used = max(threshold, threshold_minimum)

    if adaptive_threshold_enabled and len(gray_buffer) >= adaptive_window_frames:
        # 计算背景均值
        recent_frames = list(gray_buffer)[-adaptive_window_frames:]
        calculated_bg_mean = sum(recent_frames) / len(recent_frames)

        # 检查阈值保护状态
        if not threshold_protection_active:
            bg_mean = calculated_bg_mean
            bg_count = len(recent_frames)
            threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
            threshold_used = max(threshold_used, threshold_minimum)

    # ========== 步骤6: 波峰检测 ==========
    green_peaks = []
    red_peaks = []

    if hybrid_enabled and roi1_enabled and len(roi1_gray_buffer) > 0:
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
            # ... 更多配置
        }

        hybrid_peaks = hybrid_peak_detection(
            roi1_curve,
            roi2_curve,
            hybrid_config,
            processed_roi1_peaks,
            roi1_peak_counter
        )

        # 转换为传统格式
        for peak in hybrid_peaks:
            if peak['color'] == 'green':
                green_peaks.append(peak['peak_interval'])
            else:
                red_peaks.append(peak['peak_interval'])

    else:
        # 传统 ROI2 检测模式
        if gray_buffer:
            curve = list(gray_buffer)
            green_peaks_raw, red_peaks_raw = detect_peaks(
                curve,
                threshold=threshold_used,
                marginFrames=margin_frames,
                differenceThreshold=diff_threshold,
                silenceFrames=silence_frames,
                avgFrames=pre_post_avg_frames
            )

            # 应用最小宽度过滤
            green_peaks = [(s, e) for s, e in green_peaks_raw
                          if (e - s + 1) >= min_region_length]
            red_peaks = [(s, e) for s, e in red_peaks_raw
                        if (e - s + 1) >= min_region_length]

    # ========== 步骤7: 添加到统计 ==========
    current_stats = statistics_manager.current_statistics
    stats_write_results = current_stats.add_peaks_from_daemon(
        frame_index=frame_index,
        green_peaks=green_peaks,
        red_peaks=red_peaks,
        curve=list(gray_buffer),
        intersection=last_intersection_roi,
        roi2_info=roi2_info,
        gray_value=roi2_gray,
        difference_threshold=diff_threshold,
        pre_post_avg_frames=pre_post_avg_frames,
        threshold_used=threshold_used,
        bg_mean=bg_mean if bg_count > 0 else None,
        # 混合检测参数
        hybrid_enabled=hybrid_enabled,
        hybrid_peaks=hybrid_peaks,
        roi1_curve=list(roi1_gray_buffer),
        roi1_threshold_used=roi1_threshold_used,
        # ROI3 参数
        roi3_curve=list(roi3_gray_buffer),
        roi3_override_enabled=roi3_override_enabled,
        roi3_override_threshold=roi3_override_threshold
    )

    # ========== 步骤8: 保存图像和波形 ==========
    has_peak = len(green_peaks) > 0 or len(red_peaks) > 0
    should_save = (not only_delect) or has_peak

    if should_save:
        # 保存 ROI1
        if save_roi1:
            roi1_path = f"{roi1_dir}/roi1_{frame_index:06d}.png"
            roi1_image.save(roi1_path)

        # 保存 ROI2
        if save_roi2 and roi2_image is not None:
            roi2_path = f"{roi2_dir}/roi2_{frame_index:06d}.png"
            roi2_image.save(roi2_path)

        # 保存 ROI3
        if save_roi3 and roi3_image is not None:
            roi3_path = f"{roi3_dir}/roi3_{frame_index:06d}.png"
            roi3_image.save(roi3_path)

        # 保存 ROI2 波形图
        if save_wave and gray_buffer:
            save_waveform_plot(
                curve=list(gray_buffer),
                green_peaks=green_peaks,
                red_peaks=red_peaks,
                bg_mean=bg_mean,
                threshold_used=threshold_used,
                output_path=f"{wave_dir}/wave_{frame_index:06d}.png"
            )

    # ========== 步骤9: 写入分析缓存 ==========
    analysis_cache.record_frame({
        "frame_index": frame_index,
        "roi2_gray": roi2_gray,
        "threshold": {...},
        "peaks": {...},
        # ... 更多字段
    })

    # ========== 步骤10: 帧率控制 ==========
    elapsed = time.time() - loop_start
    sleep_time = max(0.0, interval_seconds - elapsed)
    time.sleep(sleep_time)
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

### 5.2 ROI2 颜色判定 (determine_roi2_color_in_interval)

**计算步骤：**
1. 计算波峰前的平均值（默认前5帧）
2. 计算波峰后的平均值（默认后10帧）
3. 计算前后差异：`frame_difference = post_avg - pre_avg`
4. 异常过滤：如果 `|frame_difference| > 15`，判定为错误数据
5. 颜色分类：
   - 绿色：`frame_difference >= color_threshold`
   - 红色：`frame_difference < color_threshold`
6. 数据质量检查：
   - 最小帧数检查（默认15帧）
   - 方差检查（默认最小0.5）
   - 灰度范围检查（5-250）

**返回格式：**
```python
{
    'color': 'green' | 'red',
    'method': 'roi2' | 'roi1_fallback' | 'error_filtered',
    'frame_difference': float,
    'threshold': float,
    'pre_avg': float,
    'post_avg': float,
    'confidence': float,
    'roi2_valid': bool,
    'quality_score': float,
    'variance': float,
    'data_range': float
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
peak_key = buffer_start_frame_index + peak_start + local_max_offset
peak_id = f"ROI1_MAX_{peak_key:06d}"
```

**防重复机制：**
```python
processed_roi1_peaks: Dict[int, str] = {}  # {peak_key: peak_id}
# 检查：
if peak_key in processed_roi1_peaks:
    continue  # 已处理过，跳过
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
- EMA 平滑：`filtered = alpha * new + (1-alpha) * old`
- 速度过滤：拒绝过快的移动
- 阈值式：小于阈值的移动完全忽略

---

## 6. 视频处理

### 6.1 视频发现 (discover_video_files)

```python
def discover_video_files(video_path: str) -> List[str]:
    """
    支持的视频格式：
    - .mp4, .avi, .mov, .mkv, .flv, .wmv
    - 按文件名排序
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(video_path, f"*{ext}")))
    return sorted(files)
```

### 6.2 视频切换逻辑

```python
def handle_video_switch():
    global current_video_index, video_cap, frame_index

    # 释放当前视频
    video_cap.release()

    # 移动到下一个视频
    current_video_index += 1

    if current_video_index < len(video_files):
        # 重置状态
        gray_buffer.clear()
        roi1_gray_buffer.clear()
        roi3_gray_buffer.clear()
        processed_roi1_peaks.clear()

        bg_count = 0
        bg_mean = 0.0
        frame_index = 0

        # 重置防抖动滤波器
        intersection_filter.reset()

        # 初始化新视频统计
        statistics_manager.initialize_for_video(
            video_files[current_video_index],
            is_batch=True
        )

        # 打开新视频
        video_cap = initialize_video_capture(video_files[current_video_index])

        # 重新计算帧率参数
        video_fps = _get_video_fps(video_cap)
        video_frame_step = int(video_fps / roi_frame_rate)

    else:
        # 所有视频处理完毕
        break
```

### 6.3 帧率控制

```python
# 视频模式：
video_fps = _get_video_fps(video_cap)
effective_frame_rate = min(roi_frame_rate, video_fps)
video_frame_step = max(1, int(video_fps / effective_frame_rate))

# 屏幕模式：
effective_frame_rate = roi_frame_rate

# 循环间隔：
interval_seconds = 1.0 / effective_frame_rate

# 主循环中：
step = 1 if first_video_frame else video_frame_step
screen = get_video_frame(video_cap, loop_enabled, frame_step=step)
```

---

## 7. 数据导出

### 7.1 CSV 导出格式

**文件命名：** `peak_statistics_{video_name}_{timestamp}.csv`

**字段列表：**
```csv
timestamp,frame_index,peak_type,peak_start,peak_end,width,roi1_frame_diff,roi2_frame_diff,
pre_peak_avg,post_peak_avg,difference_threshold_used,threshold_used,bg_mean,
peak_max_value,roi3_peak_max_value,roi3_peak_max_frame,
pre_peak_frame_start,pre_peak_frame_end,post_peak_frame_start,post_peak_frame_end,
roi1_peak_id,roi1_detection_method,roi2_color_method,
intersection_x,intersection_y,roi2_x1,roi2_y1,roi2_x2,roi2_y2,roi2_width,roi2_height,
roi3_override_applied,roi3_override_threshold
```

### 7.2 分析缓存格式

**文件格式：** JSONL（每行一个 JSON 对象）

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
  "frame_index": 93,
  "roi2_gray": 105.5,
  "buffer": {...},
  "threshold": {...},
  "peaks": {...}
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
2025-12-25T00:23:19 gray=105.5 green_peaks=1 red_peaks=0 last_green=[45,50]
2025-12-25T00:23:20 gray=98.2 green_peaks=0 red_peaks=1 last_red=[55,60]
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

1. 基础阈值：`threshold`（配置文件）
2. 最小阈值保护：`max(threshold, threshold_minimum)`
3. 自适应计算：`bg_mean * (1.0 + threshold_over_mean_ratio)`
4. 最终阈值：`max(adaptive_threshold, threshold_minimum)`

### 9.3 ROI3 覆盖逻辑

**条件：**
- `roi3_override_enabled == true`
- `roi3_curve` 不为空（或 `require_roi3_data == false`）

**规则：**
1. RED -> GREEN：如果 `roi3_peak_max_value > roi3_override_threshold`
2. 任何颜色 -> RED：如果 `roi3_peak_max_frame < 110`

### 9.4 防抖动策略

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
    filtered = filtered  # 不更新
else:
    # 接受新位置
    filtered = new_position
```

### 9.5 边界检查

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

### 9.6 错误处理原则

1. **守护进程永不崩溃**：所有异常必须捕获
2. **单帧失败不影响后续处理**：使用 try-except 包裹关键代码
3. **记录错误但继续运行**：使用 print/logging 记录错误
4. **优雅降级**：如果绿线检测失败，使用上一次的位置

### 9.7 性能考虑

1. **帧率精度**：使用 `time.time()` 计算精确间隔
2. **视频跳帧**：使用 `frame_step` 控制采样率
3. **内存管理**：固定大小循环缓冲区防止内存泄漏
4. **图像保存**：只在必要时保存（`only_delect` 模式）
5. **缓存刷新**：每 N 帧刷新一次缓存文件

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

### 10.3 可选模块

```python
# 静脉检测模式
from auto_vein_detector import AutoVeinDetector

# 改进的波峰检测
from improved_peak_detection import ImprovedPeakDetection
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
2. 增大 `movement_threshold`（如 30-40）
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
1. 确保 `reset_video_state_variables()` 正确重置所有状态
2. 清空所有缓冲区（gray_buffer, roi1_gray_buffer, roi3_gray_buffer）
3. 重置波峰 ID 管理（processed_roi1_peaks）
4. 重置防抖动滤波器（intersection_filter.reset()）
5. 为每个视频创建新的统计会话

---

## 13. 代码模板

### 13.1 最小化主循环模板

```python
def run_daemon():
    # 初始化
    config = load_fem_config()
    gray_buffer = deque(maxlen=100)
    frame_index = 0
    interval_seconds = 1.0 / config["roi_capture"]["frame_rate"]

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

            # 4. ROI2
            roi2 = extract_roi2(roi1, intersection)
            roi2_gray = compute_gray(roi2)
            gray_buffer.append(roi2_gray)

            # 5. 波峰检测
            peaks = detect_peaks(list(gray_buffer), threshold)

            # 6. 统计和导出
            statistics.add_peaks(peaks)

            # 7. 可视化（可选）
            if should_save:
                save_images(roi1, roi2)
                save_waveform(gray_buffer, peaks)

        except Exception as e:
            print(f"Error: {e}")

        # 帧率控制
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, interval_seconds - elapsed)
        time.sleep(sleep_time)
```

---

## 附录 A: 配置参数速查表

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `processing_mode` | string | "screen" | 处理模式 |
| `roi_capture.frame_rate` | float | 10 | 帧率 |
| `roi_capture.default_config` | dict | {...} | ROI1 坐标 |
| `roi_capture.roi2_config.extension_params` | dict | {...} | ROI2 扩展参数 |
| `peak_detection.threshold` | float | 95.0 | ROI2 固定阈值 |
| `peak_detection.threshold_minimum` | float | 80.0 | ROI2 最小阈值 |
| `peak_detection.adaptive_threshold_enabled` | bool | true | 自适应阈值开关 |
| `peak_detection.difference_threshold` | float | 2.1 | 绿/红分类阈值 |
| `roi2_anti_jitter.enabled` | bool | false | 防抖动开关 |
| `hybrid_detection.enabled` | bool | false | 混合检测开关 |

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
    ↑                                           ↓
    |                                    [ROI2 Peak Detection]
    |                                           ↓
    └───────────── [Hybrid Detection] ←────────┘
                        ↓
                [SafePeakStatistics]
                        ↓
                [CSV Export + Cache]
```

---

**文档版本：** 1.0
**最后更新：** 2025-12-25
**适用于：** simple_roi_daemon.py 重写
