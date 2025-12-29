# SimpleFEM 重构版 - 数据流程文档

## 系统架构概览

SimpleFEM 重构版采用模块化管理架构，将原始的单一文件（2500+行）拆分为10个独立的管理器类，每个管理器负责特定的功能域。

### 核心组件

```
Orchestrator (主编排器)
    ├── ConfigManager (配置管理器)
    ├── ROICaptureManager (ROI捕获管理器)
    ├── GreenLineManager (绿线检测管理器)
    ├── HybridDetectionManager (混合检测管理器)
    ├── ThresholdProtectionManager (阈值保护管理器)
    ├── ROI3Statistics (ROI3统计计算器)
    ├── AnalysisCacheManager (分析缓存管理器)
    ├── DataExportManager (数据导出管理器)
    └── StatisticsManager (统计管理器)
```

## 主数据流程

### 1. 系统初始化流程

```
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator.__init__()                                    │
│  ├── 加载配置: ConfigManager(config_path)                   │
│  ├── 初始化管理器:                                           │
│  │   ├── ThresholdProtectionManager                        │
│  │   ├── ROICaptureManager                                  │
│  │   ├── GreenLineManager                                   │
│  │   └── HybridDetectionManager                             │
│  ├── 创建会话ID: datetime.now().strftime("%Y%m%d_%H%M%S")   │
│  └── 初始化导出和统计:                                       │
│      ├── AnalysisCacheManager(export_dir)                   │
│      └── StatisticsManager(config)                          │
└─────────────────────────────────────────────────────────────┘
```

### 2. 帧处理主流程

```
┌────────────────┐
│ 视频帧/屏幕截图  │
└────────┬───────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤1: ROI捕获 (ROICaptureManager)                         │
│  ├── 1.1 计算ROI1平均灰度                                   │
│  │   roi1_gray = compute_average_gray(roi1_image)          │
│  │   roi1_buffer.append(roi1_gray)  # maxlen=100           │
│  │                                                          │
│  ├── 1.2 绿线交点检测 (GreenLineManager)                   │
│  │   intersection = detect_intersection(roi1_image)        │
│  │   # OpenCV HSV滤波 + Canny边缘 + Hough直线变换          │
│  │                                                          │
│  ├── 1.3 提取ROI2/ROI3                                     │
│  │   if intersection is not None:                         │
│  │       roi2_image = extract_roi2(roi1_image, ix, iy)     │
│  │       roi3_image = extract_roi3(roi1_image, ix, iy)     │
│  │                                                          │
│  └── 1.4 计算ROI2/ROI3灰度                                 │
│      roi2_gray = compute_average_gray(roi2_image)          │
│      roi2_buffer.append(roi2_gray)  # maxlen=100           │
│      roi3_gray = compute_average_gray(roi3_image)          │
│      roi3_buffer.append(roi3_gray)  # maxlen=100           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤2: ROI3统计计算 (ROI3Statistics)                        │
│  ├── G1/G2像素百分比计算                                     │
│  │   g1_percent = 像素值∈[80,255]的百分比                   │
│  │   g2_percent = 像素值∈[150,255]的百分比                  │
│  │                                                          │
│  ├── 列灰度差值计算                                         │
│  │   column_diff = abs(左半列均值 - 右半列均值)            │
│  │                                                          │
│  └── 归一化灰度值                                          │
│      normalized_80_160 = clip_to_160(roi3_avg - 80)        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤3: 自适应阈值计算                                        │
│  bg_mean = 增量更新背景均值                                   │
│  if not threshold_protection_active:                        │
│      if roi2_gray < threshold:                              │
│          bg_mean = bg_mean + (roi2_gray - bg_mean) / count  │
│                                                              │
│  threshold_used = max(threshold, bg_mean * (1 + ratio))     │
│  threshold_used = max(threshold_used, threshold_minimum)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤4: 阈值保护更新 (ThresholdProtectionManager)           │
│  ├── 第一次更新（波峰检测前）                                │
│  │   should_protect, _ = update(                           │
│  │       current_gray=roi2_gray,                           │
│  │       current_threshold=threshold,                      │
│  │       has_peaks=False,  # 稍后检测                       │
│  │       frame_time=frame_time,                             │
│  │   )                                                      │
│  │                                                          │
│  └── 触发条件:                                              │
│      ├── 波形触发: gray >= threshold                        │
│      └── 波峰触发: has_peaks=True                           │
│                                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │  保护状态机                                 │           │
│  │  ┌──────────┐                               │           │
│  │  │ 正常状态  │──灰度>=阈值→ [保护激活]       │           │
│  │  └──────────┘                               │           │
│  │       ▲                                     │           │
│  │       │  等待恢复延迟 + 稳定帧数              │           │
│  │       │                                     │           │
│  │  [保护激活]────────────────────────→┐      │           │
│  │                                     │      │           │
│  │  灰度持续<threshold且稳定帧数>=N    │      │           │
│  └─────────────────────────────────────┴──────┘           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤5: 波峰检测                                             │
│  ├── 判断检测模式                                           │
│  │   hybrid_enabled = config.hybrid_detection_enabled      │
│  │   roi1_enabled = config.roi1_peak_detection_enabled     │
│  │   roi1_buffer_len = len(roi1_buffer)                    │
│  │                                                          │
│  ├── 情况1: 混合检测（ROI1时机 + ROI2颜色）                │
│  │   if hybrid_enabled and roi1_enabled and roi1_buffer_len>0:│
│  │       hybrid_green_peaks, hybrid_red_peaks, hybrid_info =│
│  │           HybridDetectionManager.detect_hybrid_peaks(  │
│  │               roi1_curve=roi1_buffer,                   │
│  │               roi2_curve=roi2_buffer,                    │
│  │               frame_index=frame_index,                  │
│  │               roi2_intersection=intersection            │
│  │           )                                              │
│  │       green_peaks = hybrid_green_peaks                  │
│  │       red_peaks = hybrid_red_peaks                      │
│  │                                                          │
│  │   ┌────────────────────────────────────────┐            │
│  │   │ 混合检测内部流程                       │            │
│  │   │ 1. ROI1波峰检测（detect_peaks）        │            │
│  │   │    roi1_peaks = detect_peaks(         │            │
│  │   │        roi1_curve,                     │            │
│  │   │        threshold=roi1_threshold        │            │
│  │   │    )                                   │            │
│  │   │                                        │            │
│  │   │ 2. ROI2颜色判定（每个ROI1波峰）        │            │
│  │   │    for (start, end) in roi1_peaks:     │            │
│  │   │        pre_avg  = mean(roi2[start-N:start])  │     │
│  │   │        post_avg = mean(roi2[end:end+N])      │     │
│  │   │        if post_avg - pre_avg >= diff_threshold:│  │
│  │   │            color = 'green'                      │    │
│  │   │        else:                                   │    │
│  │   │            color = 'red'                         │    │
│  │   │                                        │            │
│  │   │ 3. 生成波峰ID和统计信息                │            │
│  │   │    peak_id = f"roi1_{frame_index}_{start}" │      │
│  │   └────────────────────────────────────────┘            │
│  │                                                          │
│  ├── 情况2: ROI1数据不足，跳过检测                          │
│  │   elif hybrid_enabled and roi1_enabled:                │
│  │       pass  # 保持空列表，等待ROI1缓冲区积累数据        │
│  │                                                          │
│  └── 情况3: ROI2独立检测（传统模式）                        │
│      else:                                                  │
│          green_peaks, red_peaks = detect_peaks(            │
│              roi2_buffer,                                  │
│              threshold=threshold,                          │
│              difference_threshold=difference_threshold,     │
│              margin_frames=margin_frames,                  │
│              silence_frames=silence_frames                 │
│          )                                                 │
│                                                              │
│  ├── 第二次阈值保护更新（基于实际检测结果）                  │
│  │   has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0│
│  │   if has_peaks:                                         │
│  │       threshold_protection.update(has_peaks=True)      │
│  └── 波峰信息                                               │
│      green_count = len(green_peaks)                         │
│      red_count = len(red_peaks)                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤6: 记录分析缓存 (AnalysisCacheManager)                  │
│  cache_payload = {                                          │
│      "frame_index": frame_index,                            │
│      "timestamp": datetime.fromtimestamp(frame_time),       │
│      "roi1_avg": roi1_gray,                                 │
│      "roi2_avg": roi2_gray,                                 │
│      "roi3_avg": roi3_gray,                                 │
│      "intersection": {"x": ix, "y": iy},                    │
│      "threshold": threshold,                                │
│      "green_peaks": green_peaks,                            │
│      "red_peaks": red_peaks,                                │
│      "roi3_g1_percent": g1_percent,                         │
│      "roi3_g2_percent": g2_percent,                         │
│      "roi3_column_diff": column_diff,                       │
│      "hybrid_detection_enabled": hybrid_enabled,            │
│      "hybrid_green_peaks": len(hybrid_green_peaks),         │
│      "hybrid_red_peaks": len(hybrid_red_peaks),             │
│      "protection_active": should_protect                    │
│  }                                                          │
│  analysis_cache.record_frame(cache_payload)                │
│  # 写入: export/roi_analysis_cache_{session}_{run_id}.jsonl│
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤7: 保存图像和波形 (DataExportManager)                   │
│  ├── 保存ROI1/ROI2/ROI3图像                                  │
│  │   data_export.save_roi1(roi1_image, frame_index, video_time)│
│  │   data_export.save_roi2(roi2_image, frame_index)         │
│  │   data_export.save_roi3(roi3_image, frame_index)         │
│  │                                                          │
│  ├── 保存ROI1波形图                                         │
│  │   if has_peaks or save_all:                             │
│  │       data_export.save_roi1_waveform(                   │
│  │           roi1_buffer, threshold, frame_index           │
│  │       )                                                 │
│  │                                                          │
│  └── 保存ROI2波形图（带波峰标注）                            │
│      if has_peaks:                                         │
│          data_export.save_waveform(                         │
│              roi2_buffer, green_peaks, red_peaks,           │
│              threshold, frame_index, video_time,            │
│              roi2_image_path                                │
│          )                                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤8: 添加到统计 (StatisticsManager)                       │
│  if has_peaks:                                             │
│      # 准备混合检测结果                                     │
│      hybrid_peaks_for_stats = hybrid_info if (             │
│          hybrid_enabled and roi1_enabled and               │
│          len(roi1_buffer) > 0                              │
│      ) else []                                              │
│                                                              │
│      statistics.add_peaks(                                  │
│          frame_index=frame_index,                           │
│          green_peaks=green_peaks,                           │
│          red_peaks=red_peaks,                               │
│          curve_data=roi2_buffer,                            │
│          intersection=intersection,                         │
│          roi2_info=roi2_info,                               │
│          gray_value=roi2_gray,                              │
│          threshold_used=threshold,                          │
│          bg_mean=bg_mean,                                   │
│          roi3_curve=roi3_buffer,                            │
│          hybrid_enabled=hybrid_enabled,                     │
│          hybrid_peaks=hybrid_peaks_for_stats                │
│      )                                                     │
│                                                              │
│  ┌────────────────────────────────────────┐               │
│  │ 统计去重逻辑 (StatisticsManager)       │               │
│  │ 1. 最近波峰比较（5帧窗口）            │               │
│  │ 2. 连续帧去重（40帧窗口）            │               │
│  │ 3. 跨颜色去重（绿色优先级=2）        │               │
│  │ 4. 无效数据过滤（zero avg等）        │               │
│  │                                        │               │
│  │ 输出: export/peak_statistics_*.csv   │               │
│  └────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 关键数据结构

### 1. 缓冲区（Circular Buffers）

```python
# ROI1 缓冲区 - 用于混合检测的波峰时机检测
roi1_buffer: deque[float, maxlen=100]

# ROI2 缓冲区 - 用于ROI2独立波峰检测和颜色判定
roi2_buffer: deque[float, maxlen=100]

# ROI3 缓冲区 - 用于ROI3统计和覆盖判定
roi3_buffer: deque[float, maxlen=100]

# ROI3 G1/G2 缓冲区 - 用于波峰区间的G1/G2覆盖判定
roi3_g1_buffer: deque[float, maxlen=100]
roi3_g2_buffer: deque[float, maxlen=100]
roi3_column_diff_buffer: deque[float, maxlen=100]
```

### 2. 波峰数据结构

```python
# ROI2独立检测的波峰表示
green_peaks: List[Tuple[int, int]]  # [(start, end), ...]
red_peaks: List[Tuple[int, int]]

# 混合检测波峰信息
hybrid_info: List[Dict[str, Any]] = [
    {
        "roi1_peak_id": "roi1_100_40",        # ROI1波峰唯一ID
        "peak_interval": (40, 59),            # 波峰区间
        "color": "green",                     # 颜色判定
        "detection_method": "hybrid_roi1_roi2",# 检测方法
        "quality_score": 0.85,               # 质量评分
        "confidence": 0.92,                   # 置信度
        "roi1_peak_max": 150.0,              # ROI1峰值
        "roi2_pre_avg": 75.0,                 # ROI2波峰前均值
        "roi2_post_avg": 85.0,                # ROI2波峰后均值
        "roi2_frame_diff": 10.0,              # ROI2前后帧差
        "roi1_start_frame": 90,               # ROI1波峰起始帧
        "roi1_end_frame": 109                 # ROI1波峰结束帧
    }
]
```

### 3. 统计数据格式

```python
# CSV导出格式（export/peak_statistics_*.csv）
{
    "peak_type": "green",           # 波峰类型
    "frame_index": 100,             # 帧索引
    "pre_peak_avg": 75.0,           # 波峰前均值
    "post_peak_avg": 85.0,          # 波峰后均值
    "frame_diff": 10.0,             # 前后帧差
    "difference_threshold_used": 1.8, # 使用的差值阈值
    "threshold_used": 95.0,         # 使用的检测阈值
    "bg_mean": 80.0,                 # 背景均值
    "peak_max_value": 120.0,        # 波峰最大值
    "roi3_peak_max_value": 135.0,    # ROI3波峰最大值
    "roi3_peak_max_frame": 95,      # ROI3波峰最大帧
    "pre_peak_frame_start": 90,     # 波峰前区间起始
    "pre_peak_frame_end": 94,       # 波峰前区间结束
    "post_peak_frame_start": 96,    # 波峰后区间起始
    "post_peak_frame_end": 109,     # 波峰后区间结束
    "roi1_peak_id": "roi1_100_40",  # ROI1波峰ID（混合检测）
    "g1_value": 98.5,               # G1值
    "g2_value": 25.3,               # G2值
    "g1_g2_override_applied": True, # G1/G2覆盖是否应用
    "column_diff_value": 18.5,      # 列差值
    "column_diff_override_applied": False, # 列差覆盖是否应用
    ...
}
```

## 状态机模型

### 1. 阈值保护状态机

```
┌──────────────┐
│  INACTIVE    │ ← 正常状态，背景均值正常更新
└──────┬───────┘
       │
       │ 触发条件:
       │ - 波形触发: gray >= threshold
       │ - 波峰触发: has_peaks = True
       │
       ▼
┌──────────────┐
│   ACTIVE     │ ← 保护激活，背景均值冻结
└──────┬───────┘
       │
       │ 解除条件:
       │ - 等待恢复延迟 (recovery_delay_seconds)
       │ - 连续稳定帧数 >= stability_frames
       │ - 灰度 < threshold
       │
       ▼
┌──────────────┐
│  RECOVERING  │ ← 恢复中，监控稳定性
└──────┬───────┘
       │
       │ 稳定帧数足够
       │
       ▼
┌──────────────┐
│  INACTIVE    │ ← 回到正常状态
└──────────────┘
```

### 2. 波峰检测状态机（三态逻辑）

```
┌─────────────────────────────────────┐
│  检测模式选择                        │
└────────┬────────────────────────────┘
         │
         ├─ hybrid_enabled=True AND roi1_enabled=True AND len(roi1_buffer)>0
         │   │
         │   └─► [情况1: 混合检测模式]
         │       ├── ROI1检测波峰时机
         │       ├── ROI2判定颜色
         │       └── 生成详细波峰信息
         │
         ├─ hybrid_enabled=True AND roi1_enabled=True AND len(roi1_buffer)==0
         │   │
         │   └─► [情况2: ROI1数据不足]
         │       └── 跳过检测，等待ROI1缓冲区积累
         │
         └─ hybrid_enabled=False OR roi1_enabled=False
             │
             └─► [情况3: ROI2独立检测模式]
                 ├── ROI2自适应阈值检测
                 ├── ROI2颜色判定
                 └── 简单波峰区间表示
```

**状态说明**:

1. **混合检测模式**: ROI1检测波峰时机 + ROI2判定颜色，生成丰富的波峰元数据
2. **ROI1数据不足**: ROI1缓冲区为空，跳过波峰检测（不回退到ROI2检测）
3. **ROI2独立检测**: 传统的ROI2自适应阈值检测，简单波峰区间表示

## 多视频处理流程

```
┌──────────────────────────────────────────────────────────┐
│  批量视频处理流程 (video_processing mode)                │
│                                                          │
│  ┌────────────────────────────────────────┐             │
│  │ 1. 扫描视频文件                      │             │
│  │    video_path = "video/"             │             │
│  │    video_files = list_mp4_files(video_path)        │
│  └────────────┬───────────────────────────┘             │
│               │                                           │
│               ▼                                           │
│  ┌────────────────────────────────────────┐             │
│  │ 2. 遍历视频文件                      │             │
│  │    for i, video_file in enumerate(video_files):    │
│  │        # 每个视频独立会话                        │             │
│  │        session_id = sanitize_name(video_file)      │
│  └────────────┬───────────────────────────┘             │
│               │                                           │
│               ▼                                           │
│  ┌────────────────────────────────────────┐             │
│  │ 3. 处理单个视频                      │             │
│  │    statistics.initialize_for_video(                  │
│  │        video_name,                               │             │
│  │        is_batch=True                             │             │
│  │    )                                            │             │
│  │                                                  │             │
│  │    cap = cv2.VideoCapture(video_file)            │             │
│  │    while cap.isOpened():                        │             │
│  │        ret, frame = cap.read()                   │             │
│  │        if not ret: break                          │             │
│  │                                                  │             │
│  │        # 帧处理流程（见上文）                    │             │
│  │        orchestrator._process_frame(frame, ...)    │             │
│  │                                                  │             │
│  │    cap.release()                                 │             │
│  └────────────┬───────────────────────────┘             │
│               │                                           │
│               ▼                                           │
│  ┌────────────────────────────────────────┐             │
│  │ 4. 保存统计数据                      │             │
│  │    statistics.finalize_video()                   │             │
│  │    # 生成: export/peak_statistics_{video_name}_{timestamp}.csv│
│  └────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

## 数据持久化

### 1. 文件输出结构

```
SimpleFEM/
├── export/
│   ├── peak_statistics_{video_name}_{timestamp}.csv
│   │   └── 波峰统计数据（去重后）
│   │
│   ├── roi_analysis_cache_{session_id}_{run_id}.jsonl
│   │   └── 每帧分析缓存（调试用）
│   │       ├── {type: "meta", ...}       # 会话元数据
│   │       ├── {type: "frame", ...}      # 帧数据
│   │       └── {type: "session_end", ...}# 会话结束标记
│   │
│   └── hybrid_peak_statistics_{session_id}.csv
│       └── 混合检测详细统计（如果启用）
│
├── tmp/{video_name}/
│   ├── roi1/
│   │   └── roi1_{frame_index:06d}.png
│   ├── roi2/
│   │   └── roi2_{frame_index:06d}.png
│   ├── roi3/
│   │   └── roi3_{frame_index:06d}.png
│   ├── wave/
│   │   ├── wave_{frame_index:06d}.png
│   │   └── roi1_wave_{frame_index:06d}.png
│   └── wave1/
│       └── roi1_wave_{frame_index:06d}.png
│
└── logs/
    └── roi_peak_daemon_{YYYY-MM-DD}.log
```

### 2. JSONL 缓存格式

```jsonl
{"type":"meta","cache_version":1,"created_at":"2025-12-28T14:55:24","session_id":"20251228_145523",...}
{"type":"frame","frame_index":0,"timestamp":"2025-12-28T14:55:24.625395","roi1_avg":30.77,"roi2_avg":81.11,"threshold":89.22,"green_peaks":[],"red_peaks":[],...}
{"type":"frame","frame_index":1,"timestamp":"2025-12-28T14:55:24.817426",...}
...
{"type":"session_end","ended_at":"2025-12-28T14:58:30","reason":"video_complete"}
```

## 性能考虑

### 1. 内存管理

- **固定大小缓冲区**: 使用 `deque(maxlen=100)` 防止内存无限增长
- **增量统计**: 背景均值使用增量更新算法，避免存储全部历史数据
- **及时释放**: 帧处理完成后立即释放图像资源

### 2. 处理性能

- **帧率控制**: `config.frame_rate` 控制处理间隔（1-30 FPS）
- **批量处理**: 多视频顺序处理，自动切换会话
- **异步缓存**: 分析缓存每50帧刷新一次，减少I/O开销

### 3. 并发安全

- **线程本地数据**: 每个管理器维护独立状态
- **原子写入**: CSV导出使用临时文件+重命名确保原子性
- **文件锁**: 缓存写入使用文件锁避免并发冲突

## 错误处理策略

### 1. 组件级别

```python
# 每个管理器都有独立的错误处理
try:
    result = manager.process(data)
except Exception as e:
    logging.error(f"Manager {name} failed: {e}")
    # 优雅降级，不影响其他组件
    return fallback_value
```

### 2. 帧级别

```python
# 单帧处理失败不影响后续帧
try:
    orchestrator._process_frame(frame, ...)
except Exception as e:
    logging.error(f"Frame {frame_index} processing failed: {e}")
    # 继续处理下一帧
    continue
```

### 3. 会话级别

```python
# 视频处理失败不影响其他视频
try:
    orchestrator._run_video_mode()
except Exception as e:
    logging.error(f"Video processing failed: {e}")
    # 清理资源，处理下一个视频
    orchestrator._cleanup_resources()
```

## 总结

SimpleFEM 重构版通过模块化架构实现了：

1. **清晰的职责分离**: 每个管理器专注于特定功能域
2. **完整的检测逻辑**: 实现了三态波峰检测（混合检测、ROI1数据不足、ROI2独立检测）
3. **可靠的数据持久化**: 多层数据导出，包括CSV、JSONL缓存和图像
4. **健壮的错误处理**: 组件级、帧级、会话级的错误隔离
5. **完整的可追溯性**: 分析缓存记录每帧详细信息，便于调试和验证

数据流程从视频输入开始，经过ROI捕获、绿线检测、波峰检测（三态逻辑）、统计分析等多个阶段，最终输出结构化的统计数据和可视化图像，形成完整的医疗信号处理流水线。

## 与原始代码的关键差异

### 波峰检测逻辑

**原始代码** (`simple_roi_daemon.py`):
- 实现了完整的三态逻辑
- 情况1: 混合检测（ROI1数据充足）
- 情况2: ROI1数据不足，跳过检测
- 情况3: ROI2独立检测

**重构代码** (`refactor/orchestrator.py`):
- 完全复现了原始代码的三态逻辑
- 代码结构更清晰，职责分离更明确
- 混合检测结果正确传递给统计模块

### 混合检测结果传递

**原始代码**:
```python
stats_write_results = current_stats.add_peaks_from_daemon(
    ...
    hybrid_peaks=hybrid_peaks,  # 传递实际的混合检测结果
    ...
)
```

**重构代码**:
```python
# 准备混合检测结果
hybrid_peaks_for_stats = hybrid_info if (
    hybrid_enabled and roi1_enabled and
    len(roi1_buffer) > 0
) else []

self._statistics.add_peaks(
    ...
    hybrid_peaks=hybrid_peaks_for_stats,  # 传递实际的混合检测结果
    ...
)
```

这确保了混合检测的详细信息（ROI1波峰ID、检测方法、质量评分等）不会丢失。
