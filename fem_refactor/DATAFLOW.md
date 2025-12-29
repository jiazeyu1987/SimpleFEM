# SimpleFEM ROI Daemon Dataflow（数据流）

本文描述 `simple_roi_daemon.py` 的运行数据流，目标是为后续“管理类拆分”提供边界依据；**所有判定逻辑保持不变**，仅做结构重组。

## 0. 输入/输出总览

**输入**
- 配置：`simple_fem_config.json`
- 输入源（按 `processing_mode`）：
  - `screen`：屏幕截取（`PIL.ImageGrab`）
  - `video`：视频文件/目录（OpenCV `cv2.VideoCapture`）

**主要输出**
- 日志：`logs/simple_roi_daemon_YYYYMMDD_HHMMSS.log`、`logs/roi_peak_daemon.log`
- 统计 CSV：`export/peak_statistics_*.csv`
- 临时图像/波形：`tmp/<session>/roi1|roi2|roi3|wave|wave1/...`
- 分析缓存（JSONL）：`export/roi_analysis_cache_<session>_<runid>.jsonl`

## 1. 启动阶段（Startup）

1) `setup_logging()`：初始化日志系统（全局 logging + rotating handler 等）
2) `cleanup_directories()`：按配置清理 `export/ tmp/ logs/` 等目录（如果启用）
3) `load_fem_config()`：读取 `simple_fem_config.json`

**关键副作用**
- 决定本次运行参数、模式与阈值等策略开关
- 清理可能删除上一轮产物（包括 peak_statistics CSV）

## 2. 模式初始化（Processing Mode）

按 `processing_mode` 分支：

### 2.1 screen 模式
- 初始化统计：`statistics_manager.initialize_for_video(None, is_batch=False)`

### 2.2 video 模式
- 解析 `video_processing.video_path`：
  - 如果是文件：单文件处理
  - 如果是目录：`discover_video_files()` 扫描并排序
- 打开首个视频：`initialize_video_capture(video_files[0])`
- 初始化统计：`statistics_manager.initialize_for_video(video_files[0], is_batch=True)`

**关键副作用**
- `video_cap` 的帧指针、FPS 读取与帧步长（`video_frame_step`）影响后续采样节奏

## 3. 防抖与交点（Intersection + Anti-jitter）

### 3.1 anti-jitter 初始化
- 读取 `config["roi2_anti_jitter"]`
- 构造滤波器实例：
  - `ema`：`green_detector.IntersectionFilter`
  - `threshold`：`threshold_based_anti_jitter.ThresholdIntersectionFilter`

### 3.2 每帧交点检测
每帧将 ROI1（RGB）转为 OpenCV BGR：
- `detect_green_intersection(roi_cv_image, anti_jitter_config, intersection_filter)`
  - 返回 ROI1 局部坐标系内 (x,y)，可能经过滤波
  - 失败时 `intersection=None`，并可重置滤波器

**回退策略**
- 若本帧检测失败但已有历史：使用 `last_intersection_roi`
- 若从未成功：使用 ROI1 中心作为交点（保证 ROI2/ROI3 能继续工作）

## 4. ROI 生成与信号提取（ROI1/ROI2/ROI3）

### 4.1 ROI1
- ROI1 坐标（屏幕/视频帧全局坐标系）来自 `roi_capture.default_config`
- `adjust_roi1_to_screen()` 保证 ROI1 在画面内
- `roi1_image = screen.crop((x1,y1,x2,y2))`

### 4.2 ROI2（围绕交点）
- 使用 `compute_roi2_region((roi1_w,roi1_h),(center_x,center_y),roi2_extension_params)`
- `roi2_image = roi1_image.crop(roi2_region)`
- `roi2_gray = compute_average_gray(roi2_image)`
- 更新 `gray_buffer`（deque maxlen=100，创建/重置见 `fem_refactor/signal_buffers.py`）

### 4.3 ROI3（围绕交点）
若启用 ROI3 配置：
- 同样用 `compute_roi2_region()` 得到 ROI3 区域
- 计算 ROI3 指标并写入独立 buffer：
  - `roi3_gray`
  - `g1/g2`（灰度直方图区间占比）
  - `column_diff`（列均值最大-最小差）

> 注：ROI1/ROI2/ROI3 的所有 deque buffer 统一由 `create_signal_buffers()` 创建，切换视频时相关重置逻辑集中在 `reset_video_state_variables()` / `reset_roi1_state()`。

## 5. 阈值计算与保护（Adaptive Threshold + Protection）

### 5.1 固定阈值
来自 `peak_detection.threshold`，并有 `threshold_minimum` 下限。

### 5.2 自适应阈值（如果启用）
- 窗口：`adaptive_window_seconds` → `adaptive_window_frames`
- 基线：从 `gray_buffer` 的最近窗口计算背景均值（策略由代码实现）
- 阈值：`bg_mean * (1 + threshold_over_mean_ratio)`，并 clamp 到 `threshold_minimum`

### 5.3 阈值保护（threshold_protection）
阈值保护用于避免波峰/波形段污染背景均值与自适应阈值。

- 触发条件（示例，具体以代码为准）：
  - 波形触发：`current_gray >= current_threshold`（可配置开关）
  - 波峰触发：当前帧检测到 peaks
- 保护期内：背景均值/计数更新会被抑制或延迟恢复
- 解除条件：满足延迟 + 连续低于阈值的稳定帧数

相关状态变量在主循环中随帧推进：
- `threshold_protection_active`
- `protection_end_time`
- `consecutive_below_threshold`
- `last_waveform_time`

## 6. 波峰检测（Detection）

存在两条检测路径，取决于 `hybrid_detection.enabled` 与 ROI1 数据质量等条件：

> 实现位置：`fem_refactor/detection_pipeline.py`（`run_peak_detection_step()` / `hybrid_peak_detection()`）。

### 6.1 混合检测（ROI1 peaks + ROI2 color）
1) ROI1 曲线做波峰检测（调用 `detect_peaks` 或其包装）
2) 对每个 ROI1 peak 区间，在 ROI2 曲线同一区间上计算 `pre/post avg` 与差值
3) 输出 `hybrid_peaks[]`（包含 peak_interval、color、confidence、质量分等）

### 6.2 传统检测（ROI2-only）
- 对 ROI2 曲线 `detect_peaks(curve, threshold_used, marginFrames, differenceThreshold, silenceFrames, avgFrames)`
- 再做 `min_region_length` 过滤

### 6.3 回退策略
混合检测失败/数据不足时会回退到 ROI2-only，或进入特定的 `detection_mode`（用于日志/缓存标识）。

## 7. 统计与导出（Statistics Sink）

检测结果写入统计系统：
- `SafePeakStatistics.add_peaks_from_daemon(...)`
  - 传统模式：传入 `green_peaks/red_peaks + curve`
  - 混合模式：传入 `hybrid_peaks`（优先使用）

> 实现位置：`fem_refactor/stats_sink.py`（`add_peaks_to_statistics()`）。

统计系统负责：
- 多层去重（含 ROI1 peak_id 去重、连续同色去重、跨颜色策略等）
- 计算并写入 CSV（`export/peak_statistics_<session>.csv`）

## 8. 产物保存（Artifacts）

根据 `data_processing.save_roi1/save_roi2/save_roi3/save_wave/save_roi1_wave/only_delect`：
- 保存 ROI1/2/3 截图
- 保存波形图（ROI2、ROI1 等）
- video 模式通常按“每视频”建立 `tmp/<video_name>/...` 目录

> 实现位置：`fem_refactor/artifact_saver.py`（`save_frame_artifacts()`）。

## 9. 分析缓存（Analysis Cache JSONL）

若 `analysis_cache.enabled=true`：
- 启动时 `analysis_cache.start_session(meta...)`
- 每帧 `analysis_cache.record_frame({...})` 记录关键状态快照：
  - ROI/交点、灰度、buffer 长度
  - `threshold`（fixed/minimum/used/bg_mean/bg_count/protection_active 等）
  - `detection`（模式、开关）
  - `peaks`（raw/filtered）
  - `stats_write`（写入统计系统的结果摘要）

该文件用于回归对比：重构前后同输入应产生等价的状态序列与 peaks 输出。
