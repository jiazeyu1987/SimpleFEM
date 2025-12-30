# SimpleFEM ROI Daemon Refactor TODOLIST（拆分顺序与任务清单）

目标：将 `simple_roi_daemon.py` 的代码重构到 `fem_refactor/` 下，拆成若干“管理类/模块”，**主运行文件只负责调用**，并且**所有判定逻辑不修改**（输出一致性为第一优先级）。

## 原则（必须遵守）
- 只做结构拆分/依赖注入/命名整理，不改阈值、判定条件、顺序、默认值。
- 拆分优先从“纯副作用/IO”与“纯数据结构”开始，最后再拆主循环。
- 每一步拆分都要保留同样的日志文本（或至少同级别/关键字段不变），便于对比回归。

## 建议拆分顺序（从低风险到高风险）

### 1) 纯数据/工具模块（最低风险）
- [ ] `analysis_cache.py`：`RoiAnalysisCache`、JSON 序列化辅助（已进行/可完善）
- [ ] `paths.py`：`BASE_DIR` 解析、目录常量（logs/export/tmp/video）
- [ ] `roi_math.py`：ROI 计算相关纯函数（`adjust_roi1_to_screen`, `compute_roi2_region`）
- [ ] `image_metrics.py`：灰度均值、ROI3 g1/g2、column_diff 等纯计算函数

验收：
- 模块被导入后行为一致；不触发额外 IO；单元测试可选（如果当前仓库没有测试体系则跳过）。

### 2) IO/资源管理类（低风险）
- [ ] `logging_manager.py`：`setup_logging()`、`setup_peak_logger()`（把 handler 创建集中）
- [ ] `cleanup_manager.py`：`cleanup_directories()`（配置读取、删除策略保持不变）
- [ ] `config_loader.py`：`load_fem_config()`（路径策略明确：相对 `BASE_DIR`）

验收：
- 启动日志、清理行为、配置字段读取一致。

### 3) 输入源管理（中风险）
- [ ] `video_source.py`：`discover_video_files`、`initialize_video_capture`、`get_video_frame`、fps/step 计算
- [ ] `screen_source.py`：屏幕抓取（`ImageGrab`）与失败重试策略（如现有）
- [ ] `processing_mode_manager.py`：根据 `processing_mode` 选择 source，并初始化 `SafePeakStatistics`

验收：
- video 模式切换视频/循环、帧步长采样行为一致；screen 模式频率一致。

### 4) 交点与防抖（中风险）
- [x] `anti_jitter_manager.py`：构建滤波器实例（ema/threshold）
- [x] `intersection_manager.py`：每帧 `detect_green_intersection`、失败回退、last_intersection 管理

验收：
- `roi_analysis_cache` 中 `intersection.current/used` 行为一致（特别是失败回退路径）。

### 5) 阈值与缓冲（中高风险）
- [x] `signal_buffers.py`：ROI1/ROI2/ROI3 的 deque、背景均值计数器、状态重置
- [x] `threshold_manager.py`：自适应阈值计算 + `manage_threshold_protection` 状态机（逻辑完全照搬）

验收：
- `threshold.used/bg_mean/bg_count/protection_active` 的时间序列一致（可通过 cache jsonl 对比）。

### 6) 检测管线（高风险，最后做）
- [x] `detection_pipeline.py`：
  - 混合检测：`hybrid_peak_detection(roi1_peaks -> roi2_color)`
  - 传统检测：`detect_peaks(roi2_curve)`
  - fallback 条件、data_quality 检查、min_region_length 过滤（保持顺序）
- [x] `stats_sink.py`：`SafePeakStatistics.add_peaks_from_daemon` 调用封装（参数一字不差传递）
- [x] `artifact_saver.py`：ROI1/2/3/wave 保存策略（`only_delect`、目录结构）

验收：
- `export/peak_statistics_*.csv` 行数/内容一致（同一输入视频同一配置）。

### 7) 主循环编排（最高风险）
- [x] `daemon_loop.py`：迁移 `run_daemon()` 主循环实现
- [x] `orchestrator.py`：顶层入口（委派到 `daemon_loop`）
- [x] 根入口 `simple_roi_daemon.py`：只保留入口调用（兼容 `python simple_roi_daemon.py`）
- [ ] （可选）把 `while True` 循环体进一步拆为 `step()`（需要额外回归验证）

验收：
- 运行方式兼容：
  - `python simple_roi_daemon.py`
  - `python -m fem_refactor.simple_roi_daemon`
  -（如需要）`python fem_refactor/roi_daemon_legacy.py` 调试

## 回归对比建议（强烈建议做）
- [ ] 固定同一视频（例如 `video/2（5次发射，1无效).mp4`）跑一遍，保存：
  - `fem_refactor/external/export/roi_analysis_cache_*.jsonl`
  - `export/peak_statistics_*.csv`
  - `logs/simple_roi_daemon_*.log`
- [ ] 每拆完一个模块，重跑并对比：
  - cache 中关键字段（intersection/threshold/peaks）
  - CSV 行数与关键字段（frame_index、peak_type、threshold_used 等）
