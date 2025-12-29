# HEM 检测系统数据流

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SimpleFEM HEM 检测系统                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ 输入源选择   │
│ - Screen     │
│ - Video      │
│ - Vein       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ROI1 捕获    │  ← 大范围捕获区域 (1280x80 ~ 1920x980)
│              │
│ - PIL.Image  │
│   .Grab      │
│ - OpenCV     │
│   VideoCap   │
└──────┬───────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         绿线检测模块                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │ HSV 过滤     │───▶│ Canny 边缘   │───▶│ Hough 直线   │                │
│  │ [35,85,80]  │    │ 50/150/3     │    │ 50/80/20     │                │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                │
│                                                  │                        │
│                                                  ▼                        │
│                                      ┌───────────────────┐                │
│                                      │ 交点计算 & 滤波   │                │
│                                      │ - EMA             │                │
│                                      │ - Velocity        │                │
│                                      │ - Threshold       │                │
│                                      └─────────┬─────────┘                │
└────────────────────────────────────────────────────────┼───────────────────┘
                                                         │
                              ┌──────────────────────────┴───────────────────┐
                              │              交点坐标 (x, y)                │
                              └──────────────────────────┬───────────────────┘
                                                         │
                ┌────────────────────────────────────────┴────────────────────┐
                │                                                                │
                ▼                                                                ▼
        ┌──────────────┐                                               ┌──────────────┐
        │ ROI2 提取    │                                               │ ROI3 提取    │
        │ 80x120 精确  │                                               │ 垂直扩展     │
        └──────┬───────┘                                               └──────┬───────┘
               │                                                               │
               ▼                                                               ▼
        ┌──────────────┐                                               ┌──────────────┐
        │ ROI2 灰度    │                                               │ ROI3 统计    │
        │ Buffer(100)  │                                               │ - G1/G2      │
        └──────┬───────┘                                               │ - 列差值     │
               │                                                        │ - 归一化     │
               │                                                        └──────┬───────┘
               │                                                               │
               └──────────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                         波峰检测引擎                                 │
        │                                                                     │
        │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
        │  │ 传统 ROI2    │    │ 混合检测     │    │ ROI1 独立    │          │
        │  │ 检测         │    │ ROI1+ROI2    │    │ 检测         │          │
        │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
        │         │                   │                   │                  │
        │         │                   │                   │                  │
        │         └───────────────────┴───────────────────┘                  │
        │                             │                                      │
        │                             ▼                                      │
        │                  ┌──────────────────┐                              │
        │                  │ 自适应阈值计算   │                              │
        │                  │ + 阈值保护       │                              │
        │                  └─────────┬────────┘                              │
        │                            │                                       │
        │                            ▼                                       │
        │                  ┌──────────────────┐                              │
        │                  │ 波峰区间检测     │                              │
        │                  │ - 静默帧要求     │                              │
        │                  │ - 最小宽度       │                              │
        │                  └─────────┬────────┘                              │
        └──────────────────────────────┼─────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                         颜色分类引擎                                 │
        │                                                                     │
        │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
        │  │ ROI2 前后    │    │ ROI3 G1/G2   │    │ ROI3 列差值  │          │
        │  │ 均值差分类   │    │ 覆盖         │    │ 覆盖         │          │
        │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
        │         │                   │                   │                  │
        │         ▼                   ▼                   ▼                  │
        │  ┌──────────────────────────────────────────────────────────┐      │
        │  │              最终颜色判定 (Green/Red)                     │      │
        │  └───────────────────────────┬──────────────────────────────┘      │
        └──────────────────────────────┼─────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                         三层去重系统                                 │
        │                                                                     │
        │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
        │  │ Layer 1     │───▶│ Layer 2     │───▶│ Layer 3     │             │
        │  │ 最近波峰    │    │ 连续帧去重  │    │ 跨色去重    │             │
        │  │ 5帧窗口     │    │ 40帧窗口    │    │ 绿色优先    │             │
        │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
        └─────────┼──────────────────┼──────────────────┼─────────────────────┘
                  │                  │                  │
                  └──────────────────┴──────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                         数据导出系统                                 │
        │                                                                     │
        │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
        │  │ CSV 导出     │    │ JSONL 缓存   │    │ 图像保存     │          │
        │  │ peak_stats   │    │ analysis     │    │ ROI1/2/3     │          │
        │  └──────────────┘    └──────────────┘    └──────────────┘          │
        └─────────────────────────────────────────────────────────────────────┘
```

## 主处理流程

### 1. 初始化阶段

```python
# 1. 加载配置
config = load_fem_config()

# 2. 初始化管理器
statistics_manager = VideoStatisticsManager()
analysis_cache = RoiAnalysisCache()

# 3. 初始化滤波器
intersection_filter = IntersectionFilter()

# 4. 创建会话
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
statistics_manager.initialize_for_video(video_path)
analysis_cache.start_session(session_id, ...)

# 5. 初始化缓冲区
gray_buffer = deque(maxlen=100)
roi1_gray_buffer = deque(maxlen=100)
roi3_gray_buffer = deque(maxlen=100)
roi3_g1_buffer = deque(maxlen=100)
roi3_g2_buffer = deque(maxlen=100)
roi3_column_diff_buffer = deque(maxlen=100)
```

### 2. 帧处理循环

```python
while True:
    # === 步骤 1: 捕获 ROI1 ===
    roi1_image = capture_roi1()  # Screen 或 Video

    # === 步骤 2: 绿线检测 ===
    intersection = detect_green_intersection(roi1_image)
    if intersection:
        intersection = intersection_filter.filter(intersection[0], intersection[1])

    # === 步骤 3: 提取 ROI2 ===
    if intersection:
        roi2_image = extract_roi2(roi1_image, intersection)
        roi2_gray = compute_average_gray(roi2_image)
        gray_buffer.append(roi2_gray)

    # === 步骤 4: 提取 ROI3 ===
    if intersection:
        roi3_image = extract_roi3(roi1_image, intersection)
        roi3_gray = compute_average_gray(roi3_image)
        roi3_gray_buffer.append(roi3_gray)

        # ROI3 统计
        g1, g2 = compute_roi3_g1_g2_ranges(roi3_image)
        roi3_g1_buffer.append(g1)
        roi3_g2_buffer.append(g2)

        column_diff = compute_roi3_column_mean_diff(roi3_image)
        roi3_column_diff_buffer.append(column_diff)

    # === 步骤 5: ROI1 灰度计算 ===
    roi1_gray = compute_average_gray(roi1_image)
    roi1_gray_buffer.append(roi1_gray)

    # === 步骤 6: 阈值保护更新 ===
    should_protect, ... = manage_threshold_protection(
        current_gray=roi2_gray,
        current_threshold=threshold,
        has_peaks=False,  # 稍后检测
        ...
    )

    # === 步骤 7: 波峰检测 ===
    if hybrid_detection_enabled:
        # 混合检测
        hybrid_peaks = hybrid_peak_detection(
            roi1_gray_buffer,
            gray_buffer,
            config
        )
        green_peaks = [p for p in hybrid_peaks if p['color'] == 'green']
        red_peaks = [p for p in hybrid_peaks if p['color'] == 'red']
    else:
        # 传统 ROI2 检测
        green_peaks, red_peaks = detect_peaks(
            gray_buffer,
            threshold,
            ...
        )

    # === 步骤 8: 阈值保护更新 (基于检测结果) ===
    has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
    if has_peaks:
        manage_threshold_protection(..., has_peaks=True, ...)

    # === 步骤 9: 记录分析缓存 ===
    cache_payload = {
        "frame_index": frame_index,
        "roi1_avg": roi1_gray,
        "roi2_avg": roi2_gray,
        "roi3_avg": roi3_gray,
        "intersection": intersection,
        "threshold": threshold,
        "green_peaks": green_peaks,
        "red_peaks": red_peaks,
        "roi3_g1_percent": g1,
        "roi3_g2_percent": g2,
        "roi3_column_diff": column_diff,
        ...
    }
    analysis_cache.record_frame(cache_payload)

    # === 步骤 10: 保存图像和波形 ===
    if data_export_enabled:
        save_roi1(roi1_image, frame_index)
        save_roi2(roi2_image, frame_index)
        save_roi3(roi3_image, frame_index)
        if has_peaks:
            save_waveform(gray_buffer, green_peaks, red_peaks, ...)

    # === 步骤 11: 添加到统计 ===
    if has_peaks:
        statistics_manager.add_peaks(
            frame_index=frame_index,
            green_peaks=green_peaks,
            red_peaks=red_peaks,
            curve_data=list(gray_buffer),
            ...
        )

    frame_index += 1
```

### 3. 清理阶段

```python
# 关闭分析缓存
analysis_cache.close(reason="normal")

# 导出最终 CSV
statistics_manager.export_final_csv()

# 打印汇总
summary = statistics_manager.get_global_summary()
print(f"总视频数: {summary['total_videos_processed']}")
print(f"总波峰数: {summary['total_peaks']}")
```

## 数据缓冲区管理

### 循环缓冲区 (固定大小 100)

```python
from collections import deque

# ROI2 灰度缓冲
gray_buffer = deque(maxlen=100)

# ROI1 灰度缓冲
roi1_gray_buffer = deque(maxlen=100)

# ROI3 统计缓冲
roi3_gray_buffer = deque(maxlen=100)
roi3_g1_buffer = deque(maxlen=100)
roi3_g2_buffer = deque(maxlen=100)
roi3_column_diff_buffer = deque(maxlen=100)
```

**特性**:
- 固定大小，防止内存泄漏
- 自动丢弃旧数据
- 支持 O(1) 追加和访问

### 缓冲区索引映射

```python
# 缓冲区索引 → 绝对帧索引
buffer_start_frame_index = frame_index - len(gray_buffer) + 1

# ROI1 波峰去重键
abs_peak_max = buffer_start_frame_index + peak_start + local_max_offset
peak_key = abs_peak_max
```

## 波峰检测流程

### 传统 ROI2 检测流程

```
┌─────────────────┐
│ ROI2 Buffer     │
│ [100 frames]    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 自适应阈值计算  │
│ (或固定阈值)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 阈值保护检查    │
│ - 保护中?       │
│ - 冻结背景均值  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 波峰区域检测    │
│ - > threshold   │
│ - 边界扩展      │
│ - 静默要求      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 最小宽度过滤    │
│ min_region_len  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 颜色分类        │
│ - frame_diff    │
│ - ROI3 覆盖     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Green/Red Peaks │
│ [(start,end),..]│
└─────────────────┘
```

### 混合检测流程

```
┌─────────────────┐           ┌─────────────────┐
│ ROI1 Buffer     │           │ ROI2 Buffer     │
│ [100 frames]    │           │ [100 frames]    │
└────────┬────────┘           └────────┬────────┘
         │                              │
         ▼                              │
┌─────────────────┐                    │
│ ROI1 独立阈值   │                    │
│ 波峰检测        │                    │
│ (不做颜色分类)  │                    │
└────────┬────────┘                    │
         │                              │
         ▼                              │
┌─────────────────┐                    │
│ ROI1 波峰       │                    │
│ 宽度过滤        │                    │
│ [min, max]      │                    │
└────────┬────────┘                    │
         │                              │
         ▼                              │
┌─────────────────┐                    │
│ ROI1 去重       │                    │
│ (peak_max 位置) │                    │
└────────┬────────┘                    │
         │                              │
         ▼                              │
┌─────────────────┐                    │
│ ROI1 波峰区间   │                    │
│ [(start,end),..]│                    │
└────────┬────────┘                    │
         │                              │
         └──────────┬───────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ ROI2 颜色判定   │
            │ (在 ROI1 区间)  │
            │ - 前后均值差    │
            │ - 数据质量      │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ ROI3 覆盖       │
            │ - G1/G2         │
            │ - 列差值        │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ 最终颜色判定    │
            │ Green/Red       │
            └─────────────────┘
```

## 去重流程

```
Layer 1: 最近波峰比较
┌────────────────────────────────────────┐
│ 新波峰 vs 最近 5 帧波峰                │
│ - 相同 peak_max 位置?                  │
│ - 相同颜色?                            │
└────────────┬───────────────────────────┘
             │
             ▼ (未重复)
Layer 2: 连续帧去重
┌────────────────────────────────────────┐
│ 新波峰 vs 40 帧窗口内同色波峰          │
│ - 同色?                                │
│ - 帧间隔 <= 40?                        │
└────────────┬───────────────────────────┘
             │
             ▼ (未重复)
Layer 3: 跨色去重
┌────────────────────────────────────────┐
│ 新波峰 vs 同帧不同色波峰                │
│ - 同一帧?                              │
│ - 颜色优先级: Green > Red              │
└────────────┬───────────────────────────┘
             │
             ▼ (未重复)
┌────────────────────────────────────────┐
│ 通过所有去重检查                       │
│ → 保存到统计                           │
└────────────────────────────────────────┘
```

## 阈值保护状态机

```
                   ┌──────────────────────┐
                   │     INACTIVE         │
                   │  (背景更新正常)      │
                   └──────────┬───────────┘
                              │
                              │ 触发条件:
                              │ - 波形触发
                              │   (gray >= threshold)
                              │ - 波峰触发
                              │   (has_peaks == true)
                              ▼
                   ┌──────────────────────┐
                   │      ACTIVE           │
                   │  (背景更新冻结)      │
                   └──────────┬───────────┘
                              │
                              │ 解除条件:
                              │ - 时间延迟满足
                              │   (current >= planned_end)
                              │ - 稳定性满足
                              │   (consecutive_below >= 5)
                              ▼
                   ┌──────────────────────┐
                   │     INACTIVE         │
                   └──────────────────────┘
```

## 批量视频处理流程

```
┌────────────────────────────────────────┐
│  初始化                                 │
│  - 扫描视频文件夹                       │
│  - 创建视频文件列表                     │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  视频 1 处理                            │
│  - 初始化统计实例                       │
│  - 创建输出文件夹                       │
│  - 处理所有帧                           │
│  - 导出 CSV                             │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  切换视频                               │
│  - 关闭视频资源                         │
│  - 重置所有缓冲区                       │
│  - 重置状态变量                         │
│  - 重置防抖动滤波器                     │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  视频 2 处理                            │
│  - 初始化新统计实例                     │
│  - 创建新输出文件夹                     │
│  - 处理所有帧                           │
│  - 导出 CSV                             │
└────────────┬───────────────────────────┘
             │
             ▼
              ...
             │
             ▼
┌────────────────────────────────────────┐
│  所有视频处理完成                       │
│  - 汇总统计                             │
│  - 打印报告                             │
└────────────────────────────────────────┘
```

## 数据导出格式

### CSV 导出

```csv
timestamp,frame_index,peak_start,peak_end,peak_color,peak_value,pre_avg,post_avg,...
2025-12-26T10:30:45,349,345,350,green,142.3,45.2,67.8,...
```

### JSONL 缓存

```jsonl
{"type":"meta","cache_version":1,"created_at":"2025-12-26T10:30:00",...}
{"type":"frame","frame_index":349,"roi1_avg":52.3,"roi2_avg":142.3,...}
{"type":"frame","frame_index":350,"roi1_avg":53.1,"roi2_avg":140.5,...}
...
{"type":"session_end","ended_at":"2025-12-26T10:35:00","reason":"video_complete"}
```

## 错误处理流程

```
┌────────────────────────────────────────┐
│  错误检测                               │
│  - 捕获失败                             │
│  - 检测失败                             │
│  - 导出失败                             │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  错误记录                               │
│  - 日志记录                             │
│  - 错误类型                             │
│  - 上下文信息                           │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  错误恢复                               │
│  - 使用上一有效值                       │
│  - 跳过当前帧                           │
│  - 继续处理                             │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  优雅降级                               │
│  - 不中断处理                           │
│  - 保持服务可用                         │
└────────────────────────────────────────┘
```
