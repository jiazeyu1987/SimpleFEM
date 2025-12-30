# simple_roi_daemon_重写技术文档 - 冲突分析报告

本报告详细列出了文档中发现的所有冲突、不一致和遗漏问题。

---

## 严重冲突（必须修复）

### 1. ❌ ROI3 覆盖逻辑的硬编码阈值

**位置：** 第 9.3 节

**问题：**
```python
# 文档中的描述：
2. 任何颜色 -> RED：如果 `roi3_peak_max_frame < 110`
```

**冲突：** 值 `110` 是硬编码的，在配置文件中找不到对应的配置项。

**影响：** 无法通过配置调整这个阈值，不同视频可能需要不同的阈值。

**建议修复：**
```json
// 在配置文件中添加：
"roi3_override": {
  "enabled": true,
  "threshold": 115.0,
  "require_roi3_data": true,
  "early_frame_threshold": 110  // 新增：早期波峰帧数阈值
}
```

---

### 2. ❌ ROI2 异常过滤的硬编码阈值

**位置：** 第 5.2 节

**问题：**
```python
# 文档中的描述：
4. 异常过滤：如果 `|frame_difference| > 15`，判定为错误数据
```

**冲突：** 值 `15` 是硬编码的，不在配置文件中。

**影响：** 无法根据不同信号特性调整异常检测灵敏度。

**建议修复：**
```json
// 在 hybrid_detection.data_quality 中添加：
"max_frame_difference": 15.0  // 新增：frame_difference 异常阈值
```

---

### 3. ❌ ROI1 自适应阈值更新逻辑不完整

**位置：** 第 4.2 节步骤 5

**问题：**
```python
# 文档中的描述（ROI2）：
if not threshold_protection_active:
    bg_mean = calculated_bg_mean
    bg_count = len(recent_frames)
    threshold_used = bg_mean * (1.0 + threshold_over_mean_ratio)
```

**冲突：** ROI1 的阈值更新逻辑与 ROI2 不同。ROI1 使用增量更新，但文档中描述的是替换更新。

**实际源代码逻辑（ROI1）：**
```python
# 只有当前值低于阈值时才更新背景均值
if roi1_gray < roi1_threshold_used:
    roi1_bg_count += 1
    roi1_bg_mean = roi1_bg_mean + (roi1_gray - roi1_bg_mean) / roi1_bg_count
```

**建议修复：** 在第 4.2 节步骤 5 中明确说明 ROI1 使用增量更新，ROI2 使用替换更新。

---

### 4. ❌ 初始化阶段缺少关键变量初始化

**位置：** 第 4.1 节

**问题：** 初始化阶段没有初始化以下关键变量：
- `processed_roi1_peaks` （第 5.3 节提到）
- `roi1_peak_counter` （第 5.3 节提到）
- 阈值保护的状态变量（第 9.2 节提到）

**影响：** 按照文档重写会导致运行时错误。

**建议修复：** 在第 4.1 节添加：
```python
def run_daemon():
    # ... 其他初始化 ...

    # ROI1 波峰管理
    processed_roi1_peaks: Dict[int, str] = {}
    roi1_peak_counter: int = 0

    # ROI2 阈值保护状态
    threshold_protection_active: bool = False
    protection_end_time: float = 0.0
    consecutive_below_threshold: int = 0
    last_waveform_time: float = 0.0

    # ROI1 阈值保护状态
    roi1_threshold_protection_active: bool = False
    roi1_protection_end_time: float = 0.0
    roi1_consecutive_below_threshold: int = 0
    roi1_last_waveform_time: float = 0.0

    # 背景均值
    bg_count: int = 0
    bg_mean: float = 0.0
    roi1_bg_count: int = 0
    roi1_bg_mean: float = 0.0
```

---

### 5. ❌ 视频切换时缺少变量重置

**位置：** 第 6.2 节

**问题：** `handle_video_switch()` 函数中没有重置 `roi1_peak_counter`。

**影响：** 多视频处理时，后续视频的波峰 ID 会继续累加，导致 ID 不连续。

**建议修复：** 在第 6.2 节的重置部分添加：
```python
# 重置 ROI1 波峰 ID 管理
roi1_peak_counter = 0
```

---

## 中等冲突（影响代码质量）

### 6. ⚠️ ROI1 阈值保护未实现

**位置：** 第 4.2 节步骤 5

**问题：**
- 文档第 146-163 行定义了 `roi1_peak_detection.threshold_protection` 配置
- 但在第 4.2 节步骤 5 的主循环中，没有说明如何管理 ROI1 的阈值保护状态
- 第 5.1 节只描述了 ROI2 的阈值保护机制

**影响：** 无法实现 ROI1 的阈值保护功能。

**建议：** 添加类似 ROI2 的阈值保护逻辑说明。

---

### 7. ⚠️ ROI3 提取的 extension_params 来源不明确

**位置：** 第 4.2 节步骤 4

**问题：**
```python
# ROI3 提取（类似 ROI2）
if roi3_extension_params:  # 这个变量从哪里来？
```

**冲突：** 在初始化阶段和主循环中没有说明如何获取 `roi3_extension_params`。

**建议修复：** 在步骤 4 开头添加：
```python
# 从配置文件读取
roi3_config = config.get("roi_capture", {}).get("roi3_config", {})
roi3_extension_params = roi3_config.get("extension_params", {})
```

---

### 8. ⚠️ `video_time_str` 变量作用域问题

**位置：** 第 4.2 节步骤 8

**问题：**
```python
# 保存 ROI2
roi2_path = f"{roi2_dir}/roi2_{frame_index:06d}{video_time_str}.png"
```

**冲突：** `video_time_str` 只在视频模式下定义，但在所有模式下的保存代码中都使用了这个变量。

**实际源代码逻辑：**
- 在视频模式下，`video_time_str` 格式为 `"_XXXX.XXs"`
- 在屏幕模式下，不包含时间戳后缀

**建议修复：**
```python
# 在保存 ROI2 之前
video_time_str = ""
if processing_mode == "video" and video_cap is not None:
    video_pos_msec = video_cap.get(cv2.CAP_PROP_POS_MSEC)
    video_seconds = video_pos_msec / 1000.0
    video_time_str = f"_{video_seconds:06.2f}s"

roi2_path = f"{roi2_dir}/roi2_{frame_index:06d}{video_time_str}.png"
```

---

### 9. ⚠️ ROI1 波峰检测参数与 ROI2 参数命名冲突

**位置：** 第 3.1 节配置文件结构

**问题：**
- `peak_detection.min_region_length` (ROI2 波峰最小宽度)
- `roi1_peak_detection.min_region_length` (ROI1 波峰最小宽度)

**冲突：** 虽然名称相同，但含义不同：
- ROI2 的 `min_region_length` 用于过滤 ROI2 检测到的波峰
- ROI1 的 `min_region_length` 用于过滤 ROI1 检测到的波峰

**建议：** 在文档中明确说明这两个参数的独立作用。

---

### 10. ⚠️ 缓冲区空值检查缺失

**位置：** 第 4.2 节步骤 6

**问题：**
```python
# 传统 ROI2 检测模式
if gray_buffer:
    curve = list(gray_buffer)
    green_peaks_raw, red_peaks_raw = detect_peaks(...)
```

**冲突：** 在混合检测路径中，没有检查 `gray_buffer` 是否为空。

**建议修复：** 在混合检测之前添加：
```python
if hybrid_enabled and roi1_enabled and len(roi1_gray_buffer) > 0 and len(gray_buffer) > 0:
    # 混合检测模式
    ...
```

---

## 轻微问题（不影响功能，但影响理解）

### 11. ℹ️ 函数命名不准确

**位置：** 附录 B

**问题：** `compute_roi2_region` 函数名称暗示它只能计算 ROI2，但实际上它也用于计算 ROI3。

**建议：** 重命名为 `compute_roi_region`，或在文档中说明该函数是通用的。

---

### 12. ℹ️ ROI3 图像保存时机描述不清晰

**位置：** 第 4.2 节步骤 8

**问题：** ROI3 图像保存逻辑没有在主循环伪代码中体现。

**实际逻辑：** ROI3 图像只在 `roi3_image is not None` 时保存，这与 ROI2 类似。

**建议：** 在主循环伪代码中添加 ROI3 保存逻辑。

---

### 13. ℹ️ ROI1 波峰宽度的验证规则

**位置：** 第 5.3 节

**问题：**
```python
# 应用最小宽度过滤
min_width = config.get('min_peak_width', 5)
max_width = config.get('max_peak_width', 100)
```

**冲突：** 配置文件中的实际路径是 `roi1_peak_width_range: [30, 40]`，不是单独的 `min_peak_width` 和 `max_peak_width`。

**建议修复：** 更新为：
```python
peak_width_range = config.get('roi1_peak_width_range', [5, 100])
min_width = peak_width_range[0]
max_width = peak_width_range[1]
```

---

### 14. ℹ️ 防抖动滤波器的 movement_threshold 默认值

**位置：** 第 3.1 节 vs 第 9.4 节

**问题：**
- 第 3.1 节配置：`"movement_threshold": 20.0`
- 第 12.2 节建议："增大 `movement_threshold`（如 30-40）"

**冲突：** 文档中的默认值与建议值不一致。

**建议：** 明确说明默认值是 20.0，对于抖动严重的情况可以调整到 30-40。

---

### 15. ℹ️ ROI3 覆盖逻辑的应用位置

**位置：** 第 4.2 节

**问题：** ROI3 覆盖逻辑在哪里应用？文档中没有明确说明。

**实际位置：** 在 `safe_peak_statistics.py` 的 `_create_peak_data` 或 `_create_hybrid_peak_data` 方法中应用。

**建议：** 在步骤 7（添加到统计）中说明 ROI3 覆盖逻辑会在统计模块中自动应用。

---

### 16. ℹ️ only_delect 拼写错误

**位置：** 第 3.1 节

**问题：**
```json
"only_delect": true
```

**冲突：** 应该是 `only_detect`，但代码中使用了 `only_delect`。

**影响：** 这是历史遗留的拼写错误，但为了保持一致性，文档应该保持这个拼写。

---

### 17. ℹ️ 混合检测的 ROI1 阈值计算时机

**位置：** 第 4.2 节

**问题：** 混合检测使用 `roi1_threshold_used`，但这个变量在哪里计算的？

**实际逻辑：** ROI1 的自适应阈值计算在主循环的后半部分（步骤 6 之后），但混合检测在步骤 6 中就使用了这个值。

**建议：** 说明 ROI1 阈值是在上一帧计算的，当前帧使用的是上一帧的阈值。

---

### 18. ℹ️ ROI2 配色判定参数的默认值

**位置：** 第 5.2 节

**问题：**
- 文档说："默认前5帧"、"默认后10帧"
- 配置文件：`roi2_color_frames.pre_peak: 5, post_peak: 10`

**建议：** 明确说明这些默认值来自配置文件。

---

### 19. ℹ️ 波峰 ID 生成公式的解释

**位置：** 第 5.3 节

**问题：**
```python
peak_key = buffer_start_frame_index + peak_start + local_max_offset
```

**解释不够清晰：**
- `buffer_start_frame_index` 是什么？（缓冲区第一个元素对应的全局帧索引）
- `peak_start` 是什么？（波峰在缓冲区中的起始位置）
- `local_max_offset` 是什么？（波峰最大值在波峰区间内的偏移）

**建议：** 添加详细的计算示例。

---

### 20. ℹ️ ROI3 最大值帧的计算公式

**位置：** 用户之前的问题

**问题：** 文档中没有说明 `roi3_peak_max_frame` 的计算公式。

**实际公式：**
```python
roi3_peak_max_frame = curve_start_global_frame + roi3_max_curve_idx
```

其中：
- `curve_start_global_frame = frame_index - len(roi3_curve) + 1`
- `roi3_max_curve_idx` 是 ROI3 曲线中最大值的索引

**建议：** 在第 5.2 节或第 9.3 节添加这个计算公式的说明。

---

## 配置参数缺失（建议添加）

### 21. 🔧 建议添加的配置项

```json
{
  "peak_detection": {
    // 现有配置...
    "frame_difference_max": 15.0  // ROI2 异常检测阈值
  },
  "roi3_override": {
    "enabled": true,
    "threshold": 115.0,
    "require_roi3_data": true,
    "early_frame_threshold": 110  // ROI3 早期波峰帧数阈值
  }
}
```

---

## 代码示例修正建议

### 22. 🔧 主循环初始化应该更完整

**建议添加到第 4.1 节：**
```python
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
roi1_threshold_used: float = max(roi1_threshold, roi1_threshold_minimum)

# ROI3 状态
roi3_gray_buffer: Deque[float] = deque(maxlen=100)

# ROI1 波峰管理
processed_roi1_peaks: Dict[int, str] = {}
roi1_peak_counter: int = 0

# 绿线交点回退
last_intersection_roi: Optional[Tuple[int, int]] = None
```

---

### 23. 🔧 ROI1 自适应阈值计算应该独立说明

**建议添加到第 4.2 节步骤 5 中：**
```python
# ========== 步骤 5.5: ROI1 自适应阈值计算（独立于 ROI2）==========
roi1_threshold_used = max(roi1_threshold, roi1_threshold_minimum)

if roi1_enabled and roi1_gray_buffer:
    if roi1_adaptive_threshold_enabled and len(roi1_gray_buffer) >= roi1_adaptive_window_frames:
        # 只有当前值低于阈值时才更新背景均值（防止污染）
        if roi1_gray < roi1_threshold_used:
            roi1_bg_count += 1
            roi1_bg_mean = roi1_bg_mean + (roi1_gray - roi1_bg_mean) / roi1_bg_count

        # 计算自适应阈值
        if roi1_bg_mean > 0:
            roi1_threshold_used = roi1_bg_mean * (1.0 + roi1_threshold_over_mean_ratio)
            roi1_threshold_used = max(roi1_threshold_used, roi1_threshold_minimum)
```

---

## 总结

### 必须修复的问题（5个）
1. ROI3 覆盖逻辑的硬编码阈值
2. ROI2 异常过滤的硬编码阈值
3. ROI1 自适应阈值更新逻辑不完整
4. 初始化阶段缺少关键变量
5. 视频切换时缺少变量重置

### 建议修复的问题（8个）
6-10: 中等冲突

### 可选优化的问题（10个）
11-20: 轻微问题

### 配置增强
21. 建议添加的配置项

### 代码示例改进
22-23. 更完整的代码示例

---

**文档版本：** 1.0
**分析时间：** 2025-12-25
**分析工具：** Claude Code
**严重程度：** 中等（需要修复关键冲突才能使用文档重写）
