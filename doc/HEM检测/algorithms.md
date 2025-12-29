# HEM 检测算法详解

## 目录
1. [绿线检测算法](#绿线检测算法)
2. [波峰检测算法](#波峰检测算法)
3. [颜色分类算法](#颜色分类算法)
4. [阈值保护算法](#阈值保护算法)
5. [去重算法](#去重算法)
6. [ROI3 统计算法](#roi3-统计算法)

---

## 绿线检测算法

### HSV 颜色空间过滤

绿线检测首先将 ROI1 图像从 BGR 颜色空间转换到 HSV 颜色空间：

```python
hsv = cv2.cvtColor(roi1_array, cv2.COLOR_BGR2HSV)
```

**HSV 阈值范围**:
- H (Hue): 35-85 (绿色色相)
- S (Saturation): 80-255 (高饱和度)
- V (Value): 80-255 (高亮度)

```python
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
```

### Canny 边缘检测

对二值化绿线掩码进行边缘检测：

```python
edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)
```

**参数说明**:
- `threshold1`: 50 (低阈值)
- `threshold2`: 150 (高阈值)
- `apertureSize`: 3 (Sobel 算子尺寸)

### Hough 直线变换

使用概率 Hough 变换检测直线：

```python
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=50, minLineLength=80, maxLineGap=20)
```

**参数说明**:
- `rho`: 1 (距离分辨率)
- `theta`: π/180 (角度分辨率)
- `threshold`: 50 (累加器阈值)
- `minLineLength`: 80 (最小线段长度)
- `maxLineGap`: 20 (最大间隙)

### 交点计算

选择两条不平行的直线计算几何交点：

```python
def compute_line_intersection(lines):
    # 选择两条不平行直线
    line1, line2 = select_non_parallel_lines(lines)

    # 计算交点
    x, y = line_intersection(line1, line2)
    return (x, y)
```

### 防抖动滤波

#### EMA (指数移动平均) 滤波

```python
def ema_filter(new_position, last_position, alpha=0.25):
    """
    Args:
        new_position: 当前检测到的交点
        last_position: 上一次滤波后的位置
        alpha: 平滑因子 (0-1)，越小越平滑
    """
    filtered_x = alpha * new_position[0] + (1 - alpha) * last_position[0]
    filtered_y = alpha * new_position[1] + (1 - alpha) * last_position[1]
    return (filtered_x, filtered_y)
```

#### Velocity Filter (速度滤波)

```python
def velocity_filter(new_position, last_position, velocity_threshold=20.0, fps=10):
    """
    拒绝移动速度过快的交点
    """
    dt = 1.0 / fps
    dx = new_position[0] - last_position[0]
    dy = new_position[1] - last_position[1]
    velocity = np.sqrt(dx**2 + dy**2) / dt

    if velocity > velocity_threshold:
        return last_position  # 保持旧位置
    return new_position
```

#### Threshold Filter (阈值滤波)

```python
def threshold_filter(new_position, last_position, movement_threshold=20.0):
    """
    小于阈值时完全静止，超过时直接更新
    """
    dx = abs(new_position[0] - last_position[0])
    dy = abs(new_position[1] - last_position[1])

    if dx < movement_threshold and dy < movement_threshold:
        return last_position  # 静止
    return new_position  # 直接更新
```

---

## 波峰检测算法

### 固定阈值检测

```python
def detect_peaks_fixed_threshold(curve, threshold):
    """
    Args:
        curve: 灰度值曲线
        threshold: 固定阈值
    Returns:
        green_peaks: [(start, end), ...]
        red_peaks: [(start, end), ...]
    """
    # 找到所有超过阈值的区域
    above_threshold = curve >= threshold

    # 扩展边界 (margin_frames)
    expanded = expand_boundaries(above_threshold, margin=5)

    # 应用静默帧要求
    filtered = apply_silence_requirement(expanded, silence=15)

    # 应用最小宽度要求
    peaks = apply_minimum_width(filtered, min_width=5)

    # 颜色分类
    return classify_peaks_by_color(peaks, curve, diff_threshold=2.1)
```

### 自适应阈值检测

```python
def adaptive_threshold_detection(curve, config):
    """
    基于背景均值动态计算阈值
    """
    # 计算背景均值
    bg_mean = calculate_background_mean(curve, window_frames=30)

    # 计算自适应阈值
    adaptive_threshold = bg_mean * (1 + threshold_over_mean_ratio)  # 默认 1.15

    # 应用最小阈值限制
    threshold = max(adaptive_threshold, threshold_minimum)  # 默认 80

    # 使用自适应阈值进行检测
    return detect_peaks_fixed_threshold(curve, threshold)
```

### 阈值保护机制

```python
def manage_threshold_protection(
    current_gray,
    current_threshold,
    has_peaks,
    protection_state
):
    """
    防止波峰数据污染背景计算
    """
    # 触发条件 1: 波形触发
    if current_gray >= current_threshold:
        protection_state.active = True
        protection_state.last_waveform_time = current_time

    # 触发条件 2: 波峰触发
    elif has_peaks:
        protection_state.active = True
        protection_state.last_waveform_time = current_time

    # 解除条件
    if protection_state.active:
        # 时间条件: recovery_delay_frames 后
        time_met = current_time >= protection_state.planned_end_time

        # 稳定性条件: 连续 stability_frames 低于阈值
        if current_gray < current_threshold:
            protection_state.consecutive_below += 1
        else:
            protection_state.consecutive_below = 0

        stability_met = protection_state.consecutive_below >= stability_frames

        if time_met and stability_met:
            protection_state.active = False
            protection_state.consecutive_below = 0

    return protection_state
```

### ROI1/ROI2 混合检测

```python
def hybrid_peak_detection(roi1_curve, roi2_curve, config):
    """
    ROI1 检测波峰时机，ROI2 判定颜色
    """
    # 步骤 1: ROI1 检测波峰区间
    roi1_peaks = detect_peaks(
        roi1_curve,
        threshold=config['roi1_threshold'],  # 独立阈值
        differenceThreshold=999.0  # 不做颜色分类
    )

    # 步骤 2: 宽度过滤
    filtered_peaks = filter_by_width(
        roi1_peaks,
        min_width=5,
        max_width=100
    )

    # 步骤 3: 去重 (使用 ROI1 波峰最大值位置)
    unique_peaks = dedupe_by_peak_max(filtered_peaks, processed_peaks)

    # 步骤 4: ROI2 颜色判定
    classified_peaks = []
    for peak_start, peak_end in unique_peaks:
        color = determine_roi2_color(peak_start, peak_end, roi2_curve)

        # 步骤 5: ROI3 覆盖
        if color == 'red':
            color = apply_roi3_override(peak_start, peak_end, roi3_stats)

        classified_peaks.append({
            'peak_interval': (peak_start, peak_end),
            'color': color
        })

    return classified_peaks
```

---

## 颜色分类算法

### ROI2 前后均值差分类

```python
def classify_by_frame_diff(roi2_curve, peak_start, peak_end, diff_threshold):
    """
    基于 ROI2 波峰前后均值差进行颜色分类
    """
    pre_frames = 5
    post_frames = 10

    # 计算前均值
    pre_start = max(0, peak_start - pre_frames)
    pre_values = roi2_curve[pre_start:peak_start]
    pre_avg = sum(pre_values) / len(pre_values)

    # 计算后均值
    post_end = min(len(roi2_curve), peak_end + post_frames + 1)
    post_values = roi2_curve[peak_end + 1:post_end]
    post_avg = sum(post_values) / len(post_values)

    # 计算差异
    frame_diff = post_avg - pre_avg

    # 错误数据过滤
    if abs(frame_diff) > 15.0:
        return 'red', 'error_filtered'  # 标记为错误

    # 颜色判定
    if frame_diff >= diff_threshold:
        return 'green', 'frame_diff'
    else:
        return 'red', 'frame_diff'
```

### ROI3 G1/G2 覆盖

```python
def apply_g1_g2_override(color, roi3_g1, roi3_g2, config):
    """
    使用 ROI3 G1/G2 统计覆盖颜色判定
    """
    if color != 'red':
        return color  # 只覆盖红色波峰

    g1_threshold = 98.0  # G1 阈值
    g2_threshold = 20.0  # G2 阈值

    if roi3_g1 > g1_threshold and roi3_g2 > g2_threshold:
        # 满足覆盖条件
        return 'green'
    return color
```

**G1/G2 计算**:
```python
def compute_g1_g2_ranges(roi3_image):
    """
    G1: 灰度值在 [80, 255] 范围的像素百分比
    G2: 灰度值在 [150, 255] 范围的像素百分比
    """
    roi3_array = np.array(roi3_image.convert('L'))

    # G1: 80-255 范围
    g1_pixels = np.sum((roi3_array >= 80) & (roi3_array <= 255))
    g1_percent = (g1_pixels / roi3_array.size) * 100

    # G2: 150-255 范围
    g2_pixels = np.sum((roi3_array >= 150) & (roi3_array <= 255))
    g2_percent = (g2_pixels / roi3_array.size) * 100

    return g1_percent, g2_percent
```

### ROI3 列灰度差值覆盖

```python
def apply_column_diff_override(color, roi3_g1, column_diff, config):
    """
    使用 ROI3 列灰度差值覆盖颜色判定
    """
    if color != 'red':
        return color  # 只覆盖红色波峰

    column_diff_threshold = 15.0  # 列差值阈值

    if roi3_g1 > 99.0 and column_diff > column_diff_threshold:
        # 满足覆盖条件
        return 'green'
    return color
```

**列灰度差值计算**:
```python
def compute_column_mean_diff(roi3_image):
    """
    计算每列平均灰度的最大值与最小值之差
    """
    roi3_array = np.array(roi3_image.convert('L'))

    # 计算每列平均灰度
    column_means = np.mean(roi3_array, axis=0)

    # 计算最大值与最小值之差
    diff = float(np.max(column_means) - np.min(column_means))

    return diff
```

---

## 阈值保护算法

### 状态机模型

```
[INACTIVE] --波形触发/波峰触发--> [ACTIVE]
    ^                                 |
    |                                 | 时间延迟 + 稳定性
    |                                 v
    --------------------------- [INACTIVE]
```

### 伪代码

```python
# 初始化
protection_active = False
protection_end_time = 0.0
consecutive_below = 0
last_waveform_time = 0.0

# 每帧处理
current_time = frame_time
current_gray = roi2_avg

# 检查触发条件
if current_gray >= threshold:
    protection_active = True
    last_waveform_time = current_time

# 检查解除条件
if protection_active:
    planned_end = last_waveform_time + recovery_delay

    # 检查稳定性
    if current_gray < threshold:
        consecutive_below += 1
    else:
        consecutive_below = 0

    # 判断是否解除
    if current_time >= planned_end and consecutive_below >= stability_frames:
        protection_active = False
        consecutive_below = 0
```

---

## 去重算法

### 三层去重系统

#### Layer 1: 最近波峰比较

```python
def recent_peak_deduplication(new_peak, recent_peaks, window=5):
    """
    与最近 5 帧的波峰比较
    """
    for recent_peak in recent_peaks:
        if is_same_peak(new_peak, recent_peak):
            return True  # 重复，过滤
    return False
```

#### Layer 2: 连续帧去重

```python
def consecutive_frame_deduplication(new_peak, history, window=40):
    """
    在 40 帧窗口内，相同颜色波峰去重
    """
    for old_peak in history:
        if (old_peak.color == new_peak.color and
            abs(old_peak.frame_index - new_peak.frame_index) <= window):
            return True  # 重复，过滤
    return False
```

#### Layer 3: 跨色去重

```python
def cross_color_deduplication(new_peak, history):
    """
    不同颜色波峰在同一帧，保留高优先级颜色
    """
    for old_peak in history:
        if old_peak.frame_index == new_peak.frame_index:
            # 颜色优先级: Green (2) > Red (1)
            if (new_peak.color == 'green' and old_peak.color == 'red'):
                # 新波峰优先，删除旧波峰
                remove_peak(old_peak)
            elif old_peak.color == 'green' and new_peak.color == 'red':
                # 旧波峰优先，过滤新波峰
                return True
    return False
```

### ROI1 波峰去重 (混合检测模式)

```python
def roi1_peak_deduplication(new_peak_interval, processed_peaks):
    """
    使用 ROI1 波峰最大值的绝对帧索引作为去重键
    """
    peak_start, peak_end = new_peak_interval
    roi1_curve = config['roi1_curve']

    # 找到波峰最大值位置
    peak_slice = roi1_curve[peak_start:peak_end + 1]
    local_max_offset = max(range(len(peak_slice)), key=lambda i: peak_slice[i])

    # 计算绝对帧索引
    buffer_start = config['buffer_start_frame_index']
    abs_peak_max = buffer_start + peak_start + local_max_offset

    # 检查是否已处理
    peak_key = abs_peak_max
    if peak_key in processed_peaks:
        return True  # 重复

    # 记录新波峰
    peak_id = f"ROI1_MAX_{abs_peak_max:06d}"
    processed_peaks[peak_key] = peak_id
    return False
```

---

## ROI3 统计算法

### G1/G2 像素百分比

```python
def compute_g1_g2_ranges(roi3_image):
    """
    计算特定灰度范围的像素百分比

    Returns:
        g1_percent: 灰度 [80, 255] 范围的像素百分比
        g2_percent: 灰度 [150, 255] 范围的像素百分比
    """
    roi3_array = np.array(roi3_image.convert('L'))

    total_pixels = roi3_array.size

    # G1: 80-255 范围 (高回声)
    g1_mask = (roi3_array >= 80) & (roi3_array <= 255)
    g1_pixels = np.sum(g1_mask)
    g1_percent = (g1_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # G2: 150-255 范围 (超高回声)
    g2_mask = (roi3_array >= 150) & (roi3_array <= 255)
    g2_pixels = np.sum(g2_mask)
    g2_percent = (g2_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return g1_percent, g2_percent
```

### 列灰度差值

```python
def compute_column_mean_diff(roi3_image):
    """
    计算每列平均灰度的最大值与最小值之差

    Returns:
        float: 列平均灰度差值
    """
    roi3_array = np.array(roi3_image.convert('L'))

    # 计算每列的平均灰度值 (axis=0 表示沿列方向)
    column_means = np.mean(roi3_array, axis=0)

    # 计算最大值与最小值之差
    max_mean = float(np.max(column_means))
    min_mean = float(np.min(column_means))
    diff = max_mean - min_mean

    return diff
```

### 归一化灰度百分比 (80-160)

```python
def compute_roi3_80_160_normalized(roi3_image):
    """
    计算灰度值在 [80, 160] 范围内的像素百分比

    Returns:
        float: 归一化百分比
    """
    roi3_array = np.array(roi3_image.convert('L'))

    total_pixels = roi3_array.size

    # 灰度范围 [80, 160]
    range_mask = (roi3_array >= 80) & (roi3_array <= 160)
    range_pixels = np.sum(range_mask)

    percent = (range_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return percent
```

---

## 数据质量评估

### ROI2 数据质量

```python
def calculate_roi2_data_quality(peak_start, peak_end, roi2_curve):
    """
    评估 ROI2 在波峰区间的数据质量

    Returns:
        quality_score: 质量评分 [0, 1]
        variance: 方差
        frame_count: 有效帧数
    """
    # 提取区间数据
    interval_values = roi2_curve[peak_start:peak_end + 1]

    # 计算方差
    mean_val = sum(interval_values) / len(interval_values)
    variance = sum((x - mean_val) ** 2 for x in interval_values) / len(interval_values)
    std_dev = math.sqrt(variance)

    # 计算数据范围
    data_range = max(interval_values) - min(interval_values)

    # 稳定性评分 (标准差/数据范围)
    stability = max(0, 1.0 - std_dev / max(10.0, data_range))

    # 一致性评分 (标准差/均值)
    consistency = 1.0 - min(1.0, std_dev / mean_val) if mean_val > 0 else 0

    # 综合质量评分
    quality_score = (stability + consistency) / 2.0

    return {
        'quality_score': quality_score,
        'variance': variance,
        'std_dev': std_dev,
        'frame_count': len(interval_values)
    }
```

### 质量检查阈值

```python
# 最小帧数要求
minimum_roi2_frames = 15

# 最小方差要求 (避免信号过于平坦)
roi2_minimum_variance = 0.5

# 数据有效范围
roi2_min_gray = 5.0
roi2_max_gray = 250.0

# 不合格数据过滤
if frame_count < minimum_roi2_frames:
    return 'invalid', 'insufficient_frames'

if variance < roi2_minimum_variance:
    return 'invalid', 'low_variance'
```
