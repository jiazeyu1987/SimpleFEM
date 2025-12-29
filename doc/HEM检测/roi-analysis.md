# ROI 分析详解

## 目录
1. [ROI1 - 大范围捕获区域](#roi1---大范围捕获区域)
2. [ROI2 - 精确提取区域](#roi2---精确提取区域)
3. [ROI3 - 扩展垂直区域](#roi3---扩展垂直区域)
4. [ROI 坐标系统](#roi-坐标系统)
5. [ROI 数据质量](#roi-数据质量)
6. [ROI 调试技巧](#roi-调试技巧)

---

## ROI1 - 大范围捕获区域

### 用途

ROI1 (Region of Interest 1) 是系统的主要捕获区域，用于：
- 绿线检测和交点计算
- ROI1 波峰检测 (混合检测模式)
- 作为 ROI2 和 ROI3 的提取基础

### 默认配置

```json
"default_config": {
  "x1": 1280,
  "y1": 80,
  "x2": 1920,
  "y2": 980
}
```

**尺寸**: 640 x 900 像素

### 捕获方式

#### 屏幕捕获模式

```python
from PIL import ImageGrab

# 获取屏幕尺寸
screen_width, screen_height = ImageGrab.grab().size

# 调整 ROI1 到屏幕边界
x1, y1, x2, y2 = adjust_roi1_to_screen(
    (screen_width, screen_height),
    default_config
)

# 捕获 ROI1
roi1_image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
```

#### 视频处理模式

```python
import cv2

# 读取视频帧
ret, frame = video_cap.read()

# 调整 ROI1 到视频尺寸
video_height, video_width = frame.shape[:2]
x1, y1, x2, y2 = adjust_roi1_to_screen(
    (video_width, video_height),
    default_config
)

# 裁剪 ROI1
roi1_array = frame[y1:y2, x1:x2]
roi1_image = Image.fromarray(cv2.cvtColor(roi1_array, cv2.COLOR_BGR2RGB))
```

### ROI1 灰度计算

```python
def compute_average_gray(image: Image.Image) -> float:
    """
    计算 ROI1 的平均灰度值

    Returns:
        float: 平均灰度值 [0, 255]
    """
    if image.mode != 'L':
        image = image.convert('L')

    roi1_array = np.array(image)
    avg_gray = float(np.mean(roi1_array))

    return avg_gray
```

### ROI1 缓冲区管理

```python
from collections import deque

# 100 帧循环缓冲
roi1_gray_buffer = deque(maxlen=100)

# 添加新帧
roi1_gray = compute_average_gray(roi1_image)
roi1_gray_buffer.append(roi1_gray)
```

### ROI1 阈值计算

#### 固定阈值

```python
roi1_threshold = 120.0
```

#### 自适应阈值

```python
# 背景均值
roi1_bg_mean = sum(roi1_gray_buffer) / len(roi1_gray_buffer)

# 自适应阈值
roi1_threshold = roi1_bg_mean * (1 + roi1_threshold_over_mean_ratio)
roi1_threshold = max(roi1_threshold, roi1_threshold_minimum)
```

---

## ROI2 - 精确提取区域

### 用途

ROI2 是围绕绿线交点的小范围精确分析区域，用于：
- 精确灰度值计算
- ROI2 波峰检测
- ROI2 颜色判定

### 提取方式

ROI2 基于绿线交点坐标动态提取：

```python
def compute_roi2_region(
    roi1_size: Tuple[int, int],
    center: Tuple[int, int],
    extension_params: Dict[str, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    在 ROI1 内计算 ROI2 区域

    Args:
        roi1_size: ROI1 尺寸 (width, height)
        center: 绿线交点坐标 (x, y) - ROI1 内部坐标
        extension_params: 扩展参数

    Returns:
        (x1, y1, x2, y2) ROI2 在 ROI1 内的坐标
    """
    roi_width, roi_height = roi1_size
    cx, cy = center

    # 限制中心在 ROI1 边界内
    cx = max(0, min(roi_width - 1, cx))
    cy = max(0, min(roi_height - 1, cy))

    # 扩展参数
    left = extension_params.get("left", 20)
    right = extension_params.get("right", 30)
    top = extension_params.get("top", 60)
    bottom = extension_params.get("bottom", 20)

    # 计算边界
    x1 = cx - left
    x2 = cx + right
    y1 = cy - top
    y2 = cy + bottom

    # 限制在 ROI1 边界内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(roi_width, x2)
    y2 = min(roi_height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)
```

### 默认扩展参数

```json
"roi2_config": {
  "extension_params": {
    "left": 20,
    "right": 30,
    "top": 60,
    "bottom": 20
  }
}
```

**结果尺寸**: 50 x 80 像素 (left + right = 50, top + bottom = 80)

### ROI2 提取流程

```python
# 步骤 1: 检测绿线交点
intersection = detect_green_intersection(roi1_image)

# 步骤 2: 应用防抖动滤波
if intersection:
    intersection = intersection_filter.filter(intersection[0], intersection[1])

# 步骤 3: 计算 ROI2 区域
if intersection:
    roi2_region = compute_roi2_region(
        roi1_image.size,
        intersection,
        extension_params
    )

# 步骤 4: 裁剪 ROI2
if roi2_region:
    x1, y1, x2, y2 = roi2_region
    roi2_image = roi1_image.crop((x1, y1, x2, y2))
```

### ROI2 防抖动

#### EMA (指数移动平均)

```python
class IntersectionFilter:
    def __init__(self, alpha=0.25, ...):
        self.alpha = alpha
        self.last_x = None
        self.last_y = None

    def filter_intersection(self, x, y):
        if self.last_x is None:
            # 首次初始化
            self.last_x = x
            self.last_y = y
            return (x, y)

        # EMA 平滑
        filtered_x = self.alpha * x + (1 - self.alpha) * self.last_x
        filtered_y = self.alpha * y + (1 - self.alpha) * self.last_y

        self.last_x = filtered_x
        self.last_y = filtered_y

        return (filtered_x, filtered_y)
```

**参数调优**:
- `alpha = 0.1`: 强平滑，响应慢
- `alpha = 0.25`: 中等平滑 (推荐)
- `alpha = 0.5`: 弱平滑，响应快

#### Threshold Filter (阈值滤波)

```python
def threshold_filter(new_pos, last_pos, movement_threshold=20.0):
    """
    小于阈值时完全静止，超过时直接更新
    """
    dx = abs(new_pos[0] - last_pos[0])
    dy = abs(new_pos[1] - last_pos[1])

    if dx < movement_threshold and dy < movement_threshold:
        return last_pos  # 保持旧位置
    return new_pos  # 直接更新
```

**参数调优**:
- `movement_threshold = 10.0`: 强静止 (小移动忽略)
- `movement_threshold = 20.0`: 中等静止 (推荐)
- `movement_threshold = 50.0`: 弱静止 (大移动忽略)

### ROI2 灰度缓冲

```python
from collections import deque

# 100 帧循环缓冲
roi2_gray_buffer = deque(maxlen=100)

# 添加新帧
roi2_gray = compute_average_gray(roi2_image)
roi2_gray_buffer.append(roi2_gray)
```

---

## ROI3 - 扩展垂直区域

### 用途

ROI3 是在 ROI2 基础上垂直扩展的区域，用于：
- G1/G2 像素百分比计算 (颜色覆盖判定)
- 列灰度差值计算 (颜色覆盖判定)
- 提供额外的颜色分类验证

### 扩展参数

```json
"roi3_config": {
  "extension_params": {
    "left": 20,
    "right": 30,
    "top": 80,
    "bottom": 40
  }
}
```

**与 ROI2 的区别**: 垂直方向扩展更大 (top + bottom = 120 vs ROI2 的 80)

### ROI3 提取

```python
# 计算方式与 ROI2 相同
roi3_region = compute_roi2_region(
    roi1_image.size,
    intersection,
    roi3_extension_params
)

# 裁剪 ROI3
if roi3_region:
    x1, y1, x2, y2 = roi3_region
    roi3_image = roi1_image.crop((x1, y1, x2, y2))
```

---

## ROI3 统计计算

### 1. G1/G2 像素百分比

```python
def compute_roi3_g1_g2_ranges(roi3_image: Image.Image) -> Tuple[float, float]:
    """
    计算 ROI3 中特定灰度范围的像素百分比

    G1: 灰度值 [80, 255] 范围 (高回声)
    G2: 灰度值 [150, 255] 范围 (超高回声)

    Returns:
        (g1_percent, g2_percent)
    """
    roi3_array = np.array(roi3_image.convert('L'))
    total_pixels = roi3_array.size

    # G1: 80-255 范围
    g1_mask = (roi3_array >= 80) & (roi3_array <= 255)
    g1_pixels = np.sum(g1_mask)
    g1_percent = (g1_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # G2: 150-255 范围
    g2_mask = (roi3_array >= 150) & (roi3_array <= 255)
    g2_pixels = np.sum(g2_mask)
    g2_percent = (g2_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return g1_percent, g2_percent
```

**用途**:
- `G1 > 98%` 且 `G2 > 20%`: 红色波峰覆盖为绿色
- 反映高回声像素的覆盖程度

### 2. 列灰度差值

```python
def compute_roi3_column_mean_diff(roi3_image: Image.Image) -> float:
    """
    计算 ROI3 每列平均灰度的最大值与最小值之差

    Returns:
        float: 列灰度差值
    """
    roi3_array = np.array(roi3_image.convert('L'))

    # 计算每列平均灰度
    column_means = np.mean(roi3_array, axis=0)

    # 计算最大值与最小值之差
    max_mean = float(np.max(column_means))
    min_mean = float(np.min(column_means))
    diff = max_mean - min_mean

    return diff
```

**用途**:
- `G1 > 99%` 且 `列灰度差值 > 15`: 红色波峰覆盖为绿色
- 反映垂直方向的颜色梯度

### 3. 归一化灰度百分比 (80-160)

```python
def compute_roi3_80_160_normalized(roi3_image: Image.Image) -> float:
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

## ROI 坐标系统

### 坐标层次

```
屏幕/视频坐标 (绝对坐标)
    │
    ├─ ROI1 坐标 (相对于屏幕/视频)
    │   │
    │   ├─ ROI2 坐标 (相对于 ROI1)
    │   │   │
    │   │   └─ ROI2 内部坐标 (0,0 在左上角)
    │   │
    │   └─ ROI3 坐标 (相对于 ROI1)
    │       │
    │       └─ ROI3 内部坐标 (0,0 在左上角)
    │
    └─ 绿线交点坐标 (ROI1 内部坐标)
```

### 坐标转换

#### 屏幕坐标 → ROI1 坐标

```python
# ROI1 在屏幕上的位置
screen_x1, screen_y1, screen_x2, screen_y2 = roi1_screen_coords

# 屏幕坐标 (screen_x, screen_y) 转换为 ROI1 内部坐标
roi1_x = screen_x - screen_x1
roi1_y = screen_y - screen_y1
```

#### ROI1 坐标 → ROI2 坐标

```python
# ROI2 在 ROI1 内的位置
roi2_x1, roi2_y1, roi2_x2, roi2_y2 = roi2_region_in_roi1

# ROI1 内部坐标 (roi1_x, roi1_y) 转换为 ROI2 内部坐标
roi2_x = roi1_x - roi2_x1
roi2_y = roi1_y - roi2_y1
```

#### ROI1 坐标 → 绝对帧索引

```python
# 缓冲区起始帧索引
buffer_start_frame_index = frame_index - len(roi1_gray_buffer) + 1

# ROI1 内缓冲区索引 (buffer_index) 转换为绝对帧索引
absolute_frame_index = buffer_start_frame_index + buffer_index
```

---

## ROI 数据质量

### ROI2 数据质量评估

```python
def calculate_roi2_data_quality(
    peak_start: int,
    peak_end: int,
    roi2_curve: List[float]
) -> Dict[str, float]:
    """
    评估 ROI2 在波峰区间的数据质量

    Returns:
        quality_score: 质量评分 [0, 1]
        variance: 方差
        frame_count: 有效帧数
    """
    interval_values = roi2_curve[peak_start:peak_end + 1]

    # 方差
    mean_val = sum(interval_values) / len(interval_values)
    variance = sum((x - mean_val) ** 2 for x in interval_values) / len(interval_values)

    # 标准差
    std_dev = math.sqrt(variance)

    # 数据范围
    data_range = max(interval_values) - min(interval_values)

    # 稳定性评分
    stability = max(0, 1.0 - std_dev / max(10.0, data_range))

    # 一致性评分
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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `minimum_roi2_frames` | 15 | 最小帧数要求 |
| `roi2_minimum_variance` | 0.5 | 最小方差要求 |
| `roi2_min_gray` | 5.0 | 最小灰度值 |
| `roi2_max_gray` | 250.0 | 最大灰度值 |

### 质量过滤

```python
# 帧数不足
if frame_count < minimum_roi2_frames:
    return 'invalid', 'insufficient_frames'

# 方差过低 (信号过于平坦)
if variance < roi2_minimum_variance:
    return 'invalid', 'low_variance'

# 灰度值超出范围
if avg_gray < roi2_min_gray or avg_gray > roi2_max_gray:
    return 'invalid', 'gray_out_of_range'
```

---

## ROI 调试技巧

### 1. 启用图像保存

```json
"data_processing": {
  "save_roi1": true,
  "save_roi2": true,
  "save_roi3": true
}
```

**输出**:
- `tmp/.../roi1/roi1_000001.png`
- `tmp/.../roi2/roi2_000001.png`
- `tmp/.../roi3/roi3_000001.png`

### 2. 启用波形保存

```json
"data_processing": {
  "save_wave": true,
  "save_roi1_wave": true
}
```

**输出**:
- `tmp/.../wave/wave_000001.png`
- `tmp/.../wave1/roi1_wave_000001.png`

### 3. 启用分析缓存

```json
"analysis_cache": {
  "enabled": true,
  "flush_every": 50
}
```

**输出**: `export/roi_analysis_cache_{session_id}_{run_id}.jsonl`

**分析**:
```python
import json

with open('roi_analysis_cache_...jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['type'] == 'frame':
            print(f"Frame {data['frame_index']}: "
                  f"ROI1={data['roi1_avg']:.1f}, "
                  f"ROI2={data['roi2_avg']:.1f}, "
                  f"G1={data['roi3_g1_percent']:.1f}%, "
                  f"G2={data['roi3_g2_percent']:.1f}%")
```

### 4. 使用配置 GUI

```bash
python config_gui.py
```

**功能**:
- 可视化 ROI1/ROI2/ROI3 位置
- 实时预览捕获效果
- 调整参数并立即生效

### 5. 检查缓冲区状态

```python
print(f"[DEBUG] ROI1 Buffer: len={len(roi1_gray_buffer)}, "
      f"min={min(roi1_gray_buffer):.1f}, "
      f"max={max(roi1_gray_buffer):.1f}")

print(f"[DEBUG] ROI2 Buffer: len={len(roi2_gray_buffer)}, "
      f"min={min(roi2_gray_buffer):.1f}, "
      f"max={max(roi2_gray_buffer):.1f}")

print(f"[DEBUG] ROI3 G1 Buffer: len={len(roi3_g1_buffer)}, "
      f"latest={roi3_g1_buffer[-1]:.2f}%")
```

### 6. 可视化波峰检测

```python
import matplotlib.pyplot as plt

# 绘制 ROI2 曲线
plt.figure(figsize=(12, 4))
plt.plot(roi2_curve, label='ROI2', color='blue')

# 标记波峰
for start, end in green_peaks:
    plt.axvspan(start, end, alpha=0.3, color='green')
for start, end in red_peaks:
    plt.axvspan(start, end, alpha=0.3, color='red')

# 标记阈值
plt.axhline(threshold, color='orange', linestyle='--', label='Threshold')

plt.legend()
plt.savefig('wave_debug.png')
```

### 7. ROI 有效性检查

```python
def check_roi_validity(roi_image, roi_name):
    """检查 ROI 图像是否有效"""
    if roi_image is None:
        print(f"[ERROR] {roi_name} is None")
        return False

    width, height = roi_image.size
    if width <= 0 or height <= 0:
        print(f"[ERROR] {roi_name} has invalid size: {width}x{height}")
        return False

    print(f"[OK] {roi_name}: {width}x{height}, mode={roi_image.mode}")
    return True

# 使用
check_roi_validity(roi1_image, "ROI1")
check_roi_validity(roi2_image, "ROI2")
check_roi_validity(roi3_image, "ROI3")
```

### 8. 交点调试

```python
# 在 ROI1 上绘制交点
import cv2
import numpy as np

roi1_array = np.array(roi1_image)
if intersection:
    ix, iy = intersection
    # 绘制红点标记交点
    cv2.circle(roi1_array, (ix, iy), 5, (255, 0, 0), -1)
    # 绘制 ROI2 边界
    cv2.rectangle(roi1_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('roi1_debug.png', cv2.cvtColor(roi1_array, cv2.COLOR_RGB2BGR))
```

---

## ROI 参数调优指南

### ROI1 调优

| 场景 | 调整 | 效果 |
|------|------|------|
| 绿线检测失败 | 扩大 ROI1 范围 | 增加绿线捕获概率 |
| ROI1 波峰过多 | 提高 ROI1 阈值 | 减少误检 |
| ROI1 波峰过少 | 降低 ROI1 阈值 | 减少漏检 |

### ROI2 调优

| 场景 | 调整 | 效果 |
|------|------|------|
| ROI2 抖动严重 | 启用防抖动，降低 alpha | 平滑 ROI2 位置 |
| ROI2 响应慢 | 提高 alpha 或 movement_threshold | 更快响应 |
| ROI2 区域太小 | 增加 left/right/top/bottom | 扩大 ROI2 范围 |

### ROI3 调优

| 场景 | 调整 | 效果 |
|------|------|------|
| G1/G2 覆盖过多 | 提高 g1_threshold/g2_threshold | 减少覆盖 |
| G1/G2 覆盖过少 | 降低 g1_threshold/g2_threshold | 增加覆盖 |
| 列差值覆盖过多 | 提高 column_diff_threshold | 减少覆盖 |
| 列差值覆盖过少 | 降低 column_diff_threshold | 增加覆盖 |
