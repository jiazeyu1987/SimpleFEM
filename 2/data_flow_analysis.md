# 数据流路径完整分析

## 用户案例：frame=93, roi3_peak_max_frame=81

### 1. ROI2颜色判定
**位置**: `simple_roi_daemon.py:753`
```python
color = "green" if frame_difference >= color_threshold else "red"
```
- `frame_difference = -0.072`
- `color_threshold = 1.5`
- 结果: `color = 'green'` ✓

### 2. 混合检测执行
**位置**: `simple_roi_daemon.py:1757-1760`
```python
hybrid_peaks = hybrid_peak_detection(
    roi1_curve, roi2_curve, hybrid_config_with_frame,
    processed_roi1_peaks, roi1_peak_counter
)
```
- ROI1检测到0个波峰
- `hybrid_peaks = []` (空列表)

### 3. 统计系统调用
**位置**: `simple_roi_daemon.py:1918-1938`
```python
stats_write_results = current_stats.add_peaks_from_daemon(
    frame_index=frame_index,
    green_peaks=green_peaks,      # ← ROI2独立检测的波峰
    red_peaks=red_peaks,
    hybrid_enabled=True,
    hybrid_peaks=hybrid_peaks,     # ← 空列表 []
    roi3_curve=list(roi3_gray_buffer),  # ← 可能有数据
    roi3_override_enabled=True
)
```

### 4. 统计系统内部处理
**位置**: `safe_peak_statistics.py:210-293`

#### 4a. 混合检测路径（未执行）
```python
for hybrid_peak in hybrid_peaks:  # ← hybrid_peaks=[]，循环不执行
    ...
```

#### 4b. 传统模式路径（执行）
```python
# 添加绿色波峰
for i, (start, end) in enumerate(green_peaks):
    peak_data = self._create_peak_data(
        timestamp, frame_index, "green", start, end,
        curve, intersection, roi2_info, gray_value,
        difference_threshold, pre_post_avg_frames,
        threshold_used, bg_mean,
        roi3_curve, roi3_override_enabled, roi3_override_threshold  # ← ROI3参数
    )
```

### 5. ROI3覆盖逻辑（_create_peak_data）
**位置**: `safe_peak_statistics.py:394-411`
```python
if roi3_curve and roi3_override_enabled:
    roi3_peak_max_value, roi3_max_curve_idx = self._get_peak_max_value(roi3_curve, start_frame, end_frame)
    roi3_peak_max_frame = curve_start_global_frame + roi3_max_curve_idx

    # RED -> GREEN
    if peak_type == "red" and roi3_peak_max_value > roi3_override_threshold:
        final_peak_type = "green"

    # GREEN -> RED
    elif final_peak_type == "green" and roi3_peak_max_frame < 110:  # ← 应该触发
        final_peak_type = "red"
        print(f"[DEBUG] ROI3 early-peak override applied...")
```

### 6. 写入CSV
**位置**: `safe_peak_statistics.py:_write_peak_to_csv`
```python
writer.writerow({
    'peak_type': peak_data['peak_type'],  # ← 使用覆盖后的颜色
    'roi3_peak_max_frame': peak_data['roi3_peak_max_frame'],
    ...
})
```

## 问题诊断

### 预期行为
1. ROI2判定为绿色
2. ROI3峰值位置=81 < 110
3. 触发 GREEN->RED 覆盖
4. CSV中应该是红色

### 实际结果
CSV中仍然是绿色

### 可能原因

#### 1. roi3_curve 为空
```python
roi3_curve=list(roi3_gray_buffer) if roi3_gray_buffer else []
```
- 如果 `roi3_gray_buffer` 为空列表 `[]`
- 条件 `if roi3_curve` 为 False
- ROI3覆盖逻辑不会执行 ✓ **最可能的原因**

#### 2. roi3_override_enabled 实际为False
虽然配置显示True，但可能传递时出错

#### 3. 调试日志未看到
添加了调试日志但用户可能没重新运行

## 验证方法

添加的调试日志应该显示：
```
[DEBUG ROI3] frame=93, initial_color=green, roi3_override_enabled=True, roi3_curve_len=100
[DEBUG ROI3] roi3_peak_max=133.42, roi3_peak_max_frame=81, threshold=111.0
[DEBUG ROI3] GREEN->RED override: frame=93, roi3_max_frame=81 < 110
[DEBUG ROI3] final_color=red, override_applied=True
```

如果 `roi3_curve_len=0`，说明ROI3缓冲区为空，覆盖逻辑无法执行。
