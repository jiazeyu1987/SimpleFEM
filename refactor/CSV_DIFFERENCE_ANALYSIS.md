# 第一行数据差异分析报告

## 对比文件

| 文件 | 时间戳 | 说明 |
|------|--------|------|
| 新CSV | 20251229_093536 | 重构代码生成 |
| 旧CSV | 20251228_163652 | 原始代码生成 |

## 差异总结

### 1. 表头（字段名）
**结论：完全相同** ✓
- 两个CSV都有22个字段
- 字段名称和顺序完全一致

### 2. 第一行数据

| 字段 | 新CSV | 旧CSV | 差异 |
|------|-------|-------|------|
| **帧索引** | 175 | 93 | 相差82帧 |
| **峰值颜色** | red | red | 相同 |
| **G1值** | *(empty)* | 100.0 | **缺失** |
| **G2值** | *(empty)* | 0.74 | **缺失** |
| **列差值** | *(empty)* | 11.25 | **缺失** |
| **阈值** | 87.282 | 40.0 | 自适应vs固定 |
| **背景均值** | 79.348 | 0.0 | 不同 |
| **G1/G2覆盖** | False | False | 相同 |

## 根本原因分析

### 问题1: G1/G2/列差值为空

**原因：**
重构代码的 `hybrid_detection_manager.py` 只在覆盖条件满足时才记录G1/G2值：

```python
# 旧代码逻辑（错误）
if g1_value >= g1_threshold and g2_value >= g2_threshold:
    g1_value_used = g1_value  # 只在覆盖时赋值
    g2_value_used = g2_value
```

**问题：**
- 帧175的G1=45.32%, G2=4.79%
- 阈值要求: G1>=98%, G2>=20%
- 覆盖条件不满足
- 因此 `g1_value_used` 和 `g2_value_used` 保持为 None
- CSV中显示为空

**修复：**
修改 `hybrid_detection_manager.py`，始终记录实际的G1/G2值：

```python
# 新代码逻辑（正确）
# 始终记录实际的G1/G2值（无论是否应用覆盖）
g1_value_used = g1_value
g2_value_used = g2_value

# 然后检查是否需要应用覆盖
if g1_value >= g1_threshold and g2_value >= g2_threshold:
    g1_g2_override_applied = True
    if initial_color == 'red':
        final_color = 'green'
```

同样的问题也存在于 `column_diff_value`。

### 问题2: 帧索引不同（175 vs 93）

**可能原因：**

1. **峰值检测模式不同**
   - 新CSV实际使用了混合检测（缓存显示 hybrid_red_peaks=1）
   - 但峰值检测时机可能因ROI1缓冲区填充逻辑不同而变化

2. **ROI1缓冲区填充时机**
   - 我们最近修改了代码：只在 `roi1_enabled=True` 时填充ROI1缓冲区
   - 这可能影响峰值检测的时机

3. **需要进一步验证**
   - 对比完整的峰值列表
   - 检查峰值区间是否一致（只是帧索引偏移）

## 验证数据

### 分析缓存（roi_analysis_cache_*.jsonl）

帧175的缓存数据显示：
```
roi3_g1_percent: 45.3222%
roi3_g2_percent: 4.7889%
roi3_column_diff: 15.6467
hybrid_red_peaks: 1
hybrid_detection_enabled: True
```

**关键发现：**
- ✓ 混合检测确实工作
- ✓ ROI3统计被计算
- ✗ 但CSV中G1/G2字段为空

## 已应用的修复

### 1. 修复G1/G2记录逻辑

**文件：** `refactor/hybrid_detection_manager.py`
**修改：** 始终记录实际的G1/G2值，无论是否应用覆盖

**代码：**（line 259-263）
```python
# 始终记录实际的G1/G2值（无论是否应用覆盖）
g1_value_used = g1_value
g2_value_used = g2_value
if valid_frame_indices:
    g1_g2_override_frame_idx = valid_frame_indices[max_g1_idx]
```

### 2. 修复列差值记录逻辑

**文件：** `refactor/hybrid_detection_manager.py`
**修改：** 始终记录实际的列差值，无论是否应用覆盖

**代码：**（line 293-296）
```python
# 始终记录实际的列灰度差值（无论是否应用覆盖）
column_diff_value_used = max_column_diff
if valid_frame_indices:
    column_diff_override_frame_idx = valid_frame_indices[max_idx]
```

## 预期效果

应用修复后，重新生成的CSV应该：
1. ✓ G1/G2字段有值（例如：45.32, 4.79）
2. ✓ 列差值字段有值（例如：15.65）
3. ✓ g1_g2_override_applied = False（因为不满足阈值）
4. ✓ 帧索引可能仍与旧CSV不同（需要进一步调查）

## 待验证问题

1. **帧索引差异：** 为什么新代码在帧175检测到峰值，而旧代码在帧93？
   - 检查ROI1/ROI2曲线数据
   - 对比峰值区间
   - 验证是否只是帧索引偏移

2. **峰值数量：** 新CSV有3个峰值，旧CSV有5个峰值
   - 需要对比完整的峰值列表
   - 检查是否有峰值被去重逻辑过滤

## 建议

1. **重新运行测试：** 使用修复后的代码重新处理视频2
2. **对比完整CSV：** 检查所有峰值，而不仅仅是第一个
3. **检查去重逻辑：** 确认新代码的去重参数与旧代码一致
