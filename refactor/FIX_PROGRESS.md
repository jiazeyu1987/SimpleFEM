# SimpleFEM重构代码修复进度

**目标**: 确保重构代码（refactor/）与原始代码（simple_roi_daemon.py）完全一致

**验证标准**: export/与export_2/的CSV文件完全一致

---

## ✅ 已完成的修复

### 1. G1/G2值记录逻辑 (P0 - 已完成)
**问题**: hybrid_detection_manager.py只在覆盖条件满足时记录G1/G2值
- 当G1<98%或G2<20%时，变量保持为None
- 导致CSV中G1/G2字段为空

**修复**: 修改hybrid_detection_manager.py (line 259-270)
```python
# 修复前：只在覆盖时赋值
if g1_value >= g1_threshold and g2_value >= g2_threshold:
    g1_value_used = g1_value

# 修复后：始终记录实际值
g1_value_used = g1_value
g2_value_used = g2_value
if valid_frame_indices:
    g1_g2_override_frame_idx = valid_frame_indices[max_g1_idx]

# 然后检查是否需要覆盖
if g1_value >= g1_threshold and g2_value >= g2_threshold:
    g1_g2_override_applied = True
```

**文件**: `refactor/hybrid_detection_manager.py`
**状态**: ✅ 已完成，待验证

---

### 2. 列差值记录逻辑 (P0 - 已完成)
**问题**: hybrid_detection_manager.py只在覆盖条件满足时记录列差值
- 当列差<15.0时，变量保持为None
- 导致CSV中column_diff_value字段为空

**修复**: 修改hybrid_detection_manager.py (line 293-303)
```python
# 修复前：只在覆盖时赋值
if max_column_diff >= column_diff_threshold:
    column_diff_value_used = max_column_diff

# 修复后：始终记录实际值
column_diff_value_used = max_column_diff
if valid_frame_indices:
    column_diff_override_frame_idx = valid_frame_indices[max_idx]

# 然后检查是否需要覆盖
if max_column_diff >= column_diff_threshold:
    column_diff_override_applied = True
```

**文件**: `refactor/hybrid_detection_manager.py`
**状态**: ✅ 已完成，待验证

---

### 3. ROI3列差值缓冲区实现 (P0 - 已完成)
**问题**: orchestrator.py line 357: `roi3_column_diff_curve=None`
- 列差值缓冲区未实现
- 导致混合检测无法使用列差值覆盖

**修复**:
1. 添加`roi3_column_diff_buffer`到roi_capture_manager.py (line 60)
2. 添加property方法 (line 396-399)
3. 在reset_buffers中清空 (line 440)
4. 在orchestrator.py中填充 (line 323-324)
5. 传递给混合检测 (line 360)

**文件**:
- `refactor/roi_capture_manager.py`
- `refactor/orchestrator.py`

**状态**: ✅ 已完成，待验证

---

### 4. ROI1缓冲区条件填充 (P1 - 已完成)
**问题**: orchestrator.py无条件填充ROI1缓冲区
- 原始代码只在roi1_enabled=True时填充
- 导致行为不一致

**修复**: 修改orchestrator.py (line 281-286)
```python
# 修复前：无条件填充
roi1_gray = self._roi_capture.compute_average_gray(roi1_image)
self._roi_capture.roi1_buffer.append(roi1_gray)

# 修复后：条件填充
roi1_enabled = self._config.roi1_peak_detection_enabled
roi1_gray: Optional[float] = None
if roi1_enabled:
    roi1_gray = self._roi_capture.compute_average_gray(roi1_image)
    self._roi_capture.roi1_buffer.append(roi1_gray)
```

**文件**: `refactor/orchestrator.py`
**状态**: ✅ 已完成

---

## ⚠️ 待修复的问题

### 5. 峰值数量差异 (P1 - 诊断中)
**问题**:
- 新CSV: 只有1个峰值（帧175）
- 旧CSV: 有5个峰值（帧93, 349, 408, 498, 979）

**分析**:
- 处理了188帧，其中88帧检测到峰值
- 但去重逻辑过滤后只剩1个峰值
- 说明去重逻辑过于严格或有bug

**可能原因**:
1. 连续帧去重窗口设置不当
2. ROI1缓冲区为空导致混合检测无法工作
3. 峰值边界计算不同导致去重误判

**状态**: ⚠️ 需要深入诊断

---

### 6. frame_index记录差异 (P1 - 待修复)
**问题**:
- 新CSV第一个峰值: 帧175
- 旧CSV第一个峰值: 帧93
- 相差82帧

**根本原因**: 去重逻辑选择的代表帧不同
- 缓存显示帧88-97检测到同一个峰值
- 新代码选择区间结束帧（帧175）
- 旧代码选择区间开始帧（帧93）

**影响**: 虽然是同一个峰值，但frame_index不同

**状态**: ❌ 待修复

---

### 7. 阈值计算差异 (P2 - 待修复)
**问题**:
- 新CSV: threshold_used=87.282 (自适应阈值)
- 旧CSV: threshold_used=40.0 (固定阈值)

**可能原因**:
- 自适应阈值启用状态不同
- 背景均值计算方式不同

**状态**: ❌ 待修复

---

### 8. 背景均值差异 (P2 - 待修复)
**问题**:
- 新CSV: bg_mean=79.348
- 旧CSV: bg_mean=0.0

**可能原因**:
- 背景均值初始化方式不同
- 或更新时机不同

**状态**: ❌ 待修复

---

### 9. ROI3统计精度差异 (P2 - 待修复)
**问题**:
- roi3_peak_max_value: 126.1 vs 127.46
- roi3_peak_max_frame: 82 vs 81

**说明**: ROI3统计计算有细微差异，可能是算法实现差异

**状态**: ❌ 待修复

---

## 📋 待验证项

### 验证G1/G2/列差值修复
- [ ] 重新运行代码生成新CSV
- [ ] 检查所有峰值的G1/G2字段是否有值
- [ ] 检查所有峰值的column_diff_value字段是否有值
- [ ] 验证值的大小与旧CSV相近

### 验证列差值缓冲区
- [ ] 检查roi3_column_diff_buffer是否有数据
- [ ] 验证混合检测是否使用了列差值数据
- [ ] 确认列差值覆盖逻辑正常工作

---

## 🔍 下一步行动

### 优先级1: 诊断峰值数量问题
1. 检查ROI1缓冲区是否为空
2. 检查混合检测是否正常工作
3. 检查去重逻辑参数（consecutive_frame_window等）
4. 对比原始代码和重构代码的去重实现

### 优先级2: 统一frame_index记录逻辑
1. 研究原始代码如何选择代表帧
2. 修改重构代码的记录时机
3. 确保与原始代码一致

### 优先级3: 统一阈值和背景均值计算
1. 检查自适应阈值配置
2. 统一背景均值初始化逻辑
3. 确保计算结果一致

---

## 📊 当前进度

**总任务数**: 11
**已完成**: 4 (36%)
**进行中**: 1 (9%)
**待修复**: 6 (55%)

**完成度**: ████████░░░░░░░░░░░░ 36%

---

**最后更新**: 2025-12-29 10:15
**负责人**: Claude
**文档版本**: v1.0
