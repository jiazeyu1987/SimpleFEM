# SimpleFEM 关键参数修复总结

## 修复时间
2025-12-29 10:25

## 问题诊断

### 症状
- 新CSV只有1个峰值（帧175）
- 旧CSV有5个峰值（帧93, 349, 408, 498, 979）
- 处理了188帧，88帧检测到峰值，但去重后只剩1个

### 根本原因
1. **峰值滑动问题**: 检测到77种不同的峰值边界（如[[81,87]], [[80,86]], [[79,85]]等）
2. **参数缺失**: detect_peaks调用缺少avgFrames参数，影响峰值边界计算

## 已完成的修复

### ✅ 1. 添加avgFrames参数到ROI2独立检测

**文件**: `refactor/orchestrator.py` (line 386)

**问题**: 缺少avgFrames参数，影响峰值边界和前后均值计算

**修复前**:
```python
green_peaks, red_peaks = detect_peaks(
    list(self._roi_capture.roi2_buffer),
    threshold,
    difference_threshold=self._config.difference_threshold,
    margin_frames=self._config.margin_frames,
    silence_frames=self._config.silence_frames,
    min_region_length=self._config.min_region_length
    # ← 缺少 avgFrames
)
```

**修复后**:
```python
green_peaks, red_peaks = detect_peaks(
    list(self._roi_capture.roi2_buffer),
    threshold,
    difference_threshold=self._config.difference_threshold,
    margin_frames=self._config.margin_frames,
    silence_frames=self._config.silence_frames,
    avgFrames=self._config.pre_post_avg_frames,  # ← 添加此参数
    min_region_length=self._config.min_region_length
)
```

---

### ✅ 2. 添加avgFrames参数到ROI1混合检测

**文件**: `refactor/hybrid_detection_manager.py` (line 81)

**问题**: ROI1峰值检测缺少avgFrames参数

**修复前**:
```python
roi1_green_raw, roi1_red_raw = detect_peaks(
    roi1_curve,
    threshold=self._config.roi1_threshold,
    difference_threshold=999.0,
    margin_frames=self._config.margin_frames,
    silence_frames=self._config.silence_frames,
    min_region_length=self._config.min_region_length
    # ← 缺少 avgFrames
)
```

**修复后**:
```python
roi1_green_raw, roi1_red_raw = detect_peaks(
    roi1_curve,
    threshold=self._config.roi1_threshold,
    difference_threshold=999.0,
    margin_frames=self._config.margin_frames,
    silence_frames=self._config.silence_frames,
    avgFrames=self._config.pre_post_avg_frames,  # ← 添加此参数
    min_region_length=self._config.min_region_length
)
```

---

## 参数说明

### avgFrames的作用
- 用于计算峰值前后的平均值
- 默认值: 5帧
- 影响green/red分类
- 影响峰值边界的确定

### 为什么重要
1. **峰值边界计算**: avgFrames影响峰值区域前后的均值计算
2. **颜色分类**: green/red基于前后均值差，avgFrames直接影响分类
3. **去重逻辑**: 峰值的max_value基于边界内的数据，avgFrames影响边界

---

## 预期效果

### 修复前
- 检测到77种不同的峰值边界（滑动现象）
- 去重后只剩1个峰值
- 无法正确识别连续峰值

### 修复后
- avgFrames参数统一峰值边界计算逻辑
- 减少峰值滑动现象
- 去重逻辑应该能正确合并连续峰值
- 应该能检测到5个峰值（与旧CSV一致）

---

## 验证步骤

1. **重新运行代码**
   ```bash
   cd D:\ProjectPackage\SimpleFEM
   python refactor/main.py
   ```

2. **检查生成的CSV**
   - 峰值数量应该是5个
   - 帧索引应该接近 [93, 349, 408, 498, 979]
   - G1/G2/列差值字段应该有值

3. **对比字段值**
   - pre_peak_avg/post_avg应该与旧CSV接近
   - threshold_used、bg_mean应该更一致

---

## 相关文件修改

1. `refactor/orchestrator.py` - ROI2检测添加avgFrames参数
2. `refactor/hybrid_detection_manager.py` - ROI1检测添加avgFrames参数
3. `refactor/roi_capture_manager.py` - 添加roi3_column_diff_buffer
4. `refactor/hybrid_detection_manager.py` - G1/G2/列差值始终记录

---

## 待修复问题

### P1-2: frame_index差异
- 新: 帧175
- 旧: 帧93
- 差: 82帧
- 原因: 去重逻辑选择不同的代表帧

### P2-1: 阈值计算差异
- 新: 87.282 (自适应)
- 旧: 40.0 (固定)

### P2-2: 背景均值差异
- 新: 79.348
- 旧: 0.0

### P2-3: ROI3统计精度差异
- roi3_peak_max_value: 126.1 vs 127.46
- roi3_peak_max_frame: 82 vs 81

---

**最后更新**: 2025-12-29 10:30
**修复状态**: 参数修复完成，待验证
