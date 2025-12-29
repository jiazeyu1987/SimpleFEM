# HEM 检测系统概述

## 系统简介

SimpleFEM HEM (High Echo Event - 高回声事件) 检测系统是一个独立的医学信号处理守护进程，用于从实时屏幕捕获或视频文件中检测和分析高回声事件。该系统专为医疗影像分析设计，提供可靠的波峰检测、分类和统计功能。

## 核心功能

### 1. ROI 捕获系统

系统使用三层 ROI (Region of Interest) 架构进行信号采集：

#### ROI1 - 大范围捕获区域
- **用途**: 绿线检测和初始波峰检测
- **默认尺寸**: 1280x80 到 1920x980 像素
- **特性**:
  - 支持屏幕实时捕获 (PIL.ImageGrab)
  - 支持视频文件处理 (OpenCV)
  - 自动边界调整，确保 ROI 在屏幕/视频范围内
  - 独立的灰度缓冲区 (100 帧循环缓冲)

#### ROI2 - 精确提取区域
- **用途**: 围绕绿线交点的小范围精确分析
- **默认尺寸**: 约 80x120 像素
- **提取方式**:
  - 基于绿线交点坐标动态提取
  - 使用 `extension_params` 配置扩展参数
  - 支持防抖动滤波 (EMA/Threshold/Velocity)
- **特性**:
  - 高精度灰度值计算
  - 100 帧循环缓冲区用于波峰检测
  - 支持交点缺失时使用上一有效位置

#### ROI3 - 扩展垂直区域
- **用途**: 颜色分类验证和覆盖判定
- **提取方式**: 在 ROI2 基础上垂直扩展
- **统计指标**:
  - **G1 范围 (80-255)**: 高回声像素百分比
  - **G2 范围 (150-255)**: 超高回声像素百分比
  - **列灰度差值**: 每列平均灰度的最大值与最小值之差
  - **归一化值 (80-160)**: 特定灰度范围的像素百分比

### 2. 绿线检测系统

使用 OpenCV 进行绿线交点检测：

```python
# HSV 颜色空间过滤
hsv = cv2.cvtColor(roi1_array, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Canny 边缘检测
edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)

# Hough 直线变换
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=50, minLineLength=80, maxLineGap=20)
```

**防抖动机制**:
- **EMA (指数移动平均)**: 平滑交点位置变化
- **Velocity Filter**: 拒绝超过速度阈值的快速移动
- **Threshold Filter**: 基于运动阈值的固定位置

### 3. 波峰检测系统

#### 传统 ROI2 检测
- **自适应阈值**: 基于背景均值的动态阈值计算
- **阈值保护**: 防止波峰数据污染背景计算
- **颜色分类**:
  - 绿色: `post_avg - pre_avg >= difference_threshold`
  - 红色: 所有其他波峰

#### 混合检测模式 (Hybrid Detection)
- **ROI1 检测波峰时机**: 使用 ROI1 独立阈值检测波峰区间
- **ROI2 判定颜色**: 在 ROI1 检测的波峰区间内用 ROI2 进行颜色判定
- **ROI3 覆盖机制**: 使用 G1/G2 和列灰度差值覆盖颜色判定

**ROI3 覆盖规则**:
1. **G1/G2 覆盖**:
   - 条件: `G1 > 98%` 且 `G2 > 20%`
   - 效果: 红色波峰覆盖为绿色
2. **列灰度差值覆盖**:
   - 条件: `G1 > 99%` 且 `列灰度差值 > threshold`
   - 效果: 红色波峰覆盖为绿色

### 4. 阈值保护机制

防止波峰数据污染自适应阈值的背景计算：

```python
# 触发条件
1. 波形触发: 当前灰度 >= 当前阈值
2. 波峰触发: 检测到波峰

# 解除条件
1. 时间延迟: recovery_delay_frames (默认 10 帧)
2. 稳定性检查: 连续 stability_frames (默认 5 帧) 低于阈值
```

**保护期间行为**:
- 冻结背景均值计算
- 使用最后的有效阈值
- 不更新背景统计

### 5. 三层去重系统

```
Layer 1: 最近波峰比较 (5 帧窗口)
   ↓
Layer 2: 连续帧去重 (40 帧窗口，同色波峰)
   ↓
Layer 3: 跨色去重 (不同色波峰，绿色优先)
```

**颜色优先级**:
- 绿色波峰: Priority 2 (高)
- 红色波峰: Priority 1 (低)

### 6. 数据导出系统

#### Analysis Cache (JSONL)
每帧分析数据缓存，用于调试和回溯分析：
```jsonl
{"type":"meta","cache_version":1,"created_at":"2025-12-26T10:30:00","session_id":"session_001",...}
{"type":"frame","frame_index":349,"roi1_avg":52.3,"roi2_avg":142.3,"intersection":{"x":1380,"y":150},...}
{"type":"session_end","ended_at":"2025-12-26T10:35:00","reason":"video_complete"}
```

#### CSV 导出
波峰统计数据导出，包含完整元数据：
```csv
timestamp,frame_index,peak_start,peak_end,peak_color,peak_value,pre_avg,post_avg,...
2025-12-26T10:30:45,349,345,350,green,142.3,45.2,67.8,...
```

## 处理模式

### 1. 屏幕捕获模式
- 实时屏幕捕获
- 使用 PIL.ImageGrab
- 适用于实时监控

### 2. 视频处理模式
- 支持单个视频文件
- 支持批量视频处理（文件夹）
- 自动视频切换
- 帧步长控制

### 3. 静脉跟随模式
- 基于连通组件分析的自动 ROI 跟踪
- 适用于静脉检测场景

## 配置管理

### 配置文件层次
1. **JSON 配置文件**: `simple_fem_config.json`
2. **环境变量覆盖**: 使用 `NHEM_*` 前缀
3. **运行时配置**: 无需重启即可应用

### 关键配置参数

| 参数 | 位置 | 用途 | 典型范围 |
|-----|------|-----|---------|
| `threshold` | `peak_detection.threshold` | 基础检测阈值 | 30-100 |
| `difference_threshold` | `peak_detection.difference_threshold` | 绿/红分类阈值 | 1.5-3.0 |
| `frame_rate` | `roi_capture.frame_rate` | 捕获帧率 | 5-30 FPS |
| `adaptive_threshold_enabled` | `peak_detection.adaptive_threshold_enabled` | 启用自适应阈值 | true/false |
| `hybrid_detection.enabled` | `hybrid_detection.enabled` | 启用混合检测 | true/false |

## 输出目录结构

```
SimpleFEM/
├── export/                    # 导出数据
│   ├── peak_statistics_*.csv  # 波峰统计
│   └── roi_analysis_cache_*.jsonl  # 分析缓存
├── roi1/                      # ROI1 捕获
├── roi2/                      # ROI2 提取
├── roi3/                      # ROI3 扩展
├── wave/                      # 波形图
└── logs/                      # 日志文件
```

## 技术栈

- **Python 3.7+**
- **OpenCV**: 计算机视觉和图像处理
- **PIL/Pillow**: 屏幕捕获和图像操作
- **NumPy**: 数值计算
- **Matplotlib**: 波形可视化
- **JSON**: 配置和数据导出

## 性能指标

- **处理帧率**: 1-30 FPS (可配置)
- **处理延迟**: <100ms/帧
- **内存占用**: 固定循环缓冲 (100 帧)
- **CPU 使用**: 优化的连续医学监控

## 医学应用特性

- **HEM 检测**: 高回声事件检测
- **实时分析**: 连续监控，可配置灵敏度
- **数据完整性**: 审计日志和导出功能
- **临床使用**: 适用于研究和诊断支持
- **质量保证**: 去重和验证机制确保可靠数据
