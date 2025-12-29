# HEM 检测系统配置参考

## 配置文件结构

SimpleFEM 使用 JSON 格式的配置文件 `simple_fem_config.json`，支持环境变量覆盖（`NHEM_*` 前缀）。

## 完整配置示例

```json
{
  "processing_mode": "video",
  "data_processing": {
    "save_roi1": true,
    "save_roi2": true,
    "save_roi3": true,
    "save_wave": true,
    "save_roi1_wave": true,
    "only_delect": false
  },
  "analysis_cache": {
    "enabled": true,
    "flush_every": 50
  },
  "video_processing": {
    "video_path": "video.mp4",
    "loop_enabled": false,
    "processing_frame_rate": 10.0
  },
  "roi_capture": {
    "frame_rate": 10,
    "default_config": {
      "x1": 1280,
      "y1": 80,
      "x2": 1920,
      "y2": 980
    },
    "roi2_config": {
      "x1": 100,
      "y1": 100,
      "x2": 150,
      "y2": 150,
      "extension_params": {
        "left": 20,
        "right": 30,
        "top": 60,
        "bottom": 20
      }
    },
    "roi2_anti_jitter": {
      "enabled": true,
      "algorithm": "ema",
      "movement_threshold": 20.0,
      "ema": {
        "alpha": 0.25,
        "stability_threshold": 8.0,
        "initialization_frames": 3
      },
      "threshold": {
        "movement_threshold": 20.0
      }
    },
    "roi3_config": {
      "extension_params": {
        "left": 20,
        "right": 30,
        "top": 80,
        "bottom": 40
      }
    }
  },
  "peak_detection": {
    "threshold": 95.0,
    "threshold_minimum": 80.0,
    "margin_frames": 5,
    "silence_frames": 15,
    "difference_threshold": 2.1,
    "min_region_length": 5,
    "adaptive_threshold_enabled": true,
    "threshold_over_mean_ratio": 0.15,
    "adaptive_window_seconds": 3.0,
    "threshold_protection": {
      "enabled": true,
      "recovery_delay_seconds": 1.0,
      "stability_frames": 5,
      "waveform_trigger_enabled": true
    },
    "g1_g2_override": {
      "enabled": true,
      "g1_threshold": 98.0,
      "g2_threshold": 20.0,
      "use_peak_max": true
    },
    "roi3_column_diff_override": {
      "enabled": true,
      "threshold": 15.0,
      "use_peak_max": true
    }
  },
  "roi1_peak_detection": {
    "enabled": false,
    "threshold": 120.0,
    "threshold_minimum": 110.0,
    "margin_frames": 5,
    "silence_frames": 5,
    "difference_threshold": 2.0,
    "min_region_length": 5,
    "adaptive_threshold_enabled": true,
    "threshold_over_mean_ratio": 0.08,
    "adaptive_window_seconds": 3.0,
    "threshold_protection": {
      "enabled": true,
      "recovery_delay_seconds": 1.0,
      "stability_frames": 5,
      "waveform_trigger_enabled": true
    }
  },
  "hybrid_detection": {
    "enabled": false,
    "detection_strategy": "roi1_peaks_roi2_color",
    "fusion_strategy": "roi2_priority",
    "require_intersection": true,
    "roi2_color_frames": {
      "pre_peak": 5,
      "post_peak": 10
    },
    "roi1_peak_width_range": [30, 40],
    "data_quality": {
      "minimum_roi2_frames": 15,
      "roi2_minimum_variance": 0.5,
      "roi2_min_gray": 5.0,
      "roi2_max_gray": 250.0,
      "skip_peaks_when_roi2_invalid": true
    },
    "fallback_enabled": true
  },
  "deduplication": {
    "consecutive_frame_window": 40,
    "color_priority": ["green", "red"],
    "recent_peak_window": 5,
    "cross_color_deduplication_enabled": true
  },
  "startup_cleanup": {
    "enabled": true,
    "cleanup_export": true,
    "cleanup_tmp": true,
    "cleanup_logs": false
  }
}
```

## 配置节详解

### 1. processing_mode

处理模式选择

**可选值**:
- `"screen"`: 屏幕实时捕获模式
- `"video"`: 视频文件处理模式
- `"vein_following"`: 静脉跟随模式

**示例**:
```json
"processing_mode": "video"
```

**环境变量覆盖**:
```bash
export NHEM_PROCESSING_MODE=video
```

---

### 2. data_processing

数据导出控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_roi1` | boolean | false | 保存 ROI1 大范围捕获图像 |
| `save_roi2` | boolean | true | 保存 ROI2 精确提取图像 |
| `save_roi3` | boolean | true | 保存 ROI3 扩展垂直区域图像 |
| `save_wave` | boolean | true | 保存 ROI2 波形图 |
| `save_roi1_wave` | boolean | false | 保存 ROI1 波形图 |
| `only_delect` | boolean | false | 仅保存检测到波峰的帧数据 |

**示例**:
```json
"data_processing": {
  "save_roi1": true,
  "save_roi2": true,
  "save_roi3": true,
  "save_wave": true,
  "only_delect": false
}
```

---

### 3. analysis_cache

分析缓存配置 (JSONL 格式)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 启用分析缓存 |
| `flush_every` | integer | 50 | 每N帧刷新一次缓存文件 |

**示例**:
```json
"analysis_cache": {
  "enabled": true,
  "flush_every": 50
}
```

**输出文件**: `export/roi_analysis_cache_{session_id}_{run_id}.jsonl`

---

### 4. video_processing

视频处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `video_path` | string | - | 视频文件路径或文件夹路径 |
| `loop_enabled` | boolean | false | 循环播放视频 |
| `processing_frame_rate` | float | 10.0 | 处理帧率覆盖 |

**示例** (单个视频):
```json
"video_processing": {
  "video_path": "path/to/video.mp4",
  "loop_enabled": false,
  "processing_frame_rate": 10.0
}
```

**示例** (批量视频):
```json
"video_processing": {
  "video_path": "path/to/video_folder",
  "loop_enabled": false
}
```

**支持的视频格式**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`

---

### 5. roi_capture

ROI 捕获配置

#### 5.1 frame_rate

捕获帧率 (1-30 FPS)

**示例**:
```json
"roi_capture": {
  "frame_rate": 10
}
```

#### 5.2 default_config (ROI1)

ROI1 大范围捕获区域坐标

| 参数 | 类型 | 说明 |
|------|------|------|
| `x1, y1` | integer | 左上角坐标 |
| `x2, y2` | integer | 右下角坐标 |

**示例**:
```json
"default_config": {
  "x1": 1280,
  "y1": 80,
  "x2": 1920,
  "y2": 980
}
```

#### 5.3 roi2_config (ROI2)

ROI2 精确提取区域配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `extension_params` | object | 基于交点向四周扩展的参数 |
| `extension_params.left` | integer | 向左扩展像素 |
| `extension_params.right` | integer | 向右扩展像素 |
| `extension_params.top` | integer | 向上扩展像素 |
| `extension_params.bottom` | integer | 向下扩展像素 |

**示例**:
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

#### 5.4 roi2_anti_jitter

ROI2 防抖动配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `enabled` | boolean | 启用防抖动 |
| `algorithm` | string | 算法类型: `"ema"` 或 `"threshold"` |
| `movement_threshold` | float | 运动阈值 (像素) |
| `ema.alpha` | float | EMA 平滑因子 (0.05-0.95) |
| `ema.stability_threshold` | float | EMA 稳定阈值 (像素) |
| `ema.initialization_frames` | integer | 初始化帧数 |

**EMA 模式示例**:
```json
"roi2_anti_jitter": {
  "enabled": true,
  "algorithm": "ema",
  "movement_threshold": 20.0,
  "ema": {
    "alpha": 0.25,
    "stability_threshold": 8.0,
    "initialization_frames": 3
  }
}
```

**Threshold 模式示例**:
```json
"roi2_anti_jitter": {
  "enabled": true,
  "algorithm": "threshold",
  "threshold": {
    "movement_threshold": 20.0
  }
}
```

#### 5.5 roi3_config (ROI3)

ROI3 扩展垂直区域配置

**示例**:
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

---

### 6. peak_detection

ROI2 波峰检测配置

#### 6.1 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 95.0 | 固定阈值 |
| `threshold_minimum` | float | 80.0 | 阈值下限 |
| `margin_frames` | integer | 5 | 波峰边界扩展帧数 |
| `silence_frames` | integer | 15 | 最小静默帧数 |
| `difference_threshold` | float | 2.1 | 绿/红分类阈值 |
| `min_region_length` | integer | 5 | 最小波峰宽度 (帧) |

**示例**:
```json
"peak_detection": {
  "threshold": 95.0,
  "threshold_minimum": 80.0,
  "margin_frames": 5,
  "silence_frames": 15,
  "difference_threshold": 2.1,
  "min_region_length": 5
}
```

#### 6.2 自适应阈值

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `adaptive_threshold_enabled` | boolean | false | 启用自适应阈值 |
| `threshold_over_mean_ratio` | float | 0.15 | 阈值/均值比率 (15%) |
| `adaptive_window_seconds` | float | 3.0 | 自适应窗口时间 (秒) |

**示例**:
```json
"adaptive_threshold_enabled": true,
"threshold_over_mean_ratio": 0.15,
"adaptive_window_seconds": 3.0
```

**计算公式**:
```
adaptive_threshold = bg_mean * (1 + threshold_over_mean_ratio)
final_threshold = max(adaptive_threshold, threshold_minimum)
```

#### 6.3 阈值保护

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold_protection.enabled` | boolean | false | 启用阈值保护 |
| `threshold_protection.recovery_delay_seconds` | float | 1.0 | 恢复延迟时间 |
| `threshold_protection.stability_frames` | integer | 5 | 稳定性检查帧数 |
| `threshold_protection.waveform_trigger_enabled` | boolean | true | 启用波形触发 |

**示例**:
```json
"threshold_protection": {
  "enabled": true,
  "recovery_delay_seconds": 1.0,
  "stability_frames": 5,
  "waveform_trigger_enabled": true
}
```

#### 6.4 ROI3 覆盖配置

##### G1/G2 覆盖

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `g1_g2_override.enabled` | boolean | true | 启用 G1/G2 覆盖 |
| `g1_g2_override.g1_threshold` | float | 98.0 | G1 阈值 (%) |
| `g1_g2_override.g2_threshold` | float | 20.0 | G2 阈值 (%) |
| `g1_g2_override.use_peak_max` | boolean | true | 使用波峰最大值帧 |

**覆盖条件**: `G1 > g1_threshold` 且 `G2 > g2_threshold`

**示例**:
```json
"g1_g2_override": {
  "enabled": true,
  "g1_threshold": 98.0,
  "g2_threshold": 20.0,
  "use_peak_max": true
}
```

##### 列灰度差值覆盖

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `roi3_column_diff_override.enabled` | boolean | true | 启用列差值覆盖 |
| `roi3_column_diff_override.threshold` | float | 15.0 | 列差值阈值 |
| `roi3_column_diff_override.use_peak_max` | boolean | true | 使用波峰最大值帧 |

**覆盖条件**: `G1 > 99%` 且 `列灰度差值 > threshold`

**示例**:
```json
"roi3_column_diff_override": {
  "enabled": true,
  "threshold": 15.0,
  "use_peak_max": true
}
```

---

### 7. roi1_peak_detection

ROI1 独立波峰检测配置 (用于混合检测模式)

配置结构同 `peak_detection`，但参数独立：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | false | 启用 ROI1 波峰检测 |
| `threshold` | 120.0 | ROI1 阈值 |
| `threshold_minimum` | 110.0 | ROI1 阈值下限 |
| `threshold_over_mean_ratio` | 0.08 | ROI1 阈值比率 (8%) |

**示例**:
```json
"roi1_peak_detection": {
  "enabled": true,
  "threshold": 120.0,
  "threshold_minimum": 110.0,
  "margin_frames": 5,
  "silence_frames": 5,
  "difference_threshold": 2.0,
  "min_region_length": 5,
  "adaptive_threshold_enabled": true,
  "threshold_over_mean_ratio": 0.08,
  "threshold_protection": {
    "enabled": true,
    "recovery_delay_seconds": 1.0,
    "stability_frames": 5
  }
}
```

---

### 8. hybrid_detection

混合检测模式配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | false | 启用混合检测 |
| `detection_strategy` | string | "roi1_peaks_roi2_color" | 检测策略 |
| `fusion_strategy` | string | "roi2_priority" | 融合策略 |
| `require_intersection` | boolean | true | 要求检测到绿线交点 |

#### 8.1 ROI2 颜色判定配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `roi2_color_frames.pre_peak` | integer | 5 | 波峰前帧数 |
| `roi2_color_frames.post_peak` | integer | 10 | 波峰后帧数 |

#### 8.2 ROI1 波峰宽度验证

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `roi1_peak_width_range` | array | [30, 40] | 最小/最大宽度 (帧) |

#### 8.3 数据质量检查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_quality.minimum_roi2_frames` | integer | 15 | 最小 ROI2 帧数 |
| `data_quality.roi2_minimum_variance` | float | 0.5 | 最小方差 |
| `data_quality.skip_peaks_when_roi2_invalid` | boolean | true | ROI2 无效时跳过波峰 |

**示例**:
```json
"hybrid_detection": {
  "enabled": true,
  "detection_strategy": "roi1_peaks_roi2_color",
  "fusion_strategy": "roi2_priority",
  "require_intersection": true,
  "roi2_color_frames": {
    "pre_peak": 5,
    "post_peak": 10
  },
  "roi1_peak_width_range": [30, 40],
  "data_quality": {
    "minimum_roi2_frames": 15,
    "roi2_minimum_variance": 0.5,
    "roi2_min_gray": 5.0,
    "roi2_max_gray": 250.0,
    "skip_peaks_when_roi2_invalid": true
  },
  "fallback_enabled": true
}
```

---

### 9. deduplication

去重配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `consecutive_frame_window` | integer | 40 | 连续帧去重窗口 |
| `color_priority` | array | ["green", "red"] | 颜色优先级 |
| `recent_peak_window` | integer | 5 | 最近波峰比较窗口 |
| `cross_color_deduplication_enabled` | boolean | true | 启用跨色去重 |

**示例**:
```json
"deduplication": {
  "consecutive_frame_window": 40,
  "color_priority": ["green", "red"],
  "recent_peak_window": 5,
  "cross_color_deduplication_enabled": true
}
```

---

### 10. startup_cleanup

启动清理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 启用启动清理 |
| `cleanup_export` | boolean | true | 清理 export 目录 |
| `cleanup_tmp` | boolean | true | 清理 tmp 目录 |
| `cleanup_logs` | boolean | false | 清理 logs 目录 |

**示例**:
```json
"startup_cleanup": {
  "enabled": true,
  "cleanup_export": true,
  "cleanup_tmp": true,
  "cleanup_logs": false
}
```

---

## 环境变量覆盖

使用 `NHEM_` 前缀覆盖 JSON 配置值：

```bash
# 覆盖处理模式
export NHEM_PROCESSING_MODE=video

# 覆盖视频路径
export NHEM_VIDEO_PATH=path/to/video.mp4

# 覆盖帧率
export NHEM_FRAME_RATE=10

# 覆盖阈值
export NHEM_THRESHOLD=95.0
```

**嵌套配置访问**: 使用 `__` (双下划线) 分隔：

```bash
# 覆盖 roi_capture.frame_rate
export NHEM_ROI_CAPTURE__FRAME_RATE=15

# 覆盖 peak_detection.threshold
export NHEM_PEAK_DETECTION__THRESHOLD=100.0
```

---

## 配置调优指南

### 1. 波峰检测灵敏度

**场景**: 检测过多/过少波峰

```json
// 降低灵敏度 (减少误检)
"peak_detection": {
  "threshold": 105.0,           // 提高阈值
  "silence_frames": 20,         // 增加静默帧要求
  "min_region_length": 8        // 增加最小宽度
}

// 提高灵敏度 (减少漏检)
"peak_detection": {
  "threshold": 85.0,            // 降低阈值
  "silence_frames": 10,         // 减少静默帧要求
  "min_region_length": 3        // 减少最小宽度
}
```

### 2. 颜色分类准确性

**场景**: 绿/红分类不准确

```json
// 更严格的绿色判定
"peak_detection": {
  "difference_threshold": 3.0   // 提高阈值，只有明显上升才判定为绿色
}

// 更宽松的绿色判定
"peak_detection": {
  "difference_threshold": 1.5   // 降低阈值，轻微上升也判定为绿色
}
```

### 3. ROI2 稳定性

**场景**: ROI2 区域抖动

```json
// 更强的平滑
"roi2_anti_jitter": {
  "algorithm": "ema",
  "ema": {
    "alpha": 0.1,               // 降低 alpha，更平滑
    "stability_threshold": 5.0  // 降低稳定阈值
  }
}

// 完全静止模式
"roi2_anti_jitter": {
  "algorithm": "threshold",
  "threshold": {
    "movement_threshold": 30.0  // 提高运动阈值
  }
}
```

### 4. 自适应阈值调优

**场景**: 自适应阈值不稳定

```json
// 更保守的自适应
"peak_detection": {
  "adaptive_threshold_enabled": true,
  "threshold_over_mean_ratio": 0.20,    // 提高比率
  "adaptive_window_seconds": 5.0,       // 增加窗口时间
  "threshold_protection": {
    "enabled": true,
    "recovery_delay_seconds": 2.0,      // 延长恢复时间
    "stability_frames": 10              // 增加稳定性要求
  }
}
```

---

## 配置验证

启动时会自动验证以下参数：

1. **ROI 坐标**: 确保在屏幕/视频范围内
2. **帧率**: 1-30 FPS
3. **阈值**: 阈值 >= 阈值下限
4. **缓冲区大小**: 固定为 100 帧
5. **防抖动参数**: alpha 在 [0.05, 0.95] 范围内

无效参数会自动调整到合理范围，并输出警告日志。
