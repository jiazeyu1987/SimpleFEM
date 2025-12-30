# simple_roi_daemon.py 文件依赖清单

## 1. Python 标准库（内置模块）

| 模块 | 用途 |
|------|------|
| `json` | JSON 配置文件读写 |
| `logging` | 日志记录 |
| `logging.handlers` | 日志文件轮转 |
| `os` | 文件路径操作、目录创建 |
| `sys` | 系统参数、可执行文件路径 |
| `time` | 时间戳、睡眠控制 |
| `uuid` | 生成唯一会话 ID |
| `collections.deque` | 循环缓冲区实现 |
| `datetime` | 时间戳转换 |
| `typing` | 类型提示 |
| `platform` | 获取系统平台信息 |
| `glob` | 文件模式匹配（视频发现） |

---

## 2. 第三方库（需要 pip 安装）

| 库名 | 版本要求 | 用途 |
|------|----------|------|
| `numpy` | 必需 | 数值计算、数组操作 |
| `Pillow` | 必需 | 图像捕获 (PIL.ImageGrab)、图像处理 |
| `opencv-python` | 必需 | 计算机视觉、视频读取 |
| `matplotlib` | 必需 | 波形图绘制 |

**安装命令：**
```bash
pip install numpy opencv-python Pillow matplotlib
```

---

## 3. 本地模块（项目内文件）

| 文件名 | 导入内容 | 用途 |
|--------|----------|------|
| `green_detector.py` | `detect_green_intersection`, `IntersectionFilter` | 绿线交点检测、EMA 防抖动滤波 |
| `peak_detection.py` | `detect_peaks` | 波峰检测算法 |
| `safe_peak_statistics.py` | `SafePeakStatistics` | 统计数据管理、CSV 导出、去重 |
| `threshold_based_anti_jitter.py` | `ThresholdIntersectionFilter` | 阈值式防抖动滤波器（可选） |

---

## 4. 配置文件

### 4.1 主配置文件

**文件名：** `simple_fem_config.json`

**位置：** 与 `simple_roi_daemon.py` 同目录

**用途：** 系统主配置文件

**关键配置项：**
```json
{
  "processing_mode": "video",           // 处理模式
  "roi_capture": {...},                 // ROI 捕获配置
  "peak_detection": {...},              // 波峰检测配置
  "roi1_peak_detection": {...},         // ROI1 波峰检测配置
  "roi2_anti_jitter": {...},            // 防抖动配置
  "hybrid_detection": {...},            // 混合检测配置
  "data_processing": {...},             // 数据保存配置
  "video_processing": {...},            // 视频处理配置
  "analysis_cache": {...},              // 分析缓存配置
  "startup_cleanup": {...}              // 启动清理配置
}
```

### 4.2 静脉检测配置（可选）

**文件名：** `vein_detection_config.json`

**位置：** 与 `simple_roi_daemon.py` 同目录

**用途：** 静脉检测和跟随模式配置

---

## 5. 输入文件

### 5.1 视频文件

**支持格式：**
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`

**配置路径：** `simple_fem_config.json` 中的 `video_processing.video_path`

**示例：**
```json
{
  "video_processing": {
    "video_path": "video.mp4"           // 单个视频
    // 或
    "video_path": "video/"               // 视频文件夹（批量处理）
  }
}
```

**发现逻辑：**
- 如果 `video_path` 是文件 → 处理该文件
- 如果 `video_path` 是目录 → 扫描目录中所有支持的视频文件（按文件名排序）

---

## 6. 输出文件

### 6.1 导出目录结构

```
SimpleFEM/
├── export/                              // 导出目录
│   ├── peak_statistics_*.csv            // 统计数据 CSV
│   ├── roi_analysis_cache_*.jsonl       // 分析缓存（每帧 JSONL）
│   └── tmp_{session_id}/                // 临时数据（每视频）
│       ├── roi1/                        // ROI1 图像
│       │   └── roi1_XXXXXX.png
│       ├── roi2/                        // ROI2 图像
│       │   ├── roi2_XXXXXX.png
│       │   └── roi2_XXXXXX_XXXX.XXs.png // 带时间戳（视频模式）
│       ├── roi3/                        // ROI3 图像
│       │   ├── roi3_XXXXXX.png
│       │   └── roi3_XXXXXX_XXXX.XXs.png
│       ├── wave/                        // ROI2 波形图
│       │   └── wave_XXXXXX.png
│       └── wave1/                       // ROI1 波形图
│           └── roi1_wave_XXXXXX.png
├── logs/                                // 日志目录
│   └── roi_peak_daemon.log              // 运行日志（每日轮转）
└── tmp/                                 // 临时目录（屏幕模式）
    └── {session_id}/
        ├── roi1/
        ├── roi2/
        ├── roi3/
        └── wave/
```

### 6.2 CSV 统计文件

**文件命名格式：**
```
peak_statistics_{video_name}_{timestamp}.csv
```

**示例：**
```
peak_statistics_video1_20251225_132129.csv
peak_statistics_2（5次发射，1无效）_20251225_002319.csv
```

**字段列表：**
```csv
timestamp,frame_index,peak_type,peak_start,peak_end,width,
roi1_frame_diff,roi2_frame_diff,pre_peak_avg,post_peak_avg,
difference_threshold_used,threshold_used,bg_mean,peak_max_value,
roi3_peak_max_value,roi3_peak_max_frame,
pre_peak_frame_start,pre_peak_frame_end,post_peak_frame_start,post_peak_frame_end,
roi1_peak_id,roi1_detection_method,roi2_color_method,
intersection_x,intersection_y,roi2_x1,roi2_y1,roi2_x2,roi2_y2,
roi2_width,roi2_height,roi3_override_applied,roi3_override_threshold
```

### 6.3 分析缓存文件（JSONL）

**文件命名格式：**
```
roi_analysis_cache_{session_id}_{run_id}.jsonl
```

**示例：**
```
roi_analysis_cache_20251225_132129_a1b2c3d4e5f6.jsonl
```

**文件格式：** 每行一个 JSON 对象

**内容类型：**
- `meta` - 会话元数据（第一行）
- `frame` - 帧分析数据（每帧一行）
- `session_end` - 会话结束标记（最后一行）

### 6.4 日志文件

**文件名：** `roi_peak_daemon.log`

**位置：** `logs/roi_peak_daemon.log`

**轮转规则：**
- 每天午夜轮转一次
- 保留最近 7 天的日志
- 文件命名：`roi_peak_daemon.log.YYYY-MM-DD`

**日志格式：**
```
2025-12-25T00:23:19 gray=105.5 green_peaks=1 red_peaks=0 last_green=[45,50]
2025-12-25T00:23:20 gray=98.2 green_peaks=0 red_peaks=1 last_red=[55,60]
```

### 6.5 图像文件

**ROI1 图像：**
- 命名：`roi1_{frame_index:06d}.png`
- 示例：`roi1_000123.png`

**ROI2 图像：**
- 屏幕模式：`roi2_{frame_index:06d}.png`
- 视频模式：`roi2_{frame_index:06d}_{video_seconds:06.2f}s.png`
- 示例：`roi2_000123_0005.23s.png`

**ROI3 图像：**
- 格式同 ROI2
- 示例：`roi3_000123_0005.23s.png`

**波形图：**
- ROI2 波形：`wave_{frame_index:06d}.png`
- ROI1 波形：`roi1_wave_{frame_index:06d}.png`

---

## 7. 临时文件

### 7.1 临时图像文件

**位置：** `tmp/{session_id}/`

**命名规则：** 同 export 目录

**清理时机：**
- 启动时清理（如果 `startup_cleanup.enabled = true`）
- 程序正常退出时保留
- 程序异常退出时保留

### 7.2 CSV 原子写入

**机制：** 使用临时文件确保数据完整性

**流程：**
1. 写入临时文件：`peak_statistics_{video_name}_{timestamp}.tmp.csv`
2. 验证数据完整性
3. 重命名为正式文件名
4. 如果失败，临时文件保留用于调试

---

## 8. 文件访问模式

### 8.1 读取操作

| 文件类型 | 访问时机 | 访问方式 |
|---------|---------|---------|
| `simple_fem_config.json` | 启动时 | 一次性读取 |
| 视频文件 | 运行时 | 逐帧读取（OpenCV） |
| ROI2 图像（波形标注） | 保存波形时 | 按需读取（glob 搜索） |

### 8.2 写入操作

| 文件类型 | 访问时机 | 访问方式 |
|---------|---------|---------|
| ROI 图像 | 每帧（可选） | 直接写入 |
| 波形图 | 每帧（可选） | 直接写入 |
| CSV 统计 | 检测到波峰时 | 追加写入 |
| 分析缓存 | 每帧 | 追加写入（每 N 帧刷新） |
| 日志文件 | 每帧 | 追加写入（自动刷新） |

---

## 9. 文件大小估算

### 9.1 单个视频处理

假设 7 秒视频，10 FPS：
- 总帧数：70 帧
- ROI1 图像：70 × 500 KB ≈ 35 MB
- ROI2 图像：70 × 50 KB ≈ 3.5 MB
- ROI3 图像：70 × 50 KB ≈ 3.5 MB
- 波形图：70 × 200 KB ≈ 14 MB
- CSV 文件：< 1 MB
- 缓存文件：约 5-10 MB

**总计：** 约 60-70 MB/视频（如果保存所有数据）

### 9.2 批量视频处理

假设 10 个视频：
- 总数据量：600-700 MB
- CSV 文件：< 10 MB
- 缓存文件：50-100 MB

---

## 10. 关键文件依赖关系图

```
simple_roi_daemon.py
    ├─ 读取 → simple_fem_config.json (配置)
    ├─ 导入 → green_detector.py (绿线检测)
    ├─ 导入 → peak_detection.py (波峰检测)
    ├─ 导入 → safe_peak_statistics.py (统计管理)
    ├─ 导入 → threshold_based_anti_jitter.py (防抖动，可选)
    ├─ 读取 → 视频文件 (*.mp4, *.avi, ...)
    ├─ 写入 → export/peak_statistics_*.csv (统计数据)
    ├─ 写入 → export/roi_analysis_cache_*.jsonl (缓存)
    ├─ 写入 → export/tmp_*/roi1/*.png (ROI1 图像)
    ├─ 写入 → export/tmp_*/roi2/*.png (ROI2 图像)
    ├─ 写入 → export/tmp_*/roi3/*.png (ROI3 图像)
    ├─ 写入 → export/tmp_*/wave/*.png (波形图)
    ├─ 写入 → export/tmp_*/wave1/*.png (ROI1 波形图)
    ├─ 写入 → logs/roi_peak_daemon.log (日志)
    └─ 读取 → export/tmp_*/roi2/*.png (波形标注，按需)
```

---

## 11. 部署检查清单

### 11.1 必需文件

- [ ] `simple_roi_daemon.py` - 主程序
- [ ] `simple_fem_config.json` - 配置文件
- [ ] `green_detector.py` - 绿线检测模块
- [ ] `peak_detection.py` - 波峰检测模块
- [ ] `safe_peak_statistics.py` - 统计管理模块

### 11.2 可选文件

- [ ] `threshold_based_anti_jitter.py` - 阈值式防抖动（如果配置启用）
- [ ] `vein_detection_config.json` - 静脉检测配置（如果使用该模式）

### 11.3 依赖库

- [ ] `numpy` - pip install numpy
- [ ] `opencv-python` - pip install opencv-python
- [ ] `Pillow` - pip install Pillow
- [ ] `matplotlib` - pip install matplotlib

### 11.4 目录结构

- [ ] `export/` - 导出目录（自动创建）
- [ ] `logs/` - 日志目录（自动创建）
- [ ] `tmp/` - 临时目录（自动创建）

---

## 12. 文件路径相关函数

### 12.1 基础目录获取

```python
def _get_base_dir() -> str:
    """
    获取基础目录：
    - PyInstaller 打包后：.exe 所在目录
    - 源码运行：.py 文件所在目录
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))
```

**用途：** 确保配置文件和数据文件路径正确（支持打包和源码两种模式）

### 12.2 视频文件夹创建

```python
def _create_video_folders(video_path, session_id, processing_mode, ...) -> str:
    """
    为每个视频创建独立的输出文件夹：
    export/tmp_{session_id}/
        ├── roi1/
        ├── roi2/
        ├── roi3/
        ├── wave/
        └── wave1/
    """
```

**返回：** 临时根目录路径

---

## 13. 特殊文件操作

### 13.1 启动清理

**配置项：** `startup_cleanup`

**清理目标：**
- `export/` - 导出目录
- `tmp/` - 临时目录
- `logs/` - 日志目录

**清理时机：** 程序启动时

**清理策略：**
- 删除所有文件和子文件夹
- 可分别配置是否清理各个目录

### 13.2 多视频文件名清理

**函数：** `_sanitize_video_name(video_name)`

**处理：**
- 移除文件扩展名
- 替换特殊字符为下划线
- 确保文件名安全

**示例：**
```
input: "视频#1@test.mp4"
output: "___1_test_"
```

---

## 14. 文件锁定和并发

### 14.1 写入安全

**CSV 写入：**
- 使用临时文件 + 原子重命名
- 避免部分写入导致数据损坏

**JSONL 缓存：**
- 追加模式写入
- 定期刷新（每 50 帧）
- Session end 标记确保完整性

### 14.2 多进程安全

**当前实现：** 单进程设计，无并发写入

**未来扩展：** 如需多进程，建议使用：
- 文件锁（`fcntl` 或 `msvcrt`）
- 数据库（SQLite）
- 消息队列

---

**文档版本：** 1.0
**最后更新：** 2025-12-25
**适用于：** simple_roi_daemon.py 完整文件依赖分析
