# ROI Analysis Cache Analyzer 使用说明

`analyze_cache.py` 是一个独立的Python程序，用于读取和分析 SimpleFEM 生成的 `roi_analysis_cache_*.jsonl` 文件。

## 功能特性

1. **基本统计摘要** - 显示ROI2/ROI3灰度值统计、波峰检测统计、交点分布统计
2. **波峰详细分析** - 按帧和颜色分类展示波峰检测结果
3. **交点分布分析** - 分析绿线交点的位置和抖动情况
4. **波形可视化** - 生成ROI2/ROI3波形图，标记波峰位置
5. **CSV导出** - 将缓存数据导出为CSV格式，方便在Excel中分析

## 安装依赖

```bash
pip install numpy matplotlib
```

或者使用项目自带的 requirements.txt:

```bash
pip install -r requirements.txt
```

## 基本用法

### 1. 显示基本统计摘要

```bash
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --summary
```

输出示例：
```
================================================================================
ROI 分析缓存统计摘要
================================================================================

总帧数: 802
有效ROI2数据: 802 帧
有效ROI3数据: 802 帧

--------------------------------------------------------------------------------
ROI2 灰度值统计:
  均值: 65.61
  标准差: 22.92
  范围: [1.45, 120.69]

ROI3 灰度值统计:
  均值: 64.85
  标准差: 23.32
  范围: [1.29, 131.07]

--------------------------------------------------------------------------------
波峰检测统计:
  检测到波峰的帧数: 5
  绿色波峰总数: 3
  红色波峰总数: 2
  总波峰数: 5
  波峰检测率: 0.62%

--------------------------------------------------------------------------------
绿线交点统计:
  有效交点数量: 802
  无效交点数量: 0

交点X坐标:
  均值: 321.01
  标准差: 0.15
  范围: [321.00, 324.00]

交点Y坐标:
  均值: 468.00
  标准差: 0.00
  范围: [468.00, 468.00]
================================================================================
```

### 2. 详细波峰分析

```bash
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --peaks
```

输出示例：
```
================================================================================
波峰检测详细分析
================================================================================

总共检测到 5 个波峰
绿色波峰: 3 个
红色波峰: 2 个
检测到波峰的帧数: 5

绿色波峰分布:
  首个波峰: 帧索引 63
  最后波峰: 帧索引 739

红色波峰分布:
  首个波峰: 帧索引 349
  最后波峰: 帧索引 784

前10个有波峰的帧:
  帧 63: 绿色=1, 红色=0
  帧 349: 绿色=0, 红色=1
  帧 650: 绿色=1, 红色=0
  帧 739: 绿色=1, 红色=0
  帧 784: 绿色=0, 红色=1
================================================================================
```

### 3. 绿线交点分布分析

```bash
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --intersection
```

输出示例：
```
================================================================================
绿线交点分布分析
================================================================================

有效交点数量: 802
无效交点数量: 0

X坐标统计:
  均值: 321.01
  标准差: 0.15
  范围: [321.00, 324.00]
  最大抖动: 3.00
  平均抖动: 0.00

Y坐标统计:
  均值: 468.00
  标准差: 0.00
  范围: [468.00, 468.00]
  最大抖动: 0.00
  平均抖动: 0.00
================================================================================
```

### 4. 生成波形图

```bash
# 显示波形图（需要GUI环境）
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --waveform

# 保存波形图到文件
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --waveform --output-waveform waveform.png
```

波形图包含：
- ROI2 灰度值曲线，标记检测阈值和波峰位置
- ROI3 灰度值曲线（如果可用）

### 5. 导出为CSV

```bash
# 自动生成文件名
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --export-csv

# 指定输出文件名
python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --export-csv analysis.csv
```

### 6. 组合多个选项

```bash
# 同时显示摘要、波峰分析、交点分析
python analyze_cache.py cache.jsonl --summary --peaks --intersection

# 生成波形图并导出CSV
python analyze_cache.py cache.jsonl --waveform --output-waveform plot.png --export-csv data.csv
```

### 7. 指定帧范围分析

```bash
# 只分析第100-200帧
python analyze_cache.py cache.jsonl --waveform --filter-frames 100-200 --output-waveform partial.png
```

## 高级用法

### 批量分析多个缓存文件

Windows PowerShell:
```powershell
Get-ChildItem export\roi_analysis_cache_*.jsonl | ForEach-Object {
    python analyze_cache.py $_.FullName --summary --peaks
}
```

Linux/Mac bash:
```bash
for file in export/roi_analysis_cache_*.jsonl; do
    python analyze_cache.py "$file" --summary --peaks
done
```

### 生成完整报告

```bash
python analyze_cache.py cache.jsonl \
    --summary \
    --peaks \
    --intersection \
    --waveform \
    --output-waveform "waveform_$(date +%Y%m%d_%H%M%S).png" \
    --export-csv "data_$(date +%Y%m%d_%H%M%S).csv"
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `cache_file` | ROI分析缓存文件路径（必需） |
| `--summary` | 显示基本统计摘要 |
| `--peaks` | 详细分析波峰检测 |
| `--intersection` | 分析绿线交点分布 |
| `--waveform` | 绘制波形图 |
| `--export-csv [OUTPUT]` | 导出到CSV文件（可选文件名） |
| `--filter-frames X-Y` | 只分析指定帧范围（例如：100-200） |
| `--output-waveform FILE` | 保存波形图到指定文件 |
| `--help` | 显示帮助信息 |

## 输出说明

### 统计指标解释

1. **灰度值统计**
   - 均值：平均灰度值
   - 标准差：灰度值的波动程度
   - 范围：最小值和最大值

2. **波峰检测统计**
   - 检测到波峰的帧数：至少有一个波峰的帧数
   - 绿色/红色波峰总数：所有帧中检测到的波峰总数
   - 波峰检测率：有波峰的帧占总帧数的百分比

3. **交点统计**
   - 有效/无效交点数量
   - X/Y坐标的均值、标准差、范围
   - 最大抖动：相邻帧之间的最大位置变化
   - 平均抖动：相邻帧之间的平均位置变化

## 常见问题

### Q: 中文显示乱码怎么办？
A: 程序已自动配置中文字体支持。如果仍有问题，请确保系统安装了中文字体（如SimHei或Microsoft YaHei）。

### Q: 波形图无法显示？
A: 在无GUI环境下，使用 `--output-waveform` 参数将波形图保存到文件。

### Q: 内存不足怎么办？
A: 使用 `--filter-frames` 参数只分析部分帧，或者分批分析大文件。

### Q: CSV文件用什么软件打开？
A: 可以使用Excel、LibreOffice Calc、或任何文本编辑器打开。

## 技术支持

如有问题或建议，请联系 SimpleFEM 开发团队或提交 Issue。
