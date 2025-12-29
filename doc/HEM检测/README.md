# HEM 检测文档目录

本目录包含 SimpleFEM HEM (High Echo Event - 高回声事件) 检测系统的完整技术文档。

## 文档结构

```
doc/HEM检测/
├── README.md              # 本文件 - 文档导航
├── overview.md            # 系统概述 - 快速了解系统架构
├── algorithms.md          # 算法详解 - 深入理解检测算法
├── configuration.md       # 配置参考 - 完整配置参数说明
├── data-flow.md           # 数据流图 - 系统流程和架构
└── roi-analysis.md        # ROI 分析 - 区域捕获和统计分析
```

## 快速导航

### 新手入门

1. **系统概述** ([overview.md](overview.md))
   - 系统简介和核心功能
   - ROI 捕获系统
   - 绿线检测系统
   - 波峰检测系统
   - 处理模式和输出结构

2. **配置参考** ([configuration.md](configuration.md))
   - 配置文件结构
   - 关键配置参数
   - 环境变量覆盖
   - 配置调优指南

### 深入学习

3. **算法详解** ([algorithms.md](algorithms.md))
   - 绿线检测算法 (HSV, Canny, Hough)
   - 波峰检测算法 (固定/自适应阈值)
   - 颜色分类算法 (前后均值差, ROI3 覆盖)
   - 阈值保护算法
   - 三层去重算法
   - ROI3 统计算法 (G1/G2, 列差值)

4. **数据流图** ([data-flow.md](data-flow.md))
   - 系统架构图
   - 主处理流程
   - 数据缓冲区管理
   - 波峰检测流程
   - 去重流程
   - 批量视频处理流程

5. **ROI 分析** ([roi-analysis.md](roi-analysis.md))
   - ROI1/ROI2/ROI3 详细说明
   - ROI 坐标系统
   - ROI 数据质量评估
   - ROI 调试技巧
   - ROI 参数调优指南

## 常见问题

### Q: 如何快速了解系统？

**A**: 阅读顺序：
1. [overview.md](overview.md) - 了解系统整体架构
2. [data-flow.md](data-flow.md) - 理解数据流程
3. [configuration.md](configuration.md) - 学习如何配置

### Q: 如何调优检测参数？

**A**: 参考：
- [configuration.md](configuration.md) 中的"配置调优指南"
- [algorithms.md](algorithms.md) 中的算法原理
- [roi-analysis.md](roi-analysis.md) 中的参数调优指南

### Q: 波峰检测不准确怎么办？

**A**: 检查：
1. [configuration.md](configuration.md) - 阈值和灵敏度参数
2. [algorithms.md](algorithms.md) - 波峰检测算法原理
3. [roi-analysis.md](roi-analysis.md) - ROI2 数据质量

### Q: 如何理解混合检测模式？

**A**: 阅读：
1. [overview.md](overview.md) - "混合检测模式"章节
2. [algorithms.md](algorithms.md) - "ROI1/ROI2 混合检测"
3. [data-flow.md](data-flow.md) - "混合检测流程"

### Q: 如何调试 ROI 问题？

**A**: 参考：
- [roi-analysis.md](roi-analysis.md) - "ROI 调试技巧"章节

## 核心概念速查

| 概念 | 说明 | 文档 |
|------|------|------|
| **ROI1** | 大范围捕获区域 (1280x80 ~ 1920x980) | [overview.md](overview.md), [roi-analysis.md](roi-analysis.md) |
| **ROI2** | 围绕绿线交点的精确分析区域 (~80x120) | [overview.md](overview.md), [roi-analysis.md](roi-analysis.md) |
| **ROI3** | 扩展垂直区域，用于颜色分类验证 | [overview.md](overview.md), [roi-analysis.md](roi-analysis.md) |
| **绿线检测** | 使用 OpenCV 检测绿色线段并计算交点 | [overview.md](overview.md), [algorithms.md](algorithms.md) |
| **波峰检测** | 检测灰度曲线中的波峰区间 | [algorithms.md](algorithms.md) |
| **颜色分类** | Green/Red 判定 (基于前后均值差) | [algorithms.md](algorithms.md) |
| **阈值保护** | 防止波峰污染背景计算 | [algorithms.md](algorithms.md), [overview.md](overview.md) |
| **三层去重** | 最近波峰、连续帧、跨色去重 | [algorithms.md](algorithms.md), [overview.md](overview.md) |
| **混合检测** | ROI1 检测波峰时机 + ROI2 判定颜色 | [algorithms.md](algorithms.md), [data-flow.md](data-flow.md) |
| **G1/G2** | ROI3 高回声像素百分比 (用于覆盖判定) | [algorithms.md](algorithms.md), [roi-analysis.md](roi-analysis.md) |

## 相关代码文件

| 文件 | 说明 |
|------|------|
| `simple_roi_daemon.py` | 原始单文件实现 (2514 行) |
| `green_detector.py` | 绿线检测模块 |
| `peak_detection.py` | 波峰检测模块 |
| `safe_peak_statistics.py` | 统计和去重模块 |
| `refactor/` | 重构版本 (模块化架构) |

## 技术栈

- **Python 3.7+**
- **OpenCV** - 计算机视觉和图像处理
- **PIL/Pillow** - 屏幕捕获和图像操作
- **NumPy** - 数值计算
- **Matplotlib** - 波形可视化
- **JSON** - 配置和数据导出

## 系统特性

- ✅ **实时检测**: 支持 1-30 FPS 实时处理
- ✅ **多种模式**: 屏幕捕获、视频处理、静脉跟随
- ✅ **自适应阈值**: 动态调整检测灵敏度
- ✅ **阈值保护**: 防止波峰污染背景计算
- ✅ **三层去重**: 确保数据准确性
- ✅ **混合检测**: ROI1+ROI2 联合检测
- ✅ **ROI3 覆盖**: 使用 G1/G2 和列差值覆盖颜色判定
- ✅ **批量处理**: 支持多视频自动切换
- ✅ **数据导出**: CSV、JSONL、图像、波形

## 医学应用

- **HEM 检测**: 高回声事件检测
- **实时分析**: 连续监控，可配置灵敏度
- **数据完整性**: 审计日志和导出功能
- **临床使用**: 适用于研究和诊断支持
- **质量保证**: 去重和验证机制

## 反馈和贡献

如有问题或建议，请通过以下方式反馈：
- 提交 Issue
- 代码贡献 (Pull Request)
- 文档改进

---

**文档版本**: v2.0
**最后更新**: 2025-12-28
**维护者**: SimpleFEM Team
