# SimpleFEM 重构版本

## 概述

这是 SimpleFEM 的重构版本，将原本 2500+ 行的单文件代码拆分为多个管理类，提高了代码的可维护性和可扩展性。

## 目录结构

```
refactor/
├── __init__.py                    # 包初始化文件
├── main.py                        # 主入口文件
├── orchestrator.py                # 主编排器 - 协调所有组件
├── config_manager.py              # 配置管理器 - 加载和管理配置
├── threshold_protection_manager.py # 阈值保护管理器 - 防止波峰污染背景计算
├── roi_capture_manager.py         # ROI捕获管理器 - 屏幕/视频捕获
├── green_line_manager.py          # 绿线检测管理器 - 绿线交点检测和滤波
├── data_export_manager.py         # 数据导出管理器 - 图像和波形文件导出
├── analysis_cache_manager.py      # 分析缓存管理器 - JSONL缓存管理
├── statistics_manager.py          # 统计数据管理器 - 封装SafePeakStatistics
├── hybrid_detection_manager.py    # 混合检测管理器 - ROI1+ROI2联合检测
├── roi3_statistics.py             # ROI3统计计算 - G1/G2和列差值
├── test_refactor.py               # 单元测试脚本
└── test_integration.py            # 集成测试脚本
```

## 架构设计

### 管理类职责

#### 1. ConfigManager（配置管理器）
- 从 JSON 文件加载配置
- 支持环境变量覆盖（NHEM_* 前缀）
- 提供类型安全的配置访问
- 配置验证和默认值处理

#### 2. ThresholdProtectionManager（阈值保护管理器）
- 防止波峰数据污染自适应阈值的背景均值计算
- 支持波形触发和波峰触发两种激活模式
- 智能退出机制（时间延迟 + 稳定性检查）

#### 3. ROICaptureManager（ROI捕获管理器）
- 屏幕捕获（PIL.ImageGrab）
- 视频文件捕获（OpenCV）
- ROI1/ROI2/ROI3 提取
- 批量视频处理
- 帧率控制

#### 4. GreenLineManager（绿线检测管理器）
- 绿线交点检测
- EMA/Velocity/Threshold 滤波
- 防抖动处理

#### 5. DataExportManager（数据导出管理器）
- 创建和管理输出目录结构
- 保存 ROI1/ROI2/ROI3 图像
- 保存波形图
- 生成 ROI2 波形标注图

#### 6. AnalysisCacheManager（分析缓存管理器）
- 写入 JSONL 格式的每帧分析缓存
- 支持会话元数据
- 自动刷新和关闭处理

#### 7. StatisticsManager（统计数据管理器）
- 管理每视频的统计实例
- 批量模式支持
- 添加波峰数据
- 导出 CSV

#### 8. HybridDetectionManager（混合检测管理器）
- ROI1 波峰检测（检测波峰发生时机）
- ROI2 颜色判定（确定波峰颜色）
- 融合策略：roi2_priority（优先 ROI2 颜色判定）
- ROI1 峰值 ID 跟踪

#### 9. ROI3Statistics（ROI3统计计算器）
- G1/G2 像素百分比计算（用于绿/红覆盖判定）
- 列灰度差值计算（用于绿/红覆盖判定）
- 归一化灰度值计算（0-160范围）

#### 10. Orchestrator（主编排器）
- 协调所有组件完成完整的检测流程
- 管理帧处理循环
- 处理多视频批量处理
- 统一日志记录

## 使用方法

### 基本用法

```bash
# 直接运行主入口
python -m refactor.main

# 或者
python refactor/main.py
```

### 配置文件

重构版本使用原有的 `simple_fem_config.json` 配置文件，无需修改配置格式。

### 作为模块使用

```python
from refactor.orchestrator import Orchestrator

# 创建编排器
orchestrator = Orchestrator()

# 运行
orchestrator.run()

# 关闭
orchestrator.close()
```

### 自定义配置路径

```python
from refactor.orchestrator import Orchestrator

# 使用自定义配置文件
orchestrator = Orchestrator(config_path="/path/to/config.json")
orchestrator.run()
```

## 主要改进

### 1. 模块化设计
- 将 2500+ 行代码拆分为 **10 个管理类**
- 每个类职责单一，易于理解和维护
- 支持独立测试和替换

### 2. 类型安全
- 使用类型提示（Type Hints）
- 配置访问提供类型安全的属性方法
- 减少运行时错误

### 3. 更好的错误处理
- 每个管理器独立处理错误
- 主编排器统一协调错误恢复
- 详细的日志记录

### 4. 易于扩展
- 新增功能只需添加新的管理器
- 现有管理器可以独立升级
- 支持插件式架构

### 5. 代码复用
- 管理类可以在其他项目中复用
- 清晰的接口定义
- 降低耦合度

### 6. 高级功能支持
- **混合检测**: ROI1 检测波峰时机 + ROI2 判定颜色
- **ROI3 统计**: G1/G2 像素百分比和列差值计算
- **增强日志**: 包含 ROI3 统计信息的详细日志

## 测试

### 单元测试

```bash
# 运行基础单元测试
python refactor/test_refactor.py
```

测试内容:
- ConfigManager - 配置加载和访问
- ThresholdProtectionManager - 阈值保护机制
- ROICaptureManager - ROI 捕获功能
- GreenLineManager - 绿线检测功能

### 集成测试

```bash
# 运行集成测试
python refactor/test_integration.py
```

测试内容:
- ROI3Statistics - ROI3 统计计算
- HybridDetectionManager - 混合检测功能
- ROI2 颜色判定 - 颜色分类逻辑
- 数据质量计算 - ROI2 数据质量评估

## 依赖关系

```
main.py
    └── orchestrator.py
            ├── config_manager.py
            ├── threshold_protection_manager.py
            ├── roi_capture_manager.py
            │       └── green_detector.py (原始模块)
            ├── green_line_manager.py
            │       └── green_detector.py (原始模块)
            ├── data_export_manager.py
            ├── analysis_cache_manager.py
            ├── statistics_manager.py
            │       └── safe_peak_statistics.py (原始模块)
            └── peak_detection.py (原始模块)
```

## 兼容性

- **配置兼容**: 完全兼容原有的 `simple_fem_config.json`
- **模块兼容**: 复用原始的 `green_detector.py`, `peak_detection.py`, `safe_peak_statistics.py`
- **输出兼容**: 输出格式和目录结构与原版相同

## 测试

重构版本需要测试以下方面：

1. **功能测试**: 验证所有功能与原版一致
2. **性能测试**: 确保性能没有下降
3. **配置测试**: 验证配置加载和环境变量覆盖
4. **多视频测试**: 测试批量处理功能
5. **异常测试**: 测试错误处理和恢复

## 已实现功能

1. ✅ **混合检测**: ROI1/ROI2 混合检测 (HybridDetectionManager)
   - ROI1 波峰检测（检测波峰发生时机）
   - ROI2 颜色判定（确定波峰颜色）
   - 融合策略：roi2_priority（优先 ROI2 颜色判定）
   - ROI1 峰值 ID 跟踪

2. ✅ **ROI3 统计**: G1/G2 像素百分比、列灰度差值计算 (ROI3Statistics)
   - G1 范围 [80, 255] 像素百分比
   - G2 范围 [150, 255] 像素百分比
   - 列灰度差值（最大列均值 - 最小列均值）
   - 归一化灰度值 [0, 160] 计算

3. ✅ **阈值保护**: 完整的阈值保护机制 (ThresholdProtectionManager)
   - 波形触发（当前灰度 >= 阈值）
   - 波峰触发（检测到波峰）
   - 智能退出（时间延迟 + 稳定性检查）

4. ✅ **多层去重**: 三层去重系统 (SafePeakStatistics)
   - Layer 1: 最近波峰比较（5 帧窗口）
   - Layer 2: 连续帧去重（40 帧窗口）
   - Layer 3: 跨色去重（绿色优先）

## 待完善功能

1. ⚠️ **静脉跟随模式**: `vein_following` 模式框架已搭建，完整实现待开发
2. ⚠️ **ROI3 覆盖机制**: 统计计算已实现，覆盖逻辑集成到混合检测待完善

## 开发指南

### 添加新的管理器

1. 在 `refactor/` 目录下创建新的管理器文件
2. 继承基本的设计模式（初始化、处理、清理）
3. 在 `Orchestrator` 中集成新的管理器
4. 更新 `__init__.py` 导出新的管理器

### 扩展现有管理器

1. 修改相应的管理器文件
2. 保持接口兼容性
3. 添加配置项到 `ConfigManager`
4. 更新文档

## 版本历史

- **v2.0.0** (2025-12-28): 初始重构版本
  - 创建 10 个管理器
  - 实现所有核心功能框架
  - 支持视频和屏幕模式
  - 支持混合检测和 ROI3 统计
