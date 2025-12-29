# SimpleFEM 重构版本测试文档

## 概述

本目录包含 SimpleFEM 重构版本的所有单元测试。测试覆盖了所有 10 个管理器的核心功能。

## 测试目录结构

```
refactor/test/
├── __init__.py                          # 包初始化
├── README.md                             # 本文档
├── run_all_tests.py                      # 运行所有测试的脚本
├── test_config_manager.py                # ConfigManager 测试
├── test_threshold_protection_manager.py  # ThresholdProtectionManager 测试
├── test_roi3_statistics.py               # ROI3Statistics 测试
├── test_hybrid_detection_manager.py      # HybridDetectionManager 测试
├── test_green_line_manager.py            # GreenLineManager 测试
├── test_analysis_cache_manager.py        # AnalysisCacheManager 测试
└── test_data_export_manager.py           # DataExportManager 测试
```

## 快速开始

### 运行所有测试

```bash
# 方式1: 使用测试运行脚本
python refactor/test/run_all_tests.py

# 方式2: 使用 unittest
python -m unittest discover -s refactor/test -p "test_*.py" -v

# 方式3: 运行特定测试文件
python -m unittest refactor.test.test_config_manager -v
```

### 运行单个测试类

```bash
# 测试配置管理器
python -m unittest refactor.test.test_config_manager.TestConfigManager -v

# 测试阈值保护管理器
python -m unittest refactor.test.test_threshold_protection_manager.TestThresholdProtectionManager -v

# 测试 ROI3 统计
python -m unittest refactor.test.test_roi3_statistics.TestROI3Statistics -v
```

### 运行单个测试方法

```bash
# 测试特定方法
python -m unittest refactor.test.test_config_manager.TestConfigManager.test_load_config -v
```

## 测试覆盖说明

### 1. ConfigManager (配置管理器)

测试内容：
- ✅ 配置文件加载
- ✅ 配置值获取（嵌套、默认值）
- ✅ 环境变量覆盖 (NHEM_*)
- ✅ 所有属性方法 (processing_mode, frame_rate, roi1_config 等)
- ✅ 错误处理（文件不存在）

测试数量：约 30 个测试用例

### 2. ThresholdProtectionManager (阈值保护管理器)

测试内容：
- ✅ 初始化和重置
- ✅ 波形触发保护 (gray >= threshold)
- ✅ 波峰触发保护 (has_peaks)
- ✅ 智能退出机制（时间 + 稳定性）
- ✅ 稳定性计数器重置
- ✅ 属性访问 (is_active, frames_since_end)

测试数量：约 10 个测试用例

### 3. ROI3Statistics (ROI3统计计算器)

测试内容：
- ✅ G1/G2 像素百分比计算
- ✅ 边界值处理 (80, 150)
- ✅ 列灰度差值计算
- ✅ 归一化灰度值计算 [0, 160]
- ✅ compute_all() 一致性
- ✅ 静态方法调用

测试数量：约 15 个测试用例

### 4. HybridDetectionManager (混合检测管理器)

测试内容：
- ✅ 混合检测启用/禁用
- ✅ ROI1 波峰检测
- ✅ ROI2 颜色判定 (green/red)
- ✅ 波峰 ID 生成和跟踪
- ✅ 数据质量计算
- ✅ 多波峰检测

测试数量：约 12 个测试用例

### 5. GreenLineManager (绿线检测管理器)

测试内容：
- ✅ 初始化和重置
- ✅ 交点历史记录
- ✅ 防抖动配置 (EMA/Threshold)
- ✅ 交点滤波器初始化
- ✅ 有效交点检测
- ✅ 属性访问

测试数量：约 10 个测试用例

### 6. AnalysisCacheManager (分析缓存管理器)

测试内容：
- ✅ 会话启动和切换
- ✅ 帧数据记录
- ✅ JSONL 格式验证
- ✅ 元数据写入
- ✅ 会话结束标记
- ✅ numpy/datetime JSON 序列化
- ✅ 自动刷新机制
- ✅ 禁用缓存模式

测试数量：约 12 个测试用例

### 7. DataExportManager (数据导出管理器)

测试内容：
- ✅ 目录创建 (视频/屏幕模式)
- ✅ ROI1/ROI2/ROI3 图像保存
- ✅ 波形图生成
- ✅ ROI1 波形图生成
- ✅ 文件名生成（带/不带视频时间）
- ✅ 视频名称清理
- ✅ ROI2 标注到波形图

测试数量：约 15 个测试用例

## 测试统计

| 管理器 | 测试类 | 测试方法数 | 覆盖功能 |
|--------|--------|------------|----------|
| ConfigManager | TestConfigManager | ~30 | 配置加载、环境变量、属性访问 |
| ThresholdProtectionManager | TestThresholdProtectionManager | ~10 | 波形/波峰触发、智能退出 |
| ROI3Statistics | TestROI3Statistics | ~15 | G1/G2、列差值、归一化 |
| HybridDetectionManager | TestHybridDetectionManager | ~12 | ROI1/ROI2 混合检测 |
| GreenLineManager | TestGreenLineManager | ~10 | 绿线检测、防抖动 |
| AnalysisCacheManager | TestAnalysisCacheManager | ~12 | JSONL 缓存、会话管理 |
| DataExportManager | TestDataExportManager | ~15 | 图像/波形导出 |
| **总计** | **7 个测试类** | **~114 个测试** | **7 个管理器** |

## 测试框架

本测试套件使用 Python 标准库 `unittest` 框架：

```python
import unittest

class MyTest(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        pass

    def tearDown(self):
        """测试后清理"""
        pass

    def test_something(self):
        """测试用例"""
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()
```

## 测试最佳实践

### 1. 测试命名规范

- 测试文件：`test_<module_name>.py`
- 测试类：`Test<ClassName>`
- 测试方法：`test_<method_name>_<scenario>`

### 2. 测试结构

每个测试文件包含：
1. 导入必要的模块
2. 测试类定义
3. `setUp()` - 测试前准备
4. `tearDown()` - 测试后清理
5. 测试方法

### 3. 断言使用

```python
# 相等断言
self.assertEqual(a, b)
self.assertNotEqual(a, b)

# 布尔断言
self.assertTrue(condition)
self.assertFalse(condition)

# 类型断言
self.assertIsInstance(obj, ClassName)

# 异常断言
with self.assertRaises(ExceptionType):
    function_that_raises()

# 数值比较
self.assertAlmostEqual(a, b, places=2)
self.assertGreater(a, b)
self.assertLess(a, b)
```

## 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install numpy opencv-python Pillow matplotlib
      - name: Run tests
        run: python -m unittest discover -s refactor/test -v
```

## 待添加测试

以下管理器暂时没有独立的测试文件：

1. **ROICaptureManager** - 需要模拟屏幕捕获和视频文件
2. **StatisticsManager** - 需要模拟 SafePeakStatistics
3. **Orchestrator** - 需要集成测试

这些可以通过集成测试来覆盖。

## 调试测试

### 运行单个测试并查看详细输出

```bash
python -m unittest refactor.test.test_config_manager.TestConfigManager.test_load_config -v
```

### 使用 pdb 调试

```python
def test_something(self):
    import pdb; pdb.set_trace()
    # 测试代码
    self.assertEqual(1, 1)
```

### 查看测试覆盖率

```bash
pip install coverage
coverage run -m unittest discover -s refactor/test
coverage report -m
coverage html
```

## 贡献指南

添加新测试时，请遵循：

1. **命名规范**：文件名为 `test_<module>.py`
2. **测试类**：继承 `unittest.TestCase`
3. **文档字符串**：为每个测试方法添加描述
4. **独立性**：每个测试应该独立运行
5. **清理**：在 `tearDown()` 中清理资源

## 反馈

如有问题或建议，请：
- 提交 Issue
- 创建 Pull Request
- 联系维护团队

---

**版本**: v1.0
**最后更新**: 2025-12-28
**维护者**: SimpleFEM Team
