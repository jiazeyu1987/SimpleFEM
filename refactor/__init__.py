"""
SimpleFEM 重构版本包

SimpleFEM Refactored Package

模块结构:
- config_manager: 配置管理
- threshold_protection_manager: 阈值保护管理
- roi_capture_manager: ROI捕获管理
- green_line_manager: 绿线检测管理
- data_export_manager: 数据导出管理
- analysis_cache_manager: 分析缓存管理
- statistics_manager: 统计数据管理
- hybrid_detection_manager: 混合检测管理
- roi3_statistics: ROI3统计计算
- orchestrator: 主编排器
- main: 主入口
"""

from refactor.config_manager import ConfigManager
from refactor.threshold_protection_manager import ThresholdProtectionManager
from refactor.roi_capture_manager import ROICaptureManager
from refactor.green_line_manager import GreenLineManager
from refactor.data_export_manager import DataExportManager
from refactor.analysis_cache_manager import AnalysisCacheManager
from refactor.statistics_manager import StatisticsManager
from refactor.hybrid_detection_manager import HybridDetectionManager
from refactor.roi3_statistics import ROI3Statistics
from refactor.orchestrator import Orchestrator

__all__ = [
    'ConfigManager',
    'ThresholdProtectionManager',
    'ROICaptureManager',
    'GreenLineManager',
    'DataExportManager',
    'AnalysisCacheManager',
    'StatisticsManager',
    'HybridDetectionManager',
    'ROI3Statistics',
    'Orchestrator',
]

__version__ = '2.0.0'
