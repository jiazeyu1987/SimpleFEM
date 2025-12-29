"""
配置管理器 - 负责加载、验证和访问配置

SimpleFEM Refactored Version
"""

import json
import os
import sys
from typing import Any, Dict, Optional


class ConfigManager:
    """
    配置管理器，负责加载和管理所有配置项

    功能:
    - 从JSON文件加载配置
    - 支持环境变量覆盖 (NHEM_* 前缀)
    - 提供类型安全的配置访问
    - 配置验证和默认值处理
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self._base_dir = self._get_base_dir()
        self._config: Dict[str, Any] = {}
        self._config_path = config_path

        if config_path is None:
            # 使用默认配置文件路径（在父目录中）
            self._config_path = os.path.join(os.path.dirname(self._base_dir), "simple_fem_config.json")

        self.load()

    @staticmethod
    def _get_base_dir() -> str:
        """
        获取基础目录（支持源码和打包模式）
        """
        if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
            return os.path.dirname(os.path.abspath(sys.executable))
        return os.path.dirname(os.path.abspath(__file__))

    def load(self) -> None:
        """加载配置文件"""
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"配置文件不存在: {self._config_path}")

        with open(self._config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)

        # 应用环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖配置"""
        for key, value in os.environ.items():
            if key.startswith("NHEM_"):
                config_key = key[5:].lower()  # 移除 NHEM_ 前缀
                self._set_nested_value(config_key, self._parse_env_value(value))

    def _set_nested_value(self, key: str, value: Any) -> None:
        """设置嵌套配置值（支持点号分隔的路径）"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值的类型"""
        # 尝试解析为布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 返回字符串
        return value

    def get(self, *keys, default=None) -> Any:
        """
        获取配置值

        Args:
            *keys: 配置键路径，可以多级，例如 get('peak_detection', 'threshold')
            default: 默认值

        Returns:
            配置值，如果不存在则返回默认值
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # ========== 便捷访问方法 ==========

    @property
    def processing_mode(self) -> str:
        """处理模式: screen, video, vein_following"""
        return self.get('processing_mode', default='video')

    @property
    def frame_rate(self) -> int:
        """捕获帧率"""
        return self.get('roi_capture', 'frame_rate', default=10)

    @property
    def roi1_config(self) -> Dict[str, int]:
        """ROI1 配置"""
        return self.get('roi_capture', 'default_config', default={
            'x1': 1280, 'y1': 80, 'x2': 1920, 'y2': 980
        })

    @property
    def roi2_extension_params(self) -> Dict[str, int]:
        """ROI2 扩展参数"""
        return self.get('roi_capture', 'roi2_config', 'extension_params', default={
            'left': 40, 'right': 40, 'top': 50, 'bottom': 30
        })

    @property
    def roi3_extension_params(self) -> Dict[str, int]:
        """ROI3 扩展参数"""
        return self.get('roi_capture', 'roi3_config', 'extension_params', default={
            'left': 30, 'right': 30, 'top': 50, 'bottom': 100
        })

    @property
    def peak_detection_threshold(self) -> float:
        """波峰检测阈值"""
        return self.get('peak_detection', 'threshold', default=95.0)

    @property
    def adaptive_threshold_enabled(self) -> bool:
        """是否启用自适应阈值"""
        return self.get('peak_detection', 'adaptive_threshold_enabled', default=True)

    @property
    def threshold_minimum(self) -> float:
        """阈值下限"""
        return self.get('peak_detection', 'threshold_minimum', default=40.0)

    @property
    def threshold_over_mean_ratio(self) -> float:
        """自适应阈值上浮比例"""
        return self.get('peak_detection', 'threshold_over_mean_ratio', default=0.1)

    @property
    def adaptive_window_seconds(self) -> float:
        """自适应阈值时间窗口（秒）"""
        return self.get('peak_detection', 'adaptive_window_seconds', default=3.0)

    @property
    def difference_threshold(self) -> float:
        """绿/红判定阈值"""
        return self.get('peak_detection', 'difference_threshold', default=1.8)

    @property
    def margin_frames(self) -> int:
        """峰间最小间隔"""
        return self.get('peak_detection', 'margin_frames', default=5)

    @property
    def silence_frames(self) -> int:
        """干净区间长度"""
        return self.get('peak_detection', 'silence_frames', default=5)

    @property
    def min_region_length(self) -> int:
        """最小波峰宽度"""
        return self.get('peak_detection', 'min_region_length', default=5)

    @property
    def pre_post_avg_frames(self) -> int:
        """前后平均帧数"""
        return self.get('peak_detection', 'pre_post_avg_frames', default=5)

    @property
    def threshold_protection_enabled(self) -> bool:
        """是否启用阈值保护"""
        return self.get('peak_detection', 'threshold_protection', 'enabled', default=True)

    @property
    def threshold_protection_recovery_delay(self) -> float:
        """阈值保护恢复延迟（秒）"""
        return self.get('peak_detection', 'threshold_protection', 'recovery_delay_seconds', default=1.0)

    @property
    def threshold_protection_stability_frames(self) -> int:
        """阈值保护稳定性帧数"""
        return self.get('peak_detection', 'threshold_protection', 'stability_frames', default=5)

    @property
    def threshold_protection_waveform_trigger(self) -> bool:
        """阈值保护波形触发"""
        return self.get('peak_detection', 'threshold_protection', 'waveform_trigger_enabled', default=True)

    @property
    def g1_g2_override_enabled(self) -> bool:
        """G1/G2 覆盖是否启用"""
        return self.get('peak_detection', 'g1_g2_override', 'enabled', default=True)

    @property
    def g1_threshold(self) -> float:
        """G1 阈值（百分比）"""
        return self.get('peak_detection', 'g1_g2_override', 'g1_threshold', default=98.0)

    @property
    def g2_threshold(self) -> float:
        """G2 阈值（百分比）"""
        return self.get('peak_detection', 'g1_g2_override', 'g2_threshold', default=20.0)

    @property
    def roi3_column_diff_override_enabled(self) -> bool:
        """ROI3 列差值覆盖是否启用"""
        return self.get('peak_detection', 'roi3_column_diff_override', 'enabled', default=True)

    @property
    def roi3_column_diff_threshold(self) -> float:
        """ROI3 列差值阈值"""
        return self.get('peak_detection', 'roi3_column_diff_override', 'threshold', default=15.0)

    @property
    def roi1_peak_detection_enabled(self) -> bool:
        """ROI1 波峰检测是否启用"""
        return self.get('roi1_peak_detection', 'enabled', default=True)

    @property
    def roi1_threshold(self) -> float:
        """ROI1 阈值"""
        return self.get('roi1_peak_detection', 'threshold', default=63.0)

    @property
    def hybrid_detection_enabled(self) -> bool:
        """混合检测是否启用"""
        return self.get('hybrid_detection', 'enabled', default=True)

    @property
    def hybrid_detection_strategy(self) -> str:
        """混合检测策略"""
        return self.get('hybrid_detection', 'detection_strategy', default='roi1_peaks_roi2_color')

    @property
    def roi2_anti_jitter_enabled(self) -> bool:
        """ROI2 防抖动是否启用"""
        return self.get('roi2_anti_jitter', 'enabled', default=True)

    @property
    def roi2_anti_jitter_algorithm(self) -> str:
        """ROI2 防抖动算法"""
        return self.get('roi2_anti_jitter', 'algorithm', default='threshold')

    @property
    def roi2_movement_threshold(self) -> float:
        """ROI2 移动阈值"""
        return self.get('roi2_anti_jitter', 'movement_threshold', default=20.0)

    @property
    def roi2_anti_jitter_alpha(self) -> float:
        """ROI2 EMA 平滑因子"""
        return self.get('roi2_anti_jitter', 'ema', 'alpha', default=0.25)

    @property
    def video_path(self) -> str:
        """视频路径"""
        return self.get('video_processing', 'video_path', default='video')

    @property
    def loop_enabled(self) -> bool:
        """是否循环视频"""
        return self.get('video_processing', 'loop_enabled', default=False)

    @property
    def processing_frame_rate(self) -> Optional[float]:
        """处理帧率覆盖"""
        fps = self.get('video_processing', 'processing_frame_rate', default='')
        return float(fps) if fps else None

    @property
    def save_roi1(self) -> bool:
        """是否保存 ROI1"""
        return self.get('data_processing', 'save_roi1', default=True)

    @property
    def save_roi2(self) -> bool:
        """是否保存 ROI2"""
        return self.get('data_processing', 'save_roi2', default=True)

    @property
    def save_roi3(self) -> bool:
        """是否保存 ROI3"""
        return self.get('data_processing', 'save_roi3', default=True)

    @property
    def save_wave(self) -> bool:
        """是否保存波形"""
        return self.get('data_processing', 'save_wave', default=True)

    @property
    def save_roi1_wave(self) -> bool:
        """是否保存 ROI1 波形"""
        return self.get('data_processing', 'save_roi1_wave', default=True)

    @property
    def only_detect(self) -> bool:
        """仅检测模式"""
        return self.get('data_processing', 'only_delect', default=False)

    @property
    def analysis_cache_enabled(self) -> bool:
        """分析缓存是否启用"""
        return self.get('analysis_cache', 'enabled', default=True)

    @property
    def analysis_cache_flush_every(self) -> int:
        """分析缓存刷新间隔"""
        return self.get('analysis_cache', 'flush_every', default=50)

    @property
    def deduplication_enabled(self) -> bool:
        """去重是否启用"""
        return self.get('deduplication', 'consecutive_deduplication_enabled', default=True)

    @property
    def consecutive_frame_window(self) -> int:
        """连续去重窗口"""
        return self.get('deduplication', 'consecutive_frame_window', default=40)

    @property
    def cross_color_deduplication_enabled(self) -> bool:
        """跨颜色去重是否启用"""
        return self.get('deduplication', 'cross_color_deduplication_enabled', default=True)

    @property
    def color_priority(self) -> Dict[str, int]:
        """颜色优先级"""
        return self.get('deduplication', 'color_priority', default={'green': 2, 'red': 1})

    @property
    def startup_cleanup_enabled(self) -> bool:
        """启动清理是否启用"""
        return self.get('startup_cleanup', 'enabled', default=True)

    @property
    def cleanup_export(self) -> bool:
        """清理导出目录"""
        return self.get('startup_cleanup', 'cleanup_export', default=True)

    @property
    def cleanup_tmp(self) -> bool:
        """清理临时目录"""
        return self.get('startup_cleanup', 'cleanup_tmp', default=True)

    @property
    def cleanup_logs(self) -> bool:
        """清理日志目录"""
        return self.get('startup_cleanup', 'cleanup_logs', default=True)

    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()
