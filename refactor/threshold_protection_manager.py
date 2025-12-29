"""
阈值保护管理器 - 防止波峰污染自适应阈值计算

SimpleFEM Refactored Version
"""

import logging
from typing import Tuple
from refactor.config_manager import ConfigManager


class ThresholdProtectionManager:
    """
    阈值保护管理器

    功能:
    - 防止波峰数据污染自适应阈值的背景均值计算
    - 支持波形触发和波峰触发两种激活模式
    - 智能退出机制（时间延迟 + 稳定性检查）
    """

    def __init__(self, config: ConfigManager):
        """
        初始化阈值保护管理器

        Args:
            config: 配置管理器
        """
        self._config = config

        # 状态变量
        self._protection_active = False
        self._protection_end_time = 0.0
        self._consecutive_below = 0
        self._last_waveform_time = 0.0
        self._frames_since_end = 0

    def reset(self) -> None:
        """重置保护状态"""
        self._protection_active = False
        self._protection_end_time = 0.0
        self._consecutive_below = 0
        self._last_waveform_time = 0.0
        self._frames_since_end = 0
        logging.info("[阈值保护] 状态已重置")

    def update(
        self,
        current_gray: float,
        current_threshold: float,
        has_peaks: bool,
        frame_time: float,
        frame_index: int,
        fps: float = 10.0
    ) -> Tuple[bool, int]:
        """
        更新阈值保护状态

        Args:
            current_gray: 当前灰度值
            current_threshold: 当前阈值
            has_peaks: 当前帧是否检测到波峰
            frame_time: 当前帧的时间戳
            frame_index: 当前帧索引
            fps: 帧率（用于计算帧数）

        Returns:
            Tuple[bool, int]: (should_protect, frames_since_end)
                should_protect: 是否应该保护（不更新背景均值）
                frames_since_end: 距离保护结束的帧数
        """
        if not self._config.threshold_protection_enabled:
            return False, 0

        current_time = frame_time
        self._frames_since_end = max(0, int((current_time - self._protection_end_time) / (1.0 / fps)))

        # 检查是否需要触发保护
        should_protect = self._protection_active

        # 1. 波形触发：当前灰度超过阈值时立即触发保护
        if self._config.threshold_protection_waveform_trigger and current_gray >= current_threshold:
            should_protect = True
            self._last_waveform_time = current_time
            if not self._protection_active:
                msg = f"[阈值保护] 帧{frame_index} 波形触发保护: 灰度={current_gray:.1f} >= 阈值={current_threshold:.1f}"
                logging.info(msg)
                print(msg)

        # 2. 波峰结果触发：检测到波峰时激活保护
        elif has_peaks and not self._protection_active:
            should_protect = True
            self._last_waveform_time = current_time
            msg = f"[阈值保护] 帧{frame_index} 波峰触发保护: 检测到波峰"
            logging.info(msg)
            print(msg)

        # 3. 检查是否可以解除保护
        if should_protect:
            # 计算应该的结束时间
            recovery_delay_frames = int(self._config.threshold_protection_recovery_delay * fps)
            planned_end_time = self._last_waveform_time + (recovery_delay_frames / fps)

            # 检查稳定性条件：连续多帧低于阈值
            if current_gray < current_threshold:
                self._consecutive_below += 1
            else:
                self._consecutive_below = 0

            # 智能退出：满足延迟时间和稳定性条件
            time_condition = current_time >= planned_end_time
            stability_condition = self._consecutive_below >= self._config.threshold_protection_stability_frames

            if time_condition and stability_condition:
                should_protect = False
                self._consecutive_below = 0
                self._frames_since_end = 0
                msg = (f"[阈值保护] 帧{frame_index} 解除保护: "
                       f"满足时间延迟({recovery_delay_frames}帧)和稳定性({self._config.threshold_protection_stability_frames}帧)条件")
                logging.info(msg)
                print(msg)
            else:
                # 更新结束时间
                self._protection_end_time = planned_end_time

        self._protection_active = should_protect
        return should_protect, self._frames_since_end

    @property
    def is_active(self) -> bool:
        """保护是否激活"""
        return self._protection_active

    @property
    def frames_since_end(self) -> int:
        """距离保护结束的帧数"""
        return self._frames_since_end
