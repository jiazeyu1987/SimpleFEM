from __future__ import annotations

import logging
from typing import Tuple


def manage_threshold_protection(
    current_gray: float,
    current_threshold: float,
    has_peaks: bool,
    frame_time: float,
    frame_index: int,
    # State variables (passed by reference)
    protection_active: bool,
    protection_end_time: float,
    consecutive_below: int,
    last_waveform_time: float,
    # Configuration
    enabled: bool = True,
    recovery_delay_frames: int = 10,
    stability_frames: int = 5,
    waveform_trigger: bool = True,
    threshold_minimum: float = 80.0,
) -> Tuple[bool, float, int, int, float]:
    """
    管理阈值保护状态

    Args:
        current_gray: 当前灰度值
        current_threshold: 当前阈值
        has_peaks: 当前帧是否检测到波峰
        frame_time: 当前帧的时间戳
        frame_index: 当前帧索引
        protection_active: 保护状态是否激活
        protection_end_time: 保护结束时间
        consecutive_below: 连续低于阈值的帧数
        last_waveform_time: 上次波形时间
        enabled: 是否启用保护机制
        recovery_delay_frames: 恢复延迟帧数
        stability_frames: 稳定性帧数要求
        waveform_trigger: 是否启用波形触发
        threshold_minimum: 阈值下限，确保阈值不会低于此值

    Returns:
        Tuple[bool, float, int, int, float]:
            (should_protect, new_end_time, new_consecutive_below, frames_since_end, new_waveform_time)
    """
    current_time = frame_time
    frames_since_end = max(0, int((current_time - protection_end_time) / (1.0 / 10)))  # 假设10fps

    if not enabled:
        return False, protection_end_time, consecutive_below, frames_since_end, last_waveform_time

    # 检查是否需要触发保护
    should_protect = protection_active

    # 1. 波形触发：当前灰度超过阈值时立即触发保护
    if waveform_trigger and current_gray >= current_threshold:
        should_protect = True
        last_waveform_time = current_time
        if not protection_active:
            msg = (
                f"[阈值保护] 帧{frame_index} 波形触发保护: 灰度={current_gray:.1f} >= 阈值={current_threshold:.1f}"
            )
            logging.info(msg)
            print(msg)

    # 2. 波峰结果触发：检测到波峰时激活保护
    elif has_peaks and not protection_active:
        should_protect = True
        last_waveform_time = current_time
        msg = f"[阈值保护] 帧{frame_index} 波峰触发保护: 检测到波峰"
        logging.info(msg)
        print(msg)

    # 3. 检查是否可以解除保护
    if should_protect:
        # 计算应该的结束时间
        planned_end_time = last_waveform_time + (recovery_delay_frames * 0.1)  # 0.1秒每帧

        # 检查稳定性条件：连续多帧低于阈值
        if current_gray < current_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        # 智能退出：满足延迟时间和稳定性条件
        time_condition = current_time >= planned_end_time
        stability_condition = consecutive_below >= stability_frames

        if time_condition and stability_condition:
            should_protect = False
            consecutive_below = 0
            frames_since_end = 0
            msg = (
                f"[阈值保护] 帧{frame_index} 解除保护: 满足时间延迟({recovery_delay_frames}帧)和稳定性({stability_frames}帧)条件"
            )
            logging.info(msg)
            print(msg)
        else:
            # 更新结束时间
            protection_end_time = planned_end_time

    return should_protect, protection_end_time, consecutive_below, frames_since_end, last_waveform_time

