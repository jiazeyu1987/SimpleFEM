"""
数据导出管理器 - 处理图像和波形文件导出

SimpleFEM Refactored Version
"""

import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple
from collections import deque

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from refactor.config_manager import ConfigManager


class DataExportManager:
    """
    数据导出管理器

    功能:
    - 创建和管理输出目录结构
    - 保存ROI1/ROI2/ROI3图像
    - 保存波形图
    - 生成ROI2波形标注图
    """

    def __init__(self, config: ConfigManager, session_id: str, video_path: Optional[str] = None):
        """
        初始化数据导出管理器

        Args:
            config: 配置管理器
            session_id: 会话ID
            video_path: 视频路径（可选）
        """
        self._config = config
        self._session_id = session_id
        self._video_path = video_path

        # 输出目录
        self._tmp_root: Optional[str] = None
        self._roi1_dir: Optional[str] = None
        self._roi2_dir: Optional[str] = None
        self._roi3_dir: Optional[str] = None
        self._wave_dir: Optional[str] = None
        self._wave1_dir: Optional[str] = None

        # 初始化目录
        self._create_directories()

    def _create_directories(self) -> None:
        """创建输出目录结构"""
        if self._config.processing_mode == "video" and self._video_path:
            # 批量模式：使用视频名称
            video_name = os.path.splitext(os.path.basename(self._video_path))[0]
            sanitized_name = self._sanitize_video_name(video_name)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._tmp_root = os.path.join(base_dir, "tmp", sanitized_name)
        else:
            # 屏幕模式：使用基于会话的命名
            session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._tmp_root = os.path.join(base_dir, "tmp", session_start)

        # 创建子目录
        self._roi1_dir = os.path.join(self._tmp_root, "roi1")
        self._roi2_dir = os.path.join(self._tmp_root, "roi2")
        self._roi3_dir = os.path.join(self._tmp_root, "roi3")
        self._wave_dir = os.path.join(self._tmp_root, "wave")
        self._wave1_dir = os.path.join(self._tmp_root, "wave1")

        # 根据配置创建目录
        if self._config.save_roi1 or self._config.save_roi2 or self._config.save_roi3 or \
           self._config.save_wave or self._config.save_roi1_wave:
            os.makedirs(self._tmp_root, exist_ok=True)

        if self._config.save_roi1:
            os.makedirs(self._roi1_dir, exist_ok=True)
        if self._config.save_roi2:
            os.makedirs(self._roi2_dir, exist_ok=True)
        if self._config.save_roi3:
            os.makedirs(self._roi3_dir, exist_ok=True)
        if self._config.save_wave:
            os.makedirs(self._wave_dir, exist_ok=True)
        if self._config.save_roi1_wave:
            os.makedirs(self._wave1_dir, exist_ok=True)

    def _sanitize_video_name(self, video_name: str) -> str:
        """清理视频名称用于文件夹创建"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', video_name)
        sanitized = sanitized.strip('._')[:50]
        return sanitized or f"video_{int(time.time())}"

    def save_roi1(self, roi1_image: Image.Image, frame_index: int, video_time: Optional[float] = None) -> None:
        """
        保存ROI1图像

        Args:
            roi1_image: ROI1图像
            frame_index: 帧索引
            video_time: 视频时间（秒，可选）- 注意：ROI1文件名不包含时间戳
        """
        if not self._config.save_roi1 or self._roi1_dir is None:
            return

        # ROI1文件名格式与原始代码保持一致：roi1_000093.png（无时间戳）
        filename = f"roi1_{frame_index:06d}.png"
        filepath = os.path.join(self._roi1_dir, filename)
        roi1_image.save(filepath)

    def save_roi2(self, roi2_image: Image.Image, frame_index: int, video_time: Optional[float] = None) -> None:
        """
        保存ROI2图像

        Args:
            roi2_image: ROI2图像
            frame_index: 帧索引
            video_time: 视频时间（秒，可选）
        """
        if not self._config.save_roi2 or self._roi2_dir is None:
            return

        filename = self._get_filename(frame_index, video_time, prefix="roi2")
        filepath = os.path.join(self._roi2_dir, filename)
        roi2_image.save(filepath)

    def save_roi3(self, roi3_image: Image.Image, frame_index: int, video_time: Optional[float] = None) -> None:
        """
        保存ROI3图像

        Args:
            roi3_image: ROI3图像
            frame_index: 帧索引
            video_time: 视频时间（秒，可选）
        """
        if not self._config.save_roi3 or self._roi3_dir is None:
            return

        filename = self._get_filename(frame_index, video_time, prefix="roi3")
        filepath = os.path.join(self._roi3_dir, filename)
        roi3_image.save(filepath)

    def save_waveform(
        self,
        gray_buffer: deque,
        green_peaks: list,
        red_peaks: list,
        threshold: float,
        frame_index: int,
        video_time: Optional[float] = None,
        roi2_image_path: Optional[str] = None
    ) -> None:
        """
        保存波形图

        Args:
            gray_buffer: 灰度值缓冲区
            green_peaks: 绿色波峰列表
            red_peaks: 红色波峰列表
            threshold: 阈值
            frame_index: 帧索引
            video_time: 视频时间（秒，可选）
            roi2_image_path: ROI2图像路径（用于标注）
        """
        if not self._config.save_wave or self._wave_dir is None:
            return

        filename = self._get_filename(frame_index, video_time, prefix="wave")
        filepath = os.path.join(self._wave_dir, filename)

        # 创建波形图
        plt.figure(figsize=(12, 6))
        plt.plot(list(gray_buffer), label='Gray Value', color='blue', linewidth=2)

        # 绘制阈值线
        plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold:.1f})')

        # 标注波峰
        for start, end in green_peaks:
            plt.axvspan(start, end, alpha=0.2, color='green')
        for start, end in red_peaks:
            plt.axvspan(start, end, alpha=0.2, color='red')

        # 标注ROI2图像（如果存在）
        if roi2_image_path and os.path.exists(roi2_image_path):
            try:
                roi2_img = plt.imread(roi2_image_path)
                # 在波形图下方添加ROI2图像
                fig = plt.gcf()
                new_ax = fig.add_axes([0.15, 0.02, 0.7, 0.15])
                new_ax.imshow(roi2_img)
                new_ax.axis('off')
            except Exception as e:
                print(f"警告: 无法加载ROI2图像进行标注: {e}")

        plt.xlabel('Frame Index')
        plt.ylabel('Gray Value')
        plt.title(f'Waveform - Frame {frame_index}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close()

    def save_roi1_waveform(
        self,
        roi1_buffer: deque,
        threshold: float,
        frame_index: int,
        video_time: Optional[float] = None,
        bg_mean: float = 0.0,
        protection_active: bool = False,
        roi1_green_peaks: list = None,
        roi1_red_peaks: list = None,
        roi3_80_160_buffer: deque = None
    ) -> None:
        """
        保存ROI1波形图（与原始代码保持一致）

        Args:
            roi1_buffer: ROI1灰度值缓冲区
            threshold: 阈值
            frame_index: 帧索引
            video_time: 视频时间（秒，可选）
            bg_mean: ROI1背景均值
            protection_active: 阈值保护是否激活
            roi1_green_peaks: ROI1绿色波峰列表 [(start, end), ...]
            roi1_red_peaks: ROI1红色波峰列表 [(start, end), ...]
            roi3_80_160_buffer: ROI3(80-160)百分比缓冲区
        """
        if not self._config.save_roi1_wave or self._wave1_dir is None:
            return

        # 文件名格式与原始代码保持一致：roi1_wave_000005.png
        filename = f"roi1_wave_{frame_index:06d}.png"
        filepath = os.path.join(self._wave1_dir, filename)

        # 创建ROI1波形图（与原始代码保持一致的样式）
        fig, ax = plt.subplots(figsize=(8, 3))
        x = list(range(len(roi1_buffer)))
        ax.plot(x, roi1_buffer, color="darkblue", linewidth=1, label="ROI1")

        # 绘制ROI1背景均值
        if bg_mean > 0:
            ax.axhline(
                bg_mean,
                color="blue",
                linestyle="--",
                linewidth=1,
                label="bg_mean",
            )

        # 绘制ROI1阈值（根据保护状态显示不同颜色和样式）
        threshold_color = "red" if protection_active else "orange"
        threshold_style = "--" if protection_active else "-"
        ax.axhline(
            threshold,
            color=threshold_color,
            linestyle=threshold_style,
            linewidth=1.5,
            label=f"threshold ({threshold:.1f}{'[PROTECTED]' if protection_active else ''})",
        )

        # 添加ROI3(80-160)百分比红色曲线
        if roi3_80_160_buffer and len(roi3_80_160_buffer) > 0:
            x3_80_160 = list(range(len(roi3_80_160_buffer)))
            ax.plot(x3_80_160, list(roi3_80_160_buffer), color="red", linewidth=1, label="ROI3(80-160)%")

        # 高亮ROI1绿色波峰区域
        if roi1_green_peaks:
            for start, end in roi1_green_peaks:
                s = max(0, start - 1)
                e = min(len(roi1_buffer) - 1, end + 1)
                xs = list(range(s, e + 1))
                ys = list(roi1_buffer)[s : e + 1]
                ax.plot(xs, ys, color="green", linewidth=2)

        # 高亮ROI1红色波峰区域
        if roi1_red_peaks:
            for start, end in roi1_red_peaks:
                s = max(0, start - 1)
                e = min(len(roi1_buffer) - 1, end + 1)
                xs = list(range(s, e + 1))
                ys = list(roi1_buffer)[s : e + 1]
                ax.plot(xs, ys, color="red", linewidth=2)

        # 设置图表标题和标签（与原始代码保持一致）
        ax.set_title(f"ROI1 Waveform - Frame {frame_index} (len={len(roi1_buffer)})")
        ax.set_xlabel("Frame Index (relative)")
        ax.set_ylabel("Gray Value (0-255)")
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_filename(self, frame_index: int, video_time: Optional[float], prefix: str) -> str:
        """
        生成文件名

        Args:
            frame_index: 帧索引
            video_time: 视频时间（秒）
            prefix: 文件名前缀

        Returns:
            文件名
        """
        if video_time is not None:
            return f"{prefix}_{frame_index:06d}_{video_time:06.2f}s.png"
        return f"{prefix}_{frame_index:06d}.png"

    @property
    def tmp_root(self) -> Optional[str]:
        """临时根目录"""
        return self._tmp_root
