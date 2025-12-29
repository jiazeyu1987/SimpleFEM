"""
ROI捕获管理器 - 处理屏幕和视频捕获

SimpleFEM Refactored Version
"""

import glob
import os
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageGrab

from refactor.config_manager import ConfigManager


class ROICaptureManager:
    """
    ROI捕获管理器

    功能:
    - 屏幕捕获（PIL.ImageGrab）
    - 视频文件捕获（OpenCV）
    - ROI1/ROI2/ROI3 提取
    - 批量视频处理
    - 帧率控制
    """

    def __init__(self, config: ConfigManager):
        """
        初始化ROI捕获管理器

        Args:
            config: 配置管理器
        """
        self._config = config
        self._processing_mode = config.processing_mode

        # 视频捕获相关
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._video_files: List[str] = []
        self._current_video_index = 0
        self._video_fps = 30.0

        # 屏幕尺寸
        self._screen_width = 1920
        self._screen_height = 1080

        # ROI1 缓冲区
        self._roi1_buffer: deque = deque(maxlen=100)

        # ROI2/ROI3 缓冲区
        self._roi2_buffer: deque = deque(maxlen=100)
        self._roi3_buffer: deque = deque(maxlen=100)
        self._roi3_80_160_buffer: deque = deque(maxlen=100)  # ROI3(80-160)百分比缓冲区
        self._roi3_g1_buffer: deque = deque(maxlen=100)  # ROI3 G1(80-255)百分比缓冲区
        self._roi3_g2_buffer: deque = deque(maxlen=100)  # ROI3 G2(150-255)百分比缓冲区
        self._roi3_column_diff_buffer: deque = deque(maxlen=100)  # ROI3列灰度差值缓冲区

        # 首帧标志（用于视频模式第一帧不跳帧）
        self._first_video_frame = True

        self._initialize_capture_mode()

    def _initialize_capture_mode(self) -> None:
        """初始化捕获模式"""
        if self._processing_mode == "video":
            self._initialize_video_mode()
        elif self._processing_mode == "screen":
            self._initialize_screen_mode()
        elif self._processing_mode == "vein_following":
            self._initialize_screen_mode()

    def _initialize_screen_mode(self) -> None:
        """初始化屏幕模式"""
        try:
            screen = ImageGrab.grab()
            self._screen_width, self._screen_height = screen.size
        except Exception as e:
            print(f"警告: 无法获取屏幕尺寸，使用默认值 1920x1080: {e}")

    def _initialize_video_mode(self) -> None:
        """初始化视频模式"""
        video_path = self._config.video_path

        if os.path.isdir(video_path):
            # 批量视频处理
            self._video_files = self._discover_video_files(video_path)
            print(f"[视频模式] 发现 {len(self._video_files)} 个视频文件")
        elif os.path.isfile(video_path):
            # 单个视频文件
            self._video_files = [video_path]
        else:
            raise ValueError(f"视频路径不存在: {video_path}")

        if self._video_files:
            self._open_video(self._video_files[0])

    def _discover_video_files(self, video_path: str) -> List[str]:
        """发现文件夹中的所有视频文件"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']

        video_files = []
        for ext in video_extensions:
            pattern = os.path.join(video_path, ext)
            video_files.extend(glob.glob(pattern))
            pattern = os.path.join(video_path, ext.upper())
            video_files.extend(glob.glob(pattern))

        return sorted(list(set(video_files)))

    def _open_video(self, video_path: str) -> None:
        """打开视频文件"""
        self._video_cap = cv2.VideoCapture(video_path)
        if not self._video_cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self._video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._video_fps = self._get_video_fps(self._video_cap)

        print(f"[视频模式] 打开视频: {os.path.basename(video_path)}, FPS: {self._video_fps:.2f}")

    def _get_video_fps(self, video_cap: cv2.VideoCapture) -> float:
        """获取视频帧率"""
        try:
            fps = float(video_cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = 0.0

        if not fps or fps <= 0:
            return 30.0

        return fps

    def discover_video_files(self, video_path: str) -> List[str]:
        """发现文件夹中的所有视频文件"""
        if not os.path.exists(video_path):
            raise ValueError(f"视频目录不存在: {video_path}")

        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']

        video_files = []
        for ext in video_extensions:
            pattern = os.path.join(video_path, ext)
            video_files.extend(glob.glob(pattern))
            pattern = os.path.join(video_path, ext.upper())
            video_files.extend(glob.glob(pattern))

        return sorted(list(set(video_files)))

    def initialize_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """初始化视频捕获器"""
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return video_cap

    def get_video_frame(self, loop_enabled: bool = False, frame_step: int = 1) -> Optional[Image.Image]:
        """
        从视频获取帧

        Args:
            loop_enabled: 是否循环播放
            frame_step: 跳帧步数（用于降低采样率）

        Returns:
            PIL图像或None
        """
        if self._video_cap is None:
            return None

        if frame_step is None:
            frame_step = 1
        try:
            frame_step = int(frame_step)
        except Exception:
            frame_step = 1
        frame_step = max(1, frame_step)

        frame = None
        for _ in range(frame_step):
            ret, frame = self._video_cap.read()
            if ret:
                continue
            if not loop_enabled:
                return None
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._video_cap.read()
            if not ret:
                return None

        if frame is not None:
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

        return None

    def capture_screen_roi(self, x1: int, y1: int, x2: int, y2: int) -> Optional[Image.Image]:
        """
        捕获屏幕ROI

        Args:
            x1, y1, x2, y2: ROI坐标

        Returns:
            PIL图像或None
        """
        try:
            # 调整坐标到屏幕边界
            x1 = max(0, min(x1, self._screen_width))
            y1 = max(0, min(y1, self._screen_height))
            x2 = max(0, min(x2, self._screen_width))
            y2 = max(0, min(y2, self._screen_height))

            if x2 <= x1 or y2 <= y1:
                return None

            roi = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            return roi
        except Exception as e:
            print(f"屏幕捕获失败: {e}")
            return None

    def capture_roi1(self) -> Optional[Image.Image]:
        """
        捕获ROI1

        Returns:
            PIL图像或None
        """
        roi1_cfg = self._config.roi1_config
        x1, y1 = roi1_cfg['x1'], roi1_cfg['y1']
        x2, y2 = roi1_cfg['x2'], roi1_cfg['y2']

        if self._processing_mode in ["screen", "vein_following"]:
            return self.capture_screen_roi(x1, y1, x2, y2)
        elif self._processing_mode == "video":
            # 计算帧步长（与原始代码保持一致）
            effective_frame_rate = min(self._config.frame_rate, self._video_fps)
            if self._video_fps > 0 and effective_frame_rate > 0:
                video_frame_step = max(1, int(round(self._video_fps / effective_frame_rate)))
            else:
                video_frame_step = 1

            # 第一帧使用step=1以避免跳过视频开头（与原始代码保持一致）
            frame_step = 1 if self._first_video_frame else video_frame_step
            self._first_video_frame = False

            # 先获取完整的视频帧
            full_frame = self.get_video_frame(
                loop_enabled=self._config.loop_enabled,
                frame_step=frame_step
            )
            if full_frame is None:
                return None

            # 然后根据ROI1配置裁剪（与原始代码保持一致）
            frame_width, frame_height = full_frame.size

            # 调试：打印坐标信息（每50帧打印一次）
            import sys
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 0
            if self._frame_count % 50 == 0:
                print(f"[DEBUG] 帧{self._frame_count}: 视频帧尺寸={frame_width}x{frame_height}, ROI1配置=({x1},{y1},{x2},{y2})")

            # 调整坐标到帧边界（与原始代码的adjust_roi1_to_screen逻辑完全一致）
            x1_adj = max(0, min(x1, frame_width - 1))
            y1_adj = max(0, min(y1, frame_height - 1))
            x2_adj = max(x1_adj + 1, min(x2, frame_width))
            y2_adj = max(y1_adj + 1, min(y2, frame_height))

            # 调试：打印调整后的坐标
            if self._frame_count % 50 == 0:
                print(f"[DEBUG] 帧{self._frame_count}: 调整后ROI1=({x1_adj},{y1_adj},{x2_adj},{y2_adj}), 尺寸={x2_adj-x1_adj}x{y2_adj-y1_adj}")

            return full_frame.crop((x1_adj, y1_adj, x2_adj, y2_adj))

        return None

    def extract_roi2(self, roi1_image: Image.Image, intersection_x: int, intersection_y: int) -> Optional[Image.Image]:
        """
        从ROI1提取ROI2

        Args:
            roi1_image: ROI1图像
            intersection_x: 交点X坐标（相对于ROI1）
            intersection_y: 交点Y坐标（相对于ROI1）
            extension_params: 扩展参数

        Returns:
            ROI2图像或None
        """
        ext_params = self._config.roi2_extension_params

        # 计算ROI2坐标（相对于ROI1）
        x1 = intersection_x - ext_params['left']
        y1 = intersection_y - ext_params['top']
        x2 = intersection_x + ext_params['right']
        y2 = intersection_y + ext_params['bottom']

        # 调整到ROI1边界
        roi1_width, roi1_height = roi1_image.size
        x1 = max(0, min(x1, roi1_width))
        y1 = max(0, min(y1, roi1_height))
        x2 = max(0, min(x2, roi1_width))
        y2 = max(0, min(y2, roi1_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return roi1_image.crop((x1, y1, x2, y2))

    def extract_roi3(self, roi1_image: Image.Image, intersection_x: int, intersection_y: int) -> Optional[Image.Image]:
        """
        从ROI1提取ROI3

        Args:
            roi1_image: ROI1图像
            intersection_x: 交点X坐标（相对于ROI1）
            intersection_y: 交点Y坐标（相对于ROI1）

        Returns:
            ROI3图像或None
        """
        ext_params = self._config.roi3_extension_params

        # 计算ROI3坐标（相对于ROI1）
        x1 = intersection_x - ext_params['left']
        y1 = intersection_y - ext_params['top']
        x2 = intersection_x + ext_params['right']
        y2 = intersection_y + ext_params['bottom']

        # 调整到ROI1边界
        roi1_width, roi1_height = roi1_image.size
        x1 = max(0, min(x1, roi1_width))
        y1 = max(0, min(y1, roi1_height))
        x2 = max(0, min(x2, roi1_width))
        y2 = max(0, min(y2, roi1_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return roi1_image.crop((x1, y1, x2, y2))

    def compute_average_gray(self, image: Image.Image) -> float:
        """
        计算图像平均灰度值

        Args:
            image: PIL图像

        Returns:
            平均灰度值
        """
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        return float(np.mean(gray_array))

    @property
    def roi1_buffer(self) -> deque:
        """ROI1缓冲区"""
        return self._roi1_buffer

    @property
    def roi2_buffer(self) -> deque:
        """ROI2缓冲区"""
        return self._roi2_buffer

    @property
    def roi3_buffer(self) -> deque:
        """ROI3缓冲区"""
        return self._roi3_buffer

    @property
    def roi3_80_160_buffer(self) -> deque:
        """ROI3(80-160)百分比缓冲区"""
        return self._roi3_80_160_buffer

    @property
    def roi3_g1_buffer(self) -> deque:
        """ROI3 G1(80-255)百分比缓冲区"""
        return self._roi3_g1_buffer

    @property
    def roi3_g2_buffer(self) -> deque:
        """ROI3 G2(150-255)百分比缓冲区"""
        return self._roi3_g2_buffer

    @property
    def roi3_column_diff_buffer(self) -> deque:
        """ROI3列灰度差值缓冲区"""
        return self._roi3_column_diff_buffer

    @property
    def current_video_path(self) -> Optional[str]:
        """当前视频路径"""
        if self._current_video_index < len(self._video_files):
            return self._video_files[self._current_video_index]
        return None

    @property
    def video_count(self) -> int:
        """视频总数"""
        return len(self._video_files)

    def next_video(self) -> bool:
        """
        切换到下一个视频

        Returns:
            是否成功切换
        """
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

        self._current_video_index += 1

        if self._current_video_index < len(self._video_files):
            self._open_video(self._video_files[self._current_video_index])
            return True

        return False

    def reset_buffers(self) -> None:
        """重置所有缓冲区"""
        self._roi1_buffer.clear()
        self._roi2_buffer.clear()
        self._roi3_buffer.clear()
        self._roi3_80_160_buffer.clear()
        self._roi3_g1_buffer.clear()
        self._roi3_g2_buffer.clear()
        self._roi3_column_diff_buffer.clear()
        self._first_video_frame = True  # 重置首帧标志

    def close(self) -> None:
        """关闭资源"""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
