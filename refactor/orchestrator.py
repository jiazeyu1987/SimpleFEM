"""
主编排器 - 协调所有组件完成ROI检测和波峰分析

SimpleFEM Refactored Version
"""

import logging
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 导入管理类
from refactor.config_manager import ConfigManager
from refactor.threshold_protection_manager import ThresholdProtectionManager
from refactor.roi_capture_manager import ROICaptureManager
from refactor.green_line_manager import GreenLineManager
from refactor.data_export_manager import DataExportManager
from refactor.analysis_cache_manager import AnalysisCacheManager
from refactor.statistics_manager import StatisticsManager
from refactor.hybrid_detection_manager import HybridDetectionManager
from refactor.roi3_statistics import ROI3Statistics

# 添加父目录到路径以导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_detection import detect_peaks


class Orchestrator:
    """
    主编排器

    功能:
    - 协调所有组件完成完整的检测流程
    - 管理帧处理循环
    - 处理多视频批量处理
    - 统一日志记录
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化编排器

        Args:
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        self._config = ConfigManager(config_path)

        # 初始化管理器
        self._threshold_protection = ThresholdProtectionManager(self._config)
        self._roi_capture = ROICaptureManager(self._config)
        self._green_line = GreenLineManager(self._config)
        self._hybrid_detection = HybridDetectionManager(self._config)

        # 会话ID
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 导出目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        export_dir = os.path.join(base_dir, "export")
        os.makedirs(export_dir, exist_ok=True)

        # 数据导出和分析缓存
        self._data_export: Optional[DataExportManager] = None
        self._analysis_cache = AnalysisCacheManager(self._config, export_dir)
        self._statistics = StatisticsManager(self._config)

        # 处理状态
        self._frame_index = 0
        self._bg_count = 0
        self._bg_mean = 0.0
        self._roi1_peak_counter: Dict[str, int] = {}

        # 设置日志
        self._setup_logging()

    def _setup_logging(self) -> None:
        """设置日志"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"roi_peak_daemon_{datetime.now().strftime('%Y-%m-%d')}.log")

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logging.info(f"SimpleFEM 重构版本启动 - 会话ID: {self._session_id}")

    def _cleanup_directories(self) -> None:
        """清理目录"""
        if not self._config.startup_cleanup_enabled:
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        dirs_to_clean = []
        if self._config.cleanup_export:
            dirs_to_clean.append(os.path.join(base_dir, "export"))
        if self._config.cleanup_tmp:
            dirs_to_clean.append(os.path.join(base_dir, "tmp"))
        if self._config.cleanup_logs:
            dirs_to_clean.append(os.path.join(base_dir, "logs"))

        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                try:
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                    logging.info(f"清理目录: {dir_path}")
                except Exception as e:
                    logging.warning(f"清理目录失败 {dir_path}: {e}")

    def run(self) -> None:
        """运行主循环"""
        # 清理目录
        self._cleanup_directories()

        # 启动分析缓存会话
        self._analysis_cache.start_session(
            session_id=self._session_id,
            processing_mode=self._config.processing_mode,
            video_path=self._config.video_path if self._config.processing_mode == "video" else None,
            config=self._config.get_full_config()
        )

        if self._config.processing_mode == "video":
            self._run_video_mode()
        elif self._config.processing_mode == "screen":
            self._run_screen_mode()
        elif self._config.processing_mode == "vein_following":
            self._run_vein_following_mode()

        # 关闭分析缓存
        self._analysis_cache.close(reason="normal")
        self._statistics.export_final_csv()

        # 打印汇总
        summary = self._statistics.get_global_summary()
        logging.info(f"\n处理完成汇总:")
        logging.info(f"  总视频数: {summary['total_videos_processed']}")
        logging.info(f"  总波峰数: {summary['total_peaks']}")
        logging.info(f"  绿色波峰: {summary['total_green_peaks']}")
        logging.info(f"  红色波峰: {summary['total_red_peaks']}")
        logging.info(f"  会话时长: {summary['session_duration']}")

    def _run_video_mode(self) -> None:
        """运行视频模式"""
        video_count = self._roi_capture.video_count

        for video_idx in range(video_count):
            video_path = self._roi_capture.current_video_path
            if video_path is None:
                break

            logging.info(f"\n处理视频 [{video_idx + 1}/{video_count}]: {os.path.basename(video_path)}")

            # 初始化统计
            self._statistics.initialize_for_video(video_path, is_batch=(video_count > 1))

            # 初始化数据导出
            self._data_export = DataExportManager(self._config, self._session_id, video_path)

            # 重置状态
            self._reset_video_state()

            # 处理视频帧
            self._process_video_frames()

            # 切换到下一个视频
            if video_idx < video_count - 1:
                self._roi_capture.next_video()

    def _run_screen_mode(self) -> None:
        """运行屏幕模式"""
        logging.info("屏幕捕获模式启动")

        # 初始化统计
        self._statistics.initialize_for_video("screen_capture", is_batch=False)

        # 初始化数据导出
        self._data_export = DataExportManager(self._config, self._session_id, None)

        # 重置状态
        self._reset_video_state()

        # 处理屏幕帧
        self._process_screen_frames()

    def _run_vein_following_mode(self) -> None:
        """运行静脉跟随模式"""
        logging.info("静脉跟随模式启动")
        # TODO: 实现静脉跟随模式
        pass

    def _process_video_frames(self) -> None:
        """处理视频帧"""
        frame_time = time.time()
        fps = self._roi_capture._video_fps

        while True:
            # 捕获ROI1
            roi1_image = self._roi_capture.capture_roi1()
            if roi1_image is None:
                logging.info("视频处理完成")
                break

            # 计算视频时间
            video_time = self._frame_index / fps

            # 处理帧
            self._process_frame(roi1_image, frame_time, video_time)

            # 更新帧索引
            self._frame_index += 1
            frame_time = time.time()

            # 帧率控制
            target_interval = 1.0 / self._config.frame_rate
            elapsed = time.time() - frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def _process_screen_frames(self) -> None:
        """处理屏幕帧"""
        frame_time = time.time()

        try:
            while True:
                # 捕获ROI1
                roi1_image = self._roi_capture.capture_roi1()
                if roi1_image is None:
                    logging.warning("屏幕捕获失败，跳过此帧")
                    time.sleep(1.0 / self._config.frame_rate)
                    continue

                # 处理帧
                self._process_frame(roi1_image, frame_time, None)

                # 更新帧索引
                self._frame_index += 1
                frame_time = time.time()

                # 帧率控制
                target_interval = 1.0 / self._config.frame_rate
                elapsed = time.time() - frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        except KeyboardInterrupt:
            logging.info("用户中断，停止处理")

    def _process_frame(self, roi1_image: Image.Image, frame_time: float, video_time: Optional[float]) -> None:
        """
        处理单帧（增强版 - 包含 ROI3 统计和混合检测）

        Args:
            roi1_image: ROI1图像
            frame_time: 帧时间戳
            video_time: 视频时间（秒，可选）
        """
        # 1. 计算ROI1平均灰度（仅在启用ROI1检测时填充缓冲区）
        roi1_enabled = self._config.roi1_peak_detection_enabled
        roi1_gray: Optional[float] = None
        if roi1_enabled:
            roi1_gray = self._roi_capture.compute_average_gray(roi1_image)
            self._roi_capture.roi1_buffer.append(roi1_gray)

        # 2. 检测绿线交点
        intersection = self._green_line.detect_intersection(roi1_image)
        if intersection is None:
            intersection = self._green_line.get_last_valid_intersection()

        # 3. 提取ROI2/ROI3
        roi2_image = None
        roi3_image = None
        roi2_gray = 0.0
        roi3_gray = 0.0
        roi3_stats = {}

        if intersection is not None:
            ix, iy = intersection
            roi2_image = self._roi_capture.extract_roi2(roi1_image, ix, iy)
            roi3_image = self._roi_capture.extract_roi3(roi1_image, ix, iy)

            if roi2_image is not None:
                roi2_gray = self._roi_capture.compute_average_gray(roi2_image)
                self._roi_capture.roi2_buffer.append(roi2_gray)

            if roi3_image is not None:
                roi3_gray = self._roi_capture.compute_average_gray(roi3_image)
                self._roi_capture.roi3_buffer.append(roi3_gray)

                # 计算 ROI3 统计（G1/G2, column diff, 80-160 percentage）
                roi3_stats = ROI3Statistics.compute_all(roi3_image)

                # 将80-160百分比加入缓冲区（用于ROI1波形图）
                self._roi_capture.roi3_80_160_buffer.append(roi3_stats.get('percentage_80_160', 0.0))

                # 将G1/G2百分比加入缓冲区（用于混合检测G1/G2覆盖）
                self._roi_capture.roi3_g1_buffer.append(roi3_stats.get('g1_percent', 0.0))
                self._roi_capture.roi3_g2_buffer.append(roi3_stats.get('g2_percent', 0.0))

                # 将列灰度差值加入缓冲区（用于混合检测列差值覆盖）
                self._roi_capture.roi3_column_diff_buffer.append(roi3_stats.get('column_diff', 0.0))

        # 4. 计算自适应阈值
        threshold = self._compute_adaptive_threshold(roi2_gray)

        # 5. 阈值保护更新
        should_protect, _ = self._threshold_protection.update(
            current_gray=roi2_gray,
            current_threshold=threshold,
            has_peaks=False,  # 稍后检测
            frame_time=frame_time,
            frame_index=self._frame_index,
            fps=self._config.frame_rate
        )

        # 6. 波峰检测（混合检测或ROI2独立检测）
        green_peaks, red_peaks = [], []
        hybrid_green_peaks, hybrid_red_peaks, roi1_green_peaks, roi1_red_peaks, hybrid_info = [], [], [], [], []

        # 判断是否使用混合检测
        hybrid_enabled = self._config.hybrid_detection_enabled
        roi1_enabled = self._config.roi1_peak_detection_enabled
        roi1_buffer = list(self._roi_capture.roi1_buffer)

        if hybrid_enabled and roi1_enabled and len(roi1_buffer) > 0:
            # 情况1: 混合检测（ROI1 检测波峰时机 + ROI2 判定颜色）
            if self._frame_index % 50 == 0:
                logging.info(f"帧{self._frame_index}: 使用混合检测模式（ROI1缓冲区={len(roi1_buffer)}帧）")

            hybrid_green_peaks, hybrid_red_peaks, roi1_green_peaks, roi1_red_peaks, hybrid_info = self._hybrid_detection.detect_hybrid_peaks(
                roi1_buffer,
                list(self._roi_capture.roi2_buffer),
                self._frame_index,
                intersection,
                roi3_g1_curve=list(self._roi_capture.roi3_g1_buffer),
                roi3_g2_curve=list(self._roi_capture.roi3_g2_buffer),
                roi3_column_diff_curve=list(self._roi_capture.roi3_column_diff_buffer)
            )
            # 使用混合检测结果（包括绿色和红色波峰）
            green_peaks = hybrid_green_peaks
            red_peaks = hybrid_red_peaks

        elif hybrid_enabled and roi1_enabled:
            # 情况2: ROI1 数据不足
            if self._frame_index % 50 == 0:
                logging.warning(f"帧{self._frame_index}: ROI1数据不足（len={len(roi1_buffer)}），跳过波峰检测（不回退到ROI2）")
                logging.warning(f"提示：如果一直处于此状态，请考虑关闭混合检测或ROI1检测以使用ROI2独立检测")

            # 保持空列表，等待ROI1缓冲区积累足够数据
            pass

        else:
            # 情况3: ROI2 独立检测（传统模式）
            if self._frame_index % 50 == 0:
                logging.info(f"帧{self._frame_index}: 使用ROI2独立检测模式")

            green_peaks, red_peaks = detect_peaks(
                list(self._roi_capture.roi2_buffer),
                threshold,
                difference_threshold=self._config.difference_threshold,
                margin_frames=self._config.margin_frames,
                silence_frames=self._config.silence_frames,
                avgFrames=self._config.pre_post_avg_frames,  # 添加缺失的参数
                min_region_length=self._config.min_region_length
            )

        # 更新阈值保护（基于波峰检测结果）
        has_peaks = len(green_peaks) > 0 or len(red_peaks) > 0
        if has_peaks:
            self._threshold_protection.update(
                current_gray=roi2_gray,
                current_threshold=threshold,
                has_peaks=True,
                frame_time=frame_time,
                frame_index=self._frame_index,
                fps=self._config.frame_rate
            )

        # 7. 记录分析缓存（增强版 - 包含 ROI3 统计）
        cache_payload = {
            "frame_index": self._frame_index,
            "timestamp": datetime.fromtimestamp(frame_time).isoformat(),
            "roi1_avg": roi1_gray if roi1_gray is not None else 0.0,
            "roi2_avg": roi2_gray,
            "roi3_avg": roi3_gray,
            "intersection": {"x": intersection[0], "y": intersection[1]} if intersection else None,
            "threshold": threshold,
            "green_peaks": green_peaks,
            "red_peaks": red_peaks,
            "protection_active": should_protect,
            "roi3_g1_percent": roi3_stats.get('g1_percent', 0),
            "roi3_g2_percent": roi3_stats.get('g2_percent', 0),
            "roi3_column_diff": roi3_stats.get('column_diff', 0),
            "hybrid_detection_enabled": self._config.hybrid_detection_enabled,
            "hybrid_green_peaks": len(hybrid_green_peaks) if self._config.hybrid_detection_enabled else 0,
            "hybrid_red_peaks": len(hybrid_red_peaks) if self._config.hybrid_detection_enabled else 0
        }
        self._analysis_cache.record_frame(cache_payload)

        # 8. 保存图像和波形
        if self._data_export:
            self._data_export.save_roi1(roi1_image, self._frame_index, video_time)
            if roi2_image:
                self._data_export.save_roi2(roi2_image, self._frame_index, video_time)
            if roi3_image:
                self._data_export.save_roi3(roi3_image, self._frame_index, video_time)

            # 如果有波峰，保存波形图
            if has_peaks:
                roi2_path = os.path.join(self._data_export._roi2_dir, f"roi2_{self._frame_index:06d}.png") if self._data_export._roi2_dir else None
                self._data_export.save_waveform(
                    self._roi_capture.roi2_buffer,
                    green_peaks,
                    red_peaks,
                    threshold,
                    self._frame_index,
                    video_time,
                    roi2_path
                )

            # 保存ROI1波形（传递所有参数以与原始代码保持一致）
            # 注意：传递的是ROI1的原始波峰（roi1_green_peaks, roi1_red_peaks），而不是混合检测后的波峰
            self._data_export.save_roi1_waveform(
                self._roi_capture.roi1_buffer,
                self._config.roi1_threshold,
                self._frame_index,
                video_time,
                bg_mean=self._bg_mean,
                protection_active=should_protect,
                roi1_green_peaks=roi1_green_peaks if (hybrid_enabled and roi1_enabled and len(roi1_buffer) > 0) else green_peaks,
                roi1_red_peaks=roi1_red_peaks if (hybrid_enabled and roi1_enabled and len(roi1_buffer) > 0) else red_peaks,
                roi3_80_160_buffer=self._roi_capture.roi3_80_160_buffer
            )

        # 9. 添加到统计
        if has_peaks:
            roi2_info = {
                "x1": 0, "y1": 0, "x2": 0, "y2": 0,  # TODO: 填充实际坐标
                "width": roi2_image.size[0] if roi2_image else 0,
                "height": roi2_image.size[1] if roi2_image else 0
            }

            # 将混合检测的info列表转换为原始代码期望的格式
            # 如果使用混合检测，传递hybrid_info；否则传递空列表
            hybrid_peaks_for_stats = hybrid_info if (hybrid_enabled and roi1_enabled and len(roi1_buffer) > 0) else []

            self._statistics.add_peaks(
                frame_index=self._frame_index,
                green_peaks=green_peaks,
                red_peaks=red_peaks,
                curve_data=list(self._roi_capture.roi2_buffer),
                intersection=intersection,
                roi2_info=roi2_info,
                gray_value=roi2_gray,
                threshold_used=threshold,
                bg_mean=self._bg_mean,
                roi3_curve=list(self._roi_capture.roi3_buffer) if len(self._roi_capture.roi3_buffer) > 0 else None,
                hybrid_enabled=self._config.hybrid_detection_enabled,
                hybrid_peaks=hybrid_peaks_for_stats
            )

        # 日志输出（增强版 - 包含 ROI3 统计）
        roi1_display = roi1_gray if roi1_gray is not None else 0.0
        if roi3_stats:
            logging.info(
                f"帧{self._frame_index:06d} | ROI1={roi1_display:.1f} | ROI2={roi2_gray:.1f} | "
                f"阈值={threshold:.1f} | 绿峰={len(green_peaks)} | 红峰={len(red_peaks)} | "
                f"保护={should_protect} | G1={roi3_stats.get('g1_percent', 0):.1f}% | "
                f"G2={roi3_stats.get('g2_percent', 0):.1f}% | 列差={roi3_stats.get('column_diff', 0):.1f}"
            )
        else:
            logging.info(
                f"帧{self._frame_index:06d} | ROI1={roi1_display:.1f} | ROI2={roi2_gray:.1f} | "
                f"阈值={threshold:.1f} | 绿峰={len(green_peaks)} | 红峰={len(red_peaks)} | "
                f"保护={should_protect}"
            )

    def _compute_adaptive_threshold(self, current_gray: float) -> float:
        """
        计算自适应阈值

        Args:
            current_gray: 当前灰度值

        Returns:
            阈值
        """
        # 如果未启用自适应阈值，返回固定阈值
        if not self._config.adaptive_threshold_enabled:
            return self._config.peak_detection_threshold

        # 如果阈值保护激活，不更新背景
        if self._threshold_protection.is_active:
            return max(self._config.threshold_minimum, self._bg_mean * (1 + self._config.threshold_over_mean_ratio))

        # 更新背景均值
        self._bg_count += 1
        self._bg_mean += (current_gray - self._bg_mean) / self._bg_count

        # 计算自适应阈值
        adaptive_threshold = self._bg_mean * (1 + self._config.threshold_over_mean_ratio)

        # 确保不低于最小阈值
        return max(self._config.threshold_minimum, adaptive_threshold)

    def _reset_video_state(self) -> None:
        """重置视频处理状态"""
        self._frame_index = 0
        self._bg_count = 0
        self._bg_mean = 0.0
        self._roi1_peak_counter.clear()
        self._roi_capture.reset_buffers()
        self._threshold_protection.reset()
        self._green_line.reset()

    def close(self) -> None:
        """关闭资源"""
        self._roi_capture.close()
        self._analysis_cache.close()
        self._statistics.export_final_csv()
