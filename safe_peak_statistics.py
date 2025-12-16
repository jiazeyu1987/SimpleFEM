#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleFEM 安全波峰统计模块
实现波峰去重、差值分析和Excel数据收集功能
根据 task/info1.txt 要求实现
"""

import csv
import os
import sys
import json
import threading
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import logging


def _get_base_dir() -> str:
    """获取基础目录，支持源码和打包模式"""
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _get_base_dir()


class SafePeakStatistics:
    """安全的波峰统计管理类"""

    def __init__(self):
        self.lock = threading.Lock()
        self.recent_peaks: List[Dict[str, Any]] = []
        self.max_recent_peaks = 5  # 去重检查窗口
        self.stats_data: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        # 文件路径 - 保存到export文件夹
        export_dir = os.path.join(BASE_DIR, "export")
        os.makedirs(export_dir, exist_ok=True)

        self.csv_filename = f"peak_statistics_{self.session_id}.csv"
        self.csv_path = os.path.join(export_dir, self.csv_filename)
        self.final_export_path = os.path.join(export_dir, f"peak_statistics_final_{self.session_id}.csv")

        # 配置参数（根据task要求）
        self.duplicate_check_window = 5  # 检查最近5个波峰
        self.height_tolerance = 0.1      # 高度容差≤0.1
        self.update_count = 0

        # 新增：连续同色波峰去重配置
        self.consecutive_frame_window = 10  # 连续检查窗口（10帧）
        self.consecutive_deduplication_enabled = True  # 是否启用连续同色去重
        self.consecutive_peak_groups: Dict[str, List[Dict[str, Any]]] = {}  # 跟踪连续同色波峰组

        # 初始化CSV文件（程序开始时记录）
        self._initialize_csv_file()
        self._add_log(f"SafePeakStatistics初始化完成，会话ID: {self.session_id}")
        self._add_log(f"连续同色去重配置: 窗口={self.consecutive_frame_window}帧, 启用={self.consecutive_deduplication_enabled}")

    def _initialize_csv_file(self):
        """初始化CSV文件，写入表头（程序开始时记录）"""
        try:
            # 确保目录存在
            os.makedirs(BASE_DIR, exist_ok=True)

            file_exists = os.path.exists(self.csv_path)

            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                # 包含所有必要字段：peak_type, frame_index, 前后X帧平均值, 波峰最大值
                fieldnames = [
                    'peak_type',
                    'frame_index',
                    'pre_peak_avg',
                    'post_peak_avg',
                    'frame_diff',
                    'difference_threshold_used',
                    'threshold_used',
                    'bg_mean',
                    'peak_max_value',
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()
                    self._add_log(f"CSV文件初始化完成: {self.csv_path}")
                    self._add_log(f"包含字段: {', '.join(fieldnames)}")
                else:
                    self._add_log(f"CSV文件已存在，继续追加数据: {self.csv_path}")

        except Exception as e:
            self._add_log(f"初始化CSV文件失败: {e}", level="ERROR")

    def add_peaks_from_daemon(self,
                            frame_index: int,
                            green_peaks: List[Tuple[int, int]],
                            red_peaks: List[Tuple[int, int]],
                            curve: List[float],
                            intersection: Optional[Tuple[int, int]] = None,
                            roi2_info: Optional[Dict[str, int]] = None,
                            gray_value: Optional[float] = None,
                            difference_threshold: float = 1.1,
                            pre_post_avg_frames: int = 5,
                            threshold_used: Optional[float] = None,
                            bg_mean: Optional[float] = None):
        """
        从守护进程添加波峰数据

        Args:
            frame_index: 帧索引
            green_peaks: 绿色波峰列表 [(start_frame, end_frame), ...]
            red_peaks: 红色波峰列表 [(start_frame, end_frame), ...]
            curve: 当前灰度曲线数据
            intersection: ROI1中的绿线交点坐标 (x, y)
            roi2_info: ROI2区域信息 {'x1':, 'y1':, 'x2':, 'y2':}
            gray_value: ROI2的平均灰度值
            difference_threshold: 用于分类的差值阈值
            pre_post_avg_frames: 前后平均灰度的帧数（默认5）
            threshold_used: 当次检测使用的阈值（固定阈值或自适应阈值）
            bg_mean: 当次检测时的背景均值（用于自适应阈值基线）
        """
        try:
            timestamp = datetime.now()

            with self.lock:
                # 添加绿色波峰
                for i, (start, end) in enumerate(green_peaks):
                    peak_data = self._create_peak_data(
                        timestamp, frame_index, "green", start, end,
                        curve, intersection, roi2_info, gray_value,
                        difference_threshold, pre_post_avg_frames,
                        threshold_used, bg_mean
                    )

                    # 第一层去重：基于前后帧平均值
                    if self._is_duplicate_peak(peak_data):
                        continue

                    # 第二层去重：连续同色波峰去重
                    if self._is_consecutive_duplicate(peak_data, curve, start, end):
                        continue

                    # 第三层去重：排除前后帧平均值为0的波峰
                    if self._is_invalid_peak_data(peak_data):
                        continue

                    # 三层去重都通过，才记录波峰
                    self._add_peak_to_memory(peak_data)
                    self._write_peak_to_csv(peak_data)
                    self._add_log(f"添加绿色波峰: [{start},{end}], 最大值: {peak_data['peak_max_value']:.1f}")

                # 添加红色波峰
                for i, (start, end) in enumerate(red_peaks):
                    peak_data = self._create_peak_data(
                        timestamp, frame_index, "red", start, end,
                        curve, intersection, roi2_info, gray_value,
                        difference_threshold, pre_post_avg_frames,
                        threshold_used, bg_mean
                    )

                    # 第一层去重：基于前后帧平均值
                    if self._is_duplicate_peak(peak_data):
                        continue

                    # 第二层去重：连续同色波峰去重
                    if self._is_consecutive_duplicate(peak_data, curve, start, end):
                        continue

                    # 第三层去重：排除前后帧平均值为0的波峰
                    if self._is_invalid_peak_data(peak_data):
                        continue

                    # 三层去重都通过，才记录波峰
                    self._add_peak_to_memory(peak_data)
                    self._write_peak_to_csv(peak_data)
                    self._add_log(f"添加红色波峰: [{start},{end}], 最大值: {peak_data['peak_max_value']:.1f}")

                self.update_count += 1

        except Exception as e:
            self._add_log(f"添加波峰数据失败: {e}", level="ERROR")

    def _create_peak_data(self,
                         timestamp: datetime,
                         frame_index: int,
                         peak_type: str,
                         start_frame: int,
                         end_frame: int,
                         curve: List[float],
                         intersection: Optional[Tuple[int, int]],
                         roi2_info: Optional[Dict[str, int]],
                         gray_value: Optional[float],
                         difference_threshold: float,
                         pre_post_avg_frames: int = 5,
                         threshold_used: Optional[float] = None,
                         bg_mean: Optional[float] = None) -> Dict[str, Any]:
        """创建简化的波峰数据结构（只保留必要字段）"""

        # 计算前X帧平均值（波峰开始前5帧）
        pre_frames = int(pre_post_avg_frames) if int(pre_post_avg_frames) > 0 else 5
        pre_start = max(0, start_frame - pre_frames)
        pre_end = start_frame - 1
        pre_avg = 0.0

        if pre_start <= pre_end and pre_end < len(curve):
            pre_values = curve[pre_start:pre_end + 1]
            pre_avg = sum(pre_values) / len(pre_values) if pre_values else 0.0

        # 计算后X帧平均值（波峰结束后5帧）
        post_frames = int(pre_post_avg_frames) if int(pre_post_avg_frames) > 0 else 5
        post_start = end_frame + 1
        post_end = min(len(curve) - 1, end_frame + post_frames)
        post_avg = 0.0

        if 0 <= post_start <= post_end:
            post_values = curve[post_start:post_end + 1]
            post_avg = sum(post_values) / len(post_values) if post_values else 0.0

        # 计算波峰区域最大值（用于连续同色去重）
        frame_diff = post_avg - pre_avg
        peak_max_value = self._get_peak_max_value(curve, start_frame, end_frame)

        return {
            'peak_type': peak_type,
            'frame_index': frame_index,
            'pre_peak_avg': round(pre_avg, 2),
            'post_peak_avg': round(post_avg, 2),
            'frame_diff': round(frame_diff, 3),
            'difference_threshold_used': round(float(difference_threshold), 3),
            'threshold_used': round(float(threshold_used), 3) if threshold_used is not None else 0.0,
            'bg_mean': round(float(bg_mean), 3) if bg_mean is not None else 0.0,
            'peak_max_value': round(peak_max_value, 2)  # 新增：波峰最大值
        }

    def _calculate_classification_reason(self, frame_diff: float, threshold: float, peak_type: str) -> str:
        """计算波峰分类原因（为什么是红色/绿色）"""
        if peak_type == "green":
            if frame_diff > threshold:
                return f"稳定波峰: 帧差值{frame_diff:.2f} > 阈值{threshold:.2f}"
            else:
                return f"稳定波峰: 帧差值{frame_diff:.2f} >= 阈值{threshold:.2f}"
        else:  # red
            return f"不稳定波峰: 帧差值{frame_diff:.2f} < 阈值{threshold:.2f}"

    def _calculate_peak_stability(self, peak_curve: List[float]) -> float:
        """计算波峰稳定性（0-1，越接近1越稳定）"""
        try:
            if len(peak_curve) < 2:
                return 0.0

            # 计算标准差
            mean_val = sum(peak_curve) / len(peak_curve)
            variance = sum((x - mean_val) ** 2 for x in peak_curve) / len(peak_curve)
            std_dev = variance ** 0.5

            # 归一化稳定性评分
            max_val = max(peak_curve) if peak_curve else 1
            stability_score = max(0, 1 - (std_dev / max_val))

            return min(1.0, stability_score)
        except Exception:
            return 0.0

    def _calculate_quality_score(self, max_val: float, min_val: float, duration: int, frame_diff: float) -> float:
        """计算波峰质量评分（0-100）"""
        try:
            # 基础评分：波峰高度
            height_score = max_val - min_val

            # 持续时间评分
            duration_score = min(duration / 10.0, 1.0) * 20  # 最高20分

            # 稳定性评分（帧差值越小越稳定）
            stability_score = max(0, 20 - frame_diff)  # 最高20分

            # 总分（最高100分）
            total_score = height_score + duration_score + stability_score

            return max(0, min(100, total_score))
        except Exception:
            return 0.0

    def _is_duplicate_peak(self, peak_data: Dict[str, Any]) -> bool:
        """检查是否为重复波峰（基于前后帧平均值去重）"""
        try:
            current_pre_avg = peak_data['pre_peak_avg']
            current_post_avg = peak_data['post_peak_avg']

            # 检查最近的5个波峰（task要求的窗口大小）
            for recent_peak in self.recent_peaks[-self.duplicate_check_window:]:
                recent_pre_avg = recent_peak.get('pre_peak_avg', 0)
                recent_post_avg = recent_peak.get('post_peak_avg', 0)

                # 前后帧平均值都接近（容差0.5）则视为重复
                if (abs(recent_pre_avg - current_pre_avg) <= 0.5 and
                    abs(recent_post_avg - current_post_avg) <= 0.5):
                    return True
            return False
        except Exception as e:
            self._add_log(f"去重检查失败: {e}", level="ERROR")
            return False

    def _is_consecutive_duplicate(self, peak_data: Dict[str, Any], curve: List[float], start_frame: int, end_frame: int) -> bool:
        """检查是否为连续同色重复波峰（10帧内的同色波峰，只保留最高的）"""
        try:
            if not self.consecutive_deduplication_enabled:
                return False

            current_type = peak_data['peak_type']
            current_frame_index = peak_data['frame_index']
            current_max_value = self._get_peak_max_value(curve, start_frame, end_frame)

            # 检查最近的波峰中是否有同色连续波峰
            for recent_peak in self.recent_peaks[-20:]:  # 检查更多波峰以覆盖10帧窗口
                if recent_peak['peak_type'] != current_type:
                    continue  # 不同颜色，跳过

                recent_frame_index = recent_peak['frame_index']
                recent_max_value = recent_peak.get('peak_max_value', 0)

                # 检查是否在10帧窗口内
                if abs(current_frame_index - recent_frame_index) <= self.consecutive_frame_window:
                    # 如果当前波峰更低，则是重复波峰
                    if current_max_value <= recent_max_value:
                        self._add_log(f"连续同色去重: {current_type}波峰帧{current_frame_index}(高度{current_max_value:.1f}) 低于 帧{recent_frame_index}(高度{recent_max_value:.1f})")
                        return True
                    else:
                        # 如果当前波峰更高，移除之前的较低波峰
                        self._remove_lower_consecutive_peak(recent_peak)
                        self._add_log(f"连续同色去重: {current_type}波峰帧{current_frame_index}(高度{current_max_value:.1f}) 高于 帧{recent_frame_index}(高度{recent_max_value:.1f})，移除较低的")
                        return False

            return False
        except Exception as e:
            self._add_log(f"连续同色去重检查失败: {e}", level="ERROR")
            return False

    def _is_invalid_peak_data(self, peak_data: Dict[str, Any]) -> bool:
        """检查波峰数据是否无效（前后帧平均值为0）"""
        try:
            pre_avg = peak_data.get('pre_peak_avg', 0)
            post_avg = peak_data.get('post_peak_avg', 0)

            # 如果前后帧平均值都是0，则认为数据无效
            if pre_avg == 0 or post_avg == 0:
                peak_type = peak_data.get('peak_type', 'unknown')
                frame_index = peak_data.get('frame_index', 0)
                self._add_log(f"无效波峰数据过滤: {peak_type}波峰帧{frame_index}, 前帧平均={pre_avg:.2f}, 后帧平均={post_avg:.2f}")
                return True

            return False
        except Exception as e:
            self._add_log(f"检查无效波峰数据失败: {e}", level="ERROR")
            return False

    def _get_peak_max_value(self, curve: List[float], start_frame: int, end_frame: int) -> float:
        """计算波峰区域的最大灰度值"""
        try:
            if start_frame < 0 or end_frame >= len(curve) or start_frame > end_frame:
                return 0.0

            peak_region = curve[start_frame:end_frame + 1]
            return max(peak_region) if peak_region else 0.0
        except Exception as e:
            self._add_log(f"计算波峰最大值失败: {e}", level="ERROR")
            return 0.0

    def _remove_lower_consecutive_peak(self, peak_to_remove: Dict[str, Any]):
        """从统计记录中移除较低的连续同色波峰"""
        try:
            # 从内存缓存中移除
            if peak_to_remove in self.recent_peaks:
                self.recent_peaks.remove(peak_to_remove)

            if peak_to_remove in self.stats_data:
                self.stats_data.remove(peak_to_remove)

            # 从CSV文件中移除（重新写入整个文件）
            self._rewrite_csv_without_peak(peak_to_remove)

            self._add_log(f"已移除较低的同色波峰: {peak_to_remove['peak_type']} 帧{peak_to_remove['frame_index']}")
        except Exception as e:
            self._add_log(f"移除较低波峰失败: {e}", level="ERROR")

    def _rewrite_csv_without_peak(self, peak_to_remove: Dict[str, Any]):
        """重新写入CSV文件，移除指定的波峰"""
        try:
            if not os.path.exists(self.csv_path):
                return

            # 读取现有数据
            rows_to_keep = []
            with open(self.csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames

                for row in reader:
                    # 检查是否为要移除的波峰
                    if (row['peak_type'] == peak_to_remove['peak_type'] and
                        int(row['frame_index']) == peak_to_remove['frame_index'] and
                        float(row['pre_peak_avg']) == peak_to_remove['pre_peak_avg'] and
                        float(row['post_peak_avg']) == peak_to_remove['post_peak_avg']):
                        continue  # 跳过要移除的行
                    rows_to_keep.append(row)

            # 重新写入文件
            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_keep)

            self._add_log(f"CSV文件已重写，移除了1个较低的同色波峰，保留{len(rows_to_keep)}个波峰")
        except Exception as e:
            self._add_log(f"重写CSV文件失败: {e}", level="ERROR")

    def _add_peak_to_memory(self, peak_data: Dict[str, Any]):
        """添加波峰到内存缓存"""
        self.recent_peaks.append(peak_data)
        self.stats_data.append(peak_data)

        # 保持内存缓存大小（最近5个波峰用于去重）
        if len(self.recent_peaks) > self.max_recent_peaks * 2:
            self.recent_peaks = self.recent_peaks[-self.max_recent_peaks:]

    def _write_peak_to_csv(self, peak_data: Dict[str, Any]):
        """写入单个波峰到CSV文件"""
        try:
            # 简化的字段列表（只保存CSV中需要的字段）
            fieldnames = [
                'peak_type',
                'frame_index',
                'pre_peak_avg',
                'post_peak_avg',
                'frame_diff',
                'difference_threshold_used',
                'threshold_used',
                'bg_mean',
                'peak_max_value',
            ]

            # 过滤数据，只包含CSV需要的字段
            csv_data = {key: peak_data[key] for key in fieldnames if key in peak_data}

            # 原子性写入：先写临时文件，再重命名
            temp_file = self.csv_path + '.tmp'

            # 如果原文件存在，复制内容
            if os.path.exists(self.csv_path):
                shutil.copy2(self.csv_path, temp_file)

            # 追加新数据
            with open(temp_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(csv_data)

            # 原子性重命名
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
            os.rename(temp_file, self.csv_path)

        except Exception as e:
            self._add_log(f"写入CSV失败: {e}", level="ERROR")

    
    def export_final_csv(self) -> Optional[str]:
        """程序结束时导出最终CSV文件"""
        try:
            self._add_log("开始导出最终CSV文件...")

            if not os.path.exists(self.csv_path):
                self._add_log("没有数据文件可导出")
                return None

            # 创建最终导出文件
            shutil.copy2(self.csv_path, self.final_export_path)

            # 添加导出时间戳
            with open(self.final_export_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    f"# EXPORT_SUMMARY",
                    f"export_time,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"total_peaks,{len(self.stats_data)}",
                    f"session_duration,{str(datetime.now() - self.start_time).split('.')[0]}",
                    f"session_id,{self.session_id}"
                ])

            self._add_log(f"最终CSV文件已导出至: {self.final_export_path}")
            return os.path.abspath(self.final_export_path)

        except Exception as e:
            self._add_log(f"导出最终CSV文件失败: {e}", level="ERROR")
            return None

    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        try:
            with self.lock:
                total_peaks = len(self.stats_data)
                green_peaks = len([p for p in self.stats_data if p['peak_type'] == 'green'])
                red_peaks = len([p for p in self.stats_data if p['peak_type'] == 'red'])

                avg_duration = 0
                avg_max_value = 0
                avg_frame_diff = 0
                if self.stats_data:
                    avg_duration = sum(p['duration'] for p in self.stats_data) / total_peaks
                    avg_max_value = sum(p['max_value'] for p in self.stats_data) / total_peaks
                    avg_frame_diff = sum(p['frame_diff'] for p in self.stats_data) / total_peaks

                return {
                    'total_peaks': total_peaks,
                    'green_peaks': green_peaks,
                    'red_peaks': red_peaks,
                    'avg_duration': round(avg_duration, 2),
                    'avg_max_value': round(avg_max_value, 2),
                    'avg_frame_diff': round(avg_frame_diff, 3),
                    'session_id': self.session_id,
                    'session_duration': str(datetime.now() - self.start_time).split('.')[0],
                    'csv_file_path': self.csv_path,
                    'final_export_path': self.final_export_path,
                    'csv_exists': os.path.exists(self.csv_path),
                    'csv_size_mb': round(os.path.getsize(self.csv_path) / (1024*1024), 2) if os.path.exists(self.csv_path) else 0
                }
        except Exception as e:
            self._add_log(f"获取统计摘要失败: {e}", level="ERROR")
            return {}

    def save_csv_file(self) -> Optional[str]:
        """保存CSV文件并返回路径（用于UI调用）"""
        try:
            if os.path.exists(self.csv_path):
                return os.path.abspath(self.csv_path)
            else:
                return None
        except Exception as e:
            self._add_log(f"保存CSV文件失败: {e}", level="ERROR")
            return None

    def _add_log(self, message: str, level: str = "INFO"):
        """添加日志记录"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] {level}: SafePeakStatistics - {message}"
            print(log_message)  # 输出到控制台
        except Exception:
            pass  # 日志记录失败不应该影响主要功能


# 全局实例
safe_statistics = SafePeakStatistics()
