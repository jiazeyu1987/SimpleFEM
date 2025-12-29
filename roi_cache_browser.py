#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Analysis Cache Browser
独立的 GUI 应用程序，用于浏览和对比 roi_analysis_cache_*.jsonl 文件

Usage:
    python roi_cache_browser.py

Features:
    - 分页卡片浏览：一次显示一帧的完整数据
    - 双帧对比：左右分屏并排显示，差异高亮
    - 波形趋势图：显示 roi2_gray 随帧变化，标记峰值
    - 智能提示：鼠标悬停显示字段说明
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 字段说明文档
# =============================================================================

FIELD_DESCRIPTIONS = {
    # 基本信息
    'frame_index': '帧的序列索引号，从1开始计数',
    'ts_wall': 'Wall 时间戳（Unix 时间戳，浮点数）',
    'ts_local': '本地时间戳，ISO 8601 格式字符串',
    'video_seconds': '视频播放时间（秒）',
    'screen_size': '屏幕/视频分辨率 [宽度, 高度]',

    # ROI 区域信息
    'roi1': 'ROI1 区域坐标 {x1, y1, x2, y2}，大区域用于绿线检测',
    'roi1.x1': 'ROI1 左上角 X 坐标',
    'roi1.y1': 'ROI1 左上角 Y 坐标',
    'roi1.x2': 'ROI1 右下角 X 坐标',
    'roi1.y2': 'ROI1 右下角 Y 坐标',

    'intersection': '绿色线交点信息',
    'intersection.current': '当前帧检测到的交点坐标 [x, y]',
    'intersection.used': '实际使用的交点坐标（经过滤波处理后）',

    'roi2_region': 'ROI2 区域坐标 [x1, y1, x2, y2]，小区域用于灰度分析',
    'roi2_gray': 'ROI2 区域的平均灰度值 (0-255)，用于波峰检测',

    'roi3': 'ROI3 数据对象',
    'roi3.g1': 'ROI3 区域 g1 值（ROI3 上部分的平均灰度）',
    'roi3.g2': 'ROI3 区域 g2 值（ROI3 下部分的平均灰度）',
    'roi3.gray': 'ROI3 区域的整体平均灰度值',

    # 缓冲区和阈值
    'buffer': '环形缓冲区状态',
    'buffer.len': '缓冲区当前长度（已存储的帧数）',
    'buffer.start_frame_index': '缓冲区起始帧索引',
    'buffer.maxlen': '缓冲区最大长度（固定为100）',

    'threshold': '阈值相关信息',
    'threshold.fixed': '固定阈值配置值（配置文件中设定的基础阈值）',
    'threshold.minimum': '阈值下限（最小阈值，确保检测灵敏度）',
    'threshold.used': '实际使用的检测阈值（固定阈值或自适应阈值）',
    'threshold.adaptive_enabled': '是否启用自适应阈值（根据背景均值动态调整）',
    'threshold.adaptive_window_frames': '自适应阈值计算的窗口帧数',
    'threshold.bg_mean': '背景均值（用于计算自适应阈值的背景灰度平均值）',
    'threshold.bg_count': '背景样本计数（用于计算背景均值的样本数量）',
    'threshold.protection_active': '阈值保护是否激活（防止波峰污染背景计算）',
    'threshold.consecutive_below_threshold': '连续低于阈值的帧数',

    'detect_params': '波峰检测参数',
    'detect_params.margin_frames': '峰间最小间隔帧数（两个波峰间隔小于此值只保留峰值更高的）',
    'detect_params.silence_frames': '干净区间长度（波峰前后需要低于阈值的帧数）',
    'detect_params.difference_threshold': '绿/红判定阈值 (post_peak_avg - pre_peak_avg >= 此值判为绿色)',
    'detect_params.pre_post_avg_frames': '计算前后平均值的窗口帧数',
    'detect_params.min_region_length': '最小波峰宽度（帧数）',

    'detection': '检测模式信息',
    'detection.mode': '当前检测模式（如 hybrid_roi1_peaks_roi2_color）',
    'detection.hybrid_enabled': '是否启用混合检测（ROI1检测波峰+ROI2判定颜色）',
    'detection.roi1_enabled': '是否启用 ROI1 检测',

    # 波峰检测
    'peaks': '波峰检测结果对象',
    'peaks.green_raw': '原始检测到的绿色波峰列表（未去重）',
    'peaks.red_raw': '原始检测到的红色波峰列表（未去重）',
    'peaks.green': '最终绿色波峰列表（经过去重处理的绿色波峰）',
    'peaks.red': '最终红色波峰列表（经过去重处理的红色波峰）',

    'stats_write': '写入CSV的统计数据列表（检测到波峰时非空）',
}


# =============================================================================
# ToolTip - 工具提示类
# =============================================================================

class ToolTip:
    """
    创建鼠标悬停提示控件

    使用方法：
        label = ttk.Label(parent, text="字段名")
        ToolTip(label, "这是字段的说明文字")
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        """
        初始化 ToolTip

        Args:
            widget: 要绑定提示的控件
            text: 提示文本内容
            delay: 延迟显示时间（毫秒），默认500ms
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.widget.bind('<Enter>', self.schedule_show)
        self.widget.bind('<Leave>', self.hide)
        self.widget.bind('<ButtonPress>', self.hide)

    def schedule_show(self, event=None):
        """延迟显示提示"""
        self.widget.after(self.delay, self.show)

    def show(self, event=None):
        """显示提示窗口"""
        if self.tip_window or not self.text:
            return

        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("tahoma", "8", "normal"), padx=2, pady=1)
        label.pack(ipadx=1)

    def hide(self, event=None):
        """隐藏提示窗口"""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# =============================================================================
# CacheLoader - 数据加载器
# =============================================================================

class CacheLoader:
    """加载和解析 roi_analysis_cache_*.jsonl 文件"""

    def __init__(self, file_path: str):
        """
        初始化加载器

        Args:
            file_path: JSONL 文件路径
        """
        self.file_path = Path(file_path)
        self.meta: Dict[str, Any] = {}
        self.frames: List[Dict[str, Any]] = []
        self.session_end: Dict[str, Any] = {}
        self.frame_index_map: Dict[int, Dict[str, Any]] = {}
        self.peaks_map: Dict[str, List[Tuple[int, Dict]]] = {'green': [], 'red': []}

    def load(self) -> bool:
        """
        加载 JSONL 文件

        Returns:
            加载成功返回 True，失败返回 False
        """
        if not self.file_path.exists():
            return False

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        record_type = data.get('type')

                        if record_type == 'meta':
                            self.meta = data
                        elif record_type == 'frame':
                            self.frames.append(data)
                            frame_idx = data.get('frame_index', -1)
                            self.frame_index_map[frame_idx] = data
                        elif record_type == 'session_end':
                            self.session_end = data
                    except json.JSONDecodeError:
                        continue

            # 构建峰值索引
            self._build_peaks_index()
            return True

        except Exception as e:
            print(f"加载文件失败: {e}")
            return False

    def _build_peaks_index(self):
        """构建峰值索引，用于快速查找"""
        self.peaks_map = {'green': [], 'red': []}

        for frame in self.frames:
            frame_idx = frame.get('frame_index', -1)
            peaks = frame.get('peaks', {})

            for color in ['green', 'red']:
                color_peaks = peaks.get(color, [])
                if color_peaks:
                    self.peaks_map[color].append((frame_idx, color_peaks))

    def get_frame(self, frame_index: int) -> Optional[Dict[str, Any]]:
        """获取指定索引的帧"""
        return self.frame_index_map.get(frame_index)

    def get_total_frames(self) -> int:
        """获取总帧数"""
        return len(self.frames)

    def find_next_peak(self, current_frame: int, color: str = 'green') -> Optional[int]:
        """查找下一个指定颜色的峰值帧"""
        peaks = self.peaks_map.get(color, [])
        for frame_idx, _ in peaks:
            if frame_idx > current_frame:
                return frame_idx
        return None

    def find_prev_peak(self, current_frame: int, color: str = 'green') -> Optional[int]:
        """查找上一个指定颜色的峰值帧"""
        peaks = self.peaks_map.get(color, [])
        for frame_idx, _ in reversed(peaks):
            if frame_idx < current_frame:
                return frame_idx
        return None


# =============================================================================
# SingleFrameView - 单帧浏览视图
# =============================================================================

class SingleFrameView:
    """单帧浏览视图"""

    def __init__(self, parent: tk.Widget, on_frame_change=None):
        """
        初始化单帧视图

        Args:
            parent: 父容器
            on_frame_change: 帧变化回调函数
        """
        self.parent = parent
        self.on_frame_change = on_frame_change
        self.current_frame = None
        self.value_labels = {}  # field_name -> label widget

        self._create_ui()

    def _create_ui(self):
        """创建用户界面"""
        # 滚动容器
        self.canvas = tk.Canvas(self.parent)
        self.scrollbar = ttk.Scrollbar(self.parent, orient="vertical",
                                       command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 创建字段组
        self._create_basic_info_group()
        self._create_roi_group()
        self._create_detection_group()
        self._create_peaks_group()

    def _create_field_label(self, parent: tk.Widget, field_name: str,
                           display_name: str) -> ttk.Label:
        """创建带 tooltip 的字段标签"""
        label = ttk.Label(parent, text=display_name + ':',
                         font=('Arial', 9))
        tooltip_text = FIELD_DESCRIPTIONS.get(field_name, '暂无说明')
        ToolTip(label, tooltip_text)
        return label

    def _create_value_label(self, parent: tk.Widget, field_name: str) -> ttk.Label:
        """创建值标签并保存引用"""
        label = ttk.Label(parent, text='-',
                         font=('Consolas', 9),
                         foreground='#333333')
        self.value_labels[field_name] = label
        return label

    def _create_basic_info_group(self):
        """创建基本信息组"""
        group = ttk.LabelFrame(self.scrollable_frame, text="基本信息",
                              padding=10)
        group.pack(fill='x', padx=5, pady=5)

        fields = [
            ('frame_index', '帧索引'),
            ('ts_local', '本地时间'),
            ('video_seconds', '视频时间(秒)'),
            ('screen_size', '屏幕尺寸'),
        ]

        for i, (field, display) in enumerate(fields):
            label = self._create_field_label(group, field, display)
            label.grid(row=i, column=0, sticky='w', padx=5, pady=3)

            value_label = self._create_value_label(group, field)
            value_label.grid(row=i, column=1, sticky='w', padx=5, pady=3)

    def _create_roi_group(self):
        """创建 ROI 区域组"""
        group = ttk.LabelFrame(self.scrollable_frame, text="ROI 区域信息",
                              padding=10)
        group.pack(fill='x', padx=5, pady=5)

        fields = [
            ('roi1', 'ROI1 坐标'),
            ('intersection.current', '绿线交点(当前)'),
            ('intersection.used', '绿线交点(使用)'),
            ('roi2_region', 'ROI2 区域'),
            ('roi2_gray', 'ROI2 灰度'),
            ('roi3.g1', 'ROI3 g1值'),
            ('roi3.g2', 'ROI3 g2值'),
            ('roi3.gray', 'ROI3 平均灰度'),
        ]

        for i, (field, display) in enumerate(fields):
            label = self._create_field_label(group, field, display)
            label.grid(row=i, column=0, sticky='w', padx=5, pady=3)

            value_label = self._create_value_label(group, field)
            value_label.grid(row=i, column=1, sticky='w', padx=5, pady=3)

    def _create_detection_group(self):
        """创建检测参数组"""
        group = ttk.LabelFrame(self.scrollable_frame, text="检测参数",
                              padding=10)
        group.pack(fill='x', padx=5, pady=5)

        fields = [
            ('buffer.len', '缓冲区长度'),
            ('buffer.start_frame_index', '缓冲区起始帧'),
            ('threshold.fixed', '固定阈值'),
            ('threshold.minimum', '最小阈值'),
            ('threshold.used', '使用阈值'),
            ('threshold.adaptive_enabled', '自适应阈值'),
            ('threshold.protection_active', '阈值保护'),
            ('detect_params.difference_threshold', '绿红判定阈值'),
            ('detect_params.margin_frames', '峰间最小间隔'),
            ('detect_params.silence_frames', '干净区间长度'),
        ]

        for i, (field, display) in enumerate(fields):
            label = self._create_field_label(group, field, display)
            label.grid(row=i, column=0, sticky='w', padx=5, pady=3)

            value_label = self._create_value_label(group, field)
            value_label.grid(row=i, column=1, sticky='w', padx=5, pady=3)

    def _create_peaks_group(self):
        """创建波峰检测组"""
        group = ttk.LabelFrame(self.scrollable_frame, text="波峰检测",
                              padding=10)
        group.pack(fill='x', padx=5, pady=5)

        # Green peaks
        ttk.Label(group, text='绿色波峰:', font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=3)
        green_label = self._create_value_label(group, 'peaks.green')
        green_label.grid(row=0, column=1, sticky='w', padx=5, pady=3)

        # Red peaks
        ttk.Label(group, text='红色波峰:', font=('Arial', 9, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=3)
        red_label = self._create_value_label(group, 'peaks.red')
        red_label.grid(row=1, column=1, sticky='w', padx=5, pady=3)

    def update_frame(self, frame_data: Dict[str, Any]):
        """更新显示帧数据"""
        self.current_frame = frame_data

        for field, label in self.value_labels.items():
            value = self._get_nested_value(frame_data, field)
            formatted_value = self._format_value(value)

            # 为 roi3.g1 和 roi3.g2 添加百分号
            if field in ['roi3.g1', 'roi3.g2'] and value is not None:
                formatted_value = f'{formatted_value}%'

            label.config(text=formatted_value)

    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """获取嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _format_value(self, value: Any) -> str:
        """格式化显示值"""
        if value is None:
            return '-'
        elif isinstance(value, bool):
            return '是' if value else '否'
        elif isinstance(value, (list, tuple)):
            if len(value) <= 4:
                return str(list(value))
            else:
                return f'[{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] (共{len(value)}项)'
        elif isinstance(value, dict):
            items = [f'{k}={v}' for k, v in list(value.items())[:3]]
            return '{' + ', '.join(items) + ('...' if len(value) > 3 else '') + '}'
        elif isinstance(value, float):
            return f'{value:.2f}'
        else:
            return str(value)


# =============================================================================
# CompareView - 双帧对比视图
# =============================================================================

class CompareView:
    """双帧对比视图（带差异高亮）"""

    def __init__(self, parent: tk.Widget, get_loader_func):
        """
        初始化对比视图

        Args:
            parent: 父容器
            get_loader_func: 获取数据加载器的函数
        """
        self.parent = parent
        self.get_loader_func = get_loader_func
        self.frame_a = None
        self.frame_b = None
        self.view_a = None
        self.view_b = None
        self.diffs = {}  # 存储差异计算结果

        self._create_ui()

    def _create_ui(self):
        """创建用户界面"""
        # 顶部选择器
        selector_frame = ttk.Frame(self.parent, padding=10)
        selector_frame.pack(fill='x')

        # 帧 A 选择
        ttk.Label(selector_frame, text="帧 A:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.frame_a_var = tk.IntVar(value=1)
        entry_a = ttk.Entry(selector_frame, textvariable=self.frame_a_var, width=10)
        entry_a.pack(side='left', padx=5)
        entry_a.bind('<Return>', lambda e: self.do_compare())

        # 帧 B 选择
        ttk.Label(selector_frame, text="帧 B:", font=('Arial', 10, 'bold')).pack(side='left', padx=(20, 5))
        self.frame_b_var = tk.IntVar(value=1)
        entry_b = ttk.Entry(selector_frame, textvariable=self.frame_b_var, width=10)
        entry_b.pack(side='left', padx=5)
        entry_b.bind('<Return>', lambda e: self.do_compare())

        # 对比按钮
        ttk.Button(selector_frame, text="开始对比",
                  command=self.do_compare).pack(side='left', padx=20)

        # 快速预设
        ttk.Separator(selector_frame, orient='vertical').pack(side='left', padx=10, fill='y')
        ttk.Label(selector_frame, text="快速预设:").pack(side='left', padx=5)
        ttk.Button(selector_frame, text="对比相邻帧",
                  command=self.compare_adjacent).pack(side='left', padx=2)
        ttk.Button(selector_frame, text="对比波峰帧",
                  command=self.compare_peaks).pack(side='left', padx=2)

        # 差异统计
        self.diff_stats_var = tk.StringVar(value="")
        ttk.Label(selector_frame, textvariable=self.diff_stats_var,
                 foreground='blue', font=('Arial', 9)).pack(side='left', padx=20)

        # 左右分屏
        content_frame = ttk.Frame(self.parent)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 左侧面板
        left_container = ttk.LabelFrame(content_frame, text="帧 A", padding=5)
        left_container.pack(side='left', fill='both', expand=True, padx=5)
        self.view_a = CompareSingleFrameView(left_container, 'A')

        # 右侧面板
        right_container = ttk.LabelFrame(content_frame, text="帧 B", padding=5)
        right_container.pack(side='right', fill='both', expand=True, padx=5)
        self.view_b = CompareSingleFrameView(right_container, 'B')

    def do_compare(self):
        """执行对比并高亮差异"""
        loader = self.get_loader_func()
        if not loader:
            return

        frame_a_idx = self.frame_a_var.get()
        frame_b_idx = self.frame_b_var.get()

        self.frame_a = loader.get_frame(frame_a_idx)
        self.frame_b = loader.get_frame(frame_b_idx)

        if not self.frame_a or not self.frame_b:
            return

        # 计算差异
        self.diffs = self._calculate_differences(self.frame_a, self.frame_b)

        # 更新显示（带差异高亮）
        self.view_a.update_frame(self.frame_a, self.diffs, is_side_a=True)
        self.view_b.update_frame(self.frame_b, self.diffs, is_side_a=False)

        # 更新统计
        diff_count = len([d for d in self.diffs.values() if d.get('has_diff', False)])
        self.diff_stats_var.set(f"发现 {diff_count} 处差异")

    def compare_adjacent(self):
        """对比相邻帧"""
        loader = self.get_loader_func()
        if loader:
            idx = self.frame_a_var.get()
            if idx < loader.get_total_frames():
                self.frame_a_var.set(idx)
                self.frame_b_var.set(idx + 1)
                self.do_compare()

    def compare_peaks(self):
        """对比波峰帧"""
        loader = self.get_loader_func()
        if loader:
            # 查找最近的两个波峰
            peaks = loader.peaks_map.get('green', [])
            if len(peaks) >= 2:
                # 使用前两个波峰
                self.frame_a_var.set(peaks[0][0])
                self.frame_b_var.set(peaks[1][0])
                self.do_compare()

    def _calculate_differences(self, frame_a: Dict, frame_b: Dict) -> Dict:
        """计算两帧之间的差异"""
        diffs = {}

        # 数值字段对比
        numeric_fields = [
            'roi2_gray', 'video_seconds',
            'roi3.gray', 'threshold.used'
        ]

        for field in numeric_fields:
            val_a = self._get_nested_value(frame_a, field)
            val_b = self._get_nested_value(frame_b, field)

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                if val_a != 0:
                    diff_pct = (val_b - val_a) / val_a * 100
                    has_diff = abs(diff_pct) > 5  # 差异超过5%
                    diffs[field] = {
                        'value_a': val_a,
                        'value_b': val_b,
                        'diff': val_b - val_a,
                        'diff_pct': diff_pct,
                        'has_diff': has_diff
                    }
                else:
                    diffs[field] = {'has_diff': False}

        # peaks 数量对比
        green_count_a = len(self._get_nested_value(frame_a, 'peaks.green') or [])
        green_count_b = len(self._get_nested_value(frame_b, 'peaks.green') or [])
        red_count_a = len(self._get_nested_value(frame_a, 'peaks.red') or [])
        red_count_b = len(self._get_nested_value(frame_b, 'peaks.red') or [])

        diffs['peaks.green'] = {
            'value_a': green_count_a,
            'value_b': green_count_b,
            'diff': green_count_b - green_count_a,
            'has_diff': green_count_a != green_count_b
        }

        diffs['peaks.red'] = {
            'value_a': red_count_a,
            'value_b': red_count_b,
            'diff': red_count_b - red_count_a,
            'has_diff': red_count_a != red_count_b
        }

        return diffs

    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """获取嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def set_frames(self, frame_a_idx: int, frame_b_idx: int):
        """设置要对比的帧"""
        self.frame_a_var.set(frame_a_idx)
        self.frame_b_var.set(frame_b_idx)
        self.do_compare()


class CompareSingleFrameView(SingleFrameView):
    """用于对比的单帧视图（支持差异高亮）"""

    def __init__(self, parent: tk.Widget, side_name: str):
        """
        初始化对比视图

        Args:
            parent: 父容器
            side_name: 侧标识 ('A' 或 'B')
        """
        self.side_name = side_name
        self.diffs = {}
        self.is_side_a = True
        # 不调用父类的 __init__，避免重复创建 UI
        self.parent = parent
        self.current_frame = None
        self.value_labels = {}
        self.on_frame_change = None
        self._create_ui()

    def update_frame(self, frame_data: Dict[str, Any], diffs: Dict, is_side_a: bool):
        """更新显示帧数据（带差异高亮）"""
        self.current_frame = frame_data
        self.diffs = diffs
        self.is_side_a = is_side_a

        for field, label in self.value_labels.items():
            value = self._get_nested_value(frame_data, field)
            formatted_value = self._format_value(value)

            # 为 roi3.g1 和 roi3.g2 添加百分号
            if field in ['roi3.g1', 'roi3.g2'] and value is not None:
                formatted_value = f'{formatted_value}%'

            # 检查是否有差异
            diff_info = diffs.get(field, {})
            if diff_info.get('has_diff', False):
                # 添加差异标记
                if is_side_a:
                    # 显示原值
                    label.config(text=formatted_value + ' →', background='')
                else:
                    # 显示新值和差值百分比
                    diff_pct = diff_info.get('diff_pct', 0)
                    sign = '+' if diff_pct > 0 else ''
                    bg_color = '#d4edda' if diff_pct > 0 else '#f8d7da'  # 绿色或红色背景
                    label.config(
                        text=f'→ {formatted_value} ({sign}{diff_pct:.1f}%)',
                        background=bg_color
                    )
            else:
                label.config(text=formatted_value, background='')


# =============================================================================
# WaveformView - 波形趋势图
# =============================================================================

class WaveformView:
    """波形趋势图视图"""

    def __init__(self, parent: tk.Widget, loader: CacheLoader,
                 on_frame_click=None):
        """
        初始化波形视图

        Args:
            parent: 父容器
            loader: 数据加载器（初始可能是None，加载文件后会更新）
            on_frame_click: 点击帧的回调函数
        """
        self.parent = parent
        self.loader_ref = lambda: loader  # 使用函数延迟获取 loader
        self.on_frame_click = on_frame_click
        self.fig = None
        self.canvas = None
        self.ax = None
        self.current_frame_line = None

        self._create_ui()

    def _create_ui(self):
        """创建用户界面"""
        # 创建 matplotlib 图形
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 嵌入到 Tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # 绑定点击事件
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def plot_waveform(self, current_frame_idx: int = None):
        """绘制波形图"""
        loader = self.loader_ref()
        if not loader or not loader.frames:
            return

        self.ax.clear()

        # 提取数据
        frame_indices = [f['frame_index'] for f in loader.frames]
        roi2_values = [f.get('roi2_gray', 0) for f in loader.frames]
        thresholds = [f.get('threshold', {}).get('used', 0) for f in loader.frames]

        # 绘制曲线
        self.ax.plot(frame_indices, roi2_values, 'g-',
                    linewidth=1, label='ROI2 灰度', alpha=0.7)
        self.ax.plot(frame_indices, thresholds, 'r--',
                    linewidth=1, label='阈值', alpha=0.5)

        # 标记峰值
        for color in ['green', 'red']:
            peaks = loader.peaks_map.get(color, [])
            for frame_idx, _ in peaks:
                if frame_idx in frame_indices:
                    idx = frame_indices.index(frame_idx)
                    marker_color = 'green' if color == 'green' else 'red'
                    self.ax.plot(frame_idx, roi2_values[idx], 'o',
                               color=marker_color, markersize=8,
                               markeredgecolor='black', markeredgewidth=1)

        # 绘制当前帧指示线
        if current_frame_idx and current_frame_idx in frame_indices:
            self.ax.axvline(x=current_frame_idx, color='blue',
                           linestyle=':', linewidth=2, label='当前帧')

        # 设置标签
        self.ax.set_xlabel('帧索引')
        self.ax.set_ylabel('灰度值')
        self.ax.set_title('ROI2 波形趋势图')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')

        self.canvas.draw()

    def _on_click(self, event):
        """处理图表点击事件"""
        if event.inaxes != self.ax:
            return

        frame_idx = int(round(event.xdata))
        if self.on_frame_click:
            self.on_frame_click(frame_idx)


# =============================================================================
# CacheBrowserGUI - 主窗口
# =============================================================================

class CacheBrowserGUI:
    """ROI 分析缓存浏览器主窗口"""

    def __init__(self, root: tk.Tk):
        """
        初始化主窗口

        Args:
            root: Tkinter 根窗口
        """
        self.root = root
        self.root.title("ROI Analysis Cache Browser")
        self.root.geometry("1400x900")

        # 数据模型
        self.loader: Optional[CacheLoader] = None
        self.current_frame_idx = 1

        # UI 组件
        self.single_frame_view = None
        self.compare_view = None
        self.waveform_view = None
        self.frame_var = tk.IntVar(value=1)
        self.total_frames_var = tk.StringVar(value="未加载文件")
        self.status_var = tk.StringVar(value="请选择一个 JSONL 文件")

        # 创建界面
        self._create_menu()
        self._create_toolbar()
        self._create_main_content()
        self._create_status_bar()

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开缓存文件...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill='x')

        # 文件选择
        ttk.Button(toolbar, text="选择文件",
                  command=self.open_file).pack(side='left', padx=5)

        ttk.Separator(toolbar, orient='vertical').pack(side='left', padx=10, fill='y')

        # 导航控制
        ttk.Button(toolbar, text="|<", width=4,
                  command=self.first_frame).pack(side='left', padx=2)
        ttk.Button(toolbar, text="<", width=4,
                  command=self.prev_frame).pack(side='left', padx=2)
        ttk.Button(toolbar, text=">", width=4,
                  command=self.next_frame).pack(side='left', padx=2)
        ttk.Button(toolbar, text=">|", width=4,
                  command=self.last_frame).pack(side='left', padx=2)

        # 帧输入
        ttk.Label(toolbar, text="帧:").pack(side='left', padx=(10, 5))
        ttk.Entry(toolbar, textvariable=self.frame_var, width=10).pack(side='left', padx=2)
        ttk.Button(toolbar, text="跳转",
                  command=self.goto_frame).pack(side='left', padx=2)

        self.total_frames_var = tk.StringVar(value="/ 0")
        ttk.Label(toolbar, textvariable=self.total_frames_var).pack(side='left', padx=5)

        ttk.Separator(toolbar, orient='vertical').pack(side='left', padx=10, fill='y')

        # 快速导航
        ttk.Button(toolbar, text="上一峰",
                  command=lambda: self.goto_peak('prev')).pack(side='left', padx=2)
        ttk.Button(toolbar, text="下一峰",
                  command=lambda: self.goto_peak('next')).pack(side='left', padx=2)

    def _create_main_content(self):
        """创建主内容区域"""
        # 创建多标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # 单帧浏览标签页
        single_frame = ttk.Frame(self.notebook)
        self.notebook.add(single_frame, text="单帧浏览")
        self.single_frame_view = SingleFrameView(
            single_frame,
            on_frame_change=self._on_frame_changed
        )

        # 双帧对比标签页
        compare = ttk.Frame(self.notebook)
        self.notebook.add(compare, text="双帧对比")
        self.compare_view = CompareView(compare, lambda: self.loader)

        # 波形趋势标签页
        waveform = ttk.Frame(self.notebook)
        self.notebook.add(waveform, text="波形趋势")
        self.waveform_view = WaveformView(
            waveform,
            self.loader,
            on_frame_click=self._on_waveform_click
        )

        # 绑定标签页切换事件
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _create_status_bar(self):
        """创建状态栏"""
        status_bar = ttk.Frame(self.root, padding=2)
        status_bar.pack(fill='x', side='bottom')
        ttk.Label(status_bar, textvariable=self.status_var).pack(side='left')

    def open_file(self):
        """打开文件对话框"""
        filename = filedialog.askopenfilename(
            title="选择 ROI 分析缓存文件",
            filetypes=[
                ("JSONL 文件", "*.jsonl"),
                ("所有文件", "*.*")
            ]
        )

        if filename:
            self.load_file(filename)

    def load_file(self, filename: str):
        """加载缓存文件"""
        self.loader = CacheLoader(filename)
        if self.loader.load():
            self.current_frame_idx = 1
            self.frame_var.set(1)
            self.total_frames_var.set(f"/ {self.loader.get_total_frames()}")
            self.status_var.set(f"已加载: {Path(filename).name}")
            self.update_display()
        else:
            self.status_var.set(f"加载失败: {filename}")

    def update_display(self):
        """更新所有视图显示"""
        if not self.loader:
            return

        # 更新单帧视图
        frame = self.loader.get_frame(self.current_frame_idx)
        if frame:
            self.single_frame_view.update_frame(frame)

        # 更新波形图
        if self.waveform_view:
            self.waveform_view.plot_waveform(self.current_frame_idx)

    def _on_frame_changed(self):
        """帧变化回调"""
        self.frame_var.set(self.current_frame_idx)

    def _on_tab_changed(self, event):
        """标签页切换回调"""
        tab_index = self.notebook.index("current")
        if tab_index == 2:  # 波形趋势标签页
            if self.waveform_view:
                self.waveform_view.plot_waveform(self.current_frame_idx)

    def _on_waveform_click(self, frame_idx: int):
        """波形图点击回调"""
        if self.loader and 1 <= frame_idx <= self.loader.get_total_frames():
            self.current_frame_idx = frame_idx
            self.frame_var.set(frame_idx)
            self.notebook.select(0)  # 切换到单帧浏览标签页
            self.update_display()

    def first_frame(self):
        """跳转到首帧"""
        if self.loader:
            self.current_frame_idx = 1
            self.frame_var.set(1)
            self.update_display()

    def prev_frame(self):
        """跳转到前一帧"""
        if self.loader and self.current_frame_idx > 1:
            self.current_frame_idx -= 1
            self.frame_var.set(self.current_frame_idx)
            self.update_display()

    def next_frame(self):
        """跳转到后一帧"""
        if self.loader and self.current_frame_idx < self.loader.get_total_frames():
            self.current_frame_idx += 1
            self.frame_var.set(self.current_frame_idx)
            self.update_display()

    def last_frame(self):
        """跳转到末帧"""
        if self.loader:
            self.current_frame_idx = self.loader.get_total_frames()
            self.frame_var.set(self.current_frame_idx)
            self.update_display()

    def goto_frame(self):
        """跳转到指定帧"""
        try:
            idx = self.frame_var.get()
            if self.loader and 1 <= idx <= self.loader.get_total_frames():
                self.current_frame_idx = idx
                self.update_display()
        except ValueError:
            pass

    def goto_peak(self, direction: str = 'next', color: str = 'green'):
        """跳转到峰值帧"""
        if not self.loader:
            return

        if direction == 'next':
            next_idx = self.loader.find_next_peak(self.current_frame_idx, color)
            if next_idx:
                self.current_frame_idx = next_idx
                self.frame_var.set(next_idx)
                self.update_display()
        else:
            prev_idx = self.loader.find_prev_peak(self.current_frame_idx, color)
            if prev_idx:
                self.current_frame_idx = prev_idx
                self.frame_var.set(prev_idx)
                self.update_display()

    def show_about(self):
        """显示关于对话框"""
        message = (
            "ROI Analysis Cache Browser\n\n"
            "版本: 1.0\n"
            "用于浏览和对比 roi_analysis_cache_*.jsonl 文件\n\n"
            "功能:\n"
            "- 分页卡片浏览帧数据\n"
            "- 双帧对比并高亮差异\n"
            "- 波形趋势图可视化\n"
            "- 智能字段说明"
        )
        tk.messagebox.showinfo("关于", message)


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主程序入口"""
    root = tk.Tk()
    app = CacheBrowserGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
